"""Conv/GN/SiLU patch embedding + DPE + DualMamba + mean-concat fusion.

X and S [B,H,W,D] are spatially averaged; center spectral P [B,1,D]
is squeezed. Linear(concat(mean(X), mean(S), P)) produces [B,D].
No spectral broadcasting, complementary gate, shared fusion residual,
branch normalization, raw-HSI encoder, or convolutional enhancement.
The existing FFN, final LayerNorm and classifier process the fused vector.

The input stem is Conv2d(in_chans,D,1) -> GroupNorm -> SiLU.
All downstream features originate from this embedding. DPE_Mamba_CFAM.py
remains the original version. DPE, Mamba, FFN and classifier definitions
are unchanged; replacing the embedding changes seeded initial weights.
The public factory remains basic_spa_spe_mamba. mean_concat is the
explicit selector; previous CFAM selectors are compatibility aliases.
Pooled fusion supports one block. New checkpoints must be trained.
"""

import copy
import math
from numbers import Integral

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

from .rs_mamba_ss import Mlp, OSSM


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def _probability(value, name):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError(f'{name} must lie in [0, 1)')
    return value


def _attention_logits(query, key):
    """Promote before QK multiplication, including inside CUDA autocast."""
    dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=query.device.type, enabled=False):
        return query.to(dtype) @ key.to(dtype).transpose(-2, -1)


def _working_float(value):
    return value if value.dtype == torch.float64 else value.float()


class BidirectionalFusionState(nn.Module):
    """BFS group fusion over aligned forward/backward recurrent states."""

    def __init__(self, d_inner, d_state, groups=8):
        super().__init__()
        self.d_inner = _integer(d_inner, 'd_inner')
        self.d_state = _integer(d_state, 'd_state')
        self.state_width = self.d_inner * self.d_state
        self.groups = _integer(groups, 'BFS groups')

        if self.state_width % self.groups != 0:
            raise ValueError(
                f"BFS state width {self.state_width} must be divisible by "
                f"groups={self.groups}"
            )

        self.group_fusion = nn.Conv1d(
            self.state_width,
            self.state_width,
            kernel_size=1,
            groups=self.groups,
        )

    def forward(self, state):
        if state.ndim != 4:
            raise ValueError(
                f"Expected BFS state [B,d_inner,d_state,L], got {tuple(state.shape)}"
            )
        if state.shape[1] != self.d_inner or state.shape[2] != self.d_state:
            raise ValueError(
                f"Expected BFS state channels [{self.d_inner},{self.d_state}], "
                f"got [{state.shape[1]},{state.shape[2]}]"
            )

        batch, _, _, length = state.shape
        state = state.reshape(batch, self.state_width, length)
        state = F.gelu(self.group_fusion(state))
        return state.reshape(
            batch,
            self.d_inner,
            self.d_state,
            length,
        )


class _DPEAttention(nn.Module):
    """Spatial or channel attention on learned features, with geometric bias."""
    def __init__(self, dim, channel=False, center=False, radius=1.5, num_heads=None):
        super().__init__()
        dim = _integer(dim, 'attention dim')
        self.heads = (4 if dim % 4 == 0 else 1) if num_heads is None else _integer(num_heads, 'attention heads')
        if dim % self.heads:
            raise ValueError('attention dim must be divisible by num_heads')
        self.channel, self.center = bool(channel), bool(center)
        self.radius = _positive(radius, 'attention radius')
        self.qkv = nn.Conv2d(dim, 3 * dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.temperature = nn.Parameter(torch.ones(self.heads, 1, 1))

    def forward(self, x):
        b, d, h, w = x.shape
        n, k = h * w, d // self.heads
        q, key, v = self.qkv(x).reshape(b, 3, self.heads, k, n).unbind(1)
        if self.channel:
            q, key = F.normalize(_working_float(q), dim=-1), F.normalize(_working_float(key), dim=-1)
            weights = _attention_logits(q, key) * self.temperature.clamp(0.1, 8)
            result = weights.softmax(-1).to(v.dtype) @ v
        else:
            q, key, v = q.transpose(-2, -1), key.transpose(-2, -1), v.transpose(-2, -1)
            yy, xx = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing='ij')
            coords = torch.stack((yy, xx), -1).reshape(n, 2).to(_working_float(q).dtype)
            distance = (coords[:, None] - coords[None]).square().sum(-1)
            if self.center:
                center = (h // 2) * w + w // 2
                q = q[:, :, center:center + 1]
                prior = -distance[center:center + 1] / (2 * self.radius ** 2)
                weights = _attention_logits(q, key) / math.sqrt(k)
                weights = weights * self.temperature.clamp(0.1, 8) + prior
                context = (weights.softmax(-1).to(v.dtype) @ v).transpose(-2, -1).reshape(b, d, 1, 1)
                # Center correction extends smoothly to its spatial neighborhood.
                footprint = (-distance[center] / (2 * self.radius ** 2)).exp().reshape(1, 1, h, w)
                return self.proj(context - x[:, :, h // 2:h // 2 + 1, w // 2:w // 2 + 1]) * footprint.to(x.dtype)
            weights = _attention_logits(q, key) / math.sqrt(k)
            weights = weights * self.temperature.clamp(0.1, 8) - distance / (2 * self.radius ** 2)
            result = (weights.softmax(-1).to(v.dtype) @ v).transpose(-2, -1)
        return self.proj(result.reshape(b, d, h, w))


class DualMambaDPE(nn.Module):
    """D20: dual attention with spatial correction excluded at the center.

    The original depthwise positional residual is retained. Spatial and
    channel attention act on the normalized embedding. By default RMS is
    measured after excluding the center spatial correction, so the actual
    update obeys the limit even when the two branches cancel at the center.
    The per-channel correction gain starts at 0.05 and is bounded by 0.5.
    """

    config = {
        "mechanism": "dual_attention",
        "radius": 1.5,
        "start": 0.05,
        "preserve_spatial_center": True,
    }

    def __init__(self, hidden_dim, num_heads=None, radius=1.5, gain_init=0.05,
                 gain_bound=0.5, limit_after_center_mask=True):
        super().__init__()
        hidden_dim = _integer(hidden_dim, 'DPE hidden_dim')
        self.gain_bound = _positive(gain_bound, 'DPE gain_bound')
        gain_init = float(gain_init)
        if not math.isfinite(gain_init) or abs(gain_init) >= self.gain_bound:
            raise ValueError('DPE gain_init must lie strictly inside its gain_bound')
        if not isinstance(limit_after_center_mask, bool):
            raise ValueError('limit_after_center_mask must be a boolean')
        self.limit_after_center_mask = limit_after_center_mask
        self.dwconv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim
        )
        self.dim = int(hidden_dim)
        # Preserve the experiment's initialization order and global RNG state.
        with torch.random.fork_rng(devices=[]):
            self.norm = nn.LayerNorm(hidden_dim)
            self.gain = nn.Parameter(torch.full(
                (1, hidden_dim, 1, 1), math.atanh(gain_init / self.gain_bound)
            ))
            self.core = _DPEAttention(hidden_dim, radius=radius, num_heads=num_heads)
            self.channel = _DPEAttention(hidden_dim, channel=True, num_heads=num_heads)

    def forward(self, x, raw_patch=None):
        # The shared block interface also carries raw_patch to V139 CFAM.
        if x.ndim != 4 or x.shape[-1] != self.dim:
            raise ValueError(f'DPE expects [B,H,W,{self.dim}], got {tuple(x.shape)}')
        values = x.permute(0, 3, 1, 2)
        baseline = values + self.dwconv(values)
        normalized = self.norm(x).permute(0, 3, 1, 2)
        spatial_correction = self.core(normalized)
        channel_correction = self.channel(normalized)
        correction = spatial_correction + channel_correction

        height, width = values.shape[-2:]
        center = torch.zeros(1, 1, height, width, device=values.device, dtype=torch.bool)
        center[:, :, height // 2, width // 2] = True
        protected = torch.where(center, channel_correction, correction)
        limited = protected if self.limit_after_center_mask else correction
        reference_rms = _working_float(baseline).square().mean((1, 2, 3), keepdim=True).add(1e-6).sqrt()
        correction_rms = _working_float(limited).square().mean((1, 2, 3), keepdim=True).add(1e-6).sqrt()
        scale = (reference_rms / correction_rms).clamp(max=1.0).to(values.dtype)
        output = baseline + self.gain_bound * self.gain.tanh() * (protected * scale)
        return output.permute(0, 2, 3, 1)


class DualMambaSpatialMamba(nn.Module):
    """DualMamba-style lightweight spatial OSSM."""

    def __init__(
        self,
        hidden_dim,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        d_conv=3,
        conv_bias=True,
        dropout=0.0,
        ssm_init="v0",
        forward_type="v3",
    ):
        super().__init__()

        self.norm = nn.LayerNorm(hidden_dim)

        self.mamba = OSSM(
            d_model=hidden_dim,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            act_layer=nn.SiLU,
            d_conv=d_conv,
            conv_bias=conv_bias,
            dropout=dropout,
            initialize=ssm_init,
            forward_type=forward_type,
            k_group=1,
        )

    def forward(self, x):
        return self.mamba(self.norm(x))


SpatialMamba = DualMambaSpatialMamba
SpatialSnakeMamba = DualMambaSpatialMamba


class SpectralOSSMWithBFS(OSSM):
    """Fuller spectral OSSM with optional BFS hidden-state fusion.

    The stock OSSM in ``rs_mamba_ss.py`` expects Conv2d when d_conv > 1,
    which is appropriate for spatial feature maps but not for the spectral
    sequence [B,L,1]. Therefore this class:

    1) inherits the complete OSSM projection / SSM / gate parameters;
    2) keeps the z branch enabled;
    3) uses a depthwise Conv1d over the spectral/feature sequence as the
       sequence-domain equivalent of OSSM's internal depthwise convolution;
    4) exposes recurrent states so BFS can still be inserted.

    The scan remains bidirectional (K=2), exactly as in the existing spectral
    branch. Thus this experiment changes block completeness, not scan topology.
    """

    def __init__(
        self,
        d_model=1,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        d_conv=3,
        conv_bias=True,
        dropout=0.0,
        initialize="v0",
        forward_type="v3",
        bfs_groups=8,
        bfs_enabled=True,
    ):
        if not isinstance(bfs_enabled, bool):
            raise ValueError("bfs_enabled must be a boolean")

        requested_d_conv = _integer(d_conv, 'spectral d_conv', minimum=0)
        if requested_d_conv > 1 and requested_d_conv % 2 == 0:
            raise ValueError('spectral d_conv must be 0, 1, or a positive odd kernel')

        # Use OSSM's full z/gate parameterization, but suppress its Conv2d
        # because the spectral input is 3-D [B,L,C]. A dedicated Conv1d is
        # created below.
        super().__init__(
            d_model=d_model,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            act_layer=nn.SiLU,
            d_conv=0,
            conv_bias=conv_bias,
            dropout=dropout,
            initialize=initialize,
            forward_type="spe" + forward_type,
            k_group=2,
        )

        self.spectral_d_conv = requested_d_conv
        self.d_model_bfs = int(d_model)
        self.d_inner_bfs = int(ssm_ratio * d_model)
        self.d_state_bfs = int(d_state)
        self.dt_rank_bfs = (
            math.ceil(d_model / 16)
            if dt_rank == "auto"
            else int(dt_rank)
        )
        self.num_directions_bfs = 2

        if self.spectral_d_conv > 1:
            self.spectral_conv1d = nn.Conv1d(
                in_channels=self.d_inner_bfs,
                out_channels=self.d_inner_bfs,
                kernel_size=self.spectral_d_conv,
                padding=(self.spectral_d_conv - 1) // 2,
                groups=self.d_inner_bfs,
                bias=conv_bias,
            )
        else:
            self.spectral_conv1d = nn.Identity()

        self.bfs_enabled = bfs_enabled
        self.state_fusion = (
            BidirectionalFusionState(
                d_inner=self.d_inner_bfs,
                d_state=self.d_state_bfs,
                groups=bfs_groups,
            )
            if self.bfs_enabled
            else None
        )

        # Keep the previously validated BFS direction readout.
        self.direction_logits = nn.Parameter(torch.zeros(2))

    def forward_bfs(self, sequence):
        """Spectral processing with optional BFS; retain the legacy method name."""

        if sequence.ndim != 3:
            raise ValueError(
                f"Expected spectral sequence [B,L,D], got {tuple(sequence.shape)}"
            )
        if sequence.shape[-1] != self.d_model_bfs:
            raise ValueError(
                f"Expected d_model={self.d_model_bfs}, got {sequence.shape[-1]}"
            )

        batch, length, _ = sequence.shape

        # --------------------------------------------------------------
        # 1) Full OSSM input projection.
        #    Unlike v3noz, v3 keeps x and z/gate branches.
        # --------------------------------------------------------------
        projected = self.in_proj(sequence)

        if self.disable_z:
            x_content = projected
            z = None
        else:
            x_content, z = projected.chunk(2, dim=-1)
            if not self.disable_z_act:
                z = self.act(z)

        # --------------------------------------------------------------
        # 2) Restore internal local convolution for the spectral sequence.
        #    [B,L,d_inner] -> [B,d_inner,L] -> depthwise Conv1d.
        # --------------------------------------------------------------
        x_content = x_content.transpose(1, 2).contiguous()
        x_content = self.spectral_conv1d(x_content)
        x_content = self.act(x_content)

        # --------------------------------------------------------------
        # 3) Same bidirectional spectral scan topology as before.
        # --------------------------------------------------------------
        u = torch.stack(
            [
                x_content,
                torch.flip(x_content, dims=[-1]),
            ],
            dim=1,
        )  # [B,2,d_inner,L]

        # --------------------------------------------------------------
        # 4) OSSM x/dt/B/C projections.
        # --------------------------------------------------------------
        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l",
            u,
            self.x_proj_weight,
        )

        dt, state_b, state_c = torch.split(
            x_dbl,
            [
                self.dt_rank_bfs,
                self.d_state_bfs,
                self.d_state_bfs,
            ],
            dim=2,
        )

        raw_delta = torch.einsum(
            "b k r l, k d r -> b k d l",
            dt,
            self.dt_projs_weight,
        )

        delta = F.softplus(
            raw_delta.float()
            + self.dt_projs_bias.float()[None, :, :, None]
        )

        u_float = u.float()

        a = -torch.exp(
            self.A_logs.float()
        ).view(
            self.num_directions_bfs,
            self.d_inner_bfs,
            self.d_state_bfs,
        )

        # --------------------------------------------------------------
        # 5) Explicit SSM evolution to expose hidden states for BFS.
        # --------------------------------------------------------------
        state = torch.zeros(
            batch,
            self.num_directions_bfs,
            self.d_inner_bfs,
            self.d_state_bfs,
            device=sequence.device,
            dtype=torch.float32,
        )

        state_steps = []

        for index in range(length):
            delta_t = delta[..., index]
            input_t = u_float[..., index]

            decay = torch.exp(
                delta_t.unsqueeze(-1)
                * a.unsqueeze(0)
            )

            state = (
                decay * state
                + delta_t.unsqueeze(-1)
                * state_b[..., index].float().unsqueeze(2)
                * input_t.unsqueeze(-1)
            )

            state_steps.append(state)

        states = torch.stack(
            state_steps,
            dim=-1,
        )

        # --------------------------------------------------------------
        # 6) Align directions. Without BFS, preserve each direction's own
        #    state for its C readout; an identity on their sum is NOT no-BFS.
        # --------------------------------------------------------------
        forward_state = states[:, 0]
        backward_state = torch.flip(
            states[:, 1],
            dims=[-1],
        )

        if self.bfs_enabled:
            fused_state = self.state_fusion(
                forward_state + backward_state
            )
            forward_state = fused_state
            backward_state = fused_state

        forward_c = state_c[:, 0].float()
        backward_c = torch.flip(
            state_c[:, 1].float(),
            dims=[-1],
        )

        forward_u = u_float[:, 0]
        backward_u = torch.flip(
            u_float[:, 1],
            dims=[-1],
        )

        skip = self.Ds.float().view(
            self.num_directions_bfs,
            self.d_inner_bfs,
        )

        forward_output = (
            (
                forward_state
                * forward_c.unsqueeze(1)
            ).sum(dim=2)
            + skip[0][None, :, None] * forward_u
        )

        backward_output = (
            (
                backward_state
                * backward_c.unsqueeze(1)
            ).sum(dim=2)
            + skip[1][None, :, None] * backward_u
        )

        forward_output = forward_output.transpose(1, 2).contiguous()
        backward_output = backward_output.transpose(1, 2).contiguous()

        # --------------------------------------------------------------
        # 7) Merge directions BEFORE out_norm / gate / out_proj, matching
        #    the standard OSSM/Mamba ordering more closely.
        # --------------------------------------------------------------
        direction_weights = self.direction_logits.softmax(dim=0)

        y = (
            direction_weights[0] * forward_output
            + direction_weights[1] * backward_output
        )

        y = self.out_norm(y)

        # Restore z/gating branch.
        if z is not None:
            y = y * z

        y = self.out_proj(y)
        return self.dropout(y)


class SpectralCenterMamba(nn.Module):
    """DualMamba-style center-pixel spectral branch with optional BFS.

    Shared feature x_dpe:
        [B,H,W,D]

    Center extraction:
        [B,D]

    DualMamba-style spectral sequence:
        [B,D] -> [B,1,D] -> LayerNorm(D) -> [B,D,1]

    The sequence length is D, and OSSM d_model is 1.
    """

    def __init__(
        self,
        patch_size,
        hidden_dim,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        conv_bias=True,
        dropout=0.0,
        spectral_ssm_conv=3,
        spectral_ssm_init="v0",
        spectral_forward_type="v3",
        bfs_groups=8,
        bfs_enabled=True,
    ):
        super().__init__()

        self.patch_size = int(patch_size)
        self.hidden_dim = int(hidden_dim)
        self.center = self.patch_size // 2

        # This follows DualMamba: normalize the center embedding over D.
        self.norm_spe = nn.LayerNorm(self.hidden_dim)

        # DualMamba spectral OSSM uses d_model=1 and k_group=2.
        self.mamba = SpectralOSSMWithBFS(
            d_model=1,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            d_conv=spectral_ssm_conv,
            conv_bias=conv_bias,
            dropout=dropout,
            initialize=spectral_ssm_init,
            forward_type=spectral_forward_type,
            bfs_groups=bfs_groups,
            bfs_enabled=bfs_enabled,
        )

    def forward(self, x_dpe):
        if x_dpe.ndim != 4:
            raise ValueError(
                f"Expected shared DPE feature [B,H,W,D], got {tuple(x_dpe.shape)}"
            )

        batch, height, width, channels = x_dpe.shape

        if (height, width) != (
            self.patch_size,
            self.patch_size,
        ):
            raise ValueError(
                "Spatial patch does not match configured patch_size"
            )

        if channels != self.hidden_dim:
            raise ValueError(
                "Feature dimension does not match configured hidden_dim"
            )

        # --------------------------------------------------------------
        # Shared embedding + shared DPE + center pixel.
        # --------------------------------------------------------------
        center_feature = x_dpe[
            :,
            self.center,
            self.center,
            :
        ]  # [B,D]

        # Exactly the DualMamba data-layout idea:
        # [B,D] -> [B,1,D] -> LN(D) -> [B,D,1]
        center_feature = self.norm_spe(
            center_feature.unsqueeze(1)
        )
        sequence = center_feature.permute(
            0, 2, 1
        ).contiguous()  # [B,D,1]

        spectral_output = self.mamba.forward_bfs(
            sequence
        )  # [B,D,1]

        # Return the same shape style used by DualMamba CAS2F: [B,1,D].
        return spectral_output.permute(
            0, 2, 1
        ).contiguous()


# Backward-compatible alias.
SpectralBiMamba = SpectralCenterMamba


class CrossAttentionSpaSpeFusion(nn.Module):
    """CAS2F-style Spa/Spe interaction, now very close to DualMamba.

    Inputs
    ------
    x_residual : [B,H,W,D]
        Shared embedding after DPE.

    spa_feature : [B,H,W,D]
        Spatial OSSM output.

    spe_feature : [B,1,D]
        Center-pixel spectral OSSM+BFS output.

    Fusion
    ------
        A_spe = Softmax(G_spe)
        A_spa = Softmax(AvgPool(G_spa))

        G_spa <- A_spe * G_spa
        G_spe <- A_spa * G_spe

        inject G_spe into the center pixel of G_spa
        add x_residual
    """

    def __init__(self, patch_size, hidden_dim):
        super().__init__()

        self.patch_size = int(patch_size)
        self.hidden_dim = int(hidden_dim)
        self.center = self.patch_size // 2

        self.softmax_spe = nn.Softmax(dim=-1)
        self.softmax_spa = nn.Softmax(dim=-1)
        self.spa_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x_residual, spa_feature, spe_feature):
        if x_residual.ndim != 4:
            raise ValueError(
                f"x_residual must be [B,H,W,D], got {tuple(x_residual.shape)}"
            )

        if spa_feature.shape != x_residual.shape:
            raise ValueError(
                "spa_feature must have the same shape as x_residual"
            )

        batch, height, width, channels = x_residual.shape

        if (height, width) != (
            self.patch_size,
            self.patch_size,
        ):
            raise ValueError(
                "Feature-map size does not match configured patch_size"
            )

        if channels != self.hidden_dim:
            raise ValueError(
                "Feature dimension does not match configured hidden_dim"
            )

        if spe_feature.shape != (
            batch,
            1,
            channels,
        ):
            raise ValueError(
                f"Expected spe_feature [B,1,D] = "
                f"[{batch},1,{channels}], got {tuple(spe_feature.shape)}"
            )

        # Spectral attention: [B,1,D] -> [B,1,1,D]
        attn_spe = self.softmax_spe(
            spe_feature
        ).unsqueeze(1)

        # Spatial attention:
        # [B,H,W,D] -> [B,D,H,W] -> [B,D,1,1]
        # -> [B,D,1] -> [B,1,D]
        spa_global = self.spa_pool(
            spa_feature.permute(0, 3, 1, 2)
        )
        spa_global = spa_global.squeeze(-1).permute(
            0, 2, 1
        )

        attn_spa = self.softmax_spa(
            spa_global
        )  # [B,1,D]

        # Cross modulation.
        spa_cross = spa_feature * attn_spe
        spe_cross = spe_feature * attn_spa

        # Center injection, matching DualMamba CAS2F logic.
        fused = spa_cross.clone()
        fused[
            :,
            self.center,
            self.center,
            :
        ] = (
            fused[
                :,
                self.center,
                self.center,
                :
            ]
            + spe_cross[:, 0, :]
        )

        # Residual positional feature.
        return x_residual + fused


class MeanConcatFusion(nn.Module):
    """Fuse mean-pooled X/S and center P into a single [B,D] vector."""

    def __init__(self, patch_size, hidden_dim):
        super().__init__()
        self.patch_size = _integer(patch_size, 'fusion patch_size')
        self.hidden_dim = _integer(hidden_dim, 'fusion hidden_dim')
        if self.patch_size % 2 == 0:
            raise ValueError('fusion patch_size must be positive and odd')
        self.fusion_proj = nn.Linear(3 * self.hidden_dim, self.hidden_dim)

    def forward(self, x_residual, spa_feature, spe_feature):
        # x_residual is a legacy argument name: X is concatenated, not added.
        if x_residual.ndim != 4:
            raise ValueError('X must be [B,H,W,D]')
        b, h, w, d = x_residual.shape
        if (h, w, d) != (self.patch_size, self.patch_size, self.hidden_dim):
            raise ValueError('X shape does not match fusion configuration')
        if spa_feature.shape != x_residual.shape or spe_feature.shape != (b, 1, d):
            raise ValueError('Spa/Spe feature shapes do not match fusion inputs')
        features = torch.cat((
            x_residual.mean(dim=(1, 2)),
            spa_feature.mean(dim=(1, 2)),
            spe_feature.squeeze(1),
        ), dim=-1)
        return self.fusion_proj(features)


class BasicSpaSpeBlock(nn.Module):
    """Unchanged DPE/Spa/Spe -> mean-pool/concatenate/project -> vector FFN."""

    def __init__(
        self,
        patch_size,
        hidden_dim,
        drop_path=0.0,
        mlp_ratio=4.0,
        mlp_drop_rate=0.0,
        spatial_forward_type="v3noz",
        spatial_ssm_conv=0,
        spatial_ssm_init="v0",
        spectral_forward_type="v3",
        spectral_ssm_conv=3,
        spectral_ssm_init="v0",
        bfs_groups=8,
        bfs_enabled=False,
        fusion_type="mean_concat",
        fusion_heads=8,
        dpe_options=None,
        cfam_options=None,
        **mamba_kwargs,
    ):
        super().__init__()

        self.patch_size = int(patch_size)

        # --------------------------------------------------------------
        # One shared DPE. Both branches consume x_dpe.
        # --------------------------------------------------------------
        self.dpe = DualMambaDPE(hidden_dim, **dict(dpe_options or {}))

        # Spatial branch unchanged.
        self.spatial = DualMambaSpatialMamba(
            hidden_dim=hidden_dim,
            d_state=mamba_kwargs["d_state"],
            ssm_ratio=mamba_kwargs["ssm_ratio"],
            dt_rank=mamba_kwargs["dt_rank"],
            d_conv=spatial_ssm_conv,
            conv_bias=mamba_kwargs["conv_bias"],
            dropout=mamba_kwargs["dropout"],
            ssm_init=spatial_ssm_init,
            forward_type=spatial_forward_type,
        )

        # New DualMamba-like center-pixel spectral branch.
        self.spectral = SpectralCenterMamba(
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            d_state=mamba_kwargs["d_state"],
            ssm_ratio=mamba_kwargs["ssm_ratio"],
            dt_rank=mamba_kwargs["dt_rank"],
            conv_bias=mamba_kwargs["conv_bias"],
            dropout=mamba_kwargs["dropout"],
            spectral_ssm_conv=spectral_ssm_conv,
            spectral_ssm_init=spectral_ssm_init,
            spectral_forward_type=spectral_forward_type,
            bfs_groups=bfs_groups,
            bfs_enabled=bfs_enabled,
        )

        # Keep the existing JSON selectors usable without changing the protocol.
        del fusion_heads  # Legacy attention heads; mean_concat has no attention.
        if fusion_type in ("mean_concat", "concat_residual", "center_cfam", "center_cfam_residual"):
            if cfam_options:
                raise ValueError("mean_concat has no CFAM enhancement options")
            # Isolate fusion initialization from the shared backbone RNG.
            with torch.random.fork_rng(devices=[]):
                self.cross_fusion = MeanConcatFusion(
                    patch_size=patch_size,
                    hidden_dim=hidden_dim,
                )
                self.cross_fusion.apply(BasicSpaSpeMamba._init_weights)
                for module in self.cross_fusion.modules():
                    if isinstance(module, nn.Linear):
                        module._fusion_initialized = True
        elif fusion_type == "cas2f":
            if cfam_options:
                raise ValueError('cas2f does not accept CFAM enhancement options')
            self.cross_fusion = CrossAttentionSpaSpeFusion(patch_size, hidden_dim)
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        self.drop_path = (
            DropPath(drop_path)
            if drop_path > 0
            else nn.Identity()
        )

        self.norm_ffn = nn.LayerNorm(hidden_dim)

        self.ffn = Mlp(
            in_features=hidden_dim,
            hidden_features=int(hidden_dim * mlp_ratio),
            out_features=hidden_dim,
            drop=mlp_drop_rate,
        )

    def forward(self, x, raw_patch=None):
        # --------------------------------------------------------------
        # Shared representation:
        # embedding has already been computed before entering the block.
        # DPE is applied once and shared by Spa and Spe.
        # --------------------------------------------------------------
        x_dpe = self.dpe(x, raw_patch=raw_patch)

        # Spatial OSSM on the full shared feature map.
        spa_delta = self.spatial(x_dpe)

        # Spectral OSSM+BFS on center-pixel embedding along feature dimension.
        spe_delta = self.spectral(x_dpe)  # [B,1,D]

        # Mean-concat produces a vector without adding the shared X residual.
        fusion_inputs=dict(x_residual=x_dpe,spa_feature=spa_delta,spe_feature=spe_delta)
        x=self.cross_fusion(**fusion_inputs)

        # Existing residual FFN now operates on the fused [B,D] vector.
        return x + self.drop_path(
            self.ffn(
                self.norm_ffn(x)
            )
        )


class BasicSpaSpeMamba(nn.Module):
    """Shared DPE/DualMamba backbone with mean-pooled concatenation fusion.

    DPE and both Mamba branches are unchanged. CFAM enhancement is disabled.
    """

    def __init__(
        self,
        n_groups=(4,),
        patch_size=7,
        in_chans=200,
        num_classes=16,
        dims=(64,),
        depths=(1,),
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0,
        use_gate=False,
        mlp_ratio=4.0,
        mlp_drop_rate=0.0,
        drop_path_rate=0.1,

        # Spatial OSSM settings.
        spatial_forward_type="v3noz",
        spatial_ssm_conv=0,
        spatial_ssm_init="v0",

        # Spectral OSSM settings.
        spectral_forward_type="v3",
        spectral_ssm_conv=3,
        spectral_ssm_init="v0",
        bfs_groups=8,
        bfs_enabled=False,
        fusion_type="mean_concat",
        fusion_heads=8,
        dpe_options=None,
        cfam_options=None,

        **kwargs,
    ):
        super().__init__()

        # Legacy fields are retained in the signature for JSON compatibility.
        if kwargs:
            raise TypeError('Unknown model options: ' + ', '.join(sorted(kwargs)))
        del use_gate, ssm_conv

        if (
            len(n_groups) != 1
            or len(dims) != 1
            or len(depths) != 1
        ):
            raise ValueError(
                "BasicSpaSpeMamba currently supports one encoder stage"
            )

        patch_size = _integer(patch_size, 'patch_size')
        if patch_size % 2 == 0:
            raise ValueError(
                "patch_size must be a positive odd integer"
            )

        depth = _integer(depths[0], 'depths[0]')
        if fusion_type != 'cas2f' and depth != 1:
            raise ValueError('mean-pooled fusion requires depths=[1]: its output is a vector')
        in_chans = _integer(in_chans, 'in_chans')
        num_classes = _integer(num_classes, 'num_classes')
        hidden_dim = _integer(dims[0], 'dims[0]')
        group_count = _integer(n_groups[0], 'n_groups[0]')
        if hidden_dim % group_count:
            raise ValueError('dims[0] must be divisible by n_groups[0]')
        drop_path_rate = _probability(drop_path_rate, 'drop_path_rate')
        cfam_options = dict(cfam_options or {})

        self.patch_size = int(patch_size)
        self.in_chans = int(in_chans)

        # HyPyraMamba-style shared stem. GroupNorm groups feature channels,
        # so the input bands need no padding (UP's 103 bands are used directly).
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_chans, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_count, hidden_dim),
            nn.SiLU(),
        )

        # A one-block stage must use the requested rate, not linspace(..., 1)=0.
        drop_paths = ([drop_path_rate] if depth == 1
                      else torch.linspace(0, drop_path_rate, depth).tolist())

        mamba_kwargs = dict(
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
        )

        self.blocks = nn.ModuleList(
            [
                BasicSpaSpeBlock(
                    patch_size=patch_size,
                    hidden_dim=hidden_dim,
                    drop_path=drop_paths[index],
                    mlp_ratio=mlp_ratio,
                    mlp_drop_rate=mlp_drop_rate,

                    spatial_forward_type=spatial_forward_type,
                    spatial_ssm_conv=spatial_ssm_conv,
                    spatial_ssm_init=spatial_ssm_init,

                    spectral_forward_type=spectral_forward_type,
                    spectral_ssm_conv=spectral_ssm_conv,
                    spectral_ssm_init=spectral_ssm_init,
                    bfs_groups=bfs_groups,
                    bfs_enabled=bfs_enabled,
                    fusion_type=fusion_type,
                    fusion_heads=fusion_heads,
                    dpe_options=dpe_options,
                    cfam_options=cfam_options,

                    **mamba_kwargs,
                )
                for index in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(
            hidden_dim,
            num_classes,
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear) and not getattr(module, "_fusion_initialized", False):
            trunc_normal_(
                module.weight,
                std=0.02,
            )
            if module.bias is not None:
                nn.init.constant_(
                    module.bias,
                    0,
                )

        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(
                module.bias,
                0,
            )
            nn.init.constant_(
                module.weight,
                1.0,
            )

    def forward_features(self, x):
        expected = (
            self.in_chans,
            self.patch_size,
            self.patch_size,
        )

        if (
            x.ndim != 5
            or x.shape[1] != 1
            or tuple(x.shape[2:]) != expected
        ):
            raise ValueError(
                f"Expected [B,1,{expected[0]},"
                f"{expected[1]},{expected[2]}], "
                f"got {tuple(x.shape)}"
            )

        # Every downstream module receives the shared embedded feature map.
        x = self.patch_embedding(x.squeeze(1))  # [B,D,H,W]
        x = x.permute(0, 2, 3, 1).contiguous()  # [B,H,W,D]

        for block in self.blocks:
            x = block(x)

        features = self.norm(x)
        if features.ndim == 2:
            return features
        return features.mean(dim=(1, 2))

    def forward(self, x):
        return self.head(
            self.forward_features(x)
        )

    def flops(self, shape=None, verbose=True):
        del verbose

        expected_shape = (
            1,
            self.in_chans,
            self.patch_size,
            self.patch_size,
        )

        shape = (
            expected_shape
            if shape is None
            else tuple(shape)
        )

        if shape != expected_shape:
            raise ValueError(
                f"Expected FLOPs input shape "
                f"{expected_shape}, got {shape}"
            )

        device = next(
            self.parameters()
        ).device

        model = copy.deepcopy(
            self
        ).to(device).eval()

        # Profiling is called only for run 0 by main.py. Preserve CPU and the
        # selected CUDA RNG so this inspection cannot perturb that run's seed.
        devices = [device.index] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            sample = torch.randn(
                (1, *shape), device=device, dtype=next(model.parameters()).dtype,
            )
            model(sample)

        params = sum(
            parameter.numel()
            for parameter
            in model.parameters()
        )

        print(
            f"params {params} "
            f"GFLOPs not reported for "
            f"basic_spa_spe_mamba"
        )


def basic_spa_spe_mamba(model_config):
    return BasicSpaSpeMamba(
        **model_config
    )


__all__ = [
    "BidirectionalFusionState",
    "DualMambaDPE",
    "DualMambaSpatialMamba",
    "SpatialMamba",
    "SpatialSnakeMamba",
    "SpectralOSSMWithBFS",
    "SpectralCenterMamba",
    "SpectralBiMamba",
    "CrossAttentionSpaSpeFusion",
    "MeanConcatFusion",
    "BasicSpaSpeBlock",
    "BasicSpaSpeMamba",
    "basic_spa_spe_mamba",
]

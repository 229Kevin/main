"""D20 DPE + V139 CFAM with configurable, numerically stable refinements.

DPE uses spatial/channel attention with selective center protection. V139
CFAM combines channel fusion, raw-patch encoding and locally biased
cross-attention to spatial/spectral branch values. The public factory stays
``basic_spa_spe_mamba`` and accepts the existing IP configuration.

Source: dpe_experiments_20261004/d20_selective_center_dual/mamba_CFAM.py.
Default parameter names/shapes retain IP checkpoint compatibility. DPE
limits the correction after center protection; single-block DropPath now
honors its configured rate. These training changes require fresh evaluation.
``dpe_options`` and ``cfam_options`` expose module tuning without changing
the shared embedding or the two Mamba scan topologies.
"""

import copy
import math
from numbers import Integral

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_

from .rs_mamba_ss import GroupedPixelEmbedding_mm, Mlp, OSSM


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


class CenterPairedCFAM(nn.Module):
    """V139 channel fusion with raw-to-branch local cross-attention.

    Raw tokens query actual Spa/Spe values through configurable cross-attention
    with a Gaussian spatial prior. Separate center-guided pooling
    produces a bounded raw-context correction. The optional legacy token
    attention is disabled by default. Local refinement acts on the DPE
    residual. Defaults retain V139: cross/pool heads 4/1, width 4D,
    raw dropout 0.2, gain 0.25, RMS capacity 1.75 and prior radii 1.0.
    ``num_heads`` only controls the optional legacy token attention.
    """
    def __init__(self, patch_size, hidden_dim, num_heads=8, spatial_residual=False,
                 context_enabled=False, center_readout=False, pool_readout=False,
                 fusion_drop_rate=0.0, multiplicative_gate=True,
                 token_attention=False, local_refinement=True,
                 raw_in_chans=200, raw_width_ratio=4.0,
                 raw_encoder_drop_rate=0.2, raw_gain_init=0.25,
                 raw_rms_limit=1.75, cross_heads=4, pool_heads=1,
                 spatial_prior_radius=1.0, raw_pool_radius=1.0):
        super().__init__()
        patch_size = _integer(patch_size, 'CFAM patch_size')
        hidden_dim = _integer(hidden_dim, 'CFAM hidden_dim')
        num_heads = _integer(num_heads, 'legacy token attention heads')
        if patch_size <= 0 or patch_size % 2 == 0:
            raise ValueError("patch_size must be positive and odd")
        if token_attention and hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if context_enabled or center_readout or pool_readout:
            raise ValueError('V139 uses raw context and mean readout; legacy readout flags are unsupported')
        if not multiplicative_gate:
            raise ValueError('V139 requires multiplicative_gate=True')
        self.raw_in_chans = _integer(raw_in_chans, 'raw_in_chans')
        self.raw_width_ratio = _positive(raw_width_ratio, 'raw_width_ratio')
        width = max(1, round(self.raw_width_ratio * hidden_dim))
        self.raw_rms_limit = _positive(raw_rms_limit, 'raw_rms_limit')
        self.spatial_prior_radius = _positive(spatial_prior_radius, 'spatial_prior_radius')
        self.raw_pool_radius = _positive(raw_pool_radius, 'raw_pool_radius')
        self.raw_cross_heads = _integer(cross_heads, 'cross_heads')
        self.raw_heads = _integer(pool_heads, 'pool_heads')
        if hidden_dim % self.raw_cross_heads or hidden_dim % self.raw_heads:
            raise ValueError('hidden_dim must be divisible by cross_heads and pool_heads')
        raw_gain_init = float(raw_gain_init)
        if not math.isfinite(raw_gain_init) or abs(raw_gain_init) >= 1:
            raise ValueError('raw_gain_init must lie strictly between -1 and 1')
        raw_encoder_drop_rate = _probability(raw_encoder_drop_rate, 'raw_encoder_drop_rate')
        fusion_drop_rate = _probability(fusion_drop_rate, 'fusion_drop_rate')
        self.patch_size=int(patch_size)
        self.hidden_dim=int(hidden_dim)
        self.num_heads=int(num_heads)
        self.head_dim=hidden_dim//num_heads
        self.center=patch_size//2
        self.token_attention=bool(token_attention)
        self.local_refinement=bool(local_refinement)
        self.context_enabled=False
        self.center_readout=False
        self.pool_readout=False
        self.multiplicative_gate=True
        self.norm_spa=nn.LayerNorm(hidden_dim)
        self.norm_spe=nn.LayerNorm(hidden_dim)
        bottleneck=max(hidden_dim//8,8)
        self.channel_mlp=nn.Sequential(nn.Linear(2*hidden_dim,bottleneck),nn.GELU(),nn.Linear(bottleneck,2*hidden_dim))
        self.channel_gate=nn.Parameter(torch.zeros(2,hidden_dim))
        self.spectral_gate=nn.Parameter(torch.zeros(hidden_dim))
        if self.token_attention:
            self.attention_norm=nn.LayerNorm(hidden_dim)
            self.query_dim=max(self.head_dim//2,4)
            self.attention_query=nn.Linear(2*hidden_dim,num_heads*self.query_dim,bias=False)
            self.attention_key=nn.Linear(hidden_dim,num_heads*self.query_dim,bias=False)
            self.attention_temperature=nn.Parameter(torch.full((num_heads,),math.log(9.0/2.0)))
            self.attention_gate=nn.Parameter(torch.zeros(hidden_dim))
        self.fusion_dropout=nn.Dropout2d(float(fusion_drop_rate))
        if self.local_refinement:
            with torch.random.fork_rng(devices=[]):
                self.local_conv=nn.Conv2d(hidden_dim,hidden_dim,3,padding=1,groups=hidden_dim)
                nn.init.dirac_(self.local_conv.weight,groups=hidden_dim)
                nn.init.zeros_(self.local_conv.bias)
            self.local_logit=nn.Parameter(torch.zeros(hidden_dim))

        with torch.random.fork_rng(devices=[]):
            self.raw_encoder=nn.Sequential(
                nn.Conv2d(self.raw_in_chans,width,1,bias=False),nn.BatchNorm2d(width),nn.GELU(),
                nn.Conv2d(width,width,3,padding=1,groups=width,bias=False),nn.BatchNorm2d(width),nn.GELU(),
                nn.Conv2d(width,width,1,bias=False),nn.BatchNorm2d(width),nn.GELU(),
                nn.Conv2d(width,width,3,padding=1,bias=False),nn.BatchNorm2d(width),nn.GELU(),
                nn.Conv2d(width,width,3,padding=1,bias=False),nn.BatchNorm2d(width),nn.GELU(),
                nn.Conv2d(width,hidden_dim,1),
            )
        self.raw_encoder_dropout=nn.Dropout2d(raw_encoder_drop_rate)
        self.raw_token_norm=nn.LayerNorm(hidden_dim)
        self.raw_spatial_query=nn.Linear(2*hidden_dim,hidden_dim,bias=False)
        self.raw_spatial_key=nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.raw_query_norm=nn.LayerNorm(hidden_dim)
        self.raw_norm=nn.LayerNorm(hidden_dim)
        self.raw_channel=nn.Sequential(nn.Linear(2*hidden_dim,bottleneck),nn.GELU(),nn.Linear(bottleneck,hidden_dim),nn.Sigmoid())
        self.raw_gate=nn.Parameter(torch.full((hidden_dim,),math.atanh(raw_gain_init)))

        with torch.random.fork_rng(devices=[]):
            self.raw_branch_query=nn.Linear(2*hidden_dim,hidden_dim,bias=False)
            nn.init.zeros_(self.raw_branch_query.weight)
            self.raw_branch_query._cfam_initialized=True

        with torch.random.fork_rng(devices=[]):
            self.raw_band_gate=nn.Sequential(nn.Linear(3*hidden_dim,max(hidden_dim//2,8)),nn.GELU(),nn.Linear(max(hidden_dim//2,8),self.raw_in_chans))
            trunc_normal_(self.raw_band_gate[0].weight,std=.02)
            nn.init.zeros_(self.raw_band_gate[0].bias)
            nn.init.zeros_(self.raw_band_gate[-1].weight)
            nn.init.zeros_(self.raw_band_gate[-1].bias)
            self.raw_band_gate[0]._cfam_initialized=True
            self.raw_band_gate[-1]._cfam_initialized=True


        # Actual Spa/Spe values guide raw tokens inside the fusion module.
        self.raw_cross_local=True
        with torch.random.fork_rng(devices=[]):
            self.raw_cross_qnorm=nn.LayerNorm(hidden_dim)
            self.raw_cross_spanorm=nn.LayerNorm(hidden_dim)
            self.raw_cross_spenorm=nn.LayerNorm(hidden_dim)
            self.raw_cross_query=nn.Linear(hidden_dim,hidden_dim,bias=False)
            self.raw_cross_key=nn.Linear(hidden_dim,hidden_dim,bias=False)
            self.raw_cross_value=nn.Linear(hidden_dim,hidden_dim,bias=False)
            self.raw_cross_project=nn.Linear(hidden_dim,hidden_dim,bias=False)
            for module in (self.raw_cross_query,self.raw_cross_key,self.raw_cross_value,self.raw_cross_project):
                trunc_normal_(module.weight,std=.02)
                module._cfam_initialized=True
            nn.init.zeros_(self.raw_cross_project.weight)

    def raw_cross_prior(self,height,width,device,dtype):
        count=height*width
        if self.raw_cross_local:
            rows,cols=torch.meshgrid(torch.arange(height,device=device),torch.arange(width,device=device),indexing="ij")
            positions=torch.stack((rows,cols),-1).reshape(count,2).to(dtype)
            spatial=-(positions[:,None]-positions[None,:]).square().sum(-1)/(2*self.spatial_prior_radius**2)
            spatial=spatial-spatial.logsumexp(-1,keepdim=True)
        else:
            spatial=torch.full((count,count),-math.log(count),device=device,dtype=dtype)
        return torch.cat((spatial,torch.zeros(count,1,device=device,dtype=dtype)),-1)

    def cross_attend_raw(self,features,spa_feature,spe_feature):
        b,d,height,width=features.shape
        count=height*width
        tokens=features.flatten(2).transpose(1,2)
        branch_tokens=torch.cat((self.raw_cross_spanorm(spa_feature.reshape(b,count,d)),self.raw_cross_spenorm(spe_feature)),1)
        heads=self.raw_cross_heads
        head_dim=d//heads
        query=self.raw_cross_query(self.raw_cross_qnorm(tokens)).reshape(b,count,heads,head_dim).transpose(1,2)
        keys=self.raw_cross_key(branch_tokens).reshape(b,count+1,heads,head_dim).transpose(1,2)
        values=self.raw_cross_value(branch_tokens).reshape(b,count+1,heads,head_dim).transpose(1,2)
        scores=_attention_logits(query,keys)/math.sqrt(head_dim)
        scores=scores+self.raw_cross_prior(height,width,features.device,scores.dtype)
        attended=(scores.softmax(-1).to(values.dtype)@values).transpose(1,2).reshape(b,count,d)
        output=tokens+self.raw_cross_project(attended)
        return output.transpose(1,2).reshape(b,d,height,width)

    def raw_context(self,raw_patch,residual,spa_feature,spe_feature):
        if raw_patch.ndim!=4 or raw_patch.shape[0]!=residual.shape[0] or raw_patch.shape[2:]!=(self.patch_size,self.patch_size) or raw_patch.shape[1]<1:
            raise ValueError("raw_patch must be [B,C,H,W] matching the configured spatial patch")
        if raw_patch.shape[1]!=self.raw_in_chans:
            b,c,h,w=raw_patch.shape
            # Compatibility fallback when a legacy checkpoint fixes raw bands.
            raw_patch=F.adaptive_avg_pool1d(raw_patch.permute(0,2,3,1).reshape(b*h*w,1,c),self.raw_in_chans).reshape(b,h,w,self.raw_in_chans).permute(0,3,1,2)
        band_descriptor=torch.cat((self.raw_query_norm(residual[:,self.center,self.center]),self.norm_spa(spa_feature.mean((1,2))),self.norm_spe(spe_feature[:,0])),-1)
        band_weights=2*self.raw_band_gate(band_descriptor).sigmoid()
        raw_patch=raw_patch*band_weights[:,:,None,None]
        features=raw_patch
        for layer in list(self.raw_encoder.children())[:-1]:
            features=layer(features)
        features=self.raw_encoder[-1](self.raw_encoder_dropout(features))
        features=self.cross_attend_raw(features,spa_feature,spe_feature)
        b,d,h,w=features.shape
        tokens=features.flatten(2).transpose(1,2)
        normalized=self.raw_token_norm(tokens)
        center=self.center*w+self.center
        query=self.raw_spatial_query(torch.cat((self.raw_query_norm(residual[:,self.center,self.center]),normalized[:,center]),-1)).unsqueeze(1)
        keys=self.raw_spatial_key(normalized)
        branch_descriptor=torch.cat((self.norm_spa(spa_feature.mean((1,2))),self.norm_spe(spe_feature[:,0])),-1)
        query=query+self.raw_branch_query(branch_descriptor).unsqueeze(1)
        per_head=d//self.raw_heads
        query=query.reshape(b,1,self.raw_heads,per_head).transpose(1,2)
        keys=keys.reshape(b,h*w,self.raw_heads,per_head).transpose(1,2)
        values=tokens.reshape(b,h*w,self.raw_heads,per_head).transpose(1,2)
        scores=_attention_logits(query,keys)/math.sqrt(per_head)
        positions=torch.arange(h,device=features.device,dtype=scores.dtype)-self.center
        distance=positions[:,None].square()+positions[None,:].square()
        weights=(scores-distance.reshape(1,1,h*w)/(2*self.raw_pool_radius**2)).softmax(-1).to(features.dtype)
        encoded=(weights@values).transpose(1,2).reshape(b,d)
        descriptors=torch.cat((self.raw_query_norm(residual[:,self.center,self.center]),self.raw_norm(encoded)),dim=-1)
        attention=2*self.raw_channel(descriptors)
        correction=self.raw_gate.tanh()*attention*encoded
        # Smooth sample-wise capacity bound, evaluated outside reduced precision.
        reference=_working_float(residual).square().mean((1,2,3)).clamp_min(1e-12)
        reference=reference*(self.raw_rms_limit**2)
        energy=_working_float(correction).square().mean(-1)
        factor=(reference/(reference+energy)).sqrt().to(correction.dtype)
        return (correction*factor[:,None]).unsqueeze(1)

    def attention_weights(self, features, spe_feature):
        b,h,w,d=features.shape
        tokens=features.reshape(b,h*w,d)
        norm=self.attention_norm(tokens)
        center=(h//2)*w+w//2
        query=torch.cat((norm[:,center:center+1],self.norm_spe(spe_feature)),dim=-1)
        q=self.attention_query(query).reshape(b,1,self.num_heads,self.query_dim).transpose(1,2)
        k=self.attention_key(norm).reshape(b,h*w,self.num_heads,self.query_dim).transpose(1,2)
        q,k=F.normalize(_working_float(q),dim=-1),F.normalize(_working_float(k),dim=-1)
        temperature=(1+11*self.attention_temperature.sigmoid()).view(1,self.num_heads,1,1)
        scores=_attention_logits(q,k)*temperature
        return scores.softmax(-1).to(features.dtype)

    def forward(self,x_residual,spa_feature,spe_feature,raw_patch=None):
        if x_residual.ndim!=4:
            raise ValueError("x_residual must be [B,H,W,D]")
        b,h,w,d=x_residual.shape
        if (h,w,d)!=(self.patch_size,self.patch_size,self.hidden_dim):
            raise ValueError("x_residual shape does not match configuration")
        if spa_feature.shape!=x_residual.shape or spe_feature.shape!=(b,1,d):
            raise ValueError("Spa/Spe fusion input shapes do not match configuration")
        residual=x_residual
        if self.local_refinement:
            local=self.local_conv(residual.permute(0,3,1,2)).permute(0,2,3,1)
            residual=residual+self.local_logit.sigmoid()*(local-residual)
        spa_global=F.adaptive_avg_pool2d(spa_feature.permute(0,3,1,2),1).squeeze(-1).permute(0,2,1)
        joint=torch.cat((self.norm_spa(spa_global),self.norm_spe(spe_feature)),-1)
        spe_logits,spa_logits=self.channel_mlp(joint).tanh().chunk(2,-1)
        attn_spe=spe_feature.softmax(-1)*(1+self.channel_gate[0].tanh()*spe_logits)
        attn_spa=spa_global.softmax(-1)*(1+self.channel_gate[1].tanh()*spa_logits)
        spa_cross=spa_feature*attn_spe.unsqueeze(1)
        spe_cross=spe_feature*attn_spa
        fused=spa_cross.clone()
        fused[:,self.center,self.center,:]=fused[:,self.center,self.center,:]+spe_cross[:,0,:]
        output=residual+fused
        output=output+(self.spectral_gate.tanh()*spe_cross).unsqueeze(1)
        if raw_patch is not None:
            output=output+self.raw_context(raw_patch,residual,spa_feature,spe_feature).unsqueeze(1)
        if self.token_attention:
            attn=self.attention_weights(residual,spe_feature)
            tokens=residual.reshape(b,h*w,d)
            values=tokens.reshape(b,h*w,self.num_heads,self.head_dim).transpose(1,2)
            context=(attn@values).transpose(1,2).reshape(b,1,d)
            correction=context-tokens.mean(1,keepdim=True)
            output=output+(self.attention_gate.tanh()*correction).unsqueeze(1)
        if self.fusion_dropout.p>0:
            output=self.fusion_dropout(output.permute(0,3,1,2)).permute(0,2,3,1)
        return output

    def pool(self,features):
        return features.mean((1,2))




class BasicSpaSpeBlock(nn.Module):
    """D20 DPE -> Spa / center Spe (optional BFS) -> V139 CFAM -> FFN."""

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
        fusion_type="center_cfam_residual",
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

        if fusion_type in ("center_cfam", "center_cfam_residual",
                            "center_cfam_center_controlled", "center_cfam_pool_controlled",
                            "center_cfam_regularized05", "center_cfam_regularized10",
                            "center_cfam_multiplicative", "center_cfam_multiplicative05"):
            # New fusion parameters must not change the baseline embedding,
            # Mamba/FFN/head initialization through RNG consumption.
            with torch.random.fork_rng(devices=[]):
                fusion_options = dict(
                    context_enabled=False,
                    center_readout=(fusion_type == "center_cfam_center_controlled"),
                    pool_readout=(fusion_type == "center_cfam_pool_controlled"),
                    fusion_drop_rate={
                        "center_cfam_regularized05": 0.05,
                        "center_cfam_regularized10": 0.1,
                        "center_cfam_multiplicative05": 0.05,
                    }.get(fusion_type, 0.0),
                    multiplicative_gate=True,
                )
                fusion_options.update(dict(cfam_options or {}))
                self.cross_fusion = CenterPairedCFAM(
                    patch_size=patch_size,
                    hidden_dim=hidden_dim,
                    num_heads=fusion_heads,
                    **fusion_options,
                )
                self.cross_fusion.apply(BasicSpaSpeMamba._init_weights)
                for module in self.cross_fusion.modules():
                    if isinstance(module, nn.Linear):
                        module._cfam_initialized = True
        elif fusion_type in ("center_cfam_channel", "center_cfam_center", "center_cfam_pool"):
            fusion_options = dict(
                spatial_residual=False,
                context_enabled=False,
                center_readout=(fusion_type == "center_cfam_center"),
                pool_readout=(fusion_type == "center_cfam_pool"),
            )
            fusion_options.update(dict(cfam_options or {}))
            self.cross_fusion = CenterPairedCFAM(
                patch_size=patch_size,
                hidden_dim=hidden_dim,
                num_heads=fusion_heads,
                **fusion_options,
            )
        elif fusion_type == "cas2f":
            if cfam_options:
                raise ValueError('cfam_options require a CFAM fusion_type')
            self.cross_fusion = CrossAttentionSpaSpeFusion(
                patch_size=patch_size,
                hidden_dim=hidden_dim,
            )
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

        # Configured fusion adds the shared x_dpe residual exactly once.
        fusion_inputs=dict(x_residual=x_dpe,spa_feature=spa_delta,spe_feature=spe_delta)
        if isinstance(self.cross_fusion,CenterPairedCFAM):
            fusion_inputs["raw_patch"]=raw_patch
        x=self.cross_fusion(**fusion_inputs)

        # Same residual FFN as the previous basic model.
        return x + self.drop_path(
            self.ffn(
                self.norm_ffn(x)
            )
        )


class BasicSpaSpeMamba(nn.Module):
    """Shared D20/V139 backbone with explicit module tuning dictionaries.

    Example: dpe_options={"radius": 1.5, "gain_init": 0.05};
    cfam_options={"cross_heads": 4, "raw_width_ratio": 4.0}.
    Raw width/bands and DPE head count change checkpoint tensor shapes.
    Legacy token attention and local-refinement switches change parameter
    keys. Gain, radius, dropout, RMS capacity and raw cross/pool head counts
    retain parameter shapes, but may change training/evaluation.
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
        fusion_type="center_cfam_residual",
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
        in_chans = _integer(in_chans, 'in_chans')
        num_classes = _integer(num_classes, 'num_classes')
        hidden_dim = _integer(dims[0], 'dims[0]')
        group_count = _integer(n_groups[0], 'n_groups[0]')
        if hidden_dim % group_count:
            raise ValueError('dims[0] must be divisible by n_groups[0]')
        drop_path_rate = _probability(drop_path_rate, 'drop_path_rate')
        cfam_options = dict(cfam_options or {})
        # Use physical input bands by default; an explicit override can restore
        # a legacy encoder's fixed band count when loading its checkpoint.
        if fusion_type != 'cas2f':
            cfam_options.setdefault('raw_in_chans', in_chans)

        self.patch_size = int(patch_size)
        self.in_chans = int(in_chans)

        padded_bands = (
            math.ceil(in_chans / group_count)
            * group_count
        )

        # --------------------------------------------------------------
        # Single shared initial embedding, just like the intended design.
        # There is NO independent raw-HSI spectral tokenizer anymore.
        # --------------------------------------------------------------
        self.pad = nn.ReplicationPad3d(
            (
                0, 0,
                0, 0,
                0, padded_bands - in_chans,
            )
        )

        self.group_emb = GroupedPixelEmbedding_mm(
            in_feature_map_size=patch_size,
            in_chans=padded_bands,
            embed_dim=hidden_dim,
            n_groups=group_count,
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
        if isinstance(module, nn.Linear) and not getattr(module, "_cfam_initialized", False):
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

        # --------------------------------------------------------------
        # Shared embedding only.
        # --------------------------------------------------------------
        raw_patch=x[:,0]
        x = self.pad(x).squeeze(1)

        x = self.group_emb(
            rearrange(
                x,
                "b c h w -> b h w c",
            )
        )

        for block in self.blocks:
            x = block(x,raw_patch=raw_patch)

        features = self.norm(x)
        fusion = self.blocks[-1].cross_fusion
        if isinstance(fusion, CenterPairedCFAM):
            return fusion.pool(features)
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
    "CenterPairedCFAM",
    "BasicSpaSpeBlock",
    "BasicSpaSpeMamba",
    "basic_spa_spe_mamba",
]

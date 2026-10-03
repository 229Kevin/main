"""DualMamba with scale-preserving, center-guided CFAM channel fusion.

Embedding, DPE, both Mamba branches, optional BFS, FFN and classifier are
preserved. CFAM keeps the CAS2F local path and learns bounded channel,
spectral and center-context corrections. ``fusion_type='cas2f'`` remains
an unchanged ablation; existing CFAM configuration names are supported.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_

from .rs_mamba_ss import GroupedPixelEmbedding_mm, Mlp, OSSM


class BidirectionalFusionState(nn.Module):
    """BFS group fusion over aligned forward/backward recurrent states."""

    def __init__(self, d_inner, d_state, groups=8):
        super().__init__()
        self.d_inner = int(d_inner)
        self.d_state = int(d_state)
        self.state_width = self.d_inner * self.d_state
        self.groups = int(groups)

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


class DualMambaDPE(nn.Module):
    """DualMamba-style dynamic positional embedding (3x3 depthwise Conv2d)."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.dwconv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
        )

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected [B,H,W,C], got {tuple(x.shape)}")

        pos = self.dwconv(x.permute(0, 3, 1, 2))
        return x + pos.permute(0, 2, 3, 1)


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

        requested_d_conv = int(d_conv)

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
    """Scale-preserving channel fusion with a center-conditioned context path.

    The parameter-free CAS2F interaction remains the local skip path. Learned
    channel corrections, direct spectral broadcast and center-guided context
    have bounded gates initialized to zero, so the initial output is exactly
    CAS2F. Spectral and spatial sources no longer compete in one token softmax.
    ``spatial_residual`` is retained for existing config compatibility; both
    legacy CFAM names now use the scale-preserving skip instead of raw Spa.
    """

    def __init__(self, patch_size, hidden_dim, num_heads=8, spatial_residual=False,
                 context_enabled=True, center_readout=False, pool_readout=False, fusion_drop_rate=0.0, multiplicative_gate=False):
        super().__init__()
        if patch_size <= 0 or patch_size % 2 == 0:
            raise ValueError("patch_size must be positive and odd")
        if num_heads <= 0 or hidden_dim <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.patch_size = int(patch_size)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = hidden_dim // num_heads
        self.center = (patch_size // 2) * patch_size + patch_size // 2
        self.spatial_residual = bool(spatial_residual)
        self.context_enabled = bool(context_enabled)
        self.center_readout = bool(center_readout)
        self.pool_readout = bool(pool_readout)
        self.fusion_dropout = nn.Dropout2d(float(fusion_drop_rate))
        self.multiplicative_gate = bool(multiplicative_gate)
        if self.pool_readout:
            self.readout_logit = nn.Parameter(torch.tensor(math.log(1.0 / 3.0)))
        if self.center_readout:
            self.center_gate = nn.Parameter(torch.zeros(hidden_dim))
        self.norm_spa = nn.LayerNorm(hidden_dim)
        self.norm_spe = nn.LayerNorm(hidden_dim)
        bottleneck = max(hidden_dim // 8, 8)
        self.channel_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, 2 * hidden_dim),
        )
        self.channel_gate = nn.Parameter(torch.zeros(2, hidden_dim))
        self.spectral_gate = nn.Parameter(torch.zeros(hidden_dim))
        if self.context_enabled:
            self.norm_dpe = nn.LayerNorm(hidden_dim)
            self.spectral_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.context_gate = nn.Parameter(torch.zeros(hidden_dim))

    def _heads(self, x):
        b, n, _ = x.shape
        return x.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x_residual, spa_feature, spe_feature):
        if x_residual.ndim != 4:
            raise ValueError("x_residual must be [B,H,W,D]")
        b, h, w, d = x_residual.shape
        if (h, w, d) != (self.patch_size, self.patch_size, self.hidden_dim):
            raise ValueError("x_residual shape does not match configuration")
        if spa_feature.shape != x_residual.shape:
            raise ValueError("spa_feature must match x_residual")
        if spe_feature.shape != (b, 1, d):
            raise ValueError("spe_feature must be [B,1,D]")

        # Preserve the amplitude of both Mamba outputs on the local skip.
        spa_global = F.adaptive_avg_pool2d(
            spa_feature.permute(0, 3, 1, 2), 1
        ).squeeze(-1).permute(0, 2, 1)
        joint = torch.cat((self.norm_spa(spa_global), self.norm_spe(spe_feature)), -1)
        spe_logits, spa_logits = self.channel_mlp(joint).tanh().chunk(2, -1)
        if self.multiplicative_gate:
            # Positive independent gains permit multiple channels to rise
            # together instead of making every correction compete in softmax.
            attn_spe = spe_feature.softmax(-1) * (1 + self.channel_gate[0].tanh() * spe_logits)
            attn_spa = spa_global.softmax(-1) * (1 + self.channel_gate[1].tanh() * spa_logits)
        else:
            attn_spe = (spe_feature + self.channel_gate[0].tanh() * spe_logits).softmax(-1)
            attn_spa = (spa_global + self.channel_gate[1].tanh() * spa_logits).softmax(-1)
        spa_cross = spa_feature * attn_spe.unsqueeze(1)
        spe_cross = spe_feature * attn_spa
        fused = spa_cross.clone()
        center = self.patch_size // 2
        fused[:, center, center, :] = fused[:, center, center, :] + spe_cross[:, 0, :]
        output = x_residual + fused

        # A separate branch gate avoids 1-versus-HW token competition and
        # makes the spectral readout visible beyond the single center pixel.
        output = output + (self.spectral_gate.tanh() * spe_cross).unsqueeze(1)
        if self.center_readout:
            # The task predicts the center label. Broadcast a centered readout
            # correction without replacing the per-position local skip.
            center_value = output[:, center:center + 1, center:center + 1, :]
            output = output + self.center_gate.tanh() * (
                center_value - output.mean((1, 2), keepdim=True)
            )
        if self.context_enabled:
            tokens = x_residual.reshape(b, h * w, d)
            e = self.norm_dpe(tokens)
            query = e[:, self.center:self.center + 1] + self.spectral_query(
                self.norm_spe(spe_feature)
            )
            q, k = self._heads(query), self._heads(e)
            scores = (q @ k.transpose(-2, -1)) * self.head_dim ** -0.5
            attn = scores.float().softmax(-1).to(tokens.dtype)
            # Raw values retain feature amplitude; a centered correction
            # changes the readout without duplicating the residual mean.
            context = (attn @ self._heads(tokens)).transpose(1, 2).reshape(b, 1, d)
            correction = context - tokens.mean(1, keepdim=True)
            output = output + (self.context_gate.tanh() * correction).unsqueeze(1)
        if self.fusion_dropout.p > 0:
            # Drop whole channels consistently at every spatial position.
            output = self.fusion_dropout(output.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return output

    def pool(self, features):
        """Pool normalized block outputs for a center-pixel class label.

        An optional convex center/global readout acts after the FFN and
        final normalization, so it cannot distort the local fusion path.
        """
        if features.ndim != 4 or tuple(features.shape[1:]) != (
            self.patch_size, self.patch_size, self.hidden_dim
        ):
            raise ValueError("readout features must match [B,H,W,D] configuration")
        global_feature = features.mean((1, 2))
        if not self.pool_readout:
            return global_feature
        center = self.patch_size // 2
        center_feature = features[:, center, center, :]
        alpha = self.readout_logit.sigmoid()
        return (1 - alpha) * global_feature + alpha * center_feature


class BasicSpaSpeBlock(nn.Module):
    """Shared DPE -> Spa / center Spe (optional BFS) -> CAS2F -> FFN."""

    def __init__(
        self,
        patch_size,
        hidden_dim,
        drop_path=0.0,
        mlp_ratio=4.0,
        mlp_drop_rate=0.0,
        spatial_forward_type="v3",
        spatial_ssm_conv=3,
        spatial_ssm_init="v0",
        spectral_forward_type="v3",
        spectral_ssm_conv=3,
        spectral_ssm_init="v0",
        bfs_groups=8,
        bfs_enabled=True,
        fusion_type="center_cfam",
        fusion_heads=8,
        **mamba_kwargs,
    ):
        super().__init__()

        self.patch_size = int(patch_size)

        # --------------------------------------------------------------
        # One shared DPE. Both branches consume x_dpe.
        # --------------------------------------------------------------
        self.dpe = DualMambaDPE(hidden_dim)

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
                self.cross_fusion = CenterPairedCFAM(
                    patch_size=patch_size,
                    hidden_dim=hidden_dim,
                    num_heads=fusion_heads,
                    context_enabled=False,
                    center_readout=(fusion_type == "center_cfam_center_controlled"),
                    pool_readout=(fusion_type == "center_cfam_pool_controlled"),
                    fusion_drop_rate={
                        "center_cfam_regularized05": 0.05,
                        "center_cfam_regularized10": 0.1,
                        "center_cfam_multiplicative05": 0.05,
                    }.get(fusion_type, 0.0),
                    multiplicative_gate=(fusion_type in ("center_cfam", "center_cfam_residual", "center_cfam_multiplicative", "center_cfam_multiplicative05")),
                )
                self.cross_fusion.apply(BasicSpaSpeMamba._init_weights)
                for module in self.cross_fusion.modules():
                    if isinstance(module, nn.Linear):
                        module._cfam_initialized = True
        elif fusion_type in ("center_cfam_channel", "center_cfam_center", "center_cfam_pool"):
            self.cross_fusion = CenterPairedCFAM(
                patch_size=patch_size,
                hidden_dim=hidden_dim,
                num_heads=fusion_heads,
                spatial_residual=(fusion_type == "center_cfam_residual"),
                context_enabled=(fusion_type not in ("center_cfam_channel", "center_cfam_center", "center_cfam_pool")),
                center_readout=(fusion_type == "center_cfam_center"),
                pool_readout=(fusion_type == "center_cfam_pool"),
            )
        elif fusion_type == "cas2f":
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

    def forward(self, x):
        # --------------------------------------------------------------
        # Shared representation:
        # embedding has already been computed before entering the block.
        # DPE is applied once and shared by Spa and Spe.
        # --------------------------------------------------------------
        x_dpe = self.dpe(x)

        # Spatial OSSM on the full shared feature map.
        spa_delta = self.spatial(x_dpe)

        # Spectral OSSM+BFS on center-pixel embedding along feature dimension.
        spe_delta = self.spectral(x_dpe)  # [B,1,D]

        # Configured fusion adds the shared x_dpe residual exactly once.
        x = self.cross_fusion(
            x_residual=x_dpe,
            spa_feature=spa_delta,
            spe_feature=spe_delta,
        )

        # Same residual FFN as the previous basic model.
        return x + self.drop_path(
            self.ffn(
                self.norm_ffn(x)
            )
        )


class BasicSpaSpeMamba(nn.Module):
    """CAS2MM-style backbone with fuller OSSM blocks and optional BFS."""

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
        spatial_forward_type="v3",
        spatial_ssm_conv=3,
        spatial_ssm_init="v0",

        # Spectral OSSM settings.
        spectral_forward_type="v3",
        spectral_ssm_conv=3,
        spectral_ssm_init="v0",
        bfs_groups=8,
        bfs_enabled=True,
        fusion_type="center_cfam",
        fusion_heads=8,

        **kwargs,
    ):
        super().__init__()

        # Legacy fields are retained in the signature for JSON compatibility.
        del kwargs, use_gate, ssm_conv

        if (
            len(n_groups) != 1
            or len(dims) != 1
            or len(depths) != 1
        ):
            raise ValueError(
                "BasicSpaSpeMamba currently supports one encoder stage"
            )

        if patch_size <= 0 or patch_size % 2 == 0:
            raise ValueError(
                "patch_size must be a positive odd integer"
            )

        if depths[0] < 1:
            raise ValueError(
                "depths[0] must be positive"
            )

        self.patch_size = int(patch_size)
        self.in_chans = int(in_chans)

        hidden_dim = int(dims[0])
        group_count = int(n_groups[0])

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

        drop_paths = torch.linspace(
            0,
            drop_path_rate,
            depths[0],
        ).tolist()

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

                    **mamba_kwargs,
                )
                for index in range(depths[0])
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
        x = self.pad(x).squeeze(1)

        x = self.group_emb(
            rearrange(
                x,
                "b c h w -> b h w c",
            )
        )

        for block in self.blocks:
            x = block(x)

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

        sample = torch.randn(
            (1, *shape),
            device=device,
        )

        with torch.no_grad():
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

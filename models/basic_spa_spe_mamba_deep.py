"""Shared-center CAS2F model with a controlled spectral BFS ablation.

Set ``bfs_enabled=False`` to remove hidden-state fusion: each spectral
direction is read out from its own state before the existing weighted output
merge. No BFS parameters are created in this mode. ``True`` preserves the
original BFS path and checkpoint layout. All other processing is unchanged.

Purpose
-------
This version tests whether restoring the components removed by DualMamba's
lightweight OSSM improves HSI classification.

Relative to ``basic_spa_spe_mamba_sharedcenter_cas2f_bfs.py``:

Kept unchanged:
    - GroupedPixelEmbedding_mm
    - shared DPE
    - center-pixel spectral input
    - spatial/spectral scan directions used by the current model
    - BFS hidden-state fusion
    - CAS2F-style interaction
    - residual FFN and classifier

Restored toward a fuller OSSM/Mamba block:
    - z/gating branch is ENABLED (forward_type='v3', not 'v3noz')
    - spatial internal depthwise convolution is ENABLED (kernel=3)
    - spectral internal depthwise sequence convolution is ENABLED
      with Conv1d(kernel=3), the 1-D analogue required by a [B,L,1]
      spectral sequence
    - OSSM output order is kept close to standard Mamba:
          scan/merge -> out_norm -> z gate -> out_proj

Spectral branch revision
------------------------
SpectralOSSMWithBFS no longer forces d_model=1. The center embedding is
tokenized into [B, token_num, token_dim], and d_model = token_dim, so
d_inner = ssm_ratio * token_dim is a usable width instead of 2. BFS is
residual: fused = forward + backward + fusion(forward + backward), so the
two directions keep their own identity. The center pixel can be optionally
pooled over a 3x3 window. Everything else (CAS2F fusion, spatial branch,
DPE, embedding, FFN, classifier) is unchanged.

CAS2F gating revision
---------------------
CrossAttentionSpaSpeFusion previously used Softmax(dim=-1) along the D axis
for both the spectral and spatial attention weights. Since D is the channel
axis, the softmax output summed to 1 across channels, which (a) had no
spatial selectivity at all (weights were broadcast across H, W), and (b)
shrank the modulations to ~1/D of their inputs, so the spatial and spectral
branches barely contributed to the fused feature early in training. The
revision replaces both softmaxes with sigmoid gates:

    gate_spe = sigmoid(spe_feature)                # per-channel gate
    gate_spa = sigmoid(avgpool(spa_feature))       # per-channel gate
    spa_cross = spa_feature * gate_spe
    spe_cross = spe_feature * gate_spa

This keeps the gate values in (0, 1) without a sum-to-one constraint, so
both branches keep their natural magnitudes. Initialization is unbiased
(sigmoid(0) = 0.5). Everything else in the fusion path is unchanged.

FFN spatial refinement (new)
----------------------------
A lightweight ConvNeXt-style spatial refinement is inserted between CAS2F
and the residual FFN:

    fused -> [B,H,W,D]
          -> depthwise 3x3 -> GroupNorm(1) -> SiLU -> 1x1
          -> residual add
          -> pre-LN FFN (unchanged)
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
    sequence [B,L,C]. Therefore this class:

    1) inherits the complete OSSM projection / SSM / gate parameters;
    2) keeps the z branch enabled;
    3) uses a depthwise Conv1d over the token sequence as the sequence-domain
       equivalent of OSSM's internal depthwise convolution;
    4) exposes recurrent states so BFS can still be inserted.

    The scan remains bidirectional (K=2). Revision: d_model is no longer
    forced to 1, so d_inner = ssm_ratio * d_model is usable; BFS is residual
    by default so the two directions do not collapse into a single state.
    """

    def __init__(
        self,
        d_model=16,
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
        bfs_residual=True,
    ):
        if not isinstance(bfs_enabled, bool):
            raise ValueError("bfs_enabled must be a boolean")
        if not isinstance(bfs_residual, bool):
            raise ValueError("bfs_residual must be a boolean")
        if int(d_model) < 1:
            raise ValueError("d_model must be an integer >= 1")

        requested_d_conv = int(d_conv)
        if requested_d_conv < 0:
            raise ValueError("spectral d_conv must be >= 0")
        if requested_d_conv > 1 and requested_d_conv % 2 == 0:
            raise ValueError(
                "spectral d_conv must be 0, 1, or a positive odd kernel"
            )

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
        self.bfs_residual = bfs_residual

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
        """Spectral processing with optional BFS; retain the legacy method name.

        sequence: [B, L, d_model], where d_model is the token dimension.
        """

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
        #    length == token_num (typically 4~8), so this loop is cheap.
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
        #    state for its C readout.
        # --------------------------------------------------------------
        forward_state = states[:, 0]
        backward_state = torch.flip(
            states[:, 1],
            dims=[-1],
        )

        if self.bfs_enabled:
            combined = forward_state + backward_state
            fused = self.state_fusion(combined)
            if self.bfs_residual:
                forward_state = forward_state + fused
                backward_state = backward_state + fused
            else:
                forward_state = fused
                backward_state = fused

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

    Center extraction (optionally pooled over a 3x3 window):
        [B,D]

    Tokenized spectral sequence:
        [B,D] -> [B, token_num, token_dim], d_model = token_dim

    d_model is token_dim (hidden_dim // token_num), no longer 1, so
    d_inner = ssm_ratio * token_dim is a usable SSM width. Output shape
    stays [B,1,D] so the CAS2F fusion is unchanged.
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
        token_num=4,
        center_pool_size=3,
    ):
        super().__init__()

        self.patch_size = int(patch_size)
        self.hidden_dim = int(hidden_dim)
        self.center = self.patch_size // 2

        if hidden_dim % token_num != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by token_num={token_num}"
            )
        self.token_num = int(token_num)
        self.token_dim = hidden_dim // token_num

        if center_pool_size < 1 or center_pool_size % 2 == 0:
            raise ValueError("center_pool_size must be a positive odd integer")
        if center_pool_size > self.patch_size:
            raise ValueError("center_pool_size cannot exceed patch_size")
        self.center_pool_size = int(center_pool_size)

        # This follows DualMamba: normalize the center embedding over D.
        self.norm_spe = nn.LayerNorm(self.hidden_dim)

        # Spectral OSSM operates with d_model = token_dim (not 1).
        self.mamba = SpectralOSSMWithBFS(
            d_model=self.token_dim,
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
        # Center extraction. Optional 3x3 mean for noise robustness.
        # --------------------------------------------------------------
        c = self.center
        if self.center_pool_size > 1:
            half = self.center_pool_size // 2
            center_feature = x_dpe[
                :,
                c - half : c + half + 1,
                c - half : c + half + 1,
                :,
            ].mean(dim=(1, 2))  # [B,D]
        else:
            center_feature = x_dpe[:, c, c, :]  # [B,D]

        # LayerNorm over D.
        center_feature = self.norm_spe(center_feature.unsqueeze(1))  # [B,1,D]

        # Tokenize: [B,1,D] -> [B, token_num, token_dim].
        sequence = center_feature.reshape(
            batch, self.token_num, self.token_dim
        ).contiguous()

        # Spectral SSM + optional BFS: [B, token_num, token_dim].
        spectral_output = self.mamba.forward_bfs(sequence)

        # Flatten tokens back to [B,1,D] for CAS2F.
        spectral_output = spectral_output.reshape(
            batch, 1, self.hidden_dim
        ).contiguous()

        return spectral_output


# Backward-compatible alias.
SpectralBiMamba = SpectralCenterMamba


class CrossAttentionSpaSpeFusion(nn.Module):
    """CAS2F-style Spa/Spe interaction with sigmoid gating.

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
        gate_spe = sigmoid(G_spe)              # per-channel gate, [B,1,1,D]
        gate_spa = sigmoid(AvgPool(G_spa))     # per-channel gate, [B,1,1,D]

        G_spa <- gate_spe * G_spa
        G_spe <- gate_spa * G_spe

        inject G_spe into the center pixel of G_spa
        add x_residual

    Previous version used Softmax(dim=-1) along the channel axis. Since the
    channel axis has length D, the softmax outputs summed to 1, which shrank
    the modulations to about 1/D of their inputs and gave no spatial
    selectivity at all. Sigmoid gates keep values in (0, 1) without a
    sum-to-one constraint, so both branches keep their natural magnitudes.
    """

    def __init__(self, patch_size, hidden_dim):
        super().__init__()

        self.patch_size = int(patch_size)
        self.hidden_dim = int(hidden_dim)
        self.center = self.patch_size // 2

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

        # Spectral gate: [B,1,D] -> [B,1,1,D], values in (0, 1).
        gate_spe = torch.sigmoid(
            spe_feature
        ).unsqueeze(1)

        # Spatial gate:
        # [B,H,W,D] -> [B,D,H,W] -> [B,D,1,1]
        # -> [B,D,1] -> [B,1,D] -> sigmoid
        spa_global = self.spa_pool(
            spa_feature.permute(0, 3, 1, 2)
        )
        spa_global = spa_global.squeeze(-1).permute(
            0, 2, 1
        )

        gate_spa = torch.sigmoid(
            spa_global
        )  # [B,1,D]

        # Cross modulation.
        spa_cross = spa_feature * gate_spe
        spe_cross = spe_feature * gate_spa

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


class BasicSpaSpeBlock(nn.Module):
    """Shared DPE -> Spa / center Spe (optional BFS) -> CAS2F -> spatial refine -> FFN.

    Spatial refinement (ConvNeXt-style, inserted between CAS2F and the FFN):

        fused -> [B,H,W,D]
              -> depthwise 3x3 -> GroupNorm(1) -> SiLU -> 1x1
              -> residual add
              -> pre-LN FFN (unchanged)

    The FFN definition itself is untouched.
    """

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
        spectral_token_num=4,
        spectral_center_pool=3,
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

        # Center-pixel spectral branch with a usable d_model and optional BFS.
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
            token_num=spectral_token_num,
            center_pool_size=spectral_center_pool,
        )

        self.cross_fusion = CrossAttentionSpaSpeFusion(
            patch_size=patch_size,
            hidden_dim=hidden_dim,
        )

        # --------------------------------------------------------------
        # Lightweight ConvNeXt-style spatial refinement, applied between
        # CAS2F and the residual FFN.
        # --------------------------------------------------------------
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                groups=hidden_dim,
                bias=True,
            ),
            nn.GroupNorm(1, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=True),
        )

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

        # Spectral OSSM+BFS on tokenized center embedding.
        spe_delta = self.spectral(x_dpe)  # [B,1,D]

        # CAS2F-style interaction + residual x_dpe.
        x = self.cross_fusion(
            x_residual=x_dpe,
            spa_feature=spa_delta,
            spe_feature=spe_delta,
        )

        # --------------------------------------------------------------
        # Spatial refinement (residual) before the pointwise FFN.
        # [B,H,W,D] -> [B,D,H,W] -> refine -> [B,H,W,D]
        # --------------------------------------------------------------
        x = x + self.spatial_refine(
            x.permute(0, 3, 1, 2).contiguous()
        ).permute(0, 2, 3, 1).contiguous()

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

        # Spectral tokenization (new).
        spectral_token_num=4,
        spectral_center_pool=3,

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
                    spectral_token_num=spectral_token_num,
                    spectral_center_pool=spectral_center_pool,

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
        if isinstance(module, nn.Linear):
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

        return self.norm(x).mean(
            dim=(1, 2)
        )

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
    "BasicSpaSpeBlock",
    "BasicSpaSpeMamba",
    "basic_spa_spe_mamba",
]
"""PatchEmbedding + optional PSA (HyPyraMamba) or DPE + DualMamba + CAS2F.

This is the merged backbone that keeps every module validated so far:

    - shared PatchEmbedding (Conv1x1 -> GroupNorm -> SiLU)
    - shared preprocessing: PSA (psa_enabled=True) OR DualMambaDPE (False)
    - spatial OSSM with z-gate and internal depthwise Conv2d
    - center-pixel spectral OSSM with tokenization and residual BFS
    - CAS2F-style cross fusion with sigmoid gating (default), or mean-concat
    - ConvNeXt-style spatial refinement before the FFN (CAS2F path only)
    - pre-LN residual FFN, optional AEDC, final LayerNorm, classifier

PSA (single-scale ablation)
--------------------------
Keep only the original-resolution path (7x7 with the current presets),
with one initial attention, two refinement layers and four heads.
No pooled 3x3/1x1 paths or interpolation remain. The existing connection
is unchanged: shared = x + PSA(x), consumed by both Mamba branches and
used as the fusion residual.

AEDC (optional)
---------------
HyPyraMamba DynamicConvBlock is reproduced verbatim: GAP-conditioned
softmax weights mix four independent depthwise 3x3 experts, followed by
GroupNorm(1), SiLU and Dropout(0.1). It receives the completed spatial
fusion block output before final LayerNorm and global readout. No extra
residual or AvgPool2d is inserted. Requires a spatial fusion (cas2f).

Spectral branch
---------------
SpectralOSSMWithBFS no longer forces d_model=1. The center embedding is
tokenized into [B, token_num, token_dim] with d_model = token_dim, so
d_inner = ssm_ratio * token_dim. BFS is residual by default:
    fused = forward + backward + fusion(forward + backward).
The center pixel is optionally pooled over a 3x3 window.

CAS2F fusion
------------
CrossAttentionSpaSpeFusion uses sigmoid gates instead of softmax along D.
The old softmax summed to 1 across channels and shrank the modulations to
about 1/D of their inputs; the sigmoid version keeps both branches at
their natural magnitudes and is unbiased at initialization (sigmoid(0)=0.5).

Spatial refinement
------------------
A ConvNeXt-style block is inserted between the fusion and the FFN:
    depthwise 3x3 -> GroupNorm(1) -> SiLU -> 1x1
with a residual add. It is skipped automatically when the fusion returns
a vector (mean_concat path).

Register as ``mamba_shared_psa`` (psa_enabled=True) or
``mamba_shared_dpe`` (psa_enabled=False). New checkpoints must be trained.
"""

import copy
import math
from numbers import Integral

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_

from .rs_mamba_ss import Mlp, OSSM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _probability(value, name):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError(f'{name} must lie in [0, 1)')
    return value


# ---------------------------------------------------------------------------
# BFS
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# DPE (used when psa_enabled=False)
# ---------------------------------------------------------------------------

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

    def forward(self, x, raw_patch=None):
        del raw_patch
        if x.ndim != 4:
            raise ValueError(f"Expected [B,H,W,C], got {tuple(x.shape)}")

        pos = self.dwconv(x.permute(0, 3, 1, 2))
        return x + pos.permute(0, 2, 3, 1)


# ---------------------------------------------------------------------------
# PSA (used when psa_enabled=True)
# ---------------------------------------------------------------------------

class PyramidAttention(nn.Module):
    """Single-scale QK attention with QKV 3x3 dwconv (dilation=2)."""

    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, dilation=2,
            padding=2, groups=dim * 3, bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class PyramidRefinedChannelAttention(nn.Module):
    """PSA ablation retaining only the original-resolution path (7x7).

    Keep one initial attention and two refinement layers. No spatial
    downsampling, interpolation, or multi-scale concatenation is used.
    """

    def __init__(self, dim, num_heads, bias, num_scales=1, num_layers=2):
        super().__init__()
        if num_scales != 1:
            raise ValueError('This PSA version supports only the original-resolution scale')
        self.num_heads = num_heads
        # Retained for compatibility with the original single-scale reference.
        # As in HyPyraMamba, this outer temperature is not used in forward.
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.attention_modules = nn.ModuleList([
            PyramidAttention(dim, num_heads, bias),
        ])
        self.attention_layers = nn.ModuleList([
            nn.ModuleList([PyramidAttention(dim, num_heads, bias) for _ in range(num_layers)]),
        ])
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        output = self.attention_modules[0](x)
        for layer in self.attention_layers[0]:
            output = layer(output)
        return self.project_out(output)


# AEDC: verbatim DynamicConvBlock from HyPyraMamba/HypraMamba/model.py.
# Expert kernels all have the same 3x3 receptive field, with independent weights.
class DynamicConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, num_experts=4, reduction=4, dropout=0.1):
        super(DynamicConvBlock, self).__init__()
        self.channels = channels
        self.num_experts = num_experts
        self.kernel_size = kernel_size

        # Generate dynamic weights using a lightweight attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, num_experts, kernel_size=1),
            nn.Softmax(dim=1)
        )

        # Define a set of expert convolutions
        self.convs = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=channels)
            for _ in range(num_experts)
        ])

        self.norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

    def forward(self, x):
        B, C, H, W = x.shape

        # Get the attention weights for each expert
        attention_weights = self.attention(x).view(B, self.num_experts, 1, 1, 1)

        # Apply each expert convolution
        outputs = [conv(x).unsqueeze(1) for conv in self.convs]
        outputs = torch.cat(outputs, dim=1)  # Shape: (B, num_experts, C, H, W)

        # Weighted sum of the outputs from experts
        x = (outputs * attention_weights).sum(dim=1)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


# ---------------------------------------------------------------------------
# Spatial branch
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Spectral branch: tokenized OSSM with residual BFS
# ---------------------------------------------------------------------------

class SpectralOSSMWithBFS(OSSM):
    """Spectral OSSM with tokenized input and optional residual BFS.

    d_model is the token dimension (hidden_dim // token_num), not 1, so
    d_inner = ssm_ratio * d_model is a usable width. BFS is residual by
    default: the two scan directions keep their own identity and receive
    a shared group-fused increment.
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

        requested_d_conv = _integer(d_conv, 'spectral d_conv', minimum=0)
        if requested_d_conv > 1 and requested_d_conv % 2 == 0:
            raise ValueError('spectral d_conv must be 0, 1, or a positive odd kernel')

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
            math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)
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

        self.direction_logits = nn.Parameter(torch.zeros(2))

    def forward_bfs(self, sequence):
        if sequence.ndim != 3:
            raise ValueError(
                f"Expected spectral sequence [B,L,D], got {tuple(sequence.shape)}"
            )
        if sequence.shape[-1] != self.d_model_bfs:
            raise ValueError(
                f"Expected d_model={self.d_model_bfs}, got {sequence.shape[-1]}"
            )

        batch, length, _ = sequence.shape

        # 1) Full OSSM input projection.
        projected = self.in_proj(sequence)
        if self.disable_z:
            x_content = projected
            z = None
        else:
            x_content, z = projected.chunk(2, dim=-1)
            if not self.disable_z_act:
                z = self.act(z)

        # 2) Depthwise Conv1d over the token sequence.
        x_content = x_content.transpose(1, 2).contiguous()
        x_content = self.spectral_conv1d(x_content)
        x_content = self.act(x_content)

        # 3) Bidirectional scan.
        u = torch.stack(
            [x_content, torch.flip(x_content, dims=[-1])],
            dim=1,
        )  # [B,2,d_inner,L]

        # 4) OSSM x/dt/B/C projections.
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", u, self.x_proj_weight)
        dt, state_b, state_c = torch.split(
            x_dbl,
            [self.dt_rank_bfs, self.d_state_bfs, self.d_state_bfs],
            dim=2,
        )
        raw_delta = torch.einsum("b k r l, k d r -> b k d l", dt, self.dt_projs_weight)
        delta = F.softplus(
            raw_delta.float() + self.dt_projs_bias.float()[None, :, :, None]
        )
        u_float = u.float()
        a = -torch.exp(self.A_logs.float()).view(
            self.num_directions_bfs, self.d_inner_bfs, self.d_state_bfs
        )

        # 5) Explicit SSM evolution (length is token_num, small).
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
            decay = torch.exp(delta_t.unsqueeze(-1) * a.unsqueeze(0))
            state = (
                decay * state
                + delta_t.unsqueeze(-1)
                * state_b[..., index].float().unsqueeze(2)
                * input_t.unsqueeze(-1)
            )
            state_steps.append(state)
        states = torch.stack(state_steps, dim=-1)

        # 6) Align directions and apply residual BFS.
        forward_state = states[:, 0]
        backward_state = torch.flip(states[:, 1], dims=[-1])

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
        backward_c = torch.flip(state_c[:, 1].float(), dims=[-1])
        forward_u = u_float[:, 0]
        backward_u = torch.flip(u_float[:, 1], dims=[-1])

        skip = self.Ds.float().view(self.num_directions_bfs, self.d_inner_bfs)

        forward_output = (
            (forward_state * forward_c.unsqueeze(1)).sum(dim=2)
            + skip[0][None, :, None] * forward_u
        )
        backward_output = (
            (backward_state * backward_c.unsqueeze(1)).sum(dim=2)
            + skip[1][None, :, None] * backward_u
        )

        forward_output = forward_output.transpose(1, 2).contiguous()
        backward_output = backward_output.transpose(1, 2).contiguous()

        # 7) Merge directions, then out_norm, gate, out_proj.
        direction_weights = self.direction_logits.softmax(dim=0)
        y = (
            direction_weights[0] * forward_output
            + direction_weights[1] * backward_output
        )
        y = self.out_norm(y)
        if z is not None:
            y = y * z
        y = self.out_proj(y)
        return self.dropout(y)


class SpectralCenterMamba(nn.Module):
    """Center-pixel spectral branch with tokenized OSSM and optional BFS.

    Center extraction: [B,H,W,D] -> [B,D] (optionally 3x3 mean).
    Tokenize: [B,D] -> [B, token_num, token_dim], d_model = token_dim.
    Output: [B,1,D] (kept compatible with the fusion interface).
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

        self.norm_spe = nn.LayerNorm(self.hidden_dim)

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
        if (height, width) != (self.patch_size, self.patch_size):
            raise ValueError("Spatial patch does not match configured patch_size")
        if channels != self.hidden_dim:
            raise ValueError("Feature dimension does not match configured hidden_dim")

        c = self.center
        if self.center_pool_size > 1:
            half = self.center_pool_size // 2
            center_feature = x_dpe[
                :, c - half:c + half + 1, c - half:c + half + 1, :
            ].mean(dim=(1, 2))
        else:
            center_feature = x_dpe[:, c, c, :]

        center_feature = self.norm_spe(center_feature.unsqueeze(1))  # [B,1,D]
        sequence = center_feature.reshape(
            batch, self.token_num, self.token_dim
        ).contiguous()

        spectral_output = self.mamba.forward_bfs(sequence)

        spectral_output = spectral_output.reshape(
            batch, 1, self.hidden_dim
        ).contiguous()
        return spectral_output


SpectralBiMamba = SpectralCenterMamba


# ---------------------------------------------------------------------------
# Fusion modules
# ---------------------------------------------------------------------------

class CrossAttentionSpaSpeFusion(nn.Module):
    """CAS2F-style Spa/Spe interaction with sigmoid gating.

    gate_spe = sigmoid(spe_feature)               # [B,1,1,D]
    gate_spa = sigmoid(avgpool(spa_feature))      # [B,1,1,D]
    spa_cross = spa_feature * gate_spe
    spe_cross = spe_feature * gate_spa
    fused = spa_cross, with spe_cross added at the center pixel
    return x_residual + fused
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
            raise ValueError("spa_feature must have the same shape as x_residual")

        batch, height, width, channels = x_residual.shape
        if (height, width) != (self.patch_size, self.patch_size):
            raise ValueError("Feature-map size does not match configured patch_size")
        if channels != self.hidden_dim:
            raise ValueError("Feature dimension does not match configured hidden_dim")
        if spe_feature.shape != (batch, 1, channels):
            raise ValueError(
                f"Expected spe_feature [B,1,D] = "
                f"[{batch},1,{channels}], got {tuple(spe_feature.shape)}"
            )

        gate_spe = torch.sigmoid(spe_feature).unsqueeze(1)  # [B,1,1,D]

        spa_global = self.spa_pool(spa_feature.permute(0, 3, 1, 2))
        spa_global = spa_global.squeeze(-1).permute(0, 2, 1)  # [B,1,D]
        gate_spa = torch.sigmoid(spa_global)  # [B,1,D]

        spa_cross = spa_feature * gate_spe
        spe_cross = spe_feature * gate_spa

        fused = spa_cross.clone()
        fused[:, self.center, self.center, :] = (
            fused[:, self.center, self.center, :] + spe_cross[:, 0, :]
        )

        return x_residual + fused


class MeanConcatFusion(nn.Module):
    """Legacy mean-pooled X/S + center P -> [B,D] vector."""

    def __init__(self, patch_size, hidden_dim):
        super().__init__()
        self.patch_size = _integer(patch_size, 'fusion patch_size')
        self.hidden_dim = _integer(hidden_dim, 'fusion hidden_dim')
        if self.patch_size % 2 == 0:
            raise ValueError('fusion patch_size must be positive and odd')
        self.fusion_proj = nn.Linear(3 * self.hidden_dim, self.hidden_dim)

    def forward(self, x_residual, spa_feature, spe_feature):
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


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------

class BasicSpaSpeBlock(nn.Module):
    """PatchEmbedding -> PSA or DPE -> Spa/Spe -> CAS2F -> spatial refine -> FFN."""

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
        fusion_type="cas2f",
        fusion_heads=8,
        dpe_options=None,
        cfam_options=None,
        psa_enabled=False,
        spectral_token_num=4,
        spectral_center_pool=3,
        spatial_refine_enabled=True,
        **mamba_kwargs,
    ):
        super().__init__()

        self.patch_size = int(patch_size)
        self.hidden_dim = int(hidden_dim)
        self.fusion_type = fusion_type
        if not isinstance(psa_enabled, bool):
            raise ValueError('psa_enabled must be boolean')
        self.psa_enabled = psa_enabled

        # --------------------------------------------------------------
        # Shared preprocessing: PSA or DPE.
        # --------------------------------------------------------------
        if self.psa_enabled:
            if dpe_options:
                raise ValueError('psa_enabled=True does not accept dpe_options')
            if hidden_dim % 4:
                raise ValueError(
                    'Single-scale PSA requires hidden_dim divisible by 4'
                )
            # Isolate PSA init from the shared backbone RNG.
            with torch.random.fork_rng(devices=[]):
                self.psa = PyramidRefinedChannelAttention(
                    dim=hidden_dim,
                    num_heads=4,
                    bias=True,
                    num_scales=1,
                    num_layers=2,
                )
        else:
            self.dpe = DualMambaDPE(hidden_dim)

        # --------------------------------------------------------------
        # Spatial branch.
        # --------------------------------------------------------------
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

        # --------------------------------------------------------------
        # Spectral branch: tokenized OSSM + optional residual BFS.
        # --------------------------------------------------------------
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

        # --------------------------------------------------------------
        # Fusion.
        # --------------------------------------------------------------
        del fusion_heads  # legacy field, unused by both fusions

        if fusion_type == "cas2f":
            if cfam_options:
                raise ValueError('cas2f does not accept CFAM enhancement options')
            self.cross_fusion = CrossAttentionSpaSpeFusion(
                patch_size=patch_size, hidden_dim=hidden_dim,
            )
        elif fusion_type in ("mean_concat", "concat_residual",
                             "center_cfam", "center_cfam_residual"):
            if cfam_options:
                raise ValueError("mean_concat has no CFAM enhancement options")
            with torch.random.fork_rng(devices=[]):
                self.cross_fusion = MeanConcatFusion(
                    patch_size=patch_size, hidden_dim=hidden_dim,
                )
                self.cross_fusion.apply(BasicSpaSpeMamba._init_weights)
                for module in self.cross_fusion.modules():
                    if isinstance(module, nn.Linear):
                        module._fusion_initialized = True
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        # --------------------------------------------------------------
        # Spatial refinement (only meaningful on the CAS2F path).
        # --------------------------------------------------------------
        if spatial_refine_enabled:
            self.spatial_refine = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1,
                          groups=hidden_dim, bias=True),
                nn.GroupNorm(1, hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=True),
            )
        else:
            self.spatial_refine = None

        # --------------------------------------------------------------
        # FFN.
        # --------------------------------------------------------------
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0 else nn.Identity()
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
        # Shared preprocessing: PSA or DPE.
        # --------------------------------------------------------------
        if self.psa_enabled:
            shared = self.psa(
                x.permute(0, 3, 1, 2).contiguous()
            ).permute(0, 2, 3, 1).contiguous()
            x_dpe = x + shared  # residual so PSA is a gentle preprocessing
        else:
            x_dpe = self.dpe(x, raw_patch=raw_patch)

        # --------------------------------------------------------------
        # Mamba branches (both consume the shared preprocessing output).
        # --------------------------------------------------------------
        spa_delta = self.spatial(x_dpe)
        spe_delta = self.spectral(x_dpe)

        # --------------------------------------------------------------
        # Fusion.
        # --------------------------------------------------------------
        x = self.cross_fusion(
            x_residual=x_dpe,
            spa_feature=spa_delta,
            spe_feature=spe_delta,
        )

        # --------------------------------------------------------------
        # Spatial refinement (CAS2F path only; skipped for mean_concat).
        # --------------------------------------------------------------
        if self.spatial_refine is not None and x.ndim == 4:
            x = x + self.spatial_refine(
                x.permute(0, 3, 1, 2).contiguous()
            ).permute(0, 2, 3, 1).contiguous()

        # --------------------------------------------------------------
        # Residual FFN.
        # --------------------------------------------------------------
        return x + self.drop_path(self.ffn(self.norm_ffn(x)))


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class BasicSpaSpeMamba(nn.Module):
    """Shared PatchEmbedding + optional PSA or DPE + DualMamba + CAS2F."""

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

        # Spatial OSSM.
        spatial_forward_type="v3",
        spatial_ssm_conv=3,
        spatial_ssm_init="v0",

        # Spectral OSSM.
        spectral_forward_type="v3",
        spectral_ssm_conv=3,
        spectral_ssm_init="v0",
        bfs_groups=8,
        bfs_enabled=True,

        # Fusion.
        fusion_type="cas2f",
        fusion_heads=8,
        dpe_options=None,
        cfam_options=None,

        # Shared preprocessing selector.
        psa_enabled=False,

        # Spectral tokenization.
        spectral_token_num=4,
        spectral_center_pool=3,

        # Spatial refinement toggle.
        spatial_refine_enabled=True,

        # Optional single-input enhancement after the completed fusion block.
        aedc_enabled=False,

        **kwargs,
    ):
        super().__init__()

        if kwargs:
            raise TypeError('Unknown model options: ' + ', '.join(sorted(kwargs)))
        del use_gate, ssm_conv

        if len(n_groups) != 1 or len(dims) != 1 or len(depths) != 1:
            raise ValueError(
                "BasicSpaSpeMamba currently supports one encoder stage"
            )

        patch_size = _integer(patch_size, 'patch_size')
        if patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd integer")

        depth = _integer(depths[0], 'depths[0]')
        if fusion_type != 'cas2f' and depth != 1:
            raise ValueError(
                'mean-pooled fusion requires depths=[1]: its output is a vector'
            )
        if fusion_type == 'cas2f' and depth != 1:
            raise ValueError(
                'BasicSpaSpeMamba supports one encoder stage (depths=[1])'
            )

        in_chans = _integer(in_chans, 'in_chans')
        num_classes = _integer(num_classes, 'num_classes')
        hidden_dim = _integer(dims[0], 'dims[0]')
        group_count = _integer(n_groups[0], 'n_groups[0]')
        if hidden_dim % group_count:
            raise ValueError('dims[0] must be divisible by n_groups[0]')
        drop_path_rate = _probability(drop_path_rate, 'drop_path_rate')
        cfam_options = dict(cfam_options or {})

        if not isinstance(psa_enabled, bool):
            raise ValueError('psa_enabled must be boolean')
        self.psa_enabled = psa_enabled
        if not isinstance(aedc_enabled, bool):
            raise ValueError('aedc_enabled must be boolean')
        if aedc_enabled and fusion_type != 'cas2f':
            raise ValueError('AEDC requires fusion_type="cas2f" to retain a spatial feature map')
        self.aedc_enabled = aedc_enabled

        self.patch_size = int(patch_size)
        self.in_chans = int(in_chans)

        # Shared embedding.
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_chans, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_count, hidden_dim),
            nn.SiLU(),
        )

        drop_paths = (
            [drop_path_rate] if depth == 1
            else torch.linspace(0, drop_path_rate, depth).tolist()
        )

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
                    psa_enabled=psa_enabled,

                    spectral_token_num=spectral_token_num,
                    spectral_center_pool=spectral_center_pool,
                    spatial_refine_enabled=spatial_refine_enabled,

                    **mamba_kwargs,
                )
                for index in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

        self.aedc = None
        if self.aedc_enabled:
            # Added module initialization must not change retained weights.
            with torch.random.fork_rng(devices=[]):
                self.aedc = DynamicConvBlock(channels=hidden_dim)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear) and not getattr(module, "_fusion_initialized", False):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward_features(self, x):
        expected = (self.in_chans, self.patch_size, self.patch_size)

        if (x.ndim != 5 or x.shape[1] != 1 or tuple(x.shape[2:]) != expected):
            raise ValueError(
                f"Expected [B,1,{expected[0]},{expected[1]},{expected[2]}], "
                f"got {tuple(x.shape)}"
            )

        x = self.patch_embedding(x.squeeze(1))     # [B,D,H,W]
        x = x.permute(0, 2, 3, 1).contiguous()      # [B,H,W,D]

        for block in self.blocks:
            x = block(x)

        if self.aedc_enabled:
            # Original AEDC has one BCHW input and no external residual add.
            x = self.aedc(x.permute(0, 3, 1, 2).contiguous())
            x = x.permute(0, 2, 3, 1).contiguous()

        features = self.norm(x)
        if features.ndim == 2:
            return features
        return features.mean(dim=(1, 2))

    def forward(self, x):
        return self.head(self.forward_features(x))

    def flops(self, shape=None, verbose=True):
        del verbose
        expected_shape = (1, self.in_chans, self.patch_size, self.patch_size)
        shape = expected_shape if shape is None else tuple(shape)
        if shape != expected_shape:
            raise ValueError(
                f"Expected FLOPs input shape {expected_shape}, got {shape}"
            )
        device = next(self.parameters()).device
        model = copy.deepcopy(self).to(device).eval()
        devices = [device.index] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            sample = torch.randn(
                (1, *shape), device=device,
                dtype=next(model.parameters()).dtype,
            )
            model(sample)
        params = sum(p.numel() for p in model.parameters())
        tag = 'mamba_shared_psa' if self.psa_enabled else 'mamba_shared_dpe'
        if self.aedc_enabled:
            tag += '_aedc'
        print(f"params {params} GFLOPs not reported for {tag}")


def basic_spa_spe_mamba(model_config):
    return BasicSpaSpeMamba(**model_config)


__all__ = [
    "DynamicConvBlock",
    "BidirectionalFusionState",
    "DualMambaDPE",
    "PyramidAttention",
    "PyramidRefinedChannelAttention",
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
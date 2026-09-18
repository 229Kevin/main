"""Basic spatial-spectral dual-branch Mamba for HSI classification.

The model intentionally keeps the network around Mamba simple.  It reuses
DualMamba's CUDA selective-scan backend, while keeping the parameter generation
and scan layout in this file so future changes to the Mamba internals do not
require editing ``rs_mamba_ss.py``.
"""

import copy
import math

import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_

from .rs_mamba_ss import GroupedPixelEmbedding_mm, Mlp, SelectiveScanOflex


class EditableMambaCore(nn.Module):
    """A small, editable Mamba core backed by DualMamba selective scan.

    ``num_directions`` controls independent SSM parameter groups.  Input and
    output projections are shared, matching DualMamba's lightweight design,
    while x/dt/A/D parameters remain direction-specific.
    """

    def __init__(
        self,
        d_model,
        num_directions=1,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        d_conv=0,
        conv_bias=True,
        dropout=0.0,
        use_gate=False,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_scale=1.0,
        dt_init="random",
        dt_init_floor=1e-4,
    ):
        super().__init__()
        if num_directions < 1:
            raise ValueError("num_directions must be positive")
        if d_conv > 1 and d_conv % 2 == 0:
            raise ValueError("d_conv must be odd so sequence length is preserved")

        self.d_model = int(d_model)
        self.num_directions = int(num_directions)
        self.d_inner = int(ssm_ratio * d_model)
        self.d_state = int(d_state)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.d_conv = int(d_conv)
        self.use_gate = bool(use_gate)

        projected_dim = self.d_inner * (2 if self.use_gate else 1)
        self.in_proj = nn.Linear(d_model, projected_dim, bias=bias)
        self.act = nn.SiLU()

        if self.d_conv > 1:
            self.conv1d = nn.Conv1d(
                self.d_inner,
                self.d_inner,
                kernel_size=self.d_conv,
                padding=(self.d_conv - 1) // 2,
                groups=self.d_inner,
                bias=conv_bias,
            )

        x_projs = [
            nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
            for _ in range(self.num_directions)
        ]
        self.x_proj_weight = nn.Parameter(
            torch.stack([projection.weight for projection in x_projs], dim=0)
        )

        dt_projs = [
            self._make_dt_projection(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
            )
            for _ in range(self.num_directions)
        ]
        self.dt_proj_weight = nn.Parameter(
            torch.stack([projection.weight for projection in dt_projs], dim=0)
        )
        self.dt_proj_bias = nn.Parameter(
            torch.stack([projection.bias for projection in dt_projs], dim=0)
        )

        state_ids = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        a_log = torch.log(state_ids).repeat(
            self.num_directions * self.d_inner, 1
        )
        self.A_logs = nn.Parameter(a_log)
        self.A_logs._no_weight_decay = True
        self.Ds = nn.Parameter(torch.ones(self.num_directions * self.d_inner))
        self.Ds._no_weight_decay = True

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @staticmethod
    def _make_dt_projection(
        dt_rank,
        d_inner,
        dt_scale,
        dt_init,
        dt_min,
        dt_max,
        dt_init_floor,
    ):
        projection = nn.Linear(dt_rank, d_inner, bias=True)
        init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(projection.weight, init_std)
        elif dt_init == "random":
            nn.init.uniform_(projection.weight, -init_std, init_std)
        else:
            raise ValueError(f"Unsupported dt_init: {dt_init}")

        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inverse_softplus = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            projection.bias.copy_(inverse_softplus)
        return projection

    def forward(self, sequences):
        """Process ``[B,K,L,D]`` sequences and preserve their shape."""
        if sequences.ndim != 4:
            raise ValueError(f"Expected [B,K,L,D], got {tuple(sequences.shape)}")
        batch, directions, length, dimension = sequences.shape
        if directions != self.num_directions or dimension != self.d_model:
            raise ValueError(
                f"Expected K={self.num_directions}, D={self.d_model}; "
                f"got K={directions}, D={dimension}"
            )

        projected = self.in_proj(sequences)
        if self.use_gate:
            projected, gate = projected.chunk(2, dim=-1)
            gate = self.act(gate)
        else:
            gate = None

        if self.d_conv > 1:
            projected = rearrange(projected, "b k l d -> (b k) d l")
            projected = self.conv1d(projected)
            projected = rearrange(
                projected, "(b k) d l -> b k l d", b=batch, k=directions
            )
        projected = self.act(projected)

        u = rearrange(projected, "b k l d -> b k d l")
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", u, self.x_proj_weight)
        dt, state_b, state_c = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2
        )
        delta = torch.einsum("b k r l, k d r -> b k d l", dt, self.dt_proj_weight)

        u = u.reshape(batch, directions * self.d_inner, length).contiguous()
        delta = delta.reshape(batch, directions * self.d_inner, length).contiguous()
        state_b = state_b.contiguous()
        state_c = state_c.contiguous()
        a = -torch.exp(self.A_logs.float())

        output = SelectiveScanOflex.apply(
            u,
            delta,
            a,
            state_b,
            state_c,
            self.Ds.float(),
            self.dt_proj_bias.float().reshape(-1),
            True,
            -1,
            -1,
            True,
        )
        output = output.view(batch, directions, self.d_inner, length)
        output = rearrange(output, "b k d l -> b k l d").to(projected.dtype)
        output = self.out_norm(output)
        if gate is not None:
            output = output * gate
        return self.dropout(self.out_proj(output))


def _snake_indices(size):
    grid = torch.arange(size * size, dtype=torch.long).view(size, size)
    grid[1::2] = torch.flip(grid[1::2], dims=[1])
    order = grid.flatten()
    return order, torch.argsort(order)


class SpatialSnakeMamba(nn.Module):
    """Unidirectional Mamba over an S-shaped spatial sequence."""

    def __init__(self, patch_size, hidden_dim, **mamba_kwargs):
        super().__init__()
        self.patch_size = int(patch_size)
        order, inverse_order = _snake_indices(self.patch_size)
        self.register_buffer("snake_order", order, persistent=False)
        self.register_buffer("inverse_snake_order", inverse_order, persistent=False)
        self.dynamic_pos = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.mamba = EditableMambaCore(
            d_model=hidden_dim, num_directions=1, **mamba_kwargs
        )

    def forward(self, x):
        batch, height, width, channels = x.shape
        if (height, width) != (self.patch_size, self.patch_size):
            raise ValueError("Spatial patch does not match configured patch_size")
        spatial = x.permute(0, 3, 1, 2)
        spatial = spatial + self.dynamic_pos(spatial)
        x = spatial.permute(0, 2, 3, 1)
        sequence = self.norm(x).reshape(batch, height * width, channels)
        sequence = sequence.index_select(1, self.snake_order)
        output = self.mamba(sequence.unsqueeze(1)).squeeze(1)
        output = output.index_select(1, self.inverse_snake_order)
        return output.view(batch, height, width, channels)


class SpectralBiMamba(nn.Module):
    """Bidirectional Mamba over the latent spectral/channel sequence."""

    def __init__(self, patch_size, hidden_dim, **mamba_kwargs):
        super().__init__()
        self.patch_size = int(patch_size)
        self.hidden_dim = int(hidden_dim)
        spatial_dim = self.patch_size * self.patch_size
        self.norm = nn.LayerNorm(spatial_dim)
        self.mamba = EditableMambaCore(
            d_model=spatial_dim, num_directions=2, **mamba_kwargs
        )

    def forward(self, x):
        batch, height, width, channels = x.shape
        if (height, width) != (self.patch_size, self.patch_size):
            raise ValueError("Spatial patch does not match configured patch_size")
        if channels != self.hidden_dim:
            raise ValueError("Channel dimension does not match configured hidden_dim")

        sequence = rearrange(x, "b h w d -> b d (h w)")
        sequence = self.norm(sequence)
        directions = torch.stack(
            [sequence, torch.flip(sequence, dims=[1])], dim=1
        )
        output = self.mamba(directions)
        forward = output[:, 0]
        backward = torch.flip(output[:, 1], dims=[1])
        output = 0.5 * (forward + backward)
        return rearrange(
            output,
            "b d (h w) -> b h w d",
            h=self.patch_size,
            w=self.patch_size,
        )


class BasicSpaSpeBlock(nn.Module):
    """Parallel spatial/spectral Mamba, static addition, residual FFN."""

    def __init__(
        self,
        patch_size,
        hidden_dim,
        drop_path=0.0,
        mlp_ratio=4.0,
        mlp_drop_rate=0.0,
        **mamba_kwargs,
    ):
        super().__init__()
        self.spatial = SpatialSnakeMamba(
            patch_size, hidden_dim, **mamba_kwargs
        )
        self.spectral = SpectralBiMamba(
            patch_size, hidden_dim, **mamba_kwargs
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = Mlp(
            in_features=hidden_dim,
            hidden_features=int(hidden_dim * mlp_ratio),
            out_features=hidden_dim,
            drop=mlp_drop_rate,
        )

    def forward(self, x):
        fused = self.spatial(x) + self.spectral(x)
        x = x + self.drop_path(fused)
        return x + self.drop_path(self.ffn(self.norm_ffn(x)))


class BasicSpaSpeMamba(nn.Module):
    """Minimal dual-branch Mamba classifier for Indian Pines patches."""

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
        **kwargs,
    ):
        super().__init__()
        del kwargs
        if len(n_groups) != 1 or len(dims) != 1 or len(depths) != 1:
            raise ValueError("BasicSpaSpeMamba currently supports one encoder stage")
        if patch_size <= 0 or patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd integer")
        if depths[0] < 1:
            raise ValueError("depths[0] must be positive")

        self.patch_size = int(patch_size)
        self.in_chans = int(in_chans)
        hidden_dim = int(dims[0])
        group_count = int(n_groups[0])
        padded_bands = math.ceil(in_chans / group_count) * group_count

        self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 0, padded_bands - in_chans))
        self.group_emb = GroupedPixelEmbedding_mm(
            in_feature_map_size=patch_size,
            in_chans=padded_bands,
            embed_dim=hidden_dim,
            n_groups=group_count,
        )

        drop_paths = torch.linspace(0, drop_path_rate, depths[0]).tolist()
        mamba_kwargs = dict(
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            use_gate=use_gate,
        )
        self.blocks = nn.Sequential(
            *[
                BasicSpaSpeBlock(
                    patch_size=patch_size,
                    hidden_dim=hidden_dim,
                    drop_path=drop_paths[index],
                    mlp_ratio=mlp_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    **mamba_kwargs,
                )
                for index in range(depths[0])
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward_features(self, x):
        expected = (self.in_chans, self.patch_size, self.patch_size)
        if x.ndim != 5 or x.shape[1] != 1 or tuple(x.shape[2:]) != expected:
            raise ValueError(f"Expected [B,1,{expected[0]},{expected[1]},{expected[2]}], got {tuple(x.shape)}")
        x = self.pad(x).squeeze(1)
        x = self.group_emb(rearrange(x, "b c h w -> b h w c"))
        x = self.blocks(x)
        return self.norm(x).mean(dim=(1, 2))

    def forward(self, x):
        return self.head(self.forward_features(x))

    def flops(self, shape=None, verbose=True):
        del verbose
        expected_shape = (1, self.in_chans, self.patch_size, self.patch_size)
        shape = expected_shape if shape is None else tuple(shape)
        if shape != expected_shape:
            raise ValueError(f"Expected FLOPs input shape {expected_shape}, got {shape}")

        device = next(self.parameters()).device
        model = copy.deepcopy(self).to(device).eval()
        sample = torch.randn((1, *shape), device=device)
        with torch.no_grad():
            model(sample)
        params = sum(parameter.numel() for parameter in model.parameters())
        print(f"params {params} GFLOPs not reported for basic_spa_spe_mamba")


def basic_spa_spe_mamba(model_config):
    return BasicSpaSpeMamba(**model_config)


__all__ = [
    "EditableMambaCore",
    "SpatialSnakeMamba",
    "SpectralBiMamba",
    "BasicSpaSpeBlock",
    "BasicSpaSpeMamba",
    "basic_spa_spe_mamba",
]

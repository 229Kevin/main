"""Standalone ablation of DualMamba's lightweight spectral Mamba path."""

import copy
import math

import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import trunc_normal_

from .rs_mamba_ss import GroupedPixelEmbedding_mm, OSSM


class LightweightSpectralMambaClassifier(nn.Module):
    """Keep the original grouped stem, DPE and lightweight spectral OSSM only.

    The original spectral output is used by CAS2F as channel modulation and has
    no standalone classifier. This ablation adds only LN + Linear to measure the
    discriminative information in that raw spectral output.
    """

    def __init__(
        self,
        n_groups=(4,),
        patch_size=7,
        in_chans=200,
        num_classes=16,
        dims=(64,),
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",
        ssm_conv=0,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        forward_type="v3noz",
        **kwargs,
    ):
        super().__init__()
        del kwargs
        if len(n_groups) != 1 or len(dims) != 1:
            raise ValueError("The spectral-only ablation requires one group stage and one dimension")
        if patch_size <= 0 or patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd integer")
        if forward_type != "v3noz":
            raise ValueError("Use forward_type='v3noz' to match the active IP DualMamba config")
        if ssm_conv != 0:
            raise ValueError("Use ssm_conv=0 to match the active IP DualMamba config")

        hidden_dim = int(dims[0])
        group_count = int(n_groups[0])
        padded_bands = math.ceil(in_chans / group_count) * group_count
        self.patch_size = int(patch_size)
        self.in_chans = int(in_chans)
        self.hidden_dim = hidden_dim
        self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 0, padded_bands - in_chans))
        self.group_emb = GroupedPixelEmbedding_mm(
            in_feature_map_size=patch_size,
            in_chans=padded_bands,
            embed_dim=hidden_dim,
            n_groups=group_count,
        )

        # Same DPE and center-token preparation as ASF_SSBlock._forward.
        self.dynamic_pos = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim
        )
        self.norm_spe = nn.LayerNorm(hidden_dim)
        act_layers = {"silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU}
        if isinstance(ssm_act_layer, str):
            try:
                ssm_act_layer = act_layers[ssm_act_layer.lower()]
            except KeyError as error:
                raise ValueError(f"Unsupported SSM activation {ssm_act_layer!r}") from error
        self.op_spe = OSSM(
            d_model=1,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            initialize=ssm_init,
            forward_type="spe" + forward_type,
            k_group=2,
        )

        # Minimal head required because the original branch only generates CAS2F features.
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        # Match ASF_RSM_Group initialization, including OSSM input/output projections.
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward_features(self, x):
        if x.ndim != 5 or x.shape[1] != 1 or x.shape[2] != self.in_chans:
            raise ValueError(
                f"Expected [B,1,{self.in_chans},H,W], got {tuple(x.shape)}"
            )
        if x.shape[-2:] != (self.patch_size, self.patch_size):
            raise ValueError(
                f"Expected spatial patch {self.patch_size}x{self.patch_size}, "
                f"got {tuple(x.shape[-2:])}"
            )
        x = self.pad(x).squeeze(1)
        x = self.group_emb(rearrange(x, "b c h w -> b h w c"))
        x = x.permute(0, 3, 1, 2)
        x = x + self.dynamic_pos(x)
        x = x.permute(0, 2, 3, 1)
        center = self.norm_spe(
            x[:, self.patch_size // 2, self.patch_size // 2, :].unsqueeze(1)
        )
        spectral = self.op_spe(center.permute(0, 2, 1)).permute(0, 2, 1)
        return spectral.squeeze(1)

    def forward(self, x):
        return self.classifier(self.forward_features(x))

    def flops(self, shape=(1, 200, 7, 7), verbose=True):
        del verbose
        model = copy.deepcopy(self).cuda().eval()
        sample = torch.randn((1, *shape), device=next(model.parameters()).device)
        with torch.no_grad():
            model(sample)
        params = sum(parameter.numel() for parameter in model.parameters())
        print(f"params {params} GFLOPs not reported for spectral-only ablation")


def lightweight_spectral_mamba(model_config):
    return LightweightSpectralMambaClassifier(**model_config)


__all__ = ["LightweightSpectralMambaClassifier", "lightweight_spectral_mamba"]

"""DualMamba CAS2MM-only model.

This module keeps the original DualMamba model/factory name:
    asf_rsm_group(model_config)

It is compatible with the official `asf_rsm_group_ip.json`, while removing:
    1) the lightweight spectral-spatial residual convolution module
    2) the adaptive global-local fusion module

Retained path:
    GroupedPixelEmbedding_mm
        -> OSSBlock
           (DPE + Spatial Mamba + Spectral Mamba + CAS2F)
        -> classifier
"""

import math
from collections import OrderedDict

import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import trunc_normal_

from .rs_mamba_ss import GroupedPixelEmbedding_mm, OSSBlock, Permute


class ASF_RSM_Group(nn.Module):
    """CAS2MM-only variant compatible with official DualMamba JSON."""

    def __init__(
        self,
        n_groups=[4, 4, 4, 4],
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        depths=[2, 2, 9, 2],
        dims=[96, 192, 384, 768],
        ssm_d_state=16,
        ssm_ratio=2,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0,
        ssm_init="v0",
        forward_type="v2",
        spe_branch=0,
        split_ratio=0,
        mlp_ratio=4,
        mlp_act_layer="gelu",
        mlp_drop_rate=0,
        drop_path_rate=0.1,
        patch_norm=True,
        norm_layer="LN",
        use_checkpoint=False,
        post_norm=0,
        **kwargs,
    ):
        super().__init__()

        if int(spe_branch) != 1:
            raise ValueError("CAS2MM-only requires spe_branch=1.")
        if float(split_ratio) != 0:
            raise ValueError(
                "CAS2MM-only requires split_ratio=0 because the local "
                "convolution branch is removed."
            )
        if float(mlp_ratio) != 0:
            raise ValueError(
                "CAS2MM-only requires mlp_ratio=0 to match the official "
                "CAS2MM-only setting."
            )

        if len(depths) != len(dims):
            raise ValueError("depths and dims must have the same length.")
        if len(n_groups) < 1:
            raise ValueError("n_groups must contain at least one value.")
        if len(dims) != 1:
            raise ValueError(
                "This CAS2MM-only implementation targets the official "
                "one-stage HSI setting, e.g. depths=[1], dims=[64]."
            )

        self.num_classes = int(num_classes)
        self.num_features = int(dims[-1])
        self.dims = [int(v) for v in dims]
        self.patch_size = int(patch_size)

        self.dpr = [
            x.item()
            for x in torch.linspace(0, float(drop_path_rate), sum(depths))
        ]

        norm_layers = {
            "ln": nn.LayerNorm,
            "bn": nn.BatchNorm2d,
        }
        act_layers = {
            "silu": nn.SiLU,
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "sigmoid": nn.Sigmoid,
        }

        if isinstance(norm_layer, str):
            key = norm_layer.lower()
            if key not in norm_layers:
                raise ValueError(f"Unsupported norm_layer: {norm_layer}")
            norm_layer = norm_layers[key]

        if isinstance(ssm_act_layer, str):
            key = ssm_act_layer.lower()
            if key not in act_layers:
                raise ValueError(f"Unsupported ssm_act_layer: {ssm_act_layer}")
            ssm_act_layer = act_layers[key]

        if isinstance(mlp_act_layer, str):
            key = mlp_act_layer.lower()
            if key not in act_layers:
                raise ValueError(f"Unsupported mlp_act_layer: {mlp_act_layer}")
            mlp_act_layer = act_layers[key]

        new_bands = math.ceil(in_chans / n_groups[0]) * n_groups[0]

        self.pad = nn.ReplicationPad3d(
            (0, 0, 0, 0, 0, new_bands - in_chans)
        )

        self.group_emb = GroupedPixelEmbedding_mm(
            in_feature_map_size=patch_size,
            in_chans=new_bands,
            embed_dim=self.dims[0],
            n_groups=n_groups[0],
        )

        blocks = []
        offset = 0

        for i_layer, depth in enumerate(depths):
            if i_layer != 0:
                raise ValueError(
                    "CAS2MM-only does not define stage transitions; "
                    "use the official one-stage IP setup."
                )

            for d in range(int(depth)):
                blocks.append(
                    OSSBlock(
                        hidden_dim=self.dims[i_layer],
                        drop_path=self.dpr[offset + d],
                        norm_layer=norm_layer,
                        patch_size=patch_size,
                        ssm_d_state=ssm_d_state,
                        ssm_ratio=ssm_ratio,
                        ssm_dt_rank=ssm_dt_rank,
                        ssm_act_layer=ssm_act_layer,
                        ssm_conv=ssm_conv,
                        ssm_conv_bias=ssm_conv_bias,
                        ssm_drop_rate=ssm_drop_rate,
                        ssm_init=ssm_init,
                        forward_type=forward_type,
                        spe_branch=spe_branch,
                        split_ratio=split_ratio,
                        mlp_ratio=mlp_ratio,
                        mlp_act_layer=mlp_act_layer,
                        mlp_drop_rate=mlp_drop_rate,
                        use_checkpoint=use_checkpoint,
                        post_norm=post_norm,
                    )
                )

            offset += int(depth)

        self.encoder_layers = nn.Sequential(*blocks)

        self.classifier = nn.Sequential(
            OrderedDict(
                norm=norm_layer(self.num_features),
                permute=Permute(0, 3, 1, 2),
                avgpool=nn.AdaptiveAvgPool2d(1),
                flatten=nn.Flatten(1),
                head=nn.Linear(self.num_features, self.num_classes),
            )
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                f"Expected input [B,1,C,H,W], got {tuple(x.shape)}"
            )
        if x.shape[1] != 1:
            raise ValueError(
                f"Expected x.shape[1] == 1, got {x.shape[1]}"
            )

        x = self.pad(x).squeeze(1)
        x = rearrange(x, "b c h w -> b h w c")
        x = self.group_emb(x)

        for block in self.encoder_layers:
            x = block(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.classifier(x)


def asf_rsm_group(model_config):
    """Keep the original DualMamba factory name."""
    return ASF_RSM_Group(**model_config)


if __name__ == "__main__":
    config = {
        "n_groups": [4],
        "patch_size": 7,
        "in_chans": 200,
        "num_classes": 16,
        "depths": [1],
        "dims": [64],
        "split_ratio": 0,
        "forward_type": "v3noz",
        "spe_branch": 1,
        "mlp_ratio": 0,
        "ssm_conv": 0,
        "post_norm": 0,
    }

    model = asf_rsm_group(config)
    print(model)

"""CAS2MM-only variant of DualMamba.

Goal
----
Keep only the original DualMamba CAS2MM path:
    GroupedPixelEmbedding_mm
        -> OSSBlock
           (DPE + Spatial Mamba + Spectral Mamba + CAS2F)
        -> classifier

Removed
-------
1. lightweight spectral-spatial residual convolution module
2. adaptive global-local fusion

Compatibility
-------------
- Keeps the factory name: asf_rsm_group(model_config)
- Directly accepts the official asf_rsm_group_ip.json
- Provides flops() so the original main.py can call:
      model.flops(shape=(1, num_bands, patch_size, patch_size))
"""

import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import trunc_normal_
from fvcore.nn import flop_count, parameter_count

from .rs_mamba_ss import (
    GroupedPixelEmbedding_mm,
    OSSBlock,
    Permute,
    selective_scan_flop_jit,
)


class ASF_RSM_Group(nn.Module):
    """DualMamba CAS2MM-only classifier."""

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

        # Official IP CAS2MM-only settings should satisfy these.
        if int(spe_branch) != 1:
            raise ValueError(
                "CAS2MM-only requires spe_branch=1 "
                "(Spatial Mamba + Spectral Mamba + CAS2F)."
            )
        if float(split_ratio) != 0:
            raise ValueError(
                "CAS2MM-only requires split_ratio=0. "
                "The local convolution branch is removed."
            )
        if float(mlp_ratio) != 0:
            raise ValueError(
                "CAS2MM-only requires mlp_ratio=0 to match "
                "the official IP CAS2MM setting."
            )

        if len(depths) != len(dims):
            raise ValueError("depths and dims must have the same length.")

        # Official HSI configuration is one stage, e.g. depths=[1], dims=[64].
        if len(dims) != 1:
            raise ValueError(
                "This CAS2MM-only implementation currently targets "
                "the official one-stage HSI setting, e.g. depths=[1], dims=[64]."
            )

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.num_features = dims[-1]
        self.dims = dims
        self.patch_size = patch_size

        self.dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]

        # Same layer mapping style as original DualMamba.
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

        # ------------------------------------------------------------------
        # Original DualMamba grouped pixel embedding
        # ------------------------------------------------------------------
        new_bands = math.ceil(in_chans / n_groups[0]) * n_groups[0]

        self.pad = nn.ReplicationPad3d(
            (0, 0, 0, 0, 0, new_bands - in_chans)
        )

        self.group_emb = GroupedPixelEmbedding_mm(
            in_feature_map_size=patch_size,
            in_chans=new_bands,
            embed_dim=dims[0],
            n_groups=n_groups[0],
        )

        # ------------------------------------------------------------------
        # CAS2MM ONLY
        #
        # Directly reuse rs_mamba_ss.OSSBlock.
        # OSSBlock contains:
        #   - DPE (depthwise 3x3 conv)
        #   - Spatial OSSM, k_group=1
        #   - Spectral OSSM, k_group=2 when spe_branch=1
        #   - CAS2F cross spectral-spatial interaction
        #
        # It does NOT contain the local convolution branch of ASF_SSBlock.
        # ------------------------------------------------------------------
        blocks = []
        dpr_offset = 0

        for i_layer in range(self.num_layers):
            if i_layer != 0:
                raise ValueError(
                    "CAS2MM-only currently supports the official "
                    "single-stage HSI setup only."
                )

            for d in range(depths[i_layer]):
                blocks.append(
                    OSSBlock(
                        hidden_dim=dims[i_layer],
                        drop_path=self.dpr[dpr_offset + d],
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

            dpr_offset += depths[i_layer]

        self.encoder_layers = nn.Sequential(*blocks)

        # Same classifier pattern as original DualMamba.
        self.classifier = nn.Sequential(
            OrderedDict(
                norm=norm_layer(self.num_features),
                permute=Permute(0, 3, 1, 2),
                avgpool=nn.AdaptiveAvgPool2d(1),
                flatten=nn.Flatten(1),
                head=nn.Linear(self.num_features, num_classes),
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

    def forward_features(self, x: torch.Tensor):
        # Original DualMamba input convention: [B, 1, C, H, W]
        if x.ndim != 5:
            raise ValueError(
                f"Expected input shape [B,1,C,H,W], got {tuple(x.shape)}"
            )
        if x.shape[1] != 1:
            raise ValueError(
                f"Expected input channel dimension x.shape[1] == 1, "
                f"got {x.shape[1]}"
            )

        x = self.pad(x).squeeze(1)
        x = rearrange(x, "b c h w -> b h w c")
        x = self.group_emb(x)

        for block in self.encoder_layers:
            x = block(x)

        return x

    def forward(self, x: torch.Tensor):
        x = self.forward_features(x)
        return self.classifier(x)

    def flops(self, shape=(1, 32, 15, 15), verbose=True):
        """Match the FLOPs API used by the original DualMamba main.py."""

        supported_ops = {
            "aten::silu": None,
            "aten::neg": None,
            "aten::exp": None,
            "aten::flip": None,
            "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanNRow": selective_scan_flop_jit,
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input_tensor = torch.randn(
            (1, *shape),
            device=next(model.parameters()).device,
        )

        params = parameter_count(model)[""]

        gflops, unsupported = flop_count(
            model=model,
            inputs=(input_tensor,),
            supported_ops=supported_ops,
        )

        total_gflops = sum(gflops.values())

        if verbose:
            print(f"params {params} GFLOPs {total_gflops}")
            if unsupported:
                print(f"unsupported ops: {unsupported}")

        del model, input_tensor

        return total_gflops


def asf_rsm_group(model_config):
    """Keep the original DualMamba model/factory name."""
    return ASF_RSM_Group(**model_config)


if __name__ == "__main__":
    # Same fields as official asf_rsm_group_ip.json
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

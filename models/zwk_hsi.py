"""最小 HSI patch 分类模型，支持 PSA/DAM 并行、串行或串行输出相加。

流程：HSI patch -> Conv2d -> LayerNorm -> GELU
     -> 可选浅层 CNN（Conv3d-BN-PReLU -> 合并深度 -> Conv2d-BN-PReLU）
     -> parallel: x + α * PSA(x) + β * DAM(x)
        serial: PSA(x) -> DAM（直接串联，增强模块之间无残差/系数）
        serial_sum: F + PSA(F) + DAM(PSA(F))（相加后再池化，无系数）
     -> 全局平均池化
     -> x + FFN(LayerNorm(x)) -> Linear 分类器。

α 是可学习标量，初始值为 0.1；β 是固定标量，默认 0.1。
系数仅用于 parallel 模式；serial 模式分类头只接收池化后的 DAM 输出。
DAM 实现独立保存在 zwk_DAM.py，dam_enabled=False 可恢复原 CNN+PSA。
cnn_enabled=False 时输入层直接连接增强分支。

支持 [B, bands, H, W] 和数据集使用的 [B, 1, bands, H, W]。
Conv2d 将原始波段映射为 embed_dim 个特征；启用浅层 CNN 时，
这些特征随后作为 Conv3d 的深度维，而不是原始光谱波段。

Example::

    model = ZWKHSI(num_bands=200, num_classes=16)
    logits = model(torch.randn(4, 1, 200, 13, 13))  # [4, 16]
    loss = torch.nn.functional.cross_entropy(logits, labels)
    predictions = logits.argmax(dim=1)
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class ChannelLayerNorm(nn.Module):
    """对每个位置的特征通道做 LayerNorm，兼容 2D/3D 特征图。"""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # [B, C, ...] -> [B, ..., C] -> LayerNorm -> [B, C, ...]
        return self.norm(x.movedim(1, -1)).movedim(-1, 1).contiguous()


class ShallowCNNBlock(nn.Module):
    """图示的 Conv3d-BN-PReLU-Conv2d-BN-PReLU 浅层特征提取模块。

    输入 [B, 1, depth, H, W]，输出 [B, conv2d_channels, H, W]。
    Conv3d 保持深度和空间尺寸，然后将通道与深度合并给 Conv2d。
    """

    def __init__(self, depth, conv3d_channels=8, conv2d_channels=64):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, conv3d_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(conv3d_channels),
            nn.PReLU(num_parameters=conv3d_channels),
        )
        self.conv2d = nn.Sequential(
            nn.Conv2d(conv3d_channels * depth, conv2d_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(conv2d_channels),
            nn.PReLU(num_parameters=conv2d_channels),
        )

    def forward(self, x):
        x = self.conv3d(x)       # [B, conv3d_channels, depth, H, W]
        x = x.flatten(1, 2)     # [B, conv3d_channels * depth, H, W]
        return self.conv2d(x)   # [B, conv2d_channels, H, W]


class SingleLayerPSA(nn.Module):
    """基于 HyPyraMamba 的单层 PSA，使用普通深度卷积适配 patch。

    来源：HyPyraMamba/HypraMamba/model.py 中的 PyramidAttention。
    QKV 深度卷积采用连续的 3x3 邻域（dilation=1, padding=1）。
    仅执行一次注意力，不包含多尺度分支、下采样或后续细化层。
    q/k 沿空间维归一化，注意力矩阵计算每个 head 内的通道关系。
    默认用固定高斯权重突出 patch 中心；只加权 Q/K，不加权 V。
    center_sigma 是以像素为单位的标准差，关闭 center_weighted 可恢复均匀统计。
    默认在 QKV 生成前对每个位置的通道做 LayerNorm，使用 2 个注意力头。
    """

    def __init__(self, dim, num_heads=2, bias=True,
                 center_weighted=True, center_sigma=3.0, pre_norm=True):
        super().__init__()
        if dim < 1 or num_heads < 1 or dim % num_heads:
            raise ValueError("PSA dim must be positive and divisible by a positive num_heads")
        if not isinstance(center_weighted, bool):
            raise ValueError("center_weighted must be boolean")
        if not isinstance(pre_norm, bool):
            raise ValueError("pre_norm must be boolean")
        self.center_sigma = float(center_sigma)
        if not math.isfinite(self.center_sigma) or self.center_sigma <= 0:
            raise ValueError("center_sigma must be positive and finite")
        self.center_weighted = center_weighted
        self.num_heads = num_heads
        self.pre_norm = pre_norm
        self.norm = ChannelLayerNorm(dim) if pre_norm else nn.Identity()
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1,
            dilation=1, padding=1, groups=dim * 3, bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        batch, channels, height, width = x.shape
        x = self.norm(x)
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        # [B, C, H, W] -> [B, heads, C / heads, H * W]
        q = q.reshape(batch, self.num_heads, channels // self.num_heads, height * width)
        k = k.reshape(batch, self.num_heads, channels // self.num_heads, height * width)
        v = v.reshape(batch, self.num_heads, channels // self.num_heads, height * width)
        if self.center_weighted:
            # w = exp(-distance^2 / (2 * sigma^2)); use sqrt(w) for weighted cosine.
            rows = torch.arange(height, device=q.device, dtype=q.dtype) - (height - 1) / 2
            cols = torch.arange(width, device=q.device, dtype=q.dtype) - (width - 1) / 2
            distance_squared = rows[:, None].square() + cols[None, :].square()
            sqrt_weight = torch.exp(-distance_squared / (4 * self.center_sigma ** 2))
            sqrt_weight = sqrt_weight.reshape(1, 1, 1, height * width)
            q = q * sqrt_weight
            k = k * sqrt_weight
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attention = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        output = (attention @ v).reshape(batch, channels, height, width)
        return self.project_out(output)


class ResidualFFN(nn.Module):
    """预归一化 FFN：输出 x + FFN(LayerNorm(x))，保留输入特征维度。"""

    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return x + self.ffn(self.norm(x))


class ZWKHSI(nn.Module):
    """一个 patch 输出一个类别，patch 的空间大小可变化。

    Args:
        num_bands: 输入 HSI 的波段数。
        num_classes: 分类类别数。
        embed_dim: 前端 Conv2d 输出通道数，也是后续 Conv3d 的深度。
        conv3d_channels: CNN 模块中 Conv3d 的输出通道数。
        conv2d_channels: 启用浅层 CNN 时的输出通道数，也是 PSA/FFN 输入维度。
        ffn_hidden_dim: 分类 FFN 的隐藏维度。
        dropout: 分类 FFN 隐藏层的 dropout 概率。
        psa_enabled: 是否在 CNN 后加入单层、原分辨率 PSA。
        psa_num_heads: PSA 注意力头数，默认 2，必须整除实际输入特征通道数。
        psa_residual_scale: parallel 模式下 PSA 可学习残差系数的初始值。
        psa_center_weighted: 是否用中心高斯权重计算 PSA 的通道关系。
        psa_center_sigma: 高斯权重的标准差，单位为像素，默认 3.0。
        psa_pre_norm: 是否在 PSA 的 QKV 生成前使用通道 LayerNorm。
        cnn_enabled: 是否使用浅层 CNN，默认 True 兼容原结构；False 时
            stem 直接连接 PSA，PSA/FFN 输入维度为 embed_dim。
        dam_enabled: 是否启用单层 DAM，默认 False 兼容已有模型/检查点。
        dam_residual_scale: parallel 模式下 DAM 固定融合系数，默认 0.1。
        dam_d_state: DAM 状态维度，默认 8。
        dam_ssm_ratio: DAM 内部宽度倍率，默认 1.0。
        dam_dt_rank: DAM 时间步投影秩，默认 auto（64 通道时为 4）。
        dam_direction_enabled: 是否启用轻量方向补偿，默认 True。
        dam_scan_backend: auto、reference 或 cuda，默认 auto。
        fusion_mode: parallel 保留加权残差融合；serial 将 PSA 输出直接送给
            DAM 后直接池化；serial_sum 保留串行提取，再将 CNN、PSA、DAM
            三个输出直接相加后池化。FFN 沿用已有内部残差。

    Returns:
        [B, num_classes] 的 logits；训练时直接交给 CrossEntropyLoss。
    """

    def __init__(
        self,
        num_bands,
        num_classes,
        embed_dim=32,
        conv3d_channels=8,
        conv2d_channels=64,
        ffn_hidden_dim=64,
        dropout=0.1,
        psa_enabled=True,
        psa_num_heads=2,
        psa_residual_scale=0.1,
        psa_center_weighted=True,
        psa_center_sigma=3.0,
        psa_pre_norm=True,
        cnn_enabled=True,
        dam_enabled=False,
        dam_residual_scale=0.1,
        dam_d_state=8,
        dam_ssm_ratio=1.0,
        dam_dt_rank="auto",
        dam_direction_enabled=True,
        dam_scan_backend="auto",
        fusion_mode="parallel",
    ):
        super().__init__()
        if min(num_bands, num_classes, embed_dim, conv3d_channels,
               conv2d_channels, ffn_hidden_dim) < 1:
            raise ValueError("Band count, class count and feature dimensions must be positive")
        self.num_bands = num_bands
        self.num_classes = num_classes
        if fusion_mode not in ("parallel", "serial", "serial_sum"):
            raise ValueError("fusion_mode must be parallel, serial or serial_sum")
        self.fusion_mode = fusion_mode
        if not isinstance(cnn_enabled, bool):
            raise ValueError("cnn_enabled must be boolean")
        self.cnn_enabled = cnn_enabled
        feature_dim = conv2d_channels if cnn_enabled else embed_dim
        if not isinstance(psa_enabled, bool):
            raise ValueError("psa_enabled must be boolean")
        self.psa_enabled = psa_enabled
        self.register_parameter("psa_residual_scale", (
            nn.Parameter(torch.tensor(float(psa_residual_scale)), requires_grad=psa_enabled)
            if fusion_mode == "parallel" else None
        ))
        if not isinstance(dam_enabled, bool):
            raise ValueError("dam_enabled must be boolean")
        if not math.isfinite(float(dam_residual_scale)):
            raise ValueError("dam_residual_scale must be finite")
        self.dam_enabled = dam_enabled
        self.register_buffer("dam_residual_scale", (
            torch.tensor(float(dam_residual_scale))
            if dam_enabled and fusion_mode == "parallel" else None
        ))

        self.stem = nn.Sequential(
            nn.Conv2d(num_bands, embed_dim, kernel_size=3, padding=1),
            ChannelLayerNorm(embed_dim),
            nn.GELU(),
        )
        self.cnn = (
            ShallowCNNBlock(
                depth=embed_dim,
                conv3d_channels=conv3d_channels,
                conv2d_channels=conv2d_channels,
            )
            if cnn_enabled else nn.Identity()
        )
        # 隔离 PSA 初始化，便于开关 PSA 时保留相同种子的 CNN/FFN 初始权重。
        with torch.random.fork_rng(devices=[]):
            self.psa = (
                SingleLayerPSA(
                    feature_dim, num_heads=psa_num_heads, bias=True,
                    center_weighted=psa_center_weighted, center_sigma=psa_center_sigma,
                    pre_norm=psa_pre_norm,
                )
                if psa_enabled else nn.Identity()
            )
        self.pool = nn.AdaptiveAvgPool2d(1)
        # DAM 初始化不消耗已有 CNN/PSA/FFN 的随机数流，便于开关消融。
        with torch.random.fork_rng(devices=[]):
            if dam_enabled:
                from .zwk_DAM import ZWKDAM
                self.dam = ZWKDAM(
                    d_model=feature_dim, d_state=dam_d_state, ssm_ratio=dam_ssm_ratio,
                    dt_rank=dam_dt_rank, direction_enabled=dam_direction_enabled,
                    scan_backend=dam_scan_backend,
                )
            else:
                self.dam = nn.Identity()
        self.classifier = nn.Sequential(
            ResidualFFN(feature_dim, ffn_hidden_dim, dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x):
        if x.ndim == 5:
            if x.shape[1] != 1:
                raise ValueError("5D HSI input must have shape [B, 1, bands, H, W]")
            x = x.squeeze(1)
        if x.ndim != 4 or x.shape[1] != self.num_bands:
            raise ValueError(
                f"Expected [B, {self.num_bands}, H, W] or "
                f"[B, 1, {self.num_bands}, H, W], got {tuple(x.shape)}"
            )

        x = self.stem(x)                 # [B, embed_dim, H, W]
        if self.cnn_enabled:
            x = self.cnn(x.unsqueeze(1))  # [B, conv2d_channels, H, W]
        if self.fusion_mode == "serial":
            if self.psa_enabled:
                x = self._align_branch(self.psa(x), x, "PSA")
            if self.dam_enabled:
                x = self._align_branch(self.dam(x), x, "DAM")
        elif self.fusion_mode == "serial_sum":
            features = x
            psa = self._align_branch(self.psa(features), features, "PSA") if self.psa_enabled else None
            dam_input = psa if psa is not None else features
            dam = self._align_branch(self.dam(dam_input), dam_input, "DAM") if self.dam_enabled else None
            x = features
            if psa is not None:
                x = x + psa
            if dam is not None:
                x = x + dam
        else:
            features = x  # 两个分支均接收这份原特征，避免误接成串联。
            if self.psa_enabled:
                psa = self._align_branch(self.psa(features), features, "PSA")
                x = x + self.psa_residual_scale * psa
            if self.dam_enabled:
                dam = self._align_branch(self.dam(features), features, "DAM")
                x = x + self.dam_residual_scale * dam
        x = self.pool(x).flatten(1)       # [B, feature_dim]
        return self.classifier(x)        # [B, num_classes]

    @staticmethod
    def _align_branch(output, features, name):
        """输出匹配模块输入的空间/通道、设备和精度，禁止广播融合。"""
        if output.shape != features.shape or output.device != features.device:
            raise RuntimeError(f"{name} output must match input feature shape/device: "
                               f"{tuple(output.shape)} on {output.device} vs "
                               f"{tuple(features.shape)} on {features.device}")
        return output.to(dtype=features.dtype)

    def no_weight_decay(self):
        """仅对新增 DAM 的状态转移与直接通路参数关闭权重衰减。"""
        return {f"dam.{name}" for name in self.dam.no_weight_decay()} if self.dam_enabled else set()

    def flops(self, shape=None, verbose=True):
        """兼容训练入口的参数量/FLOPs 统计，不改变模型或随机数状态。

        shape 是完整输入形状，默认 [1, num_bands, 13, 13]。
        使用项目已有的 fvcore，按其约定一次乘加计一次操作；
        未支持的算子会在 verbose=True 时列出。
        """
        import copy
        from fvcore.nn import FlopCountAnalysis

        shape = (1, self.num_bands, 13, 13) if shape is None else tuple(shape)
        model = copy.deepcopy(self).eval()
        parameter = next(model.parameters())
        sample = torch.zeros(shape, device=parameter.device, dtype=parameter.dtype)
        with torch.no_grad():
            analysis = FlopCountAnalysis(model, (sample,))
            analysis.set_op_handle("aten::movedim", None)
            analysis.unsupported_ops_warnings(False).tracer_warnings("none")
            gflops = analysis.total() / 1e9
        if verbose:
            params = sum(p.numel() for p in model.parameters())
            print(f"params {params} GFLOPs {gflops:.6f}")
            unsupported = dict(analysis.unsupported_ops())
            if unsupported:
                print(f"FLOPs excludes unsupported operations: {unsupported}")
        return gflops

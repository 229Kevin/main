"""Cross-attention-guided variant of GPSMamba's dynamic mixed convolution.

Standalone patch classifier: HSI -> pointwise spectral projection -> one DMC
-> global average pooling -> FFN -> classifier (unnormalized class logits).

The original paper-based reconstruction mapped pixel-to-region affinities
directly to spatial kernels. This variant replaces only that kernel generator:
pixel Queries attend to pooled regional Keys/Values; a group-shared MLP maps
the concatenation of each pixel's raw features and retrieved context to a
dynamic spatial kernel. Dynamic convolution is added to the local RepConv.

Reconstruction choices where the paper does not specify implementation details:
* Retain the HW axis in Kd: [B, groups, HW, K*K]. Each position has its own
  kernel; channels within a group share that kernel.
* Q/K/V use three bias-free 1x1 projections C -> C; the head/group dimension
  is d=C/groups. Attention Softmax runs over S*S regional representatives.
* Kernel MLP: Linear(2*d, d) -> GELU -> Linear(d, K*K), shared across groups
  and positions. A SECOND Softmax runs over valid K*K spatial neighbors.
  S*S regions and K*K neighbor offsets have distinct meanings, even when
  their counts happen to match. Padded neighbors are masked before Softmax.
* No attention output projection, positional encoding, extra normalization,
  dropout or residual is added to the generator in this first variant.
* RepConv uses dilated 3x3+BN, 1x1+BN, and identity+BN training branches.
  Their sum is exactly reparameterizable in evaluation mode. Default dilation
  is 2. Folding follows the standard RepVGG Conv/BN algebra:
  https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
* Dynamic-branch BN and one GELU follow the two-branch addition. No additional
  input skip is used. A 1x1 projection maps raw spectral bands to 64 features;
  there is no separate CNN/PSA/DAM backbone and no FFN residual in this model.

This is a proposed architectural variant, not an official implementation or
an exact reproduction of the paper's equations or reported accuracy. Old
affinity-generator checkpoints are not strictly compatible; retrain this
version. The standalone network/config/registration interfaces are unchanged.
"""

import copy

import torch
from torch import nn
from torch.nn import functional as F


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _check_features(x, channels):
    if x.ndim != 4 or x.shape[1] != channels or min(x.shape) <= 0:
        raise ValueError(f"Expected nonempty [B, {channels}, H, W], got {tuple(x.shape)}")


class ChannelLayerNorm(nn.Module):
    """Normalize channels independently at each spatial location."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class GlobalKernelGenerator(nn.Module):
    """Regional cross-attention -> context-conditioned grouped spatial kernels.

    Input: [B, C, H, W]. Output: [B, groups, H*W, kernel_size**2].
    Attention heads align with the channel groups used by dynamic convolution.
    Projection weights mix all C channels before splitting into heads; kernel
    MLP weights are shared between heads, while their features/kernels differ.
    """

    def __init__(self, channels, groups=4, region_size=3, kernel_size=3):
        super().__init__()
        for name, value in (("channels", channels), ("groups", groups),
                            ("region_size", region_size), ("kernel_size", kernel_size)):
            _positive_integer(value, name)
        if channels % groups:
            raise ValueError("channels must be divisible by groups")
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.channels = channels
        self.groups = groups
        self.region_size = region_size
        self.kernel_size = kernel_size
        self.group_dim = channels // groups
        self.scale = self.group_dim ** -0.5
        self.pool = nn.AdaptiveAvgPool2d(region_size)
        self.query_projection = nn.Conv2d(channels, channels, 1, bias=False)
        self.key_projection = nn.Conv2d(channels, channels, 1, bias=False)
        self.value_projection = nn.Conv2d(channels, channels, 1, bias=False)
        self.kernel_mlp = nn.Sequential(
            nn.Linear(2 * self.group_dim, self.group_dim), nn.GELU(),
            nn.Linear(self.group_dim, kernel_size ** 2))

    def forward(self, x):
        _check_features(x, self.channels)
        batch, _, height, width = x.shape
        pixels, regions = height * width, self.region_size ** 2
        pooled = self.pool(x)
        queries = self.query_projection(x).reshape(
            batch, self.groups, self.group_dim, pixels).transpose(-2, -1)
        keys = self.key_projection(pooled).reshape(
            batch, self.groups, self.group_dim, regions)
        values = self.value_projection(pooled).reshape(
            batch, self.groups, self.group_dim, regions).transpose(-2, -1)
        local = x.reshape(batch, self.groups, self.group_dim, pixels).transpose(-2, -1)
        # Keep scores, retrieved content, MLP logits and both Softmax operations
        # in FP32 under AMP; preserve FP64 for numerical checks. Differentiable
        # weight casts also support explicitly half-precision parameters.
        compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            scores = (queries.to(compute_dtype) @ keys.to(compute_dtype)) * self.scale
            attention = scores.softmax(dim=-1)  # [B, G, HW, S*S], over regions
            context = attention @ values.to(compute_dtype)  # [B, G, HW, d]
            conditioning = torch.cat((local.to(compute_dtype), context), dim=-1)
            hidden_layer, activation, output_layer = self.kernel_mlp
            hidden = activation(F.linear(conditioning, hidden_layer.weight.to(compute_dtype),
                                         hidden_layer.bias.to(compute_dtype)))
            logits = F.linear(hidden, output_layer.weight.to(compute_dtype),
                              output_layer.bias.to(compute_dtype))
            valid = F.unfold(x.new_ones(1, 1, height, width, dtype=compute_dtype),
                             self.kernel_size, padding=self.kernel_size // 2)
            valid = valid.transpose(1, 2).unsqueeze(1).bool()
            # [B, G, HW, K*K], over valid LOCAL offsets rather than regions.
            weights = logits.masked_fill(~valid, float('-inf')).softmax(dim=-1)
        return weights.to(x.dtype)


def apply_grouped_dynamic_kernel(features, weights, kernel_size):
    """Apply [B, G, HW, K*K] kernels to [B, C, H, W], preserving its shape."""
    _positive_integer(kernel_size, "kernel_size")
    if kernel_size % 2 != 1:
        raise ValueError("kernel_size must be odd")
    if features.ndim != 4 or weights.ndim != 4:
        raise ValueError("features and weights must both have four dimensions")
    batch, channels, height, width = features.shape
    groups = weights.shape[1]
    if min(features.shape) <= 0 or groups <= 0 or channels % groups:
        raise ValueError("features must be nonempty and channels divisible by kernel groups")
    if weights.shape != (batch, groups, height * width, kernel_size ** 2):
        raise ValueError("Dynamic kernels do not match the input batch/spatial/kernel dimensions")
    neighborhoods = F.unfold(features, kernel_size, padding=kernel_size // 2)
    neighborhoods = neighborhoods.reshape(batch, groups, channels // groups,
                                           kernel_size ** 2, height * width)
    weights = weights.transpose(-2, -1).unsqueeze(2)
    return (neighborhoods * weights).sum(dim=3).reshape(batch, channels, height, width)


class RepConv(nn.Module):
    """Linear local branches, foldable into one dilated 3x3 convolution."""

    def __init__(self, channels, groups=1, dilation=2, deploy=False):
        super().__init__()
        for name, value in (("channels", channels), ("groups", groups), ("dilation", dilation)):
            _positive_integer(value, name)
        if channels % groups:
            raise ValueError("channels must be divisible by groups")
        self.channels = channels
        self.groups = groups
        self.dilation = dilation
        if deploy:
            self.reparam = nn.Conv2d(channels, channels, 3, padding=dilation,
                                    dilation=dilation, groups=groups, bias=True)
        else:
            self.spatial = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation,
                          groups=groups, bias=False), nn.BatchNorm2d(channels))
            self.pointwise = nn.Sequential(
                nn.Conv2d(channels, channels, 1, groups=groups, bias=False),
                nn.BatchNorm2d(channels))
            self.identity = nn.BatchNorm2d(channels)

    def forward(self, x):
        if hasattr(self, 'reparam'):
            return self.reparam(x)
        return self.spatial(x) + self.pointwise(x) + self.identity(x)

    @staticmethod
    def _fold(kernel, bn):
        scale = bn.weight / (bn.running_var + bn.eps).sqrt()
        return kernel * scale[:, None, None, None], bn.bias - bn.running_mean * scale

    def switch_to_deploy(self):
        """Mutate an eval-mode block; instantiate with deploy=True to reload it."""
        if self.training:
            raise RuntimeError("Call eval() before converting RepConv to deployment")
        if hasattr(self, 'reparam'):
            return self
        spatial, spatial_bias = self._fold(self.spatial[0].weight, self.spatial[1])
        pointwise, pointwise_bias = self._fold(self.pointwise[0].weight, self.pointwise[1])
        identity = torch.zeros_like(spatial)
        indices = torch.arange(self.channels, device=identity.device)
        identity[indices, indices % (self.channels // self.groups), 1, 1] = 1
        identity, identity_bias = self._fold(identity, self.identity)
        kernel = spatial + F.pad(pointwise, (1, 1, 1, 1)) + identity
        bias = spatial_bias + pointwise_bias + identity_bias
        # Constructing a new Conv initializes weights. Preserve the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            reparam = nn.Conv2d(self.channels, self.channels, 3, padding=self.dilation,
                               dilation=self.dilation, groups=self.groups, bias=True)
        self.reparam = reparam.to(device=kernel.device, dtype=kernel.dtype)
        with torch.no_grad():
            self.reparam.weight.copy_(kernel)
            self.reparam.bias.copy_(bias)
        self.reparam.train(self.training)
        del self.spatial, self.pointwise, self.identity
        return self


class DMCBlock(nn.Module):
    """Add the globally conditioned dynamic branch and local RepConv branch."""

    def __init__(self, channels, groups=4, region_size=3, kernel_size=3,
                 local_dilation=2, deploy=False):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.kernel_generator = GlobalKernelGenerator(channels, groups, region_size, kernel_size)
        self.dynamic_norm = nn.BatchNorm2d(channels)
        self.local = RepConv(channels, dilation=local_dilation, deploy=deploy)
        self.activation = nn.GELU()

    def forward(self, x):
        weights = self.kernel_generator(x)
        dynamic = apply_grouped_dynamic_kernel(x, weights, self.kernel_size)
        return self.activation(self.dynamic_norm(dynamic) + self.local(x))


class ZWKDMC(nn.Module):
    """Single attention-guided DMC classifier; outputs [B, num_classes] logits."""

    def __init__(self, num_bands=200, num_classes=16, feature_dim=64, groups=4,
                 region_size=3, kernel_size=3, local_dilation=2, ffn_hidden_dim=64,
                 dropout=0.1, deploy=False):
        super().__init__()
        for name, value in (("num_bands", num_bands), ("num_classes", num_classes),
                            ("feature_dim", feature_dim), ("ffn_hidden_dim", ffn_hidden_dim)):
            _positive_integer(value, name)
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.num_bands = num_bands
        self.num_classes = num_classes
        self.input_projection = nn.Sequential(
            nn.Conv2d(num_bands, feature_dim, 1, bias=False),
            ChannelLayerNorm(feature_dim), nn.GELU())
        self.dmc = DMCBlock(feature_dim, groups, region_size, kernel_size, local_dilation, deploy)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.ffn = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, ffn_hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_hidden_dim, feature_dim))
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        if x.ndim == 5:
            if x.shape[1] != 1:
                raise ValueError("Five-dimensional HSI input must have shape [B, 1, bands, H, W]")
            x = x[:, 0]
        _check_features(x, self.num_bands)
        x = self.dmc(self.input_projection(x))
        return self.classifier(self.ffn(self.pool(x).flatten(1)))

    def switch_to_deploy(self):
        if self.training:
            raise RuntimeError("Call eval() before converting the classifier to deployment")
        self.dmc.local.switch_to_deploy()
        return self

    def flops(self, shape=None, verbose=True):
        """Training-entry-compatible FVCore count (one multiply-add is one op).

        FVCore counts Conv/Linear/affinity matmul and normalization. Include
        dynamic weighted reduction explicitly; reshaping/gathering cost zero.
        Elementwise activations/Softmax and the nonlearned boundary mask are
        excluded, as in common convolutional-model FLOP reporting.
        """
        from fvcore.nn import FlopCountAnalysis

        shape = tuple(shape or (1, self.num_bands, 15, 15))
        model = copy.deepcopy(self).eval()
        parameter = next(model.parameters())
        sample = torch.zeros(shape, device=parameter.device, dtype=parameter.dtype)
        analysis = FlopCountAnalysis(model, sample)
        analysis.set_op_handle(**{
            'aten::im2col': None, 'aten::mul': None, 'aten::sum': None,
            'aten::softmax': None, 'aten::gelu': None, 'aten::pad': None,
            'aten::add': None, 'aten::sub': None, 'aten::reciprocal': None,
            'aten::masked_fill': None, 'aten::bitwise_not': None,
        })
        analysis.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        with torch.no_grad():
            total = analysis.total()
        batch, height, width = shape[0], shape[-2], shape[-1]
        # Each output channel/location multiplies K*K samples by its dynamic kernel.
        total += batch * self.dmc.channels * height * width * self.dmc.kernel_size ** 2
        if verbose:
            print(f"DMC FLOPs (Conv/Linear/matmul/normalization/dynamic reduction): {total / 1e9:.4f} GFLOPs")
        return float(total)

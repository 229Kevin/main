"""Standalone, patch-oriented Direction-Aware Mamba (DAM) adaptation.

Input/output: [B, C, H, W]. This module returns ONLY an enhancement feature;
the caller owns residual scaling and fusion with the CNN/PSA features.

Architecture:
    channel LayerNorm -> shared 1x1 projection -> feature/gate split
    -> 3x3 spatial depthwise convolution + SiLU
    -> row/column serpentine paths and their reversals
    -> four independent selective SSM parameter groups
    -> inverse permutations + mean -> LayerNorm -> SiLU gate -> 1x1 projection

Direction compensation is the lightweight, kernel-compatible adaptation:
    B_eff[t] = B[t] + theta[arrival_direction[t]]
    h[t] = exp(delta[t] * A) * h[t-1] + delta[t] * B_eff[t] * u[t]
    y[t] = C[t] * h[t] + D_skip * u[t]
The five labels (start/up/down/left/right), mean aggregation, expansion=1,
and absence of extra positional encoding are OUR explicit adaptation choices.
This is not a claim to reproduce GPSMamba's exact ZOH discretization of theta.

Only PyTorch is needed for the CPU reference path. CUDA lazily imports the
already installed mamba_ssm selective_scan_fn; no custom kernels are compiled.
SSM initialization follows the existing OSSM/Mamba implementation.

Example:
    module = ZWKDAM(d_model=64)
    enhanced = module(torch.randn(2, 64, 15, 15))
    # Later, outside this module: fused = cnn + alpha * psa + beta * enhanced
"""

import math
import warnings

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ['ZWKDAM', 'build_scan_indices', 'selective_scan_reference']


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def build_scan_indices(height, width, device=None):
    """Return [4,H*W] permutations, inverses and arrival direction labels.

    Path order: row snake, reversed row snake, column snake, reversed column.
    Labels: 0=start, 1=up, 2=down, 3=left, 4=right. Each reversed path computes
    its own movement vectors; reversing the forward labels would be incorrect.
    """
    _positive_integer(height, 'height')
    _positive_integer(width, 'width')
    grid = torch.arange(height * width, device=device).reshape(height, width)
    row_grid = grid.clone()
    row_grid[1::2] = row_grid[1::2].flip(-1)
    col_grid = grid.transpose(0, 1).clone()
    col_grid[1::2] = col_grid[1::2].flip(-1)
    row, col = row_grid.reshape(-1), col_grid.reshape(-1)
    orders = torch.stack((row, row.flip(0), col, col.flip(0)))
    inverses = orders.argsort(dim=-1)
    directions = torch.zeros_like(orders)
    row_step = orders[:, 1:] // width - orders[:, :-1] // width
    col_step = orders[:, 1:] % width - orders[:, :-1] % width
    labels = directions[:, 1:]
    labels[row_step == -1] = 1
    labels[row_step == 1] = 2
    labels[col_step == -1] = 3
    labels[col_step == 1] = 4
    return orders, inverses, directions


def selective_scan_reference(u, delta, a, state_b, state_c, skip):
    """Differentiable reference recurrence, with explicit scan groups.

    u/delta: [B,K,D,L]; a: [K,D,N]; B/C: [B,K,N,L]; skip: [K,D].
    delta is already positive; B already includes optional direction correction.
    Every call starts from zero state. This path is for CPU/numerical checking,
    not production GPU throughput. FP64 is retained for numerical tests.
    """
    batch, paths, channels, length = u.shape
    state = u.new_zeros(batch, paths, channels, a.shape[-1])
    outputs = []
    for index in range(length):
        step = delta[..., index, None]
        decay = torch.exp(step * a.unsqueeze(0))
        write = step * state_b[..., index].unsqueeze(2) * u[..., index, None]
        state = decay * state + write
        read = (state * state_c[..., index].unsqueeze(2)).sum(-1)
        outputs.append(read + skip.unsqueeze(0) * u[..., index])
    return torch.stack(outputs, dim=-1)


class ZWKDAM(nn.Module):
    """One spatial DAM layer, preserving channels and spatial resolution.

    Defaults are adapted to the current CNN output [B,64,15,15]:
    d_model=64, d_inner=64, d_state=8, dt_rank=4, four scan paths.

    scan_backend:
        auto: CUDA Mamba kernel on CUDA when available, reference otherwise.
        reference: explicit differentiable recurrence (also supports FP64).
        cuda: require CUDA and the installed Mamba kernel, fail if unavailable.

    direction_enabled=False removes theta from state writing, allowing a
    controlled comparison against the SAME four-path ordinary spatial Mamba.
    Direction parameters are zero-initialized and frozen when disabled.
    No internal outer residual, residual coefficient, downsampling, extra FFN,
    or positional encoding is added. Per-path parameters are independent;
    input convolution, gate, and output projection are shared across paths.
    """

    def __init__(self, d_model=64, d_state=8, ssm_ratio=1.0, dt_rank='auto',
                 d_conv=3, dropout=0.0, direction_enabled=True,
                 scan_backend='auto', dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = _positive_integer(d_model, 'd_model')
        self.d_state = _positive_integer(d_state, 'd_state')
        if not math.isfinite(ssm_ratio) or ssm_ratio <= 0:
            raise ValueError('ssm_ratio must be positive and finite')
        self.d_inner = _positive_integer(int(d_model * ssm_ratio), 'd_inner')
        self.dt_rank = (math.ceil(d_model / 16) if dt_rank == 'auto'
                        else _positive_integer(dt_rank, 'dt_rank'))
        _positive_integer(d_conv, 'd_conv')
        if d_conv % 2 == 0:
            raise ValueError('d_conv must be odd to preserve spatial size')
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError('dropout must be finite and in [0,1)')
        if not isinstance(direction_enabled, bool):
            raise ValueError('direction_enabled must be boolean')
        if scan_backend not in ('auto', 'reference', 'cuda'):
            raise ValueError('scan_backend must be auto, reference or cuda')
        if not (math.isfinite(dt_min) and math.isfinite(dt_max)
                and 0 < dt_min < dt_max):
            raise ValueError('Require finite 0 < dt_min < dt_max')
        self.direction_enabled = direction_enabled
        self.scan_backend = scan_backend
        self.num_paths = 4

        self.input_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.in_proj = nn.Conv2d(d_model, 2 * self.d_inner, 1, bias=False)
        self.dwconv = nn.Conv2d(self.d_inner, self.d_inner, d_conv,
                                padding=d_conv // 2, groups=self.d_inner, bias=True)
        self.x_proj_weight = nn.Parameter(torch.empty(
            self.num_paths, self.dt_rank + 2 * d_state, self.d_inner))
        nn.init.uniform_(self.x_proj_weight, -self.d_inner ** -0.5,
                         self.d_inner ** -0.5)
        self.dt_proj_weight = nn.Parameter(torch.empty(
            self.num_paths, self.d_inner, self.dt_rank))
        nn.init.uniform_(self.dt_proj_weight, -self.dt_rank ** -0.5,
                         self.dt_rank ** -0.5)
        dt = torch.exp(torch.rand(self.num_paths, self.d_inner)
                       * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        self.dt_proj_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_proj_bias._no_reinit = True
        self.A_log = nn.Parameter(torch.arange(1, d_state + 1, dtype=torch.float32)
                                  .log().repeat(self.num_paths, self.d_inner, 1))
        self.D_skip = nn.Parameter(torch.ones(self.num_paths, self.d_inner))
        self.A_log._no_weight_decay = True
        self.D_skip._no_weight_decay = True
        self.direction_table = nn.Parameter(torch.zeros(5, d_state),
                                            requires_grad=direction_enabled)
        self.output_norm = nn.LayerNorm(self.d_inner, eps=1e-5)
        self.out_proj = nn.Conv2d(self.d_inner, d_model, 1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

        for name in ('_orders', '_inverses', '_directions'):
            self.register_buffer(name, torch.empty(4, 0, dtype=torch.long), persistent=False)
        self._cached_shape = None
        self._cuda_scan = None
        self._cuda_import_attempted = False

    def no_weight_decay(self):
        """Names for a future optimizer group; flags alone do not configure AdamW."""
        return {'A_log', 'D_skip'}

    def _indices(self, height, width, device):
        if self._cached_shape != (height, width) or self._orders.device != device:
            self._orders, self._inverses, self._directions = build_scan_indices(
                height, width, device=device)
            self._cached_shape = (height, width)
        return self._orders, self._inverses, self._directions

    def _resolve_cuda_scan(self):
        if not self._cuda_import_attempted:
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
                import selective_scan_cuda  # Ensure the actual extension is installed.
                self._cuda_scan = selective_scan_fn
            except (ImportError, OSError) as error:
                if self.scan_backend == 'cuda':
                    raise RuntimeError('CUDA DAM requires the installed mamba_ssm '
                                       'and selective_scan_cuda extension') from error
                warnings.warn(f'DAM uses the slower reference scan: {error}', RuntimeWarning)
            self._cuda_import_attempted = True
        if self._cuda_scan is None and self.scan_backend == 'cuda':
            raise RuntimeError('CUDA selective scan is unavailable')
        return self._cuda_scan

    def _scan(self, features, direction_ids):
        # Cast scan parameters and recurrence to FP32 under mixed precision.
        # Preserve FP64 only for the explicit numerical reference path.
        dtype = torch.float64 if features.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=features.device.type, enabled=False):
            u = features.to(dtype)
            projected = torch.einsum('bkdl,kpd->bkpl', u, self.x_proj_weight.to(dtype))
            dt, state_b, state_c = torch.split(
                projected, (self.dt_rank, self.d_state, self.d_state), dim=2)
            delta = torch.einsum('bkrl,kdr->bkdl', dt, self.dt_proj_weight.to(dtype))
            delta = F.softplus(delta + self.dt_proj_bias.to(dtype)[None, :, :, None])
            if self.direction_enabled:
                correction = F.embedding(direction_ids, self.direction_table.to(dtype))
                state_b = state_b + correction.permute(0, 2, 1).unsqueeze(0)
            a, skip = -self.A_log.to(dtype).exp(), self.D_skip.to(dtype)
            use_cuda = self.scan_backend != 'reference' and u.is_cuda and dtype != torch.float64
            if self.scan_backend == 'cuda' and not use_cuda:
                raise ValueError('scan_backend=cuda requires CUDA input with FP32/FP16/BF16 dtype')
            cuda_scan = self._resolve_cuda_scan() if use_cuda else None
            if cuda_scan is None:
                return selective_scan_reference(u, delta, a, state_b, state_c, skip)
            batch, paths, channels, length = u.shape
            output = cuda_scan(
                u.reshape(batch, paths * channels, length).contiguous(),
                delta.reshape(batch, paths * channels, length).contiguous(),
                a.reshape(paths * channels, self.d_state).contiguous(),
                state_b.contiguous(), state_c.contiguous(), skip.reshape(-1).contiguous(),
                delta_softplus=False,
            )
            return output.reshape(batch, paths, channels, length)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.d_model or min(x.shape[0], *x.shape[2:]) < 1:
            raise ValueError(f'Expected nonempty [B,{self.d_model},H,W], got {tuple(x.shape)}')
        # Index permutations are static for each traced patch shape. JIT shape
        # values can be scalar tensors; convert before integer validation/cache.
        height, width = int(x.shape[-2]), int(x.shape[-1])
        orders, inverses, directions = self._indices(height, width, x.device)
        normalized = self.input_norm(x.movedim(1, -1)).movedim(-1, 1).contiguous()
        features, gate = self.in_proj(normalized).chunk(2, dim=1)
        features = F.silu(self.dwconv(features))
        flat = features.flatten(2)
        sequences = torch.stack([flat.index_select(-1, order) for order in orders], dim=1)
        outputs = self._scan(sequences, directions)
        aligned = torch.stack([outputs[:, path].index_select(-1, inverses[path])
                               for path in range(self.num_paths)], dim=1)
        merged = aligned.mean(dim=1).reshape(x.shape[0], self.d_inner, height, width)
        merged = merged.to(features.dtype)
        merged = self.output_norm(merged.movedim(1, -1)).movedim(-1, 1).contiguous()
        return self.dropout(self.out_proj(merged * F.silu(gate)))

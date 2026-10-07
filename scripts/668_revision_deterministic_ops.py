"""Deterministic adjoints for R3's fixed-coordinate spatial operations.

Native forward values are retained. Backward sums use deterministic PyTorch
index/scatter-add with strict mode enabled; never silently disable that mode.
The grid contract intentionally excludes learned coordinates and higher-order
derivatives, neither of which is used by R3.
"""
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.function import once_differentiable

ROOT = Path(__file__).resolve().parents[1]
NATIVE_GRID = F.grid_sample
NATIVE_RESIZE = F.interpolate
NATIVE_POOL = F.adaptive_avg_pool2d


def configure():
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('set CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts')
    if os.environ.get('PYTHONHASHSEED') != '42':
        raise ValueError('set PYTHONHASHSEED=42 before Python starts')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def require_strict():
    if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
        raise RuntimeError('deterministic adjoints require strict deterministic mode')


def grid_terms(coordinates, height, width, mode):
    # Match grid_sample's normalization and border clipping (align_corners=False).
    grid = coordinates * 2 - 1
    xy = ((grid + 1) * grid.new_tensor([width, height]) - 1) / 2
    x, y = xy[..., 0].clamp(0, width - 1), xy[..., 1].clamp(0, height - 1)
    if mode == 'nearest':
        return [(y.round().long() * width + x.round().long(), torch.ones_like(x))]
    x0, y0 = x.floor().long(), y.floor().long()
    x1, y1 = (x0 + 1).clamp_max(width - 1), (y0 + 1).clamp_max(height - 1)
    dx, dy = x - x0, y - y0
    return [(y0 * width + x0, (1 - dx) * (1 - dy)),
            (y0 * width + x1, dx * (1 - dy)),
            (y1 * width + x0, (1 - dx) * dy),
            (y1 * width + x1, dx * dy)]


class FixedGrid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, coordinates, mode):
        if features.ndim != 4 or coordinates.ndim != 3 or coordinates.shape[-1] != 2:
            raise ValueError('expected NCHW features and NP2 coordinates')
        if coordinates.requires_grad or mode not in ('bilinear', 'nearest'):
            raise ValueError('only fixed coordinates with bilinear/nearest sampling are supported')
        if coordinates.shape[0] != features.shape[0] or coordinates.device != features.device or coordinates.dtype != features.dtype:
            raise ValueError('batch, device and dtype must match')
        if not torch.isfinite(coordinates).all():
            raise ValueError('nonfinite sampling coordinate')
        ctx.shape, ctx.mode = features.shape, mode
        ctx.save_for_backward(coordinates)
        return NATIVE_GRID(features, coordinates[:, None] * 2 - 1,
                           mode=mode, padding_mode='border', align_corners=False)[:, :, 0]

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        require_strict()
        coordinates, = ctx.saved_tensors
        batch, channels, height, width = ctx.shape
        result = gradient.new_zeros(batch, channels, height * width)
        for index, weight in grid_terms(coordinates, height, width, ctx.mode):
            result.scatter_add_(2, index[:, None].expand(-1, channels, -1), gradient * weight[:, None])
        return result.reshape(ctx.shape), None, None


def sample_points(features, coordinates, mode='bilinear'):
    return FixedGrid.apply(features, coordinates, mode)


def axis_terms(size_in, size_out, device, dtype):
    source = ((torch.arange(size_out, device=device, dtype=dtype) + .5) * (size_in / size_out) - .5).clamp_min(0)
    lower = source.floor().long().clamp_max(size_in - 1)
    upper = (lower + 1).clamp_max(size_in - 1)
    fraction = source - lower
    return lower, upper, 1 - fraction, fraction


class BilinearResize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, size):
        ctx.shape, ctx.size = value.shape, size
        return NATIVE_RESIZE(value, size=size, mode='bilinear', align_corners=False)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        require_strict()
        batch, channels, height, width = ctx.shape
        oh, ow = ctx.size
        x0, x1, wx0, wx1 = axis_terms(width, ow, gradient.device, gradient.dtype)
        y0, y1, wy0, wy1 = axis_terms(height, oh, gradient.device, gradient.dtype)
        horizontal = gradient.new_zeros(batch, channels, oh, width)
        horizontal.index_add_(3, x0, gradient * wx0[None, None, None])
        horizontal.index_add_(3, x1, gradient * wx1[None, None, None])
        result = gradient.new_zeros(ctx.shape)
        result.index_add_(2, y0, horizontal * wy0[None, None, :, None])
        result.index_add_(2, y1, horizontal * wy1[None, None, :, None])
        return result, None


def interpolate(value, size=None, scale_factor=None, mode='nearest', align_corners=None,
                recompute_scale_factor=None, antialias=False):
    if value.ndim == 4 and mode == 'bilinear' and torch.is_grad_enabled() and value.requires_grad:
        if align_corners not in (False, None) or scale_factor is not None or antialias or recompute_scale_factor:
            raise ValueError('R3 deterministic resize supports explicit size, align_corners=False, no antialias')
        shape = (size, size) if isinstance(size, int) else tuple(size)
        return BilinearResize.apply(value, shape)
    return NATIVE_RESIZE(value, size=size, scale_factor=scale_factor, mode=mode,
                        align_corners=align_corners, recompute_scale_factor=recompute_scale_factor, antialias=antialias)


class DivisiblePool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, size):
        height, width = value.shape[-2:]
        if height % size[0] or width % size[1]:
            raise ValueError('R3 pool requires integer, nonoverlapping cells')
        ctx.factors = height // size[0], width // size[1]
        return NATIVE_POOL(value, size)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        kh, kw = ctx.factors
        return gradient.repeat_interleave(kh, 2).repeat_interleave(kw, 3) / (kh * kw), None


def adaptive_avg_pool2d(value, output_size):
    if torch.is_grad_enabled() and value.requires_grad:
        size = (output_size, output_size) if isinstance(output_size, int) else tuple(output_size)
        return DivisiblePool.apply(value, size)
    return NATIVE_POOL(value, output_size)


def stable_topk(values, count):
    return torch.argsort(values, dim=1, descending=True, stable=True)[:, :count]


def install(architecture):
    """Process-local wrappers cover SMP's resize calls as well as the R3 head."""
    if F.interpolate is not NATIVE_RESIZE or F.adaptive_avg_pool2d is not NATIVE_POOL:
        raise RuntimeError('spatial wrappers already installed')
    F.interpolate, F.adaptive_avg_pool2d = interpolate, adaptive_avg_pool2d
    architecture.sample_points = sample_points

    def shared_points(coarse, objects, dense, count, training):
        batch, queries, height, width = coarse.shape
        count = min(count, height * width)
        candidates = (torch.rand(batch, count * 3, 2, device=coarse.device) if training else
                      architecture.index_coordinates(torch.arange(height * width, device=coarse.device)[None].expand(batch, -1), height, width))
        p = sample_points(coarse, candidates).sigmoid()
        scores = objects.sigmoid()[..., None]
        priority = (4 * p * (1 - p) * scores).amax(1) + (p * scores).topk(min(2, queries), dim=1).values[:, -1]
        priority = priority + .25 * sample_points(dense, candidates).sigmoid()[:, 0]
        top_count = int(count * .75)
        top = stable_topk(priority, top_count)
        available = torch.ones_like(priority, dtype=torch.bool).scatter(1, top, False)
        if training:
            coverage = torch.rand_like(priority)
        else:
            k = torch.arange(priority.shape[1], device=priority.device, dtype=priority.dtype) + .5
            coverage = (k * .61803398875).remainder(1)[None].expand(batch, -1)
        rest = stable_topk(coverage.masked_fill(~available, -torch.inf), count - top_count)
        chosen = torch.cat((top, rest), 1)
        return candidates.gather(1, chosen[..., None].expand(-1, -1, 2))
    architecture.select_shared_points = shared_points

    def find_forward(self, quarter, fine, dense):
        batch, _, height, width = quarter.shape
        pixels = adaptive_avg_pool2d(fine, (height, width))
        indices = torch.arange(height * width, device=quarter.device)[None].expand(batch, -1)
        memory = quarter.flatten(2).transpose(1, 2) + self.position(architecture.index_coordinates(indices, height, width))
        seed_score = dense[:, 0].detach()
        maxima = seed_score == F.max_pool2d(seed_score[:, None], 9, 1, 4)[:, 0]
        seeds = stable_topk((seed_score + maxima.to(seed_score.dtype) * 100).flatten(1), self.query_count)
        coordinates = architecture.index_coordinates(seeds, *fine.shape[-2:])
        query = (sample_points(quarter, coordinates).transpose(1, 2) + self.position(coordinates)
                 + .1 * self.seed_fine_projection(sample_points(fine, coordinates).transpose(1, 2)) + self.queries[None])
        auxiliary = []
        for index, layer in enumerate(self.layers):
            query = layer(query, memory)
            coarse = torch.einsum('bqc,bchw->bqhw', self.mask_embed(query), pixels)
            if index < len(self.layers) - 1:
                auxiliary.append((coarse, self.objectness(query).squeeze(-1)))
        return query, coarse, self.objectness(query).squeeze(-1), auxiliary, coordinates
    architecture.FindInstances.forward = find_forward


def configuration():
    return {'strict': True, 'cublas': ':4096:8', 'tf32': False, 'attention': 'math',
            'spatial_forward': 'native', 'grid_backward': 'four deterministic scatter_add sums',
            'resize_backward': 'separable deterministic index_add sums', 'pool_backward': 'nonoverlapping repeat',
            'tie_order': 'descending value, ascending original index',
            'rng': 'original R3 global torch schedule; dedicated data random.Random; full states checkpointed',
            'optimizer': 'AdamW foreach=False fused=False', 'no_new_parameters': True}

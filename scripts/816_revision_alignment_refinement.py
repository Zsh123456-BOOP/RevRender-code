"""Isolated score-target and sparse ROI-context experiments on split-B r50_ctrl.

score03 changes ONLY quality-label binarization from .5 to delivery's .3.
render03 additionally computes detached quality features/labels on the same
ROI subdivision/pasting path used at inference. The detector/renderer retain
their original losses and ownership; quality gradients remain detached.
point_ms adds a zero-initialized multiscale residual at existing sparse ROI points.
point_context adds cross-attention to 32 spatial memory points in the same ROI.
The latter pair isolates cross-point context from simply adding multiscale input.
These are mechanism adaptations, not a reproduction of Mask Transfiner's quadtree
or a learned error-region selector. Existing uncertainty point selection is kept.

The old architecture and scripts are untouched. No test split is opened here.
"""
import importlib.util
import os
import sys
from pathlib import Path
from types import MethodType

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('off', 'score03', 'render03', 'point_ms', 'point_context')
VARIANT = os.environ.get('HME_ALIGN_REFINE', 'off')
if VARIANT not in VARIANTS:
    raise ValueError('undeclared alignment/refinement variant')
os.environ['HME_MECHANISM'] = 'off'
os.environ['HME_MECHANISM_BASE'] = 'ctrl'
RESIDUAL_SEED = 81601
QUALITY_LEVEL = .3 if VARIANT in ('score03', 'render03') else .5


def load(name):
    spec = importlib.util.spec_from_file_location('align816_'+name[:3], ROOT/'scripts'/name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load('775_revision_mechanism_trials.py')
COARSE, TRANSFER, CORE, ARMS, RULE = BASE.COARSE, BASE.TRANSFER, BASE.CORE, BASE.ARMS, BASE.RULE
SCORE = COARSE.BASE.SCORE
losses = BASE.losses
_original_overlap = SCORE.overlap_targets


def aligned_overlap(masks, targets, level=None):
    if level is not None and level != QUALITY_LEVEL:
        raise ValueError('quality target level disagrees with the declared variant')
    return _original_overlap(masks, targets, level=QUALITY_LEVEL)


if VARIANT in ('score03', 'render03'):
    SCORE.overlap_targets = aligned_overlap


class PointResidual(nn.Module):
    """32-dimensional tokens; queries attend only to 32 fixed points in their ROI."""
    def __init__(self, contextual):
        super().__init__()
        self.contextual = contextual
        self.encode = nn.Sequential(nn.Linear(99, 32), nn.ReLU())
        # Construct shared layers first, so point_ms/context have identical common weights.
        self.output = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
        if contextual:
            self.query = nn.Linear(32, 32, bias=False)
            self.key = nn.Linear(32, 32, bias=False)
            self.value = nn.Linear(32, 32, bias=False)
            self.norm = nn.LayerNorm(32)
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, features, memory=None):
        token = self.encode(features)
        if self.contextual:
            if memory is None:
                raise ValueError('contextual points require ROI memory')
            mem = self.encode(memory)
            attention = (self.query(token) @ self.key(mem).transpose(-1, -2))/(32**.5)
            token = self.norm(token+attention.softmax(-1) @ self.value(mem))
        return self.output(token).squeeze(-1)


def point_features(draw, coarse, coordinates):
    b, q, p = coordinates.shape[:3]
    features = [CORE.sample_points(f, coordinates.reshape(b, q*p, 2)).transpose(1, 2).reshape(b, q, p, 32)
                for f in draw._alignment_pyramid]
    mask = TRANSFER.sample_learned(coarse.reshape(b*q, 1, *coarse.shape[-2:]),
                                   coordinates.reshape(b*q, p, 2)).reshape(b, q, p, 1)
    roi = draw._alignment_roi
    relative = (coordinates-roi[:, :, None, :2])/(roi[:, :, None, 2:]-roi[:, :, None, :2]).clamp_min(1e-5)
    return torch.cat([*features, mask, relative], -1)


def install_point_residual(draw, contextual):
    # Preserve the old point head's parameters, initialization and RNG stream.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(RESIDUAL_SEED)
        draw.alignment_residual = PointResidual(contextual)
    original_forward, original_points = draw.forward, draw.at_points

    def forward(self, coarse, fine, boxes, size):
        self._alignment_pyramid = (fine, CORE.F.avg_pool2d(fine, 2), CORE.F.avg_pool2d(fine, 4))
        self._alignment_roi = TRANSFER.expanded_roi(boxes, size)
        self._alignment_memory = None
        try:
            return original_forward(coarse, fine, boxes, size)
        finally:
            # Tensors returned by forward retain their graphs; do not retain them on the module.
            self._alignment_pyramid = self._alignment_roi = self._alignment_memory = None

    def at_points(self, coarse, fine, coordinates):
        base = original_points(coarse, fine, coordinates)
        features = point_features(self, coarse, coordinates)
        memory = None
        if contextual:
            if self._alignment_memory is None:
                roi = self._alignment_roi
                grid = TRANSFER.grid(4, 8, coarse)[None]
                coords = roi[:, :, None, :2]+grid*(roi[:, :, None, 2:]-roi[:, :, None, :2])
                self._alignment_memory = point_features(self, coarse, coords)
            memory = self._alignment_memory
        return base+self.alignment_residual(features, memory)

    draw.forward = MethodType(forward, draw)
    draw.at_points = MethodType(at_points, draw)


def inference_render(draw, coarse, fine, boxes, size):
    modes = [(module, module.training) for module in draw.modules()]
    try:
        draw.eval()
        with torch.no_grad():
            return draw(coarse.detach(), fine.detach(), boxes.detach(), size)[0]
    finally:
        for module, training in modes:
            module.training = training


def install_rendered_quality(model):
    original = model.forward

    def forward(self, image, *args):
        result = original(image, *args)
        if self.training:
            inputs = self._mask_iou_inputs
            rendered = inference_render(self.draw, self.cache['coarse'], inputs['fine'],
                                        self.cache['boxes'], image.shape[-2:])
            pixels = CORE.F.adaptive_avg_pool2d(inputs['fine'], self.cache['coarse'].shape[-2:])
            self.cache['mask_iou'] = self.find.mask_iou(inputs['query'], self.cache['coarse'], pixels,
                                                     rendered, detach=True)
            self.cache['mask_iou_delivered'] = rendered
        return result

    model.forward = MethodType(forward, model)


def build_model(baseline, args):
    model = BASE.build_model(baseline, args)
    if VARIANT.startswith('point_'):
        install_point_residual(model.draw, VARIANT == 'point_context')
    if VARIANT == 'render03':
        install_rendered_quality(model)
    return model


def configuration(arm):
    return dict(BASE.configuration(arm), alignment_refinement={
        'variant': VARIANT, 'quality_target_level': QUALITY_LEVEL, 'delivery_mask_level': .3,
        'quality_target_source': 'inference_roi_detached' if VARIANT == 'render03' else 'training_coarse',
        'quality_gradients_to_trunk': False, 'matching_and_denoising': 'unchanged',
        'point_selection': 'original uncertainty plus random training points',
        'point_scales': [1, 2, 4] if VARIANT.startswith('point_') else [1],
        'roi_memory_points': 32 if VARIANT == 'point_context' else 0,
        'residual_seed': RESIDUAL_SEED, 'residual_zero_initialization': True,
        'scope': 'component adaptation, not complete Mask Transfiner'} )

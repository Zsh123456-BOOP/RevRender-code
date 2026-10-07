"""Rebuild v2 of point_context: a context encoder, a dense-guided renderer, one-to-many training.

outputs/revision_rebuild_v2/design.md (approved 2026-09-27) splits the corrected
model's 123 misses and 87 false alarms into three structural failures and gives
each one module.  ``HME_REBUILD`` picks the modules: ``off`` (point_context,
which must replay 816 value for value), ``m1``, ``m2``, ``m3`` or ``all``.

  M1  context encoder.  674 scores 6,120 multi-scale feature positions for its
      32 proposals after one linear layer -- no encoder at all (MaskDINO runs
      six deformable layers).  Three pre-norm multi-scale deformable layers
      (674's DeformableRead as self-attention, each position reading 4 points on
      each of 4 levels around its own anchor, 4 heads; FFN 640) now run on the
      levels before the finder sees them; output projections start at zero, so
      the encoder starts as the identity.  Layers are checkpointed to bound
      memory; recomputation changes no value.
  M2  dense-guided renderer.  The see stage's full-resolution revision map
      (trained every step, dropped at inference) becomes evidence for the ROI
      point head: at each point it reads the map's logit and its 3x3 and 7x7
      maxima (the maxima from the detached map), through a small MLP whose last
      layer starts at zero, added to the point logit.  The quarter-level
      features (encoded when M1 is on) reach the full-resolution detail map
      through a zero-initialised 1x1 projection.  The rendered path's point
      loss, weight 1 beside the coarse mask's 5 + 5, gets weight 5: the extra
      4x is computed here on the final main stage's own assignment (recorded
      from the matcher's calls), because 772's speed path accepts only 767's
      point weight of one.
  M3  one-to-many training (H-DETR's hybrid branch).  During training only, the
      top 96 proposals enter the decoder as a second query set that cannot see,
      or be seen by, the 32 delivered queries and the denoising groups; each
      gold component is repeated 4 times in its Hungarian matching (674's own
      cost), so up to 4 queries learn to be a positive; its stage losses are
      674's, divided by 4 and added last.  Inference builds none of it.

Second round (869's diagnosis, 2026-09-28): M3's one-to-many loss was 58% of
the first step's total and trained the delivered queries' classifier, and its
fit on training lines rose to .944 (dev .663); M2's 5x point loss raised
training fit to .927 (dev .658); M1 found more good candidates (coverage 304)
that were not delivered.  Three modes answer them:

  m3d     M3 with its own classifier for the one-to-many queries (mask and box
          heads stay shared) and the one-to-many loss at weight 0.5
  m2lite  M2 with the rendered point loss back at weight 1 (the dense evidence
          and the context projection stay)
  m1m3d   M1 with m3d

At start the forward pass of m1 and m2 equals point_context's exactly and so do
the first step's loss pieces; m3's shared queries differ only by attention to
masked keys.  Nothing else changes: backbone, data, recipe, gates and freeze.

This module deliberately does NOT: change the matcher's cost, the denoising
groups, the quality head, the score rule or the number of delivered queries;
use the dense map to cut masks; read the frozen common test; or contact the
network.
"""
import importlib.util
import math
import os
import sys
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
MODES = {'off': (), 'm1': ('m1',), 'm2': ('m2',), 'm3': ('m3',), 'all': ('m1', 'm2', 'm3'),
         # 869-guided second round: see the v2 note in the docstring.
         'm3d': ('m3', 'decoupled'), 'm2lite': ('m2', 'point1'), 'm1m3d': ('m1', 'm3', 'decoupled')}
MODE = os.environ.get('HME_REBUILD', 'off')
if MODE not in MODES:
    raise ValueError('undeclared rebuild mode: '+MODE)
ON = set(MODES[MODE])
os.environ['HME_ALIGN_REFINE'] = 'point_context'
DIMENSION, CHANNELS, QUERIES = 160, 32, 32
ENCODER_LAYERS, ENCODER_FFN = 3, 640
POINT_WEIGHT = 5.
O2M_QUERIES, O2M_REPEAT = 96, 4
O2M_WEIGHT = .5 if 'decoupled' in MODES[MODE] else 1.
SEED = 86501
CHECKED_STEPS = 4


def load(name):
    spec = importlib.util.spec_from_file_location('rebuild865_'+name[:3], ROOT/'scripts'/name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ARCH = load('816_revision_alignment_refinement.py')
COARSE, TRANSFER, CORE, ARMS, RULE = ARCH.COARSE, ARCH.TRANSFER, ARCH.CORE, ARCH.ARMS, ARCH.RULE
if TRANSFER.CORE is not CORE:
    raise RuntimeError('the finder and the architecture read different spatial operators')
STATE = {'checked': 0, 'recorder': None, 'pairs': None}


def require(condition, message):
    if not condition:
        raise RuntimeError('865 rebuild refused: '+message)


# ---------------------------------------------------------------- M1 context encoder

def level_anchors(levels):
    """674's anchors: each position's centre and a square of 0.05 * 2**level."""
    b = levels[0].shape[0]
    out = []
    for level, feature in enumerate(levels):
        h, w = feature.shape[-2:]
        centre = TRANSFER.grid(h, w, feature).expand(b, -1, -1)
        out.append(torch.cat((centre, torch.full_like(centre, .05*2**level)), -1))
    return torch.cat(out, 1)


class ContextLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_read, self.norm_ffn = nn.LayerNorm(DIMENSION), nn.LayerNorm(DIMENSION)
        self.read = TRANSFER.DeformableRead()
        self.ffn = nn.Sequential(nn.Linear(DIMENSION, ENCODER_FFN), nn.ReLU(), nn.Linear(ENCODER_FFN, DIMENSION))
        for last in (self.read.output, self.ffn[-1]):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, tokens, position, anchors, shapes):
        normed = self.norm_read(tokens)
        maps, start = [], 0
        for h, w in shapes:
            maps.append(normed[:, start:start+h*w].transpose(1, 2).reshape(len(tokens), DIMENSION, h, w))
            start += h*w
        tokens = tokens+self.read(normed+position, maps, anchors)
        return tokens+self.ffn(self.norm_ffn(tokens))


class ContextEncoder(nn.Module):
    def __init__(self, layers=ENCODER_LAYERS):
        super().__init__()
        self.position = nn.Sequential(nn.Linear(4, DIMENSION), nn.ReLU(), nn.Linear(DIMENSION, DIMENSION))
        self.layers = nn.ModuleList([ContextLayer() for _ in range(layers)])

    def forward(self, levels):
        from torch.utils.checkpoint import checkpoint
        shapes = [tuple(f.shape[-2:]) for f in levels]
        tokens = torch.cat([f.flatten(2).transpose(1, 2) for f in levels], 1)
        anchors = level_anchors(levels)
        position = self.position(torch.logit(anchors.clamp(.001, .999)))
        for layer in self.layers:
            tokens = (checkpoint(layer, tokens, position, anchors, shapes, use_reentrant=False)
                      if self.training and torch.is_grad_enabled() else layer(tokens, position, anchors, shapes))
        out, start = [], 0
        for h, w in shapes:
            out.append(tokens[:, start:start+h*w].transpose(1, 2).reshape(len(tokens), DIMENSION, h, w))
            start += h*w
        return out


# ---------------------------------------------------------------- M2 dense-guided renderer

def dense_evidence(dense):
    """The map's logit and its 3x3 / 7x7 maxima (maxima from the detached map)."""
    held = dense.detach()
    return torch.cat((dense, F.max_pool2d(held, 3, 1, 1), F.max_pool2d(held, 7, 1, 3)), 1)


def install(model):
    """Attach the modules the mode declares; their outputs start at exactly zero."""
    see, draw = model.see, model.draw
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(SEED)
        if 'm1' in ON:
            see.context_encoder = ContextEncoder()
        if 'm2' in ON:
            see.context_to_fine = nn.Conv2d(DIMENSION, CHANNELS, 1)
            draw.dense_residual = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))
            for last in (see.context_to_fine, draw.dense_residual[-1]):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
        if 'decoupled' in ON:
            head = nn.Linear(DIMENSION, 1)
            nn.init.constant_(head.bias, math.log(.01/.99))
            model.find.o2m_class_embed = head
    if not ON & {'m1', 'm2'}:
        return model
    original_see = see.forward

    def see_forward(self, image):
        quarter, fine, dense, levels = original_see(image)
        if 'm1' in ON:
            levels = self.context_encoder(levels)
        if 'm2' in ON:
            fine = fine+F.interpolate(self.context_to_fine(levels[0]), size=fine.shape[-2:], mode='bilinear',
                                      align_corners=False)
            draw._rebuild_dense = dense
        return quarter, fine, dense, levels
    see.forward = MethodType(see_forward, see)
    if 'm2' in ON:
        original_draw, original_points = draw.forward, draw.at_points

        def draw_forward(self, coarse, fine, boxes, size):
            dense = getattr(self, '_rebuild_dense', None)
            require(dense is not None, 'the dense map did not reach the renderer')
            self._rebuild_evidence = dense_evidence(dense)
            try:
                return original_draw(coarse, fine, boxes, size)
            finally:
                # Returned tensors keep their graphs; the module must not.
                self._rebuild_dense = self._rebuild_evidence = None

        def at_points(self, coarse, fine, coordinates):
            base = original_points(coarse, fine, coordinates)
            b, q, p = coordinates.shape[:3]
            values = CORE.sample_points(self._rebuild_evidence, coordinates.reshape(b, q*p, 2))
            values = values.transpose(1, 2).reshape(b, q, p, 3)
            return base+self.dense_residual(values).squeeze(-1)
        draw.forward = MethodType(draw_forward, draw)
        draw.at_points = MethodType(at_points, draw)
    return model


# ---------------------------------------------------------------- M3 one-to-many training

def hybrid_forward(self, levels, fine, targets=None):
    """674's RegionFinder.forward, line for line, plus a training-only one-to-many query set."""
    b = fine.shape[0]
    pixels = F.adaptive_avg_pool2d(fine, levels[0].shape[-2:])
    memories, anchors = [], []
    for level, feature in enumerate(levels):
        h, w = feature.shape[-2:]
        memories.append(feature.flatten(2).transpose(1, 2)+self.level_embed[level])
        centre = TRANSFER.grid(h, w, feature).expand(b, -1, -1)
        anchors.append(torch.cat((centre, torch.full_like(centre, .05*2**level)), -1))
    memory = self.enc_output(torch.cat(memories, 1))
    anchors = torch.cat(anchors, 1)
    boxes = (torch.logit(anchors.clamp(.001, .999))+self.box_embed(memory)).sigmoid()
    scores = self.class_embed(memory).squeeze(-1)
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    indices = order[:, :QUERIES]
    content = memory.gather(1, indices[..., None].expand(-1, -1, DIMENSION))
    reference = boxes.gather(1, indices[..., None].expand(-1, -1, 4))
    stages = [self.prediction(content, pixels, reference)]
    query, reference = content.detach(), reference.detach()
    dn = self.denoising(targets, b, fine) if self.training and targets is not None else None
    count, attention = 0, None
    if dn is not None:
        dc, db, attention, _, _ = dn
        count = dc.shape[1]
        query, reference = torch.cat((dc, query), 1), torch.cat((db, reference), 1)
    many = self.training and targets is not None
    o2m = None
    if many:
        extra = order[:, :O2M_QUERIES]
        m_content = memory.gather(1, extra[..., None].expand(-1, -1, DIMENSION))
        m_reference = boxes.gather(1, extra[..., None].expand(-1, -1, 4))
        o2m = [self.prediction(m_content, pixels, m_reference)]
        head = getattr(self, 'o2m_class_embed', None)
        if head is not None:
            o2m[0] = (o2m[0][0], head(self.output_norm(m_content)).squeeze(-1), o2m[0][2])
        n = query.shape[1]
        query = torch.cat((query, m_content.detach()), 1)
        reference = torch.cat((reference, m_reference.detach()), 1)
        mask = torch.zeros(n+O2M_QUERIES, n+O2M_QUERIES, dtype=torch.bool, device=fine.device)
        if attention is not None:
            mask[:n, :n] = attention
        mask[:n, n:] = True
        mask[n:, :n] = True
        attention = mask
    end = count+QUERIES
    dn_stages = []
    for layer in self.layers:
        query = layer(query, levels, reference, attention)
        updated = (torch.logit(reference.clamp(.001, .999))+self.box_embed(query)).sigmoid()
        logits, objects, _ = self.prediction(query, pixels, updated)
        stages.append((logits[:, count:end], objects[:, count:end], updated[:, count:end]))
        if count:
            dn_stages.append((logits[:, :count], objects[:, :count], updated[:, :count]))
        if many:
            head = getattr(self, 'o2m_class_embed', None)
            o2m_objects = objects[:, end:] if head is None else head(self.output_norm(query[:, end:])).squeeze(-1)
            o2m.append((logits[:, end:], o2m_objects, updated[:, end:]))
        reference = updated.detach()
    self._o2m_stages = o2m
    return query[:, count:end], stages, dn_stages, dn


def o2m_match(stage, row, truth, repeat=O2M_REPEAT):
    """674's hybrid-match cost with every gold component repeated `repeat` times."""
    from scipy.optimize import linear_sum_assignment
    masks, scores, boxes = stage
    if not truth:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    with torch.no_grad():
        gold = torch.stack(truth)
        coordinates = torch.rand(1, 4096, 2, device=masks.device)
        target = CORE.sample_points(gold[None], coordinates)[0]
        logits = CORE.sample_points(masks[row:row+1], coordinates)[0]
        probability = logits.sigmoid()
        bce = F.softplus(logits).mean(1)[:, None]-logits@target.T/logits.shape[1]
        dice = 1-(2*probability@target.T+1)/(probability.sum(1)[:, None]+target.sum(1)[None]+1)
        p = scores[row].sigmoid()
        focal = .25*(1-p).square()*(-(p+1e-8).log()) - .75*p.square()*(-(1-p+1e-8).log())
        gt_boxes = TRANSFER.mask_boxes(gold)
        cost = (4*focal[:, None]+5*bce+5*dice+5*torch.cdist(boxes[row], gt_boxes, p=1)
                - 2*TRANSFER.giou(TRANSFER.xyxy(boxes[row]), TRANSFER.xyxy(gt_boxes)))
    rows, columns = linear_sum_assignment(cost.repeat(1, repeat).cpu().numpy())
    return rows, columns % len(truth)


if 'm3' in ON:
    TRANSFER.RegionFinder.forward = hybrid_forward


# ---------------------------------------------------------------- model and losses

def build_model(baseline, args):
    return install(ARCH.build_model(baseline, args))


def record_pairs():
    """Wrap the matcher as it stands at the first loss call (772 installs its own first)."""
    current = TRANSFER.stage_loss
    if STATE['recorder'] is current:
        return

    def stage_loss(stage, targets, pairs=None):
        term, chosen = current(stage, targets, pairs)
        if pairs is None:              # a main stage; denoising and one-to-many hand theirs in
            STATE['pairs'] = chosen
        return term, chosen
    TRANSFER.stage_loss = STATE['recorder'] = stage_loss


def extra_point_term(model, targets, pairs):
    """674's ROI point BCE on the final main stage's assignment."""
    coords, logits = model.cache['roi_points']
    terms = []
    for row, (q, g) in enumerate(pairs):
        if len(g):
            truth = torch.stack([targets[row][int(i)] for i in g])[:, None]
            gold = CORE.sample_points(truth, coords[row, q])[:, 0]
            terms.append(F.binary_cross_entropy_with_logits(logits[row, q], gold, reduction='none').mean(1))
    return torch.cat(terms).mean() if terms else logits.sum()*0


def losses(model, masks, scores, targets, probe, fullres):
    extra_points = 'm2' in ON and 'point1' not in ON
    if extra_points:
        record_pairs()
        STATE['pairs'] = None
    total, pieces = ARCH.losses(model, masks, scores, targets, probe, fullres)
    pieces = dict(pieces)
    if extra_points:
        require(STATE['pairs'] is not None and 'roi_points' in model.cache, 'the final assignment was not seen')
        extra = extra_point_term(model, targets, STATE['pairs'])
        total = total+(POINT_WEIGHT-1.)*extra
        pieces['roi_point_extra'] = TRANSFER.metric(extra)
    if 'm3' in ON:
        stages = getattr(model.find, '_o2m_stages', None)
        require(stages is not None, 'the one-to-many branch was not built')
        term = masks.sum()*0
        for stage in stages:
            pairs = [o2m_match(stage, row, truth) for row, truth in enumerate(targets)]
            value, _ = TRANSFER.stage_loss(stage, targets, pairs)
            term = term+value/O2M_REPEAT
        total = total+O2M_WEIGHT*term
        pieces['one_to_many'] = TRANSFER.metric(term)
        model.find._o2m_stages = None
    if STATE['checked'] < CHECKED_STEPS:
        STATE['checked'] += 1
        require(('roi_point_extra' in pieces) == extra_points and ('one_to_many' in pieces) == ('m3' in ON),
                'the loss pieces do not show the declared modules')
    return total, pieces


def configuration(arm):
    return dict(ARCH.configuration(arm), rebuild_v2={
        'mode': MODE, 'modules': sorted(ON), 'seed': SEED,
        'm1_context_encoder': {'layers': ENCODER_LAYERS, 'ffn': ENCODER_FFN, 'norm': 'pre', 'heads': 4,
                               'levels': 4, 'points': 4, 'zero_init_outputs': True, 'checkpointed': True}
        if 'm1' in ON else None,
        'm2_dense_renderer': {'evidence': ['dense logit', 'max3x3 (detached)', 'max7x7 (detached)'],
                              'residual': '3-16-1, last layer zero', 'context_to_fine': '1x1 zero-init',
                              'rendered_point_weight': 1. if 'point1' in ON else POINT_WEIGHT} if 'm2' in ON else None,
        'm3_one_to_many': {'queries': O2M_QUERIES, 'repeat': O2M_REPEAT, 'weight': O2M_WEIGHT,
                           'classifier': 'own (decoupled)' if 'decoupled' in ON else 'shared',
                           'isolation': 'mutually masked attention', 'inference': 'not built'}
        if 'm3' in ON else None,
        'design': 'outputs/revision_rebuild_v2/design.md'})

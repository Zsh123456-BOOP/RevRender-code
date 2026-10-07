"""Independent mechanism adaptations on a declared seed42 base.

rank: Rank-DETR-style rank embeddings and classification bias; original owners.
fusion: FreqFusion-inspired adaptive low/high-pass and offset residual at half scale.
proposal: Co-DETR-inspired train-only ATSS auxiliary encoder supervision.
rank_proposal: rank and proposal together, each built exactly as it is alone.
These are component adaptations, not reproductions of the complete cited methods.

Base (``--base``): ``stride2`` is the 767 stride-2 single-plane model the first
campaign used and stays the default, so that campaign replays unchanged.
``ctrl`` is 767 at its defaults (stride 4, one plane), which is r50_ctrl: its
step-2000 checkpoint under the speed path is bit-identical to
revision_r50_ctrl_s42 (weights, optimizer, sampler, RNG, curve).

This script deliberately does NOT: change the matcher, owners, denoising, the
schedule, the budget or the gate; combine rank_proposal in any way other than
both modules side by side; re-select an operating point; read, transcribe or
otherwise consume the frozen test split; or contact the network.
"""
import argparse
import importlib.util
import math
import os
from pathlib import Path
import sys
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
MODES = ('off', 'rank', 'fusion', 'proposal', 'rank_proposal')
BASES = {'stride2': '2', 'ctrl': '4'}
MODE = os.environ.get('HME_MECHANISM', 'off')
BASE = os.environ.get('HME_MECHANISM_BASE', 'stride2')
if MODE not in MODES:
    raise ValueError('undeclared mechanism')
if BASE not in BASES:
    raise ValueError('undeclared base')
RANKED = MODE in ('rank', 'rank_proposal')
AUXILIARY = MODE in ('proposal', 'rank_proposal')
# Each run has exactly one declared base, independent of inherited shell state.
os.environ.update(HME_COARSE_STRIDE=BASES[BASE], HME_MASK_PLANES='1',
                  HME_POINT_WEIGHT='1.0', HME_SCORE_RULE='product_rescaled')


def load(name):
    spec = importlib.util.spec_from_file_location('mechanism775_' + name[:3], ROOT/'scripts'/name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COARSE = load('767_revision_coarse_mask.py')
TRANSFER, CORE, ARMS, RULE = COARSE.TRANSFER, COARSE.CORE, COARSE.ARMS, COARSE.RULE


def ranks_of(scores):
    return torch.argsort(torch.argsort(scores.detach(), dim=1, descending=True, stable=True),
                         dim=1, stable=True)


def adaptive_filter(value, logits):
    """Nine spatially varying taps; normalize valid support at image boundaries."""
    h, w = value.shape[-2:]
    padded = F.pad(value, (1, 1, 1, 1))
    valid = F.pad(torch.ones_like(value[:, :1]), (1, 1, 1, 1))
    weights = logits.softmax(1)
    total, norm = torch.zeros_like(value), torch.zeros_like(value[:, :1])
    for i in range(9):
        y, x = divmod(i, 3)
        weight = weights[:, i:i+1] * valid[:, :, y:y+h, x:x+w]
        total = total + padded[:, :, y:y+h, x:x+w] * weight
        norm = norm + weight
    return total / norm.clamp_min(1e-6)


class DetailFusion(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.context = nn.Sequential(nn.Conv2d(2*channels, 32, 1), nn.ReLU())
        self.filters = nn.Conv2d(32, 18, 3, padding=1)
        self.offsets = nn.Conv2d(32, 2, 3, padding=1)
        self.project = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.offsets.weight); nn.init.zeros_(self.offsets.bias)
        nn.init.zeros_(self.project.weight); nn.init.zeros_(self.project.bias)

    def forward(self, semantic, detail):
        b, c, h, w = semantic.shape
        context = self.context(torch.cat((semantic, detail), 1))
        kernels = self.filters(context)
        low = adaptive_filter(semantic, kernels[:, :9])
        offsets = 2*self.offsets(context).tanh().flatten(2).transpose(1, 2)
        coordinates = TRANSFER.grid(h, w, semantic).expand(b, -1, -1)
        coordinates = coordinates + offsets/semantic.new_tensor([w, h])
        aligned = TRANSFER.sample_learned(low, coordinates).reshape(b, c, h, w)
        high = detail-adaptive_filter(detail, kernels[:, 9:])
        return semantic + detail + self.project(aligned-semantic+high)


def fused_pixel_forward(self, quarter, image):
    half = CORE.F.interpolate(quarter, size=(image.shape[-2]//2, image.shape[-1]//2),
                              mode='bilinear', align_corners=False)
    half = self.refine_half(self.mechanism_fusion(
        self.reduce_half(half), self.skip_half(CORE.F.avg_pool2d(image, 2))))
    full = CORE.F.interpolate(half, size=image.shape[-2:], mode='bilinear', align_corners=False)
    return self.output(self.refine_full(self.reduce_full(full)+self.skip_full(image)))


class DenseAuxiliary(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(160, 1)
        self.box = nn.Sequential(nn.Linear(160, 160), nn.ReLU(), nn.Linear(160, 4))
        nn.init.constant_(self.classifier.bias, math.log(.01/.99))
        nn.init.zeros_(self.box[-1].weight); nn.init.zeros_(self.box[-1].bias)

    def forward(self, memory, anchors):
        return (self.classifier(memory).squeeze(-1),
                (torch.logit(anchors.clamp(.001, .999))+self.box(memory)).sigmoid())


@torch.no_grad()
def atss_assign(anchors, counts, gold, size, k=9):
    """ATSS centers in pixel units, candidate IoU mean + sample std (correction=1).

    Each anchor has at most one auxiliary owner. No fallback positive is invented
    if no candidate clears the ATSS threshold and center-inside test.
    """
    assigned = torch.full((len(anchors),), -1, device=anchors.device, dtype=torch.long)
    if not len(gold):
        return assigned
    a, g = TRANSFER.xyxy(anchors), TRANSFER.xyxy(gold)
    lo = torch.maximum(a[:, None, :2], g[None, :, :2])
    hi = torch.minimum(a[:, None, 2:], g[None, :, 2:])
    inter = (hi-lo).clamp_min(0).prod(-1)
    union = anchors[:, 2:].prod(-1)[:, None]+gold[:, 2:].prod(-1)[None]-inter
    iou = inter/union.clamp_min(1e-7)
    distance = ((anchors[:, None, :2]-gold[None, :, :2])*anchors.new_tensor([size[1], size[0]])).square().sum(-1)
    candidates, start = [], 0
    for count in counts:
        candidates.append(torch.argsort(distance[start:start+count], dim=0, stable=True)[:min(k, count)]+start)
        start += count
    if start != len(anchors):
        raise ValueError('anchor level counts disagree')
    candidates = torch.cat(candidates)
    values = iou.gather(0, candidates)
    threshold = values.mean(0)+values.std(0, correction=1 if len(values) > 1 else 0)
    eligible = torch.zeros_like(iou, dtype=torch.bool)
    eligible.scatter_(0, candidates, values >= threshold[None])
    inside = ((anchors[:, None, :2] > g[None, :, :2]) &
              (anchors[:, None, :2] < g[None, :, 2:])).all(-1)
    best, owner = iou.masked_fill(~(eligible & inside), -1).max(1)
    return torch.where(best >= 0, owner, assigned)


def aligned_giou(a, b):
    a, b = TRANSFER.xyxy(a), TRANSFER.xyxy(b)
    inter = (torch.minimum(a[:, 2:], b[:, 2:])-torch.maximum(a[:, :2], b[:, :2])).clamp_min(0).prod(-1)
    union = (a[:, 2:]-a[:, :2]).prod(-1)+(b[:, 2:]-b[:, :2]).prod(-1)-inter
    enclosing = (torch.maximum(a[:, 2:], b[:, 2:])-torch.minimum(a[:, :2], b[:, :2])).clamp_min(0).prod(-1)
    return inter/union.clamp_min(1e-7)-(enclosing-union)/enclosing.clamp_min(1e-7)


def auxiliary_loss(cache, targets):
    scores, boxes, anchors, counts, size = cache
    cls, regression = scores.sum()*0, boxes.sum()*0
    positives = scores.new_zeros(())
    for row, truth in enumerate(targets):
        gold = TRANSFER.mask_boxes(torch.stack(truth)) if truth else boxes.new_empty((0, 4))
        owner = atss_assign(anchors[row], counts, gold, size)
        chosen = owner >= 0
        label = chosen.to(scores.dtype)
        p = scores[row].sigmoid()
        pt = p*label+(1-p)*(1-label)
        cls = cls + ((.25*label+.75*(1-label))*(1-pt).square()*
                     F.binary_cross_entropy_with_logits(scores[row], label, reduction='none')).sum()
        selected, expected = boxes[row, chosen], gold[owner[chosen]]
        regression = regression + 5*(selected-expected).abs().sum()+2*(1-aligned_giou(selected, expected)).sum()
        positives = positives + chosen.sum()
    return (cls+regression)/positives.clamp_min(1), positives


def mechanism_forward(self, levels, fine, targets=None):
    """767 forward with rank hooks or a train-only auxiliary memory head."""
    b = fine.shape[0]
    height, width = fine.shape[-2:]
    pixels = TRANSFER.F.adaptive_avg_pool2d(fine, (height//COARSE.STRIDE, width//COARSE.STRIDE))
    memories, anchors, counts = [], [], []
    for level, feature in enumerate(levels):
        h, w = feature.shape[-2:]
        counts.append(h*w)
        memories.append(feature.flatten(2).transpose(1, 2)+self.level_embed[level])
        centre = TRANSFER.grid(h, w, feature).expand(b, -1, -1)
        anchors.append(torch.cat((centre, torch.full_like(centre, .05*2**level)), -1))
    memory = self.enc_output(torch.cat(memories, 1))
    anchors = torch.cat(anchors, 1)
    boxes = (torch.logit(anchors.clamp(.001, .999))+self.box_embed(memory)).sigmoid()
    scores = self.class_embed(memory).squeeze(-1)
    self._mechanism_aux = None
    if hasattr(self, 'mechanism_aux') and self.training and targets is not None:
        auxiliary_scores, auxiliary_boxes = self.mechanism_aux(memory, anchors)
        self._mechanism_aux = auxiliary_scores, auxiliary_boxes, anchors, counts, (height, width)
    indices = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :32]
    content = memory.gather(1, indices[..., None].expand(-1, -1, 160))
    reference = boxes.gather(1, indices[..., None].expand(-1, -1, 4))
    stage = self.prediction(content, pixels, reference)
    ranked = hasattr(self, 'mechanism_rank')
    if ranked:
        rank = ranks_of(scores.gather(1, indices))
        stage = stage[0], stage[1]+self.mechanism_bias[0][rank], stage[2]
    stages = [stage]
    query, reference = content.detach(), reference.detach()
    dn = self.denoising(targets, b, fine) if self.training and targets is not None else None
    count, attention = 0, None
    if dn is not None:
        dc, db, attention, _, _ = dn
        count = dc.shape[1]
        query, reference = torch.cat((dc, query), 1), torch.cat((db, reference), 1)
    dn_stages = []
    for index, layer in enumerate(self.layers):
        if ranked:
            rank = ranks_of(stages[-1][1])
            regular = query[:, count:]+self.mechanism_rank[index][rank]
            query = torch.cat((query[:, :count], regular), 1) if count else regular
        query = layer(query, levels, reference, attention)
        updated = (torch.logit(reference.clamp(.001, .999))+self.box_embed(query)).sigmoid()
        logits, objects, _ = self.prediction(query, pixels, updated)
        regular_objects = objects[:, count:]
        if ranked:
            regular_objects = regular_objects+self.mechanism_bias[index+1][rank]
        stages.append((logits[:, count:], regular_objects, updated[:, count:]))
        if count:
            dn_stages.append((logits[:, :count], objects[:, :count], updated[:, :count]))
        reference = updated.detach()
    return query[:, count:], stages, dn_stages, dn


# Install before 685.attach captures the finder's bound method.
if RANKED or AUXILIARY:
    TRANSFER.RegionFinder.forward = mechanism_forward


def attach_mechanism(model, mode=MODE):
    if mode not in MODES:
        raise ValueError('undeclared mechanism')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(775)
        # Zero-initialized rank tensors draw nothing, so rank_proposal's
        # auxiliary head is drawn exactly as proposal's is.
        if mode in ('rank', 'rank_proposal'):
            model.find.mechanism_rank = nn.Parameter(torch.zeros(len(model.find.layers), 32, 160))
            model.find.mechanism_bias = nn.Parameter(torch.zeros(len(model.find.layers)+1, 32))
        if mode == 'fusion':
            decoder = model.see.pixel_decoder
            decoder.mechanism_fusion = DetailFusion()
            decoder.forward = MethodType(fused_pixel_forward, decoder)
        if mode in ('proposal', 'rank_proposal'):
            model.find.mechanism_aux = DenseAuxiliary()
    return model


def build_model(baseline, args):
    return attach_mechanism(COARSE.build_model(baseline, args))


def losses(model, masks, scores, targets, probe, fullres):
    total, pieces = COARSE.losses(model, masks, scores, targets, probe, fullres)
    if AUXILIARY:
        cache = model.find._mechanism_aux
        if cache is None:
            raise RuntimeError('proposal training requires dense auxiliary predictions')
        extra, positives = auxiliary_loss(cache, targets)
        pieces = dict(pieces, dense_aux=TRANSFER.metric(extra), dense_aux_positive=TRANSFER.metric(positives))
        total = total+extra
        model.find._mechanism_aux = None
    return total, pieces


def configuration(arm):
    value = dict(COARSE.configuration(arm))
    value.update(experiment='775 mechanism adaptations', mechanism=MODE, base=BASE,
                 control=('existing 767 stride2 single-plane seed42' if BASE == 'stride2' else
                          'revision_r50_ctrl_s42 (767 defaults); bit-identical twin verified at step 2000'),
                 matching='original one-to-one assignment and binary ownership, unchanged',
                 auxiliary_weight=1.0 if AUXILIARY else 0.0,
                 atss_topk_per_level=9 if AUXILIARY else None,
                 new_module_seed=775,
                 finder='rank conditioning' if RANKED else 'original top32 and anchor prior',
                 bars={'rank': 'reduce cut losses with FP/F1/AP guards',
                       'fusion': 'improve coarse/rendered IoU and partial errors with FP/F1/AP guards',
                       'proposal': 'reduce no-box losses and improve candidate coverage with FP/F1/AP guards',
                       'rank_proposal': 'keep the rank cut reduction while the auxiliary head holds clean-row false positives'})
    return value


def main(argv=None):
    ahead = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ahead.add_argument('--mechanism', choices=MODES, required=True)
    ahead.add_argument('--base', choices=sorted(BASES), default='stride2')
    picked, rest = ahead.parse_known_args(sys.argv[1:] if argv is None else argv)
    os.environ['HME_MECHANISM'] = picked.mechanism
    os.environ['HME_MECHANISM_BASE'] = picked.base
    train = COARSE.RESNET.repoint(load('675_revision_region_point_train.py'), COARSE.RESNET.MIT_B0_SHA)
    args = train.parse(rest, arms=ARMS)
    train.run(args, architecture_file=Path(__file__).name,
              extra_dependencies=('685_revision_mask_iou_score.py', '687_revision_region_point_maskiou.py',
                                  '750_revision_resnet50_base.py', '767_revision_coarse_mask.py'))


if __name__ == '__main__':
    main()

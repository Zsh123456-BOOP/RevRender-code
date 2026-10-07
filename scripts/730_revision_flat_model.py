"""A1empty as one file: see, find, draw, score and loss, with nothing patched at import.

The model this project reports is currently spelled across six files that each
rewrite the previous one at import time.  654 defines the parts, 674 replaces
two of them and rebinds ``stage_loss``, 685 rewrites ``ThreePartTransfer.forward``
and ``losses`` and wraps ``find.forward``, 687 re-exports the pair, 717 wraps
``stage_loss`` again to record an assignment in a module global, and 718 wraps
``losses`` once more to read that global.  Forty-seven files are loaded to
express one 6.1M-parameter model, thirty-six of them predating 654.

That layering is not only slow to read, it produced the one real bug this line
has had.  674 calls the stage loss once per decoder stage and then once per
denoising stage; 717's recorder wrote every call into a module global, so the
terms built on it read the denoising assignment and pushed matched queries
towards empty while training unmatched ones onto gold shapes.  A wrapper on a
wrapper on a rebound function is what made that invisible.  Here the assignment
is a local variable handed to the term that needs it, so the same mistake
cannot be written.

The arithmetic is unchanged and that is checked rather than asserted: this
file's four-step smoke trace must equal, value for value, the trace
``revision_ablation726_empty_only_s42`` already wrote.  Two things are load
bearing for that and are kept exactly as they are rather than tidied:

  initialisation  674 builds a whole 654 model in the ``d_competition`` arm and
                  keeps only its ``see``.  The discarded ``find`` and ``draw``
                  consume the random numbers every completed run's weights
                  depend on, so the base is still built and still discarded.
  the CUDA reseed ``torch.manual_seed`` inside ``fork_rng(devices=[])`` restores
                  the CPU generator and leaves CUDA reseeded.  That moves every
                  later CUDA draw, and every run in this line was trained with
                  it, so it stays -- with this sentence attached to it.

Cleaning either one is a different model and needs its own three seeds.

The configuration is A1empty, which 64 measured against its own ablation at
seed 42: the empty-query term alone is +0.0130 on the two-term control and the
asymmetric ROI term alone is -0.0207, so only the first is carried here.  It is
not a knob: an arm that wants the other term has 718.

This script deliberately does NOT: change the architecture, the schedule, the
budget, the seed, the gate or the operating-point rule; delete or modify the
files it replaces, which every completed run's hashed evidence names; offer the
asymmetric ROI term, the other two arms of 674, or any other switch; re-select
an operating point; read, transcribe or otherwise consume the frozen test
split; or contact the network.
"""
import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
BASE_FILE = '654_revision_three_part_model.py'
ARMS = ('d_region_roi_point',)
RULE = os.environ.get('HME_SCORE_RULE', 'product_rescaled')
RULES = ('product', 'product_rescaled', 'quality_only')

DIMENSION = 160
CHANNELS = 32
QUERIES = 32
LAYERS = 4
REGION_SEED = 67401
DRAW_SEED = 67402
# 685 computes the overlap label at .5, which is NOT the .3 the readouts
# deliver at; the first draft of this file used .3 and the smoke trace caught
# it as a different mask_iou target mean on an otherwise identical step.
OVERLAP_LEVEL = .5
MASK_IOU_WEIGHT = 1.
EMPTY_WEIGHT = .25


def load(name):
    spec = importlib.util.spec_from_file_location('flat730_' + name[:3],
                                                  ROOT / 'scripts' / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load(BASE_FILE)
# 675 hands this to 668, which replaces ``sample_points`` and
# ``select_shared_points`` on it and reads ``index_coordinates`` and
# ``FindInstances``.  Pointing it at this module means the deterministic
# operators land on the names the code below actually calls.
CORE = sys.modules[__name__]
FindInstances = BASE.FindInstances


def sample_points(features, coordinates, mode='bilinear'):
    """B,C,H,W and B,P,2 normalized x,y pixel-centre coordinates -> B,C,P."""
    return F.grid_sample(features, coordinates[:, None] * 2 - 1,
                         mode=mode, padding_mode='border', align_corners=False)[:, :, 0]


def index_coordinates(indices, height, width):
    return torch.stack(((indices % width + .5) / width,
                        (indices // width + .5) / height), -1)


def select_shared_points(coarse, objects, dense, count, training):
    """Present because 668 replaces it; this arm's renderer never calls it."""
    raise RuntimeError('the ROI renderer samples its own points')


def sample_learned(features, coordinates):
    """Bilinear, zero padded, align_corners=False; deterministic gather adjoint.

    Unlike 668's fixed grid, this differentiates coordinates as well as values.
    Four explicit gathers avoid native CUDA grid_sample backward atomics.
    """
    b, c, h, w = features.shape
    if coordinates.ndim != 3 or coordinates.shape[0] != b or coordinates.shape[-1] != 2:
        raise ValueError('expected matching NCHW and NP2')
    if not torch.isfinite(coordinates).all():
        raise ValueError('nonfinite coordinates')
    xy = ((coordinates * 2 - 1 + 1) * coordinates.new_tensor([w, h]) - 1) / 2
    x, y = xy.unbind(-1)
    x0, y0 = x.floor(), y.floor()
    dx, dy = x - x0, y - y0
    flat = features.flatten(2)
    output = 0
    for xx, yy, weight in ((x0, y0, (1-dx)*(1-dy)), (x0+1, y0, dx*(1-dy)),
                           (x0, y0+1, (1-dx)*dy), (x0+1, y0+1, dx*dy)):
        valid = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        index = yy.long().clamp(0, h-1)*w + xx.long().clamp(0, w-1)
        output = output + flat.gather(2, index[:, None].expand(-1, c, -1)) * (weight*valid)[:, None]
    return output


def xyxy(boxes):
    return torch.cat((boxes[..., :2]-boxes[..., 2:]/2, boxes[..., :2]+boxes[..., 2:]/2), -1)


def giou(a, b):
    lower = torch.maximum(a[:, None, :2], b[None, :, :2])
    upper = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (upper-lower).clamp_min(0).prod(-1)
    union = (a[:, 2:]-a[:, :2]).clamp_min(0).prod(-1)[:, None] + (b[:, 2:]-b[:, :2]).clamp_min(0).prod(-1)[None] - inter
    enclosing = (torch.maximum(a[:, None, 2:], b[None, :, 2:])-torch.minimum(a[:, None, :2], b[None, :, :2])).clamp_min(0).prod(-1)
    return inter/union.clamp_min(1e-7) - (enclosing-union)/enclosing.clamp_min(1e-7)


def mask_boxes(masks):
    """cxcywh from nonempty binary masks; empty predicted masks use full image."""
    h, w = masks.shape[-2:]
    xs, ys = masks.bool().any(-2), masks.bool().any(-1)
    x, y = torch.arange(w, device=masks.device), torch.arange(h, device=masks.device)
    left = x.expand_as(xs).masked_fill(~xs, w).amin(-1)/w
    right = (x.expand_as(xs).masked_fill(~xs, -1).amax(-1)+1)/w
    top = y.expand_as(ys).masked_fill(~ys, h).amin(-1)/h
    bottom = (y.expand_as(ys).masked_fill(~ys, -1).amax(-1)+1)/h
    boxes = torch.stack(((left+right)/2, (top+bottom)/2, right-left, bottom-top), -1)
    return torch.where(xs.any(-1)[..., None], boxes, boxes.new_tensor([.5, .5, 1., 1.]))


def expanded_roi(boxes, size):
    boxes = boxes.detach()
    minimum = boxes.new_tensor([4/size[1], 4/size[0]])
    extent = torch.maximum(boxes[..., 2:]*1.5, minimum)
    low = (boxes[..., :2]-extent/2).clamp(0, 1)
    high = (boxes[..., :2]+extent/2).clamp(0, 1)
    return torch.cat((low, high), -1)


def grid(h, w, like):
    return index_coordinates(torch.arange(h*w, device=like.device)[None], h, w).to(like.dtype)


def paste_roi_probabilities(logits, local, size):
    """As in native mask pasting: interpolate probabilities, zero outside ROI."""
    probability = sample_learned(logits.sigmoid(), local).reshape(len(local), *size)
    inside = ((local >= 0) & (local <= 1)).all(-1).reshape(len(local), *size)
    return torch.logit(probability.clamp(1e-6, 1-1e-6)).masked_fill(~inside, -20.)


class MultiScaleSee(nn.Module):
    """654's encoder, decoder and full-resolution detail map, plus three laterals."""

    def __init__(self, base):
        super().__init__()
        for name, part in base.named_children():
            setattr(self, name, part)
        self.laterals = nn.ModuleList([nn.Sequential(nn.Conv2d(c, DIMENSION, 1), nn.GroupNorm(8, DIMENSION))
                                       for c in self.encoder.out_channels[-3:]])

    def forward(self, image):
        encoded = self.encoder(image)
        quarter = self.shared_projection(self.dense_decoder(encoded))
        levels = [quarter] + [layer(value) for layer, value in zip(self.laterals, encoded[-3:])]
        fine = self.pixel_decoder(quarter, image)
        return quarter, fine, self.dense_head(fine), levels


class DeformableRead(nn.Module):
    def __init__(self, dim=DIMENSION, heads=4, levels=4, points=4):
        super().__init__()
        self.heads, self.levels, self.points = heads, levels, points
        self.offsets = nn.Linear(dim, heads*levels*points*2)
        self.weights = nn.Linear(dim, heads*levels*points)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        nn.init.zeros_(self.offsets.weight)
        angles = torch.arange(heads)*2*math.pi/heads
        direction = torch.stack((angles.cos(), angles.sin()), -1)
        direction /= direction.abs().amax(-1, keepdim=True)
        bias = direction[:, None, None]*torch.arange(1, points+1)[None, None, :, None]
        with torch.no_grad():
            self.offsets.bias.copy_(bias.expand(heads, levels, points, 2).reshape(-1))
        nn.init.zeros_(self.weights.weight)
        nn.init.zeros_(self.weights.bias)

    def forward(self, query, features, reference):
        b, q, d = query.shape
        offsets = self.offsets(query).reshape(b, q, self.heads, self.levels, self.points, 2)
        locations = reference[:, :, None, None, None, :2] + offsets/self.points * reference[:, :, None, None, None, 2:] * .5
        weights = self.weights(query).reshape(b, q, self.heads, self.levels*self.points).softmax(-1)
        weights = weights.reshape(b, q, self.heads, self.levels, self.points)
        result = 0
        for level, feature in enumerate(features):
            h, w = feature.shape[-2:]
            values = self.value(feature.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(b*self.heads, d//self.heads, h, w)
            coordinates = locations[:, :, :, level].permute(0, 2, 1, 3, 4).reshape(b*self.heads, q*self.points, 2)
            sampled = sample_learned(values, coordinates).reshape(b, self.heads, d//self.heads, q, self.points).permute(0, 3, 1, 4, 2)
            result = result + (sampled*weights[:, :, :, level, :, None]).sum(3)
        return self.output(result.reshape(b, q, d))


class RegionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.position = nn.Sequential(nn.Linear(4, DIMENSION), nn.ReLU(), nn.Linear(DIMENSION, DIMENSION))
        self.self_attention = nn.MultiheadAttention(DIMENSION, 4, dropout=0., batch_first=True)
        self.cross_attention = DeformableRead()
        self.norms = nn.ModuleList([nn.LayerNorm(DIMENSION) for _ in range(3)])
        self.ffn = nn.Sequential(nn.Linear(DIMENSION, 640), nn.ReLU(), nn.Linear(640, DIMENSION))

    def forward(self, query, features, reference, attention_mask):
        position = self.position(torch.logit(reference.clamp(.001, .999)))
        value = self.self_attention(query+position, query+position, query, attn_mask=attention_mask, need_weights=False)[0]
        query = self.norms[0](query+value)
        query = self.norms[1](query+self.cross_attention(query+position, features, reference))
        return self.norms[2](query+self.ffn(query))


class RegionFinder(nn.Module):
    """Two-stage proposal selection, deformable decoding and denoising groups."""

    def __init__(self):
        super().__init__()
        self.query_count = QUERIES
        self.enc_output = nn.Sequential(nn.Linear(DIMENSION, DIMENSION), nn.LayerNorm(DIMENSION))
        self.level_embed = nn.Parameter(torch.randn(4, DIMENSION)*.02)
        self.output_norm = nn.LayerNorm(DIMENSION)
        self.class_embed = nn.Linear(DIMENSION, 1)
        self.mask_embed = nn.Sequential(nn.Linear(DIMENSION, DIMENSION), nn.ReLU(),
                                        nn.Linear(DIMENSION, DIMENSION), nn.ReLU(),
                                        nn.Linear(DIMENSION, CHANNELS))
        self.box_embed = nn.Sequential(nn.Linear(DIMENSION, DIMENSION), nn.ReLU(),
                                       nn.Linear(DIMENSION, DIMENSION), nn.ReLU(),
                                       nn.Linear(DIMENSION, 4))
        nn.init.zeros_(self.box_embed[-1].weight)
        nn.init.zeros_(self.box_embed[-1].bias)
        nn.init.constant_(self.class_embed.bias, math.log(.01/.99))
        self.label_embedding = nn.Embedding(1, DIMENSION)
        self.layers = nn.ModuleList([RegionLayer() for _ in range(LAYERS)])

    def prediction(self, query, pixels, boxes):
        value = self.output_norm(query)
        return (torch.einsum('bqc,bchw->bqhw', self.mask_embed(value), pixels),
                self.class_embed(value).squeeze(-1), boxes)

    def denoising(self, targets, batch, like):
        maximum = max((len(t) for t in targets), default=0)
        groups = min(4, QUERIES//maximum) if maximum else 0
        count = maximum*groups
        if not count:
            return None
        boxes = like.new_full((batch, count, 4), .5)
        pairs = []
        for row, truth in enumerate(targets):
            n = len(truth)
            if n:
                boxes[row, torch.cat([torch.arange(n, device=like.device)+group*maximum for group in range(groups)])] = mask_boxes(torch.stack(truth)).repeat(groups, 1)
            pairs.append((torch.cat([torch.arange(n, device=like.device)+g*maximum for g in range(groups)]), torch.arange(n, device=like.device).repeat(groups)))
        extent = torch.cat((boxes[..., 2:]/2, boxes[..., 2:]), -1)
        noisy = (boxes+(torch.rand_like(boxes)*2-1)*extent*.4).clamp(.001, .999)
        content = self.label_embedding.weight[0][None, None].expand(batch, count, -1)
        attention = torch.zeros(count+QUERIES, count+QUERIES, dtype=torch.bool, device=like.device)
        attention[count:, :count] = True
        for group in range(groups):
            attention[group*maximum:(group+1)*maximum, :count] = True
            attention[group*maximum:(group+1)*maximum, group*maximum:(group+1)*maximum] = False
        return content, noisy, attention, pairs, groups

    def forward(self, levels, fine, targets=None):
        b = fine.shape[0]
        pixels = F.adaptive_avg_pool2d(fine, levels[0].shape[-2:])
        memories, anchors = [], []
        for level, feature in enumerate(levels):
            h, w = feature.shape[-2:]
            memories.append(feature.flatten(2).transpose(1, 2)+self.level_embed[level])
            centre = grid(h, w, feature).expand(b, -1, -1)
            anchors.append(torch.cat((centre, torch.full_like(centre, .05*2**level)), -1))
        memory = self.enc_output(torch.cat(memories, 1))
        anchors = torch.cat(anchors, 1)
        boxes = (torch.logit(anchors.clamp(.001, .999))+self.box_embed(memory)).sigmoid()
        scores = self.class_embed(memory).squeeze(-1)
        indices = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :QUERIES]
        content = memory.gather(1, indices[..., None].expand(-1, -1, DIMENSION))
        reference = boxes.gather(1, indices[..., None].expand(-1, -1, 4))
        # The selected encoder predictions receive their own hybrid supervision.
        stages = [self.prediction(content, pixels, reference)]
        query, reference = content.detach(), reference.detach()
        dn = self.denoising(targets, b, fine) if self.training and targets is not None else None
        count, attention = 0, None
        if dn is not None:
            dc, db, attention, _, _ = dn
            count = dc.shape[1]
            query, reference = torch.cat((dc, query), 1), torch.cat((db, reference), 1)
        dn_stages = []
        for layer in self.layers:
            query = layer(query, levels, reference, attention)
            updated = (torch.logit(reference.clamp(.001, .999))+self.box_embed(query)).sigmoid()
            logits, objects, _ = self.prediction(query, pixels, updated)
            stages.append((logits[:, count:], objects[:, count:], updated[:, count:]))
            if count:
                dn_stages.append((logits[:, :count], objects[:, :count], updated[:, :count]))
            reference = updated.detach()
        return query[:, count:], stages, dn_stages, dn


class RoiPointDraw(nn.Module):
    """Per-instance PointRend-style ROI, original coarse features, current uncertainty."""

    def __init__(self):
        super().__init__()
        self.point_head = nn.Sequential(nn.Linear(CHANNELS+1, 64), nn.ReLU(),
                                        nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def at_points(self, coarse, fine, coords):
        b, q, p = coords.shape[:3]
        feature = sample_points(fine, coords.reshape(b, q*p, 2)).transpose(1, 2).reshape(b, q, p, CHANNELS)
        values = sample_learned(coarse.reshape(b*q, 1, *coarse.shape[-2:]), coords.reshape(b*q, p, 2)).reshape(b, q, p)
        return self.point_head(torch.cat((feature, values[..., None]), -1)).squeeze(-1)

    def forward(self, coarse, fine, boxes, size):
        b, q = coarse.shape[:2]
        roi = expanded_roi(boxes, size)
        low, extent = roi[..., :2], (roi[..., 2:]-roi[..., :2]).clamp_min(1e-5)
        if self.training:
            with torch.no_grad():
                local = torch.rand(b, q, 768, 2, device=coarse.device)
                global_coords = low[:, :, None]+local*extent[:, :, None]
                sampled = sample_learned(coarse.reshape(b*q, 1, *coarse.shape[-2:]), global_coords.reshape(b*q, 768, 2)).reshape(b, q, 768)
                top = torch.argsort(-sampled.abs(), dim=-1, descending=True, stable=True)[..., :192]
                uncertain = local.gather(2, top[..., None].expand(-1, -1, -1, 2))
                random = torch.rand(b, q, 64, 2, device=coarse.device)
                coords = low[:, :, None]+torch.cat((uncertain, random), 2)*extent[:, :, None]
            logits = self.at_points(coarse, fine, coords)
            return F.interpolate(coarse, size=size, mode='bilinear', align_corners=False), {'roi_points': (coords, logits)}
        local = grid(14, 56, coarse).expand(b*q, -1, -1).reshape(b, q, -1, 2)
        coords = low[:, :, None]+local*extent[:, :, None]
        rendered = sample_learned(coarse.reshape(b*q, 1, *coarse.shape[-2:]), coords.reshape(b*q, -1, 2)).reshape(b*q, 1, 14, 56)
        for h, w in ((28, 112), (56, 224)):
            rendered = F.interpolate(rendered, size=(h, w), mode='bilinear', align_corners=False)
            indices = torch.argsort(-rendered.flatten(1).abs(), dim=-1, descending=True, stable=True)[:, :1024]
            local = index_coordinates(indices, h, w).reshape(b, q, -1, 2)
            coords = low[:, :, None]+local*extent[:, :, None]
            logits = self.at_points(coarse, fine, coords).reshape(b*q, 1, -1)
            rendered = rendered.flatten(2).scatter(2, indices[:, None], logits).reshape(b*q, 1, h, w)
        # Paste in small groups to limit temporary allocation; outside ROI is background.
        full_grid = grid(*size, coarse)
        masks = []
        for row in range(b):
            local = (full_grid-low[row, :, None])/extent[row, :, None]
            masks.append(paste_roi_probabilities(rendered[row*q:(row+1)*q], local, size))
        return torch.stack(masks), {}


class MaskIoUHead(nn.Module):
    """Reads the query and a summary of the mask that query just drew."""

    def __init__(self, dimension=DIMENSION, channels=CHANNELS):
        super().__init__()
        self.project = nn.Linear(channels, dimension, bias=False)
        # The pooled feature is a weighted average, so it is unchanged by how
        # much of the map the mask covers -- and coverage is most of what an
        # IoU depends on.  These two statistics put the extent back in.
        self.extent = nn.Linear(2, dimension, bias=False)
        self.score = nn.Sequential(nn.LayerNorm(dimension * 3),
                                   nn.Linear(dimension * 3, dimension),
                                   nn.ReLU(),
                                   nn.Linear(dimension, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, query, coarse, pixels, delivered):
        """Detached from the trunk; ``delivered`` is the mask that is reported."""
        query, coarse, pixels = query.detach(), coarse.detach(), pixels.detach()
        delivered = None if delivered is None else delivered.detach()
        weight = coarse.sigmoid()
        total = weight.flatten(2).sum(-1).clamp(min=1e-4)
        pooled = torch.einsum('bqhw,bchw->bqc', weight, pixels) / total[..., None]
        source = weight if delivered is None else delivered.sigmoid()
        area = source.flatten(2).sum(-1)
        extent = torch.stack([area / (source.shape[-1] * source.shape[-2]),
                              source.flatten(2).amax(-1)], -1)
        return self.score(torch.cat(
            [query, self.project(pooled), self.extent(extent)], -1)).squeeze(-1)


def overlap_targets(masks, targets, level=OVERLAP_LEVEL):
    """The IoU each query's delivered mask reaches against its best gold piece."""
    batch, queries = masks.shape[:2]
    coarse = masks
    values = torch.zeros(batch, queries, device=coarse.device)
    binary = (coarse.detach().sigmoid() >= level)
    areas = binary.flatten(2).sum(-1).to(coarse.dtype)
    for index, truth in enumerate(targets):
        if not len(truth):
            continue
        gold = torch.stack([t.to(coarse.dtype) for t in truth]).flatten(1)
        if gold.shape[-1] != binary.shape[-2] * binary.shape[-1]:
            raise ValueError('gold and coarse mask resolutions differ')
        flat = binary[index].flatten(1).to(coarse.dtype)
        intersection = flat @ gold.t()
        union = areas[index][:, None] + gold.sum(-1)[None] - intersection
        values[index] = (intersection / union.clamp(min=1.)).max(-1).values
    return values


def delivered_score(objectness, quality, rule):
    """The logit the operating-point search and the archive will see."""
    if rule not in RULES:
        raise ValueError('undeclared scoring rule: ' + str(rule))
    predicted = quality.sigmoid()
    if rule == 'quality_only':
        combined = predicted
    else:
        combined = objectness.sigmoid() * predicted
        if rule == 'product_rescaled':
            combined = combined.clamp(min=0.).sqrt()
    return torch.logit(combined.clamp(1e-6, 1. - 1e-6))


class FlatModel(nn.Module):
    """see / find / draw, with the overlap head living inside find."""

    def __init__(self, base):
        super().__init__()
        self.arm, self.query_count = ARMS[0], QUERIES
        # Both seeds, and the fact that manual_seed inside fork_rng(devices=[])
        # leaves CUDA reseeded, reproduce 674's construction exactly; see the
        # module docstring before changing either.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(REGION_SEED)
            self.see, self.find = MultiScaleSee(base.see), RegionFinder()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(DRAW_SEED)
            self.draw = RoiPointDraw()
        self.find.mask_iou = MaskIoUHead()
        self.cache = {}

    def forward(self, image, targets=None):
        quarter, fine, dense, levels = self.see(image)
        query, stages, dn_stages, dn = self.find(levels, fine, targets)
        coarse, objects, boxes = stages[-1]
        rendered, points = self.draw(coarse, fine, boxes, image.shape[-2:])
        self.cache = {'coarse': coarse, 'dense': dense, 'boxes': boxes,
                      'stages': stages, 'dn_stages': dn_stages, 'dn': dn,
                      'auxiliary': [], 'seed_coordinates': boxes[..., :2], **points}
        pixels = F.adaptive_avg_pool2d(fine, coarse.shape[-2:])
        quality = self.find.mask_iou(query, coarse, pixels, rendered)
        self.cache['mask_iou'] = quality
        self.cache['mask_iou_delivered'] = rendered
        if self.training:
            return rendered, objects, boxes
        return rendered, delivered_score(objects, quality, RULE), boxes


def build_model(baseline, args):
    """674's construction, including the base whose random draws the weights need."""
    from types import SimpleNamespace

    if args.arm not in ARMS:
        raise ValueError('unknown arm')
    base_args = SimpleNamespace(**{**vars(args), 'arm': 'd_competition'})
    base = BASE.build_model(baseline, base_args)
    return FlatModel(base)


def hybrid_match(stage, row, truth):
    from scipy.optimize import linear_sum_assignment

    masks, scores, boxes = stage
    if not truth:
        return [], []
    with torch.no_grad():
        gold = torch.stack(truth)
        coordinates = torch.rand(1, 4096, 2, device=masks.device)
        target = sample_points(gold[None], coordinates)[0]
        logits = sample_points(masks[row:row+1], coordinates)[0]
        probability = logits.sigmoid()
        bce = F.softplus(logits).mean(1)[:, None]-logits@target.T/logits.shape[1]
        dice = 1-(2*probability@target.T+1)/(probability.sum(1)[:, None]+target.sum(1)[None]+1)
        p = scores[row].sigmoid()
        focal = .25*(1-p).square()*(-(p+1e-8).log()) - .75*p.square()*(-(1-p+1e-8).log())
        gt_boxes = mask_boxes(gold)
        cost = 4*focal[:, None]+5*bce+5*dice+5*torch.cdist(boxes[row], gt_boxes, p=1)-2*giou(xyxy(boxes[row]), xyxy(gt_boxes))
        return linear_sum_assignment(cost.cpu().numpy())


def stage_loss(stage, targets, pairs=None):
    masks, objects, boxes = stage
    labels = torch.zeros_like(objects)
    count = max(1, sum(len(t) for t in targets))
    mask_term, box_term = masks.sum()*0, boxes.sum()*0
    chosen = []
    for row, truth in enumerate(targets):
        q, g = hybrid_match(stage, row, truth) if pairs is None else pairs[row]
        chosen.append((q, g))
        if not len(g):
            continue
        labels[row, q] = 1
        gold = torch.stack([truth[int(i)] for i in g])
        logits = F.interpolate(masks[row, q, None], size=gold.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
        probability = logits.sigmoid().flatten(1)
        target = gold.flatten(1)
        bce = F.binary_cross_entropy_with_logits(logits, gold, reduction='none').flatten(1).mean(1)
        dice = 1-(2*(probability*target).sum(1)+1)/(probability.sum(1)+target.sum(1)+1)
        mask_term = mask_term + (5*bce+5*dice).sum()/count
        gold_boxes = mask_boxes(gold)
        box_term = box_term + (5*(boxes[row, q]-gold_boxes).abs().sum(1)+2*(1-giou(xyxy(boxes[row, q]), xyxy(gold_boxes)).diag())).sum()/count
    p = objects.sigmoid()
    pt = p*labels+(1-p)*(1-labels)
    alpha = .25*labels+.75*(1-labels)
    classification = 4*(alpha*(1-pt).square()*F.binary_cross_entropy_with_logits(objects, labels, reduction='none')).sum()/count
    return mask_term+box_term+classification, chosen


def empty_mask_term(masks, chosen):
    """Push every unmatched query's mask towards empty.

    The assignment already tells the classification head that these queries
    hold no object; it says nothing to the mask head, which is why they still
    draw.  Normalised by the number of unmatched queries so the term does not
    grow with the query budget.  ``chosen`` is the main stages' assignment,
    handed in by the caller -- never a module global, and never the denoising
    one, which names different queries.
    """
    total, counted = masks.sum() * 0, 0
    for row, (q, g) in enumerate(chosen):
        matched = set(int(i) for i in q) if len(g) else set()
        rest = [i for i in range(masks.shape[1]) if i not in matched]
        if not rest:
            continue
        logits = masks[row, rest]
        total = total + F.binary_cross_entropy_with_logits(
            logits, torch.zeros_like(logits), reduction='none').flatten(1).mean(1).sum()
        counted += len(rest)
    return total / max(1, counted)


def losses(model, masks, scores, targets, probe, fullres):
    """674's region loss, 685's overlap term, and the empty-query term, in that order."""
    if any(len(t) > QUERIES for t in targets):
        raise ValueError('targets exceed query budget')
    total = masks.sum()*0
    pieces = {}
    pairs = None
    for index, stage in enumerate(model.cache['stages']):
        term, pairs = stage_loss(stage, targets)
        total = total+term
        pieces['stage_'+str(index)] = float(term.detach())
    dn = model.cache['dn']
    if dn is not None:
        dn_loss = masks.sum()*0
        for stage in model.cache['dn_stages']:
            term, _ = stage_loss(stage, targets, dn[3])
            dn_loss = dn_loss+term/dn[4]
        total = total+dn_loss
        pieces['denoising'] = float(dn_loss.detach())
    union = torch.stack([torch.stack(t).amax(0) if t else torch.zeros_like(masks[0, 0]) for t in targets])[:, None]
    dense = fullres.matched_mask_loss(model.cache['dense'], union, 'dice', .0)
    total = total+dense
    pieces['dense'] = float(dense.detach())
    coords, logits = model.cache['roi_points']
    terms = []
    for row, (q, g) in enumerate(pairs):
        if len(g):
            truth = torch.stack([targets[row][int(i)] for i in g])[:, None]
            gold = sample_points(truth, coords[row, q])[:, 0]
            terms.append(F.binary_cross_entropy_with_logits(logits[row, q], gold, reduction='none').mean(1))
    point = torch.cat(terms).mean() if terms else logits.sum()*0
    total = total+point
    pieces['roi_point'] = float(point.detach())

    quality = model.cache['mask_iou']
    truth = overlap_targets(model.cache['mask_iou_delivered'], targets)
    overlap = F.binary_cross_entropy_with_logits(quality, truth)
    total = total + MASK_IOU_WEIGHT * overlap
    pieces['mask_iou'] = float(overlap.detach())
    pieces['mask_iou_target_mean'] = float(truth.mean().detach())

    empty = empty_mask_term(model.cache['coarse'], pairs)
    total = total + EMPTY_WEIGHT * empty
    pieces['empty_mask'] = float(empty.detach())
    pieces['empty_weight'] = EMPTY_WEIGHT
    return total, pieces


def configuration(arm):
    return {'arm': arm, 'experiment': '730 A1empty as one file, nothing patched at import',
            'parts': ['see', 'find', 'draw'], 'encoder': 'MiT-B0', 'queries': QUERIES,
            'layers': LAYERS, 'dimension': DIMENSION, 'mask_channels': CHANNELS,
            'all_joint': True, 'region': True, 'roi_point': True,
            'dn': 'up to 4 isolated positive noisy-box groups, training only',
            'region_supervision': 'focal class4, mask BCE5/Dice5, box L1 5/GIoU2; each layer and encoder selection',
            'roi_train_points_per_query': 256, 'roi_render_points_per_query_level': 1024,
            'roi_canvas': [14, 56], 'roi_subdivision': [[28, 112], [56, 224]], 'roi_expansion': 1.5,
            'mask_iou_score': {'rule': RULE, 'weight': MASK_IOU_WEIGHT, 'mask_level': OVERLAP_LEVEL,
                               'detached_from_the_trunk': True,
                               'source': 'Mask Scoring R-CNN (Huang et al., CVPR 2019)'},
            'empty_query_term': {'weight': EMPTY_WEIGHT,
                                 'assignment': 'the main stages\' Hungarian pairs, passed as an argument'},
            'asymmetric_roi_term': None,
            'replaces': ['654', '674', '685', '687', '715', '717', '718'],
            'equivalent_to': 'revision_ablation726_empty_only_s42, checked by an identical smoke trace',
            'determinism': '668 fixed-grid operators plus four-gather bilinear sampling'}


def main(argv=None):
    ahead = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ahead.add_argument('--score-rule', default='product_rescaled')
    picked, rest = ahead.parse_known_args(
        sys.argv[1:] if argv is None else list(argv))
    os.environ['HME_SCORE_RULE'] = picked.score_rule

    train = load('675_revision_region_point_train.py')
    architecture = load(Path(__file__).name)
    if architecture.RULE != picked.score_rule:
        raise ValueError('the architecture did not take the requested rule')
    args = train.parse(rest, arms=architecture.ARMS)
    if 'find.mask_iou' not in args.clip_groups:
        raise ValueError('the overlap head is clipped as its own group')
    train.run(args, architecture_file=Path(__file__).name,
              extra_dependencies=(BASE_FILE, Path(__file__).name))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

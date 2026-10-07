"""Joint see/find/draw mechanism transfer, NOT a native MaskDINO reproduction.

References: IDEA-Research/MaskDINO (3831d851), maskdino_decoder.py,
matcher.py; facebookresearch/detectron2 PointRend mask_head.py.
Keep MiT-B0/32 queries/4 layers. Transfer two-stage proposal supervision,
detached reference refinement, hybrid matching, denoising, and ROI subdivision.
Historical 654/668 implementations and their evidence remain immutable.
"""
import importlib.util
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location('transfer674_' + name[:3], ROOT / 'scripts' / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORE = load('654_revision_three_part_model.py')
ARMS = ('b_region', 'c_roi_point', 'd_region_roi_point')
DEFER_METRICS = False
CONSTANT_CACHE = None


def metric(value):
    return value.detach() if DEFER_METRICS else float(value.detach())


def constant(like, values):
    """Opt-in reuse of immutable device constants; no arithmetic is changed."""
    if CONSTANT_CACHE is None:
        return like.new_tensor(values)
    key = like.device, like.dtype, tuple(values)
    if key not in CONSTANT_CACHE:
        CONSTANT_CACHE[key] = like.new_tensor(values)
    return CONSTANT_CACHE[key]


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
    xy = ((coordinates * 2 - 1 + 1) * constant(coordinates, [w, h]) - 1) / 2
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
    return torch.where(xs.any(-1)[..., None], boxes, constant(boxes, [.5, .5, 1., 1.]))


def expanded_roi(boxes, size):
    boxes = boxes.detach()
    minimum = constant(boxes, [4/size[1], 4/size[0]])
    extent = torch.maximum(boxes[..., 2:]*1.5, minimum)
    low = (boxes[..., :2]-extent/2).clamp(0, 1)
    high = (boxes[..., :2]+extent/2).clamp(0, 1)
    return torch.cat((low, high), -1)


def grid(h, w, like):
    return CORE.index_coordinates(torch.arange(h*w, device=like.device)[None], h, w).to(like.dtype)


def paste_roi_probabilities(logits, local, size):
    """As in native mask pasting: interpolate probabilities, zero outside ROI."""
    probability = sample_learned(logits.sigmoid(), local).reshape(len(local), *size)
    inside = ((local >= 0) & (local <= 1)).all(-1).reshape(len(local), *size)
    return torch.logit(probability.clamp(1e-6, 1-1e-6)).masked_fill(~inside, -20.)


class MultiScaleSee(nn.Module):
    def __init__(self, base):
        super().__init__()
        for name, part in base.named_children():
            setattr(self, name, part)
        self.laterals = nn.ModuleList([nn.Sequential(nn.Conv2d(c, 160, 1), nn.GroupNorm(8, 160))
                                      for c in self.encoder.out_channels[-3:]])

    def forward(self, image):
        encoded = self.encoder(image)
        quarter = self.shared_projection(self.dense_decoder(encoded))
        levels = [quarter] + [layer(value) for layer, value in zip(self.laterals, encoded[-3:])]
        fine = self.pixel_decoder(quarter, image)
        return quarter, fine, self.dense_head(fine), levels


class DeformableRead(nn.Module):
    def __init__(self, dim=160, heads=4, levels=4, points=4):
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
        self.position = nn.Sequential(nn.Linear(4, 160), nn.ReLU(), nn.Linear(160, 160))
        self.self_attention = nn.MultiheadAttention(160, 4, dropout=0., batch_first=True)
        self.cross_attention = DeformableRead()
        self.norms = nn.ModuleList([nn.LayerNorm(160) for _ in range(3)])
        self.ffn = nn.Sequential(nn.Linear(160, 640), nn.ReLU(), nn.Linear(640, 160))

    def forward(self, query, features, reference, attention_mask):
        position = self.position(torch.logit(reference.clamp(.001, .999)))
        value = self.self_attention(query+position, query+position, query, attn_mask=attention_mask, need_weights=False)[0]
        query = self.norms[0](query+value)
        query = self.norms[1](query+self.cross_attention(query+position, features, reference))
        return self.norms[2](query+self.ffn(query))


class RegionFinder(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_count = 32
        self.enc_output = nn.Sequential(nn.Linear(160, 160), nn.LayerNorm(160))
        self.level_embed = nn.Parameter(torch.randn(4, 160)*.02)
        self.output_norm = nn.LayerNorm(160)
        self.class_embed = nn.Linear(160, 1)
        self.mask_embed = nn.Sequential(nn.Linear(160, 160), nn.ReLU(), nn.Linear(160, 160), nn.ReLU(), nn.Linear(160, 32))
        self.box_embed = nn.Sequential(nn.Linear(160, 160), nn.ReLU(), nn.Linear(160, 160), nn.ReLU(), nn.Linear(160, 4))
        nn.init.zeros_(self.box_embed[-1].weight)
        nn.init.zeros_(self.box_embed[-1].bias)
        nn.init.constant_(self.class_embed.bias, math.log(.01/.99))
        self.label_embedding = nn.Embedding(1, 160)
        self.layers = nn.ModuleList([RegionLayer() for _ in range(4)])

    def prediction(self, query, pixels, boxes):
        value = self.output_norm(query)
        return (torch.einsum('bqc,bchw->bqhw', self.mask_embed(value), pixels),
                self.class_embed(value).squeeze(-1), boxes)

    def denoising(self, targets, batch, like):
        maximum = max((len(t) for t in targets), default=0)
        groups = min(4, 32//maximum) if maximum else 0
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
        attention = torch.zeros(count+32, count+32, dtype=torch.bool, device=like.device)
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
        indices = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :32]
        content = memory.gather(1, indices[..., None].expand(-1, -1, 160))
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
        self.point_head = nn.Sequential(nn.Linear(33, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def at_points(self, coarse, fine, coords):
        b, q, p = coords.shape[:3]
        feature = CORE.sample_points(fine, coords.reshape(b, q*p, 2)).transpose(1, 2).reshape(b, q, p, 32)
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
            local = CORE.index_coordinates(indices, h, w).reshape(b, q, -1, 2)
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


class ThreePartTransfer(nn.Module):
    def __init__(self, base, arm):
        super().__init__()
        self.arm, self.query_count = arm, 32
        region = arm in ('b_region', 'd_region_roi_point')
        roi = arm in ('c_roi_point', 'd_region_roi_point')
        self.see, self.find, self.draw = base.see, base.find, base.draw
        if region:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(67401)
                self.see, self.find = MultiScaleSee(base.see), RegionFinder()
        if roi:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(67402)
                self.draw = RoiPointDraw()
        self.cache = {}

    def forward(self, image, targets=None):
        if isinstance(self.find, RegionFinder):
            quarter, fine, dense, levels = self.see(image)
            query, stages, dn_stages, dn = self.find(levels, fine, targets)
            coarse, objects, boxes = stages[-1]
            cache = {'stages': stages, 'dn_stages': dn_stages, 'dn': dn, 'auxiliary': [], 'seed_coordinates': boxes[..., :2]}
        else:
            quarter, fine, dense = self.see(image)
            query, coarse, objects, auxiliary, seeds = self.find(quarter, fine, dense)
            boxes = mask_boxes(coarse.detach().sigmoid() >= .3)
            cache = {'auxiliary': auxiliary, 'seed_coordinates': seeds}
        if isinstance(self.draw, RoiPointDraw):
            rendered, points = self.draw(coarse, fine, boxes, image.shape[-2:])
        else:
            rendered, points = self.draw(query, coarse, fine, dense, objects, image.shape[-2:])
        self.cache = {'coarse': coarse, 'dense': dense, 'boxes': boxes, **cache, **points}
        return rendered, objects, boxes


def build_model(baseline, args):
    # Build the identical R3 first so every retained module has the same seed42 initialization.
    from types import SimpleNamespace
    base_args = SimpleNamespace(**{**vars(args), 'arm': 'd_competition'})
    base = CORE.build_model(baseline, base_args)
    if args.arm not in ARMS:
        raise ValueError('unknown transfer arm')
    return ThreePartTransfer(base, args.arm)


def hybrid_match(stage, row, truth):
    from scipy.optimize import linear_sum_assignment
    masks, scores, boxes = stage
    if not truth:
        return [], []
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


def losses(model, masks, scores, targets, probe, fullres):
    if any(len(t) > 32 for t in targets):
        raise ValueError('targets exceed query budget')
    if 'stages' not in model.cache:
        total, pieces = CORE.losses(model, masks, scores, targets, probe, fullres)
        pairs = [probe.hungarian_match(masks[i], scores[i], truth) for i, truth in enumerate(targets)]
    else:
        total = masks.sum()*0
        pieces = {}
        for index, stage in enumerate(model.cache['stages']):
            term, pairs = stage_loss(stage, targets)
            total = total+term
            pieces['stage_'+str(index)] = metric(term)
        dn = model.cache['dn']
        if dn is not None:
            dn_loss = masks.sum()*0
            for stage in model.cache['dn_stages']:
                term, _ = stage_loss(stage, targets, dn[3])
                dn_loss = dn_loss+term/dn[4]
            total = total+dn_loss
            pieces['denoising'] = metric(dn_loss)
        union = torch.stack([torch.stack(t).amax(0) if t else torch.zeros_like(masks[0, 0]) for t in targets])[:, None]
        dense = fullres.matched_mask_loss(model.cache['dense'], union, 'dice', .0)
        total = total+dense
        pieces['dense'] = metric(dense)
        # B keeps the current point and ownership mechanism, with the new final assignment.
        if 'points' in model.cache:
            aligned = torch.zeros_like(masks)
            active = torch.zeros_like(scores, dtype=torch.bool)
            for row, (q, g) in enumerate(pairs):
                for qi, gi in zip(q, g):
                    aligned[row, qi], active[row, qi] = targets[row][int(gi)], True
            coords, logits = model.cache['points']
            gold = CORE.sample_points(aligned, coords)
            point = F.binary_cross_entropy_with_logits(logits[active], gold[active]) if active.any() else logits.sum()*0
            total = total+point
            pieces['point'] = metric(point)
            coords, logits = model.cache['ownership']
            ids = torch.full(masks.shape[:1]+masks.shape[2:], 32, device=masks.device, dtype=torch.long)
            for row in range(len(targets)):
                for q in active[row].nonzero().flatten():
                    ids[row][aligned[row, q].bool()] = q
            labels = CORE.sample_points(ids[:, None].float(), coords, 'nearest')[:, 0].long()
            ce = F.cross_entropy(logits, labels, reduction='none')
            groups = [ce[labels != 32].mean()] if (labels != 32).any() else []
            if (labels == 32).any():
                groups.append(ce[labels == 32].mean())
            ownership = torch.stack(groups).mean()
            total = total+.2*ownership
            pieces['ownership'] = metric(ownership)
    if 'roi_points' in model.cache:
        coords, logits = model.cache['roi_points']
        terms = []
        for row, (q, g) in enumerate(pairs):
            if len(g):
                truth = torch.stack([targets[row][int(i)] for i in g])[:, None]
                gold = CORE.sample_points(truth, coords[row, q])[:, 0]
                terms.append(F.binary_cross_entropy_with_logits(logits[row, q], gold, reduction='none').mean(1))
        point = torch.cat(terms).mean() if terms else logits.sum()*0
        total = total+point
        pieces['roi_point'] = metric(point)
    return total, pieces


def configuration(arm):
    return {'arm': arm, 'parts': ['see', 'find', 'draw'], 'encoder': 'MiT-B0', 'queries': 32,
            'layers': 4, 'dimension': 160, 'mask_channels': 32, 'all_joint': True,
            'region': arm != 'c_roi_point', 'roi_point': arm != 'b_region',
            'dn': 'up to 4 isolated positive noisy-box groups, training only' if arm != 'c_roi_point' else None,
            'region_supervision': 'focal class4, mask BCE5/Dice5, box L1 5/GIoU2; each layer and encoder selection',
            'roi_train_points_per_query': 256, 'roi_render_points_per_query_level': 1024,
            'roi_canvas': [14, 56], 'roi_subdivision': [[28, 112], [56, 224]], 'roi_expansion': 1.5,
            'scope': 'mechanism adaptation; no native six-layer deformable pixel encoder, no full detector port',
            'determinism': '668 fixed-grid operators plus 674 differentiable four-gather bilinear sampling'}

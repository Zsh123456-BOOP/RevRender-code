"""Opt-in FP32 scheduling optimizations; model math and training budget unchanged.

The legacy path remains the reference. Batch matching changes only when costs
are copied to the CPU, not cost arithmetic, draw order, or the SciPy solver.
"""
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'deterministic-fp32-speed-v1'


def materialize(pieces):
    """Copy detached scalar metrics together, preserving Python float values."""
    names = [name for name, value in pieces.items() if torch.is_tensor(value)]
    values = torch.stack([pieces[name].detach().reshape(()) for name in names]).cpu().tolist() if names else []
    return {name: (values[names.index(name)] if name in names else value)
            for name, value in pieces.items()}


def prepare_batch(batch, probe):
    images, masks, _ = batch
    values = np.stack(images)[:, None].astype(np.float32)/127.5-1
    blocks = []
    for mask in masks:
        instances = probe.gold_instances(mask.astype(np.float32)/255)
        blocks.append(np.stack([t.astype(np.float32) for t in instances]) if instances
                      else np.empty((0, *mask.shape), dtype=np.float32))
    return values, blocks


def device_batch(batch, device='cuda'):
    values, blocks = batch
    images = torch.from_numpy(values).to(device)
    targets = [list(torch.from_numpy(block).to(device).unbind(0)) for block in blocks]
    return images, targets


def prepare_stream(stream, probe):
    """Prepare on the existing producer before its consumed-state snapshot.

    No extra producer, randomness, or queue is introduced. The checkpointed
    stream continues to report the state of the last CONSUMED batch.
    """
    original_put = stream._put

    def put(value):
        if value[0] == 'batch':
            value = ('batch', prepare_batch(value[1], probe))
        return original_put(value)

    stream._put = put
    return stream


class Clipper:
    def __init__(self, model, prefixes, limit=5.):
        named = list(model.named_parameters())
        claimed, self.groups = set(), []
        self.limit = limit
        for prefix in prefixes:
            selected = [p for name, p in named if name.startswith(prefix) and name not in claimed]
            if not selected:
                raise ValueError('no parameter matches clip group ' + prefix)
            claimed.update(name for name, _ in named if name.startswith(prefix))
            self.groups.append(selected)
        self.groups.append([p for name, p in named if name not in claimed])

    def __call__(self):
        for group in self.groups:
            if group:
                torch.nn.utils.clip_grad_norm_(group, self.limit, foreach=False, error_if_nonfinite=True)


def matching_cost(transfer, stage, row, truth, class_weight, cache):
    """Same arithmetic order as 674/769; no gradients or random draws added."""
    masks, scores, boxes = stage
    if not truth:
        return None
    with torch.no_grad():
        key = id(truth)
        if key not in cache:
            gold = torch.stack(truth)
            cache[key] = gold, transfer.mask_boxes(gold)
        gold, gt_boxes = cache[key]
        coordinates = torch.rand(1, 4096, 2, device=masks.device)
        # Both coordinates are generated just above by rand: finite by
        # construction, and no adjoint is needed inside matching/no_grad.
        # This is exactly 668.FixedGrid.forward's native operation. Public
        # sampling of supplied/learned coordinates retains its finite guard.
        target = torch.nn.functional.grid_sample(gold[None], coordinates[:, None]*2-1,
            mode='bilinear', padding_mode='border', align_corners=False)[:, :, 0][0]
        logits = torch.nn.functional.grid_sample(masks[row:row+1], coordinates[:, None]*2-1,
            mode='bilinear', padding_mode='border', align_corners=False)[:, :, 0][0]
        probability = logits.sigmoid()
        bce = torch.nn.functional.softplus(logits).mean(1)[:, None]-logits@target.T/logits.shape[1]
        dice = 1-(2*probability@target.T+1)/(probability.sum(1)[:, None]+target.sum(1)[None]+1)
        p = scores[row].sigmoid()
        focal = .25*(1-p).square()*(-(p+1e-8).log()) - .75*p.square()*(-(1-p+1e-8).log())
        return (class_weight*focal[:, None]+5*bce+5*dice
                +5*torch.cdist(boxes[row], gt_boxes, p=1)
                -2*transfer.giou(transfer.xyxy(boxes[row]), transfer.xyxy(gt_boxes)))


def batched_match(transfer, stage, targets, class_weight=4., cache=None):
    from scipy.optimize import linear_sum_assignment
    cache = {} if cache is None else cache
    costs = [matching_cost(transfer, stage, row, truth, class_weight, cache)
             for row, truth in enumerate(targets)]
    nonempty = [cost for cost in costs if cost is not None]
    if not nonempty:
        return [([], []) for _ in targets]
    flat = torch.cat([cost.flatten() for cost in nonempty]).cpu().numpy()
    pairs, offset = [], 0
    for cost in costs:
        if cost is None:
            pairs.append(([], []))
        else:
            end = offset + cost.numel()
            pairs.append(linear_sum_assignment(flat[offset:end].reshape(cost.shape)))
            offset = end
    return pairs


def install(architecture):
    """Only registered 767/769 models are supported; never silently generalize."""
    coarse = getattr(architecture, 'COARSE', architecture)
    if not hasattr(coarse, 'PLANES') or coarse.POINT_WEIGHT != 1.:
        raise ValueError('speed v1 supports 767/769 with point weight one')
    transfer, score = coarse.TRANSFER, coarse.BASE.SCORE
    if getattr(transfer, '_speed_installed', False):
        raise ValueError('speed implementation already installed')
    transfer.DEFER_METRICS = score.DEFER_METRICS = True
    transfer.CONSTANT_CACHE = {}
    original_stage, original_losses = transfer.stage_loss, architecture.losses
    cache = {}
    weight = getattr(architecture, 'CLASS_WEIGHT', 4.)

    def stage_loss(stage, targets, pairs=None):
        if pairs is None:
            pairs = batched_match(transfer, stage, targets, weight, cache)
        device_pairs = [(torch.as_tensor(q, device=stage[0].device, dtype=torch.long), g) for q, g in pairs]
        term, _ = original_stage(stage, targets, device_pairs)
        return term, pairs

    def losses(*args, **kwargs):
        cache.clear()
        try:
            return original_losses(*args, **kwargs)
        finally:
            cache.clear()

    transfer.stage_loss = stage_loss
    architecture.losses = losses
    transfer._speed_installed = True
    return {'version': VERSION, 'deferred_metrics': True, 'batched_cost_transfer': True,
            'target_preparation': 'existing single producer', 'cached_clip_groups': True,
            'cached_constants': True, 'query_indices_transfer': 'once per stage and row',
            'matching_sampling': 'native forward on freshly generated finite rand coordinates; no_grad',
            'precision': 'FP32', 'determinism': 'unchanged 668',
            'class_weight': weight, 'cost_rng_order': 'stage then row, unchanged',
            'fill_uninitialized_memory': torch.utils.deterministic.fill_uninitialized_memory}

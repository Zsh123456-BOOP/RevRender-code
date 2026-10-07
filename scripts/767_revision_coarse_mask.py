"""Three ways at the one tensor every other stage copies: the coarse mask.

766 asked the renderer's point head about 8 percent of the finest level's cells,
then 33, then every one of them, and component F1 moved by 0.0000 at every
budget.  The head is not short of opportunities.  763 had already shown it is
not free of the coarse mask either: removing that input made "the coarse mask
reached it and the renderer lost it" go 3 to 10.  Put together, the renderer
copies the coarse mask, and asking it more often copies it more often.

So the coarse mask is where the shortfall is, and 744 measures it: a median IoU
of 0.40 on the misshapen targets, which the renderer lifts to 0.43.  708 says
what its errors look like -- the delivered mask is 0.66 of its target's area and
covers 41 percent of it, which is the densest part and not the rest.

674 builds it in one line:

    torch.einsum('bqc,bchw->bqhw', self.mask_embed(value), pixels)

a 32-dimensional query embedding dotted with a 32-channel map pooled to a
quarter of the strip.  That is one hyperplane per query in feature space,
evaluated on a 24 by 192 grid, and it is supervised at weight 5 BCE plus 5 dice
while the full-resolution point term is added at weight 1.

Each of those three is a candidate and they are separable:

  stride    ``HME_COARSE_STRIDE=2`` pools the detail map to half the strip
            rather than a quarter, so the mask is decided on 48 by 384.  The
            gold components have a median height of 33 pixels, which is eight
            cells at a quarter and sixteen at a half, and the targets that go
            missing are 27 by 10.
  planes    ``HME_MASK_PLANES`` predicts that many embeddings per query and
            takes the largest response, so the mask is a maximum of hyperplanes
            instead of one.  A revision whose ink is uneven -- a filled blot
            with a thin tail -- cannot be cut out by a single hyperplane, and
            "keeps the densest part" is what that failure looks like.
  point     ``HME_POINT_WEIGHT`` multiplies the ROI point term, which 674 adds
            at 1 beside a coarse mask at 5 and 5.  The stage that sees full
            resolution has a tenth of the say, and copying what is ten times
            louder is what a learner should be expected to do.

Each defaults to 674's value, so an arm moves one thing.  The copied forward
must reproduce 674's outputs exactly at stride 4 and one plane, and the
recomputed point term must equal the one 674 recorded; both are asserted,
because a copy that drifts would make every comparison meaningless.

Bars, registered before the run, against r50_ctrl at the same seed:
  gate      675's train-only learning gate must pass.
  primary   708's partials in [0.40, 0.50) must fall with true positives up and
            false positives down together, and 760's "a box reached it and the
            mask did not" must fall below 28.
  guard     overall component F1 must not fall below the control's 0.6165, and
            760's "no box reached it" must not move -- the finder sees nothing
            different in the stride and plane arms, so if it moves, something
            else changed and the reading is void.

This script deliberately does NOT: change the finder's selection, the anchor
prior, the number of proposals or queries, the renderer's point budget or its
subdivisions, the ROI or its expansion, the matcher, the box or class terms, the
denoising groups, the schedule, the budget, the seed or the gate; touch 674,
687, 751 or any file another run is executing; re-select an operating point;
read, transcribe or otherwise consume the frozen test split; or contact the
network.
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
BASE_FILE = '687_revision_region_point_maskiou.py'
RESNET_FILE = '750_revision_resnet50_base.py'
STRIDE_VARIABLE = 'HME_COARSE_STRIDE'
PLANES_VARIABLE = 'HME_MASK_PLANES'
WEIGHT_VARIABLE = 'HME_POINT_WEIGHT'
STRIDE_ALLOWED = (4, 2)
PLANES_ALLOWED = (1, 2, 4)
BASE_STRIDE = 4
BASE_PLANES = 1
BASE_WEIGHT = 1.
MASK_CHANNELS = 32
DIMENSION = 160


def load(name):
    spec = importlib.util.spec_from_file_location('coarse767_' + name[:3],
                                                  ROOT / 'scripts' / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load(BASE_FILE)
RESNET = load(RESNET_FILE)
TRANSFER = BASE.TRANSFER
ARMS, CORE, RULE = BASE.ARMS, BASE.CORE, BASE.RULE
STRIDE = int(os.environ.get(STRIDE_VARIABLE, BASE_STRIDE))
PLANES = int(os.environ.get(PLANES_VARIABLE, BASE_PLANES))
POINT_WEIGHT = float(os.environ.get(WEIGHT_VARIABLE, BASE_WEIGHT))
if STRIDE not in STRIDE_ALLOWED:
    raise ValueError(STRIDE_VARIABLE + ' is one of ' + ', '.join(map(str, STRIDE_ALLOWED)))
if PLANES not in PLANES_ALLOWED:
    raise ValueError(PLANES_VARIABLE + ' is one of ' + ', '.join(map(str, PLANES_ALLOWED)))
if POINT_WEIGHT <= 0.:
    raise ValueError(WEIGHT_VARIABLE + ' multiplies a loss, so it is positive')


class ResNet50Baseline:
    """422, except that the Segformer it builds has the baselines' ResNet-50."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def _smp_model(self, name, args):
        import segmentation_models_pytorch as smp

        if name != 'segformer_b0':
            raise ValueError('only the MiT-B0 Segformer is replaced, not ' + name)
        segmenter = smp.Segformer(encoder_name='resnet50', encoder_weights=None,
                                  in_channels=1, classes=1)
        RESNET.load_resnet50(segmenter.encoder, args.segformer_b0_weights)
        return segmenter


def widen_mask_embed(find):
    """The last layer of the mask head predicts PLANES embeddings, not one.

    Built inside ``fork_rng`` so every other parameter is drawn from the stream
    674 draws it from and the arm differs in this layer alone.
    """
    last = find.mask_embed[-1]
    if last.out_features != MASK_CHANNELS:
        raise ValueError("674's mask head no longer ends in 32 channels")
    with torch.random.fork_rng(devices=[]):
        wide = torch.nn.Linear(last.in_features, PLANES*MASK_CHANNELS,
                               bias=last.bias is not None)
    find.mask_embed[-1] = wide
    return find


def build_model(baseline, args):
    model = BASE.build_model(ResNet50Baseline(baseline), args)
    if PLANES != BASE_PLANES:
        widen_mask_embed(model.find)
    return model


def planed_prediction(self, query, pixels, boxes):
    """674's prediction with the mask as the largest of PLANES responses.

    At one plane the reshape is a no-op and ``amax`` over a length-one axis is
    the identity, so this is 674's own einsum.
    """
    value = self.output_norm(query)
    embedding = self.mask_embed(value)
    batch, queries = embedding.shape[:2]
    embedding = embedding.reshape(batch, queries, PLANES, MASK_CHANNELS)
    masks = torch.einsum('bqkc,bchw->bqkhw', embedding, pixels).amax(2)
    return masks, self.class_embed(value).squeeze(-1), boxes


def strided_forward(self, levels, fine, targets=None):
    """674's RegionFinder.forward with the detail map pooled at STRIDE.

    Every other line is 674's, in 674's order, so at stride 4 -- which is what
    ``levels[0]`` already is -- the outputs are the original's.
    """
    b = fine.shape[0]
    height, width = fine.shape[-2:]
    pixels = TRANSFER.F.adaptive_avg_pool2d(fine, (height//STRIDE, width//STRIDE))
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
    indices = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :32]
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


ORIGINAL_LOSSES = BASE.losses


def point_term(model, targets):
    """674's ROI point term, recomputed from the same cache and the same match.

    674 keeps the assignment local to ``losses``; the matcher is deterministic
    and this is the line ``stage_loss`` runs for the final stage, so the pairs
    are the ones the original used.  The test asserts the value matches what
    674 recorded.
    """
    if 'roi_points' not in model.cache or 'stages' not in model.cache:
        return None
    stage = model.cache['stages'][-1]
    pairs = [TRANSFER.hybrid_match(stage, row, truth) for row, truth in enumerate(targets)]
    coords, logits = model.cache['roi_points']
    terms = []
    for row, (q, g) in enumerate(pairs):
        if len(g):
            truth = torch.stack([targets[row][int(i)] for i in g])[:, None]
            gold = CORE.sample_points(truth, coords[row, q])[:, 0]
            terms.append(TRANSFER.F.binary_cross_entropy_with_logits(
                logits[row, q], gold, reduction='none').mean(1))
    return torch.cat(terms).mean() if terms else logits.sum()*0


def losses(model, masks, scores, targets, probe, fullres):
    """674's losses, with the ROI point term carrying POINT_WEIGHT.

    674 already added it once at weight one, so only the remainder is added
    here and at weight one nothing is computed and nothing is added.
    """
    total, pieces = ORIGINAL_LOSSES(model, masks, scores, targets, probe, fullres)
    if POINT_WEIGHT == BASE_WEIGHT:
        return total, pieces
    extra = point_term(model, targets)
    if extra is None:
        return total, pieces
    pieces = dict(pieces)
    pieces['roi_point_weight'] = POINT_WEIGHT
    return total+(POINT_WEIGHT-BASE_WEIGHT)*extra, pieces


if STRIDE != BASE_STRIDE:
    TRANSFER.RegionFinder.forward = strided_forward
if PLANES != BASE_PLANES:
    TRANSFER.RegionFinder.prediction = planed_prediction


def configuration(arm):
    value = dict(BASE.configuration(arm))
    height, width = 96, 768
    value.update(experiment='767 three ways at the coarse mask',
                 encoder='ResNet-50, R-50-torchvision.pkl; 767.ResNet50Baseline',
                 coarse_stride=STRIDE,
                 coarse_grid=[height//STRIDE, width//STRIDE],
                 mask_planes=PLANES,
                 mask_embed_out=PLANES*MASK_CHANNELS,
                 roi_point_weight=POINT_WEIGHT,
                 coarse_mask_weight="674's 5 BCE and 5 dice, unchanged",
                 finder="674's selection, anchors and proposals, unchanged",
                 renderer="674's point budget and subdivisions, unchanged",
                 control='r50_ctrl at the same seed')
    value['bars'] = {'learning_gate': 'must pass',
                     'primary': "708's partials in [0.40, 0.50) and 760's "
                                'mask-did-not-reach count',
                     'guard': "overall F1 must not fall and 760's no-box count "
                              'must not move'}
    return value


def main(argv=None):
    ahead = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ahead.add_argument('--coarse-stride', type=int, choices=STRIDE_ALLOWED, default=BASE_STRIDE)
    ahead.add_argument('--mask-planes', type=int, choices=PLANES_ALLOWED, default=BASE_PLANES)
    ahead.add_argument('--point-weight', type=float, default=BASE_WEIGHT)
    ahead.add_argument('--score-rule', default='product_rescaled')
    picked, rest = ahead.parse_known_args(sys.argv[1:] if argv is None else list(argv))
    os.environ[STRIDE_VARIABLE] = str(picked.coarse_stride)
    os.environ[PLANES_VARIABLE] = str(picked.mask_planes)
    os.environ[WEIGHT_VARIABLE] = repr(picked.point_weight)
    os.environ['HME_SCORE_RULE'] = picked.score_rule
    train = RESNET.repoint(load('675_revision_region_point_train.py'),
                           RESNET.MIT_B0_SHA)
    architecture = load(Path(__file__).name)
    if (architecture.STRIDE, architecture.PLANES, architecture.POINT_WEIGHT) != \
            (picked.coarse_stride, picked.mask_planes, picked.point_weight):
        raise ValueError('the architecture did not take the requested switches')
    args = train.parse(rest, arms=architecture.ARMS)
    train.run(args, architecture_file=Path(__file__).name,
              extra_dependencies=('685_revision_mask_iou_score.py', BASE_FILE,
                                  RESNET_FILE, Path(__file__).name))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

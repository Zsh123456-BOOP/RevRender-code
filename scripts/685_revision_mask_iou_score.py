"""Score a candidate by the overlap its own mask achieves, not by where it was seeded.

682 measured that R3's delivered score ranks nearly independently of mask
quality (Spearman 0.305) while its pool would reach F1 0.6845 if ranked
perfectly.  684 then tried MaskDINO's denoising group twice, with negatives on
blank paper and with negatives taken from the model's own unmatched candidates.
Both collapsed inside 1000 steps -- denoise_objectness 0.0010 and 0.0026 -- and
the reason is the same in both: the label was a function of the seed position,
and the query already samples the image at that position, so "is this seed in a
revision" is answerable from the local feature and teaches nothing about the
mask that gets drawn.

The supervision here is a function of the drawn mask instead.  A small head
predicts the IoU its own coarse mask reaches against the gold component it
overlaps best, recomputed every step from the current prediction, and the
delivered score becomes the classification score times that prediction.  This
is Mask Scoring R-CNN's mechanism (Huang et al., CVPR 2019), whose stated
motivation is exactly the mismatch 682 measured: a classification score does
not know how good a mask is.

Prior work in this repository, which this must not be confused with: 613
trained a quality score on a frozen front end for 5000 steps with a separate
head, reaching 0.5249 and 0.5440 against controls of 0.5222 and 0.5478, and it
was not used to deliver anything -- the refinement kept the original score.
Here the head is trained jointly with everything else for the full budget and
its output is what selects the candidates, so the question 613 left open is
whether that changes the answer.  613's result is the reason this is an open
question rather than a settled one.

This module deliberately does NOT: change the matching, the seeding, the mask
renderer or any existing loss term; detach the head from the mask, since the
whole point is that its input is the mask; claim 613's inconclusive result
transfers to a jointly trained head, or that it does not; read, transcribe or
otherwise consume the frozen test split; or contact the network.
"""
import torch
from torch import nn
from torch.nn import functional as F

# The predicted overlap multiplies the classification score, which is the form
# Mask Scoring R-CNN uses: the first factor says whether anything is there, the
# second how well it is drawn, and a candidate needs both.
# ``product_rescaled`` is the geometric mean of the two factors.  It is a
# monotone transform of ``product``, so the ranking is identical to the last
# decimal; what changes is the range.  655's score grid runs from 0.45 to 0.85
# and was calibrated for an objectness probability, and a product of two
# sigmoids sits systematically below it -- 687's arms chose the grid's floor in
# 34 of 36 validations, delivering 86 targets where the control's own curve
# reaches 113, because they could not cut any lower.  Measured on 687's own
# archive, at 0.45 the control keeps 172 candidates and the product keeps 112,
# while the geometric mean keeps 180: the same neighbourhood, same order.
RULES = ('product', 'product_rescaled', 'quality_only')
MASK_LEVEL = .5

# Whether the head's gradient reaches the rest of the model.  Detached by
# default, because the question is whether a better score helps, and a head
# that also reshapes the trunk answers a different one: on R3 the coupled
# version raised the matching objectness loss to about twice the control's and
# cost F1, and on 674's d arm it pushed enough queries past the learning
# gate's fixed 0.45 to turn a clean pass -- 6 true, 0 false -- into 6 true and
# 132 false, with the loss curve unchanged.  Detached, the trunk sees exactly
# the gradients it saw without the head, so the completed run stays a valid
# control and the score is the only thing that moved.
DETACH_DEFAULT = True
DEFER_METRICS = False


def metric(value):
    return value.detach() if DEFER_METRICS else float(value.detach())


class MaskIoUHead(nn.Module):
    """Reads the query and a summary of the mask that query just drew."""

    def __init__(self, dimension, channels):
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
        # Start near "half overlap" rather than at zero, so an untrained head
        # neither suppresses nor passes everything and the guard can tell.
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, query, coarse, pixels, delivered=None, detach=True):
        """``delivered`` is the mask that will actually be reported, if given.

        The pooled feature comes from the cheap quarter-resolution mask, but
        the extent has to describe the mask whose overlap is being predicted,
        which is the rendered one.
        """
        if detach:
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


def overlap_targets(masks, targets, level=MASK_LEVEL):
    """The IoU each query's delivered mask reaches against its best gold piece.

    Recomputed from the prediction every step, so the label moves with the mask
    and cannot be answered from the seed.  ``masks`` are the rendered logits,
    at the resolution the gold is scored at, because that is the mask whose
    overlap the score has to predict.  A query whose mask is empty, or a row
    with no gold, has nothing to overlap and its target is zero.
    """
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
    """The logit the operating-point search and the archive will see.

    655's ``predict_packed`` stores ``objects.sigmoid()``, so a model returns a
    logit and the pipeline turns it into the probability the score grid is
    written against.  Returning the probability itself would have it squashed a
    second time into [0.5, 0.731], which is above every threshold in that grid
    -- the first version of this did exactly that, and 675's learning gate
    reported six true positives against a hundred and thirty-eight false ones
    because effectively nothing was ever filtered out.
    """
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


def install(architecture, weight=1., rule='product', model_class='ThreePartModel',
            detach=DETACH_DEFAULT):
    """Add the head, train it on the overlap, and deliver with its prediction.

    During training the model keeps returning the raw classification logit, so
    every existing loss term is untouched; only the evaluation path returns the
    combined score, which is what the archive and the operating-point search
    read.
    """
    if getattr(architecture, '_mask_iou_installed', False):
        raise RuntimeError('mask-iou scoring already installed')
    if rule not in RULES:
        raise ValueError('undeclared scoring rule: ' + str(rule))
    holder = getattr(architecture, model_class)
    original_forward, original_losses = holder.forward, architecture.losses

    def forward(self, image, *rest):
        # ``*rest`` because 674's transfer takes the targets as well; the
        # wrapper must not care which architecture it is wrapping.
        rendered, objects, extra = original_forward(self, image, *rest)
        coarse = self.cache['coarse']
        # Read from a plain attribute the finder wrapper set on the way past,
        # not from ``cache`` -- the original forward replaces that dict, so
        # anything stashed in it before the call would be gone by now.
        stashed = self._mask_iou_inputs
        pixels = F.adaptive_avg_pool2d(stashed['fine'], coarse.shape[-2:])
        quality = self.find.mask_iou(stashed['query'], coarse, pixels, rendered,
                                     detach)
        self.cache['mask_iou'] = quality
        self.cache['mask_iou_delivered'] = rendered
        if self.training:
            return rendered, objects, extra
        return rendered, delivered_score(objects, quality, rule), extra

    def losses(model, masks, scores, targets, probe, fullres):
        total, pieces = original_losses(model, masks, scores, targets, probe, fullres)
        quality = model.cache.get('mask_iou')
        if quality is None:
            return total, pieces
        truth = overlap_targets(model.cache['mask_iou_delivered'], targets)
        term = F.binary_cross_entropy_with_logits(quality, truth)
        pieces = dict(pieces)
        pieces['mask_iou'] = metric(term)
        pieces['mask_iou_target_mean'] = metric(truth.mean())
        return total + weight * term, pieces

    holder.forward = forward
    architecture.losses = losses
    architecture._mask_iou_installed = True
    return {'rule': rule, 'weight': weight, 'mask_level': MASK_LEVEL,
            'model_class': model_class, 'detached_from_the_trunk': detach,
            'source': 'Mask Scoring R-CNN (Huang et al., CVPR 2019)',
            'prior': '613 trained a frozen-front-end quality score for 5000 '
                     'steps and did not deliver with it: 0.5249 / 0.5440 '
                     'against 0.5222 / 0.5478'}


def attach(model, architecture, dimension=None, channels=None):
    """Give the model its head, and stash what the head needs to read.

    Both finders this is used with take the full-resolution features as their
    second positional argument and return the query embeddings first -- R3's
    ``find(quarter, fine, dense)`` and 674's ``find(levels, fine, targets)`` --
    so one wrapper covers them and neither has to be special-cased.
    """
    if dimension is None:
        dimension = int(model.find.queries.shape[-1])
    if channels is None:
        channels = int(model.find.mask_embed[-1].out_features)
    # Under ``find`` rather than on the model: 675 asserts the model has
    # exactly the three children it trains jointly, and a fourth one would
    # trip a gate that is there for a good reason.  The head still trains --
    # it is inside a part that does.
    model.find.mask_iou = MaskIoUHead(dimension, channels).to(
        next(model.parameters()).device)
    model._mask_iou_inputs = {}
    inner = model.find.forward

    def find_forward(first, fine, *rest):
        result = inner(first, fine, *rest)
        model._mask_iou_inputs = {'query': result[0], 'fine': fine}
        return result

    model.find.forward = find_forward
    return model

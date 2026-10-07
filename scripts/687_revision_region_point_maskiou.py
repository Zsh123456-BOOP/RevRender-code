"""676-D's region locator with 685's overlap score, as one architecture module.

682 and 686 together located the headroom.  R3's ranking is already reasonable
-- average precision 0.721 -- and its perfect-selection ceiling is only 0.6845,
so its limit is the pool, and putting the overlap score on it moved average
precision by 0.013 and cost F1.  676-D is the opposite model: its pool covers
174 of the 221 targets and its ceiling is 0.8810, while its average precision
is 0.295 and it delivers 0.620 of that ceiling against R3's 0.777.  So this
puts the score where the ranking is actually bad and the pool is actually
large.

Nothing is reimplemented here.  674 supplies the architecture, 685 supplies the
head, and this module exists so 675 can train the pair without being modified:
it already takes the architecture file as an argument, so a module that
re-exports 674's surface with the head installed is the whole change.

Pre-registered, and the mechanism bar is average precision this time rather
than the rank correlation, which 686 showed can be doubled without moving
anything that is delivered:

    mechanism   average precision must exceed 0.60, against 0.295.
    result      component F1 at IoU 0.50 must exceed 0.5751, which is 676-D's
                own figure, and MaskDINO's 0.5753.
    guardrail   pool coverage must stay at or above 157, nine tenths of 174, so
                a gain cannot come from quietly proposing less.

This module deliberately does NOT: change 674's locator, its denoising group,
its losses or its renderer; change 675's training loop, schedule or budget;
re-select 676-D's operating point rule; read, transcribe or otherwise consume
the frozen test split; or contact the network.
"""
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Read once, at import, because installing the head patches a class and cannot
# be deferred.  688 sets it; the value lands in the profile and therefore in
# the run manifest and the gate's hashed evidence, so a run cannot claim one
# rule and have used another.
RULE = os.environ.get('HME_SCORE_RULE', 'product_rescaled')


def load(name):
    spec = importlib.util.spec_from_file_location('maskiou687_' + name[:3],
                                                  ROOT / 'scripts' / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TRANSFER = load('674_revision_region_point_model.py')
SCORE = load('685_revision_mask_iou_score.py')

# 675 reaches through to the original R3 module to install the deterministic
# spatial wrappers, so it has to keep pointing at the same object.
CORE = TRANSFER.CORE
ARMS = TRANSFER.ARMS
# RegionFinder carries no ``queries`` parameter and its mask embedding ends at
# 32 channels, so the head is told both rather than guessing from R3's shapes.
DIMENSION = 160
CHANNELS = 32
MECHANISM_BAR = .60
RESULT_BAR = .5751
POOL_GUARDRAIL = 157

# Rescaled: the ranking is identical to the plain product, but the values
# reach the score grid 675 searches.  The plain rule pinned both arms to
# the grid's floor for 34 of 36 validations.
PROFILE = SCORE.install(TRANSFER, rule=RULE,
                        model_class='ThreePartTransfer')


def configuration(arm):
    value = dict(TRANSFER.configuration(arm))
    value['mask_iou_score'] = dict(PROFILE)
    value['bars'] = {'average_precision': MECHANISM_BAR,
                     'instance_f1': RESULT_BAR,
                     'pool_coverage': POOL_GUARDRAIL}
    return value


def build_model(baseline, args):
    model = TRANSFER.build_model(baseline, args)
    return SCORE.attach(model, TRANSFER, dimension=DIMENSION,
                        channels=CHANNELS)


def losses(model, masks, scores, targets, probe, fullres):
    # Looked up at call time: installing the head replaced the attribute.
    return TRANSFER.losses(model, masks, scores, targets, probe, fullres)

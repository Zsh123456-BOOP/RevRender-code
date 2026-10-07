"""The corrected model (point_context) with one declared recipe change: the tuning campaign of 2026-09-28.

The user asked on 2026-09-28 for the recipe to be tuned (it never was on the
corrected data) and for in-domain pretraining to be tried in parallel.  869
measured the corrected model's fit on training lines at .889 against .686 on
validation writers, so every change here aims at generalisation.  675 hard-codes
one AdamW group (lr 3e-4, weight decay 1e-4) for all 27.7M parameters,
including the ImageNet ResNet-50, and refuses a model with any frozen
parameter; the changes therefore act inside the optimizer step and the
modules, and 675 is not edited.  ``HME_TUNE`` picks the mode:

  control   nothing (865 ``off``); on validation gold v2 it is the reference
  bb01      the ResNet-50 (see.encoder) at 0.1x the learning rate
  wd1e3     weight decay 1e-4 -> 1e-3
  lr05      every learning rate at 0.5x (3e-4 -> 1.5e-4 at the peak)
  frozenbn  the ResNet-50's batch norms keep their ImageNet statistics (eval
            mode) and, with the stem and layer1, learn at rate 0
  drop01    dropout 0.1 in the four decoder layers: attention weights, and the
            self-attention, deformable-read and FFN outputs before each residual
  ssl       the see stage (encoder and decoders, not the revision head)
            starts from 879's in-domain masked-reconstruction pretraining
  nofine    ablation of the full-resolution detail pathway: the pixel decoder's
            image skips (full and half resolution) are fed zeros, so the detail
            map the renderer and the coarse masks read carries only what the
            quarter-resolution features hold; every module and parameter stays
  nocontext ablation of the renderer's ROI context: 816's ``point_ms`` instead of
            ``point_context`` -- the same point residual and multi-scale point
            features, without the cross-attention of each point to the 32
            fixed points of its box (the shared layers are built first in both,
            so every common weight starts identical)
  nodense   ablation of the per-pixel supervision of the detail map: 674's
            dense Dice term is replaced by an exact zero that keeps the graph
            (the dense head still exists and gets zero gradient); everything
            else, including every other use of the detail map, is unchanged
  nofine_nodense  both of the above detail-map ablations together: with the
            renderer read off by 886's coarse variant, the model without FSR

Learning-rate multipliers are applied inside AdamW.step() around 675's own
per-step schedule, so the schedule is unchanged and 675's recipe string still
states the base; the profile carries this module's ``tuning880`` record with
the effective values.  A rate of 0 moves nothing: AdamW's decoupled decay and
its update are both scaled by the rate.  At mode ``control`` nothing is
installed.

This module deliberately does NOT: change the data, the losses, the matcher,
the schedule's shape, the step budget, early stopping or the freeze rule; read
the frozen common test; or contact the network.
"""
import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
MODES = {'control': {}, 'bb01': {'backbone_lr': .1}, 'wd1e3': {'weight_decay': 1e-3}, 'lr05': {'lr_scale': .5},
         'frozenbn': {'frozen_bn': True}, 'drop01': {'dropout': .1}, 'ssl': {'ssl': True},
         'nofine': {'ablation': 'no_fullres_detail'}, 'nocontext': {'ablation': 'no_context_attention'},
         'nodense': {'ablation': 'no_pixel_supervision'},
         'nofine_nodense': {'ablation': 'no_fullres_detail+no_pixel_supervision'}}
MODE = os.environ.get('HME_TUNE', 'control')
if MODE not in MODES:
    raise ValueError('undeclared tuning mode: '+MODE)
KNOBS = dict(MODES[MODE])
ABLATIONS = set(KNOBS['ablation'].split('+')) if 'ablation' in KNOBS else set()
SSL_CHECKPOINT = os.environ.get('HME_TUNE_SSL', '')
SSL_SHA = os.environ.get('HME_TUNE_SSL_SHA', '')
if ('ssl' in KNOBS) != bool(SSL_CHECKPOINT):
    raise ValueError('the ssl mode needs HME_TUNE_SSL (and only it)')
os.environ['HME_REBUILD'] = 'off'
BASE_LR, BASE_WD = 3e-4, 1e-4
STEM = ('see.encoder.conv1.', 'see.encoder.bn1.', 'see.encoder.layer1.')
NOT_TRANSFERRED = ('dense_head.',)


def load(name):
    spec = importlib.util.spec_from_file_location('tune880_'+name[:3], ROOT/'scripts'/name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ARCH = load('865_revision_rebuild_v2.py')
if ARCH.MODE != 'off' or ARCH.ON:
    raise RuntimeError('880 builds on 865 off only')
COARSE, TRANSFER, CORE, ARMS, RULE = ARCH.COARSE, ARCH.TRANSFER, ARCH.CORE, ARCH.ARMS, ARCH.RULE
if ARCH.ARCH.VARIANT != 'point_context':
    raise RuntimeError('880 expects 865 to build 816 point_context')
if 'no_context_attention' in ABLATIONS:
    ARCH.ARCH.VARIANT = 'point_ms'      # read by 816's build_model and configuration at call time
STATE = {'installed': False, 'groups': None}


def require(condition, message):
    if not condition:
        raise RuntimeError('880 tuning refused: '+message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def rate_of(name, knobs=None):
    """The learning-rate multiplier of one parameter under the mode's knobs."""
    knobs = KNOBS if knobs is None else knobs
    rate = knobs.get('lr_scale', 1.)
    if name.startswith('see.encoder.'):
        rate *= knobs.get('backbone_lr', 1.)
    return rate


def tuned_adamw():
    """AdamW whose groups carry lr multipliers applied around 675's per-step lr, and the mode's weight decay."""
    import torch

    class TunedAdamW(torch.optim.AdamW):
        def __init__(self, params, **options):
            params = list(params)
            groups = {}
            for p in params:
                groups.setdefault(p._tune880_rate, []).append(p)
            options['weight_decay'] = KNOBS.get('weight_decay', options.get('weight_decay', BASE_WD))
            super().__init__([{'params': ps, 'lr_multiplier': rate} for rate, ps in groups.items()], **options)
            STATE['groups'] = {str(rate): len(ps) for rate, ps in groups.items()}

        def step(self, closure=None):
            bases = [g['lr'] for g in self.param_groups]
            for g in self.param_groups:
                g['lr'] = g['lr']*g['lr_multiplier']
            try:
                return super().step(closure)
            finally:
                for g, base in zip(self.param_groups, bases):
                    g['lr'] = base
    return TunedAdamW


def freeze_batch_norms(encoder):
    """The encoder's batch norms stay in eval mode whatever train() is called with."""
    import torch
    frozen = 0
    for module in encoder.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.train = MethodType(lambda self, mode=True: torch.nn.Module.train(self, False), module)
            module.eval()
            frozen += 1
    return frozen


def add_dropout(layers, p):
    """Dropout p on attention weights and on the three sublayer outputs of each decoder layer."""
    import torch
    from torch.nn import functional as F

    def on_attention(module, inputs, output):
        return (F.dropout(output[0], p, module.training),) + tuple(output[1:])

    def on_tensor(module, inputs, output):
        return F.dropout(output, p, module.training)
    for layer in layers:
        require(isinstance(layer.self_attention, torch.nn.MultiheadAttention), 'unexpected decoder layer')
        layer.self_attention.dropout = p
        layer.self_attention.register_forward_hook(on_attention)
        layer.cross_attention.register_forward_hook(on_tensor)
        layer.ffn.register_forward_hook(on_tensor)
    return len(layers)


def remove_image_skips(pixel_decoder):
    """The pixel decoder sees a blank image, so its full- and half-resolution skips carry nothing."""
    import torch
    original = pixel_decoder.forward

    def forward(self, quarter, image):
        return original(quarter, torch.zeros_like(image))
    pixel_decoder.forward = MethodType(forward, pixel_decoder)
    return pixel_decoder


class WithoutPixelSupervision:
    """675's full-resolution loss module, except that the detail map's dense Dice term is an exact zero."""

    def __init__(self, fullres, model):
        self._fullres, self._model, self.silenced = fullres, model, 0

    def __getattr__(self, name):
        return getattr(self._fullres, name)

    def matched_mask_loss(self, prediction, target, *args, **kwargs):
        if prediction is self._model.cache.get('dense'):
            self.silenced += 1
            return prediction.sum()*0
        return self._fullres.matched_mask_loss(prediction, target, *args, **kwargs)


def load_pretrained(see, path, sha):
    """879's see-stage weights, all but the revision head; the checkpoint's digest must match."""
    import torch
    require(digest(path) == sha, 'the pretraining checkpoint is not the declared one')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    require(payload.get('schema') == 'hme-revision-ssl-pretrain879-v1', 'not an 879 checkpoint')
    state = {k: v for k, v in payload['see_state_dict'].items() if not k.startswith(NOT_TRANSFERRED)}
    missing, unexpected = see.load_state_dict(state, strict=False)
    require(not unexpected and all(k.startswith(NOT_TRANSFERRED) or k.endswith('num_batches_tracked') for k in missing),
            f'see-stage mismatch: missing={missing}, unexpected={unexpected}')
    return len(state)


def build_model(baseline, args):
    import torch
    model = ARCH.build_model(baseline, args)
    if not KNOBS:
        return model
    if 'ssl' in KNOBS:
        STATE['ssl_tensors'] = load_pretrained(model.see, SSL_CHECKPOINT, SSL_SHA)
    if KNOBS.get('frozen_bn'):
        STATE['frozen_bn'] = freeze_batch_norms(model.see.encoder)
    if 'dropout' in KNOBS:
        STATE['dropout_layers'] = add_dropout(model.find.layers, KNOBS['dropout'])
    if 'no_fullres_detail' in ABLATIONS:
        remove_image_skips(model.see.pixel_decoder)
    if 'no_context_attention' in ABLATIONS:
        require(not model.draw.alignment_residual.contextual, 'the renderer still attends to its ROI points')
    frozen_affine = {id(p) for m in model.see.encoder.modules() if isinstance(m, torch.nn.BatchNorm2d)
                     for p in m.parameters()} if KNOBS.get('frozen_bn') else set()
    for name, p in model.named_parameters():
        rate = rate_of(name)
        if KNOBS.get('frozen_bn') and (name.startswith(STEM) or id(p) in frozen_affine):
            rate = 0.
        p._tune880_rate = rate
    if not STATE['installed'] and ({'backbone_lr', 'weight_decay', 'lr_scale', 'frozen_bn'} & set(KNOBS)):
        torch.optim.AdamW = tuned_adamw()
        STATE['installed'] = True
    return model


def losses(model, masks, scores, targets, probe, fullres):
    if 'no_pixel_supervision' not in ABLATIONS:
        return ARCH.losses(model, masks, scores, targets, probe, fullres)
    silent = WithoutPixelSupervision(fullres, model)
    total, pieces = ARCH.losses(model, masks, scores, targets, probe, silent)
    require(silent.silenced == 1 and float(pieces.get('dense', -1.)) == 0.,
            'the per-pixel supervision was not removed exactly once')
    return total, pieces


def configuration(arm):
    return dict(ARCH.configuration(arm), tuning880={
        'mode': MODE, 'knobs': KNOBS, 'base_lr': BASE_LR, 'base_weight_decay': BASE_WD,
        'effective': {'peak_lr_heads': BASE_LR*KNOBS.get('lr_scale', 1.),
                      'peak_lr_backbone': BASE_LR*KNOBS.get('lr_scale', 1.)*KNOBS.get('backbone_lr', 1.),
                      'weight_decay': KNOBS.get('weight_decay', BASE_WD),
                      'frozen': 'encoder batch norms (eval, rate 0), stem and layer1 at rate 0' if KNOBS.get('frozen_bn') else None,
                      'decoder_dropout': KNOBS.get('dropout', 0.),
                      'ablation': KNOBS.get('ablation')},
        'ssl_checkpoint_sha256': SSL_SHA or None,
        'where': "multipliers act inside AdamW.step around 675's schedule; 675's recipe string states the base"})

"""The same model on the same backbone as the baselines: MiT-B0 out, ResNet-50 in.

76 left one thing no change to the score, the matcher or the renderer can
reach.  Of the 87 misshapen targets 744 dissected over three seeds, the
decoder had a coarse mask that cleared IoU 0.50 on any query for only 12; for
the rest the coarse masks themselves are short of the target, at a median of
0.40, and the renderer, which lifts that to 0.43, is not what holds them back.
That is a representation limit, and it is also the one place a reviewer will
point first: the two baselines run on ResNet-50 and this model on a 3.3M-
parameter MiT-B0, so any comparison is between methods and backbones at once.

So this swaps exactly the backbone.  SMP's Segformer decoder takes a
ResNet-50 encoder and produces the same four maps at the same strides -- 24x192,
12x96, 6x48, 3x24 -- and the same 256-channel quarter-resolution output, so
654's see stage, 674's finder and laterals, the renderer, the head, every loss
and every weight of the rest of 730 are unchanged; only the three laterals
widen to the ResNet's 512, 1024 and 2048 channels.  The encoder is initialised
from R-50-torchvision.pkl, the very file MaskDINO and PointRend load in
516 -- sha256 16fd03f0..., recorded in their manifests -- converted from
detectron2's names to torchvision's and averaged over the three input channels
for the grayscale strip, as 422 does for every ResNet in this project.

675 takes the pretrained file through ``--segformer-b0-weights`` and pins it to
655's digest.  The argument name is historical; this passes the ResNet file
through it and replaces 655's pin with the ResNet's digest, having asserted
that 655 still pins MiT-B0's, so the run's profile records the weights it
actually loaded and cannot be mistaken for a MiT-B0 run.  Two things differ
from 516's recipe and are kept as this line's protocol rather than borrowed:
the input is scaled to [-1, 1] rather than ImageNet-normalised, which batch
normalisation absorbs in fine-tuning, and every parameter shares one learning
rate rather than the backbone taking a tenth of it.

Bars, registered before the run, against revision_ablation726_empty_only_s42
(730's own seed-42 run) and md_s42:
  gate        675's train-only learning gate must pass.
  mechanism   pixel IoU at its own best point above the control's 0.5408, and
              711's misshapen count below the control's 35: a better backbone
              is supposed to move the coarse masks.
  result      component F1 above 0.6065, the control's; reported beside
              MaskDINO's 0.5777 at the same seed and on the same backbone.

This script deliberately does NOT: change the see stage's structure, the
finder, the renderer, the head, any loss, the matcher, the schedule, the
budget, the seed or the gate; use a pretrained file other than the baselines'
own; re-select an operating point; read, transcribe or otherwise consume the
frozen test split; or contact the network.
"""
import argparse
import importlib.util
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE_FILE = '730_revision_flat_model.py'
MIT_B0_SHA = 'df468f7f13c4186f25bd3e2caf09e4f927b5b5ac0abccac84011dae747d4c49c'
RESNET50_SHA = '16fd03f0b0123e85b959cb9b64aaf9eeb4a7dc68bd9b0b134a9220d4b24c15c6'
RESNET50_KEYS = 265
F1_BAR = .6065
PIXEL_IOU_BAR = .5408
MISSHAPEN_BAR = 35
MASKDINO_S42 = .5777
_STAGE = re.compile(r'^res(\d)\.(\d+)\.(conv(\d)|shortcut)(\.norm)?\.(.+)$')


def load(name):
    spec = importlib.util.spec_from_file_location('resnet750_' + name[:3],
                                                  ROOT / 'scripts' / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load(BASE_FILE)
ARMS, CORE, RULE = BASE.ARMS, BASE.CORE, BASE.RULE
losses = BASE.losses


def torchvision_names(state):
    """detectron2's converted ResNet names -> torchvision's; the fc is dropped."""
    out = {}
    for key, value in state.items():
        if key.startswith('stem.fc.'):
            continue
        if key == 'stem.conv1.weight':
            name = 'conv1.weight'
        elif key.startswith('stem.conv1.norm.'):
            name = 'bn1.' + key[len('stem.conv1.norm.'):]
        else:
            match = _STAGE.match(key)
            if match is None:
                raise ValueError('unrecognised ResNet key: ' + key)
            stage, block, part, conv, norm, field = match.groups()
            layer = f'layer{int(stage) - 1}.{block}'
            if part == 'shortcut':
                name = f'{layer}.downsample.{1 if norm else 0}.{field}'
            else:
                name = f'{layer}.{"bn" if norm else "conv"}{conv}.{field}'
        out[name] = torch.as_tensor(np.asarray(value), dtype=torch.float32)
    return out


def load_resnet50(encoder, path):
    """Load the baselines' R-50 into an SMP ResNet encoder that reads one channel."""
    with open(path, 'rb') as handle:
        payload = pickle.load(handle, encoding='latin1')
    state = torchvision_names(payload['model'] if 'model' in payload else payload)
    if len(state) != RESNET50_KEYS:
        raise ValueError(f'{len(state)} ResNet tensors, expected {RESNET50_KEYS}')
    if encoder.conv1.weight.shape[1] == 1 and state['conv1.weight'].shape[1] == 3:
        state['conv1.weight'] = state['conv1.weight'].mean(1, keepdim=True)
    missing, unexpected = torch.nn.Module.load_state_dict(encoder, state, strict=False)
    if unexpected or any(not key.endswith('num_batches_tracked') for key in missing):
        raise ValueError(f'ResNet state mismatch: missing={missing}, unexpected={unexpected}')
    return len(state)


def build_model(baseline, args):
    """654's three parts on a ResNet-50 Segformer, then 730's model on top."""
    import segmentation_models_pytorch as smp

    if args.arm not in ARMS:
        raise ValueError('unknown arm')
    segmenter = smp.Segformer(encoder_name='resnet50', encoder_weights=None,
                              in_channels=1, classes=1)
    load_resnet50(segmenter.encoder, args.segformer_b0_weights)
    base = BASE.BASE.ThreePartModel(segmenter.encoder, segmenter.decoder,
                                    int(segmenter.segmentation_head[0].in_channels),
                                    'd_competition')
    return BASE.FlatModel(base)


def repoint(train, mit_b0_sha):
    """Make 675 pin the ResNet file it is handed, having checked what it pinned."""
    original = train.load

    def loader(name):
        module = original(name)
        if name.startswith('655'):
            if module.PRETRAINED_SHA != mit_b0_sha:
                raise ValueError('655 no longer pins the MiT-B0 weights this replaces')
            module.PRETRAINED_SHA = RESNET50_SHA
        return module

    train.load = loader
    return train


def configuration(arm):
    value = dict(BASE.configuration(arm))
    value.update(experiment='750 the same model on the baselines\' ResNet-50',
                 encoder='ResNet-50, R-50-torchvision.pkl (sha256 ' + RESNET50_SHA[:12] +
                         '...), the file 516 loads for MaskDINO and PointRend',
                 decoder='SMP Segformer, 256 channels at quarter resolution, as for MiT-B0',
                 laterals=[512, 1024, 2048],
                 input_scaling='[-1, 1], this line\'s protocol, not ImageNet normalisation',
                 backbone_learning_rate='shared, this line\'s protocol, not 516\'s 0.1 multiplier',
                 control='revision_ablation726_empty_only_s42 (730 on MiT-B0)')
    value['bars'] = {'learning_gate': 'must pass', 'pixel_iou': PIXEL_IOU_BAR,
                     'misshapen_below': MISSHAPEN_BAR, 'component_f1': F1_BAR,
                     'maskdino_same_seed_same_backbone': MASKDINO_S42}
    return value


def main(argv=None):
    ahead = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ahead.add_argument('--score-rule', default='product_rescaled')
    picked, rest = ahead.parse_known_args(sys.argv[1:] if argv is None else list(argv))
    os.environ['HME_SCORE_RULE'] = picked.score_rule
    train = repoint(load('675_revision_region_point_train.py'), MIT_B0_SHA)
    architecture = load(Path(__file__).name)
    args = train.parse(rest, arms=architecture.ARMS)
    if 'find.mask_iou' not in args.clip_groups:
        raise ValueError('the overlap head is clipped as its own group')
    train.run(args, architecture_file=Path(__file__).name,
              extra_dependencies=(BASE_FILE, Path(__file__).name))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

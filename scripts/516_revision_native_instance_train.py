#!/usr/bin/env python3
"""Native R50 instance baselines under a declared common revision-task budget.

This is a task adaptation using upstream architectures/losses, not a claim to
reproduce each paper's COCO score or tuned COCO training recipe. No test input.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('mask2former', 'maskdino', 'pointrend', 'condinst', 'maskrcnn', 'solov2', 'dimaskdino')
# 422's declared seeds.  A baseline has to be read across the same seeds as
# the model it is compared with, or the margin is one run against one run.
# The added three are for variance estimation, never for choosing among, and
# are declared here rather than in 422, whose digest 514 pins.
SEEDS = (42, 2354, 3407)
VARIANCE_SEEDS = (8, 88, 888)
ALL_SEEDS = SEEDS + VARIANCE_SEEDS


def load_module(filename):
    path = ROOT / 'scripts' / filename
    spec = importlib.util.spec_from_file_location('native_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def learning_rate(step, total, base=3e-4, warmup=500):
    if step <= warmup:
        return base * step / warmup
    progress = min(1, (step - warmup) / max(1, total - warmup))
    return base * (.05 + .95 * .5 * (1 + math.cos(math.pi * progress)))


class Cycle:
    def __init__(self, rows, rng):
        self.rows, self.rng, self.order = rows, rng, []

    def take(self):
        if not self.order:
            self.order = list(range(len(self.rows)))
            self.rng.shuffle(self.order)
        return self.rows[self.order.pop()]


def make_input(record, dataset, augment=None, rng=None):
    import numpy as np
    import torch
    from detectron2.structures import BitMasks, Instances

    image = np.asarray(record['image']).copy()
    mask = np.asarray(record['mask']).copy()
    if augment is not None:
        gray, label, _, _ = augment.paired_affine_with_aux(
            image[0], mask.astype(np.uint8) * 255, np.zeros(mask.shape, np.uint8), None, rng)
        image, mask = np.repeat(gray[None], 3, axis=0), label > 0
    masks = dataset.instances_from_mask(mask)
    instance = Instances(mask.shape)
    instance.gt_masks = BitMasks(torch.from_numpy(masks.copy()))
    instance.gt_boxes = instance.gt_masks.get_bounding_boxes()
    instance.gt_classes = torch.zeros(len(masks), dtype=torch.int64)
    return {'image': torch.from_numpy(image.astype(np.float32)),
            'height': mask.shape[0], 'width': mask.shape[1], 'instances': instance}


def predict_record(model, record, device):
    import numpy as np
    import torch
    item = {'image': torch.from_numpy(np.asarray(record['image'], dtype=np.float32).copy()).to(device),
            'height': record['mask'].shape[0], 'width': record['mask'].shape[1]}
    result = model([item])[0]['instances'].to('cpu')
    return {'sample_id': record['sample_id'], 'masks': result.pred_masks.numpy().astype(bool),
            'scores': result.scores.float().numpy()}


def evaluate(model, rows, metrics, device, archive=None):
    import numpy as np
    import torch
    model.eval()
    prepared = []
    if archive is not None:
        archive.mkdir()
    with torch.no_grad():
        for index, row in enumerate(rows):
            prediction = predict_record(model, row, device)
            prepared.append(metrics.prepare_record(row, prediction))
            if archive is not None:
                masks = prediction['masks']
                np.savez_compressed(archive / f'{index:04d}.npz',
                                    sample_id=np.asarray(row['sample_id']), scores=prediction['scores'],
                                    shape=np.asarray(masks.shape, dtype=np.int64),
                                    masks_packed=np.packbits(masks.reshape(len(masks), -1), axis=1)
                                    if len(masks) else np.zeros((0, math.ceil(row['mask'].size / 8)), np.uint8))
    reading = metrics.select_operating_point(prepared)
    model.train()
    return reading


def probe_native_head(model, method, dataset):
    """Smoke-only rectangle checks the actual head, even for very thin data GT.

    No optimizer update or reported dev metric uses these synthetic fixtures.
    This catches a registered-but-disconnected point/dynamic/mask head.
    """
    import numpy as np
    import torch
    positive = np.zeros((96, 768), bool)
    positive[24:64, 256:336] = True
    records = []
    for mask in (positive, np.zeros_like(positive)):
        gray = np.where(mask, 0, 255).astype(np.uint8)
        records.append({'image': np.repeat(gray[None], 3, axis=0), 'mask': mask})
    model.train()
    model.zero_grad(set_to_none=True)
    losses = model([make_input(record, dataset) for record in records])
    required = {'maskrcnn': {'loss_cls', 'loss_box_reg', 'loss_mask', 'loss_rpn_cls', 'loss_rpn_loc'},
                'solov2': {'loss_ins', 'loss_cate'},
                'dimaskdino': {'loss_ce', 'loss_mask', 'loss_dice', 'loss_bbox', 'loss_giou'},
                'mask2former': {'loss_ce', 'loss_mask', 'loss_dice'},
                'maskdino': {'loss_ce', 'loss_mask', 'loss_dice', 'loss_bbox', 'loss_giou'},
                'pointrend': {'loss_mask', 'loss_mask_point'},
                'condinst': {'loss_mask', 'loss_fcos_cls', 'loss_fcos_loc', 'loss_fcos_ctr'}}[method]
    if not required <= losses.keys() or not torch.isfinite(sum(losses.values())):
        raise RuntimeError(f'native {method} losses missing or nonfinite: {list(losses)}')
    sum(losses.values()).backward()
    tokens = {'mask2former': ('mask_embed',), 'maskdino': ('mask_embed',),
              'pointrend': ('point_head',), 'condinst': ('controller',),
              'maskrcnn': ('backbone', 'proposal_generator', 'box_predictor', 'mask_head'),
              'solov2': ('backbone', 'ins_head.cate_pred', 'ins_head.kernel_pred', 'mask_head'),
              'dimaskdino': ('backbone', 'pixel_decoder', 'mask_embed', 'class_embed',
                            'bbox_embed', 'oldm', 'query_guide_feature_update')}[method]
    gradients = {name: float(value.grad.norm()) for name, value in model.named_parameters()
                 if any(token in name for token in tokens) and value.grad is not None}
    for token in tokens:
        values = [value for name, value in gradients.items() if token in name]
        if not values or not all(math.isfinite(value) for value in values) or not any(value > 0 for value in values):
            raise RuntimeError(f'native {method} component {token} has no finite nonzero gradient')
    result = {'smoke_only': True, 'fixture': 'one synthetic rectangle and one clean image',
              'optimizer_updates': 0, 'losses': {k: float(v.detach()) for k, v in losses.items()},
              'native_head_gradients': gradients}
    model.zero_grad(set_to_none=True)
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument('--method', choices=METHODS, required=True)
    for name in ('dataset-dir', 'upstream-root', 'weights-dir', 'output-dir'):
        result.add_argument('--' + name, type=Path, required=True)
    result.add_argument('--seed', type=int, default=42, choices=ALL_SEEDS)
    result.add_argument('--microbatch', type=int, default=2)
    result.add_argument('--smoke-steps', type=int, default=0)
    return result


def training_event_storage():
    from detectron2.utils.events import EventStorage
    return EventStorage(0)


def run(args):
    if args.seed not in ALL_SEEDS or args.microbatch not in (1, 2, 4, 8) or args.smoke_steps not in (0, 4):
        raise ValueError('this declared protocol requires one of 422\'s seeds, divisible microbatch, smoke=0/4')
    if args.smoke_steps and args.microbatch > 4:
        raise ValueError('smoke microbatch<=4 is required to exercise all-clean and all-positive batches')
    dataset = load_module('514_revision_native_dataset.py')
    if args.output_dir.expanduser().is_symlink():
        raise ValueError('output symlink refused')
    output = dataset.safe_path(args.output_dir)
    if output.exists():
        raise FileExistsError(f'create-only run output already exists: {output}')
    for path in (args.dataset_dir, args.upstream_root, args.weights_dir):
        dataset.safe_path(path)
    bundle = dataset.load_bundle(args.dataset_dir)
    factory = load_module('515_revision_native_model_factory.py')
    metrics = load_module('520_revision_native_metrics.py')
    augmentation = load_module('357_revision_localizer_extensions.py')
    kind = factory.get_pretraining_kind(args.method)
    weight = args.weights_dir / f'R-50-{"MSRA" if kind == "msra" else kind}.pkl'
    if not weight.is_file():
        raise FileNotFoundError(weight)
    output.mkdir(parents=True)
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('native baseline requires CUDA')
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Keep all native custom CUDA loss/matching paths in float32 in this first run.
    device = 'cuda'
    started = time.time()
    model = factory.build_model(args.method, args.upstream_root, weight, device=device,
                                seed=args.seed, smoke=bool(args.smoke_steps))
    native_description = factory.describe_model(args.method)
    config_yaml = getattr(model, 'native_config_yaml', None)
    if config_yaml:
        (output / 'native_config.yaml').write_text(config_yaml)
    groups = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        multiplier = .1 if 'backbone' in name and args.method in ('mask2former', 'maskdino', 'dimaskdino') else 1.
        no_decay = parameter.ndim == 1 or 'relative_position_bias_table' in name or 'absolute_pos_embed' in name
        groups.append({'params': [parameter], 'lr': 3e-4 * multiplier,
                       'lr_multiplier': multiplier, 'weight_decay': 0. if no_decay else 1e-4})
    optimiser = torch.optim.AdamW(groups, lr=3e-4, weight_decay=1e-4)
    total = args.smoke_steps or 28300
    warmup = 1 if args.smoke_steps else 500
    select_start = 2 if args.smoke_steps else 14150
    files = [Path(__file__), Path(dataset.__file__), Path(factory.__file__),
             Path(metrics.__file__), Path(augmentation.__file__)]
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    manifest = {
        'schema_version': 'hme-native-instance-baseline-v1', 'method': args.method,
        'smoke_only': bool(args.smoke_steps), 'formal_comparison_eligible': not bool(args.smoke_steps),
        'git_revision': commit, 'source_sha256': {str(p): digest(p) for p in files},
        'dataset_manifest_sha256': digest(args.dataset_dir / 'run_manifest.json'),
        'pretraining': {'kind': kind, 'path': str(weight), 'sha256': digest(weight)},
        'native_description': native_description,
        'native_metadata': getattr(model, 'baseline_metadata', {}),
        'upstream_compatibility': json.loads((args.upstream_root / 'compatibility_manifest.json').read_text()),
        'protocol': {'seed': args.seed, 'steps': total, 'effective_batch_size': 8,
                     'microbatch': args.microbatch, 'negative_ratio': 1,
                     'accumulation_note': '8 images/optimizer step; native per-microbatch target normalization is retained, not numerically equivalent to one batch of 8',
                     'optimizer': 'AdamW', 'lr': 3e-4, 'weight_decay': 1e-4,
                     'transformer_backbone_lr_multiplier': .1, 'normalization_bias_weight_decay': 0,
                     'warmup_steps': warmup, 'schedule': 'cosine, terminal lr=0.05*base',
                     'gradient_clip_norm': 5., 'amp': False, 'canvas': [96, 768],
                     'loss': 'native upstream instance losses including configured auxiliaries',
                     'augmentation': 'unchanged 357 paired affine+photometry; 8CC targets after warp',
                     'evaluation': '520; all 505 dev; mask=.5; shared score grid; clean FP<=.15',
                     'selection_start_step': select_start, 'validation_interval_early': 2000,
                     'validation_interval_late': 500, 'fixed_test_accessed': False},
        'interpretation': 'Native architectures under a common task budget; not per-method optimized COCO recipes or SOTA evidence.',
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
                        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                        'parameters': sum(p.numel() for p in model.parameters())},
    }
    write_json(output / 'run_manifest.json', manifest)
    with training_event_storage() as storage:
        if args.smoke_steps:
            write_json(output / 'native_head_smoke.json', probe_native_head(model, args.method, dataset))
        positives = [r for r in bundle['train'] if r['is_positive']]
        clean = [r for r in bundle['train'] if not r['is_positive']]
        rng = random.Random(args.seed)
        pos_cycle, neg_cycle = Cycle(positives, rng), Cycle(clean, rng)
        dev = bundle['dev']
        if args.smoke_steps:
            dev = [r for r in dev if r['is_positive']][:2] + [r for r in dev if not r['is_positive']][:2]
        curve, best_key, best_step = [], None, -1
        print(json.dumps({'event': 'model_ready', 'method': args.method, 'manifest': str(output / 'run_manifest.json')}), flush=True)
        for step in range(1, total + 1):
            model.train()
            lr = learning_rate(step, total, warmup=warmup)
            for group in optimiser.param_groups:
                group['lr'] = lr * group['lr_multiplier']
            rows = [pos_cycle.take() for _ in range(4)] + [neg_cycle.take() for _ in range(4)]
            if not args.smoke_steps:
                rng.shuffle(rows)
            optimiser.zero_grad(set_to_none=True)
            loss_report = {}
            for start in range(0, 8, args.microbatch):
                inputs = [make_input(row, dataset, augmentation, rng) for row in rows[start:start + args.microbatch]]
                losses = model(inputs)
                loss = sum(losses.values())
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'nonfinite native loss at step {step}: {losses}')
                (loss * args.microbatch / 8).backward()
                for key, value in losses.items():
                    loss_report[key] = loss_report.get(key, 0.) + float(value.detach()) * args.microbatch / 8
            if step == 1:
                gradients = {name: float(parameter.grad.detach().norm()) for name, parameter in model.named_parameters()
                             if parameter.grad is not None and parameter.requires_grad}
                if not gradients or not any(value > 0 for value in gradients.values()):
                    raise RuntimeError('no native model gradients')
                write_json(output / 'first_step_gradients.json', gradients)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f'nonfinite gradients at step {step}')
            optimiser.step()
            storage.step()
            if step == 1 or step % 100 == 0 or args.smoke_steps:
                print(json.dumps({'event': 'train', 'method': args.method, 'step': step,
                                  'loss': sum(loss_report.values()), 'elapsed_seconds': round(time.time() - started, 2),
                                  'max_allocated_mib': round(torch.cuda.max_memory_allocated() / 2**20)}), flush=True)
            interval = 2 if args.smoke_steps else (2000 if step < select_start else 500)
            if step % interval and step != total:
                continue
            reading = evaluate(model, dev, metrics, device)
            if not args.smoke_steps and (reading['rows'] != 505 or reading['gold_components'] != 221):
                raise RuntimeError('validation coverage changed')
            eligible = step >= select_start
            item = {'step': step, 'checkpoint_eligible': eligible, 'loss': loss_report,
                    'learning_rate': lr, 'elapsed_seconds': time.time() - started, **reading}
            curve.append(item)
            write_json(output / 'validation_curve.json', curve)
            key = (reading['component_f1'], reading['pixel_f1'])
            if eligible and (best_key is None or key > best_key):
                best_key, best_step = key, step
                torch.save({'model': model.state_dict(), 'step': step, 'threshold': reading['threshold']}, output / 'best_model.pth.partial')
                (output / 'best_model.pth.partial').replace(output / 'best_model.pth')
            torch.save({'model': model.state_dict(), 'optimizer': optimiser.state_dict(), 'step': step}, output / 'latest_checkpoint.pth.partial')
            (output / 'latest_checkpoint.pth.partial').replace(output / 'latest_checkpoint.pth')
            print(json.dumps({'event': 'validation', 'method': args.method, 'step': step,
                              'component_f1': reading['component_f1'], 'pixel_f1': reading['pixel_f1'],
                              'clean_fp': reading['clean_fp'], 'threshold': reading['threshold'], 'best_step': best_step}), flush=True)
        best = torch.load(output / 'best_model.pth', map_location='cpu', weights_only=True)
        model.load_state_dict(best['model'])
        final = evaluate(model, dev, metrics, device, archive=output / 'best_dev_instances')
        eligible_curve = [item for item in curve if item['checkpoint_eligible']]
        summary = {'schema_version': 'hme-native-instance-baseline-v1', 'status': 'complete',
                   'smoke_only': bool(args.smoke_steps), 'optimizer_steps': total,
                   'method': args.method, 'seed': args.seed, 'steps': total, 'best_step': best_step,
                   'best_dev': final, 'late_mean_component_f1': float(np.mean([x['component_f1'] for x in eligible_curve])),
                   'max_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                   'elapsed_seconds': time.time() - started, 'formal_comparison_eligible': not bool(args.smoke_steps),
                   'run_manifest_sha256': digest(output / 'run_manifest.json')}
        write_json(output / 'summary.json', summary)
        write_json(output / ('smoke_complete.json' if args.smoke_steps else 'train_complete.json'), summary)
        print(json.dumps({'event': 'complete', 'method': args.method, 'best_step': best_step}), flush=True)


if __name__ == '__main__':
    run(parser().parse_args())

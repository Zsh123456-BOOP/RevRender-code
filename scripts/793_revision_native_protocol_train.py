"""Train a native baseline under the protocol: 788 training gold, 784 validation gold, early stopping.

516 trained every native baseline on the registered bundle, chose its
checkpoint against the old validation answers and ran all 28,300 steps.
涂改定位流程.md now fixes one procedure for every model: the training split
the user reviewed in full (788), the validation answers the user redrew (784),
and early stopping -- from step 14,150, stop once 6 validations in a row (3,000
steps) have not beaten the best one, and keep the best.  This is 516's run with
exactly those three changes and nothing else: the same upstream architecture
and native losses (515), the same budget ceiling, schedule, optimiser, batch
composition, augmentation (357), validation grid and selection key
(component F1, then pixel F1, from 520 at native mask 0.5).  The learning-rate
schedule still runs to 28,300 steps; stopping early only cuts it short.

With ``--split B`` the same run reads split B (798) -- every annotated line
re-split 8:1:1 by student, stratified by revision type -- through 798's
contract instead of 792's; nothing else changes.

This script deliberately does NOT: change a baseline's architecture, loss,
optimiser, schedule or augmentation; read the test split or the test gold;
read any output of this project's own models; choose anything on a split other
than validation; or contact the network.
"""
import argparse
import importlib.util
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('maskdino', 'pointrend', 'mask2former', 'condinst', 'maskrcnn', 'solov2', 'dimaskdino')
SEED = 42
PATIENCE = 6
SELECT_START, TOTAL = 14150, 28300


def load(name, key):
    spec = importlib.util.spec_from_file_location(key, ROOT/'scripts'/name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


class EarlyStop:
    """Stop once PATIENCE eligible validations in a row have not beaten the best."""

    def __init__(self, patience=PATIENCE):
        self.patience, self.best, self.best_step, self.stale = patience, None, None, 0

    def update(self, step, key):
        if self.best is None or key > self.best:
            self.best, self.best_step, self.stale = key, step, 0
            return True
        self.stale += 1
        return False

    @property
    def stop(self):
        return self.stale >= self.patience


def parser():
    result = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    result.add_argument('--method', choices=METHODS, required=True)
    for name in ('dataset-dir', 'upstream-root', 'weights-dir', 'output-dir'):
        result.add_argument('--'+name, type=Path, required=True)
    result.add_argument('--seed', type=int, default=SEED, choices=(SEED,))
    result.add_argument('--microbatch', type=int, default=2)
    result.add_argument('--smoke-steps', type=int, default=0)
    result.add_argument('--split', choices=('original', 'B'), default='original',
                        help="original: 792's bundle (788 training, 784 validation); B: 798's split B")
    return result


GOLDS = {'original': ('788', '784'), 'B': ('split_b_798', 'split_b_798')}


def run(args):
    base = load('516_revision_native_instance_train.py', 'protocol793_516')
    protocol = load('792_revision_protocol_bundle.py' if args.split == 'original' else '798_revision_split_b.py',
                    'protocol793_'+args.split)
    if args.microbatch not in (1, 2, 4, 8) or args.smoke_steps not in (0, 4):
        raise ValueError("516's protocol requires a divisible microbatch and smoke=0/4")
    dataset = protocol.dataset_for(base.load_module('514_revision_native_dataset.py'), args.dataset_dir)
    if args.output_dir.expanduser().is_symlink():
        raise ValueError('output symlink refused')
    output = dataset.safe_path(args.output_dir)
    if output.exists():
        raise FileExistsError(f'create-only run output already exists: {output}')
    for path in (args.dataset_dir, args.upstream_root, args.weights_dir):
        dataset.safe_path(path)
    bundle = dataset.load_bundle(args.dataset_dir)
    factory = base.load_module('515_revision_native_model_factory.py')
    metrics = base.load_module('520_revision_native_metrics.py')
    augmentation = base.load_module('357_revision_localizer_extensions.py')
    kind = factory.get_pretraining_kind(args.method)
    weight = args.weights_dir/f'R-50-{"MSRA" if kind == "msra" else kind}.pkl'
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
    device = 'cuda'
    started = time.time()
    model = factory.build_model(args.method, args.upstream_root, weight, device=device,
                                seed=args.seed, smoke=bool(args.smoke_steps))
    config_yaml = getattr(model, 'native_config_yaml', None)
    if config_yaml:
        (output/'native_config.yaml').write_text(config_yaml)
    groups = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        multiplier = .1 if 'backbone' in name and args.method in ('mask2former', 'maskdino', 'dimaskdino') else 1.
        no_decay = parameter.ndim == 1 or 'relative_position_bias_table' in name or 'absolute_pos_embed' in name
        groups.append({'params': [parameter], 'lr': 3e-4*multiplier,
                       'lr_multiplier': multiplier, 'weight_decay': 0. if no_decay else 1e-4})
    optimiser = torch.optim.AdamW(groups, lr=3e-4, weight_decay=1e-4)
    total = args.smoke_steps or TOTAL
    warmup = 1 if args.smoke_steps else 500
    select_start = 2 if args.smoke_steps else SELECT_START
    patience = 1 if args.smoke_steps else PATIENCE
    expected_components = sum(len(r['instances']) for r in bundle['dev'])
    files = [Path(__file__), Path(base.__file__), Path(protocol.__file__), Path(dataset.__file__),
             Path(factory.__file__), Path(metrics.__file__), Path(augmentation.__file__)]
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    manifest = {
        'schema_version': 'hme-native-instance-baseline-protocol-v1', 'method': args.method,
        'smoke_only': bool(args.smoke_steps), 'formal_comparison_eligible': not bool(args.smoke_steps),
        'git_revision': commit, 'source_sha256': {str(p): base.digest(p) for p in files},
        'dataset_manifest_sha256': base.digest(args.dataset_dir/'run_manifest.json'),
        'pretraining': {'kind': kind, 'path': str(weight), 'sha256': base.digest(weight)},
        'native_description': factory.describe_model(args.method),
        'native_metadata': getattr(model, 'baseline_metadata', {}),
        'upstream_compatibility': json.loads((args.upstream_root/'compatibility_manifest.json').read_text()),
        'protocol': {'document': '涂改定位流程.md', 'seed': args.seed, 'max_steps': total,
                     'early_stopping': {'from_step': select_start, 'patience_validations': patience,
                                        'interval_steps': 500},
                     'split': args.split, 'training_gold': GOLDS[args.split][0], 'validation_gold': GOLDS[args.split][1],
                     'validation_components': expected_components,
                     'effective_batch_size': 8, 'microbatch': args.microbatch, 'negative_ratio': 1,
                     'optimizer': 'AdamW', 'lr': 3e-4, 'weight_decay': 1e-4,
                     'transformer_backbone_lr_multiplier': .1, 'warmup_steps': warmup,
                     'schedule': f'cosine to {total} steps, terminal lr=0.05*base; early stopping only cuts it short',
                     'gradient_clip_norm': 5., 'amp': False, 'canvas': [96, 768],
                     'loss': 'native upstream instance losses including configured auxiliaries',
                     'augmentation': 'unchanged 357 paired affine+photometry; 8CC targets after warp',
                     'evaluation': f"520; all {len(bundle['dev'])} validation rows; mask=.5; shared score grid; clean FP<=.15",
                     'validation_interval_early': 2000, 'validation_interval_late': 500,
                     'test_accessed': False},
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
                        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                        'parameters': sum(p.numel() for p in model.parameters())},
    }
    base.write_json(output/'run_manifest.json', manifest)
    with base.training_event_storage() as storage:
        if args.smoke_steps:
            base.write_json(output/'native_head_smoke.json', base.probe_native_head(model, args.method, dataset))
        positives = [r for r in bundle['train'] if r['is_positive']]
        clean = [r for r in bundle['train'] if not r['is_positive']]
        rng = random.Random(args.seed)
        pos_cycle, neg_cycle = base.Cycle(positives, rng), base.Cycle(clean, rng)
        dev = bundle['dev']
        if args.smoke_steps:
            dev = [r for r in dev if r['is_positive']][:2]+[r for r in dev if not r['is_positive']][:2]
        curve, stopper, stopped_at = [], EarlyStop(patience), None
        print(json.dumps({'event': 'model_ready', 'method': args.method, 'manifest': str(output/'run_manifest.json')}), flush=True)
        for step in range(1, total+1):
            model.train()
            lr = base.learning_rate(step, total, warmup=warmup)
            for group in optimiser.param_groups:
                group['lr'] = lr*group['lr_multiplier']
            rows = [pos_cycle.take() for _ in range(4)]+[neg_cycle.take() for _ in range(4)]
            if not args.smoke_steps:
                rng.shuffle(rows)
            optimiser.zero_grad(set_to_none=True)
            loss_report = {}
            for start in range(0, 8, args.microbatch):
                inputs = [base.make_input(row, dataset, augmentation, rng) for row in rows[start:start+args.microbatch]]
                losses = model(inputs)
                loss = sum(losses.values())
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'nonfinite native loss at step {step}: {losses}')
                (loss*args.microbatch/8).backward()
                for key, value in losses.items():
                    loss_report[key] = loss_report.get(key, 0.)+float(value.detach())*args.microbatch/8
            if step == 1:
                gradients = {name: float(parameter.grad.detach().norm()) for name, parameter in model.named_parameters()
                             if parameter.grad is not None and parameter.requires_grad}
                if not gradients or not any(value > 0 for value in gradients.values()):
                    raise RuntimeError('no native model gradients')
                base.write_json(output/'first_step_gradients.json', gradients)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f'nonfinite gradients at step {step}')
            optimiser.step()
            storage.step()
            if step == 1 or step % 100 == 0 or args.smoke_steps:
                print(json.dumps({'event': 'train', 'method': args.method, 'step': step,
                                  'loss': sum(loss_report.values()), 'elapsed_seconds': round(time.time()-started, 2),
                                  'max_allocated_mib': round(torch.cuda.max_memory_allocated()/2**20)}), flush=True)
            interval = 2 if args.smoke_steps else (2000 if step < select_start else 500)
            if step % interval and step != total:
                continue
            reading = base.evaluate(model, dev, metrics, device)
            if not args.smoke_steps and (reading['rows'] != len(bundle['dev']) or reading['gold_components'] != expected_components):
                raise RuntimeError('validation coverage changed')
            eligible = step >= select_start
            item = {'step': step, 'checkpoint_eligible': eligible, 'loss': loss_report,
                    'learning_rate': lr, 'elapsed_seconds': time.time()-started, **reading}
            curve.append(item)
            base.write_json(output/'validation_curve.json', curve)
            if eligible and stopper.update(step, (reading['component_f1'], reading['pixel_f1'])):
                torch.save({'model': model.state_dict(), 'step': step, 'threshold': reading['threshold']},
                           output/'best_model.pth.partial')
                (output/'best_model.pth.partial').replace(output/'best_model.pth')
            print(json.dumps({'event': 'validation', 'method': args.method, 'step': step,
                              'component_f1': reading['component_f1'], 'pixel_f1': reading['pixel_f1'],
                              'clean_fp': reading['clean_fp'], 'threshold': reading['threshold'],
                              'best_step': stopper.best_step, 'stale': stopper.stale}), flush=True)
            if eligible and stopper.stop:
                stopped_at = step
                break
        best = torch.load(output/'best_model.pth', map_location='cpu', weights_only=True)
        model.load_state_dict(best['model'])
        final = base.evaluate(model, dev, metrics, device, archive=output/'best_dev_instances')
        eligible_curve = [item for item in curve if item['checkpoint_eligible']]
        summary = {'schema_version': 'hme-native-instance-baseline-protocol-v1', 'status': 'complete',
                   'smoke_only': bool(args.smoke_steps), 'method': args.method, 'seed': args.seed,
                   'max_steps': total, 'stopped_early_at': stopped_at, 'best_step': stopper.best_step,
                   'best_dev': final, 'late_mean_component_f1': float(np.mean([x['component_f1'] for x in eligible_curve])),
                   'max_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                   'elapsed_seconds': time.time()-started, 'formal_comparison_eligible': not bool(args.smoke_steps),
                   'run_manifest_sha256': base.digest(output/'run_manifest.json')}
        base.write_json(output/'summary.json', summary)
        base.write_json(output/('smoke_complete.json' if args.smoke_steps else 'train_complete.json'), summary)
        print(json.dumps({'event': 'complete', 'method': args.method, 'best_step': stopper.best_step,
                          'stopped_early_at': stopped_at}), flush=True)


if __name__ == '__main__':
    run(parser().parse_args())

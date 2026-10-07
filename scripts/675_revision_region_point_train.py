"""Deterministic joint transfer on a declared seed: replay smoke -> train-only learning gate -> full run.

The seed was written into eight places as the literal 42 while every arm was a
single run, which was honest then and is not now: 719 measured the alpha-alone
series at 0.5970, 0.6262, 0.5714, 0.5707 for alpha .70/.75/.78/.85, a 0.0548
swing between adjacent doses, against a 0.0392 margin over MaskDINO.  A margin
smaller than the spread between neighbouring configurations cannot be read off
one run, so the seed is an argument.

It is restricted to the triple 422 already declares, so a seed cannot be
searched over: choosing among seeds after seeing their scores would inflate
exactly the margin this is meant to measure.  Seed 42 is unchanged, byte for
byte, which the smoke trace is checked against.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('b_region', 'c_roi_point', 'd_region_roi_point')
# 422's triple is the original contract and is cross-checked in run().  The
# three added here are for variance estimation and are declared here rather
# than in 422, whose digest 514 pins.  They are not a set to choose from: a
# number is the mean over every declared seed that was run, never over a
# chosen subset of them.  Three seeds resolve only about 0.027 of component
# F1, which is wider than every margin this project has measured.
SEEDS = (42, 2354, 3407)
VARIANCE_SEEDS = (8, 88, 888)
ALL_SEEDS = SEEDS + VARIANCE_SEEDS


def load(name):
    spec = importlib.util.spec_from_file_location('train675_'+name[:3], ROOT/'scripts'/name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def clip_groups(model, prefixes, limit=5.):
    """Clip each declared group on its own; everything else together.

    With one global norm, adding a loss on a detached head still rescales the
    trunk's gradients through the clip coefficient, so two runs that differ
    only in a head loss do not share a trunk.  Splitting the groups makes the
    trunk's update depend on the trunk's gradients alone.  With no prefixes
    this is the original single call, unchanged.
    """
    import torch          # imported here as everywhere else in this file

    if not prefixes:
        torch.nn.utils.clip_grad_norm_(model.parameters(), limit,
                                       foreach=False, error_if_nonfinite=True)
        return
    named = list(model.named_parameters())
    claimed, groups = set(), []
    for prefix in prefixes:
        picked = [p for name, p in named
                  if name.startswith(prefix) and name not in claimed]
        claimed.update(name for name, _ in named if name.startswith(prefix))
        if not picked:
            raise ValueError('no parameter matches clip group ' + prefix)
        groups.append(picked)
    groups.append([p for name, p in named if name not in claimed])
    for group in groups:
        if group:
            torch.nn.utils.clip_grad_norm_(group, limit, foreach=False,
                                           error_if_nonfinite=True)


def parse(argv=None, arms=ARMS):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--arm', choices=arms, required=True)
    p.add_argument('--stage', choices=('smoke', 'learning', 'formal'), required=True)
    for name in ('bundle', 'segformer-b0-weights', 'output-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--gate-root', type=Path)
    # Off by default, so every completed run replays byte for byte.
    p.add_argument('--clip-groups', action='append', default=[],
                   help='parameter-name prefix whose gradients are clipped on '
                        'their own; repeatable. Global clipping couples an '
                        'added head loss to the trunk through the clip '
                        'coefficient even when the head is detached, so a head '
                        'ablation is not attributable without this.')
    p.add_argument('--seed', type=int, default=42, choices=ALL_SEEDS,
                   help='one of the three seeds 422 declares; not a knob to search')
    p.add_argument('--replay-of', type=Path)
    p.add_argument('--speed-v1', action='store_true', help='verified FP32 scheduling optimizations, 767/769 only')
    p.add_argument('--speed-no-empty-fill', action='store_true', help='audited initialized outputs; keep deterministic algorithms, omit diagnostic empty-memory filling')
    p.add_argument('--augment', choices=('current', 'off', 'strong'), default='current',
                   help='780 augmentation dose; current is 357 unchanged and loads nothing new')
    p.add_argument('--early-stop-patience', type=int, default=0,
                   help='涂改定位流程.md: in formal, stop once this many validations from step 14150 in a row '
                        'have not beaten the best; 0 (the default) runs the full budget, so completed runs still replay')
    a = p.parse_args(argv)
    if a.speed_no_empty_fill and not a.speed_v1:
        p.error('--speed-no-empty-fill requires --speed-v1')
    if a.stage == 'formal' and a.gate_root is None:
        p.error('formal requires completed learning and replay gates')
    if a.replay_of and a.stage != 'smoke':
        p.error('replay-of is smoke only')
    a.batch_size, a.negative_ratio = 8, 1.
    a.optimizer_steps, a.warmup_steps, a.validation_interval = 28300, 500, 500
    a.learning_rate, a.weight_decay = .0003, .0001
    a.run_steps = {'smoke': 4, 'learning': 512, 'formal': 28300}[a.stage]
    return a


def learning_rows(rows):
    single = [r for r in rows if len(r['instances']) == 1][:2]
    multi = [r for r in rows if len(r['instances']) >= 2][:2]
    clean = [r for r in rows if len(r['instances']) == 0][:4]
    if list(map(len, (single, multi, clean))) != [2, 2, 4]:
        raise ValueError('missing TRAIN learning strata')
    return single+multi+clean


def learning_gate(initial, final, tp, fp, gold, multi_tp, clean_fired):
    return (final < .7*initial and tp >= int(np.ceil(.7*gold)) and
            tp/max(1, tp+fp) >= .7 and multi_tp >= 2 and clean_fired <= 1)


def make_optimizer(model, args):
    """Use the declared recipe for the actual optimizer, not just its manifest."""
    import torch
    return torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                            weight_decay=args.weight_decay, foreach=False, fused=False)


def effective_learning_rate(step, args, common):
    return args.learning_rate if args.stage == 'learning' else common.learning_rate_at(step, args)


def optimizer_description(args):
    # Preserve the historical default string, including its leading-dot style.
    rate = format(args.learning_rate, 'g').removeprefix('0')
    decay = format(args.weight_decay, 'g').removeprefix('0')
    return f'AdamW lr={rate} wd={decay} foreach=False fused=False; clip5; full 28300 cosine'


def validate_evidence(path, profile, common):
    c = json.loads((path/'complete.json').read_text())
    m = json.loads((path/'run_manifest.json').read_text())
    if (path/'failure.json').exists() or m['profile'] != profile or m['frozen_test_touched']:
        raise ValueError('gate profile mismatch or failure')
    for name, sha in c['files'].items():
        if Path(name).name != name or common.digest(path/name) != sha:
            raise ValueError('gate evidence changed: '+name)
    return c, json.loads((path/'summary.json').read_text())


def run(args, architecture_file='674_revision_region_point_model.py', extra_dependencies=()):
    common, dataset = load('655_revision_three_part_train.py'), load('514_revision_native_dataset.py')
    common.validate_paths(args, dataset)
    for path in (args.gate_root, args.replay_of):
        if path is not None:
            dataset.safe_path(path)
    if common.digest(args.bundle/'run_manifest.json') != common.BUNDLE_SHA or common.digest(args.segformer_b0_weights) != common.PRETRAINED_SHA:
        raise ValueError('registered data/pretrained hash changed')
    dependencies = (*common.DEPENDENCIES, '668_revision_deterministic_ops.py',
                    '674_revision_region_point_model.py', architecture_file, Path(__file__).name, *extra_dependencies)
    if args.speed_v1:
        dependencies += ('772_revision_training_speed.py',)
    if args.augment != 'current':
        dependencies += ('780_revision_augmentation_dose.py',)
    hashes = {n: common.digest(ROOT/'scripts'/n) for n in dependencies}
    import torch
    ops = load('668_revision_deterministic_ops.py')
    ops.configure()
    torch.autograd.set_multithreading_enabled(False)
    architecture = load(architecture_file)
    ops.install(architecture.CORE)
    speed = load('772_revision_training_speed.py') if args.speed_v1 else None
    speed_profile = speed.install(architecture) if speed else None
    if args.speed_no_empty_fill:
        torch.utils.deterministic.fill_uninitialized_memory = False
        speed_profile['fill_uninitialized_memory'] = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    profile = {'source_sha256': hashes, 'seed': args.seed, 'arm': args.arm,
               'dataset_sha256': common.BUNDLE_SHA, 'pretrained_sha256': common.PRETRAINED_SHA,
               'configuration': architecture.configuration(args.arm),
               'torch': torch.__version__, 'cuda': torch.version.cuda, 'python': platform.python_version(),
               'gpu': torch.cuda.get_device_name(0), 'cudnn': torch.backends.cudnn.version(),
               'strict_determinism': True, 'autograd_multithreading': False,
               'optimizer': optimizer_description(args),
               'environment': {n: os.environ.get(n) for n in ('CUBLAS_WORKSPACE_CONFIG', 'PYTHONHASHSEED', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}}
    if speed:
        profile['speed_implementation'] = speed_profile
    if args.augment != 'current':
        profile['augmentation'] = {'dose': args.augment}
    if args.early_stop_patience:
        profile['early_stopping'] = {'patience_validations': args.early_stop_patience, 'from_step': 14150,
                                     'key': 'validation F1@0.50, then pixel IoU; stop only in formal'}
    expected = None
    if args.replay_of:
        c, s = validate_evidence(args.replay_of, profile, common)
        if c['stage'] != 'smoke' or c['steps'] != 4:
            raise ValueError('four-step smoke replay required')
        expected = json.loads((args.replay_of/'trace.json').read_text())
    if args.stage == 'formal':
        a, first = validate_evidence(args.gate_root/'smoke_a', profile, common)
        b, replay = validate_evidence(args.gate_root/'smoke_b', profile, common)
        c, learn = validate_evidence(args.gate_root/'learning', profile, common)
        if (a['stage'], b['stage'], c['stage']) != ('smoke', 'smoke', 'learning') or not replay['exact_replay'] or not learn['passed']:
            raise ValueError('replay and training-only learning gates required')
        expected = json.loads((args.gate_root/'smoke_a'/'trace.json').read_text())
    baseline = load('422_revision_paper_baselines.py')
    # 422 is pinned by 514's digest, so the widening lives here rather than
    # there; what is checked is that 422's own triple has not drifted.
    if tuple(baseline.SEEDS) != SEEDS:
        raise ValueError('declared seeds drifted from 422')
    base, extension = load('354_revision_localizer_optimized.py'), load('357_revision_localizer_extensions.py')
    probe, fullres = load('442_revision_instance_head_probe.py'), load('480_revision_fullres_instance_seed42.py')
    fast = load('483_revision_fast_train_loop.py')
    bundle = dataset.load_bundle(args.bundle)
    data = common.make_rows(bundle)
    augmentation_install = None
    if args.augment != 'current':
        # Training positives only; the pool never sees a dev row.  The profile
        # carries only the dose, so replay and gate checks compare like with
        # like; what the install built goes into the run metadata.
        augmentation_install = load('780_revision_augmentation_dose.py').install(
            extension, args.augment, data['fit_positive'])
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = architecture.build_model(baseline, args).cuda()
    if list(dict(model.named_children())) != ['see', 'find', 'draw'] or not all(p.requires_grad for p in model.parameters()):
        raise ValueError('three fully trainable parts required')
    initial = {name: common.state_digest(part) for name, part in model.named_children()}
    optimizer = make_optimizer(model, args)
    clipper = speed.Clipper(model, args.clip_groups) if speed else None
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    metadata = {'profile': profile, 'stage': args.stage, 'seed': args.seed, 'arm': args.arm,
                'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'git_revision': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                'command': [sys.executable, str(Path(__file__)), *sys.argv[1:]],
                'parameters': sum(p.numel() for p in model.parameters()), 'initial_part_sha256': initial,
                'full_joint_training': True, 'frozen_test_touched': False,
                'gpu_uuid': str(torch.cuda.get_device_properties(0).uuid), 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    metadata['encoder_class'] = type(model.see.encoder).__name__
    if augmentation_install is not None:
        metadata['augmentation_install'] = augmentation_install
    metadata['cpu_affinity'] = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
    metadata['numa_node'] = os.environ.get('HME_NUMA_NODE')
    origin = ROOT/'deployment_origin.json'
    if origin.is_file():
        metadata['deployment_origin'] = json.loads(origin.read_text())
    if args.stage == 'learning':
        selected = learning_rows(bundle['train'])
        dev = [{**r, 'image': r['image'][0]} for r in selected]
        fixed_images = torch.from_numpy(np.stack([r['image'] for r in dev])[:, None].astype(np.float32)/127.5-1).cuda()
        fixed_targets = [[torch.from_numpy(t.astype(np.float32)).cuda() for t in r['instances']] for r in selected]
        metadata['train_sample_ids'] = [r['sample_id'] for r in selected]
        metadata['learning_gate'] = '512 steps; loss < .7 initial; recall>=.7 precision>=.7; multi TP>=2; <=1 clean row fired; mask=.5 score=.45'
        stream = None
        positive_count = 4
    else:
        stream = common.make_stream(fast, base, extension, data, args, random.Random(args.seed))
        if speed:
            stream = speed.prepare_stream(stream, probe)
        dev = data['validation_positive']+data['validation_clean']
        positive_count = len(data['validation_positive'])
        if args.stage == 'smoke':
            dev = data['validation_positive'][:4]+data['validation_clean'][:4]
            positive_count = 4
    truths = [r['instances'] for r in dev]
    common.write_json(out/'run_manifest.json', metadata)
    curve, trace, losses = [], [], []
    best_key = best_step = best_reading = None
    final_reading = stopped_at = None
    stale = 0
    started = time.monotonic()
    if stream:
        stream.start()
    try:
        for step in range(1, args.run_steps+1):
            model.train()
            rate = effective_learning_rate(step, args, common)
            for group in optimizer.param_groups:
                group['lr'] = rate
            if stream:
                if speed:
                    images, targets = speed.device_batch(stream.next())
                else:
                    images, masks, _ = stream.next()
                    image_array = np.stack(images)[:, None].astype(np.float32)/127.5-1
                    images = torch.from_numpy(image_array).cuda()
                    targets = [[torch.from_numpy(t.astype(np.float32)).cuda() for t in probe.gold_instances(mask.astype(np.float32)/255)] for mask in masks]
            else:
                images, targets = fixed_images, fixed_targets
            prediction, scores, _ = model(images, targets)
            loss, pieces = architecture.losses(model, prediction, scores, targets, probe, fullres)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clipper:
                clipper()
            else:
                clip_groups(model, args.clip_groups)
            if step == 1:
                checks = {name: {'all_require_grad': all(p.requires_grad for p in part.parameters()),
                                'nonzero_gradient': any(p.grad is not None and bool((p.grad != 0).any()) for p in part.parameters())}
                          for name, part in model.named_children()}
                checks['backbone'] = {'nonzero_gradient': any(p.grad is not None and bool((p.grad != 0).any()) for p in model.see.encoder.parameters())}
                if not all(v['nonzero_gradient'] for v in checks.values()):
                    raise RuntimeError('missing joint learning gradient')
                common.write_json(out/'gradient_check.json', checks)
            optimizer.step()
            if not speed:
                losses.append(float(loss.detach()))
            elif args.stage == 'learning':
                losses.append(loss.detach())
            due = (fast.validation_is_due(step, 28300, 500, 14150) if args.stage == 'formal' else step == args.run_steps)
            if speed and (due or (step <= 4 and args.stage != 'learning')):
                pieces = speed.materialize(pieces)
            if step <= 4 and args.stage != 'learning':
                import hashlib
                entry = {'step': step, 'loss': float(loss.detach()), 'pieces': pieces,
                         'input_sha256': hashlib.sha256(images.detach().cpu().numpy().tobytes()).hexdigest(),
                         'state_sha256': common.state_digest(model)}
                trace.append(entry)
                if expected is not None and entry != expected[step-1]:
                    raise RuntimeError('independent run differs at step '+str(step))
            if step == 1 or step % 64 == 0 or step == args.run_steps:
                progress = {'arm': args.arm, 'stage': args.stage, 'step': step, 'steps': args.run_steps,
                            'loss': float(loss.detach()), 'elapsed_seconds': time.monotonic()-started,
                            'gpu_gib': torch.cuda.max_memory_allocated()/2**30}
                common.write_json(out/'progress.json', progress)
                print(json.dumps(progress), flush=True)
            due = (fast.validation_is_due(step, 28300, 500, 14150) if args.stage == 'formal' else step == args.run_steps)
            if not due:
                continue
            cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
            bits, scores_np, seconds = common.predict_packed(model, dev, args)
            reading = common.evaluate(bits, scores_np, truths, positive_count, packed=True)
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(cuda_rng)
            curve.append({'step': step, 'validation': reading, 'loss_terms': pieces})
            with (out/'curve.jsonl').open('a') as f:
                f.write(json.dumps(curve[-1], allow_nan=False)+'\n')
            key = reading['instance']['0.50']['f1'], reading['pixel_iou']
            if (args.stage != 'formal' or step >= 14150) and (best_key is None or key > best_key):
                best_key, best_step, best_reading = key, step, reading
                torch.save({'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 'step': step, 'profile': profile}, out/'best.pt')
            if args.stage == 'formal' and step >= 14150:
                stale = 0 if best_step == step else stale+1
            if stream:
                state = common.capture_state(model, optimizer, stream, step, metadata, curve, {'step': best_step, 'key': best_key, 'reading': best_reading})
                common.save_state(out/f'resume_{step:06d}.pt', state)
                del state
            if step == args.run_steps:
                final_reading = reading
                common.save_predictions(out/'final_predictions.npz', bits, scores_np, dev)
            print(json.dumps({'event': 'validation', 'step': step, 'best_step': best_step, **reading['instance']['0.50']}), flush=True)
            if args.early_stop_patience and args.stage == 'formal' and stale >= args.early_stop_patience:
                stopped_at = step
                print(json.dumps({'event': 'early_stop', 'step': step, 'best_step': best_step}), flush=True)
                break
        changed = common.training_parts(model, initial)
        if not all(v['changed'] for v in changed.values()):
            raise RuntimeError('one of the three parts did not update')
        checkpoint = torch.load(out/'best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        bits, scores_np, seconds = common.predict_packed(model, dev, args)
        reading = common.evaluate(bits, scores_np, truths, positive_count, packed=True)
        if reading != best_reading:
            raise RuntimeError('checkpoint reload changed metrics')
        common.save_predictions(out/'dev_predictions.npz', bits, scores_np, dev)
        summary = {'arm': args.arm, 'seed': args.seed, 'stage': args.stage, 'steps': args.run_steps,
                   'best_step': best_step, 'stopped_early_at': stopped_at, 'validation': reading, 'final_validation': final_reading,
                   'late_mean_f1': float(np.mean([r['validation']['instance']['0.50']['f1'] for r in curve if args.stage != 'formal' or r['step'] >= 14150])),
                   'parameters': metadata['parameters'], 'parts_after_training': changed,
                   'gpu_gib': torch.cuda.max_memory_allocated()/2**30, 'elapsed_seconds': time.monotonic()-started,
                   'checkpoint_reload_verified': True, 'exact_replay': bool(args.replay_of), 'frozen_test_touched': False}
        if args.stage == 'learning':
            if speed:
                losses = torch.stack(losses).cpu().tolist()
            reader = load('656_revision_three_part_readout.py')
            # The query count is the model's, not a literal: an arm that keeps
            # more of the encoder's proposals archives that many columns, and a
            # written-in 32 reshapes its predictions into the wrong array.  For
            # every arm that keeps 32 this is the same reshape it always was.
            queries = model.query_count
            binary = np.unpackbits(bits[common.MASK_THRESHOLDS.index(.5)], axis=-1, count=96*768).reshape(len(dev), queries, 96, 768).astype(bool)
            tp = fp = multi_tp = clean_fired = 0
            for row, gold in enumerate(truths):
                pred = binary[row].reshape(queries, -1)
                target = gold.reshape(len(gold), 96*768)
                inter = np.stack([pred[:, t].sum(1) for t in target], 1) if len(gold) else np.zeros((queries, 0))
                iou = inter/np.maximum(1, pred.sum(1)[:, None]+target.sum(1)[None]-inter)
                keep = np.flatnonzero((scores_np[row] >= .45) & pred.any(1))
                _, found = reader.match_pairs(iou, keep)
                tp += len(found)
                fp += len(keep)-len(found)
                multi_tp += len(found) if len(gold) >= 2 else 0
                clean_fired += int(len(keep) > 0) if not len(gold) else 0
            initial_loss, final_loss = float(np.mean(losses[:16])), float(np.mean(losses[-16:]))
            gold = sum(map(len, truths))
            passed = learning_gate(initial_loss, final_loss, tp, fp, gold, multi_tp, clean_fired)
            summary.update(initial_loss=initial_loss, final_loss=final_loss, tp=tp, fp=fp, fn=gold-tp,
                           multi_tp=multi_tp, clean_fired=clean_fired, passed=passed, training_only=True)
        common.write_json(out/'trace.json', trace)
        common.write_json(out/'summary.json', summary)
        if args.stage == 'learning' and not summary['passed']:
            raise RuntimeError('train-only learning gate failed; no formal training allowed')
        if hashes != {n: common.digest(ROOT/'scripts'/n) for n in dependencies}:
            raise RuntimeError('source changed during run')
        common.write_json(out/'complete.json', {'stage': args.stage, 'steps': args.run_steps,
                          'files': {p.name: common.digest(p) for p in out.iterdir() if p.is_file()}})
        print(json.dumps({'event': 'complete', 'arm': args.arm, 'stage': args.stage}), flush=True)
    except BaseException as exc:
        common.write_json(out/'failure.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        if stream:
            stream.close()


if __name__ == '__main__':
    run(parse())

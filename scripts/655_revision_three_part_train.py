"""Seed42, full-network ablation of the three-part model; immutable train/dev bundle."""
from __future__ import annotations
import argparse
import copy
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
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'hme-three-part-joint-training-v1'
ARMS = ('a_coarse','b_point','c_identity','d_competition')
MASK_THRESHOLDS = (.2,.3,.4,.5,.6,.7,.8)
SCORE_THRESHOLDS = (.45,.55,.65,.75,.85)
MATCH_IOUS = (.25,.5,.75)
DEPENDENCIES = ('354_revision_localizer_optimized.py','357_revision_localizer_extensions.py',
    '422_revision_paper_baselines.py','442_revision_instance_head_probe.py',
    '445_revision_joint_head_probe.py','480_revision_fullres_instance_seed42.py',
    '483_revision_fast_train_loop.py','514_revision_native_dataset.py',
    '654_revision_three_part_model.py','655_revision_three_part_train.py')
BUNDLE_SHA = 'd9cacdf88844768cf415e1b954ac4234f7908bb6c5896f6cbe1961da6f54f250'
PRETRAINED_SHA = 'df468f7f13c4186f25bd3e2caf09e4f927b5b5ac0abccac84011dae747d4c49c'

def load(name):
    path = ROOT / "scripts" / name
    module_name = "threepart655_" + name.split("_")[0]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

def metric_from_ious(iou_rows, area_rows, scores, threshold, match_iou, positive_count):
    tp = fp = fn = clean_fp = 0
    for row, (ious, areas, row_scores) in enumerate(zip(iou_rows, area_rows, scores)):
        keep = np.flatnonzero((row_scores >= threshold) & (areas > 0))
        pairs = [(float(ious[p, g]), int(p), int(g)) for p in keep
                 for g in range(ious.shape[1]) if ious[p, g] >= match_iou]
        pairs.sort(key=lambda value: (-value[0], value[1], value[2]))
        used_p, used_g = set(), set()
        for _, p, g in pairs:
            if p not in used_p and g not in used_g:
                used_p.add(p)
                used_g.add(g)
        tp += len(used_p)
        fp += len(keep) - len(used_p)
        fn += ious.shape[1] - len(used_g)
        if row >= positive_count:
            clean_fp += len(keep)
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    return {"precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "clean_false_positive": clean_fp, "positive_false_positive": fp - clean_fp,
            "fp_components_per_line": fp / len(scores)}

def evaluate(masks, scores, truths, positive_count, packed=False):
    """Same greedy one-to-one matching/grid as 480; reuse IoUs across score thresholds.

    packed=True takes 7,N,Q,ceil(H*W/8) uint8 exact threshold bitmasks.
    Pixel metrics are measured at the SAME selected operating point as instances.
    """
    best, grid = None, []
    pixels = int(np.prod(truths[0].shape[-2:]))
    for m_index, mask_threshold in enumerate(MASK_THRESHOLDS):
        iou_rows, area_rows = [], []
        for row, truth in enumerate(truths):
            binary = (np.unpackbits(masks[m_index, row], axis=-1, count=pixels).astype(bool)
                      if packed else (masks[row] >= mask_threshold).reshape(len(scores[row]), pixels))
            gold = truth.reshape(len(truth), pixels)
            areas = binary.sum(1)
            intersections = np.stack([binary[:, target].sum(1) for target in gold], 1) if len(gold) else np.zeros((len(binary), 0))
            denominator = areas[:, None] + gold.sum(1)[None] - intersections
            iou_rows.append(intersections / np.maximum(denominator, 1))
            area_rows.append(areas)
        for score_threshold in SCORE_THRESHOLDS:
            metrics = metric_from_ious(iou_rows, area_rows, scores, score_threshold, .5, positive_count)
            grid.append({"mask_threshold": mask_threshold, "score_threshold": score_threshold, **metrics})
            key = (metrics["f1"], metrics["recall"], metrics["precision"],
                   -metrics["false_positive"], mask_threshold, score_threshold)
            if best is None or key > best[0]:
                readings = {f"{level:.2f}": metric_from_ious(iou_rows, area_rows, scores, score_threshold,
                                                          level, positive_count) for level in MATCH_IOUS}
                best = key, m_index, mask_threshold, score_threshold, readings
    _, m_index, mask_threshold, score_threshold, readings = best
    intersection = predicted_area = gold_area = 0
    for row, truth in enumerate(truths):
        binary = (np.unpackbits(masks[m_index, row], axis=-1, count=pixels).astype(bool)
                  if packed else (masks[row] >= mask_threshold).reshape(len(scores[row]), pixels))
        pred = binary[scores[row] >= score_threshold].any(0)
        gold = truth.reshape(len(truth), pixels).any(0)
        intersection += int((pred & gold).sum())
        predicted_area += int(pred.sum())
        gold_area += int(gold.sum())
    return {"instance": readings, "mask_threshold": mask_threshold, "score_threshold": score_threshold,
            "pixel_iou": intersection / max(1, predicted_area + gold_area - intersection),
            "pixel_precision": intersection / max(1, predicted_area),
            "pixel_recall": intersection / max(1, gold_area),
            "pixel_metric_scope": "same instance operating point, union of retained independent masks",
            "grid": grid, "gold_instances": sum(len(row) for row in truths)}

def make_rows(bundle):
    rows = {name: [] for name in ("fit_positive", "fit_clean", "validation_positive", "validation_clean")}
    for split in ("train", "dev"):
        for row in bundle[split]:
            rows[row["legacy_split"]].append({**row, "image": row["image"][0],
                "mask": row["mask"].astype(np.uint8) * 255,
                # Historical geometric augmentation accepts a skeleton; no skeleton target is used here.
                "line_skeleton": np.zeros_like(row["mask"], dtype=np.uint8)})
    return rows

def parser(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--segformer-b0-weights", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--optimizer-steps", type=int, default=28300)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--negative-ratio", type=float, default=1.)
    p.add_argument("--learning-rate", type=float, default=.0003)
    p.add_argument("--weight-decay", type=float, default=.0001)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--validation-interval", type=int, default=500)
    p.add_argument("--smoke", action="store_true", help="4 updates and 8 dev rows; cannot qualify as formal result")
    a = p.parse_args(argv)
    if a.optimizer_steps <= 0 or a.batch_size < 2 or a.batch_size % 2 or a.negative_ratio != 1.:
        p.error("positive steps and an even, 1:1 balanced batch are required")
    if a.smoke:
        a.optimizer_steps, a.validation_interval, a.warmup_steps = 4, 4, 1
    if a.seed != 42:
        p.error('this ablation is registered for seed42 only')
    if (not a.smoke and (a.optimizer_steps != 28300 or a.validation_interval != 500 or a.warmup_steps != 500)):
        p.error('formal budget and validation schedule are fixed')
    if a.batch_size != 8 or a.learning_rate != .0003 or a.weight_decay != .0001:
        p.error('registered batch and optimizer settings are fixed')
    return a

def validate_paths(args, dataset):
    for path in (args.bundle, args.segformer_b0_weights, args.output_dir):
        dataset.safe_path(path)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError(f"create-only output exists: {args.output_dir}")
    if args.output_dir.resolve().is_relative_to(ROOT):
        raise ValueError("runtime outputs must be outside source checkout")
    if not args.segformer_b0_weights.is_file():
        raise ValueError("local pretrained MiT-B0 weights missing")


def make_stream(fast, *args):
    class CheckpointedStream(fast.OneDeepBatchPrefetcher):
        def __init__(self, *args, state=None, **kwargs):
            super().__init__(*args, **kwargs)
            if state is not None:
                self.rng.setstate(state['rng'])
                for name in ('positive_cycle','clean_cycle'):
                    cycle = getattr(self,name); saved = state[name]
                    if saved['count'] != len(cycle.rows): raise ValueError('sampler pool size differs')
                    order = list(saved['order']); position = saved['position']
                    if (order and sorted(order) != list(range(len(cycle.rows)))) or not 0 <= position <= len(order):
                        raise ValueError('invalid sampler cycle')
                    cycle.order, cycle.position = order, position
            self.consumed_state = self._snapshot()

        def _snapshot(self):
            return {'rng':self.rng.getstate(), **{name:{'order':list(getattr(self,name).order),
                'position':getattr(self,name).position,'count':len(getattr(self,name).rows)}
                for name in ('positive_cycle','clean_cycle')}}

        def _put(self, value):
            if value[0] == 'batch': value = ('batch',(value[1],self._snapshot()))
            return super()._put(value)

        def next(self):
            batch,state = super().next()
            self.consumed_state = state
            return batch

        def state_dict(self):
            return copy.deepcopy(self.consumed_state)

        def close(self):
            super().close()
            if self.thread.is_alive(): self.thread.join(timeout=2)
    return CheckpointedStream(*args)

def capture_state(head, optimizer, stream, step, metadata, curve, best):
    import torch
    return {'schema':'hme-consumed-batch-training-state-v1','step':step,
        'state_dict':{k:v.detach().cpu().clone() for k,v in head.state_dict().items()},
        'optimizer_state_dict':copy.deepcopy(optimizer.state_dict()), 'sampler_state':stream.state_dict(),
        'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
        'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        'metadata':copy.deepcopy(metadata),'curve':copy.deepcopy(curve),'best':copy.deepcopy(best),
        'head_training':head.training}


def save_state(path, state):
    """Keep every validation checkpoint; atomic state + a verified latest pointer."""
    import torch
    if path.exists(): raise ValueError('training state is create-only')
    temporary = path.with_suffix('.tmp')
    torch.save(state,temporary); temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    pointer = {'file':path.name,'step':state['step'],'sha256':digest}
    temporary = path.parent/'resume_latest.tmp'
    temporary.write_text(json.dumps(pointer,indent=2)+'\n'); temporary.replace(path.parent/'resume_latest.json')
    return pointer


def state_digest(module):
    h = hashlib.sha256()
    for name,value in module.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def predict_packed(model,rows,args):
    """Keep exact threshold bits; avoid allocating several GiB of float predictions."""
    import torch
    pixels = 96*768
    bits = np.empty((len(MASK_THRESHOLDS),len(rows),model.query_count,(pixels+7)//8),np.uint8)
    scores = np.empty((len(rows),model.query_count),np.float32)
    model.eval();started=time.monotonic()
    with torch.no_grad():
        for start in range(0,len(rows),args.batch_size):
            values=np.stack([r['image'] for r in rows[start:start+args.batch_size]])
            images=torch.from_numpy(values[:,None].astype(np.float32)/127.5-1).cuda()
            masks,objects,_=model(images)
            probability=masks.sigmoid().cpu().numpy()
            scores[start:start+len(images)]=objects.sigmoid().cpu().numpy()
            for index,threshold in enumerate(MASK_THRESHOLDS):
                bits[index,start:start+len(images)]=np.packbits((probability>=threshold).reshape(len(images),model.query_count,-1),axis=-1)
    return bits,scores,(time.monotonic()-started)/len(rows)


def save_predictions(path,bits,scores,rows):
    np.savez_compressed(path,packed_masks=bits,scores=scores,
                        mask_shape=np.array([len(rows),32,96,768]),
                        mask_thresholds=np.array(MASK_THRESHOLDS),
                        sample_ids=np.array([r['sample_id'] for r in rows]))


def learning_rate_at(step,args):
    if step<=args.warmup_steps:return args.learning_rate*step/max(1,args.warmup_steps)
    progress=(step-args.warmup_steps)/max(1,args.optimizer_steps-args.warmup_steps)
    return args.learning_rate*(.05+.95*.5*(1+math.cos(math.pi*min(1.,progress))))


def training_parts(model,initial):
    return {name:{'parameters':sum(p.numel() for p in part.parameters()),
                  'all_require_grad':all(p.requires_grad for p in part.parameters()),
                  'changed':state_digest(part)!=initial[name],
                  'sha256':state_digest(part)} for name,part in model.named_children()}


def run(args):
    dataset=load('514_revision_native_dataset.py')
    validate_paths(args,dataset)
    if digest(args.bundle/'run_manifest.json')!=BUNDLE_SHA or digest(args.segformer_b0_weights)!=PRETRAINED_SHA:
        raise ValueError('registered data or pretrained weights changed')
    hashes={name:digest(ROOT/'scripts'/name) for name in DEPENDENCIES}
    bundle=dataset.load_bundle(args.bundle);data=make_rows(bundle)
    import torch
    architecture=load('654_revision_three_part_model.py')
    baseline=load('422_revision_paper_baselines.py')
    base=load('354_revision_localizer_optimized.py');extension=load('357_revision_localizer_extensions.py')
    probe=load('442_revision_instance_head_probe.py');fullres=load('480_revision_fullres_instance_seed42.py')
    fast=load('483_revision_fast_train_loop.py')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(4);fast.configure_determinism()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    model=architecture.build_model(baseline,args).cuda()
    if list(dict(model.named_children()))!=['see','find','draw']:raise RuntimeError('unexpected model parts')
    if not all(p.requires_grad for p in model.parameters()):raise RuntimeError('frozen parameter in joint model')
    initial={name:state_digest(part) for name,part in model.named_children()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    stream=make_stream(fast,base,extension,data,args,random.Random(args.seed))
    dev=data['validation_positive']+data['validation_clean'];positive_count=len(data['validation_positive'])
    if args.smoke:dev=data['validation_positive'][:4]+data['validation_clean'][:4];positive_count=4
    truths=[r['instances'] for r in dev]
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    manifest={'schema_version':SCHEMA,'status':'running','arm':args.arm,'seed':args.seed,
        'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'command':[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]],
        'git_revision':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        'source_sha256':hashes,'dataset_manifest_sha256':BUNDLE_SHA,'pretrained_sha256':PRETRAINED_SHA,
        'counts':dataset.COUNTS,'corrected_dev_components':dataset.DEV_COMPONENTS,
        'model':architecture.configuration(args.arm),'initial_part_digests':initial,
        'model_parameters':sum(p.numel() for p in model.parameters()),
        'optimizer_parameter_names':[name for name,_ in model.named_parameters()],
        'selection':'same 7 mask x 5 score grid as594; checkpoint max F1 then pixel IoU from second half; no early stop',
        'training':'joint network from common pretrained MiT-B0; other parameters freshly initialized; no R3 checkpoint',
        'frozen_test_touched':False,'correction_applied_in_training':False,
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
                       'gpu':torch.cuda.get_device_name(0),'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}}
    write_json(out/'run_manifest.json',manifest)
    curve=[];best_key=None;best_step=None;best_reading=None;gradient_checks={}
    start=time.monotonic();stream.start()
    try:
        for step in range(1,args.optimizer_steps+1):
            model.train();rate=learning_rate_at(step,args)
            for group in optimizer.param_groups:group['lr']=rate
            images,masks,_=stream.next()
            value=torch.from_numpy(np.stack(images)[:,None].astype(np.float32)/127.5-1).cuda()
            targets=[[torch.from_numpy(piece.astype(np.float32)).cuda()
                      for piece in probe.gold_instances(mask.astype(np.float32)/255)] for mask in masks]
            prediction,objects,_=model(value)
            loss,pieces=architecture.losses(model,prediction,objects,targets,probe,fullres)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            if step==1:
                for name,part in model.named_children():
                    count=sum(p.numel() for p in part.parameters())
                    nonzero=any(p.grad is not None and bool((p.grad!=0).any()) for p in part.parameters())
                    if count and not nonzero:raise RuntimeError(f'{name} has no learning gradient')
                    gradient_checks[name]={'parameters':count,'nonzero_gradient':nonzero,'joint':True}
                encoder=model.see.encoder
                if not any(p.grad is not None and bool((p.grad!=0).any()) for p in encoder.parameters()):
                    raise RuntimeError('backbone gradient missing')
                gradient_checks['backbone_nonzero']=True
                write_json(out/'gradient_preflight.json',gradient_checks)
            optimizer.step()
            if step==1 or step%100==0:
                progress={'arm':args.arm,'step':step,'steps':args.optimizer_steps,'loss':float(loss.detach()),
                          'gradient_norm':float(norm),'elapsed_seconds':time.monotonic()-start,
                          'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30}
                write_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
            if not fast.validation_is_due(step,args.optimizer_steps,args.validation_interval,round(args.optimizer_steps*.5)):continue
            bits,scores,seconds=predict_packed(model,dev,args)
            reading=evaluate(bits,scores,truths,positive_count,packed=True)
            entry={'step':step,'learning_rate':rate,'validation':reading,'loss_terms':pieces}
            curve.append(entry)
            with (out/'curve.jsonl').open('a') as f:f.write(json.dumps(entry,allow_nan=False)+'\n')
            key=(reading['instance']['0.50']['f1'],reading['pixel_iou'])
            if step>=round(args.optimizer_steps*.5) and (best_key is None or key>best_key):
                best_key,best_step,best_reading=key,step,reading
                torch.save({'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                            'step':step,'arm':args.arm,'seed':args.seed},out/'best.pt')
            state=capture_state(model,optimizer,stream,step,manifest,curve,
                                {'step':best_step,'key':best_key,'reading':best_reading})
            pointer=save_state(out/f'resume_{step:06d}.pt',state)
            del state
            if step==args.optimizer_steps:
                save_predictions(out/'final_predictions.npz',bits,scores,dev)
                final_reading=reading
            print(json.dumps({'event':'validation','arm':args.arm,'step':step,'best_step':best_step,
                              **reading['instance']['0.50']}),flush=True)
            del bits,scores
        changed=training_parts(model,initial)
        if any(x['parameters'] and not x['changed'] for x in changed.values()):raise RuntimeError('model part did not update')
        saved=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(saved['state_dict'],strict=True)
        bits,scores,seconds=predict_packed(model,dev,args)
        reading=evaluate(bits,scores,truths,positive_count,packed=True)
        if reading!=best_reading:raise RuntimeError('checkpoint reload changed predictions')
        save_predictions(out/'dev_predictions.npz',bits,scores,dev)
        with np.load(out/'dev_predictions.npz',allow_pickle=False) as packed:
            if evaluate(packed['packed_masks'],packed['scores'],truths,positive_count,packed=True)!=reading:
                raise RuntimeError('serialized predictions changed metrics')
        if hashes!={name:digest(ROOT/'scripts'/name) for name in DEPENDENCIES}:raise RuntimeError('source changed')
        late=[r['validation']['instance']['0.50']['f1'] for r in curve if r['step']>=round(args.optimizer_steps*.5)]
        summary={'schema_version':SCHEMA,'arm':args.arm,'seed':args.seed,'optimizer_steps':args.optimizer_steps,
                 'best_step':best_step,'validation':reading,'final_validation':final_reading,
                 'late_mean_f1':float(np.mean(late)),'late_last10_f1':float(np.mean(late[-10:])),
                 'model_parameters':manifest['model_parameters'],'initial_part_digests':initial,'parts_after_training':changed,
                 'gradient_preflight':gradient_checks,'checkpoint_reload_verified':True,'prediction_recompute_verified':True,
                 'last_saved_training_state':pointer,'validation_points':len(curve),
                 'inference_seconds_per_image':seconds,'elapsed_seconds':time.monotonic()-start,
                 'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'smoke':args.smoke,'frozen_test_touched':False}
        write_json(out/'summary.json',summary);manifest['status']='completed';write_json(out/'run_manifest.json',manifest)
        write_json(out/'complete.json',{'schema_version':SCHEMA,'smoke':args.smoke,'steps':args.optimizer_steps,
                    'files':{p.name:digest(p) for p in out.iterdir() if p.is_file()}})
        print(json.dumps({'event':'complete','arm':args.arm,'f1':reading['instance']['0.50']['f1']}),flush=True)
    except BaseException as exc:
        write_json(out/'failure.json',{'error':type(exc).__name__,'message':str(exc)})
        raise
    finally:stream.close()


if __name__=='__main__':
    run(parser())

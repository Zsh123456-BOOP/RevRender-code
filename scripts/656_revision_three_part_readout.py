"""Recount completed three-part ablations and verify genuine joint training."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def match_pairs(ious, keep, threshold=.5):
    edges = sorted([(float(ious[p, g]), int(p), int(g)) for p in keep
                    for g in range(ious.shape[1]) if ious[p, g] >= threshold],
                   key=lambda value: (-value[0], value[1], value[2]))
    used_p, used_g = set(), set()
    for _, p, g in edges:
        if p not in used_p and g not in used_g:
            used_p.add(p); used_g.add(g)
    return used_p, used_g

def selected_diagnostics(bits, scores, truths, rows, reading, train):
    threshold_index = list(train.MASK_THRESHOLDS).index(reading["mask_threshold"])
    pixels = int(np.prod(truths[0].shape[-2:]))
    totals = {"matched": 0, "pool_coverage_050": 0, "pool_one_to_one_050": 0,
              "no_pool_iou025": 0, "pool_iou025_to050": 0,
              "score_filtered": 0, "assignment_conflict": 0}
    grouped = {"single": [0, 0], "multi": [0, 0], "area_le256": [0, 0], "area_gt256": [0, 0]}
    matches = []
    for row, truth in enumerate(truths):
        binary = np.unpackbits(bits[threshold_index, row], axis=-1, count=pixels).astype(bool)
        gold = truth.reshape(len(truth), pixels)
        areas = binary.sum(1)
        inter = np.stack([binary[:, target].sum(1) for target in gold], 1) if len(gold) else np.zeros((len(binary), 0))
        ious = inter / np.maximum(1, areas[:, None] + gold.sum(1)[None] - inter)
        keep = np.flatnonzero((scores[row] >= reading["score_threshold"]) & (areas > 0))
        _, matched = match_pairs(ious, keep)
        _, pool_matched = match_pairs(ious, np.flatnonzero(areas > 0))
        totals["matched"] += len(matched)
        totals["pool_one_to_one_050"] += len(pool_matched)
        matches.extend(f"{rows[row]['sample_id']}#{g}" for g in sorted(matched))
        best = ious.max(0) if len(gold) else []
        totals["pool_coverage_050"] += int(np.sum(np.asarray(best) >= .5))
        for g in range(len(gold)):
            for group in ("single" if len(gold) == 1 else "multi", "area_le256" if gold[g].sum() <= 256 else "area_gt256"):
                grouped[group][0] += int(g in matched); grouped[group][1] += 1
            if g in matched:
                continue
            if best[g] < .25:
                totals["no_pool_iou025"] += 1
            elif best[g] < .5:
                totals["pool_iou025_to050"] += 1
            elif not (ious[keep, g] >= .5).any():
                totals["score_filtered"] += 1
            else:
                totals["assignment_conflict"] += 1
    return {**totals, "grouped_tp_total": grouped, "matched_target_ids": matches,
            "scope": "all32 queries at this run's selected mask threshold; thresholds may differ across arms"}


def validate_joint_state(saved,manifest,train):
    if saved['step']!=manifest['arguments']['optimizer_steps']:
        raise ValueError('joint training budget not completed')
    if saved['metadata']['source_sha256']!=manifest['source_sha256']:
        raise ValueError('saved training source differs')
    names=list(saved['state_dict'])
    if not all(n.split('.')[0] in ('see','find','draw') for n in names):
        raise ValueError('unexpected fourth model part')
    # Optimizer parameter names are captured in the manifest rather than guessed from state_dict buffers.
    groups=saved['optimizer_state_dict']['param_groups'];states=saved['optimizer_state_dict']['state']
    parameter_names=manifest['optimizer_parameter_names']
    ids=[i for g in groups for i in g['params']]
    if len(ids)!=len(parameter_names):raise ValueError('optimizer identity mismatch')
    steps={name:int(states[index]['step']) if index in states else 0 for name,index in zip(parameter_names,ids)}
    if any(not 0<=step<=saved['step'] for step in steps.values()):raise ValueError('invalid optimizer steps')
    required=['see.pixel_decoder.output.weight','find.queries','find.seed_fine_projection.weight']
    required.append(next(n for n in parameter_names if n.startswith('see.encoder.') and n.endswith('weight')))
    arm=manifest['arm']
    if arm=='b_point':required.append('draw.point_head.0.weight')
    if arm in ('c_identity','d_competition'):required.extend(['draw.query_code.1.weight','draw.comp_attention.in_proj_weight'])
    if arm=='d_competition':required.append('draw.owner_mix')
    if any(steps.get(n)!=saved['step'] for n in required):raise ValueError('joint path did not update at every step')
    return {'required_parameter_steps':{n:steps[n] for n in required},
            'inactive_parameter_names':[n for n,step in steps.items() if step==0]}


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--run-root',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    spec=importlib.util.spec_from_file_location('threepart655',ROOT/'scripts/655_revision_three_part_train.py')
    train=importlib.util.module_from_spec(spec);spec.loader.exec_module(train)
    dataset=train.load('514_revision_native_dataset.py')
    for path in (a.run_root,a.bundle,a.output_dir):dataset.safe_path(path)
    if a.output_dir.exists() or a.output_dir.is_symlink():raise ValueError('create-only output exists')
    if train.digest(a.bundle/'run_manifest.json')!=train.BUNDLE_SHA:raise ValueError('dataset mismatch')
    data=train.make_rows(dataset.load_bundle(a.bundle))
    rows=data['validation_positive']+data['validation_clean'];truths=[r['instances'] for r in rows]
    positive_count=len(data['validation_positive']);results={};common=None
    import torch
    for arm in train.ARMS:
        path=a.run_root/(arm+'_seed42')
        complete=json.loads((path/'complete.json').read_text());m=json.loads((path/'run_manifest.json').read_text())
        s=json.loads((path/'summary.json').read_text())
        if complete['smoke'] or complete['steps']!=28300 or s['seed']!=42 or m['arm']!=arm or m['status']!='completed':
            raise ValueError('incomplete or wrong arm')
        if (path/'failure.json').exists():raise ValueError('failure present')
        for name,sha in complete['files'].items():
            if Path(name).name!=name or train.digest(path/name)!=sha:raise ValueError('artifact changed')
        for name,sha in m['source_sha256'].items():
            if Path(name).name!=name or train.digest(ROOT/'scripts'/name)!=sha:raise ValueError('source changed')
        key={k:m[k] for k in ['dataset_manifest_sha256','pretrained_sha256','selection','source_sha256']}
        key['initial_see_find']={n:m['initial_part_digests'][n] for n in ('see','find')}
        if common is None:common=key
        elif common!=key:raise ValueError('ablation inputs or shared initialization differ')
        for name,state in s['parts_after_training'].items():
            if state['parameters'] and (not state['changed'] or not state['all_require_grad']):raise ValueError('frozen model part')
        pointer=s['last_saved_training_state']
        if train.digest(path/pointer['file'])!=pointer['sha256']:raise ValueError('last state hash mismatch')
        saved=torch.load(path/pointer['file'],map_location='cpu',weights_only=False)
        state_check=validate_joint_state(saved,m,train);del saved
        readings={}
        for filename,expected in [('dev_predictions.npz',s['validation']),('final_predictions.npz',s['final_validation'])]:
            with np.load(path/filename,allow_pickle=False) as packed:
                if packed['sample_ids'].tolist()!=[r['sample_id'] for r in rows]:raise ValueError('sample order changed')
                reading=train.evaluate(packed['packed_masks'],packed['scores'],truths,positive_count,packed=True)
                if reading!=expected:raise ValueError('saved predictions do not recount')
                diagnostic=selected_diagnostics(packed['packed_masks'],packed['scores'],truths,rows,reading,train)
                readings[filename]={'reading':reading,'diagnostic':diagnostic}
        results[arm]={'summary':s,'readings':readings,'joint_optimizer_verified':state_check,
                      'complete_sha256':train.digest(path/'complete.json')}
    reference=set(results[train.ARMS[0]]['readings']['dev_predictions.npz']['diagnostic']['matched_target_ids'])
    for arm,result in results.items():
        found=set(result['readings']['dev_predictions.npz']['diagnostic']['matched_target_ids'])
        result['changes_vs_coarse']={'gained':sorted(found-reference),'lost':sorted(reference-found)}
    a.output_dir.mkdir(parents=True,exist_ok=False)
    output={'schema_version':'hme-three-part-readout-v1','arms':results,'all_four_recounted':True,
            'shared_initialization_verified':True,'frozen_test_touched':False,'source_inputs':common}
    train.write_json(a.output_dir/'results.json',output)
    lines=['|方案|找到|漏掉|误报|F1|后期F1|像素IoU|小目标找到|','|---|---|---|---|---|---|---|---|']
    for arm,result in results.items():
        s=result['summary'];v=s['validation'];m=v['instance']['0.50']
        small=result['readings']['dev_predictions.npz']['diagnostic']['grouped_tp_total']['area_le256'][0]
        lines.append(f"|{arm}|{m['true_positive']}|{m['false_negative']}|{m['false_positive']}|{m['f1']:.4f}|{s['late_mean_f1']:.4f}|{v['pixel_iou']:.4f}|{small}|")
    (a.output_dir/'table.md').write_text('\n'.join(lines)+'\n')
    train.write_json(a.output_dir/'complete.json',{'all_four_recounted':True,'files':{n:train.digest(a.output_dir/n) for n in ['results.json','table.md']}})
    print('\n'.join(lines))


if __name__=='__main__':main()

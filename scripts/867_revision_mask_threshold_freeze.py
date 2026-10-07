"""Freeze one of our C832 runs with the mask threshold chosen on validation too: 0.3, 0.4 or 0.5.

835 fixed our masks at 0.30 and chose only NMS and cut; 775 found most of the
pixel-precision gap to MaskDINO was that convention, and the corrected model
reads F1 .6799 / .6860 / .6616 at 0.3 / 0.4 / 0.5.  The user decided on
2026-09-27 that from now on the threshold is chosen on validation with the
other two knobs, for every one of our models alike.  This is 835 with that one
change: 703's frontier at each threshold, the best F1 over all of them (ties:
fewer false positives, then the lower threshold), pixel metrics at that point.
MaskDINO and PointRend archive binary masks, so they keep their native 0.5.

This script deliberately does NOT: look at the test split; choose on anything
but the validation split; or change a checkpoint.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = (.3, .4, .5)


def load(name):
    spec = importlib.util.spec_from_file_location('freeze867_' + name[:3], ROOT/'scripts'/name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pick(readings):
    """Best F1; ties to fewer false positives, then the lower threshold."""
    return min(readings, key=lambda r: (-r['f1'], r['false_positive'], r['mask_threshold']))


def main(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False)
    for name in ('bundle', 'run', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--bundle-sha', required=True)
    args = p.parse_args(argv)
    data = load('833_revision_candidate_bundle.py')
    data.require(not args.output.exists(), 'create-only output exists')
    summary = json.loads((args.run/'summary.json').read_text())
    manifest = json.loads((args.run/'run_manifest.json').read_text())
    data.require(summary['stage'] == 'formal' and summary['checkpoint_reload_verified'] and (args.run/'complete.json').is_file(),
                 'incomplete formal run')
    data.require(manifest['profile']['dataset_sha256'] == args.bundle_sha, 'wrong run dataset')
    checkpoint, archive = args.run/'best.pt', args.run/'dev_predictions.npz'
    bundle = data.load_bundle(args.bundle, args.bundle_sha)
    matched = load('703_revision_matched_operating_points.py')
    dev = sorted(bundle['dev'], key=lambda r: not r['is_positive'])
    order = [r['sample_id'].split(':', 1)[1] for r in dev]
    truths = [list(r['instances']) for r in dev]
    gold = sum(map(len, truths))
    readings, frontiers = [], {}
    for level in CANDIDATES:
        provide, available = matched.TABLE.packed_provider(archive, level)
        data.require(set(order) == available, 'validation archive coverage mismatch')
        points = matched.frontier(provide, order, truths, gold)
        best = dict(max(points, key=lambda x: (x['f1'], -x['false_positive'])))
        best.update(matched.pixel_iou(provide, order, truths, best), mask_threshold=level)
        readings.append(best)
        frontiers[level] = points
    best = pick(readings)
    points = frontiers[best['mask_threshold']]
    result = {'schema': 'hme-revision-c832-freeze-v2', 'split': 'C832', 'model': 'point_context',
              'candidate': bundle['manifest']['candidate'], 'test_touched': False,
              'common_test_sample_ids_sha256': data.COMMON_TEST_SHA, 'bundle_sha256': args.bundle_sha,
              'chosen_step': summary['best_step'], 'checkpoint': str(checkpoint), 'checkpoint_sha256': data.digest(checkpoint),
              'selection_record_sha256': data.digest(args.run/'summary.json'), 'validation_archive': str(archive),
              'mask_threshold': best['mask_threshold'], 'mask_threshold_rule': 'chosen on validation among 0.3/0.4/0.5 (user, 2026-09-27)',
              'score_threshold': best['score_threshold'], 'nms': best['nms'],
              'validation': {'gold_components': gold, 'own_best': best, 'by_mask_threshold': {str(r['mask_threshold']): r for r in readings},
                             'tp_within_fp_budget': {str(b): (matched.true_positives_at(points, b) or {}).get('true_positive')
                                                     for b in (41, 55, 65, 70, 83, 111)}},
              'frozen_at': datetime.now(timezone.utc).isoformat()}
    data.write_json(args.output, result)
    print(json.dumps({'mask_threshold': best['mask_threshold'], 'f1': round(best['f1'], 4),
                      'tp_fp': [best['true_positive'], best['false_positive']], 'pixel_iou': round(best['pixel_iou'], 4)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

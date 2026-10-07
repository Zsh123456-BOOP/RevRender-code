"""COCO-style mask AP for every frozen model on validation gold v2, beside the component F1 we report.

Instance segmentation papers (MaskDINO, PointRend) rank by COCO mask AP: the
area under the precision-recall curve of all scored masks, averaged over
match thresholds IoU 0.50:0.05:0.95, with no score cut to choose.  Our main
metric is component F1 at IoU 0.50 at a validation-chosen operating point.
This reads, from each frozen record's validation archive (its own mask
threshold), with one class and COCO's rules -- per line, detections in
descending score each take the best still-unmatched gold at or above the
threshold; the curve over all lines is made monotone and sampled at 101 recall
points; at most 100 detections per line:

  AP, AP50, AP75   over the model's masks after its own frozen mask NMS
  AP_raw, AP50_raw the same with no NMS

against 878's gold-v2 bundle.  This script deliberately does NOT: change a
frozen point or a mask; choose anything; read the frozen common test; or
contact the network.
"""
import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'hme-revision-mask-ap883-v1'
IOUS = tuple(round(.5+.05*i, 2) for i in range(10))
MAX_DETS = 100


def load(name, key):
    spec = importlib.util.spec_from_file_location(key, ROOT/'scripts'/name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def match_line(iou, scores, threshold):
    """COCO's per-image matching: detections in descending score take the best unmatched gold with IoU >= threshold."""
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))[:MAX_DETS]
    taken, out = set(), []
    for i in order:
        best, best_iou = None, threshold-1e-12
        for g in range(iou.shape[1]):
            if g not in taken and iou[i, g] >= best_iou and iou[i, g] >= threshold:
                best, best_iou = g, iou[i, g]
        if best is not None:
            taken.add(best)
        out.append((float(scores[i]), best is not None))
    return out


def average_precision(records, gold):
    """COCO 101-point interpolated AP from (score, is_tp) over all lines."""
    import numpy as np
    if not gold:
        return 0.
    records = sorted(records, key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in records], dtype=np.float64)
    fp = np.cumsum([not r[1] for r in records], dtype=np.float64)
    recall = tp/gold
    precision = tp/np.maximum(tp+fp, np.finfo(np.float64).eps)
    for i in range(len(precision)-2, -1, -1):
        precision[i] = max(precision[i], precision[i+1])
    points = np.linspace(0, 1, 101)
    index = np.searchsorted(recall, points, side='left')
    sampled = [precision[i] if i < len(precision) else 0. for i in index]
    return float(np.mean(sampled))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--freeze', action='append', required=True, help='label=freeze.json')
    parser.add_argument('--bundle', type=Path, required=True, help="878's gold-v2 bundle")
    parser.add_argument('--bundle-sha', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit('883 refused: create-only output exists')
    data_code = load('833_revision_candidate_bundle.py', 'ap883_833')
    matched = load('703_revision_matched_operating_points.py', 'ap883_703')
    extent = load('872_revision_extent_disagreement.py', 'ap883_872')
    manifest = data_code.manifest_at(args.bundle, args.bundle_sha)
    source = (manifest.get('gold_v2') or {}).get('from_bundle_sha256')
    bundle = data_code.load_bundle(args.bundle, args.bundle_sha)
    dev = sorted(bundle['dev'], key=lambda r: not r['is_positive'])
    gold = sum(len(r['instances']) for r in dev)
    report = {}
    for entry in args.freeze:
        label, path = entry.split('=', 1)
        frozen = json.loads(Path(path).read_text())
        if frozen['test_touched'] is not False or frozen['bundle_sha256'] not in (args.bundle_sha, source):
            raise SystemExit(f'883 refused: {label} is not a validation freeze of this bundle or its gold-v2 source')
        archive = Path(frozen['validation_archive'])
        provide = (matched.TABLE.packed_provider(archive, float(frozen['mask_threshold']))[0] if frozen['model'] == 'point_context'
                   else matched.TABLE.native_provider(archive)[0])
        nms = frozen['validation']['own_best']['nms']
        records = {(kind, t): [] for kind in ('nms', 'raw') for t in IOUS}
        for row in dev:
            masks, scores = provide(row['sample_id'].split(':', 1)[1])
            truth = list(row['instances'])
            for kind, level in (('nms', nms), ('raw', 1.)):
                keep = [i for i in matched.SUPPRESS.suppressed(masks, scores, level) if masks[i].any()]
                kept = [masks[i] for i in keep]
                iou = extent.iou_matrix(kept, truth)
                for t in IOUS:
                    records[(kind, t)] += match_line(iou, [scores[i] for i in keep], t) if kept else []
        values = {kind: {str(t): average_precision(records[(kind, t)], gold) for t in IOUS} for kind in ('nms', 'raw')}
        report[label] = {'model': frozen['model'], 'mask_threshold': frozen['mask_threshold'], 'nms': nms,
                         'AP': sum(values['nms'].values())/len(IOUS), 'AP50': values['nms']['0.5'], 'AP75': values['nms']['0.75'],
                         'AP_raw': sum(values['raw'].values())/len(IOUS), 'AP50_raw': values['raw']['0.5'],
                         'by_iou': values, 'freeze_sha256': digest(path)}
        r = report[label]
        print(json.dumps({'label': label, **{k: round(r[k], 4) for k in ('AP', 'AP50', 'AP75', 'AP_raw', 'AP50_raw')}}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'schema': SCHEMA, 'gold_components': gold, 'models': report, 'test_touched': False,
                                       'bundle_sha256': args.bundle_sha, 'script_sha256': digest(Path(__file__).resolve()),
                                       'python': platform.python_version()}, indent=1, sort_keys=True)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

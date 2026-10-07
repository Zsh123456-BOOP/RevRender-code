"""How much of the remaining error is where a revision is, and how much is how far it extends.

At a model's frozen point every validation error is either a target nothing
was delivered near, a delivered mask near nothing, or a near pair: a delivered
mask and a gold component that overlap (IoU 0.10-0.50) and so count once as a
false positive and once as a miss.  811 split near pairs by shape on split B;
this reads, on morphology_s42 and for any frozen record (ours or native):

  sweep     F1 of the frozen delivery when the matching IoU is 0.1 ... 0.75
            instead of 0.50 -- nothing re-selected
  pairs     after 703's greedy matching at 0.50, the unmatched delivered masks
            and unmatched components paired greedily at IoU >= 0.10, each pair
            classified in this order:
              merged    the mask covers >= 0.2 of two or more components
              boundary  within 2 px of each other both ways (>= 0.9 of each lies
                        within 2 px of the other): the same strokes, drawn
                        thicker, thinner or with a different edge
              piece     >= 0.9 of the mask lies within 2 px of the component,
                        which it does not cover: a piece or a short line
              overreach >= 0.9 of the component lies within 2 px of the mask,
                        which spills beyond it: too long, too fat, extra ink
              offset    anything else
  singles   misses with no delivered mask at IoU >= 0.10; false positives with
            no component at IoU >= 0.10 (on a clean line, or elsewhere)
  what-if   F1 if each kind of pair were a hit (TP +1, FP -1, FN -1 per pair)

This script deliberately does NOT: change a frozen point, a mask or the
matching rule of the main metric; train or choose anything; read the frozen
common test; or contact the network.  The what-if numbers are a decomposition,
not a result.
"""
import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'hme-revision-extent-disagreement872-v1'
SWEEP = (.1, .2, .3, .4, .5, .6, .7, .75)
KINDS = ('merged', 'boundary', 'piece', 'overreach', 'offset')
NEAR, TOLERANCE_PX, WITHIN = .1, 2, .9


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


def dilate(mask, radius=TOLERANCE_PX):
    import numpy as np
    from scipy import ndimage
    return ndimage.binary_dilation(mask, structure=np.ones((3, 3), bool), iterations=radius)


def iou_matrix(masks, truth):
    import numpy as np
    if not masks or not truth:
        return np.zeros((len(masks), len(truth)))
    a = np.stack([m.reshape(-1) for m in masks]).astype(np.float32)
    b = np.stack([t.reshape(-1) for t in truth]).astype(np.float32)
    inter = a @ b.T
    return inter/np.maximum(a.sum(1)[:, None]+b.sum(1)[None]-inter, 1)


def greedy(iou, rows, columns, threshold):
    """703's rule on the given rows and columns: highest IoU first, one-to-one, at or above the threshold."""
    pairs = sorted(((iou[r, g], r, g) for r in rows for g in columns if iou[r, g] >= threshold),
                   key=lambda t: (-t[0], t[1], t[2]))
    used_r, used_g, out = set(), set(), {}
    for _, r, g in pairs:
        if r not in used_r and g not in used_g:
            used_r.add(r)
            used_g.add(g)
            out[g] = r
    return out


def kind_of(mask, component, others):
    """One near pair's kind; `others` are the line's other gold components."""
    area = float(mask.sum())
    if sum(float((mask & c).sum()) >= .2*float(c.sum()) for c in [component, *others]) >= 2:
        return 'merged'
    inside = float((mask & dilate(component)).sum())/area
    covered = float((component & dilate(mask)).sum())/float(component.sum())
    if inside >= WITHIN and covered >= WITHIN:
        return 'boundary'
    if inside >= WITHIN:
        return 'piece'
    if covered >= WITHIN:
        return 'overreach'
    return 'offset'


def read_line(masks, scores, truth, nms, cut, suppressed):
    """The line's delivered masks, its matches at each sweep IoU, and its near pairs and singles at 0.50."""
    delivered = [masks[i] for i in suppressed(masks, scores, nms) if masks[i].any() and float(scores[i]) >= cut]
    iou = iou_matrix(delivered, truth)
    rows, columns = range(len(delivered)), range(len(truth))
    sweep = {t: len(greedy(iou, rows, columns, t)) for t in SWEEP}
    won = greedy(iou, rows, columns, .5)
    free_rows = [r for r in rows if r not in won.values()]
    free_columns = [g for g in columns if g not in won]
    near = greedy(iou, free_rows, free_columns, NEAR)
    pairs = [{'kind': kind_of(delivered[r], truth[g], [truth[h] for h in columns if h != g]), 'iou': float(iou[r, g]),
              'area_ratio': float(delivered[r].sum())/float(truth[g].sum())} for g, r in near.items()]
    lone_rows = [r for r in free_rows if r not in near.values()]
    singles = Counter({'miss_nothing_near': len([g for g in free_columns if g not in near]),
                       'false_positive_clean_line': len(lone_rows) if not truth else 0,
                       'false_positive_elsewhere': len(lone_rows) if truth else 0})
    return len(delivered), len(truth), sweep, pairs, singles


def f1(tp, delivered, gold):
    return 2*tp/max(1, delivered+gold)


def summarise(lines):
    import numpy as np
    delivered = sum(d for d, _, _, _, _ in lines)
    gold = sum(g for _, g, _, _, _ in lines)
    sweep = {str(t): sum(s[t] for _, _, s, _, _ in lines) for t in SWEEP}
    pairs = [p for _, _, _, ps, _ in lines for p in ps]
    singles = sum((s for _, _, _, _, s in lines), Counter())
    tp = sweep['0.5']
    kinds = Counter(p['kind'] for p in pairs)
    what_if = {k: f1(tp+kinds[k], delivered, gold) for k in KINDS}
    what_if['all_pairs'] = f1(tp+len(pairs), delivered, gold)
    return {'delivered': delivered, 'gold': gold, 'true_positive': tp, 'false_positive': delivered-tp, 'false_negative': gold-tp,
            'f1_by_match_iou': {t: f1(v, delivered, gold) for t, v in sweep.items()}, 'tp_by_match_iou': sweep,
            'near_pairs': len(pairs), 'pairs_by_kind': {k: kinds[k] for k in KINDS},
            'pair_iou_median': {k: float(np.median([p['iou'] for p in pairs if p['kind'] == k])) if kinds[k] else None for k in KINDS},
            'pair_area_ratio_median': {k: float(np.median([p['area_ratio'] for p in pairs if p['kind'] == k])) if kinds[k] else None
                                       for k in KINDS},
            'singles': dict(singles), 'f1_if_pairs_were_hits': what_if}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--freeze', action='append', required=True, help='label=freeze.json')
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--bundle-sha', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit('872 refused: create-only output exists')
    data_code = load('833_revision_candidate_bundle.py', 'extent872_833')
    matched = load('703_revision_matched_operating_points.py', 'extent872_703')
    bundle = data_code.load_bundle(args.bundle, args.bundle_sha)
    dev = sorted(bundle['dev'], key=lambda r: not r['is_positive'])
    report = {}
    for entry in args.freeze:
        label, path = entry.split('=', 1)
        frozen = json.loads(Path(path).read_text())
        if frozen['test_touched'] is not False or frozen['bundle_sha256'] != args.bundle_sha:
            raise SystemExit(f'872 refused: {label} is not a validation freeze of this bundle')
        archive = Path(frozen['validation_archive'])
        provide = (matched.TABLE.packed_provider(archive, float(frozen['mask_threshold']))[0] if frozen['model'] == 'point_context'
                   else matched.TABLE.native_provider(archive)[0])
        best = frozen['validation']['own_best']
        lines = []
        for row in dev:
            masks, scores = provide(row['sample_id'].split(':', 1)[1])
            lines.append(read_line(masks, scores, list(row['instances']), best['nms'], best['score_threshold'],
                                   matched.SUPPRESS.suppressed))
        summary = summarise(lines)
        if summary['true_positive'] != best['true_positive'] or summary['false_positive'] != best['false_positive']:
            raise SystemExit(f'872 refused: {label} recount disagrees with its freeze')
        report[label] = dict(summary, mask_threshold=frozen['mask_threshold'], freeze_sha256=digest(path))
        print(json.dumps({'label': label, 'f1_by_iou': {k: round(v, 4) for k, v in summary['f1_by_match_iou'].items()},
                          'pairs': summary['pairs_by_kind'], 'singles': summary['singles'],
                          'what_if': {k: round(v, 4) for k, v in summary['f1_if_pairs_were_hits'].items()}}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'schema': SCHEMA, 'models': report, 'test_touched': False,
                                       'definitions': {'near': NEAR, 'tolerance_px': TOLERANCE_PX, 'within': WITHIN,
                                                       'kinds': list(KINDS), 'sweep': list(SWEEP)},
                                       'script_sha256': digest(Path(__file__).resolve()), 'python': platform.python_version()},
                                      indent=1, sort_keys=True)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Compare models where they are comparable: at each other's precision, not at each other's F1.

701 and 702 each report one point per model -- whichever maximises its own F1.
That is enough to rank F1 and nothing else.  Our best arm reads 0.6029
precision at its own best point and MaskDINO reads 0.7211 at its, so the
sentence "MaskDINO is more precise" is not yet a claim about the models: both
would move if either turned its score knob.  The question a reader actually has
is whether, at the precision MaskDINO delivers, our recall is higher or lower.

So this walks each model's whole two-knob frontier and reads it at anchors
every model can be read at: the precision each model achieves at its own best
point, and the false-positive budgets the baselines spend.  It also reports
pixel IoU at each reported point, because pixel IoU too is a function of the
operating point and quoting one model's from a loose point against another's
from a tight one is the same error a level up.

A frontier point is kept only if no other point has both higher precision and
higher recall, so "recall at precision p" is the best this model can do, not
the best it happens to do at a threshold someone chose for it.

This script deliberately does NOT: train anything, run inference, or load a
checkpoint; re-derive the gold, the matcher or the 0.50 threshold; move any
model's mask threshold; interpolate between measured points or extrapolate
past a model's frontier, reporting no reading instead; claim a dev-tuned
operating point is a test result; read, transcribe or otherwise consume the
frozen test split; or contact the network.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'hme-revision-matched-operating-points-v1'
TABLE_FILE = '701_revision_one_protocol_table.py'
SUPPRESS_FILE = '702_revision_suppression_protocol.py'
FALSE_POSITIVE_BUDGETS = (41, 55, 83, 111)


class Matched703Error(RuntimeError):
    """A refusal that must stop the run rather than degrade its output."""


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TABLE = load_module(ROOT / 'scripts' / TABLE_FILE, 'hme701_for703')
SUPPRESS = load_module(ROOT / 'scripts' / SUPPRESS_FILE, 'hme702_for703')


def sweep_every_cut(tables, gold_total: int) -> list[dict[str, Any]]:
    """Every distinct threshold in one pass instead of one pass per threshold.

    Lowering the cut past a candidate's score extends exactly one row's prefix
    by one, so the totals are a running sum over the globally score-ordered
    candidates rather than a fresh walk over all rows per threshold.  703 reads
    the whole frontier rather than only its best point, so the quadratic form
    701 and 702 use would dominate the run.
    """
    events = []
    for row, (matched, kept, ordered) in enumerate(tables):
        for rank, score in enumerate(ordered):
            events.append((-float(score), row, rank))
    events.sort()
    readings, true_positive, kept_total, index = [], 0, 0, 0
    while index < len(events):
        cut = -events[index][0]
        while index < len(events) and events[index][0] == -cut:
            _, row, rank = events[index]
            matched, kept, _ = tables[row]
            true_positive += int(matched[rank + 1]) - int(matched[rank])
            kept_total += int(kept[rank + 1]) - int(kept[rank])
            index += 1
        false_positive = kept_total - true_positive
        precision = true_positive / (true_positive + false_positive) \
            if true_positive + false_positive else 0.
        recall = true_positive / gold_total if gold_total else 0.
        readings.append({
            'score_threshold': cut, 'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall)
            if precision + recall else 0.,
            'true_positive': true_positive, 'false_positive': false_positive,
            'false_negative': gold_total - true_positive})
    return readings


def frontier(provide, order, truths, gold_total: int) -> list[dict[str, Any]]:
    """Every (nms, score) reading, reduced to the precision/recall frontier."""
    points = []
    for level in SUPPRESS.NMS_LEVELS:
        tables, _ = SUPPRESS.tabulate_at(provide, order, truths, level)
        for reading in sweep_every_cut(tables, gold_total):
            reading['nms'] = level
            points.append(reading)
    points.sort(key=lambda p: (-p['precision'], -p['recall']))
    kept, best_recall = [], -1.
    for point in points:
        if point['recall'] > best_recall:
            kept.append(point)
            best_recall = point['recall']
    return kept


def recall_at_precision(points: Sequence[dict[str, Any]], floor: float):
    """The most recall this model reaches without dropping below that precision."""
    reachable = [p for p in points if p['precision'] >= floor]
    if not reachable:
        return None
    return max(reachable, key=lambda p: (p['recall'], p['precision']))


def true_positives_at(points: Sequence[dict[str, Any]], budget: int):
    affordable = [p for p in points if p['false_positive'] <= budget]
    if not affordable:
        return None
    return max(affordable, key=lambda p: (p['true_positive'], -p['false_positive']))


def pixel_iou(provide, order, truths, point: dict[str, Any]) -> dict[str, float]:
    """Pixels of the kept union against pixels of the gold union, at one point."""
    intersection = predicted = gold_pixels = 0
    for line, truth in zip(order, truths):
        masks, scores = provide(line)
        keep = SUPPRESS.suppressed(masks, scores, point['nms'])
        union = None
        for index in keep:
            if float(scores[index]) < point['score_threshold']:
                continue
            union = masks[index] if union is None else np.logical_or(union, masks[index])
        if union is None and not truth:
            continue
        shape = truth[0].shape if truth else union.shape
        union = np.zeros(shape, bool) if union is None else union
        gold = np.zeros(shape, bool)
        for component in truth:
            gold = np.logical_or(gold, component)
        intersection += int(np.logical_and(union, gold).sum())
        predicted += int(union.sum())
        gold_pixels += int(gold.sum())
    denominator = predicted + gold_pixels - intersection
    return {'pixel_iou': intersection / denominator if denominator else 0.,
            'pixel_precision': intersection / predicted if predicted else 0.,
            'pixel_recall': intersection / gold_pixels if gold_pixels else 0.}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--native', type=str, action='append', default=[])
    parser.add_argument('--ours', type=str, action='append', default=[])
    parser.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args(argv)

    dataset = load_module(ROOT / 'scripts/514_revision_native_dataset.py', 'hme514_for703')
    train = load_module(ROOT / 'scripts/655_revision_three_part_train.py', 'hme655_for703')
    dual = load_module(ROOT / 'scripts/547_revision_dual_definition_scoring.py', 'hme547_for703')

    try:
        report_dir = args.report_dir.expanduser().resolve()
        dataset.safe_path(args.bundle)
        if report_dir.exists() or report_dir.is_symlink():
            raise Matched703Error(f'output exists: {report_dir}')
        if not args.native and not args.ours:
            raise Matched703Error('nothing to read')
        if dataset.digest(args.bundle / 'run_manifest.json') != train.BUNDLE_SHA:
            raise Matched703Error('bundle manifest does not match')

        rows = sorted(dataset.load_bundle(args.bundle)['dev'],
                      key=lambda r: not r['is_positive'])
        order = [str(row['sample_id']).split(':', 1)[1] for row in rows]
        stroke = [list(row['instances']) for row in rows]
        if sum(len(t) for t in stroke) != dual.EXPECTED_STROKE_GOLD:
            raise Matched703Error('stroke gold is not the expected total')
        grouping = dual.read_grouping(dual.GROUPING_PATH)
        event = [dual.event_truths(components, grouping.get(line))
                 for line, components in zip(order, stroke)]
        if sum(len(t) for t in event) != dual.EXPECTED_EVENT_GOLD:
            raise Matched703Error('event gold is not the expected total')

        providers, frontiers, own = {}, {}, {}
        for kind, value in ([('native', v) for v in args.native]
                            + [('ours', v) for v in args.ours]):
            label, source = TABLE.split_label(value)
            if kind == 'ours':
                path, _, mask = source.partition('@')
                provide, available = TABLE.packed_provider(
                    Path(path).expanduser().resolve(), float(mask))
            else:
                provide, available = TABLE.native_provider(
                    Path(source).expanduser().resolve())
            if set(order) - available:
                raise Matched703Error(f'{label}: rows without an archive')
            providers[label] = provide
            frontiers[label] = frontier(provide, order, stroke,
                                        dual.EXPECTED_STROKE_GOLD)
            own[label] = max(frontiers[label], key=lambda p: p['f1'])

        anchors = {label: point['precision'] for label, point in own.items()}
        records = []
        for label, points in frontiers.items():
            best = dict(own[label])
            best.update(pixel_iou(providers[label], order, stroke, best))
            at_precision = {}
            for other, floor in sorted(anchors.items()):
                found = recall_at_precision(points, floor)
                at_precision[other] = None if found is None else {
                    'precision_floor': floor, **found,
                    **pixel_iou(providers[label], order, stroke, found)}
            at_budget = {}
            for budget in FALSE_POSITIVE_BUDGETS:
                found = true_positives_at(points, budget)
                at_budget[str(budget)] = None if found is None else dict(found)
            records.append({
                'model': label, 'own_best_f1_point': best,
                'frontier_points': len(points),
                'recall_at_each_models_precision': at_precision,
                'true_positives_within_false_positive_budget': at_budget,
                'event_level_best': max(
                    frontier(providers[label], order, event,
                             dual.EXPECTED_EVENT_GOLD), key=lambda p: p['f1'])})

        report = {'schema_version': SCHEMA, 'match_iou': TABLE.MATCH_IOU,
                  'rows': len(order), 'matcher': '442 IoU-ordered greedy one-to-one',
                  'protocol': "each model's own mask; score and mask NMS swept for all",
                  'anchors_are_each_models_own_best_f1_precision': anchors,
                  'false_positive_budgets': list(FALSE_POSITIVE_BUDGETS),
                  'both_knobs_selected_on': 'the validation split, for every model alike',
                  'training_performed': False, 'inference_performed': False,
                  'frozen_test_touched': False, 'results': records}
        report_dir.mkdir(parents=True)
        (report_dir / 'matched_operating_points.json').write_text(
            json.dumps(report, indent=1, sort_keys=True, ensure_ascii=False) + '\n')
        (report_dir / 'run_manifest.json').write_text(json.dumps({
            'script_sha256': dual.digest_file(Path(__file__).resolve()),
            'python': platform.python_version(), 'numpy': np.__version__,
            'create_only': True, 'training_performed': False,
            'frozen_test_touched': False}, indent=1, sort_keys=True) + '\n')

        names = sorted(anchors)
        lines = ['recall at each anchor precision (— means the model never reaches it)',
                 f"{'model':<22}" + ''.join(f'{n[:9]:>11}' for n in names)]
        for record in records:
            cells = ''
            for other in names:
                found = record['recall_at_each_models_precision'][other]
                cells += f"{'—':>11}" if found is None else f"{found['recall']:>11.4f}"
            lines.append(f"{record['model']:<22}{cells}")
        lines += ['', f"{'model':<22}{'F1':>8}{'P':>8}{'R':>8}{'pxIoU':>8}{'nms':>6}"
                      f"{'cut':>8}" + ''.join(f'{"TP@" + str(b):>8}'
                                              for b in FALSE_POSITIVE_BUDGETS)]
        for record in sorted(records, key=lambda r: -r['own_best_f1_point']['f1']):
            best = record['own_best_f1_point']
            cells = ''
            for budget in FALSE_POSITIVE_BUDGETS:
                found = record['true_positives_within_false_positive_budget'][str(budget)]
                cells += f"{'—':>8}" if found is None else f"{found['true_positive']:>8}"
            lines.append(f"{record['model']:<22}{best['f1']:>8.4f}{best['precision']:>8.4f}"
                         f"{best['recall']:>8.4f}{best['pixel_iou']:>8.4f}"
                         f"{best['nms']:>6.2f}{best['score_threshold']:>8.4f}{cells}")
        table = '\n'.join(lines) + '\n'
        (report_dir / 'table.txt').write_text(table)
        print(table)
    except Matched703Error as exc:
        raise SystemExit('703 matched refused: ' + str(exc))

    print(json.dumps({'models': len(records), 'report_dir': str(report_dir),
                      'training_performed': False}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

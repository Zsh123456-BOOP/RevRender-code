"""Score every model -- ours and the native baselines -- on one protocol.

588's table is internally consistent but its score sweep is not symmetric.
The native instance baselines were read on seven points,
``POINTREND_SCORE_GRID = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90)``, while
our own runs were read on whatever grid their runner declared, and 700 has just
measured that moving from a coarse grid to every distinct score is worth
+0.009 to +0.013 F1 on our side alone.  A ranking built across two different
sweeps is not a ranking of the models.

This gives every model the same one knob and turns it all the way: the mask
stays exactly where each model's own run put it, and the score threshold is
swept over every distinct value the model actually produced.  The matcher is
442's -- IoU-ordered greedy one-to-one at IoU 0.50 -- for everyone, because
that is the one 588's published table used.  As a check that nothing else
changed, each native baseline is also read on the old seven points and must
reproduce its published number exactly before the fine sweep is reported.

Sweeping is cheap because the matching is recomputed only when the kept set
changes: candidates enter in descending score order, so each row's outcome is
tabulated once per prefix and every threshold is then a lookup.

This script deliberately does NOT: train anything, run inference, or load a
checkpoint; re-derive the gold, the event grouping, the matcher or the 0.50
threshold; move any model's mask threshold, which stays where its own run put
it; claim that a dev-selected operating point is a test result; read,
transcribe or otherwise consume the frozen test split; or contact the network.
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
SCHEMA = 'hme-revision-one-protocol-table-v1'
MATCH_IOU = .50
EXPECTED_ROWS = 505
COARSE_GRID = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90)
# 588's published readings, from revision_full_metric_table_v3: the coarse
# sweep here must land on these exactly, or the two are not the same measurement.
PUBLISHED = {'maskdino': {'score': .5, 'f1': 0.5706371191135733,
                          'true_positive': 103, 'false_positive': 37},
             'pointrend': {'score': .7, 'f1': 0.5366876310272536,
                           'true_positive': 128, 'false_positive': 128}}


class Table701Error(RuntimeError):
    """A refusal that must stop the run rather than degrade its output."""


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def split_label(value: str) -> tuple[str, str]:
    label, separator, rest = value.partition('=')
    if not separator or not rest:
        raise Table701Error(f'expected label=source, got {value!r}')
    return label, rest


def native_provider(directory: Path):
    """Per line id, the archived instance masks and their scores, on demand.

    Loaded lazily because our own side unpacks to about a gigabyte of masks if
    every row is held at once, and nothing here needs two rows at a time.
    """
    index: dict[str, Path] = {}
    for path in sorted(directory.glob('*.npz')):
        line = str(np.load(path, allow_pickle=True)['sample_id']).split(':', 1)[1]
        if line in index:
            raise Table701Error(f'two archives for {line}')
        index[line] = path

    def provide(line: str):
        if line not in index:
            raise Table701Error(f'no archive for {line}')
        payload = np.load(index[line], allow_pickle=True)
        shape = tuple(int(v) for v in payload['shape'])
        masks = np.unpackbits(payload['masks_packed'], axis=1)
        masks = masks[:, :shape[1] * shape[2]].reshape(*shape).astype(bool)
        return [masks[i] for i in range(shape[0])], \
            np.asarray(payload['scores'], np.float64)

    return provide, set(index)


def packed_provider(path: Path, mask_threshold: float):
    """The same, from one of our runs' packed dev predictions."""
    with np.load(path, allow_pickle=False) as archive:
        thresholds = archive['mask_thresholds'].tolist()
        matches = [i for i, t in enumerate(thresholds)
                   if abs(t - mask_threshold) < 1e-9]
        if len(matches) != 1:
            raise Table701Error(f'{path}: mask {mask_threshold} not stored once')
        bits = np.array(archive['packed_masks'][matches[0]])
        scores = np.asarray(archive['scores'], np.float64)
        shape = tuple(int(v) for v in archive['mask_shape'])
        rows = {str(s).split(':', 1)[1]: i
                for i, s in enumerate(archive['sample_ids'].tolist())}
    height, width = shape[-2], shape[-1]

    def provide(line: str):
        if line not in rows:
            raise Table701Error(f'no packed row for {line}')
        row = rows[line]
        masks = np.unpackbits(bits[row], axis=-1, count=height * width)
        masks = masks.reshape(-1, height, width).astype(bool)
        return [masks[i] for i in range(len(masks))], scores[row]

    return provide, set(rows)


def prefix_outcomes(masks: Sequence[np.ndarray], scores: np.ndarray,
                    truth: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Matched and kept counts after admitting each score-ordered prefix.

    442 orders the pairs by IoU, not by score, so admitting a lower-scored
    candidate can move an earlier match; the prefix is therefore rematched from
    scratch rather than extended, which is what makes this equal to calling 442
    on the filtered list at every threshold.
    """
    order = sorted(range(len(masks)),
                   key=lambda i: (-float(scores[i]), i))
    areas = [int(masks[i].sum()) for i in order]
    pairs_by_rank: list[list[tuple[float, int, int]]] = []
    for rank, index in enumerate(order):
        row = []
        if areas[rank]:
            for t_index, t_mask in enumerate(truth):
                intersection = int(np.logical_and(masks[index], t_mask).sum())
                if not intersection:
                    continue
                union = areas[rank] + int(t_mask.sum()) - intersection
                iou = intersection / union
                if iou >= MATCH_IOU:
                    row.append((iou, rank, t_index))
        pairs_by_rank.append(row)

    matched = np.zeros(len(masks) + 1, np.int64)
    kept = np.zeros(len(masks) + 1, np.int64)
    pairs: list[tuple[float, int, int]] = []
    running_kept = 0
    for rank in range(len(masks)):
        pairs.extend(pairs_by_rank[rank])
        running_kept += 1 if areas[rank] else 0
        ordered = sorted(pairs, key=lambda item: (-item[0], item[1], item[2]))
        used_p: set[int] = set()
        used_t: set[int] = set()
        for _iou, p_index, t_index in ordered:
            if p_index in used_p or t_index in used_t:
                continue
            used_p.add(p_index)
            used_t.add(t_index)
        matched[rank + 1] = len(used_t)
        kept[rank + 1] = running_kept
    return matched, kept, np.array([float(scores[i]) for i in order], np.float64)


def totals_at(tables: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
              cut: float) -> tuple[int, int]:
    true_positive = kept_total = 0
    for matched, kept, ordered in tables:
        admitted = int(np.searchsorted(-ordered, -cut, side='right'))
        true_positive += int(matched[admitted])
        kept_total += int(kept[admitted])
    return true_positive, kept_total - true_positive


def reading(tables, cut: float, gold_total: int) -> dict[str, Any]:
    true_positive, false_positive = totals_at(tables, cut)
    precision = true_positive / (true_positive + false_positive) \
        if true_positive + false_positive else 0.
    recall = true_positive / gold_total if gold_total else 0.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    return {'score_threshold': cut, 'f1': f1, 'precision': precision,
            'recall': recall, 'true_positive': true_positive,
            'false_positive': false_positive,
            'false_negative': gold_total - true_positive}


def best_over(tables, cuts: Sequence[float], gold_total: int) -> dict[str, Any]:
    best = None
    for cut in sorted(set(float(c) for c in cuts)):
        candidate = reading(tables, cut, gold_total)
        if best is None or (candidate['f1'], -candidate['score_threshold']) > \
                (best['f1'], -best['score_threshold']):
            best = candidate
    if best is None:
        raise Table701Error('no score threshold to read')
    return best


def tabulate(provide, order: Sequence[str], truths: Sequence[Sequence[np.ndarray]]):
    """One prefix table per row, plus every distinct score the model produced."""
    tables, cuts = [], set()
    for line, truth in zip(order, truths):
        masks, scores = provide(line)
        if truth and masks and masks[0].shape != truth[0].shape:
            raise Table701Error(
                f'{line}: candidate {masks[0].shape} against gold {truth[0].shape}')
        table = prefix_outcomes(masks, scores, truth)
        tables.append(table)
        cuts.update(float(v) for v in table[2])
    return tables, sorted(cuts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--native', type=str, action='append', default=[],
                        help='label=directory of per-line instance archives')
    parser.add_argument('--ours', type=str, action='append', default=[],
                        help='label=dev_predictions.npz@mask, e.g. r3=/x/y.npz@0.3')
    parser.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args(argv)

    dataset = load_module(ROOT / 'scripts/514_revision_native_dataset.py', 'hme514_for701')
    train = load_module(ROOT / 'scripts/655_revision_three_part_train.py', 'hme655_for701')
    dual = load_module(ROOT / 'scripts/547_revision_dual_definition_scoring.py', 'hme547_for701')

    try:
        report_dir = args.report_dir.expanduser().resolve()
        dataset.safe_path(args.bundle)
        if report_dir.exists() or report_dir.is_symlink():
            raise Table701Error(f'output exists: {report_dir}')
        if not args.native and not args.ours:
            raise Table701Error('nothing to read')
        if dataset.digest(args.bundle / 'run_manifest.json') != train.BUNDLE_SHA:
            raise Table701Error('bundle manifest does not match')

        # The gold comes from the bundle rather than from 547's eight-module
        # preparation chain, whose pinned hashes no longer match this tree.
        # Nothing is taken on trust: if this were not 547's gold, the
        # seven-point reproduction of 588's published baselines below would
        # not land on the published numbers, and the run refuses.
        rows = sorted(dataset.load_bundle(args.bundle)['dev'],
                      key=lambda r: not r['is_positive'])
        if len(rows) != EXPECTED_ROWS:
            raise Table701Error(f'{len(rows)} rows, expected {EXPECTED_ROWS}')
        order = [str(row['sample_id']).split(':', 1)[1] for row in rows]
        stroke = [list(row['instances']) for row in rows]
        if sum(len(t) for t in stroke) != dual.EXPECTED_STROKE_GOLD:
            raise Table701Error('stroke gold is not the expected total')
        grouping = dual.read_grouping(dual.GROUPING_PATH)
        event = [dual.event_truths(components, grouping.get(line))
                 for line, components in zip(order, stroke)]
        if sum(len(t) for t in event) != dual.EXPECTED_EVENT_GOLD:
            raise Table701Error('event gold is not the expected total')

        records = []
        for kind, value in ([('native', v) for v in args.native]
                            + [('ours', v) for v in args.ours]):
            label, source = split_label(value)
            if kind == 'ours':
                path, _, mask = source.partition('@')
                provide, available = packed_provider(Path(path).expanduser().resolve(),
                                                     float(mask))
                family, mask_level = 'this work', float(mask)
            else:
                provide, available = native_provider(Path(source).expanduser().resolve())
                family, mask_level = 'native instance baseline', .5
            missing = set(order) - available
            if missing:
                raise Table701Error(f'{label}: {len(missing)} rows have no archive')
            tables, cuts = tabulate(provide, order, stroke)
            fine = best_over(tables, cuts, dual.EXPECTED_STROKE_GOLD)
            coarse = best_over(tables, COARSE_GRID, dual.EXPECTED_STROKE_GOLD)
            if label in PUBLISHED:
                expected = PUBLISHED[label]
                if abs(coarse['f1'] - expected['f1']) > 1e-9 \
                        or coarse['true_positive'] != expected['true_positive'] \
                        or coarse['false_positive'] != expected['false_positive'] \
                        or abs(coarse['score_threshold'] - expected['score']) > 1e-9:
                    raise Table701Error(
                        f'{label}: the seven-point sweep does not reproduce 588 '
                        f'({coarse} against {expected}); the two are not the '
                        'same measurement and the fine sweep cannot be compared')
            event_tables, event_cuts = tabulate(provide, order, event)
            records.append({
                'model': label, 'family': family, 'mask': mask_level,
                'stroke_level': {'fine': fine, 'coarse_seven_point': coarse,
                                 'distinct_scores': len(cuts),
                                 'coarse_understated_f1_by': fine['f1'] - coarse['f1']},
                'event_level': {'fine': best_over(event_tables, event_cuts,
                                                  dual.EXPECTED_EVENT_GOLD),
                                'coarse_seven_point': best_over(
                                    event_tables, COARSE_GRID, dual.EXPECTED_EVENT_GOLD)}})

        report = {'schema_version': SCHEMA, 'match_iou': MATCH_IOU,
                  'rows': len(order), 'matcher': '442 IoU-ordered greedy one-to-one',
                  'protocol': "each model's own mask; every distinct score it produced",
                  'gold_read_from': 'the native bundle, proven equal to 547\'s by the '
                                    'seven-point reproduction of 588',
                  'operating_point_selected_on': 'the validation split, for every model alike',
                  'training_performed': False, 'inference_performed': False,
                  'frozen_test_touched': False, 'results': records}
        report_dir.mkdir(parents=True)
        (report_dir / 'one_protocol_table.json').write_text(
            json.dumps(report, indent=1, sort_keys=True, ensure_ascii=False) + '\n')
        (report_dir / 'run_manifest.json').write_text(json.dumps({
            'script_sha256': dual.digest_file(Path(__file__).resolve()),
            'dual_script_sha256': dual.digest_file(
                ROOT / 'scripts/547_revision_dual_definition_scoring.py'),
            'python': platform.python_version(), 'numpy': np.__version__,
            'create_only': True, 'training_performed': False,
            'frozen_test_touched': False}, indent=1, sort_keys=True) + '\n')

        lines = [f"{'model':<22}{'F1':>8}{'P':>8}{'R':>8}{'TP':>5}{'FP':>5}"
                 f"{'cut':>8}{'7ptF1':>8}{'delta':>8}{'cuts':>7}{'eventF1':>9}"]
        for record in sorted(records, key=lambda r: -r['stroke_level']['fine']['f1']):
            fine = record['stroke_level']['fine']
            coarse = record['stroke_level']['coarse_seven_point']
            lines.append(
                f"{record['model']:<22}{fine['f1']:>8.4f}{fine['precision']:>8.4f}"
                f"{fine['recall']:>8.4f}{fine['true_positive']:>5}"
                f"{fine['false_positive']:>5}{fine['score_threshold']:>8.4f}"
                f"{coarse['f1']:>8.4f}"
                f"{record['stroke_level']['coarse_understated_f1_by']:>8.4f}"
                f"{record['stroke_level']['distinct_scores']:>7}"
                f"{record['event_level']['fine']['f1']:>9.4f}")
        table = '\n'.join(lines) + '\n'
        (report_dir / 'table.txt').write_text(table)
        print(table)
    except Table701Error as exc:
        raise SystemExit('701 table refused: ' + str(exc))

    print(json.dumps({'models': len(records), 'report_dir': str(report_dir),
                      'training_performed': False}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

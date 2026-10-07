"""Give every model the same two knobs -- score and mask NMS -- and read them all.

701 gave every model one knob and found our best arm ahead of MaskDINO on
stroke-level F1 by 0.0099 while losing on precision (0.6029 against 0.7211)
and on the event definition (0.3896 against 0.4698).  Both losses point the
same way: our decoder emits several candidates per revision where MaskDINO
emits one.  682 measured the size of it indirectly -- suppression was worth
+0.0195 F1 on the geometric-mean arm at mask 0.5 -- but only for our runs and
only at one mask level.

MaskDINO needs no suppression by construction; Mask R-CNN with PointRend
applies it natively.  So the honest comparison is not "should we be allowed
NMS" but "every model gets both knobs, tuned on the same split, and we report
where each one lands".  A model that gains nothing from suppression simply
selects the off setting, which is on the grid.

Suppression is a prefix computation like the score sweep: NMS walks candidates
in descending score and keeps one only against those already kept, so a lower
score threshold can only extend the kept sequence, never rewrite it.  The
per-threshold tabulation from 701 therefore still applies, once per NMS level.

This script deliberately does NOT: train anything, run inference, or load a
checkpoint; re-derive the gold, the matcher or the 0.50 threshold; move any
model's mask threshold; merge, dilate or otherwise edit a candidate mask, since
suppression only ever discards; claim a dev-tuned operating point is a test
result; read, transcribe or otherwise consume the frozen test split; or contact
the network.
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
SCHEMA = 'hme-revision-suppression-protocol-v1'
BASE = '701_revision_one_protocol_table.py'
# 1.0 is suppression off: no pair can exceed an IoU of one, so nothing is cut.
NMS_LEVELS = (1., .9, .7, .5, .3)


class Suppress702Error(RuntimeError):
    """A refusal that must stop the run rather than degrade its output."""


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TABLE = load_module(ROOT / 'scripts' / BASE, 'hme701_for702')


def suppressed(masks: Sequence[np.ndarray], scores: np.ndarray,
               level: float) -> list[int]:
    """Indices surviving mask NMS at this level, in descending score order.

    Empty candidates are kept in the sequence so the caller's bookkeeping still
    counts them exactly as 442 does; they suppress nothing, having no area.
    """
    order = sorted(range(len(masks)), key=lambda i: (-float(scores[i]), i))
    if level >= 1.:
        return order
    kept: list[int] = []
    areas = {i: int(masks[i].sum()) for i in order}
    for index in order:
        if areas[index]:
            duplicate = False
            for other in kept:
                if not areas[other]:
                    continue
                intersection = int(np.logical_and(masks[index], masks[other]).sum())
                if not intersection:
                    continue
                union = areas[index] + areas[other] - intersection
                if intersection / union > level:
                    duplicate = True
                    break
            if duplicate:
                continue
        kept.append(index)
    return kept


def tabulate_at(provide, order: Sequence[str],
                truths: Sequence[Sequence[np.ndarray]], level: float):
    tables, cuts = [], set()
    for line, truth in zip(order, truths):
        masks, scores = provide(line)
        if truth and masks and masks[0].shape != truth[0].shape:
            raise Suppress702Error(
                f'{line}: candidate {masks[0].shape} against gold {truth[0].shape}')
        keep = suppressed(masks, scores, level)
        table = TABLE.prefix_outcomes([masks[i] for i in keep],
                                      np.array([float(scores[i]) for i in keep]),
                                      truth)
        tables.append(table)
        cuts.update(float(v) for v in table[2])
    return tables, sorted(cuts)


def sweep(provide, order, truths, gold_total: int) -> dict[str, Any]:
    """Best reading per NMS level, and the best over the whole two-knob grid."""
    per_level, best = {}, None
    for level in NMS_LEVELS:
        tables, cuts = tabulate_at(provide, order, truths, level)
        reading = TABLE.best_over(tables, cuts, gold_total)
        reading['nms'] = level
        per_level[f'{level:.2f}'] = reading
        if best is None or (reading['f1'], reading['nms']) > (best['f1'], best['nms']):
            best = dict(reading)
    return {'best': best, 'per_nms_level': per_level,
            'suppression_bought': best['f1'] - per_level['1.00']['f1']}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--native', type=str, action='append', default=[])
    parser.add_argument('--ours', type=str, action='append', default=[])
    parser.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args(argv)

    dataset = load_module(ROOT / 'scripts/514_revision_native_dataset.py', 'hme514_for702')
    train = load_module(ROOT / 'scripts/655_revision_three_part_train.py', 'hme655_for702')
    dual = load_module(ROOT / 'scripts/547_revision_dual_definition_scoring.py', 'hme547_for702')

    try:
        report_dir = args.report_dir.expanduser().resolve()
        dataset.safe_path(args.bundle)
        if report_dir.exists() or report_dir.is_symlink():
            raise Suppress702Error(f'output exists: {report_dir}')
        if not args.native and not args.ours:
            raise Suppress702Error('nothing to read')
        if dataset.digest(args.bundle / 'run_manifest.json') != train.BUNDLE_SHA:
            raise Suppress702Error('bundle manifest does not match')

        rows = sorted(dataset.load_bundle(args.bundle)['dev'],
                      key=lambda r: not r['is_positive'])
        if len(rows) != TABLE.EXPECTED_ROWS:
            raise Suppress702Error(f'{len(rows)} rows')
        order = [str(row['sample_id']).split(':', 1)[1] for row in rows]
        stroke = [list(row['instances']) for row in rows]
        if sum(len(t) for t in stroke) != dual.EXPECTED_STROKE_GOLD:
            raise Suppress702Error('stroke gold is not the expected total')
        grouping = dual.read_grouping(dual.GROUPING_PATH)
        event = [dual.event_truths(components, grouping.get(line))
                 for line, components in zip(order, stroke)]
        if sum(len(t) for t in event) != dual.EXPECTED_EVENT_GOLD:
            raise Suppress702Error('event gold is not the expected total')

        records = []
        for kind, value in ([('native', v) for v in args.native]
                            + [('ours', v) for v in args.ours]):
            label, source = TABLE.split_label(value)
            if kind == 'ours':
                path, _, mask = source.partition('@')
                provide, available = TABLE.packed_provider(
                    Path(path).expanduser().resolve(), float(mask))
                family, mask_level = 'this work', float(mask)
            else:
                provide, available = TABLE.native_provider(
                    Path(source).expanduser().resolve())
                family, mask_level = 'native instance baseline', .5
            if set(order) - available:
                raise Suppress702Error(f'{label}: rows without an archive')

            stroke_sweep = sweep(provide, order, stroke, dual.EXPECTED_STROKE_GOLD)
            if label in TABLE.PUBLISHED:
                # With suppression off this must still be 588's measurement.
                expected = TABLE.PUBLISHED[label]
                tables, _ = tabulate_at(provide, order, stroke, 1.)
                coarse = TABLE.best_over(tables, TABLE.COARSE_GRID,
                                         dual.EXPECTED_STROKE_GOLD)
                if abs(coarse['f1'] - expected['f1']) > 1e-9 \
                        or coarse['true_positive'] != expected['true_positive'] \
                        or coarse['false_positive'] != expected['false_positive']:
                    raise Suppress702Error(
                        f'{label}: with suppression off the seven-point sweep no '
                        f'longer reproduces 588 ({coarse} against {expected})')
            records.append({
                'model': label, 'family': family, 'mask': mask_level,
                'stroke_level': stroke_sweep,
                'event_level': sweep(provide, order, event, dual.EXPECTED_EVENT_GOLD)})

        report = {'schema_version': SCHEMA, 'match_iou': TABLE.MATCH_IOU,
                  'rows': len(order), 'nms_levels': list(NMS_LEVELS),
                  'matcher': '442 IoU-ordered greedy one-to-one',
                  'protocol': "each model's own mask; every distinct score; mask NMS "
                              'from the same grid for every model, 1.0 meaning off',
                  'both_knobs_selected_on': 'the validation split, for every model alike',
                  'training_performed': False, 'inference_performed': False,
                  'frozen_test_touched': False, 'results': records}
        report_dir.mkdir(parents=True)
        (report_dir / 'suppression_protocol.json').write_text(
            json.dumps(report, indent=1, sort_keys=True, ensure_ascii=False) + '\n')
        (report_dir / 'run_manifest.json').write_text(json.dumps({
            'script_sha256': dual.digest_file(Path(__file__).resolve()),
            'table_script_sha256': dual.digest_file(ROOT / 'scripts' / BASE),
            'python': platform.python_version(), 'numpy': np.__version__,
            'create_only': True, 'training_performed': False,
            'frozen_test_touched': False}, indent=1, sort_keys=True) + '\n')

        lines = [f"{'model':<22}{'F1':>8}{'P':>8}{'R':>8}{'TP':>5}{'FP':>5}"
                 f"{'cut':>8}{'nms':>6}{'noNMS':>8}{'gain':>8}{'eventF1':>9}{'evNMS':>7}"]
        for record in sorted(records, key=lambda r: -r['stroke_level']['best']['f1']):
            best = record['stroke_level']['best']
            off = record['stroke_level']['per_nms_level']['1.00']
            lines.append(
                f"{record['model']:<22}{best['f1']:>8.4f}{best['precision']:>8.4f}"
                f"{best['recall']:>8.4f}{best['true_positive']:>5}"
                f"{best['false_positive']:>5}{best['score_threshold']:>8.4f}"
                f"{best['nms']:>6.2f}{off['f1']:>8.4f}"
                f"{record['stroke_level']['suppression_bought']:>8.4f}"
                f"{record['event_level']['best']['f1']:>9.4f}"
                f"{record['event_level']['best']['nms']:>7.2f}")
        table = '\n'.join(lines) + '\n'
        (report_dir / 'table.txt').write_text(table)
        print(table)
    except Suppress702Error as exc:
        raise SystemExit('702 suppression refused: ' + str(exc))

    print(json.dumps({'models': len(records), 'report_dir': str(report_dir),
                      'training_performed': False}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

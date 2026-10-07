#!/usr/bin/env python3
"""Streaming dev-only common evaluation for native instance baselines.

Gold is already corrected, then split into 8-connected components by 514.
Predicted instances are NOT relabelled into connected components. Matching is
the IoU-descending greedy rule of 442. Mask binarization is native 0.5; only
the instance score is calibrated, on all dev rows including adjudicated clean.
"""
from __future__ import annotations

SCORE_THRESHOLDS = tuple(round(i / 20, 2) for i in range(1, 20)) + (0.99, 1.000001)


def prepare_record(record, prediction, thresholds=None):
    import numpy as np

    thresholds = tuple(SCORE_THRESHOLDS if thresholds is None else thresholds)
    mask = np.asarray(record['mask'], dtype=bool)
    gold = np.asarray(record['instances'], dtype=bool).reshape(-1, *mask.shape)
    predicted = np.asarray(prediction['masks'], dtype=bool).reshape(-1, *mask.shape)
    scores = np.asarray(prediction['scores'], dtype=float)
    if prediction.get('sample_id', record['sample_id']) != record['sample_id']:
        raise ValueError('prediction/sample identity mismatch')
    if scores.shape != (len(predicted),) or not np.isfinite(scores).all():
        raise ValueError('invalid instance scores')
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError('instance scores must be probabilities')
    areas = predicted.sum(axis=(1, 2))
    keep = areas > 0
    predicted, scores, areas = predicted[keep], scores[keep], areas[keep]
    intersections = np.zeros((len(predicted), len(gold)), dtype=np.int64)
    for index, truth in enumerate(gold):
        intersections[:, index] = predicted[:, truth].sum(axis=1)
    unions = areas[:, None] + gold.sum(axis=(1, 2))[None, :] - intersections
    iou = np.divide(intersections, unions, out=np.zeros_like(unions, dtype=float),
                    where=unions > 0)
    points = []
    for threshold in thresholds:
        indices = np.flatnonzero(scores >= threshold)
        pairs = [(float(iou[p, t]), int(p), int(t))
                 for p in indices for t in range(len(gold)) if iou[p, t] >= 0.5]
        pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
        used_p, used_t = set(), set()
        for _, p, t in pairs:
            if p not in used_p and t not in used_t:
                used_p.add(p)
                used_t.add(t)
        union = (predicted[indices].any(axis=0) if len(indices)
                 else np.zeros(mask.shape, dtype=bool))
        intersection = int((union & mask).sum())
        points.append({'threshold': float(threshold), 'tp': len(used_p),
                       'fp': len(indices) - len(used_p), 'fn': len(gold) - len(used_t),
                       'pixel_tp': intersection, 'pixel_fp': int(union.sum()) - intersection,
                       'pixel_fn': int(mask.sum()) - intersection,
                       'clean_fired': int(not record['is_positive'] and union.sum() >= 8),
                       'positive_fired': int(record['is_positive'] and union.sum() >= 8)})
    return {'sample_id': record['sample_id'], 'is_positive': bool(record['is_positive']),
            'gold_count': len(gold), 'operating_points': points}


def select_operating_point(prepared, thresholds=None, max_clean_fp=0.15):
    thresholds = tuple(SCORE_THRESHOLDS if thresholds is None else thresholds)
    if not prepared or len({x['sample_id'] for x in prepared}) != len(prepared):
        raise ValueError('evaluation needs nonempty unique dev rows')
    clean_count = sum(not x['is_positive'] for x in prepared)
    positive_count = len(prepared) - clean_count
    if not clean_count or not positive_count:
        raise ValueError('evaluation must include both positive and clean rows')
    points = []
    for index, threshold in enumerate(thresholds):
        rows = [item['operating_points'][index] for item in prepared]
        if any(row['threshold'] != threshold for row in rows):
            raise ValueError('threshold grids disagree')
        sums = {key: sum(row[key] for row in rows) for key in
                ('tp', 'fp', 'fn', 'pixel_tp', 'pixel_fp', 'pixel_fn',
                 'clean_fired', 'positive_fired')}
        tp, fp, fn = (sums[key] for key in ('tp', 'fp', 'fn'))
        pt, pf, pn = (sums[key] for key in ('pixel_tp', 'pixel_fp', 'pixel_fn'))
        clean_fp = sums['clean_fired'] / clean_count
        points.append(dict(sums, threshold=threshold,
                           component_f1=2 * tp / max(1, 2 * tp + fp + fn),
                           precision=tp / max(1, tp + fp), recall=tp / max(1, tp + fn),
                           pixel_f1=2 * pt / max(1, 2 * pt + pf + pn),
                           pixel_iou=pt / max(1, pt + pf + pn), clean_fp=clean_fp,
                           row_recall=sums['positive_fired'] / positive_count,
                           eligible=clean_fp <= max_clean_fp))
    eligible = [point for point in points if point['eligible']]
    if not eligible:
        raise ValueError('no operating point satisfies clean FP bound')
    best = max(eligible, key=lambda x: (x['component_f1'], x['pixel_f1'], x['threshold']))
    return dict(best, rows=len(prepared), positive_rows=positive_count, clean_rows=clean_count,
                gold_components=sum(item['gold_count'] for item in prepared),
                operating_points=points,
                protocol='all-dev; native instances; mask=0.5; IoU=0.5; clean union>=8px')

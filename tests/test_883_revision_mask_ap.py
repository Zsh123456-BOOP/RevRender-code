import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ap883_test', ROOT/'scripts/883_revision_mask_ap.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def test_perfect_ranking_scores_one_and_a_leading_false_positive_costs():
    assert M.average_precision([(.9, True), (.8, True)], 2) == pytest.approx(1.)
    assert M.average_precision([(.9, False), (.8, True), (.7, True)], 2) < 1.
    assert M.average_precision([(.9, True)], 2) == pytest.approx(51/101)      # recall reaches only 0.5
    assert M.average_precision([], 3) == 0.


def test_matching_is_greedy_by_score_and_one_to_one():
    iou = np.array([[.8, .0], [.7, .6], [.55, .0]])
    out = M.match_line(iou, [.9, .8, .7], .5)
    assert out == [(.9, True), (.8, True), (.7, False)]                       # the third finds gold 0 taken
    assert M.match_line(iou, [.9, .8, .7], .75) == [(.9, True), (.8, False), (.7, False)]
    assert M.IOUS[0] == .5 and M.IOUS[-1] == .95 and len(M.IOUS) == 10

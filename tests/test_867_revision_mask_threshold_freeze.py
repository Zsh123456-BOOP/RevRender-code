import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('freeze867_test', ROOT/'scripts/867_revision_mask_threshold_freeze.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def test_pick_best_f1_then_fewer_false_positives_then_lower_threshold():
    r = lambda t, f1, fp: {'mask_threshold': t, 'f1': f1, 'false_positive': fp}
    assert M.pick([r(.3, .68, 87), r(.4, .686, 85), r(.5, .66, 93)])['mask_threshold'] == .4
    assert M.pick([r(.3, .70, 80), r(.4, .70, 70)])['mask_threshold'] == .4
    assert M.pick([r(.3, .70, 70), r(.5, .70, 70)])['mask_threshold'] == .3

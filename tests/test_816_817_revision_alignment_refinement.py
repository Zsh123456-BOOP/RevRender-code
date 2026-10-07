import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def architecture(monkeypatch, variant):
    pytest.importorskip('torch')
    monkeypatch.setenv('HME_ALIGN_REFINE', variant)
    path = ROOT/'scripts/816_revision_alignment_refinement.py'
    spec = importlib.util.spec_from_file_location('test816_'+variant, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quality_label_matches_delivered_threshold(monkeypatch):
    torch = pytest.importorskip('torch')
    m = architecture(monkeypatch, 'score03')
    logits = torch.logit(torch.tensor([[[[.4, .6]]]]))
    truth = [[torch.tensor([[1., 0.]])]]
    torch.testing.assert_close(m.SCORE.overlap_targets(logits, truth), torch.tensor([[.5]]))
    torch.testing.assert_close(m._original_overlap(logits, truth), torch.tensor([[0.]]))


@pytest.mark.parametrize('contextual', [False, True])
def test_point_extension_is_initially_identity_and_preserves_rng(monkeypatch, contextual):
    torch = pytest.importorskip('torch')
    m = architecture(monkeypatch, 'point_context' if contextual else 'point_ms')
    original = m.TRANSFER.RoiPointDraw()
    extended = m.TRANSFER.RoiPointDraw()
    extended.load_state_dict(original.state_dict())
    before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    m.install_point_residual(extended, contextual)
    assert torch.equal(before, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all()))
    fine = torch.randn(1, 32, 16, 32, requires_grad=True)
    coarse = torch.randn(1, 2, 4, 8, requires_grad=True)
    boxes = torch.tensor([[[.4, .5, .5, .7], [.7, .4, .4, .5]]])
    rng = torch.get_rng_state()
    expected, points = original(coarse, fine, boxes, (16, 32))
    torch.set_rng_state(rng)
    actual, extended_points = extended(coarse, fine, boxes, (16, 32))
    assert torch.equal(expected, actual)
    assert torch.equal(points['roi_points'][1], extended_points['roi_points'][1])
    extended_points['roi_points'][1].sum().backward()
    assert extended.alignment_residual.output[-1].weight.grad.abs().sum() > 0
    assert extended._alignment_memory is None


def test_context_output_reads_other_roi_points(monkeypatch):
    torch = pytest.importorskip('torch')
    m = architecture(monkeypatch, 'point_context')
    head = m.PointResidual(True)
    with torch.no_grad():
        head.output[-1].weight.fill_(.1)
    x = torch.randn(1, 2, 4, 99, requires_grad=True)
    memory = torch.randn(1, 2, 32, 99, requires_grad=True)
    out = head(x, memory)
    out.sum().backward()
    assert memory.grad is not None and memory.grad.abs().sum() > 0
    assert torch.isfinite(memory.grad).all()


def test_multiscale_and_context_share_identical_common_initialization(monkeypatch):
    torch = pytest.importorskip('torch')
    m = architecture(monkeypatch, 'point_context')
    torch.manual_seed(m.RESIDUAL_SEED)
    multi = m.PointResidual(False).state_dict()
    torch.manual_seed(m.RESIDUAL_SEED)
    context = m.PointResidual(True).state_dict()
    assert all(torch.equal(value, context[key]) for key, value in multi.items())


def test_inference_teacher_restores_mode_and_detaches(monkeypatch):
    torch = pytest.importorskip('torch')
    m = architecture(monkeypatch, 'render03')
    class Draw(torch.nn.Module):
        def forward(self, coarse, fine, boxes, size):
            assert not self.training
            return coarse+fine.sum()*0, {}
    draw = Draw().train()
    coarse = torch.randn(1, 1, 2, 2, requires_grad=True)
    rendered = m.inference_render(draw, coarse, coarse, torch.zeros(1, 1, 4), (2, 2))
    assert draw.training and not rendered.requires_grad


def test_inference_teacher_restores_mode_on_failure(monkeypatch):
    torch = pytest.importorskip('torch')
    m = architecture(monkeypatch, 'render03')
    class Draw(torch.nn.Module):
        def forward(self, *args):
            raise RuntimeError('intentional')
    draw = Draw().train()
    x = torch.zeros(1)
    with pytest.raises(RuntimeError, match='intentional'):
        m.inference_render(draw, x, x, x, (2, 2))
    assert draw.training

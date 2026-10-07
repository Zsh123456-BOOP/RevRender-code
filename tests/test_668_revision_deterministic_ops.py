import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
spec = importlib.util.spec_from_file_location('det668', Path(__file__).resolve().parents[1] / 'scripts/668_revision_deterministic_ops.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture(autouse=True)
def strict():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


@pytest.mark.parametrize('mode', ['bilinear', 'nearest'])
def test_grid_matches_native_forward_and_gradient_at_borders_duplicates(mode):
    torch.manual_seed(42)
    x = torch.randn(2, 3, 5, 7, dtype=torch.double, requires_grad=True)
    coords = torch.tensor([[[-.2, .7], [0, 0], [.5, .5], [.5, .5], [1, 1], [1.2, .4]]], dtype=torch.double).repeat(2, 1, 1)
    a = m.sample_points(x, coords, mode)
    b = m.NATIVE_GRID(x, coords[:, None] * 2 - 1, mode=mode, padding_mode='border', align_corners=False)[:, :, 0]
    assert torch.equal(a, b)
    gradient = torch.randn_like(a)
    ga = torch.autograd.grad(a, x, gradient)[0]
    gb = torch.autograd.grad(b, x, gradient)[0]
    torch.testing.assert_close(ga, gb, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('source,target', [((3, 5), (6, 10)), ((5, 7), (3, 4)), ((1, 5), (4, 13)), ((5, 7), (5, 7))])
def test_resize_matches_forward_and_gradient(source, target):
    x = torch.randn(2, 3, *source, dtype=torch.double, requires_grad=True)
    a = m.BilinearResize.apply(x, target)
    b = m.NATIVE_RESIZE(x, size=target, mode='bilinear', align_corners=False)
    assert torch.equal(a, b)
    g = torch.randn_like(a)
    torch.testing.assert_close(torch.autograd.grad(a, x, g)[0], torch.autograd.grad(b, x, g)[0], rtol=1e-12, atol=1e-12)


def test_pool_gradient_and_invalid_size():
    x = torch.randn(2, 3, 8, 12, dtype=torch.double, requires_grad=True)
    a, b = m.DivisiblePool.apply(x, (2, 3)), m.NATIVE_POOL(x, (2, 3))
    assert torch.equal(a, b)
    g = torch.randn_like(a)
    assert torch.equal(torch.autograd.grad(a, x, g)[0], torch.autograd.grad(b, x, g)[0])
    with pytest.raises(ValueError):
        m.DivisiblePool.apply(x, (3, 3))


def test_ties_have_a_fixed_order_without_reordering_distinct_values():
    x = torch.tensor([[.5, .7, .7, .2, .7]])
    assert m.stable_topk(x, 4).tolist() == [[1, 2, 4, 0]]


def test_refuses_learned_coordinates_and_non_strict_backward():
    x = torch.randn(1, 2, 3, 4, requires_grad=True)
    coords = torch.rand(1, 4, 2, requires_grad=True)
    with pytest.raises(ValueError):
        m.sample_points(x, coords)
    y = m.sample_points(x, coords.detach())
    torch.use_deterministic_algorithms(False)
    with pytest.raises(RuntimeError):
        y.sum().backward()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('operation', ['grid', 'resize', 'pool'])
def test_cuda_adjoint_repeats_exactly_and_agrees_with_cpu(operation):
    torch.manual_seed(123)
    x = torch.randn(2, 4, 8, 12, dtype=torch.double, requires_grad=True)
    coords = torch.rand(2, 17, 2, dtype=torch.double)
    coords[:, 5:8] = coords[:, 1:2]
    def apply(value, native):
        if operation == 'grid':
            c = coords.to(value.device)
            return (m.NATIVE_GRID(value, c[:, None] * 2 - 1, padding_mode='border', align_corners=False)[:, :, 0]
                    if native else m.sample_points(value, c))
        if operation == 'resize':
            return (m.NATIVE_RESIZE(value, size=(16, 24), mode='bilinear', align_corners=False)
                    if native else m.BilinearResize.apply(value, (16, 24)))
        return m.NATIVE_POOL(value, (2, 3)) if native else m.DivisiblePool.apply(value, (2, 3))
    reference = apply(x, True)
    gradient = torch.randn_like(reference)
    expected = torch.autograd.grad(reference, x, gradient)[0]
    actual = []
    for _ in range(2):
        value = x.detach().cuda().requires_grad_(True)
        y = apply(value, False)
        with torch.no_grad():
            assert torch.equal(y, apply(value, True))
        actual.append(torch.autograd.grad(y, value, gradient.cuda())[0].cpu())
    assert torch.equal(*actual)
    torch.testing.assert_close(actual[0], expected, rtol=1e-11, atol=1e-11)

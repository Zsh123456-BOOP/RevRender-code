import importlib.util
import os
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('segmentation_models_pytorch')
ROOT = Path(__file__).resolve().parents[1]


def module(mode):
    os.environ['HME_TUNE'] = mode
    spec = importlib.util.spec_from_file_location('tune880_test_'+mode, ROOT/'scripts/880_revision_tuning.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_rates_per_mode():
    m = module('bb01')
    assert m.rate_of('see.encoder.layer2.0.conv1.weight') == pytest.approx(.1)
    assert m.rate_of('find.layers.0.ffn.0.weight') == 1.
    assert m.rate_of('see.encoder.layer1.0.conv1.weight', {'lr_scale': .5, 'backbone_lr': .1}) == pytest.approx(.05)
    assert module('control').KNOBS == {}
    with pytest.raises(ValueError):
        module('nonsense')


def test_tuned_adamw_scales_each_group_and_restores_the_schedule():
    m = module('bb01')
    fast, slow, still = (torch.nn.Parameter(torch.zeros(3)) for _ in range(3))
    fast._tune880_rate, slow._tune880_rate, still._tune880_rate = 1., .1, 0.
    optimizer = m.tuned_adamw()([fast, slow, still], lr=.01, weight_decay=.5)
    for p in (fast, slow, still):
        p.grad = torch.ones(3)
    for g in optimizer.param_groups:
        g['lr'] = .01                                   # what 675's schedule writes each step
    optimizer.step()
    assert torch.allclose(fast, torch.full((3,), -.01), atol=1e-6)
    assert torch.allclose(slow, torch.full((3,), -.001), atol=1e-6)
    assert torch.equal(still, torch.zeros(3))
    assert [g['lr'] for g in optimizer.param_groups] == [.01, .01, .01]


def test_frozen_batch_norms_stay_in_eval_mode():
    m = module('frozenbn')
    net = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 1), torch.nn.BatchNorm2d(2))
    assert m.freeze_batch_norms(net) == 1
    net.train()
    assert net.training and not net[1].training
    before = net[1].running_mean.clone()
    net(torch.randn(4, 1, 5, 5))
    assert torch.equal(before, net[1].running_mean)


def test_dropout_hooks_are_identity_in_eval_and_at_zero():
    m = module('drop01')

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attention = torch.nn.MultiheadAttention(8, 2, batch_first=True)
            self.cross_attention = torch.nn.Linear(8, 8)
            self.ffn = torch.nn.Linear(8, 8)
    layer = Layer()
    x = torch.randn(2, 5, 8)
    reference = layer.ffn(x)
    m.add_dropout([layer], .5)
    layer.eval()
    assert torch.equal(layer.ffn(x), reference)
    layer.train()
    assert not torch.equal(layer.ffn(x), reference)


def test_nofine_feeds_the_pixel_decoder_a_blank_image():
    m = module('nofine')
    assert m.KNOBS == {'ablation': 'no_fullres_detail'}

    class Decoder(torch.nn.Module):
        def forward(self, quarter, image):
            return quarter.sum()+image.sum()
    decoder = m.remove_image_skips(Decoder())
    assert decoder(torch.ones(1, 1, 2, 2), torch.ones(1, 1, 4, 4)).item() == 4.


def test_nocontext_builds_816_point_ms_and_nothing_else_changes():
    m = module('nocontext')
    assert m.ABLATIONS == {'no_context_attention'} and m.ARCH.ARCH.VARIANT == 'point_ms'
    assert module('control').ARCH.ARCH.VARIANT == 'point_context'
    both = module('nofine_nodense')
    assert both.ABLATIONS == {'no_fullres_detail', 'no_pixel_supervision'} and both.ARCH.ARCH.VARIANT == 'point_context'


def test_nodense_silences_only_the_detail_maps_dense_term():
    m = module('nodense')

    class Full:
        def matched_mask_loss(self, prediction, target, kind, weight):
            return (prediction-target).abs().sum()
        other = 'kept'

    class Model:
        cache = {}
    model = Model()
    dense, other = torch.ones(2, 3, requires_grad=True), torch.ones(2, 3, requires_grad=True)
    model.cache['dense'] = dense
    silent = m.WithoutPixelSupervision(Full(), model)
    zero = silent.matched_mask_loss(dense, torch.zeros(2, 3), 'dice', 0.)
    kept = silent.matched_mask_loss(other, torch.zeros(2, 3), 'dice', 0.)
    assert zero.item() == 0. and kept.item() == 6. and silent.silenced == 1 and silent.other == 'kept'
    (zero+kept).backward()
    assert torch.equal(dense.grad, torch.zeros(2, 3)) and torch.equal(other.grad, torch.ones(2, 3))

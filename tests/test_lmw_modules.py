import warnings

import torch

from ultralytics.nn.tasks import parse_model
from ultralytics.nn.modules import DWRBottleneck, LKABottleneck, LKCA, MSDP


def _assert_finite_backward(module, x):
    y = module(x)
    loss = y.square().mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for parameter in module.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    return y


def test_lkabottleneck_preserves_shape_dtype_and_backward():
    module = LKABottleneck(64)
    x = torch.randn(2, 64, 16, 16, requires_grad=True)

    y = _assert_finite_backward(module, x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert torch.isfinite(y).all()


def test_lkca_output_shape_and_backward():
    module = LKCA(64, 128, n=2)
    x = torch.randn(2, 64, 16, 16, requires_grad=True)

    y = _assert_finite_backward(module, x)

    assert y.shape == (2, 128, 16, 16)
    assert torch.isfinite(y).all()


def test_dwrbottleneck_preserves_shape_and_backward():
    module = DWRBottleneck(128)
    x = torch.randn(2, 128, 16, 16, requires_grad=True)

    y = _assert_finite_backward(module, x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_msdp_output_shape_and_backward():
    module = MSDP(128, 256, n=2)
    x = torch.randn(2, 128, 16, 16, requires_grad=True)

    y = _assert_finite_backward(module, x)

    assert y.shape == (2, 256, 16, 16)
    assert torch.isfinite(y).all()


@torch.no_grad()
def test_lkca_and_msdp_trace_smoke():
    lkca = LKCA(64, 128, n=1).eval()
    msdp = MSDP(128, 256, n=1).eval()

    x1 = torch.randn(1, 64, 16, 16)
    x2 = torch.randn(1, 128, 16, 16)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        traced_lkca = torch.jit.trace(lkca, x1)
        traced_msdp = torch.jit.trace(msdp, x2)

    y1 = traced_lkca(x1)
    y2 = traced_msdp(x2)

    assert y1.shape == (1, 128, 16, 16)
    assert y2.shape == (1, 256, 16, 16)
    assert torch.isfinite(y1).all()
    assert torch.isfinite(y2).all()


def test_parse_model_registers_lkca_and_msdp_repeat_modules():
    model_dict = {
        "nc": 80,
        "depth_multiple": 1.0,
        "width_multiple": 1.0,
        "backbone": [
            [-1, 2, "LKCA", [128, True, 0.5]],
            [-1, 2, "MSDP", [256, True, 0.5]],
        ],
        "head": [],
    }

    model, save = parse_model(model_dict, ch=64, verbose=False)

    assert save == []
    assert isinstance(model[0], LKCA)
    assert isinstance(model[1], MSDP)
    assert len(model[0].m) == 2
    assert len(model[1].m) == 2

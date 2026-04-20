"""Smoke tests for the YOLO26 LMW model YAML."""

import torch

from ultralytics import YOLO
from ultralytics.nn.modules.block import C3k2
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.lmw import LKCA, MSDP

CFG = "ultralytics/cfg/models/26/yolo26-lmw.yaml"


def _build_model():
    """Build the YOLO26 LMW detection model."""
    return YOLO(CFG).model


def _assert_branch_outputs(branch, batch_size):
    """Validate one2many/one2one branch tensor structure."""
    assert set(branch) == {"boxes", "scores", "feats"}
    assert branch["boxes"].shape[0] == batch_size
    assert branch["scores"].shape[0] == batch_size
    assert branch["boxes"].shape[-1] == branch["scores"].shape[-1]
    assert torch.isfinite(branch["boxes"]).all()
    assert torch.isfinite(branch["scores"]).all()


def test_yolo26_lmw_model_builds():
    """The YAML builds and exposes an end-to-end Detect head."""
    model = YOLO(CFG)
    head = model.model.model[-1]

    assert isinstance(head, Detect)
    assert head.end2end is True


def test_yolo26_lmw_has_expected_modules():
    """The YAML swaps only the intended head modules."""
    layers = _build_model().model

    assert isinstance(layers[16], LKCA)
    assert isinstance(layers[19], MSDP)
    assert isinstance(layers[22], C3k2)
    assert layers[-1].f == [16, 19, 22]


def test_yolo26_lmw_forward_train_shape_smoke():
    """Train-mode forward returns the current end-to-end dict structure."""
    model = _build_model()
    model.train()

    with torch.no_grad():
        outputs_batch_1 = model(torch.randn(1, 3, 64, 64))
        outputs_batch_2 = model(torch.randn(2, 3, 64, 64))

    assert set(outputs_batch_1) == {"one2many", "one2one"}
    assert set(outputs_batch_2) == {"one2many", "one2one"}
    _assert_branch_outputs(outputs_batch_1["one2many"], batch_size=1)
    _assert_branch_outputs(outputs_batch_1["one2one"], batch_size=1)
    _assert_branch_outputs(outputs_batch_2["one2many"], batch_size=2)
    _assert_branch_outputs(outputs_batch_2["one2one"], batch_size=2)


def test_yolo26_lmw_detect_outputs_end2end():
    """Eval-mode forward returns the current end-to-end tuple output."""
    model = _build_model()
    model.eval()

    with torch.no_grad():
        outputs = model(torch.randn(2, 3, 64, 64))

    assert isinstance(outputs, tuple)
    assert len(outputs) == 2
    predictions, preds = outputs
    assert predictions.shape[0] == 2
    assert set(preds) == {"one2many", "one2one"}

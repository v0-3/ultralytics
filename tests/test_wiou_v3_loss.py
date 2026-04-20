"""Tests for optional WIoU v3 bbox loss wiring."""

from types import SimpleNamespace

import torch

from ultralytics import YOLO
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist

CFG = "ultralytics/cfg/models/26/yolo26-lmw.yaml"


def _make_bbox_inputs():
    """Build a small, coherent bbox-loss input batch."""
    anchor_points = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    pred_bboxes = torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.25, 1.25, 3.75, 3.75]])
    target_bboxes = torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 4.0, 4.0]])
    pred_dist = bbox2dist(anchor_points, pred_bboxes)
    target_scores = torch.tensor([[1.0], [0.5]])
    target_scores_sum = target_scores.sum()
    fg_mask = torch.tensor([True, True])
    imgsz = torch.tensor([8.0, 8.0])
    stride = torch.ones((2, 1))
    return pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, imgsz, stride


def _wiou_v3_terms(
    pred_bboxes: torch.Tensor,
    target_bboxes: torch.Tensor,
    iou_loss_mean: torch.Tensor,
    alpha: float,
    delta: float,
    eps: float,
):
    """Compute the expected WIoU v3 per-box terms for a batch."""
    iou = bbox_iou(pred_bboxes, target_bboxes, xywh=False)
    iou_loss = (1.0 - iou).clamp(min=0)

    b1_x1, b1_y1, b1_x2, b1_y2 = pred_bboxes.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = target_bboxes.chunk(4, -1)
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = (cw.pow(2) + ch.pow(2)).clamp_min(eps)
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4

    r_wiou = ((rho2 / c2).clamp(min=0, max=50)).exp()
    beta = iou_loss.detach() / iou_loss_mean.clamp_min(eps)
    r = beta / (delta * alpha ** (beta - delta))
    return iou_loss, r.detach() * r_wiou * iou_loss


def test_wiou_v3_loss_finite_for_simple_boxes():
    """WIoU v3 stays finite for simple overlapping boxes."""
    bbox_loss = BboxLoss(reg_max=1, box_loss="wiou_v3")
    bbox_loss.eval()

    loss_iou, loss_dfl = bbox_loss(*_make_bbox_inputs())

    assert loss_iou.ndim == 0
    assert loss_dfl.ndim == 0
    assert torch.isfinite(loss_iou)
    assert torch.isfinite(loss_dfl)


def test_wiou_v3_loss_reduces_to_valid_positive_scalar():
    """WIoU v3 matches the expected weighted scalar loss."""
    inputs = _make_bbox_inputs()
    bbox_loss = BboxLoss(reg_max=1, box_loss="wiou_v3", wiou_alpha=1.9, wiou_delta=3.0, wiou_eps=1e-7)
    bbox_loss.eval()

    loss_iou, _ = bbox_loss(*inputs)

    _, per_box = _wiou_v3_terms(
        pred_bboxes=inputs[1][inputs[6]],
        target_bboxes=inputs[3][inputs[6]],
        iou_loss_mean=bbox_loss.iou_loss_mean,
        alpha=bbox_loss.wiou_alpha,
        delta=bbox_loss.wiou_delta,
        eps=bbox_loss.wiou_eps,
    )
    expected = (per_box * inputs[4].sum(-1)[inputs[6]].unsqueeze(-1)).sum() / inputs[5]

    assert loss_iou.ndim == 0
    assert loss_iou > 0
    assert torch.isfinite(loss_iou)
    assert torch.allclose(loss_iou, expected, atol=1e-6, rtol=1e-6)


def test_wiou_v3_running_mean_updates_in_train_mode():
    """WIoU v3 updates its running IoU-loss mean during training."""
    inputs = _make_bbox_inputs()
    bbox_loss = BboxLoss(reg_max=1, box_loss="wiou_v3", wiou_momentum=0.5)

    iou_loss, _ = _wiou_v3_terms(
        pred_bboxes=inputs[1][inputs[6]],
        target_bboxes=inputs[3][inputs[6]],
        iou_loss_mean=bbox_loss.iou_loss_mean,
        alpha=bbox_loss.wiou_alpha,
        delta=bbox_loss.wiou_delta,
        eps=bbox_loss.wiou_eps,
    )

    bbox_loss(*inputs)

    expected_mean = 0.5 * torch.tensor(1.0) + 0.5 * iou_loss.mean()
    assert torch.allclose(bbox_loss.iou_loss_mean, expected_mean, atol=1e-6, rtol=1e-6)


def test_ciou_default_path_unchanged_shape_and_finite():
    """The default bbox-loss path remains equivalent to explicit CIoU."""
    inputs = _make_bbox_inputs()
    default_loss = BboxLoss(reg_max=1)
    explicit_ciou_loss = BboxLoss(reg_max=1, box_loss="ciou")
    default_loss.eval()
    explicit_ciou_loss.eval()

    default_iou, default_dfl = default_loss(*inputs)
    explicit_iou, explicit_dfl = explicit_ciou_loss(*inputs)

    assert default_iou.ndim == 0
    assert default_dfl.ndim == 0
    assert torch.isfinite(default_iou)
    assert torch.isfinite(default_dfl)
    assert torch.allclose(default_iou, explicit_iou, atol=1e-6, rtol=1e-6)
    assert torch.allclose(default_dfl, explicit_dfl, atol=1e-6, rtol=1e-6)


def test_e2e_loss_uses_configured_box_loss_for_both_paths():
    """End-to-end detection loss forwards the configured bbox-loss mode to both branches."""
    model = YOLO(CFG).model
    model.args = SimpleNamespace(
        box_loss="wiou_v3",
        wiou_alpha=1.9,
        wiou_delta=3.0,
        wiou_momentum=0.99,
        wiou_eps=1e-7,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        epochs=1,
    )

    criterion = model.init_criterion()

    assert criterion.one2many.bbox_loss.box_loss == "wiou_v3"
    assert criterion.one2one.bbox_loss.box_loss == "wiou_v3"

# YOLO26-LMW Codex Implementation Plan

Generated: 2026-04-20

This is a Codex-ready implementation plan for forking Ultralytics YOLO26 and adding an LMW-YOLO-inspired detection variant based on the Scientific Reports paper and the public LMW-YOLO repository.

Use this document as a repository-local execution plan. Recommended path:

```text
docs/plans/yolo26-lmw-execplan.md
```

Recommended Codex prompt:

```text
Read docs/plans/yolo26-lmw-execplan.md. Implement Milestone 1 only. Before editing, inspect the current upstream code paths named in the plan and restate the exact files you will touch. Keep the change small, run the specified tests, and stop after the milestone acceptance criteria pass.
```

For later milestones, replace `Milestone 1` with the specific milestone number. Do not ask Codex to implement the entire plan in one pass unless you are prepared to review a large PR.

---

## 1. Verified upstream context

### 1.1 Source materials

Primary sources checked on 2026-04-20:

- Ultralytics YOLO26 docs: https://docs.ultralytics.com/models/yolo26/
- YOLO26 model YAML: https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/models/26/yolo26.yaml
- YOLO26 P2 model YAML: https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/models/26/yolo26-p2.yaml
- LMW-YOLO paper: https://www.nature.com/articles/s41598-026-45055-6
- LMW-YOLO repository: https://github.com/qqqqqq-ch/LMW-YOLO
- LMW-YOLO model YAML: https://raw.githubusercontent.com/qqqqqq-ch/LMW-YOLO/main/YOLO-UAV.yaml
- LMW-YOLO custom block file: https://raw.githubusercontent.com/qqqqqq-ch/LMW-YOLO/main/ultralytics/modifiednn/modules/block.py
- OpenAI Codex `AGENTS.md` guidance: https://developers.openai.com/codex/guides/agents-md
- OpenAI Codex execution plan guidance: https://developers.openai.com/cookbook/articles/codex_exec_plans
- OpenAI Codex best practices: https://developers.openai.com/codex/learn/best-practices

### 1.2 YOLO26 facts that constrain the implementation

The upstream YOLO26 detection YAML currently has:

```yaml
end2end: True
reg_max: 1
```

The upstream YOLO26 detection head uses P3/P4/P5 outputs and ends with:

```yaml
- [[16, 19, 22], 1, Detect, [nc]]
```

YOLO26 is designed around native end-to-end / NMS-free inference, DFL removal, and a newer loss/training stack. Do not overwrite the YOLO26 detection head or replace the whole training loss stack with YOLO11-era code.

### 1.3 LMW-YOLO facts that constrain the implementation

The paper proposes a Context-Scale Decoupled strategy:

```text
P3 / shallow small-object branch: LKCA
P4 / medium branch:              MSDP
P5 / deep branch:                C3k2
Box regression loss:             WIoU v3
```

The public LMW-YOLO YAML mirrors this head placement:

```yaml
- [-1, 2, LKCA, [256, True]] # 16, P3/8-small
- [-1, 2, MSDP, [512, True]] # 19, P4/16-medium
- [-1, 2, C3k2, [1024, True]] # 22, P5/32-large
- [[16, 19, 22], 1, Detect, [nc]]
```

The public LMW-YOLO custom module file currently defines `RFAM`, `MSDP`, and `LKCA` as empty stubs. Treat this as a paper-to-YOLO26 port, not a direct transplant of working module code.

### 1.4 Codex usage constraints

Codex works best when the repository has durable guidance such as `AGENTS.md`, and larger tasks can be driven from execution plans. Keep this plan in the repository and ask Codex to implement one milestone at a time.

Suggested repo-root `AGENTS.md` addition:

```markdown
# YOLO26-LMW implementation

When implementing YOLO26-LMW, read `docs/plans/yolo26-lmw-execplan.md` before editing. Implement one milestone at a time. Preserve YOLO26 end-to-end detection semantics unless the plan explicitly says otherwise. After each milestone, run the specified tests and summarize changed files, commands run, and remaining risks.
```

---

## 2. Goal

Create a fork of Ultralytics YOLO26 that adds a detection model variant inspired by LMW-YOLO:

```text
ultralytics/cfg/models/26/yolo26-lmw.yaml
```

Optional tiny-object variant:

```text
ultralytics/cfg/models/26/yolo26-lmw-p2.yaml
```

The implementation must:

1. Preserve YOLO26's detection head, end-to-end mode, and `reg_max: 1` behavior.
2. Add paper-described `LKCA` and `MSDP` modules.
3. Replace the P3 head block with `LKCA` and the P4 head block with `MSDP`.
4. Keep the P5 head block as `C3k2`.
5. Add WIoU v3 as an optional box loss mode, not as the unconditional default.
6. Include smoke tests, shape tests, backward tests, and one-epoch training checks.
7. Keep the main model near YOLO26n / LMW-YOLO scale.

---

## 3. Non-goals

Do not do the following in the first implementation:

- Do not rewrite the YOLO26 detection head.
- Do not remove YOLO26 `end2end: True`.
- Do not restore YOLO11-style DFL behavior.
- Do not copy incomplete LMW-YOLO stub classes as final code.
- Do not force WIoU v3 globally across all models.
- Do not target oriented bounding boxes in this plan.
- Do not require RS-STOD or VisDrone full training to pass before merging the initial architecture PR.

---

## 4. Branching and PR strategy

Create focused branches:

```text
feature/yolo26-lmw-modules
feature/yolo26-lmw-yaml
feature/yolo26-lmw-wiou-v3
experiment/yolo26-lmw-ablation
```

Recommended PR sequence:

1. `PR 1`: Add LKCA/MSDP modules and parser integration.
2. `PR 2`: Add `yolo26-lmw.yaml` and smoke tests.
3. `PR 3`: Add optional WIoU v3 loss mode.
4. `PR 4`: Add dataset conversion scripts and training recipes.
5. `PR 5`: Add ablation results and documentation.

---

## 5. Repository setup

```bash
git clone git@github.com:<your-user-or-org>/ultralytics.git
cd ultralytics
git checkout -b feature/yolo26-lmw-modules

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Baseline checks before modifications:

```bash
yolo detect train model=yolo26n.yaml data=coco8.yaml epochs=1 imgsz=640
python - <<'PY'
from ultralytics import YOLO
model = YOLO('yolo26n.yaml')
model.info()
PY
```

If either baseline check fails, stop and fix environment issues before changing architecture code.

---

## 6. File map

Codex must inspect the current versions of these files before editing because Ultralytics internals can change:

```text
ultralytics/nn/modules/__init__.py
ultralytics/nn/modules/block.py
ultralytics/nn/modules/conv.py
ultralytics/nn/tasks.py
ultralytics/utils/loss.py
ultralytics/utils/metrics.py
ultralytics/cfg/default.yaml
ultralytics/cfg/models/26/yolo26.yaml
ultralytics/cfg/models/26/yolo26-p2.yaml
```

Expected new files:

```text
ultralytics/nn/modules/lmw.py
ultralytics/cfg/models/26/yolo26-lmw.yaml
ultralytics/cfg/models/26/yolo26-lmw-p2.yaml              # optional
examples/train_yolo26_lmw_nwpu.sh                         # later milestone
examples/train_yolo26_lmw_visdrone.sh                     # later milestone
docs/models/yolo26-lmw.md                                  # later milestone
tests/test_lmw_modules.py
tests/test_yolo26_lmw_yaml.py
tests/test_wiou_v3_loss.py
```

Expected modified files:

```text
ultralytics/nn/modules/__init__.py
ultralytics/nn/tasks.py
ultralytics/utils/loss.py
ultralytics/cfg/default.yaml
```

---

## 7. Architecture implementation

### 7.1 Module names

Add these public classes:

```python
LKABottleneck
LKCA
DWRBottleneck
MSDP
```

Place them in:

```text
ultralytics/nn/modules/lmw.py
```

Export from:

```text
ultralytics/nn/modules/__init__.py
```

Import into parser scope in:

```text
ultralytics/nn/tasks.py
```

### 7.2 Parser registration

In `parse_model`, add `LKCA` and `MSDP` to the same parser categories as C3/C2f-style repeated channel-changing modules.

Expected behavior:

```yaml
- [-1, 2, LKCA, [256, True]]
```

should instantiate as roughly:

```python
LKCA(c1, c2, n, shortcut=True)
```

and:

```yaml
- [-1, 2, MSDP, [512, True]]
```

should instantiate as roughly:

```python
MSDP(c1, c2, n, shortcut=True)
```

Implementation guidance:

```text
- Add LKCA and MSDP to base_modules.
- Add LKCA and MSDP to repeat_modules.
- Do not add special-case legacy logic unless tests show it is needed.
- Keep signatures compatible with Ultralytics depth/width scaling.
```

### 7.3 `LKABottleneck`

Paper-derived behavior:

```text
X
 -> 5x5 depth-wise convolution
 -> 7x7 depth-wise dilated convolution with dilation = 3
 -> 1x1 point-wise convolution
 -> additive residual connection
```

Formula from paper:

```text
F_local  = DWConv_5x5(X)
F_global = DWConv_7x7,d=3(F_local)
Y        = Conv_1x1(F_global) + X
```

Implementation requirements:

```text
- Preserve spatial shape.
- Use groups=channels for depth-wise convolutions.
- Padding for 5x5 DWConv should preserve H/W.
- Padding for 7x7 dilation=3 should preserve H/W.
- Residual add only when input and output channels match.
- Keep tensor dtype and device unchanged.
- Avoid fragile in-place ops.
```

Recommended signature:

```python
class LKABottleneck(nn.Module):
    def __init__(self, c: int, shortcut: bool = True): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

### 7.4 `LKCA`

Paper-derived behavior:

```text
input
 -> CSP/C3k2-like split into two branches
 -> branch A: one or more LKABottleneck layers
 -> branch B: identity/light shortcut branch
 -> concatenate
 -> 1x1 fuse
```

Recommended implementation pattern:

```python
class LKCA(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, 2 * c_, 1, 1)
        self.m = nn.Sequential(*(LKABottleneck(c_, shortcut) for _ in range(n)))
        self.cv2 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        y1, y2 = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(y1), y2), 1))
```

Codex may adjust the exact structure to match local Ultralytics conventions, but the public API and shape behavior must remain stable.

### 7.5 `DWRBottleneck`

Paper-derived behavior:

```text
input
 -> 3x3 convolution for region residualization / local regional texture
 -> parallel dilated convolutions with dilation rates 1, 3, 5
 -> concatenate or sum branches
 -> 1x1 fuse
 -> residual add
```

Recommended implementation:

```text
- Start with a normal 3x3 Conv.
- Create three depth-wise or standard conv branches with dilation 1, 3, 5.
- Preserve spatial shape in all branches.
- Fuse with 1x1 Conv.
- Residual add when enabled and channels match.
```

Preferred lightweight version:

```text
- Use depth-wise dilated branches plus 1x1 fuse to constrain params/FLOPs.
```

### 7.6 `MSDP`

Paper-derived behavior:

```text
input
 -> split channels in half
 -> branch A: bypass identity
 -> branch B: DWRBottleneck
 -> concatenate
 -> 1x1 restore
 -> residual fuse
```

Recommended signature:

```python
class MSDP(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, e: float = 0.5): ...
```

Recommended behavior:

```text
- Project c1 -> 2*c_ channels.
- Split with chunk(2, 1).
- Apply DWRBottleneck stack to one half.
- Bypass the other half unchanged.
- Concatenate and fuse to c2.
- Apply outer residual only when c1 == c2 and shortcut is true.
```

---

## 8. YOLO26-LMW YAML

Create:

```text
ultralytics/cfg/models/26/yolo26-lmw.yaml
```

Start from upstream `yolo26.yaml`. Keep:

```yaml
end2end: True
reg_max: 1
```

Modify only the P3 and P4 head refinement blocks. Target head:

```yaml
# YOLO26n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, C3k2, [512, True]] # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, LKCA, [256, True]] # 16 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 2, MSDP, [512, True]] # 19 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 1, C3k2, [1024, True, 0.5, True]] # 22 (P5/32-large)

  - [[16, 19, 22], 1, Detect, [nc]] # Detect(P3, P4, P5)
```

Important: use the current upstream YOLO26 P5 block form, not the YOLO11-style P5 form, unless tests show the parser requires a different signature.

### 8.1 Optional P2 variant

Create only after the main model passes tests:

```text
ultralytics/cfg/models/26/yolo26-lmw-p2.yaml
```

Start from upstream `yolo26-p2.yaml`. Suggested mapping:

```text
P2: C3k2 or LKCA-lite
P3: LKCA
P4: MSDP
P5: C3k2
Detect: P2/P3/P4/P5
```

Keep it separate from the main model because P2 changes FLOPs and comparison fairness.

---

## 9. WIoU v3 implementation

### 9.1 Config keys

Add to `ultralytics/cfg/default.yaml`:

```yaml
box_loss: ciou # choices: ciou, wiou_v3
wiou_alpha: 1.9
wiou_delta: 3.0
wiou_momentum: 0.99
wiou_eps: 1.0e-7
```

The default must remain functionally equivalent to upstream YOLO26 behavior.

### 9.2 Loss integration point

Current YOLO-style detection loss uses `BboxLoss` inside `v8DetectionLoss`; YOLO26 end-to-end training wraps two `v8DetectionLoss` instances through `E2ELoss`. Therefore:

```text
- Modify BboxLoss to accept box_loss, wiou_alpha, wiou_delta, wiou_momentum, wiou_eps.
- Instantiate BboxLoss from v8DetectionLoss using model.args.
- Do not bypass E2ELoss.
- Ensure both one-to-many and one-to-one losses use the same configured box loss mode.
```

### 9.3 Mathematical target

For foreground boxes:

```text
R_WIoU = exp(center_distance_squared / enclosing_diagonal_squared)
L_WIoUv1 = (1 - IoU) * R_WIoU
beta = detached_iou_loss / running_mean_iou_loss
r = beta / (delta * alpha ** (beta - delta))
L_WIoUv3 = detach(r) * L_WIoUv1
```

Use defaults:

```text
alpha = 1.9
delta = 3.0
```

### 9.4 Numerical constraints

```text
- Clamp IoU to a safe range before computing 1 - IoU.
- Clamp denominator for enclosing diagonal squared with eps.
- Clamp exponent input before exp for AMP stability.
- Detach beta/r from gradient unless intentionally experimenting.
- Update running mean under torch.no_grad().
- Handle no-foreground batches by returning zero box loss as upstream does.
```

### 9.5 DDP/resume constraints

Minimum acceptable implementation:

```text
- Running IoU mean is a BboxLoss buffer.
- It updates during training only.
- It does not break single-GPU, CPU, AMP, or DDP smoke tests.
```

Preferred implementation:

```text
- Synchronize the batch IoU-loss mean across distributed ranks before EMA update.
- Serialize the loss state if the trainer supports criterion state checkpointing.
```

Do not block Milestone 3 on criterion-state checkpointing if the upstream trainer does not already persist criterion state. Document this limitation.

---

## 10. Tests

### 10.1 Module tests

Create:

```text
tests/test_lmw_modules.py
```

Required tests:

```python
def test_lka_bottleneck_preserves_shape(): ...
def test_lka_bottleneck_backward(): ...
def test_lkca_changes_channels_and_backward(): ...
def test_dwr_bottleneck_preserves_shape(): ...
def test_msdp_changes_channels_and_backward(): ...
def test_lmw_modules_torchscript_or_export_friendly_smoke(): ...
```

Test shapes:

```text
batch=2, channels=64, h=32, w=32
LKCA: c1=64, c2=128
MSDP: c1=128, c2=256
```

Assertions:

```text
- output shape is expected
- output dtype equals input dtype
- no NaN/Inf
- backward produces finite gradients for at least one parameter
```

### 10.2 YAML/model tests

Create:

```text
tests/test_yolo26_lmw_yaml.py
```

Required tests:

```python
def test_yolo26_lmw_model_builds(): ...
def test_yolo26_lmw_forward_train_shape_smoke(): ...
def test_yolo26_lmw_has_expected_modules(): ...
def test_yolo26_lmw_detect_outputs_end2end(): ...
```

Suggested command:

```bash
python - <<'PY'
from ultralytics import YOLO
model = YOLO('ultralytics/cfg/models/26/yolo26-lmw.yaml')
model.info()
PY
```

### 10.3 WIoU v3 tests

Create:

```text
tests/test_wiou_v3_loss.py
```

Required tests:

```python
def test_wiou_v3_loss_finite_for_simple_boxes(): ...
def test_wiou_v3_loss_reduces_to_valid_positive_scalar(): ...
def test_wiou_v3_running_mean_updates_in_train_mode(): ...
def test_ciou_default_path_unchanged_shape_and_finite(): ...
def test_e2e_loss_uses_configured_box_loss_for_both_paths(): ...
```

### 10.4 Smoke training/export checks

Run after YAML and loss milestones:

```bash
yolo detect train model=ultralytics/cfg/models/26/yolo26-lmw.yaml data=coco8.yaml epochs=1 imgsz=640 batch=2

yolo detect train model=ultralytics/cfg/models/26/yolo26-lmw.yaml data=coco8.yaml epochs=1 imgsz=640 batch=2 box_loss=wiou_v3

python - <<'PY'
from ultralytics import YOLO
model = YOLO('runs/detect/train/weights/best.pt')
model.export(format='onnx', imgsz=640)
PY
```

---

## 11. Training recipes

### 11.1 Architecture-only run

```bash
yolo detect train \
  model=ultralytics/cfg/models/26/yolo26-lmw.yaml \
  data=datasets/nwpu-vhr10.yaml \
  imgsz=640 \
  epochs=300 \
  batch=32 \
  close_mosaic=30
```

### 11.2 Full LMW-style run with WIoU v3

```bash
yolo detect train \
  model=ultralytics/cfg/models/26/yolo26-lmw.yaml \
  data=datasets/nwpu-vhr10.yaml \
  imgsz=640 \
  epochs=300 \
  batch=32 \
  close_mosaic=30 \
  box_loss=wiou_v3 \
  wiou_alpha=1.9 \
  wiou_delta=3.0
```

### 11.3 Paper-style optimizer comparison

The paper reports SGD with initial learning rate 0.01 and final learning rate 0.001. In Ultralytics config terms, verify the exact optimizer arguments in the current version before running. A likely starting point is:

```bash
yolo detect train \
  model=ultralytics/cfg/models/26/yolo26-lmw.yaml \
  data=datasets/nwpu-vhr10.yaml \
  imgsz=640 \
  epochs=300 \
  batch=32 \
  optimizer=SGD \
  lr0=0.01 \
  lrf=0.1 \
  close_mosaic=30 \
  box_loss=wiou_v3 \
  wiou_alpha=1.9 \
  wiou_delta=3.0
```

---

## 12. Dataset plan

Create dataset YAMLs:

```text
datasets/nwpu-vhr10.yaml
datasets/rs-stod.yaml
datasets/visdrone2019.yaml
```

Create conversion scripts if not already available:

```text
tools/datasets/convert_nwpu_vhr10.py
tools/datasets/convert_rs_stod.py
tools/datasets/convert_visdrone2019.py
```

Dataset requirements:

```text
- Convert all labels to YOLO normalized xywh format.
- Preserve ignored/negative images where valid.
- Generate deterministic train/val/test splits with a seed.
- Write a conversion manifest containing source path, output path, seed, date, and class map.
- Add a label visualization sanity-check script.
```

Paper reference splits:

```text
NWPU VHR-10: 8:1:1 random split
RS-STOD:     7:1:2 random split
VisDrone:    official train/val/test split, ignoring ignored regions and others
```

---

## 13. Ablation matrix

Run these after implementation is stable:

| Run | Model                          | P3   | P4   | Loss    | Purpose                  |
| --: | ------------------------------ | ---- | ---- | ------- | ------------------------ |
|   A | `yolo26n.yaml`                 | C3k2 | C3k2 | default | YOLO26 baseline          |
|   B | `yolo26-lmw.yaml` LKCA-only    | LKCA | C3k2 | default | isolate LKCA             |
|   C | `yolo26-lmw.yaml` architecture | LKCA | MSDP | default | architecture-only effect |
|   D | `yolo26-lmw.yaml` full         | LKCA | MSDP | WIoU v3 | full model               |
|   E | `yolo26-lmw-p2.yaml` full      | LKCA | MSDP | WIoU v3 | tiny-object P2 variant   |

Suggested metrics:

```text
mAP@0.5
mAP@0.5:0.95
precision
recall
params
FLOPs
CPU latency
GPU latency
ONNX export success
TensorRT export success, if applicable
```

---

## 14. Acceptance criteria

### 14.1 Functional acceptance

```text
- `YOLO('ultralytics/cfg/models/26/yolo26-lmw.yaml')` builds.
- `model.info()` runs without parser errors.
- A forward pass works for batch size 1 and batch size 2.
- One-epoch coco8 training completes with default loss.
- One-epoch coco8 training completes with `box_loss=wiou_v3`.
- ONNX export completes for a trained or randomly initialized checkpoint.
```

### 14.2 Model-size acceptance

Main P3/P4/P5 variant target:

```text
params <= 2.8M for n scale, unless justified
FLOPs  <= 7.0G at 640, unless justified
```

The P2 variant may exceed this because it adds an extra detection scale.

### 14.3 Performance acceptance

```text
- Full YOLO26-LMW beats the local YOLO26n baseline on NWPU VHR-10 mAP@0.5.
- mAP@0.5:0.95 does not regress materially versus the local YOLO26n baseline.
- Ablation trend is coherent, even if exact paper numbers are not reproduced.
```

Do not require the paper's exact reported mAP as a merge blocker, because this is a YOLO26 port rather than the original YOLO11n implementation.

---

## 15. Milestones for Codex

### Milestone 1: Baseline fork validation

Prompt:

```text
Read docs/plans/yolo26-lmw-execplan.md. Implement Milestone 1 only: validate that the current fork can build and smoke-train upstream YOLO26. Do not edit architecture or loss code. Add a short `docs/models/yolo26-lmw.md` placeholder with source links and a checklist if docs do not exist. Run the baseline commands in the plan and report results.
```

Acceptance:

```text
- Upstream YOLO26 model builds.
- `coco8.yaml` one-epoch training succeeds or environment failure is documented.
- No architecture/loss changes are made.
```

### Milestone 2: Add LKCA/MSDP modules

Prompt:

```text
Read docs/plans/yolo26-lmw-execplan.md. Implement Milestone 2 only: add `ultralytics/nn/modules/lmw.py` with LKABottleneck, LKCA, DWRBottleneck, and MSDP. Export the classes and register LKCA/MSDP in the model parser as repeated base modules. Add module unit tests. Do not add WIoU yet. Run the module tests and a parser smoke test.
```

Acceptance:

```text
- LKCA and MSDP are importable from `ultralytics.nn.modules`.
- YAML parser can resolve `LKCA` and `MSDP`.
- Shape and backward tests pass.
- No changes to loss behavior.
```

### Milestone 3: Add `yolo26-lmw.yaml`

Prompt:

```text
Read docs/plans/yolo26-lmw-execplan.md. Implement Milestone 3 only: create `ultralytics/cfg/models/26/yolo26-lmw.yaml` from current YOLO26 YAML, replacing only the P3 head block with LKCA and the P4 head block with MSDP. Keep end2end True, reg_max 1, and the YOLO26 Detect head. Add YAML/model smoke tests. Run model build and one forward-pass check.
```

Acceptance:

```text
- `YOLO('ultralytics/cfg/models/26/yolo26-lmw.yaml')` builds.
- Model contains LKCA at P3 and MSDP at P4.
- Detect head still consumes [16, 19, 22].
- Batch-1 and batch-2 forward smoke tests pass.
```

### Milestone 4: One-epoch training and export

Prompt:

```text
Read docs/plans/yolo26-lmw-execplan.md. Implement Milestone 4 only: make the yolo26-lmw model pass one-epoch detection training on coco8 and ONNX export. Fix only issues required for training/export. Do not add WIoU yet. Document commands and results.
```

Acceptance:

```text
- `yolo detect train model=...yolo26-lmw.yaml data=coco8.yaml epochs=1 imgsz=640 batch=2` succeeds.
- ONNX export succeeds.
- Any export limitations are documented.
```

### Milestone 5: Add WIoU v3 as optional box loss

Prompt:

```text
Read docs/plans/yolo26-lmw-execplan.md. Implement Milestone 5 only: add optional `box_loss=wiou_v3` support with `wiou_alpha`, `wiou_delta`, `wiou_momentum`, and `wiou_eps` config keys. Preserve the default CIoU behavior. Ensure YOLO26 end-to-end loss uses the configured BboxLoss in both one-to-many and one-to-one paths. Add WIoU unit tests and run one-epoch coco8 training with and without WIoU.
```

Acceptance:

```text
- Default training path remains finite and passes smoke training.
- `box_loss=wiou_v3` path is finite and passes smoke training.
- WIoU v3 running mean updates during training.
- E2ELoss still works.
```

### Milestone 6: Dataset configs and conversion scripts

Prompt:

```text
Read docs/plans/yolo26-lmw-execplan.md. Implement Milestone 6 only: add dataset YAML templates and conversion scripts for NWPU VHR-10, RS-STOD, and VisDrone2019. Do not download datasets. Scripts should validate paths, produce deterministic splits, and write YOLO labels plus a manifest. Add dry-run tests using tiny synthetic annotations.
```

Acceptance:

```text
- Dataset YAML templates exist.
- Conversion scripts have clear CLI help.
- Dry-run tests pass.
- Scripts do not require proprietary/local data to run tests.
```

### Milestone 7: Ablation and documentation

Prompt:

```text
Read docs/plans/yolo26-lmw-execplan.md. Implement Milestone 7 only: add training scripts, ablation config notes, and documentation. Do not fabricate metrics. Add tables with placeholders for local results and commands that reproduce each run.
```

Acceptance:

```text
- `docs/models/yolo26-lmw.md` explains architecture, loss, commands, and limitations.
- Example scripts are executable or clearly documented.
- No unsupported performance claims are made.
```

---

## 16. Review checklist

Before merging each PR, verify:

```text
[ ] No unrelated formatting churn.
[ ] No wholesale copy of YOLO11 loss code.
[ ] YOLO26 `end2end: True` preserved.
[ ] YOLO26 `reg_max: 1` preserved.
[ ] New modules have shape and backward tests.
[ ] New YAML builds from a clean checkout.
[ ] WIoU v3 is optional and default behavior is preserved.
[ ] Smoke training command and result are documented.
[ ] Licenses and source attribution are documented.
```

---

## 17. License and attribution notes

Ultralytics is AGPL-3.0. The public LMW-YOLO repository is GPL-3.0. Preserve upstream notices. Because the visible LMW-YOLO custom module implementations are stubs, implement the modules from the paper description instead of copying nonexistent or incomplete code.

Add attribution in docs:

```text
This model variant is a YOLO26 port inspired by Qiu and Lin, "Lightweight model LMW-YOLO for small object detection in remote sensing images," Scientific Reports 16, 11644 (2026), and the public qqqqqq-ch/LMW-YOLO repository. It is not a byte-for-byte reproduction of the original YOLO11n-based implementation.
```

---

## 18. Known risks

```text
1. The public LMW-YOLO module classes are stubs, so exact architecture may require interpretation from figures/text.
2. YOLO26 loss behavior differs from YOLO11n; WIoU v3 must be integrated as a controlled option.
3. WIoU v3 has running state, which can be tricky under DDP and resume-from-checkpoint.
4. P2 may improve tiny objects but changes FLOPs and comparison fairness.
5. Paper metrics may not reproduce exactly because this is a YOLO26 port with different training internals.
```

---

## 19. Minimal first Codex task

Use this as the first actual implementation prompt:

```text
Read `docs/plans/yolo26-lmw-execplan.md`. Implement Milestone 2 only.

Specific constraints:
- Create `ultralytics/nn/modules/lmw.py`.
- Add `LKABottleneck`, `LKCA`, `DWRBottleneck`, and `MSDP`.
- Use Ultralytics `Conv` where practical.
- Register `LKCA` and `MSDP` in `ultralytics/nn/modules/__init__.py` and `ultralytics/nn/tasks.py`.
- Treat LKCA/MSDP like C3k2-style repeated modules in the parser.
- Add tests in `tests/test_lmw_modules.py`.
- Do not add WIoU v3 and do not add `yolo26-lmw.yaml` yet.

Done when:
- `pytest tests/test_lmw_modules.py` passes.
- A temporary in-memory YAML using LKCA/MSDP can be parsed or an equivalent parser smoke test passes.
- You summarize changed files and commands run.
```

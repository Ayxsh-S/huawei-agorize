# ROLE B — GLOBAL LANDMARK MODEL ENGINEER

**Owner: Ojas**

> Give this file to your AI after `MASTER_AI_CONTEXT.md` and `DATA_SPEC.md`.

Tell the AI:

> I am Ojas and I own Role B. Inspect the current repository before editing. Follow this role in order. Do not redesign the architecture. Do not modify another owner's modules unless a shared interface genuinely requires it. Do not commit or push.

---

# Mission

Build the smallest reliable PointNet-style neural model that predicts `85 x 3` canonical landmark residuals from A's processed ear point cloud and beats C's canonical-template baseline.

---

# Primary ownership

```text
src/model.py
src/losses.py
src/train.py
configs/
tests/test_model.py
```

You own:
- architecture
- losses
- augmentation
- training
- checkpoints
- GPU experiments

You do not own:
- PLY geometry/preprocessing
- template statistics
- evaluation metric
- final inference CLI

---

# Input contract

A supplies canonical ears.

Initial:

```text
points [B,2048,3]
```

Later optional:

```text
features [B,2048,6]
```

Training target:

```text
canonical_landmarks [B,85,3]
```

C supplies canonical template(s).

---

# Prediction contract

Your model outputs residuals:

```python
residual = model(points)
assert residual.shape == (B,85,3)
```

Then:

```python
pred_canonical = canonical_template + residual
```

Template selection should be supplied/configured externally.

Do not recompute template statistics from validation data.

---

# First architecture

Keep it small.

Suggested:

```text
[B,N,F]
→ shared point MLP
64
→ 128
→ 256
→ global max pool
→ MLP 256→512→256→255
→ reshape [B,85,3]
```

Use standard PyTorch.

Do not begin with:
- PointNet++
- transformer
- graph model
- heatmaps
- uncertainty head
- bilateral branch
- contour branch

---

# Loss

Start with Smooth-L1/Huber coordinate loss in canonical space.

Role C owns the official Euclidean evaluation metric.

Later, compare at most one simple alternative if useful.

---

# Required development order

## 1. Tensor-shape test

```python
x = torch.randn(B,2048,3)
r = model(x)
assert r.shape == (B,85,3)
```

---

## 2. Synthetic training-loop smoke test

Before A's cache exists:
- random/synthetic point tensors
- random/synthetic target residuals
- verify forward/backward/checkpoint save/load

This lets you code independently on day 1.

---

## 3. Tiny real-data overfit

Once A supplies canonical ears and C supplies template:

Train on ~4–8 ears.

Augmentation OFF.

The model should substantially overfit.

If not, investigate:
- canonical coordinate scale
- target construction
- tensor order
- template
- loss
- optimiser
- interface bugs

Do not just make the network bigger.

---

## 4. First full training

Use C's frozen subject split.

Log run config/results.

---

# Augmentation

Only after baseline works.

Possible conservative options:
- small rotation
- small scale
- jitter
- point dropout/resampling

Ensure target landmarks are transformed consistently for geometric augmentations.

Own/document all augmentation logic.

---

# Checkpoints

Store enough metadata to reproduce:
- model `state_dict`
- input feature dimension
- architecture config
- template version reference
- epoch
- seed
- training config
- validation score if available

Do not leave D guessing which architecture a checkpoint uses.

---

# Controlled experiments

After first full model:

Potential experiments:
- XYZ vs XYZ+normals
- 1024 vs 2048 points
- hidden width
- conservative augmentation
- Smooth-L1 vs simple alternative
- 2–3 seeds

C evaluates all outputs.

---

# Multiple seeds

If baseline is stable:
- train same best config using 2–3 seeds
- hand checkpoint set to C/D
- C tests coordinate averaging

Do not independently invent ensemble logic.

---

# Optional local refiner

Not owned by you in the locked plan.

C owns it if the team explicitly unlocks it.

You may provide model-code advice, but do not let it interrupt global model work.

---

# First work session

1. Read master context.
2. Inspect repo.
3. Implement minimal PointNet.
4. Add `[B,N,F] → [B,85,3]` shape test.
5. Implement simple loss.
6. Implement training/checkpoint skeleton against synthetic tensors.
7. Make feature dimension configurable.
8. Wait for A's real preprocessing contract rather than inventing a second data pipeline.

---

# Definition of done

Role B is done when a documented checkpoint consumes A's canonical ear tensors and produces canonical residual predictions that beat C's canonical mean-template baseline on the frozen validation split.

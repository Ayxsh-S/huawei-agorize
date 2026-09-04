# ROLE B — MODEL & TRAINING OWNER

> Paste this file **after** `01_MASTER_AI_CONTEXT.md` into your AI.  
> Tell it: **"I am Role B. Follow the master plan and work only on Model & Training unless an interface change is necessary."**

---

# Mission

Build the smallest neural model that learns useful residual corrections from canonical ear point clouds and reliably beats the canonical-template baseline.

Do not build the full PinnaTwin-Zoom proposal.

---

# Files you primarily own

```text
src/model.py
src/losses.py
src/train.py
configs/
tests/test_model.py
```

Avoid changing Role A geometry or Role C metric code without coordination.

---

# Input contract

Expect canonical ears from Role A.

Initial model input:

```text
points: [B,2048,3]
```

Optional later:

```text
features: [B,2048,6]  # XYZ + normals
```

Targets for training:

```text
canonical_landmarks: [B,85,3]
```

---

# Template contract

Do not confuse templates.

The model uses the **canonical-ear mean template**, not the original global-frame template.

Conceptually:

```python
residual = model(points)                         # [B,85,3]
prediction = canonical_template + residual       # [B,85,3]
```

If preprocessing successfully maps both sides into one shared orientation, try one shared template.

If not, use side-specific canonical templates as supplied by the data/config layer.

Do not secretly recompute templates inside the model from validation data.

---

# First model

Use a small PointNet-style network.

Suggested:

```text
[B,N,F]
↓ shared point MLP
64
↓
128
↓
256
↓ max pool over N
[B,256]
↓ MLP
512
↓
256
↓
255
↓
[B,85,3]
```

Exact widths are flexible.

Do not introduce:
- PointNet++;
- transformer;
- graph network;
- heatmap head;
- uncertainty head;
- multiple branches.

---

# Loss

Start with Smooth-L1/Huber in canonical coordinates.

Metric is still Euclidean distance and is owned by Role C.

Optionally compare one other basic loss later.

Do not invent a complex anatomical loss in the first baseline.

---

# Required development sequence

## 1. Shape test

Random input:

```python
x = torch.randn(B, 2048, 3)
y = model(x)
assert y.shape == (B, 85, 3)
```

## 2. Tiny-data overfit

Train on 4–8 ears.

Augmentation OFF.

The network should substantially overfit.

If it does not:
- inspect scale;
- inspect targets;
- inspect template addition;
- inspect batch/landmark dimensions;
- inspect learning rate/loss.

Do not make the model huge to hide a data bug.

## 3. Full baseline

Train on frozen subject-level training split.

Log every run.

## 4. Only then cheap improvements

Try controlled changes:
- normals;
- augmentation;
- point count;
- hidden width;
- loss;
- seed.

---

# Checkpoint format

Save enough information to reproduce:
- `state_dict`;
- model config;
- point feature count;
- template choice/version;
- epoch;
- validation metric;
- seed.

Do not make Role D guess the architecture.

---

# Augmentation

Only after baseline works.

Candidate conservative augmentations:
- small rotation;
- small isotropic scaling;
- point jitter;
- point dropout/resampling.

Targets must receive the same geometric transform when necessary.

Keep ranges in config.

---

# Training machine

Use the team's agreed canonical GPU environment for official runs.

Local machine can run:
- shape tests;
- tiny subsets;
- CPU debugging.

---

# Experiment discipline

Every meaningful run goes into `EXPERIMENTS.md` via the agreed format.

One change per experiment when possible.

Never say "this checkpoint seems best" without a frozen validation number.

---

# Optional ensemble

If the baseline is stable:
- train 2–3 seeds;
- pass checkpoints to Role D/C;
- validate coordinate averaging.

Do not change architecture for each seed.

---

# Local refiner

Not your default task.

Only implement if the team explicitly unlocks Stage 5 after the global pipeline is stable.

---

# Today / first work session

1. Read `DATA_SPEC.md` if it exists.
2. Implement minimal PointNet.
3. Implement config-driven feature dimension.
4. Implement residual output `[B,85,3]`.
5. Add tensor-shape unit test.
6. Implement a minimal training loop skeleton that can consume synthetic tensors.
7. Wait for Role A's real canonical dataset interface rather than inventing a second loader.

---

# Your definition of done

Role B is done when a documented checkpoint can take Role A's canonical point tensors and produce `[85,3]` residuals that beat the canonical-template baseline on Role C's frozen validation split.

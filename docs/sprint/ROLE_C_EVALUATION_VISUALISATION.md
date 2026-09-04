# ROLE C — EVALUATION & VISUALISATION OWNER

> Paste this file **after** `01_MASTER_AI_CONTEXT.md` into your AI.  
> Tell it: **"I am Role C. Follow the master plan and work only on Evaluation & Visualisation unless an interface change is necessary."**

---

# Mission

Be the project's truth system.

Your job is to answer:
- does the pipeline actually improve Huawei's metric?
- where does it fail?
- are coordinates/order/sides correct?
- does an optional feature help or hurt?

Do not optimise models based on visual vibes alone.

---

# Files you primarily own

```text
src/evaluate.py
src/visualise.py
tests/test_evaluate.py
validation-analysis scripts if needed
```

---

# 1. Frozen subject split

Create one subject-level train/validation split.

Default:
```text
160 train subjects
40 validation subjects
seed 42
```

unless the real dataset/example code provides a reason to change it.

Both ears stay with the same subject.

Save the split.

Do not change it after seeing model performance.

---

# 2. Official metric

Implement Euclidean error.

For prediction/target:

```text
[B,85,3]
```

per-landmark distance:

```python
sqrt(sum((pred-target)**2, axis=-1))
```

Then compute the required mean.

Expose:
- global mean;
- per-ear/per-example mean;
- per-landmark errors.

Unit-test with simple hand-checkable coordinates.

---

# 3. Diagnostic metrics

Also report:
- median;
- p95;
- per contour;
- per landmark;
- worst subjects;
- left vs right if useful.

Contour index ranges must come from verified `DATA_SPEC.md`.

Do not assume indices solely from proposal prose if the actual annotation files disagree.

---

# 4. Visualisation

Build tools to plot:
- ear mesh/point cloud;
- GT landmarks;
- predicted landmarks;
- lines prediction→GT;
- landmark index optionally;
- contour colouring optionally.

Create a "worst cases" output folder.

Make it fast to inspect 20 bad ears.

---

# 5. Baseline tracking

Evaluate:

## Baseline 0
Global-frame mean left/right template.

## Baseline 1
Canonical template transformed back to each validation ear.

## Baseline 2
PointNet residual model.

Later optional:
- + normals/augmentation;
- + surface projection;
- + ensemble;
- + fixed refiner.

The project should always know which is best.

---

# 6. Surface projection experiment

Own this as an evaluation-controlled feature.

Compare:
- raw predicted coordinate;
- projection onto nearest mesh surface/vertex.

Because nearest-surface projection can snap to the wrong fold, do not automatically enable it.

Report validation delta.

---

# 7. Failure diagnosis

When a model is bad, identify likely category:

### Geometry failure
- crop truncated;
- side swapped;
- transform incorrect;
- inverse wrong.

### Model failure
- predictions clustered;
- underfitting;
- unstable landmark groups.

### Data/order failure
- landmark indices mismatched;
- contour sequence wrong.

### Projection failure
- points snap to nearby wrong fold.

Provide evidence and figures rather than vague guesses.

---

# 8. Required comparison table

Maintain something like:

| Run | System | Mean | Median | P95 | Notes |
|---|---|---:|---:|---:|---|
| B0 | Global mean | | | | |
| B1 | Canonical template | | | | |
| M1 | PointNet XYZ | | | | |
| M2 | + normals | | | | |
| M3 | + augmentation | | | | |
| E1 | ensemble | | | | |

---

# 9. Leakage watch

You are also the person most likely to catch validation leakage.

Check:
- split is by subject;
- template uses training only;
- crop constants are training-derived only;
- canonical transform on val uses val mesh only;
- no val landmarks are fed into preprocessing;
- hyperparameter decisions are based on the fixed validation set only.

Flag anything suspicious immediately.

---

# Today / first work session

1. Implement metric with tests.
2. Create/freeze subject split.
3. Implement simple landmark/mesh visualisation.
4. Prepare functions for per-example errors.
5. Coordinate with Role D on the expected prediction file/object format.

---

# Your definition of done

Role C is done when every proposed model/component can be compared objectively on one frozen validation split, and the team can inspect exactly where bad predictions occur.

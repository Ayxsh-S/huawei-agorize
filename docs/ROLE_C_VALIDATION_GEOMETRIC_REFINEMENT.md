# ROLE C — VALIDATION & GEOMETRIC REFINEMENT ENGINEER

> Give this file to your AI after `MASTER_AI_CONTEXT.md` and `DATA_SPEC.md`.

Tell the AI:

> I own Role C. Inspect the current repository before editing. Follow this role in order. Do not redesign the architecture. Do not modify another owner's modules unless a shared interface genuinely requires it. Do not commit or push.

---

# Mission

Build the project's quantitative truth system and all low-risk geometry-based improvements around the learned model.

This is an implementation role, not passive evaluation.

You own:
- baselines
- templates
- metrics
- analysis
- surface projection
- prediction averaging
- optional local refinement if unlocked

---

# Primary ownership

Suggested:

```text
src/evaluate.py
src/visualise.py
src/postprocess.py
src/templates.py      # if useful
tests/test_evaluate.py
```

Exact filenames may follow the existing repo.

Do not duplicate A's transforms or B's training loop.

---

# 1. Frozen subject-level split

Create and save one split.

Default:

```text
160 train
40 validation
seed 42
```

unless actual dataset/example code suggests otherwise.

Both ears from the same person stay together.

Never:

```text
subject X left → train
subject X right → validation
```

Freeze split before model selection.

---

# 2. Official metric

Implement Euclidean landmark error for:

```text
pred   [B,85,3]
target [B,85,3]
```

Expose:
- overall mean
- per-ear/per-example mean
- per-landmark error

Add hand-checkable unit tests.

---

# 3. Diagnostic metrics

Also implement:
- median
- p95
- per-contour
- per-landmark
- worst subjects
- optionally left vs right

Read verified contour indices from `DATA_SPEC.md`.

Do not invent them if not yet verified.

---

# 4. Template 0 — global-frame mean

YOU own it.

From TRAINING annotations only:

```python
mean_left_global
mean_right_global
```

Evaluate on validation.

This is the first non-neural baseline and emergency fallback.

Save/document it according to NDA/repo rules.

---

# 5. Template 1 — canonical-ear mean

YOU own it.

Once A's mesh-derived transform is frozen:

For each TRAINING ear:
- ask/use A's preprocessing/transform
- transform training GT landmarks into canonical coordinates
- average them

If A's mirroring creates a reliable shared orientation:
- test one shared canonical template

Otherwise:
- test side-specific canonical templates

Use validation/visual evidence.

B consumes the selected template.

Do not rewrite A's transform logic.

---

# 6. Canonical-template baseline

Evaluate:

```text
validation mesh
→ A preprocessing
→ canonical template
→ A inverse transform
→ original coordinate prediction
```

This is Baseline 1.

Compare against global mean.

---

# 7. Visualisation

Build fast tooling for:
- mesh/point cloud
- GT landmarks
- predicted landmarks
- error lines
- landmark indices
- contours if verified
- worst-case gallery

The team should be able to inspect 20 bad ears quickly.

---

# 8. Failure analysis

Classify evidence into likely:
- crop/transform problem
- side/mirror problem
- landmark-order problem
- model underfit
- catastrophic outlier
- postprocessing failure

Give A/B/D reproducible cases rather than vague impressions.

---

# 9. Surface projection

YOU own this optional improvement.

Compare:

```text
raw coordinate prediction
```

vs

```text
nearest valid target-ear surface projection
```

Projection can hurt if it snaps to the wrong fold.

Enable only if held-out MED improves.

---

# 10. Test-time multi-sample averaging

Once B model works:

Run several point samples for same ear:

```text
sample 1 → prediction
sample 2 → prediction
sample 3 → prediction
→ average coordinates
```

Measure benefit.

Own the evaluation/averaging logic.

Coordinate with D for final inference integration if retained.

---

# 11. Multi-seed checkpoint averaging

If B provides multiple compatible checkpoints:

```text
checkpoint 1 → prediction
checkpoint 2 → prediction
checkpoint 3 → prediction
→ average
```

Measure validation delta.

Only retain if helpful.

---

# 12. Optional fixed local refiner — stretch only

This belongs to you IF the team explicitly unlocks it.

Unlock only if:
- A stable
- B stable
- D end-to-end stable
- global model beats template
- enough time remains

Initial design:

```text
coarse landmark
→ K=256 nearest ear points
→ local coordinates/features
→ tiny shared local PointNet
→ XYZ correction
```

No:
- adaptive patch size
- uncertainty router
- bilateral risk
- contour-risk router

Keep only if validation improves.

---

# Baseline/experiment table

Maintain quantitative comparison such as:

| ID | System | Mean | Median | P95 | Decision |
|---|---|---:|---:|---:|---|
| B0 | Global mean | | | | |
| B1 | Canonical template | | | | |
| M1 | PointNet XYZ | | | | |
| M2 | + normals | | | | |
| P1 | + projection | | | | |
| E1 | multi-sample avg | | | | |
| E2 | multi-seed avg | | | | |
| R1 | fixed refiner | | | | |

---

# Leakage watch

You are also responsible for catching validation leakage.

Check:
- split by subject
- global template train-only
- canonical template train-only
- A transform on val uses val mesh only
- no val landmark-dependent preprocessing
- model selection uses frozen validation set

Flag anything suspicious immediately.

---

# First work session

1. Read master context.
2. Inspect repo/data naming info.
3. Create/freeze subject split.
4. Implement official metric with unit tests.
5. Implement global mean template baseline.
6. Evaluate it.
7. Build simple prediction-vs-GT plotting.
8. Create template utility interfaces that can later consume A's transforms.

You have meaningful code from day 1.

---

# Definition of done

Role C is done when:
- the team has trustworthy baselines and metrics
- canonical template is available to B
- every model change can be objectively compared
- useful post-processing/ensembling is validated rather than guessed
- optional refiner is either proven useful or cleanly dropped

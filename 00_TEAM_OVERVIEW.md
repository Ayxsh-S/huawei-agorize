# PinnaTwin-Zoom — Team Sprint Overview

## Read this part only if you are a human and short on time

**Challenge:** Huawei Munich Tech Arena 2026 — Automatic Extraction of Anatomical Landmarks of Pinna Shape  
**Team:** Imperial_Huawei  
**Deadline:** 15 September 2026 (check the logged-in portal for the exact cutoff time)  
**Dataset:** 200 subjects, full head/upper-torso PLY meshes, separate left/right `85 x 3` pinna landmark annotations  
**Metric:** mean Euclidean distance between predicted and ground-truth landmarks, averaged over both ears and hidden subjects  
**Team capacity:** 4 people, roughly 3–4 focused hours each per day  
**Goal:** do **not** implement the full research proposal. Ship the simplest reliable end-to-end product first, then reduce validation error.

---

# What we are actually building

The final product must do this:

```text
raw Huawei head PLY
        ↓
find/crop left and right ear
        ↓
mesh-only canonical transform
        ↓
sample point cloud
        ↓
small PointNet
        ↓
canonical mean template + predicted residuals
        ↓
optional surface projection
        ↓
inverse transform
        ↓
left  85 x 3
right 85 x 3
in original Huawei coordinates
```

We maintain an even simpler emergency fallback:

```text
mean left landmarks in Huawei global coordinates
mean right landmarks in Huawei global coordinates
```

So we should have a valid prediction method before the neural model is finished.

---

# What we are NOT building unless everything else is already finished

Do **not** spend sprint time on:

- full PinnaTwin bilateral-risk routing;
- ContourGuard inference routing;
- heatmap uncertainty;
- adaptive 128/256/512-point risk patches;
- learned risk weights;
- confidence gating;
- PointNet++ migration;
- a complicated multi-stage research architecture.

The original proposal remains the research direction. The sprint implementation is the minimum viable version.

---

# Four roles

## Role A — Data & Geometry
Owns:
- PLY/annotation loading;
- ear crops;
- mirroring/canonicalisation;
- point sampling/normals;
- transform metadata;
- inverse transforms;
- preprocessing cache;
- crop sanity checks.

## Role B — Model & Training
Owns:
- small PointNet;
- residual prediction;
- loss;
- training/validation loop;
- checkpoints;
- augmentation;
- training configs;
- model experiments.

## Role C — Evaluation & Visualisation
Owns:
- frozen subject-level train/validation split;
- official mean Euclidean error;
- median/p95/per-contour/per-landmark error;
- prediction visualisation;
- worst-case analysis;
- optional surface-projection experiment;
- comparison tables.

## Role D — Integration & Submission
Owns:
- interfaces between A/B/C;
- end-to-end `predict_subject(...)`;
- daily `main` smoke test;
- experiment log;
- configuration;
- final output format;
- README/reproducibility;
- final package/submission.

**Roles are equal. Role D integrates; Role D is not the boss.**

---

# The two template types — do not confuse them

## Template 0: global-frame mean
Average training landmarks directly in Huawei's aligned global XYZ frame.

Purpose:
- immediate non-neural baseline;
- emergency fallback;
- first validation number.

## Template 1: canonical-ear mean
After Role A freezes a **mesh-derived** ear transform, transform TRAINING landmarks using the same transform and compute the canonical mean.

Purpose:
- residual base for PointNet.

At inference:

```text
test mesh
→ mesh-derived transform only
→ canonical ear
→ canonical template + PointNet residual
→ inverse transform
```

**Never use validation/test ground-truth landmarks to choose the crop, centre, scale, rotation, mirror transform, or any preprocessing parameter.**

---

# Critical NDA rule

The full Huawei dataset is NDA-protected.

Do not put raw PLYs, annotations, processed caches, or checkpoints in GitHub.

Until the NDA terms are checked, do not assume raw NDA data can be pasted/uploaded into arbitrary external AI chats. Give AI:
- code;
- schemas;
- file names;
- tensor shapes;
- coordinate conventions;
- stack traces;
- logs;
- aggregate statistics;
- synthetic examples.

---

# Daily outcome we care about

## Weekend
Repository works; Huawei notebook understood; `DATA_SPEC.md` written; mean-template baseline exists.

## Monday
Safe ear preprocessing + exact inverse transform work.

## Tuesday
Small PointNet can overfit 4–8 ears.

## Wednesday
First full learned pipeline:
`raw PLY -> left/right 85x3`.

## Thursday–Friday
Improve the number with cheap changes only.

## Saturday
Only if stable: consider fixed local refiner.

## Sunday
Choose final model.

## Monday 14 Sept
Code freeze, reproduce from scratch, package/upload.

## Tuesday 15 Sept
Emergency/submission only.

---

# One rule for the whole sprint

Before adding any feature, ask:

> Does this measurably improve held-out mean Euclidean error or make the existing end-to-end system more reliable before the deadline?

If the answer is unclear, **do not build it**.

---

# What each teammate should do

1. Clone the same repo.
2. Read/paste `01_MASTER_AI_CONTEXT.md` into their AI.
3. Paste their own role file too.
4. Tell the AI: **"I am Role A/B/C/D. Follow the master plan. Inspect the current repo before editing. Do not redesign the architecture."**
5. Work only on the owned deliverables.
6. Merge at least daily.
7. Tell the team immediately if a shared interface changes.
8. Humans perform commits/pushes; AI does not.

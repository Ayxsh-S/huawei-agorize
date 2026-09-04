# PinnaTwin-Zoom — Final Team Sprint Overview

## Read this if you are human and short on time

**Challenge:** Huawei Munich Tech Arena 2026 — Automatic Extraction of Anatomical Landmarks of Pinna Shape  
**Team:** Imperial_Huawei  
**Deadline:** 15 September 2026  
**Dataset:** 200 subjects, full head/upper-torso PLY meshes, separate left/right `85 x 3` pinna landmark annotations  
**Metric:** mean Euclidean distance over predicted landmarks, ears and hidden subjects  
**Working constraint:** 4 people, roughly 3–4 focused hours/person/day, heavy use of AI coding assistants  
**Goal:** ship a working end-to-end product first; only then improve accuracy.

---

# What we are actually building

```text
raw Huawei head PLY
        ↓
A — crop/canonicalise/sample ears
        ↓
B — PointNet residual landmark prediction
        ↓
C — validate/post-process/ensemble if useful
        ↓
D — end-to-end inference + output
        ↓
left  85 x 3
right 85 x 3
in original Huawei coordinates
```

We are **not** trying to implement the full original PinnaTwin-Zoom research proposal.

---

# Final engineering roles

## A — 3D Preprocessing & Geometry Engineer — Alfred

Builds:

- PLY loading
- training annotation loading
- left/right ear cropping
- mirroring/canonicalisation
- mesh-only centring/scaling
- point sampling
- normals if useful
- transform metadata
- exact inverse transforms
- processed data cache
- crop sanity checks

Primary contract:

```text
raw PLY
→ canonical ear tensor + invertible transform
```

This is the highest-risk foundation. If A is wrong, every downstream model can silently produce garbage.

---

## B — Global Landmark Model Engineer — Ojas

Builds:

- small PointNet
- residual landmark prediction
- losses
- training loop
- augmentation
- checkpoints
- GPU experiments
- model configs

Primary contract:

```text
canonical ear tensor
→ 85 x 3 canonical landmark prediction
```

The model predicts residuals from C's canonical mean template.

---

## C — Validation & Geometric Refinement Engineer

Builds:

- subject-level train/validation split
- Huawei metric
- global-frame mean template
- canonical-ear mean template
- per-landmark/per-contour metrics
- visualisation/worst-case analysis
- nearest-surface projection
- test-time multiple point-sample averaging
- multi-seed checkpoint averaging
- optional fixed local refiner if explicitly unlocked

Primary contract:

```text
predictions
→ reliable metrics + validated post-processing
```

C owns **both templates**. B consumes the canonical template; A supplies the transforms needed to construct it.

---

## D — Inference & Pipeline Engineer

Builds:

- `predict_subject(...)`
- checkpoint/template/config loading
- left/right orchestration
- output writer
- CLI
- integration tests
- daily main smoke test
- reproducibility path
- README/inference instructions
- final executable/package

Primary contract:

```text
raw PLY
→ left (85,3) + right (85,3)
```

D starts on day 1 using stubs/mean-template output and progressively swaps in A/B/C modules.

---

# Two templates — do not confuse them

## Template 0 — Global-frame mean template

Average the TRAINING left/right landmark coordinates directly in Huawei's aligned global coordinate frame.

Purpose:
- immediate non-neural baseline
- emergency fallback
- first validation number

Owner: **C**

---

## Template 1 — Canonical-ear mean template

After A freezes a mesh-derived canonical transform:

```text
training mesh
→ A's mesh-derived transform
→ transform TRAINING landmarks
→ average 85 canonical landmark positions
```

Purpose:
- residual base for B's PointNet

Owner: **C**

B loads this template.  
A does not own template statistics.  
D does not recompute it.

---

# Critical no-leakage rule

Validation/test ground-truth landmarks must NEVER be used to:

- locate the ear
- choose a crop
- calculate centre
- calculate scale
- rotate
- mirror
- define canonicalisation
- select points
- derive any inference transform

At validation/test time the pipeline may use only:

- the supplied mesh
- frozen training-derived constants/configuration

Ground-truth landmarks may be used as target coordinates and to test transform mathematics during development, but must not define the transform itself.

---

# What we are cutting

Unless the baseline is already stable and the whole team explicitly unlocks them, do not implement:

- PointNet++
- 85 heatmap outputs
- uncertainty routing
- ContourGuard routing
- bilateral PinnaTwin routing
- adaptive 128/256/512 patches
- learned risk weights
- confidence gating
- complicated multi-stage research architecture

Optional stretch only:
- fixed local refiner, owned by C if unlocked

---

# Shared ownership — no duplication

| Responsibility | Owner |
|---|---|
| PLY/annotation loading | A |
| Crop/canonicalisation/mirroring | A |
| Point sampling/normals | A |
| Transform/inverse/cache | A |
| PointNet architecture | B |
| Loss/augmentation/training/checkpoints | B |
| Subject split | C |
| Global mean template | C |
| Canonical mean template | C |
| Metric/error analysis | C |
| Surface projection | C |
| TTA/multi-sample averaging | C |
| Multi-seed prediction averaging | C |
| Fixed local refiner if unlocked | C |
| `predict_subject()` | D |
| CLI/config/loading orchestration | D |
| Prediction output format | D |
| Integration/smoke tests | D |
| README/final runnable package | D |

---

# Day-1 independent work

## A
- open Huawei example notebook
- load one PLY + annotations
- plot both ears with landmarks
- document file/coordinate conventions
- prototype first crop

## B
- implement PointNet residual model
- random tensor forward test
- synthetic training-loop skeleton

## C
- freeze subject split
- implement metric with tests
- compute/evaluate global mean baseline
- start template utilities

## D
- create `predict_subject(mesh_path, ...)` skeleton
- return global mean template initially
- build CLI/output writer
- add one smoke test

All four people code from the start.

---

# Daily sprint milestones

## Weekend
- repo/context ready
- Huawei notebook understood
- `DATA_SPEC.md` started
- global mean baseline exists

## Monday
- A's safe crop/canonical transform/inverse works

## Tuesday
- A's cache exists
- B can overfit 4–8 ears
- C can build canonical template
- D has an inference path

## Wednesday
Critical milestone:

```text
raw PLY
→ A preprocessing
→ B model
→ inverse transform
→ left/right 85 x 3
```

## Thursday–Friday
Only cheap, validated improvements.

## Saturday
If global pipeline is stable, C may attempt fixed local refiner.

## Sunday
Choose final model.

## Monday 14 Sept
Code freeze and reproducibility.

## Tuesday 15 Sept
Emergency/submission only.

---

# NDA / repository rule

Do not commit Huawei raw or derived confidential data.

At minimum ignore:

```gitignore
data/
outputs/
checkpoints/
*.ply
*.npz
*.npy
*.pt
*.pth
```

Until the NDA terms are confirmed, do not assume raw Huawei data can be uploaded into arbitrary external AI chats.

---

# What each person gives their AI

Each AI should read:

1. `MASTER_AI_CONTEXT.md`
2. that person's role file
3. `DATA_SPEC.md`
4. current relevant repo files

Then tell the AI:

> I own Role X. Follow the master plan exactly. Inspect the current repository before editing. Do not redesign the architecture. Do not modify another owner's module unless a shared interface genuinely requires it. Do not commit or push. Start with my role's first-work-session tasks.

---

# Team rule

Before adding anything, ask:

> Does this measurably improve held-out mean Euclidean error or reduce the risk that the end-to-end pipeline fails before the deadline?

If not, do not build it.

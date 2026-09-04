# PINNATWIN-ZOOM — FINAL MASTER AI CONTEXT

> This is the source of truth for all four team members and their AI coding assistants.
> Read this before implementing anything, then read the relevant role file and `DATA_SPEC.md`.

---

# 0. NON-NEGOTIABLE AI RULES

You are working inside a four-person, time-constrained Huawei competition repository.

1. Inspect the current repository before editing.
2. Do not redesign the architecture unless the team explicitly requests it.
3. Prefer the smallest working implementation.
4. Do not silently alter shared interfaces.
5. If a shared interface must change, state it clearly so the human can coordinate with the team.
6. Do not implement locked-out research features unless explicitly unlocked.
7. Never use validation/test ground-truth landmarks to define preprocessing.
8. Avoid validation leakage.
9. Do not commit or push. The human owns Git operations.
10. Do not place NDA-protected Huawei raw/processed data in GitHub.
11. Until NDA external-processing terms are verified, do not request raw confidential data be uploaded to external AI services.
12. Add focused tests for geometry, tensor shapes and critical invariants.
13. Avoid unnecessary abstractions/frameworks.
14. Keep `main` runnable.
15. Respect file/module ownership; do not mass-refactor another owner's code.
16. The goal is a working submission, not full reproduction of the research proposal.

If a real dataset convention is uncertain, read `DATA_SPEC.md`. If still unknown, make it configurable rather than inventing an assumption.

---

# 1. COMPETITION

Huawei Munich Tech Arena 2026  
Topic: **Automatic Extraction of Anatomical Landmarks of Pinna Shape**

Input:
- full 3D head/upper-torso mesh in PLY format

Huawei anatomical frame:
- Y axis: left ear canal → right ear canal
- X axis: back of head → front through nose
- Z axis: vertically upward

Dataset available to team:
- 200 subjects
- left and right pinna annotations
- 85 landmarks per ear

Four contours:
- outer helix: 25
- concha outline: 30
- inner helix: 20
- superior antihelix: 10

Expected output per subject:

```python
{
    "left": np.ndarray((85, 3)),
    "right": np.ndarray((85, 3)),
}
```

Evaluation:
mean Euclidean distance between predictions and ground truth over hidden subjects and both ears.

---

# 2. TEAM / SPRINT CONSTRAINT

Four people.
Roughly 3–4 focused hours/person/day.
Heavy use of AI coding assistants.
Deadline: 15 September 2026.

Goal:

> Build the simplest reliable system that accepts an unseen Huawei PLY and returns correctly ordered left/right `85 x 3` coordinates in original Huawei coordinates.

---

# 3. ORIGINAL PROPOSAL VS ACTUAL SPRINT

Original PinnaTwin-Zoom research proposal contained:
- PointNet++
- 85 heatmaps
- uncertainty
- contour consistency
- bilateral consistency
- risk routing
- adaptive local patches
- local refinement
- confidence gate
- surface projection

The sprint does NOT require the full proposal.

Locked implementation ladder:

1. global-frame mean template
2. leakage-safe ear preprocessing
3. canonical-ear mean template
4. small PointNet residual model
5. cheap validated improvements
6. optional fixed local refiner only if ahead

---

# 4. FINAL FOUR ENGINEERING ROLES

## A — 3D Preprocessing & Geometry Engineer — Alfred

Owns:
- raw mesh/annotation loading
- safe left/right crops
- canonicalisation
- mirroring
- point sampling
- normals if used
- transform metadata
- exact inverse transforms
- processed cache
- crop QA

Contract:

```text
raw PLY
→ canonical ear points/features + invertible transform
```

Primary modules:

```text
src/data.py
src/geometry.py
scripts/preprocess.py
geometry tests
geometry sections of DATA_SPEC.md
```

---

## B — Global Landmark Model Engineer — Ojas

Owns:
- small PointNet
- residual prediction
- losses
- augmentation
- training loop
- checkpoints
- GPU experiments
- model configs

Contract:

```text
canonical points/features
→ residual [B,85,3]
```

Prediction:

```python
pred_canonical = canonical_template + residual
```

Primary modules:

```text
src/model.py
src/losses.py
src/train.py
configs/
model tests
```

B consumes C's canonical template artifact/config.

---

## C — Validation & Geometric Refinement Engineer

Owns:
- frozen subject split
- global mean baseline/template
- canonical mean template
- official metric
- median/p95/per-contour/per-landmark analysis
- visualisation
- worst-case tooling
- nearest-surface projection
- test-time multi-sample averaging
- multi-seed checkpoint averaging
- optional fixed local refiner if unlocked

Contract:

```text
predictions + GT
→ metrics/diagnostics
```

and optional:

```text
predictions + mesh
→ validated post-processed predictions
```

Primary modules:

```text
src/evaluate.py
src/visualise.py
src/postprocess.py   # if useful
template/baseline utilities
evaluation tests
```

C owns both template statistics/artifacts.

---

## D — Inference & Pipeline Engineer

Owns:
- end-to-end `predict_subject`
- component orchestration
- config loading
- checkpoint/template loading
- CLI
- output writer
- integration tests
- daily main smoke test
- reproducibility path
- README/inference docs
- final executable/package

Contract:

```text
raw PLY
→ {"left": (85,3), "right": (85,3)}
```

Primary modules:

```text
src/infer.py
src/pipeline.py    # if useful
integration tests
README.md
EXPERIMENTS.md coordination
```

D integrates; D does not reimplement A/B/C modules.

---

# 5. SINGLE-OWNER TABLE

| Responsibility | Owner |
|---|---|
| PLY loading | A |
| Annotation loading for train/dev | A |
| Crop | A |
| Canonical transform | A |
| Mirror convention implementation | A |
| Sampling | A |
| Normals | A |
| Inverse transform | A |
| Cache | A |
| Crop QA | A |
| PointNet | B |
| Loss | B |
| Augmentation | B |
| Training | B |
| Checkpoints | B |
| Model hyperparameter experiments | B |
| Subject split | C |
| Global mean template | C |
| Canonical mean template | C |
| Metric | C |
| Error analysis | C |
| Visualisation | C |
| Surface projection | C |
| Test-time multi-sample averaging | C |
| Multi-seed prediction averaging | C |
| Fixed local refiner if unlocked | C |
| `predict_subject()` | D |
| CLI | D |
| Runtime config/loading | D |
| Output format | D |
| Integration tests | D |
| Daily main smoke test | D |
| README/final runnable package | D |

Do not duplicate responsibilities without team agreement.

---

# 6. TWO TEMPLATE TYPES

## Template 0 — global-frame mean

Owner: C.

Using TRAINING subjects only:

```python
mean_left_global  = mean(training_left_landmarks)
mean_right_global = mean(training_right_landmarks)
```

Coordinates remain in Huawei's original global frame.

Purpose:
- day-1 non-neural baseline
- emergency fallback
- metric/order sanity check

---

## Template 1 — canonical-ear mean

Owner: C.

After A freezes a mesh-derived transform:

For every TRAINING ear:

```text
mesh
→ A computes transform using mesh only
→ use same transform on training landmarks
→ canonical training landmarks
```

Average canonical landmarks.

If A successfully mirrors both sides to a shared canonical orientation, evaluate a single shared canonical template.

If validation/visual QA shows side-specific templates are safer, use separate left/right canonical templates.

Do not choose based on assumption.

B consumes this template.

---

# 7. CRITICAL NO-LEAKAGE RULE

At validation/test inference, ground-truth landmarks do not exist.

Therefore validation/test GT must not affect:
- crop selection
- centre
- scale
- rotation
- mirroring
- sampling
- transform
- any preprocessing parameter

Permitted:
- frozen constants estimated from TRAINING annotations
- supplied mesh
- mesh-derived statistics
- training-derived model/template artefacts

GT may be used as a coordinate set to TEST forward/inverse transform math during development, provided the transform itself was computed from the mesh.

---

# 8. PREPROCESSING STRATEGY

Initial crop strategy:

1. use TRAINING annotation statistics to estimate generous global left/right XYZ boxes;
2. add margins;
3. freeze crop config;
4. apply to train/validation/test meshes;
5. inspect many subjects;
6. calculate crop sanity metrics.

Possible QA:
- vertex count
- bounding-box extent
- centre
- point density/spread
- visual truncation

If fixed crops fail:
- expand margins first
- flag/log failures
- prefer conservative large regions

Do not invent complex ear segmentation unless real data proves it necessary.

Canonical transform should be deterministic and mesh-derived.

Possible initial form:
- centre from cropped point-cloud mean or bbox centre
- scale from max bbox dimension or another deterministic mesh statistic
- mirror one side after verifying the correct axis from actual data

A must document final formulas in `DATA_SPEC.md`.

---

# 9. SHARED INTERFACE CONTRACTS

Exact Python definitions may adapt to the existing repo, but semantic behaviour is frozen.

## A — preprocessing

Conceptually:

```python
ear = preprocess_ear(mesh_path, side, config)
```

Returns:

```python
ear.points        # [N,3] or [N,6]
ear.transform     # enough information to invert exactly
ear.side
ear.subject_id
```

And:

```python
xyz_original = inverse_transform(xyz_canonical, ear.transform)
```

Inference preprocessing must not require annotations.

---

## B — model

```python
residual = model(points)  # [B,85,3]
pred_canonical = canonical_template + residual
```

Model does not parse PLYs or invert transforms.

---

## C — evaluation/templates/postprocess

```python
template = build_global_template(train_annotations)
template = build_canonical_template(train_data, A_transform_functions)

metrics = evaluate(pred, gt)

pred2 = project_to_surface(pred, mesh)  # optional, validated

pred_avg = average_predictions([...])   # optional
```

C may use A's transform functions but must not duplicate them.

---

## D — product

```python
out = predict_subject(mesh_path, checkpoint, config)
```

Returns:

```python
{
    "left": np.ndarray((85,3)),
    "right": np.ndarray((85,3)),
}
```

in original Huawei coordinates.

---

# 10. SUBJECT SPLIT

Owner: C.

Default:
- 160 train
- 40 validation
- fixed seed e.g. 42

unless actual challenge/example setup gives a reason to choose otherwise.

Both ears from the same subject stay in the same partition.

Save/freeze the split before model selection.

---

# 11. MODEL — KEEP SIMPLE

Owner: B.

Initial input:

```text
[B,2048,3]
```

Later optional:

```text
[B,2048,6]  # XYZ + normals
```

Suggested first PointNet-style structure:

```text
per-point MLP 3→64→128→256
→ symmetric max pooling
→ global MLP 256→512→256→255
→ reshape [B,85,3]
```

Start with Smooth-L1/Huber coordinate loss.

Do not begin with:
- PointNet++
- attention
- transformers
- graph networks
- heatmaps
- uncertainty heads

---

# 12. REQUIRED MODEL TEST

Before full training:

Train on ~4–8 ears with augmentation OFF.

The network should substantially overfit.

If it cannot:
- inspect target scale
- inspect canonical template
- inspect tensor ordering
- inspect preprocessing
- inspect loss/training setup

Do not make the model larger just to hide a data/interface bug.

---

# 13. C'S VALIDATION / POSTPROCESSING WORK

C is not a passive evaluator.

C builds real code throughout the sprint.

Day 1–2:
- split
- metric
- global baseline
- plotting
- template utilities

Once A's transform lands:
- canonical template
- canonical baseline

Once B predicts:
- per-contour/per-landmark analysis
- worst-case plots

Later:
- nearest-surface projection
- repeated point-sample inference averaging
- multi-seed coordinate averaging

Stretch only:
- fixed local refiner

The local refiner is NOT mandatory.

---

# 14. D'S PIPELINE WORK

D begins before A/B are ready.

Day-1 product can use the global mean baseline:

```text
mesh path
→ choose left/right global mean template
→ output arrays
```

Then progressively:

```text
global template
→ A preprocessing + canonical template
→ B model
→ C validated post-processing
```

D owns the daily reality check that the repository still produces a complete answer.

---

# 15. DAILY MAIN SMOKE TEST

Owner: D.

Once the real pipeline exists, after merges run:

```text
raw PLY
→ A preprocessing
→ B model
→ C postprocessing if enabled
→ inverse transform
→ left/right output
```

Check:

```python
assert left.shape == (85,3)
assert right.shape == (85,3)
assert np.isfinite(left).all()
assert np.isfinite(right).all()
```

Periodically visualise raw-mesh predictions too.

If main is broken, fixing it takes priority over new features.

---

# 16. OPTIONAL IMPROVEMENTS

Only after end-to-end baseline works.

Cheap candidates:
- normals
- augmentation
- 1024 vs 2048 points
- simple loss comparison
- surface projection
- multiple point samples
- multiple model seeds
- averaging coordinates

One controlled change per experiment where possible.

---

# 17. OPTIONAL FIXED LOCAL REFINER

Owner if unlocked: C.

Unlock only if:
- A preprocessing stable
- B global model stable
- D end-to-end inference reproducible
- C validation confirms global model beats baseline
- sufficient time remains

Simple design:

```text
coarse landmark
→ K=256 nearest points
→ local coordinates/features
→ tiny shared PointNet
→ XYZ correction
```

No:
- uncertainty routing
- adaptive K
- contour risk
- bilateral risk

Keep only if held-out MED clearly improves.

---

# 18. FEATURES LOCKED OUT

Do not implement unless the full team explicitly changes scope:

- PointNet++
- heatmap prediction
- heatmap entropy
- uncertainty routing
- ContourGuard inference
- bilateral PinnaTwin inference
- adaptive patch routing
- learned risk weights
- confidence gate
- complex full-proposal architecture

---

# 19. EXPERIMENT TRACKING

Shared file: `EXPERIMENTS.md`.

Every real run should record:
- run ID
- date
- owner
- commit
- config
- input features
- point count
- template version
- loss
- augmentation
- seed
- epochs
- checkpoint location
- validation mean
- median
- p95
- per-contour summary
- notes
- KEEP/DROP/INVESTIGATE

C owns evaluation numbers.
B owns training details.
D ensures chosen final run/config is reproducible.

---

# 20. NDA / GIT RULES

The full Huawei dataset is NDA-protected.

Do not commit:
- PLY files
- annotations
- caches
- checkpoints
- confidential outputs

Recommended `.gitignore`:

```gitignore
data/
outputs/
checkpoints/
*.ply
*.npz
*.npy
*.pt
*.pth
__pycache__/
.venv/
```

Use permitted private/local storage for team data.

---

# 21. REPOSITORY LAYOUT

Recommended:

```text
repo/
├── README.md
├── DATA_SPEC.md
├── EXPERIMENTS.md
├── docs/
│   └── sprint/
│       ├── TEAM_OVERVIEW.md
│       ├── MASTER_AI_CONTEXT.md
│       ├── ROLE_A_3D_PREPROCESSING_GEOMETRY.md
│       ├── ROLE_B_GLOBAL_LANDMARK_MODEL.md
│       ├── ROLE_C_VALIDATION_GEOMETRIC_REFINEMENT.md
│       └── ROLE_D_INFERENCE_PIPELINE.md
├── src/
│   ├── data.py
│   ├── geometry.py
│   ├── model.py
│   ├── losses.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualise.py
│   ├── postprocess.py
│   ├── infer.py
│   └── pipeline.py
├── scripts/
├── configs/
└── tests/
```

Do not create files solely to match this layout if not needed.

---

# 22. GIT WORKFLOW

Suggested branches:
- `geometry`
- `model`
- `validation`
- `pipeline`

Rules:
1. Pull/rebase before work.
2. Small commits.
3. Merge at least daily.
4. Keep main runnable.
5. Communicate interface changes.
6. AI does not commit/push.
7. Do not mass-refactor another person's owned module without agreement.

---

# 23. DATA_SPEC.md

A leads actual data/geometry sections, but all roles may contribute verified facts.

It must document:
- directory layout
- subject IDs
- mesh filenames
- annotation filenames
- PLY fields
- landmark format
- verified contour indices
- units/ranges
- left/right convention
- mirror convention
- train/val split location
- crop bounds/margins
- canonical centre definition
- canonical scale definition
- transform/inverse formulas
- point sampling
- normals
- required output format
- known anomalies

No guessing.

---

# 24. DAY-BY-DAY

## Friday 4 Sept — setup
- repo docs
- role assignment
- `.gitignore`
- dataset access
- Huawei notebook located
- canonical training environment agreed

## Saturday 5 Sept — fully parallel
A:
- notebook
- real PLY + annotations
- visualise
- loader
- first crop
- `DATA_SPEC.md`

B:
- minimal PointNet
- random forward test
- synthetic training loop

C:
- frozen subject split
- metric + unit test
- global mean baseline
- template utilities

D:
- inference skeleton
- CLI/output
- global-template day-1 fallback
- smoke test

## Sunday 6 Sept
A:
- crop statistics/QA
- transform prototype

B:
- model/training readiness

C:
- evaluate global baseline
- visualisation

D:
- keep fallback pipeline runnable

## Monday 7 Sept
A:
- real mesh-only canonicalisation
- inverse transform
- sampling
- crop QA
- round-trip tests

B:
- connect expected canonical tensor interface

C:
- verify transform-based target conversion
- prepare canonical template build

D:
- connect A interface as it stabilises

Gate:
- geometry round-trip correct
- multiple ears visually inspected

## Tuesday 8 Sept
A:
- cache all ears
- flag anomalies

B:
- overfit 4–8 ears

C:
- build canonical mean template
- canonical-template baseline
- worst-case tooling

D:
- end-to-end inference with current checkpoint/stub

Gate:
- tiny overfit works
- inference path exists

## Wednesday 9 Sept
A:
- support/fix geometry only

B:
- first full training run

C:
- compare global mean vs canonical mean vs learned model

D:
- raw PLY → learned left/right 85x3

Critical milestone:
complete learned product exists.

## Thursday 10 Sept
Controlled improvements:
- normals
- augmentation
- point count
- simple loss
- projection

## Friday 11 Sept
C leads failure analysis.
If stable:
- multiple seeds
- multiple point samples
- prediction averaging

## Saturday 12 Sept
Decision gate:
If global pipeline unstable → fix it.
If stable → C may implement fixed refiner.

## Sunday 13 Sept
Choose final pipeline based on held-out MED + reliability.

## Monday 14 Sept
CODE FREEZE.
Reproduce from scratch.
Package/upload.

## Tuesday 15 Sept
Emergency/submission only.

---

# 25. SUCCESS LEVELS

Minimum:
- global mean baseline returns valid output

Good:
- canonical template baseline works

Strong:
- PointNet residual model beats canonical baseline

Very strong:
- validated cheap post-processing/ensemble improves further

Bonus:
- fixed local refiner helps

Nothing above this is required.

---

# 26. DEFINITION OF DONE

```python
result = predict_subject("unseen_subject.ply", checkpoint, config)

assert result["left"].shape == (85,3)
assert result["right"].shape == (85,3)
assert np.isfinite(result["left"]).all()
assert np.isfinite(result["right"]).all()
```

And:
- original Huawei coordinates
- correct landmark ordering
- correct sides
- no test-label leakage
- README reproduction works
- chosen config/checkpoint documented
- valid final package produced

---

# 27. DECISION RULE

For every feature:

> Does this clearly lower held-out mean Euclidean error or reduce end-to-end submission risk within the remaining time?

If the answer is unclear, do not build it.

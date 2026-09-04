# PINNATWIN-ZOOM — MASTER AI CONTEXT AND SPRINT PLAN

> **Purpose of this file:** This is the single source of truth for all four teammates and their AI coding assistants.  
> Every AI working on this project should receive this file before implementing anything.  
> The AI should then also receive the relevant role-specific file.

---

# 0. NON-NEGOTIABLE AI INSTRUCTIONS

You are helping one member of a four-person engineering team implement a time-constrained Huawei competition project.

Follow these rules:

1. **Do not redesign the project unless explicitly asked by the team.**
2. **Inspect the existing repository before proposing edits.**
3. Prefer the **smallest working implementation** over a sophisticated one.
4. Do not silently change shared function signatures.
5. If a shared interface genuinely needs to change, state the change clearly so the human can tell the other teammates.
6. Do not implement features marked `CUT` or `LOCKED OUT` unless the team explicitly unlocks them.
7. Do not use ground-truth validation/test landmarks to define preprocessing.
8. Do not create validation leakage.
9. Do not commit or push Git changes. The human does all commits and pushes.
10. Raw Huawei NDA data must not be copied into the repository.
11. Until the team verifies the NDA allows it, do not request that raw Huawei meshes/annotations be uploaded to external AI systems.
12. Add focused tests for geometry, tensor shapes and key invariants.
13. Keep code readable and boring. Avoid unnecessary frameworks or abstractions.
14. Preserve a runnable `main` branch.
15. The sprint objective is **shipping a working end-to-end system**, not reproducing the entire proposal.

If information about the actual dataset is unknown, read `DATA_SPEC.md`. If it is still unknown, write code so that the uncertain convention is configurable rather than inventing a convention.

---

# 1. PROJECT CONTEXT

## Competition

Huawei Munich Tech Arena 2026  
Topic: **Automatic Extraction of Anatomical Landmarks of Pinna Shape**

The supplied data consists of full 3D head/upper-torso meshes in PLY format with annotated left/right pinna landmarks.

Huawei describes a consistent anatomical coordinate frame:

- Y axis: left ear canal to right ear canal;
- X axis: back of head to front of head through nose tip;
- Z axis: vertically toward top of head;
- centre: intersection of the anatomical axes.

Each ear has 85 landmarks representing four pinna contours:

- outer helix: 25;
- concha outline: 30;
- inner helix: 20;
- superior antihelix: 10.

Expected prediction per subject:

```text
left:  (85, 3)
right: (85, 3)
```

Evaluation uses mean Euclidean distance between predicted and ground-truth landmark coordinates over hidden subjects and both ears.

Dataset size available to the team: 200 subjects.

The team already submitted a broader research proposal called **PinnaTwin-Zoom**, based on:
- global point-cloud landmark prediction;
- uncertainty;
- contour-consistency risk;
- bilateral-consistency risk;
- adaptive local refinement.

We are **not attempting to implement the full proposal in this sprint**.

---

# 2. SPRINT CONSTRAINT

Team:
- 4 people;
- approximately 3–4 focused hours per person per day;
- heavy use of AI coding assistants;
- final challenge deadline: 15 September 2026;
- current sprint begins before Monday 7 September 2026.

The actual objective is:

> Build the simplest reliable system that accepts an unseen Huawei PLY mesh and returns correctly ordered left/right `85 x 3` coordinates in the original Huawei coordinate frame.

A working simple solution is more valuable than an unfinished sophisticated architecture.

---

# 3. LOCKED IMPLEMENTATION LADDER

Implement in this order.

## Stage 0 — global-frame mean template

Using the frozen TRAINING subjects only:

```python
mean_left_global  = mean(training_left_landmarks, axis=subject)
mean_right_global = mean(training_right_landmarks, axis=subject)
```

These remain in Huawei's original aligned global XYZ coordinate frame.

This is:
- the first non-neural baseline;
- an emergency fallback;
- a way to verify the metric and annotation ordering.

It does not require ear cropping or a neural network.

---

## Stage 1 — leakage-safe ear preprocessing

For each raw head mesh:

1. identify/crop the left or right ear using only the mesh and constants learned from training;
2. optionally mirror one side to a shared orientation;
3. compute a deterministic centre and scale from the cropped **mesh points**, not annotations;
4. map ear points into canonical coordinates;
5. sample a fixed number of points;
6. optionally compute/use normals;
7. save enough transform metadata to invert predicted coordinates exactly;
8. perform crop sanity checks.

Important:

> Validation/test landmarks must never contribute to crop selection, centring, scaling, rotation, mirroring, point sampling decisions, or any other preprocessing step.

Training annotations may be used offline to estimate fixed crop bounds/statistics. Those parameters are then frozen.

---

## Stage 2 — canonical-ear mean template

After the Stage 1 transform is frozen:

For each TRAINING ear:
- compute its mesh-derived canonical transform;
- transform that ear's training landmarks with the same transform.

Then compute the canonical mean landmark configuration.

If left and right ears are successfully mirrored into one consistent canonical orientation, first evaluate using **one shared canonical template** over all training ears.

If visual checks or validation show that shared canonicalisation is unreliable, keep:
- `canonical_template_left`;
- `canonical_template_right`;
while still using a shared model if practical.

Do not make this decision by assumption. Validate it.

This canonical template is the residual base for the neural model.

---

## Stage 3 — small PointNet residual model

Initial model input:

```text
B x N x F
N = 2048
F = 3 (XYZ)
```

Optional later:

```text
F = 6 (XYZ + normals)
```

Output:

```text
B x 85 x 3
```

The model predicts residuals:

```python
canonical_prediction = canonical_template + model(points)
```

Then:

```text
canonical prediction
→ optional surface projection
→ inverse transform
→ original Huawei XYZ prediction
```

Start with a small PointNet-style encoder:
- per-point MLP;
- symmetric global pooling;
- global MLP;
- `85 * 3` output.

Do **not** start with PointNet++.

---

## Stage 4 — cheap improvements

Only after Stage 3 works end-to-end.

Candidates:

- normals;
- mild point-cloud augmentation;
- 1024 vs 2048 points;
- different hidden widths;
- Smooth-L1 vs simple coordinate loss;
- surface projection;
- multiple random point samples at inference;
- 2–3 independently trained seeds;
- averaging predictions.

Each candidate survives only if validation supports it.

---

## Stage 5 — optional fixed local refiner

Unlock only if the global pipeline has been stable for at least one full day.

For each coarse landmark:

```text
coarse prediction
→ fixed K nearest surface points (start K=256)
→ local coordinates
→ tiny shared PointNet
→ XYZ correction
→ refined point
```

Train first using synthetic offsets around training ground-truth landmarks, then evaluate around actual coarse predictions if time permits.

No adaptive risk routing.

Keep the refiner only if it clearly reduces held-out error.

---

# 4. FEATURES LOCKED OUT OF THIS SPRINT

Unless the four-person team explicitly agrees that the full baseline is already complete and stable, do not implement:

- PointNet++ migration;
- heatmap outputs;
- heatmap entropy;
- uncertainty routing;
- ContourGuard inference;
- bilateral PinnaTwin inference;
- learned risk weights;
- adaptive 128/256/512 patches;
- correction confidence gating;
- large architecture rewrites.

These are research extensions, not sprint requirements.

---

# 5. DATA / NDA RULES

The full dataset is provided under NDA.

Repository `.gitignore` should include at minimum:

```gitignore
data/
outputs/
checkpoints/
*.ply
*.npz
*.pt
*.pth
__pycache__/
.venv/
```

Do not commit:
- raw meshes;
- raw annotations;
- preprocessed Huawei samples;
- model checkpoints;
- confidential screenshots/exports if covered by NDA.

Before exposing raw data to an external AI provider, check the NDA and team policy.

AI can work effectively from:
- code;
- schemas;
- array shapes;
- file naming conventions;
- stack traces;
- logs;
- aggregate coordinate statistics;
- synthetic examples;
- non-confidential descriptions.

---

# 6. REPOSITORY LAYOUT

Use a simple shared repo:

```text
PinnaTwin-Zoom/
│
├── src/
│   ├── data.py
│   ├── geometry.py
│   ├── model.py
│   ├── losses.py
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   └── visualise.py
│
├── scripts/
│   └── preprocess.py
│
├── configs/
│
├── tests/
│
├── DATA_SPEC.md
├── EXPERIMENTS.md
├── SPRINT_PLAN.md
├── README.md
├── requirements.txt
└── .gitignore
```

Do not create a complex package architecture unless necessary.

---

# 7. OWNERSHIP

## Role A — Data & Geometry

Primary files:
- `src/data.py`
- `src/geometry.py`
- `scripts/preprocess.py`
- geometry-focused tests
- geometry section of `DATA_SPEC.md`

Owns:
- PLY and annotation format;
- safe ear crops;
- mirroring/canonicalisation;
- point sampling;
- normals if needed;
- transform metadata;
- exact inverse transforms;
- preprocessing cache;
- crop QA.

---

## Role B — Model & Training

Primary files:
- `src/model.py`
- `src/losses.py`
- `src/train.py`
- `configs/`
- model-focused tests

Owns:
- PointNet;
- residual output;
- losses;
- optimiser;
- augmentation;
- training loop;
- checkpointing;
- model experiments.

---

## Role C — Evaluation & Visualisation

Primary files:
- `src/evaluate.py`
- `src/visualise.py`
- evaluation tests
- validation analysis outputs

Owns:
- frozen train/validation split;
- official metric;
- median/p95;
- per-contour and per-landmark metrics;
- plots;
- worst cases;
- optional surface projection experiment;
- experiment comparison.

---

## Role D — Integration & Submission

Primary files:
- `src/infer.py`
- `README.md`
- `EXPERIMENTS.md`
- integration tests
- final packaging/config glue

Owns:
- shared interfaces;
- end-to-end inference;
- daily `main` smoke test;
- experiment registry;
- final model/config selection;
- output format;
- reproducibility;
- submission package.

Role D integrates but does not control the other owners' technical work.

---

# 8. SHARED INTERFACE CONTRACTS

These contracts exist to stop four AI agents from inventing incompatible code.

Exact Python syntax may change once the actual Huawei notebook is inspected, but semantic behaviour must remain stable.

## 8.1 `EarSide`

Represent side explicitly:

```python
Literal["left", "right"]
```

Do not rely on ambiguous booleans.

---

## 8.2 Loaded raw subject

Conceptual structure:

```python
@dataclass
class RawSubject:
    subject_id: str
    vertices: np.ndarray        # [V, 3]
    faces: np.ndarray | None    # [F, 3] if available
    vertex_normals: np.ndarray | None
```

Training loader may also return annotations, but inference functions must not require them.

---

## 8.3 Transform metadata

Conceptual structure:

```python
@dataclass
class EarTransform:
    side: str
    center: np.ndarray          # [3]
    scale: float
    mirror_applied: bool
    # add only mesh-derived/frozen metadata needed to invert exactly
```

No field may contain validation/test ground-truth information.

---

## 8.4 Canonical ear sample

Conceptual structure:

```python
@dataclass
class CanonicalEar:
    subject_id: str
    side: str
    points: np.ndarray          # [N, 3]
    normals: np.ndarray | None  # [N, 3]
    transform: EarTransform
```

Training dataset may separately attach:

```python
target_landmarks: np.ndarray    # [85, 3] canonical, training only
```

---

## 8.5 Required geometry functions

Role A should expose semantically equivalent functions:

```python
load_mesh(path) -> RawSubject

crop_ear(raw_subject, side, crop_config) -> raw ear geometry

canonicalize_ear(raw_ear, side, config) -> CanonicalEar

transform_points_to_canonical(points_xyz, transform) -> np.ndarray

inverse_transform_points(points_xyz, transform) -> np.ndarray
```

The transform pair must be mathematically inverse.

---

## 8.6 Model interface

Role B should expose something equivalent to:

```python
class LandmarkPointNet(nn.Module):
    def forward(self, points):
        # points: [B, N, F]
        # return residuals: [B, 85, 3]
```

Do not make the model responsible for parsing files or undoing coordinate transforms.

---

## 8.7 Evaluation interface

Role C should expose:

```python
mean_euclidean_error(pred, target) -> float
```

where:

```text
pred   [B, 85, 3]
target [B, 85, 3]
```

Also provide per-example errors for analysis.

Contour slices must be read from verified `DATA_SPEC.md`; do not invent indices until verified.

---

## 8.8 Inference interface

Role D should ultimately expose:

```python
predict_subject(mesh_path, checkpoint, config) -> dict
```

returning:

```python
{
    "left":  np.ndarray(shape=(85, 3)),
    "right": np.ndarray(shape=(85, 3)),
}
```

Both arrays must be:
- correctly ordered;
- finite;
- in original Huawei coordinates.

---

# 9. CROP STRATEGY

Because Huawei aligns all scans anatomically, the first approach is training-derived fixed global crop bounds.

Procedure:

1. use TRAINING landmark distributions only;
2. obtain left/right XYZ bounding ranges;
3. add generous margins;
4. freeze these ranges;
5. apply them to raw meshes;
6. visually inspect many training and validation subjects;
7. calculate crop sanity statistics.

Possible sanity checks:
- nonzero and plausible vertex count;
- plausible bounding-box dimensions;
- enough spatial spread;
- no obviously truncated pinna in visual QA.

Do not blindly trust fixed bounds.

Fallback strategy should remain conservative:

1. expand the frozen crop margins;
2. flag/log the sample;
3. prefer a larger safe region over an unvalidated clever segmentation algorithm.

Do not introduce a "densest patch" heuristic unless actual data inspection shows it is necessary and validation proves it works.

---

# 10. CANONICAL TRANSFORM

The transform must be based only on the ear mesh/crop and frozen settings.

A simple initial choice:

1. crop ear using frozen global bounds;
2. centre using cropped point-cloud centre or bounding-box centre;
3. scale using a deterministic mesh statistic such as maximum bounding-box extent;
4. mirror one side if needed to create a common orientation.

Huawei's Y axis is left-to-right, so mirroring may plausibly be a Y-axis reflection, but **do not hardcode this solely from theory**. Verify using sample visualisation and document the final convention in `DATA_SPEC.md`.

The canonical transform must support exact inverse mapping.

---

# 11. REQUIRED GEOMETRY TESTS

Before serious training:

## Test A — round trip

For arbitrary original coordinates:

```text
original
→ canonical transform
→ inverse transform
→ original
```

Error should be at numerical precision.

Ground-truth landmarks may be used as test coordinates during TRAINING/DEVELOPMENT to verify transform mathematics, but the transform itself must be calculated from mesh information only.

## Test B — crop QA

Visualise several subjects, including extremes.

## Test C — target transform

For a training ear:
- calculate transform from mesh;
- transform its GT landmarks;
- inverse them;
- recover original annotations.

## Test D — deterministic behaviour

With fixed seed/config, repeated preprocessing gives the same sampled points if determinism is desired.

---

# 12. TRAIN / VALIDATION SPLIT

Use a subject-level split.

Default unless the dataset/example code dictates something better:

```text
160 subjects training
40 subjects validation
```

Use fixed seed, e.g. 42.

Save the split to files/config.

Both ears from a subject must remain in the same partition.

Never do:

```text
subject X left  → train
subject X right → validation
```

Freeze the split before model selection.

Do not change the validation subjects because a model performs badly.

---

# 13. BASELINES

## Baseline 0 — global mean

Predict the same:
- global mean left landmarks for every left ear;
- global mean right landmarks for every right ear.

Record validation MED.

## Baseline 1 — canonical template only

After safe canonicalisation:
- canonicalise the test ear using mesh only;
- output canonical mean template;
- inverse-transform it.

This is an important stronger non-neural baseline.

## Baseline 2 — PointNet residual

```text
canonical template + learned residual
```

The learned model should beat Baseline 1 to justify itself.

---

# 14. MODEL DETAILS — KEEP SIMPLE

Suggested first architecture:

```text
input [B,N,3]
↓
shared MLP 3→64→128→256
↓
global max pooling
↓
MLP 256→512→256
↓
output 255
↓
reshape [B,85,3]
```

Exact widths are not sacred.

Start with:
- ReLU/GELU;
- batch norm if stable;
- dropout only if needed;
- no attention;
- no graph neural network;
- no transformer.

Loss:

Start with Smooth-L1/Huber coordinate loss in canonical coordinates.

Also report Euclidean distance metric separately.

Do not optimise architecture before confirming the pipeline works.

---

# 15. TINY-DATA OVERFIT TEST

Before full training:

Train on approximately 4–8 ears with augmentation OFF.

Expected result:
- training error should become very small relative to starting error;
- visual predictions should approach GT.

If the model cannot overfit:
- inspect labels;
- inspect coordinate scales;
- inspect template;
- inspect tensor ordering;
- inspect loss;
- inspect transforms.

Do not solve an overfit failure by making the network larger.

---

# 16. AUGMENTATION

Only add after baseline training works.

Safe candidates:
- small rotations;
- small isotropic scale;
- point jitter;
- random point dropout/resampling.

Any geometric augmentation applied to input points must also be applied consistently to target landmarks.

Avoid unrealistic transformations that break ear anatomy or coordinate conventions.

Start conservative.

Log exact augmentation ranges.

---

# 17. SURFACE PROJECTION

This is an experiment, not automatically part of the final pipeline.

Compare:

```text
raw predicted coordinate
```

vs

```text
prediction projected to nearest valid mesh surface
```

Projection can help if predictions drift off-surface, but it can hurt if the nearest surface is the wrong fold.

Role C evaluates it.

Keep only if validation improves.

---

# 18. ENSEMBLING

Only after one model works.

Cheap options:

## Multiple seeds
Train 2–3 copies of the same best configuration.

Average final coordinates.

## Multiple point samples
At inference, sample the same ear point cloud several times and average predictions.

Do not assume it helps; validate.

---

# 19. OPTIONAL LOCAL REFINER

Unlock only if:
- preprocessing is stable;
- global inference is stable;
- model clearly beats template baseline;
- end-to-end reproduction works;
- at least one full day remains for validation.

First implementation:
- fixed K=256 nearest points;
- shared local model;
- local relative XYZ/normals;
- landmark identity may be included;
- output XYZ correction.

No adaptive patch size.

No uncertainty router.

No contour/bilateral risk.

If refiner does not clearly improve held-out MED, remove it.

---

# 20. EXPERIMENT TRACKING

Every meaningful run goes into `EXPERIMENTS.md`.

Minimum record:

```markdown
## Run XXX

Date:
Owner:
Git commit:
Config:
Model:
Input features:
Point count:
Template:
Loss:
Augmentation:
Seed:
Epochs:
Checkpoint location:

Validation mean:
Validation median:
Validation p95:
Per-contour summary:

Notes:
Decision:
```

Never rely on Slack/Discord/chat memory to identify the best run.

---

# 21. DAILY MAIN SMOKE TEST

Role D runs this after daily merges once inference exists.

One command should:

```text
raw PLY
→ both ears
→ preprocessing
→ model
→ inverse transform
→ output left/right predictions
```

Verify:

```python
left.shape == (85, 3)
right.shape == (85, 3)

np.isfinite(left).all()
np.isfinite(right).all()
```

Also visually inspect at least one sample periodically.

If `main` is broken:
- fix it before adding new features.

---

# 22. GIT WORKFLOW

One shared repository.

Suggested role branches:
- `geometry`
- `model`
- `evaluation`
- `integration`

Rules:
1. Pull/rebase before work.
2. Small commits.
3. Merge at least daily.
4. `main` should remain runnable.
5. Communicate shared interface changes.
6. Do not let AI make commits/pushes.
7. Do not let AI mass-refactor another owner's files without agreement.
8. Resolve merge conflicts deliberately; do not accept generated conflict resolutions blindly.

---

# 23. TRAINING MACHINE

Choose one canonical GPU environment.

Reasons:
- avoids dependency drift;
- keeps checkpoints in one place;
- makes run comparison consistent.

Other teammates can:
- develop locally on CPU/small data;
- run tests;
- inspect visualisations.

Preprocessing should be deterministic enough that any machine with permitted local data access can rebuild its own cache.

Do not sync NDA cache files through Git.

---

# 24. DATA_SPEC.md — MUST EXIST

After inspecting Huawei's example notebook and real files, document:

```markdown
# DATA SPEC

## Directory layout

## Subject IDs

## Mesh filename convention

## Annotation filename convention

## PLY fields
- vertices:
- faces:
- normals:
- colours:

## Landmark shape

## Left/right annotation convention

## Confirmed contour indices

## Coordinate units

## Approximate coordinate ranges

## X/Y/Z interpretation

## Confirmed mirror convention

## Crop bounds and margins

## Canonical centre definition

## Canonical scale definition

## Point count

## Normal handling

## Output format required by challenge

## Known anomalies
```

Every AI should read this before touching geometry/model interfaces.

---

# 25. DAY-BY-DAY SPRINT

## Friday 4 September — setup

Goal: no setup friction later.

Team:
- create/join repo;
- add `.gitignore`;
- add this master file;
- add role files;
- agree branch names;
- confirm dataset access;
- locate Huawei example notebook;
- agree canonical GPU/training machine;
- assign Roles A/B/C/D.

Do not waste the day debating architectures.

---

## Saturday 5 September — understand data

Everyone starts from Huawei's example notebook.

Role A:
- load one real mesh;
- inspect fields;
- load annotations;
- visualise left/right landmarks;
- begin `DATA_SPEC.md`.

Role B:
- create minimal PointNet skeleton;
- forward-pass test on random tensors.

Role C:
- implement metric;
- prepare frozen subject split;
- create basic landmark visualisation.

Role D:
- establish repository interfaces;
- create baseline/evaluation CLI skeleton;
- ensure modules can import cleanly.

Team output:
- usable `DATA_SPEC.md`;
- first real data visualisations;
- no confusion about shapes/naming.

---

## Sunday 6 September — non-neural safety net

Role A:
- help establish training-derived global landmark statistics;
- prototype crop boxes.

Role C/D:
- compute Baseline 0 global mean left/right templates;
- score validation;
- visualise on held-out subjects.

Role B:
- finish model and loss skeleton;
- prepare dataset expectations.

Output:
- valid non-neural predictor;
- first logged MED;
- fixed split;
- basic repo works.

---

## Monday 7 September — safe geometry

Primary goal: leakage-safe preprocessing.

Role A:
- implement crop;
- mesh-derived centre/scale;
- mirror convention;
- point sampling;
- transform metadata;
- inverse transform;
- crop QA.

Role B:
- connect model to expected canonical tensors.

Role C:
- test transformed GT round-trip;
- enhance visualisation.

Role D:
- connect preprocessing interfaces;
- add integration tests.

Gate:
- transform round-trip works;
- at least 5–10 subjects visually inspected;
- no GT-dependent test preprocessing.

---

## Tuesday 8 September — prove learning

Role A:
- preprocess/cache full train/validation set;
- flag crop anomalies.

Role B:
- compute/use canonical template;
- overfit 4–8 ears;
- fix model/training until successful.

Role C:
- worst-case plot tooling;
- per-example metric.

Role D:
- create end-to-end inference using a checkpoint;
- verify output returns to original frame.

Gate:
- tiny-data overfit works;
- inference path exists.

---

## Wednesday 9 September — first real learned system

Role B:
- train PointNet on full training partition.

Role A:
- support data issues only; avoid new geometry ideas unless necessary.

Role C:
- evaluate Baseline 0, Baseline 1, Baseline 2;
- inspect failures.

Role D:
- make raw-Ply-to-two-85x3 pipeline work.

Critical milestone by end of day:

```text
raw PLY
→ safe ear processing
→ learned model
→ original-frame left/right 85x3
```

Once this exists, the project is safe to submit in principle.

---

## Thursday 10 September — cheap improvements

Candidate controlled experiments:
- XYZ vs XYZ+normals;
- 1024 vs 2048 points;
- conservative augmentation;
- Smooth-L1 vs another simple coordinate loss;
- surface projection.

Do not test everything at once.

One change per run when possible.

---

## Friday 11 September — analyse and ensemble

Role C leads error breakdown:
- mean;
- median;
- p95;
- per contour;
- per landmark;
- worst 20 ears.

Team fixes actual bottleneck.

If stable:
- train 2–3 seeds;
- test multiple point samples;
- average predictions.

No new research architecture.

---

## Saturday 12 September — decision gate

Question:

> Is the global system fully reproducible and stable?

If NO:
- fix it;
- no refiner.

If YES:
- optionally implement fixed local refiner.

The refiner must earn its place through validation.

---

## Sunday 13 September — final model decision

Complete comparison table.

Choose final pipeline based on:
1. held-out mean Euclidean error;
2. reliability;
3. reproducibility;
4. no catastrophic failures.

Delete/disable features that hurt.

Begin final write-up/package.

---

## Monday 14 September — CODE FREEZE

No new architecture.

Tasks:
- select checkpoint(s);
- final inference config;
- fresh reproduction from README;
- dependency verification;
- output shape/order verification;
- original-coordinate verification;
- test several raw subjects;
- visual spot checks;
- final package;
- upload a valid submission.

Someone other than the author of `infer.py` must follow the README from scratch.

---

## Tuesday 15 September — emergency/submission only

Do not plan normal development.

Use only for:
- verifying uploaded package;
- replacing genuinely broken upload;
- critical fixes.

Check the logged-in challenge portal for the exact final cutoff time.

---

# 26. STOP CONDITIONS

Stop new feature work immediately if any of these are not true:

```text
[ ] raw data loads
[ ] left/right handling is verified
[ ] crop works
[ ] transform is leakage-safe
[ ] transform round-trip works
[ ] train/validation split is subject-safe
[ ] metric works
[ ] model tensor shapes are correct
[ ] tiny-data overfit works
[ ] predictions inverse-transform correctly
[ ] inference returns both (85,3) arrays
[ ] main branch remains runnable
```

Fix foundations first.

---

# 27. SUCCESS LEVELS

## Minimum
Global mean template works and output formatting is valid.

## Good
Canonical template baseline works.

## Strong sprint result
PointNet residual model beats canonical template and runs end-to-end.

## Very strong
Normals/augmentation/projection/ensemble provide validated improvement.

## Bonus only
Fixed local refiner improves validation.

No other feature is required.

---

# 28. DEFINITION OF DONE

The sprint is complete when:

```python
result = predict_subject("unseen_subject.ply", checkpoint, config)

assert result["left"].shape == (85, 3)
assert result["right"].shape == (85, 3)
assert np.isfinite(result["left"]).all()
assert np.isfinite(result["right"]).all()
```

and:

- predictions are in original Huawei coordinates;
- ordering matches Huawei annotation convention;
- process uses no hidden GT information;
- README reproduction works;
- best configuration is documented;
- final package is uploaded successfully.

---

# 29. TEAM DECISION RULE

For every proposed feature:

> Does this clearly improve held-out mean Euclidean error or reduce end-to-end failure risk within the time remaining?

If not, do not build it.

---

# 30. ONE-SENTENCE PROJECT MISSION

> Build the simplest reliable system that transforms an unseen Huawei head mesh into correctly ordered left and right 85-point pinna landmarks, get the complete path working early, and spend the remaining sprint reducing validation error instead of expanding architecture.

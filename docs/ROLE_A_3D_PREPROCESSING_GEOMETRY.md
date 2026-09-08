# ROLE A — 3D PREPROCESSING & GEOMETRY ENGINEER

**Owner: Alfred**

> Give this file to your AI after `MASTER_AI_CONTEXT.md` and `DATA_SPEC.md`.

Tell the AI:

> I am Alfred and I own Role A. Inspect the current repository before editing. Follow this role in order. Do not redesign the architecture. Do not modify another owner's modules unless a shared interface genuinely requires it. Do not commit or push.

---

# Mission

Turn each raw Huawei head/torso PLY into trustworthy canonical left/right ear point clouds, and provide an exact inverse transform so downstream landmark predictions return to Huawei's original XYZ frame.

This is the highest-risk foundation of the project.

---

# Primary ownership

```text
src/data.py
src/geometry.py
scripts/preprocess.py
tests/test_geometry.py
geometry/data sections of DATA_SPEC.md
```

You do not own:
- PointNet/training
- metric/template statistics
- end-to-end CLI

Coordinate interface changes rather than duplicating those modules.

---

# What you must build

## 1. Understand actual Huawei files

Start with Huawei's example notebook.

Verify and document:
- dataset directory structure
- mesh names
- annotation names
- subject IDs
- PLY vertex/face/normals fields
- annotation shape
- left/right conventions
- coordinate ranges/units
- landmark ordering
- actual contour indexing

Update `DATA_SPEC.md`.

Do not guess.

---

## 2. Raw loaders

Expose semantically equivalent functions:

```python
load_mesh(path) -> RawSubject
load_landmarks(path, side) -> np.ndarray  # train/dev only
```

Inference mesh loading must not require annotations.

---

## 3. Ear crop system

Use TRAINING annotation statistics to estimate generous left/right crop regions in the aligned Huawei frame.

Then freeze crop parameters.

At val/test inference, crop uses:
- mesh
- frozen config

not GT landmarks.

Build:

```python
crop_ear(raw_subject, side, crop_config)
```

Validate across many subjects.

Track QA such as:
- vertex count
- crop bounding box
- spatial spread
- centre
- suspicious/truncated crop flag

If fixed crop is insufficient:
1. increase frozen margins
2. flag/log
3. prefer safe large regions

Do not invent complex segmentation unless the data proves it is necessary.

---

## 4. Canonical transform

Build deterministic mesh-derived canonicalisation.

Initial simple strategy:
- centre from cropped mesh points or bbox centre
- scale from deterministic crop extent
- optional side mirror

Huawei Y is left-to-right, so mirroring may involve Y, but verify visually before locking.

Never use GT val/test landmarks to calculate transform values.

Expose:

```python
canonicalize_ear(raw_ear, side, config) -> CanonicalEar
```

---

## 5. Inverse transform

Store enough metadata to recover original coordinates exactly.

Expose:

```python
transform_points_to_canonical(points_xyz, transform)
inverse_transform_points(points_xyz, transform)
```

These must be mathematical inverses.

---

## 6. Point sampling

Start with:

```text
N = 2048
```

Simple deterministic/random sampling is sufficient.

Do not block baseline on farthest-point sampling.

Return:

```text
points [N,3]
```

Optionally later:

```text
points/features [N,6]
```

if normals are used.

---

## 7. Normals

Support if easy from PLY or mesh processing.

Do not make normals mandatory for initial pipeline.

---

## 8. Processed cache

Build local cache generation so B can train efficiently.

Example local artefacts:

```text
subject_001_left.npz
subject_001_right.npz
```

May include:
- sampled canonical points
- normals
- transform metadata
- subject ID
- side
- canonical targets for training/dev

Do not commit cache.

---

# Critical tests

## Test 1 — mathematical round trip

```python
canonical = transform_points_to_canonical(original, transform)
recovered = inverse_transform_points(canonical, transform)

assert np.allclose(original, recovered)
```

Use multiple points/subjects.

---

## Test 2 — GT coordinate round trip

For a TRAINING/DEV ear:
- calculate transform from mesh only
- transform GT landmarks
- inverse them

Recover original GT coordinates.

GT is only the test coordinate set; it must not define the transform.

---

## Test 3 — side correctness

Verify left/right never swap.

---

## Test 4 — mirror correctness

Overlay/inspect canonical left/right ears.

Verify shared orientation makes anatomical sense.

---

## Test 5 — crop QA

Visualise at least 5–10 subjects initially and additional extreme cases.

---

# Contract to the rest of team

Your output should conceptually provide:

```python
@dataclass
class CanonicalEar:
    subject_id: str
    side: str
    points: np.ndarray
    normals: np.ndarray | None
    transform: EarTransform
```

Role B should not need to know how PLY loading works.

Role C uses your transform functions when converting training landmarks to canonical coordinates.

Role D uses your preprocessing/inverse functions in inference.

---

# First work session

1. Read master context.
2. Open Huawei example notebook.
3. Inspect one real subject.
4. Plot full mesh + left/right landmarks.
5. Write verified conventions into `DATA_SPEC.md`.
6. Implement raw loader.
7. Calculate rough training landmark ranges.
8. Prototype left/right crop.
9. Plot crop result.
10. Stop before over-engineering.

---

# Next milestones

## By Monday
- safe crop
- canonical transform
- inverse
- sampling
- round-trip tests

## By Tuesday
- full processed cache
- anomaly flags
- stable interface for B/C/D

## After Tuesday
- fix geometry edge cases
- normals only if useful
- support integration

---

# Definition of done

Role A is done when:

```text
raw Huawei PLY
→ left CanonicalEar
→ right CanonicalEar
```

works without GT annotations, and any canonical prediction can be inverse-transformed exactly into Huawei's original coordinate frame.

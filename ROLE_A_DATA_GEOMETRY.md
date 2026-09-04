# ROLE A — DATA & GEOMETRY OWNER

> Paste this file **after** `01_MASTER_AI_CONTEXT.md` into your AI.  
> Tell it: **"I am Role A. Follow the master plan and work only on Data & Geometry unless an interface change is necessary."**

---

# Mission

Make every raw Huawei PLY mesh become a trustworthy canonical left/right ear point cloud, and make every canonical prediction map exactly back into the original Huawei coordinates.

You own the geometry foundation. If this is wrong, every model score is meaningless.

---

# Files you primarily own

```text
src/data.py
src/geometry.py
scripts/preprocess.py
tests/test_geometry.py
geometry-related sections of DATA_SPEC.md
```

Avoid editing model/evaluation/integration files unless coordinating a shared interface change.

---

# Required deliverables

## 1. Understand Huawei's real data

Start from Huawei's provided example notebook.

Document in `DATA_SPEC.md`:
- folder structure;
- subject naming;
- PLY structure;
- vertex/face/normals availability;
- annotation files;
- annotation shape;
- left/right convention;
- contour ordering;
- coordinate ranges/units if known;
- any anomalies.

Do not guess.

---

## 2. Raw loading

Expose clean functions equivalent to:

```python
load_mesh(path) -> RawSubject
load_landmarks(path, side) -> np.ndarray  # training/dev only
```

Inference mesh loading must not require annotations.

---

## 3. Training-derived crop configuration

Use TRAINING annotations to estimate generous fixed left/right XYZ regions in Huawei's aligned frame.

Then freeze them.

At inference, only the mesh + frozen crop config may be used.

Add margins.

Validate crops visually over many subjects, including extremes.

---

## 4. Crop sanity checking

Every crop should return/log useful QA:
- number of vertices;
- bounding-box extent;
- centre;
- scale;
- whether fallback/expanded crop was used.

Flag suspicious crops.

Prefer a larger safe crop to an unvalidated clever heuristic.

---

## 5. Canonical transform

Define deterministic mesh-derived:
- centre;
- scale;
- optional side mirror.

Likely mirroring relates to Huawei's left-right Y axis, but verify from data before locking.

All transform parameters must come from:
- cropped mesh;
- frozen training-derived constants.

Never GT at val/test.

---

## 6. Exact inverse

Implement:

```python
transform_points_to_canonical(...)
inverse_transform_points(...)
```

They must be inverses.

---

## 7. Point sampling

Start with:
```text
N = 2048
```

Simple random sampling is acceptable initially.

If sampling with replacement is ever necessary, document it.

Farthest point sampling is optional and not a priority.

---

## 8. Normals

Support normals if cheaply available from PLY or mesh computation.

Do not block the baseline on normals.

Return `None` or configurable normals cleanly if not used.

---

## 9. Cache

Create deterministic preprocessing script producing local cache such as:

```text
subject_001_left.npz
subject_001_right.npz
```

Potential contents:
- sampled canonical points;
- normals;
- transform metadata;
- canonical training landmarks for train/val development;
- side;
- subject ID.

Never commit these files.

---

# Required tests

## Round-trip transform

For multiple arbitrary coordinates:

```text
original → canonical → inverse → original
```

Numerical error ~0.

## Training annotation round-trip

Compute transform from mesh only, then transform/inverse GT coordinates.

Must recover GT.

## Visual crop test

Plot at least 5–10 subjects initially and more once automation exists.

## Side test

Verify left/right are not accidentally swapped.

## Mirror test

Overlay canonical left/right ears and verify the chosen mirroring convention makes anatomical sense.

---

# Shared output contract

You must provide Role B/D with a stable object containing:

```text
points:    [N,3]
normals:   [N,3] or None
side:      "left"/"right"
transform: invertible metadata
subject_id
```

Training code may additionally receive canonical targets.

Tell the team before changing the interface.

---

# Today / first work session

1. Open Huawei notebook.
2. Load one real subject.
3. Plot mesh + both landmark sets.
4. Write actual conventions into `DATA_SPEC.md`.
5. Implement raw loader.
6. Prototype generous left/right crops.
7. Show visual result.

Do not spend the first session implementing sophisticated geometry.

---

# Your definition of done

Role A is done when Role D can give you a raw PLY and receive two canonical ears plus invertible transforms, without using any test labels.

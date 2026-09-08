---
paths:
  - "src/data.py"
  - "src/geometry.py"
  - "scripts/**"
  - "tests/test_geometry.py"
  - "tests/test_data.py"
---

# Role A — preprocessing & geometry contract

Mission: raw Huawei head PLY -> trustworthy canonical left/right ear point clouds
+ an exact inverse back to Huawei's original XYZ frame. Mesh-derived only; GT
landmarks may be pushed *through* these functions, never used to define one.

## Interface (as implemented — frozen; change only on Alfred's say-so)
`src/data.py`
- `DATA_ROOT = Path($HUAWEI_DATA_ROOT or "data")`, `SIDES = ("left", "right")`
- `RawSubject(subject_id: str, vertices [V,3] f64, faces [F,3] int|None, normals [V,3] f64|None, mesh_path: Path)`
- `load_mesh(path) -> RawSubject` — must work with no annotations (inference entry point)
- `load_landmarks(path, side) -> [85,3] f64` — TRAIN/DEV ONLY; `load_subject_landmarks(sid, root) -> {side: [85,3]}`
- `list_subjects(root=DATA_ROOT) -> list[str]` — full IDs `"P0001"`; `mesh_path(sid, root)`, `landmark_path(sid, side, root)`

`src/geometry.py` (`N_LANDMARKS = 85`, `N_POINTS = 2048`)
- `EarTransform(centre [3], scale float>0, mirror_axis int|None)` + `to_dict()` / `from_dict()`
  canonical = `(p - centre) / scale`, then negate component `mirror_axis`
- `CropConfig(side, lo [3], hi [3], min_vertices=500)` — frozen axis-aligned box, original frame; `load_crop_config(side, path="configs/crop.yaml")` reads the one written by `scripts/crop_stats.py`
- `CanonicalEar(subject_id, side, points [N,3], normals [N,3]|None, transform, qa: dict)`
- `crop_ear(raw, side, cfg) -> (points [M,3], normals [M,3]|None, qa)`
  qa: `subject_id, side, n_vertices, bbox_lo, bbox_hi, centre, suspicious`
- `make_transform(crop_points, side, mirror_side, mirror_axis=1) -> EarTransform` (bbox centre, half max extent)
- `sample_points(points, normals, n=N_POINTS, seed=0) -> ([n,3], [n,3]|None)` — deterministic RNG
- `transform_points_to_canonical(points, t)` / `inverse_transform_points(points, t)` — exact inverses
- `canonicalize_ear(raw, side, cfg, mirror_side="right", n_points=N_POINTS, seed=0, mirror_axis=1) -> CanonicalEar`

## Required tests (synthetic data constructed in the test; no real fixtures)
1. Mathematical round trip: `inverse(to_canonical(p)) ≈ p` over many points/transforms.
2. GT coordinate round trip: transform derived from mesh only, GT pushed through and recovered.
3. Side correctness: left/right never swap.
4. Mirror correctness: canonical left/right share an anatomically sensible orientation.
5. Crop QA: `n_vertices`, bbox, centre, `suspicious` flag behave on known geometry.

## Cache layout — `cache/<subject>_<side>.npz` (never committed)
`points [N,3]`, `normals [N,3]|None`, transform dict (`centre`, `scale`, `mirror_axis`),
`subject_id`, `side`, canonical GT targets `[85,3]` (train/dev only).

## Milestone order — do not skip ahead
loader -> crop -> transform/inverse -> sampling -> cache.

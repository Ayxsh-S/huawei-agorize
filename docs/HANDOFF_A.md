# Role A — frozen interface for B / C / D

Import from `src.data` / `src.geometry`. Frozen — ping Alfred before relying on anything else. Constants: `N_LANDMARKS = 85`, `N_POINTS = 2048`, `SIDES = ("left", "right")`.

## Data types
- `RawSubject(subject_id: str, vertices: [V,3] f64, faces: [F,3] int|None, normals: [V,3] f64|None, mesh_path: Path)`
- `EarTransform(centre: [3] f64, scale: float > 0, mirror_axis: int|None)` — `.to_dict()` / `.from_dict(d)`
- `CropConfig(side: str, lo: [3], hi: [3], min_vertices: int = 500)`
- `CanonicalEar(subject_id: str, side: str, points: [N,3] f64, normals: [N,3] f64|None, transform: EarTransform, qa: dict)`

## Functions
```python
load_mesh(path) -> RawSubject                      # no annotations needed
load_landmarks(path, side) -> np.ndarray           # [85,3], TRAIN/DEV ONLY
load_subject_landmarks(subject_id, root=DATA_ROOT) -> {"left": [85,3], "right": [85,3]}
list_subjects(root=DATA_ROOT) -> list[str]        # full IDs, e.g. "P0001"
mesh_path(subject_id, root=DATA_ROOT) -> Path      # <root>/mesh/P0001.ply
landmark_path(subject_id, side, root=DATA_ROOT) -> Path  # <root>/landmarks/P0001_<side>_ear_landmarks.csv

load_crop_config(side, path="configs/crop.yaml") -> CropConfig   # frozen box from scripts/crop_stats.py

crop_ear(raw, side, cfg) -> (points[M,3], normals[M,3]|None, qa)
make_transform(crop_points, side, mirror_side, mirror_axis=1) -> EarTransform
sample_points(points, normals, n=2048, seed=0) -> (points[n,3], normals[n,3]|None)
transform_points_to_canonical(points, t) -> np.ndarray   # (p - centre)/scale, then mirror
inverse_transform_points(points, t) -> np.ndarray        # exact inverse
canonicalize_ear(raw, side, cfg, mirror_side="right", n_points=2048, seed=0, mirror_axis=1) -> CanonicalEar
```

## Notes
- B: train on `CanonicalEar.points` `[2048,3]`; predict `[85,3]` in canonical space.
- C: use `transform_points_to_canonical` to move training landmarks into canonical space — never derive a transform from GT.
- D: inference is `load_mesh -> canonicalize_ear(raw, side, load_crop_config(side)) -> model -> inverse_transform_points`, mesh + frozen config only.
- Subject IDs are the full `"P0001"` string everywhere (cache keys, logs, submission). Real PLYs carry no normals: `RawSubject.normals` is `None`.
- Cache: `cache/<subject>_<side>.npz` — `points`, `normals`, transform dict, `subject_id`, `side`, canonical GT (train/dev only). Not committed.

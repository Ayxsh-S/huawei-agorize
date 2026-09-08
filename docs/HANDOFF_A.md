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
- Cache: `cache/<subject_id>_<side>.npz`, read ONLY through `src.cache` (next section). Not committed. No normals are cached — the real scans carry none.

## Processed cache — `src.cache` (schema `"1"`)

Built by Alfred with `python scripts/preprocess.py --subject-list splits/<split>_ids.txt [--with-targets]`
(one file per ear, `cache/<subject_id>_<side>.npz`, e.g. `cache/P0001_left.npz`). B and C never
parse npz keys themselves — use these two functions:

```python
from src.cache import list_cached, load_cached_ear

list_cached(cache_dir) -> list[tuple[str, str, Path]]   # (subject_id, side, path), sorted by
                                                        # subject then left/right; ignores anything
                                                        # that is not <id>_<side>.npz (temp files etc.)
load_cached_ear(path) -> CanonicalEar                   # transform rebuilt via EarTransform.from_dict:
                                                        # bit-exact, so inverse_transform_points on a
                                                        # prediction is as exact as in check_roundtrip
```

`load_cached_ear` returns `CanonicalEar(subject_id, side, points, normals=None, transform, qa)` with
`points` float64 `[n_points, 3]` (float32-exact values — the file stores float32) and a `qa` dict
carrying `n_vertices` (crop size), `suspicious`, `sampled_with_replacement`, `n_points`, `seed`,
`ear_seed`, `scale`, `mirror_axis`, `crop_config_sha256`, `split_file`, `split_sha256`,
`schema_version`, and — only for files built with `--with-targets` — **`qa["targets"]`: the
`[85, 3]` float64 canonical GT landmarks** (the training label; C maps them back with
`inverse_transform_points(qa["targets"], ear.transform)`, float32-exact, ~1e-6 mm).
It raises `ValueError` on a wrong `schema_version`, a missing key, or a file whose stored
`subject_id`/`side` disagree with its filename.

Exact npz layout (plain numpy arrays, `allow_pickle=False`):

| key | dtype / shape | meaning |
|---|---|---|
| `points` | float32 `[n_points, 3]` | canonical-frame sample (n_points = 2048 by default) |
| `transform_centre` | float64 `[3]` | `EarTransform.centre` |
| `transform_scale` | float64 scalar | `EarTransform.scale` |
| `transform_mirror_axis` | int64 scalar | `EarTransform.mirror_axis`, **-1 for None** (left ears) |
| `subject_id` | str scalar | `"P0001"` |
| `side` | str scalar | `"left"` / `"right"` |
| `n_crop_vertices` | int64 scalar | vertices inside the frozen crop box |
| `qa_suspicious` | bool scalar | `n_crop_vertices < CropConfig.min_vertices` |
| `n_points` | int64 scalar | `== points.shape[0]` |
| `seed` | int64 scalar | the run's `--seed` |
| `ear_seed` | int64 scalar | per-ear sampling seed: sha256 of `"<seed>\|<subject_id>\|<side>"`, first 8 bytes, top bit dropped |
| `crop_config_sha256` | str scalar | sha256 of the `configs/crop.yaml` used |
| `split_file` | str scalar | the `--subject-list` path the run was given |
| `split_sha256` | str scalar | sha256 of that subject list |
| `schema_version` | str scalar | `"1"` |
| `targets` | float32 `[85, 3]` | canonical GT landmarks — **present only with `--with-targets`** |

Guarantees: the **transform** is exact — for every ear with targets, the float64 canonical GT was
mapped back through the transform *as stored* (`from_dict(to_dict())`) and reproduced the original
GT to < 1e-9 mm before the float32 cast, or the run aborted. The stored float32 `targets`
themselves invert to ~1e-6 mm (float32 quantisation), which is the number to use for any tolerance
on cached labels; predictions inverted through `ear.transform` carry no such loss. A suspicious
crop (`n_crop_vertices < CropConfig.min_vertices`) is never written — it fails the run; files are
written atomically; a file left over from a run with a different `n_points`, `seed`, crop config or
target policy is refused, not silently reused (`--overwrite` rebuilds). Every ear draws its sample
from its own `ear_seed`, so a cache entry is reproducible on its own and the two ears of one
subject never share a draw.

# Role A — handover (COMPLETE, 2026-09-08)

Raw Huawei head PLY -> left/right `CanonicalEar` (2048 canonical points + an exactly invertible transform). FROZEN — ask Alfred before relying on anything else. Subject IDs are the full string `"P0001"` everywhere; `N_LANDMARKS=85`, `N_POINTS=2048`, `SIDES=("left","right")`; real PLYs carry **no normals**, so `normals is None` throughout.

## Frozen signatures — `src.data`, `src.geometry`
```python
RawSubject(subject_id, vertices[V,3]f64, faces[F,3]i64|None, normals|None, mesh_path)
EarTransform(centre[3]f64, scale>0, mirror_axis:int|None)    # .to_dict() / .from_dict(d)
CropConfig(side, lo[3], hi[3], min_vertices=500)
CanonicalEar(subject_id, side, points[N,3]f64, normals, transform, qa: dict)
load_mesh(path) -> RawSubject                        # no annotations needed
load_landmarks(path, side) -> [85,3]                 # TRAIN/DEV ONLY
load_subject_landmarks(sid, root=DATA_ROOT) -> {"left":[85,3], "right":[85,3]}
list_subjects(root); mesh_path(sid, root); landmark_path(sid, side, root)
load_crop_config(side, path="configs/crop.yaml") -> CropConfig
crop_ear(raw, side, cfg) -> (points[M,3], normals|None, qa)
make_transform(crop_points, side, mirror_side, mirror_axis=1) -> EarTransform
sample_points(points, normals, n=2048, seed=0) -> (points[n,3], normals|None)
transform_points_to_canonical(points, t)             # (p - centre)/scale, then mirror
inverse_transform_points(points, t)                  # exact inverse
canonicalize_ear(raw, side, cfg, mirror_side="right", n_points=2048, seed=0, mirror_axis=1)
```
D: inference is `load_mesh -> canonicalize_ear(raw, side, load_crop_config(side)) -> model -> inverse_transform_points`. Mesh + frozen config only; GT never defines a transform.

## Cache — read ONLY through `src.cache` (schema `"1"`)
```python
list_cached(cache_dir) -> [(subject_id, side, Path)]   # sorted; skips non-<id>_<side>.npz
load_cached_ear(path)  -> CanonicalEar                 # transform via from_dict, bit-exact
```
`qa`: `subject_id, side, n_vertices, suspicious, sampled_with_replacement, n_points, seed, ear_seed, scale, mirror_axis, crop_config_sha256, split_file, split_sha256, schema_version`, plus — only for `--with-targets` files — **`qa["targets"]`, the `[85,3]` f64 canonical GT landmarks** (the training label). `ValueError` on a wrong schema, a missing key, or a stored `subject_id`/`side` disagreeing with the filename.

npz keys: `points` f32[2048,3] · `transform_centre` f64[3] · `transform_scale` f64 · `transform_mirror_axis` i64 (**-1 = None**) · `subject_id`/`side` str · `n_crop_vertices` i64 · `qa_suspicious` bool · `n_points`/`seed`/`ear_seed` i64 · `crop_config_sha256`/`split_file`/`split_sha256`/`schema_version`("1") str · `targets` f32[85,3] *(only with `--with-targets`)*. Stored f32 targets invert to ~1e-6 mm; predictions inverted through `ear.transform` do not.

**Reproducing a cached ear — the cache is NOT `seed=0`.** Each ear was drawn with `seed = sha256("<run seed>|<subject_id>|<side>")`, first 8 bytes, top bit dropped, stored as `qa["ear_seed"]` (so the two ears of a subject never share a draw). `canonicalize_ear(raw, side, cfg)` at its default `seed=0` therefore returns a **different** 2048-point sample than the cached one — that is expected, not a cache bug. To reproduce a cached ear exactly: `canonicalize_ear(raw, side, cfg, seed=ear.qa["ear_seed"])`. The helper itself is `scripts/preprocess.py:ear_seed`, not in `src/` — ask Alfred if you need it importable from the package.

## Frozen numbers
- crop box (`configs/crop.yaml`, from `splits/train_ids.txt` ONLY, sha256 `b2d5ff90...`, `min_vertices` 500): left lo[-52.01, 40.77, -53.14] hi[29.29, 122.22, 58.69] · right lo[-55.39, -120.96, -50.96] hi[27.49, -43.49, 59.68]
- canonical GT envelope: left X[-0.384,0.459] Y[-0.119,0.521] Z[-0.613,0.682] · right X[-0.356,0.474] Y[-0.105,0.522] Z[-0.667,0.616] — every ear well inside [-1.5,1.5]; C sizes the template from this.
- transform scale ~55.9 mm left / ~55.3 mm right, i.e. **1 mm ~ 0.018 canonical units** (B: canonical loss -> mm).

## Mirror finding, and what it means for C
`mirror_side="right"`, `mirror_axis=1` (Y); **+Y = the subject's LEFT**, per the data, not the challenge page. Mirror beats no-mirror **0.0882 vs 0.5169** canonical (4.90 vs 28.74 mm), a **5.86x** margin; per-contour means 0.0903 / 0.0811 / 0.0968 / 0.0869, **spread 0.0157**. Left and right landmark orderings are therefore anatomically matched: **ONE SHARED canonical template is valid for both ears** — no per-side template, no per-side permutation, index-matched losses are sound. The 0.0882 residual is genuine inter-ear asymmetry plus ~2.6 mm X / 1.6 mm Z crop-frame mismatch, not a correspondence error; never subtract that floor from the metric.

## Rebuild the cache (needs `HUAWEI_DATA_ROOT` set)
```
python scripts/preprocess.py --subject-list splits/train_ids.txt --with-targets
python scripts/preprocess.py --subject-list splits/val_ids.txt   --with-targets
```
Defaults `--out cache/ --crop configs/crop.yaml --n-points 2048 --seed 0`; `--overwrite` rebuilds, `--workers N` parallelises. If the split ever changes, regenerate `configs/crop.yaml` FIRST (`python scripts/crop_stats.py --subject-list splits/train_ids.txt`) — a stale config is caught by the sha256 stored in every npz.

## Verifications that passed
- loaders on real data: 200/200 meshes and 400/400 landmark CSVs -> [85,3] f64, **0 failures**.
- crop freeze criterion, 160 train subjects: min leave-one-out headroom **10.94 mm**, positive on every axis both sides — PASS.
- crop on the 40 HELD-OUT val subjects: **85/85** GT landmarks inside on every ear, 0 suspicious, smallest crop **10260** vertices.
- round trip through the mesh-only transform: max |inverse(canonical(GT)) - GT| = **7.1e-15 mm** (tolerance 1e-9).
- mirror, 40 train subjects: **5.86x** margin for mirror over no-mirror, per-contour spread **0.0157**.
- cache build over both splits: **400 files / 11.6 MB**, 0 failures, every ear sampled without replacement, targets invert at **7.105e-15 mm** in f64.
- test suite: **169 passed** (`pytest tests -q`), synthetic data only; 4 of them pin the committed `configs/crop.yaml` and split against the numbers above.

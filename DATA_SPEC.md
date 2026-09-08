# DATA_SPEC — Huawei conventions (Role A)
Source: challenge topic page (public) + local dataset inspection (Alfred, 2026-09-08).

## Task
- 200 subjects, head + upper-torso PLY, EinScan Pro 2X, holes filled.
- Per subject: left and right pinna landmarks, each [85, 3].
- Metric: mean Euclidean distance per ear, averaged over all ears and hidden subjects.

## Coordinate frame (VERIFIED from challenge page — meshes are pre-aligned)
- Origin: head centre (intersection of axes below).
- X: back of head -> front (nose tip). +X = anterior.
- Y: left ear canal -> right ear canal. So left ear at Y<0, right ear at Y>0 (page wording; confirm sign with data).
- Z: vertical, +Z = up.
- Consequence for Role A: left/right mirror is a flip of Y (axis 1). Head centre ~ origin, so crop boxes can be fixed in absolute coordinates.
- Units: mm (VERIFIED from vertex ranges: a head+torso spans a few hundred units per axis).

## Landmark ordering (VERIFIED counts + listed order; index ranges assumed sequential — confirm with data/notebook)
| contour | count | assumed indices |
|---|---|---|
| outer helix | 25 | 0–24 |
| concha outline | 30 | 25–54 |
| inner helix | 20 | 55–74 |
| superior antihelix | 10 | 75–84 |
- Within-contour point order: ?
- Left and right use the same ordering: ?

## Directory layout (VERIFIED 2026-09-08)
- data root: $HUAWEI_DATA_ROOT (on Alfred's machine this is inside the repo:
  `2026 Munich Tech Arena - Datas/...` — git-ignored, never committed)
- mesh filename pattern: `$HUAWEI_DATA_ROOT/mesh/P<id>.ply`
- annotation filename pattern (left / right):
  `$HUAWEI_DATA_ROOT/landmarks/P<id>_left_ear_landmarks.csv` /
  `$HUAWEI_DATA_ROOT/landmarks/P<id>_right_ear_landmarks.csv`
- annotation file format: CSV (text, see "Annotations" below)
- subject ID pattern: `"P" + 4 digits, zero-padded` (regex `^P\d{4}`). IDs are
  NON-contiguous: P0001 .. P0308. **Codebase-wide convention: the subject ID is
  the full string, e.g. `"P0001"`** (cache keys, submission, logs).
- subject count found: 200. Every mesh has both annotation files.

## PLY (VERIFIED 2026-09-08)
- format: `binary_little_endian 1.0`
- vertex properties: `x y z red green blue alpha` (xyz float, colour uchar)
- vertex count range: 0.28M – 0.84M (median 0.40M) over 20 sampled meshes
- faces present: yes
- normals present: NO (no nx/ny/nz stored -> `RawSubject.normals is None`)
- vertex dtype: float (32-bit in file, float64 as loaded)
- global bbox X/Y/Z ranges (mm, torso included):
  - first mesh: X[-147.7, 128.2]  Y[-242.5, 211.6]  Z[-263.6, 155.8]
  - across 20 meshes: X[-257.4, 132.7]  Y[-270.5, 270.3]  Z[-309.8, 158.9]

## Annotations (PARTIALLY VERIFIED 2026-09-08)
- file: 85 lines, CRLF line endings, no header, no BOM.
- each line: exactly one comma and 3 numbers, whitespace-separated with
  variable spacing (some lines have 4–5 whitespace tokens).
- exact column layout: ? — either `"x, y z"` (3 numeric tokens) or
  `"idx,x y z"` (4 tokens, leading integer index). `load_landmarks` handles
  both; `scripts/inspect_dataset.py` section 5 prints the token-count histogram
  to settle it.
- shape and dtype as loaded: [85, 3] float64 (by construction of the loader)
- left landmark Y sign: ?   right landmark Y sign: ?
- left ear landmark bbox: ?  right ear landmark bbox: ?

## Crop configuration (?)
- source of bounds: training landmark min/max per side + margin (frozen once)
- margin per axis: max(0.25 * median per-subject ear extent on that axis, 15 mm)
  (`scripts/crop_stats.py`; values recorded in `configs/crop.yaml`)
- left crop lo / hi: ?  (fill from configs/crop.yaml once generated)
- right crop lo / hi: ?
- min_vertices threshold for "suspicious": 500
- config file path: configs/crop.yaml (derived statistic — fine to commit)

## Canonical transform (?)
- centre definition (crop bbox centre): ?
- scale definition (half max bbox extent): ?
- mirror_axis (expect 1 = Y): ?
- mirror_side (which side is mirrored; default "right"): ?
- visually confirmed on real data: ?

## Point sampling (?)
- N points per ear (default 2048): ?
- typical crop vertex count (is 2048 without replacement always possible?): ?
- sampling seed / determinism policy: ?

## Normals (VERIFIED 2026-09-08)
- present in PLY header: NO
- per-vertex or per-face: n/a
- unit length as loaded: n/a
- used by model (B decision): cannot be — no real normals exist; do not synthesise.

## Processed cache format (?)
- path pattern (cache/<subject>_<side>.npz): ?
- keys and dtypes: ?
- transform stored as (dict / separate arrays): ?
- canonical GT targets included for train/dev: ?
- total cache size on disk: ?

## Output format (?)
- submission file format expected by Huawei: ?
- per-subject file or single file: ?
- landmark array order (left then right): ?
- coordinate frame (original Huawei XYZ): ?
- dtype / precision: ?

## Train/val split (owned by C — reference only) (?)
- split definition file: ?
- number of train / val subjects: ?
- split seed or fixed list: ?

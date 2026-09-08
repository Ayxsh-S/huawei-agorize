# DATA_SPEC — Huawei conventions (Role A)
Source: challenge topic page (public) + local dataset inspection (Alfred, 2026-09-08).

## Task
- 200 subjects, head + upper-torso PLY, EinScan Pro 2X, holes filled.
- Per subject: left and right pinna landmarks, each [85, 3].
- Metric: mean Euclidean distance per ear, averaged over all ears and hidden subjects.

## Coordinate frame (meshes are pre-aligned)
- Origin: head centre (intersection of axes below).
- X: back of head -> front (nose tip). +X = anterior.
- Y: **+Y = the subject's LEFT** (VERIFIED on the real landmarks 2026-09-08:
  left-ear landmarks sit at Y ~ +75 mm, right-ear landmarks at Y ~ -80 mm).
  This is the OPPOSITE of the challenge page's wording ("left ear canal ->
  right ear canal"); the data wins. Anything that assumes the page's sign —
  including a Y-sign sanity check — is wrong.
- Z: vertical, +Z = up.
- Consequence for Role A: left/right mirror is still a flip of Y (axis 1); only
  the sign convention's meaning changed, not the mirror axis. Head centre ~
  origin, so crop boxes can be fixed in absolute coordinates.
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

## Annotations (VERIFIED 2026-09-08)
- file: 85 lines, CRLF line endings, no header, no BOM.
- exact column layout (VERIFIED): `<idx>,[<x> <y> <z>]` — a landmark index, one
  comma, then a **numpy array repr in square brackets**: variable spacing
  between the coordinates and occasionally scientific notation (`5.56e-02`).
- `idx` runs 0..84 and equals the landmark's 0-based position in the file.
  `load_landmarks` requires this, so a reordered/duplicated/gappy annotation is
  refused rather than silently loaded in the wrong order.
- `load_landmarks` accepts this layout only (the earlier `"x, y z"` /
  `"idx,x y z"` candidates are gone). Errors name file + line number + reason
  and never quote the line, so annotation coordinates cannot leak into logs.
- shape and dtype as loaded: [85, 3] float64 (by construction of the loader)
- left landmark Y sign: **+** (mean Y ~ +75 mm)
  right landmark Y sign: **-** (mean Y ~ -80 mm)   -> see "Coordinate frame"
- left ear landmark bbox: ?  right ear landmark bbox: ?
  (fill from `scripts/inspect_dataset.py` section 6)

## Crop configuration (?)
- **derived from `splits/train_ids.txt` ONLY** — never the full dataset. Val
  subjects must not influence a frozen preprocessing parameter (no-leakage rule,
  `CLAUDE.md`). `crop_stats.py` requires `--subject-list` and records that file's
  path, sha256 and subject count inside `configs/crop.yaml`, so a config left over
  from a different split is detectable. Regenerate `crop.yaml` whenever the train
  set changes. The split itself is provisional (Role A default 160/40, seed 42);
  Role C owns the real one — see `splits/README.md`.
- source of bounds: training landmark min/max per side + margin (frozen once)
- margin per axis: max(0.25 * median per-subject ear extent on that axis, 15 mm)
  (`scripts/crop_stats.py`; values recorded in `configs/crop.yaml`)
- left crop lo / hi: ?  (fill from configs/crop.yaml once generated)
- right crop lo / hi: ?
- min_vertices threshold for "suspicious": 500
- config file path: configs/crop.yaml (derived statistic — fine to commit)
- freeze criterion (printed by `crop_stats.py`): the minimum **leave-one-out**
  headroom must be > 0 on every axis for both sides. Each subject is measured
  against a box rebuilt without it, which is what an unseen subject would face;
  "no training subject is outside its own envelope" is true by construction and
  proves nothing.
- leave-one-out min headroom, left / right: ?  (fill once run on real data)
- a run that fails the criterion writes `crop.rejected.yaml`, never `crop.yaml`,
  and `load_crop_config` refuses any file carrying `freeze_criterion_passed:
  false`.

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

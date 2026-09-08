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
- vertex count range (VERIFIED over all 200 meshes, 200/200 loaded, 0 failed):
  min 1.101e5, median 3.346e5, max 9.325e5
- faces present: yes — face count min 2.203e5, median 6.693e5, max 1.863e6
- normals present: NO (no nx/ny/nz stored -> `RawSubject.normals is None`)
- vertex dtype: float (32-bit in file, float64 as loaded)
- global vertex envelope over ALL 200 meshes (mm, torso included; VERIFIED):
  X[-257.4, 140.2]  Y[-298.1, 270.3]  Z[-312.3, 164.7]
  (the head is a small part of this box — the torso and shoulders dominate it;
  the ear crop boxes below are what actually matter.)

## Annotations (VERIFIED 2026-09-08)
- file: 85 lines, CRLF line endings, no header, no BOM. VERIFIED over all
  400 files (200 subjects x 2 sides): 400/400 parsed, every one [85, 3] float64,
  every line 4 tokens with exactly one comma and one bracket pair.
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
- landmark envelopes over ALL 200 subjects (mm, VERIFIED 2026-09-08):
  - left:  X[-37.01, 14.29]  Y[ 52.10, 107.80]  Z[-38.01, 43.56]
  - right: X[-40.39, 12.49]  Y[-106.0, -54.68]  Z[-35.96, 44.68]
- per-subject ear bbox extent, median over all 200 subjects (mm):
  - left:  X 34.63  Y 24.10  Z 60.42   bbox diagonal median 74.14
  - right: X 34.48  Y 23.78  Z 59.83   bbox diagonal median 73.80
  An ear is therefore ~35 x 24 x 60 mm — Z (vertical) is the long axis, so
  `make_transform`'s "half the largest bbox extent" scale is driven by Z.
- NOTE: these envelopes are over all 200 subjects, whereas `configs/crop.yaml`
  is built from the 160 TRAINING subjects only, so the all-subject envelope
  sticks out past the training envelope (e.g. left Y max 107.80 vs training
  107.22; right Y max -54.68 vs training -58.49). Both stay well inside the
  frozen crop box — that gap is exactly the headroom the margin buys, and
  `scripts/check_crop_all.py` re-checks it on the 40 held-out subjects.

## Crop configuration (FROZEN + VERIFIED 2026-09-08)
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
- **FROZEN** in `configs/crop.yaml`, generated 2026-09-08 from
  `splits/train_ids.txt` (sha256 `b2d5ff90...`, 160 subjects, 0 failures):
  - left  lo [-52.01,   40.77, -53.14]   hi [29.29,  122.22, 58.69]
  - right lo [-55.39, -120.96, -50.96]   hi [27.49,  -43.49, 59.68]
- the 15 mm floor binds on every axis except left Z (margin 15.13 there), i.e.
  0.25 * median ear extent < 15 mm for X and Y; only the long Z axis exceeds it.
- min_vertices threshold for "suspicious": 500
- config file path: configs/crop.yaml (derived statistic — fine to commit)
- freeze criterion (printed by `crop_stats.py`): the minimum **leave-one-out**
  headroom must be > 0 on every axis for both sides. Each subject is measured
  against a box rebuilt without it, which is what an unseen subject would face;
  "no training subject is outside its own envelope" is true by construction and
  proves nothing.
- leave-one-out min headroom (mm, VERIFIED — freeze criterion **PASS**):
  - left:  X 13.83  Y 11.36  Z 11.32
  - right: X 12.21  Y 14.35  Z 10.94
  All positive on every axis, both sides -> safe to freeze. The tightest
  training subjects were P0130, P0307, P0174, P0013, P0111.
- observed crop size: ~23-25k vertices per ear on P0001/P0002, both with
  85/85 GT landmarks inside the box.
- **held-out generalisation check PASSED** (Alfred, 2026-09-08,
  `scripts/check_crop_all.py --subject-list splits/val_ids.txt`): on all 40
  validation subjects the box was NOT derived from, every ear keeps 85/85 GT
  landmarks inside the box, no crop is suspicious, and the smallest crop is
  10260 vertices (~5x the 2048 sampled, so sampling stays without
  replacement). **The crop config is FROZEN from here on.**
- a run that fails the criterion writes `crop.rejected.yaml`, never `crop.yaml`,
  and `load_crop_config` refuses any file carrying `freeze_criterion_passed:
  false`.

## Canonical transform (frozen semantics; mirror axis VERIFIED)
- centre definition: the **cropped mesh** bbox centre, `0.5 * (lo + hi)` over
  the cropped vertices. Mesh-derived only — GT never enters it.
- scale definition: **half the largest** cropped bbox extent, so the canonical
  ear spans about [-1, 1] on its longest axis (Z, per the extents above).
- mirror_axis: **1 (Y)** — VERIFIED: +Y is the subject's left on all 200
  subjects, so left and right ears are Y-reflections of one another.
- mirror_side: `"right"` (`canonicalize_ear` default); the left ear is left
  untouched and the right ear gets its Y component negated.
- visually confirmed on real data: NOT YET (§3 of `docs/STATUS_A.md` — plot
  5-10 subjects). The Y **sign convention** is confirmed numerically on all 200
  subjects; the visual sanity check of the canonicalised ears is still open.
- **exact invertibility VERIFIED on real data** (Alfred, 2026-09-08,
  `scripts/check_roundtrip.py`, transform built from the mesh only, GT pushed
  through `to_dict()`/`from_dict()` and back): max
  |inverse(canonical(GT)) - GT| = **7.1e-15 mm** over every ear checked,
  i.e. float64 rounding — PASS against the 1e-9 mm tolerance.
- canonical envelope of the GT landmarks: reported by the same script (tells
  Role C whether the frame suits a template) — Alfred to paste the numbers.

## Point sampling (VERIFIED 2026-09-08)
- N points per ear: **2048** (`geometry.N_POINTS`).
- typical crop vertex count: ~23-25k vertices per ear (P0001/P0002); the
  smallest crop over the 40 held-out val subjects is 10260 vertices (VERIFIED
  2026-09-08), i.e. >= 5x the 2048 needed, so sampling is **without
  replacement** in practice. `sample_points` still falls back to replacement
  if a crop ever came up short, `canonicalize_ear` records
  `sampled_with_replacement` in the QA dict, and `scripts/preprocess.py`
  counts such ears in its summary.
- sampling seed / determinism policy (SHIPPED in `scripts/preprocess.py`): a
  deterministic `numpy.default_rng` seeded per ear with
  `ear_seed = sha256("<seed>|<subject_id>|<side>")[:8]` as a big-endian int
  with the top bit dropped (fits int64; `--seed` defaults to 0). A cache entry
  is therefore reproducible on its own and the two ears of one subject never
  share a draw. The per-ear seed is stored in the npz (`ear_seed`).

## Normals (VERIFIED 2026-09-08)
- present in PLY header: NO — confirmed on all 200 meshes (not one file
  declares nx/ny/nz), so `RawSubject.normals is None` throughout and no normal
  feature channel exists anywhere in the pipeline.
- per-vertex or per-face: n/a
- unit length as loaded: n/a
- used by model (B decision): cannot be — no real normals exist; do not synthesise.

## Processed cache format (schema "1", `src/cache.py`, built by `scripts/preprocess.py`)
- path pattern: `cache/<subject_id>_<side>.npz`, e.g. `cache/P0001_left.npz`
  (git-ignored). Read ONLY via `src.cache.load_cached_ear` / `list_cached`;
  full key table in `docs/HANDOFF_A.md`.
- keys and dtypes: `points` float32 [2048,3] canonical; `transform_centre`
  float64 [3], `transform_scale` float64, `transform_mirror_axis` int64 (-1 =
  None); `subject_id`, `side` str; `n_crop_vertices` int64; `qa_suspicious`
  bool; `n_points`, `seed`, `ear_seed` int64; `crop_config_sha256`,
  `split_file`, `split_sha256`, `schema_version` ("1") str.
- transform stored as: separate float64 arrays holding exactly the
  `EarTransform.to_dict()` fields, rebuilt with `from_dict` (bit-exact).
- canonical GT targets included for train/dev: yes, `targets` float32 [85,3],
  ONLY when built with `--with-targets`; each ear's targets are verified at
  write time to invert back to the original GT to < 1e-9 mm (float64, through
  the stored transform) or the run aborts naming the subject.
- staleness guard: an existing file built with a different `n_points`, `seed`,
  crop config sha256 or target policy fails the run instead of being skipped.
- total cache size on disk: ~25 KB per ear without targets (2048 x 3 float32 +
  metadata), ~26 KB with; ~10 MB for all 400 ears. Exact figure printed by
  `preprocess.py` — Alfred to record after the real run.

## Output format (?)
- submission file format expected by Huawei: ?
- per-subject file or single file: ?
- landmark array order (left then right): ?
- coordinate frame (original Huawei XYZ): ?
- dtype / precision: ?

## Train/val split (owned by C — reference only) (PROVISIONAL Role A default)
- split definition file: `splits/train_ids.txt` / `splits/val_ids.txt` (IDs only,
  committed). Train list sha256 `b2d5ff90...`, recorded inside `configs/crop.yaml`.
- number of train / val subjects: 160 / 40, subject-level (both ears of a
  subject always land in the same fold)
- split seed or fixed list: fixed lists, generated by `scripts/make_split.py`
  with `numpy.default_rng(42)` over the sorted IDs
- STATUS: this is Role A's provisional split, made only so the crop box had a
  training set to be derived from. **Role C owns the real one** — if it changes,
  `configs/crop.yaml` must be regenerated (the sha256 above is how a stale
  config is detected). See `splits/README.md`.

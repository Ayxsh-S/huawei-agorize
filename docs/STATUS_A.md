# STATUS — Role A (Alfred)

Working plan and running status. Read at the start of every task; tick items off
when done. Milestone order is fixed: **loader → crop → transform/inverse →
sampling → cache**.

Created 2026-09-08. Deadline 2026-09-15.

## 1. Verify the data conventions (blocking most of the rest)
- [x] `scripts/inspect_dataset.py` written (aggregates only, never per-point dumps)
- [x] Run it on the real data and fill in `DATA_SPEC.md` (Alfred, 2026-09-08):
  - [x] units mm, global bbox X/Y/Z ranges, vertex-count range (full run,
        200 meshes: 0.11M–0.93M, median 0.33M; the earlier 0.28M–0.84M came
        from a 20-mesh sample)
  - [x] faces present, NO normals, float xyz + uchar rgba in the PLYs
  - [x] `mesh/P<id>.ply`, `landmarks/P<id>_{left,right}_ear_landmarks.csv`,
        subject ID = full `"P0001"` string, 200 subjects, IDs non-contiguous
  - [x] annotation file format: 85-line CSV, CRLF, one comma per line
  - [x] **CSV column layout** (Alfred, 2026-09-08): `<idx>,[<x> <y> <z>]` — a
        numpy array repr in brackets, variable spacing, sometimes scientific
        notation; `idx` = the landmark's 0-based position. `load_landmarks`
        accepts this layout only; `inspect_dataset.py` section 5 now reports
        conformance to it instead of guessing between two candidates.
  - [x] left/right landmark **Y signs** (Alfred, 2026-09-08): **+Y = subject's
        left** — left ears at Y ≈ +75, right ears at Y ≈ -80. This is the
        OPPOSITE of the challenge page; `DATA_SPEC.md` and the
        `inspect_dataset.py` sanity check now follow the data. `mirror_axis`
        stays 1 (Y) and `mirror_side` is unchanged.
  - [x] left/right landmark bboxes (Alfred, 2026-09-08, all 200 subjects):
        left X[-37.01,14.29] Y[52.10,107.80] Z[-38.01,43.56]; right
        X[-40.39,12.49] Y[-106.0,-54.68] Z[-35.96,44.68]; median ear extent
        ~35 x 24 x 60 mm (Z is the long axis) -> `DATA_SPEC.md`
  - [ ] within-contour point order; contour index ranges really sequential (§3)
  - [ ] left and right use the same ordering? (§3)

## 2. Raw loaders — `src/data.py`
- [x] `RawSubject` dataclass (frozen interface, see `docs/HANDOFF_A.md`)
- [x] **`load_mesh(path)`** — trimesh `process=False` (no vertex merge/reorder),
      float64 `[V,3]` vertices, int64 `[F,3]` faces or `None`, normals only when
      the PLY header declares `nx/ny/nz` (never `trimesh.vertex_normals`, which
      synthesises them). Works with no annotations — the inference entry point.
- [x] `subject_id_from_path(path)` — `^(P\d{4})` at the start of the stem,
      returns the full `"P0001"` string (provisional digit-run rule removed)
- [x] `list_subjects(root)` — sorted IDs of `mesh/P*.ply`
- [x] `mesh_path(sid, root)`, `landmark_path(sid, side, root)` helpers
- [x] `load_landmarks(path, side)` — `[85,3]` float64; the one verified layout
      `<idx>,[<x> <y> <z>]`, refuses wrong line counts / malformed lines / an
      `idx` that disagrees with its position (naming the line number but never
      its contents), refuses a file named for the other side.
      `parse_landmark_line` (now returning `(index, xyz)`) and
      `landmark_line_tokens` exposed.
- [x] `load_subject_landmarks(sid, root)` — `{"left": [85,3], "right": [85,3]}`
- [x] `tests/test_data.py` — synthetic PLYs + CSVs in `tmp_path`: the real
      layout (incl. a scientific-notation line and a double-space line),
      CRLF/LF, wrong line count, bad-line numbering, index mismatch, non-layout
      lines, "errors never quote the line", side mismatch, ID rule on mesh and
      landmark filenames, `list_subjects` ignores stray files
- [x] Run the loaders on real data once (Alfred, 2026-09-08): 200/200 meshes
      loaded, 0 failures, no stored normals in any file; 400/400 landmark
      CSVs parsed to [85,3] float64 on the one verified layout

## 3. QA visualisation
- [x] `scripts/plot_ear.py` — grey mesh points (subsampled to 20k) + landmarks
      coloured per contour with start-index labels and in-order contour lines;
      `--subject P0001` resolves paths via `src.data`; `--crop configs/crop.yaml`
      draws both crop boxes as wireframes on the head view and plots the cropped
      points per side (prints crop vertex count + landmarks-inside-box count).
      Smoke-tested on synthetic data only.
- [ ] Plot 5–10 subjects, eyeball crop/ordering, feed findings back into §1

## 4. Ear crop — `src/geometry.py`
- [x] `CropConfig` + `crop_ear(raw, side, cfg)` with QA dict
- [x] `scripts/crop_stats.py` — per side: landmark envelope over the **training**
      subjects, median per-subject ear extent, margin = max(0.25·median extent,
      15 mm); writes `configs/crop.yaml` (lo/hi/min_vertices, the stats used, and
      the split's path + sha256 + subject count). `--subject-list` is **required**
      (no accidental whole-dataset run); any load failure aborts with a non-zero
      exit and no YAML unless `--allow-failures`. Generalisation is measured by a
      **leave-one-out** check (each subject vs a box rebuilt without it), printing
      the 5 worst subjects, the min LOO headroom per side/axis and the freeze
      criterion (min LOO headroom > 0 on every axis, both sides). The old
      by-construction "outside the envelope" check is gone — it proved nothing.
      **Run on the real training split 2026-09-08; bounds frozen (see below).**
- [x] `scripts/make_split.py` — **provisional** subject-level split (Role C owns
      the real one): 160/40, `default_rng(42)`, IDs sorted before shuffling, both
      ears together by construction. Writes `splits/{train,val}_ids.txt` +
      `splits/README.md`. `splits/` is in the repo (IDs only, no data).
- [x] Run `make_split.py` on real data → `splits/train_ids.txt` (160) +
      `splits/val_ids.txt` (40); train list sha256 `b2d5ff90...`, recorded in
      `configs/crop.yaml`. Still the PROVISIONAL Role A split — Role C ratifies.
- [x] `load_crop_config(side, path="configs/crop.yaml") -> CropConfig`
- [x] Run `crop_stats.py --subject-list splits/train_ids.txt` on real data →
      `configs/crop.yaml` **FROZEN** (Alfred, 2026-09-08, 160 subjects, 0
      failures). Freeze criterion **PASS**; lo/hi and LOO headroom copied into
      `DATA_SPEC.md`. From here on, inference = mesh + this config only.
- [x] `scripts/check_crop_all.py` written — runs the frozen box over a subject
      list and reports, per side, min/median/max cropped vertex count, suspicious
      crops, ears with < 85/85 GT landmarks inside, and the offending IDs. Exit 1
      on any truncated ear or suspicious crop. Synthetic tests only; not yet run.
- [x] Alfred: run `check_crop_all.py --subject-list splits/val_ids.txt` — the
      real generalisation test, on the 40 subjects the box was NOT derived from
      (the LOO headroom is only an estimate of this). **PASSED** (Alfred,
      2026-09-08): 85/85 landmarks inside on every ear of all 40 held-out
      subjects, no suspicious crop, min crop 10260 vertices. Recorded in
      `DATA_SPEC.md`. **`configs/crop.yaml` is now FROZEN.**

## 5. Canonical transform + inverse — `src/geometry.py`
- [x] `EarTransform`, `make_transform`, `transform_points_to_canonical`,
      `inverse_transform_points`, `canonicalize_ear`
- [x] Round-trip tests in `tests/test_geometry.py`
- [x] `scripts/check_roundtrip.py` written — builds the transform from the
      MESH only (frozen crop + `make_transform`), pushes the GT landmarks to
      canonical and back, and fails (exit 1) above 1e-9 mm. Also reports the
      canonical envelope of the GT landmarks and how many ears leave
      [-1.5, 1.5], which is what Role C needs to size the template. Synthetic
      tests only; not yet run.
- [x] Alfred: run `check_roundtrip.py` — **PASSED** (Alfred, 2026-09-08): max
      |inverse(canonical(GT)) - GT| = 7.1e-15 mm (float64 rounding), recorded
      in `DATA_SPEC.md`.
- [x] Canonical envelope pasted into `DATA_SPEC.md` (2026-09-08): left
      X[-0.384,0.459] Y[-0.119,0.521] Z[-0.613,0.682]; right X[-0.356,0.474]
      Y[-0.105,0.522] Z[-0.667,0.616]; transform scale ~55.9 mm left / ~55.3 mm
      right (1 mm ~ 0.018 canonical units). Every ear well inside [-1.5, 1.5] —
      Role C can size the template from this.
- [x] Mirror **axis** confirmed as Y on real data: +Y = subject's left on all
      200 subjects, so left/right are Y-reflections (`mirror_axis=1`,
      `mirror_side="right"`)
- [x] `scripts/check_mirror.py` written — compares each subject's own canonical
      left ear against its own canonical right ear under three configurations
      (A no mirror, B mirror right = current default, C mirror left), rebuilding
      both ears from the mesh with the frozen crop config every time (the cache
      is never read). Two metrics, mean/median/p95 over subjects: symmetric
      Chamfer on a 512-point subsample of each ear's 2048 canonical points, and
      the mean index-matched distance over the 85 GT landmarks (decisive).
      Prints the verdict with the margin over the runner-up, the per-contour
      means for the winner (so a misordered contour shows up), the worst 5
      subjects, and writes `outputs/mirror_check.png` (winning configuration,
      one subject, two viewing angles). B and C are an exact tie by construction
      — the run decides mirror vs no mirror. **Changes no default**: if the
      default loses it shouts and exits 0. Synthetic tests only; not yet run.
- [ ] **Alfred: run `python scripts/check_mirror.py --subject-list
      splits/train_ids.txt`** and paste the verdict into the PENDING block under
      `DATA_SPEC.md` → "Mirror". This is the last open assumption in §5.
- [ ] Visual check of the canonicalised ears themselves (§3) — the sign
      convention is confirmed numerically, the picture is not
      (`outputs/mirror_check.png` from the run above covers the canonical frame)

## 6. Sampling
- [x] `sample_points(points, normals, n=2048, seed=0)` — deterministic
- [x] Normals: the PLYs carry none (verified), so no normal feature channel.
      `RawSubject.normals` will always be `None` on real data.

## 7. Processed cache — `scripts/preprocess.py` + `src/cache.py`
- [x] `src/cache.py` — schema `"1"` in one place: `write_cached_ear` (atomic
      temp-file + rename), `load_cached_ear(path) -> CanonicalEar` (transform
      via `EarTransform.from_dict`, targets in `qa["targets"]`, refuses a wrong
      schema version / missing key / renamed file), `list_cached(cache_dir) ->
      [(subject_id, side, path)]`, `cache_path`. B and C never parse npz keys.
- [x] `scripts/preprocess.py` — `--subject-list` (required) `--out cache/`
      `--crop configs/crop.yaml` `--n-points 2048` `--seed 0` `[--with-targets]
      [--overwrite] [--workers N] [--limit N] [--root]`. Per-ear seed =
      sha256(seed|subject|side) (stored as `ear_seed`); targets verified to
      invert to < 1e-9 mm through the stored transform or the run ABORTS
      naming the ear; stale entries (other n_points/seed/crop sha/target
      policy) fail instead of being skipped; progress every 25 subjects;
      summary with written/skipped/failed, cache size, min/median/max crop
      vertices; a suspicious crop (< min_vertices) fails the ear and writes
      nothing, ears sampled with replacement are written but named; exit 1 on
      any failure. Synthetic tests only (35 in
      `tests/test_preprocess.py` + `tests/test_cache.py`, incl. a 2-worker run
      on an on-disk synthetic root).
- [x] `docs/HANDOFF_A.md` — exact npz key table + the two `src.cache` functions.
- [x] Alfred ran `preprocess.py --with-targets` on both splits (2026-09-08):
      **400 files / 11.6 MB**, 0 failures, crop vertices min 10260 / median
      ~25000 / max 66654 over all ears, targets invert at 7.105e-15 mm in
      float64 (stored float32 targets invert to ~1e-6 mm). Recorded in
      `DATA_SPEC.md`. **Tell B and C the cache is ready** (read it only via
      `src.cache.load_cached_ear` / `list_cached`, key table in
      `docs/HANDOFF_A.md`).

## Open questions for Alfred
- PyYAML is used for `configs/crop.yaml` (already installed in the venv, 6.0.3);
  add it to the requirements file when one exists.
- The workspace is not a git repository yet, so `.gitignore` could not be
  verified with `git status`. Before `git init`, note that the dataset folder,
  its zip and `*.csv` are now ignored.

## Done log
- 2026-09-08 — **Cache numbers recorded; `scripts/check_mirror.py` written.**
  `DATA_SPEC.md` now carries Alfred's real-run figures: cache 400 files /
  11.6 MB, crop vertices min 10260 / median ~25000 / max 66654 over all 400
  ears, targets inverting at 7.105e-15 mm in float64 (float32 as stored:
  ~1e-6 mm), the canonical GT envelope (left X[-0.384,0.459] Y[-0.119,0.521]
  Z[-0.613,0.682]; right X[-0.356,0.474] Y[-0.105,0.522] Z[-0.667,0.616]) and
  the transform scales (~55.9 / ~55.3 mm, so 1 mm ~ 0.018 canonical units).
  New `scripts/check_mirror.py --subject-list ... [--limit N] [--out outputs/]`
  settles the last open assumption (`mirror_side="right"`, `mirror_axis=1`)
  numerically: three configurations (A none, B mirror right, C mirror left),
  each rebuilt from the mesh with the frozen crop config — the cache is never
  read, since it is fixed to the current setting; crop and point draw are done
  once per ear and shared, so only the mirror differs. Metrics per subject:
  symmetric Chamfer (mean of both directed mean NN distances) on a 512-point
  subsample of each ear's 2048 canonical points, and the mean index-matched
  distance over the 85 landmarks (decisive), aggregated mean/median/p95, plus
  per-contour means for the winner, the worst 5 subjects, and
  `outputs/mirror_check.png` (winning configuration, two viewing angles).
  B and C are proved an exact tie (mirroring both clouds is an isometry) and
  the verdict says so; a losing default triggers a loud banner but **no default
  is touched** and the exit code stays 0 — only an unrunnable check exits 1.
  Config B is guarded at import against `canonicalize_ear`'s defaults drifting,
  and the contour ranges are guarded against gaps/overlaps. 20 new synthetic
  tests in `tests/test_scripts.py` (mirrored world -> B/C exact and A 0.4 off;
  translated world -> A exact and the banner fires; B == C bit-for-bit; a
  reversed concha shows up in its own contour row only; empty/suspicious crops
  refused; --limit / --allow-failures / rejected-config paths; the plot is
  written from the WINNING configuration, not the default). 157 tests pass;
  every new guard mutation-checked (the plot-uses-the-winner test was added
  because the mutation survived without it). senior-reviewer round 1
  **FIX-FIRST**, all items applied: the verdict now prints BEFORE the figure and
  the whole drawing (not just the matplotlib import) is inside one try, so a
  backend or `--out` failure can no longer eat the numbers a full dataset pass
  just bought; a new `frame_floor` block reports the residual a perfectly
  mirror-symmetric subject would still score (the two frozen crop boxes are not
  exact Y-reflections — mirrored centres ~3 mm apart, scales 55.9 vs 55.3 mm,
  so the decisive metric has a ~0.06 canonical floor), and `DATA_SPEC.md` now
  says to read the per-contour SPREAD, not the level, as the ordering signal;
  new tests exercise `subsample`'s real 2048→512 branch (seed-determinism, no
  duplicates), pin `check_mirror`'s crop→sample→transform against
  `canonicalize_ear` bit-for-bit under configuration B, and pin the floor's
  formula and the plot-failure path. The reviewer also caught a stale
  `make_transform` docstring in `src/geometry.py` still repeating the disproven
  "Y runs left ear canal to right ear canal" — corrected to the verified
  convention (docstring only; no signature touched). Reviewer round 2 caught a
  real error in that new floor: `‖flip(c_right) − c_left‖` is the true residual
  only if the subject's own mid-sagittal plane is exactly y = 0 — a subject
  perfectly mirrored about y = y0 scores 0 on the decisive metric yet shows
  2·y0/scale on the mirror axis — so the floor is now reported **per axis**
  (`FrameFloor`): X and Z are a genuine floor (a Y-reflection cannot move a
  point in X or Z), Y is printed as an upper bound, and both the script and
  `DATA_SPEC.md` state that none of it may be subtracted from the metric
  (a constant frame offset and per-landmark errors combine as vectors). A test
  builds the y0 = 3 case (perfect subject, 0.6 on the Y term, 0 on X/Z), the
  stale `# pragma: no cover` on the plot guard is gone, and the four new floor
  guards were mutation-checked. Reviewer round 3: **SHIP**, no open items.
  165 tests pass. **Alfred still has to run the script** — Role A never touches
  the data — and paste the verdict into the PENDING block under `DATA_SPEC.md`
  → "Mirror".
- 2026-09-08 — **Both real-data checks PASSED, crop config FROZEN, cache
  built.** Alfred ran `check_roundtrip.py` (max round-trip error 7.1e-15 mm)
  and `check_crop_all.py --subject-list splits/val_ids.txt` (85/85 landmarks
  inside on all 40 held-out subjects, no suspicious crop, min crop 10260
  vertices); both recorded in `DATA_SPEC.md`. `scripts/preprocess.py` is real:
  `--subject-list` (required) `--out --crop --n-points --seed [--with-targets]
  [--overwrite] [--workers N] [--limit N] [--root]`; per-ear sampling seed =
  sha256("seed|subject|side") first 8 bytes, top bit dropped (stored as
  `ear_seed`, value for (0, P0001, left) pinned by a test); targets are
  pushed through the mesh-only transform and verified to invert to < 1e-9 mm
  in float64 through `from_dict(to_dict())` before the float32 cast, else the
  run ABORTS naming the ear and writes nothing for it; an existing file with
  another n_points / seed / crop sha256 / target policy (either direction)
  is a failure, not a skip; suspicious crops fail the ear and are not
  written; with-replacement ears are written but named; progress every 25
  subjects; summary with written/skipped/failed, cache size, min/median/max
  crop vertices per side; stray temp files swept; exit 1 on any failure.
  New `src/cache.py` owns schema "1" (`write_cached_ear` atomic,
  `load_cached_ear` -> `CanonicalEar` with the transform rebuilt via
  `from_dict` and targets in `qa["targets"]`, `list_cached`, `cache_path`);
  `docs/HANDOFF_A.md` has the exact key table; `DATA_SPEC.md` cache section
  filled. 35 new synthetic tests (`tests/test_cache.py`,
  `tests/test_preprocess.py`), including a 2-worker in-process run and a
  subprocess run of the script itself with `--workers 2` and a corrupt PLY.
  The CLI was also smoke-run by hand with two spawned workers on a synthetic
  root. senior-reviewer round 1: **FIX-FIRST**, all four items applied (the
  symmetric targets staleness check, naming suspicious / with-replacement
  ears and failing suspicious ones, the HANDOFF wording that the 1e-9
  guarantee is on the float64 values and stored float32 targets invert to
  ~1e-6 mm, and the `__mp_main__` subprocess test); nice-to-have temp sweep
  applied; left as-is: `sampled_with_replacement` is derived on load rather
  than stored (schema stays as briefed) and the "cwd then repo root" config
  resolution stays private in `src/geometry.py` (a public helper would touch
  the frozen module — Alfred's call). Reviewer round 2: **SHIP**; its last
  nice-to-have applied too (no pre-run temp sweep, so a concurrent run
  sharing `cache/` cannot lose its in-flight file; the post-run sweep
  tolerates a file another process holds open). 137 tests pass.
- 2026-09-08 — **CROP FROZEN** from Alfred's full-dataset run. `configs/crop.yaml`
  built from `splits/train_ids.txt` (160 subjects, sha256 `b2d5ff90...`, 0
  failures): left lo[-52.01, 40.77, -53.14] hi[29.29, 122.22, 58.69]; right
  lo[-55.39, -120.96, -50.96] hi[27.49, -43.49, 59.68]. Margin is the 15 mm
  floor on every axis except left Z (15.13). Freeze criterion **PASS** — min
  leave-one-out headroom left X 13.83 / Y 11.36 / Z 11.32, right X 12.21 /
  Y 14.35 / Z 10.94 mm, all > 0; tightest training subjects P0130, P0307,
  P0174, P0013, P0111. Observed crops ~23-25k vertices per ear (P0001/P0002),
  85/85 landmarks inside on both, so 2048 points are sampled without
  replacement with ~11x headroom. Dataset facts recorded in `DATA_SPEC.md`:
  200/200 meshes (vertices 1.101e5 / 3.346e5 / 9.325e5 min/median/max, faces
  2.203e5 / 6.693e5 / 1.863e6, no stored normals anywhere), 400/400 landmark
  files, global vertex envelope X[-257.4, 140.2] Y[-298.1, 270.3]
  Z[-312.3, 164.7] mm, and the per-side landmark envelopes. Two new scripts for
  Alfred to run: `scripts/check_roundtrip.py` (mesh-only transform, GT pushed
  through and back, exit 1 above 1e-9 mm, plus the canonical envelope for Role
  C) and `scripts/check_crop_all.py` (frozen box on the held-out val subjects;
  exit 1 on any truncated ear or suspicious crop). `crop_stats._describe_existing`
  became the public `describe_crop_config` so both new scripts print which
  frozen config and split they validated. 99 tests pass (20 new, synthetic
  only); every new guard was mutation-checked.
- 2026-09-08 - senior-reviewer FIX-FIRST applied to the two new scripts: a
  listed subject with no mesh now goes into `failures` instead of a warning
  line, so a wrong `--root` or a stale subject list can no longer check 3 ears
  of 400 and still print PASS (both scripts gained `--allow-failures` with the
  same meaning as `crop_stats.py`, and `--limit` is applied before the mesh
  lookup so an out-of-scope subject is not reported missing). The only
  failing-verdict round-trip test used `--tol -1`, which pinned nothing: a test
  now monkeypatches `inverse_transform_points` to be wrong by 1e-3 and asserts
  the run fails at the default tolerance and reports that error. Also from the
  review: `check_roundtrip` inverts through `EarTransform.from_dict(to_dict())`
  (the cache path Role D will actually use, so serialisation loss is measured),
  asserts `MIRROR_SIDE`/`MIRROR_AXIS` against `canonicalize_ear`'s defaults at
  import so the check cannot drift from the pipeline, and both scripts echo
  `--limit` in the verdict; `check_crop_all` warns when the subject list is the
  very split `crop.yaml` was built from (a PASS there is true by construction).
  Left as-is by decision: a crop below `N_POINTS` but above `min_vertices` is
  reported, not failed - Alfred's call, and the brief says exit 1 only on a
  truncated ear or a suspicious crop. Reviewer verdict: **SHIP**; its three
  nice-to-haves were applied too (the "same split as the config" caveat is
  repeated in the verdict block, `check_crop_all --allow-failures` is now
  pinned by a test, and the convention guards raise `RuntimeError` instead of
  `assert`, which `python -O` would strip). 102 tests pass.
- 2026-09-08 — real-data findings applied: annotation layout confirmed as
  `<idx>,[<x> <y> <z>]`, so `parse_landmark_line` strips brackets/commas,
  demands exactly 4 tokens and returns `(index, xyz)`; `load_landmarks` requires
  `idx` to equal the landmark's 0-based position and no error path quotes a
  line's contents any more (file + line number + reason only). Tests rewritten
  onto the real layout (scientific notation, double spacing) plus new
  index-mismatch and "never quotes the line" tests; the three new guards were
  mutation-checked. Y convention corrected to **+Y = subject's left** in
  `DATA_SPEC.md` and in `inspect_dataset.py` (left mean Y > 0, right mean Y < 0;
  prints "NOT CHECKED" when no ear loaded); `mirror_side`/`mirror_axis`
  untouched. Stray empty `test.py` at the repo root deleted.
- 2026-09-08 - reviewer round 4 FIX-FIRST applied to the loader: the
  non-numeric-token error is raised `from None` (float()'s own message
  quotes the token, so a chained traceback leaked a coordinate), the
  index-mismatch message no longer echoes the parsed index, and the
  "never quotes the line" tests now assert over the whole printed
  exception chain rather than `str(exc)`. New tests pin the finiteness
  guard (a `nan`/`inf` line) and BOM tolerance, both previously untested.
  79 tests pass; each new guard was mutation-checked (breaking it turns
  the tests red).
- 2026-09-08 — `load_mesh`, `RawSubject`, `CropConfig`/`crop_ear`, transform +
  inverse + sampling, `plot_ear.py`, `inspect_dataset.py`, synthetic tests.
- 2026-09-08 — Alfred inspected the real data; `DATA_SPEC.md` filled in
  (layout, IDs, PLY format, CSV shape). `.gitignore` now excludes the dataset
  folder/zip and `*.csv`; superseded bundle folders deleted.
- 2026-09-08 — `src/data.py` loaders done (`subject_id_from_path` = `P\d{4}`,
  `list_subjects`, `mesh_path`, `landmark_path`, `load_landmarks`,
  `parse_landmark_line`, `load_subject_landmarks`); `inspect_dataset.py` uses
  the real loader + token histogram; `scripts/crop_stats.py` + `load_crop_config`
  written; `plot_ear.py` gained `--subject` / `--crop`. 51 tests pass. All
  scripts smoke-tested on a synthetic root only.
- 2026-09-08 — senior-reviewer FIX-FIRST round applied: `load_landmarks` now
  requires one consistent token count (3 or 4) across all 85 lines and names the
  first line that disagrees (mixed-layout test added); `scripts/make_split.py` +
  `splits/` added (provisional 160/40 seed-42 split, Role C to replace/ratify);
  `crop_stats.py` gained a required `--subject-list`, split provenance
  (path/sha256/count) in `crop.yaml`, fail-fast on load errors unless
  `--allow-failures`, and a leave-one-out headroom check with an explicit freeze
  criterion replacing the by-construction envelope check; `test_geometry.py`
  checks sample_points membership by row-tuple sets and adds a two-blob subject
  (per-side crop boxes either side of Y=0) asserting each side inverts back into
  its own blob and that side/config mismatch raises; `DATA_SPEC.md` records that
  `crop.yaml` derives from `splits/train_ids.txt` only. 54 tests pass. Both
  scripts smoke-tested on a synthetic root only — no real data touched.
- 2026-09-08 — reviewer round 2 FIX-FIRST applied: `crop_stats.py` no longer
  writes `--out` when the freeze criterion fails (bounds go to
  `<out>.rejected.yaml` for inspection, exit 1), since `load_crop_config` does
  not check `freeze_criterion_passed`; leave-one-out now reports the room below
  and above each face so a FAIL names the tight face; new `tests/test_scripts.py`
  covers `make_split` (disjoint/covering/seed-stable/order-independent, bad
  input) and the crop-box helpers (margin rule, uniform set -> headroom ==
  margin, planted outlier -> negative headroom on exactly one axis and face,
  outlier inside the full envelope yet failing leave-one-out, worst-subject
  ordering and tight-face reporting). Side-mismatch assertions now `match="side"`.
  64 tests pass.
- 2026-09-08 - reviewer round 3 FIX-FIRST applied: `load_crop_config` now raises
  on `freeze_criterion_passed: false`, so rejected bounds cannot be loaded as if
  frozen even if the file is copied; a failed `crop_stats.py` run that leaves an
  earlier `configs/crop.yaml` in place now says so loudly and prints that file's
  split path, sha256, subject count and freeze verdict; the rejected file gets a
  "# REJECTED - DO NOT USE" header instead of the frozen one. New `main()` tests
  cover pass, fail (frozen path absent, rejected present, exit 1), the stale-file
  warning, `--dry-run` and a missing subject list. 69 tests pass; each new guard
  was mutation-checked (breaking it turns the tests red).


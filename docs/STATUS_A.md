# STATUS — Role A (Alfred)

Working plan and running status. Read at the start of every task; tick items off
when done. Milestone order is fixed: **loader → crop → transform/inverse →
sampling → cache**.

Created 2026-09-08. Deadline 2026-09-15.

## 1. Verify the data conventions (blocking most of the rest)
- [x] `scripts/inspect_dataset.py` written (aggregates only, never per-point dumps)
- [x] Run it on the real data and fill in `DATA_SPEC.md` (Alfred, 2026-09-08):
  - [x] units mm, global bbox X/Y/Z ranges, vertex-count range (0.28M–0.84M)
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
  - [ ] left/right landmark bboxes (section 6 of the same run)
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
- [ ] Run the loaders on real data once (`inspect_dataset.py` does this)

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
      Smoke-tested on a synthetic root only; **not yet run on real data.**
- [x] `scripts/make_split.py` — **provisional** subject-level split (Role C owns
      the real one): 160/40, `default_rng(42)`, IDs sorted before shuffling, both
      ears together by construction. Writes `splits/{train,val}_ids.txt` +
      `splits/README.md`. `splits/` is in the repo (IDs only, no data).
- [ ] Run `make_split.py` on real data → `splits/train_ids.txt` (200 subjects)
- [x] `load_crop_config(side, path="configs/crop.yaml") -> CropConfig`
- [ ] Run `crop_stats.py --subject-list splits/train_ids.txt` on real data →
      `configs/crop.yaml`; check the freeze criterion PASSES; copy lo/hi and the
      LOO headroom into `DATA_SPEC.md`; then **freeze** (inference: mesh + frozen
      config only)
- [ ] Validate across all subjects; check the `suspicious` flag catches truncation

## 5. Canonical transform + inverse — `src/geometry.py`
- [x] `EarTransform`, `make_transform`, `transform_points_to_canonical`,
      `inverse_transform_points`, `canonicalize_ear`
- [x] Round-trip tests in `tests/test_geometry.py`
- [ ] Confirm the mirror axis is Y on real data (visual check, §3)

## 6. Sampling
- [x] `sample_points(points, normals, n=2048, seed=0)` — deterministic
- [x] Normals: the PLYs carry none (verified), so no normal feature channel.
      `RawSubject.normals` will always be `None` on real data.

## 7. Processed cache — `scripts/preprocess.py`
- [ ] Placeholder only. Blocked on frozen crop bounds (§4). Writes
      `cache/<subject>_<side>.npz`; never committed.

## Open questions for Alfred
- PyYAML is used for `configs/crop.yaml` (already installed in the venv, 6.0.3);
  add it to the requirements file when one exists.
- The workspace is not a git repository yet, so `.gitignore` could not be
  verified with `git status`. Before `git init`, note that the dataset folder,
  its zip and `*.csv` are now ignored.

## Done log
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


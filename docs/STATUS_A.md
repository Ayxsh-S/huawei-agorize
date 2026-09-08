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
  - [ ] **CSV column layout**: `"x, y z"` or `"idx,x y z"`? — re-run
        `scripts/inspect_dataset.py` (section 5 now prints the per-line
        token-count histogram: all-3 → no index column, all-4 → index column)
  - [ ] left/right landmark Y signs and bboxes (section 6 of the same run)
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
- [x] `load_landmarks(path, side)` — `[85,3]` float64; accepts both candidate
      CSV layouts, refuses wrong line counts / malformed lines (names the line),
      refuses a file named for the other side. `parse_landmark_line` exposed.
- [x] `load_subject_landmarks(sid, root)` — `{"left": [85,3], "right": [85,3]}`
- [x] `tests/test_data.py` — synthetic PLYs + CSVs in `tmp_path`: both layouts,
      CRLF/LF, wrong line count, bad-line numbering, side mismatch, ID rule on
      mesh and landmark filenames, `list_subjects` ignores stray files
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
- [x] `scripts/crop_stats.py` — per side: landmark envelope over all subjects,
      median per-subject ear extent, margin = max(0.25·median extent, 15 mm);
      writes `configs/crop.yaml` (lo/hi/min_vertices + the stats used); prints
      summary table, #subjects outside the unexpanded envelope (must be 0) and
      min headroom to the expanded faces. `--subject-list` restricts to a
      training split. **Not yet run on real data.**
- [x] `load_crop_config(side, path="configs/crop.yaml") -> CropConfig`
- [ ] Run `crop_stats.py` on real data → `configs/crop.yaml`; copy lo/hi into
      `DATA_SPEC.md`; then **freeze** (inference: mesh + frozen config only)
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
- CSV layout (3 vs 4 tokens per line) — see §1; the loader handles both, but
  `DATA_SPEC.md` should record which one is real.
- PyYAML is used for `configs/crop.yaml` (already installed in the venv, 6.0.3);
  add it to the requirements file when one exists.
- The workspace is not a git repository yet, so `.gitignore` could not be
  verified with `git status`. Before `git init`, note that the dataset folder,
  its zip and `*.csv` are now ignored.

## Done log
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

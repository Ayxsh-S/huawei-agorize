# EXPERIMENTS

Use this file for every meaningful training/evaluation run.

---

## Baselines

### B0 — Global-frame mean template
- Date: 07/09/2026 
- Owner: Pravi  
- Split: seed=42, subject-level, train=160, val=40, file=configs/split_seed42.json
- Validation mean: 6.3932424711307645
- Median: 6.159525629613377
- P95: 11.46832211778311
- Notes: Built mean left/right templates from TRAIN subjects only; evaluated on VAL subjects only. Output artifacts: outputs/role_c/b0_metrics.json and outputs/role_c/global_template.npz. Left has heavier high-error tail (p95) than right - more extreme outliers.

### B0 — Global-frame mean template
- Date: 2026-09-08
- Owner: Pravi (Role C)
- Split: seed=42, subject-level, train=160, val=40 (configs/split_seed42.json)
- Validation mean: 6.3932424711307645
- Median: 6.159525629613377
- P95: 11.46832211778311
- Notes:
  - Side metrics:
    - left_mean: 6.348421204522823
    - right_mean: 6.438063737738706
    - left_median: 6.005782685040604
    - right_median: 6.267835125708471
    - left_p95: 13.40592973521226
    - right_p95: 10.81508953959883
  - Worst-case analysis saved: outputs/role_c/worst_b0_subjects.json
  - Top-5 worst left: P0190, P0058, P0041, P0102, P0223
  - Top-5 worst right: P0041, P0298, P0134, P0027, P0223
  - Visual check: apparent “distorted ear” cases are mostly viewpoint-dependent in 3D plots.
- Decision: KEEP (baseline + fallback), INVESTIGATE left-tail outliers.


### B1 — Canonical template
- Date:
- Owner:
- Transform version:
- Validation mean:
- Median:
- P95:
- Notes:

---

## Run Template

### Run XXX

- **Date:**
- **Owner:**
- **Git commit:**
- **Config:**
- **Model:**
- **Input features:**
- **Point count:**
- **Template version:**
- **Loss:**
- **Augmentation:**
- **Seed:**
- **Epochs:**
- **Checkpoint path/location:**

#### Validation
- Mean:
- Median:
- P95:
- Outer helix:
- Concha:
- Inner helix:
- Superior antihelix:

#### Notes
-

#### Decision
KEEP / DROP / INVESTIGATE

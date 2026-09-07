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
- Notes: Built mean left/right templates from TRAIN subjects only; evaluated on VAL subjects only. Output artifacts: outputs/role_c/b0_metrics.json and outputs/role_c/global_template.npz. Decision: KEEP (baseline + fallback).



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

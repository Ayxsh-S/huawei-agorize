# Huawei ear-landmark sprint — Role A (Alfred)
Goal: raw Huawei head PLY -> left/right CanonicalEar (2048 canonical points + exact invertible transform). Deadline 15 Sep 2026. Ship working first, optimise second.

## Ownership
- Alfred (Role A) owns: src/data.py, src/geometry.py, scripts/, tests/test_geometry.py, tests/test_data.py, DATA_SPEC.md.
- Do not edit src/model.py, src/losses.py, src/train.py, configs/model* (B), src/evaluate.py, src/postprocess.py, src/templates.py, src/visualise.py (C), src/infer.py, src/pipeline.py (D). Frozen signatures in src/geometry.py and src/data.py change only on Alfred's say-so.
- Never commit or push.

## Hard rules
- NEVER open, read, list, plot or print anything under the data root ($HUAWEI_DATA_ROOT) or any .ply/.npz/.npy file. Alfred runs code on real data and reports results back. Tests use synthetic data only.
- GT landmarks never define a transform, crop, centre, scale, rotation or mirror. Inference uses mesh + frozen training-derived config only.
- Do not redesign the architecture or add PointNet++/heatmaps/segmentation/FPS. Ask before adding any dependency.

## Conventions
- Windows, Python 3.11, venv at C:\Munich\.venv, run tests with `pytest tests -q`. Data root env var HUAWEI_DATA_ROOT (folders mesh/ and landmarks/).
- Verified data facts live in DATA_SPEC.md; if a field is still "?", ask Alfred rather than guessing.
- After each implementation task, invoke the senior-reviewer subagent and act on FIX-FIRST/BLOCK items before ticking STATUS_A.
- Current work plan and status: docs/STATUS_A.md — read it at the start of every task and tick items off when done. Deeper context on demand only: docs/ROLE_A_3D_PREPROCESSING_GEOMETRY.md (my full role), docs/TEAM_OVERVIEW.md (other roles/interfaces). Read those only when the task needs them.

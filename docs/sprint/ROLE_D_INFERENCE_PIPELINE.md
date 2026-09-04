# ROLE D — INFERENCE & PIPELINE ENGINEER

> Give this file to your AI after `MASTER_AI_CONTEXT.md` and `DATA_SPEC.md`.

Tell the AI:

> I own Role D. Inspect the current repository before editing. Follow this role in order. Do not redesign the architecture. Do not modify another owner's modules unless a shared interface genuinely requires it. Do not commit or push.

---

# Mission

Turn A, B and C's components into one continuously runnable product.

Your output is the thing Huawei actually needs:

```text
raw PLY
→ left (85,3)
→ right (85,3)
```

You are not a submission/admin role. You are the runtime/integration engineer.

---

# Primary ownership

Suggested:

```text
src/infer.py
src/pipeline.py
integration tests
runtime config/loading glue
README.md inference/reproduction sections
```

Coordinate `EXPERIMENTS.md` final run references.

Do not reimplement A/B/C internals.

---

# 1. Day-1 pipeline

Do not wait for A/B.

Build the product shell immediately.

Initial version can use C's/global mean baseline:

```python
def predict_subject(mesh_path, config):
    return {
        "left": global_mean_left.copy(),
        "right": global_mean_right.copy(),
    }
```

This proves:
- CLI
- output type
- side handling
- writer
- integration test
- runtime config path

A valid fallback exists from the start.

---

# 2. Final conceptual pipeline

Eventually:

```python
def predict_subject(mesh_path, checkpoint, config):
    raw = load_mesh(mesh_path)

    result = {}

    for side in ("left", "right"):
        ear = preprocess_ear(mesh_path, side, config)

        template = load_canonical_template(side, config)

        residual = model_predict(ear, checkpoint, config)
        pred_canonical = template + residual

        pred_canonical = apply_enabled_postprocessing(
            pred_canonical,
            ear,
            config,
        )

        pred_original = inverse_transform_points(
            pred_canonical,
            ear.transform,
        )

        result[side] = pred_original

    return result
```

Exact code must follow real repo interfaces.

---

# 3. Required output invariants

```python
assert result["left"].shape == (85,3)
assert result["right"].shape == (85,3)

assert np.isfinite(result["left"]).all()
assert np.isfinite(result["right"]).all()
```

Also ensure:
- no side swap
- correct landmark order
- original Huawei coordinate frame

---

# 4. Shared component integration

## A

Consume:
- preprocessing
- canonical ear
- inverse transform

Do not duplicate crop/geometry logic.

## B

Consume:
- model class
- checkpoint format
- model config

Do not duplicate architecture definitions.

## C

Consume:
- selected canonical template
- enabled/validated projection/ensemble settings
- chosen final run metrics

Do not independently invent post-processing.

---

# 5. Config system

Keep simple.

Final config should identify things like:
- crop config/version
- point count
- normals on/off
- canonical template path/version
- model architecture params
- checkpoint(s)
- surface projection on/off
- TTA/multi-sample count
- ensemble checkpoint list if enabled

Avoid building a giant configuration framework.

---

# 6. CLI

Aim for one obvious command, conceptually:

```bash
python -m src.infer   --input /path/to/subject.ply   --config configs/final.yaml   --output predictions/
```

Exact challenge output formatting must follow Huawei requirements.

---

# 7. Output writer

YOU own challenge-facing output formatting.

Verify:
- left/right arrays
- 85 landmarks each
- XYZ column order
- numeric types
- filenames/structure
- any official expected format

Do not assume; check challenge docs/example code.

---

# 8. Daily main smoke test

After merges, run one complete case.

Expected:

```text
raw PLY
→ current A
→ current B/fallback
→ current C settings
→ original coordinate output
```

Check shapes and finite values.

If `main` is broken, repair integration before new features.

---

# 9. Integration tests

Use stubs where needed initially.

Tests should catch:
- missing side
- wrong output shape
- NaN/Inf
- missing template
- incompatible checkpoint config
- failed inverse call
- accidental annotation requirement at inference

---

# 10. Experiment/checkpoint selection

You do not decide by intuition.

C provides validation numbers.
B provides checkpoint/config details.

Your job:
- make chosen run reproducible
- wire exact config/checkpoint/template into final runtime
- ensure no ambiguity

Avoid filenames like:

```text
best_final_REAL_v3.pt
```

without a run ID.

---

# 11. README

Before code freeze, document:
- environment setup
- data path configuration
- preprocessing if needed
- training command
- evaluation command
- inference command
- checkpoint/config required
- output format
- assumptions

Another teammate must follow it from scratch.

---

# 12. Final package QA

Verify:
- no raw NDA data included accidentally
- no local absolute paths
- no missing template/config
- no missing model code
- dependencies work
- archive opens
- output writer is correct
- final model/checkpoint available as required
- README reproduction works

---

# First work session

1. Read master context.
2. Inspect repo.
3. Create inference/pipeline skeleton.
4. Add a global-mean-template fallback interface.
5. Build CLI.
6. Build output object/writer.
7. Add one end-to-end smoke test using mocks/stubs.
8. Define expected integration boundaries with A/B/C.
9. Do not wait for their real implementations.

---

# Definition of done

Role D is done when a teammate who did not write the pipeline can follow the README and run one command on a raw Huawei mesh to obtain valid left/right `85 x 3` predictions using the chosen final components.

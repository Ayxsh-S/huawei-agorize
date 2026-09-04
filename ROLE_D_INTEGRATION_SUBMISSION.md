# ROLE D — INTEGRATION & SUBMISSION OWNER

> Paste this file **after** `01_MASTER_AI_CONTEXT.md` into your AI.  
> Tell it: **"I am Role D. Follow the master plan and work only on Integration & Submission unless an interface change is necessary."**

---

# Mission

Make the four people's work become one runnable product.

You are not responsible for inventing all modules yourself. You own:
- interfaces;
- end-to-end execution;
- reproducibility;
- daily `main` health;
- final output/submission.

Your success criterion is that another teammate can run one documented command on a raw Huawei mesh and obtain two valid `85 x 3` arrays.

---

# Files you primarily own

```text
src/infer.py
README.md
EXPERIMENTS.md
integration tests
top-level configs/glue
final packaging scripts if needed
```

Coordinate rather than overwrite A/B/C-owned modules.

---

# 1. Lock shared interfaces

Work with the existing repo to ensure:

Role A produces:
```text
CanonicalEar(points, normals, transform, side, subject_id)
```

Role B consumes canonical tensors and produces:
```text
residuals [B,85,3]
```

Role C consumes:
```text
prediction [B,85,3]
target     [B,85,3]
```

You assemble them.

---

# 2. End-to-end inference target

Implement conceptually:

```python
def predict_subject(mesh_path, checkpoint, config):
    raw = load_mesh(mesh_path)

    outputs = {}
    for side in ["left", "right"]:
        ear = canonicalize_ear(crop_ear(raw, side, config), side, config)

        template = get_canonical_template(side, config)

        residual = model_predict(ear.points, ear.normals)
        canonical_pred = template + residual

        if config.surface_projection:
            canonical_pred = maybe_project(...)

        pred_original = inverse_transform_points(
            canonical_pred,
            ear.transform,
        )

        outputs[side] = pred_original

    return outputs
```

Exact implementation depends on repo conventions.

---

# 3. Output invariants

Always check:

```python
assert result["left"].shape == (85, 3)
assert result["right"].shape == (85, 3)

assert np.isfinite(result["left"]).all()
assert np.isfinite(result["right"]).all()
```

Also verify:
- sides not swapped;
- correct landmark ordering;
- original Huawei coordinate frame.

---

# 4. Daily main smoke test

Once the first learned checkpoint exists, after merges run one end-to-end test.

Required outcome:

```text
raw PLY
→ both ears
→ model
→ inverse transform
→ two valid arrays
```

If `main` breaks, integration repair takes priority over new experiments.

Do not wait until submission day to discover merge incompatibilities.

---

# 5. Experiment registry

Maintain `EXPERIMENTS.md`.

Make sure each checkpoint has:
- run ID;
- commit;
- config;
- validation number;
- checkpoint location;
- decision.

Do not let the team have unnamed files like:
```text
best_final_v2_REAL.pt
```
without a logged run.

---

# 6. Baseline CLI

Before model completion, support Baseline 0/1 through the same general inference/evaluation framework if practical.

The product should always have a fallback.

---

# 7. README

By code-freeze day, README must state:

1. environment setup;
2. where permitted local dataset paths are configured;
3. preprocessing command if needed;
4. training command;
5. evaluation command;
6. inference command;
7. expected output;
8. checkpoint/config needed;
9. known assumptions.

Another teammate must test it from scratch.

---

# 8. Final package

Before upload, verify:
- no NDA raw data accidentally included;
- no absolute local Windows paths;
- no missing config/template file;
- no missing model definition;
- checkpoint path is documented;
- dependency versions are usable;
- prediction writer matches competition requirements;
- archive opens cleanly.

Check challenge portal requirements separately.

---

# 9. Shared template files

Make sure the final system can load the correct:
- global baseline templates if needed;
- canonical template(s);
- crop configuration;
- transform configuration.

These should be training-derived artefacts/configuration, not recomputed from hidden test annotations.

Store them in a safe non-confidential form only if allowed. If derived data is restricted by NDA, package according to challenge rules rather than public GitHub.

---

# 10. Role D is not "do everything"

If A's crop breaks, tell A with a reproducible failing case.
If B's checkpoint API breaks, tell B.
If C's metric is unclear, tell C.

Do not silently fork their module and create a second implementation.

Your job is integration, not replacing everyone.

---

# Today / first work session

1. Create/verify repo skeleton.
2. Add `.gitignore`.
3. Add master/role docs.
4. Establish common config strategy.
5. Create `src/infer.py` skeleton.
6. Create `EXPERIMENTS.md` template.
7. Make imports/tests runnable before real code lands.
8. Coordinate one agreed training machine.
9. Ensure everyone knows the shared interface contract.

---

# Your definition of done

Role D is done when a teammate who did not write the integration code can follow the README from a clean environment and run the final inference path successfully.

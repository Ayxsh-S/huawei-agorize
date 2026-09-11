from pathlib import Path

import numpy as np
import pytest

from src.geometry import EarTransform
from src.pipeline import (
    InferenceConfig,
    PipelineError,
    load_canonical_template,
    predict_subject,
    validate_result,
    write_predictions,
)


def _valid_template(offset: float = 0.0):
    x = np.zeros((85, 3), dtype=np.float64) + offset
    return {"left": x.copy(), "right": x.copy() + 1.0}


def test_validate_result_shape_and_finiteness():
    result = validate_result(_valid_template())
    assert result["left"].shape == (85, 3)
    assert result["right"].shape == (85, 3)


def test_validate_result_rejects_bad_shape():
    bad = _valid_template()
    bad["left"] = np.zeros((84, 3))
    with pytest.raises(PipelineError, match="shape"):
        validate_result(bad)


def test_validate_result_rejects_nonfinite():
    bad = _valid_template()
    bad["right"][0, 0] = np.nan
    with pytest.raises(PipelineError, match="NaN/Inf"):
        validate_result(bad)


def test_global_template_pipeline_does_not_require_annotations(tmp_path: Path):
    mesh = tmp_path / "P9999.ply"
    mesh.write_text("not actually parsed in fallback mode\n", encoding="utf-8")
    result = predict_subject(mesh, InferenceConfig(mode="global_template"), global_template=_valid_template())
    assert result["left"].shape == (85, 3)
    assert result["right"].shape == (85, 3)


def test_learned_path_requires_model_and_template(tmp_path: Path):
    mesh = tmp_path / "P9999.ply"
    mesh.write_text("placeholder\n", encoding="utf-8")
    with pytest.raises(PipelineError, match="model instance"):
        predict_subject(
            mesh,
            InferenceConfig(mode="learned"),
            model=None,
            canonical_template=np.zeros((85, 3)),
        )


def test_write_npz_and_json(tmp_path: Path):
    result = _valid_template()
    npz = write_predictions(result, tmp_path / "pred.npz", "npz")
    with np.load(npz, allow_pickle=False) as data:
        assert data["left"].shape == (85, 3)
        assert data["right"].shape == (85, 3)

    js = write_predictions(result, tmp_path / "pred.json", "json")
    assert js.read_text(encoding="utf-8").startswith("{")

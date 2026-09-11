"""End-to-end inference orchestration for Role D.

Role D owns orchestration only. Geometry, model architecture, template statistics,
and optional post-processing remain owned by Roles A/B/C respectively.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .data import load_mesh
from .geometry import N_LANDMARKS, canonicalize_ear, inverse_transform_points, load_crop_config

SIDES = ("left", "right")
OUTPUT_SHAPE = (N_LANDMARKS, 3)


class PipelineError(RuntimeError):
    """Raised when an inference dependency or invariant is invalid."""


@dataclass(frozen=True)
class InferenceConfig:
    """Small, explicit runtime config; no framework or registry required."""

    mode: str = "global_template"
    crop_config: str = "configs/crop.yaml"
    point_count: int = 2048
    seed: int = 0
    mirror_side: str = "right"
    mirror_axis: int = 1
    template_path: str | None = None
    checkpoint_path: str | None = None
    output_format: str = "npz"
    tta_samples: int = 1
    projection_enabled: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "InferenceConfig":
        model = data.get("model") if isinstance(data.get("model"), Mapping) else {}
        preprocessing = data.get("preprocessing") if isinstance(data.get("preprocessing"), Mapping) else {}
        output = data.get("output") if isinstance(data.get("output"), Mapping) else {}
        post = data.get("postprocessing") if isinstance(data.get("postprocessing"), Mapping) else {}

        mode = str(data.get("mode", "global_template"))
        template = data.get("template_path", data.get("template"))
        checkpoint = data.get("checkpoint_path", data.get("checkpoint"))
        return cls(
            mode=mode,
            crop_config=str(data.get("crop_config", preprocessing.get("crop_config", cls.crop_config))),
            point_count=int(data.get("point_count", preprocessing.get("point_count", cls.point_count))),
            seed=int(data.get("seed", preprocessing.get("seed", cls.seed))),
            mirror_side=str(data.get("mirror_side", preprocessing.get("mirror_side", cls.mirror_side))),
            mirror_axis=int(data.get("mirror_axis", preprocessing.get("mirror_axis", cls.mirror_axis))),
            template_path=str(template) if template is not None else None,
            checkpoint_path=str(checkpoint) if checkpoint is not None else None,
            output_format=str(data.get("output_format", output.get("format", cls.output_format))),
            tta_samples=int(data.get("tta_samples", post.get("tta_samples", 1))),
            projection_enabled=bool(data.get("projection_enabled", post.get("projection_enabled", False))),
        )


def _validate_prediction(pred: Any, side: str) -> np.ndarray:
    arr = np.asarray(pred, dtype=np.float64)
    if arr.shape != OUTPUT_SHAPE:
        raise PipelineError(f"{side} prediction must have shape {OUTPUT_SHAPE}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise PipelineError(f"{side} prediction contains NaN/Inf")
    return arr


def validate_result(result: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if set(result.keys()) != set(SIDES):
        raise PipelineError(f"prediction result must contain exactly {SIDES}, got {sorted(result.keys())}")
    return {side: _validate_prediction(result[side], side) for side in SIDES}


def load_global_template(path: str | Path) -> dict[str, np.ndarray]:
    """Load C's existing global-frame fallback template."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"global template not found: {p}")
    with np.load(p, allow_pickle=False) as data:
        missing = [side for side in SIDES if side not in data.files]
        if missing:
            raise PipelineError(f"global template {p} missing keys: {missing}")
        result = {side: _validate_prediction(data[side], side) for side in SIDES}
    return result


def load_canonical_template(path: str | Path) -> np.ndarray:
    """Load C's shared canonical template artifact."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"canonical template not found: {p}")
    with np.load(p, allow_pickle=False) as data:
        if "template" not in data.files:
            raise PipelineError(f"canonical template {p} must contain key 'template'")
        return _validate_prediction(data["template"], "template")


def _seed_for_ear(base_seed: int, subject_id: str, side: str) -> int:
    # Match Role A's deterministic per-ear seed policy without reading cache internals.
    import hashlib
    digest = hashlib.sha256(f"{base_seed}|{subject_id}|{side}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def _default_preprocess(mesh_path: str | Path, side: str, cfg: InferenceConfig):
    raw = load_mesh(mesh_path)
    crop = load_crop_config(side, cfg.crop_config)
    return canonicalize_ear(
        raw,
        side,
        crop,
        mirror_side=cfg.mirror_side,
        n_points=cfg.point_count,
        seed=_seed_for_ear(cfg.seed, raw.subject_id, side),
        mirror_axis=cfg.mirror_axis,
    )


def _call_model(model: Any, points: np.ndarray) -> np.ndarray:
    """Strictly call B's model as documented: [B,N,F] -> [B,85,3]."""
    # Lazy import is intentional: baseline inference needs no torch.
    try:
        import torch
    except ImportError as exc:
        raise PipelineError("learned inference requires PyTorch, but torch is not installed") from exc

    x = np.asarray(points)
    if x.ndim != 2 or x.shape[1] not in (3, 6):
        raise PipelineError(f"model input must be [N,3] or [N,6], got {x.shape}")
    tensor = torch.from_numpy(x.astype(np.float32, copy=False)).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        residual = model(tensor)
    if hasattr(residual, "detach"):
        residual = residual.detach().cpu().numpy()
    residual = np.asarray(residual, dtype=np.float64)
    if residual.shape != (1, N_LANDMARKS, 3):
        raise PipelineError(
            "B model contract violation: expected residual shape "
            f"(1,{N_LANDMARKS},3), got {residual.shape}"
        )
    return residual[0]


def predict_subject(
    mesh_path: str | Path,
    config: InferenceConfig,
    *,
    global_template: Mapping[str, np.ndarray] | None = None,
    preprocess_fn: Callable[[str | Path, str, InferenceConfig], Any] | None = None,
    model: Any | None = None,
    canonical_template: np.ndarray | None = None,
    postprocess_fn: Callable[[np.ndarray, Any, str], np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Predict both ears, returning original Huawei XYZ coordinates.

    ``global_template`` is the explicit emergency fallback. Learned mode requires
    both a model and a shared canonical template. No annotation path is accepted.
    """
    mesh_path = Path(mesh_path)
    if not mesh_path.is_file():
        raise FileNotFoundError(f"input mesh not found: {mesh_path}")
    if config.tta_samples < 1:
        raise PipelineError("tta_samples must be >= 1")
    if config.tta_samples != 1:
        raise PipelineError(
            "tta_samples > 1 is owned by Role C; pass a C-validated averaging wrapper "
            "to this pipeline before enabling it"
        )
    if config.projection_enabled and postprocess_fn is None:
        raise PipelineError("projection_enabled=True but no C-owned postprocess_fn was provided")

    mode = config.mode.lower()
    if mode in {"global_template", "baseline", "fallback"}:
        if global_template is None:
            raise PipelineError("global_template mode requires C's global template")
        return validate_result({side: np.asarray(global_template[side]) for side in SIDES})

    if mode not in {"learned", "pointnet", "residual"}:
        raise PipelineError(f"unknown inference mode: {config.mode!r}")
    if model is None:
        raise PipelineError("learned inference requires a B model instance")
    if canonical_template is None:
        raise PipelineError("learned inference requires C's canonical template")
    canonical_template = _validate_prediction(canonical_template, "canonical template")

    preprocess = preprocess_fn or _default_preprocess
    result: dict[str, np.ndarray] = {}
    for side in SIDES:
        ear = preprocess(mesh_path, side, config)
        residual = _call_model(model, ear.points)
        pred_can = canonical_template + residual
        if postprocess_fn is not None:
            pred_can = postprocess_fn(pred_can, ear, side)
        pred_global = inverse_transform_points(pred_can, ear.transform)
        result[side] = _validate_prediction(pred_global, side)

    return validate_result(result)


def write_predictions(result: Mapping[str, np.ndarray], output_path: str | Path, fmt: str = "npz") -> Path:
    """Write an internal, explicit prediction artifact.

    Huawei submission formatting is deliberately not hard-coded because DATA_SPEC
    still marks the official challenge format as unverified.
    """
    result = validate_result(result)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = fmt.lower()
    if fmt == "npz":
        np.savez(out, left=result["left"], right=result["right"])
    elif fmt == "json":
        import json
        out.write_text(json.dumps({side: result[side].tolist() for side in SIDES}), encoding="utf-8")
    else:
        raise ValueError(f"unsupported internal output format {fmt!r}; use 'npz' or 'json'")
    return out

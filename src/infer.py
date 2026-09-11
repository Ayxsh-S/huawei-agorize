"""Command-line entry point for Role D inference."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

import numpy as np

from .pipeline import (
    InferenceConfig,
    PipelineError,
    load_canonical_template,
    load_global_template,
    predict_subject,
    write_predictions,
)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required for --config") from exc
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"config not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"config must contain a YAML mapping: {p}")
    return data


def _load_model_from_entry(entry: dict[str, Any] | None):
    """Load B's model only when a config explicitly identifies the factory."""
    if not entry:
        raise PipelineError(
            "learned mode requires a 'model' mapping with module/class/checkpoint "
            "after B's checkpoint contract is frozen"
        )
    module_name = entry.get("module")
    class_name = entry.get("class")
    if not module_name or not class_name:
        raise PipelineError("learned model config requires model.module and model.class")
    module = importlib.import_module(str(module_name))
    cls = getattr(module, str(class_name))
    kwargs = entry.get("kwargs", {})
    if not isinstance(kwargs, dict):
        raise PipelineError("model.kwargs must be a mapping")
    model = cls(**kwargs)

    checkpoint = entry.get("checkpoint")
    if not checkpoint:
        raise PipelineError("learned model config requires model.checkpoint")
    try:
        import torch
    except ImportError as exc:
        raise PipelineError("learned inference requires PyTorch") from exc
    ckpt_path = Path(checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = raw.get("state_dict") if isinstance(raw, dict) else raw
    if not isinstance(state, dict):
        raise PipelineError("checkpoint must contain a state_dict mapping or be a raw state_dict")
    model.load_state_dict(state, strict=True)
    return model


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PinnaTwin-Zoom Role D inference")
    ap.add_argument("--input", required=True, help="unseen Huawei PLY")
    ap.add_argument("--config", required=True, help="runtime YAML config")
    ap.add_argument("--output", required=True, help="output file path")
    args = ap.parse_args(argv)

    raw_cfg = _load_yaml(args.config)
    cfg = InferenceConfig.from_mapping(raw_cfg)
    mode = cfg.mode.lower()

    if mode in {"global_template", "baseline", "fallback"}:
        template_path = cfg.template_path or "outputs/role_c/global_template.npz"
        result = predict_subject(args.input, cfg, global_template=load_global_template(template_path))
    else:
        canonical_path = cfg.template_path or "outputs/role_c/canonical_template_shared.npz"
        model_entry = raw_cfg.get("model") if isinstance(raw_cfg.get("model"), dict) else None
        model = _load_model_from_entry(model_entry)
        result = predict_subject(
            args.input,
            cfg,
            model=model,
            canonical_template=load_canonical_template(canonical_path),
        )

    out = write_predictions(result, args.output, cfg.output_format)
    print(f"Wrote {out}")
    print(f"left={result['left'].shape} right={result['right'].shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

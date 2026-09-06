from __future__ import annotations
from typing import Dict, Optional
import numpy as np


def landmark_errors(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")
    if pred.ndim != 3 or pred.shape[1:] != (85, 3):
        raise ValueError(f"Expected [B,85,3], got {pred.shape}")

    return np.linalg.norm(pred - target, axis=-1)  # [B,85]


def summarize_errors(err: np.ndarray, contour_indices: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, object]:
    err = np.asarray(err, dtype=np.float64)
    if err.ndim != 2 or err.shape[1] != 85:
        raise ValueError(f"Expected [B,85], got {err.shape}")

    out: Dict[str, object] = {
        "mean": float(err.mean()),
        "median": float(np.median(err)),
        "p95": float(np.percentile(err, 95)),
        "per_example_mean": err.mean(axis=1),
        "per_landmark_mean": err.mean(axis=0),
    }

    if contour_indices:
        per_contour = {}
        for name, idx in contour_indices.items():
            idx = np.asarray(idx, dtype=np.int64)
            per_contour[name] = float(err[:, idx].mean())
        out["per_contour_mean"] = per_contour

    return out


def top_k_worst(per_example_mean: np.ndarray, k: int = 20) -> np.ndarray:
    x = np.asarray(per_example_mean, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("per_example_mean must be 1D")
    k = min(k, len(x))
    return np.argsort(-x)[:k]
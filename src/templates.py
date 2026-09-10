from __future__ import annotations
from typing import Dict
import numpy as np
from .evaluate import landmark_errors, summarize_errors


def build_global_template(train_left: np.ndarray, train_right: np.ndarray) -> Dict[str, np.ndarray]:
    train_left = np.asarray(train_left, dtype=np.float64)
    train_right = np.asarray(train_right, dtype=np.float64)

    if train_left.ndim != 3 or train_left.shape[1:] != (85, 3):
        raise ValueError(f"train_left must be [N,85,3], got {train_left.shape}")
    if train_right.ndim != 3 or train_right.shape[1:] != (85, 3):
        raise ValueError(f"train_right must be [N,85,3], got {train_right.shape}")

    return {
        "left": train_left.mean(axis=0),
        "right": train_right.mean(axis=0),
    }


def predict_from_global_template(template: Dict[str, np.ndarray], n_subjects: int) -> Dict[str, np.ndarray]:
    left = np.repeat(template["left"][None, ...], n_subjects, axis=0)
    right = np.repeat(template["right"][None, ...], n_subjects, axis=0)
    return {"left": left, "right": right}


def evaluate_global_template(
    template: Dict[str, np.ndarray],
    val_left: np.ndarray,
    val_right: np.ndarray,
) -> Dict[str, object]:
    n = val_left.shape[0]
    pred = predict_from_global_template(template, n)

    err_left = landmark_errors(pred["left"], val_left)
    err_right = landmark_errors(pred["right"], val_right)

    # Combined across both ears:
    err_all = np.concatenate([err_left, err_right], axis=0)

    return {
        "left": summarize_errors(err_left),
        "right": summarize_errors(err_right),
        "all_ears": summarize_errors(err_all),
    }
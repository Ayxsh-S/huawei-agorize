from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


def plot_pred_vs_gt(pred: np.ndarray, gt: np.ndarray, title: str = "Pred vs GT"):
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    if pred.shape != (85, 3) or gt.shape != (85, 3):
        raise ValueError("Expected shape (85,3) for pred and gt")

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], s=10, label="GT")
    ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], s=10, label="Pred")
    ax.set_title(title)
    ax.legend()
    return fig
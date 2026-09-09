import numpy as np
from src.templates import build_global_template, evaluate_global_template


def test_global_template_shapes():
    train_left = np.zeros((10, 85, 3), dtype=np.float32)
    train_right = np.ones((10, 85, 3), dtype=np.float32)
    tpl = build_global_template(train_left, train_right)
    assert tpl["left"].shape == (85, 3)
    assert tpl["right"].shape == (85, 3)


def test_global_template_eval_runs():
    train_left = np.zeros((10, 85, 3), dtype=np.float32)
    train_right = np.zeros((10, 85, 3), dtype=np.float32)
    val_left = np.zeros((4, 85, 3), dtype=np.float32)
    val_right = np.zeros((4, 85, 3), dtype=np.float32)

    tpl = build_global_template(train_left, train_right)
    m = evaluate_global_template(tpl, val_left, val_right)
    assert "all_ears" in m
    assert m["all_ears"]["mean"] == 0.0
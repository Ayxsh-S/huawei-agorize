import numpy as np
from src.evaluate import landmark_errors, summarize_errors, top_k_worst


def test_landmark_errors_zero():
    p = np.zeros((2, 85, 3), dtype=np.float32)
    t = np.zeros((2, 85, 3), dtype=np.float32)
    e = landmark_errors(p, t)
    assert e.shape == (2, 85)
    assert np.allclose(e, 0.0)


def test_landmark_errors_unit_offset():
    p = np.zeros((1, 85, 3), dtype=np.float32)
    t = np.zeros((1, 85, 3), dtype=np.float32)
    p[..., 0] = 1.0
    e = landmark_errors(p, t)
    assert np.allclose(e, 1.0)


def test_summarize_and_topk():
    err = np.ones((3, 85), dtype=np.float32)
    s = summarize_errors(err)
    assert s["mean"] == 1.0
    assert s["median"] == 1.0
    assert s["p95"] == 1.0
    assert s["per_example_mean"].shape == (3,)
    assert s["per_landmark_mean"].shape == (85,)

    idx = top_k_worst(np.array([0.1, 2.0, 1.0]), k=2)
    assert idx.tolist() == [1, 2]
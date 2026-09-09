from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from src.cache import list_cached, load_cached_ear
from src.geometry import inverse_transform_points
from src.evaluate import landmark_errors, summarize_errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="cache", type=str)
    ap.add_argument("--split", default="configs/split_seed42.json", type=str)
    ap.add_argument("--template_npz", default="outputs/role_c/canonical_template_shared.npz", type=str)
    ap.add_argument("--out_json", default="outputs/role_c/b1_metrics.json", type=str)
    ap.add_argument("--topk", default=20, type=int)
    args = ap.parse_args()

    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    val_ids = set(split["val_subjects"])

    tpl = np.load(args.template_npz)
    template = np.asarray(tpl["template"], dtype=np.float64)
    if template.shape != (85, 3):
        raise ValueError(f"Template shape must be (85,3), got {template.shape}")

    err_all = []
    err_left = []
    err_right = []
    ear_keys = []
    ear_mean = []

    for sid, side, p in list_cached(args.cache_dir):
        if sid not in val_ids:
            continue

        ear = load_cached_ear(p)
        if "targets" not in ear.qa:
            raise ValueError(f"Missing qa['targets'] in {p}. Rebuild cache with --with-targets.")

        gt_can = np.asarray(ear.qa["targets"], dtype=np.float64)   # [85,3]
        pred_can = template                                        # [85,3]

        # Evaluate in global coordinates (official space)
        gt_global = inverse_transform_points(gt_can, ear.transform)
        pred_global = inverse_transform_points(pred_can, ear.transform)

        e = landmark_errors(pred_global[None, ...], gt_global[None, ...])[0]  # [85]
        err_all.append(e)
        ear_keys.append({"subject_id": sid, "side": side})
        ear_mean.append(float(e.mean()))

        if side == "left":
            err_left.append(e)
        else:
            err_right.append(e)

    if not err_all:
        raise RuntimeError("No validation ears found in cache for provided split.")

    err_all = np.stack(err_all, axis=0)
    err_left = np.stack(err_left, axis=0) if err_left else np.zeros((0, 85))
    err_right = np.stack(err_right, axis=0) if err_right else np.zeros((0, 85))

    m_all = summarize_errors(err_all)
    m_left = summarize_errors(err_left) if len(err_left) else None
    m_right = summarize_errors(err_right) if len(err_right) else None

    order = np.argsort(-np.asarray(ear_mean))
    k = min(args.topk, len(order))
    worst = [
        {
            "subject_id": ear_keys[i]["subject_id"],
            "side": ear_keys[i]["side"],
            "mean_error": ear_mean[i],
        }
        for i in order[:k]
    ]

    out = {
        "id": "B1",
        "system": "Shared canonical mean template",
        "n_val_ears": int(err_all.shape[0]),
        "all_ears_mean": float(m_all["mean"]),
        "all_ears_median": float(m_all["median"]),
        "all_ears_p95": float(m_all["p95"]),
        "left_mean": float(m_left["mean"]) if m_left else None,
        "right_mean": float(m_right["mean"]) if m_right else None,
        "left_p95": float(m_left["p95"]) if m_left else None,
        "right_p95": float(m_right["p95"]) if m_right else None,
        "worst_ears_topk": worst,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Saved B1 metrics: {out_path}")
    print(json.dumps({
        "B1_shared_canonical_template": {
            "mean": out["all_ears_mean"],
            "median": out["all_ears_median"],
            "p95": out["all_ears_p95"],
        }
    }, indent=2))


if __name__ == "__main__":
    main()
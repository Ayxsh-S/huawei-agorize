from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import numpy as np

from src.split import load_split
from src.templates import build_global_template, evaluate_global_template

FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_landmark_file(path: Path) -> np.ndarray:
    pts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        nums = [float(x) for x in FLOAT_RE.findall(line)]
        if len(nums) >= 3:
            pts.append(nums[-3:])  # handles "idx [x y z]" format
    arr = np.asarray(pts, dtype=np.float64)
    if arr.shape != (85, 3):
        raise ValueError(f"{path.name} parsed shape {arr.shape}, expected (85,3)")
    return arr


def load_side_arrays(data_root: Path, subject_ids: list[str], side: str) -> np.ndarray:
    lm_dir = data_root / "landmarks"
    suffix = f"_{side}_ear_landmarks.csv"
    arrays = []
    for sid in subject_ids:
        p = lm_dir / f"{sid}{suffix}"
        if not p.exists():
            raise FileNotFoundError(f"Missing {side} landmarks for {sid}: {p}")
        arrays.append(parse_landmark_file(p))
    return np.stack(arrays, axis=0)  # [N,85,3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, type=str)
    ap.add_argument("--split", default="configs/split_seed42.json", type=str)
    ap.add_argument("--out_json", default="outputs/role_c/b0_metrics.json", type=str)
    ap.add_argument("--out_template_npz", default="outputs/role_c/global_template.npz", type=str)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    split = load_split(args.split)

    train_ids = list(split["train_subjects"])
    val_ids = list(split["val_subjects"])

    train_left = load_side_arrays(data_root, train_ids, "left")
    train_right = load_side_arrays(data_root, train_ids, "right")
    val_left = load_side_arrays(data_root, val_ids, "left")
    val_right = load_side_arrays(data_root, val_ids, "right")

    template = build_global_template(train_left, train_right)
    metrics = evaluate_global_template(template, val_left, val_right)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "id": "B0",
        "system": "Global mean template",
        "n_train_subjects": len(train_ids),
        "n_val_subjects": len(val_ids),
        "all_ears_mean": metrics["all_ears"]["mean"],
        "all_ears_median": metrics["all_ears"]["median"],
        "all_ears_p95": metrics["all_ears"]["p95"],
    }, indent=2), encoding="utf-8")

    out_tpl = Path(args.out_template_npz)
    out_tpl.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_tpl, left=template["left"], right=template["right"])

    print(f"Saved metrics: {out_json}")
    print(f"Saved template: {out_tpl}")
    print(json.dumps({
        "B0_global_mean": {
            "mean": metrics["all_ears"]["mean"],
            "median": metrics["all_ears"]["median"],
            "p95": metrics["all_ears"]["p95"],
            "left_mean": metrics["left"]["mean"],
            "right_mean": metrics["right"]["mean"],
            "left_median": metrics["left"]["median"],
            "right_median": metrics["right"]["median"],
            "left_p95": metrics["left"]["p95"],
            "right_p95": metrics["right"]["p95"],
        }
    }, indent=2))


if __name__ == "__main__":
    main()
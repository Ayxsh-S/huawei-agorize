from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import numpy as np

from src.split import load_split
from src.evaluate import landmark_errors

FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_landmark_file(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        nums = [float(x) for x in FLOAT_RE.findall(line)]
        if len(nums) >= 4:
            idx = int(round(nums[0]))
            xyz = nums[1:4]
        elif len(nums) == 3:
            idx = len(rows)
            xyz = nums
        else:
            continue
        rows.append((idx, xyz))

    if len(rows) != 85:
        raise ValueError(f"{path.name}: expected 85 rows, got {len(rows)}")

    idxs = [i for i, _ in rows]
    if sorted(idxs) != list(range(85)):
        raise ValueError(f"{path.name}: index set not 0..84")

    rows.sort(key=lambda t: t[0])
    return np.asarray([xyz for _, xyz in rows], dtype=np.float64)


def load_side(data_root: Path, subject_ids: list[str], side: str) -> np.ndarray:
    lm_dir = data_root / "landmarks"
    arrs = []
    for sid in subject_ids:
        p = lm_dir / f"{sid}_{side}_ear_landmarks.csv"
        arrs.append(parse_landmark_file(p))
    return np.stack(arrs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, type=str)
    ap.add_argument("--split", default="configs/split_seed42.json")
    ap.add_argument("--template_npz", default="outputs/role_c/global_template.npz")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--out_json", default="outputs/role_c/worst_b0_subjects.json")
    args = ap.parse_args()

    split = load_split(args.split)
    val_ids = list(split["val_subjects"])

    tpl = np.load(args.template_npz)
    t_left = tpl["left"][None, ...]
    t_right = tpl["right"][None, ...]

    val_left = load_side(Path(args.data_root), val_ids, "left")
    val_right = load_side(Path(args.data_root), val_ids, "right")

    pred_left = np.repeat(t_left, len(val_ids), axis=0)
    pred_right = np.repeat(t_right, len(val_ids), axis=0)

    e_left = landmark_errors(pred_left, val_left).mean(axis=1)   # per-subject left mean
    e_right = landmark_errors(pred_right, val_right).mean(axis=1)

    k = min(args.k, len(val_ids))
    idx_l = np.argsort(-e_left)[:k]
    idx_r = np.argsort(-e_right)[:k]

    out = {
        "worst_left": [{"subject_id": val_ids[i], "mean_error": float(e_left[i])} for i in idx_l],
        "worst_right": [{"subject_id": val_ids[i], "mean_error": float(e_right[i])} for i in idx_r],
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")
    print("Top 5 worst left:", out["worst_left"][:5])
    print("Top 5 worst right:", out["worst_right"][:5])


if __name__ == "__main__":
    main()
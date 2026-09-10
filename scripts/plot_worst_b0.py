from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np

from src.visualise import plot_pred_vs_gt
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

    rows.sort(key=lambda t: t[0])
    arr = np.asarray([xyz for _, xyz in rows], dtype=np.float64)
    if arr.shape != (85, 3):
        raise ValueError(f"{path.name}: expected (85,3), got {arr.shape}")
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--worst_json", default="outputs/role_c/worst_b0_subjects.json")
    ap.add_argument("--template_npz", default="outputs/role_c/global_template.npz")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out_dir", default="outputs/role_c/plots_worst_b0")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    lm_dir = data_root / "landmarks"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    worst = json.loads(Path(args.worst_json).read_text(encoding="utf-8"))
    tpl = np.load(args.template_npz)
    t_left, t_right = tpl["left"], tpl["right"]

    for side, key, pred in [("left", "worst_left", t_left), ("right", "worst_right", t_right)]:
        for item in worst[key][:args.k]:
            sid = item["subject_id"]
            gt = parse_landmark_file(lm_dir / f"{sid}_{side}_ear_landmarks.csv")
            mean_err = float(landmark_errors(pred[None], gt[None]).mean())
            fig = plot_pred_vs_gt(pred, gt, title=f"B0 {sid} {side} | mean={mean_err:.3f}")
            fig.savefig(out_dir / f"{sid}_{side}_b0.png", dpi=150, bbox_inches="tight")

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
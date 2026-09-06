from __future__ import annotations
import argparse
import json
import numpy as np
from src.split import load_split
from src.templates import build_global_template, evaluate_global_template


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, type=str, help="Prepared dataset file")
    parser.add_argument("--split", required=True, type=str, help="configs/split_seed42.json")
    parser.add_argument("--subjects_key", default="subject_ids", type=str)
    parser.add_argument("--left_key", default="left_landmarks", type=str)
    parser.add_argument("--right_key", default="right_landmarks", type=str)
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    subject_ids = data[args.subjects_key].astype(str)
    left = data[args.left_key]   # [S,85,3]
    right = data[args.right_key] # [S,85,3]

    split = load_split(args.split)
    train_set = set(split["train_subjects"])
    val_set = set(split["val_subjects"])

    train_mask = np.array([sid in train_set for sid in subject_ids], dtype=bool)
    val_mask = np.array([sid in val_set for sid in subject_ids], dtype=bool)

    train_left, train_right = left[train_mask], right[train_mask]
    val_left, val_right = left[val_mask], right[val_mask]

    template = build_global_template(train_left, train_right)
    metrics = evaluate_global_template(template, val_left, val_right)

    print(json.dumps({
        "B0_global_template": {
            "all_ears_mean": metrics["all_ears"]["mean"],
            "all_ears_median": metrics["all_ears"]["median"],
            "all_ears_p95": metrics["all_ears"]["p95"],
        }
    }, indent=2))


if __name__ == "__main__":
    main()
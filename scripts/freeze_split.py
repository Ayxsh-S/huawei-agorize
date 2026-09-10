from __future__ import annotations
import argparse
from src.split import load_subject_ids_txt, freeze_subject_split, save_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects_file", required=True, type=str, help="txt file, one subject id per line")
    parser.add_argument("--train_size", default=160, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--out", default="configs/split_seed42.json", type=str)
    args = parser.parse_args()

    subject_ids = load_subject_ids_txt(args.subjects_file)
    split = freeze_subject_split(subject_ids, train_size=args.train_size, seed=args.seed)
    save_split(split, args.out)

    print(f"Saved split: {args.out}")
    print(f"Train: {len(split['train_subjects'])}, Val: {len(split['val_subjects'])}")


if __name__ == "__main__":
    main()
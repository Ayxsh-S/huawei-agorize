from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from src.cache import list_cached, load_cached_ear


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="cache", type=str)
    ap.add_argument("--split", default="configs/split_seed42.json", type=str)
    ap.add_argument("--out", default="outputs/role_c/canonical_template_shared.npz", type=str)
    args = ap.parse_args()

    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    train_ids = set(split["train_subjects"])

    targets = []
    n_left, n_right = 0, 0

    for sid, side, p in list_cached(args.cache_dir):
        if sid not in train_ids:
            continue
        ear = load_cached_ear(p)
        if "targets" not in ear.qa:
            raise ValueError(f"Missing qa['targets'] in {p}. Rebuild cache with --with-targets.")
        t = np.asarray(ear.qa["targets"], dtype=np.float64)
        if t.shape != (85, 3):
            raise ValueError(f"Bad target shape in {p}: {t.shape}")
        targets.append(t)
        if side == "left":
            n_left += 1
        elif side == "right":
            n_right += 1

    if not targets:
        raise RuntimeError("No training ears found in cache for provided split.")

    arr = np.stack(targets, axis=0)  # [N_ears,85,3]
    template = arr.mean(axis=0)      # [85,3]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        template=template.astype(np.float32),
        n_train_ears=np.int64(arr.shape[0]),
        n_train_left=np.int64(n_left),
        n_train_right=np.int64(n_right),
    )

    print(f"Saved shared canonical template: {out}")
    print(f"Train ears used: {arr.shape[0]} (left={n_left}, right={n_right})")


if __name__ == "__main__":
    main()
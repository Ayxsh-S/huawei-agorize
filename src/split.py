from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict
import numpy as np


def freeze_subject_split(subject_ids: List[str], train_size: int = 160, seed: int = 42) -> Dict[str, object]:
    ids = [str(x) for x in subject_ids]
    unique = sorted(set(ids))
    if len(unique) != len(ids):
        raise ValueError("Duplicate subject IDs found.")
    if not (1 <= train_size < len(unique)):
        raise ValueError(f"train_size must be in [1, {len(unique)-1}]")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique)

    train_subjects = perm[:train_size].tolist()
    val_subjects = perm[train_size:].tolist()

    return {
        "seed": int(seed),
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
    }


def validate_split(split: Dict[str, object]) -> None:
    train = set(split["train_subjects"])
    val = set(split["val_subjects"])
    overlap = train.intersection(val)
    if overlap:
        raise ValueError(f"Train/val overlap found: {sorted(overlap)[:5]}")


def save_split(split: Dict[str, object], out_path: str | Path) -> None:
    validate_split(split)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, indent=2), encoding="utf-8")


def load_split(path: str | Path) -> Dict[str, object]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_split(data)
    return data


def load_subject_ids_txt(path: str | Path) -> List[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [x.strip() for x in lines if x.strip()]
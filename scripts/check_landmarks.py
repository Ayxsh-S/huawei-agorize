from pathlib import Path
import re
import argparse
import numpy as np

FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_landmark_file(path: Path) -> np.ndarray:
    pts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        nums = [float(x) for x in FLOAT_RE.findall(line)]
        if len(nums) >= 3:
            pts.append(nums[-3:])  # supports lines with index + xyz
    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Bad parsed shape for {path.name}: {arr.shape}")
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, type=str)
    args = parser.parse_args()

    root = Path(args.data_root)
    mesh_dir = root / "mesh"
    lm_dir = root / "landmarks"

    mesh_ids = {p.stem for p in mesh_dir.glob("*.ply")}
    left = {p.name.replace("_left_ear_landmarks.csv", ""): p for p in lm_dir.glob("*_left_ear_landmarks.csv")}
    right = {p.name.replace("_right_ear_landmarks.csv", ""): p for p in lm_dir.glob("*_right_ear_landmarks.csv")}

    both = set(left).intersection(right)
    usable = sorted(mesh_ids.intersection(both))

    bad_shape = []
    bad_finite = []
    mins, maxs = [], []

    for sid in usable:
        l = parse_landmark_file(left[sid])
        r = parse_landmark_file(right[sid])

        if l.shape != (85, 3):
            bad_shape.append((sid, "left", l.shape))
        if r.shape != (85, 3):
            bad_shape.append((sid, "right", r.shape))

        if not np.isfinite(l).all():
            bad_finite.append((sid, "left"))
        if not np.isfinite(r).all():
            bad_finite.append((sid, "right"))

        mins.append(np.minimum(l.min(axis=0), r.min(axis=0)))
        maxs.append(np.maximum(l.max(axis=0), r.max(axis=0)))

    mins = np.stack(mins).min(axis=0) if mins else np.array([np.nan, np.nan, np.nan])
    maxs = np.stack(maxs).max(axis=0) if maxs else np.array([np.nan, np.nan, np.nan])

    print(f"mesh count: {len(mesh_ids)}")
    print(f"left lm count: {len(left)}")
    print(f"right lm count: {len(right)}")
    print(f"usable subjects (mesh+left+right): {len(usable)}")
    print(f"bad shape entries: {len(bad_shape)}")
    if bad_shape[:10]:
        print("first bad shapes:", bad_shape[:10])
    print(f"non-finite entries: {len(bad_finite)}")
    print(f"global xyz min: {mins}")
    print(f"global xyz max: {maxs}")


if __name__ == "__main__":
    main()
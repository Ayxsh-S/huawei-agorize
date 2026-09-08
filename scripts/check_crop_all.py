"""Validate the FROZEN crop box on subjects it was never derived from (Role A).

Run by Alfred against the NDA-protected data, normally on the held-out split::

    python scripts/check_crop_all.py --subject-list splits/val_ids.txt

``configs/crop.yaml`` is built from the TRAINING landmarks only, and the
leave-one-out headroom printed by ``scripts/crop_stats.py`` is an *estimate* of
how an unseen subject would fare. This script measures the real thing: for
every listed subject and side it runs ``crop_ear`` with the frozen config and
reports

* the cropped vertex count (min / median / max) — is there enough surface to
  sample 2048 points from, and is any crop suspiciously small,
* how many crops trip ``CropConfig.min_vertices`` (the ``suspicious`` QA flag),
* how many ears lose a GT landmark, i.e. fewer than 85/85 fall inside the box.

A landmark outside the box is the failure that matters: the model can only ever
predict inside the crop, so a truncated ear caps the achievable error for that
subject no matter how good the model is.

**Exit code 1 if any ear has fewer than 85 landmarks inside, or any crop is
flagged suspicious.** Offending subject IDs are listed so they can be plotted
with ``scripts/plot_ear.py --subject <id> --crop configs/crop.yaml``.

Prints aggregates and subject IDs only — never a landmark coordinate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent
sys.path.insert(0, str(REPO_ROOT))  # for `src`
sys.path.insert(0, str(_HERE))         # sibling scripts, for shared helpers

from crop_stats import describe_crop_config, read_subject_list, sha256_file  # noqa: E402
from src.data import (  # noqa: E402
    DATA_ROOT,
    SIDES,
    RawSubject,
    list_subjects,
    load_mesh,
    load_subject_landmarks,
    mesh_path,
)
from src.geometry import (  # noqa: E402
    DEFAULT_CROP_CONFIG,
    N_LANDMARKS,
    N_POINTS,
    CropConfig,
    crop_ear,
    load_crop_config,
)

MAX_IDS_LISTED = 20
PROGRESS_EVERY = 10


def config_split_sha(path: Path) -> str | None:
    """sha256 of the subject list ``configs/crop.yaml`` was built from, if recorded.

    Used to tell Alfred when he is "validating" the frozen box on the very
    subjects that defined it, where a PASS is true by construction.
    """
    import yaml

    if not path.is_file() and not path.is_absolute() and (REPO_ROOT / path).is_file():
        path = REPO_ROOT / path
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        return (doc.get("split") or {}).get("sha256")
    except Exception:
        return None


def count_landmarks_inside(
    landmarks: np.ndarray, side: str, cfg: CropConfig, subject_id: str
) -> int:
    """How many of the 85 GT landmarks the frozen crop box keeps.

    Deliberately routed through ``crop_ear`` itself, on a throwaway
    ``RawSubject`` whose "vertices" are the landmarks, rather than re-writing
    the inclusive bounds test here: a hand-rolled copy could drift from the real
    crop (``>=`` vs ``>``, per-axis vs all-axes) and then report that everything
    is fine while the pipeline quietly truncates ears.
    """
    fake = RawSubject(
        subject_id=subject_id,
        vertices=np.asarray(landmarks, dtype=np.float64),
        faces=None,
        normals=None,
        mesh_path=Path("<landmarks>"),
    )
    inside, _, _ = crop_ear(fake, side, cfg)
    return int(inside.shape[0])


class SideAccumulator:
    """Running aggregates for one side — subject IDs and counts, no coordinates."""

    def __init__(self) -> None:
        self.n_ears = 0
        self.crop_sizes: list[int] = []
        self.suspicious_ids: list[str] = []
        self.truncated_ids: list[str] = []      # < 85 landmarks inside
        self.thin_ids: list[str] = []           # crop smaller than N_POINTS
        self.min_inside = N_LANDMARKS

    def add(self, subject_id: str, n_vertices: int, suspicious: bool, n_inside: int) -> None:
        self.n_ears += 1
        self.crop_sizes.append(int(n_vertices))
        if suspicious:
            self.suspicious_ids.append(subject_id)
        if n_inside < N_LANDMARKS:
            self.truncated_ids.append(subject_id)
        if n_vertices < N_POINTS:
            self.thin_ids.append(subject_id)
        self.min_inside = min(self.min_inside, n_inside)

    @property
    def ok(self) -> bool:
        return not self.suspicious_ids and not self.truncated_ids


def _id_list(ids: list[str]) -> str:
    shown = ", ".join(ids[:MAX_IDS_LISTED])
    more = "" if len(ids) <= MAX_IDS_LISTED else f" (+{len(ids) - MAX_IDS_LISTED} more)"
    return f"{shown}{more}"


def print_side_report(side: str, acc: SideAccumulator, cfg: CropConfig) -> None:
    print(f"\n--- {side} ({acc.n_ears} ears) ---")
    if acc.n_ears == 0:
        print("  no ears processed")
        return

    sizes = np.asarray(acc.crop_sizes, dtype=np.float64)
    print(f"  cropped vertices:  min {sizes.min():.0f}   "
          f"median {np.median(sizes):.0f}   max {sizes.max():.0f}")
    print(f"  suspicious crops (< {cfg.min_vertices} vertices): "
          f"{len(acc.suspicious_ids)}/{acc.n_ears}")
    if acc.suspicious_ids:
        print(f"    {_id_list(acc.suspicious_ids)}")
    print(f"  ears with < {N_LANDMARKS}/{N_LANDMARKS} GT landmarks inside the box: "
          f"{len(acc.truncated_ids)}/{acc.n_ears}   "
          f"(worst ear kept {acc.min_inside}/{N_LANDMARKS})")
    if acc.truncated_ids:
        print(f"    {_id_list(acc.truncated_ids)}")
    print(f"  crops smaller than N_POINTS={N_POINTS} (would sample WITH replacement): "
          f"{len(acc.thin_ids)}/{acc.n_ears}")
    if acc.thin_ids:
        print(f"    {_id_list(acc.thin_ids)}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=DATA_ROOT,
                    help="dataset root (default: $HUAWEI_DATA_ROOT)")
    ap.add_argument("--subject-list", type=Path, required=True,
                    help="REQUIRED text file of subject IDs, one per line. Use the "
                         "HELD-OUT split (splits/val_ids.txt): the point is to test "
                         "the frozen box on subjects it never saw.")
    ap.add_argument("--limit", type=int, default=0,
                    help="check only the first N listed subjects (0 = all)")
    ap.add_argument("--crop-config", type=Path, default=Path(DEFAULT_CROP_CONFIG),
                    help="frozen crop config (default: configs/crop.yaml)")
    ap.add_argument("--allow-failures", action="store_true",
                    help="still report subjects that are missing or fail to load, but "
                         "let the verdict stand on the ears that were checked; without "
                         "it any unchecked ear fails the run")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).expanduser()
    split_path = Path(args.subject_list).expanduser()

    if not split_path.is_file():
        print(f"!! subject list not found: {split_path}")
        return 1
    try:
        wanted_ids = read_subject_list(split_path)
    except ValueError as exc:
        print(f"!! {exc}")
        return 1

    try:
        configs = {side: load_crop_config(side, args.crop_config) for side in SIDES}
    except (FileNotFoundError, ValueError) as exc:
        print(f"!! could not load the crop config: {exc}")
        return 1

    # --limit truncates the list *before* the mesh lookup, so a subject this run
    # was never going to reach is not reported as missing.
    selected = wanted_ids[: args.limit] if args.limit > 0 else wanted_ids
    available = set(list_subjects(root))
    missing = [sid for sid in selected if sid not in available]
    subjects = [sid for sid in selected if sid in available]

    print(f"root: {root}")
    print(f"subject list: {split_path}  ({len(wanted_ids)} IDs)")
    print(f"crop config: {args.crop_config}")
    print(f"  {describe_crop_config(Path(args.crop_config))}")
    split_sha = sha256_file(split_path)
    print(f"  this run's subject list sha256 {split_sha[:16]}...")
    same_split_as_config = split_sha == config_split_sha(Path(args.crop_config))
    if same_split_as_config:
        print("  !! this is the SAME subject list the crop box was derived from - a "
              "PASS here says nothing about unseen subjects. Use splits/val_ids.txt.")
    for side in SIDES:
        cfg = configs[side]
        print(f"  {side:<6} lo {np.array2string(cfg.lo, precision=2)}  "
              f"hi {np.array2string(cfg.hi, precision=2)}  "
              f"min_vertices {cfg.min_vertices}")
    print(f"subjects to check: {len(subjects)}"
          + (f"  (--limit {args.limit})" if args.limit > 0 else ""))
    if missing:
        print(f"  !! {len(missing)} listed subject(s) have no mesh under {root}: "
              f"{missing[:10]}")
    if not subjects:
        print("!! no subjects found")
        return 1

    acc = {side: SideAccumulator() for side in SIDES}
    # An ear that never gets cropped is a failure, not a footnote: a wrong --root
    # or a stale subject list could otherwise check 3 ears out of 80 and still
    # print PASS.
    failures: list[str] = [f"{sid}: no mesh under {root}" for sid in missing]

    for i, sid in enumerate(subjects, 1):
        try:
            raw = load_mesh(mesh_path(sid, root))
            landmarks = load_subject_landmarks(sid, root)
        except Exception as exc:
            failures.append(f"{sid}: load failed: {type(exc).__name__}: {exc}")
            continue
        for side in SIDES:
            try:
                _, _, qa = crop_ear(raw, side, configs[side])
                n_inside = count_landmarks_inside(landmarks[side], side, configs[side], sid)
            except Exception as exc:
                failures.append(f"{sid} {side}: {type(exc).__name__}: {exc}")
                continue
            acc[side].add(sid, qa["n_vertices"], bool(qa["suspicious"]), n_inside)
        if i % PROGRESS_EVERY == 0:
            print(f"  ... {i}/{len(subjects)}", flush=True)

    for side in SIDES:
        print_side_report(side, acc[side], configs[side])

    if failures:
        print(f"\n!! {len(failures)} subject(s)/ear(s) could not be checked:")
        for f in failures[:20]:
            print(f"   {f}")
        if len(failures) > 20:
            print(f"   ... and {len(failures) - 20} more")

    checked = sum(a.n_ears for a in acc.values())
    n_truncated = sum(len(a.truncated_ids) for a in acc.values())
    n_suspicious = sum(len(a.suspicious_ids) for a in acc.values())

    print("\n=== FROZEN CROP ON HELD-OUT SUBJECTS ===")
    print(f"  ears checked: {checked}")
    print(f"  ears losing a GT landmark: {n_truncated}")
    print(f"  suspicious crops: {n_suspicious}")
    if args.limit > 0:
        print(f"  NOTE: --limit {args.limit} - this covered only the first "
              f"{len(subjects)} listed subject(s), not the whole list.")
    if same_split_as_config:
        # Repeated here so a verdict read off the tail of a long log cannot look
        # like evidence of generalisation when it is true by construction.
        print("  NOTE: this subject list IS the split configs/crop.yaml was built "
              "from, so the box holding here is true by construction. Re-run on "
              "splits/val_ids.txt for the real test.")

    unchecked_fatal = bool(failures) and not args.allow_failures
    ok = (checked > 0 and n_truncated == 0 and n_suspicious == 0 and not unchecked_fatal)
    if checked == 0:
        print("  => FAIL - nothing was checked.")
    elif ok:
        suffix = (f" ({len(failures)} unchecked, ignored via --allow-failures)"
                  if failures else "")
        print("  => PASS - the frozen box keeps all 85 landmarks on every ear it was "
              f"run over and no crop is suspicious{suffix}.")
    else:
        if n_truncated:
            print("  => FAIL - the frozen box truncates at least one ear: those "
                  "subjects' landmarks can never be predicted. Widen the margin "
                  "(scripts/crop_stats.py --min-margin) and re-freeze.")
        if n_suspicious:
            print("  => FAIL - at least one crop is suspiciously small; plot it "
                  "before trusting the mesh or the box.")
        if unchecked_fatal:
            print("  => FAIL - some subjects/ears could not be checked at all (see "
                  "above). Fix --root or the subject list, or pass --allow-failures.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

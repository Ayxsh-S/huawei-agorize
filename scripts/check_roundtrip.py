"""Verify the canonical transform is EXACTLY invertible on real data (Role A).

Run by Alfred against the NDA-protected data. For every subject and side it

1. loads the mesh and the GT landmarks,
2. builds the ear transform **from the mesh only** — ``crop_ear`` with the
   frozen ``configs/crop.yaml``, then ``make_transform`` on the cropped
   vertices (GT never touches the transform, see the no-leakage rule in
   ``CLAUDE.md``; the landmarks are only pushed *through* it),
3. maps the GT landmarks into canonical space and straight back, and
4. records the largest absolute coordinate error over all 85 landmarks.

The whole point of the canonical frame is that a predicted landmark can be
mapped back to Huawei's original millimetres with no loss, so anything above
``--tol`` (1e-9 mm) means the round-trip is not exact and every submitted
coordinate is already wrong by that much before the model has made a single
error. **Exit code 1 if the tolerance is exceeded.**

It also reports, per side, the canonical-coordinate envelope of the GT
landmarks and how many ears put any landmark outside ``[-1.5, 1.5]``. That is
for Role C: it says whether the canonical frame is well scaled for a mean-shape
template (landmarks clustered around the origin at ~±1, no side sitting off in
a corner). It is informational and never fails this check.

Prints aggregates only — envelopes, counts and per-ear maxima reduced over
subjects. No individual landmark coordinate is ever printed.

Usage::

    python scripts/check_roundtrip.py --subject-list splits/train_ids.txt
    python scripts/check_roundtrip.py --subject-list splits/val_ids.txt --limit 5
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # repo root, for `src`
sys.path.insert(0, str(_HERE))         # sibling scripts, for shared helpers

from crop_stats import describe_crop_config, read_subject_list  # noqa: E402
from src.data import (  # noqa: E402
    DATA_ROOT,
    SIDES,
    list_subjects,
    load_mesh,
    load_subject_landmarks,
    mesh_path,
)
from src.geometry import (  # noqa: E402
    DEFAULT_CROP_CONFIG,
    EarTransform,
    canonicalize_ear,
    crop_ear,
    inverse_transform_points,
    load_crop_config,
    make_transform,
    transform_points_to_canonical,
)

AXES = ("X", "Y", "Z")

# The frozen canonicalisation convention (canonicalize_ear's defaults, recorded
# in DATA_SPEC.md). Deliberately NOT command-line options: this script must
# check the transform the pipeline actually uses, not a variant of it.
MIRROR_SIDE = "right"
MIRROR_AXIS = 1

# ... and they must stay the pipeline's, not this script's: a silent drift here
# would mean checking a transform canonicalize_ear no longer builds.
_PIPELINE_DEFAULTS = inspect.signature(canonicalize_ear).parameters
for _name, _expected in (("mirror_side", MIRROR_SIDE), ("mirror_axis", MIRROR_AXIS)):
    # A RuntimeError, not an assert: `python -O` would strip an assert and let
    # this script quietly validate a transform the pipeline no longer builds.
    if _PIPELINE_DEFAULTS[_name].default != _expected:
        raise RuntimeError(
            f"check_roundtrip.{_name.upper()} is {_expected!r} but canonicalize_ear "
            f"now defaults to {_PIPELINE_DEFAULTS[_name].default!r}: this script would "
            "check a transform the pipeline does not build. Update both together."
        )

DEFAULT_TOL = 1e-9        # mm
CANONICAL_LIMIT = 1.5     # "well scaled for the template" band, informational
MAX_IDS_LISTED = 20
PROGRESS_EVERY = 25


def ear_roundtrip(
    landmarks: np.ndarray, crop_points: np.ndarray, side: str
) -> tuple[float, np.ndarray, float]:
    """``(max abs round-trip error in mm, [85,3] canonical landmarks, scale)``.

    The transform is derived from ``crop_points`` (mesh) alone; ``landmarks``
    are only mapped through it and back.

    The inverse deliberately uses a transform rebuilt from ``to_dict()`` rather
    than the original object: that is the path the pipeline really takes (the
    transform is written to the npz cache and reloaded before Role D inverts a
    prediction), so precision lost in serialisation is measured here instead of
    surfacing in the submission.
    """
    transform = make_transform(crop_points, side, MIRROR_SIDE, MIRROR_AXIS)
    canonical = transform_points_to_canonical(landmarks, transform)
    cached = EarTransform.from_dict(transform.to_dict())
    restored = inverse_transform_points(canonical, cached)
    error = float(np.max(np.abs(restored - np.asarray(landmarks, dtype=np.float64))))
    return error, canonical, float(transform.scale)


class SideAccumulator:
    """Running aggregates for one side — never keeps a landmark coordinate."""

    def __init__(self) -> None:
        self.n_ears = 0
        self.max_error = 0.0
        self.worst_subject: str | None = None
        self.canonical_lo = np.full(3, np.inf)
        self.canonical_hi = np.full(3, -np.inf)
        self.max_abs: list[float] = []      # per ear, reduced to min/median/max
        self.outside_ids: list[str] = []
        self.scales: list[float] = []       # mm per canonical unit
        self.crop_sizes: list[int] = []

    def add(self, subject_id: str, error: float, canonical: np.ndarray,
            scale: float, n_crop: int) -> None:
        self.n_ears += 1
        if error > self.max_error:
            self.max_error = error
            self.worst_subject = subject_id
        self.canonical_lo = np.minimum(self.canonical_lo, canonical.min(axis=0))
        self.canonical_hi = np.maximum(self.canonical_hi, canonical.max(axis=0))
        self.max_abs.append(float(np.max(np.abs(canonical))))
        if np.any(np.abs(canonical) > CANONICAL_LIMIT):
            self.outside_ids.append(subject_id)
        self.scales.append(float(scale))
        self.crop_sizes.append(int(n_crop))


def _stats_line(label: str, values: list[float], fmt: str = ".2f") -> str:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return f"  {label:<28}(none)"
    return (
        f"  {label:<28}min {arr.min():{fmt}}   median {np.median(arr):{fmt}}   "
        f"max {arr.max():{fmt}}"
    )


def print_side_report(side: str, acc: SideAccumulator, tol: float) -> None:
    print(f"\n--- {side} ({acc.n_ears} ears) ---")
    if acc.n_ears == 0:
        print("  no ears processed")
        return

    print(f"  max round-trip error: {acc.max_error:.3e} mm "
          f"(worst subject {acc.worst_subject}, tolerance {tol:.0e})")
    print(f"  {'':<28}" + "".join(f"{ax:>12}" for ax in AXES))
    print(f"  {'canonical GT min':<28}" + "".join(f"{v:>12.3f}" for v in acc.canonical_lo))
    print(f"  {'canonical GT max':<28}" + "".join(f"{v:>12.3f}" for v in acc.canonical_hi))
    print(_stats_line("per-ear max |canonical|", acc.max_abs, ".3f"))
    print(_stats_line("transform scale (mm)", acc.scales))
    print(_stats_line("crop vertices", [float(v) for v in acc.crop_sizes], ".0f"))
    n_out = len(acc.outside_ids)
    print(f"  ears with any canonical landmark outside "
          f"[{-CANONICAL_LIMIT}, {CANONICAL_LIMIT}]: {n_out}/{acc.n_ears}")
    if n_out:
        shown = acc.outside_ids[:MAX_IDS_LISTED]
        more = "" if n_out <= MAX_IDS_LISTED else f" (+{n_out - MAX_IDS_LISTED} more)"
        print(f"    {', '.join(shown)}{more}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=DATA_ROOT,
                    help="dataset root (default: $HUAWEI_DATA_ROOT)")
    ap.add_argument("--subject-list", type=Path, required=True,
                    help="REQUIRED text file of subject IDs, one per line "
                         "(e.g. splits/train_ids.txt)")
    ap.add_argument("--limit", type=int, default=0,
                    help="check only the first N listed subjects (0 = all)")
    ap.add_argument("--crop-config", type=Path, default=Path(DEFAULT_CROP_CONFIG),
                    help="frozen crop config (default: configs/crop.yaml)")
    ap.add_argument("--allow-failures", action="store_true",
                    help="still report subjects that are missing or fail to load, but "
                         "let the verdict stand on the ears that were checked; without "
                         "it any unchecked ear fails the run")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL,
                    help=f"max allowed absolute round-trip error, mm "
                         f"(default {DEFAULT_TOL:.0e})")
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
    print(f"transform: mesh-derived only, mirror_side={MIRROR_SIDE!r}, "
          f"mirror_axis={MIRROR_AXIS} (GT is never used to build it)")
    print(f"subjects to check: {len(subjects)}"
          + (f"  (--limit {args.limit})" if args.limit > 0 else ""))
    if missing:
        print(f"  !! {len(missing)} listed subject(s) have no mesh under {root}: "
              f"{missing[:10]}")
    if not subjects:
        print("!! no subjects found")
        return 1

    acc = {side: SideAccumulator() for side in SIDES}
    # A subject that never gets checked is a failure, not a footnote: a wrong
    # --root or a stale subject list could otherwise check 3 ears out of 400 and
    # still print PASS.
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
                crop_points, _, qa = crop_ear(raw, side, configs[side])
                if crop_points.shape[0] == 0:
                    raise ValueError("the frozen crop box contains no vertices")
                error, canonical, scale = ear_roundtrip(landmarks[side], crop_points, side)
            except Exception as exc:
                failures.append(f"{sid} {side}: {type(exc).__name__}: {exc}")
                continue
            acc[side].add(sid, error, canonical, scale, qa["n_vertices"])
        if i % PROGRESS_EVERY == 0:
            print(f"  ... {i}/{len(subjects)}", flush=True)

    for side in SIDES:
        print_side_report(side, acc[side], args.tol)

    if failures:
        print(f"\n!! {len(failures)} subject(s)/ear(s) could not be checked:")
        for f in failures[:20]:
            print(f"   {f}")
        if len(failures) > 20:
            print(f"   ... and {len(failures) - 20} more")

    checked = sum(a.n_ears for a in acc.values())
    worst = max((a.max_error for a in acc.values()), default=0.0)
    print("\n=== ROUND-TRIP ===")
    print(f"  ears checked: {checked}")
    print(f"  max |inverse(canonical(GT)) - GT| over all ears: {worst:.3e} mm")
    print(f"  tolerance: {args.tol:.0e} mm")

    if args.limit > 0:
        print(f"  NOTE: --limit {args.limit} - this covered only the first "
              f"{len(subjects)} listed subject(s), not the whole list.")

    if checked == 0:
        print("  => FAIL - nothing was checked.")
    elif worst > args.tol:
        print("  => FAIL - the canonical transform is NOT exactly invertible; "
              "predictions cannot be mapped back to Huawei coordinates safely.")
    elif failures and not args.allow_failures:
        print("  => FAIL - the round-trip was exact on every ear that was checked, "
              "but some subjects/ears could not be checked at all (see above). "
              "Fix --root or the subject list, or pass --allow-failures.")
    elif failures:
        print("  => PASS (--allow-failures) - exact on every ear that was checked; "
              f"{len(failures)} could not be checked and were ignored.")
    else:
        print("  => PASS - the round-trip is exact to within the tolerance.")

    print("\n(The canonical envelope above is informational, for Role C's template; "
          "it never fails this check.)")
    ok = checked > 0 and worst <= args.tol and (not failures or args.allow_failures)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

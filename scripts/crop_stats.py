"""Estimate and FREEZE the per-side ear crop boxes from training landmarks (Role A).

Run by Alfred against the NDA-protected data. Reads every subject's left and
right landmark CSVs, and per side computes

* the per-axis min/max **envelope** of all landmarks over all subjects,
* the **median per-subject ear extent** per axis (bbox hi - lo),
* a per-axis **margin** = max(margin_frac * median extent, min_margin_mm),

and writes ``configs/crop.yaml`` with ``lo = envelope_lo - margin`` and
``hi = envelope_hi + margin`` per side, plus the statistics used.

The output is an *aggregate* statistic over the training annotations. It is the
one place GT is allowed to influence preprocessing: bounds are estimated once,
frozen in the repo, and from then on cropping uses the mesh + this config only
(see the no-leakage rule in ``CLAUDE.md``). The YAML is fine to commit; nothing
in it identifies an individual point.

Prints a summary table only (envelope values, medians, margins, sanity counts);
never an individual landmark.

Usage::

    python scripts/crop_stats.py                       # all subjects under $HUAWEI_DATA_ROOT
    python scripts/crop_stats.py --subject-list train_ids.txt   # restrict to a training split
    python scripts/crop_stats.py --out configs/crop.yaml --margin-frac 0.25 --min-margin 15
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `src`

from src.data import DATA_ROOT, SIDES, list_subjects, load_subject_landmarks  # noqa: E402

AXES = ("X", "Y", "Z")


def side_stats(bbox_lo: np.ndarray, bbox_hi: np.ndarray, margin_frac: float, min_margin: float) -> dict:
    """Aggregate statistics for one side from per-subject landmark bboxes.

    Parameters
    ----------
    bbox_lo, bbox_hi : np.ndarray
        ``[S, 3]`` per-subject landmark bbox corners.
    """
    envelope_lo = bbox_lo.min(axis=0)
    envelope_hi = bbox_hi.max(axis=0)
    extents = bbox_hi - bbox_lo                       # [S, 3]
    median_extent = np.median(extents, axis=0)
    margin = np.maximum(margin_frac * median_extent, min_margin)
    lo = envelope_lo - margin
    hi = envelope_hi + margin

    # Sanity: by construction no subject can exceed the unexpanded envelope.
    outside = np.any((bbox_lo < envelope_lo) | (bbox_hi > envelope_hi), axis=1)
    headroom = np.minimum(bbox_lo - lo, hi - bbox_hi)  # [S, 3] distance to each face pair
    return {
        "lo": lo,
        "hi": hi,
        "envelope_lo": envelope_lo,
        "envelope_hi": envelope_hi,
        "median_extent": median_extent,
        "min_extent": extents.min(axis=0),
        "max_extent": extents.max(axis=0),
        "margin": margin,
        "n_subjects": int(bbox_lo.shape[0]),
        "n_outside_envelope": int(outside.sum()),
        "min_headroom": float(headroom.min()),
        "min_headroom_per_axis": headroom.min(axis=0),
    }


def collect_bboxes(subjects: list[str], root: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per side: ``([S,3] bbox lo, [S,3] bbox hi)`` of each subject's landmarks."""
    lo = {side: [] for side in SIDES}
    hi = {side: [] for side in SIDES}
    failures: list[str] = []
    for i, sid in enumerate(subjects, 1):
        try:
            lm = load_subject_landmarks(sid, root)
        except Exception as exc:  # keep going; report at the end
            failures.append(f"{sid}: {type(exc).__name__}: {exc}")
            continue
        for side in SIDES:
            lo[side].append(lm[side].min(axis=0))
            hi[side].append(lm[side].max(axis=0))
        if i % 50 == 0:
            print(f"  ... {i}/{len(subjects)}", flush=True)
    if failures:
        print(f"  !! {len(failures)} subject(s) failed to load (excluded):")
        for f in failures[:20]:
            print(f"     {f}")
    return {side: (np.asarray(lo[side]), np.asarray(hi[side])) for side in SIDES}


def _row(label: str, values: np.ndarray, width: int = 10) -> str:
    return f"  {label:<26}" + "".join(f"{float(v):>{width}.2f}" for v in values)


def print_summary(stats: dict[str, dict]) -> None:
    for side in SIDES:
        st = stats[side]
        print(f"\n--- {side} ({st['n_subjects']} subjects) ---")
        print(f"  {'':<26}" + "".join(f"{ax:>10}" for ax in AXES))
        print(_row("envelope lo", st["envelope_lo"]))
        print(_row("envelope hi", st["envelope_hi"]))
        print(_row("envelope extent", st["envelope_hi"] - st["envelope_lo"]))
        print(_row("ear extent min", st["min_extent"]))
        print(_row("ear extent median", st["median_extent"]))
        print(_row("ear extent max", st["max_extent"]))
        print(_row("margin", st["margin"]))
        print(_row("crop lo", st["lo"]))
        print(_row("crop hi", st["hi"]))
        print(_row("crop extent", st["hi"] - st["lo"]))
        print(_row("min headroom per axis", st["min_headroom_per_axis"]))
        print(f"  subjects outside unexpanded envelope : {st['n_outside_envelope']}   (must be 0)")
        print(f"  min headroom bbox -> crop face (mm)  : {st['min_headroom']:.2f}")


def _floats(a: np.ndarray) -> list[float]:
    return [round(float(v), 4) for v in a]


def write_yaml(path: Path, stats: dict[str, dict], args: argparse.Namespace, n_subjects: int) -> None:
    header = (
        "# configs/crop.yaml - frozen per-side ear crop boxes, Huawei original frame, mm.\n"
        "# Generated by scripts/crop_stats.py from TRAINING landmark statistics.\n"
        "# This is a DERIVED AGGREGATE statistic (envelope + margin), not raw data:\n"
        "# it is fine to keep in the repo. Do not hand-edit lo/hi - re-run the script.\n"
        "# Inference uses mesh + this config only (no-leakage rule, CLAUDE.md).\n"
    )
    body = {
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "source": "scripts/crop_stats.py",
        "n_subjects": int(n_subjects),
        "margin_frac": float(args.margin_frac),
        "min_margin_mm": float(args.min_margin),
        "axes": list(AXES),
        "sides": {},
    }
    for side in SIDES:
        st = stats[side]
        body["sides"][side] = {
            "lo": _floats(st["lo"]),
            "hi": _floats(st["hi"]),
            "min_vertices": int(args.min_vertices),
            "stats": {
                "n_subjects": st["n_subjects"],
                "envelope_lo": _floats(st["envelope_lo"]),
                "envelope_hi": _floats(st["envelope_hi"]),
                "median_extent": _floats(st["median_extent"]),
                "margin": _floats(st["margin"]),
                "n_outside_envelope": st["n_outside_envelope"],
                "min_headroom_mm": round(st["min_headroom"], 4),
            },
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(header)
        yaml.safe_dump(body, fh, sort_keys=False, default_flow_style=None)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DATA_ROOT, help="dataset root (default: $HUAWEI_DATA_ROOT)")
    ap.add_argument("--out", type=Path, default=Path("configs/crop.yaml"), help="output YAML path")
    ap.add_argument("--subject-list", type=Path, default=None,
                    help="optional text file, one subject ID per line, to restrict to a training split")
    ap.add_argument("--margin-frac", type=float, default=0.25, help="margin as a fraction of median ear extent")
    ap.add_argument("--min-margin", type=float, default=15.0, help="minimum margin per axis, mm")
    ap.add_argument("--min-vertices", type=int, default=500, help="CropConfig.min_vertices to record")
    ap.add_argument("--dry-run", action="store_true", help="print the summary but write nothing")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).expanduser()

    subjects = list_subjects(root)
    if args.subject_list is not None:
        wanted = {line.strip() for line in args.subject_list.read_text().splitlines() if line.strip()}
        missing = sorted(wanted - set(subjects))
        if missing:
            print(f"  !! {len(missing)} listed subject(s) have no mesh: {missing[:10]}")
        subjects = [s for s in subjects if s in wanted]
    print(f"root: {root}\nsubjects: {len(subjects)}")
    if not subjects:
        print("!! no subjects found")
        return 1

    bboxes = collect_bboxes(subjects, root)
    stats = {}
    for side in SIDES:
        lo, hi = bboxes[side]
        if lo.shape[0] == 0:
            print(f"!! no landmarks loaded for side {side}")
            return 1
        stats[side] = side_stats(lo, hi, args.margin_frac, args.min_margin)

    print_summary(stats)

    bad = [side for side in SIDES if stats[side]["n_outside_envelope"] != 0]
    if bad:
        print(f"\n!! internal error: subjects outside their own envelope on {bad}")
        return 1

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0
    write_yaml(args.out, stats, args, stats["left"]["n_subjects"])
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

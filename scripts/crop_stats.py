"""Estimate and FREEZE the per-side ear crop boxes from TRAINING landmarks (Role A).

Run by Alfred against the NDA-protected data. Reads the landmark CSVs of the
subjects named in ``--subject-list`` (a training split — required, never the
whole dataset), and per side computes

* the per-axis min/max **envelope** of all landmarks over those subjects,
* the **median per-subject ear extent** per axis (bbox hi - lo),
* a per-axis **margin** = max(margin_frac * median extent, min_margin_mm),

and writes ``configs/crop.yaml`` with ``lo = envelope_lo - margin`` and
``hi = envelope_hi + margin`` per side, plus the statistics used and the
provenance of the split it was derived from (path, sha256, subject count).

The output is an *aggregate* statistic over the training annotations. It is the
one place GT is allowed to influence preprocessing: bounds are estimated once,
frozen in the repo, and from then on cropping uses the mesh + this config only
(see the no-leakage rule in ``CLAUDE.md``). The YAML is fine to commit; nothing
in it identifies an individual point.

**Generalisation check (leave-one-out).** "No training subject sticks out of the
envelope" is true by construction and therefore says nothing. Instead, for every
subject the envelope and margin are recomputed *without* that subject, and the
subject's own landmark bbox is measured against the resulting box. That is an
honest estimate of the headroom an *unseen* subject can expect.

**Freeze criterion, printed at the end:** the minimum leave-one-out headroom
must be **> 0 on every axis for both sides**.

Prints a summary table only (envelope values, medians, margins, LOO headroom);
never an individual landmark.

Usage::

    python scripts/crop_stats.py --subject-list splits/train_ids.txt
    python scripts/crop_stats.py --subject-list splits/train_ids.txt --dry-run
    python scripts/crop_stats.py --subject-list splits/train_ids.txt \\
        --out configs/crop.yaml --margin-frac 0.25 --min-margin 15
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # for `src`

from src.data import DATA_ROOT, SIDES, list_subjects, load_subject_landmarks  # noqa: E402

AXES = ("X", "Y", "Z")
N_WORST = 5  # how many worst-headroom subjects to print per side


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file, so a stale crop.yaml can be spotted."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def box_from_bboxes(
    bbox_lo: np.ndarray, bbox_hi: np.ndarray, margin_frac: float, min_margin: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(lo, hi, envelope_lo, envelope_hi, margin)`` for a set of per-subject bboxes.

    The single place the crop box is defined, so the leave-one-out check below
    reproduces the real recipe (margin included) rather than an approximation
    of it.
    """
    envelope_lo = bbox_lo.min(axis=0)
    envelope_hi = bbox_hi.max(axis=0)
    median_extent = np.median(bbox_hi - bbox_lo, axis=0)
    margin = np.maximum(margin_frac * median_extent, min_margin)
    return envelope_lo - margin, envelope_hi + margin, envelope_lo, envelope_hi, margin


def leave_one_out_headroom(
    bbox_lo: np.ndarray, bbox_hi: np.ndarray, margin_frac: float, min_margin: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Headroom of each subject against the box built *without* it.

    Returns ``(headroom, lower_slack, upper_slack)``, each ``[S, 3]``::

        lower_slack[i, a] = bbox_lo[i, a] - lo_loo[i, a]     # room below
        upper_slack[i, a] = hi_loo[i, a] - bbox_hi[i, a]     # room above
        headroom[i, a]    = min(lower_slack[i, a], upper_slack[i, a])

    Negative means that subject's landmarks would fall outside a crop box
    estimated from the other subjects — i.e. an unseen subject like it would be
    truncated. The two slacks are kept separately so a failure says *which* face
    is too tight, not just that one of them is.
    """
    n = bbox_lo.shape[0]
    lower = np.empty((n, 3), dtype=np.float64)
    upper = np.empty((n, 3), dtype=np.float64)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        keep[i] = False
        lo, hi, *_ = box_from_bboxes(bbox_lo[keep], bbox_hi[keep], margin_frac, min_margin)
        keep[i] = True
        lower[i] = bbox_lo[i] - lo
        upper[i] = hi - bbox_hi[i]
    return np.minimum(lower, upper), lower, upper


def side_stats(
    subjects: list[str],
    bbox_lo: np.ndarray,
    bbox_hi: np.ndarray,
    margin_frac: float,
    min_margin: float,
) -> dict:
    """Aggregate statistics for one side from per-subject landmark bboxes.

    Parameters
    ----------
    subjects : list[str]
        Subject IDs, row-aligned with ``bbox_lo`` / ``bbox_hi``.
    bbox_lo, bbox_hi : np.ndarray
        ``[S, 3]`` per-subject landmark bbox corners.
    """
    lo, hi, envelope_lo, envelope_hi, margin = box_from_bboxes(
        bbox_lo, bbox_hi, margin_frac, min_margin
    )
    extents = bbox_hi - bbox_lo                        # [S, 3]

    loo, loo_lower, loo_upper = leave_one_out_headroom(
        bbox_lo, bbox_hi, margin_frac, min_margin
    )
    loo_per_subject = loo.min(axis=1)                  # worst axis per subject
    worst_order = np.argsort(loo_per_subject)[:N_WORST]

    return {
        "lo": lo,
        "hi": hi,
        "envelope_lo": envelope_lo,
        "envelope_hi": envelope_hi,
        "median_extent": np.median(extents, axis=0),
        "min_extent": extents.min(axis=0),
        "max_extent": extents.max(axis=0),
        "margin": margin,
        "n_subjects": int(bbox_lo.shape[0]),
        "loo_min_headroom_per_axis": loo.min(axis=0),
        "loo_min_headroom": float(loo.min()),
        # Which face binds, per axis, across all subjects: "lo" if the minimum
        # room below the box beats the minimum room above it, "hi" otherwise.
        # This names the face attaining the reported per-axis minimum, which
        # need not come from a single subject.
        "loo_tight_face_per_axis": [
            "lo" if loo_lower[:, a].min() <= loo_upper[:, a].min() else "hi"
            for a in range(3)
        ],
        "loo_min_lower_per_axis": loo_lower.min(axis=0),
        "loo_min_upper_per_axis": loo_upper.min(axis=0),
        "loo_worst": [
            (
                subjects[int(i)],
                float(loo_per_subject[int(i)]),
                loo[int(i)].copy(),
                ["lo" if lo_v <= hi_v else "hi"
                 for lo_v, hi_v in zip(loo_lower[int(i)], loo_upper[int(i)])],
            )
            for i in worst_order
        ],
    }


def read_subject_list(path: Path) -> list[str]:
    """Subject IDs from a text file, one per line; blanks and ``#`` comments skipped."""
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        sid = line.strip()
        if sid and not sid.startswith("#"):
            ids.append(sid)
    if not ids:
        raise ValueError(f"{path}: no subject IDs")
    duplicates = sorted({sid for sid in ids if ids.count(sid) > 1})
    if duplicates:
        raise ValueError(f"{path}: duplicate subject IDs: {duplicates[:10]}")
    return ids


def collect_bboxes(
    subjects: list[str], root: Path
) -> tuple[list[str], dict[str, tuple[np.ndarray, np.ndarray]], list[str]]:
    """``(loaded_ids, {side: ([S,3] lo, [S,3] hi)}, failures)``.

    A subject is included only if *both* sides loaded, so the per-side arrays
    stay row-aligned with ``loaded_ids``.
    """
    lo: dict[str, list[np.ndarray]] = {side: [] for side in SIDES}
    hi: dict[str, list[np.ndarray]] = {side: [] for side in SIDES}
    loaded: list[str] = []
    failures: list[str] = []
    for i, sid in enumerate(subjects, 1):
        try:
            lm = load_subject_landmarks(sid, root)
        except Exception as exc:  # keep going; the caller decides whether to abort
            failures.append(f"{sid}: {type(exc).__name__}: {exc}")
            continue
        for side in SIDES:
            lo[side].append(lm[side].min(axis=0))
            hi[side].append(lm[side].max(axis=0))
        loaded.append(sid)
        if i % 50 == 0:
            print(f"  ... {i}/{len(subjects)}", flush=True)
    boxes = {side: (np.asarray(lo[side]), np.asarray(hi[side])) for side in SIDES}
    return loaded, boxes, failures


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
        print(_row("LOO min room below face", st["loo_min_lower_per_axis"]))
        print(_row("LOO min room above face", st["loo_min_upper_per_axis"]))
        print(_row("LOO min headroom / axis", st["loo_min_headroom_per_axis"]))
        print(f"  worst {N_WORST} leave-one-out subjects (mm; own bbox vs box built without them;")
        print("  'lo'/'hi' marks which face of that axis is the tight one):")
        print(f"    {'subject':<12}{'worst':>10}" + "".join(f"{ax:>13}" for ax in AXES))
        for sid, worst, per_axis, faces in st["loo_worst"]:
            print(
                f"    {sid:<12}{worst:>10.2f}"
                + "".join(f"{float(v):>10.2f} {f}" for v, f in zip(per_axis, faces))
            )


def freeze_verdict(stats: dict[str, dict]) -> bool:
    """True iff the min LOO headroom is > 0 on every axis for both sides."""
    return all(
        bool(np.all(stats[side]["loo_min_headroom_per_axis"] > 0.0)) for side in SIDES
    )


def print_freeze_criterion(stats: dict[str, dict]) -> bool:
    ok = freeze_verdict(stats)
    print("\n=== FREEZE CRITERION ===")
    print("  minimum leave-one-out headroom > 0 on every axis, both sides")
    for side in SIDES:
        per_axis = stats[side]["loo_min_headroom_per_axis"]
        faces = stats[side]["loo_tight_face_per_axis"]
        detail = "  ".join(
            f"{ax}={float(v):.2f}({f})" for ax, v, f in zip(AXES, per_axis, faces)
        )
        mark = "OK  " if np.all(per_axis > 0.0) else "FAIL"
        print(f"  [{mark}] {side:<6} {detail}")
    print("  => " + ("PASS - safe to freeze these bounds."
                     if ok else
                     "FAIL - an unseen subject would be truncated; widen the margin "
                     "(--margin-frac / --min-margin) and re-run."))
    return ok


def _floats(a: np.ndarray) -> list[float]:
    return [round(float(v), 4) for v in a]


def write_yaml(
    path: Path,
    stats: dict[str, dict],
    args: argparse.Namespace,
    split: dict,
    freeze_ok: bool,
) -> None:
    """Write the crop document. A rejected run gets a header that says so."""
    if not freeze_ok:
        # A rejected file must not read like a frozen one, or it invites a copy
        # into configs/. load_crop_config() also refuses a document carrying
        # freeze_criterion_passed: false, so this header is the warning, not
        # the only guard.
        _write_doc(
            path,
            "# REJECTED - these bounds FAILED the leave-one-out freeze criterion.\n"
            "# DO NOT copy this into configs/crop.yaml and DO NOT use it for training\n"
            "# or inference: an unseen subject would be truncated. Kept only so the\n"
            "# failing numbers can be inspected. Re-run scripts/crop_stats.py with a\n"
            "# wider margin (--margin-frac / --min-margin) and use that result.\n"
            "# src.geometry.load_crop_config() refuses to load this file.\n",
            stats, args, split, freeze_ok,
        )
        return
    header = (
        "# configs/crop.yaml - frozen per-side ear crop boxes, Huawei original frame, mm.\n"
        "# Generated by scripts/crop_stats.py from TRAINING landmark statistics.\n"
        "# Derived from the subject list in `split` below ONLY - regenerate whenever\n"
        "# that split changes (the sha256 lets you detect a stale config).\n"
        "# This is a DERIVED AGGREGATE statistic (envelope + margin), not raw data:\n"
        "# it is fine to keep in the repo. Do not hand-edit lo/hi - re-run the script.\n"
        "# Inference uses mesh + this config only (no-leakage rule, CLAUDE.md).\n"
    )
    _write_doc(path, header, stats, args, split, freeze_ok)


def _write_doc(
    path: Path,
    header: str,
    stats: dict[str, dict],
    args: argparse.Namespace,
    split: dict,
    freeze_ok: bool,
) -> None:
    """Serialise one crop document under the given header."""
    body = {
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "source": "scripts/crop_stats.py",
        "split": split,
        "n_subjects": int(split["n_subjects_used"]),
        "margin_frac": float(args.margin_frac),
        "min_margin_mm": float(args.min_margin),
        "freeze_criterion": "min leave-one-out headroom > 0 on every axis, both sides",
        "freeze_criterion_passed": bool(freeze_ok),
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
                "loo_min_headroom_per_axis_mm": _floats(st["loo_min_headroom_per_axis"]),
                "loo_min_headroom_mm": round(st["loo_min_headroom"], 4),
                "loo_min_room_below_face_mm": _floats(st["loo_min_lower_per_axis"]),
                "loo_min_room_above_face_mm": _floats(st["loo_min_upper_per_axis"]),
                "loo_tight_face_per_axis": list(st["loo_tight_face_per_axis"]),
            },
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(header)
        yaml.safe_dump(body, fh, sort_keys=False, default_flow_style=None)



def describe_crop_config(path: Path) -> str:
    """One-line provenance of a crop config: which split it came from and its verdict.

    Used by the stale-file warning below and by the check_* scripts, so a run
    always says which frozen box it validated.
    """
    if not path.is_file() and not path.is_absolute() and (REPO_ROOT / path).is_file():
        # Same "cwd first, then repo root" rule as geometry.load_crop_config, so
        # the provenance line describes the file that was actually loaded.
        path = REPO_ROOT / path
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        split = doc.get("split") or {}
        return (
            f"generated {doc.get('generated', '?')} from "
            f"{split.get('path', '?')} (sha256 {str(split.get('sha256', '?'))[:16]}..., "
            f"{split.get('n_subjects_used', '?')} subjects), "
            f"freeze_criterion_passed={doc.get('freeze_criterion_passed', '?')}"
        )
    except Exception as exc:  # a corrupt existing file must not mask the real failure
        return f"could not be read ({type(exc).__name__}: {exc})"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=DATA_ROOT,
                    help="dataset root (default: $HUAWEI_DATA_ROOT)")
    ap.add_argument("--out", type=Path, default=Path("configs/crop.yaml"),
                    help="output YAML path")
    ap.add_argument("--subject-list", type=Path, required=True,
                    help="REQUIRED text file of TRAINING subject IDs, one per line "
                         "(e.g. splits/train_ids.txt). Never the full dataset: the crop "
                         "box must not see validation landmarks.")
    ap.add_argument("--margin-frac", type=float, default=0.25,
                    help="margin as a fraction of median ear extent")
    ap.add_argument("--min-margin", type=float, default=15.0,
                    help="minimum margin per axis, mm")
    ap.add_argument("--min-vertices", type=int, default=500,
                    help="CropConfig.min_vertices to record")
    ap.add_argument("--allow-failures", action="store_true",
                    help="continue (and still write the YAML) when some subjects fail to "
                         "load; without it any load failure aborts with no YAML written")
    ap.add_argument("--dry-run", action="store_true", help="print the summary but write nothing")
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
    split_sha = sha256_file(split_path)

    available = set(list_subjects(root))
    missing = sorted(set(wanted_ids) - available)
    subjects = [sid for sid in wanted_ids if sid in available]

    print(f"root: {root}")
    print(f"subject list: {split_path}  (sha256 {split_sha[:16]}..., {len(wanted_ids)} IDs)")
    print(f"subjects to use: {len(subjects)}")
    if missing:
        print(f"  !! {len(missing)} listed subject(s) have no mesh: {missing[:10]}")
        if not args.allow_failures:
            print("!! aborting: listed subjects are missing (pass --allow-failures to proceed)")
            return 1
    if not subjects:
        print("!! no subjects found")
        return 1

    loaded, bboxes, failures = collect_bboxes(subjects, root)
    if failures:
        print(f"  !! {len(failures)} subject(s) failed to load:")
        for f in failures[:20]:
            print(f"     {f}")
        if not args.allow_failures:
            print("!! aborting: no YAML written (pass --allow-failures to proceed anyway)")
            return 1
        print("  --allow-failures: continuing without them")

    stats = {}
    for side in SIDES:
        lo, hi = bboxes[side]
        if lo.shape[0] < 2:
            print(f"!! need at least 2 subjects for the leave-one-out check, "
                  f"side {side} has {lo.shape[0]}")
            return 1
        stats[side] = side_stats(loaded, lo, hi, args.margin_frac, args.min_margin)

    print_summary(stats)
    freeze_ok = print_freeze_criterion(stats)

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0 if freeze_ok else 1

    split = {
        "path": split_path.as_posix(),
        "sha256": split_sha,
        "n_subjects_listed": len(wanted_ids),
        "n_subjects_used": len(loaded),
        "n_subjects_failed": len(failures) + len(missing),
    }
    if not freeze_ok:
        # Never write the frozen path with bounds that failed the criterion:
        # load_crop_config() does not inspect freeze_criterion_passed, so a
        # rejected config left at configs/crop.yaml would be picked up silently
        # by the rest of the team. Write it beside the target for inspection.
        rejected = args.out.with_name(args.out.stem + ".rejected" + args.out.suffix)
        write_yaml(rejected, stats, args, split, freeze_ok)
        print(f"\n!! freeze criterion FAILED - {args.out} NOT written")
        print(f"   bounds written to {rejected} for inspection only - do not freeze them")
        if args.out.exists():
            # Refusing to write leaves whatever was there before, which is the
            # file the rest of the team loads. Say so loudly and name the split
            # it came from, rather than letting a stale freeze pass unnoticed.
            print(f"   !! {args.out} STILL EXISTS from an earlier run and is what")
            print("      load_crop_config() will return. Check its `split` block:")
            print(f"      {describe_crop_config(args.out)}")
            print("      Delete or rename it unless you are sure it is still the one you want.")
        return 1

    write_yaml(args.out, stats, args, split, freeze_ok)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

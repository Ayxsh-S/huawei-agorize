"""QA plots: head mesh + optional per-side landmarks and frozen crop boxes (Role A).

Usage::

    python scripts/plot_ear.py --subject P0001                      # $HUAWEI_DATA_ROOT layout
    python scripts/plot_ear.py --subject P0001 --crop configs/crop.yaml
    python scripts/plot_ear.py --mesh <head.ply> [--left <l.csv>] [--right <r.csv>] [--crop ...]

Produces, under ``--out`` (default ``outputs/``):

* ``<subject>_head.png``   — whole mesh, any landmarks, both crop boxes (with ``--crop``)
* ``<subject>_left.png``   — zoomed left ear: cropped mesh points (with ``--crop``) or the
                             mesh near the landmarks, plus landmarks if available
* ``<subject>_right.png``  — same for the right ear

Runs fine with no landmarks at all (inference-style view): with ``--crop`` you
still get the per-side cropped points, without it just the head view.

Without ``--crop`` the zoom window of a per-side figure is taken from that
side's landmark bbox. That is a *display* choice in a QA script — it never
becomes a crop, centre, scale or transform. Real crop bounds come from the
frozen training statistics in ``configs/crop.yaml`` (see the no-leakage rule in
``CLAUDE.md``).

Contour index ranges follow ``DATA_SPEC.md``. They are marked there as assumed
sequential; ``--lines`` (on by default) joins each contour in index order and
labels each contour's first point with its start index, which is how you check
that assumption by eye.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script only ever writes files

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `src`

from src.data import DATA_ROOT, SIDES, landmark_path, load_landmarks, load_mesh, mesh_path  # noqa: E402
from src.geometry import CropConfig, crop_ear, load_crop_config  # noqa: E402

# Landmark contours, per DATA_SPEC.md (85 = 25 + 30 + 20 + 10).
CONTOURS: tuple[tuple[str, int, int, str], ...] = (
    ("outer helix", 0, 25, "tab:red"),
    ("concha outline", 25, 55, "tab:blue"),
    ("inner helix", 55, 75, "tab:green"),
    ("superior antihelix", 75, 85, "tab:orange"),
)

BOX_COLOURS = {"left": "tab:purple", "right": "tab:cyan"}

DEFAULT_MAX_POINTS = 20_000

# The 12 edges of an axis-aligned box, as index pairs into its 8 corners.
_BOX_EDGES = (
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def box_corners(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """``[8, 3]`` corners of the box ``[lo, hi]``, bit ``i`` of the index picks hi on axis ``i``."""
    corners = np.empty((8, 3))
    for k in range(8):
        corners[k] = [hi[a] if (k >> a) & 1 else lo[a] for a in range(3)]
    return corners


def subsample(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    """Random subset of at most ``max_points`` rows, deterministic per seed."""
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), size=max_points, replace=False)]


def _set_equal_aspect(ax, points: np.ndarray) -> None:
    """Equal data aspect, so an ear is not silently stretched."""
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    centre = (lo + hi) / 2.0
    radius = float((hi - lo).max()) / 2.0 or 1.0
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_zlim(centre[2] - radius, centre[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def draw_box(ax, lo: np.ndarray, hi: np.ndarray, colour: str, label: str) -> None:
    """Wireframe of an axis-aligned box."""
    corners = box_corners(lo, hi)
    for n, (i, j) in enumerate(_BOX_EDGES):
        seg = corners[[i, j]]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], c=colour, lw=1.2, alpha=0.9,
                label=label if n == 0 else None)


def plot_view(
    vertices: np.ndarray,
    landmarks: dict[str, np.ndarray],
    boxes: dict[str, CropConfig],
    title: str,
    out_path: Path,
    point_size: float,
    draw_lines: bool,
) -> Path:
    """One 3-D figure: grey mesh points, landmarks by contour, crop boxes as wireframes."""
    figure = plt.figure(figsize=(10, 9))
    ax = figure.add_subplot(111, projection="3d")

    ax.scatter(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        s=point_size,
        c="0.65",
        alpha=0.35,
        linewidths=0,
        depthshade=False,
    )

    for side, points in landmarks.items():
        marker = "o" if side == "left" else "^"
        for name, start, stop, colour in CONTOURS:
            contour = points[start:stop]
            if not len(contour):
                continue
            ax.scatter(
                contour[:, 0],
                contour[:, 1],
                contour[:, 2],
                s=26,
                c=colour,
                marker=marker,
                depthshade=False,
                label=f"{side} {name}",
            )
            if draw_lines and len(contour) > 1:
                ax.plot(contour[:, 0], contour[:, 1], contour[:, 2], c=colour, lw=0.8, alpha=0.7)
            # Mark each contour's first point, to read off ordering direction.
            ax.text(*contour[0], f" {start}", color=colour, fontsize=7)

    for side, cfg in boxes.items():
        draw_box(ax, cfg.lo, cfg.hi, BOX_COLOURS[side], f"{side} crop box")

    extent_parts = [vertices, *landmarks.values()]
    extent_parts += [box_corners(cfg.lo, cfg.hi) for cfg in boxes.values()]
    _set_equal_aspect(ax, np.vstack(extent_parts))
    ax.set_xlabel("X (anterior +)")
    ax.set_ylabel("Y (subject's left +)")
    ax.set_zlabel("Z (up +)")
    ax.set_title(title)
    if landmarks or boxes:
        ax.legend(loc="upper left", fontsize=7, markerscale=1.2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return out_path


def _near_landmarks(vertices: np.ndarray, points: np.ndarray, margin: float) -> np.ndarray:
    """Mesh vertices inside the landmark bbox grown by ``margin`` (display only)."""
    lo = points.min(axis=0) - margin
    hi = points.max(axis=0) + margin
    inside = np.all((vertices >= lo) & (vertices <= hi), axis=1)
    return vertices[inside]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_argument_group("input (one of --subject / --mesh)")
    src.add_argument("--subject", type=str, default=None,
                     help="subject ID, e.g. P0001; mesh and landmarks resolved under --root")
    src.add_argument("--root", type=Path, default=DATA_ROOT,
                     help="dataset root for --subject (default: $HUAWEI_DATA_ROOT)")
    src.add_argument("--mesh", type=Path, default=None, help="explicit head PLY to plot")
    src.add_argument("--left", type=Path, default=None, help="explicit left landmark CSV")
    src.add_argument("--right", type=Path, default=None, help="explicit right landmark CSV")
    parser.add_argument("--crop", type=Path, default=None,
                        help="configs/crop.yaml: draw both crop boxes and plot the cropped points per side")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="output directory")
    parser.add_argument(
        "--max-points",
        type=int,
        default=DEFAULT_MAX_POINTS,
        help=f"mesh vertices drawn per figure (default {DEFAULT_MAX_POINTS})",
    )
    parser.add_argument("--seed", type=int, default=0, help="subsampling seed")
    parser.add_argument(
        "--margin",
        type=float,
        default=15.0,
        help="padding around the landmark bbox in the zoomed views without --crop, in mm",
    )
    parser.add_argument(
        "--no-lines",
        dest="lines",
        action="store_false",
        help="do not join landmarks within a contour in index order",
    )
    args = parser.parse_args(argv)
    if (args.subject is None) == (args.mesh is None):
        parser.error("give exactly one of --subject or --mesh")
    if args.subject is not None and (args.left or args.right):
        parser.error("--left/--right are for --mesh; with --subject the landmark paths are resolved automatically")
    return args


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, dict[str, Path]]:
    """Mesh path and per-side landmark paths (only those that exist)."""
    if args.subject is not None:
        mesh = mesh_path(args.subject, args.root)
        candidates = {side: landmark_path(args.subject, side, args.root) for side in SIDES}
        landmarks = {side: p for side, p in candidates.items() if p.is_file()}
        for side in SIDES:
            if side not in landmarks:
                print(f"  note: no {side} landmark file for {args.subject} (expected {candidates[side].name})")
        return mesh, landmarks
    landmarks = {side: p for side, p in (("left", args.left), ("right", args.right)) if p is not None}
    return args.mesh, landmarks


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mesh_file, landmark_files = resolve_inputs(args)

    raw = load_mesh(mesh_file)
    vertices = raw.vertices
    print(f"{raw.subject_id}: {len(vertices)} vertices, faces={raw.faces is not None}, normals={raw.normals is not None}")
    print(f"  bbox lo {np.round(vertices.min(axis=0), 3)}  hi {np.round(vertices.max(axis=0), 3)}")

    landmarks: dict[str, np.ndarray] = {}
    for side, path in landmark_files.items():
        landmarks[side] = load_landmarks(path, side)
        print(f"  {side}: {len(landmarks[side])} landmarks from {path.name}")

    boxes: dict[str, CropConfig] = {}
    if args.crop is not None:
        for side in SIDES:
            boxes[side] = load_crop_config(side, args.crop)
            print(f"  {side} crop box lo {np.round(boxes[side].lo, 2)}  hi {np.round(boxes[side].hi, 2)}")

    written = [
        plot_view(
            subsample(vertices, args.max_points, args.seed),
            landmarks,
            boxes,
            f"{raw.subject_id} — full head",
            args.out / f"{raw.subject_id}_head.png",
            point_size=0.6,
            draw_lines=args.lines,
        )
    ]

    for side in SIDES:
        if side not in landmarks and side not in boxes:
            continue
        side_landmarks = {side: landmarks[side]} if side in landmarks else {}
        side_boxes = {side: boxes[side]} if side in boxes else {}

        if side in boxes:
            near, _, qa = crop_ear(raw, side, boxes[side])
            flag = "  !! SUSPICIOUS (below min_vertices)" if qa["suspicious"] else ""
            print(f"  {side} crop: {qa['n_vertices']} vertices{flag}")
            if side in landmarks:
                inside = np.all((landmarks[side] >= boxes[side].lo) & (landmarks[side] <= boxes[side].hi), axis=1)
                print(f"  {side} landmarks inside crop box: {int(inside.sum())}/{len(inside)}")
            if not len(near):
                print(f"  warning: {side} crop box contains no vertices — showing the whole mesh")
                near = vertices
        else:
            near = _near_landmarks(vertices, landmarks[side], args.margin)
            if not len(near):
                print(f"  warning: no mesh vertices within {args.margin} of the {side} landmarks")
                near = vertices

        written.append(
            plot_view(
                subsample(near, args.max_points, args.seed),
                side_landmarks,
                side_boxes,
                f"{raw.subject_id} — {side} ear",
                args.out / f"{raw.subject_id}_{side}.png",
                point_size=3.0,
                draw_lines=args.lines,
            )
        )

    if not landmarks and not boxes:
        print("  no landmarks or crop config given - head view only")
    for path in written:
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

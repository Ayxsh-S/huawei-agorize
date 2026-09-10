"""Settle the left/right mirror convention numerically (Role A).

``canonicalize_ear`` mirrors ONE side so both ears share a canonical
orientation. Which side, and on which axis, was assumed rather than measured:
``DATA_SPEC.md`` records ``mirror_side="right"``, ``mirror_axis=1`` (Y) as the
default, and the Y axis itself is verified (+Y is the subject's left on all 200
subjects), but nobody has checked that flipping Y actually makes a subject's two
canonical ears land on top of each other. That is what this script measures.

For every subject it compares the subject's OWN canonical left ear against its
OWN canonical right ear under three configurations::

    A   no mirror         (mirror_axis=None on both sides)
    B   mirror the RIGHT  (mirror_side="right", mirror_axis=1) - current default
    C   mirror the LEFT   (mirror_side="left",  mirror_axis=1)

Two metrics per configuration, aggregated over subjects as mean / median / p95:

1. **Symmetric Chamfer distance** between the two canonical point clouds - the
   mean of the two directed mean nearest-neighbour distances - computed on a
   deterministic 512-point subsample of each ear's 2048 canonical points.
   Shape agreement, no correspondence assumed.
2. **Mean per-landmark Euclidean distance** between the subject's canonical left
   and canonical right GT landmark sets, matched index for index. This is the
   decisive metric: it is the only one that can tell a genuinely aligned pair
   from two clouds that merely occupy the same box.

Both are in canonical units - the CROP is ~2 units on its longest axis, the GT
landmarks inside it span ~1.3 (see DATA_SPEC.md). The approximate millimetre
figure alongside is the canonical value times the mean of the two ears'
transform scales, printed for readability only.

The decisive metric has a FLOOR that is nothing to do with the mirror: the two
frozen crop boxes are not exact Y-reflections of each other, so each ear's
canonical origin sits in a slightly different place and even a perfectly
mirror-symmetric subject scores above zero. The run measures it
(:class:`FrameFloor`) and prints it per axis next to the metrics, because the
LEVEL of the decisive metric cannot be read without it - but see that class:
only the axes off the mirror are a true floor, the mirror axis itself is an
upper bound (it also carries the subject's own mid-sagittal offset), and none of
it may be subtracted from the metric. What is NOT affected is the comparison
between configurations - A, B and C share one crop, one centre, one scale and
one point draw per ear, so the floor is identical in all three.

Everything is rebuilt from the mesh with the frozen ``configs/crop.yaml`` - the
processed cache is NOT read, because it is fixed to the current mirror setting
and could only ever confirm it. The crop and the point draw are done once per
ear and shared by the three configurations, so the only thing that differs
between them is the mirror. That is what the pipeline would produce anyway:
neither the crop box nor the sampling seed depends on the mirror setting.

The GT landmarks are only *measured* here - every transform is derived from the
mesh alone (``crop_ear`` + ``make_transform``), so the no-leakage rule in
``CLAUDE.md`` holds: no annotation defines a crop, centre, scale or mirror.

Note on B vs C: applying the Y flip to both clouds instead of one is an
isometry, so B and C are mathematically indistinguishable by any distance-based
metric and this script reports them as an exact tie. What the run really decides
is mirror versus no mirror; choosing between B and C is a convention.

Also printed, for the winning configuration: the per-contour mean landmark
distance (0-24 outer helix, 25-54 concha, 55-74 inner helix, 75-84 superior
antihelix), so a single misordered contour shows up instead of hiding inside the
85-landmark average, plus the worst few subjects by landmark distance.

Writes ``<out>/mirror_check.png``: one subject under the winning configuration,
canonical left (blue) and canonical right (red) point clouds overlaid with both
landmark sets, from two viewing angles.

Prints aggregates, distances and subject IDs only - never a coordinate.

**This script changes no default.** If the winner is not the current default it
says so loudly and stops there; the decision is Alfred's. Exit code 1 only when
the check could not be run (nothing compared, or subjects that could not be
checked without ``--allow-failures``), never because of which configuration won.

Usage::

    python scripts/check_mirror.py --subject-list splits/train_ids.txt
    python scripts/check_mirror.py --subject-list splits/train_ids.txt --limit 20 --out outputs/
"""

from __future__ import annotations

import argparse
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent
sys.path.insert(0, str(REPO_ROOT))  # repo root, for `src`
sys.path.insert(0, str(_HERE))      # sibling scripts, for shared helpers

from crop_stats import describe_crop_config, read_subject_list  # noqa: E402
from preprocess import ear_seed  # noqa: E402
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
    canonicalize_ear,
    crop_ear,
    load_crop_config,
    make_transform,
    sample_points,
    transform_points_to_canonical,
)


@dataclass(frozen=True)
class MirrorSetting:
    """One candidate mirror convention, as passed to ``make_transform``."""

    key: str
    label: str
    mirror_side: str
    mirror_axis: int | None


DEFAULT_KEY = "B"

CONFIGS: tuple[MirrorSetting, ...] = (
    MirrorSetting("A", "no mirror (neither side flipped)", "right", None),
    MirrorSetting("B", "mirror the RIGHT side on Y  [CURRENT DEFAULT]", "right", 1),
    MirrorSetting("C", "mirror the LEFT side on Y", "left", 1),
)

# Configuration B must stay *exactly* the pipeline's default, or this script
# would compare three settings none of which is the one that ships. A
# RuntimeError, not an assert: `python -O` strips asserts.
_PIPELINE_DEFAULTS = inspect.signature(canonicalize_ear).parameters
_DEFAULT_SETTING = next(c for c in CONFIGS if c.key == DEFAULT_KEY)
for _name, _expected in (
    ("mirror_side", _DEFAULT_SETTING.mirror_side),
    ("mirror_axis", _DEFAULT_SETTING.mirror_axis),
):
    if _PIPELINE_DEFAULTS[_name].default != _expected:
        raise RuntimeError(
            f"check_mirror config {DEFAULT_KEY} has {_name}={_expected!r} but "
            f"canonicalize_ear now defaults to {_PIPELINE_DEFAULTS[_name].default!r}: "
            "the 'current default' column would be a fiction. Update both together."
        )

# Landmark contours, per DATA_SPEC.md (85 = 25 + 30 + 20 + 10).
CONTOURS: tuple[tuple[str, int, int], ...] = (
    ("outer helix", 0, 25),
    ("concha outline", 25, 55),
    ("inner helix", 55, 75),
    ("superior antihelix", 75, 85),
)
if CONTOURS[0][1] != 0 or CONTOURS[-1][2] != N_LANDMARKS or any(
    a[2] != b[1] for a, b in zip(CONTOURS, CONTOURS[1:])
):
    raise RuntimeError("CONTOURS must tile 0..N_LANDMARKS without gaps or overlaps")

AXIS_NAMES = ("X", "Y", "Z")

CHAMFER_POINTS = 512      # subsample per ear for the Chamfer distance
TIE_REL = 1e-9            # two configurations closer than this are a tie
ZERO_EPS = 1e-12          # below this a mean distance is reported as ~0
PROGRESS_EVERY = 25
WORST_N = 5
MAX_IDS_LISTED = 10
PLOT_NAME = "mirror_check.png"


# --- metrics ---------------------------------------------------------------- #

def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Chamfer distance between two point clouds.

    The mean of the two directed mean nearest-neighbour distances,
    ``0.5 * (mean_a min_b ||a-b|| + mean_b min_a ||a-b||)``. Symmetric in its
    arguments and zero exactly when the two clouds cover each other.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 3 or b.shape[1] != 3:
        raise ValueError(f"clouds must be [N,3], got {a.shape} and {b.shape}")
    if a.shape[0] == 0 or b.shape[0] == 0:
        raise ValueError("clouds must be non-empty")
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return 0.5 * (float(d.min(axis=1).mean()) + float(d.min(axis=0).mean()))


def subsample(points: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Deterministic subset of at most ``n`` rows (all of them if fewer)."""
    m = points.shape[0]
    if m <= n:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(m, size=n, replace=False)]


def contour_means(distances: np.ndarray) -> np.ndarray:
    """Mean of ``[85]`` per-landmark distances within each contour, ``[4]``."""
    return np.asarray(
        [float(distances[start:stop].mean()) for _, start, stop in CONTOURS],
        dtype=np.float64,
    )


# --- per-ear / per-subject work --------------------------------------------- #

@dataclass
class EarInputs:
    """One ear's mesh-derived inputs, shared by all three configurations."""

    crop_points: np.ndarray   # [M,3] mm, cropped mesh vertices
    sampled: np.ndarray       # [n_points,3] mm, the pipeline's draw
    landmarks: np.ndarray     # [85,3] mm GT - measured only, never a transform
    n_crop: int
    seed: int


def prepare_ear(
    raw: RawSubject,
    side: str,
    cfg: CropConfig,
    landmarks: np.ndarray,
    n_points: int = N_POINTS,
    seed: int = 0,
) -> EarInputs:
    """Crop and sample one ear once, mirror-independently.

    Raises ``ValueError`` for an empty or suspicious crop: the frozen box did
    not find that ear, so comparing it against the other side would measure
    nothing but the failure.
    """
    crop_points, _, qa = crop_ear(raw, side, cfg)
    if crop_points.shape[0] == 0:
        raise ValueError(f"{side}: the frozen crop box contains no vertices")
    if qa["suspicious"]:
        raise ValueError(
            f"{side}: suspicious crop, {qa['n_vertices']} vertices < min_vertices "
            f"{cfg.min_vertices}"
        )
    ear_rng_seed = ear_seed(seed, raw.subject_id, side)
    sampled, _ = sample_points(crop_points, None, n_points, ear_rng_seed)
    return EarInputs(
        crop_points=crop_points,
        sampled=sampled,
        landmarks=np.asarray(landmarks, dtype=np.float64),
        n_crop=int(qa["n_vertices"]),
        seed=ear_rng_seed,
    )


def canonicalise(
    ear: EarInputs, side: str, setting: MirrorSetting
) -> tuple[np.ndarray, np.ndarray, float]:
    """``(canonical points, canonical landmarks, scale mm)`` under one setting.

    The transform comes from the cropped MESH only; the landmarks are pushed
    through it, exactly as ``preprocess.py`` builds its cached targets.
    """
    transform = make_transform(
        ear.crop_points, side, setting.mirror_side, setting.mirror_axis
    )
    return (
        transform_points_to_canonical(ear.sampled, transform),
        transform_points_to_canonical(ear.landmarks, transform),
        float(transform.scale),
    )


@dataclass
class SubjectMetrics:
    """One subject under one configuration. Distances only, never coordinates."""

    chamfer: float
    landmark_mean: float
    contours: np.ndarray      # [4] per-contour mean landmark distance
    scale_mm: float           # mean of the two ears' scales, for the mm view

    @property
    def chamfer_mm(self) -> float:
        return self.chamfer * self.scale_mm

    @property
    def landmark_mean_mm(self) -> float:
        return self.landmark_mean * self.scale_mm


def compare_ears(
    ears: dict[str, EarInputs],
    setting: MirrorSetting,
    chamfer_points: int = CHAMFER_POINTS,
) -> tuple[SubjectMetrics, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compare one subject's two canonical ears under one configuration.

    Returns the metrics plus the canonical clouds and canonical landmark sets
    per side (the caller keeps those only for the plotted subject).
    """
    clouds: dict[str, np.ndarray] = {}
    lms: dict[str, np.ndarray] = {}
    scales: list[float] = []
    for side in SIDES:
        points, landmarks, scale = canonicalise(ears[side], side, setting)
        clouds[side] = points
        lms[side] = landmarks
        scales.append(scale)

    chamfer = chamfer_distance(
        subsample(clouds["left"], chamfer_points, ears["left"].seed),
        subsample(clouds["right"], chamfer_points, ears["right"].seed),
    )
    per_landmark = np.linalg.norm(lms["left"] - lms["right"], axis=1)
    metrics = SubjectMetrics(
        chamfer=chamfer,
        landmark_mean=float(per_landmark.mean()),
        contours=contour_means(per_landmark),
        scale_mm=float(np.mean(scales)),
    )
    return metrics, clouds, lms


@dataclass
class FrameFloor:
    """How far one subject's two canonical frames are from being exact mirrors.

    ``offset_mm`` is the per-axis ``|flip(c_right) - c_left|``, mesh-derived
    (crop bbox centres and scales only; no landmark is involved).

    Read the axes separately, because they do not mean the same thing:

    * **X and Z are pure frame mismatch.** A reflection in Y cannot move a point
      in X or Z, so anything here is the two frozen crop boxes (and the surface
      inside them) failing to be reflections of each other. This part really is
      a floor: a perfectly mirror-symmetric subject still carries it.
    * **Y is an upper bound, not a floor.** It also contains the subject's own
      mid-sagittal offset from ``y = 0``, which is *not* an error at all: a
      subject perfectly mirrored about ``y = y0`` scores exactly 0 on the
      decisive metric, yet shows ``2*y0 / scale`` here. The two cannot be
      separated from the crop centres alone - the midpoint estimator that would
      remove ``y0`` sets this term to zero identically - so Y is reported as
      the ceiling it is.

    Nothing here may be subtracted from the decisive metric: a constant frame
    offset and the per-landmark errors combine as vectors, not as scalars. It is
    a reference for reading the metric's LEVEL, and only that.

    ``mirror_axis`` defaults to the pipeline's; pass ``None`` for the no-mirror
    frame separation (simply how far apart the two ears sit).
    """

    offset_mm: np.ndarray            # [3] per axis
    mean_scale: float                # mm per canonical unit
    scale_ratio: float               # left / right
    mirror_axis_used: int | None     # the axis the flip was applied to

    @property
    def canonical(self) -> np.ndarray:
        """``[3]`` per-axis offset in canonical units."""
        return self.offset_mm / self.mean_scale

    @property
    def total(self) -> float:
        """All three axes together, canonical units - an upper bound."""
        return float(np.linalg.norm(self.offset_mm) / self.mean_scale)

    @property
    def off_axis(self) -> float:
        """The axes the mirror cannot touch, canonical units - a true floor."""
        keep = [a for a in range(3) if a != self.mirror_axis_used]
        return float(np.linalg.norm(self.offset_mm[keep]) / self.mean_scale)


def frame_floor(
    ears: dict[str, EarInputs],
    mirror_axis: int | None = _DEFAULT_SETTING.mirror_axis,
) -> FrameFloor:
    """Measure one subject's frame mismatch; see :class:`FrameFloor`."""
    transforms = {
        side: make_transform(ears[side].crop_points, side, side, None)  # no mirror
        for side in SIDES
    }
    flip = np.ones(3)
    if mirror_axis is not None:
        flip[mirror_axis] = -1.0
    offset_mm = np.abs(transforms["right"].centre * flip - transforms["left"].centre)
    return FrameFloor(
        offset_mm=offset_mm,
        mean_scale=0.5 * (transforms["left"].scale + transforms["right"].scale),
        scale_ratio=transforms["left"].scale / transforms["right"].scale,
        mirror_axis_used=mirror_axis,
    )


# --- aggregation ------------------------------------------------------------ #

@dataclass
class ConfigAccumulator:
    """Per-configuration aggregates over subjects."""

    setting: MirrorSetting
    subjects: list[str] = field(default_factory=list)
    metrics: list[SubjectMetrics] = field(default_factory=list)

    def add(self, subject_id: str, m: SubjectMetrics) -> None:
        self.subjects.append(subject_id)
        self.metrics.append(m)

    @property
    def n(self) -> int:
        return len(self.metrics)

    def values(self, attr: str) -> np.ndarray:
        return np.asarray([getattr(m, attr) for m in self.metrics], dtype=np.float64)

    def contour_table(self) -> np.ndarray:
        """``[4]`` mean over subjects of each contour's mean distance."""
        if not self.metrics:
            return np.full(len(CONTOURS), np.nan)
        return np.mean([m.contours for m in self.metrics], axis=0)

    def worst_subjects(self, k: int = WORST_N) -> list[tuple[str, float]]:
        pairs = list(zip(self.subjects, self.values("landmark_mean")))
        return sorted(pairs, key=lambda p: -p[1])[:k]


def aggregate(values: np.ndarray) -> tuple[float, float, float]:
    """``(mean, median, p95)``; NaNs if empty."""
    if values.size == 0:
        return (float("nan"),) * 3
    return (
        float(values.mean()),
        float(np.median(values)),
        float(np.percentile(values, 95)),
    )


def winning_keys(means: dict[str, float]) -> list[str]:
    """Every configuration within :data:`TIE_REL` of the smallest mean.

    B and C are provably tied (mirroring both clouds instead of one is an
    isometry), so "the minimum" is a set, not a single key.
    """
    best = min(means.values())
    tol = TIE_REL * max(abs(best), 1.0)
    return [k for k, v in means.items() if v <= best + tol]


# --- reporting -------------------------------------------------------------- #

def print_metric_table(title: str, accs: dict[str, ConfigAccumulator], attr: str) -> None:
    print(f"\n--- {title} ---")
    print(f"  {'cfg':<4}{'mean':>10}{'median':>10}{'p95':>10}   "
          f"{'mean (mm)':>11}   configuration")
    for key, acc in accs.items():
        mean, median, p95 = aggregate(acc.values(attr))
        mean_mm, _, _ = aggregate(acc.values(f"{attr}_mm"))
        print(f"  {key:<4}{mean:>10.4f}{median:>10.4f}{p95:>10.4f}   "
              f"{mean_mm:>11.2f}   {acc.setting.label}")


def print_frame_floor(floors: list[FrameFloor]) -> None:
    """What a perfectly mirror-symmetric subject would still show, per axis.

    Printed next to the metrics because the decisive metric's *level* cannot be
    read without it. It shifts all three configurations equally, so it never
    changes which one wins. See :class:`FrameFloor` for why the mirror axis is
    an upper bound while the other two are a genuine floor.
    """
    if not floors:
        return
    axis = floors[0].mirror_axis_used
    axis_name = AXIS_NAMES[axis] if axis is not None else "none"
    print(f"\n--- frame floor: how close the two ears' canonical frames are to "
          f"being {axis_name}-mirrors ---")
    print("  |flip(centre_right) - centre_left| per axis, canonical units "
          "(and mm):")
    for a, name in enumerate(AXIS_NAMES):
        canonical = np.asarray([f.canonical[a] for f in floors])
        millimetres = np.asarray([f.offset_mm[a] for f in floors])
        mean, median, p95 = aggregate(canonical)
        mean_mm, _, _ = aggregate(millimetres)
        tag = "  <- upper bound, not a floor" if a == axis else ""
        print(f"    {name}   mean {mean:.4f}   median {median:.4f}   "
              f"p95 {p95:.4f}   ({mean_mm:.2f} mm){tag}")
    for label, values in (
        ("true floor (axes off the mirror)",
         np.asarray([f.off_axis for f in floors])),
        ("all three axes (upper bound)", np.asarray([f.total for f in floors])),
        ("left/right scale ratio", np.asarray([f.scale_ratio for f in floors])),
    ):
        mean, median, p95 = aggregate(values)
        print(f"  {label:<34}mean {mean:.4f}   median {median:.4f}   "
              f"p95 {p95:.4f}")
    off_axis_names = ", ".join(n for i, n in enumerate(AXIS_NAMES) if i != axis)
    print(f"  Only the axes OFF the mirror ({off_axis_names}) are pure frame "
          f"mismatch: a reflection in {axis_name}\n  cannot move a point in them, so a "
          "perfectly mirror-symmetric subject still\n  carries that much. The "
          f"{axis_name} term also contains the subject's own mid-sagittal\n  "
          f"offset from {axis_name} = 0, which is NOT an error - a subject "
          f"mirrored about {axis_name} = y0 scores\n  exactly 0 on the decisive "
          "metric yet shows 2*y0/scale here - so read it as a\n  ceiling. None "
          "of these may be SUBTRACTED from the decisive metric: a constant\n  "
          "frame offset and the per-landmark errors combine as vectors, not as "
          "scalars.\n  They are a reference for the metric's LEVEL; the "
          "comparison BETWEEN\n  configurations is unaffected (all three share "
          "one crop, centre, scale and\n  point draw per ear).")


def print_verdict(accs: dict[str, ConfigAccumulator]) -> str:
    """Print the verdict block; returns the key of the configuration reported."""
    lm_means = {k: aggregate(a.values("landmark_mean"))[0] for k, a in accs.items()}
    ch_means = {k: aggregate(a.values("chamfer"))[0] for k, a in accs.items()}
    lm_winners = winning_keys(lm_means)
    ch_winners = winning_keys(ch_means)
    best = lm_means[lm_winners[0]]

    ordered = sorted(lm_means, key=lambda k: lm_means[k])
    runner_up = next((k for k in ordered if k not in lm_winners), None)
    # With a tie, report the tied configuration the pipeline actually uses.
    winner = DEFAULT_KEY if DEFAULT_KEY in lm_winners else lm_winners[0]
    joined = " = ".join(lm_winners)

    print("\n=== MIRROR VERDICT ===")
    print("  decisive metric - mean index-matched landmark distance "
          "(canonical units):")
    print(f"    winner: {joined}  ({best:.4f})")
    if runner_up is not None:
        margin = lm_means[runner_up] - best
        # Guard the ratio, not just a literal zero: a winner at 1e-17 would
        # otherwise print a 16-digit "times worse" that means nothing.
        ratio = (f"{lm_means[runner_up] / best:.2f}x worse" if best > ZERO_EPS
                 else "the winner is ~0")
        print(f"    runner-up: {runner_up} ({lm_means[runner_up]:.4f}) - margin "
              f"{margin:.4f} canonical units ({ratio})")
    else:
        print("    runner-up: none - every configuration tied")
    print(f"  Chamfer metric winner: {' = '.join(ch_winners)} "
          f"({ch_means[ch_winners[0]]:.4f})")

    if set(ch_winners) != set(lm_winners):
        print("  !! THE TWO METRICS DISAGREE. The landmark metric is decisive "
              "(Chamfer cannot see a correspondence error), but a disagreement "
              "means the two ears overlap as shapes without their landmarks "
              "lining up - read the per-contour table and the plot before "
              "trusting either.")
    if len(lm_winners) > 1:
        print(f"  ({joined} are tied by construction: flipping Y on both clouds "
              "instead of one is an isometry, so no distance metric can separate "
              "them. The finding is mirror vs no mirror; picking between the tied "
              "ones is a convention.)")

    print(f"\n  per-contour mean landmark distance for {joined} "
          "(canonical units):")
    contours = accs[winner].contour_table()
    for (name, start, stop), value in zip(CONTOURS, contours):
        print(f"    {name:<20} [{start:>2}-{stop - 1:<2}]  {value:.4f}")
    spread = float(np.nanmax(contours) - np.nanmin(contours))
    print(f"    spread across contours: {spread:.4f} - one contour far above the "
          "others would mean that contour is ordered differently on the two "
          "sides, not that the mirror is wrong.")

    worst = accs[winner].worst_subjects()
    if worst:
        print(f"\n  worst {len(worst)} subjects under {winner} "
              "(mean landmark distance, canonical units):")
        print("    " + ", ".join(f"{sid} {value:.3f}" for sid, value in worst))

    if DEFAULT_KEY in lm_winners:
        print(f"\n  => The current default ({DEFAULT_KEY}: mirror_side='right', "
              "mirror_axis=1) is among the winners. No change indicated.")
    else:
        print("\n  " + "!" * 68)
        print("  !! THE CURRENT DEFAULT IS NOT THE WINNER.")
        print(f"  !! default {DEFAULT_KEY} ({lm_means[DEFAULT_KEY]:.4f}) loses to "
              f"{joined} ({best:.4f}) on the decisive metric, by "
              f"{lm_means[DEFAULT_KEY] - best:.4f} canonical units.")
        print("  !! This script changes NOTHING - canonicalize_ear, crop.yaml and "
              "the cache are untouched.")
        print("  !! Alfred decides whether to switch mirror_side/mirror_axis; a "
              "switch means rebuilding the whole cache.")
        print("  " + "!" * 68)

    return winner


# --- plot ------------------------------------------------------------------- #

def write_mirror_plot(
    out_path: Path,
    subject_id: str,
    setting: MirrorSetting,
    clouds: dict[str, np.ndarray],
    landmarks: dict[str, np.ndarray],
    metrics: SubjectMetrics,
) -> Path | None:
    """Canonical left (blue) vs canonical right (red), from two viewing angles.

    Returns the path written, or ``None`` if the figure could not be produced -
    a missing backend, an unwritable ``--out`` or any other drawing failure must
    not sink a verdict that took a full pass over the dataset to compute, so
    everything here is inside one try and reported as a warning.
    """
    try:
        return _draw_mirror_plot(out_path, subject_id, setting, clouds,
                                 landmarks, metrics)
    except Exception as exc:
        print(f"  !! plot skipped ({type(exc).__name__}: {exc}); the numbers "
              "above are unaffected")
        return None


def _draw_mirror_plot(
    out_path: Path,
    subject_id: str,
    setting: MirrorSetting,
    clouds: dict[str, np.ndarray],
    landmarks: dict[str, np.ndarray],
    metrics: SubjectMetrics,
) -> Path:
    """The figure itself; see :func:`write_mirror_plot`, which guards it."""
    import matplotlib

    matplotlib.use("Agg")  # headless: this script only ever writes files
    import matplotlib.pyplot as plt

    colours = {"left": "tab:blue", "right": "tab:red"}
    lm_colours = {"left": "navy", "right": "darkred"}
    markers = {"left": "o", "right": "^"}
    views = ((20, -60), (20, 30))

    figure = plt.figure(figsize=(16, 8))
    everything = np.vstack([*clouds.values(), *landmarks.values()])
    lo, hi = everything.min(axis=0), everything.max(axis=0)
    centre = 0.5 * (lo + hi)
    radius = float((hi - lo).max()) / 2.0 or 1.0

    for i, (elev, azim) in enumerate(views, 1):
        ax = figure.add_subplot(1, len(views), i, projection="3d")
        for side in SIDES:
            ax.scatter(clouds[side][:, 0], clouds[side][:, 1], clouds[side][:, 2],
                       s=2, c=colours[side], alpha=0.25, linewidths=0,
                       depthshade=False, label=f"{side} points")
            ax.scatter(landmarks[side][:, 0], landmarks[side][:, 1],
                       landmarks[side][:, 2], s=22, c=lm_colours[side],
                       marker=markers[side], depthshade=False,
                       label=f"{side} landmarks")
        # Equal data aspect, so an ear is not silently stretched.
        ax.set_xlim(centre[0] - radius, centre[0] + radius)
        ax.set_ylim(centre[1] - radius, centre[1] + radius)
        ax.set_zlim(centre[2] - radius, centre[2] + radius)
        ax.set_box_aspect((1.0, 1.0, 1.0))
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("canonical X")
        ax.set_ylabel("canonical Y")
        ax.set_zlabel("canonical Z")
        ax.set_title(f"elev {elev}, azim {azim}")
        if i == 1:
            ax.legend(loc="upper left", fontsize=7, markerscale=2.0)

    figure.suptitle(
        f"{subject_id} - canonical left vs right, configuration {setting.key}: "
        f"{setting.label}\nmean landmark distance {metrics.landmark_mean:.4f} "
        f"(~{metrics.landmark_mean_mm:.2f} mm), Chamfer {metrics.chamfer:.4f} "
        f"(~{metrics.chamfer_mm:.2f} mm)"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return out_path


# --- CLI -------------------------------------------------------------------- #

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
    ap.add_argument("--out", type=Path, default=Path("outputs"),
                    help=f"directory for {PLOT_NAME} (default: outputs/)")
    ap.add_argument("--crop-config", type=Path, default=Path(DEFAULT_CROP_CONFIG),
                    help="frozen crop config (default: configs/crop.yaml)")
    ap.add_argument("--plot-subject", default=None,
                    help="subject to plot (default: the first one compared)")
    ap.add_argument("--n-points", type=int, default=N_POINTS,
                    help=f"canonical points per ear (default {N_POINTS}, the "
                         "pipeline's value)")
    ap.add_argument("--seed", type=int, default=0,
                    help="base sampling seed, as in preprocess.py (default 0)")
    ap.add_argument("--no-plot", action="store_true",
                    help="skip the figure, print the numbers only")
    ap.add_argument("--allow-failures", action="store_true",
                    help="still report subjects that are missing or fail to load, but "
                         "let the verdict stand on the subjects that were compared; "
                         "without it any unchecked subject fails the run")
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
    print(f"subjects to check: {len(subjects)}"
          + (f"  (--limit {args.limit})" if args.limit > 0 else ""))
    print("configurations compared (each rebuilt from the mesh with the frozen "
          "crop config; the cache is never read):")
    for setting in CONFIGS:
        print(f"  {setting.key}: mirror_side={setting.mirror_side!r} "
              f"mirror_axis={setting.mirror_axis}  - {setting.label}")
    print(f"metrics: symmetric Chamfer on a deterministic {CHAMFER_POINTS}-point "
          f"subsample of each ear's {args.n_points} canonical points, and the mean "
          "index-matched distance over the 85 GT landmarks (decisive). Canonical "
          "units; the mm column is canonical x the mean of the two ears' scales.")
    if missing:
        print(f"  !! {len(missing)} listed subject(s) have no mesh under {root}: "
              f"{missing[:MAX_IDS_LISTED]}")
    if not subjects:
        print("!! no subjects found")
        return 1

    accs = {c.key: ConfigAccumulator(c) for c in CONFIGS}
    # A subject that never gets compared is a failure, not a footnote: a wrong
    # --root or a stale subject list could otherwise compare 2 subjects of 160
    # and still print a verdict.
    failures: list[str] = [f"{sid}: no mesh under {root}" for sid in missing]
    plot_subject = args.plot_subject
    plot_data: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray],
                               SubjectMetrics]] = {}
    floors: list[FrameFloor] = []

    for i, sid in enumerate(subjects, 1):
        try:
            raw = load_mesh(mesh_path(sid, root))
            landmarks = load_subject_landmarks(sid, root)
        except Exception as exc:
            failures.append(f"{sid}: load failed: {type(exc).__name__}: {exc}")
            continue
        try:
            ears = {
                side: prepare_ear(raw, side, configs[side], landmarks[side],
                                  args.n_points, args.seed)
                for side in SIDES
            }
        except Exception as exc:
            failures.append(f"{sid}: {type(exc).__name__}: {exc}")
            continue

        floors.append(frame_floor(ears))
        keep = (sid == plot_subject) or (plot_subject is None and not plot_data)
        for setting in CONFIGS:
            metrics, clouds, lms = compare_ears(ears, setting)
            accs[setting.key].add(sid, metrics)
            if keep:
                plot_data[setting.key] = (clouds, lms, metrics)
        if keep and plot_subject is None:
            plot_subject = sid
        if i % PROGRESS_EVERY == 0:
            print(f"  ... {i}/{len(subjects)}", flush=True)

    compared = accs[DEFAULT_KEY].n
    print(f"\nsubjects compared: {compared}")
    if compared == 0:
        print("!! nothing was compared - no verdict.")
        if failures:
            print(f"!! {len(failures)} subject(s) could not be checked:")
            for f in failures[:MAX_IDS_LISTED]:
                print(f"   {f}")
        return 1

    print_metric_table("metric 1: symmetric Chamfer (canonical units)", accs, "chamfer")
    print_metric_table(
        "metric 2: mean index-matched landmark distance (DECISIVE)",
        accs, "landmark_mean",
    )

    if failures:
        print(f"\n!! {len(failures)} subject(s) could not be checked:")
        for f in failures[:MAX_IDS_LISTED]:
            print(f"   {f}")
        if len(failures) > MAX_IDS_LISTED:
            print(f"   ... and {len(failures) - MAX_IDS_LISTED} more")

    print_frame_floor(floors)

    # The verdict goes out BEFORE the figure: a full pass over the dataset has
    # just been paid for, and a plotting failure must not take the numbers with
    # it. The plot then shows whichever configuration the verdict picked.
    winner = print_verdict(accs)

    if args.no_plot:
        pass
    elif not plot_data:
        print(f"\n  !! plot skipped: subject {args.plot_subject!r} was not among "
              "the subjects compared")
    else:
        clouds, lms, metrics = plot_data[winner]
        plotted = write_mirror_plot(
            Path(args.out).expanduser() / PLOT_NAME, str(plot_subject),
            accs[winner].setting, clouds, lms, metrics,
        )
        if plotted is not None:
            print(f"\n  plot ({plot_subject}, configuration {winner}): {plotted}")

    if args.limit > 0:
        print(f"\n  NOTE: --limit {args.limit} - this covered only the first "
              f"{len(subjects)} listed subject(s), not the whole list.")
    if failures and not args.allow_failures:
        print("\n  => INCOMPLETE - the verdict above stands only on the subjects "
              "that were compared; some could not be checked at all (see above). "
              "Fix --root or the subject list, or pass --allow-failures.")
        return 1
    if failures:
        print(f"\n  => complete (--allow-failures): {len(failures)} subject(s) "
              "were skipped and ignored.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

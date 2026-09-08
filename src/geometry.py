"""Ear cropping, canonicalisation, sampling and exact inverse transforms — Role A.

Contract owned by Role A::

    raw PLY  ->  canonical ear points/features + exactly invertible transform

Everything here is pure numpy and mesh-derived. Ground-truth landmarks never
enter a transform: they may only be pushed *through* these functions. See the
no-leakage rule in ``CLAUDE.md``.

Canonical transform (frozen semantics)::

    canonical = (p - centre) / scale,  then negate component `mirror_axis`
    original  = negate component `mirror_axis`,  then canonical * scale + centre
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .data import SIDES, RawSubject

# Frozen crop boxes, written by scripts/crop_stats.py. Relative paths are tried
# against the current directory first, then the repo root.
DEFAULT_CROP_CONFIG = "configs/crop.yaml"
_REPO_ROOT = Path(__file__).resolve().parents[1]

# Number of annotated landmarks per ear (four contours: 25 + 30 + 20 + 10).
N_LANDMARKS = 85

# Default number of sampled points per canonical ear.
N_POINTS = 2048

_AXES = (0, 1, 2)


def _as_points(points: Any, name: str = "points") -> np.ndarray:
    """Validate and return an ``[..., 3]`` float64 array."""
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim < 1 or arr.shape[-1] != 3:
        raise ValueError(f"{name} must have last dimension 3, got shape {arr.shape}")
    return arr


def _check_side(side: str, name: str = "side") -> str:
    if side not in SIDES:
        raise ValueError(f"{name} must be one of {SIDES}, got {side!r}")
    return side


def _check_mirror_axis(mirror_axis: int | None) -> int | None:
    if mirror_axis is None:
        return None
    if mirror_axis not in _AXES:
        raise ValueError(f"mirror_axis must be one of {_AXES} or None, got {mirror_axis!r}")
    return int(mirror_axis)


@dataclass
class EarTransform:
    """Mesh-derived, exactly invertible canonicalisation of one ear.

    Attributes
    ----------
    centre : np.ndarray
        ``[3]`` translation removed from the original coordinates.
    scale : float
        Positive isotropic scale divided out.
    mirror_axis : int | None
        Axis (0=X, 1=Y, 2=Z) negated after centring/scaling so both ears share a
        canonical orientation, or ``None`` for the side that is not mirrored.
    """

    centre: np.ndarray
    scale: float
    mirror_axis: int | None = None

    def __post_init__(self) -> None:
        self.centre = np.asarray(self.centre, dtype=np.float64).reshape(3)
        self.scale = float(self.scale)
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError(f"scale must be finite and > 0, got {self.scale!r}")
        self.mirror_axis = _check_mirror_axis(self.mirror_axis)

    def to_dict(self) -> dict[str, Any]:
        """Plain-Python representation, safe for JSON / npz metadata."""
        return {
            "centre": [float(c) for c in self.centre],
            "scale": float(self.scale),
            "mirror_axis": None if self.mirror_axis is None else int(self.mirror_axis),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EarTransform":
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            centre=np.asarray(d["centre"], dtype=np.float64),
            scale=float(d["scale"]),
            mirror_axis=d.get("mirror_axis"),
        )


@dataclass
class CanonicalEar:
    """One ear in canonical space, plus everything needed to get back."""

    subject_id: str
    side: str
    points: np.ndarray                      # [N, 3] float64
    normals: np.ndarray | None              # [N, 3] float64 or None
    transform: EarTransform
    qa: dict[str, Any] = field(default_factory=dict)


@dataclass
class CropConfig:
    """Frozen axis-aligned crop box for one side, in Huawei's original frame.

    Bounds are estimated once from TRAINING annotation statistics (plus generous
    margins), then frozen: at validation/test time the crop uses the mesh and
    this config only.
    """

    side: str
    lo: np.ndarray                          # [3] inclusive lower bound
    hi: np.ndarray                          # [3] inclusive upper bound
    min_vertices: int = 500

    def __post_init__(self) -> None:
        self.side = _check_side(self.side, "CropConfig.side")
        self.lo = np.asarray(self.lo, dtype=np.float64).reshape(3)
        self.hi = np.asarray(self.hi, dtype=np.float64).reshape(3)
        if not np.all(self.hi > self.lo):
            raise ValueError(
                f"crop bounds must satisfy hi > lo, got lo={self.lo}, hi={self.hi}"
            )
        self.min_vertices = int(self.min_vertices)


def _resolve_config_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    alt = _REPO_ROOT / p
    if not p.is_absolute() and alt.is_file():
        return alt
    raise FileNotFoundError(
        f"crop config {p} not found (also tried {alt}); run scripts/crop_stats.py first"
    )


def load_crop_config(side: str, path: str | Path = DEFAULT_CROP_CONFIG) -> CropConfig:
    """Read one side's frozen crop box from ``configs/crop.yaml``.

    Expected layout (written by ``scripts/crop_stats.py``)::

        sides:
          left:  {lo: [x, y, z], hi: [x, y, z], min_vertices: 500, ...}
          right: {...}
    """
    import yaml  # local import: geometry stays importable without PyYAML

    side = _check_side(side)
    cfg_path = _resolve_config_path(path)
    with open(cfg_path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    try:
        entry = doc["sides"][side]
        lo, hi = entry["lo"], entry["hi"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{cfg_path}: no 'sides.{side}.lo/hi' entry") from exc
    return CropConfig(
        side=side,
        lo=np.asarray(lo, dtype=np.float64),
        hi=np.asarray(hi, dtype=np.float64),
        min_vertices=int(entry.get("min_vertices", CropConfig.min_vertices)),
    )


def transform_points_to_canonical(points: np.ndarray, t: EarTransform) -> np.ndarray:
    """Map original Huawei coordinates into canonical ear space.

    ``(p - centre) / scale``, then negate component ``t.mirror_axis`` if set.
    """
    p = (_as_points(points) - t.centre) / t.scale
    if t.mirror_axis is not None:
        p[..., t.mirror_axis] *= -1.0
    return p


def inverse_transform_points(points: np.ndarray, t: EarTransform) -> np.ndarray:
    """Exact inverse of :func:`transform_points_to_canonical`."""
    p = _as_points(points).copy()
    if t.mirror_axis is not None:
        p[..., t.mirror_axis] *= -1.0
    return p * t.scale + t.centre


def crop_ear(
    raw: RawSubject,
    side: str,
    cfg: CropConfig,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Crop one ear region out of a raw mesh using frozen bounds.

    Mesh plus frozen config only — no ground truth.

    Returns
    -------
    points : np.ndarray
        ``[M, 3]`` vertices inside the box.
    normals : np.ndarray | None
        ``[M, 3]`` matching normals, or ``None`` if the mesh carries none.
    qa : dict
        ``n_vertices``, ``bbox_lo``, ``bbox_hi``, ``centre``, ``suspicious``.
    """
    side = _check_side(side)
    if cfg.side != side:
        raise ValueError(
            f"CropConfig is for side {cfg.side!r} but crop_ear was called with {side!r}"
        )

    vertices = _as_points(raw.vertices, "raw.vertices")
    if vertices.ndim != 2:
        raise ValueError(f"raw.vertices must be [V,3], got shape {vertices.shape}")

    inside = np.all((vertices >= cfg.lo) & (vertices <= cfg.hi), axis=1)
    points = vertices[inside]

    normals = None
    if raw.normals is not None:
        all_normals = _as_points(raw.normals, "raw.normals")
        if all_normals.shape != vertices.shape:
            raise ValueError(
                f"raw.normals shape {all_normals.shape} does not match "
                f"vertices {vertices.shape}"
            )
        normals = all_normals[inside]

    n_vertices = int(points.shape[0])
    if n_vertices > 0:
        bbox_lo = points.min(axis=0)
        bbox_hi = points.max(axis=0)
        centre = 0.5 * (bbox_lo + bbox_hi)
    else:
        bbox_lo = np.full(3, np.nan)
        bbox_hi = np.full(3, np.nan)
        centre = np.full(3, np.nan)

    qa: dict[str, Any] = {
        "subject_id": raw.subject_id,
        "side": side,
        "n_vertices": n_vertices,
        "bbox_lo": bbox_lo,
        "bbox_hi": bbox_hi,
        "centre": centre,
        "suspicious": bool(n_vertices < cfg.min_vertices),
    }
    return points, normals, qa


def make_transform(
    crop_points: np.ndarray,
    side: str,
    mirror_side: str,
    mirror_axis: int = 1,
) -> EarTransform:
    """Derive the canonical transform from cropped mesh points alone.

    Centre is the crop bounding-box centre; scale is half the largest bbox
    extent, so the canonical ear roughly spans ``[-1, 1]`` on its longest axis.
    The side equal to ``mirror_side`` gets ``mirror_axis`` negated, so both ears
    end up in a shared canonical orientation.

    Huawei's Y axis runs left ear canal to right ear canal, hence the default
    ``mirror_axis=1`` — but this must be confirmed visually on real data and
    recorded in ``DATA_SPEC.md`` before it is trusted.
    """
    side = _check_side(side)
    mirror_side = _check_side(mirror_side, "mirror_side")
    mirror_axis = _check_mirror_axis(mirror_axis)

    pts = _as_points(crop_points, "crop_points")
    if pts.ndim != 2 or pts.shape[0] == 0:
        raise ValueError(
            f"crop_points must be a non-empty [M,3] array, got shape {pts.shape}"
        )

    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    centre = 0.5 * (lo + hi)
    scale = float(np.max(hi - lo) / 2.0)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"degenerate crop: half-extent {scale} is not positive")

    return EarTransform(
        centre=centre,
        scale=scale,
        mirror_axis=mirror_axis if side == mirror_side else None,
    )


def sample_points(
    points: np.ndarray,
    normals: np.ndarray | None,
    n: int = N_POINTS,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Draw exactly ``n`` points (and matching normals) with a deterministic RNG.

    Sampling is without replacement when enough points exist, and with
    replacement otherwise, so the output is always ``[n, 3]``.
    """
    pts = _as_points(points, "points")
    if pts.ndim != 2 or pts.shape[0] == 0:
        raise ValueError(f"points must be a non-empty [M,3] array, got shape {pts.shape}")
    n = int(n)
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")

    m = pts.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(m, size=n, replace=m < n)

    sampled_normals = None
    if normals is not None:
        nrm = _as_points(normals, "normals")
        if nrm.shape != pts.shape:
            raise ValueError(f"normals shape {nrm.shape} does not match points {pts.shape}")
        sampled_normals = nrm[idx]

    return pts[idx], sampled_normals


def canonicalize_ear(
    raw: RawSubject,
    side: str,
    cfg: CropConfig,
    mirror_side: str = "right",
    n_points: int = N_POINTS,
    seed: int = 0,
    mirror_axis: int = 1,
) -> CanonicalEar:
    """Full Role A pipeline for one ear: crop, transform, sample, canonicalise.

    Uses the mesh and the frozen ``cfg`` only; never annotations.
    """
    side = _check_side(side)
    crop_points, crop_normals, qa = crop_ear(raw, side, cfg)
    if crop_points.shape[0] == 0:
        raise ValueError(
            f"empty crop for subject {raw.subject_id!r} side {side!r}: "
            "the frozen crop box contains no vertices"
        )

    transform = make_transform(crop_points, side, mirror_side, mirror_axis)
    sampled, sampled_normals = sample_points(crop_points, crop_normals, n_points, seed)

    points = transform_points_to_canonical(sampled, transform)

    normals = None
    if sampled_normals is not None:
        normals = sampled_normals.copy()
        if transform.mirror_axis is not None:
            # Normals are directions: translation and positive scaling leave them
            # unchanged, but the mirror flips that component.
            normals[:, transform.mirror_axis] *= -1.0

    qa = dict(qa)
    qa.update(
        {
            "n_points": int(n_points),
            "sampled_with_replacement": bool(crop_points.shape[0] < n_points),
            "seed": int(seed),
            "scale": float(transform.scale),
            "mirror_axis": transform.mirror_axis,
        }
    )

    return CanonicalEar(
        subject_id=raw.subject_id,
        side=side,
        points=points,
        normals=normals,
        transform=transform,
        qa=qa,
    )

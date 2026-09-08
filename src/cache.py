"""Processed-cache I/O — Role A.

The one place that knows the layout of ``cache/<subject_id>_<side>.npz``.
Roles B and C call :func:`load_cached_ear` / :func:`list_cached` and never touch
npz keys themselves; ``scripts/preprocess.py`` writes the files through
:func:`write_cached_ear`, so writer and reader cannot drift apart.

Schema ``"1"`` (every key is a plain numpy array, no pickled objects)::

    points                 float32 [n_points, 3]   canonical-frame sample
    transform_centre       float64 [3]             EarTransform.centre
    transform_scale        float64 scalar          EarTransform.scale
    transform_mirror_axis  int64 scalar            EarTransform.mirror_axis, -1 for None
    subject_id             str                     e.g. "P0001"
    side                   str                     "left" | "right"
    n_crop_vertices        int64 scalar            vertices inside the frozen crop box
    qa_suspicious          bool scalar             n_crop_vertices < CropConfig.min_vertices
    n_points               int64 scalar            == points.shape[0]
    seed                   int64 scalar            the run's --seed
    ear_seed               int64 scalar            per-ear sampling seed (from seed, subject, side)
    crop_config_sha256     str                     sha256 of the configs/crop.yaml used
    split_file             str                     the --subject-list path the run was given
    split_sha256           str                     sha256 of that subject list
    schema_version         str                     "1"
    targets                float32 [85, 3]         canonical GT landmarks — ONLY with --with-targets

The transform is stored as float64 exactly as :meth:`EarTransform.to_dict`
holds it, so ``from_dict`` rebuilds it bit-for-bit and
``inverse_transform_points`` on a prediction is as exact as it is on the
in-memory transform. ``points`` and ``targets`` are float32 (model inputs and
training labels only); the cache never round-trips a *stored* target back to
millimetres, that guarantee is about the transform.

Files are written atomically (temp file + ``os.replace``), so a crash can
never leave a half-written npz that a later run would skip as "existing".
The cache is NDA-protected and stays out of git (``.gitignore``).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from .data import SIDES
from .geometry import N_LANDMARKS, CanonicalEar, EarTransform

SCHEMA_VERSION = "1"
CACHE_SUFFIX = ".npz"

# Stored value of ``transform_mirror_axis`` when ``EarTransform.mirror_axis`` is None.
NO_MIRROR = -1

# ``<subject_id>_<side>`` — the stem of every cache file.
_STEM_RE = re.compile(r"^(?P<subject_id>.+)_(?P<side>left|right)$")

REQUIRED_KEYS = (
    "points",
    "transform_centre",
    "transform_scale",
    "transform_mirror_axis",
    "subject_id",
    "side",
    "n_crop_vertices",
    "qa_suspicious",
    "n_points",
    "seed",
    "ear_seed",
    "crop_config_sha256",
    "split_file",
    "split_sha256",
    "schema_version",
)


def cache_path(cache_dir: str | Path, subject_id: str, side: str) -> Path:
    """``<cache_dir>/<subject_id>_<side>.npz``."""
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    return Path(cache_dir) / f"{subject_id}_{side}{CACHE_SUFFIX}"


def parse_cache_filename(path: str | Path) -> tuple[str, str] | None:
    """``(subject_id, side)`` from a cache filename, or ``None`` if it is not one."""
    p = Path(path)
    if p.suffix != CACHE_SUFFIX:
        return None
    match = _STEM_RE.match(p.stem)
    if match is None:
        return None
    return match.group("subject_id"), match.group("side")


def list_cached(cache_dir: str | Path) -> list[tuple[str, str, Path]]:
    """Every cached ear under ``cache_dir`` as ``(subject_id, side, path)``.

    Sorted by subject then side. Files that are not ``<id>_<side>.npz`` (temp
    files, stray downloads) are ignored, so a partial write can never be listed.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"cache directory {cache_dir} does not exist")
    entries = []
    for p in cache_dir.iterdir():
        if not p.is_file():
            continue
        parsed = parse_cache_filename(p)
        if parsed is not None:
            entries.append((parsed[0], parsed[1], p))
    return sorted(entries, key=lambda e: (e[0], SIDES.index(e[1])))


def transform_to_arrays(t: EarTransform) -> dict[str, np.ndarray]:
    """The ``transform_*`` npz keys for one transform (lossless, float64)."""
    d = t.to_dict()
    return {
        "transform_centre": np.asarray(d["centre"], dtype=np.float64).reshape(3),
        "transform_scale": np.asarray(d["scale"], dtype=np.float64),
        "transform_mirror_axis": np.asarray(
            NO_MIRROR if d["mirror_axis"] is None else int(d["mirror_axis"]), dtype=np.int64
        ),
    }


def transform_from_arrays(arrays: Any) -> EarTransform:
    """Rebuild the transform from the ``transform_*`` keys via ``EarTransform.from_dict``."""
    centre = np.asarray(arrays["transform_centre"], dtype=np.float64)
    if centre.shape != (3,):
        raise ValueError(f"transform_centre must have shape (3,), got {centre.shape}")
    mirror = int(np.asarray(arrays["transform_mirror_axis"]))
    return EarTransform.from_dict(
        {
            "centre": centre,
            "scale": float(np.asarray(arrays["transform_scale"])),
            "mirror_axis": None if mirror == NO_MIRROR else mirror,
        }
    )


def _scalar_str(npz: Any, key: str) -> str:
    value = npz[key]
    if value.ndim != 0 or value.dtype.kind != "U":
        raise ValueError(f"{key} must be a scalar string, got dtype {value.dtype} shape {value.shape}")
    return str(value.item())


def write_cached_ear(
    path: str | Path,
    ear: CanonicalEar,
    *,
    seed: int,
    ear_seed: int,
    crop_config_sha256: str,
    split_file: str,
    split_sha256: str,
    targets: np.ndarray | None = None,
) -> Path:
    """Serialise one canonical ear (schema ``"1"``) atomically to ``path``.

    ``ear.qa`` must carry ``n_vertices`` and ``suspicious`` as ``crop_ear`` /
    ``canonicalize_ear`` set them. ``targets`` are the ``[85, 3]`` canonical GT
    landmarks (train/dev only) and are stored only when given. The caller is
    responsible for having verified them — this function stores, it does not
    judge.
    """
    path = Path(path)
    if ear.side not in SIDES:
        raise ValueError(f"ear.side must be one of {SIDES}, got {ear.side!r}")
    points = np.asarray(ear.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"ear.points must be a non-empty [N,3] array, got shape {points.shape}")
    if not isinstance(ear.transform, EarTransform):
        raise TypeError(f"ear.transform must be an EarTransform, got {type(ear.transform).__name__}")
    try:
        n_crop = int(ear.qa["n_vertices"])
        suspicious = bool(ear.qa["suspicious"])
    except KeyError as exc:
        raise ValueError(f"ear.qa lacks {exc}; pass a CanonicalEar from canonicalize_ear") from exc

    arrays: dict[str, np.ndarray] = {
        "points": points.astype(np.float32),
        **transform_to_arrays(ear.transform),
        "subject_id": np.asarray(str(ear.subject_id)),
        "side": np.asarray(str(ear.side)),
        "n_crop_vertices": np.asarray(n_crop, dtype=np.int64),
        "qa_suspicious": np.asarray(suspicious, dtype=np.bool_),
        "n_points": np.asarray(points.shape[0], dtype=np.int64),
        "seed": np.asarray(int(seed), dtype=np.int64),
        "ear_seed": np.asarray(int(ear_seed), dtype=np.int64),
        "crop_config_sha256": np.asarray(str(crop_config_sha256)),
        "split_file": np.asarray(str(split_file)),
        "split_sha256": np.asarray(str(split_sha256)),
        "schema_version": np.asarray(SCHEMA_VERSION),
    }
    if targets is not None:
        tgt = np.asarray(targets, dtype=np.float64)
        if tgt.shape != (N_LANDMARKS, 3):
            raise ValueError(f"targets must have shape ({N_LANDMARKS}, 3), got {tgt.shape}")
        if not np.all(np.isfinite(tgt)):
            raise ValueError("targets contain non-finite values")
        arrays["targets"] = tgt.astype(np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    # Temp file in the same directory, then an atomic rename: a crash mid-write
    # leaves a ".tmp" that list_cached ignores, never a truncated .npz. The pid
    # keeps parallel workers from ever sharing a temp name.
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as fh:
            # A file object, not a path: np.savez would silently append ".npz"
            # to a path that lacks it, and the temp name deliberately lacks it.
            np.savez(fh, **arrays)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def load_cached_ear(path: str | Path) -> CanonicalEar:
    """Read one cached ear back into a :class:`CanonicalEar`.

    ``points`` come back as float64 ``[n_points, 3]`` (the dataclass contract;
    the values are float32-exact). ``normals`` is always ``None`` — the real
    scans carry none. ``transform`` is rebuilt through
    :meth:`EarTransform.from_dict` and is exactly what the writer held.

    ``qa`` carries ``n_vertices``, ``suspicious``, ``sampled_with_replacement``,
    ``n_points``, ``seed``, ``ear_seed``, ``scale``, ``mirror_axis``,
    ``crop_config_sha256``, ``split_file``, ``split_sha256``,
    ``schema_version`` and, when the file was written with ``--with-targets``,
    ``qa["targets"]``: the ``[85, 3]`` float64 canonical GT landmarks.

    Raises
    ------
    ValueError
        On a wrong schema version, a missing key, a malformed array, or a file
        whose stored ``subject_id``/``side`` disagree with its filename.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as npz:
        keys = set(npz.files)
        missing = [k for k in REQUIRED_KEYS if k not in keys]
        if missing:
            raise ValueError(f"{path}: not a Role A cache file, missing keys {missing}")
        version = _scalar_str(npz, "schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"{path}: cache schema_version {version!r} != {SCHEMA_VERSION!r}; "
                "rebuild the cache with scripts/preprocess.py --overwrite"
            )

        subject_id = _scalar_str(npz, "subject_id")
        side = _scalar_str(npz, "side")
        if side not in SIDES:
            raise ValueError(f"{path}: side must be one of {SIDES}, got {side!r}")
        parsed = parse_cache_filename(path)
        if parsed is not None and parsed != (subject_id, side):
            raise ValueError(
                f"{path}: filename says {parsed} but the file holds "
                f"{(subject_id, side)}; a renamed cache file is a bug, not a feature"
            )

        points = np.asarray(npz["points"], dtype=np.float64)
        n_points = int(npz["n_points"])
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] != n_points:
            raise ValueError(
                f"{path}: points must have shape ({n_points}, 3), got {points.shape}"
            )
        if not np.all(np.isfinite(points)):
            raise ValueError(f"{path}: non-finite points")

        transform = transform_from_arrays(npz)
        n_vertices = int(npz["n_crop_vertices"])
        qa: dict[str, Any] = {
            "subject_id": subject_id,
            "side": side,
            "n_vertices": n_vertices,
            "suspicious": bool(npz["qa_suspicious"]),
            "n_points": n_points,
            "sampled_with_replacement": n_vertices < n_points,
            "seed": int(npz["seed"]),
            "ear_seed": int(npz["ear_seed"]),
            "scale": float(transform.scale),
            "mirror_axis": transform.mirror_axis,
            "crop_config_sha256": _scalar_str(npz, "crop_config_sha256"),
            "split_file": _scalar_str(npz, "split_file"),
            "split_sha256": _scalar_str(npz, "split_sha256"),
            "schema_version": version,
        }
        if "targets" in keys:
            targets = np.asarray(npz["targets"], dtype=np.float64)
            if targets.shape != (N_LANDMARKS, 3):
                raise ValueError(
                    f"{path}: targets must have shape ({N_LANDMARKS}, 3), got {targets.shape}"
                )
            qa["targets"] = targets

    return CanonicalEar(
        subject_id=subject_id,
        side=side,
        points=points,
        normals=None,
        transform=transform,
        qa=qa,
    )

"""Raw Huawei data access — Role A.

Loading of raw meshes and (train/dev only) landmark annotations.

Verified conventions (``DATA_SPEC.md``, 2026-09-08)::

    $HUAWEI_DATA_ROOT/mesh/P<id>.ply
    $HUAWEI_DATA_ROOT/landmarks/P<id>_left_ear_landmarks.csv
    $HUAWEI_DATA_ROOT/landmarks/P<id>_right_ear_landmarks.csv

Subject IDs are the full ``"P" + 4 digits`` string (``"P0001"``), non-contiguous.
Landmark CSVs are 85 lines, no header, one comma per line, whitespace-separated
with variable spacing; ``load_landmarks`` accepts both candidate layouts
(``"x, y z"`` and ``"idx,x y z"``).

Invariant: `load_mesh` must never require annotations — inference sees meshes
only.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

# Dataset root. Override with the HUAWEI_DATA_ROOT environment variable so no
# machine-specific (NDA-protected) path is ever hard-coded in the repository.
DATA_ROOT = Path(os.environ.get("HUAWEI_DATA_ROOT", "data"))

SIDES = ("left", "right")

# Vertex properties a PLY header must declare for us to believe it carries real
# per-vertex normals. Anything else is synthesised, and synthesised normals are
# worse than none (see `load_mesh`).
_NORMAL_PROPERTIES = ("nx", "ny", "nz")

# Sanity bound while scanning for `end_header` in a possibly binary file.
_MAX_HEADER_LINES = 10_000

# Directory layout and filename conventions (verified, see DATA_SPEC.md).
MESH_DIR = "mesh"
LANDMARK_DIR = "landmarks"
MESH_SUFFIX = ".ply"
LANDMARK_NAME = "{subject_id}_{side}_ear_landmarks.csv"

# Subject ID: "P" + 4 zero-padded digits at the start of the filename stem.
_SUBJECT_ID_RE = re.compile(r"^(P\d{4})")

# Landmarks per ear (four contours: 25 + 30 + 20 + 10).
N_LANDMARKS = 85


@dataclass
class RawSubject:
    """One subject's raw mesh, exactly as read from disk.

    Attributes
    ----------
    subject_id : str
        Subject identifier, convention per ``DATA_SPEC.md``.
    vertices : np.ndarray
        ``[V, 3]`` float64 vertex coordinates in Huawei's original frame.
    faces : np.ndarray | None
        ``[F, 3]`` integer triangle indices, or ``None`` if unused/absent.
    normals : np.ndarray | None
        ``[V, 3]`` per-vertex normals, or ``None`` if absent.
    mesh_path : Path
        Path the mesh was read from.
    """

    subject_id: str
    vertices: np.ndarray
    faces: np.ndarray | None
    normals: np.ndarray | None
    mesh_path: Path


def subject_id_from_path(path: str | Path) -> str:
    """Derive a subject ID from a mesh or annotation filename.

    The ID is the ``"P" + 4 digits`` token at the start of the filename stem
    (``P0001.ply`` and ``P0001_left_ear_landmarks.csv`` both give ``"P0001"``).
    Convention for the whole codebase: the full string, never the bare digits.

    Raises
    ------
    ValueError
        If the stem does not start with ``P`` followed by four digits.
    """
    stem = Path(path).stem
    match = _SUBJECT_ID_RE.match(stem)
    if match is None:
        raise ValueError(
            f"cannot derive a subject id from {stem!r}: expected the filename to "
            "start with 'P' + 4 digits (e.g. P0001), see DATA_SPEC.md"
        )
    return match.group(1)


def mesh_path(subject_id: str, root: str | Path = DATA_ROOT) -> Path:
    """``<root>/mesh/<subject_id>.ply``."""
    return Path(root) / MESH_DIR / f"{subject_id}{MESH_SUFFIX}"


def landmark_path(subject_id: str, side: str, root: str | Path = DATA_ROOT) -> Path:
    """``<root>/landmarks/<subject_id>_<side>_ear_landmarks.csv`` (train/dev only)."""
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    return Path(root) / LANDMARK_DIR / LANDMARK_NAME.format(subject_id=subject_id, side=side)


def list_subjects(root: str | Path = DATA_ROOT) -> list[str]:
    """Sorted subject IDs of every ``mesh/P*.ply`` under ``root``.

    Files whose stem is not exactly ``P`` + 4 digits are ignored, so a stray
    file in ``mesh/`` cannot masquerade as a subject.
    """
    mesh_dir = Path(root) / MESH_DIR
    if not mesh_dir.is_dir():
        raise FileNotFoundError(
            f"no {MESH_DIR}/ directory under {Path(root)!s} — is HUAWEI_DATA_ROOT set?"
        )
    ids = {
        p.stem
        for p in mesh_dir.glob(f"P*{MESH_SUFFIX}")
        if p.is_file() and _SUBJECT_ID_RE.fullmatch(p.stem)
    }
    return sorted(ids)


def _read_ply_header(path: Path) -> list[str]:
    """Return the PLY header lines, ``end_header`` included.

    Read as bytes and decoded as ASCII: a PLY header is ASCII even when the
    body is binary.
    """
    lines: list[str] = []
    with open(path, "rb") as fh:
        magic = fh.readline().strip()
        if magic != b"ply":
            raise ValueError(f"{path} is not a PLY file (first line: {magic!r})")
        lines.append("ply")
        for _ in range(_MAX_HEADER_LINES):
            raw = fh.readline()
            if not raw:
                break
            line = raw.decode("ascii", errors="replace").strip()
            lines.append(line)
            if line == "end_header":
                return lines
    raise ValueError(f"{path}: no end_header within {_MAX_HEADER_LINES} lines")


def _header_declares_normals(header_lines: list[str]) -> bool:
    """True iff the header declares nx/ny/nz on the ``vertex`` element."""
    element: str | None = None
    vertex_properties: set[str] = set()
    for line in header_lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element":
            element = parts[1] if len(parts) > 1 else None
        elif parts[0] == "property" and element == "vertex" and len(parts) > 1:
            vertex_properties.add(parts[-1].lower())
    return all(name in vertex_properties for name in _NORMAL_PROPERTIES)


def _normals_from_ply_raw(loaded: object, path: Path, n_vertices: int) -> np.ndarray:
    """Pull the *file's own* nx/ny/nz columns out of trimesh's raw PLY tables.

    Deliberately not ``trimesh.Trimesh.vertex_normals``: that property happily
    synthesises normals from face geometry when the file has none, and a
    synthesised normal silently pretending to be a scanned one is exactly the
    kind of thing that poisons a downstream feature channel.
    """
    metadata = getattr(loaded, "metadata", None) or {}
    raw = metadata.get("_ply_raw")
    if not raw or "vertex" not in raw or "data" not in raw["vertex"]:
        raise ValueError(
            f"{path}: header declares {_NORMAL_PROPERTIES} but trimesh exposed no raw "
            "vertex table to read them from (trimesh version change?)."
        )
    data = raw["vertex"]["data"]
    columns = [np.asarray(data[name], dtype=np.float64).reshape(-1) for name in _NORMAL_PROPERTIES]
    normals = np.column_stack(columns)
    if normals.shape != (n_vertices, 3):
        raise ValueError(
            f"{path}: normals shape {normals.shape} does not match {n_vertices} vertices"
        )
    return normals


def load_mesh(path: str | Path) -> RawSubject:
    """Read one PLY mesh into a :class:`RawSubject`.

    Loaded with ``process=False``, so vertices are neither merged nor reordered:
    vertex ``i`` in the file stays vertex ``i`` here, which is what lets any
    per-vertex annotation or index-based QA stay meaningful.

    Normals are returned only when the PLY header actually declares ``nx``,
    ``ny`` and ``nz``; otherwise ``normals`` is ``None`` rather than a
    face-derived guess.

    Must work without any annotation file: this is the inference-time entry
    point.

    Returns
    -------
    RawSubject
        ``vertices`` ``[V, 3]`` float64, ``faces`` ``[F, 3]`` int64 or ``None``
        when the file carries no faces, ``normals`` ``[V, 3]`` float64 or
        ``None``.
    """
    path = Path(path)
    header = _read_ply_header(path)

    loaded = trimesh.load(path, file_type="ply", process=False)
    if not hasattr(loaded, "vertices"):
        raise ValueError(
            f"{path}: trimesh returned {type(loaded).__name__}, which has no vertices"
        )

    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"{path}: expected [V, 3] vertices, got shape {vertices.shape}")

    faces = getattr(loaded, "faces", None)
    if faces is not None:
        faces = np.asarray(faces, dtype=np.int64)
        if faces.size == 0:
            faces = None
        elif faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"{path}: expected [F, 3] faces, got shape {faces.shape}")

    normals = None
    if _header_declares_normals(header):
        normals = _normals_from_ply_raw(loaded, path, len(vertices))

    return RawSubject(
        subject_id=subject_id_from_path(path),
        vertices=vertices,
        faces=faces,
        normals=normals,
        mesh_path=path,
    )


def parse_landmark_line(line: str) -> tuple[int, np.ndarray]:
    """Parse one landmark CSV line into ``(n_tokens, xyz)``.

    Commas are treated as whitespace, so both candidate layouts work:
    ``"x, y z"`` (3 tokens) and ``"idx,x y z"`` (4 tokens, integer-valued
    index dropped). ``n_tokens`` is the count *after* comma replacement, before
    any index column is dropped — ``scripts/inspect_dataset.py`` histograms it
    to tell the two layouts apart.

    Raises
    ------
    ValueError
        On a non-numeric token, or if anything other than exactly 3 coordinates
        remain.
    """
    tokens = line.replace(",", " ").split()
    n_tokens = len(tokens)
    try:
        values = [float(t) for t in tokens]
    except ValueError as exc:
        raise ValueError(f"non-numeric token in landmark line {line!r}") from exc
    if n_tokens == 4 and values[0].is_integer():
        values = values[1:]  # leading index column
    if len(values) != 3:
        raise ValueError(
            f"expected 3 coordinates (or index + 3) per landmark line, got {n_tokens} "
            f"tokens in {line!r}"
        )
    return n_tokens, np.asarray(values, dtype=np.float64)


def _check_side_in_filename(path: Path, side: str) -> None:
    """Refuse to read a file that is clearly named for the other side."""
    stem = path.stem.lower()
    other = "right" if side == "left" else "left"
    if f"_{other}_" in f"_{stem}_" and f"_{side}_" not in f"_{stem}_":
        raise ValueError(f"{path.name} looks like a {other!r} file but side={side!r} was requested")


def load_landmarks(path: str | Path, side: str) -> np.ndarray:
    """Read one side's ground-truth landmarks — TRAIN/DEV ONLY.

    Text file, 85 non-empty lines, each ``"x, y z"`` or ``"idx,x y z"`` with
    arbitrary whitespace (CRLF fine, no header, no BOM expected but tolerated).
    Every line must use the *same* layout: a file mixing 3-token and 4-token
    lines is rejected, naming the first line that disagrees.

    Returns
    -------
    np.ndarray
        ``[85, 3]`` float64 landmarks in Huawei's original frame.

    Notes
    -----
    Never call this from the inference path, and never let its output influence
    any preprocessing parameter (see the no-leakage rule in ``CLAUDE.md``).

    Raises
    ------
    ValueError
        Naming the offending line number on a malformed line or on the first
        line whose token count disagrees with the rest of the file, or if the
        file does not contain exactly 85 landmark lines.
    """
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    path = Path(path)
    _check_side_in_filename(path, side)

    text = path.read_text(encoding="utf-8-sig")
    rows: list[np.ndarray] = []
    layout_tokens: int | None = None      # 3 ("x, y z") or 4 ("idx,x y z")
    layout_lineno: int | None = None      # where that layout was first seen
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            n_tokens, xyz = parse_landmark_line(line)
        except ValueError as exc:
            raise ValueError(f"{path}: line {lineno}: {exc}") from exc
        if n_tokens not in (3, 4):
            raise ValueError(
                f"{path}: line {lineno}: expected 3 or 4 tokens per landmark line, got {n_tokens}"
            )
        # One file, one layout. A file that mixes "x, y z" and "idx,x y z" lines
        # is not a layout we can trust to mean what it looks like, so refuse it
        # rather than silently guessing per line (see DATA_SPEC.md).
        if layout_tokens is None:
            layout_tokens, layout_lineno = n_tokens, lineno
        elif n_tokens != layout_tokens:
            raise ValueError(
                f"{path}: line {lineno}: inconsistent CSV layout — {n_tokens} tokens here but "
                f"{layout_tokens} on line {layout_lineno}; the whole file must use one layout "
                f"(either 'x, y z' or 'idx,x y z')"
            )
        rows.append(xyz)

    if len(rows) != N_LANDMARKS:
        raise ValueError(
            f"{path}: expected {N_LANDMARKS} landmark lines, found {len(rows)}"
        )
    landmarks = np.vstack(rows)
    if not np.all(np.isfinite(landmarks)):
        raise ValueError(f"{path}: non-finite landmark coordinates")
    return landmarks


def load_subject_landmarks(subject_id: str, root: str | Path = DATA_ROOT) -> dict[str, np.ndarray]:
    """Both sides' landmarks for one subject — TRAIN/DEV ONLY.

    Returns ``{"left": [85,3], "right": [85,3]}``.
    """
    return {side: load_landmarks(landmark_path(subject_id, side, root), side) for side in SIDES}

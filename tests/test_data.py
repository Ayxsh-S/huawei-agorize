"""Raw loader invariants for Role A.

All data here is synthetic and written into ``tmp_path`` by the tests
themselves — no Huawei data is ever read (see ``CLAUDE.md``).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from src.data import (
    RawSubject,
    landmark_path,
    list_subjects,
    load_landmarks,
    load_mesh,
    load_subject_landmarks,
    mesh_path,
    parse_landmark_line,
    subject_id_from_path,
)

# 20 vertices on a coarse grid, deliberately including an exact duplicate
# (index 7 repeats index 0) so a merging loader would be caught by the count.
_VERTICES = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.5],
        [2.0, 1.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [2.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.5],
        [2.0, 1.0, 1.0],
        [0.0, 2.0, 1.0],
        [1.0, 2.0, 1.0],
        [2.0, 2.0, 1.0],
        [0.5, 0.5, 2.0],
        [1.5, 1.5, 2.0],
    ],
    dtype=np.float64,
)

# A few faces. They reference only some vertices: an unreferenced vertex must
# still survive, in its original slot.
_FACES = np.array(
    [
        [0, 1, 4],
        [1, 2, 5],
        [3, 4, 6],
        [9, 10, 13],
        [12, 13, 15],
    ],
    dtype=np.int64,
)


def _unit_normals(vertices: np.ndarray) -> np.ndarray:
    """Deterministic, distinct per-vertex normals — never all equal."""
    directions = vertices - vertices.mean(axis=0) + np.array([0.1, 0.2, 0.3])
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


_NORMALS = _unit_normals(_VERTICES)


def write_ascii_ply(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray | None = None,
    normals: np.ndarray | None = None,
    colours: np.ndarray | None = None,
) -> Path:
    """Write a minimal ASCII PLY. Test-local, so nothing under test writes it."""
    header = ["ply", "format ascii 1.0", f"element vertex {len(vertices)}"]
    header += ["property float x", "property float y", "property float z"]
    if normals is not None:
        header += ["property float nx", "property float ny", "property float nz"]
    if colours is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    if faces is not None:
        header += [f"element face {len(faces)}", "property list uchar int vertex_indices"]
    header.append("end_header")

    body = []
    for i, vertex in enumerate(vertices):
        row = [f"{value:.6f}" for value in vertex]
        if normals is not None:
            row += [f"{value:.6f}" for value in normals[i]]
        if colours is not None:
            row += [str(int(value)) for value in colours[i]]
        body.append(" ".join(row))
    if faces is not None:
        for face in faces:
            body.append("3 " + " ".join(str(int(index)) for index in face))

    path.write_text("\n".join(header + body) + "\n", encoding="ascii")
    return path


def write_binary_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray) -> Path:
    """Little-endian binary PLY — the format the real scans are likely in."""
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    buffer = bytearray(header.encode("ascii"))
    for vertex, normal in zip(vertices, normals):
        buffer += struct.pack("<6f", *vertex, *normal)
    for face in faces:
        buffer += struct.pack("<B3i", 3, *(int(index) for index in face))
    path.write_bytes(bytes(buffer))
    return path


def test_load_mesh_with_faces_and_normals(tmp_path: Path) -> None:
    path = write_ascii_ply(tmp_path / "P0001_head.ply", _VERTICES, _FACES, _NORMALS)

    raw = load_mesh(path)

    assert isinstance(raw, RawSubject)
    assert raw.vertices.shape == (20, 3)
    assert raw.vertices.dtype == np.float64
    assert raw.faces is not None
    assert raw.faces.shape == (5, 3)
    assert np.issubdtype(raw.faces.dtype, np.integer)
    assert np.array_equal(raw.faces, _FACES)
    assert raw.normals is not None
    assert raw.normals.shape == (20, 3)
    assert raw.normals.dtype == np.float64
    assert raw.mesh_path == path


def test_load_mesh_preserves_vertex_order_and_duplicates(tmp_path: Path) -> None:
    """process=False: no merging, no reordering, no dropping."""
    path = write_ascii_ply(tmp_path / "P0002_head.ply", _VERTICES, _FACES, _NORMALS)

    raw = load_mesh(path)

    assert len(raw.vertices) == len(_VERTICES)
    np.testing.assert_allclose(raw.vertices, _VERTICES, atol=1e-6)
    # The duplicate pair survives as two separate vertices ...
    np.testing.assert_allclose(raw.vertices[7], raw.vertices[0], atol=1e-6)
    # ... and vertex 19, referenced by no face, keeps its slot.
    np.testing.assert_allclose(raw.vertices[19], _VERTICES[19], atol=1e-6)
    np.testing.assert_allclose(raw.normals, _NORMALS, atol=1e-6)


def test_load_mesh_without_normals(tmp_path: Path) -> None:
    """No nx/ny/nz in the header means None — not a face-derived guess."""
    path = write_ascii_ply(tmp_path / "P0003_head.ply", _VERTICES, _FACES, normals=None)

    raw = load_mesh(path)

    assert raw.normals is None
    assert raw.faces is not None
    np.testing.assert_allclose(raw.vertices, _VERTICES, atol=1e-6)


def test_load_mesh_without_faces(tmp_path: Path) -> None:
    """A vertex-only PLY loads as a point cloud with faces None."""
    path = write_ascii_ply(tmp_path / "P0004_head.ply", _VERTICES, faces=None, normals=_NORMALS)

    raw = load_mesh(path)

    assert raw.faces is None
    assert raw.vertices.shape == (20, 3)
    assert raw.normals is not None
    np.testing.assert_allclose(raw.normals, _NORMALS, atol=1e-6)


def test_load_mesh_without_faces_or_normals(tmp_path: Path) -> None:
    path = write_ascii_ply(tmp_path / "P0005_head.ply", _VERTICES, faces=None, normals=None)

    raw = load_mesh(path)

    assert raw.faces is None
    assert raw.normals is None
    np.testing.assert_allclose(raw.vertices, _VERTICES, atol=1e-6)


def test_load_mesh_ignores_non_normal_vertex_properties(tmp_path: Path) -> None:
    """Extra vertex properties (e.g. colour) must not be mistaken for normals."""
    colours = np.tile(np.array([10, 20, 30]), (len(_VERTICES), 1))
    path = write_ascii_ply(tmp_path / "P0006_head.ply", _VERTICES, _FACES, normals=None, colours=colours)

    raw = load_mesh(path)

    assert raw.normals is None
    np.testing.assert_allclose(raw.vertices, _VERTICES, atol=1e-6)


def test_load_mesh_binary_ply(tmp_path: Path) -> None:
    path = write_binary_ply(tmp_path / "P0007_head.ply", _VERTICES, _FACES, _NORMALS)

    raw = load_mesh(path)

    assert raw.vertices.shape == (20, 3)
    assert raw.vertices.dtype == np.float64
    np.testing.assert_allclose(raw.vertices, _VERTICES, atol=1e-6)
    assert raw.normals is not None
    np.testing.assert_allclose(raw.normals, _NORMALS, atol=1e-6)
    assert raw.faces is not None
    np.testing.assert_array_equal(raw.faces, _FACES)


def test_load_mesh_rejects_non_ply(tmp_path: Path) -> None:
    path = tmp_path / "not_a_mesh_0001.ply"
    path.write_text("this is not a PLY\n", encoding="ascii")

    with pytest.raises(ValueError):
        load_mesh(path)


def test_load_mesh_accepts_str_path(tmp_path: Path) -> None:
    path = write_ascii_ply(tmp_path / "P0008_head.ply", _VERTICES, _FACES, _NORMALS)

    raw = load_mesh(str(path))

    assert raw.mesh_path == Path(path)
    assert raw.subject_id == "P0008"


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("P0001.ply", "P0001"),
        ("P0308.ply", "P0308"),
        ("P0001_left_ear_landmarks.csv", "P0001"),
        ("P0042_right_ear_landmarks.csv", "P0042"),
        ("mesh/P0123.ply", "P0123"),
        ("P00170_extra.ply", "P0017"),  # only the first four digits form the ID
    ],
)
def test_subject_id_from_path(filename: str, expected: str) -> None:
    assert subject_id_from_path(filename) == expected
    assert subject_id_from_path(Path("some") / "dir" / filename) == expected


@pytest.mark.parametrize("filename", ["head_mesh.ply", "0001.ply", "p0001.ply", "X_P0001.ply", "P001.ply"])
def test_subject_id_from_path_rejects_other_patterns(filename: str) -> None:
    with pytest.raises(ValueError):
        subject_id_from_path(filename)


# --------------------------------------------------------------------------- #
# landmark CSVs — synthetic, both candidate layouts
# --------------------------------------------------------------------------- #

def _synthetic_landmarks(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=[40.0, -70.0, 10.0], scale=[8.0, 4.0, 12.0], size=(85, 3))


def _write_csv_xyz_layout(path: Path, points: np.ndarray, newline: str = "\r\n") -> Path:
    """``x, y z`` — one comma after x, variable spacing, CRLF like the real files."""
    lines = []
    for i, (x, y, z) in enumerate(points):
        gap = " " * (1 + i % 3)
        lines.append(f"{x:.6f},{gap}{y:.6f}{gap}{z:.6f}")
    path.write_bytes((newline.join(lines) + newline).encode("ascii"))
    return path


def _write_csv_index_layout(path: Path, points: np.ndarray, newline: str = "\r\n") -> Path:
    """``idx,x y z`` — leading integer index column."""
    lines = []
    for i, (x, y, z) in enumerate(points):
        gap = " " * (1 + i % 3)
        lines.append(f"{i},{x:.6f}{gap}{y:.6f} {z:.6f}")
    path.write_bytes((newline.join(lines) + newline).encode("ascii"))
    return path


def test_parse_landmark_line_layouts() -> None:
    n, xyz = parse_landmark_line("1.5, 2.5 3.5")
    assert n == 3
    np.testing.assert_allclose(xyz, [1.5, 2.5, 3.5])

    n, xyz = parse_landmark_line("7,   1.5  2.5    3.5")
    assert n == 4
    np.testing.assert_allclose(xyz, [1.5, 2.5, 3.5])

    with pytest.raises(ValueError):
        parse_landmark_line("1.5, 2.5")
    with pytest.raises(ValueError):
        parse_landmark_line("1.5, 2.5 3.5 4.5")  # 4 tokens but no integer index
    with pytest.raises(ValueError):
        parse_landmark_line("a, 2.5 3.5")


def test_load_landmarks_xyz_layout(tmp_path: Path) -> None:
    expected = _synthetic_landmarks(1)
    path = _write_csv_xyz_layout(tmp_path / "P0001_left_ear_landmarks.csv", expected)

    got = load_landmarks(path, "left")

    assert got.shape == (85, 3)
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_load_landmarks_index_layout(tmp_path: Path) -> None:
    expected = _synthetic_landmarks(2)
    path = _write_csv_index_layout(tmp_path / "P0001_right_ear_landmarks.csv", expected)

    got = load_landmarks(path, "right")

    assert got.shape == (85, 3)
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_load_landmarks_lf_and_trailing_blank_lines(tmp_path: Path) -> None:
    expected = _synthetic_landmarks(3)
    path = _write_csv_xyz_layout(tmp_path / "P0002_left_ear_landmarks.csv", expected, newline="\n")
    path.write_bytes(path.read_bytes() + b"\n\n")

    np.testing.assert_allclose(load_landmarks(path, "left"), expected, atol=1e-6)


def test_load_landmarks_wrong_line_count(tmp_path: Path) -> None:
    path = _write_csv_xyz_layout(tmp_path / "P0003_left_ear_landmarks.csv", _synthetic_landmarks()[:84])
    with pytest.raises(ValueError, match="85"):
        load_landmarks(path, "left")

    path = _write_csv_xyz_layout(tmp_path / "P0004_left_ear_landmarks.csv", np.vstack([_synthetic_landmarks(), [[0.0, 0.0, 0.0]]]))
    with pytest.raises(ValueError, match="85"):
        load_landmarks(path, "left")


def test_load_landmarks_names_bad_line(tmp_path: Path) -> None:
    path = _write_csv_xyz_layout(tmp_path / "P0005_left_ear_landmarks.csv", _synthetic_landmarks())
    lines = path.read_bytes().split(b"\r\n")
    lines[41] = b"1.0, 2.0"  # line 42 (1-based) loses a coordinate
    path.write_bytes(b"\r\n".join(lines))

    with pytest.raises(ValueError, match="line 42"):
        load_landmarks(path, "left")


def test_load_landmarks_rejects_side_mismatch(tmp_path: Path) -> None:
    path = _write_csv_xyz_layout(tmp_path / "P0006_left_ear_landmarks.csv", _synthetic_landmarks())
    with pytest.raises(ValueError):
        load_landmarks(path, "right")
    with pytest.raises(ValueError):
        load_landmarks(path, "middle")


# --------------------------------------------------------------------------- #
# directory layout helpers
# --------------------------------------------------------------------------- #

def test_path_helpers() -> None:
    root = Path("R")
    assert mesh_path("P0001", root) == root / "mesh" / "P0001.ply"
    assert landmark_path("P0001", "left", root) == root / "landmarks" / "P0001_left_ear_landmarks.csv"
    assert landmark_path("P0308", "right", root) == root / "landmarks" / "P0308_right_ear_landmarks.csv"
    with pytest.raises(ValueError):
        landmark_path("P0001", "up", root)


def test_list_subjects_and_subject_landmarks(tmp_path: Path) -> None:
    (tmp_path / "mesh").mkdir()
    (tmp_path / "landmarks").mkdir()
    for sid in ("P0308", "P0001", "P0042"):
        write_ascii_ply(mesh_path(sid, tmp_path), _VERTICES, _FACES)
    (tmp_path / "mesh" / "notes.ply").write_text("junk", encoding="ascii")  # ignored
    (tmp_path / "mesh" / "P0001.txt").write_text("junk", encoding="ascii")  # ignored

    assert list_subjects(tmp_path) == ["P0001", "P0042", "P0308"]

    left = _synthetic_landmarks(10)
    right = _synthetic_landmarks(11)
    _write_csv_xyz_layout(landmark_path("P0042", "left", tmp_path), left)
    _write_csv_index_layout(landmark_path("P0042", "right", tmp_path), right)

    got = load_subject_landmarks("P0042", tmp_path)
    assert set(got) == {"left", "right"}
    np.testing.assert_allclose(got["left"], left, atol=1e-6)
    np.testing.assert_allclose(got["right"], right, atol=1e-6)

    raw = load_mesh(mesh_path("P0042", tmp_path))
    assert raw.subject_id == "P0042"


def test_list_subjects_requires_mesh_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list_subjects(tmp_path)

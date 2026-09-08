"""Print METADATA ONLY about the local Huawei dataset (Role A).

Run by Alfred, on his machine, against the NDA-protected data. It prints
aggregate metadata: counts, dtypes, filename patterns and per-axis min/max
envelopes. It NEVER prints the coordinates of an individual point or landmark,
and it NEVER writes a file.

Usage::

    python scripts/inspect_dataset.py                     # uses $HUAWEI_DATA_ROOT
    python scripts/inspect_dataset.py --root "D:/path/to/data"
    python scripts/inspect_dataset.py --max-meshes 0      # 0 = every mesh (slow)

Feed the printed output into DATA_SPEC.md to replace the `?` fields.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `src`

from src.data import load_landmarks, parse_landmark_line, subject_id_from_path  # noqa: E402

MESH_EXTS = {".ply"}
ANN_EXTS = {".csv", ".txt", ".npy", ".json", ".npz"}
SKIP_NAME_HINTS = ("readme", "license", "licence", "notice", "requirements")

AXES = ("X", "Y", "Z")


# --------------------------------------------------------------------------- #
# formatting helpers — every printed number here is an aggregate, never a point
# --------------------------------------------------------------------------- #

def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def fmt(v) -> str:
    try:
        return f"{float(v):.4g}"
    except (TypeError, ValueError):
        return str(v)


def print_axis_ranges(label: str, lo, hi, indent: str = "  ") -> None:
    """Print per-axis [min, max] for a 3-vector pair."""
    print(f"{indent}{label}")
    for i, ax in enumerate(AXES):
        print(f"{indent}  {ax}: [{fmt(lo[i]):>12} , {fmt(hi[i]):>12} ]   extent {fmt(hi[i] - lo[i])}")


def print_stats(label: str, values, indent: str = "  ") -> None:
    """Print min / median / max of a 1-D collection."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        print(f"{indent}{label}: (none)")
        return
    print(
        f"{indent}{label}: min {fmt(arr.min())}  median {fmt(np.median(arr))}  "
        f"max {fmt(arr.max())}   (n={arr.size})"
    )


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #

def is_noise(path: Path) -> bool:
    low = path.name.lower()
    return any(h in low for h in SKIP_NAME_HINTS)


def find_files(root: Path) -> tuple[list[Path], list[Path]]:
    """Return (mesh paths, candidate annotation paths), sorted."""
    meshes: list[Path] = []
    anns: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or is_noise(p):
            continue
        ext = p.suffix.lower()
        if ext in MESH_EXTS:
            meshes.append(p)
        elif ext in ANN_EXTS:
            anns.append(p)
    return meshes, anns


def pattern_of(name: str) -> str:
    """Filename with every digit run replaced by <id>."""
    return re.sub(r"\d+", "<id>", name)


def subject_id_of(name: str) -> str | None:
    """Subject ID per ``src.data`` (``P`` + 4 digits); longest digit run as fallback."""
    try:
        return subject_id_from_path(name)
    except ValueError:
        runs = re.findall(r"\d+", name)
        return max(runs, key=len) if runs else None


def detect_side(name: str) -> str | None:
    low = name.lower()
    if "left" in low:
        return "left"
    if "right" in low:
        return "right"
    tokens = re.split(r"[^a-z0-9]+", low)
    if "l" in tokens:
        return "left"
    if "r" in tokens:
        return "right"
    return None


# --------------------------------------------------------------------------- #
# 1 — directory tree
# --------------------------------------------------------------------------- #

def print_tree(root: Path, max_depth: int = 2, max_entries: int = 10) -> None:
    section("1. DIRECTORY TREE (depth 2, filenames only)")
    print(f"root: {root}")
    if not root.exists():
        print("  !! root does not exist")
        return

    def walk(d: Path, depth: int, prefix: str) -> None:
        try:
            entries = sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            print(f"{prefix}(permission denied)")
            return
        dirs = [e for e in entries if e.is_dir()]
        files = [e for e in entries if e.is_file()]

        for e in dirs[:max_entries]:
            print(f"{prefix}{e.name}/")
            if depth < max_depth:
                walk(e, depth + 1, prefix + "    ")
        if len(dirs) > max_entries:
            print(f"{prefix}... and {len(dirs) - max_entries} more directories")

        for e in files[:max_entries]:
            print(f"{prefix}{e.name}")
        if len(files) > max_entries:
            print(f"{prefix}... and {len(files) - max_entries} more files")

        by_ext = Counter(e.suffix.lower() or "(no ext)" for e in files)
        if by_ext:
            summary = ", ".join(f"{n}x {ext}" for ext, n in sorted(by_ext.items()))
            print(f"{prefix}[{len(files)} files: {summary}]")

    walk(root, 1, "  ")


# --------------------------------------------------------------------------- #
# 2 — filename patterns
# --------------------------------------------------------------------------- #

def print_patterns(meshes: list[Path], anns: list[Path]) -> None:
    section("2. FILENAME PATTERNS (digit runs shown as <id>)")

    def report(label: str, paths: list[Path]) -> set[str]:
        print(f"\n{label}: {len(paths)} files")
        if not paths:
            return set()
        pats = Counter(pattern_of(p.name) for p in paths)
        for pat, n in pats.most_common(5):
            print(f"  {n:5d}x  {pat}")
        if len(pats) > 5:
            print(f"  ... and {len(pats) - 5} more distinct patterns")
        ids = {sid for p in paths if (sid := subject_id_of(p.name))}
        print(f"  distinct subject IDs extracted: {len(ids)}")
        if ids:
            widths = Counter(len(i) for i in ids)
            print(f"  ID digit widths: {dict(sorted(widths.items()))}")
            ordered = sorted(ids)
            print(f"  first ID: {ordered[0]}   last ID: {ordered[-1]}")
        return ids

    mesh_ids = report("MESHES (.ply)", meshes)
    ann_ids = report("ANNOTATION CANDIDATES", anns)

    sides = Counter(detect_side(p.name) for p in anns)
    print(f"\n  annotation side detection: {dict(sides)}")

    if mesh_ids and ann_ids:
        print(f"\n  mesh IDs without annotations: {len(mesh_ids - ann_ids)}")
        print(f"  annotation IDs without a mesh: {len(ann_ids - mesh_ids)}")


# --------------------------------------------------------------------------- #
# meshes
# --------------------------------------------------------------------------- #

def ply_header_info(path: Path) -> dict:
    """Parse the PLY header only. Reads no vertex data."""
    info: dict = {"format": "?", "has_normals": False, "vertex_props": [], "declared_dtype": "?"}
    try:
        with open(path, "rb") as fh:
            raw = fh.read(8192)
        text = raw.split(b"end_header")[0].decode("ascii", errors="replace")
    except OSError as exc:
        info["error"] = str(exc)
        return info

    in_vertex = False
    for line in text.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "format" and len(parts) >= 2:
            info["format"] = " ".join(parts[1:])
        elif parts[0] == "element":
            in_vertex = len(parts) >= 2 and parts[1] == "vertex"
        elif parts[0] == "property" and in_vertex and len(parts) >= 3:
            info["vertex_props"].append((parts[-2], parts[-1]))

    names = [n for _, n in info["vertex_props"]]
    info["has_normals"] = all(n in names for n in ("nx", "ny", "nz"))
    xyz = [t for t, n in info["vertex_props"] if n in ("x", "y", "z")]
    if xyz:
        info["declared_dtype"] = xyz[0]
    return info


def as_mesh(obj):
    """Collapse a trimesh Scene to a single geometry."""
    import trimesh

    if isinstance(obj, trimesh.Scene):
        geoms = list(obj.geometry.values())
        if not geoms:
            return None
        return geoms[0] if len(geoms) == 1 else trimesh.util.concatenate(geoms)
    return obj


def load_mesh_quiet(path: Path):
    import trimesh

    return as_mesh(trimesh.load(path, process=False))


def print_first_mesh(meshes: list[Path]) -> None:
    section("3. FIRST MESH (trimesh)")
    if not meshes:
        print("  no .ply files found")
        return

    path = meshes[0]
    print(f"  file: {path.name}")

    hdr = ply_header_info(path)
    print(f"  PLY header format      : {hdr['format']}")
    print(f"  PLY vertex properties  : {[n for _, n in hdr['vertex_props']]}")
    print(f"  normals stored in file : {hdr['has_normals']}")
    print(f"  declared vertex dtype  : {hdr['declared_dtype']}")

    try:
        mesh = load_mesh_quiet(path)
    except Exception as exc:
        print(f"  !! trimesh failed to load: {type(exc).__name__}: {exc}")
        return
    if mesh is None:
        print("  !! trimesh returned an empty scene")
        return

    v = np.asarray(mesh.vertices)
    faces = getattr(mesh, "faces", None)
    n_faces = 0 if faces is None else int(len(faces))
    print(f"  vertex count           : {len(v)}")
    print(f"  face count             : {n_faces}")
    print(f"  vertices dtype (loaded): {v.dtype}")
    print(f"  trimesh type           : {type(mesh).__name__}")
    if len(v):
        print_axis_ranges("vertex per-axis min/max:", v.min(axis=0), v.max(axis=0))


def print_mesh_envelope(meshes: list[Path], limit: int) -> None:
    section(f"4. MESH ENVELOPE ACROSS SUBJECTS (limit={limit or 'all'})")
    if not meshes:
        print("  no .ply files found")
        return

    subset = meshes if limit in (0, None) else meshes[:limit]
    print(f"  scanning {len(subset)} of {len(meshes)} meshes ...")

    counts: list[int] = []
    face_counts: list[int] = []
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    normals_flags = Counter()
    failed = 0

    for i, path in enumerate(subset, 1):
        normals_flags[ply_header_info(path)["has_normals"]] += 1
        try:
            mesh = load_mesh_quiet(path)
            v = np.asarray(mesh.vertices)
        except Exception:
            failed += 1
            continue
        if not len(v):
            failed += 1
            continue
        counts.append(len(v))
        f = getattr(mesh, "faces", None)
        face_counts.append(0 if f is None else int(len(f)))
        lo = np.minimum(lo, v.min(axis=0))
        hi = np.maximum(hi, v.max(axis=0))
        if i % 25 == 0:
            print(f"    ... {i}/{len(subset)}", flush=True)

    print(f"  meshes loaded ok: {len(counts)}   failed: {failed}")
    print(f"  normals stored in file: {dict(normals_flags)}")
    print_stats("vertex count", counts)
    print_stats("face count", face_counts)
    if np.isfinite(lo).all():
        print_axis_ranges("global vertex envelope (min/max over scanned meshes):", lo, hi)


# --------------------------------------------------------------------------- #
# annotations
# --------------------------------------------------------------------------- #

def load_annotation(path: Path, side: str) -> tuple[np.ndarray | None, str]:
    """Parse with the real loader, ``src.data.load_landmarks``. Returns (array, note)."""
    try:
        return load_landmarks(path, side), "src.data.load_landmarks"
    except Exception as exc:  # report, never crash the inspection
        return None, f"load_landmarks failed: {type(exc).__name__}: {exc}"


def token_histogram(path: Path) -> tuple[Counter, int, int]:
    """Per-line token counts (after comma->space) plus comma/BOM facts. No coordinates."""
    raw = path.read_bytes()
    has_bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig", errors="replace")
    counts: Counter = Counter()
    commas: Counter = Counter()
    for line in text.splitlines():
        if not line.strip():
            continue
        commas[line.count(",")] += 1
        try:
            n_tokens, _ = parse_landmark_line(line)
        except ValueError:
            n_tokens = -1  # unparseable
        counts[n_tokens] += 1
    crlf = raw.count(b"\r\n")
    print(f"    BOM       : {has_bom}   CRLF line endings: {crlf}")
    print(f"    commas per line : {dict(sorted(commas.items()))}")
    print(f"    tokens per line : {dict(sorted(counts.items()))}   (-1 = unparseable)")
    layout = {3: "'x, y z'  (3 numbers, no index column)", 4: "'idx,x y z'  (leading integer index)"}
    for n, name in layout.items():
        if counts.get(n):
            print(f"      -> {counts[n]} line(s) match layout {name}")
    return counts, crlf, int(has_bom)


def print_first_subject_annotations(by_subject: dict[str, dict[str, Path]]) -> None:
    section("5. FIRST SUBJECT'S ANNOTATIONS (parsed by src.data.load_landmarks)")
    if not by_subject:
        print("  no annotation files matched a subject ID + side")
        return

    sid = sorted(by_subject)[0]
    print(f"  subject ID: {sid}")
    first = True
    for side in ("left", "right"):
        path = by_subject[sid].get(side)
        print(f"\n  --- {side} ---")
        if path is None:
            print("    (no file found for this side)")
            continue
        print(f"    file      : {path.name}")
        print(f"    size      : {path.stat().st_size} bytes")
        if first:
            print("    per-line token-count histogram (first file only):")
            token_histogram(path)
            first = False
        xyz, how = load_annotation(path, side)
        print(f"    parsed by : {how}")
        if xyz is None:
            continue
        print(f"    shape     : {xyz.shape}   dtype: {xyz.dtype}")
        print(f"    n_landmarks: {xyz.shape[0]}   (expected 85)")
        print_axis_ranges("per-axis min/max:", xyz.min(axis=0), xyz.max(axis=0), indent="    ")
        mean_y = float(xyz[:, 1].mean())
        print(f"    mean Y sign: {'+' if mean_y > 0 else '-'}  (|mean Y| = {fmt(abs(mean_y))})")


def print_annotation_envelope(by_subject: dict[str, dict[str, Path]], limit: int) -> None:
    section(f"6. ANNOTATION ENVELOPE ACROSS SUBJECTS (limit={limit or 'all'})")
    if not by_subject:
        print("  no annotation files matched a subject ID + side")
        return

    sids = sorted(by_subject)
    if limit:
        sids = sids[:limit]
    print(f"  scanning {len(sids)} subjects ...")

    lo = {s: np.full(3, np.inf) for s in ("left", "right")}
    hi = {s: np.full(3, -np.inf) for s in ("left", "right")}
    extents = defaultdict(list)
    n_ok = Counter()
    shapes = Counter()
    failures: list[str] = []
    sign_warnings: list[str] = []

    for sid in sids:
        for side in ("left", "right"):
            path = by_subject[sid].get(side)
            if path is None:
                continue
            xyz, how = load_annotation(path, side)
            if xyz is None:
                if len(failures) < 10:
                    failures.append(f"{sid}/{side}: {how}")
                continue
            n_ok[side] += 1
            shapes[(side, xyz.shape)] += 1
            lo[side] = np.minimum(lo[side], xyz.min(axis=0))
            hi[side] = np.maximum(hi[side], xyz.max(axis=0))
            extents[side].append(xyz.max(axis=0) - xyz.min(axis=0))

            mean_y = float(xyz[:, 1].mean())
            if side == "left" and mean_y > 0:
                sign_warnings.append(f"LEFT ear {sid} has mean Y > 0 ({fmt(mean_y)})")
            if side == "right" and mean_y < 0:
                sign_warnings.append(f"RIGHT ear {sid} has mean Y < 0 ({fmt(mean_y)})")

    print(f"  loaded ok: {dict(n_ok)}")
    print(f"  shapes seen: { {f'{s}:{sh}': n for (s, sh), n in shapes.items()} }")
    if failures:
        print(f"  parse failures (first {len(failures)}):")
        for f in failures:
            print(f"    {f}")

    for side in ("left", "right"):
        print(f"\n  --- {side} ({n_ok[side]} ears) ---")
        if not n_ok[side]:
            print("    (none loaded)")
            continue
        print_axis_ranges("landmark envelope (min/max over subjects):", lo[side], hi[side], indent="    ")
        ext = np.asarray(extents[side])
        for i, ax in enumerate(AXES):
            print_stats(f"bbox extent {ax}", ext[:, i], indent="    ")
        print_stats("bbox diagonal", np.linalg.norm(ext, axis=1), indent="    ")

    print()
    print("  " + "-" * 70)
    if sign_warnings:
        print(f"  !! Y-SIGN WARNINGS: {len(sign_warnings)} ear(s) violate the expected convention")
        print("     (expected: left ear mean Y < 0, right ear mean Y > 0)")
        for w in sign_warnings[:20]:
            print(f"     {w}")
        if len(sign_warnings) > 20:
            print(f"     ... and {len(sign_warnings) - 20} more")
    else:
        print("  OK: every left ear has mean Y < 0 and every right ear has mean Y > 0.")


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("HUAWEI_DATA_ROOT", "data"),
                    help="dataset root (default: $HUAWEI_DATA_ROOT, else 'data')")
    ap.add_argument("--max-meshes", type=int, default=20,
                    help="meshes to scan in section 4; 0 = all (default: 20)")
    ap.add_argument("--max-subjects", type=int, default=0,
                    help="subjects to scan in section 6; 0 = all (default: 0)")
    ap.add_argument("--entries", type=int, default=10,
                    help="entries listed per folder in section 1 (default: 10)")
    args = ap.parse_args()

    root = Path(args.root).expanduser()

    print("=" * 78)
    print("HUAWEI DATASET INSPECTION — METADATA ONLY")
    print("Prints counts, dtypes, patterns and per-axis min/max envelopes.")
    print("Never prints individual point/landmark coordinates. Writes nothing.")
    print("=" * 78)
    print(f"root       : {root}")
    print(f"exists     : {root.exists()}")
    print(f"numpy      : {np.__version__}")
    try:
        import trimesh
        print(f"trimesh    : {trimesh.__version__}")
    except ImportError:
        print("trimesh    : NOT INSTALLED (sections 3 and 4 will be skipped)")

    if not root.exists():
        print("\n!! Dataset root does not exist. Set HUAWEI_DATA_ROOT or pass --root.")
        return 1

    print_tree(root, max_depth=2, max_entries=args.entries)

    meshes, anns = find_files(root)
    print_patterns(meshes, anns)

    by_subject: dict[str, dict[str, Path]] = defaultdict(dict)
    for p in anns:
        sid = subject_id_of(p.name)
        side = detect_side(p.name)
        if sid and side:
            by_subject[sid][side] = p

    try:
        import trimesh  # noqa: F401
        print_first_mesh(meshes)
        print_mesh_envelope(meshes, args.max_meshes)
    except ImportError:
        section("3 + 4. MESHES — SKIPPED (trimesh not installed)")

    print_first_subject_annotations(by_subject)
    print_annotation_envelope(by_subject, args.max_subjects)

    section("DONE")
    print("Copy the values above into DATA_SPEC.md, replacing the `?` fields.")
    print("Nothing was written to disk.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Unit tests for the Role A scripts' correctness-critical helpers.

``scripts/make_split.py`` decides which subjects the frozen crop box may see,
and ``scripts/crop_stats.py`` decides whether that box is safe to freeze. Both
only ever *run* against data these tests may not touch, so the logic itself is
pinned here on synthetic bboxes constructed in the test — no Huawei data is read
(see ``CLAUDE.md``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.geometry import load_crop_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import crop_stats  # noqa: E402
from crop_stats import (  # noqa: E402
    box_from_bboxes,
    leave_one_out_headroom,
    read_subject_list,
    side_stats,
)
from make_split import make_split  # noqa: E402

# --------------------------------------------------------------------------- #
# make_split
# --------------------------------------------------------------------------- #

SUBJECTS = [f"P{3 * i:04d}" for i in range(1, 201)]  # 200 non-contiguous IDs


def test_make_split_is_disjoint_and_covering():
    train, val = make_split(SUBJECTS, n_train=160, seed=42)

    assert len(train) == 160 and len(val) == 40
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(SUBJECTS)
    assert train == sorted(train) and val == sorted(val)


def test_make_split_is_seed_stable_and_order_independent():
    train, val = make_split(SUBJECTS, n_train=160, seed=42)

    # Same seed -> same split, whatever order the IDs arrived in.
    shuffled = list(np.random.default_rng(7).permutation(SUBJECTS))
    assert make_split(shuffled, n_train=160, seed=42) == (train, val)
    assert make_split(SUBJECTS, n_train=160, seed=42) == (train, val)

    # A different seed must actually move subjects between folds.
    other_train, _ = make_split(SUBJECTS, n_train=160, seed=43)
    assert other_train != train


def test_make_split_rejects_bad_input():
    with pytest.raises(ValueError):
        make_split(SUBJECTS + [SUBJECTS[0]], n_train=160)      # duplicate ID
    with pytest.raises(ValueError):
        make_split(SUBJECTS, n_train=0)                        # empty train fold
    with pytest.raises(ValueError):
        make_split(SUBJECTS, n_train=len(SUBJECTS))            # empty val fold


def test_read_subject_list_skips_blanks_and_comments(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text("# a comment\nP0001\n\n  P0002  \nP0003\n", encoding="utf-8")
    assert read_subject_list(path) == ["P0001", "P0002", "P0003"]

    dupes = tmp_path / "dupes.txt"
    dupes.write_text("P0001\nP0002\nP0001\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_subject_list(dupes)

    empty = tmp_path / "empty.txt"
    empty.write_text("# nothing here\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_subject_list(empty)


# --------------------------------------------------------------------------- #
# crop box + leave-one-out headroom
# --------------------------------------------------------------------------- #

def _identical_bboxes(n: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """``n`` subjects with the exact same [0,40] x [0,20] x [0,10] landmark bbox."""
    lo = np.tile(np.array([0.0, 0.0, 0.0]), (n, 1))
    hi = np.tile(np.array([40.0, 20.0, 10.0]), (n, 1))
    return lo, hi


def test_box_from_bboxes_applies_the_margin_rule():
    lo, hi = _identical_bboxes()

    box_lo, box_hi, env_lo, env_hi, margin = box_from_bboxes(
        lo, hi, margin_frac=0.25, min_margin=15.0
    )

    np.testing.assert_allclose(env_lo, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(env_hi, [40.0, 20.0, 10.0])
    # margin = max(0.25 * median extent, 15): X 10 -> 15 floor, Y 5 -> 15, Z 2.5 -> 15
    np.testing.assert_allclose(margin, [15.0, 15.0, 15.0])
    np.testing.assert_allclose(box_lo, [-15.0, -15.0, -15.0])
    np.testing.assert_allclose(box_hi, [55.0, 35.0, 25.0])

    # With a small floor the fractional term wins on the widest axis.
    _, _, _, _, margin2 = box_from_bboxes(lo, hi, margin_frac=0.25, min_margin=1.0)
    np.testing.assert_allclose(margin2, [10.0, 5.0, 2.5])


def test_loo_headroom_equals_the_margin_for_a_uniform_set():
    """Identical subjects: dropping one changes nothing, so headroom == margin."""
    lo, hi = _identical_bboxes()

    headroom, lower, upper = leave_one_out_headroom(lo, hi, margin_frac=0.25, min_margin=15.0)

    assert headroom.shape == lo.shape
    np.testing.assert_allclose(headroom, 15.0)
    np.testing.assert_allclose(lower, 15.0)
    np.testing.assert_allclose(upper, 15.0)


def test_loo_headroom_goes_negative_for_a_planted_outlier():
    """One subject reaching well past the others is caught on exactly one axis.

    This is the case the by-construction envelope check could never see: the
    outlier defines the envelope, so it is trivially "inside" it, yet a box built
    from the other subjects would truncate it.
    """
    lo, hi = _identical_bboxes(n=20)
    lo, hi = lo.copy(), hi.copy()
    hi[0, 0] = 40.0 + 25.0        # subject 0 reaches 25 mm past everyone else in +X

    headroom, lower, upper = leave_one_out_headroom(lo, hi, margin_frac=0.25, min_margin=15.0)

    # The outlier fails, and only on the +X face (25 mm out, 15 mm of margin).
    assert headroom[0, 0] == pytest.approx(-10.0)
    assert upper[0, 0] == pytest.approx(-10.0)
    assert lower[0, 0] > 0.0
    assert np.all(headroom[0, 1:] > 0.0)

    # Everyone else keeps positive headroom: the outlier only widens their box.
    assert np.all(headroom[1:] > 0.0)


def test_loo_headroom_is_not_the_by_construction_check():
    """The outlier is inside the full envelope yet fails leave-one-out."""
    lo, hi = _identical_bboxes(n=20)
    lo, hi = lo.copy(), hi.copy()
    hi[0, 0] = 40.0 + 25.0

    box_lo, box_hi, *_ = box_from_bboxes(lo, hi, margin_frac=0.25, min_margin=15.0)
    assert np.all(lo >= box_lo) and np.all(hi <= box_hi)   # nobody is "outside"

    headroom, _, _ = leave_one_out_headroom(lo, hi, margin_frac=0.25, min_margin=15.0)
    assert headroom.min() < 0.0                            # but leave-one-out fails


def test_side_stats_reports_worst_subjects_and_tight_face():
    lo, hi = _identical_bboxes(n=10)
    lo, hi = lo.copy(), hi.copy()
    hi[3, 0] = 40.0 + 25.0                        # worst: pushes past the +X face
    lo[7, 1] = -18.0                              # second: pushes past the -Y face
    subjects = [f"P{i:04d}" for i in range(10)]

    st = side_stats(subjects, lo, hi, margin_frac=0.25, min_margin=15.0)

    assert st["n_subjects"] == 10
    assert st["loo_min_headroom"] < 0.0
    assert st["loo_min_headroom_per_axis"][0] < 0.0        # X fails
    assert st["loo_min_headroom_per_axis"][2] > 0.0        # Z is fine

    # The worst subjects are reported worst-first, and name the right faces.
    worst_ids = [row[0] for row in st["loo_worst"]]
    assert worst_ids[:2] == ["P0003", "P0007"]
    assert st["loo_tight_face_per_axis"][0] == "hi"        # X binds above
    assert st["loo_tight_face_per_axis"][1] == "lo"        # Y binds below

    worst_row = st["loo_worst"][0]
    assert worst_row[1] == pytest.approx(worst_row[2].min())
    assert worst_row[3][0] == "hi"


def test_side_stats_passes_for_a_well_separated_set():
    """A set with no outlier keeps positive headroom on every axis."""
    rng = np.random.default_rng(0)
    centres = rng.normal(loc=[40.0, -70.0, 10.0], scale=[2.0, 1.0, 2.0], size=(40, 3))
    half = np.array([19.0, 9.0, 29.0])
    lo, hi = centres - half, centres + half
    subjects = [f"P{i:04d}" for i in range(40)]

    st = side_stats(subjects, lo, hi, margin_frac=0.25, min_margin=15.0)

    assert np.all(st["loo_min_headroom_per_axis"] > 0.0)
    assert len(st["loo_worst"]) == 5

# --------------------------------------------------------------------------- #
# crop_stats.main(): what it writes, and what it refuses to write
# --------------------------------------------------------------------------- #

def _fake_dataset(monkeypatch, lo: np.ndarray, hi: np.ndarray, subjects: list[str]) -> None:
    """Patch crop_stats' data access so main() runs without touching any file.

    Both sides get the same synthetic bboxes; no mesh, no CSV, no data root.
    """
    monkeypatch.setattr(crop_stats, "list_subjects", lambda root: list(subjects))

    def fake_collect(subs, root):
        boxes = {side: (lo, hi) for side in ("left", "right")}
        return list(subs), boxes, []

    monkeypatch.setattr(crop_stats, "collect_bboxes", fake_collect)


def _run_main(tmp_path, out_name="crop.yaml", extra_args=()):
    ids = tmp_path / "train_ids.txt"
    if not ids.exists():
        ids.write_text("\n".join(f"P{i:04d}" for i in range(10)) + "\n", encoding="utf-8")
    out = tmp_path / out_name
    code = crop_stats.main(
        ["--root", str(tmp_path), "--subject-list", str(ids), "--out", str(out), *extra_args]
    )
    return code, out, out.with_name(out.stem + ".rejected" + out.suffix)


def test_main_writes_the_config_when_the_criterion_passes(tmp_path, monkeypatch, capsys):
    subjects = [f"P{i:04d}" for i in range(10)]
    lo, hi = _identical_bboxes(n=10)
    _fake_dataset(monkeypatch, lo, hi, subjects)

    code, out, rejected = _run_main(tmp_path)
    printed = capsys.readouterr().out

    assert code == 0
    assert out.exists() and not rejected.exists()
    assert "PASS" in printed

    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert doc["freeze_criterion_passed"] is True
    assert doc["split"]["n_subjects_used"] == 10
    assert len(doc["split"]["sha256"]) == 64
    # And the frozen file is loadable by the rest of the team.
    assert load_crop_config("left", out).lo.shape == (3,)


def test_main_refuses_to_write_the_config_when_the_criterion_fails(tmp_path, monkeypatch, capsys):
    """The consequential behaviour: a failing run must not produce configs/crop.yaml."""
    subjects = [f"P{i:04d}" for i in range(10)]
    lo, hi = _identical_bboxes(n=10)
    lo, hi = lo.copy(), hi.copy()
    hi[0, 0] = 40.0 + 25.0                      # outlier: 25 mm out, 15 mm of margin
    _fake_dataset(monkeypatch, lo, hi, subjects)

    code, out, rejected = _run_main(tmp_path)
    printed = capsys.readouterr().out

    assert code == 1
    assert not out.exists()                     # the frozen path is untouched
    assert rejected.exists()                    # the numbers are kept for inspection
    assert "FAIL" in printed and "NOT written" in printed

    doc = yaml.safe_load(rejected.read_text(encoding="utf-8"))
    assert doc["freeze_criterion_passed"] is False
    assert rejected.read_text(encoding="utf-8").startswith("# REJECTED")

    # ... and the rejected file cannot be loaded as if it were frozen.
    with pytest.raises(ValueError, match="freeze_criterion_passed"):
        load_crop_config("left", rejected)


def test_main_warns_that_an_earlier_config_survives_a_failed_run(tmp_path, monkeypatch, capsys):
    """Refusing to write leaves the old file in place; the user must be told."""
    subjects = [f"P{i:04d}" for i in range(10)]
    lo, hi = _identical_bboxes(n=10)
    _fake_dataset(monkeypatch, lo, hi, subjects)
    code, out, _ = _run_main(tmp_path)           # first run passes and writes
    assert code == 0 and out.exists()
    before = out.read_bytes()

    lo, hi = lo.copy(), hi.copy()
    hi[0, 0] = 40.0 + 25.0
    _fake_dataset(monkeypatch, lo, hi, subjects)
    capsys.readouterr()
    code, out, rejected = _run_main(tmp_path)    # second run fails
    printed = capsys.readouterr().out

    assert code == 1
    assert out.read_bytes() == before            # untouched, not half-written
    assert "STILL EXISTS" in printed
    assert "freeze_criterion_passed=True" in printed


def test_main_dry_run_writes_nothing(tmp_path, monkeypatch):
    subjects = [f"P{i:04d}" for i in range(10)]
    lo, hi = _identical_bboxes(n=10)
    _fake_dataset(monkeypatch, lo, hi, subjects)

    code, out, rejected = _run_main(tmp_path, extra_args=("--dry-run",))

    assert code == 0
    assert not out.exists() and not rejected.exists()


def test_main_requires_an_existing_subject_list(tmp_path, monkeypatch):
    subjects = [f"P{i:04d}" for i in range(10)]
    lo, hi = _identical_bboxes(n=10)
    _fake_dataset(monkeypatch, lo, hi, subjects)
    out = tmp_path / "crop.yaml"

    code = crop_stats.main(
        ["--root", str(tmp_path), "--subject-list", str(tmp_path / "nope.txt"), "--out", str(out)]
    )

    assert code == 1
    assert not out.exists()


# --------------------------------------------------------------------------- #
# check_roundtrip / check_crop_all
#
# Both scripts only ever run against data these tests may not touch, so the
# logic is pinned here on a synthetic two-ear "subject": a grid of vertices in
# each side's crop box and 85 fake landmarks inside it. What is being checked is
# the scripts' own arithmetic and verdicts, not numpy's.
# --------------------------------------------------------------------------- #

import inspect  # noqa: E402

import check_crop_all  # noqa: E402
import check_roundtrip  # noqa: E402
from src.data import RawSubject  # noqa: E402
from src.geometry import CropConfig, N_LANDMARKS, canonicalize_ear  # noqa: E402

LEFT_LO, LEFT_HI = np.array([0.0, 0.0, 0.0]), np.array([10.0, 10.0, 20.0])
RIGHT_LO, RIGHT_HI = np.array([0.0, -10.0, 0.0]), np.array([10.0, 0.0, 20.0])


def _grid(lo: np.ndarray, hi: np.ndarray, n: int = 11) -> np.ndarray:
    """``n**3`` points filling the box, corners included."""
    axes = [np.linspace(lo[a], hi[a], n) for a in range(3)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def _fake_landmarks(seed: int = 0, lo=LEFT_LO, hi=LEFT_HI) -> np.ndarray:
    """85 pseudo-random points strictly inside a box."""
    rng = np.random.default_rng(seed)
    return lo + (hi - lo) * (0.1 + 0.8 * rng.random((N_LANDMARKS, 3)))


def _write_crop_yaml(tmp_path: Path, min_vertices: int = 500) -> Path:
    path = tmp_path / "crop.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "freeze_criterion_passed": True,
                "sides": {
                    "left": {"lo": LEFT_LO.tolist(), "hi": LEFT_HI.tolist(),
                             "min_vertices": min_vertices},
                    "right": {"lo": RIGHT_LO.tolist(), "hi": RIGHT_HI.tolist(),
                              "min_vertices": min_vertices},
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _fake_ears(
    monkeypatch,
    module,
    subjects: list[str],
    landmarks: dict[str, dict[str, np.ndarray]] | None = None,
    grid_n: int = 11,
    failing: set[str] | None = None,
) -> None:
    """Point one script module's loaders at synthetic two-ear subjects."""
    vertices = np.vstack([_grid(LEFT_LO, LEFT_HI, grid_n), _grid(RIGHT_LO, RIGHT_HI, grid_n)])
    failing = failing or set()

    def fake_load_mesh(path):
        sid = Path(path).name
        if sid in failing:
            raise OSError("synthetic mesh failure")
        return RawSubject(sid, vertices.copy(), None, None, Path(path))

    def fake_load_subject_landmarks(sid, root):
        if landmarks is not None:
            return landmarks[sid]
        return {
            "left": _fake_landmarks(seed=abs(hash(sid)) % 1000),
            "right": _fake_landmarks(seed=abs(hash(sid)) % 1000, lo=RIGHT_LO, hi=RIGHT_HI),
        }

    monkeypatch.setattr(module, "list_subjects", lambda root: list(subjects))
    monkeypatch.setattr(module, "mesh_path", lambda sid, root=None: Path(sid))
    monkeypatch.setattr(module, "load_mesh", fake_load_mesh)
    monkeypatch.setattr(module, "load_subject_landmarks", fake_load_subject_landmarks)


def _subject_list(tmp_path: Path, subjects: list[str]) -> Path:
    path = tmp_path / "ids.txt"
    path.write_text("\n".join(subjects) + "\n", encoding="utf-8")
    return path


# --- check_roundtrip -------------------------------------------------------- #

def test_ear_roundtrip_is_exact_and_mesh_derived():
    crop_points = _grid(LEFT_LO, LEFT_HI)
    landmarks = _fake_landmarks(seed=1)

    error, canonical, scale = check_roundtrip.ear_roundtrip(landmarks, crop_points, "left")

    assert error < 1e-9
    assert scale == pytest.approx(10.0)          # half the largest extent (Z = 20)
    # Landmarks inside the crop box land inside the canonical unit-ish box.
    assert np.max(np.abs(canonical)) <= 1.0


def test_ear_roundtrip_mirrors_only_the_mirror_side():
    left_points, right_points = _grid(LEFT_LO, LEFT_HI), _grid(RIGHT_LO, RIGHT_HI)
    # The same ear geometry either side of Y=0, so canonical Y must agree in
    # sign once the right side has been mirrored.
    left_lm = _fake_landmarks(seed=2)
    right_lm = left_lm * np.array([1.0, -1.0, 1.0])

    _, left_canon, _ = check_roundtrip.ear_roundtrip(left_lm, left_points, "left")
    _, right_canon, _ = check_roundtrip.ear_roundtrip(right_lm, right_points, "right")

    assert check_roundtrip.MIRROR_SIDE == "right"
    assert np.allclose(left_canon, right_canon)


def test_side_accumulator_tracks_envelope_and_the_outside_band():
    acc = check_roundtrip.SideAccumulator()
    inside = np.zeros((3, 3))
    inside[0] = [0.5, -0.25, 1.0]
    outside = np.zeros((3, 3))
    outside[0] = [0.0, 0.0, check_roundtrip.CANONICAL_LIMIT + 0.1]

    acc.add("P0001", 1e-15, inside, scale=10.0, n_crop=1000)
    acc.add("P0002", 1e-12, outside, scale=11.0, n_crop=2000)

    assert acc.n_ears == 2
    assert acc.max_error == 1e-12 and acc.worst_subject == "P0002"
    assert acc.canonical_lo == pytest.approx([0.0, -0.25, 0.0])
    assert acc.canonical_hi == pytest.approx([0.5, 0.0, check_roundtrip.CANONICAL_LIMIT + 0.1])
    assert acc.outside_ids == ["P0002"]


def test_check_roundtrip_main_passes_on_exact_data(tmp_path, monkeypatch, capsys):
    subjects = ["P0001", "P0002", "P0003"]
    _fake_ears(monkeypatch, check_roundtrip, subjects)

    code = check_roundtrip.main([
        "--root", str(tmp_path),
        "--subject-list", str(_subject_list(tmp_path, subjects)),
        "--crop-config", str(_write_crop_yaml(tmp_path)),
    ])
    out = capsys.readouterr().out

    assert code == 0
    assert "=> PASS" in out
    assert "ears checked: 6" in out


def test_check_roundtrip_main_fails_on_an_impossible_tolerance(tmp_path, monkeypatch, capsys):
    # A negative tolerance no round-trip can meet: proves the verdict is driven
    # by the measured error, not hard-coded to PASS.
    subjects = ["P0001"]
    _fake_ears(monkeypatch, check_roundtrip, subjects)

    code = check_roundtrip.main([
        "--root", str(tmp_path),
        "--subject-list", str(_subject_list(tmp_path, subjects)),
        "--crop-config", str(_write_crop_yaml(tmp_path)),
        "--tol", "-1",
    ])

    assert code == 1
    assert "=> FAIL" in capsys.readouterr().out


def test_check_roundtrip_main_fails_when_an_ear_cannot_be_checked(tmp_path, monkeypatch, capsys):
    subjects = ["P0001", "P0002"]
    _fake_ears(monkeypatch, check_roundtrip, subjects, failing={"P0002"})

    code = check_roundtrip.main([
        "--root", str(tmp_path),
        "--subject-list", str(_subject_list(tmp_path, subjects)),
        "--crop-config", str(_write_crop_yaml(tmp_path)),
    ])
    out = capsys.readouterr().out

    assert code == 1
    assert "could not be checked" in out and "P0002" in out


def test_check_roundtrip_main_refuses_a_rejected_crop_config(tmp_path, monkeypatch, capsys):
    subjects = ["P0001"]
    _fake_ears(monkeypatch, check_roundtrip, subjects)
    cfg = _write_crop_yaml(tmp_path)
    doc = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    doc["freeze_criterion_passed"] = False
    cfg.write_text(yaml.safe_dump(doc), encoding="utf-8")

    code = check_roundtrip.main([
        "--root", str(tmp_path),
        "--subject-list", str(_subject_list(tmp_path, subjects)),
        "--crop-config", str(cfg),
    ])

    assert code == 1
    assert "crop config" in capsys.readouterr().out


# --- check_crop_all --------------------------------------------------------- #

def test_count_landmarks_inside_matches_crop_ear_including_the_boundary():
    cfg = CropConfig(side="left", lo=LEFT_LO, hi=LEFT_HI)
    landmarks = np.vstack([
        LEFT_LO,                                   # on the lower corner: inside
        LEFT_HI,                                   # on the upper corner: inside
        0.5 * (LEFT_LO + LEFT_HI),                 # middle: inside
        LEFT_HI + np.array([0.0, 0.0, 1e-9]),      # a hair above Z: outside
        LEFT_LO - np.array([1.0, 0.0, 0.0]),       # outside
    ])

    assert check_crop_all.count_landmarks_inside(landmarks, "left", cfg, "P0001") == 3


def test_check_crop_all_main_passes_when_every_landmark_is_inside(tmp_path, monkeypatch, capsys):
    subjects = ["P0001", "P0002"]
    _fake_ears(monkeypatch, check_crop_all, subjects)

    code = check_crop_all.main([
        "--root", str(tmp_path),
        "--subject-list", str(_subject_list(tmp_path, subjects)),
        "--crop-config", str(_write_crop_yaml(tmp_path)),
    ])
    out = capsys.readouterr().out

    assert code == 0
    assert "=> PASS" in out
    assert "ears losing a GT landmark: 0" in out


def test_check_crop_all_main_fails_and_names_a_truncated_ear(tmp_path, monkeypatch, capsys):
    subjects = ["P0001", "P0002"]
    bad = _fake_landmarks(seed=3)
    bad[7] = LEFT_HI + np.array([0.0, 0.0, 5.0])          # one landmark above the box
    landmarks = {
        "P0001": {"left": _fake_landmarks(seed=4),
                  "right": _fake_landmarks(seed=4, lo=RIGHT_LO, hi=RIGHT_HI)},
        "P0002": {"left": bad,
                  "right": _fake_landmarks(seed=5, lo=RIGHT_LO, hi=RIGHT_HI)},
    }
    _fake_ears(monkeypatch, check_crop_all, subjects, landmarks=landmarks)

    code = check_crop_all.main([
        "--root", str(tmp_path),
        "--subject-list", str(_subject_list(tmp_path, subjects)),
        "--crop-config", str(_write_crop_yaml(tmp_path)),
    ])
    out = capsys.readouterr().out

    assert code == 1
    assert "ears losing a GT landmark: 1" in out
    assert f"worst ear kept {N_LANDMARKS - 1}/{N_LANDMARKS}" in out
    assert "P0002" in out and "truncates" in out


def test_check_crop_all_main_fails_on_a_suspicious_crop(tmp_path, monkeypatch, capsys):
    subjects = ["P0001"]
    _fake_ears(monkeypatch, check_crop_all, subjects)

    # min_vertices above the synthetic crop size (11**3 = 1331 per side).
    code = check_crop_all.main([
        "--root", str(tmp_path),
        "--subject-list", str(_subject_list(tmp_path, subjects)),
        "--crop-config", str(_write_crop_yaml(tmp_path, min_vertices=5000)),
    ])
    out = capsys.readouterr().out

    assert code == 1
    assert "suspicious crops: 2" in out
    assert "suspiciously small" in out


def test_check_crop_all_main_flags_crops_too_small_to_sample(tmp_path, monkeypatch, capsys):
    subjects = ["P0001"]
    # 8**3 = 512 vertices per side: above min_vertices=500, below N_POINTS=2048,
    # so it is not suspicious but would force sampling with replacement.
    _fake_ears(monkeypatch, check_crop_all, subjects, grid_n=8)

    code = check_crop_all.main([
        "--root", str(tmp_path),
        "--subject-list", str(_subject_list(tmp_path, subjects)),
        "--crop-config", str(_write_crop_yaml(tmp_path)),
    ])
    out = capsys.readouterr().out

    assert code == 0
    assert "WITH replacement): 1/1" in out


# --- reviewer round 5: unchecked ears, an inexact inverse, split provenance -- #

def test_check_roundtrip_main_fails_when_a_listed_subject_has_no_mesh(
    tmp_path, monkeypatch, capsys
):
    listed = ["P0001", "P0002", "P0003"]
    _fake_ears(monkeypatch, check_roundtrip, listed)
    # Only two of the three listed subjects actually exist under --root.
    monkeypatch.setattr(check_roundtrip, "list_subjects", lambda root: ["P0001", "P0002"])

    code = check_roundtrip.main([
        "--root", str(tmp_path),
        "--subject-list", str(_subject_list(tmp_path, listed)),
        "--crop-config", str(_write_crop_yaml(tmp_path)),
    ])
    out = capsys.readouterr().out

    # An exact round-trip on the ears that were reached must NOT hide the fact
    # that a third of the run never happened.
    assert code == 1
    assert "no mesh under" in out and "P0003" in out
    assert "=> FAIL" in out


def test_check_roundtrip_allow_failures_lets_the_verdict_stand(tmp_path, monkeypatch, capsys):
    listed = ["P0001", "P0002", "P0003"]
    _fake_ears(monkeypatch, check_roundtrip, listed)
    monkeypatch.setattr(check_roundtrip, "list_subjects", lambda root: ["P0001", "P0002"])

    code = check_roundtrip.main([
        "--root", str(tmp_path),
        "--subject-list", str(_subject_list(tmp_path, listed)),
        "--crop-config", str(_write_crop_yaml(tmp_path)),
        "--allow-failures",
    ])
    out = capsys.readouterr().out

    assert code == 0
    assert "PASS (--allow-failures)" in out


def test_check_roundtrip_limit_does_not_invent_missing_subjects(tmp_path, monkeypatch, capsys):
    listed = ["P0001", "P0002", "P0003"]
    _fake_ears(monkeypatch, check_roundtrip, listed)
    monkeypatch.setattr(check_roundtrip, "list_subjects", lambda root: ["P0001", "P0002"])

    # --limit 2 stops before P0003, so P0003 is out of scope, not missing.
    code = check_roundtrip.main([
        "--root", str(tmp_path),
        "--subject-list", str(_subject_list(tmp_path, listed)),
        "--crop-config", str(_write_crop_yaml(tmp_path)),
        "--limit", "2",
    ])
    out = capsys.readouterr().out

    assert code == 0
    assert "no mesh under" not in out
    assert "--limit 2" in out


def test_check_roundtrip_main_detects_an_inexact_inverse(tmp_path, monkeypatch, capsys):
    """The verdict must follow the measured error, not the happy path."""
    subjects = ["P0001"]
    _fake_ears(monkeypatch, check_roundtrip, subjects)
    real_inverse = check_roundtrip.inverse_transform_points
    monkeypatch.setattr(
        check_roundtrip,
        "inverse_transform_points",
        lambda points, t: real_inverse(points, t) + 1e-3,
    )

    code = check_roundtrip.main([
        "--root", str(tmp_path),
        "--subject-list", str(_subject_list(tmp_path, subjects)),
        "--crop-config", str(_write_crop_yaml(tmp_path)),
    ])
    out = capsys.readouterr().out

    assert code == 1
    assert "1.000e-03 mm" in out
    assert "NOT exactly invertible" in out


def test_ear_roundtrip_goes_through_the_serialised_transform(monkeypatch):
    """The inverse must use the transform as the cache will hand it back."""
    seen: list[dict] = []
    real_from_dict = check_roundtrip.EarTransform.from_dict

    def spy(d):
        seen.append(d)
        return real_from_dict(d)

    monkeypatch.setattr(check_roundtrip.EarTransform, "from_dict", staticmethod(spy))
    check_roundtrip.ear_roundtrip(_fake_landmarks(seed=9), _grid(LEFT_LO, LEFT_HI), "left")

    assert len(seen) == 1
    assert set(seen[0]) == {"centre", "scale", "mirror_axis"}


def test_check_crop_all_main_fails_when_a_listed_subject_has_no_mesh(
    tmp_path, monkeypatch, capsys
):
    listed = ["P0001", "P0002"]
    _fake_ears(monkeypatch, check_crop_all, listed)
    monkeypatch.setattr(check_crop_all, "list_subjects", lambda root: ["P0001"])

    code = check_crop_all.main([
        "--root", str(tmp_path),
        "--subject-list", str(_subject_list(tmp_path, listed)),
        "--crop-config", str(_write_crop_yaml(tmp_path)),
    ])
    out = capsys.readouterr().out

    assert code == 1
    assert "no mesh under" in out and "P0002" in out


def test_check_crop_all_warns_when_run_on_the_split_the_box_came_from(
    tmp_path, monkeypatch, capsys
):
    subjects = ["P0001"]
    _fake_ears(monkeypatch, check_crop_all, subjects)
    ids = _subject_list(tmp_path, subjects)
    cfg = _write_crop_yaml(tmp_path)
    doc = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    doc["split"] = {"path": "splits/train_ids.txt", "sha256": crop_stats.sha256_file(ids)}
    cfg.write_text(yaml.safe_dump(doc), encoding="utf-8")

    code = check_crop_all.main([
        "--root", str(tmp_path), "--subject-list", str(ids), "--crop-config", str(cfg),
    ])
    out = capsys.readouterr().out

    # Still a PASS - the box does hold on these ears - but Alfred is told the
    # PASS is true by construction and proves nothing about unseen subjects.
    assert code == 0
    assert "SAME subject list" in out


def test_check_crop_all_does_not_warn_on_a_held_out_split(tmp_path, monkeypatch, capsys):
    subjects = ["P0001"]
    _fake_ears(monkeypatch, check_crop_all, subjects)
    cfg = _write_crop_yaml(tmp_path)
    doc = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    doc["split"] = {"path": "splits/train_ids.txt", "sha256": "0" * 64}
    cfg.write_text(yaml.safe_dump(doc), encoding="utf-8")

    code = check_crop_all.main([
        "--root", str(tmp_path),
        "--subject-list", str(_subject_list(tmp_path, subjects)),
        "--crop-config", str(cfg),
    ])
    out = capsys.readouterr().out

    assert code == 0
    assert "SAME subject list" not in out


def test_check_crop_all_allow_failures_lets_the_verdict_stand(tmp_path, monkeypatch, capsys):
    listed = ["P0001", "P0002"]
    _fake_ears(monkeypatch, check_crop_all, listed)
    monkeypatch.setattr(check_crop_all, "list_subjects", lambda root: ["P0001"])

    code = check_crop_all.main([
        "--root", str(tmp_path),
        "--subject-list", str(_subject_list(tmp_path, listed)),
        "--crop-config", str(_write_crop_yaml(tmp_path)),
        "--allow-failures",
    ])
    out = capsys.readouterr().out

    assert code == 0
    assert "unchecked, ignored via --allow-failures" in out


def test_check_crop_all_repeats_the_same_split_warning_in_the_verdict(
    tmp_path, monkeypatch, capsys
):
    subjects = ["P0001"]
    _fake_ears(monkeypatch, check_crop_all, subjects)
    ids = _subject_list(tmp_path, subjects)
    cfg = _write_crop_yaml(tmp_path)
    doc = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    doc["split"] = {"path": "splits/train_ids.txt", "sha256": crop_stats.sha256_file(ids)}
    cfg.write_text(yaml.safe_dump(doc), encoding="utf-8")

    check_crop_all.main([
        "--root", str(tmp_path), "--subject-list", str(ids), "--crop-config", str(cfg),
    ])
    verdict = capsys.readouterr().out.split("=== FROZEN CROP ON HELD-OUT SUBJECTS ===")[1]

    # A verdict read off the tail of a long log must carry the caveat too.
    assert "true by construction" in verdict


def test_check_roundtrip_convention_matches_the_pipeline():
    """The script's frozen convention must be canonicalize_ear's, not its own."""
    params = inspect.signature(canonicalize_ear).parameters

    assert check_roundtrip.MIRROR_SIDE == params["mirror_side"].default
    assert check_roundtrip.MIRROR_AXIS == params["mirror_axis"].default


# --- check_mirror ----------------------------------------------------------- #

import check_mirror  # noqa: E402


def _wedge(rng: np.random.Generator, n: int = 6000) -> np.ndarray:
    """A triangular prism filling half of the LEFT crop box.

    Its bounding box is (near enough) the whole box, so the shape is asymmetric
    *inside* its own bbox — which is the only asymmetry a mirror about the bbox
    centre can see.
    """
    p = rng.random((n, 3)) * (LEFT_HI - LEFT_LO) + LEFT_LO
    return p[(p[:, 1] - LEFT_LO[1]) < (p[:, 0] - LEFT_LO[0])]


#: y -> -y maps the LEFT box exactly onto the RIGHT box; so does y -> y - 10.
_MIRROR = np.array([1.0, -1.0, 1.0])
_SHIFT = np.array([0.0, -10.0, 0.0])


def _mirror_world(sid: str, mirrored: bool, seed: int = 0):
    """``(vertices, {side: [85,3]})`` for one synthetic subject.

    ``mirrored=True``: the right ear is the exact Y-reflection of the left one,
    so exactly one side must be flipped (configurations B and C).
    ``mirrored=False``: the right ear is the left one translated, so no flip is
    right (configuration A).
    """
    rng = np.random.default_rng(sum(map(ord, sid)) + seed)
    left = _wedge(rng)
    left_lm = _fake_landmarks(seed=sum(map(ord, sid)), lo=LEFT_LO, hi=LEFT_HI)
    if mirrored:
        right, right_lm = left * _MIRROR, left_lm * _MIRROR
    else:
        right, right_lm = left + _SHIFT, left_lm + _SHIFT
    return np.vstack([left, right]), {"left": left_lm, "right": right_lm}


def _fake_mirror_dataset(
    monkeypatch,
    subjects: list[str],
    mirrored: bool = True,
    failing: set[str] | None = None,
    tweak=None,
) -> None:
    """Point check_mirror's loaders at synthetic mirrored/translated subjects.

    ``tweak`` may mutate each subject's landmark dict, e.g. to plant a contour
    that is ordered differently on the two sides.
    """
    failing = failing or set()
    worlds = {sid: _mirror_world(sid, mirrored) for sid in subjects}
    if tweak is not None:
        for sid in subjects:
            tweak(worlds[sid][1])

    def fake_load_mesh(path):
        sid = Path(path).name
        if sid in failing:
            raise OSError("synthetic mesh failure")
        return RawSubject(sid, worlds[sid][0].copy(), None, None, Path(path))

    monkeypatch.setattr(check_mirror, "list_subjects", lambda root: list(subjects))
    monkeypatch.setattr(check_mirror, "mesh_path", lambda sid, root=None: Path(sid))
    monkeypatch.setattr(check_mirror, "load_mesh", fake_load_mesh)
    monkeypatch.setattr(check_mirror, "load_subject_landmarks",
                        lambda sid, root: worlds[sid][1])


def _mirror_boxes(min_vertices: int = 10) -> dict:
    return {"left": CropConfig("left", LEFT_LO, LEFT_HI, min_vertices),
            "right": CropConfig("right", RIGHT_LO, RIGHT_HI, min_vertices)}


def _prepared(sid: str = "P0001", mirrored: bool = True, landmarks=None):
    """Both ears of one synthetic subject, ready for ``compare_ears``."""
    vertices, own_landmarks = _mirror_world(sid, mirrored)
    landmarks = own_landmarks if landmarks is None else landmarks
    raw = RawSubject(sid, vertices, None, None, Path(sid))
    boxes = _mirror_boxes()
    return {
        side: check_mirror.prepare_ear(raw, side, boxes[side], landmarks[side],
                                       n_points=512)
        for side in ("left", "right")
    }


def _setting(key: str) -> "check_mirror.MirrorSetting":
    return next(c for c in check_mirror.CONFIGS if c.key == key)


def _run_mirror(tmp_path, subjects, extra=()):
    return check_mirror.main([
        "--root", str(tmp_path),
        "--subject-list", str(_subject_list(tmp_path, subjects)),
        "--crop-config", str(_write_crop_yaml(tmp_path, min_vertices=10)),
        "--n-points", "512",
        *extra,
    ])


def test_chamfer_distance_is_zero_symmetric_and_measures_offsets():
    rng = np.random.default_rng(0)
    cloud = rng.random((60, 3))

    assert check_mirror.chamfer_distance(cloud, cloud) == 0.0
    assert check_mirror.chamfer_distance(cloud, cloud[:20]) == pytest.approx(
        check_mirror.chamfer_distance(cloud[:20], cloud)
    )
    # Two single points 3 apart: the distance is the distance.
    assert check_mirror.chamfer_distance(
        np.zeros((1, 3)), np.array([[3.0, 0.0, 0.0]])
    ) == pytest.approx(3.0, abs=1e-12)
    # A rigid offset can only ever shrink under nearest-neighbour matching.
    assert 0.0 < check_mirror.chamfer_distance(
        cloud, cloud + np.array([3.0, 0.0, 0.0])
    ) <= 3.0


def test_chamfer_distance_rejects_malformed_clouds():
    with pytest.raises(ValueError, match=r"\[N,3\]"):
        check_mirror.chamfer_distance(np.zeros((4, 2)), np.zeros((4, 3)))
    with pytest.raises(ValueError, match="non-empty"):
        check_mirror.chamfer_distance(np.zeros((0, 3)), np.zeros((4, 3)))


def test_contour_means_follows_the_documented_ranges():
    distances = np.zeros(N_LANDMARKS)
    distances[25:55] = 2.0                      # concha outline only

    means = check_mirror.contour_means(distances)

    assert means.tolist() == [0.0, 2.0, 0.0, 0.0]
    assert [(s, e) for _, s, e in check_mirror.CONTOURS] == [
        (0, 25), (25, 55), (55, 75), (75, 85)
    ]


def test_winning_keys_groups_exact_ties_only():
    assert check_mirror.winning_keys({"A": 0.5, "B": 0.1, "C": 0.1}) == ["B", "C"]
    assert check_mirror.winning_keys({"A": 0.1, "B": 0.1000001, "C": 0.5}) == ["A"]


def test_subsample_draws_without_replacement_and_only_by_its_seed():
    """The branch the pipeline's 2048 points actually take (2048 -> 512)."""
    cloud = np.random.default_rng(0).random((2048, 3))

    first = check_mirror.subsample(cloud, 512, seed=7)
    again = check_mirror.subsample(cloud, 512, seed=7)
    other = check_mirror.subsample(cloud, 512, seed=8)

    assert first.shape == (512, 3)
    assert len({tuple(row) for row in first}) == 512      # no duplicates
    assert np.array_equal(first, again)                   # seed-determined
    assert not np.array_equal(first, other)
    # Fewer points than asked for: everything, untouched.
    assert np.array_equal(check_mirror.subsample(cloud[:100], 512, seed=7), cloud[:100])


def test_the_two_ears_of_a_subject_never_share_a_draw():
    ears = _prepared(mirrored=True)

    assert ears["left"].seed != ears["right"].seed


def test_canonical_points_match_the_pipelines_own(monkeypatch):
    """cfg B must reproduce canonicalize_ear exactly, or the verdict is fiction."""
    from preprocess import ear_seed

    vertices, landmarks = _mirror_world("P0001", mirrored=True)
    raw = RawSubject("P0001", vertices, None, None, Path("P0001"))
    boxes = _mirror_boxes()

    for side in ("left", "right"):
        ear = check_mirror.prepare_ear(raw, side, boxes[side], landmarks[side],
                                       n_points=2048)
        points, _, scale = check_mirror.canonicalise(ear, side, _setting("B"))
        expected = canonicalize_ear(raw, side, boxes[side], n_points=2048,
                                    seed=ear_seed(0, "P0001", side))

        assert np.array_equal(points, expected.points)
        assert scale == expected.transform.scale


def _ear_from(points: np.ndarray, landmarks=None) -> "check_mirror.EarInputs":
    if landmarks is None:
        landmarks = np.zeros((N_LANDMARKS, 3))
    return check_mirror.EarInputs(crop_points=points, sampled=points,
                                  landmarks=landmarks, n_crop=len(points), seed=0)


def test_frame_floor_measures_the_mismatch_between_the_two_crop_boxes():
    """The reference the decisive metric's level has to be read against."""
    left = _grid(LEFT_LO, LEFT_HI)                     # centre (5, 5, 10), scale 10
    exact = {"left": _ear_from(left), "right": _ear_from(left * _MIRROR)}
    # Same shape, but its box sits 2 mm further along X than a true reflection.
    offset = {"left": _ear_from(left),
              "right": _ear_from(left * _MIRROR + np.array([2.0, 0.0, 0.0]))}
    # ... and here it is half the size, so the two ears do not share a scale.
    smaller = {"left": _ear_from(left), "right": _ear_from(left * _MIRROR * 0.5)}

    assert check_mirror.frame_floor(exact).total == pytest.approx(0.0, abs=1e-12)
    assert check_mirror.frame_floor(exact).scale_ratio == pytest.approx(1.0, abs=1e-12)

    off = check_mirror.frame_floor(offset)
    assert off.offset_mm.tolist() == pytest.approx([2.0, 0.0, 0.0], abs=1e-9)
    assert off.canonical.tolist() == pytest.approx([0.2, 0.0, 0.0], abs=1e-9)
    assert off.off_axis == pytest.approx(0.2, abs=1e-9)   # X is off the mirror

    # Unequal scales: the canonical figure divides by the MEAN of the two (the
    # mm offset here is |(2.5, 2.5, 5)| over a mean scale of 7.5).
    small = check_mirror.frame_floor(smaller)
    assert small.scale_ratio == pytest.approx(2.0, abs=1e-9)
    assert small.total == pytest.approx(6.123724356957945 / 7.5, abs=1e-9)


def test_the_mirror_axis_term_is_an_upper_bound_not_a_floor():
    """A subject mirrored about y = 3 is perfect, yet the Y term reports 0.6.

    The mirror-axis component cannot separate crop-box mismatch from the
    subject's own mid-sagittal offset, so it must never be read as a floor.
    """
    left = _grid(LEFT_LO, LEFT_HI)
    shift = np.array([0.0, 6.0, 0.0])                  # reflection about y = 3
    left_lm = _fake_landmarks(seed=3, lo=LEFT_LO, hi=LEFT_HI)
    ears = {"left": _ear_from(left, left_lm),
            "right": _ear_from(left * _MIRROR + shift, left_lm * _MIRROR + shift)}

    metrics, _, _ = check_mirror.compare_ears(ears, _setting("B"))
    floor = check_mirror.frame_floor(ears)

    assert metrics.landmark_mean == pytest.approx(0.0, abs=1e-12)  # nothing wrong
    assert floor.canonical[1] == pytest.approx(0.6, abs=1e-9)      # Y overstates it
    assert floor.off_axis == pytest.approx(0.0, abs=1e-12)         # X and Z: nothing


def test_frame_floor_defaults_to_the_pipelines_mirror_axis():
    ears = _prepared(mirrored=True)
    default = check_mirror.frame_floor(ears)
    unmirrored = check_mirror.frame_floor(ears, mirror_axis=None)

    assert default.mirror_axis_used == 1
    assert default.total == pytest.approx(0.0, abs=1e-12)
    # Without the flip the two frames are a whole ear separation apart.
    assert unmirrored.mirror_axis_used is None
    assert unmirrored.total * unmirrored.mean_scale > 5.0


def test_main_prints_the_frame_floor_per_axis(tmp_path, monkeypatch, capsys):
    subjects = ["P0001"]
    _fake_mirror_dataset(monkeypatch, subjects, mirrored=True)

    _run_mirror(tmp_path, subjects, extra=["--no-plot"])
    out = capsys.readouterr().out

    axis_lines = {l.strip()[0]: l for l in out.splitlines()
                  if l.startswith("    ") and l.strip()[:1] in set("XYZ")
                  and "mean" in l}

    assert "frame floor" in out and "left/right scale ratio" in out
    assert "true floor (axes off the mirror)" in out
    assert "combine as vectors" in out                # never subtract it
    # The caveat belongs to the mirror axis and to no other.
    assert "upper bound, not a floor" in axis_lines["Y"]
    assert "upper bound" not in axis_lines["X"] and "upper bound" not in axis_lines["Z"]


def test_mirrored_subject_is_matched_exactly_by_the_mirror_configs():
    """Right ear = exact Y-reflection of the left: B nails it, A does not."""
    ears = _prepared(mirrored=True)

    a, _, _ = check_mirror.compare_ears(ears, _setting("A"))
    b, _, _ = check_mirror.compare_ears(ears, _setting("B"))

    assert b.landmark_mean == pytest.approx(0.0, abs=1e-12)   # index-matched, exact
    assert a.landmark_mean > 0.05
    assert b.chamfer < a.chamfer / 2.0


def test_translated_subject_is_matched_by_no_mirror_at_all():
    """Right ear = left ear translated: A nails it, the mirror configs do not."""
    ears = _prepared(mirrored=False)

    a, _, _ = check_mirror.compare_ears(ears, _setting("A"))
    b, _, _ = check_mirror.compare_ears(ears, _setting("B"))

    assert a.landmark_mean == pytest.approx(0.0, abs=1e-12)
    assert b.landmark_mean > 0.05
    assert a.chamfer < b.chamfer / 2.0


def test_configs_b_and_c_are_indistinguishable():
    """Flipping Y on both clouds instead of one is an isometry: an exact tie."""
    ears = _prepared(mirrored=True)

    b, b_clouds, _ = check_mirror.compare_ears(ears, _setting("B"))
    c, c_clouds, _ = check_mirror.compare_ears(ears, _setting("C"))

    assert b.chamfer == c.chamfer
    assert b.landmark_mean == c.landmark_mean
    assert b.contours.tolist() == c.contours.tolist()
    # ... and they really are different canonical frames, not the same numbers.
    assert not np.allclose(b_clouds["left"], c_clouds["left"])


def test_a_misordered_contour_shows_up_in_its_own_row():
    """One contour reversed on the right side: only that contour's mean moves."""
    _, landmarks = _mirror_world("P0001", mirrored=True)
    landmarks["right"][25:55] = landmarks["right"][25:55][::-1]

    ears = _prepared(mirrored=True, landmarks=landmarks)
    metrics, _, _ = check_mirror.compare_ears(ears, _setting("B"))

    outer, concha, inner, superior = metrics.contours
    assert concha > 0.1
    assert max(outer, inner, superior) == pytest.approx(0.0, abs=1e-12)


def test_prepare_ear_refuses_an_empty_or_suspicious_crop():
    vertices, landmarks = _mirror_world("P0001", mirrored=True)
    raw = RawSubject("P0001", vertices, None, None, Path("P0001"))

    with pytest.raises(ValueError, match="suspicious crop"):
        check_mirror.prepare_ear(raw, "left",
                                 CropConfig("left", LEFT_LO, LEFT_HI, 10 ** 9),
                                 landmarks["left"])
    empty = CropConfig("left", np.array([100.0, 100.0, 100.0]),
                       np.array([110.0, 110.0, 110.0]), 1)
    with pytest.raises(ValueError, match="no vertices"):
        check_mirror.prepare_ear(raw, "left", empty, landmarks["left"])


def test_main_keeps_the_default_on_mirrored_data_and_writes_the_plot(
    tmp_path, monkeypatch, capsys
):
    subjects = ["P0001", "P0002", "P0003"]
    _fake_mirror_dataset(monkeypatch, subjects, mirrored=True)

    code = _run_mirror(tmp_path, subjects, extra=["--out", str(tmp_path / "outputs")])
    out = capsys.readouterr().out

    assert code == 0
    assert "subjects compared: 3" in out
    assert "winner: B = C" in out
    assert "is among the winners" in out
    assert "THE CURRENT DEFAULT IS NOT THE WINNER" not in out
    assert (tmp_path / "outputs" / check_mirror.PLOT_NAME).is_file()


def test_main_shouts_when_the_default_loses_but_changes_nothing(
    tmp_path, monkeypatch, capsys
):
    subjects = ["P0001", "P0002"]
    _fake_mirror_dataset(monkeypatch, subjects, mirrored=False)

    code = _run_mirror(tmp_path, subjects, extra=["--no-plot"])
    out = capsys.readouterr().out

    assert code == 0                       # a finding, not a broken run
    assert "THE CURRENT DEFAULT IS NOT THE WINNER" in out
    assert "winner: A" in out
    assert "changes NOTHING" in out
    # The pipeline's own defaults are untouched by the run.
    params = inspect.signature(canonicalize_ear).parameters
    assert params["mirror_side"].default == "right"
    assert params["mirror_axis"].default == 1


def test_main_reports_the_margin_over_the_runner_up(tmp_path, monkeypatch, capsys):
    subjects = ["P0001"]
    _fake_mirror_dataset(monkeypatch, subjects, mirrored=True)

    _run_mirror(tmp_path, subjects, extra=["--no-plot"])
    out = capsys.readouterr().out

    assert "runner-up: A" in out and "margin" in out
    assert f"{check_mirror.CHAMFER_POINTS}-point" in out   # subsampling is stated
    assert "per-contour mean landmark distance" in out


def test_main_names_the_offending_contour_and_the_ratio(tmp_path, monkeypatch, capsys):
    """A contour reversed on the right side: B still wins, but not at zero."""
    def reverse_concha(landmarks):
        landmarks["right"][25:55] = landmarks["right"][25:55][::-1]

    subjects = ["P0001", "P0002"]
    _fake_mirror_dataset(monkeypatch, subjects, mirrored=True, tweak=reverse_concha)

    code = _run_mirror(tmp_path, subjects, extra=["--no-plot"])
    out = capsys.readouterr().out
    per_contour = out.split("per-contour mean landmark distance")[1]
    rows = {name: float(line.split()[-1])
            for name, _, _ in check_mirror.CONTOURS
            for line in per_contour.splitlines() if line.strip().startswith(name)}

    assert code == 0
    assert "winner: B = C" in out and "x worse" in out
    assert rows["concha outline"] > 0.1
    assert max(rows["outer helix"], rows["inner helix"],
               rows["superior antihelix"]) < 1e-9


def test_main_fails_when_a_listed_subject_cannot_be_checked(
    tmp_path, monkeypatch, capsys
):
    subjects = ["P0001", "P0002"]
    _fake_mirror_dataset(monkeypatch, subjects, mirrored=True, failing={"P0002"})

    code = _run_mirror(tmp_path, subjects, extra=["--no-plot"])
    out = capsys.readouterr().out

    assert code == 1
    assert "P0002" in out and "INCOMPLETE" in out


def test_main_allow_failures_lets_the_verdict_stand(tmp_path, monkeypatch, capsys):
    subjects = ["P0001", "P0002"]
    _fake_mirror_dataset(monkeypatch, subjects, mirrored=True, failing={"P0002"})

    code = _run_mirror(tmp_path, subjects, extra=["--no-plot", "--allow-failures"])
    out = capsys.readouterr().out

    assert code == 0
    assert "complete (--allow-failures)" in out


def test_main_limit_does_not_invent_missing_subjects(tmp_path, monkeypatch, capsys):
    subjects = ["P0001", "P0002"]
    _fake_mirror_dataset(monkeypatch, subjects, mirrored=True)
    # P0009 is listed but out of scope: --limit 1 must not report it as missing.
    listed = ["P0001", "P0009"]
    code = check_mirror.main([
        "--root", str(tmp_path),
        "--subject-list", str(_subject_list(tmp_path, listed)),
        "--crop-config", str(_write_crop_yaml(tmp_path, min_vertices=10)),
        "--n-points", "512", "--no-plot", "--limit", "1",
    ])
    out = capsys.readouterr().out

    assert code == 0
    assert "P0009" not in out
    assert "--limit 1" in out


def test_main_plots_the_winning_configuration_not_the_default(
    tmp_path, monkeypatch, capsys
):
    """The figure must show whichever configuration won, default or not."""
    subjects = ["P0001"]
    _fake_mirror_dataset(monkeypatch, subjects, mirrored=False)   # A wins here
    seen = {}

    def spy(out_path, subject_id, setting, clouds, landmarks, metrics):
        seen["key"] = setting.key
        seen["landmark_mean"] = metrics.landmark_mean
        return out_path

    monkeypatch.setattr(check_mirror, "write_mirror_plot", spy)
    _run_mirror(tmp_path, subjects)

    assert seen["key"] == "A"
    assert seen["landmark_mean"] == pytest.approx(0.0, abs=1e-12)


def test_a_plotting_failure_does_not_swallow_the_verdict(tmp_path, monkeypatch, capsys):
    """The numbers cost a full pass over the dataset; a dead backend must not eat them."""
    subjects = ["P0001"]
    _fake_mirror_dataset(monkeypatch, subjects, mirrored=True)

    def explode(*args, **kwargs):
        raise RuntimeError("no display for you")

    monkeypatch.setattr(check_mirror, "_draw_mirror_plot", explode)
    code = _run_mirror(tmp_path, subjects, extra=["--out", str(tmp_path / "o")])
    out = capsys.readouterr().out

    assert code == 0
    assert "MIRROR VERDICT" in out and "winner:" in out
    assert "plot skipped" in out and "no display for you" in out


def test_main_skips_the_plot_for_an_unknown_subject(tmp_path, monkeypatch, capsys):
    subjects = ["P0001"]
    _fake_mirror_dataset(monkeypatch, subjects, mirrored=True)

    code = _run_mirror(tmp_path, subjects,
                       extra=["--plot-subject", "P0404", "--out", str(tmp_path / "o")])
    out = capsys.readouterr().out

    assert code == 0
    assert "plot skipped" in out
    assert not (tmp_path / "o" / check_mirror.PLOT_NAME).exists()


def test_main_refuses_a_rejected_crop_config(tmp_path, monkeypatch, capsys):
    subjects = ["P0001"]
    _fake_mirror_dataset(monkeypatch, subjects, mirrored=True)
    cfg = _write_crop_yaml(tmp_path, min_vertices=10)
    doc = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    doc["freeze_criterion_passed"] = False
    cfg.write_text(yaml.safe_dump(doc), encoding="utf-8")

    code = check_mirror.main([
        "--root", str(tmp_path),
        "--subject-list", str(_subject_list(tmp_path, subjects)),
        "--crop-config", str(cfg), "--no-plot",
    ])

    assert code == 1
    assert "could not load the crop config" in capsys.readouterr().out


def test_check_mirror_default_config_matches_the_pipeline():
    """Configuration B must be canonicalize_ear's default, or the run lies."""
    params = inspect.signature(canonicalize_ear).parameters
    default = next(c for c in check_mirror.CONFIGS
                   if c.key == check_mirror.DEFAULT_KEY)

    assert default.mirror_side == params["mirror_side"].default
    assert default.mirror_axis == params["mirror_axis"].default

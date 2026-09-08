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

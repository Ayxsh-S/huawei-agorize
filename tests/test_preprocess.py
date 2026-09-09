"""Tests for ``scripts/preprocess.py`` — the cache builder.

Synthetic subjects only: either monkeypatched loaders (single worker) or a
real on-disk synthetic root in ``tmp_path`` (multi-worker, where monkeypatches
cannot cross the process boundary). No Huawei data is touched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import preprocess  # noqa: E402
from preprocess import (  # noqa: E402
    TARGET_TOL,
    TargetRoundTripError,
    canonical_targets,
    ear_seed,
)
from src.cache import cache_path, list_cached, load_cached_ear  # noqa: E402
from src.geometry import (  # noqa: E402
    N_LANDMARKS,
    EarTransform,
    inverse_transform_points,
    make_transform,
)
from test_data import _write_csv, write_ascii_ply  # noqa: E402
from test_scripts import (  # noqa: E402
    LEFT_HI,
    LEFT_LO,
    RIGHT_HI,
    RIGHT_LO,
    _fake_ears,
    _fake_landmarks,
    _grid,
    _subject_list,
    _write_crop_yaml,
)

SUBJECTS = ["P0001", "P0002", "P0005"]
N_POINTS = 32   # a fake ear has N_CROP crop vertices, plenty for 32 without replacement
# 11**3 grid points per side, plus the 11*11 face at Y=0 that the two synthetic
# boxes share (LEFT Y in [0, 10], RIGHT Y in [-10, 0]; crop bounds are inclusive).
N_CROP = 11 ** 3 + 11 * 11


def _run(tmp_path, subjects=SUBJECTS, extra=(), out="cache"):
    return preprocess.main([
        "--subject-list", str(_subject_list(tmp_path, subjects)),
        "--out", str(tmp_path / out),
        "--crop", str(_write_crop_yaml(tmp_path)),
        "--n-points", str(N_POINTS),
        "--root", str(tmp_path / "root"),
        *extra,
    ])


# --- per-ear seed ----------------------------------------------------------- #

def test_ear_seed_is_stable_distinct_and_int64_sized():
    a = ear_seed(0, "P0001", "left")
    assert a == ear_seed(0, "P0001", "left")                 # deterministic
    assert a != ear_seed(0, "P0001", "right")                # the two ears differ
    assert a != ear_seed(0, "P0002", "left")                 # subjects differ
    assert a != ear_seed(1, "P0001", "left")                 # the run seed matters
    assert 0 <= a < 2 ** 63
    # Pinned: changing the derivation silently changes every cached sample, so
    # it must be a deliberate, visible decision.
    assert ear_seed(0, "P0001", "left") == 1448645556787510418
    with pytest.raises(ValueError, match="side"):
        ear_seed(0, "P0001", "up")


# --- target verification ---------------------------------------------------- #

def test_canonical_targets_are_exact_and_go_through_the_serialised_transform(monkeypatch):
    transform = make_transform(_grid(LEFT_LO, LEFT_HI), "right", "right", 1)
    gt = _fake_landmarks(seed=5)
    seen = []
    original = EarTransform.from_dict
    monkeypatch.setattr(EarTransform, "from_dict", classmethod(lambda cls, d: seen.append(d) or original(d)))

    canonical, err = canonical_targets(gt, transform)

    assert seen == [transform.to_dict()]
    assert err < TARGET_TOL
    assert np.allclose(inverse_transform_points(canonical, transform), gt, atol=1e-12)
    assert canonical.dtype == np.float64 and canonical.shape == (N_LANDMARKS, 3)


def test_canonical_targets_refuse_an_inexact_inverse(monkeypatch):
    transform = make_transform(_grid(LEFT_LO, LEFT_HI), "left", "right", 1)
    real = preprocess.inverse_transform_points
    monkeypatch.setattr(preprocess, "inverse_transform_points",
                        lambda p, t: real(p, t) + 1e-3)

    with pytest.raises(TargetRoundTripError, match="1.000e-03"):
        canonical_targets(_fake_landmarks(seed=1), transform)


# --- main: single worker with fake loaders --------------------------------- #

def test_main_writes_both_ears_for_every_subject(tmp_path, monkeypatch, capsys):
    _fake_ears(monkeypatch, preprocess, SUBJECTS)

    assert _run(tmp_path) == 0

    listed = list_cached(tmp_path / "cache")
    assert [(s, d) for s, d, _ in listed] == [
        (s, d) for s in SUBJECTS for d in ("left", "right")
    ]
    for sid, side, path in listed:
        ear = load_cached_ear(path)
        assert (ear.subject_id, ear.side) == (sid, side)
        assert ear.points.shape == (N_POINTS, 3)
        assert ear.qa["n_vertices"] == N_CROP and ear.qa["suspicious"] is False
        assert ear.qa["ear_seed"] == ear_seed(0, sid, side)
        assert ear.qa["seed"] == 0 and ear.qa["n_points"] == N_POINTS
        assert ear.qa["split_file"].endswith("ids.txt")
        assert len(ear.qa["crop_config_sha256"]) == 64
        assert "targets" not in ear.qa                       # not requested
        assert ear.transform.mirror_axis == (1 if side == "right" else None)
    out = capsys.readouterr().out
    assert "ears written: 6" in out
    assert "ears skipped (already cached, same settings): 0" in out
    assert "=> OK" in out
    assert f"min {N_CROP}   median {N_CROP}   max {N_CROP}" in out


def test_main_with_targets_stores_verified_canonical_gt(tmp_path, monkeypatch, capsys):
    landmarks = {
        sid: {"left": _fake_landmarks(seed=i),
              "right": _fake_landmarks(seed=i, lo=RIGHT_LO, hi=RIGHT_HI)}
        for i, sid in enumerate(SUBJECTS)
    }
    _fake_ears(monkeypatch, preprocess, SUBJECTS, landmarks=landmarks)

    assert _run(tmp_path, extra=["--with-targets"]) == 0

    for sid, side, path in list_cached(tmp_path / "cache"):
        ear = load_cached_ear(path)
        gt = landmarks[sid][side]
        # Transform is mesh-derived: identical to make_transform on the crop alone.
        crop = _grid(LEFT_LO, LEFT_HI) if side == "left" else _grid(RIGHT_LO, RIGHT_HI)
        expected = make_transform(crop, side, "right", 1)
        assert np.array_equal(ear.transform.centre, expected.centre)
        assert ear.transform.scale == expected.scale
        restored = inverse_transform_points(ear.qa["targets"], ear.transform)
        assert np.max(np.abs(restored - gt)) < 1e-4       # float32 storage of targets
    assert "every stored target inverts exactly" in capsys.readouterr().out


def test_main_aborts_naming_the_subject_when_targets_do_not_invert(tmp_path, monkeypatch, capsys):
    _fake_ears(monkeypatch, preprocess, SUBJECTS)
    real = preprocess.inverse_transform_points
    monkeypatch.setattr(preprocess, "inverse_transform_points",
                        lambda p, t: real(p, t) + 1e-6)

    assert _run(tmp_path, extra=["--with-targets"]) == 1

    out = capsys.readouterr().out
    assert "ABORTED on P0001 left" in out
    assert not (tmp_path / "cache" / "P0001_left.npz").exists()   # nothing written for it
    assert list_cached(tmp_path / "cache") == []                   # and the run stopped there


def test_main_skips_existing_files_unless_overwrite(tmp_path, monkeypatch, capsys):
    _fake_ears(monkeypatch, preprocess, SUBJECTS)
    assert _run(tmp_path) == 0
    first = {p: p.stat().st_mtime_ns for _, _, p in list_cached(tmp_path / "cache")}
    p = tmp_path / "cache" / "P0002_right.npz"
    p.unlink()

    assert _run(tmp_path) == 0
    out = capsys.readouterr().out
    assert "ears written: 1" in out and "skipped (already cached, same settings): 5" in out
    for path, mtime in first.items():
        if path != p:
            assert path.stat().st_mtime_ns == mtime           # untouched

    assert _run(tmp_path, extra=["--overwrite"]) == 0
    out = capsys.readouterr().out
    assert "ears written: 6" in out and "skipped (already cached, same settings): 0" in out


def test_main_rebuild_is_bit_identical(tmp_path, monkeypatch):
    _fake_ears(monkeypatch, preprocess, SUBJECTS)
    assert _run(tmp_path, extra=["--with-targets"]) == 0
    before = {p.name: p.read_bytes() for _, _, p in list_cached(tmp_path / "cache")}

    assert _run(tmp_path, extra=["--with-targets", "--overwrite"]) == 0

    after = {p.name: p.read_bytes() for _, _, p in list_cached(tmp_path / "cache")}
    assert after == before


def test_main_treats_a_stale_cache_entry_as_a_failure(tmp_path, monkeypatch, capsys):
    _fake_ears(monkeypatch, preprocess, SUBJECTS)
    assert _run(tmp_path) == 0

    # Different seed -> the existing files were built with other settings.
    assert _run(tmp_path, extra=["--seed", "1"]) == 1
    out = capsys.readouterr().out
    assert "stale cache entry" in out and "seed=0" in out and "ears failed:  6" in out

    # Same settings but now wanting targets the files do not carry.
    assert _run(tmp_path, extra=["--with-targets"]) == 1
    assert "no targets" in capsys.readouterr().out

    # And the mirror image: a label-free run must not inherit GT from a
    # --with-targets cache in the same directory.
    assert _run(tmp_path, extra=["--with-targets", "--overwrite"]) == 0
    capsys.readouterr()
    assert _run(tmp_path) == 1
    out = capsys.readouterr().out
    assert "carries targets" in out and "ears failed:  6" in out


def test_main_fails_a_suspicious_crop_and_writes_nothing_for_it(tmp_path, monkeypatch, capsys):
    _fake_ears(monkeypatch, preprocess, SUBJECTS)
    crop = _write_crop_yaml(tmp_path, min_vertices=N_CROP + 1)   # every crop is "suspicious"

    rc = preprocess.main(["--subject-list", str(_subject_list(tmp_path, SUBJECTS)),
                          "--out", str(tmp_path / "cache"), "--crop", str(crop),
                          "--n-points", str(N_POINTS), "--root", str(tmp_path / "root")])

    assert rc == 1
    out = capsys.readouterr().out
    assert list_cached(tmp_path / "cache") == []
    assert f"P0001 left: suspicious crop: {N_CROP} vertices < min_vertices {N_CROP + 1}" in out
    assert "suspicious crops (below CropConfig.min_vertices; FAILED, not written): 6  P0001 left, P0001 right" in out
    assert "=> FAIL" in out


def test_main_names_ears_sampled_with_replacement(tmp_path, monkeypatch, capsys):
    _fake_ears(monkeypatch, preprocess, ["P0001"])

    assert _run(tmp_path, subjects=["P0001"], extra=["--n-points", str(N_CROP + 1)]) == 0

    out = capsys.readouterr().out
    assert "WITH replacement (crop smaller than n_points): 2  P0001 left, P0001 right" in out
    assert "!! written, but those ears repeat points" in out
    ear = load_cached_ear(cache_path(tmp_path / "cache", "P0001", "left"))
    assert ear.qa["sampled_with_replacement"] is True and ear.points.shape == (N_CROP + 1, 3)


def test_main_sweeps_temp_files_left_by_an_interrupted_write(tmp_path, monkeypatch, capsys):
    _fake_ears(monkeypatch, preprocess, ["P0001"])
    cache = tmp_path / "cache"
    cache.mkdir()
    stray = cache / ".P0009_left.npz.4242.tmp"
    stray.write_bytes(b"partial")

    assert _run(tmp_path, subjects=["P0001"]) == 0

    assert not stray.exists()
    assert sorted(p.name for p in cache.iterdir()) == ["P0001_left.npz", "P0001_right.npz"]


def test_main_fails_when_a_listed_subject_has_no_mesh_or_fails_to_load(tmp_path, monkeypatch, capsys):
    _fake_ears(monkeypatch, preprocess, ["P0001", "P0002"], failing={"P0002"})

    assert _run(tmp_path, subjects=["P0001", "P0002", "P0009"]) == 1

    out = capsys.readouterr().out
    assert "P0009 left: no mesh" in out
    assert "P0002 left: load failed: OSError" in out
    assert "ears written: 2" in out and "ears failed:  4" in out
    assert "=> FAIL" in out
    assert [(s, d) for s, d, _ in list_cached(tmp_path / "cache")] == [
        ("P0001", "left"), ("P0001", "right")
    ]


def test_main_prints_progress_every_25_subjects(tmp_path, monkeypatch, capsys):
    many = [f"P{i:04d}" for i in range(1, 27)]
    _fake_ears(monkeypatch, preprocess, many)

    assert _run(tmp_path, subjects=many) == 0

    out = capsys.readouterr().out
    assert "... 25/26 subjects  (written 50," in out
    assert "... 26/26 subjects  (written 52," in out
    assert "... 24/26" not in out


def test_main_limit_does_not_invent_missing_subjects(tmp_path, monkeypatch, capsys):
    _fake_ears(monkeypatch, preprocess, ["P0001"])

    assert _run(tmp_path, subjects=["P0001", "P0404"], extra=["--limit", "1"]) == 0
    assert "no mesh" not in capsys.readouterr().out


def test_main_refuses_a_rejected_crop_config_and_a_missing_list(tmp_path, monkeypatch, capsys):
    _fake_ears(monkeypatch, preprocess, SUBJECTS)
    crop = _write_crop_yaml(tmp_path)
    crop.write_text(crop.read_text().replace("freeze_criterion_passed: true",
                                             "freeze_criterion_passed: false"))
    rc = preprocess.main(["--subject-list", str(_subject_list(tmp_path, SUBJECTS)),
                          "--out", str(tmp_path / "cache"), "--crop", str(crop)])
    assert rc == 1 and "freeze_criterion_passed" in capsys.readouterr().out

    rc = preprocess.main(["--subject-list", str(tmp_path / "missing.txt"),
                          "--out", str(tmp_path / "cache"), "--crop", str(_write_crop_yaml(tmp_path))])
    assert rc == 1


def test_main_records_the_sha_of_the_crop_config_it_used(tmp_path, monkeypatch):
    _fake_ears(monkeypatch, preprocess, ["P0001"])
    crop = _write_crop_yaml(tmp_path)
    import hashlib
    expected = hashlib.sha256(crop.read_bytes()).hexdigest()

    assert _run(tmp_path, subjects=["P0001"]) == 0

    ear = load_cached_ear(cache_path(tmp_path / "cache", "P0001", "left"))
    assert ear.qa["crop_config_sha256"] == expected


# --- main: real synthetic root on disk, two worker processes ---------------- #

def _synthetic_root(tmp_path: Path, subjects: list[str]) -> dict[str, dict[str, np.ndarray]]:
    root = tmp_path / "root"
    (root / "mesh").mkdir(parents=True)
    (root / "landmarks").mkdir()
    vertices = np.vstack([_grid(LEFT_LO, LEFT_HI), _grid(RIGHT_LO, RIGHT_HI)])
    landmarks = {}
    for i, sid in enumerate(subjects):
        write_ascii_ply(root / "mesh" / f"{sid}.ply", vertices)
        lm = {"left": _fake_landmarks(seed=10 + i),
              "right": _fake_landmarks(seed=10 + i, lo=RIGHT_LO, hi=RIGHT_HI)}
        for side in ("left", "right"):
            _write_csv(root / "landmarks" / f"{sid}_{side}_ear_landmarks.csv", lm[side])
        landmarks[sid] = lm
    return landmarks


def test_main_with_two_workers_matches_a_single_worker_run(tmp_path, capsys):
    _synthetic_root(tmp_path, SUBJECTS)

    assert _run(tmp_path, extra=["--with-targets", "--workers", "2"], out="par") == 0
    assert _run(tmp_path, extra=["--with-targets", "--workers", "1"], out="seq") == 0

    par = {p.name: p.read_bytes() for _, _, p in list_cached(tmp_path / "par")}
    seq = {p.name: p.read_bytes() for _, _, p in list_cached(tmp_path / "seq")}
    assert set(par) == {f"{s}_{d}.npz" for s in SUBJECTS for d in ("left", "right")}
    assert par == seq
    for _, _, p in list_cached(tmp_path / "par"):
        ear = load_cached_ear(p)
        assert ear.qa["targets"].shape == (85, 3)
        assert ear.qa["n_vertices"] == N_CROP
    assert "ears written: 6" in capsys.readouterr().out


def test_script_entry_point_with_two_workers_reports_a_worker_failure(tmp_path):
    """The real ``python scripts/preprocess.py --workers 2`` path.

    Under pytest the module is imported normally; run as a script it is
    ``__main__`` and spawn re-executes it as ``__mp_main__`` in each child, with
    ``Job`` / ``EarResult`` pickled across that boundary. A corrupt PLY makes
    one subject fail inside a worker, so the failure crosses it too.
    """
    import subprocess

    _synthetic_root(tmp_path, SUBJECTS)
    (tmp_path / "root" / "mesh" / "P0002.ply").write_bytes(b"not a ply\n")
    script = Path(__file__).resolve().parents[1] / "scripts" / "preprocess.py"

    proc = subprocess.run(
        [sys.executable, str(script),
         "--subject-list", str(_subject_list(tmp_path, SUBJECTS)),
         "--out", str(tmp_path / "cache"), "--crop", str(_write_crop_yaml(tmp_path)),
         "--n-points", str(N_POINTS), "--root", str(tmp_path / "root"),
         "--with-targets", "--workers", "2"],
        capture_output=True, text=True, timeout=120,
    )

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "ears written: 4" in proc.stdout and "ears failed:  2" in proc.stdout
    assert "P0002 left: load failed" in proc.stdout
    assert [(s, d) for s, d, _ in list_cached(tmp_path / "cache")] == [
        ("P0001", "left"), ("P0001", "right"), ("P0005", "left"), ("P0005", "right")
    ]
    assert [p.name for p in (tmp_path / "cache").iterdir() if p.name.endswith(".tmp")] == []

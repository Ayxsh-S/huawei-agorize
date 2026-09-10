"""Tests for ``src/cache.py`` — the npz schema Roles B and C read through.

Synthetic ears only, written to ``tmp_path``; no Huawei data is touched.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.cache import (
    NO_MIRROR,
    REQUIRED_KEYS,
    SCHEMA_VERSION,
    cache_path,
    list_cached,
    load_cached_ear,
    parse_cache_filename,
    transform_from_arrays,
    transform_to_arrays,
    write_cached_ear,
)
from src.data import RawSubject
from src.geometry import (
    N_LANDMARKS,
    CanonicalEar,
    CropConfig,
    EarTransform,
    canonicalize_ear,
    inverse_transform_points,
)

LO, HI = np.array([0.0, 0.0, 0.0]), np.array([10.0, 10.0, 20.0])
PROVENANCE = dict(
    seed=0, ear_seed=123456789012345, crop_config_sha256="c" * 64,
    split_file="splits/train_ids.txt", split_sha256="s" * 64,
)


def _grid(n: int = 11) -> np.ndarray:
    axes = [np.linspace(LO[a], HI[a], n) for a in range(3)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def _ear(subject_id: str = "P0001", side: str = "left", n_points: int = 64,
         seed: int = 0) -> CanonicalEar:
    raw = RawSubject(subject_id, _grid(), None, None, Path("<synthetic>"))
    cfg = CropConfig(side, LO, HI, min_vertices=10)
    return canonicalize_ear(raw, side, cfg, n_points=n_points, seed=seed)


def _targets(seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(N_LANDMARKS, 3))


# --- transform round trip --------------------------------------------------- #

@pytest.mark.parametrize("mirror_axis", [None, 0, 1, 2])
def test_transform_arrays_round_trip_exactly(mirror_axis):
    t = EarTransform(centre=[1.1234567890123456, -2.5, 1e-7], scale=31.4159265358979,
                     mirror_axis=mirror_axis)

    arrays = transform_to_arrays(t)
    rebuilt = transform_from_arrays(arrays)

    assert arrays["transform_centre"].dtype == np.float64
    assert arrays["transform_scale"].dtype == np.float64
    assert arrays["transform_mirror_axis"].dtype == np.int64
    assert int(arrays["transform_mirror_axis"]) == (NO_MIRROR if mirror_axis is None else mirror_axis)
    assert np.array_equal(rebuilt.centre, t.centre)       # bit-exact, not approx
    assert rebuilt.scale == t.scale
    assert rebuilt.mirror_axis == t.mirror_axis


# --- write / load ----------------------------------------------------------- #

def test_write_then_load_reproduces_the_ear(tmp_path):
    ear = _ear("P0007", "right")
    path = write_cached_ear(cache_path(tmp_path, "P0007", "right"), ear, **PROVENANCE)

    assert path == tmp_path / "P0007_right.npz"
    loaded = load_cached_ear(path)

    assert isinstance(loaded, CanonicalEar)
    assert (loaded.subject_id, loaded.side) == ("P0007", "right")
    assert loaded.normals is None
    assert loaded.points.dtype == np.float64 and loaded.points.shape == (64, 3)
    # Stored as float32: the values must be float32-exact copies of the originals.
    assert np.array_equal(loaded.points, ear.points.astype(np.float32).astype(np.float64))
    assert np.array_equal(loaded.transform.centre, ear.transform.centre)
    assert loaded.transform.scale == ear.transform.scale
    assert loaded.transform.mirror_axis == ear.transform.mirror_axis == 1   # right is mirrored
    assert loaded.qa["n_vertices"] == ear.qa["n_vertices"] == 1331
    assert loaded.qa["suspicious"] is False
    assert loaded.qa["sampled_with_replacement"] is False
    assert loaded.qa["n_points"] == 64
    assert loaded.qa["seed"] == 0
    assert loaded.qa["ear_seed"] == PROVENANCE["ear_seed"]
    assert loaded.qa["crop_config_sha256"] == "c" * 64
    assert loaded.qa["split_file"] == "splits/train_ids.txt"
    assert loaded.qa["split_sha256"] == "s" * 64
    assert loaded.qa["schema_version"] == SCHEMA_VERSION
    assert "targets" not in loaded.qa


def test_file_holds_exactly_the_documented_keys_and_dtypes(tmp_path):
    ear = _ear()
    path = write_cached_ear(tmp_path / "P0001_left.npz", ear, targets=_targets(), **PROVENANCE)

    with np.load(path, allow_pickle=False) as npz:
        assert set(npz.files) == set(REQUIRED_KEYS) | {"targets"}
        assert npz["points"].dtype == np.float32 and npz["points"].shape == (64, 3)
        assert npz["targets"].dtype == np.float32 and npz["targets"].shape == (85, 3)
        assert npz["transform_centre"].dtype == np.float64
        assert npz["transform_scale"].dtype == np.float64
        assert npz["transform_mirror_axis"].dtype == np.int64
        assert int(npz["transform_mirror_axis"]) == NO_MIRROR       # left is not mirrored
        assert npz["n_crop_vertices"].dtype == np.int64
        assert npz["qa_suspicious"].dtype == np.bool_
        assert npz["n_points"].dtype == np.int64 and npz["seed"].dtype == np.int64
        assert npz["ear_seed"].dtype == np.int64
        for key in ("subject_id", "side", "crop_config_sha256", "split_file",
                    "split_sha256", "schema_version"):
            assert npz[key].dtype.kind == "U" and npz[key].ndim == 0
        assert str(npz["schema_version"]) == "1"


def test_targets_are_stored_only_when_given_and_invert_through_the_loaded_transform(tmp_path):
    ear = _ear("P0002", "right")
    gt = LO + (HI - LO) * np.random.default_rng(3).random((N_LANDMARKS, 3))
    canonical = (gt - ear.transform.centre) / ear.transform.scale
    canonical[:, 1] *= -1.0

    write_cached_ear(tmp_path / "P0002_right.npz", ear, targets=canonical, **PROVENANCE)
    loaded = load_cached_ear(tmp_path / "P0002_right.npz")

    assert loaded.qa["targets"].shape == (85, 3) and loaded.qa["targets"].dtype == np.float64
    restored = inverse_transform_points(loaded.qa["targets"], loaded.transform)
    # The stored targets are float32, so the inverse is float32-exact, not 1e-9.
    assert np.max(np.abs(restored - gt)) < 1e-4


def test_write_is_atomic_and_leaves_no_temp_file(tmp_path):
    write_cached_ear(tmp_path / "P0001_left.npz", _ear(), **PROVENANCE)
    assert [p.name for p in tmp_path.iterdir()] == ["P0001_left.npz"]


def test_write_refuses_bad_targets_and_bad_ears(tmp_path):
    ear = _ear()
    with pytest.raises(ValueError, match="targets"):
        write_cached_ear(tmp_path / "a.npz", ear, targets=np.zeros((84, 3)), **PROVENANCE)
    with pytest.raises(ValueError, match="non-finite"):
        bad = _targets(); bad[0, 0] = np.nan
        write_cached_ear(tmp_path / "a.npz", ear, targets=bad, **PROVENANCE)
    ear_without_qa = CanonicalEar("P0001", "left", ear.points, None, ear.transform, qa={})
    with pytest.raises(ValueError, match="qa"):
        write_cached_ear(tmp_path / "a.npz", ear_without_qa, **PROVENANCE)
    assert list(tmp_path.iterdir()) == []       # nothing half-written


def test_load_rejects_a_wrong_schema_version(tmp_path):
    path = write_cached_ear(tmp_path / "P0001_left.npz", _ear(), **PROVENANCE)
    with np.load(path, allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}
    arrays["schema_version"] = np.asarray("0")
    with open(path, "wb") as fh:
        np.savez(fh, **arrays)

    with pytest.raises(ValueError, match="schema_version"):
        load_cached_ear(path)


def test_load_rejects_a_missing_key(tmp_path):
    path = write_cached_ear(tmp_path / "P0001_left.npz", _ear(), **PROVENANCE)
    with np.load(path, allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files if k != "ear_seed"}
    with open(path, "wb") as fh:
        np.savez(fh, **arrays)

    with pytest.raises(ValueError, match="ear_seed"):
        load_cached_ear(path)


def test_load_rejects_a_renamed_file(tmp_path):
    path = write_cached_ear(tmp_path / "P0001_left.npz", _ear("P0001", "left"), **PROVENANCE)
    renamed = tmp_path / "P0002_right.npz"
    path.rename(renamed)

    with pytest.raises(ValueError, match="filename"):
        load_cached_ear(renamed)

    # A file that does not follow the naming convention is loaded on its content.
    other = tmp_path / "whatever.npz"
    renamed.rename(other)
    assert load_cached_ear(other).subject_id == "P0001"


def test_load_rejects_a_point_count_that_disagrees_with_n_points(tmp_path):
    path = write_cached_ear(tmp_path / "P0001_left.npz", _ear(), **PROVENANCE)
    with np.load(path, allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}
    arrays["points"] = arrays["points"][:-1]
    with open(path, "wb") as fh:
        np.savez(fh, **arrays)

    with pytest.raises(ValueError, match="points"):
        load_cached_ear(path)


# --- listing ---------------------------------------------------------------- #

def test_list_cached_returns_sorted_ears_and_ignores_strays(tmp_path):
    for sid, side in [("P0010", "right"), ("P0001", "right"), ("P0001", "left")]:
        write_cached_ear(cache_path(tmp_path, sid, side), _ear(sid, side), **PROVENANCE)
    (tmp_path / ".P0002_left.npz.123.tmp").write_bytes(b"partial")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "P0003.npz").write_bytes(b"x")        # no side
    (tmp_path / "subdir").mkdir()

    listed = list_cached(tmp_path)

    assert [(s, d) for s, d, _ in listed] == [
        ("P0001", "left"), ("P0001", "right"), ("P0010", "right")
    ]
    assert all(p == tmp_path / f"{s}_{d}.npz" for s, d, p in listed)
    for s, d, p in listed:
        ear = load_cached_ear(p)
        assert (ear.subject_id, ear.side) == (s, d)


def test_list_cached_requires_the_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_cached(tmp_path / "nope")


def test_cache_path_and_filename_parsing():
    assert cache_path("cache", "P0001", "left") == Path("cache") / "P0001_left.npz"
    assert parse_cache_filename("cache/P0001_left.npz") == ("P0001", "left")
    assert parse_cache_filename("cache/P0001_right.npz") == ("P0001", "right")
    assert parse_cache_filename("cache/P0001_left.npy") is None
    assert parse_cache_filename("cache/P0001.npz") is None
    assert parse_cache_filename("cache/.P0001_left.npz.4.tmp") is None
    with pytest.raises(ValueError, match="side"):
        cache_path("cache", "P0001", "up")

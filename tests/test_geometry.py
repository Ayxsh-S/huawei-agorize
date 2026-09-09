"""Geometry invariants for Role A.

All data here is synthetic and built inside the tests — no Huawei data is ever
read (see ``CLAUDE.md``).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from src.data import RawSubject
from src.geometry import (
    CropConfig,
    EarTransform,
    canonicalize_ear,
    crop_ear,
    inverse_transform_points,
    load_crop_config,
    make_transform,
    sample_points,
    transform_points_to_canonical,
)


def _synthetic_points(n: int, seed: int = 0) -> np.ndarray:
    """Random blob of points, deterministic per seed."""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=[10.0, -20.0, 5.0], scale=[3.0, 1.5, 2.0], size=(n, 3))


def _synthetic_subject(n: int = 4000, seed: int = 0) -> RawSubject:
    """Synthetic 'mesh' with normals, no faces."""
    vertices = _synthetic_points(n, seed)
    rng = np.random.default_rng(seed + 1)
    normals = rng.normal(size=(n, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return RawSubject(
        subject_id="synthetic_000",
        vertices=vertices,
        faces=None,
        normals=normals,
        mesh_path="<synthetic>",
    )


@pytest.mark.parametrize("mirror_axis", [None, 0, 1, 2])
def test_transform_round_trip(mirror_axis):
    """canonical -> inverse recovers the original coordinates exactly."""
    original = _synthetic_points(500, seed=7)
    t = EarTransform(centre=np.array([1.0, -2.5, 0.75]), scale=3.25, mirror_axis=mirror_axis)

    canonical = transform_points_to_canonical(original, t)
    recovered = inverse_transform_points(canonical, t)

    assert canonical.shape == original.shape
    assert np.allclose(recovered, original, atol=1e-9)
    # The forward pass must not mutate its input.
    assert np.allclose(original, _synthetic_points(500, seed=7), atol=1e-9)


@pytest.mark.parametrize("mirror_axis", [None, 0, 1, 2])
def test_inverse_then_forward_round_trip(mirror_axis):
    """The inverse is a true two-sided inverse, not just a left inverse."""
    canonical = _synthetic_points(200, seed=11) * 0.1
    t = EarTransform(centre=np.array([-4.0, 8.0, 2.0]), scale=0.5, mirror_axis=mirror_axis)

    original = inverse_transform_points(canonical, t)
    recovered = transform_points_to_canonical(original, t)

    assert np.allclose(recovered, canonical, atol=1e-9)


def test_make_transform_centre_scale_and_mirror():
    """Centre is the bbox centre, scale is half the largest extent."""
    # Explicit corner points: bbox [0,10] x [0,4] x [-2,2].
    crop = np.array(
        [
            [0.0, 0.0, -2.0],
            [10.0, 4.0, 2.0],
            [5.0, 2.0, 0.0],
        ]
    )

    t_left = make_transform(crop, side="left", mirror_side="right", mirror_axis=1)
    assert np.allclose(t_left.centre, [5.0, 2.0, 0.0], atol=1e-9)
    assert t_left.scale == pytest.approx(5.0)  # max extent 10 / 2
    assert t_left.mirror_axis is None          # left is not the mirrored side

    t_right = make_transform(crop, side="right", mirror_side="right", mirror_axis=1)
    assert np.allclose(t_right.centre, t_left.centre, atol=1e-9)
    assert t_right.scale == pytest.approx(t_left.scale)
    assert t_right.mirror_axis == 1

    # A non-default mirror axis is honoured.
    t_z = make_transform(crop, side="right", mirror_side="right", mirror_axis=2)
    assert t_z.mirror_axis == 2

    # The canonical bbox spans [-1, 1] on the longest axis.
    canonical = transform_points_to_canonical(crop, t_left)
    assert canonical[:, 0].min() == pytest.approx(-1.0)
    assert canonical[:, 0].max() == pytest.approx(1.0)


def test_make_transform_is_mesh_derived_only():
    """Shuffling the input points cannot change the transform."""
    crop = _synthetic_points(300, seed=3)
    rng = np.random.default_rng(99)
    shuffled = crop[rng.permutation(crop.shape[0])]

    a = make_transform(crop, "right", "right")
    b = make_transform(shuffled, "right", "right")

    assert np.allclose(a.centre, b.centre, atol=1e-9)
    assert a.scale == pytest.approx(b.scale)
    assert a.mirror_axis == b.mirror_axis


def test_sample_points_upsamples_when_too_few():
    """Fewer input points than n still yields exactly [n, 3]."""
    points = _synthetic_points(37, seed=5)
    normals = np.tile(np.array([0.0, 0.0, 1.0]), (37, 1))

    sampled, sampled_normals = sample_points(points, normals, n=2048, seed=0)

    assert sampled.shape == (2048, 3)
    assert sampled_normals is not None
    assert sampled_normals.shape == (2048, 3)
    # Every sampled ROW came from the input set. np.isin() would only check that
    # each coordinate appears *somewhere* in the input, which a shuffled or
    # recombined row would also pass.
    source_rows = {tuple(row) for row in points}
    assert {tuple(row) for row in sampled} <= source_rows


def test_sample_points_deterministic_and_no_replacement_when_plenty():
    points = _synthetic_points(500, seed=13)

    a, a_n = sample_points(points, None, n=128, seed=42)
    b, _ = sample_points(points, None, n=128, seed=42)
    c, _ = sample_points(points, None, n=128, seed=43)

    assert a.shape == (128, 3)
    assert a_n is None
    assert np.array_equal(a, b)                       # same seed -> same sample
    assert not np.array_equal(a, c)                   # different seed -> different sample
    assert len(np.unique(a, axis=0)) == 128           # no replacement when M >= n


def _two_blob_subject(n_per_blob: int = 3000, seed: int = 31) -> RawSubject:
    """One 'head' whose two ears are well-separated blobs either side of Y=0.

    Left blob sits at Y around -20, right blob at Y around +20, and the two are
    separated by an empty band across Y=0 so a per-side crop box can only ever
    contain one of them. Nothing here comes from real data.
    """
    rng = np.random.default_rng(seed)
    left = rng.normal(loc=[10.0, -20.0, 5.0], scale=[3.0, 1.5, 2.0], size=(n_per_blob, 3))
    right = rng.normal(loc=[10.0, 20.0, 5.0], scale=[3.0, 1.5, 2.0], size=(n_per_blob, 3))
    vertices = np.vstack([left, right])
    normals = rng.normal(size=vertices.shape)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    raw = RawSubject(
        subject_id="synthetic_two_blob",
        vertices=vertices,
        faces=None,
        normals=normals,
        mesh_path="<synthetic>",
    )
    return raw


# Boxes generous on X and Z (so Gaussian tails are not clipped) but tight on Y,
# where the two blobs are separated by an empty band across Y = 0.
_LEFT_BLOB_CFG = CropConfig(
    side="left", lo=[-40.0, -40.0, -45.0], hi=[60.0, -8.0, 55.0], min_vertices=10
)
_RIGHT_BLOB_CFG = CropConfig(
    side="right", lo=[-40.0, 8.0, -45.0], hi=[60.0, 40.0, 55.0], min_vertices=10
)


def test_two_blob_subject_each_side_inverts_into_its_own_blob():
    """Per-side crops stay in their own blob, and the inverse lands back there.

    This is the property the whole canonical pipeline rests on: a prediction made
    in canonical space must come back into the *same* ear it came from.
    """
    raw = _two_blob_subject()

    for side, cfg in (("left", _LEFT_BLOB_CFG), ("right", _RIGHT_BLOB_CFG)):
        ear = canonicalize_ear(raw, side, cfg, mirror_side="right", n_points=512, seed=0)

        assert ear.side == side
        assert ear.points.shape == (512, 3)

        # The canonical points invert back inside this side's crop box ...
        recovered = inverse_transform_points(ear.points, ear.transform)
        assert np.all(recovered >= cfg.lo - 1e-9)
        assert np.all(recovered <= cfg.hi + 1e-9)

        # ... on the correct side of Y=0, i.e. into its own blob, not the other.
        if side == "left":
            assert np.all(recovered[:, 1] < 0.0)
        else:
            assert np.all(recovered[:, 1] > 0.0)

        # ... and they are genuinely rows of that side's cropped vertices.
        crop_points, _, _ = crop_ear(raw, side, cfg)
        crop_rows = {tuple(np.round(row, 9)) for row in crop_points}
        assert {tuple(np.round(row, 9)) for row in recovered} <= crop_rows

    # The two sides really are disjoint point sets.
    left_crop, _, _ = crop_ear(raw, "left", _LEFT_BLOB_CFG)
    right_crop, _, _ = crop_ear(raw, "right", _RIGHT_BLOB_CFG)
    n_per_blob = raw.vertices.shape[0] // 2
    assert left_crop.shape[0] == n_per_blob and right_crop.shape[0] == n_per_blob
    assert np.all(left_crop[:, 1] < 0.0) and np.all(right_crop[:, 1] > 0.0)
    assert not ({tuple(r) for r in left_crop} & {tuple(r) for r in right_crop})


def test_two_blob_subject_rejects_side_config_mismatch():
    """A left box handed to the right side (or vice versa) must raise, not crop."""
    raw = _two_blob_subject()

    # match="side" so an unrelated failure (empty crop, bad shape) cannot pass
    # this test by accident.
    with pytest.raises(ValueError, match="side"):
        crop_ear(raw, "right", _LEFT_BLOB_CFG)
    with pytest.raises(ValueError, match="side"):
        crop_ear(raw, "left", _RIGHT_BLOB_CFG)
    with pytest.raises(ValueError, match="side"):
        canonicalize_ear(raw, "right", _LEFT_BLOB_CFG)
    with pytest.raises(ValueError, match="side"):
        canonicalize_ear(raw, "left", _RIGHT_BLOB_CFG)


def test_ear_transform_dict_round_trip():
    for mirror_axis in (None, 0, 1, 2):
        t = EarTransform(centre=np.array([0.5, -1.5, 2.25]), scale=4.5, mirror_axis=mirror_axis)
        d = t.to_dict()
        back = EarTransform.from_dict(d)

        assert np.allclose(back.centre, t.centre, atol=1e-9)
        assert back.scale == pytest.approx(t.scale)
        assert back.mirror_axis == t.mirror_axis
        assert back.to_dict() == d


def test_crop_ear_qa_and_bounds():
    raw = _synthetic_subject(n=4000, seed=1)
    cfg = CropConfig(side="left", lo=[8.0, -22.0, 3.0], hi=[12.0, -18.0, 7.0], min_vertices=10)

    points, normals, qa = crop_ear(raw, "left", cfg)

    assert points.shape[1] == 3
    assert normals is not None and normals.shape == points.shape
    assert qa["n_vertices"] == points.shape[0]
    assert np.all(points >= cfg.lo) and np.all(points <= cfg.hi)
    assert np.allclose(qa["bbox_lo"], points.min(axis=0), atol=1e-9)
    assert np.allclose(qa["bbox_hi"], points.max(axis=0), atol=1e-9)
    assert np.allclose(qa["centre"], 0.5 * (qa["bbox_lo"] + qa["bbox_hi"]), atol=1e-9)
    assert qa["suspicious"] is False


def test_crop_ear_flags_suspicious_crop():
    raw = _synthetic_subject(n=200, seed=2)
    cfg = CropConfig(side="right", lo=[9.9, -20.1, 4.9], hi=[10.1, -19.9, 5.1], min_vertices=500)

    points, _, qa = crop_ear(raw, "right", cfg)

    assert qa["suspicious"] is True
    assert qa["n_vertices"] == points.shape[0]


def test_crop_ear_rejects_side_mismatch():
    raw = _synthetic_subject(n=100, seed=4)
    cfg = CropConfig(side="left", lo=[0.0, -30.0, 0.0], hi=[20.0, -10.0, 10.0])

    with pytest.raises(ValueError):
        crop_ear(raw, "right", cfg)


def test_canonicalize_ear_round_trips_landmarks():
    """A canonical prediction inverts exactly back to the original frame.

    The transform is derived from the mesh only; the 'landmarks' here are just an
    arbitrary coordinate set pushed through it, exactly as the no-leakage rule
    allows.
    """
    raw = _synthetic_subject(n=5000, seed=6)
    cfg = CropConfig(side="right", lo=[6.0, -24.0, 1.0], hi=[14.0, -16.0, 9.0], min_vertices=10)

    ear = canonicalize_ear(raw, "right", cfg, mirror_side="right", n_points=2048, seed=0)

    assert ear.subject_id == "synthetic_000"
    assert ear.side == "right"
    assert ear.points.shape == (2048, 3)
    assert ear.normals is not None and ear.normals.shape == (2048, 3)
    assert ear.transform.mirror_axis == 1
    assert ear.qa["n_points"] == 2048
    assert np.isfinite(ear.points).all()

    fake_landmarks = _synthetic_points(85, seed=21)
    canonical = transform_points_to_canonical(fake_landmarks, ear.transform)
    recovered = inverse_transform_points(canonical, ear.transform)
    assert np.allclose(recovered, fake_landmarks, atol=1e-9)


def test_canonicalize_ear_mirrors_only_the_mirror_side():
    raw = _synthetic_subject(n=5000, seed=8)
    lo, hi = [6.0, -24.0, 1.0], [14.0, -16.0, 9.0]

    left = canonicalize_ear(
        raw, "left", CropConfig(side="left", lo=lo, hi=hi, min_vertices=10), mirror_side="right"
    )
    right = canonicalize_ear(
        raw, "right", CropConfig(side="right", lo=lo, hi=hi, min_vertices=10), mirror_side="right"
    )

    assert left.transform.mirror_axis is None
    assert right.transform.mirror_axis == 1
    # Same synthetic geometry, so the mirrored side is the left side with Y negated.
    assert np.allclose(right.points[:, 1], -left.points[:, 1], atol=1e-9)
    assert np.allclose(right.points[:, [0, 2]], left.points[:, [0, 2]], atol=1e-9)
    # Normals follow the same mirror.
    assert np.allclose(right.normals[:, 1], -left.normals[:, 1], atol=1e-9)
    assert np.allclose(right.normals[:, [0, 2]], left.normals[:, [0, 2]], atol=1e-9)


def test_canonicalize_ear_raises_on_empty_crop():
    raw = _synthetic_subject(n=100, seed=9)
    cfg = CropConfig(side="left", lo=[1e6, 1e6, 1e6], hi=[2e6, 2e6, 2e6], min_vertices=1)

    with pytest.raises(ValueError):
        canonicalize_ear(raw, "left", cfg)


def test_load_crop_config_round_trip(tmp_path):
    """configs/crop.yaml as written by scripts/crop_stats.py -> CropConfig."""
    cfg_path = tmp_path / "crop.yaml"
    cfg_path.write_text(
        "# derived statistic\n"
        "sides:\n"
        "  left:\n"
        "    lo: [10.0, -120.5, -60.0]\n"
        "    hi: [80.0, -40.0, 30.0]\n"
        "    min_vertices: 700\n"
        "  right:\n"
        "    lo: [10.0, 40.0, -60.0]\n"
        "    hi: [80.0, 120.5, 30.0]\n",
        encoding="utf-8",
    )

    left = load_crop_config("left", cfg_path)
    right = load_crop_config("right", cfg_path)

    assert isinstance(left, CropConfig)
    assert left.side == "left" and right.side == "right"
    np.testing.assert_allclose(left.lo, [10.0, -120.5, -60.0])
    np.testing.assert_allclose(left.hi, [80.0, -40.0, 30.0])
    assert left.min_vertices == 700
    assert right.min_vertices == 500  # default when the key is absent
    assert right.lo.dtype == np.float64

    with pytest.raises(ValueError):
        load_crop_config("up", cfg_path)
    with pytest.raises(FileNotFoundError):
        load_crop_config("left", tmp_path / "missing.yaml")


def test_load_crop_config_rejects_missing_side(tmp_path):
    cfg_path = tmp_path / "crop.yaml"
    cfg_path.write_text("sides:\n  left:\n    lo: [0, 0, 0]\n    hi: [1, 1, 1]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_crop_config("right", cfg_path)


# ---------------------------------------------------------------------------
# The committed frozen artefact.
#
# configs/crop.yaml is the one file every downstream role depends on, and it is
# committed (a derived aggregate, not raw data). These tests pin it against the
# numbers recorded in DATA_SPEC.md and docs/HANDOFF_A.md so a hand-edit or an
# accidental re-run of scripts/crop_stats.py cannot change the frozen box
# unnoticed. No Huawei data is read — only the committed YAML and the committed
# subject list.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_CROP = REPO_ROOT / "configs" / "crop.yaml"

# Exactly the values in configs/crop.yaml, quoted to 4 dp in DATA_SPEC.md
# and docs/HANDOFF_A.md. Changing the box means changing all three together
# AND rebuilding the cache — see docs/HANDOFF_A.md "Rebuild the cache".
FROZEN_BOUNDS = {
    "left": ([-52.0145, 40.7728, -53.1446], [29.2909, 122.2154, 58.6937]),
    "right": ([-55.3936, -120.9620, -50.9584], [27.4891, -43.4874, 59.6772]),
}
FROZEN_SPLIT_SHA256 = "b2d5ff901b641fffa6f1bf1c19a587d3c4ef55e9eb32c697f0c70bbf506b1520"


def test_committed_crop_config_matches_the_frozen_numbers():
    """The committed box is exactly what DATA_SPEC.md and HANDOFF_A.md promise."""
    assert FROZEN_CROP.is_file(), f"{FROZEN_CROP} is missing — the frozen box is committed"
    for side, (lo, hi) in FROZEN_BOUNDS.items():
        cfg = load_crop_config(side, FROZEN_CROP)
        assert cfg.side == side
        np.testing.assert_allclose(cfg.lo, lo, rtol=0, atol=1e-4)
        np.testing.assert_allclose(cfg.hi, hi, rtol=0, atol=1e-4)
        assert cfg.min_vertices == 500
        assert np.all(cfg.hi > cfg.lo)


def test_committed_crop_config_passed_the_freeze_criterion():
    """A config that failed leave-one-out must never be the committed one."""
    import yaml

    doc = yaml.safe_load(FROZEN_CROP.read_text(encoding="utf-8"))
    assert doc["freeze_criterion_passed"] is True
    for side in ("left", "right"):
        headroom = doc["sides"][side]["stats"]["loo_min_headroom_per_axis_mm"]
        assert min(headroom) > 0.0, f"{side}: leave-one-out headroom must be positive"


def test_committed_crop_config_names_the_committed_training_split():
    """No-leakage provenance: the frozen box came from splits/train_ids.txt only.

    The sha256 recorded inside the YAML must still be the sha256 of the committed
    training list — otherwise the box and the split have drifted apart and the
    box would have to be regenerated (see DATA_SPEC.md "Crop configuration").
    """
    import hashlib
    import yaml

    doc = yaml.safe_load(FROZEN_CROP.read_text(encoding="utf-8"))
    split = doc["split"]
    assert split["path"] == "splits/train_ids.txt"
    assert split["n_subjects_used"] == 160 and split["n_subjects_failed"] == 0

    train_list = REPO_ROOT / "splits" / "train_ids.txt"
    digest = hashlib.sha256(train_list.read_bytes()).hexdigest()
    assert digest == FROZEN_SPLIT_SHA256 == split["sha256"], (
        "configs/crop.yaml was built from a different splits/train_ids.txt than the "
        "one committed — regenerate the crop config (scripts/crop_stats.py) and "
        "rebuild the cache."
    )


def test_committed_split_is_disjoint_and_covers_200_subjects():
    """The provisional split the frozen box rests on: 160/40, no overlap."""
    ids = {}
    for name in ("train", "val"):
        path = REPO_ROOT / "splits" / f"{name}_ids.txt"
        ids[name] = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(ids["train"]) == 160 and len(ids["val"]) == 40
    assert not set(ids["train"]) & set(ids["val"])
    assert len(set(ids["train"]) | set(ids["val"])) == 200
    assert all(re.fullmatch(r"P\d{4}", s) for group in ids.values() for s in group)

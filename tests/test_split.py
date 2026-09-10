from src.split import freeze_subject_split, validate_split


def test_split_sizes_and_disjoint():
    ids = [f"S{i:03d}" for i in range(200)]
    split = freeze_subject_split(ids, train_size=160, seed=42)
    assert len(split["train_subjects"]) == 160
    assert len(split["val_subjects"]) == 40
    validate_split(split)
    assert set(split["train_subjects"]).isdisjoint(set(split["val_subjects"]))
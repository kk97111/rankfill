from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ours"))


def test_query_validation_split_is_disjoint_deterministic_and_complete():
    from rankfill.query_validation import split_query_rows

    train_a, val_a = split_query_rows(20, val_size=4, seed=123)
    train_b, val_b = split_query_rows(20, val_size=4, seed=123)
    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(val_a, val_b)
    assert len(train_a) == 16
    assert len(val_a) == 4
    assert not set(train_a.tolist()) & set(val_a.tolist())
    np.testing.assert_array_equal(np.sort(np.concatenate([train_a, val_a])), np.arange(20))


def test_query_validation_split_zero_size_keeps_all_rows_for_training():
    from rankfill.query_validation import split_query_rows

    train, val = split_query_rows(5, val_size=0, seed=123)
    np.testing.assert_array_equal(train, np.arange(5))
    assert len(val) == 0

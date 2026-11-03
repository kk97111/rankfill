"""Deterministic held-out query split shared by training, acquisition, and validation."""
import numpy as np


def split_query_rows(num_queries, val_size=0, seed=42):
    """Return disjoint train/validation row indices covering every query exactly once."""
    if num_queries <= 0:
        raise ValueError("num_queries must be positive")
    if not 0 <= val_size < num_queries:
        raise ValueError("val_size must be in [0, num_queries)")
    if val_size == 0:
        return np.arange(num_queries, dtype=np.int64), np.empty(0, dtype=np.int64)
    perm = np.random.default_rng(seed).permutation(num_queries).astype(np.int64, copy=False)
    return perm[val_size:], perm[:val_size]

"""RankImpact scores and cumulative refresh-set extension."""
import numpy as np

def rankimpact_scores(doc_ids, scores, n_docs, k, tau):
    """Boundary impact R(d), Top-M frequency, and accumulated |rank-K|."""
    if not 1 <= k <= doc_ids.shape[1] or tau <= 0:
        raise ValueError("need 1 <= K <= M and tau > 0")
    boundary = scores[:, k - 1:k]
    weights = np.exp(-np.abs(scores - boundary) / tau).astype(np.float32)
    impact = np.zeros(n_docs, dtype=np.float32); frequency = np.zeros(n_docs, dtype=np.int32)
    rank_distance = np.zeros(n_docs, dtype=np.float64)
    np.add.at(impact, doc_ids.ravel(), weights.ravel())
    np.add.at(frequency, doc_ids.ravel(), 1)
    np.add.at(rank_distance, doc_ids.ravel(), np.broadcast_to(np.abs(np.arange(1, doc_ids.shape[1]+1) - k), doc_ids.shape).ravel())
    return impact, frequency, rank_distance

def extend_selection(current, score, target, seed):
    current = np.asarray(current, dtype=np.int64)
    if len(np.unique(current)) != len(current) or target < len(current) or target > len(score):
        raise ValueError('invalid current set or target budget')
    mask = np.ones(len(score), dtype=bool)
    mask[current] = False
    candidates = np.flatnonzero(mask)
    jitter = np.random.default_rng(seed + 12).random(len(candidates))
    order = np.lexsort((jitter, -np.asarray(score)[candidates]))
    return np.concatenate((current, candidates[order[:target-len(current)]]))

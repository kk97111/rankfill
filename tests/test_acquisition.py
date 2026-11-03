from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rankfill.acquisition import rankimpact_scores


def test_rankimpact_accumulates_score_boundary_weights_and_topm_frequency():
    ids = np.array([[4, 2, 1, 3], [2, 5, 4, 0]])
    scores = np.array([[.9, .8, .7, .6], [.95, .85, .75, .65]], dtype=np.float32)

    impact, frequency, rank_distance = rankimpact_scores(ids, scores, n_docs=6, k=2, tau=.1)

    assert np.isclose(impact[2], 1 + np.exp(-1))
    assert frequency.tolist() == [1, 1, 2, 1, 2, 1]
    assert np.isclose(rank_distance[2], 1.0)

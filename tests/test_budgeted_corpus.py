import numpy as np
import pytest
import torch

from rankfill.budgeted_corpus import SelectedNewCorpus


def test_reads_only_selected_rows_and_preserves_global_id_gather():
    class AuditedArray:
        shape = (7, 3)
        def __getitem__(self, rows):
            np.testing.assert_array_equal(rows, [5, 1])
            return np.array([[50, 51, 52], [10, 11, 12]], dtype=np.float16)
    corpus = SelectedNewCorpus(AuditedArray(), np.array([5, 1]), 'cpu')
    assert corpus.shape == (7, 3)
    assert corpus.values.shape == (2, 3)
    gathered = corpus.gather(torch.tensor([[1, 5], [5, 1]]), 'cpu')
    assert gathered.dtype == torch.float32
    np.testing.assert_array_equal(gathered.numpy()[0], [[10, 11, 12], [50, 51, 52]])


def test_nonselected_access_raises_instead_of_exposing_new_embeddings():
    corpus = SelectedNewCorpus(np.zeros((5, 2), np.float16), np.array([0, 3]), 'cpu')
    with pytest.raises(ValueError, match='outside'):
        corpus.gather(torch.tensor([0, 1]), 'cpu')


def test_rejects_duplicate_or_invalid_selected_ids():
    for selected in (np.array([0, 0]), np.array([-1]), np.array([5])):
        with pytest.raises(ValueError):
            SelectedNewCorpus(np.zeros((5, 2), np.float16), selected, 'cpu')

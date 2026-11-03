import numpy as np
import torch

from rankfill.weighted_cdf import (
    WeightedCDF, anchor_features, empirical_cdf, flip_from_cdf, episode_loss, AnchorEpisodes, collate_episodes,
)




def test_features_only_depend_on_visible_inputs():
    scores = np.array([.9, .8, .7, .6])
    a = anchor_features(scores, np.array([1, 2]), np.array([0, 3]), np.array([.1, -.1]))
    assert a.shape == (2, 2, 6)
    # API has no heldout new-score argument; ranks/score/local margin stay observable.
    assert np.isfinite(a).all()
    assert a[0, 0, 2] == np.float32(1.)


def test_kernel_distribution_and_padding():
    model = WeightedCDF()
    x = torch.randn(2, 3, 6)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    weights = model.weights(x, mask)
    assert torch.allclose(weights.sum(-1), torch.ones(2))
    assert weights[0, 2] == 0
    centers = torch.tensor([[-.1, .1, 100.], [-.2, 0., .2]])
    points = torch.linspace(-1, 1, 101).repeat(2, 1)
    cdf = model.cdf(weights, centers, points)
    assert (cdf.diff(dim=-1) >= -1e-7).all()
    assert (cdf[:, 0] < 1e-5).all() and (cdf[:, -1] > .99999).all()


def test_flip_direction_and_loss_gradient():
    f = torch.tensor([.8, .8])
    torch.testing.assert_close(flip_from_cdf(f, torch.tensor([.1, -.1])), torch.tensor([.8, .2]))
    model = WeightedCDF()
    x = torch.randn(3, 4, 6)
    mask = torch.ones(3, 4, dtype=torch.bool)
    centers = torch.tensor([[-.1, 0, .1, .2]]).repeat(3, 1)
    target = torch.tensor([.04, -.03, .08])
    mapped_gap = torch.tensor([[.02, -.04, .06, -.08]]).repeat(3, 1)
    total, nll, bce = episode_loss(model, x, centers, mask, target, mapped_gap, torch.tensor([0, 5, 20]))
    assert torch.isfinite(total) and torch.isfinite(nll) and torch.isfinite(bce)
    total.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_episode_masks_target_truth_and_checks_membership():
    compact = dict(visible_offsets=np.array([0, 2]), heldout_offsets=np.array([0, 2]),
                   query_index=np.array([0]), visible_ranks=np.array([0, 2]),
                   heldout_ranks=np.array([3, 4]), visible_new_scores=np.array([.95, .75]),
                   heldout_scores=np.array([.6, .5]))
    scores = np.array([[.9, .8, .7, .6, .5]])
    ids = np.array([[10, 11, 12, 13, 14]])
    ds = AnchorEpisodes(compact, scores, ids, np.array([10, 12, 13, 14]))
    e = ds.episode(0, 42)
    held_ranks = set(e['rank'])
    changed = {k:v.copy() for k,v in compact.items()}
    for ranks, values in [('visible_ranks','visible_new_scores'), ('heldout_ranks','heldout_scores')]:
        for i,r in enumerate(changed[ranks]):
            if r in held_ranks: changed[values][i] += .7
    ds2 = AnchorEpisodes(changed, scores, ids, np.array([10, 12, 13, 14]))
    e2 = ds2.episode(0, 42)
    np.testing.assert_array_equal(e['features'], e2['features'])
    np.testing.assert_array_equal(e['centers'], e2['centers'])
    assert not np.array_equal(e['target'],e2['target'])
    import pytest
    bad = AnchorEpisodes(compact, scores, ids, np.array([10, 12, 13]))
    with pytest.raises(AssertionError, match='non-S'):
        bad.episode(0, 42)


def test_no_support_raises_and_single_anchor_is_uniform():
    import pytest
    model=WeightedCDF()
    x=torch.randn(2, 1, 6)
    torch.testing.assert_close(model.weights(x,torch.ones(2,1,dtype=torch.bool)),torch.ones(2,1))
    with pytest.raises(ValueError,match='support'):
        model.weights(x,torch.zeros(2,1,dtype=torch.bool))


def test_teacher_ties_stay_masked_after_collation():
    compact = dict(visible_offsets=np.array([0, 2]), heldout_offsets=np.array([0, 2]),
                   query_index=np.array([0]), visible_ranks=np.array([0, 1]),
                   heldout_ranks=np.array([2, 3]), visible_new_scores=np.array([.75, .75]),
                   heldout_scores=np.array([.75, .75]))
    scores=np.array([[.9, .8, .7, .6]],dtype=np.float32)
    ids=np.array([[0, 1, 2, 3]])
    ds=AnchorEpisodes(compact,scores,ids,np.arange(4))
    batch=collate_episodes([ds.episode(0,42)],'cpu')
    assert not batch['pair_valid'].any()
    total,nll,bce=episode_loss(WeightedCDF(),batch['features'],batch['centers'],batch['mask'],
        batch['target'],batch['gap'],batch['rank'],sample_weight=batch['sample_weight'],
        pair_labels=batch['pair_labels'],pair_valid=batch['pair_valid'])
    assert bce.item()==0 and torch.isfinite(total)

"""Target-conditioned logistic-kernel residual mixtures; no target truth in features."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

FEATURES = ('relative_rank', 'mapped_score_gap', 'delta_s',
            'target_rank', 'anchor_rank', 'local_margin')
FEATURE_SETS = {'full': tuple(range(6)), 'rank_only': (0,), 'rank_score': (0, 1),
                'rank_score_residual': (0, 1, 2), 'anchor_only': (2, 4)}


def anchor_features(scores, targets, support, support_delta):
    scores = np.asarray(scores, dtype=np.float64)
    targets, support = np.asarray(targets, int), np.asarray(support, int)
    assert len(support) and len(support) == len(support_delta)
    assert not np.intersect1d(targets, support).size, 'heldout target is a source anchor'
    depth = len(scores); scale = max(1, depth - 1)
    h = targets[:, None]; s = support[None, :]
    shape = (len(targets), len(support))
    margin = (scores[np.maximum(targets-1, 0)] - scores[np.minimum(targets+1, depth-1)]) / 2
    return np.stack([
        np.broadcast_to((s-h)/scale, shape), (scores[h]-scores[s])*10,
        np.broadcast_to(np.asarray(support_delta)[None, :]*10, shape),
        np.broadcast_to(h/scale, shape), np.broadcast_to(s/scale, shape),
        np.broadcast_to(margin[:, None]*100, shape),
    ], axis=-1).astype(np.float32)


def empirical_cdf(centers, points, mask=None):
    if mask is None:
        mask = torch.ones_like(centers, dtype=torch.bool)
    count = ((centers[:, None, :] <= points[:, :, None]) & mask[:, None, :]).sum(-1)
    return (count.to(centers.dtype) + .5) / (mask.sum(-1, keepdim=True) + 1)


def flip_from_cdf(cdf, mapped_gap):
    # threshold = delta_s - (m_h-m_s); mapped h>s flips iff delta_h <= threshold.
    return torch.where(mapped_gap > 0, cdf, 1-cdf)


class WeightedCDF(nn.Module):
    def __init__(self, bandwidth=.01, feature_set='full'):
        super().__init__()
        if bandwidth <= 0:
            raise ValueError('smooth kernel bandwidth must be positive')
        self.bandwidth = float(bandwidth)
        if feature_set not in FEATURE_SETS:
            raise ValueError(f'unknown feature set: {feature_set}')
        self.feature_set = feature_set
        # Nonpersistent: old checkpoints retain exactly their original state keys.
        self.register_buffer('feature_mask', torch.tensor(
            [i in FEATURE_SETS[feature_set] for i in range(6)]), persistent=False)
        self.net = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def weights(self, features, mask, uniform=False):
        if not mask.any(-1).all():
            raise ValueError('distribution needs at least one support anchor')
        logits = self.net(features.masked_fill(~self.feature_mask, 0)).squeeze(-1)
        if uniform:
            logits = torch.zeros_like(logits)
        return logits.masked_fill(~mask, -torch.inf).softmax(-1)

    def cdf(self, weights, centers, points):
        return (weights[:, None, :] * torch.sigmoid(
            (points[:, :, None]-centers[:, None, :])/self.bandwidth)).sum(-1)

    def nll(self, weights, centers, target):
        z = (target[:, None]-centers)/self.bandwidth
        logpdf = F.logsigmoid(z) + F.logsigmoid(-z) - np.log(self.bandwidth)
        logw = weights.clamp_min(torch.finfo(weights.dtype).tiny).log()
        logw = logw.masked_fill(weights == 0, -torch.inf)
        return -torch.logsumexp(logw + logpdf, dim=-1)


def episode_loss(model, features, centers, mask, target, mapped_gap, target_rank,
                 pair_weight=.2, sample_weight=None, pair_labels=None, pair_valid=None,
                 residual_weight=1.):
    w = model.weights(features, mask)
    nll_rows = model.nll(w, centers, target)
    threshold = centers - mapped_gap
    probability = flip_from_cdf(model.cdf(w, centers, threshold), mapped_gap)
    teacher_gap = mapped_gap + target[:, None] - centers
    valid = mask & (mapped_gap != 0) & (teacher_gap != 0)
    labels = ((mapped_gap > 0) != (teacher_gap > 0)).to(probability.dtype)
    if pair_labels is not None:
        labels = pair_labels.to(probability.dtype)
        valid = mask & pair_valid
    bce = F.binary_cross_entropy(probability.clamp(1e-7, 1-1e-7), labels, reduction='none')
    bce_rows = (bce*valid).sum(-1) / valid.sum(-1).clamp_min(1)
    if sample_weight is None:
        sample_weight = torch.ones_like(target)
    head = 1/torch.log2(target_rank.to(target.dtype)+2)
    nll = (nll_rows*sample_weight).sum()/sample_weight.sum()
    bce_weight = sample_weight*head*(valid.sum(-1)>0)
    pair = (bce_rows*bce_weight).sum()/bce_weight.sum().clamp_min(1e-12)
    return residual_weight*nll+pair_weight*pair, nll, pair


class AnchorEpisodes:
    """Compact known-S anchors plus mapped scores; deliberately no full teacher input."""
    def __init__(self, compact, scores, ids, selected):
        self.data = {k: compact[k] for k in ('visible_offsets', 'heldout_offsets', 'query_index',
                     'visible_ranks', 'heldout_ranks', 'visible_new_scores', 'heldout_scores')}
        self.scores, self.ids = scores, ids
        self.selected = np.zeros(int(max(np.max(selected), ids.max()))+1, bool)
        self.selected[selected] = True
        self.query_rows = {int(q): row for row, q in enumerate(self.data['query_index'])}

    def anchors(self, qi):
        row = self.query_rows[int(qi)]; d = self.data
        a, b = d['visible_offsets'][row:row+2]; c, e = d['heldout_offsets'][row:row+2]
        ranks = np.r_[d['visible_ranks'][a:b], d['heldout_ranks'][c:e]].astype(int)
        new = np.r_[d['visible_new_scores'][a:b], d['heldout_scores'][c:e]].astype(np.float64)
        order = np.argsort(ranks, kind='stable'); ranks, new = ranks[order], new[order]
        assert len(np.unique(ranks)) == len(ranks)
        assert np.all(self.selected[self.ids[qi, ranks]]), 'non-S teacher score in anchor cache'
        assert np.isfinite(new).all()
        return ranks, new-np.asarray(self.scores[qi, ranks], np.float64)

    def episode(self, qi, seed, max_targets=4):
        ranks, delta = self.anchors(qi)
        if len(ranks) < 3:
            return None
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(qi)]))
        order = rng.permutation(len(ranks)); cut = max(2, len(ranks)//2)
        support, held = order[:cut], order[cut:cut+max_targets]
        scores = np.asarray(self.scores[qi], np.float64)
        h, s = ranks[held], ranks[support]
        gap = scores[h,None]-scores[s][None,:]
        teacher_gap = gap + delta[held,None] - delta[support][None,:]
        return dict(features=anchor_features(scores, h, s, delta[support]),
                    centers=np.broadcast_to(delta[support], (len(h),len(s))).copy(),
                    target=delta[held], gap=gap, rank=h,
                    pair_labels=(gap>0)!=(teacher_gap>0),
                    pair_valid=(gap!=0)&(teacher_gap!=0))


def collate_episodes(episodes, device):
    episodes = [e for e in episodes if e is not None]
    if not episodes:
        return None
    n = sum(len(e['target']) for e in episodes); support = max(e['centers'].shape[1] for e in episodes)
    arrays = dict(features=np.zeros((n,support,6),np.float32), centers=np.zeros((n,support),np.float32),
                  mask=np.zeros((n,support),bool), target=np.zeros(n,np.float32),
                  gap=np.zeros((n,support),np.float32), rank=np.zeros(n,np.int64),
                  pair_labels=np.zeros((n,support),bool),pair_valid=np.zeros((n,support),bool),
                  sample_weight=np.zeros(n,np.float32))
    offset=0
    for e in episodes:
        h,s=e['centers'].shape; sl=slice(offset,offset+h)
        for key in ('features','centers','gap','pair_labels','pair_valid'):
            arrays[key][sl,:s]=e[key]
        for key in ('target','rank'):
            arrays[key][sl]=e[key]
        arrays['mask'][sl,:s]=True;arrays['sample_weight'][sl]=1/h;offset+=h
    return {k:torch.as_tensor(v,device=device) for k,v in arrays.items()}

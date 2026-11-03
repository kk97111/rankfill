"""Retrieval distribution alignment (RDA)."""
import torch
import torch.nn.functional as F

def rank_distill_loss(qb, h_cand, eN_cand, temp):
    """Query-anchored listwise ranking distillation (ours, replaces doc-doc RPL).

    qb       [Qb,d]  phi_new(q) for a batch of TRAIN queries (deploy query side)
    h_cand   [M,d]   h(phi_old(d)) for the union of the batch's candidate docs (student)
    eN_cand  [M,d]   phi_new(d) for the same docs (teacher)
    The order that matters is the QUERY->DOC ranking the metric scores, so we distill
    the teacher ranking <phi_new(q),phi_new(d)> into the deploy ranking
    <phi_new(q),h(phi_old(d))>. Other queries' docs in the union act as in-batch negatives.
    """
    T = qb @ eN_cand.T                                   # [Qb,M] teacher scores
    Sc = qb @ F.normalize(h_cand, dim=1).T               # [Qb,M] student (h not unit -> norm)
    p = F.softmax(T / temp, dim=1)
    logq = F.log_softmax(Sc / temp, dim=1)
    return -(p * logq).sum(dim=1).mean()                 # listwise KL (up to teacher entropy)

def cosine_alignment_loss(mapped, target):
    """Global selected-doc pointwise cosine alignment and its mean cosine."""
    cos_sim = F.cosine_similarity(mapped, target, dim=-1)
    return (1.0 - cos_sim).mean(), cos_sim.mean()

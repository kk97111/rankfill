"""Refreshed-only ARC caches, held-out calibration, and ranking evaluation."""
import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .arc import query_proposals, decode_query
from .mapper import mapped_gallery
from .metrics import per_query_metrics
from .retrieval import retrieve, normalized_gallery
from .weighted_cdf import WeightedCDF, AnchorEpisodes


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
def build_cache(emb, old_tag, new_tag, selected, checkpoint, out, device, query_batch=32):
    """Keep all non-S new-document access in the separate oracle evaluation stage."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    old = np.load(emb / f"{old_tag}_corpus.fp16.npy", mmap_mode="r")
    new = np.load(emb / f"{new_tag}_corpus.fp16.npy", mmap_mode="r")
    qtrain = np.load(emb / f"{new_tag}_qtrain.fp16.npy", mmap_mode="r")
    qdev = np.load(emb / f"{new_tag}_qdev.fp16.npy", mmap_mode="r")
    qids = np.asarray(json.loads((emb / f"{new_tag}_qdev.qids.json").read_text()))
    gallery = mapped_gallery(old, checkpoint, device)
    # Match the original evaluator's CPU normalization followed by retrieval normalization.
    queries = F.normalize(torch.tensor(np.asarray(qtrain, np.float32)), dim=1).numpy()
    train_ids, _ = retrieve(queries, gallery, batch=query_batch, prefix=out / "qtrain_mapped_top1000")
    queries = F.normalize(torch.tensor(np.asarray(qdev, np.float32)), dim=1).numpy()
    dev_ids, dev_scores = retrieve(queries, gallery, batch=query_batch)
    del gallery, queries
    release_memory()
    np.save(out / "qdev_mapped_scores.npy", dev_scores)
    np.savez_compressed(out / "qdev_rankings.npz", mapped=dev_ids, qids=qids)
    local = np.full(len(old), -1, np.int64)
    local[selected] = np.arange(len(selected))
    anchors = normalized_gallery(new[selected], device)
    offsets = np.zeros(len(qtrain) + 1, np.int64)
    ranks, values = [], []
    for start in range(0, len(qtrain), 256):
        ids = train_ids[start:start + 256]
        rows, cols = np.nonzero(local[ids] >= 0)
        q = F.normalize(torch.tensor(np.asarray(qtrain[start:start + len(ids)], np.float32), device=device), dim=1)
        known = (q[torch.as_tensor(rows, device=device)] * anchors[torch.as_tensor(local[ids[rows, cols]], device=device)]).sum(-1).cpu().numpy()
        offsets[start + 1:start + len(ids) + 1] = offsets[start] + np.cumsum(np.bincount(rows, minlength=len(ids)))
        ranks.append(cols.astype(np.int16))
        values.append(known)
    np.savez_compressed(out / "qtrain_s_compact.npz", query_index=np.arange(len(qtrain)),
                        visible_offsets=offsets, visible_ranks=np.concatenate(ranks),
                        visible_new_scores=np.concatenate(values),
                        heldout_offsets=np.zeros(len(qtrain) + 1, np.int64),
                        heldout_ranks=np.empty(0, np.int16), heldout_scores=np.empty(0, np.float32))
    # Non-anchor score slots remain NaN: inference cannot accidentally use them.
    known_scores = np.full(dev_ids.shape, np.nan, np.float32)
    for start in range(0, len(qdev), 16):
        ids = dev_ids[start:start + 16]
        rows, cols = np.nonzero(local[ids] >= 0)
        q = F.normalize(torch.tensor(np.asarray(qdev[start:start + len(ids)], np.float32), device=device), dim=1)
        known_scores[start + rows, cols] = (q[torch.as_tensor(rows, device=device)] * anchors[torch.as_tensor(local[ids[rows, cols]], device=device)]).sum(-1).cpu().numpy()
    np.save(out / "qdev_anchor_scores.npy", known_scores)
    (out / "config.json").write_text(json.dumps(dict(candidate_depth=1000, stage="mapped",
        checkpoint=str(checkpoint), selected_count=len(selected), non_S_new_access=False), indent=2) + "\n")


@torch.inference_mode()
def evaluate(cache, train, selected, emb, new_tag, output, device, qrels=None, query_batch=32):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(train / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    model = WeightedCDF(ckpt["config"]["bandwidth"], ckpt["config"].get("feature_set", "full")).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    scores = np.load(cache / "qtrain_mapped_top1000.scores.npy", mmap_mode="r")
    ids = np.load(cache / "qtrain_mapped_top1000.ids.npy", mmap_mode="r")
    with np.load(cache / "qtrain_s_compact.npz") as compact:
        data = AnchorEpisodes(compact, scores, ids, selected)
    validation = np.load(train / "query_split.npz")["validation"]
    probabilities = []
    for qi in validation:
        ranks, delta = data.anchors(int(qi))
        _, pools = query_proposals(model, ids[qi], scores[qi], ranks, delta + scores[qi, ranks], device, methods=("LearnedCDF",))
        probabilities.extend(p[0] for p in pools["LearnedCDF"])
    threshold = float(np.quantile(probabilities, 0.85)) if probabilities else 1.0
    (output / "threshold.json").write_text(json.dumps(dict(threshold=threshold, quantile=0.85,
        validation_queries=len(validation), proposals=len(probabilities), source="held-out training queries; S-only anchors"), indent=2) + "\n")
    z = np.load(cache / "qdev_rankings.npz")
    mapped, qids = z["mapped"], z["qids"].astype(str)
    scores = np.load(cache / "qdev_mapped_scores.npy", mmap_mode="r")
    known = np.load(cache / "qdev_anchor_scores.npy", mmap_mode="r")
    predictions = np.empty_like(mapped)
    selected_mask = np.zeros(max(int(mapped.max()), int(selected.max())) + 1, dtype=bool)
    selected_mask[selected] = True
    for qi, docs in enumerate(mapped):
        ranks = np.flatnonzero(selected_mask[docs])
        fixed, pools = query_proposals(model, docs, scores[qi], ranks, known[qi, ranks], device, methods=("LearnedCDF",))
        predictions[qi] = decode_query(fixed, docs[ranks], pools["LearnedCDF"], threshold)[0]
    assert np.array_equal(np.sort(predictions, axis=1), np.sort(mapped, axis=1))
    np.savez_compressed(output / "predictions.npz", qids=qids, rankfill=predictions, mapped=mapped)
    # Predictions and threshold are frozen before the full-new oracle is accessed.
    new = np.load(emb / f"{new_tag}_corpus.fp16.npy", mmap_mode="r")
    qdev = np.load(emb / f"{new_tag}_qdev.fp16.npy", mmap_mode="r")
    gallery = normalized_gallery(new, device)
    teacher, _ = retrieve(qdev, gallery, depth=10, batch=query_batch)
    np.save(output / "oracle_top10.npy", teacher)
    result, per_query = {}, {}
    for k in (5, 10):
        recall, ndcg = per_query_metrics(predictions, teacher, k)
        result[f"alignment_recall@{k}"] = float(recall.mean())
        result[f"alignment_ndcg@{k}"] = float(ndcg.mean())
        per_query[f"alignment_recall@{k}"] = recall
        per_query[f"alignment_ndcg@{k}"] = ndcg
    if qrels is not None:
        pids = json.loads((emb / f"{new_tag}_corpus.pids.json").read_text())
        relevant = {}
        for line in Path(qrels).read_text().splitlines():
            qid, pid, score = line.split("\t")
            if float(score) > 0:
                relevant.setdefault(qid, set()).add(pid)
        for k in (5, 10):
            recall, ndcg = [], []
            weights = 1 / np.log2(np.arange(k) + 2)
            for qid, docs in zip(qids, predictions):
                gold = relevant.get(qid, set())
                hit = np.array([pids[int(d)] in gold for d in docs[:k]])
                recall.append(float(hit.sum()) / max(1, len(gold)))
                ndcg.append(float(np.dot(hit, weights)) / max(1e-12, weights[:min(k, len(gold))].sum()))
            result[f"qrels_recall@{k}"] = float(np.mean(recall))
            result[f"qrels_ndcg@{k}"] = float(np.mean(ndcg))
    np.savez_compressed(output / "per_query_metrics.npz", qids=qids, **per_query)
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result

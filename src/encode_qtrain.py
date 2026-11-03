"""Encode MS MARCO TRAIN queries with phi_new (teacher) for the ranking-distillation loss.

Train queries (NOT dev -> no eval contamination) anchor the query-aware ranking-preserving
loss: deployment score is <phi_new(q), h(phi_old(d))>, teacher is <phi_new(q), phi_new(d)>.
Both sides use phi_new(q), so we only need phi_new for the queries.

Run:
  CUDA_VISIBLE_DEVICES=3,4,5 python src/encode_qtrain.py --backbone contriever --which phi_new_x10
"""
import argparse
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
EMB = os.path.join(ROOT, "data", "emb")


def load_tsv(path):
    ids, texts = [], []
    with open(path) as f:
        for line in f:
            a, b = line.rstrip("\n").split("\t", 1)
            ids.append(a); texts.append(b)
    return ids, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="contriever")
    ap.add_argument("--which", default="phi_new_x10")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--batch", type=int, default=1024)
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]
    mpath = os.path.join(ROOT, "models", args.backbone, "backbone", args.which)
    out_dir = os.path.join(EMB, args.backbone)

    model = SentenceTransformer(mpath); model.half()
    qids, qtexts = load_tsv(os.path.join(DATA, "queries.train.tsv"))
    print(f"[enc] {len(qtexts):,} train queries")
    devices = [f"cuda:{g}" for g in gpus]
    if len(devices) == 1:
        emb = model.encode(qtexts, batch_size=args.batch, device=devices[0],
                           normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
    else:
        pool = model.start_multi_process_pool(target_devices=devices)
        emb = model.encode_multi_process(qtexts, pool, batch_size=args.batch, normalize_embeddings=True)
        model.stop_multi_process_pool(pool)
    emb = emb.astype(np.float16)
    np.save(os.path.join(out_dir, f"{args.which}_qtrain.fp16.npy"), emb)
    json.dump(qids, open(os.path.join(out_dir, f"{args.which}_qtrain.qids.json"), "w"))
    print(f"[enc] saved {args.which}_qtrain {emb.shape}")


if __name__ == "__main__":
    main()

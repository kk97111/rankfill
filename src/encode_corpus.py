"""Phase 2: encode the full gallery (and dev queries) with a given encoder,
multi-GPU + fp16, cache to data/emb/.

  data/emb/<tag>_corpus.fp16.npy   [N, 768] float16, L2-normalized
  data/emb/<tag>_corpus.pids.json  pid order
  data/emb/<tag>_qdev.fp16.npy     [Q, 768]
  data/emb/<tag>_qdev.qids.json    qid order

Run:
  python src/encode_corpus.py --backbone contriever --which phi_new_x10 --gpus 1,3,4,5
  (loads models/<backbone>/backbone/<which>)
"""
import argparse
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
EMB = os.path.join(DATA, "emb")
os.makedirs(EMB, exist_ok=True)


def load_corpus(path, limit=None):
    pids, texts = [], []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            ex = json.loads(line)
            pids.append(ex["pid"])
            texts.append(ex["text"])
    return pids, texts


def load_tsv(path):
    ids, texts = [], []
    with open(path) as f:
        for line in f:
            a, b = line.rstrip("\n").split("\t", 1)
            ids.append(a)
            texts.append(b)
    return ids, texts


def encode(model, texts, gpus, batch):
    devices = [f"cuda:{g}" for g in gpus]
    if len(devices) == 1:
        return model.encode(texts, batch_size=batch, device=devices[0],
                            normalize_embeddings=True, show_progress_bar=True,
                            convert_to_numpy=True)
    pool = model.start_multi_process_pool(target_devices=devices)
    emb = model.encode_multi_process(texts, pool, batch_size=batch,
                                     normalize_embeddings=True)
    model.stop_multi_process_pool(pool)
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, help="contriever")
    ap.add_argument("--which", required=True, help="phi_old or phi_new_x10 ...")
    ap.add_argument("--gpus", default="3")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None, help="benchmark on first N passages")
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]
    mpath = os.path.join(ROOT, "models", args.backbone, "backbone", args.which)
    out_dir = os.path.join(EMB, args.backbone)
    os.makedirs(out_dir, exist_ok=True)
    tag = args.which

    model = SentenceTransformer(mpath)
    model.half()

    # corpus
    pids, texts = load_corpus(os.path.join(DATA, "corpus.jsonl"), limit=args.limit)
    import time
    t = time.time()
    emb = encode(model, texts, gpus, args.batch).astype(np.float16)
    dt = time.time() - t
    print(f"[enc] {args.backbone}/{tag} corpus {emb.shape} in {dt:.1f}s = {len(texts)/dt:.0f} p/s")
    suffix = f"_lim{args.limit}" if args.limit else ""
    np.save(os.path.join(out_dir, f"{tag}_corpus{suffix}.fp16.npy"), emb)
    json.dump(pids, open(os.path.join(out_dir, f"{tag}_corpus{suffix}.pids.json"), "w"))

    if args.limit:
        return  # benchmark mode, skip queries

    # dev queries
    qids, qtexts = load_tsv(os.path.join(DATA, "queries.dev.tsv"))
    qemb = encode(model, qtexts, gpus, 512).astype(np.float16)
    np.save(os.path.join(out_dir, f"{tag}_qdev.fp16.npy"), qemb)
    json.dump(qids, open(os.path.join(out_dir, f"{tag}_qdev.qids.json"), "w"))
    print(f"[enc] dev queries {qemb.shape}")
    print("[enc] done")


if __name__ == "__main__":
    main()

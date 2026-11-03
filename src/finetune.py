"""Fine-tune a tied Contriever bi-encoder on a fraction of MS MARCO training queries."""
import argparse
import json
import os
import random

os.environ.setdefault("WANDB_DISABLED", "true")  # system wandb is numpy-2 incompatible

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    models,
    losses,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")


def build_encoder(base="facebook/contriever", max_len=256, pooling="mean"):
    """Shared bi-encoder: backbone -> pooling -> L2 normalize."""
    word = models.Transformer(base, max_seq_length=max_len)
    pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode=pooling)
    norm = models.Normalize()
    return SentenceTransformer(modules=[word, pool, norm])


def load_corpus(path):
    print(f"[data] loading corpus {path}")
    corpus = {}
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            corpus[ex["pid"]] = ex["text"]
    print(f"[data] corpus {len(corpus):,} passages")
    return corpus


def load_queries(path):
    q = {}
    with open(path) as f:
        for line in f:
            qid, text = line.rstrip("\n").split("\t", 1)
            q[qid] = text
    return q


def build_pairs(frac, seed):
    """Sample query IDs, then include their positive passages."""
    corpus = load_corpus(os.path.join(DATA, "corpus.jsonl"))
    queries = load_queries(os.path.join(DATA, "queries.train.tsv"))
    # positives from train qrels
    qrels = {}
    with open(os.path.join(DATA, "qrels.train.tsv")) as f:
        for line in f:
            qid, pid, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                qrels.setdefault(qid, []).append(pid)
    qids = sorted(qrels.keys())
    rng = random.Random(seed)
    rng.shuffle(qids)
    total = len(qids)
    n = int(total * frac)
    qids = qids[:n]
    print(f"[data] sampling {n:,} / {total:,} train queries (frac={frac})")
    anchors, positives = [], []
    miss = 0
    for qid in qids:
        if qid not in queries:
            continue
        for pid in qrels[qid]:
            if pid in corpus:
                anchors.append(queries[qid])
                positives.append(corpus[pid])
            else:
                miss += 1
    print(f"[data] {len(anchors):,} (query, positive) pairs (missing pid: {miss})")
    return Dataset.from_dict({"anchor": anchors, "positive": positives})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backbone", default="contriever")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None, help="output dir name under backbone/ (overrides default)")
    ap.add_argument("--smoke", action="store_true", help="tiny run to verify pipeline")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, os.path.join(ROOT, "configs"))
    from backbones import get as get_backbone
    bb = get_backbone(args.backbone)

    pct = int(round(args.frac * 100))
    default_tag = args.tag or f"phi_new_x{pct}"
    out = args.out or os.path.join(ROOT, "models", args.backbone, "backbone", default_tag)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    model = build_encoder(bb["hf"], args.max_len, bb["pooling"])
    train_ds = build_pairs(args.frac, args.seed)
    if args.smoke:
        train_ds = train_ds.select(range(min(2048, len(train_ds))))
        args.epochs = 1
        out = os.path.join(ROOT, "models", args.backbone, "backbone", "phi_new_smoke")

    loss = losses.MultipleNegativesRankingLoss(model)  # scale=20, cosine on normalized emb

    targs = SentenceTransformerTrainingArguments(
        output_dir=os.path.join(ROOT, "logs", os.path.basename(out)),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        bf16=True,
        dataloader_num_workers=8,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
    )

    trainer = SentenceTransformerTrainer(
        model=model, args=targs, train_dataset=train_ds, loss=loss
    )
    trainer.train()
    model.save(out)
    print(f"[done] saved phi_new -> {out}")


if __name__ == "__main__":
    main()

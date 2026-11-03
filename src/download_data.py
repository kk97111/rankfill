"""Download MS MARCO passage data (BeIR/msmarco) and materialize to simple files.

Outputs under data/:
  corpus.jsonl        {"pid": str, "text": str}   ~8.84M passages
  queries.train.tsv   qid \t text
  queries.dev.tsv     qid \t text
  qrels.train.tsv     qid \t pid \t score
  qrels.dev.tsv       qid \t pid \t score

We only need (query -> positive passage) for in-batch-negative training,
plus dev qrels for the sanity-check Recall@10. Negatives come from the batch.
"""
import json
import os
from datasets import load_dataset

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
DATA = os.path.abspath(DATA)
os.makedirs(DATA, exist_ok=True)


def dump_corpus():
    out = os.path.join(DATA, "corpus.jsonl")
    if os.path.exists(out):
        print(f"[corpus] exists, skip: {out}")
        return
    print("[corpus] loading BeIR/msmarco corpus ...")
    ds = load_dataset("BeIR/msmarco", "corpus", split="corpus")
    print(f"[corpus] {len(ds):,} passages; writing {out}")
    with open(out, "w") as f:
        for ex in ds:
            title = (ex.get("title") or "").strip()
            text = (ex.get("text") or "").strip()
            full = (title + " " + text).strip() if title else text
            f.write(json.dumps({"pid": str(ex["_id"]), "text": full}) + "\n")
    print("[corpus] done")


def dump_queries():
    print("[queries] loading BeIR/msmarco queries ...")
    ds = load_dataset("BeIR/msmarco", "queries", split="queries")
    qmap = {str(ex["_id"]): (ex.get("text") or "").strip() for ex in ds}
    print(f"[queries] {len(qmap):,} total queries")
    return qmap


def dump_qrels(qmap):
    print("[qrels] loading BeIR/msmarco-qrels ...")
    qrels = load_dataset("BeIR/msmarco-qrels")
    print(f"[qrels] available splits: {list(qrels.keys())}")
    # BeIR qrels splits are typically: train, validation (dev), test
    split_map = {}
    for s in qrels.keys():
        if "train" in s:
            split_map["train"] = s
        elif "val" in s or "dev" in s:
            split_map["dev"] = s
    for name, split in split_map.items():
        rows = qrels[split]
        qids = set()
        qrels_out = os.path.join(DATA, f"qrels.{name}.tsv")
        with open(qrels_out, "w") as f:
            for ex in rows:
                qid = str(ex["query-id"])
                pid = str(ex["corpus-id"])
                score = ex["score"]
                f.write(f"{qid}\t{pid}\t{score}\n")
                qids.add(qid)
        # write the corresponding queries
        q_out = os.path.join(DATA, f"queries.{name}.tsv")
        n = 0
        with open(q_out, "w") as f:
            for qid in sorted(qids):
                if qid in qmap:
                    f.write(f"{qid}\t{qmap[qid]}\n")
                    n += 1
        print(f"[qrels] {name}: {len(rows):,} qrels, {n:,} queries -> {qrels_out}, {q_out}")


if __name__ == "__main__":
    dump_corpus()
    qmap = dump_queries()
    dump_qrels(qmap)
    print("ALL DONE")

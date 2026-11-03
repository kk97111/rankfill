"""Minimal progressive Rankfill reproduction: RDA -> RankImpact -> ARC."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

from .acquisition import extend_selection, rankimpact_scores
from .evaluation import build_cache, evaluate, release_memory
from .mapper import mapped_gallery
from .query_validation import split_query_rows
from .retrieval import retrieve
from .training import mine, train_mapper

ROOT = Path(__file__).resolve().parents[1]


@torch.inference_mode()
def select(emb, old_tag, new_tag, current, checkpoint, budget, val_size, seed, device,
           query_sample=120000, query_batch=32):
    old = np.load(emb / f"{old_tag}_corpus.fp16.npy", mmap_mode="r")
    queries = np.load(emb / f"{new_tag}_qtrain.fp16.npy", mmap_mode="r")
    train_rows, _ = split_query_rows(len(queries), val_size, 42)
    rows = np.random.default_rng(seed + 11).choice(train_rows, min(query_sample, len(train_rows)), replace=False)
    # Acquisition reads no new-document matrix.
    gallery = mapped_gallery(old, checkpoint, device, batch=8192)
    ids, scores = retrieve(queries[rows], gallery, depth=20, batch=query_batch)
    impact, _, _ = rankimpact_scores(ids, scores, len(old), 10, 0.01)
    return extend_selection(current, impact, int(budget * len(old)), seed), rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emb-dir", type=Path, default=ROOT / "data/emb/contriever")
    p.add_argument("--old-tag", default="phi_old")
    p.add_argument("--new-tag", default="phi_new_x1")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/contriever_x1_s42")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--budgets", default="0.01,0.02,0.05,0.10", help="Increasing cumulative refresh fractions")
    p.add_argument("--seed-budget", type=float, default=0.002)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--arc-epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--query-sample", type=int, default=120000)
    p.add_argument("--query-batch", type=int, default=32)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--qrels", type=Path, help="Optional MS MARCO dev qrels: qid TAB pid TAB score")
    p.add_argument("--evaluate-all", action="store_true", help="Fit/evaluate ARC at every reported budget; default: final budget only")
    a = p.parse_args()
    a.emb_dir, a.out = a.emb_dir.resolve(), a.out.resolve()
    budgets = [float(b) for b in a.budgets.split(",")]
    if not budgets or not 0 < a.seed_budget < budgets[0] or budgets[-1] >= 1 or any(x >= y for x, y in zip(budgets, budgets[1:])):
        p.error("Require 0 < seed-budget < strictly increasing budgets < 1. The 100% endpoint is the full-new oracle.")
    if min(a.epochs, a.arc_epochs, a.batch, a.hidden, a.query_sample, a.query_batch, a.threads) < 1:
        p.error("Training and retrieval sizes must be positive.")
    if a.out.exists() and any(a.out.iterdir()):
        p.error("Output directory is nonempty; use a fresh --out to preserve existing runs.")
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    old = np.load(a.emb_dir / f"{a.old_tag}_corpus.fp16.npy", mmap_mode="r")
    new = np.load(a.emb_dir / f"{a.new_tag}_corpus.fp16.npy", mmap_mode="r")
    if old.shape != new.shape or len(old) < 1000 or int(a.seed_budget * len(old)) < 32:
        p.error("Need matching old/new matrices, at least 1000 documents, and at least 32 seed documents.")
    pid_files = [a.emb_dir / f"{tag}_corpus.pids.json" for tag in (a.old_tag, a.new_tag)]
    if json.loads(pid_files[0].read_text()) != json.loads(pid_files[1].read_text()):
        p.error("Old/new passage ID row orders differ.")
    qtrain = np.load(a.emb_dir / f"{a.new_tag}_qtrain.fp16.npy", mmap_mode="r")
    qdev = np.load(a.emb_dir / f"{a.new_tag}_qdev.fp16.npy", mmap_mode="r")
    train_ids = json.loads((a.emb_dir / f"{a.new_tag}_qtrain.qids.json").read_text())
    dev_ids = json.loads((a.emb_dir / f"{a.new_tag}_qdev.qids.json").read_text())
    if len(train_ids) != len(qtrain) or len(dev_ids) != len(qdev) or set(train_ids) & set(dev_ids):
        p.error("Query IDs must match embedding rows and training/dev queries must be disjoint.")
    val_size = round(0.1 * len(qtrain))
    if val_size < 1:
        p.error("Need at least ten training queries for held-out calibration.")
    a.out.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(query_val_size=val_size, query_val_seed=42, budgets=budgets)
    (a.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    selected = torch.randperm(len(old), generator=torch.Generator().manual_seed(a.seed)).numpy()[:int(a.seed_budget * len(old))]
    checkpoint = None
    manifest = []
    for index, budget in enumerate([a.seed_budget] + budgets):
        stage = a.out / ("seed" if index == 0 else f"budget_{budget:g}")
        stage.mkdir()
        if checkpoint is not None:
            selected, query_rows = select(a.emb_dir, a.old_tag, a.new_tag, selected, checkpoint,
                budget, val_size, a.seed, a.device, a.query_sample, a.query_batch)
            np.save(stage / "acquisition_query_rows.npy", query_rows)
            release_memory()
        selected_file = stage / "selected.npy"
        np.save(selected_file, selected)
        print(f"[round] budget={budget:g} selected={len(selected)}", flush=True)
        topn = stage / "teacher_top32.npz"
        mine(a.emb_dir, a.new_tag, selected, topn, val_size, a.device)
        release_memory()
        checkpoint = train_mapper(a.emb_dir, a.old_tag, a.new_tag, selected, topn,
            stage / "rda", val_size, a.seed, a.device, a.epochs, a.batch, a.hidden)
        release_memory()
        entry = dict(budget=budget, selected_count=len(selected),
                     selected=str(selected_file.relative_to(a.out)),
                     selected_sha256=hashlib.sha256(selected_file.read_bytes()).hexdigest(),
                     checkpoint=str(checkpoint.relative_to(a.out)))
        if index > 0 and (a.evaluate_all or index == len(budgets)):
            cache, arc = stage / "cache", stage / "arc"
            build_cache(a.emb_dir, a.old_tag, a.new_tag, selected, checkpoint, cache, a.device, a.query_batch)
            release_memory()
            cmd = [sys.executable, str(ROOT / "scripts/train_arc.py"),
                   "--compact", str(cache / "qtrain_s_compact.npz"),
                   "--scores", str(cache / "qtrain_mapped_top1000.scores.npy"),
                   "--ids", str(cache / "qtrain_mapped_top1000.ids.npy"),
                   "--selected", str(selected_file), "--out", str(arc),
                   "--mapper-checkpoint", str(checkpoint), "--new-tag", a.new_tag,
                   "--epochs", str(a.arc_epochs), "--batch", str(a.batch),
                   "--seed", str(a.seed), "--threads", str(a.threads),
                   "--query-val-size", str(val_size), "--query-val-seed", "42", "--device", a.device]
            subprocess.run(cmd, check=True, cwd=ROOT)
            entry["metrics"] = evaluate(cache, arc, selected, a.emb_dir, a.new_tag, stage / "evaluation", a.device, a.qrels, a.query_batch)
            release_memory()
        manifest.append(entry)
        (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[done] {a.out / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()

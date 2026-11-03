"""The paper's ten-epoch RDA recipe, restricted to observed new vectors."""
import json
from pathlib import Path

import numpy as np
import torch

from .budgeted_corpus import SelectedNewCorpus
from .losses import cosine_alignment_loss, rank_distill_loss
from .mapper import Mapper
from .mining import mine_extreme_local_ids
from .query_validation import split_query_rows


def mine(emb, new_tag, selected, output, val_size, device, query_chunk=512):
    """Cache top-32 teacher documents per training query using only S."""
    queries = np.load(emb / f"{new_tag}_qtrain.fp16.npy", mmap_mode="r")
    source = np.load(emb / f"{new_tag}_corpus.fp16.npy", mmap_mode="r")
    train_rows, val_rows = split_query_rows(len(queries), val_size, 42)
    if len(selected) < 32:
        raise ValueError("At least 32 refreshed documents are required for J=32.")
    documents = torch.tensor(np.asarray(source[selected], np.float32), device=device)
    ids = np.empty((len(train_rows), 32), np.int64)
    for start in range(0, len(train_rows), query_chunk):
        q = torch.tensor(np.asarray(queries[train_rows[start:start + query_chunk]], np.float32), device=device)
        with torch.no_grad():
            local = mine_extreme_local_ids(q, documents, 32, schunk=50000)
        ids[start:start + len(q)] = selected[local.cpu().numpy()]
    np.savez(output, gdocs=ids, qrows=train_rows, qval_rows=val_rows)


def train_mapper(emb, old_tag, new_tag, selected, topn, output, val_size,
                 seed=42, device="cuda:0", epochs=10, batch=256, hidden=2048):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    old_source = np.load(emb / f"{old_tag}_corpus.fp16.npy", mmap_mode="r")
    new_source = np.load(emb / f"{new_tag}_corpus.fp16.npy", mmap_mode="r")
    if old_source.shape != new_source.shape:
        raise ValueError("Old/new document matrices must have identical shapes and row order.")
    old = torch.tensor(np.asarray(old_source), device=device)
    new = SelectedNewCorpus(new_source, selected, device)
    queries = np.load(emb / f"{new_tag}_qtrain.fp16.npy", mmap_mode="r")
    train_rows, _ = split_query_rows(len(queries), val_size, 42)
    with np.load(topn) as cache:
        if not np.array_equal(cache["qrows"], train_rows):
            raise ValueError("Mined queries do not match the held-out split.")
        if not np.isin(cache["gdocs"], selected).all():
            raise ValueError("Teacher mining exposed documents outside S.")
        candidates = torch.tensor(cache["gdocs"], device=device)
    q = torch.tensor(np.asarray(queries[train_rows], np.float32), device=device)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    model = Mapper(old.shape[1], hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    steps = (len(selected) + batch - 1) // batch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001, total_steps=steps * epochs, pct_start=0.1)
    indices = torch.from_numpy(np.asarray(selected, np.int64))
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = indices[torch.randperm(len(indices), generator=generator)]
        qperm = torch.randperm(len(q), generator=generator)
        qptr = 0
        totals = np.zeros(3)
        count = 0
        for start in range(0, len(permutation), batch):
            ids = permutation[start:start + batch].to(device)
            if len(ids) < 2:
                continue
            if qptr + batch > len(qperm):
                qperm = torch.randperm(len(q), generator=generator)
                qptr = 0
            rows = qperm[qptr:qptr + batch].to(device)
            qptr += batch
            union = torch.unique(candidates[rows].reshape(-1))
            mapped = model(old[ids].float())
            cos, _ = cosine_alignment_loss(mapped, new.gather(ids, device))
            rank = rank_distill_loss(q[rows], model(old[union].float()),
                                     new.gather(union, device), 0.05)
            loss = rank + 2.0 * cos
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite RDA loss.")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            totals += [loss.item(), rank.item(), cos.item()]
            count += 1
        if count == 0:
            raise ValueError("No document minibatches were produced.")
        row = dict(epoch=epoch, **dict(zip(("loss", "rank_loss", "cos_loss"), (totals / count).tolist())))
        history.append(row)
        print("[RDA]", json.dumps(row), flush=True)
    checkpoint = output / "mapper.pt"
    torch.save(model.state_dict(), checkpoint)
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps(dict(
        seed=seed, epochs=epochs, batch=batch, hidden=hidden, selected_count=len(selected),
        query_val_size=val_size, query_val_seed=42, rank_temperature=0.05,
        cos_weight=2.0, learning_rate=0.001, checkpoint="final epoch"), indent=2) + "\n")
    return checkpoint

"""Small CPU end-to-end check; no downloads or bundled research data."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_progressive_training_correction_and_evaluation(tmp_path):
    root = Path(__file__).resolve().parents[1]
    emb, out = tmp_path / "emb", tmp_path / "run"
    emb.mkdir()
    rng = np.random.default_rng(19)

    def vectors(n):
        x = rng.normal(size=(n, 16)).astype(np.float32)
        return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)

    old = vectors(1200)
    new = old.astype(np.float32) + 0.1 * vectors(1200)
    new = (new / np.linalg.norm(new, axis=1, keepdims=True)).astype(np.float16)
    for tag, corpus in (("phi_old", old), ("phi_new_x1", new)):
        np.save(emb / f"{tag}_corpus.fp16.npy", corpus)
        (emb / f"{tag}_corpus.pids.json").write_text(json.dumps([str(i) for i in range(1200)]))
    for split, count in (("qtrain", 40), ("qdev", 4)):
        np.save(emb / f"phi_new_x1_{split}.fp16.npy", vectors(count))
        (emb / f"phi_new_x1_{split}.qids.json").write_text(json.dumps([f"{split}_{i}" for i in range(count)]))
    qrels = tmp_path / "qrels.tsv"
    qrels.write_text("".join(f"qdev_{i}\t{i}\t1\n" for i in range(4)))
    command = [sys.executable, "-m", "rankfill.run", "--emb-dir", str(emb),
               "--out", str(out), "--device", "cpu", "--seed-budget", "0.1",
               "--budgets", "0.2,0.3", "--epochs", "2", "--arc-epochs", "1",
               "--batch", "16", "--hidden", "32", "--query-sample", "30",
               "--threads", "1", "--qrels", str(qrels)]
    result = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((out / "manifest.json").read_text())
    assert [stage["selected_count"] for stage in manifest] == [120, 240, 360]
    previous = np.array([], dtype=int)
    for stage in manifest:
        selected = np.load(out / stage["selected"])
        assert np.isin(previous, selected).all()
        assert len(np.unique(selected)) == len(selected)
        previous = selected
    final = out / "budget_0.3"
    train_split = np.load(final / "arc/query_split.npz")
    acquisition = np.load(final / "acquisition_query_rows.npy")
    assert not np.intersect1d(acquisition, train_split["validation"]).size
    with np.load(final / "cache/qdev_rankings.npz") as z:
        ids = z["mapped"]
    anchor_scores = np.load(final / "cache/qdev_anchor_scores.npy")
    assert np.array_equal(np.isfinite(anchor_scores), np.isin(ids, previous))
    with np.load(final / "evaluation/predictions.npz") as z:
        assert np.array_equal(np.sort(z["mapped"], axis=1), np.sort(z["rankfill"], axis=1))
    metrics = json.loads((final / "evaluation/metrics.json").read_text())
    assert len(metrics) == 8 and all(0 <= v <= 1 for v in metrics.values())

"""Encode MS MARCO with native contracts for two public GTE-large versions."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    pooling: str
    max_length: int
    trust_remote_code: bool = False


MODEL_SPECS = {
    "gte_large_v1": ModelSpec("thenlper/gte-large", "mean", 512),
    "gte_large_v15": ModelSpec(
        "Alibaba-NLP/gte-large-en-v1.5", "cls", 8192, trust_remote_code=True
    ),
}


def pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "cls":
        return hidden[:, 0]
    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    raise ValueError(f"unknown pooling mode: {pooling}")


def validate_embedding(embedding: np.ndarray, expected_rows: int) -> None:
    expected = (expected_rows, 1024)
    if embedding.shape != expected:
        raise ValueError(f"unexpected GTE embedding shape {embedding.shape}, expected {expected}")
    if not np.isfinite(embedding).all():
        raise ValueError("GTE embedding contains non-finite values")
    norms = np.linalg.norm(embedding.astype(np.float32), axis=1)
    if len(norms) and not np.allclose(norms, 1.0, atol=2e-3):
        raise ValueError("GTE embeddings are not unit-normalized")


def load_corpus(limit: int | None = None) -> tuple[list[str], list[str]]:
    ids, texts = [], []
    with (DATA / "corpus.jsonl").open() as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            row = json.loads(line)
            ids.append(row["pid"]); texts.append(row["text"])
    return ids, texts


def load_queries(split: str, limit: int | None = None) -> tuple[list[str], list[str]]:
    if split not in {"dev", "train"}:
        raise ValueError(f"unknown query split: {split}")
    ids, texts = [], []
    with (DATA / f"queries.{split}.tsv").open() as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            qid, text = line.rstrip("\n").split("\t", 1)
            ids.append(qid); texts.append(text)
    return ids, texts


@torch.inference_mode()
def encode_texts(model, tokenizer, texts: list[str], spec: ModelSpec, device: str,
                 batch_size: int) -> np.ndarray:
    chunks = []
    for start in range(0, len(texts), batch_size):
        tokens = tokenizer(
            texts[start:start + batch_size], padding=True, truncation=True,
            max_length=spec.max_length, return_tensors="pt"
        ).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(**tokens).last_hidden_state
            embedding = F.normalize(pool_hidden(output, tokens["attention_mask"], spec.pooling), dim=1)
        chunks.append(embedding.cpu().numpy().astype(np.float16))
        if start % (batch_size * 100) == 0:
            print(f"[encode] {start}/{len(texts)}", flush=True)
    result = np.concatenate(chunks) if chunks else np.empty((0, 1024), dtype=np.float16)
    validate_embedding(result, len(texts))
    return result


def save_array(out: Path, tag: str, kind: str, ids: list[str], values: np.ndarray,
               suffix: str = "") -> None:
    np.save(out / f"{tag}_{kind}{suffix}.fp16.npy", values)
    id_name = "pids" if kind == "corpus" else "qids"
    (out / f"{tag}_{kind}{suffix}.{id_name}.json").write_text(json.dumps(ids))
    print(f"[saved] {tag} {kind}{suffix} {values.shape}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out-dir", default=str(DATA / "emb/gte-large"))
    parser.add_argument("--splits", default="corpus,dev,train")
    args = parser.parse_args()

    spec = MODEL_SPECS[args.tag]
    device = f"cuda:{args.gpu}"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id, trust_remote_code=spec.trust_remote_code)
    model = AutoModel.from_pretrained(
        spec.model_id, trust_remote_code=spec.trust_remote_code,
        torch_dtype=torch.float16
    ).to(device).eval()
    suffix = f"_lim{args.limit}" if args.limit is not None else ""

    for split in args.splits.split(","):
        split = split.strip()
        if split == "corpus":
            ids, texts = load_corpus(args.limit); kind = "corpus"
        elif split in {"dev", "train"}:
            ids, texts = load_queries(split, args.limit); kind = f"q{split}"
        else:
            raise ValueError(f"unknown split: {split}")
        values = encode_texts(model, tokenizer, texts, spec, device, args.batch)
        save_array(out, args.tag, kind, ids, values, suffix)


if __name__ == "__main__":
    main()

"""Build phi_old for a backbone: the pretrained base with the SAME architecture as
phi_new (pooling + L2 normalize) but WITHOUT MS MARCO fine-tuning.

Contriever provides the zero-shot starting point for the data-growth update.

Run:
  python src/build_phi_old.py --backbone contriever
"""
import argparse
import os
import sys

from finetune import build_encoder

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "configs"))
from backbones import get as get_backbone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="contriever")
    ap.add_argument("--max_len", type=int, default=256)
    args = ap.parse_args()
    bb = get_backbone(args.backbone)
    out = os.path.join(ROOT, "models", args.backbone, "backbone", "phi_old")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    model = build_encoder(bb["hf"], args.max_len, bb["pooling"])
    model.save(out)
    print(f"[done] saved phi_old ({args.backbone}: {bb['hf']}, {bb['pooling']} pool, "
          f"no MS MARCO fine-tune) -> {out}")


if __name__ == "__main__":
    main()

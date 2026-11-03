"""Encoder used in the canonical data-growth experiment."""
BACKBONES = {"contriever": {"hf": "facebook/contriever", "pooling": "mean"}}

def get(backbone):
    return BACKBONES[backbone]

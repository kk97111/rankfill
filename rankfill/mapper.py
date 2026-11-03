"""The residual mapper used by Rankfill (no uncertainty or baseline heads)."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Mapper(nn.Module):
    def __init__(self, dim=768, hidden=2048, alpha=0.03):
        super().__init__()
        # Parameter names match the original resscale_nobn checkpoints.
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))

    def forward(self, x):
        return x + self.alpha * self.net(x)


def load_mapper(checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    hidden, dim = state["net.0.weight"].shape
    model = Mapper(dim, hidden).to(device)
    model.load_state_dict(state)
    return model.eval()


@torch.inference_mode()
def mapped_gallery(old, checkpoint, device, batch=4096):
    model = load_mapper(checkpoint, device)
    gallery = torch.empty(old.shape, dtype=torch.float32, device=device)
    for start in range(0, len(old), batch):
        x = torch.tensor(np.asarray(old[start:start + batch], np.float32), device=device)
        # Match the original evaluator's CPU normalization of mapped vectors.
        gallery[start:start + len(x)] = F.normalize(model(x).cpu(), dim=1).to(device)
    return gallery

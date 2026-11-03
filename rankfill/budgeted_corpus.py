"""New-document storage that physically materializes only budget-counted S."""
import numpy as np
import torch


class SelectedNewCorpus:
    def __init__(self, source, selected, device):
        selected = np.asarray(selected, dtype=np.int64)
        self.shape = source.shape
        if selected.ndim != 1 or len(np.unique(selected)) != len(selected):
            raise ValueError('selected ids must be unique and one-dimensional')
        if np.any(selected < 0) or np.any(selected >= self.shape[0]):
            raise ValueError('selected ids outside corpus')
        # source can be a mmap. No unselected row is indexed or transferred.
        self.values = torch.from_numpy(np.asarray(source[selected]).copy()).to(device)
        lookup = np.full(self.shape[0], -1, dtype=np.int64)
        lookup[selected] = np.arange(len(selected))
        self.lookup = torch.from_numpy(lookup).to(device)

    def gather(self, indices, output_device):
        indices = indices.to(device=self.lookup.device, dtype=torch.long)
        if bool(((indices < 0) | (indices >= self.shape[0])).any()):
            raise ValueError('requested new ids outside corpus')
        local = self.lookup[indices]
        if bool((local < 0).any()):
            raise ValueError('requested new ids outside current selected budget S')
        return self.values[local].to(device=output_device, dtype=torch.float32)

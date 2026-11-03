"""Exact teacher top-J mining within the refreshed document set."""
import torch

def mine_extreme_local_ids(queries, documents, top_n, bottom_n=0, schunk=150000):
    """Return per-query local IDs ordered as teacher Top-N then Bottom-N."""
    if top_n < 0 or bottom_n < 0 or top_n + bottom_n <= 0:
        raise ValueError("top_n and bottom_n must be non-negative with a positive sum")
    if top_n + bottom_n > len(documents):
        raise ValueError("top_n + bottom_n cannot exceed the selected document pool")
    if schunk <= 0:
        raise ValueError("schunk must be positive")

    batch = len(queries)
    device = queries.device
    top_values = torch.full((batch, top_n), -torch.inf, device=device)
    top_ids = torch.full((batch, top_n), -1, dtype=torch.long, device=device)
    bottom_values = torch.full((batch, bottom_n), torch.inf, device=device)
    bottom_ids = torch.full((batch, bottom_n), -1, dtype=torch.long, device=device)

    for start in range(0, len(documents), schunk):
        stop = min(start + schunk, len(documents))
        scores = queries @ documents[start:stop].T
        chunk_ids = torch.arange(start, stop, device=device).unsqueeze(0).expand(batch, -1)
        if top_n:
            values = torch.cat((top_values, scores), dim=1)
            ids = torch.cat((top_ids, chunk_ids), dim=1)
            top_values, positions = values.topk(top_n, dim=1)
            top_ids = torch.gather(ids, 1, positions)
        if bottom_n:
            values = torch.cat((bottom_values, scores), dim=1)
            ids = torch.cat((bottom_ids, chunk_ids), dim=1)
            bottom_values, positions = values.topk(bottom_n, dim=1, largest=False)
            bottom_ids = torch.gather(ids, 1, positions)

    result = torch.cat((top_ids, bottom_ids), dim=1)
    if bottom_n:
        overlap = (top_ids[:, :, None] == bottom_ids[:, None, :]).any()
        assert not bool(overlap.item()), "teacher Top-N and Bottom-N candidates overlap"
    return result

"""Unstructured per-output-row pruning (magnitude / Wanda), nonuniform per-unit counts, and 2:4 masks."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch

from .calib import output_error, run_sequential
from .model_utils import GROUP_OF, block_linears, unit_name


def score_matrix(w: torch.Tensor, method: str, sq: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Pruning score S_ij. magnitude: |W_ij|; wanda: |W_ij| * ||X_j||_2."""
    if method == "magnitude":
        return w.float().abs()
    if method == "wanda":
        assert sq is not None
        return w.float().abs() * sq.float().sqrt().unsqueeze(0)
    raise ValueError(method)


@torch.no_grad()
def row_prune_mask(score: torch.Tensor, row_counts: torch.Tensor) -> torch.Tensor:
    """Boolean mask (True = pruned) removing the `row_counts[r]` lowest-score weights of each row r.

    Ties are broken by a stable sort on the score (lower column index first)."""
    rows, cols = score.shape
    order = torch.argsort(score, dim=1, stable=True)  # ascending
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(cols, device=score.device).expand(rows, cols))
    return ranks < row_counts.view(-1, 1).to(score.device)


@torch.no_grad()
def nm_mask(score: torch.Tensor, n: int = 2, m: int = 4) -> torch.Tensor:
    """N:M mask (True = pruned): in each group of m contiguous input channels keep the n highest scores."""
    rows, cols = score.shape
    assert cols % m == 0
    s = score.view(rows, cols // m, m)
    idx = torch.topk(s, m - n, dim=-1, largest=False).indices
    mask = torch.zeros_like(s, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return mask.view(rows, cols)


def uniform_row_counts(shape, sparsity: float) -> torch.Tensor:
    rows, cols = shape
    k = int(round(sparsity * cols))
    return torch.full((rows,), k, dtype=torch.long)


def prune_model(
    model,
    calib_ids: torch.Tensor,
    method: str,
    row_counts: Dict[str, torch.Tensor],
    structure: str = "unstructured",
    batch_size: int = 8,
    keep_scores: bool = False,
) -> dict:
    """Block-sequential pruning. row_counts maps unit name -> LongTensor[rows] of weights to prune per row.

    Returns per-unit achieved sparsity, layer-output errors on calibration activations, optional score
    summaries, and calibration/compression timing."""
    unit_stats: Dict[str, dict] = {}
    score_summ: Dict[str, dict] = {}

    def block_fn(li, layer, inputs, ctx):
        for p, lin in block_linears(layer).items():
            u = unit_name(li, p)
            w = lin.weight.data
            g = inputs[GROUP_OF[p]]
            score = score_matrix(w, method, g["sq"])
            if structure == "2:4":
                mask = nm_mask(score, 2, 4)
            else:
                mask = row_prune_mask(score, row_counts[u])
            w_ref = w.clone()
            w.masked_fill_(mask, 0)
            err = output_error(w_ref, w, g["X"])
            unit_stats[u] = dict(sparsity=float(mask.float().mean().item()), zeros=int(mask.sum().item()),
                                 numel=w.numel(), out_err_rel=err["rel"], out_err_abs=err["abs"])
            if keep_scores:
                s = score.flatten()
                qs = torch.quantile(s[torch.randperm(s.numel(), device=s.device)[:1_000_000]].float(),
                                    torch.tensor([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99], device=s.device))
                pruned_mag = w_ref.float().abs()[mask]
                score_summ[u] = dict(score_quantiles=qs.tolist(), score_mean=float(s.mean()),
                                     pruned_abs_w_mean=float(pruned_mag.mean()) if pruned_mag.numel() else 0.0,
                                     kept_abs_w_mean=float(w_ref.float().abs()[~mask].mean()))
            del score, mask, w_ref
        return {}

    res = run_sequential(model, calib_ids, block_fn, batch_size=batch_size, collect_outputs_error=True)
    res["units"] = unit_stats
    if keep_scores:
        res["score_summary"] = score_summ
    return res


@torch.no_grad()
def magnitude_prune_inplace(model, row_counts: Dict[str, torch.Tensor]) -> None:
    """Data-free magnitude pruning (used for quick checks; the experiments use prune_model for errors)."""
    from .model_utils import eligible_linears

    for u, lin in eligible_linears(model).items():
        mask = row_prune_mask(lin.weight.float().abs(), row_counts[u])
        lin.weight.data.masked_fill_(mask, 0)

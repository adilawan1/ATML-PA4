"""Sensitivity probes for nonuniform pruning (P3) and mixed precision (Q3).

All probes use the 128 C4 calibration sequences only.

1. Empirical Fisher diagonal F_ij = E_seq[(dL_seq/dW_ij)^2] of the calibration NLL on the unchanged FP16
   model (Molchanov et al. 2019; HAWQ-V2-style trace weighting uses the same second-order proxy).
2. Dense block-sequential probe pass. For every eligible projection u:
   * pruning cost curve  Omega_u(s) = 1/2 sum_{(i,j) pruned by Wanda at per-row sparsity s} F_ij W_ij^2
     for s on a 0.025 grid in [0.10, 0.85] (Wanda ranking rule unchanged, only the amount varies);
   * OWL statistics: fraction of Wanda scores > M * (block-mean score), M in {3,5,7,10};
   * magnitude/Wanda pruned-set overlap (Jaccard) at s in {0.25, 0.5, 0.75}  (research question 1);
   * quantization costs at b in {2,3,4,8} with the AWQ-style quantizer (alpha searched per group at
     that bit width): Fisher-weighted perturbation 1/2 sum F (W - W_eff)^2 and calibration output error
     ||(W - W_eff) X||^2, plus RTN (alpha=0) errors and exact packed bytes.
"""
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import torch

from .awq import ALPHA_GRID, _err_only, _make_scale, search_scale
from .calib import run_sequential
from .model_utils import INPUT_GROUPS, block_linears, eligible_linears, unit_name
from .prune import row_prune_mask, score_matrix
from .quant import fake_quant, packed_bytes

PRUNE_GRID = [round(0.10 + 0.025 * i, 3) for i in range(31)]  # 0.10 ... 0.85
BITS = [2, 3, 4, 8]
OWL_M = [3, 5, 7, 10]


def fisher_diagonal(model, calib_ids: torch.Tensor, loss_scale: float = 1024.0) -> dict:
    """Accumulate squared gradients of the per-sequence mean NLL (FP16 model, FP32 accumulation)."""
    elig = eligible_linears(model)
    for lin in elig.values():
        lin.weight.requires_grad_(True)
    F = {u: torch.zeros_like(lin.weight, dtype=torch.float32) for u, lin in elig.items()}
    dev = next(model.parameters()).device
    used = skipped = 0
    t0 = time.perf_counter()
    for i in range(calib_ids.shape[0]):
        x = calib_ids[i:i + 1].to(dev)
        out = model(input_ids=x, labels=x, use_cache=False)
        (out.loss * loss_scale).backward()
        grads = [lin.weight.grad for lin in elig.values()]
        if any((g is None) or (not torch.isfinite(g).all()) for g in grads):
            skipped += 1
        else:
            for u, lin in elig.items():
                F[u].add_((lin.weight.grad.float() / loss_scale).pow(2))
            used += 1
        for lin in elig.values():
            lin.weight.grad = None
    for lin in elig.values():
        lin.weight.requires_grad_(False)
    for u in F:
        F[u].div_(max(used, 1))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return dict(F=F, used=used, skipped=skipped, seconds=time.perf_counter() - t0, loss_scale=loss_scale)


@torch.no_grad()
def probe(model, calib_ids: torch.Tensor, F: Dict[str, torch.Tensor], batch_size: int = 8,
          bits: List[int] = BITS, grid=ALPHA_GRID) -> dict:
    out: Dict[str, dict] = {}
    blocks: Dict[int, dict] = {}

    def block_fn(li, layer, inputs, ctx):
        lins = block_linears(layer)
        scores = {}
        for p, lin in lins.items():
            u = unit_name(li, p)
            g = next(gname for gname, ps in INPUT_GROUPS.items() if p in ps)
            w = lin.weight.data
            Fu = F[u]
            wanda = score_matrix(w, "wanda", inputs[g]["sq"])
            scores[p] = wanda
            rows, cols = w.shape
            # Fisher-weighted pruning cost along the Wanda order
            order = torch.argsort(wanda, dim=1, stable=True)
            contrib = 0.5 * Fu * w.float().pow(2)
            csum = torch.gather(contrib, 1, order).cumsum(1)
            ks = [int(round(s * cols)) for s in PRUNE_GRID]
            omega = [float(csum[:, k - 1].sum().item()) for k in ks]
            # magnitude vs Wanda pruned sets
            mag = score_matrix(w, "magnitude")
            overlap = {}
            for s in (0.25, 0.5, 0.75):
                k = torch.full((rows,), int(round(s * cols)), dtype=torch.long)
                a = row_prune_mask(wanda, k)
                b = row_prune_mask(mag, k)
                inter = (a & b).sum().item()
                overlap[str(s)] = inter / max((a | b).sum().item(), 1)
            out[u] = dict(layer=li, proj=p, shape=[rows, cols], prune_grid=PRUNE_GRID, prune_rowk=ks,
                          prune_fisher=omega, fisher_trace=float(Fu.sum().item()),
                          fisher_w2=float((Fu * w.float().pow(2)).sum().item()),
                          mag_wanda_jaccard=overlap, wanda_mean=float(wanda.mean().item()),
                          act_norm_mean=float(inputs[g]["sq"].sqrt().mean().item()),
                          act_norm_max=float(inputs[g]["sq"].sqrt().max().item()))
            del order, contrib, csum, mag
        # OWL layerwise outlier ratio (block = layer in OWL's terminology)
        tot = sum(s.numel() for s in scores.values())
        block_mean = sum(s.sum().item() for s in scores.values()) / tot
        owl = {}
        for M in OWL_M:
            owl[str(M)] = sum((s > M * block_mean).sum().item() for s in scores.values()) / tot
            for p, s in scores.items():
                out[unit_name(li, p)].setdefault("owl_unit", {})[str(M)] = float((s > M * s.mean()).float().mean().item())
        blocks[li] = dict(owl=owl, numel=tot)
        del scores
        # quantization costs
        for b in bits:
            for g, projs in INPUT_GROUPS.items():
                X = inputs[g]["X"]
                wts = {p: lins[p].weight.data for p in projs}
                r = search_scale(wts, {p: b for p in projs}, X, inputs[g]["absmean"], grid)
                s = r["scale"]
                for p in projs:
                    u = unit_name(li, p)
                    w = wts[p]
                    w_eff = fake_quant((w.float() * s).to(w.dtype), b).float() / s
                    w_rtn = fake_quant(w, b).float()
                    d = out[u].setdefault("quant", {})
                    d[str(b)] = dict(
                        alpha=r["alpha"],
                        fisher=float((0.5 * F[u] * (w.float() - w_eff).pow(2)).sum().item()),
                        out_err=_err_only(w, w_eff, X),
                        fisher_rtn=float((0.5 * F[u] * (w.float() - w_rtn).pow(2)).sum().item()),
                        out_err_rtn=_err_only(w, w_rtn, X),
                        bytes=packed_bytes(tuple(w.shape), b)["total"],
                    )
        for p in lins:
            u = unit_name(li, p)
            X = inputs[next(g for g, ps in INPUT_GROUPS.items() if p in ps)]["X"]
            out[u]["out_energy"] = _err_only(lins[p].weight.data, torch.zeros_like(lins[p].weight.data), X)
        return {}

    t0 = time.perf_counter()
    res = run_sequential(model, calib_ids, block_fn, batch_size=batch_size)
    return dict(units=out, blocks=blocks, timing=res["timing"], seconds=time.perf_counter() - t0,
                bits=bits, prune_grid=PRUNE_GRID, owl_M=OWL_M, alpha_grid=list(grid))

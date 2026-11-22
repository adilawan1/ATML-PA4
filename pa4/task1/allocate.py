"""Allocation rules turning sensitivity measurements into constrained per-unit budgets.

* greedy_knapsack: multiple-choice knapsack by greedy marginal analysis. Each unit starts at its most
  compressed level and repeatedly the unit with the largest (cost reduction / extra resource) is upgraded
  while the global resource budget is respected (Lagrangian/greedy solution of HAWQ-V2-style constrained
  assignment; exact for convex cost curves).
* owl_sparsities: OWL (Yin et al. 2024) layerwise sparsity from outlier ratios with hyper-parameter lambda.
* exact_row_counts: converts per-unit sparsities to integer per-row prune counts and redistributes the
  rounding residual one weight per row so the total number of zeroed eligible weights equals the target
  exactly, keeping every unit inside [0.10, 0.85] up to one weight per row.
"""
from __future__ import annotations

import heapq
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

S_MIN, S_MAX = 0.10, 0.85


def greedy_knapsack(costs: Dict[str, Sequence[float]], resources: Dict[str, Sequence[float]], budget: float):
    """Levels are ordered from most to least compressed (resource increasing, cost ideally decreasing).

    Returns (level index per unit, total resource, total cost, n_upgrades)."""
    level = {u: 0 for u in costs}
    used = sum(resources[u][0] for u in costs)
    if used > budget + 1e-9:
        raise ValueError(f"Most compressed allocation ({used}) already exceeds budget ({budget})")
    heap: List[Tuple[float, str]] = []

    def push(u):
        l = level[u]
        if l + 1 < len(costs[u]):
            dr = resources[u][l + 1] - resources[u][l]
            dc = costs[u][l] - costs[u][l + 1]  # reduction
            ratio = dc / max(dr, 1e-12)
            heapq.heappush(heap, (-ratio, u))

    for u in costs:
        push(u)
    n = 0
    while heap:
        _, u = heapq.heappop(heap)
        l = level[u]
        dr = resources[u][l + 1] - resources[u][l]
        if used + dr <= budget + 1e-9:
            level[u] = l + 1
            used += dr
            n += 1
            push(u)
        # else: this unit cannot be upgraded further within budget; drop it
    total_cost = sum(costs[u][level[u]] for u in costs)
    return level, used, total_cost, n


def owl_sparsities(outlier_ratio: Sequence[float], target: float, lam: float) -> np.ndarray:
    """OWL: density_l = 2*lam*(D_l - min)/(max - min) - mean(.) + (1 - target); sparsity = 1 - density."""
    D = np.asarray(outlier_ratio, float)
    r = (D - D.min()) / max(D.max() - D.min(), 1e-12) * 2 * lam
    density = r - r.mean() + (1 - target)
    return 1 - density


def rescale_clip(s: np.ndarray, sizes: np.ndarray, target: float, lo=S_MIN, hi=S_MAX) -> np.ndarray:
    """Clip to [lo, hi] and shift by a constant c (bisection) so the size-weighted mean equals target."""
    sizes = np.asarray(sizes, float)

    def mean_at(c):
        return float((np.clip(s + c, lo, hi) * sizes).sum() / sizes.sum())

    a, b = -1.0, 1.0
    for _ in range(200):
        m = (a + b) / 2
        if mean_at(m) < target:
            a = m
        else:
            b = m
    return np.clip(s + (a + b) / 2, lo, hi)


def exact_row_counts(unit_sparsity: Dict[str, float], shapes: Dict[str, Tuple[int, int]], target_zeros: int,
                     lo=S_MIN, hi=S_MAX) -> Dict[str, torch.Tensor]:
    """Integer per-row prune counts whose sum equals target_zeros exactly."""
    counts = {}
    total = 0
    for u, s in unit_sparsity.items():
        rows, cols = shapes[u]
        k = int(math.floor(s * cols))
        k = min(max(k, int(math.ceil(lo * cols))), int(math.floor(hi * cols)))
        counts[u] = torch.full((rows,), k, dtype=torch.long)
        total += k * rows
    diff = target_zeros - total
    if diff != 0:
        step = 1 if diff > 0 else -1
        # units whose desired sparsity was rounded the most get the adjustment first (fractional part)
        def frac(u):
            x = unit_sparsity[u] * shapes[u][1]
            return x - math.floor(x)
        order = sorted(unit_sparsity, key=lambda u: -frac(u) if step > 0 else frac(u))
        capacity = sum(shapes[u][0] for u in order)
        if abs(diff) > capacity:
            raise RuntimeError(f"residual {diff} exceeds one weight per row ({capacity} rows)")
        left = abs(diff)
        for u in order:  # at most one extra (or one fewer) pruned weight per row
            if left == 0:
                break
            n = min(left, shapes[u][0])
            counts[u][:n] += step
            left -= n
    assert sum(int(c.sum()) for c in counts.values()) == target_zeros
    return counts


def fisher_prune_allocation(probe_units: Dict[str, dict], target_sparsity: float, total_eligible: int):
    """Knapsack over the PRUNE_GRID levels using the Fisher pruning-cost curves (unit = projection)."""
    costs, res, sp = {}, {}, {}
    for u, d in probe_units.items():
        rows, cols = d["shape"]
        grid = d["prune_grid"][::-1]  # most compressed first (0.85 -> 0.10)
        om = d["prune_fisher"][::-1]
        ks = d["prune_rowk"][::-1]
        costs[u] = om
        res[u] = [rows * (cols - k) for k in ks]  # kept weights
        sp[u] = grid
    target_zeros = int(round(target_sparsity * total_eligible))
    budget_kept = total_eligible - target_zeros
    level, used, cost, n = greedy_knapsack(costs, res, budget_kept)
    unit_s = {u: sp[u][level[u]] for u in costs}
    return unit_s, dict(target_zeros=target_zeros, kept_after_greedy=used, predicted_cost=cost, upgrades=n)


def quant_allocation(probe_units: Dict[str, dict], budget_bytes: int, key: str = "fisher", bits=(2, 3, 4, 8)):
    """Knapsack over bit widths using a chosen per-unit cost ('fisher' or 'out_err'); resource = packed bytes."""
    costs, res = {}, {}
    for u, d in probe_units.items():
        q = d["quant"]
        costs[u] = [q[str(b)][key] for b in bits]
        res[u] = [q[str(b)]["bytes"] for b in bits]
    level, used, cost, n = greedy_knapsack(costs, res, budget_bytes)
    return {u: bits[level[u]] for u in costs}, dict(bytes=used, budget=budget_bytes, predicted_cost=cost, upgrades=n)

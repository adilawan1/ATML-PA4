"""Task 2B training protocol (fixed for every mechanism).

AdamW(peak lr 8e-4, wd 0.01), batch 32, 4,500 updates, 200 linear warm-up updates then linear decay to
15% of peak, global-norm clipping 1.0, dev evaluation at updates {500,1000,1500,2000,3000,4500}, seeds
6304/6305/6306. Training examples come from the supplied deterministic generator (split='train'); the
stream for seed s uses indices (s-6304)*10^6 + i, so all mechanisms trained with the same (K, regime,
seed) see exactly the same sequences in the same order.
"""
from __future__ import annotations

import os
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from ..common.data import mqar_generator
from ..common.utils import ckpt_root, set_seed
from .mqar import N_CLASSES, VALUE_OFFSET, MQARModel, evaluate, n_params

STEPS, BATCH, PEAK_LR, WD, WARMUP, FINAL_FRAC, CLIP = 4500, 32, 8e-4, 0.01, 200, 0.15, 1.0
EVAL_AT = [500, 1000, 1500, 2000, 3000, 4500]
SEEDS = [6304, 6305, 6306]


def lr_at(step: int) -> float:
    """step is 1-based update index."""
    if step <= WARMUP:
        return PEAK_LR * step / WARMUP
    frac = (step - WARMUP) / (STEPS - WARMUP)
    return PEAK_LR * (1 - (1 - FINAL_FRAC) * frac)


_GEN = None


def _gen(args):
    global _GEN
    if _GEN is None:
        _GEN = mqar_generator()
    k, regime, idx = args
    ex = _GEN.make_example("train", k, regime, idx)
    return ex["tokens"], ex["query_positions"], ex["answer_ids"], ex["source_positions"]


def train_stream(k: int, regime: str, seed: int, n: int = STEPS * BATCH, workers: int = 4) -> Dict[str, np.ndarray]:
    path = ckpt_root() / "mqar_train" / f"K{k}_{regime}_seed{seed}_n{n}.npz"
    if path.exists():
        z = np.load(path)
        return {kk: z[kk] for kk in z.files}
    base = (seed - 6304) * 1_000_000
    args = [(k, regime, base + i) for i in range(n)]
    t0 = time.time()
    with Pool(workers) as pool:
        res = pool.map(_gen, args, chunksize=512)
    out = dict(tokens=np.array([r[0] for r in res], dtype=np.int16), qpos=np.array([r[1] for r in res], dtype=np.int16),
               ans=np.array([r[2] for r in res], dtype=np.int16), src=np.array([r[3] for r in res], dtype=np.int16),
               index=np.arange(base, base + n))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:  # atomic: a disconnect never leaves a truncated stream cache behind
        np.savez_compressed(f, **out)
    os.replace(tmp, path)
    print(f"generated {path.name} in {time.time() - t0:.0f}s", flush=True)
    return out


def train_run(kind: str, k: int, regime: str, seed: int, dev_rows: List[dict], device, steps: int = STEPS,
              log_every: int = 50, **model_kw) -> dict:
    data = train_stream(k, regime, seed)
    set_seed(seed)
    model = MQARModel(kind, **model_kw).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=WD)
    curve, dev_curve = [], []
    run_loss, run_n = 0.0, 0
    t_train = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    for step in range(1, steps + 1):
        sl = slice((step - 1) * BATCH, step * BATCH)
        tok = torch.from_numpy(data["tokens"][sl].astype(np.int64)).to(device)
        qp = torch.from_numpy(data["qpos"][sl].astype(np.int64)).to(device)
        ans = torch.from_numpy(data["ans"][sl].astype(np.int64) - VALUE_OFFSET).to(device)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model(tok, qp)
        loss = F.cross_entropy(logits.reshape(-1, N_CLASSES), ans.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_train += time.perf_counter() - t0
        run_loss += float(loss.item())
        run_n += 1
        if step % log_every == 0:
            curve.append(dict(step=step, loss=run_loss / run_n, lr=lr_at(step), grad_norm=float(gn)))
            run_loss, run_n = 0.0, 0
        if step in EVAL_AT:
            ev = evaluate(model, dev_rows, device)
            dev_curve.append(dict(step=step, acc=ev["acc"], loss=ev["loss"], acc_near=ev["acc_near"],
                                  acc_far=ev["acc_far"], acc_mixed=ev["acc_mixed"]))
            print(f"  [{kind} K{k} {regime} s{seed}] step {step} loss {curve[-1]['loss']:.3f} dev acc {ev['acc']:.3f}", flush=True)
    peak = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
    return dict(model=model, curve=curve, dev_curve=dev_curve, train_seconds=t_train,
                ms_per_step=1000 * t_train / steps, peak_train_alloc_bytes=peak, n_params=n_params(model))

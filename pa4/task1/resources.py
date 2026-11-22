"""Resource accounting for Task 1: storage representations, latency, and peak GPU memory.

All compressed conditions are executed with dense FP16 kernels (zeros and dequantized weights are still
stored as dense FP16 tensors). Packed / sparse sizes are therefore *serialized-storage estimates*
derived from actual bit-packed tensors (quantization) or the specified mask+value model (pruning),
reported next to the measured dense-FP16 latency and memory.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from ..common.utils import Timer, time_fn
from .model_utils import eligible_linears, parameter_account
from .quant import sparse16_bytes


def storage_report(model, quant_eligible_bytes: Optional[int] = None) -> dict:
    acc = parameter_account(model)
    other_fp16 = 2 * acc.n_other  # embeddings (tied, counted once), norms, biases: uncompressed FP16
    nnz = numel = 0
    sparse_bytes = 0
    for lin in eligible_linears(model).values():
        n = lin.weight.numel()
        z = int((lin.weight != 0).sum().item())
        nnz += z
        numel += n
        sparse_bytes += sparse16_bytes(n, z)
    rep = dict(
        n_eligible=acc.n_eligible, n_other_unique=acc.n_other, tied_embeddings=acc.tied_embeddings,
        eligible_nonzeros=nnz, eligible_sparsity=1 - nnz / numel,
        dense_fp16_eligible_bytes=2 * numel, uncompressed_other_bytes=other_fp16,
        dense_fp16_total_bytes=2 * numel + other_fp16,
        sparse16_eligible_bytes=sparse_bytes, sparse16_total_bytes=sparse_bytes + other_fp16,
        nominal_bits_per_eligible_weight_dense=16.0,
        nominal_bits_per_eligible_weight_sparse16=8 * sparse_bytes / numel,
    )
    if quant_eligible_bytes is not None:
        rep.update(packed_eligible_bytes=int(quant_eligible_bytes),
                   packed_total_bytes=int(quant_eligible_bytes) + other_fp16,
                   nominal_bits_per_eligible_weight_packed=8 * quant_eligible_bytes / numel)
    return rep


@torch.no_grad()
def latency_report(model, reps: int = 5, warmup: int = 2, prompt_lens=(512, 2048), decode_tokens: int = 64,
                   decode_prompt: int = 512, seed: int = 6304) -> dict:
    """Prefill latency at fixed shapes, 64-token greedy decode latency (KV cache), peak allocated memory."""
    dev = next(model.parameters()).device
    g = torch.Generator().manual_seed(seed)
    out = dict(unit="seconds", batch_size=1, reps=reps, warmup=warmup, kernels="dense FP16 (torch SDPA + cuBLAS)")
    for L in prompt_lens:
        ids = torch.randint(0, model.config.vocab_size - 1000, (1, L), generator=g).to(dev)

        def prefill():
            model(input_ids=ids, use_cache=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t = time_fn(prefill, reps, warmup)
        t["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        out[f"prefill_{L}"] = t
    ids = torch.randint(0, model.config.vocab_size - 1000, (1, decode_prompt), generator=g).to(dev)

    def decode():
        o = model(input_ids=ids, use_cache=True)
        past = o.past_key_values
        nxt = o.logits[:, -1:].argmax(-1)
        with Timer() as t:
            for _ in range(decode_tokens):
                o = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past = o.past_key_values
                nxt = o.logits[:, -1:].argmax(-1)
        return t.elapsed

    for _ in range(warmup):
        decode()
    if torch.cuda.is_available():
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    times = [decode() for _ in range(reps)]
    import numpy as np

    out[f"decode_{decode_tokens}"] = dict(prompt_len=decode_prompt, mean=float(np.mean(times)),
                                          std=float(np.std(times, ddof=1)), all=times,
                                          per_token_ms=1000 * float(np.mean(times)) / decode_tokens,
                                          peak_allocated_bytes=int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None)
    if torch.cuda.is_available():
        out["model_allocated_bytes"] = int(torch.cuda.memory_allocated())
    return out

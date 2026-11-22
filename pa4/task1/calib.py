"""Sequential (block-by-block) calibration runner.

For block l we (1) run the *current* block on the hidden states produced by the already-compressed
blocks 0..l-1, recording the inputs of every eligible projection; (2) call a compression callback
that may modify the block's weights using those inputs; (3) re-run the compressed block to obtain the
inputs of block l+1. This is the layer-wise protocol used by Wanda/SparseGPT/AWQ reference code.

Recorded per input group (attn_in, o_in, mlp_in, down_in):
  X      : FP16 matrix [N_tokens, d_in] of all calibration inputs (kept only for the current block);
  sq     : FP32 per-channel sum of squares  (Wanda: ||X_j||_2^2);
  absmean: FP32 per-channel mean |x|        (AWQ activation statistic).
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

__all__ = ["run_sequential", "output_error", "calib_tensor", "call_layer"]

import torch

from .model_utils import INPUT_GROUPS, block_linears, get_layers


class _Stop(Exception):
    pass


class BlockInputs:
    def __init__(self):
        self.X: Dict[str, List[torch.Tensor]] = {g: [] for g in INPUT_GROUPS}

    def finalize(self):
        out = {}
        for g, chunks in self.X.items():
            X = torch.cat(chunks, 0)
            out[g] = dict(X=X, sq=X.float().pow(2).sum(0), absmean=X.float().abs().mean(0), n=X.shape[0])
        return out


def call_layer(layer, h, kwargs):
    out = layer(h, **kwargs)
    return out[0] if isinstance(out, (tuple, list)) else out


@torch.no_grad()
def capture_first_inputs(model, token_ids: torch.Tensor, batch_size: int):
    """Hidden states entering block 0 for all calibration sequences, and the layer kwargs."""
    layers = get_layers(model)
    store = {"h": [], "kwargs": None}

    def pre_hook(mod, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        store["h"].append(h.detach())
        if store["kwargs"] is None:
            kw = {k: v for k, v in kwargs.items() if k != "hidden_states"}
            kw["use_cache"] = False
            kw["past_key_values"] = None
            kw.pop("past_key_value", None)
            store["kwargs"] = kw
        raise _Stop

    handle = layers[0].register_forward_pre_hook(pre_hook, with_kwargs=True)
    dev = next(model.parameters()).device
    try:
        for i in range(0, token_ids.shape[0], batch_size):
            try:
                model(input_ids=token_ids[i:i + batch_size].to(dev), use_cache=False)
            except _Stop:
                pass
    finally:
        handle.remove()
    return store["h"], store["kwargs"]


def _register_input_hooks(layer, rec: BlockInputs):
    lins = block_linears(layer)
    handles = []
    for g, projs in INPUT_GROUPS.items():
        lin = lins[projs[0]]

        def hook(mod, args, g=g):
            x = args[0]
            rec.X[g].append(x.detach().reshape(-1, x.shape[-1]).half())

        handles.append(lin.register_forward_pre_hook(hook))
    return handles


@torch.no_grad()
def run_sequential(
    model,
    token_ids: torch.Tensor,
    block_fn: Optional[Callable] = None,
    batch_size: int = 8,
    collect_outputs_error: bool = False,
) -> dict:
    """Run block-sequential calibration. `block_fn(layer_idx, layer, inputs, ctx) -> dict` may modify weights.

    ctx = dict(hs=list of block-input hidden-state batches, kwargs=decoder-layer kwargs)."""
    layers = get_layers(model)
    t0 = time.perf_counter()
    hs, kwargs = capture_first_inputs(model, token_ids, batch_size)
    timing = dict(capture=time.perf_counter() - t0, stats=0.0, compress=0.0, propagate=0.0)
    per_block = []
    for li, layer in enumerate(layers):
        t = time.perf_counter()
        rec = BlockInputs()
        handles = _register_input_hooks(layer, rec)
        dense_out = []
        for h in hs:
            o = call_layer(layer, h, kwargs)
            if collect_outputs_error:
                dense_out.append(o)
        for hd in handles:
            hd.remove()
        inputs = rec.finalize()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timing["stats"] += time.perf_counter() - t
        t = time.perf_counter()
        info = block_fn(li, layer, inputs, dict(hs=hs, kwargs=kwargs)) if block_fn is not None else {}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timing["compress"] += time.perf_counter() - t
        t = time.perf_counter()
        new_hs = []
        num = den = 0.0
        for bi, h in enumerate(hs):
            o = call_layer(layer, h, kwargs)
            if collect_outputs_error:
                num += (o.float() - dense_out[bi].float()).pow(2).sum().item()
                den += dense_out[bi].float().pow(2).sum().item()
            new_hs.append(o)
        if collect_outputs_error:
            info = dict(info or {})
            info["block_output_rel_err"] = num / max(den, 1e-30)
        hs = new_hs
        del inputs, rec, dense_out
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timing["propagate"] += time.perf_counter() - t
        per_block.append(info or {})
    return dict(per_block=per_block, timing=timing)


# ----------------------------------------------------------------------------- error helpers


@torch.no_grad()
def output_error(w_ref: torch.Tensor, w_new: torch.Tensor, X: torch.Tensor, chunk: int = 8192) -> Dict[str, float]:
    """||(W_ref - W_new) X^T||_F^2 and relative error vs ||W_ref X^T||_F^2 on calibration inputs."""
    dW = (w_ref.float() - w_new.float()).to(X.dtype)
    Wr = w_ref.to(X.dtype)
    num = den = 0.0
    for i in range(0, X.shape[0], chunk):
        xc = X[i:i + chunk]
        a = xc @ dW.t()
        b = xc @ Wr.t()
        if not (torch.isfinite(a).all() and torch.isfinite(b).all()):  # FP16 overflow -> recompute in FP32
            a = xc.float() @ dW.float().t()
            b = xc.float() @ Wr.float().t()
        num += a.float().pow(2).sum().item()
        den += b.float().pow(2).sum().item()
    return dict(abs=num, rel=num / max(den, 1e-30))


def calib_tensor(rows) -> torch.Tensor:
    return torch.tensor([r["token_ids"] for r in rows], dtype=torch.long)

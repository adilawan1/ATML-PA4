"""Task 2A: exact causal attention implementations.

dense_attention   : explicit QK^T, causal mask, numerically stable softmax (FP32 statistics), P @ V.
TiledAttention    : exact blockwise online-softmax forward that never materialises the L x L score
                    matrix (only Bq x Bk tiles), plus a recomputation-based tiled backward that stores only
                    O and the per-row log-sum-exp (FlashAttention-2 algorithm written in PyTorch). It is a
                    *Python reference*, not a fused kernel: every tile is a separate CUDA launch.
sdpa_attention    : torch.nn.functional.scaled_dot_product_attention with an optional forced backend.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def dense_attention(q, k, v, causal: bool = True, mask: Optional[torch.Tensor] = None):
    """q,k,v: [B,H,L,d]. mask: optional bool [.., L, L] (True = may attend). Fully-masked rows return 0."""
    d = q.shape[-1]
    acc_t = torch.promote_types(q.dtype, torch.float32)
    s = (q @ k.transpose(-1, -2)).to(acc_t) * (1.0 / math.sqrt(d))
    L, S = q.shape[-2], k.shape[-2]
    allow = torch.ones(L, S, dtype=torch.bool, device=q.device)
    if causal:
        allow = torch.tril(allow, diagonal=S - L)
    if mask is not None:
        allow = allow & mask
    s = s.masked_fill(~allow, float("-inf"))
    m = s.amax(-1, keepdim=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))  # all-masked rows
    p = torch.exp(s - m)
    denom = p.sum(-1, keepdim=True)
    p = p / denom.clamp(min=1e-30)
    return (p.to(v.dtype) @ v)


class _TiledFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bq: int, bk: int):
        B, H, L, d = q.shape
        scale = 1.0 / math.sqrt(d)
        acc_t = torch.promote_types(q.dtype, torch.float32)
        out = torch.empty_like(q)
        lse = torch.empty(B, H, L, device=q.device, dtype=acc_t)
        for qs in range(0, L, bq):
            qe = min(qs + bq, L)
            qb = q[:, :, qs:qe]
            m = torch.full((B, H, qe - qs), float("-inf"), device=q.device, dtype=acc_t)
            ell = torch.zeros((B, H, qe - qs), device=q.device, dtype=acc_t)
            acc = torch.zeros((B, H, qe - qs, d), device=q.device, dtype=acc_t)
            for ks in range(0, qe, bk):  # causal: key tiles beyond the last query of this tile are skipped
                ke = min(ks + bk, L)
                s = (qb @ k[:, :, ks:ke].transpose(-1, -2)).to(acc_t) * scale
                if ke > qs:  # tile crosses the diagonal
                    qi = torch.arange(qs, qe, device=q.device).view(-1, 1)
                    kj = torch.arange(ks, ke, device=q.device).view(1, -1)
                    s = s.masked_fill(kj > qi, float("-inf"))
                m_new = torch.maximum(m, s.amax(-1))
                p = torch.exp(s - m_new.unsqueeze(-1))
                corr = torch.exp(m - m_new)
                ell = corr * ell + p.sum(-1)
                acc = corr.unsqueeze(-1) * acc + (p.to(v.dtype) @ v[:, :, ks:ke]).to(acc_t)
                m = m_new
            out[:, :, qs:qe] = (acc / ell.unsqueeze(-1)).to(q.dtype)
            lse[:, :, qs:qe] = m + torch.log(ell)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.bq, ctx.bk = bq, bk
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        bq, bk = ctx.bq, ctx.bk
        B, H, L, d = q.shape
        scale = 1.0 / math.sqrt(d)
        acc_t = lse.dtype
        dq = torch.zeros(q.shape, device=q.device, dtype=acc_t)
        dk = torch.zeros(k.shape, device=q.device, dtype=acc_t)
        dv = torch.zeros(v.shape, device=q.device, dtype=acc_t)
        Dvec = (dout.to(acc_t) * out.to(acc_t)).sum(-1)  # [B,H,L]
        for qs in range(0, L, bq):
            qe = min(qs + bq, L)
            qb = q[:, :, qs:qe].to(acc_t)
            dob = dout[:, :, qs:qe].to(acc_t)
            for ks in range(0, qe, bk):
                ke = min(ks + bk, L)
                kb, vb = k[:, :, ks:ke].to(acc_t), v[:, :, ks:ke].to(acc_t)
                s = (qb @ kb.transpose(-1, -2)) * scale
                if ke > qs:
                    qi = torch.arange(qs, qe, device=q.device).view(-1, 1)
                    kj = torch.arange(ks, ke, device=q.device).view(1, -1)
                    s = s.masked_fill(kj > qi, float("-inf"))
                p = torch.exp(s - lse[:, :, qs:qe].unsqueeze(-1))
                dv[:, :, ks:ke] += p.transpose(-1, -2) @ dob
                dp = dob @ vb.transpose(-1, -2)
                ds = p * (dp - Dvec[:, :, qs:qe].unsqueeze(-1))
                dq[:, :, qs:qe] += (ds @ kb) * scale
                dk[:, :, ks:ke] += (ds.transpose(-1, -2) @ qb) * scale
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None


def tiled_attention(q, k, v, bq: int = 128, bk: int = 128):
    return _TiledFn.apply(q, k, v, bq, bk)


SDPA_BACKENDS = ["FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "MATH"]


def sdpa_attention(q, k, v, backend: Optional[str] = None):
    if backend is None:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel([getattr(SDPBackend, backend)]):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def detect_sdpa_kernel(q, k, v) -> dict:
    """Identify the kernel actually dispatched by default SDPA from profiler kernel names."""
    try:
        from torch.profiler import ProfilerActivity, profile

        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if q.is_cuda else [])
        with profile(activities=acts) as prof:
            sdpa_attention(q, k, v)
            if q.is_cuda:
                torch.cuda.synchronize()
        names = sorted({e.key for e in prof.key_averages()})
        joined = " ".join(names).lower()
        if "flash" in joined:
            b = "FLASH_ATTENTION"
        elif "efficient" in joined or "fmha" in joined or "cutlass" in joined and "mem_eff" in joined:
            b = "EFFICIENT_ATTENTION"
        elif "cudnn" in joined:
            b = "CUDNN_ATTENTION"
        else:
            b = "MATH(or unknown)"
        rel = [n for n in names if any(t in n.lower() for t in ("flash", "efficient", "fmha", "cudnn", "attention", "softmax", "bmm", "gemm"))]
        return dict(backend=b, kernels=rel[:20])
    except Exception as e:  # pragma: no cover
        return dict(backend="unknown", error=repr(e))

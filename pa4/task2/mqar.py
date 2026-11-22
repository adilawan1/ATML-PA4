"""Task 2B: small decoder-only Transformers for multi-query associative recall (MQAR).

Architecture (identical for every mechanism): learned token embeddings (vocab 256) + fixed sinusoidal
positions, 2 pre-LayerNorm blocks, d=128, 4 heads x 32, GELU MLP of width 512, final LayerNorm, and a
64-way classifier over value IDs 128..191 applied only at the supervised query positions
(query-position classification, not next-token LM).

Attention mechanisms (only the operator changes; q/k/v/out projections are identical):
  dense    causal softmax attention
  window   causal sliding window, query i attends j in [i-W+1, i]   (W=64)
  dilated  heads 0-1: window 64; heads 2-3: offsets {0,4,...,252} (64 positions), causal
  linear   ELU+1 kernel attention, causal prefix sums (materialises the [B,H,T,f,d] prefix tensor)
  hybrid   (B3 intervention) per-head gate between a W_loc=8 local softmax window and an ELU+1 linear
           memory on a learned 16-dim key/query feature projection; state per head = 16x32 + 16 floats
           + an 8-token K/V buffer (8 x 64 floats) = 1040 floats <= 1056 floats of the ELU+1 linear model.
  topk     (test-time only) dense scores, keep the k=64 largest causal scores per query.
Dense/window/dilated/topk masks are explicit dense boolean masks evaluated by SDPA: a correctness
implementation, not an efficient sparse kernel.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB, D_MODEL, N_HEADS, D_HEAD, D_FF, N_LAYERS, T = 256, 128, 4, 32, 512, 2, 256
N_CLASSES, VALUE_OFFSET = 64, 128


def sinusoidal(n: int, d: int) -> torch.Tensor:
    pos = torch.arange(n).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def window_mask(T_: int, W: int, device=None) -> torch.Tensor:
    i = torch.arange(T_, device=device).view(-1, 1)
    j = torch.arange(T_, device=device).view(1, -1)
    return (j <= i) & (j > i - W)


def dilated_mask(T_: int, step: int = 4, n: int = 64, device=None) -> torch.Tensor:
    i = torch.arange(T_, device=device).view(-1, 1)
    j = torch.arange(T_, device=device).view(1, -1)
    off = i - j
    return (off >= 0) & (off % step == 0) & (off <= step * (n - 1))


def causal_mask(T_: int, device=None) -> torch.Tensor:
    return torch.tril(torch.ones(T_, T_, dtype=torch.bool, device=device))


def elu1(x):
    return F.elu(x) + 1.0


def linear_attention_causal(q, k, v, eps: float = 1e-6):
    """Causal ELU+1 linear attention via prefix sums. q,k: [B,H,T,f] (already feature-mapped), v: [B,H,T,d]."""
    kv = torch.einsum("bhtf,bhtd->bhtfd", k, v).cumsum(2)
    z = k.cumsum(2)
    num = torch.einsum("bhtf,bhtfd->bhtd", q, kv)
    den = torch.einsum("bhtf,bhtf->bht", q, z).unsqueeze(-1) + eps
    return num / den


def linear_attention_recurrent(q, k, v, eps: float = 1e-6):
    """Same computation as a recurrence with a [B,H,f,d] state (used to verify the prefix implementation)."""
    B, H, T_, f = q.shape
    S = q.new_zeros(B, H, f, v.shape[-1])
    z = q.new_zeros(B, H, f)
    outs = []
    for t in range(T_):
        S = S + k[:, :, t].unsqueeze(-1) * v[:, :, t].unsqueeze(-2)
        z = z + k[:, :, t]
        num = torch.einsum("bhf,bhfd->bhd", q[:, :, t], S)
        den = (q[:, :, t] * z).sum(-1, keepdim=True) + eps
        outs.append(num / den)
    return torch.stack(outs, 2)


class Attention(nn.Module):
    def __init__(self, kind: str, window: int = 64, loc_window: int = 8, feat_dim: int = 16, topk: int = 64):
        super().__init__()
        self.kind = kind
        self.window, self.loc_window, self.feat_dim, self.topk = window, loc_window, feat_dim, topk
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)
        if kind == "hybrid":
            self.feat = nn.Linear(D_HEAD, feat_dim, bias=False)  # shared across heads; applied to q and k
            self.gate = nn.Linear(D_MODEL, N_HEADS)
        self.test_override: Optional[str] = None  # 'window' or 'topk' imposed at test time only
        self._mask_cache: Dict = {}

    def _mask(self, name, T_, device):
        key = (name, T_, str(device))
        if key not in self._mask_cache:
            if name == "causal":
                m = causal_mask(T_, device)
            elif name == "window":
                m = window_mask(T_, self.window, device)
            elif name == "locwin":
                m = window_mask(T_, self.loc_window, device)
            elif name == "dilated":
                m = torch.stack([window_mask(T_, 64, device)] * 2 + [dilated_mask(T_, 4, 64, device)] * 2, 0)
            self._mask_cache[key] = m
        return self._mask_cache[key]

    def forward(self, x):
        B, T_, _ = x.shape
        q, k, v = self.qkv(x).view(B, T_, 3, N_HEADS, D_HEAD).permute(2, 0, 3, 1, 4)
        kind = self.test_override or self.kind
        if kind == "dense":
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif kind in ("window", "dilated"):
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=self._mask(kind, T_, x.device))
        elif kind == "topk":
            s = (q @ k.transpose(-1, -2)) / math.sqrt(D_HEAD)
            s = s.masked_fill(~self._mask("causal", T_, x.device), float("-inf"))
            kk = min(self.topk, T_)
            thr = s.topk(kk, dim=-1).values[..., -1:]
            s = s.masked_fill(s < thr, float("-inf"))
            o = torch.softmax(s, -1) @ v
        elif kind == "linear":
            o = linear_attention_causal(elu1(q), elu1(k), v)
        elif kind == "hybrid":
            loc = F.scaled_dot_product_attention(q, k, v, attn_mask=self._mask("locwin", T_, x.device))
            lin = linear_attention_causal(elu1(self.feat(q)), elu1(self.feat(k)), v)
            g = torch.sigmoid(self.gate(x)).permute(0, 2, 1).unsqueeze(-1)  # [B,H,T,1]
            o = g * loc + (1 - g) * lin
        else:
            raise ValueError(kind)
        return self.out(o.transpose(1, 2).reshape(B, T_, D_MODEL))


class Block(nn.Module):
    def __init__(self, kind, **kw):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
        self.attn = Attention(kind, **kw)
        self.mlp = nn.Sequential(nn.Linear(D_MODEL, D_FF), nn.GELU(), nn.Linear(D_FF, D_MODEL))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class MQARModel(nn.Module):
    def __init__(self, kind: str = "dense", max_len: int = 8192, **kw):
        super().__init__()
        self.kind = kind
        self.emb = nn.Embedding(VOCAB, D_MODEL)
        self.register_buffer("pos", sinusoidal(max_len, D_MODEL), persistent=False)
        self.blocks = nn.ModuleList([Block(kind, **kw) for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, N_CLASSES)

    def set_test_override(self, name: Optional[str]):
        for b in self.blocks:
            b.attn.test_override = name

    def forward(self, tokens, query_pos):
        x = self.emb(tokens) + self.pos[: tokens.shape[1]]
        for b in self.blocks:
            x = b(x)
        h = self.ln_f(x)
        hq = torch.gather(h, 1, query_pos.unsqueeze(-1).expand(-1, -1, D_MODEL))
        return self.head(hq)  # [B, K, 64]


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------- theoretical costs


def theoretical_costs(kind: str, T_: int = T) -> dict:
    """Attended (query,key) pairs per head per layer and persistent decoding state per layer (FP32 floats)."""
    i = np.arange(T_)
    if kind in ("dense", "topk"):
        pairs = float((i + 1).sum())
        if kind == "topk":
            pairs_sel = float(np.minimum(i + 1, 64).sum())
            return dict(pairs_per_head=pairs, pairs_selected_per_head=pairs_sel,
                        state_floats_per_layer=float(N_HEADS * 2 * D_HEAD * T_), note="scores all pairs, keeps 64")
        return dict(pairs_per_head=pairs, state_floats_per_layer=float(N_HEADS * 2 * D_HEAD * T_),
                    note="KV cache grows with T")
    if kind == "window":
        pairs = float(np.minimum(i + 1, 64).sum())
        return dict(pairs_per_head=pairs, state_floats_per_layer=float(N_HEADS * 2 * D_HEAD * 64),
                    max_receptive_field_2layers=2 * 63)
    if kind == "dilated":
        loc = np.minimum(i + 1, 64).sum()
        dil = np.minimum(i // 4 + 1, 64).sum()
        return dict(pairs_per_head_local=float(loc), pairs_per_head_dilated=float(dil),
                    pairs_per_head=float((2 * loc + 2 * dil) / 4),
                    state_floats_per_layer=float(2 * 2 * D_HEAD * 64 + 2 * 2 * D_HEAD * 253),
                    note="dilated heads need the last 253 positions buffered (64 used)")
    if kind == "linear":
        st = N_HEADS * (D_HEAD * D_HEAD + D_HEAD)
        return dict(pairs_per_head=float(T_), state_floats_per_layer=float(st), state_bytes_per_layer_fp32=4 * st,
                    note="state independent of T")
    if kind == "hybrid":
        f = 16
        lin = N_HEADS * (f * D_HEAD + f)
        buf = N_HEADS * 8 * 2 * D_HEAD
        return dict(pairs_per_head=float(np.minimum(i + 1, 8).sum() + T_), state_floats_per_layer=float(lin + buf),
                    state_bytes_per_layer_fp32=4 * (lin + buf), linear_state_floats=lin, local_buffer_floats=buf,
                    comparator_linear_state_bytes_fp32=4 * N_HEADS * (D_HEAD * D_HEAD + D_HEAD))
    raise ValueError(kind)


# ----------------------------------------------------------------------------- data


def batchify(rows: List[dict], device) -> Dict[str, torch.Tensor]:
    tok = torch.tensor([r["tokens"] for r in rows], dtype=torch.long, device=device)
    qp = torch.tensor([r["query_positions"] for r in rows], dtype=torch.long, device=device)
    ans = torch.tensor([r["answer_ids"] for r in rows], dtype=torch.long, device=device) - VALUE_OFFSET
    src = torch.tensor([r["source_positions"] for r in rows], dtype=torch.long, device=device)
    return dict(tokens=tok, query_pos=qp, answers=ans, sources=src)


DIST_BINS = [(0, 32), (32, 64), (64, 128), (128, 256)]


@torch.no_grad()
def evaluate(model, rows: List[dict], device, batch_size: int = 256) -> dict:
    """Accuracy over supervised query positions, aggregated per (K, regime) cell and distance bin."""
    model.eval()
    recs = []
    # group by K so tensors are rectangular
    byk: Dict[int, List[dict]] = {}
    for r in rows:
        byk.setdefault(r["K"], []).append(r)
    loss_sum, n_q = 0.0, 0
    for K, rs in byk.items():
        for i in range(0, len(rs), batch_size):
            chunk = rs[i:i + batch_size]
            b = batchify(chunk, device)
            logits = model(b["tokens"], b["query_pos"])
            loss_sum += float(F.cross_entropy(logits.reshape(-1, N_CLASSES), b["answers"].reshape(-1), reduction="sum"))
            pred = logits.argmax(-1)
            # random-choice baseline restricted to the K values present in the example
            correct = (pred == b["answers"]).cpu().numpy()
            for j, r in enumerate(chunk):
                for qi in range(K):
                    recs.append((r["K"], r["regime"], r["distances"][qi], bool(correct[j, qi])))
            n_q += pred.numel()
    arr = np.array([(k, d, c) for k, _, d, c in recs], dtype=float)
    regimes = np.array([g for _, g, _, _ in recs])
    out = dict(n_queries=n_q, loss=loss_sum / max(n_q, 1), acc=float(arr[:, 2].mean()), cells={}, bins={}, heat={})
    for K in sorted(set(arr[:, 0].astype(int))):
        for g in ("near", "far", "mixed"):
            sel = (arr[:, 0] == K) & (regimes == g)
            if sel.any():
                out["cells"][f"K{K}_{g}"] = dict(acc=float(arr[sel, 2].mean()), n=int(sel.sum()),
                                                 chance_label=1 / 64, chance_present=1 / K)
        for lo, hi in DIST_BINS:
            sel = (arr[:, 0] == K) & (arr[:, 1] >= lo) & (arr[:, 1] < hi)
            out["heat"][f"K{K}_[{lo},{hi})"] = dict(acc=float(arr[sel, 2].mean()) if sel.any() else None, n=int(sel.sum()))
    for lo, hi in DIST_BINS:
        sel = (arr[:, 1] >= lo) & (arr[:, 1] < hi)
        out["bins"][f"[{lo},{hi})"] = dict(acc=float(arr[sel, 2].mean()) if sel.any() else None, n=int(sel.sum()))
    for g in ("near", "far", "mixed"):
        sel = regimes == g
        out[f"acc_{g}"] = float(arr[sel, 2].mean()) if sel.any() else None
    model.train()
    return out

"""Task 2C: KV-cache compression for Qwen2.5-1.5B-Instruct (FP16, SDPA, greedy decoding).

Protocol per prompt (identical prompt formatting for every policy):
  1. Prefill the pre-question context (tokens before "QUESTION_BEGIN") into a DynamicCache.
  2. Question-aware scoring (simplified SnapKV-style fixed policy, *not* the full head-specific
     SnapKV algorithm): run the question suffix once against the *full* context cache with the ordinary
     SDPA model, capture its q_proj outputs, apply RoPE at the suffix's absolute positions, and compute
     masked FP32 attention against the cached (already rotary-positioned) keys. Importance of a context
     position = attention mass summed over suffix queries and all query heads of the layer, average-pooled
     with kernel 7. Selection is per layer, shared by the 2 KV heads of that layer. The scoring pass is
     discarded (cache cropped back to the context).
  3. Compress the context cache with one policy at the case-specific budget = serialized bytes of the
     uniform INT4 channel-wise-K / token-wise-V cache of the context.
  4. Run question suffix (+ teacher-forced gold code for NLL) on the compressed cache with the original
     absolute position_ids; the physical cache length is the number of retained positions. Then decode
     greedily (<= 12 tokens), extracting the first 5-digit string as the answer.

Quantization: asymmetric min/max groups of 64 elements, FP16 scale + FP16 offset (=group minimum, no
integer zero-point) per group, round-half-to-even, codes in [0, 2^b-1]. Channel-wise K groups 64
consecutive tokens of one channel; token-wise V (and token-wise K) groups 64 consecutive features of one
token. Only complete 64-token groups are quantized; trailing L mod 64 tokens stay FP16 for K and V.
New tokens join the FP16 residual and are quantized when a complete 64-token group forms (streaming,
KIVI-style) for the two INT4 layouts. Quantized entries are stored *dequantized* in FP16 inside the
DynamicCache for ordinary SDPA, so allocated GPU memory != packed bytes (packed sizes are computed
from the byte model and verified against real packing in the C5 benchmark).
"""
from __future__ import annotations

import math
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..common.utils import model_revision

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
GROUP = 64
SINK, RECENT = 8, 64
POOL = 7
MAX_NEW = 12
METHODS = ["fp16", "sink_recent", "qa_evict", "int4_token", "int4_kivi", "hybrid"]


# ----------------------------------------------------------------------------- model & cache helpers


def load(device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rev = model_revision(MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=rev, torch_dtype=torch.float16,
                                                 attn_implementation="sdpa").to(device).eval()
    return model, tok, rev


def kv_dims(cfg) -> Tuple[int, int, int]:
    return cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads


def cache_tensors(cache) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    if hasattr(cache, "layers"):
        return [l.keys for l in cache.layers], [l.values for l in cache.layers]
    return list(cache.key_cache), list(cache.value_cache)


def build_cache(K: List[torch.Tensor], V: List[torch.Tensor]):
    from transformers import DynamicCache

    c = DynamicCache()
    for l, (k, v) in enumerate(zip(K, V)):
        c.update(k.contiguous(), v.contiguous(), l)
    return c


def set_layer(cache, l, k, v):
    if hasattr(cache, "layers"):
        cache.layers[l].keys, cache.layers[l].values = k, v
    else:
        cache.key_cache[l], cache.value_cache[l] = k, v


# ----------------------------------------------------------------------------- prompts


def prepare_case(tok, case: dict) -> dict:
    msgs = [{"role": "user", "content": case["user_text"]}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    ids_t = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    assert list(ids) == list(ids_t), "offset tokenization differs from chat template tokenization"
    u0 = text.index(case["user_text"])
    qchar = u0 + case["user_text"].index("QUESTION_BEGIN")
    L_ctx = next(i for i, (a, b) in enumerate(offs) if b > qchar)  # first token reaching the question
    es, ee = u0 + case["evidence_char_start"], u0 + case["evidence_char_end"]
    ev = [i for i, (a, b) in enumerate(offs) if b > es and a < ee]
    assert ev and max(ev) < L_ctx
    code_ids = tok(case["code"], add_special_tokens=False)["input_ids"]
    return dict(id=case["id"], ids=list(ids), L_ctx=L_ctx, suffix=list(ids[L_ctx:]), evidence=ev,
                code=case["code"], code_ids=code_ids, n_tokens=len(ids), target_tokens=case["target_tokens"],
                position=case["position"], distractors=case["distractors"])


def extract_code(text: str) -> Optional[str]:
    m = re.search(r"\d{5}", text)
    return m.group(0) if m else None


# ----------------------------------------------------------------------------- byte accounting


def m16(L: int, cfg) -> int:
    nl, H, d = kv_dims(cfg)
    return 4 * nl * H * d * L


def mb(L: int, b: int, cfg) -> int:
    """Formula from the handout: 2 n_l H_kv d_h (floor(L/64)(64b/8 + 4) + 2 (L mod 64))."""
    nl, H, d = kv_dims(cfg)
    return 2 * nl * H * d * ((L // GROUP) * (GROUP * b // 8 + 4) + 2 * (L % GROUP))


def evict_bytes(n: int, cfg, index_positions: int) -> int:
    nl, H, d = kv_dims(cfg)
    return 4 * nl * H * d * n + 2 * nl * index_positions  # uint16 indices per layer


def max_evict_positions(L: int, budget: int, cfg, indexed: bool) -> int:
    best = None
    for n in range(SINK + RECENT, L + 1):
        if evict_bytes(n, cfg, (n - SINK - RECENT) if indexed else 0) <= budget:
            best = n
        else:
            break
    assert best is not None, "budget below the sink+recent safeguard"
    return best


def hybrid_bytes(L: int, m: int, cfg) -> Tuple[int, dict]:
    nl, H, d = kv_dims(cfg)
    low = L - SINK - RECENT - m
    g, r = low // GROUP, low % GROUP
    per_ch = g * (GROUP * 2 // 8 + 4) + 2 * (SINK + RECENT + m + r)
    tot = 2 * nl * H * d * per_ch + 2 * nl * m
    return tot, dict(int2_groups_per_channel=g, fp16_residual=r, fp16_selected=m, fp16_total_positions=SINK + RECENT + m + r)


def max_hybrid_m(L: int, budget: int, cfg) -> int:
    best = None
    for m in range(0, L - SINK - RECENT + 1):
        if hybrid_bytes(L, m, cfg)[0] <= budget:
            best = m
    assert best is not None, f"hybrid safeguards ({SINK + RECENT} FP16 tokens) do not fit the INT4 budget at L={L}"
    return best


# ----------------------------------------------------------------------------- quantization (simulated)


def qdq(x: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    """Quantize-dequantize groups formed by `dim` (size must be the group size)."""
    xf = x.float()
    mn = xf.amin(dim, keepdim=True)
    mx = xf.amax(dim, keepdim=True)
    qmax = 2 ** bits - 1
    scale = ((mx - mn).clamp(min=1e-6) / qmax).half().float()
    off = mn.half().float()
    q = torch.clamp(torch.round((xf - off) / scale), 0, qmax)
    return (q * scale + off).to(x.dtype)


def quant_tokens(K: torch.Tensor, V: torch.Tensor, bits: int, layout: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a run of n (multiple of 64) tokens. K,V: [B,H,n,d]."""
    B, H, n, d = K.shape
    assert n % GROUP == 0 and d % GROUP == 0
    if layout == "kivi":  # channel-wise K: groups of 64 tokens per channel
        Kq = qdq(K.view(B, H, n // GROUP, GROUP, d), bits, 3).view(B, H, n, d)
    else:  # token-wise K
        Kq = qdq(K.view(B, H, n, d // GROUP, GROUP), bits, 4).view(B, H, n, d)
    Vq = qdq(V.view(B, H, n, d // GROUP, GROUP), bits, 4).view(B, H, n, d)
    return Kq, Vq


class StreamQuant:
    """Keeps the 'only complete 64-token groups are quantized' invariant while tokens are appended."""

    def __init__(self, bits: int, layout: str, n_layers: int, start: int = 0):
        self.bits, self.layout = bits, layout
        self.ptr = [start] * n_layers

    @torch.no_grad()
    def apply(self, cache):
        K, V = cache_tensors(cache)
        for l in range(len(K)):
            n = K[l].shape[2]
            while n - self.ptr[l] >= GROUP:
                a, b = self.ptr[l], self.ptr[l] + GROUP
                kq, vq = quant_tokens(K[l][:, :, a:b], V[l][:, :, a:b], self.bits, self.layout)
                K[l][:, :, a:b] = kq
                V[l][:, :, a:b] = vq
                self.ptr[l] = b

    def live_bytes(self, cache, cfg) -> int:
        nl, H, d = kv_dims(cfg)
        K, _ = cache_tensors(cache)
        tot = 0
        for l in range(len(K)):
            q, f = self.ptr[l], K[l].shape[2] - self.ptr[l]
            per_ch = (q // GROUP) * (GROUP * self.bits // 8 + 4) + 2 * f
            tot += 2 * H * d * per_ch
        return tot


# ----------------------------------------------------------------------------- scoring


@torch.no_grad()
def question_scores(model, ctx_K, ctx_V, L_ctx: int, suffix: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-layer importance [n_layers, L_ctx] of context positions; also returns the full-cache logits."""
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    dev = ctx_K[0].device
    cache = build_cache([k.clone() for k in ctx_K], [v.clone() for v in ctx_V])
    layers = model.model.layers
    caps = {}
    hooks = [layer.self_attn.q_proj.register_forward_hook(lambda m, i, o, l=l: caps.__setitem__(l, o))
             for l, layer in enumerate(layers)]
    s = len(suffix)
    pos = torch.arange(L_ctx, L_ctx + s, device=dev).unsqueeze(0)
    try:
        out = model(input_ids=torch.tensor([suffix], device=dev), past_key_values=cache, position_ids=pos, use_cache=True)
    finally:
        for h in hooks:
            h.remove()
    cfg = model.config
    nH, dh = cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads
    cos, sin = model.model.rotary_emb(caps[0], pos)
    Kall, _ = cache_tensors(cache)
    scores = []
    qi = torch.arange(s, device=dev).view(-1, 1)
    kj = torch.arange(L_ctx + s, device=dev).view(1, -1)
    allow = kj <= (L_ctx + qi)
    for l in range(len(layers)):
        q = caps[l].view(1, s, nH, dh).transpose(1, 2)
        q, _ = apply_rotary_pos_emb(q, q, cos, sin)
        k = repeat_kv(Kall[l], nH // Kall[l].shape[1]).float()
        att = (q.float() @ k.transpose(-1, -2)) / math.sqrt(dh)
        att = att.masked_fill(~allow, float("-inf")).softmax(-1)
        sc = att[..., :L_ctx].sum(dim=(0, 1, 2))
        sc = F.avg_pool1d(sc.view(1, 1, -1), POOL, stride=1, padding=POOL // 2, count_include_pad=False).view(-1)
        scores.append(sc)
    del cache
    return torch.stack(scores), out.logits[0, -1].float()


# ----------------------------------------------------------------------------- policies


def _select_qa(scores_l: torch.Tensor, L: int, n: int) -> torch.Tensor:
    keep = torch.zeros(L, dtype=torch.bool, device=scores_l.device)
    keep[:SINK] = True
    keep[L - RECENT:] = True
    extra = n - int(keep.sum())
    if extra > 0:
        sc = scores_l.clone()
        sc[keep] = float("-inf")
        keep[sc.topk(extra).indices] = True
    return keep


@torch.no_grad()
def compress(method: str, ctx_K, ctx_V, scores: Optional[torch.Tensor], cfg, evidence: List[int]) -> dict:
    """Return dict(K, V lists for the new cache, stream quantizer or None, bytes/meta)."""
    L = ctx_K[0].shape[2]
    nl, H, d = kv_dims(cfg)
    budget = mb(L, 4, cfg)
    ev = torch.tensor(evidence, device=ctx_K[0].device)
    meta = dict(L_ctx=L, budget_bytes=budget, fp16_bytes=m16(L, cfg))
    K = [k.clone() for k in ctx_K]
    V = [v.clone() for v in ctx_V]
    stream = None
    if method == "fp16":
        meta.update(bytes=m16(L, cfg), positions=L, evidence_fp16_retained=1.0)
    elif method in ("sink_recent", "qa_evict"):
        indexed = method == "qa_evict"
        n = max_evict_positions(L, budget, cfg, indexed)
        ret = []
        for l in range(nl):
            if method == "sink_recent":
                keep = torch.zeros(L, dtype=torch.bool, device=K[l].device)
                keep[:SINK] = True
                keep[L - (n - SINK):] = True
            else:
                keep = _select_qa(scores[l], L, n)
            idx = torch.nonzero(keep).flatten()
            K[l], V[l] = K[l][:, :, idx], V[l][:, :, idx]
            ret.append(float(keep[ev].float().mean()))
            if l in (0, nl // 2, nl - 1):
                meta[f"keep_mask_layer{l}"] = keep.cpu().numpy().astype(np.uint8).tolist()
        meta.update(bytes=evict_bytes(n, cfg, (n - SINK - RECENT) if indexed else 0), positions=n,
                    evidence_fp16_retained=float(np.mean(ret)), evidence_fp16_retained_per_layer=ret,
                    index_bytes=2 * nl * (n - SINK - RECENT) if indexed else 0)
    elif method in ("int4_token", "int4_kivi"):
        layout = "kivi" if method == "int4_kivi" else "token"
        nc = (L // GROUP) * GROUP
        err_k = err_v = 0.0
        nk = nv = 0.0
        for l in range(nl):
            kq, vq = quant_tokens(K[l][:, :, :nc], V[l][:, :, :nc], 4, layout)
            err_k += float((kq.float() - K[l][:, :, :nc].float()).pow(2).sum())
            err_v += float((vq.float() - V[l][:, :, :nc].float()).pow(2).sum())
            nk += float(K[l][:, :, :nc].float().pow(2).sum())
            nv += float(V[l][:, :, :nc].float().pow(2).sum())
            K[l][:, :, :nc], V[l][:, :, :nc] = kq, vq
        stream = StreamQuant(4, layout, nl, start=nc)
        meta.update(bytes=mb(L, 4, cfg), positions=L, fp16_residual=L - nc, evidence_fp16_retained=None,
                    k_rel_mse=err_k / nk, v_rel_mse=err_v / nv)
    elif method == "hybrid":
        m = max_hybrid_m(L, budget, cfg)
        tot, hb = hybrid_bytes(L, m, cfg)
        ret = []
        err_k = err_v = nk = nv = 0.0
        for l in range(nl):
            keep = _select_qa(scores[l], L, SINK + RECENT + m)
            low = torch.nonzero(~keep).flatten()
            r = low.numel() % GROUP
            q_idx = low[: low.numel() - r]  # complete groups; the last r low positions stay FP16
            if q_idx.numel():
                kq, vq = quant_tokens(K[l][:, :, q_idx], V[l][:, :, q_idx], 2, "kivi")
                err_k += float((kq.float() - K[l][:, :, q_idx].float()).pow(2).sum())
                err_v += float((vq.float() - V[l][:, :, q_idx].float()).pow(2).sum())
                nk += float(K[l][:, :, q_idx].float().pow(2).sum())
                nv += float(V[l][:, :, q_idx].float().pow(2).sum())
                K[l][:, :, q_idx], V[l][:, :, q_idx] = kq, vq
            ret.append(float(keep[ev].float().mean()))
            if l in (0, nl // 2, nl - 1):
                meta[f"keep_mask_layer{l}"] = keep.cpu().numpy().astype(np.uint8).tolist()
        meta.update(bytes=tot, positions=L, evidence_fp16_retained=float(np.mean(ret)),
                    evidence_fp16_retained_per_layer=ret, k_rel_mse=err_k / max(nk, 1e-30),
                    v_rel_mse=err_v / max(nv, 1e-30), index_bytes=2 * nl * m, **hb)
    else:
        raise ValueError(method)
    meta["realized_fraction_of_fp16"] = meta["bytes"] / meta["fp16_bytes"]
    meta["within_budget"] = meta["bytes"] <= budget
    return dict(K=K, V=V, stream=stream, meta=meta)


# ----------------------------------------------------------------------------- running one policy


@torch.no_grad()
def run_policy(model, tok, case: dict, comp: dict, ref_logits: Optional[torch.Tensor]) -> dict:
    dev = comp["K"][0].device
    cfg = model.config
    L, s = case["L_ctx"], len(case["suffix"])
    cache = build_cache(comp["K"], comp["V"])
    n_phys = comp["K"][0].shape[2]
    stream = comp["stream"]
    # teacher-forced: suffix + gold code in one pass (logits at s-1 are the post-prefill distribution)
    tf = case["suffix"] + case["code_ids"]
    pos = torch.arange(L, L + len(tf), device=dev).unsqueeze(0)
    torch.cuda.synchronize() if dev.type == "cuda" else None
    t0 = time.perf_counter()
    out = model(input_ids=torch.tensor([tf], device=dev), past_key_values=cache, position_ids=pos, use_cache=True)
    lp = torch.log_softmax(out.logits[0].float(), -1)
    first = lp[s - 1]
    code_nll = -float(sum(lp[s - 1 + i, t] for i, t in enumerate(case["code_ids"])))
    cache.crop(n_phys + s)
    if stream is not None:
        stream.apply(cache)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t_suffix = time.perf_counter() - t0
    kl = None
    if ref_logits is not None:
        pr = torch.log_softmax(ref_logits, -1)
        kl = float((pr.exp() * (pr - first)).sum())
    # greedy decode
    gen = [int(first.argmax())]
    t0 = time.perf_counter()
    abs_pos = L + s
    while len(gen) < MAX_NEW and gen[-1] != tok.eos_token_id and gen[-1] != tok.convert_tokens_to_ids("<|im_end|>"):
        o = model(input_ids=torch.tensor([[gen[-1]]], device=dev), past_key_values=cache,
                  position_ids=torch.tensor([[abs_pos]], device=dev), use_cache=True)
        abs_pos += 1
        if stream is not None:
            stream.apply(cache)
        gen.append(int(o.logits[0, -1].argmax()))
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t_dec = time.perf_counter() - t0
    text = tok.decode(gen, skip_special_tokens=True)
    pred = extract_code(text)
    nl, H, d = kv_dims(cfg)
    n_new = s + len(gen) - 1  # tokens appended after the context (suffix + generated, last not cached)
    if stream is not None:
        live = stream.live_bytes(cache, cfg)
        shared = None
    else:
        shared = 4 * nl * H * d * n_new
        live = comp["meta"]["bytes"] + shared
    return dict(gen_ids=gen, text=text, pred=pred, correct=pred == case["code"], first_token_logit_kl=kl,
                code_nll=code_nll, first_logits=first, t_suffix=t_suffix, t_decode=t_dec, n_generated=len(gen),
                shared_decode_bytes_fp16=shared, live_cache_bytes_end=live, finite=bool(torch.isfinite(first).all()))

"""Prefix banks and draft/target distribution-alignment metrics on common prefixes.

A bank record (prompt_id, bank, position, response_prefix_token_ids) asks for the next-token distributions
at response position `position` given prompt + response[:position-1]. Records of one (prompt, bank) come
from the same sampled response, so one forward over prompt + longest prefix yields every position.

Metrics per position (p = target/teacher, q = draft, full common vocabulary, temperature 1):
  forward KL(p||q), reverse KL(q||p), TV = 1/2 sum|p - q|, expected one-token acceptance 1 - TV,
  greedy agreement [argmax p == argmax q].
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List

import torch

POS_SCHEME = [(1, 16, 8), (17, 32, 8), (33, 64, 16), (65, 192, 16)]  # identical to the course bank script
POS_BINS = [(1, 16), (17, 32), (33, 64), (65, 192)]


def sample_positions(pid: str, which: str, length: int) -> List[int]:
    rng = random.Random(f"PA4-banks-v1:6304:{pid}:{which}")
    pos = []
    for lo, hi, n in POS_SCHEME:
        cand = list(range(lo, min(hi, length) + 1))
        pos.extend(sorted(rng.sample(cand, min(n, len(cand)))))
    return pos


def bank_records(pid: str, domain: str, prompt: str, which: str, seq: List[int], meta: dict) -> List[dict]:
    return [dict(prompt_id=pid, domain=domain, bank=which, position=p, prompt=prompt,
                 response_prefix_token_ids=seq[: p - 1], source_length=len(seq), **meta)
            for p in sample_positions(pid, which, len(seq))]


def group_bank(records: List[dict]) -> Dict[tuple, dict]:
    """(bank, prompt_id) -> dict(prompt, positions, longest prefix)."""
    g = defaultdict(lambda: dict(positions=[], longest=[]))
    for r in records:
        e = g[(r["bank"], r["prompt_id"])]
        e["prompt"], e["domain"] = r["prompt"], r.get("domain")
        e["positions"].append(r["position"])
        if len(r["response_prefix_token_ids"]) > len(e["longest"]):
            e["longest"] = r["response_prefix_token_ids"]
    for e in g.values():
        e["positions"] = sorted(set(e["positions"]))
    return dict(g)


@torch.no_grad()
def position_logprobs(model, prompt_ids: List[int], prefix: List[int], positions: List[int]) -> torch.Tensor:
    dev = next(model.parameters()).device
    ids = torch.tensor([prompt_ids + prefix], device=dev)
    logits = model(input_ids=ids, use_cache=False).logits[0]
    P = len(prompt_ids)
    idx = torch.tensor([P + p - 2 for p in positions], device=dev)
    return torch.log_softmax(logits[idx].float(), -1)


def pair_metrics(lp: torch.Tensor, lq: torch.Tensor) -> Dict[str, List[float]]:
    p, q = lp.exp(), lq.exp()
    fkl = (p * (lp - lq)).sum(-1)
    rkl = (q * (lq - lp)).sum(-1)
    tv = 0.5 * (p - q).abs().sum(-1)
    agree = (lp.argmax(-1) == lq.argmax(-1)).float()
    return dict(fkl=fkl.tolist(), rkl=rkl.tolist(), tv=tv.tolist(), exp_accept=(1 - tv).tolist(), greedy_agree=agree.tolist())


def pos_bin(p: int) -> str:
    for lo, hi in POS_BINS:
        if lo <= p <= hi:
            return f"[{lo},{hi}]"
    return "other"

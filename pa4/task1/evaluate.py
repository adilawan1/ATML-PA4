"""Perplexity on locked token windows and zero-shot HellaSwag (lm-evaluation-harness conventions).

Perplexity: every locked window is an independent sequence (no state carried across windows); the
targets are token_ids[a:b] for score_target_offsets [a, b] = [1, n+1], so the first token of every
window is never scored. PPL = exp(sum NLL / N). Per-window NLL sums are saved for paired bootstraps.

HellaSwag (lm-eval-harness `hellaswag` task, v1.0 preprocessing):
  query  = preprocess(activity_label + ": " + ctx_a + " " + ctx_b.capitalize())
  choice = preprocess(ending); continuation = " " + choice
  preprocess: strip, " [title]" -> ". ", drop "[...]" tags, "  " -> " "
  ll(choice) = sum log p(continuation tokens | query), with continuation tokens = enc(query+cont)[len(enc(query)):]
  acc = argmax ll;  acc_norm = argmax ll / len(continuation characters)   (not token count)
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as Fn

HELLASWAG_PROMPT = "lm-eval-harness hellaswag v1.0: query=preprocess(activity_label+': '+ctx_a+' '+ctx_b.capitalize()); continuation=' '+preprocess(ending)"


@torch.no_grad()
def window_nll(model, windows: List[dict], batch_size: int = 4, pad_id: int = 0) -> Dict[str, list]:
    dev = next(model.parameters()).device
    nll, ntok = [], []
    for i in range(0, len(windows), batch_size):
        batch = windows[i:i + batch_size]
        L = max(len(w["token_ids"]) for w in batch)
        ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for j, w in enumerate(batch):
            ids[j, : len(w["token_ids"])] = torch.tensor(w["token_ids"])
            att[j, : len(w["token_ids"])] = 1
        logits = model(input_ids=ids.to(dev), attention_mask=att.to(dev), use_cache=False).logits
        for j, w in enumerate(batch):
            a, b = w["score_target_offsets"]
            tgt = ids[j, a:b].to(dev)
            lg = logits[j, a - 1:b - 1].float()
            nll.append(float(Fn.cross_entropy(lg, tgt, reduction="sum").item()))
            ntok.append(int(b - a))
            assert ntok[-1] == w["scored_tokens"]
        del logits
    return dict(nll=nll, ntok=ntok)


def ppl_summary(res: Dict[str, list]) -> dict:
    N = int(sum(res["ntok"]))
    tot = float(sum(res["nll"]))
    return dict(N=N, mean_nll=tot / N, ppl=float(np.exp(tot / N)), n_windows=len(res["ntok"]))


# ----------------------------------------------------------------------------- HellaSwag


def hs_preprocess(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def hs_doc(example: dict) -> dict:
    ctx = example["ctx_a"] + " " + example["ctx_b"].capitalize()
    return dict(query=hs_preprocess(example["activity_label"] + ": " + ctx),
                choices=[hs_preprocess(e) for e in example["endings"]], gold=int(example["label"]))


@torch.no_grad()
def hellaswag_scores(model, tok, rows: List[dict], batch_size: int = 16) -> List[dict]:
    """Return per-example loglikelihoods of the 4 endings and their character lengths."""
    dev = next(model.parameters()).device
    reqs = []  # (example_idx, choice_idx, ctx_ids, cont_ids, cont_chars)
    docs = []
    for ei, r in enumerate(rows):
        d = hs_doc(r["example"])
        docs.append(d)
        ctx_ids = tok(d["query"], add_special_tokens=False)["input_ids"]
        for ci, c in enumerate(d["choices"]):
            cont = " " + c
            whole = tok(d["query"] + cont, add_special_tokens=False)["input_ids"]
            cont_ids = whole[len(ctx_ids):]
            reqs.append((ei, ci, whole[: len(whole) - len(cont_ids)], cont_ids, len(cont)))
    ll = np.zeros((len(rows), 4))
    greedy = np.zeros((len(rows), 4), dtype=bool)
    # sort by length for efficient batching (results are written back by index)
    order = sorted(range(len(reqs)), key=lambda i: -(len(reqs[i][2]) + len(reqs[i][3])))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        seqs = [reqs[i][2] + reqs[i][3] for i in idx]
        L = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, x in enumerate(seqs):
            ids[j, : len(x)] = torch.tensor(x)
            att[j, : len(x)] = 1
        logp = torch.log_softmax(model(input_ids=ids.to(dev), attention_mask=att.to(dev), use_cache=False).logits.float(), -1)
        for j, i in enumerate(idx):
            ei, ci, ctx_ids, cont_ids, _ = reqs[i]
            a = len(ctx_ids)
            tgt = torch.tensor(cont_ids, device=dev)
            lp = logp[j, a - 1:a - 1 + len(cont_ids)]
            ll[ei, ci] = float(lp.gather(1, tgt.view(-1, 1)).sum().item())
            greedy[ei, ci] = bool((lp.argmax(-1) == tgt).all().item())
        del logp
    out = []
    for ei, r in enumerate(rows):
        lens = [len(" " + c) for c in docs[ei]["choices"]]
        out.append(dict(order=r["order"], source_row_index=r["source_row_index"], gold=docs[ei]["gold"],
                        ll=ll[ei].tolist(), char_len=lens,
                        pred=int(np.argmax(ll[ei])), pred_norm=int(np.argmax(ll[ei] / np.asarray(lens)))))
    return out


def hellaswag_summary(scores: List[dict]) -> dict:
    acc = np.mean([s["pred"] == s["gold"] for s in scores])
    accn = np.mean([s["pred_norm"] == s["gold"] for s in scores])
    return dict(n=len(scores), acc=float(acc), acc_norm=float(accn))

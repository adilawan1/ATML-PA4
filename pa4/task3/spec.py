"""Exact speculative sampling (Leviathan et al. 2023) with explicit KV-cache rollback.

Invariant maintained between rounds: each runner has cached every token of `seq` except the last one
(the target always; the draft may lag by one token after a fully-accepted round, in which case it feeds
two pending tokens at the start of the next round).

One round with proposal length g = min(gamma, remaining - 1)  (so the bonus/correction token never
exceeds the max_new_tokens cap; g = 0 is a plain target step):
  1. draft samples x_1..x_g from q_i = q(. | seq, x_<i)            (g draft forward passes)
  2. target scores [pending, x_1..x_g] in ONE forward -> p_1..p_{g+1}
  3. for i = 1..g: accept x_i with prob min(1, p_i(x_i)/q_i(x_i)); stop at the first rejection and
     sample the correction from norm([p_i - q_i]_+); an accepted EOS ends generation immediately
  4. if all g accepted: bonus token ~ p_{g+1}
  5. rollback: target cache -> len(seq_new) - 1; draft cache -> min(current, len(seq_new) - 1)
The emitted sequence has exactly the target's sampling distribution (verified by the chi^2 test).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch


def _sync(dev):
    if isinstance(dev, torch.device) and dev.type == "cuda":
        torch.cuda.synchronize()


class HFRunner:
    """Incremental decoding of a HF causal LM with a DynamicCache that supports crop()."""

    def __init__(self, model, temperature: float = 1.0):
        self.model = model
        self.dev = next(model.parameters()).device
        self.temperature = temperature
        self.cache = None
        self.n_cached = 0
        self.forward_calls = 0

    def reset(self):
        from transformers import DynamicCache

        self.cache = DynamicCache()
        self.n_cached = 0
        self.forward_calls = 0

    @torch.no_grad()
    def feed(self, tokens: Sequence[int]) -> torch.Tensor:
        """Feed tokens; returns next-token probability rows [len(tokens), V] (float32)."""
        ids = torch.tensor([list(tokens)], device=self.dev)
        out = self.model(input_ids=ids, past_key_values=self.cache, use_cache=True)
        self.n_cached += len(tokens)
        self.forward_calls += 1
        return torch.softmax(out.logits[0].float() / self.temperature, -1)

    def crop(self, n: int):
        if n < self.n_cached:
            self.cache.crop(n)
            self.n_cached = n


class ToyMarkovRunner:
    """Order-1 Markov 'model' over a small vocabulary with the same runner interface (for exact tests).

    trans[t] is the next-token distribution after token t (the start token's row is the initial law)."""

    def __init__(self, trans: torch.Tensor):
        self.trans = trans.double()
        self.dev = torch.device("cpu")
        self.n_cached = 0
        self.forward_calls = 0
        self.fed: List[int] = []

    def reset(self):
        self.n_cached = 0
        self.forward_calls = 0
        self.fed = []

    def feed(self, tokens):
        self.fed = self.fed[: self.n_cached] + list(tokens)
        self.n_cached += len(tokens)
        self.forward_calls += 1
        return self.trans[torch.tensor(list(tokens))]

    def crop(self, n):
        if n < self.n_cached:
            self.n_cached = n
            self.fed = self.fed[:n]


def _sample(p: torch.Tensor, gen: torch.Generator) -> int:
    return int(torch.multinomial(p.float() if p.dtype == torch.float16 else p, 1, generator=gen).item())


@dataclass
class SpecStats:
    rounds: int = 0
    proposed: int = 0
    examined: int = 0
    accepted: int = 0
    bonus: int = 0
    plain_target_steps: int = 0  # rounds with g = 0 (only one token of room left before the cap)
    corrections: int = 0
    target_calls: int = 0
    draft_calls: int = 0
    accepted_per_round: List[int] = field(default_factory=list)
    proposed_per_round: List[int] = field(default_factory=list)
    t_prefill: float = 0.0
    t_draft: float = 0.0
    t_verify: float = 0.0
    t_accept: float = 0.0
    t_rollback: float = 0.0
    t_total: float = 0.0
    eos: bool = False
    truncated: bool = False
    n_out: int = 0

    def to_dict(self):
        d = dict(self.__dict__)
        d["t_host_other"] = self.t_total - (self.t_prefill + self.t_draft + self.t_verify + self.t_accept + self.t_rollback)
        d["accept_rate_examined"] = self.accepted / self.examined if self.examined else None
        d["accept_rate_proposed"] = self.accepted / self.proposed if self.proposed else None
        d["mean_accepted_prefix_per_verification"] = self.accepted / self.rounds if self.rounds else None
        d["target_passes_per_token"] = self.target_calls / self.n_out if self.n_out else None
        return d


@torch.no_grad()
def speculative_generate(target, draft, prompt: Sequence[int], gamma: int, max_new: int, gen: torch.Generator,
                         eos_ids: Sequence[int] = (), timing: bool = False, residual: str = "exact"):
    """Returns (generated token list, SpecStats). residual='p' is the deliberately wrong negative control."""
    st = SpecStats()
    dev = target.dev
    T = (lambda: (_sync(dev), time.perf_counter())[1]) if timing else time.perf_counter
    t_start = T()
    target.reset(); draft.reset()
    seq = list(prompt)
    t0 = T()
    if len(seq) > 1:
        target.feed(seq[:-1])
        if gamma > 0:
            draft.feed(seq[:-1])
    st.t_prefill = T() - t0
    base_target_calls, base_draft_calls = target.forward_calls, draft.forward_calls
    eos = set(eos_ids)
    out: List[int] = []
    while len(out) < max_new:
        g = max(0, min(gamma, max_new - len(out) - 1))
        base = len(seq)
        # 1) draft proposals
        t0 = T()
        props, qs = [], []
        if g > 0:
            probs = draft.feed(seq[draft.n_cached:])[-1]
            for i in range(g):
                x = _sample(probs, gen)
                props.append(x); qs.append(probs)
                if x in eos:
                    break  # a proposal after EOS would be meaningless
                if i < g - 1:
                    probs = draft.feed([x])[-1]
        g_eff = len(props)
        st.t_draft += T() - t0
        # 2) one target verification pass
        t0 = T()
        P = target.feed(seq[target.n_cached:] + props)[-(g_eff + 1):]
        st.t_verify += T() - t0
        # 3)-4) accept / residual / bonus
        t0 = T()
        n_acc, new, stop = 0, [], False
        for i in range(g_eff):
            x = props[i]
            st.examined += 1
            a = min(1.0, float(P[i, x]) / max(float(qs[i][x]), 1e-30))
            u = float(torch.rand(1, generator=gen, device=gen.device).item())
            if u < a:
                n_acc += 1
                new.append(x)
                if x in eos:
                    stop = True
                    break
            else:
                if residual == "exact":
                    r = (P[i] - qs[i]).clamp(min=0)
                    s = r.sum()
                    r = r / s if float(s) > 0 else P[i]
                else:  # negative control: ignores the draft's contribution
                    r = P[i]
                y = _sample(r, gen)
                new.append(y)
                st.corrections += 1
                stop = y in eos
                break
        else:
            if not stop:
                y = _sample(P[g_eff], gen)
                new.append(y)
                if g_eff > 0:
                    st.bonus += 1
                else:
                    st.plain_target_steps += 1
                stop = y in eos
        st.t_accept += T() - t0
        st.rounds += 1
        st.proposed += g_eff
        st.accepted += n_acc
        st.accepted_per_round.append(n_acc)
        st.proposed_per_round.append(g_eff)
        # 5) rollback / advance caches
        t0 = T()
        seq.extend(new)
        out.extend(new)
        target.crop(len(seq) - 1)
        draft.crop(min(draft.n_cached, len(seq) - 1))
        st.t_rollback += T() - t0
        assert target.n_cached == len(seq) - 1
        if stop:
            st.eos = True
            break
    if len(out) > max_new:  # cannot happen given g <= remaining - 1, kept as a guard
        out = out[:max_new]
    st.truncated = (not st.eos) and len(out) >= max_new
    st.n_out = len(out)
    st.target_calls = target.forward_calls - base_target_calls
    st.draft_calls = draft.forward_calls - base_draft_calls
    st.t_total = T() - t_start
    return out, st


@torch.no_grad()
def target_only_generate(target, prompt: Sequence[int], max_new: int, gen: torch.Generator, eos_ids=(),
                         timing: bool = False):
    dev = target.dev
    T = (lambda: (_sync(dev), time.perf_counter())[1]) if timing else time.perf_counter
    t_start = T()
    target.reset()
    seq = list(prompt)
    t0 = T()
    if len(seq) > 1:
        target.feed(seq[:-1])
    t_prefill = T() - t0
    calls0 = target.forward_calls
    out = []
    eos = set(eos_ids)
    t_dec = 0.0
    while len(out) < max_new:
        t0 = T()
        p = target.feed(seq[target.n_cached:])[-1]
        y = _sample(p, gen)
        t_dec += T() - t0
        seq.append(y); out.append(y)
        if y in eos:
            break
    total = T() - t_start
    return out, dict(n_out=len(out), eos=bool(out and out[-1] in eos), truncated=not (out and out[-1] in eos),
                     target_calls=target.forward_calls - calls0, t_prefill=t_prefill, t_decode=t_dec, t_total=total)


# ----------------------------------------------------------------------------- exact toy test


def toy_markov_test(n: int = 100_000, length: int = 3, gamma: int = 2, seed: int = 6304) -> dict:
    """Pearson chi^2 of empirical length-3 sequences vs the exact target joint law (3 symbols + start)."""
    import itertools

    from scipy.stats import chisquare

    START = 3
    tp = torch.tensor([[0.70, 0.20, 0.10, 0.0],
                       [0.15, 0.25, 0.60, 0.0],
                       [0.30, 0.45, 0.25, 0.0],
                       [0.50, 0.30, 0.20, 0.0]], dtype=torch.float64)
    tq = torch.tensor([[0.20, 0.30, 0.50, 0.0],
                       [0.60, 0.20, 0.20, 0.0],
                       [0.10, 0.10, 0.80, 0.0],
                       [0.15, 0.70, 0.15, 0.0]], dtype=torch.float64)
    exact = {}
    for seqt in itertools.product(range(3), repeat=length):
        pr, prev = 1.0, START
        for x in seqt:
            pr *= float(tp[prev, x]); prev = x
        exact[seqt] = pr
    res = {}
    for mode in ("exact", "p"):
        gen = torch.Generator().manual_seed(seed)
        T_, Q_ = ToyMarkovRunner(tp), ToyMarkovRunner(tq)
        counts = {k: 0 for k in exact}
        acc = exam = bonus = corr = plain = full_rounds = 0
        for _ in range(n):
            out, st = speculative_generate(T_, Q_, [START], gamma, length, gen, residual=mode)
            counts[tuple(out)] += 1
            acc += st.accepted; exam += st.examined; bonus += st.bonus; corr += st.corrections
            plain += st.plain_target_steps
            full_rounds += sum(1 for a, p_ in zip(st.accepted_per_round, st.proposed_per_round) if p_ > 0 and a == p_)
        keys = sorted(exact)
        obs = [counts[k] for k in keys]
        expv = [exact[k] * n for k in keys]
        chi = chisquare(obs, expv)
        res[mode] = dict(chi2=float(chi.statistic), p_value=float(chi.pvalue), dof=len(keys) - 1,
                         min_expected=float(min(expv)), accepted=acc, examined=exam, bonus=bonus, corrections=corr,
                         plain_target_steps=plain, fully_accepted_rounds=full_rounds,
                         table=[dict(seq="".join(map(str, k)), exact_prob=exact[k], empirical=counts[k] / n) for k in keys])
    one_step_tv = 0.5 * (tp[START, :3] - tq[START, :3]).abs().sum().item()
    res["setup"] = dict(n=n, length=length, gamma=gamma, seed=seed, target=tp.tolist(), draft=tq.tolist(),
                        expected_first_acceptance=1 - one_step_tv,
                        pass_exact=res["exact"]["p_value"] >= 0.01, rejects_negative_control=res["p"]["p_value"] < 0.01)
    return res


def toy_eos_test(n: int = 100_000, max_len: int = 3, gamma: int = 3, eos: int = 2, seed: int = 6305) -> dict:
    """Exact-law check with EOS: symbol `eos` terminates a sequence; others run to max_len."""
    from scipy.stats import chisquare

    START = 3
    tp = torch.tensor([[0.55, 0.30, 0.15, 0.0], [0.25, 0.45, 0.30, 0.0], [0.3, 0.3, 0.4, 0.0], [0.5, 0.3, 0.2, 0.0]],
                      dtype=torch.float64)
    tq = torch.tensor([[0.2, 0.3, 0.5, 0.0], [0.6, 0.2, 0.2, 0.0], [0.3, 0.3, 0.4, 0.0], [0.15, 0.7, 0.15, 0.0]],
                      dtype=torch.float64)
    exact = {}

    def rec(prefix, prob, prev):
        if (prefix and prefix[-1] == eos) or len(prefix) == max_len:
            exact[tuple(prefix)] = prob
            return
        for x in range(3):
            rec(prefix + [x], prob * float(tp[prev, x]), x)

    rec([], 1.0, START)
    gen = torch.Generator().manual_seed(seed)
    T_, Q_ = ToyMarkovRunner(tp), ToyMarkovRunner(tq)
    counts = {k: 0 for k in exact}
    eos_flags = 0
    for _ in range(n):
        out, st = speculative_generate(T_, Q_, [START], gamma, max_len, gen, eos_ids=[eos])
        counts[tuple(out)] += 1
        eos_flags += int(st.eos)
    keys = sorted(exact)
    chi = chisquare([counts[k] for k in keys], [exact[k] * n for k in keys])
    p_eos = sum(v for k, v in exact.items() if k and k[-1] == eos)
    return dict(n=n, gamma=gamma, max_len=max_len, chi2=float(chi.statistic), p_value=float(chi.pvalue),
                dof=len(keys) - 1, min_expected=float(min(exact[k] * n for k in keys)),
                eos_rate_empirical=eos_flags / n, eos_rate_exact=p_eos, pass_exact=float(chi.pvalue) >= 0.01,
                table=[dict(seq="".join(map(str, k)), exact_prob=exact[k], empirical=counts[k] / n) for k in keys])

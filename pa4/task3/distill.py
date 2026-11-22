"""Task 3 draft-model distillation (forward token-level KL from a frozen 1.5B teacher into LoRA on 0.5B).

Fixed settings: seeds 6304/6305; off-policy and on-policy each independently initialised; quota 40,000
teacher-scored response tokens per run (30,000 GSM8K + 10,000 short-form); response caps 192 / 32;
temperature 1, top-p 1; microbatch 1; one optimizer update per 256 teacher-scored tokens (a response
may be split across two updates; the final update may be short); AdamW lr 3e-4, wd 0.01; FP16 frozen
backbones, FP32 adapters and optimizer state, FP16 autocast + dynamic GradScaler, unscale before
global-norm clipping at 1.0; KL normalised by the supervised tokens of each update; LoRA r=8, alpha=16,
dropout 0 on q/k/v/o.

Prompt schedule: within each category a seeded permutation of the 500 GSM8K-train / 300 short-train
IDs is visited round-robin (re-used once exhausted); the next presentation comes from the category with
the smaller fraction of its token quota consumed (ties -> mathematics). Both methods use the same schedule.
Off-policy: one frozen teacher response per training prompt (all 800 prepared up front, timed), cycled
deterministically through the schedule. On-policy: a fresh response from the current student checkpoint at
every presentation (seed = rollout seed of that presentation; no gradient through generation).
"""
from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..common.utils import model_revision

TEACHER = "Qwen/Qwen2.5-1.5B-Instruct"
STUDENT = "Qwen/Qwen2.5-0.5B-Instruct"
QUOTA = dict(math=30_000, short=10_000)
CAP = dict(math=192, short=32)
TOKENS_PER_UPDATE = 256
LR, WD, CLIP = 3e-4, 0.01, 1.0
LORA = dict(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])


def load_causal(name: str, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rev = model_revision(name)
    tok = AutoTokenizer.from_pretrained(name, revision=rev)
    m = AutoModelForCausalLM.from_pretrained(name, revision=rev, torch_dtype=torch.float16,
                                             attn_implementation="sdpa").to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, tok, rev


def chat_ids(tok, text: str) -> List[int]:
    """Common non-thinking assistant format (identical for teacher, student, and every evaluation)."""
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True)


def eos_ids(tok) -> List[int]:
    ids = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    return sorted(i for i in ids if i is not None)


def check_tokenizers(t_tok, s_tok, prompts: List[str]) -> dict:
    bad = [p for p in prompts if chat_ids(t_tok, p) != chat_ids(s_tok, p) or t_tok.encode(p) != s_tok.encode(p)]
    return dict(n_checked=len(prompts), n_mismatch=len(bad), eos_teacher=eos_ids(t_tok), eos_student=eos_ids(s_tok),
                same_eos=eos_ids(t_tok) == eos_ids(s_tok))


# ----------------------------------------------------------------------------- schedule


def build_schedule(math_ids: List[str], short_ids: List[str], seed: int = 6304, max_len: int = 20000):
    """Deterministic list of (category, prompt_id, rollout_seed); consumed lazily against the quotas."""
    rng = random.Random(f"PA4-distill-schedule:{seed}")
    perms = dict(math=list(math_ids), short=list(short_ids))
    rng.shuffle(perms["math"]); rng.shuffle(perms["short"])
    return perms


class Scheduler:
    def __init__(self, perms: Dict[str, List[str]], seed: int):
        self.perms, self.seed = perms, seed
        self.ptr = dict(math=0, short=0)
        self.used = dict(math=0, short=0)
        self.k = 0

    def done(self):
        return all(self.used[c] >= QUOTA[c] for c in QUOTA)

    def next(self):
        open_ = [c for c in ("math", "short") if self.used[c] < QUOTA[c]]
        c = min(open_, key=lambda c: (self.used[c] / QUOTA[c], 0 if c == "math" else 1))
        pid = self.perms[c][self.ptr[c] % len(self.perms[c])]
        self.ptr[c] += 1
        self.k += 1
        return c, pid, self.seed * 1_000_003 + self.k

    def consume(self, c, n_tokens: int) -> int:
        n = min(n_tokens, QUOTA[c] - self.used[c])
        self.used[c] += n
        return n


# ----------------------------------------------------------------------------- generation


@torch.no_grad()
def sample_response(model, prompt_ids: List[int], cap: int, seed: int, eos: List[int]) -> dict:
    """Ancestral sampling (temperature 1, top-p 1, no top-k) with a dedicated seeded torch.Generator."""
    dev = next(model.parameters()).device
    gen = torch.Generator(device=dev).manual_seed(seed)
    ids = torch.tensor([prompt_ids], device=dev)
    out = _manual_sample(model, ids, cap, gen, eos)
    resp = out[0, ids.shape[1]:].tolist()
    reason = "eos" if (resp and resp[-1] in eos) else "length"
    return dict(response=resp, termination=reason, seed=seed)


@torch.no_grad()
def _manual_sample(model, ids, cap, gen, eos):
    from transformers import DynamicCache

    cache = DynamicCache()
    out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    seq = ids
    for _ in range(cap):
        p = torch.softmax(out.logits[0, -1].float(), -1)
        y = torch.multinomial(p, 1, generator=gen)
        seq = torch.cat([seq, y.view(1, 1)], 1)
        if int(y) in eos:
            break
        out = model(input_ids=y.view(1, 1), past_key_values=cache, use_cache=True)
    return seq


# ----------------------------------------------------------------------------- KL


def token_kl(t_logits: torch.Tensor, s_logits: torch.Tensor) -> torch.Tensor:
    """Per-position forward KL(p_T || q_S) over the common full vocabulary (FP32)."""
    lt = torch.log_softmax(t_logits.float(), -1)
    ls = torch.log_softmax(s_logits.float(), -1)
    return (lt.exp() * (lt - ls)).sum(-1)


def add_lora(student, seed: int):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    cfg = LoraConfig(r=LORA["r"], lora_alpha=LORA["lora_alpha"], lora_dropout=LORA["lora_dropout"],
                     target_modules=LORA["target_modules"], bias="none", task_type="CAUSAL_LM")
    pm = get_peft_model(student, cfg)
    for n, p in pm.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()  # FP32 adapters (and therefore FP32 AdamW states)
    return pm


def distill(method: str, seed: int, teacher, student, tok, prompts: Dict[str, dict], perms, pool: Optional[dict],
            log=print, ckpt_path=None, ckpt_every: int = 10) -> dict:
    """Run one distillation. prompts: pid -> dict(ids, category). pool: pid -> teacher response (off-policy).

    If ckpt_path is given, the complete training state is saved atomically every `ckpt_every` optimizer updates
    and an existing checkpoint is resumed (adapter weights, AdamW + GradScaler state, prompt-schedule position,
    partially consumed response, logs, presentations, cost counters, elapsed time). Rollout seeds are per
    presentation, so a resumed run follows the same schedule as an uninterrupted one."""
    assert method in ("off", "on")
    dev = next(teacher.parameters()).device
    eos = eos_ids(tok)
    model = add_lora(student, seed)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    n_adapter = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    scaler = torch.amp.GradScaler("cuda")
    sched = Scheduler(perms, seed)
    total_quota = sum(QUOTA.values())
    logs, presentations = [], []
    costs = dict(student_rollout_s=0.0, teacher_scoring_s=0.0, student_fwd_bwd_s=0.0, optimizer_s=0.0)
    pending = None  # (presentation record, remaining positions [a,b))
    scored_total = 0
    update = 0
    skipped = 0
    prior_elapsed, prior_peak, resumed_from = 0.0, 0, []
    if ckpt_path is not None and Path(ckpt_path).exists():
        st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert st["method"] == method and st["seed"] == seed
        named = {n: p for n, p in model.named_parameters() if p.requires_grad}
        for n, t in st["adapter"].items():
            named[n].data.copy_(t.to(named[n].device))
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        sched.ptr, sched.used, sched.k = st["sched"]["ptr"], st["sched"]["used"], st["sched"]["k"]
        logs, presentations, costs = st["logs"], st["presentations"], st["costs"]
        scored_total, update, skipped = st["scored_total"], st["update"], st["skipped"]
        pending = None if st["pending"] is None else [presentations[st["pending"][0]], st["pending"][1], st["pending"][2]]
        prior_elapsed, prior_peak = st["elapsed_s"], st["peak_alloc_bytes"]
        resumed_from = st.get("resumed_from", []) + [update]
        log(f"  [{method} s{seed}] resumed at update {update} ({scored_total} scored tokens)")

    def save_ckpt():
        if ckpt_path is None:
            return
        from ..common.utils import atomic_torch_save

        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        atomic_torch_save(dict(
            method=method, seed=seed, adapter={n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
            opt=opt.state_dict(), scaler=scaler.state_dict(), sched=dict(ptr=sched.ptr, used=sched.used, k=sched.k),
            logs=logs, presentations=presentations, costs=costs, scored_total=scored_total, update=update,
            skipped=skipped, pending=None if pending is None else [pending[0]["idx"], pending[1], pending[2]],
            elapsed_s=prior_elapsed + time.perf_counter() - t_run, peak_alloc_bytes=max(prior_peak, peak),
            resumed_from=resumed_from), ckpt_path)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_run = time.perf_counter()
    while scored_total < total_quota:
        need = min(TOKENS_PER_UPDATE, total_quota - scored_total)
        got = 0
        seg_losses = []
        opt.zero_grad(set_to_none=True)
        while got < need:
            if pending is None:
                c, pid, rseed = sched.next()
                pr = prompts[pid]
                t0 = time.perf_counter()
                if method == "off":
                    resp = pool[pid]
                else:
                    model.eval()
                    resp = sample_response(model, pr["ids"], CAP[c], rseed, eos)
                    model.train()
                    _sync()
                    costs["student_rollout_s"] += time.perf_counter() - t0
                n_allowed = sched.consume(c, len(resp["response"]))
                rec = dict(idx=len(presentations), category=c, prompt_id=pid, rollout_seed=resp["seed"],
                           response_ids=resp["response"], termination=resp["termination"],
                           scored_tokens=n_allowed, truncated_by_quota=n_allowed < len(resp["response"]))
                presentations.append(rec)
                if n_allowed == 0:
                    continue
                pending = [rec, 0, n_allowed]
            rec, a, b = pending
            take = min(b - a, need - got)
            seq = prompts[rec["prompt_id"]]["ids"] + rec["response_ids"][: a + take]
            P = len(prompts[rec["prompt_id"]]["ids"])
            ids = torch.tensor([seq], device=dev)
            t0 = time.perf_counter()
            with torch.no_grad():
                tl = teacher(input_ids=ids, use_cache=False).logits[0, P - 1 + a: P - 1 + a + take]
            _sync()
            costs["teacher_scoring_s"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16):
                sl = model(input_ids=ids, use_cache=False).logits[0, P - 1 + a: P - 1 + a + take]
            kl = token_kl(tl, sl)
            loss = kl.sum() / need
            scaler.scale(loss).backward()
            _sync()
            costs["student_fwd_bwd_s"] += time.perf_counter() - t0
            seg_losses.append(float(kl.detach().sum()))
            got += take
            pending = None if a + take >= b else [rec, a + take, b]
        t0 = time.perf_counter()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(params, CLIP)
        scale_before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        step_skipped = scaler.get_scale() < scale_before
        skipped += int(step_skipped)
        _sync()
        costs["optimizer_s"] += time.perf_counter() - t0
        update += 1
        scored_total += got
        logs.append(dict(update=update, scored_tokens=scored_total, math_tokens=sched.used["math"],
                         short_tokens=sched.used["short"], kl_per_token=sum(seg_losses) / got, lr=LR,
                         grad_norm=float(gn), skipped=step_skipped, loss_scale=scale_before,
                         elapsed_s=prior_elapsed + time.perf_counter() - t_run))
        if update % ckpt_every == 0:
            save_ckpt()
        if update % 20 == 0:
            log(f"  [{method} s{seed}] update {update} tokens {scored_total} KL/token {logs[-1]['kl_per_token']:.4f}")
    save_ckpt()
    model.eval()
    n_base = sum(p.numel() for n, p in model.named_parameters() if not p.requires_grad)
    res = dict(method=method, seed=seed, updates=update, skipped_updates=skipped, n_adapter_params=n_adapter,
               n_student_backbone_params=n_base, n_teacher_params=sum(p.numel() for p in teacher.parameters()),
               resumed_from_updates=resumed_from,
               logs=logs, costs=costs, wall_s=prior_elapsed + time.perf_counter() - t_run,
               peak_alloc_bytes=max(prior_peak, int(torch.cuda.max_memory_allocated())) if torch.cuda.is_available() else None,
               n_presentations=len(presentations),
               tokens=dict(math=sched.used["math"], short=sched.used["short"]),
               rollout_lengths=dict(math=[len(p["response_ids"]) for p in presentations if p["category"] == "math"],
                                    short=[len(p["response_ids"]) for p in presentations if p["category"] == "short"]))
    return dict(result=res, model=model, presentations=presentations)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def teacher_pool(teacher, tok, prompts: Dict[str, dict], seed: int = 6304, log=print, partial_path=None,
                 save_every: int = 25) -> dict:
    """One frozen teacher response per training prompt (all 800), seeds recorded.

    With partial_path, finished responses (and the generation time spent so far) are saved atomically every
    `save_every` prompts and reused after a disconnect; each prompt has its own seed, so the pool is identical."""
    from ..common.utils import read_json, write_json

    eos = eos_ids(tok)
    pool, prior = {}, 0.0
    if partial_path is not None and Path(partial_path).exists():
        st = read_json(partial_path)
        pool, prior = st["pool"], st["seconds"]
        log(f"  teacher pool: resuming with {len(pool)}/{len(prompts)} responses")
    t0 = time.perf_counter()
    for i, (pid, pr) in enumerate(sorted(prompts.items())):
        if pid in pool:
            continue
        rseed = seed * 1_000_003 + 10_000_000 + i
        pool[pid] = sample_response(teacher, pr["ids"], CAP[pr["category"]], rseed, eos)
        if partial_path is not None and len(pool) % save_every == 0:
            _sync()
            write_json(partial_path, dict(pool=pool, seconds=prior + time.perf_counter() - t0))
        if (i + 1) % 100 == 0:
            log(f"  teacher pool {i+1}/{len(prompts)}")
    _sync()
    return dict(pool=pool, seconds=prior + time.perf_counter() - t0,
                tokens=dict(math=sum(len(v["response"]) for k, v in pool.items() if prompts[k]["category"] == "math"),
                            short=sum(len(v["response"]) for k, v in pool.items() if prompts[k]["category"] == "short")))

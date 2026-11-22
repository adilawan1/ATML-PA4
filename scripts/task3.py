#!/usr/bin/env python
"""Task 3 driver: exact speculative sampling + off/on-policy draft distillation.

  python scripts/task3.py check        # tokenizer agreement, exact Markov chi^2 tests, real-model cache checks
  python scripts/task3.py baseline     # pretrained-draft speculative benchmark + isolated step costs
  python scripts/task3.py pool         # frozen teacher-response pool for off-policy training (timed)
  python scripts/task3.py train        # off/on-policy x seeds 6304/6305 (resumable)
  python scripts/task3.py merge        # merge adapters into fresh FP16 backbones, verify logits
  python scripts/task3.py banks        # exact-speculative bank (original draft, gamma=4) + own-student banks
  python scripts/task3.py align        # KL / TV / acceptance on all common banks for all drafts
  python scripts/task3.py spec         # EOS-aware 120-prompt speculative benchmark, all drafts, gamma in {2,4}
  python scripts/task3.py standalone   # standalone answer accuracy of each draft (and the teacher)
  python scripts/task3.py all
"""
from __future__ import annotations

import argparse
import gc
import gzip
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from pa4.common import data as D
from pa4.common.utils import (ckpt_root, environment_record, out_dir, read_json, read_jsonl, set_seed, time_fn,
                              write_json, write_jsonl)
from pa4.task3 import align as AL
from pa4.task3 import distill as DS
from pa4.task3.spec import HFRunner, speculative_generate, target_only_generate, toy_eos_test, toy_markov_test

SEEDS = [6304, 6305]
METHODS = ["off", "on"]
GAMMAS = [2, 4]
REPS = 2
SPEED_CAP = 64
DRAFTS = ["original"] + [f"{m}_s{s}" for m in METHODS for s in SEEDS]
RES = out_dir("task3")
CK = ckpt_root() / "task3"


# ----------------------------------------------------------------------------- prompts


def test_prompts():
    out = []
    for r in D.gsm8k("test"):
        gold = r["example"]["answer"].split("####")[-1].strip().replace(",", "")
        out.append(dict(pid=f"GSM8K-{r['order']:04d}", text=r["example"]["question"], domain="math", split="math", gold=gold))
    for r in D.short_form("test"):
        out.append(dict(pid=r["id"], text=r["input"], domain="short", split=r["split"], gold=r["answer"], family=r["family"]))
    assert len(out) == 120
    return out


def train_prompts(tok):
    pr = {}
    for r in D.gsm8k("train"):
        pr[f"GSM8K-train-{r['order']:04d}"] = dict(ids=DS.chat_ids(tok, r["example"]["question"]), category="math")
    for r in D.short_form("train"):
        pr[r["id"]] = dict(ids=DS.chat_ids(tok, r["input"]), category="short")
    assert len(pr) == 800
    return pr


def load_draft(name: str):
    if name == "original":
        return DS.load_causal(DS.STUDENT)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = CK / "merged" / name
    m = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
    return m, AutoTokenizer.from_pretrained(p), None


def free(*objs):
    for o in objs:
        del o
    gc.collect(); torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- check


def cmd_check():
    od = out_dir("task3", "checks")
    toy = toy_markov_test(100_000, 3, 2)
    toy_eos = toy_eos_test(100_000, 3, 3)
    toy_g1 = toy_markov_test(20_000, 3, 1, seed=6306)
    t, ttok, trev = DS.load_causal(DS.TEACHER)
    s, stok, srev = DS.load_causal(DS.STUDENT)
    texts = [p["text"] for p in test_prompts()] + [r["example"]["question"] for r in D.gsm8k("train")] + \
            [r["input"] for r in D.short_form("train")]
    tokc = DS.check_tokenizers(ttok, stok, texts)
    # real-model cache consistency: after speculative rounds with rollbacks, cached logits == full forward
    prompts = test_prompts()
    eos = DS.eos_ids(ttok)
    tr, dr = HFRunner(t), HFRunner(s)
    cons = []
    for pr in (prompts[0], prompts[60], prompts[90]):
        ids = DS.chat_ids(ttok, pr["text"])
        gen = torch.Generator(device="cuda").manual_seed(6304)
        out, st = speculative_generate(tr, dr, ids, 4, 48, gen, eos)
        seq = ids + out
        # continue one step from the speculative cache state vs full recomputation
        cached = tr.feed(seq[tr.n_cached:])[-1]
        full = torch.softmax(t(input_ids=torch.tensor([seq], device="cuda")).logits[0, -1].float(), -1)
        # split-prefill equivalence on the prompt
        tr.reset()
        a = tr.feed(ids[:-1]); b = tr.feed(ids[-1:])[-1]
        whole = torch.softmax(t(input_ids=torch.tensor([ids], device="cuda")).logits[0, -1].float(), -1)
        # crop/re-feed equivalence
        tr.crop(len(ids) - 5)
        c = tr.feed(ids[len(ids) - 5:])[-1]
        cons.append(dict(prompt=pr["pid"], rounds=st.rounds, bonus=st.bonus, corrections=st.corrections,
                         eos=st.eos, n_out=st.n_out, cache_vs_full_max_abs_prob=float((cached - full).abs().max()),
                         split_prefill_max_abs_prob=float((b - whole).abs().max()),
                         crop_refeed_max_abs_prob=float((c - whole).abs().max())))
    write_json(od / "sampler_checks.json", dict(markov_gamma2=toy, markov_eos_gamma3=toy_eos, markov_gamma1=toy_g1,
                                                tokenizer=tokc, real_model_cache=cons, target_revision=trev,
                                                draft_revision=srev, env=environment_record()))
    print("chi2 exact p=%.3f, negative control p=%.2e, eos p=%.3f" % (toy["exact"]["p_value"], toy["p"]["p_value"], toy_eos["p_value"]))
    print(cons)


# ----------------------------------------------------------------------------- isolated costs


@torch.no_grad()
def step_costs(model, lens=(64, 256, 512), gammas=(1, 2, 4), reps=20):
    from transformers import DynamicCache

    out = {}
    for L in lens:
        ids = torch.randint(100, 20000, (1, L), generator=torch.Generator().manual_seed(L)).cuda()
        for g in gammas:
            nxt = torch.randint(100, 20000, (1, g + 1 if g > 1 else 1)).cuda()
            cache = DynamicCache()
            model(input_ids=ids, past_key_values=cache, use_cache=True)

            def f():
                model(input_ids=nxt, past_key_values=cache, use_cache=True)
                cache.crop(L)

            t = time_fn(f, reps, 3)
            out[f"L{L}_tokens{nxt.shape[1]}"] = 1000 * t["mean"]
    return out


def cmd_costs(drafts=("original",)):
    t, ttok, _ = DS.load_causal(DS.TEACHER)
    res = dict(target=step_costs(t), unit="ms per forward (incl. cache crop)")
    free(t)
    for d in drafts:
        m, _, _ = load_draft(d)
        res[d] = step_costs(m)
        free(m)
    write_json(out_dir("task3", "spec") / "step_costs.json", res)
    return res


# ----------------------------------------------------------------------------- speculative benchmark


def _spec_records_path(tag):
    return out_dir("task3", "spec") / f"records_{tag}.jsonl"


@torch.no_grad()
def run_spec(draft_name: str, gammas=GAMMAS, reps=REPS, prompts=None, tag=None, cap=SPEED_CAP, ignore_eos=False,
             draft_model=None):
    tag = tag or draft_name
    path = _spec_records_path(tag)
    done = set()
    if path.exists():
        for l in open(path):
            r = json.loads(l)
            done.add((r["prompt_id"], r["gamma"], r["rep"]))
    t, ttok, _ = DS.load_causal(DS.TEACHER)
    d = draft_model if draft_model is not None else load_draft(draft_name)[0]
    tr, dr = HFRunner(t), HFRunner(d)
    eos = [] if ignore_eos else DS.eos_ids(ttok)
    prompts = prompts or test_prompts()
    # warm-up
    w = DS.chat_ids(ttok, prompts[0]["text"])
    for _ in range(2):
        speculative_generate(tr, dr, w, 4, 16, torch.Generator(device="cuda").manual_seed(0), eos, timing=True)
        target_only_generate(tr, w, 16, torch.Generator(device="cuda").manual_seed(0), eos, timing=True)
    with open(path, "a") as fo:
        for gi, g in enumerate(gammas):
            for rep in range(reps):
                for pi, pr in enumerate(prompts):
                    if (pr["pid"], g, rep) in done:
                        continue
                    ids = DS.chat_ids(ttok, pr["text"])
                    seed = 6304 + 1000 * rep + pi
                    order = ("spec", "target") if (pi + rep + gi) % 2 == 0 else ("target", "spec")
                    rec = dict(draft=draft_name, tag=tag, prompt_id=pr["pid"], domain=pr["domain"], split=pr["split"],
                               gamma=g, rep=rep, seed=seed, order=list(order), prompt_len=len(ids), cap=cap,
                               ignore_eos=ignore_eos)
                    for which in order:
                        torch.cuda.reset_peak_memory_stats()
                        gen = torch.Generator(device="cuda").manual_seed(seed)
                        if which == "spec":
                            out, st = speculative_generate(tr, dr, ids, g, cap, gen, eos, timing=True)
                            sd = st.to_dict()
                            sd.pop("accepted_per_round"); sd.pop("proposed_per_round")
                            rec["spec"] = dict(sd, accepted_hist=np.bincount(st.accepted_per_round, minlength=g + 1).tolist(),
                                               peak_alloc_bytes=int(torch.cuda.max_memory_allocated()),
                                               tokens_per_s=st.n_out / st.t_total, out_ids=out)
                        else:
                            out, info = target_only_generate(tr, ids, cap, gen, eos, timing=True)
                            rec["target"] = dict(info, peak_alloc_bytes=int(torch.cuda.max_memory_allocated()),
                                                 tokens_per_s=info["n_out"] / info["t_total"], out_ids=out)
                    rec["speedup"] = rec["target"]["t_total"] / rec["spec"]["t_total"]
                    fo.write(json.dumps(rec) + "\n"); fo.flush()
                print(f"[{tag}] gamma={g} rep={rep} done", flush=True)
    free(t, d)


def cmd_baseline():
    cmd_costs(("original",))
    run_spec("original")
    prompts = [p for p in test_prompts() if p["domain"] == "math"][:24]
    run_spec("original", prompts=prompts, tag="original_fixed64", ignore_eos=True)


# ----------------------------------------------------------------------------- distillation


def cmd_pool():
    f = CK / "teacher_pool.json"
    if f.exists():
        print("[skip] pool"); return
    t, tok, _ = DS.load_causal(DS.TEACHER)
    pool = DS.teacher_pool(t, tok, train_prompts(tok))
    f.parent.mkdir(parents=True, exist_ok=True)
    json.dump(pool, open(f, "w"))
    write_json(out_dir("task3", "distill") / "teacher_pool_cost.json",
               dict(seconds=pool["seconds"], tokens=pool["tokens"], n=len(pool["pool"]),
                    terminations={k: sum(1 for v in pool["pool"].values() if v["termination"] == k) for k in ("eos", "length")}))
    free(t)


def cmd_train(methods=METHODS, seeds=SEEDS):
    t, tok, _ = DS.load_causal(DS.TEACHER)
    prompts = train_prompts(tok)
    math_ids = sorted(k for k, v in prompts.items() if v["category"] == "math")
    short_ids = sorted(k for k, v in prompts.items() if v["category"] == "short")
    pool = json.load(open(CK / "teacher_pool.json"))["pool"] if "off" in methods else None
    for method in methods:
        for seed in seeds:
            name = f"{method}_s{seed}"
            od = out_dir("task3", "distill", name)
            if (od / "result.json").exists():
                print("[skip]", name); continue
            set_seed(seed)
            s, _, _ = DS.load_causal(DS.STUDENT)
            perms = DS.build_schedule(math_ids, short_ids, seed=6304)  # same prompt schedule for every run
            r = DS.distill(method, seed, t, s, tok, prompts, perms, pool)
            res = r["result"]
            if method == "off":
                pc = read_json(out_dir("task3", "distill") / "teacher_pool_cost.json")
                res["costs"]["teacher_pool_generation_s_shared"] = pc["seconds"]
            write_jsonl(od / "presentations.jsonl.gz", r["presentations"])
            r["model"].save_pretrained(CK / "adapters" / name)
            res["env"] = environment_record()
            write_json(od / "result.json", res)
            free(r["model"], s)
    free(t)


@torch.no_grad()
def cmd_merge():
    from peft import PeftModel

    t, tok, _ = DS.load_causal(DS.TEACHER)
    probe = [DS.chat_ids(tok, p["text"]) for p in test_prompts()[::15]]
    free(t)
    for m in METHODS:
        for s in SEEDS:
            name = f"{m}_s{s}"
            dst = CK / "merged" / name
            f = out_dir("task3", "distill", name) / "merge.json"
            if f.exists():
                print("[skip]", name); continue
            base, btok, _ = DS.load_causal(DS.STUDENT)
            t0 = time.perf_counter()
            peft_m = PeftModel.from_pretrained(base, CK / "adapters" / name)
            peft_m = peft_m.to(torch.float16).eval()
            unmerged = [peft_m(input_ids=torch.tensor([ids], device="cuda")).logits[0].float() for ids in probe]
            merged = peft_m.merge_and_unload()
            torch.cuda.synchronize()
            t_merge = time.perf_counter() - t0
            err = [float((merged(input_ids=torch.tensor([ids], device="cuda")).logits[0].float() - u).abs().max())
                   for ids, u in zip(probe, unmerged)]
            merged.save_pretrained(dst)
            btok.save_pretrained(dst)
            write_json(f, dict(merge_seconds=t_merge, max_abs_logit_err=max(err), per_prompt=err, n_prompts=len(probe)))
            free(merged, peft_m, base)


# ----------------------------------------------------------------------------- banks & alignment


@torch.no_grad()
def cmd_banks():
    od = out_dir("task3", "banks")
    t, tok, _ = DS.load_causal(DS.TEACHER)
    eos = DS.eos_ids(tok)
    prompts = test_prompts()
    f = od / "exact_spec_original_gamma4.jsonl.gz"
    if not f.exists():
        d, _, _ = load_draft("original")
        tr, dr = HFRunner(t), HFRunner(d)
        recs = []
        for j, pr in enumerate(prompts):
            seed = 6304 + 10007 * j  # same per-prompt seed rule as the course teacher bank
            gen = torch.Generator(device="cuda").manual_seed(seed)
            out, st = speculative_generate(tr, dr, DS.chat_ids(tok, pr["text"]), 4, 192, gen, eos)
            recs += AL.bank_records(pr["pid"], pr["domain"], pr["text"], "exact_spec_original_gamma4", out,
                                    dict(source_generation_seed=seed, producer="exact speculative sampler, original 0.5B draft, gamma=4"))
        write_jsonl(f, recs)
        free(d)
    free(t)
    for name in DRAFTS[1:]:
        f = od / f"own_{name}.jsonl.gz"
        if f.exists():
            continue
        d, _, _ = load_draft(name)
        recs = []
        for j, pr in enumerate(prompts):
            seed = 6304 + 10007 * j + 777
            r = DS.sample_response(d, DS.chat_ids(tok, pr["text"]), 192, seed, eos)
            recs += AL.bank_records(pr["pid"], pr["domain"], pr["text"], f"own_{name}", r["response"],
                                    dict(source_generation_seed=seed, producer=f"merged student {name}"))
        write_jsonl(f, recs)
        free(d)


@torch.no_grad()
def cmd_align():
    od = out_dir("task3", "align")
    banks = [r for r in D.prefix_banks()]
    for f in sorted(out_dir("task3", "banks").glob("*.jsonl.gz")):
        banks += read_jsonl(f)
    groups = AL.group_bank(banks)
    split_of = {p["pid"]: p["split"] for p in test_prompts()}
    t, tok, _ = DS.load_causal(DS.TEACHER)
    drafts = {n: load_draft(n)[0] for n in DRAFTS}
    rows = []
    for (bank, pid), e in sorted(groups.items()):
        ids = DS.chat_ids(tok, e["prompt"])
        lp = AL.position_logprobs(t, ids, e["longest"], e["positions"])
        resp_tokens = e["longest"]
        for name, m in drafts.items():
            lq = AL.position_logprobs(m, ids, e["longest"], e["positions"])
            met = AL.pair_metrics(lp, lq)
            for i, pos in enumerate(e["positions"]):
                rows.append(dict(bank=bank, prompt_id=pid, split=split_of[pid], position=pos, pos_bin=AL.pos_bin(pos),
                                 draft=name, **{k: v[i] for k, v in met.items()},
                                 target_logp_sampled=float(lp[i, resp_tokens[pos - 1]]) if pos - 1 < len(resp_tokens) else None))
    write_jsonl(od / "positions.jsonl.gz", rows)
    free(t, *drafts.values())


# ----------------------------------------------------------------------------- post-distillation spec + standalone


def cmd_spec():
    for name in DRAFTS:
        run_spec(name)
    cmd_costs(tuple(DRAFTS))
    # wrapper overhead of an unmerged adapter (not the primary comparison)
    from peft import PeftModel

    base, _, _ = DS.load_causal(DS.STUDENT)
    pm = PeftModel.from_pretrained(base, CK / "adapters" / "on_s6304").to(torch.float16).eval()
    run_spec("on_s6304_unmerged", gammas=[4], reps=1, prompts=test_prompts()[::6], draft_model=pm)
    free(pm, base)


def gsm_extract(text: str):
    m = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
    return m[-1].replace(",", "").rstrip(".") if m else None


FAMILY_RE = {0: r"^[A-Za-z]+$", 1: r"^\d+$", 2: r"^[A-Z]{2,3}$"}


@torch.no_grad()
def cmd_standalone():
    od = out_dir("task3", "standalone")
    prompts = test_prompts()
    for name in ["teacher"] + DRAFTS:
        f = od / f"{name}.jsonl"
        if f.exists():
            continue
        m, tok, _ = DS.load_causal(DS.TEACHER) if name == "teacher" else load_draft(name)
        tok = tok or DS.load_causal(DS.STUDENT)[1]
        eos = DS.eos_ids(tok)
        rows = []
        for pr in prompts:
            cap = 512 if pr["domain"] == "math" else 32
            ids = torch.tensor([DS.chat_ids(tok, pr["text"])], device="cuda")
            g = m.generate(ids, max_new_tokens=cap, do_sample=False, eos_token_id=eos, pad_token_id=eos[0])[0, ids.shape[1]:].tolist()
            text = tok.decode(g, skip_special_tokens=True).strip()
            truncated = not (g and g[-1] in eos)
            if pr["domain"] == "math":
                pred = gsm_extract(text)
                ok = pred is not None and abs(float(pred) - float(pr["gold"])) < 1e-6 if pred and re.match(r"^-?\d+\.?\d*$", pred) else False
                rows.append(dict(prompt_id=pr["pid"], split="math", pred=pred, gold=pr["gold"], correct=bool(ok),
                                 extractable=pred is not None, truncated=truncated, n_tokens=len(g), text=text))
            else:
                norm = text.rstrip(".").strip()
                rows.append(dict(prompt_id=pr["pid"], split=pr["split"], pred=norm, gold=pr["gold"],
                                 correct=norm == pr["gold"], format_ok=bool(re.match(FAMILY_RE[pr["family"]], norm)),
                                 truncated=truncated, n_tokens=len(g), text=text))
        write_jsonl(f, rows)
        free(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "baseline", "pool", "train", "merge", "banks", "align", "spec",
                                    "standalone", "costs", "all"])
    ap.add_argument("--methods", nargs="*", default=METHODS)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    a = ap.parse_args()
    steps = dict(check=cmd_check, baseline=cmd_baseline, pool=cmd_pool,
                 train=lambda: cmd_train(a.methods, a.seeds), merge=cmd_merge, banks=cmd_banks,
                 align=cmd_align, spec=cmd_spec, standalone=cmd_standalone, costs=lambda: cmd_costs(tuple(DRAFTS)))
    order = ["check", "baseline", "pool", "train", "merge", "banks", "align", "spec", "standalone"] if a.cmd == "all" else [a.cmd]
    for s in order:
        print(f"##### {s}", flush=True)
        steps[s]()


if __name__ == "__main__":
    main()

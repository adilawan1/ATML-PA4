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
from collections import defaultdict
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
from pa4.common.utils import (JsonlAppender, ckpt_root, environment_record, out_dir, read_json, read_jsonl, set_seed, time_fn,
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


def free(*_ignored):
    """Release cached GPU memory. Callers must `del` their own references first (a function cannot)."""
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
    # held-out hygiene: no training prompt (GSM8K train / short-form train) overlaps an evaluation prompt
    tr_math = {r["example"]["question"].strip() for r in D.gsm8k("train")}
    tr_short = D.short_form("train")
    te = test_prompts()
    overlap = dict(
        gsm8k_question_overlap=sum(p["text"].strip() in tr_math for p in te if p["domain"] == "math"),
        short_id_overlap=len({r["id"] for r in tr_short} & {p["pid"] for p in te}),
        short_text_overlap=len({r["input"].strip() for r in tr_short} & {p["text"].strip() for p in te if p["domain"] == "short"}),
        test_split_counts={k: sum(p["split"] == k for p in te) for k in ("math", "familiar", "shifted")},
        train_ids=dict(gsm8k=[f"GSM8K-train-{r['order']:04d}" for r in D.gsm8k("train")], short=[r["id"] for r in tr_short]),
        test_ids=[p["pid"] for p in te])
    assert overlap["gsm8k_question_overlap"] == overlap["short_id_overlap"] == overlap["short_text_overlap"] == 0, overlap
    write_json(od / "prompt_sets_and_overlap.json", overlap)
    write_json(od / "sampler_checks.json", dict(markov_gamma2=toy, markov_eos_gamma3=toy_eos, markov_gamma1=toy_g1,
                                                tokenizer=tokc, real_model_cache=cons, target_revision=trev,
                                                draft_revision=srev, env=environment_record(dict(precision="float16", attn_implementation=t.config._attn_implementation,
                                                tokenizer_target=f"{DS.TEACHER}@{trev}", tokenizer_draft=f"{DS.STUDENT}@{srev}"))))
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
    del t
    free()
    for d in drafts:
        m, _, _ = load_draft(d)
        res[d] = step_costs(m)
        del m
        free()
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
    log = JsonlAppender(path, key_fn=lambda r: (r["prompt_id"], r["gamma"], r["rep"]))
    done = log.done
    todo = [(p["pid"], g, rep) for g in gammas for rep in range(reps) for p in (prompts or test_prompts())]
    if all(k in done for k in todo):
        print(f"[skip] {tag}: all records present"); return
    t_setup0 = time.perf_counter()
    t, ttok, _ = DS.load_causal(DS.TEACHER)
    d = draft_model if draft_model is not None else load_draft(draft_name)[0]
    tr, dr = HFRunner(t), HFRunner(d)
    eos = [] if ignore_eos else DS.eos_ids(ttok)
    prompts = prompts or test_prompts()
    torch.cuda.synchronize()
    t_load = time.perf_counter() - t_setup0
    # warm-up (recorded separately from the timed runs)
    w = DS.chat_ids(ttok, prompts[0]["text"])
    t0 = time.perf_counter()
    for _ in range(2):
        speculative_generate(tr, dr, w, 4, 16, torch.Generator(device="cuda").manual_seed(0), eos, timing=True)
        target_only_generate(tr, w, 16, torch.Generator(device="cuda").manual_seed(0), eos, timing=True)
    torch.cuda.synchronize()
    setup_file = out_dir("task3", "spec") / f"setup_{tag}.json"
    setups = read_json(setup_file)["sessions"] if setup_file.exists() else []
    setups.append(dict(model_load_s=t_load, warmup_s=time.perf_counter() - t0, warmup_calls="2 x (spec gamma=4 + target-only), 16 tokens",
                       already_done=len(done)))
    write_json(setup_file, dict(sessions=setups))
    with log:
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
                    log.append(rec)
                print(f"[{tag}] gamma={g} rep={rep} done", flush=True)
    del t, d
    free()


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
    pool = DS.teacher_pool(t, tok, train_prompts(tok), partial_path=CK / "teacher_pool.partial.json")
    write_json(f, pool)  # atomic; the partial file is kept as a record
    write_json(out_dir("task3", "distill") / "teacher_pool_cost.json",
               dict(seconds=pool["seconds"], tokens=pool["tokens"], n=len(pool["pool"]),
                    terminations={k: sum(1 for v in pool["pool"].values() if v["termination"] == k) for k in ("eos", "length")}))
    del t
    free()


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
            r = DS.distill(method, seed, t, s, tok, prompts, perms, pool,
                           ckpt_path=CK / "distill_ckpt" / f"{name}.pt")  # resumes automatically after a disconnect
            res = r["result"]
            if method == "off":
                pc = read_json(out_dir("task3", "distill") / "teacher_pool_cost.json")
                res["costs"]["teacher_pool_generation_s_shared"] = pc["seconds"]
            write_jsonl(od / "presentations.jsonl.gz", r["presentations"])
            r["model"].save_pretrained(CK / "adapters" / name)
            res["env"] = environment_record(dict(teacher=DS.TEACHER, student=DS.STUDENT, precision="float16 backbone / float32 LoRA",
                                                 attn_implementation="sdpa"))
            write_json(od / "result.json", res)
            del r, s
            free()
    del t
    free()


@torch.no_grad()
def cmd_merge():
    from peft import PeftModel

    t, tok, _ = DS.load_causal(DS.TEACHER)
    probe = [DS.chat_ids(tok, p["text"]) for p in test_prompts()[::15]]
    del t
    free()
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
            del merged, peft_m, base
            free()


# ----------------------------------------------------------------------------- banks & alignment


def _bank_with_resume(f, prompts, make_records):
    """Generate a bank prompt by prompt; each prompt's records are fsynced to <bank>.partial.jsonl so a disconnect
    loses at most one prompt. The final .jsonl.gz is written atomically once every prompt is present."""
    part = f.with_name(f.name.replace(".jsonl.gz", ".partial.jsonl"))
    with JsonlAppender(part, key_fn=lambda r: r["prompt_id"]) as log:
        for j, pr in enumerate(prompts):
            if pr["pid"] not in log.done:
                log.append(dict(prompt_id=pr["pid"], records=make_records(j, pr)))
        recs = [x for r in sorted(log.records, key=lambda r: r["prompt_id"]) for x in r["records"]]
    write_jsonl(f, recs)


@torch.no_grad()
def cmd_banks():
    od = out_dir("task3", "banks")
    t, tok, _ = DS.load_causal(DS.TEACHER)
    eos = DS.eos_ids(tok)
    prompts = test_prompts()
    write_json(od / "test_prompt_ids.json", [dict(prompt_id=p["pid"], split=p["split"]) for p in prompts])
    f = od / "exact_spec_original_gamma4.jsonl.gz"
    if not f.exists():
        d, _, _ = load_draft("original")
        tr, dr = HFRunner(t), HFRunner(d)

        def spec_records(j, pr):
            seed = 6304 + 10007 * j  # same per-prompt seed rule as the course teacher bank
            gen = torch.Generator(device="cuda").manual_seed(seed)
            out, st = speculative_generate(tr, dr, DS.chat_ids(tok, pr["text"]), 4, 192, gen, eos)
            return AL.bank_records(pr["pid"], pr["domain"], pr["text"], "exact_spec_original_gamma4", out,
                                   dict(source_generation_seed=seed, producer="exact speculative sampler, original 0.5B draft, gamma=4"))

        _bank_with_resume(f, prompts, spec_records)
        del d, tr, dr
        free()
    del t
    free()
    for name in DRAFTS[1:]:
        f = od / f"own_{name}.jsonl.gz"
        if f.exists():
            continue
        d, _, _ = load_draft(name)

        def own_records(j, pr, d=d, name=name):
            seed = 6304 + 10007 * j + 777
            r = DS.sample_response(d, DS.chat_ids(tok, pr["text"]), 192, seed, eos)
            return AL.bank_records(pr["pid"], pr["domain"], pr["text"], f"own_{name}", r["response"],
                                   dict(source_generation_seed=seed, producer=f"merged student {name}"))

        _bank_with_resume(f, prompts, own_records)
        del d
        free()


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
    # one fsynced record per (bank, prompt): a disconnect loses at most one group
    with JsonlAppender(od / "positions.partial.jsonl", key_fn=lambda r: (r["bank"], r["prompt_id"])) as log:
        for (bank, pid), e in sorted(groups.items()):
            if (bank, pid) in log.done:
                continue
            ids = DS.chat_ids(tok, e["prompt"])
            lp = AL.position_logprobs(t, ids, e["longest"], e["positions"])
            resp_tokens = e["longest"]
            rows = []
            for name, m in drafts.items():
                lq = AL.position_logprobs(m, ids, e["longest"], e["positions"])
                met = AL.pair_metrics(lp, lq)
                for i, pos in enumerate(e["positions"]):
                    rows.append(dict(bank=bank, prompt_id=pid, split=split_of[pid], position=pos, pos_bin=AL.pos_bin(pos),
                                     draft=name, **{k: v[i] for k, v in met.items()},
                                     target_logp_sampled=float(lp[i, resp_tokens[pos - 1]]) if pos - 1 < len(resp_tokens) else None))
            log.append(dict(bank=bank, prompt_id=pid, rows=rows))
        allrows = [x for r in log.records for x in r["rows"]]
    write_jsonl(od / "positions.jsonl.gz", allrows)
    # decoded aligned disagreements (largest TV gap original vs on-policy seed 6304 on the teacher bank)
    sel = defaultdict(dict)
    for r in allrows:
        if r["bank"] == "teacher" and r["draft"] in ("original", "on_s6304"):
            sel[(r["prompt_id"], r["position"])][r["draft"]] = r
    pairs = sorted((v for v in sel.values() if len(v) == 2), key=lambda v: v["on_s6304"]["tv"] - v["original"]["tv"])
    out = []
    for v in pairs[:5] + pairs[-3:]:
        g = groups[("teacher", v["original"]["prompt_id"])]
        pos = v["original"]["position"]
        out.append(dict(prompt_id=v["original"]["prompt_id"], position=pos,
                        prefix_tail=tok.decode(g["longest"][max(0, pos - 25):pos - 1]),
                        target_argmax=tok.decode([v["original"]["target_argmax"]]),
                        original_argmax=tok.decode([v["original"]["draft_argmax"]]),
                        on_policy_argmax=tok.decode([v["on_s6304"]["draft_argmax"]]),
                        tv_original=v["original"]["tv"], tv_on_s6304=v["on_s6304"]["tv"],
                        selection="5 largest TV improvements + 3 largest TV regressions (on_s6304 vs original), teacher bank"))
    write_json(od / "aligned_disagreements.json", out)
    del t, drafts
    free()


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
    del pm, base
    free()


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
        log = JsonlAppender(f, key_fn=lambda r: r["prompt_id"])  # per-prompt, resumable
        if all(p["pid"] in log.done for p in prompts):
            log.close(); continue
        m, tok, _ = DS.load_causal(DS.TEACHER) if name == "teacher" else load_draft(name)
        tok = tok or DS.load_causal(DS.STUDENT)[1]
        eos = DS.eos_ids(tok)
        with log:
            for pr in prompts:
                if pr["pid"] in log.done:
                    continue
                cap = 512 if pr["domain"] == "math" else 32
                ids = torch.tensor([DS.chat_ids(tok, pr["text"])], device="cuda")
                g = m.generate(ids, max_new_tokens=cap, do_sample=False, eos_token_id=eos, pad_token_id=eos[0])[0, ids.shape[1]:].tolist()
                text = tok.decode(g, skip_special_tokens=True).strip()
                truncated = not (g and g[-1] in eos)
                if pr["domain"] == "math":
                    pred = gsm_extract(text)
                    ok = bool(pred is not None and re.match(r"^-?\d+\.?\d*$", pred) and abs(float(pred) - float(pr["gold"])) < 1e-6)
                    log.append(dict(prompt_id=pr["pid"], split="math", pred=pred, gold=pr["gold"], correct=ok,
                                    extractable=pred is not None, truncated=truncated, n_tokens=len(g), text=text))
                else:
                    norm = text.rstrip(".").strip()
                    log.append(dict(prompt_id=pr["pid"], split=pr["split"], pred=norm, gold=pr["gold"],
                                    correct=norm == pr["gold"], format_ok=bool(re.match(FAMILY_RE[pr["family"]], norm)),
                                    truncated=truncated, n_tokens=len(g), text=text))
        del m
        free()


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

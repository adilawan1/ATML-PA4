#!/usr/bin/env python
"""Task 2C: KV-cache compression on the 180 frozen retrieval prompts + C5 systems benchmark.

  python scripts/task2c.py eval    # C1 validation + C2/C3/C4 policies on every prompt (resumable per prompt)
  python scripts/task2c.py bench   # C5 packed encode/decode + decode-step latency, batch {1,8} x L {2048,4096}
  python scripts/task2c.py all
Per-prompt records: results/task2/kv/cases.jsonl ; aligned generations: results/task2/kv/aligned.json
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from pa4.common import data as D
from pa4.common.utils import (JsonlAppender, environment_record, out_dir, read_json, read_jsonl, set_seed, time_fn,
                              write_json)
from pa4.task2 import kv, kv_pack

ALIGNED = {f"KV-{L}-{p}-8-00" for L in (2048, 4096) for p in ("start", "middle", "end")}


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def cmd_eval(limit=None):
    set_seed(6304)
    model, tok, rev = kv.load()
    cfg = model.config
    dev = next(model.parameters()).device
    od = out_dir("task2", "kv")
    fcase = od / "cases.jsonl"
    log = JsonlAppender(fcase, key_fn=lambda r: r["id"])  # resumes after a disconnect; torn last line dropped
    done = log.done
    print(f"resuming: {len(done)} prompts already evaluated" if done else "starting KV evaluation", flush=True)
    cases = D.kv_cases()[:limit] if limit else D.kv_cases()
    aligned = read_json(od / "aligned.json") if (od / "aligned.json").exists() else {}
    write_json(od / "setup.json", dict(model=kv.MODEL, revision=rev, sink=kv.SINK, recent=kv.RECENT, group=kv.GROUP,
                                       pool=kv.POOL, max_new_tokens=kv.MAX_NEW, methods=kv.METHODS,
                                       selection="per layer, shared by both KV heads of the layer",
                                       quant="asym min/max, FP16 scale + FP16 offset (group min), no integer zero-point",
                                       answer_extractor=r"first match of \d{5} in decoded greedy output",
                                       env=environment_record(dict(model=kv.MODEL, model_revision=rev, tokenizer=f"{kv.MODEL}@{rev}",
                                                                   precision="float16",
                                                                   attn_implementation=model.config._attn_implementation))))
    with log:
        for ci, raw in enumerate(cases):
            if raw["id"] in done:
                continue
            case = kv.prepare_case(tok, raw)
            ids = torch.tensor([case["ids"]], device=dev)
            L = case["L_ctx"]
            rec = dict(id=case["id"], n_tokens=case["n_tokens"], L_ctx=L, n_suffix=len(case["suffix"]),
                       target_tokens=case["target_tokens"], position=case["position"], distractors=case["distractors"],
                       n_evidence_tokens=len(case["evidence"]), code=case["code"], methods={})
            # ---- C1: full forward vs split prefill/decode; ordinary generate as reference
            full_last = model(input_ids=ids, use_cache=False).logits[0, -1].float()
            _sync(); torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            pre = model(input_ids=ids[:, :L], use_cache=True)
            _sync()
            rec["prefill_seconds"] = time.perf_counter() - t0
            rec["prefill_peak_alloc_bytes"] = int(torch.cuda.max_memory_allocated())
            ctxK, ctxV = kv.cache_tensors(pre.past_key_values)
            ctxK, ctxV = [k.clone() for k in ctxK], [v.clone() for v in ctxV]
            del pre
            t0 = time.perf_counter()
            scores, split_last = kv.question_scores(model, ctxK, ctxV, L, case["suffix"])
            _sync()
            rec["scoring_seconds"] = time.perf_counter() - t0
            rec["c1_split_vs_full_max_abs_logit"] = float((split_last - full_last).abs().max())
            gen_ref = model.generate(ids, max_new_tokens=kv.MAX_NEW, do_sample=False, pad_token_id=tok.eos_token_id)
            gen_ref = gen_ref[0, ids.shape[1]:].tolist()
            ref_logits = None
            for meth in kv.METHODS:
                _sync(); torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                comp = kv.compress(meth, ctxK, ctxV, scores, cfg, case["evidence"])
                _sync()
                t_comp = time.perf_counter() - t0
                r = kv.run_policy(model, tok, case, comp, ref_logits)
                peak = int(torch.cuda.max_memory_allocated())
                tf_lp = r.pop("tf_logprobs")
                if meth == "fp16":
                    ref_logits = tf_lp
                    r["first_token_logit_kl"] = 0.0
                    r["teacher_forced_kl_mean"] = 0.0
                    r["c1_matches_ordinary_generate"] = r["gen_ids"] == gen_ref[: len(r["gen_ids"])]
                    first_p = tf_lp[len(case["suffix"]) - 1].exp()
                    r["c1_first_probs_vs_split_max_abs"] = float((first_p - torch.softmax(split_last, -1)).abs().max())
                meta = {k: v for k, v in comp["meta"].items() if not k.startswith("keep_mask") or case["id"] in ALIGNED}
                r.update(meta=meta, compress_seconds=t_comp, peak_alloc_bytes=peak)
                if case["id"] in ALIGNED:
                    aligned.setdefault(case["id"], dict(evidence=raw["evidence_text"], evidence_tokens=case["evidence"],
                                                        L_ctx=L, code=case["code"], methods={}))
                    aligned[case["id"]]["methods"][meth] = dict(text=r["text"], pred=r["pred"],
                                                                **{k: v for k, v in comp["meta"].items() if k.startswith("keep_mask")})
                    r["meta"] = {k: v for k, v in r["meta"].items() if not k.startswith("keep_mask")}
                rec["methods"][meth] = r
                del comp
            if aligned:
                write_json(od / "aligned.json", aligned)  # written before the case record -> never missing on resume
            log.append(rec)
            print(f"[{ci+1}/{len(cases)}] {case['id']} L={L} " +
                  " ".join(f"{m}:{int(rec['methods'][m]['correct'])}" for m in kv.METHODS), flush=True)
            del ctxK, ctxV, scores
            gc.collect(); torch.cuda.empty_cache()


@torch.no_grad()
def cmd_bench(reps=10):
    model, tok, rev = kv.load()
    cfg = model.config
    dev = next(model.parameters()).device
    nl, H, d = kv.kv_dims(cfg)
    cases = D.kv_cases()
    log = JsonlAppender(out_dir("task2", "kv") / "c5_bench_rows.jsonl",
                        key_fn=lambda r: (r["target_tokens"], r["batch"], r["layout"], r["bits"]))
    rows = log.records
    for target in (2048, 4096):
        if all((target, b, lay, bt) in log.done for b in (1, 8) for lay, bt in (("fp16", 16), ("kivi", 4), ("token", 4), ("kivi", 2))):
            continue
        raw = next(c for c in cases if c["target_tokens"] == target)
        case = kv.prepare_case(tok, raw)
        L = case["L_ctx"]
        pre = model(input_ids=torch.tensor([case["ids"][:L]], device=dev), use_cache=True)
        K1, V1 = kv.cache_tensors(pre.past_key_values)
        del pre
        for bsz in (1, 8):
            if all((target, bsz, lay, bt) in log.done for lay, bt in (("fp16", 16), ("kivi", 4), ("token", 4), ("kivi", 2))):
                continue
            K = [k.expand(bsz, -1, -1, -1).contiguous() for k in K1]
            V = [v.expand(bsz, -1, -1, -1).contiguous() for v in V1]
            nxt = torch.full((bsz, 1), case["suffix"][0], device=dev)
            pos = torch.full((bsz, 1), L, device=dev)

            def step_fp16():
                c = kv.build_cache(K, V)
                model(input_ids=nxt, past_key_values=c, position_ids=pos, use_cache=True)

            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t = time_fn(step_fp16, reps, 3)
            ref = dict(target_tokens=target, context_tokens=L, batch=bsz, layout="fp16", bits=16, unit="ms", reps=reps,
                       decode_step_ms=1000 * t["mean"], decode_step_ms_std=1000 * t["std"],
                       serialized_bytes=kv.m16(L, cfg) * bsz, peak_alloc_bytes=int(torch.cuda.max_memory_allocated()))
            if (target, bsz, "fp16", 16) not in log.done:
                log.append(ref)
            for layout, bits in (("kivi", 4), ("token", 4), ("kivi", 2)):
                if (target, bsz, layout, bits) in log.done:
                    continue
                enc = [kv_pack.encode_layer(K[l], V[l], bits, layout) for l in range(nl)]
                tp = time_fn(lambda: [kv_pack.encode_layer(K[l], V[l], bits, layout) for l in range(nl)], reps, 3)
                tu = time_fn(lambda: [kv_pack.decode_layer(enc[l], K[l].shape) for l in range(nl)], reps, 3)

                def step_unpack():
                    dec = [kv_pack.decode_layer(enc[l], K[l].shape) for l in range(nl)]
                    c = kv.build_cache([x[0] for x in dec], [x[1] for x in dec])
                    model(input_ids=nxt, past_key_values=c, position_ids=pos, use_cache=True)

                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                ts = time_fn(step_unpack, reps, 3)
                sb = kv_pack.serialized_bytes(enc)
                log.append(dict(target_tokens=target, context_tokens=L, batch=bsz, layout=layout, bits=bits, unit="ms", reps=reps,
                                 pack_ms=1000 * tp["mean"], pack_ms_std=1000 * tp["std"],
                                 unpack_ms=1000 * tu["mean"], unpack_ms_std=1000 * tu["std"],
                                 decode_step_with_unpack_ms=1000 * ts["mean"], decode_step_with_unpack_ms_std=1000 * ts["std"],
                                 serialized_bytes=sb, formula_bytes=kv.mb(L, bits, cfg) * bsz,
                                 bytes_match_formula=sb == kv.mb(L, bits, cfg) * bsz,
                                 peak_alloc_bytes=int(torch.cuda.max_memory_allocated())))
                print(log.records[-1], flush=True)
                del enc
            del K, V
            gc.collect(); torch.cuda.empty_cache()
    write_json(out_dir("task2", "kv") / "c5_bench.json", dict(rows=rows, env=environment_record(dict(model=kv.MODEL, model_revision=rev, precision="float16")),
               note="decode step = build DynamicCache from (unpacked) FP16 tensors + one forward of 1 token"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "bench", "all"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    if a.cmd in ("eval", "all"):
        cmd_eval(a.limit)
    if a.cmd in ("bench", "all"):
        cmd_bench(a.reps)


if __name__ == "__main__":
    main()

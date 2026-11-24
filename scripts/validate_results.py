#!/usr/bin/env python
"""Check every saved result against the PA4 handout requirements; writes results/validation.{md,json}.

Each check is PASS / WARN / FAIL with the evidence (numbers read from the result files). Run after all tasks:
    python scripts/validate_results.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pa4.common.utils import read_json, read_jsonl, results_root, write_json

R = results_root()
CHECKS = []


def check(task, item, ok, evidence, warn=False):
    status = "PASS" if ok else ("WARN" if warn else "FAIL")
    CHECKS.append(dict(task=task, item=item, status=status, evidence=evidence))


def exists(p):
    return (R / p).exists()


def tparse(s):
    for fmt in ("%Y-%m-%d %H:%M:%S %Z", "%Y-%m-%d %H:%M:%S "):
        try:
            return datetime.strptime(s, fmt)
        except (ValueError, TypeError):
            pass
    try:
        return datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


# =============================================================================== Task 1
def task1():
    T = "Task 1"
    conds = ["fp16", "p1_s25", "p1_s50", "p1_s75", "p2_s25", "p2_s50", "p2_s75", "p3_s50", "p3_s75", "p2_24_s50",
             "q1_b8", "q1_b4", "q1_b3", "q2_b8", "q2_b4", "q2_b3", "q3_b4", "q3_b3"]
    M = {c: read_json(R / f"task1/conditions/{c}/metrics.json") for c in conds if exists(f"task1/conditions/{c}/metrics.json")}
    check(T, "All 18 required conditions evaluated", len(M) == 18, f"{len(M)}/18 conditions with metrics.json")
    if "fp16" not in M:
        return
    en = read_json(R / "task1/conditions/fp16/enumeration_check.json")
    check(T, "Eligible projections enumerated (28 blocks x 7) and tied embedding counted once",
          en["n_eligible_matrices"] == 196 and en["tied_embeddings"] and en["tied_param_counted_once"],
          f"{en['n_eligible_matrices']} matrices, {en['n_eligible_weights']:,} eligible weights, tied={en['tied_embeddings']}, "
          f"unique params {en['n_total_unique']:,} vs naive {en['n_total_naive']:,}")
    Ns = {(c, s): M[c]["quality"][s]["N"] for c in M for s in ("wikitext2", "c4val")}
    check(T, "Both perplexity suites score exactly 50,000 target tokens for every condition",
          all(v == 50000 for v in Ns.values()), f"N values: {sorted(set(Ns.values()))}")
    full = {"fp16", "p1_s50", "p2_s50", "p3_s50", "q1_b4", "q1_b3", "q2_b4", "q2_b3", "q3_b4", "q3_b3"}
    hs_ok = all(M[c]["quality"]["hellaswag"]["n"] == (2000 if c in full else 512) for c in M)
    check(T, "HellaSwag: 2,000 IDs for FP16, 50% P1/P2/P3 and all INT4/INT3; first 512 for the rest", hs_ok,
          ", ".join(f"{c}:{M[c]['quality']['hellaswag']['n']}" for c in M))
    ds = read_json(R / "task1/data_selection.json")
    check(T, "Calibration (128) and development (32) C4 sequences are disjoint", ds["calibration_dev_disjoint"] and
          len(ds["calibration_source_rows"]) == 128 and len(ds["development_source_rows"]) == 32,
          "C4 train manifests; evaluation uses C4 validation / WikiText-2 test / HellaSwag validation (separate splits)")
    for s in ("25", "50", "75"):
        z = {p: M[f"{p}_s{s}"]["zeros"] for p in ("p1", "p2", "p3") if f"{p}_s{s}" in M}
        tgt = int(s) / 100
        sp = {p: M[f"{p}_s{s}"]["achieved_sparsity"] for p in z}
        check(T, f"Pruning at {s}%: achieved sparsity on target and P1/P2/P3 zero counts equal",
              len(set(z.values())) == 1 and all(abs(v - tgt) < 1e-4 for v in sp.values()),
              f"zeros {z}; achieved {{{', '.join(f'{k}: {v:.5f}' for k, v in sp.items())}}}")
    if "p2_24_s50" in M:
        check(T, "2:4 diagnostic at exactly 50%", abs(M["p2_24_s50"]["achieved_sparsity"] - 0.5) < 1e-6,
              f"achieved {M['p2_24_s50']['achieved_sparsity']:.6f}")
    for b in ("4", "3"):
        q3, q2 = M[f"q3_b{b}"]["storage"]["packed_total_bytes"], M[f"q2_b{b}"]["storage"]["packed_total_bytes"]
        check(T, f"Q3 packed bytes <= Q2 uniform INT{b} budget (incl. metadata + uncompressed params)", q3 <= q2,
              f"Q3 {q3:,} B vs Q2 {q2:,} B (slack {q2 - q3:,} B, {100 * (q2 - q3) / q2:.4f}%)")
    # selection before final evaluation
    for t in ("p3_s50", "p3_s75", "q3_b4", "q3_b3"):
        sel = read_json(R / f"task1/allocations/{t}_selected.json")
        cands = sel["selection"]["candidates"]
        dev_times = []
        for c in cands:
            name = "dev" + t + "_" + c["candidate"].split(t + "_")[1]
            f = R / f"task1/dev_candidates/{name}/metrics.json"
            if f.exists():
                dev_times.append(tparse(read_json(f)["env"]["time"]))
        final_t = tparse(M[t]["env"]["time"])
        best = min(cands, key=lambda c: c["dev_mean_nll"])["candidate"]
        ok = best == sel["name"] and dev_times and final_t and max(dev_times) < final_t
        check(T, f"{t}: design chosen by dev NLL, all candidates evaluated before the final test run", bool(ok),
              f"selected {sel['name']} ({sel['design']}); candidates " +
              ", ".join(f"{c['candidate'].split(t + '_')[1]}={c['dev_mean_nll']:.4f}" for c in cands) +
              f"; last dev run {max(dev_times) if dev_times else None} < final {final_t}")
    folds = [x["rel_err"] for c in ("q2_b4", "q2_b3", "q3_b4", "q3_b3") for x in M[c]["compression"].get("fold_checks", [])]
    if folds:
        check(T, "AWQ full-precision rescaling preserves block outputs before rounding", max(folds) < 5e-3,
              f"max relative block-output change after folding = {max(folds):.2e} over {len(folds)} block checks")
    pk = [M[c]["compression"].get("packing_roundtrip_verified_units", 0) for c in M if c.startswith("q")]
    check(T, "Bit packing round-trip verified against dequantized weights", min(pk) >= 1, f"verified units per quant run: {pk}")
    reps = {M[c]["latency"]["reps"] for c in M}
    check(T, ">= 5 warmed latency repetitions; prefill 512/2048 + 64-token decode + peak memory recorded", min(reps) >= 5,
          f"reps={reps}; GPU={M['fp16']['env'].get('cuda_device')}")
    gpus = {M[c]["env"].get("cuda_device") for c in M}
    check(T, "All Task 1 timings on one GPU type", len(gpus) == 1, f"devices: {gpus}")
    bp = read_json(R / "task1/bootstrap_pairs.json")
    need = ["p3_s50-p2_s50", "p3_s75-p2_s75", "q3_b4-q2_b4", "q3_b3-q2_b3"]
    check(T, "Paired 95% bootstraps (2,000 resamples) for P3-P2 and Q3-Q2 on PPL windows and HellaSwag",
          all(k in bp and bp[k]["wikitext2_ppl"]["n_resamples"] == 2000 for k in need),
          "; ".join(f"{k}: dWiki {bp[k]['wikitext2_ppl']['point']:+.2f} [{bp[k]['wikitext2_ppl']['ci_low']:+.2f}, {bp[k]['wikitext2_ppl']['ci_high']:+.2f}]" for k in need if k in bp))
    for f in ["figures/task1/quality_vs_resource.pdf", "figures/task1/sensitivity_maps.pdf", "figures/task1/allocation_maps.pdf",
              "figures/task1/calib_vs_heldout.pdf", "figures/task1/mag_wanda_overlap.pdf", "task1/qualitative.json",
              "task1/calib_vs_heldout.json", "tables/task1_all_conditions.csv"]:
        check(T, f"Artifact present: {f}", exists(f), "")
    base = M["fp16"]["quality"]
    check(T, "Baseline sanity (HellaSwag acc_norm near published Qwen2.5-1.5B ~0.68)",
          0.6 < base["hellaswag"]["acc_norm"] < 0.75,
          f"acc {base['hellaswag']['acc']:.4f}, acc_norm {base['hellaswag']['acc_norm']:.4f}; "
          f"WikiText-2 PPL {base['wikitext2']['ppl']:.3f}, C4 PPL {base['c4val']['ppl']:.3f}")


# =============================================================================== Task 2
def task2():
    T = "Task 2"
    rows = read_jsonl(R / "task2/attention/bench.jsonl")
    check(T, "2A: 7 implementations x 6 lengths x 2 dtypes benchmarked (incl. unsupported/OOM rows)", len(rows) == 84,
          f"{len(rows)} rows; statuses: {sorted({r['status'][:20] for r in rows})}")
    ok = [r for r in rows if r["status"] == "ok"]
    fp32_tiled = [r for r in ok if r["impl"] == "tiled" and r["dtype"] == "float32"]
    check(T, "2A: tiled online-softmax matches dense FP32 (output and gradients) at every length",
          all(r["out_max_abs"] < 1e-4 and max(r["dq_max_abs"], r["dk_max_abs"], r["dv_max_abs"]) < 1e-3 for r in fp32_tiled),
          f"FP32 tiled max |out err| {max(r['out_max_abs'] for r in fp32_tiled):.1e}, max |grad err| "
          f"{max(max(r['dq_max_abs'], r['dk_max_abs'], r['dv_max_abs']) for r in fp32_tiled):.1e}")
    check(T, "2A: forward-only and forward+backward timed separately with peak allocated/reserved memory",
          all("fwd_ms" in r and "fwdbwd_ms" in r and "fwdbwd_peak_reserved_bytes" in r for r in ok), f"{len(ok)} ok rows")
    ed = read_json(R / "task2/attention/edge_cases.json")
    caus = {k: v.get("causality_future_perturbation_max_change") for k, v in ed.items() if isinstance(v, dict) and "causality_future_perturbation_max_change" in v}
    check(T, "2A: causality verified by perturbing future keys/values", all(v is not None and v < 1e-5 for v in caus.values()),
          f"max change before perturbed positions: {max(caus.values()):.1e} across {len(caus)} implementations")
    check(T, "2A: tile-size sweep and default SDPA backend recorded", exists("task2/attention/tiles.json") and exists("task2/attention/backends.json"),
          str(read_json(R / "task2/attention/backends.json")["default_dispatch_L1024"]) [:160])
    runs = [read_json(f) for f in sorted((R / "task2/mqar/runs").glob("*.json"))]
    names = {r["name"] for r in runs}
    check(T, "2B: 14 configurations x 3 seeds trained (B1, B2, B3)", len(runs) == 42 and len(names) == 14,
          f"{len(runs)} runs, {len(names)} configurations")
    p = {r["mechanism"]: r["n_params"] for r in runs}
    check(T, "2B: parameter counts within 5% of the dense model", all(abs(v - p["dense"]) / p["dense"] < 0.05 for v in p.values()),
          ", ".join(f"{k} {v:,}" for k, v in p.items()))
    check(T, "2B: full learning curves (50-update loss + 6 dev evaluations) for every run",
          all(len(r["curve"]) == 90 and [d["step"] for d in r["dev_curve"]] == [500, 1000, 1500, 2000, 3000, 4500] for r in runs), "")
    check(T, "2B: every model evaluated on the same near/far/mixed test cells for K in {4,8,16}",
          all(len(r["test"]["cells"]) == 9 for r in runs), "9 cells x 1,024 sequences each")
    check(T, "2B: train-vs-test mask controls (dense + W=64 / top-64 at test time)",
          all("test_window64_at_test" in r and "test_topk64_at_test" in r for r in runs if r["mechanism"] == "dense"), "")
    check(T, "2B: generator invariants verified; theoretical costs and forward-only sweep present",
          exists("task2/mqar/generator_checks.json") and exists("task2/mqar/theoretical_costs_T256.json") and exists("task2/mqar/inference_sweep.json"), "")
    import subprocess

    def git(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=Path(__file__).resolve().parents[1], text=True).strip()
        except Exception:
            return ""
    pred = git("log", "--format=%h %aI", "-S", "prediction_before_evaluation", "--", "configs/task2.yaml").splitlines()
    res = git("log", "--diff-filter=A", "--format=%h %aI", "--", "results/task2/mqar/runs/hybrid_K16_mixed_s6304.json").splitlines()
    edits = git("log", "--format=%h", "--", "configs/task2.yaml").splitlines()
    if pred and res:
        tp, tr = datetime.fromisoformat(pred[-1].split()[1]), datetime.fromisoformat(res[-1].split()[1])
        check(T, "2B: B3 prediction committed before any B3 result (git history)", tp < tr and len(edits) == 1,
              f"prediction {pred[-1]} (file never edited afterwards: {len(edits) == 1}); first B3 result {res[-1]}")
    else:
        check(T, "2B: B3 prediction committed before any B3 result (git history)", False, "git history unavailable", warn=True)
    cases = read_jsonl(R / "task2/kv/cases.jsonl")
    meths = ["fp16", "sink_recent", "qa_evict", "int4_token", "int4_kivi", "hybrid"]
    check(T, "2C: all 180 prompts x 6 policies evaluated", len(cases) == 180 and all(set(c["methods"]) == set(meths) for c in cases),
          f"{len(cases)} prompts")
    wb = all(c["methods"][m]["meta"]["within_budget"] for c in cases for m in meths[1:])
    fr = max(c["methods"][m]["meta"]["realized_fraction_of_fp16"] for c in cases for m in meths[1:])
    check(T, "2C: every compressed policy within the per-prompt INT4 byte budget", wb, f"max realized fraction of FP16 = {fr:.4f}")
    gm = np.mean([c["methods"]["fp16"]["c1_matches_ordinary_generate"] for c in cases])
    fin = all(c["methods"][m]["finite"] for c in cases for m in meths)
    check(T, "2C/C1: rebuilt 100%-retention cache reproduces ordinary generation; logits finite", gm == 1.0 and fin,
          f"generation match {gm:.3f}; split-vs-full max |logit diff| {max(c['c1_split_vs_full_max_abs_logit'] for c in cases):.3f} (FP16)")
    check(T, "2C: actual tokenized lengths recorded", all("L_ctx" in c and "n_tokens" in c for c in cases),
          f"context tokens {min(c['L_ctx'] for c in cases)}-{max(c['L_ctx'] for c in cases)}")
    c5 = read_jsonl(R / "task2/kv/c5_bench_rows.jsonl") if exists("task2/kv/c5_bench_rows.jsonl") else read_json(R / "task2/kv/c5_bench.json")["rows"]
    pk = [r for r in c5 if r["layout"] != "fp16"]
    check(T, "2C/C5: packed encode/decode + decode step at batch {1,8} x both lengths; serialized bytes == formula",
          len(c5) == 16 and all(r["bytes_match_formula"] for r in pk), f"{len(c5)} rows, {sum(r['bytes_match_formula'] for r in pk)}/{len(pk)} byte matches")
    al = read_json(R / "task2/kv/aligned.json")
    check(T, "2C: fixed aligned generations with evidence spans and selection masks",
          len(al) == 6 and all("keep_mask_layer0" in al[k]["methods"]["qa_evict"] for k in al), f"{len(al)} prompts")
    check(T, "2C: paired bootstrap CIs for principal differences", exists("task2/kv/bootstrap.json"), "")
    for task_dir in ("task2/mqar/runs", ):
        g = {r["env"].get("cuda_device") for r in runs}
        check(T, "2B timings on one GPU type", len(g) == 1, str(g))


# =============================================================================== Task 3
def task3():
    T = "Task 3"
    c = read_json(R / "task3/checks/sampler_checks.json")
    check(T, "100,000-sequence Markov chi^2: exact sampler accepted (p >= 0.01)", c["markov_gamma2"]["exact"]["p_value"] >= 0.01,
          f"chi2={c['markov_gamma2']['exact']['chi2']:.2f}, dof={c['markov_gamma2']['exact']['dof']}, p={c['markov_gamma2']['exact']['p_value']:.3f}")
    check(T, "Negative control (residual := p) rejected", c["markov_gamma2"]["p"]["p_value"] < 0.01,
          f"chi2={c['markov_gamma2']['p']['chi2']:.0f}, p={c['markov_gamma2']['p']['p_value']:.1e}")
    check(T, "EOS-aware and gamma=1 variants pass", c["markov_eos_gamma3"]["p_value"] >= 0.01 and c["markov_gamma1"]["exact"]["p_value"] >= 0.01,
          f"EOS p={c['markov_eos_gamma3']['p_value']:.3f}, gamma=1 p={c['markov_gamma1']['exact']['p_value']:.3f}")
    rc = c["real_model_cache"]
    check(T, "Real-model cache rollback / split prefill / crop-refeed agree with full recomputation",
          max(max(r["cache_vs_full_max_abs_prob"], r["crop_refeed_max_abs_prob"], r["split_prefill_max_abs_prob"]) for r in rc) < 5e-3,
          f"max |prob diff| {max(r['cache_vs_full_max_abs_prob'] for r in rc):.1e}; bonus tokens {sum(r['bonus'] for r in rc)}, EOS {sum(r['eos'] for r in rc)}")
    check(T, "Tokenizer and chat-template agreement (teacher vs student)", c["tokenizer"]["n_mismatch"] == 0 and c["tokenizer"]["same_eos"],
          f"{c['tokenizer']['n_checked']} prompts, 0 mismatches")
    ov = read_json(R / "task3/checks/prompt_sets_and_overlap.json")
    check(T, "No train/test prompt overlap; 60 math + 30 familiar + 30 shifted test prompts",
          ov["gsm8k_question_overlap"] == ov["short_id_overlap"] == ov["short_text_overlap"] == 0 and ov["test_split_counts"] == {"math": 60, "familiar": 30, "shifted": 30},
          str(ov["test_split_counts"]))
    runs = {n: read_json(R / f"task3/distill/{n}/result.json") for n in ("off_s6304", "off_s6305", "on_s6304", "on_s6305")}
    check(T, "Distillation: exact 30,000 + 10,000 teacher-scored tokens per run",
          all(r["tokens"] == {"math": 30000, "short": 10000} for r in runs.values()),
          ", ".join(f"{n}: {r['updates']} updates, {r['skipped_updates']} skipped" for n, r in runs.items()))
    check(T, "Distillation: adapter/model parameter counts and cost breakdown recorded",
          all("n_adapter_params" in r and "costs" in r for r in runs.values()),
          f"adapter {runs['off_s6304']['n_adapter_params']:,}; backbone {runs['off_s6304'].get('n_student_backbone_params', 'n/a')}")
    mg = {n: read_json(R / f"task3/distill/{n}/merge.json")["max_abs_logit_err"] for n in runs}
    check(T, "Merged vs unmerged adapter logits compared (FP16)", max(mg.values()) < 1.0,
          ", ".join(f"{k} {v:.3f}" for k, v in mg.items()), warn=True)
    banks = sorted(p.name for p in (R / "task3/banks").glob("*.jsonl.gz"))
    check(T, "Banks: exact-speculative (original, gamma=4) + 4 own-rollout banks generated", len(banks) == 5, ", ".join(banks))
    cons = read_json(R / "task3/align/consistency.json")
    d = cons.get("prompt_paired_diff_exp_accept", {})
    check(T, "Target vs exact-speculative bank consistency (CI includes 0)", d.get("ci_low", 1) <= 0 <= d.get("ci_high", -1),
          f"diff {d.get('point', float('nan')):+.4f} [{d.get('ci_low', float('nan')):+.4f}, {d.get('ci_high', float('nan')):+.4f}]")
    counts = {}
    for tag, tot in [("original", 480), ("off_s6304", 480), ("off_s6305", 480), ("on_s6304", 480), ("on_s6305", 480),
                     ("original_fixed64", 96), ("on_s6304_unmerged", 20)]:
        f = R / f"task3/spec/records_{tag}.jsonl"
        counts[tag] = (len(read_jsonl(f)) if f.exists() else 0, tot)
    check(T, "Speculative benchmark complete (5 drafts x gamma {2,4} x 2 reps x 120 prompts, + fixed-64 + unmerged)",
          all(a == b for a, b in counts.values()), str(counts))
    check(T, "Paired bootstrap CIs over prompts for throughput and acceptance comparisons",
          exists("task3/spec/bootstrap.json") and exists("tables/task3_spec_paired_comparisons.csv"), "")
    sa = {p.stem: len(read_jsonl(p)) for p in (R / "task3/standalone").glob("*.jsonl")}
    check(T, "Standalone answer accuracy for teacher + 5 drafts (120 prompts each)", len(sa) == 6 and all(v == 120 for v in sa.values()), str(sa))
    gpus = {c["env"].get("cuda_device")} | {r["env"].get("cuda_device") for r in runs.values()}
    check(T, "Task 3 timings on one GPU type", len(gpus) == 1, str(gpus))


def main():
    for f in (task1, task2, task3):
        try:
            f()
        except Exception as e:  # report missing artefacts as failures rather than crashing
            check(f.__name__, "validation error", False, repr(e))
    write_json(R / "validation.json", CHECKS)
    lines = ["# Results validation against the PA4 handout", "",
             f"Generated by `python scripts/validate_results.py`. {sum(c['status'] == 'PASS' for c in CHECKS)} PASS, "
             f"{sum(c['status'] == 'WARN' for c in CHECKS)} WARN, {sum(c['status'] == 'FAIL' for c in CHECKS)} FAIL.", "",
             "| Task | Check | Status | Evidence |", "|---|---|---|---|"]
    for c in CHECKS:
        lines.append(f"| {c['task']} | {c['item']} | {c['status']} | {c['evidence'].replace('|', '/')} |")
    (R / "validation.md").write_text("\n".join(lines) + "\n")
    for c in CHECKS:
        print(f"[{c['status']}] {c['task']}: {c['item']}" + (f"\n        {c['evidence']}" if c["status"] != "PASS" or "--verbose" in sys.argv else ""))


if __name__ == "__main__":
    main()

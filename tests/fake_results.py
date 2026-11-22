"""Write a synthetic results tree with the drivers' schemas so the analysis scripts can be smoke-tested.

Usage: PA4_RESULTS=/tmp/fake python tests/fake_results.py && PA4_RESULTS=/tmp/fake python analysis/task1_report.py ...
Numbers are random; never use them in a report.
"""
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from pa4.common.utils import results_root, write_json, write_jsonl

rng = np.random.default_rng(0)
R = results_root()
PROJ = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
NL = 4


def task1():
    conds = ["fp16", "p1_s25", "p1_s50", "p1_s75", "p2_s25", "p2_s50", "p2_s75", "p3_s50", "p3_s75", "p2_24_s50",
             "q1_b8", "q1_b4", "q1_b3", "q2_b8", "q2_b4", "q2_b3", "q3_b4", "q3_b3"]
    for c in conds:
        d = R / "task1" / "conditions" / c
        nw = 100
        for k in ("wikitext2", "c4val"):
            write_json(d / f"ppl_{k}.json", dict(nll=(rng.random(nw) * 1000 + 1000).tolist(), ntok=[500] * nw))
        n = 2000 if c in ("fp16", "p2_s50", "p3_s50", "q2_b4") else 512
        write_jsonl(d / "hellaswag.jsonl", [dict(order=i, source_row_index=i, gold=int(rng.integers(4)), ll=rng.random(4).tolist(),
                                                 char_len=[10, 11, 12, 13], pred=int(rng.integers(4)), pred_norm=int(rng.integers(4)))
                                            for i in range(n)])
        sp = float(c.split("_s")[-1]) / 100 if c.startswith("p") else 0.0
        q = dict(wikitext2=dict(N=50000, mean_nll=2.2 + sp, ppl=float(np.exp(2.2 + sp)), n_windows=nw),
                 c4val=dict(N=50000, mean_nll=2.6 + sp, ppl=float(np.exp(2.6 + sp)), n_windows=nw),
                 hellaswag=dict(n=n, acc=0.5, acc_norm=0.6), hellaswag_diag512=dict(n=512, acc=0.5, acc_norm=0.6),
                 continuations=[dict(prompt="a", continuation="b")])
        st = dict(dense_fp16_total_bytes=3.0e9, sparse16_total_bytes=2.5e9, nominal_bits_per_eligible_weight_sparse16=9.0)
        if c.startswith("q"):
            b = int(c[-1])
            st.update(packed_total_bytes=1e9 + b * 1e8, nominal_bits_per_eligible_weight_packed=b + 0.2)
        lat = {"prefill_512": dict(mean=0.02), "prefill_2048": dict(mean=0.08, peak_allocated_bytes=4e9),
               "decode_64": dict(per_token_ms=20.0)}
        write_json(d / "metrics.json", dict(condition=c, achieved_sparsity=sp, zeros=int(sp * 1e6), quality=q, storage=st,
                                            latency=lat, dev=dict(mean_nll=2.5 + sp), calib_out_err_rel_mean=None if c == "fp16" else 0.01 + sp,
                                            compression=dict(compression_seconds_total=10.0)))
    units = {}
    for l in range(NL):
        for p in PROJ:
            units[f"layers.{l}.{p}"] = dict(layer=l, proj=p, prune_fisher=list(rng.random(31)), quant={str(b): dict(fisher=float(rng.random()), out_err=float(rng.random())) for b in (2, 3, 4, 8)},
                                            out_energy=1.0, mag_wanda_jaccard={"0.25": 0.5, "0.5": 0.6, "0.75": 0.7})
    write_json(R / "task1" / "sensitivity" / "probe.json", dict(units=units, prune_grid=[round(0.1 + 0.025 * i, 3) for i in range(31)], owl_M=[3, 5],
                                                              blocks={str(l): dict(owl={"3": 0.01, "5": 0.002}) for l in range(NL)}))
    for t, key in (("p3_s50", "unit_sparsity"), ("p3_s75", "unit_sparsity"), ("q3_b4", "bits"), ("q3_b3", "bits")):
        write_json(R / "task1" / "allocations" / f"{t}_selected.json", {key: {u: float(rng.random()) for u in units}, "design": "fisher_knapsack"})


def task2():
    rows = []
    for impl in ("dense", "tiled", "sdpa_default", "sdpa_flash_attention"):
        for dt in ("float32", "float16"):
            for L in (256, 1024, 4096):
                rows.append(dict(impl=impl, dtype=dt, L=L, status="ok" if not (impl == "sdpa_flash_attention" and dt == "float32") else "unsupported",
                                 fwd_ms=1.0 * L / 256, fwdbwd_ms=3.0 * L / 256, fwd_peak_alloc_bytes=1e6 * L, fwdbwd_peak_alloc_bytes=3e6 * L,
                                 out_max_abs=1e-4, dq_max_abs=1e-4, dk_max_abs=1e-4, dv_max_abs=1e-4, reference="dense_fp32"))
    write_jsonl(R / "task2" / "attention" / "bench.jsonl", rows)
    bins = ["[0,32)", "[32,64)", "[64,128)", "[128,256)"]
    for name, mech, K, reg in (("dense_K16_mixed", "dense", 16, "mixed"), ("window_K16_near", "window", 16, "near"), ("linear_K16_mixed", "linear", 16, "mixed")):
        for s in (6304, 6305, 6306):
            ev = lambda: dict(cells={f"K{k}_{g}": dict(acc=float(rng.random())) for k in (4, 8, 16) for g in ("near", "far", "mixed")},
                              heat={f"K{k}_{b}": dict(acc=float(rng.random())) for k in (4, 8, 16) for b in bins})
            r = dict(name=name, mechanism=mech, train_K=K, train_regime=reg, seed=s, n_params=400000, ms_per_step=5.0,
                     peak_train_alloc_bytes=1e8, curve=[dict(step=i * 50, loss=1 / (i + 1)) for i in range(1, 91)],
                     dev_curve=[dict(step=st, acc=float(rng.random())) for st in (500, 1000, 1500, 2000, 3000, 4500)], test=ev())
            if mech == "dense":
                r["test_window64_at_test"] = ev(); r["test_topk64_at_test"] = ev()
            write_json(R / "task2" / "mqar" / "runs" / f"{name}_s{s}.json", r)
    write_json(R / "task2" / "mqar" / "theoretical_costs_T256.json", dict(dense=dict(pairs_per_head=32896.0, state_floats_per_layer=1.0)))
    write_json(R / "task2" / "mqar" / "inference_sweep.json", dict(rows=[dict(T=T, mechanism=m, ms=1.0 * T, peak_alloc_bytes=1e5 * T, status="ok")
                                                                         for T in (256, 1024) for m in ("dense", "linear")]))
    M = ["fp16", "sink_recent", "qa_evict", "int4_token", "int4_kivi", "hybrid"]
    cases = []
    for i in range(36):
        cases.append(dict(id=f"c{i}", target_tokens=[2048, 4096][i % 2], position=["start", "middle", "end"][i % 3], distractors=[0, 2, 8][i % 3],
                          L_ctx=2000, prefill_seconds=0.1, scoring_seconds=0.02, prefill_peak_alloc_bytes=4e9, c1_split_vs_full_max_abs_logit=0.01,
                          methods={m: dict(correct=bool(rng.random() > 0.3), code_nll=float(rng.random()), first_token_logit_kl=float(rng.random()), teacher_forced_kl_mean=0.1, shared_decode_bytes=5e4,
                                           meta=dict(bytes=1e6, realized_fraction_of_fp16=0.28, within_budget=True, evidence_fp16_retained=0.5),
                                           compress_seconds=0.01, t_suffix=0.02, t_decode=0.1, peak_alloc_bytes=4e9, live_cache_bytes_end=1e6,
                                           finite=True, c1_matches_ordinary_generate=True) for m in M}))
    write_jsonl(R / "task2" / "kv" / "cases.jsonl", cases)


def task3():
    t = [dict(seq="000", exact_prob=0.1, empirical=0.1)]
    write_json(R / "task3" / "checks" / "sampler_checks.json", dict(
        markov_gamma2=dict(exact=dict(chi2=20, p_value=0.5, table=t), p=dict(chi2=2000, p_value=0.0, table=t)),
        markov_gamma1=dict(exact=dict(chi2=20, p_value=0.5)), markov_eos_gamma3=dict(chi2=10, p_value=0.4)))
    for n in ("off_s6304", "off_s6305", "on_s6304", "on_s6305"):
        write_json(R / "task3" / "distill" / n / "result.json", dict(
            logs=[dict(scored_tokens=256 * i, kl_per_token=1 / i) for i in range(1, 157)], updates=157, skipped_updates=1,
            n_adapter_params=1081344, tokens=dict(math=30000, short=10000), n_presentations=3000,
            costs=dict(student_rollout_s=100.0, teacher_scoring_s=50.0, student_fwd_bwd_s=80.0, optimizer_s=5.0), wall_s=300.0,
            peak_alloc_bytes=6e9, rollout_lengths=dict(math=[150, 120], short=[3, 2])))
        write_json(R / "task3" / "distill" / n / "merge.json", dict(merge_seconds=3.0, max_abs_logit_err=0.01))
    write_json(R / "task3" / "distill" / "teacher_pool_cost.json", dict(seconds=600.0))
    rows = []
    for bank in ("teacher", "original_draft", "exact_spec_original_gamma4", "own_on_s6304"):
        for pid, split in (("GSM8K-0000", "math"), ("SHORT-familiar-0000", "familiar")):
            for pos in (1, 20, 40, 100):
                for d in ("original", "off_s6304", "off_s6305", "on_s6304", "on_s6305"):
                    tv = float(rng.random() * 0.5)
                    rows.append(dict(bank=bank, prompt_id=pid, split=split, position=pos, pos_bin="[1,16]" if pos < 17 else "[17,32]" if pos < 33 else "[33,64]" if pos < 65 else "[65,192]",
                                     draft=d, fkl=tv, rkl=tv, tv=tv, exp_accept=1 - tv, greedy_agree=1.0, target_logp_sampled=-1.0))
    write_jsonl(R / "task3" / "align" / "positions.jsonl.gz", rows)
    recs = []
    for d in ("original", "off_s6304", "off_s6305", "on_s6304", "on_s6305"):
        for g in (2, 4):
            for rep in (0, 1):
                for i, split in enumerate(("math", "familiar", "shifted") * 4):
                    sp = dict(accepted=30, examined=40, proposed=45, rounds=15, n_out=48, target_calls=15, draft_calls=45,
                              tokens_per_s=40.0 + rng.random(), eos=True, truncated=False, peak_alloc_bytes=5e9,
                              t_prefill=0.02, t_draft=0.4, t_verify=0.4, t_accept=0.05, t_rollback=0.01, t_host_other=0.02, t_total=0.9, out_ids=[1, 2])
                    tg = dict(tokens_per_s=35.0 + rng.random(), peak_alloc_bytes=4e9, t_decode=1.0, target_calls=48, out_ids=[3])
                    recs.append(dict(tag=d, draft=d, gamma=g, rep=rep, prompt_id=f"p{i}", split=split, spec=sp, target=tg, speedup=1.1))
    write_jsonl(R / "task3" / "spec" / "records_all.jsonl", recs)
    for m in ("teacher", "original", "on_s6304"):
        write_jsonl(R / "task3" / "standalone" / f"{m}.jsonl",
                    [dict(split="math", correct=True, extractable=True, truncated=False)] +
                    [dict(split=s, correct=False, format_ok=True) for s in ("familiar", "shifted")])


if __name__ == "__main__":
    task1(); task2(); task3()
    print("fake results in", R)

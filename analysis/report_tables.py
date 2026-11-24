#!/usr/bin/env python
"""LaTeX table bodies + macros for report/main.tex, generated from saved results (report/generated/*.tex).

Every number in the report's tables and every \\newcommand value comes from here, so the PDF traces to files.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from pa4.common.utils import paired_bootstrap, read_json, read_jsonl, results_root

R = results_root()
OUT = Path(__file__).resolve().parents[1] / "report" / "generated"
OUT.mkdir(parents=True, exist_ok=True)
MACROS = {}


def mac(name, value):
    MACROS[name] = value


def ci(d, scale=1.0, fmt="{:+.2f}"):
    return f"{fmt.format(scale * d['point'])} [{fmt.format(scale * d['ci_low'])}, {fmt.format(scale * d['ci_high'])}]"


def write(name, body):
    (OUT / f"{name}.tex").write_text(body)


# ============================================================================ Task 1
def task1():
    C = R / "task1/conditions"
    order = ["fp16", "p1_s25", "p2_s25", "p1_s50", "p2_s50", "p3_s50", "p2_24_s50", "p1_s75", "p2_s75", "p3_s75",
             "q1_b8", "q2_b8", "q1_b4", "q2_b4", "q3_b4", "q1_b3", "q2_b3", "q3_b3"]
    lab = {"fp16": "FP16 reference", "p2_24_s50": "P2 2:4 (Wanda)"}
    M = {c: read_json(C / c / "metrics.json") for c in order}
    hs = {c: {s["order"]: s for s in read_jsonl(C / c / "hellaswag.jsonl")} for c in order}
    b = M["fp16"]
    rows = []
    for c in order:
        m, q, st = M[c], M[c]["quality"], M[c]["storage"]
        common = sorted(hs[c])
        d_pp = 100 * np.mean([(hs[c][o]["pred_norm"] == hs[c][o]["gold"]) - (hs["fp16"][o]["pred_norm"] == hs["fp16"][o]["gold"]) for o in common])
        if c.startswith("q"):
            budget = f"{st['nominal_bits_per_eligible_weight_packed']:.2f} b/w"
            mb = st["packed_total_bytes"] / 1e6
        elif c == "fp16":
            budget = "16 b/w"
            mb = st["dense_fp16_total_bytes"] / 1e6
        else:
            budget = f"{100 * m['achieved_sparsity']:.1f}\\% sp."
            mb = st["sparse16_total_bytes"] / 1e6
        name = lab.get(c, c.upper().replace("_S", " ").replace("_B", " INT") + ("\\%" if "_s" in c else ""))
        dw = q["wikitext2"]["ppl"] - b["quality"]["wikitext2"]["ppl"]
        dc = q["c4val"]["ppl"] - b["quality"]["c4val"]["ppl"]
        fmt = lambda x: f"{x:,.0f}" if x >= 1000 else f"{x:.2f}"
        rows.append(" & ".join([
            name, budget, f"{fmt(q['wikitext2']['ppl'])} ({q['wikitext2']['mean_nll']:.3f})",
            f"{fmt(q['c4val']['ppl'])} ({q['c4val']['mean_nll']:.3f})",
            f"{100 * q['hellaswag']['acc']:.1f} / {100 * q['hellaswag']['acc_norm']:.1f} ({q['hellaswag']['n']})",
            "--" if c == "fp16" else f"{d_pp:+.1f}",
            f"{mb:,.0f}", "--" if m.get("calib_out_err_rel_mean") is None else f"{m['calib_out_err_rel_mean']:.3f}",
            "--" if c == "fp16" else f"{m['compression']['compression_seconds_total']:.0f}"]) + " \\\\")
        if c in ("fp16", "p2_s25", "p2_24_s50", "p3_s75", "q2_b8", "q3_b4"):
            rows.append("\\midrule")
    if rows[-1] == "\\midrule":
        rows.pop()
    write("t1_summary", "\n".join(rows))
    bp = read_json(R / "task1/bootstrap_pairs.json")
    names = {"p3_s50-p2_s50": "P3 $-$ P2, 50\\%", "p3_s75-p2_s75": "P3 $-$ P2, 75\\%", "q3_b4-q2_b4": "Q3 $-$ Q2, INT4 budget",
             "q3_b3-q2_b3": "Q3 $-$ Q2, INT3 budget", "p2_s50-p1_s50": "P2 $-$ P1, 50\\%", "q2_b3-q1_b3": "Q2 $-$ Q1, INT3",
             "q2_b4-q1_b4": "Q2 $-$ Q1, INT4", "p2_24_s50-p2_s50": "2:4 $-$ unstructured, 50\\%"}
    rows = []
    for k in names:
        v = bp[k]
        f = (lambda d: ci(d, 1, "{:+,.0f}")) if abs(v["wikitext2_ppl"]["point"]) > 100 else (lambda d: ci(d))
        rows.append(f"{names[k]} & {f(v['wikitext2_ppl'])} & {f(v['c4val_ppl'])} & {ci(v['hellaswag_acc_norm'], 100, '{:+.1f}')} ({v['hellaswag_n']}) \\\\")
        if k == "q3_b3-q2_b3":
            rows.append("\\midrule")
    write("t1_boot", "\n".join(rows))
    # macros
    q = b["quality"]
    mac("BaseWiki", f"{q['wikitext2']['ppl']:.2f}"); mac("BaseCfour", f"{q['c4val']['ppl']:.2f}")
    mac("BaseHS", f"{100 * q['hellaswag']['acc_norm']:.1f}"); mac("BaseHSacc", f"{100 * q['hellaswag']['acc']:.1f}")
    s5 = [s for s in hs["fp16"].values()]
    acc = np.array([s["pred_norm"] == s["gold"] for s in sorted(s5, key=lambda x: x["order"])])
    mac("HSfirst", f"{100 * acc[:512].mean():.1f}"); mac("HSrest", f"{100 * acc[512:].mean():.1f}")
    st = b["storage"]
    mac("FPGB", f"{st['dense_fp16_total_bytes'] / 1e9:.2f}"); mac("NElig", f"{st['n_eligible'] / 1e9:.3f}")
    mac("OtherMB", f"{st['uncompressed_other_bytes'] / 1e6:.0f}")
    lat = {c: M[c]["latency"] for c in order}
    pre = [1000 * l["prefill_2048"]["mean"] for l in lat.values()]
    dec = [l["decode_64"]["per_token_ms"] for l in lat.values()]
    mac("PrefillLo", f"{min(pre):.0f}"); mac("PrefillHi", f"{max(pre):.0f}"); mac("DecLo", f"{min(dec):.1f}"); mac("DecHi", f"{max(dec):.1f}")
    mac("PeakGB", f"{lat['fp16']['prefill_2048']['peak_allocated_bytes'] / 1e9:.2f}")
    mac("GPUone", b["env"].get("cuda_device", "?").replace("NVIDIA ", ""))
    cv = read_json(R / "task1/calib_vs_heldout.json")
    mac("RhoCalWiki", f"{cv['spearman_calib_err_vs_log_wiki_ppl']:.2f}"); mac("RhoDevHS", f"{cv['spearman_dev_nll_vs_hs_acc_norm']:.2f}")
    pr = read_json(R / "task1/sensitivity/probe.json")
    mac("ProbeS", f"{pr['seconds'] + pr['fisher']['seconds']:.0f}"); mac("FisherS", f"{pr['fisher']['seconds']:.0f}")
    U = pr["units"]
    jac = {s: np.mean([u["mag_wanda_jaccard"][s] for u in U.values()]) for s in ("0.25", "0.5", "0.75")}
    mac("JacTwentyFive", f"{jac['0.25']:.2f}"); mac("JacSeventyFive", f"{jac['0.75']:.2f}")
    jp = {p: np.mean([u["mag_wanda_jaccard"]["0.5"] for u in U.values() if u["proj"] == p]) for p in ("o_proj", "gate_proj")}
    mac("JacO", f"{jp['o_proj']:.2f}"); mac("JacGate", f"{jp['gate_proj']:.2f}")
    mac("ActMax", f"{max(u['act_norm_max'] for u in U.values()):,.0f}")
    sel = read_json(R / "task1/allocations/selection_summary.json")
    for t, keyn in (("p3_s75", "SelSeventyFive"), ("p3_s50", "SelFifty"), ("q3_b3", "SelQthree")):
        mac(keyn, ", ".join(f"{r['candidate'].split(t + '_')[1].replace('_', ' ')}: {r['dev_mean_nll']:.3f}" for r in sel[t]))
    q3 = read_json(R / "task1/allocations/q3_b4_selected.json")
    kv8 = np.mean([v == 8 for k, v in q3["bits"].items() if k.endswith(("k_proj", "v_proj"))])
    mac("QfourKVeight", f"{100 * kv8:.0f}")
    mac("QfourHist", ", ".join(f"{q3['bit_histogram'][b]}$\\times${b}b" for b in ("2", "3", "4", "8") if q3["bit_histogram"][b]))
    q33 = read_json(R / "task1/allocations/q3_b3_selected.json")
    mac("QthreeHist", ", ".join(f"{q33['bit_histogram'][b]}$\\times${b}b" for b in ("2", "3", "4", "8") if q33["bit_histogram"][b]))
    folds = [x["rel_err"] for c in ("q2_b4", "q2_b3", "q3_b4", "q3_b3") for x in M[c]["compression"]["fold_checks"]]
    mac("FoldErr", f"{max(folds):.1e}")
    mb = read_json(R / "task1/matched_budget_checks.json")
    mac("SlackFour", f"{mb['b4']['slack_bytes'] / 1e3:.0f}"); mac("SlackThree", f"{mb['b3']['slack_bytes'] / 1e3:.0f}")
    mac("ZerosFifty", f"{mb['s50']['p3_zeros']:,}")


# ============================================================================ Task 2
def task2():
    b = pd.DataFrame(read_jsonl(R / "task2/attention/bench.jsonl"))
    rows = []
    names = [("dense", "Dense (ours)"), ("tiled", "Tiled online softmax (ours)"), ("sdpa_default", "SDPA default"),
             ("sdpa_flash_attention", "SDPA Flash"), ("sdpa_efficient_attention", "SDPA mem-efficient"),
             ("sdpa_cudnn_attention", "SDPA cuDNN"), ("sdpa_math", "SDPA math")]
    for k, n in names:
        r16 = b[(b.impl == k) & (b.dtype == "float16") & (b.L == 8192)].iloc[0]
        s32 = "yes" if (b[(b.impl == k) & (b.dtype == "float32")].status == "ok").all() else "unsupported"
        if r16.status != "ok":
            rows.append(f"{n} & {s32} & \\multicolumn{{5}}{{c}}{{{r16.status}}} \\\\")
            continue
        g = max(r16.dq_max_abs, r16.dk_max_abs, r16.dv_max_abs)
        rows.append(f"{n} & {s32} & {r16.fwd_ms:.2f} & {r16.fwdbwd_ms:.1f} & {r16.fwdbwd_peak_alloc_bytes / 1e6:,.0f} & {r16.out_max_abs:.1e} & {g:.1e} \\\\")
    write("t2_attn", "\n".join(rows))
    bk = read_json(R / "task2/attention/backends.json")["default_dispatch_L1024"]
    mac("DefaultSixteen", bk["torch.float16"]["backend"].replace("_", " ").title())
    mac("DefaultThirtyTwo", bk["torch.float32"]["backend"].replace("_", " ").title())
    t32 = b[(b.impl == "tiled") & (b.dtype == "float32") & (b.status == "ok")]
    mac("TiledErrOut", f"{t32.out_max_abs.max():.1e}"); mac("TiledErrGrad", f"{t32[['dq_max_abs', 'dk_max_abs', 'dv_max_abs']].max().max():.1e}")
    ed = read_json(R / "task2/attention/edge_cases.json")
    mac("ExtremeDenseSixteen", f"{ed['dense']['extreme_scores']['torch.float16']['max_abs_err_vs_fp64']:.3f}")
    mac("ExtremeSdpaSixteen", f"{ed['sdpa_default']['extreme_scores']['torch.float16']['max_abs_err_vs_fp64']:.4f}")
    tl = read_json(R / "task2/attention/tiles.json")["rows"]
    mac("TileMinMB", f"{min(r['fwd_peak_alloc_bytes'] for r in tl) / 1e6:.1f}"); mac("TileMaxMB", f"{max(r['fwd_peak_alloc_bytes'] for r in tl) / 1e6:.0f}")
    # MQAR
    runs = [read_json(f) for f in sorted((R / "task2/mqar/runs").glob("*.json"))]
    by = defaultdict(list)
    for r in runs:
        by[r["name"]].append(r)
    th = read_json(R / "task2/mqar/theoretical_costs_T256.json")
    order = ["dense_K4_mixed", "dense_K8_mixed", "dense_K16_mixed", "dense_K16_near", "window_K4_near", "window_K8_near",
             "window_K16_near", "window_K16_mixed", "dilated_K4_mixed", "dilated_K16_mixed", "linear_K4_mixed",
             "linear_K8_mixed", "linear_K16_mixed", "hybrid_K16_mixed"]
    rows = []
    for n in order:
        rs = by[n]
        K, mech = rs[0]["train_K"], rs[0]["mechanism"]
        cell = lambda g: np.array([r["test"]["cells"][f"K{K}_{g}"]["acc"] for r in rs])
        f = lambda a: f"{a.mean():.2f}$\\pm${a.std(ddof=1):.2f}"
        seeds = ", ".join(f"{x:.2f}" for x in cell("mixed"))
        t = th[mech]
        rows.append(f"{mech} & {K} & {rs[0]['train_regime']} & {f(cell('near'))} & {f(cell('far'))} & {f(cell('mixed'))} & {seeds} & "
                    f"{t['pairs_per_head'] / 1000:.1f}k & {t['state_floats_per_layer'] / 1000:.1f}k & {np.mean([r['ms_per_step'] for r in rs]):.1f} & "
                    f"{np.mean([r['peak_train_alloc_bytes'] for r in rs]) / 1e6:,.0f} \\\\")
        if n in ("dense_K16_near", "window_K16_mixed", "dilated_K16_mixed", "linear_K16_mixed"):
            rows.append("\\midrule")
    write("t2_mqar", "\n".join(rows))
    ctrl = pd.read_csv(R / "tables/task2b_mask_controls.csv")
    d = ctrl[ctrl.model == "dense_K16_mixed"].set_index("condition")
    mac("MaskFar", f"{d.loc['W=64 mask at test', 'far']:.3f}"); mac("MaskNear", f"{d.loc['W=64 mask at test', 'near']:.2f}")
    mac("TopkMixed", f"{d.loc['top-64 at test', 'mixed']:.2f}")
    w = ctrl[ctrl.model == "window_K16_near"].iloc[0]; mac("WinFar", f"{w.far:.3f}")
    wl = [r["curve"][-1]["loss"] for r in by["window_K16_mixed"]]
    mac("WinMixLoss", f"{np.mean(wl):.2f}"); mac("LnSixtyFour", f"{np.log(64):.2f}")
    sw = pd.DataFrame(read_json(R / "task2/mqar/inference_sweep.json")["rows"])
    p = sw.pivot_table(index="mechanism", columns="T", values="ms")
    pm = sw.pivot_table(index="mechanism", columns="T", values="peak_alloc_bytes") / 1e6
    for m_ in ("dense", "window", "linear", "topk", "dilated"):
        mac(f"Sw{m_.title()}Ms", f"{p.loc[m_, 8192]:.1f}"); mac(f"Sw{m_.title()}MB", f"{pm.loc[m_, 8192]:,.0f}")
    # KV
    cases = read_jsonl(R / "task2/kv/cases.jsonl")
    meths = [("fp16", "FP16 full cache"), ("sink_recent", "FP16 evict: sink+recent"), ("qa_evict", "FP16 evict: question-aware"),
             ("int4_token", "INT4 token-wise K and V"), ("int4_kivi", "INT4 channel-K / token-V"), ("hybrid", "Hybrid FP16-select + INT2")]
    rows = []
    for m_, n in meths:
        g = lambda k: np.array([c["methods"][m_][k] for c in cases], dtype=float)
        meta = lambda k: np.array([np.nan if c["methods"][m_]["meta"].get(k) is None else c["methods"][m_]["meta"][k] for c in cases], dtype=float)
        ev = meta("evidence_fp16_retained")
        rows.append(f"{n} & {100 * g('correct').mean():.1f} & {g('code_nll').mean():.3f} & {g('first_token_logit_kl').mean():.3f} & "
                    f"{g('teacher_forced_kl_mean').mean():.3f} & {'--' if np.isnan(ev).all() else f'{100 * np.nanmean(ev):.1f}'} & "
                    f"{meta('bytes').mean() / 1e6:.1f} & {100 * meta('realized_fraction_of_fp16').mean():.1f} & "
                    f"{g('live_cache_bytes_end').mean() / 1e6:.1f} & {g('peak_alloc_bytes').mean() / 1e9:.2f} \\\\")
    write("t2_kv", "\n".join(rows))
    bt = read_json(R / "task2/kv/bootstrap.json")
    mac("KvQaSink", ci(bt["qa_evict-sink_recent"], 100, "{:+.1f}"))
    mac("KvNllHybQa", ci(bt["NLL hybrid-qa_evict"], 1, "{:+.3f}")); mac("KvNllKiviQa", ci(bt["NLL int4_kivi-qa_evict"], 1, "{:+.4f}"))
    mac("KvHybQa", ci(bt["hybrid-qa_evict"], 100, "{:+.1f}"))
    kr = lambda m_, k: np.mean([c["methods"][m_]["meta"][k] for c in cases])
    mac("KvKmseTok", f"{kr('int4_token', 'k_rel_mse'):.4f}"); mac("KvKmseKivi", f"{kr('int4_kivi', 'k_rel_mse'):.1e}")
    mac("KvVmse", f"{kr('int4_kivi', 'v_rel_mse'):.4f}"); mac("KvVmseHyb", f"{kr('hybrid', 'v_rel_mse'):.3f}")
    mac("KvHybM", f"{np.mean([c['methods']['hybrid']['meta']['fp16_selected'] for c in cases]):.0f}")
    mac("KvEvictN", f"{np.mean([c['methods']['qa_evict']['meta']['positions'] for c in cases]):.0f}")
    mac("KvLctxLo", f"{min(c['L_ctx'] for c in cases):,}"); mac("KvLctxHi", f"{max(c['L_ctx'] for c in cases):,}")
    sy = read_json(R / "task2/kv/systems_summary.json")
    mac("KvPrefill", f"{sy['prefill_ms']:.0f}"); mac("KvScoring", f"{sy['scoring_ms']:.0f}"); mac("KvSplit", f"{sy['c1_split_vs_full_max']:.3f}")
    mac("KvPrefillPeak", f"{sy['prefill_peak_MB'] / 1000:.2f}")
    import gzip
    raw = {json.loads(l)["id"]: json.loads(l) for l in gzip.open(Path(__import__('pa4.common.utils', fromlist=['x']).data_root()) / "data/kv_retrieval/kv_retrieval_cases.jsonl.gz", "rt")} if (Path(__import__('pa4.common.utils', fromlist=['x']).data_root()) / "data/kv_retrieval/kv_retrieval_cases.jsonl.gz").exists() else None
    if raw:
        wrong = [c for c in cases if not c["methods"]["sink_recent"]["correct"]]
        inctx = sum(1 for c in wrong if c["methods"]["sink_recent"]["pred"] and c["methods"]["sink_recent"]["pred"] in raw[c["id"]]["user_text"])
        mac("SinkWrong", str(len(wrong))); mac("SinkInCtx", str(inctx))
    pl = np.array([c["methods"]["qa_evict"]["meta"]["evidence_fp16_retained_per_layer"] for c in cases])
    mac("QaEvLzero", f"{100 * pl[:, 0].mean():.0f}"); mac("QaEvLtwo", f"{100 * pl[:, 2:].mean():.1f}"); mac("QaEvLtwoMin", f"{100 * pl[:, 2:].mean(0).min():.1f}")
    br = pd.read_csv(R / "tables/task2c_kv_breakdown.csv")
    rows = []
    for _, r in br.iterrows():
        if r.factor == "target_tokens":
            lv = f"nominal {int(r.level):,} (actual {r.L_ctx_mean:,.0f} tok.)"
        elif r.factor == "position":
            lv = f"evidence at {r.level}"
        else:
            lv = f"{r.level} distractors"
        rows.append(f"{lv} & {int(r.n)} & " + " & ".join(f"{100 * r[m_]:.0f}" for m_, _ in meths) + " \\\\")
        if r.level in ("4096", 4096, "start"):
            rows.append("\\midrule")
    write("t2_kvbreak", "\n".join(rows))
    c5 = read_jsonl(R / "task2/kv/c5_bench_rows.jsonl")
    rows = []
    for tgt in (2048, 4096):
        for bs in (1, 8):
            ref = next(r for r in c5 if r["target_tokens"] == tgt and r["batch"] == bs and r["layout"] == "fp16")
            for lay, bits in (("kivi", 4), ("token", 4), ("kivi", 2)):
                r = next(x for x in c5 if x["target_tokens"] == tgt and x["batch"] == bs and x["layout"] == lay and x["bits"] == bits)
                rows.append(f"{r['context_tokens']:,} & {bs} & {'chan-K' if lay == 'kivi' else 'tok-K'} INT{bits} & {r['serialized_bytes'] / 1e6:.1f} / {ref['serialized_bytes'] / 1e6:.1f} & "
                            f"{r['pack_ms']:.1f} & {r['unpack_ms']:.1f} & {r['decode_step_with_unpack_ms']:.1f} / {ref['decode_step_ms']:.1f} & "
                            f"{r['peak_alloc_bytes'] / 1e9:.2f} / {ref['peak_alloc_bytes'] / 1e9:.2f} \\\\")
    write("t2_c5", "\n".join(rows))
    # main-text subset: longest context only (full grid in the appendix)
    longest = max(r.split(" & ")[0] for r in rows if " & " in r)
    write("t2_c5_main", "\n".join(r for r in rows if r.startswith(longest + " & ")))
    mac("GPUtwo", runs[0]["env"].get("cuda_device", "?").replace("NVIDIA ", ""))


# ============================================================================ Task 3
def task3():
    c = read_json(R / "task3/checks/sampler_checks.json")
    mac("ChiP", f"{c['markov_gamma2']['exact']['p_value']:.2f}"); mac("ChiStat", f"{c['markov_gamma2']['exact']['chi2']:.1f}")
    mac("ChiNeg", f"{c['markov_gamma2']['p']['chi2']:,.0f}"); mac("ChiEos", f"{c['markov_eos_gamma3']['p_value']:.2f}")
    mac("ChiGone", f"{c['markov_gamma1']['exact']['p_value']:.2f}")
    mac("CacheDiff", f"{max(r['cache_vs_full_max_abs_prob'] for r in c['real_model_cache']):.1e}")
    mac("GPUthree", c["env"].get("cuda_device", "?").replace("NVIDIA ", ""))
    D = R / "task3/distill"
    runs = {n: read_json(D / n / "result.json") for n in ("off_s6304", "off_s6305", "on_s6304", "on_s6305")}
    pool = read_json(D / "teacher_pool_cost.json")
    rows = []
    for n, r in runs.items():
        cst, mg = r["costs"], read_json(D / n / "merge.json")
        poolS = pool["seconds"] if n.startswith("off") else 0
        rows.append(f"{n.replace('_s', ' s')} & {r['updates']} ({r['skipped_updates']}) & {poolS / 60:.1f} & {cst['student_rollout_s'] / 60:.1f} & "
                    f"{cst['teacher_scoring_s'] / 60:.1f} & {(cst['student_fwd_bwd_s'] + cst['optimizer_s']) / 60:.1f} & {mg['merge_seconds']:.1f} & "
                    f"{(r['wall_s'] + poolS + mg['merge_seconds']) / 60:.1f} & {mg['max_abs_logit_err']:.2f} & {r['peak_alloc_bytes'] / 1e9:.1f} \\\\")
    write("t3_distill", "\n".join(rows))
    mac("AdapterParams", f"{runs['off_s6304']['n_adapter_params']:,}"); mac("StudentParams", f"{runs['off_s6304']['n_student_backbone_params'] / 1e6:.0f}M")
    mac("TeacherParams", f"{runs['off_s6304']['n_teacher_params'] / 1e9:.2f}B")
    mac("PoolLen", str(pool["terminations"]["length"])); mac("PoolEos", str(pool["terminations"]["eos"]))
    pos = pd.DataFrame(read_jsonl(R / "task3/align/positions.jsonl.gz"))
    pos["method"] = pos.draft.str.replace(r"_s\d+", "", regex=True)
    pos["bankg"] = pos.bank.str.replace(r"_s\d+$", "", regex=True)
    banks = [("teacher", "Teacher (target) samples"), ("original_draft", "Original-draft samples"),
             ("exact_spec_original_gamma4", "Exact speculative ($\\gamma$=4)"), ("own_off", "Off-policy students' own"),
             ("own_on", "On-policy students' own")]
    rows = []
    for bk, n in banks:
        s = pos[pos.bankg == bk]
        seedm = s.groupby(["method", "draft"])[["fkl", "rkl", "exp_accept"]].mean()
        cells = []
        for m_ in ("original", "off", "on"):
            v = seedm.loc[m_]
            sd = f"$\\pm${v.exp_accept.std(ddof=1):.3f}" if len(v) > 1 else ""
            cells.append(f"{v.fkl.mean():.3f} & {v.rkl.mean():.3f} & {v.exp_accept.mean():.3f}{sd}")
        rows.append(f"{n} ({s[s.draft == 'original'].shape[0]:,}) & " + " & ".join(cells) + " \\\\")
    write("t3_align", "\n".join(rows))
    cons = read_json(R / "task3/align/consistency.json")
    mac("ConsDiff", ci(cons["prompt_paired_diff_exp_accept"], 1, "{:+.3f}"))
    mac("ConsLogpT", f"{cons['teacher']['mean_target_logp_of_sampled']:.3f}")
    mac("ConsLogpS", f"{cons['exact_spec_original_gamma4']['mean_target_logp_of_sampled']:.3f}")
    sm = pd.read_csv(R / "tables/task3_spec_main.csv")
    rows = []
    for tag, n in (("original", "original"), ("off_s6304", "off s6304"), ("off_s6305", "off s6305"), ("on_s6304", "on s6304"),
                   ("on_s6305", "on s6305"), ("on_s6304_unmerged", "on s6304 unmerged"), ("original_fixed64", "original, fixed 64")):
        for g in (2, 4):
            q = sm[(sm.draft == tag) & (sm.gamma == g)]
            if not len(q):
                continue
            r = q.iloc[0]
            be = "--" if pd.isna(r.break_even_alpha) else f"{r.break_even_alpha:.2f}"
            rows.append(f"{n} & {g} & {r.accept_examined:.3f} & {r.accept_proposed:.3f} & {r.mean_accepted_prefix:.2f} & {r.target_passes_per_token:.3f} & "
                        f"{r.draft_passes_per_token:.2f} & {r.t_draft_ms:.0f} / {r.t_verify_ms:.0f} / {r.t_accept_ms + r.t_rollback_ms:.0f} / {r.t_prefill_ms:.0f} & "
                        f"{r.spec_tok_s:.1f} / {r.target_tok_s:.1f} & {r.decode_speedup:.3f} [{r.decode_speedup_ci_low:.3f}, {r.decode_speedup_ci_high:.3f}] & "
                        f"{r.c:.2f} & {r.v_gamma:.2f} & {r.predicted_S:.2f} & {be} \\\\")
        if tag in ("original", "on_s6305", "on_s6304_unmerged"):
            rows.append("\\midrule")
    write("t3_spec", "\n".join(rows))
    sc = read_json(R / "task3/spec/step_costs.json")
    mac("StepTarget", f"{sc['target']['L256_tokens1']:.1f}"); mac("StepDraft", f"{sc['original']['L256_tokens1']:.1f}")
    mac("StepVerifyFive", f"{sc['target']['L256_tokens5']:.1f}")
    pc = pd.read_csv(R / "tables/task3_spec_paired_comparisons.csv")
    g2 = pc[(pc.gamma == 2) & (pc.metric == "throughput_ratio")]
    mac("PairOffOrig", "; ".join(f"{r.comparison.split(' - ')[0].replace('_s', ' s')}: {r['diff']:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]" for _, r in g2.iterrows() if "original" in r.comparison))
    oo = pc[pc.comparison.str.contains("off_") & pc.comparison.str.startswith("on_") & (pc.metric == "accept_examined")]
    lo, hi = oo.ci_low.min(), oo.ci_high.max()
    mac("OnOffRange", f"{oo['diff'].min():+.3f} to {oo['diff'].max():+.3f}")
    mac("OnOffAllZero", "all" if ((oo.ci_low <= 0) & (oo.ci_high >= 0)).all() else "not all")
    mac("OnOffCI", f"[{lo:+.3f}, {hi:+.3f}]")
    sa = pd.read_csv(R / "tables/task3_standalone_accuracy.csv").set_index("model")
    rows = []
    for m_, n in (("teacher", "Target 1.5B"), ("original", "Original 0.5B"), ("off_s6304", "Off s6304"), ("off_s6305", "Off s6305"),
                  ("on_s6304", "On s6304"), ("on_s6305", "On s6305")):
        r = sa.loc[m_]
        rows.append(f"{n} & {100 * r.math_acc:.0f} & {100 * r.math_truncated:.0f} & {100 * r.familiar_acc:.0f} / {100 * r.familiar_answer_present:.0f} & "
                    f"{100 * r.shifted_acc:.0f} / {100 * r.shifted_answer_present:.0f} \\\\")
    write("t3_standalone", "\n".join(rows))
    be = pd.read_csv(R / "tables/task3_deployment_break_even.csv")
    r = be[(be.draft == "off_s6304") & (be.gamma == 4)].iloc[0]
    mac("BeSave", f"{r.saving_ms_per_req_vs_original_draft:.1f}"); mac("BeReq", f"{r.break_even_requests_vs_original / 1e3:,.0f}k")
    mac("BeCost", f"{r.distill_cost_ms / 6e4:.0f}")
    r2 = be[(be.draft == "off_s6304") & (be.gamma == 2)].iloc[0]
    mac("BeSaveTwo", f"{r2.saving_ms_per_req_vs_original_draft:.1f}"); mac("BeReqTwo", f"{r2.break_even_requests_vs_original / 1e6:.2f}M")
    w = pd.read_csv(R / "tables/task3_spec_workload.csv")
    get = lambda d, g, s_, k: w[(w.draft == d) & (w.gamma == g) & (w.split == s_)][k].iloc[0]
    mac("FamAccOrig", f"{get('original', 2, 'familiar', 'accept_examined'):.3f}"); mac("FamAccOff", f"{get('off_s6304', 2, 'familiar', 'accept_examined'):.3f}")
    mac("ShAccOrig", f"{get('original', 2, 'shifted', 'accept_examined'):.3f}"); mac("ShAccOff", f"{get('off_s6304', 2, 'shifted', 'accept_examined'):.3f}")
    mac("MathAccOrig", f"{get('original', 2, 'math', 'accept_examined'):.3f}"); mac("MathAccOff", f"{get('off_s6304', 2, 'math', 'accept_examined'):.3f}")
    al = pos[(pos.bank == "teacher")]
    fam_short = al[(al.split == "familiar") & (al.position <= 64)].groupby("method").exp_accept.mean()
    mac("FamExpOrig", f"{fam_short['original']:.3f}"); mac("FamExpOff", f"{fam_short['off']:.3f}")


def main():
    task1(); task2(); task3()
    body = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MACROS.items()))
    (OUT / "macros.tex").write_text("% generated by analysis/report_tables.py -- do not edit\n" + body + "\n")
    print(f"wrote {len(MACROS)} macros and tables to {OUT}")


if __name__ == "__main__":
    main()

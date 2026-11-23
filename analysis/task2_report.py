#!/usr/bin/env python
"""Task 2 tables/figures: exact attention (A), MQAR (B), KV cache fidelity and systems (C)."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from pa4.common.utils import bootstrap_mean_diff, read_json, read_jsonl, results_root, write_json
from plotting import CAT, SEQ, color_map, plt, save, write_table

R = results_root() / "task2"


# ============================================================================ A
def part_a():
    f = R / "attention" / "bench.jsonl"
    if not f.exists():
        return
    df = pd.DataFrame(read_jsonl(f))
    write_table(df[[c for c in ["impl", "dtype", "L", "status", "fwd_ms", "fwdbwd_ms", "fwd_peak_alloc_bytes",
                                "fwdbwd_peak_alloc_bytes", "out_max_abs", "dq_max_abs", "dk_max_abs", "dv_max_abs", "reference"]
                    if c in df]], "task2a_attention_full")
    status = df.pivot_table(index=["impl", "dtype"], columns="L", values="status", aggfunc="first").reset_index()
    write_table(status, "task2a_backend_status", caption="Supported / unsupported / OOM by implementation.")
    ok = df[df.status == "ok"]
    impls = ["dense", "tiled", "sdpa_default", "sdpa_flash_attention", "sdpa_efficient_attention", "sdpa_cudnn_attention", "sdpa_math"]
    cm = color_map(impls)
    fig, axes = plt.subplots(2, 3, figsize=(9, 5), sharex=True)
    for r, dt in enumerate(("float32", "float16")):
        for c, (col, lab) in enumerate((("fwd_ms", "forward ms"), ("fwdbwd_ms", "forward+backward ms"),
                                        ("fwdbwd_peak_alloc_bytes", "peak allocated (MB), fwd+bwd"))):
            ax = axes[r, c]
            for im in impls:
                s = ok[(ok.impl == im) & (ok.dtype == dt)].sort_values("L")
                if len(s):
                    y = s[col] / (1e6 if "bytes" in col else 1)
                    ax.plot(s.L, y, marker="o", color=cm[im], label=im.replace("sdpa_", "sdpa:"))
            ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_title(f"{dt}: {lab}")
            if r == 1:
                ax.set_xlabel("sequence length L")
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout(); save(fig, "task2", "attention_cost")
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))
    for ax, dt in zip(axes, ("float32", "float16")):
        for im in impls:
            s = ok[(ok.impl == im) & (ok.dtype == dt)].sort_values("L")
            if len(s):
                ax.plot(s.L, s.out_max_abs, marker="o", color=cm[im], label=f"{im} out")
                ax.plot(s.L, s[["dq_max_abs", "dk_max_abs", "dv_max_abs"]].max(axis=1), marker="x", ls=":", color=cm[im])
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("L")
        ax.set_ylabel("max |err| vs FP32 dense (solid: output, dotted: grads)"); ax.set_title(dt)
    axes[0].legend(fontsize=5)
    fig.tight_layout(); save(fig, "task2", "attention_error")


# ============================================================================ B
def part_b():
    runs_dir = R / "mqar" / "runs"
    if not runs_dir.exists():
        return
    runs = [read_json(f) for f in sorted(runs_dir.glob("*.json"))]
    by = defaultdict(list)
    for r in runs:
        by[r["name"]].append(r)
    names = list(by)
    cm = color_map(names)
    # learning curves
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for n in names:
        rs = by[n]
        steps = [c["step"] for c in rs[0]["curve"]]
        loss = np.array([[c["loss"] for c in r["curve"]] for r in rs])
        axes[0].plot(steps, loss.mean(0), color=cm[n], label=n, lw=1.2)
        st = [c["step"] for c in rs[0]["dev_curve"]]
        acc = np.array([[c["acc"] for c in r["dev_curve"]] for r in rs])
        axes[1].plot(st, acc.mean(0), marker="o", color=cm[n], lw=1.2)
        axes[1].fill_between(st, acc.min(0), acc.max(0), color=cm[n], alpha=0.12, lw=0)
    axes[0].set_xlabel("update"); axes[0].set_ylabel("train CE (mean of 3 seeds)"); axes[0].set_yscale("log")
    axes[1].set_xlabel("update"); axes[1].set_ylabel("dev accuracy (all cells), min-max over seeds")
    axes[0].legend(fontsize=5, ncol=2)
    fig.tight_layout(); save(fig, "task2", "mqar_learning_curves")
    # accuracy table: each run on near/far/mixed at its trained K and on all K
    rows = []
    for n in names:
        rs = by[n]
        K = rs[0]["train_K"]
        row = dict(run=n, mechanism=rs[0]["mechanism"], train_K=K, train_regime=rs[0]["train_regime"], seeds=len(rs),
                   params=rs[0]["n_params"], ms_per_step=np.mean([r["ms_per_step"] for r in rs]),
                   peak_train_MB=np.mean([r["peak_train_alloc_bytes"] or 0 for r in rs]) / 1e6)
        for g in ("near", "far", "mixed"):
            v = [r["test"]["cells"][f"K{K}_{g}"]["acc"] for r in rs]
            row[f"{g}@K{K}"] = np.mean(v); row[f"{g}@K{K}_sd"] = np.std(v, ddof=1) if len(v) > 1 else 0
        for kk in (4, 8, 16):
            row[f"mixed@K{kk}"] = np.mean([r["test"]["cells"][f"K{kk}_mixed"]["acc"] for r in rs])
        row["chance_label"] = 1 / 64; row["chance_present"] = 1 / K
        rows.append(row)
    df = pd.DataFrame(rows)
    write_table(df, "task2b_mqar_accuracy", caption="MQAR test accuracy (mean, sd over 3 seeds).")
    # per-seed results: exposes bimodal outcomes (some seeds solve the task, others stay near chance)
    ps = []
    for n in names:
        for r in by[n]:
            K = r["train_K"]
            ps.append(dict(run=n, seed=r["seed"], **{f"{g}@K{K}": r["test"]["cells"][f"K{K}_{g}"]["acc"] for g in ("near", "far", "mixed")},
                           final_train_loss=r["curve"][-1]["loss"], final_dev_acc=r["dev_curve"][-1]["acc"],
                           dev_acc_at_500=r["dev_curve"][0]["acc"]))
    write_table(pd.DataFrame(ps), "task2b_mqar_per_seed", caption="MQAR per-seed test accuracy at the trained K.")
    # distance x load heatmaps
    bins = ["[0,32)", "[32,64)", "[64,128)", "[128,256)"]
    n_cols = 4
    n_rows = int(np.ceil(len(names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 2.1 * n_rows))
    for ax, n in zip(np.ravel(axes), names):
        rs = by[n]
        g = np.array([[np.mean([r["test"]["heat"][f"K{k}_{b}"]["acc"] or np.nan for r in rs]) for b in bins] for k in (4, 8, 16)])
        im = ax.imshow(g, vmin=0, vmax=1, cmap=SEQ, origin="lower", aspect="auto")
        for i in range(3):
            for j in range(4):
                if not np.isnan(g[i, j]):
                    ax.text(j, i, f"{g[i, j]:.2f}", ha="center", va="center", fontsize=5,
                            color="white" if g[i, j] > 0.6 else "#0b0b0b")
        ax.set_xticks(range(4)); ax.set_xticklabels(bins, fontsize=5); ax.set_yticks(range(3)); ax.set_yticklabels(["K=4", "K=8", "K=16"])
        ax.set_title(f"{n} (trained K={rs[0]['train_K']})", fontsize=6); ax.grid(False)
    for ax in np.ravel(axes)[len(names):]:
        ax.axis("off")
    fig.tight_layout(); save(fig, "task2", "mqar_distance_load_heatmaps")
    # train-vs-test mask controls
    ctrl = []
    for n in names:
        rs = by[n]
        if rs[0]["mechanism"] != "dense":
            continue
        K = rs[0]["train_K"]
        for tag, key in (("as trained", "test"), ("W=64 mask at test", "test_window64_at_test"), ("top-64 at test", "test_topk64_at_test")):
            ctrl.append(dict(model=n, condition=tag, **{g: np.mean([r[key]["cells"][f"K{K}_{g}"]["acc"] for r in rs])
                                                        for g in ("near", "far", "mixed")}))
    for n in names:
        if by[n][0]["mechanism"] == "window":
            K = by[n][0]["train_K"]
            ctrl.append(dict(model=n, condition="window-trained", **{g: np.mean([r["test"]["cells"][f"K{K}_{g}"]["acc"] for r in by[n]])
                                                                     for g in ("near", "far", "mixed")}))
    if ctrl:
        write_table(pd.DataFrame(ctrl), "task2b_mask_controls")
    # costs + sweep
    tc = R / "mqar" / "theoretical_costs_T256.json"
    if tc.exists():
        t = read_json(tc)
        write_table(pd.DataFrame([dict(mechanism=k, **{kk: vv for kk, vv in v.items() if not isinstance(vv, str)}) for k, v in t.items()]),
                    "task2b_theoretical_costs")
    sw = R / "mqar" / "inference_sweep.json"
    if sw.exists():
        s = pd.DataFrame(read_json(sw)["rows"])
        write_table(s, "task2b_inference_sweep")
        mech = list(dict.fromkeys(s.mechanism))
        c2 = color_map(mech)
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))
        for m in mech:
            q = s[(s.mechanism == m) & (s.status == "ok")].sort_values("T")
            axes[0].plot(q["T"], q.ms, marker="o", color=c2[m], label=m)
            axes[1].plot(q["T"], q.peak_alloc_bytes / 1e6, marker="o", color=c2[m], label=m)
        for ax, yl in zip(axes, ("forward ms (1 attention layer, B=4)", "peak allocated MB")):
            ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("T"); ax.set_ylabel(yl)
        axes[0].legend()
        fig.tight_layout(); save(fig, "task2", "mqar_inference_sweep")


# ============================================================================ C
METHODS = ["fp16", "sink_recent", "qa_evict", "int4_token", "int4_kivi", "hybrid"]
LABEL = dict(fp16="FP16 full", sink_recent="Sink+recent (FP16 evict)", qa_evict="Question-aware (FP16 evict)",
             int4_token="INT4 token-wise K/V", int4_kivi="INT4 channel-K/token-V", hybrid="Hybrid FP16-select + INT2")


def part_c():
    f = R / "kv" / "cases.jsonl"
    if not f.exists():
        return
    cases = read_jsonl(f)
    print("KV cases:", len(cases))
    ref_ok = np.array([c["methods"]["fp16"]["correct"] for c in cases])
    rows = []
    for m in METHODS:
        acc = np.array([c["methods"][m]["correct"] for c in cases], float)
        rows.append(dict(
            method=LABEL[m], n=len(cases), exact_match=acc.mean(),
            acc_given_ref_correct=acc[ref_ok].mean() if ref_ok.any() else np.nan,
            code_nll=np.mean([c["methods"][m]["code_nll"] for c in cases]),
            first_token_KL=np.mean([c["methods"][m]["first_token_logit_kl"] for c in cases]),
            teacher_forced_KL=np.mean([c["methods"][m].get("teacher_forced_kl_mean", np.nan) for c in cases]),
            evidence_fp16_retained=np.nanmean([np.nan if c["methods"][m]["meta"].get("evidence_fp16_retained") is None
                                               else c["methods"][m]["meta"]["evidence_fp16_retained"] for c in cases]),
            packed_KB=np.mean([c["methods"][m]["meta"]["bytes"] for c in cases]) / 1024,
            frac_of_fp16=np.mean([c["methods"][m]["meta"]["realized_fraction_of_fp16"] for c in cases]),
            within_budget=all(c["methods"][m]["meta"]["within_budget"] for c in cases) if m != "fp16" else None,
            k_rel_mse=np.mean([c["methods"][m]["meta"].get("k_rel_mse", np.nan) for c in cases]),
            v_rel_mse=np.mean([c["methods"][m]["meta"].get("v_rel_mse", np.nan) for c in cases]),
            compress_ms=1000 * np.mean([c["methods"][m]["compress_seconds"] for c in cases]),
            suffix_ms=1000 * np.mean([c["methods"][m]["t_suffix"] for c in cases]),
            decode_ms=1000 * np.mean([c["methods"][m]["t_decode"] for c in cases]),
            peak_alloc_MB=np.mean([c["methods"][m]["peak_alloc_bytes"] for c in cases]) / 1e6,
            shared_decode_KB=np.mean([c["methods"][m].get("shared_decode_bytes", np.nan) for c in cases]) / 1024,
            live_cache_end_KB=np.mean([c["methods"][m]["live_cache_bytes_end"] for c in cases]) / 1024))
    df = pd.DataFrame(rows)
    write_table(df, "task2c_kv_matched_budget", caption="KV policies at the per-prompt INT4 channel-K/token-V byte budget.")
    sysrow = dict(prefill_ms=1000 * np.mean([c["prefill_seconds"] for c in cases]),
                  scoring_ms=1000 * np.mean([c["scoring_seconds"] for c in cases]),
                  prefill_peak_MB=np.mean([c["prefill_peak_alloc_bytes"] for c in cases]) / 1e6,
                  c1_split_vs_full_max=max(c["c1_split_vs_full_max_abs_logit"] for c in cases),
                  c1_generate_match_rate=np.mean([c["methods"]["fp16"]["c1_matches_ordinary_generate"] for c in cases]),
                  all_finite=all(c["methods"][m]["finite"] for c in cases for m in METHODS))
    write_json(R / "kv" / "systems_summary.json", sysrow)
    # C1: full-cache (FP16 reference) retrieval accuracy on every prompt cell, alongside each policy
    cells = []
    for key in sorted({(c["target_tokens"], c["position"], c["distractors"]) for c in cases}, key=str):
        sub = [c for c in cases if (c["target_tokens"], c["position"], c["distractors"]) == key]
        cells.append(dict(context=key[0], position=key[1], distractors=key[2], n=len(sub),
                          L_ctx_range=f"{min(c['L_ctx'] for c in sub)}-{max(c['L_ctx'] for c in sub)}",
                          **{m: np.mean([c["methods"][m]["correct"] for c in sub]) for m in METHODS}))
    write_table(pd.DataFrame(cells), "task2c_kv_cells", caption="Exact match per prompt cell (10 prompts each; descriptive only).")
    # breakdowns
    br = []
    for fac in ("target_tokens", "position", "distractors"):
        for lv in sorted({c[fac] for c in cases}, key=str):
            sub = [c for c in cases if c[fac] == lv]
            br.append(dict(factor=fac, level=lv, n=len(sub), L_ctx_mean=np.mean([c["L_ctx"] for c in sub]),
                           **{m: np.mean([c["methods"][m]["correct"] for c in sub]) for m in METHODS}))
    write_table(pd.DataFrame(br), "task2c_kv_breakdown")
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.6), sharey=True)
    cm = color_map(METHODS)
    bd = pd.DataFrame(br)
    for ax, fac in zip(axes, ("target_tokens", "position", "distractors")):
        s = bd[bd.factor == fac]
        x = np.arange(len(s))
        w = 0.13
        for i, m in enumerate(METHODS):
            ax.bar(x + (i - 2.5) * w, s[m], width=w * 0.9, color=cm[m], label=LABEL[m])
        ax.set_xticks(x); ax.set_xticklabels([str(v) for v in s.level]); ax.set_title(fac); ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("exact-match accuracy (10 prompts x other factors)")
    axes[0].legend(fontsize=5)
    fig.tight_layout(); save(fig, "task2", "kv_breakdown")
    # paired bootstrap on principal differences
    acc = {m: np.array([c["methods"][m]["correct"] for c in cases], float) for m in METHODS}
    pairs = [("qa_evict", "sink_recent"), ("int4_kivi", "int4_token"), ("hybrid", "qa_evict"), ("hybrid", "int4_kivi"),
             ("int4_kivi", "qa_evict")] + [(m, "fp16") for m in METHODS[1:]]
    boots = {f"{a}-{b}": bootstrap_mean_diff(acc[a], acc[b], 2000) for a, b in pairs}
    nll = {m: np.array([c["methods"][m]["code_nll"] for c in cases]) for m in METHODS}
    for a, b in pairs[:5]:
        boots[f"NLL {a}-{b}"] = bootstrap_mean_diff(nll[a], nll[b], 2000)
    write_json(R / "kv" / "bootstrap.json", boots)
    write_table(pd.DataFrame([dict(pair=k, **v) for k, v in boots.items()]), "task2c_kv_bootstrap")
    # C5 bench
    b = R / "kv" / "c5_bench.json"
    if b.exists():
        write_table(pd.DataFrame(read_json(b)["rows"]), "task2c_c5_pack_bench")


def main():
    part_a(); part_b(); part_c()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Compact figures for the 8-page NeurIPS report (text width 5.5 in). Output: results/figures/report/*.pdf.

Every value is read from the saved result files; nothing is typed in by hand.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from pa4.common.utils import read_json, read_jsonl, results_root
from plotting import CAT, INK2, SEQ, color_map, plt, save

R = results_root()
W = 5.5
PROJ = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5, "legend.fontsize": 5.5,
                     "xtick.labelsize": 5.5, "ytick.labelsize": 5.5, "lines.linewidth": 1.2, "lines.markersize": 3})


def unit_grid(vals: dict, nl=28):
    g = np.full((nl, 7), np.nan)
    for u, v in vals.items():
        _, l, p = u.split(".")
        g[int(l), PROJ.index(p)] = v
    return g


def heat(ax, g, title, cbar_label, fig, vmin=None, vmax=None):
    im = ax.imshow(g, aspect="auto", cmap=SEQ, origin="lower", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(7)); ax.set_xticklabels(["q", "k", "v", "o", "gate", "up", "down"], rotation=90, fontsize=4.8)
    ax.set_title(title); ax.grid(False)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))
    cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03)
    cb.ax.tick_params(labelsize=5); cb.set_label(cbar_label, fontsize=5.5)


def task1():
    df = pd.read_csv(R / "tables/task1_all_conditions.csv")
    base = df[df.condition == "fp16"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.75))
    cm = color_map(["P1 magnitude", "P2 Wanda", "P3 adaptive", "2:4 Wanda"])
    ax = axes[0]
    for lab, pre in (("P1 magnitude", "p1_s"), ("P2 Wanda", "p2_s"), ("P3 adaptive", "p3_s")):
        s = df[df.condition.str.startswith(pre)].sort_values("sparsity")
        ax.plot(s.sparsity, s.wiki_ppl, marker="o", color=cm[lab], label=lab)
    s = df[df.condition == "p2_24_s50"]
    ax.scatter(s.sparsity, s.wiki_ppl, marker="s", color=cm["2:4 Wanda"], label="2:4 Wanda", zorder=3, s=12)
    ax.axhline(base.wiki_ppl, color=INK2, lw=0.8, ls="--", label="FP16")
    ax.set_yscale("log"); ax.set_xlabel("eligible-weight sparsity (achieved)"); ax.set_ylabel("WikiText-2 PPL (log)")
    ax.set_title("(a) Pruning, 50k scored tokens"); ax.legend(ncol=2)
    ax = axes[1]
    cq = color_map(["Q1 RTN", "Q2 AWQ-style", "Q3 mixed"])
    for lab, pre in (("Q1 RTN", "q1_"), ("Q2 AWQ-style", "q2_"), ("Q3 mixed", "q3_")):
        s = df[df.condition.str.startswith(pre)].sort_values("packed_MB")
        ax.plot(s.packed_MB / 1000, s.wiki_ppl, marker="o", color=cq[lab], label=lab)
    ax.axhline(base.wiki_ppl, color=INK2, lw=0.8, ls="--", label=f"FP16 ({base.dense_fp16_MB / 1000:.2f} GB)")
    ax.set_yscale("log"); ax.set_xlabel("total packed bytes (GB, incl. metadata and FP16 non-eligible params)")
    ax.set_ylabel("WikiText-2 PPL (log)"); ax.set_title("(b) Weight-only quantization, G=128"); ax.legend()
    fig.tight_layout(pad=0.3); save(fig, "report", "t1_tradeoff")

    pr = read_json(R / "task1/sensitivity/probe.json")
    U = pr["units"]
    i50 = pr["prune_grid"].index(0.5)
    p50 = read_json(R / "task1/allocations/p3_s50_selected.json")
    p75 = read_json(R / "task1/allocations/p3_s75_selected.json")
    q3 = read_json(R / "task1/allocations/q3_b3_selected.json")
    fig, axes = plt.subplots(1, 5, figsize=(W, 1.9))
    heat(axes[0], unit_grid({u: np.log10(v["prune_fisher"][i50]) for u, v in U.items()}), "(a) Fisher prune\ncost @50%", "log10", fig)
    heat(axes[1], unit_grid(p50["unit_sparsity"]), "(b) P3 sparsity @50%\n(Fisher knapsack)", "sparsity", fig, 0.1, 0.85)
    heat(axes[2], unit_grid(p75["unit_sparsity"]), "(c) P3 sparsity @75%\n(OWL, $\\lambda$=0.08)", "sparsity", fig)
    heat(axes[3], unit_grid({u: np.log10(v["quant"]["3"]["fisher"]) for u, v in U.items()}), "(d) Fisher quant\ncost @3 bit", "log10", fig)
    heat(axes[4], unit_grid(q3["bits"]), "(e) Q3 bits\n(INT3 budget)", "bits", fig, 2, 8)
    axes[0].set_ylabel("decoder block")
    fig.tight_layout(pad=0.3, w_pad=0.4); save(fig, "report", "t1_maps")


def task2():
    b = pd.DataFrame(read_jsonl(R / "task2/attention/bench.jsonl"))
    ok = b[b.status == "ok"].copy()
    ok["grad"] = ok[["dq_max_abs", "dk_max_abs", "dv_max_abs"]].max(axis=1)
    impls = ["dense", "tiled", "sdpa_default", "sdpa_math"]
    cm = color_map(impls)
    lab = {"dense": "dense (ours)", "tiled": "tiled online-softmax (ours)", "sdpa_default": "SDPA default", "sdpa_math": "SDPA math"}
    fig, axes = plt.subplots(1, 3, figsize=(W, 1.75))
    for im in impls:
        for dt, ls in (("float32", "-"), ("float16", "--")):
            s = ok[(ok.impl == im) & (ok.dtype == dt)].sort_values("L")
            if im == "dense" and dt == "float32":
                continue  # it is the reference
            axes[0].plot(s.L, s.grad, ls=ls, marker="o", color=cm[im], label=f"{lab[im]} {dt[5:]}" if dt == "float16" else None)
        s = ok[(ok.impl == im) & (ok.dtype == "float16")].sort_values("L")
        axes[1].plot(s.L, s.fwdbwd_ms, marker="o", color=cm[im], label=lab[im])
        axes[1].plot(s.L, s.fwd_ms, marker="", ls=":", color=cm[im])
        axes[2].plot(s.L, s.fwdbwd_peak_alloc_bytes / 1e6, marker="o", color=cm[im], label=lab[im])
    axes[0].set_title("(a) max |grad err| vs FP32 dense\n(solid FP32, dashed FP16)")
    axes[1].set_title("(b) FP16 latency, ms\n(solid fwd+bwd, dotted fwd)")
    axes[2].set_title("(c) FP16 fwd+bwd peak\nallocated memory, MB")
    for ax in axes:
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("sequence length L")
    axes[0].legend(handles=[plt.Line2D([], [], color=cm[i], marker="o", label=lab[i]) for i in impls], loc="center", fontsize=5)
    fig.tight_layout(pad=0.3); save(fig, "report", "t2_attention")

    runs = [read_json(f) for f in sorted((R / "task2/mqar/runs").glob("*.json"))]
    by = defaultdict(list)
    for r in runs:
        by[r["name"]].append(r)
    order = ["dense_K4_mixed", "dense_K8_mixed", "dense_K16_mixed", "dense_K16_near", "window_K4_near", "window_K8_near",
             "window_K16_near", "window_K16_mixed", "dilated_K4_mixed", "dilated_K16_mixed", "linear_K4_mixed",
             "linear_K8_mixed", "linear_K16_mixed", "hybrid_K16_mixed"]
    bins = ["[0,32)", "[32,64)", "[64,128)", "[128,256)"]
    G = np.array([[np.mean([r["test"]["heat"][f"K{k}_{bn}"]["acc"] for r in by[n]]) for k in (4, 8, 16) for bn in bins] for n in order])
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.35), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axes[0]
    im = ax.imshow(G, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            ax.text(j, i, "1" if G[i, j] >= 0.995 else f"{G[i, j]:.2f}"[1:], ha="center", va="center", fontsize=4.3,
                    color="white" if G[i, j] > 0.6 else "#0b0b0b")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([n.replace("_", " ") for n in order], fontsize=5)
    ax.set_xticks(range(12)); ax.set_xticklabels(["0", "32", "64", "128"] * 3, fontsize=4.5)
    for x in (3.5, 7.5):
        ax.axvline(x, color="white", lw=1.5)
    ax.set_xlabel("distance bin start, grouped by test K = 4 | 8 | 16"); ax.grid(False)
    ax.set_title("(a) test accuracy by distance x load (rows: model, trained K, regime)")
    ax = axes[1]
    fams = ["dense", "window", "linear", "dilated", "hybrid"]
    cmf = color_map(fams)
    for f in fams:
        for n in [n for n in order if by[n][0]["mechanism"] == f]:
            rs = by[n]
            st = [c["step"] for c in rs[0]["dev_curve"]]
            acc = np.array([[c["acc"] for c in r["dev_curve"]] for r in rs])
            ax.plot(st, acc.mean(0), color=cmf[f], lw=0.9, alpha=0.9, label=f if n == [m for m in order if by[m][0]["mechanism"] == f][0] else None)
            ax.fill_between(st, acc.min(0), acc.max(0), color=cmf[f], alpha=0.08, lw=0)
    ax.axhline(1 / 64, color=INK2, lw=0.7, ls=":")
    ax.set_xlabel("optimizer update"); ax.set_ylabel("dev accuracy (all K x regimes)")
    ax.set_ylim(0, 1.3); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("(b) learning curves, 14 configs x 3 seeds\n(line = seed mean, band = min-max)"); ax.legend(ncol=3, loc="upper center")
    fig.tight_layout(pad=0.3); save(fig, "report", "t2_mqar")


def task3():
    fig, axes = plt.subplots(1, 3, figsize=(W, 1.8))
    d = R / "task3/distill"
    names = ["off_s6304", "off_s6305", "on_s6304", "on_s6305"]
    cm = color_map(names)
    for n in names:
        r = read_json(d / n / "result.json")
        x = [l["scored_tokens"] for l in r["logs"]]
        y = pd.Series([l["kl_per_token"] for l in r["logs"]]).rolling(10, min_periods=1).mean()
        axes[0].plot(x, y, color=cm[n], ls="--" if n.startswith("off") else "-", label=n.replace("_s", " s"))
    axes[0].set_xlabel("teacher-scored response tokens"); axes[0].set_ylabel("fwd KL / token (10-update mean)")
    axes[0].set_ylim(0.05, 0.33)
    axes[0].set_title("(a) training KL on own prefixes"); axes[0].legend(ncol=2, loc="upper center")
    f = pd.read_csv(R / "tables/task3_alignment_full.csv")
    f = f[f.bank == "teacher"].copy()
    f["method"] = f.draft.str.replace(r"_s\d+", "", regex=True)
    grp = f.groupby(["split", "length", "method"]).exp_accept.mean()
    cats = [(s, l) for s in ("math", "familiar", "shifted") for l in ("short(1-64)", "long(65-192)")]
    meth = ["original", "off", "on"]
    cmm = color_map(meth)
    x = np.arange(len(cats))
    for i, m in enumerate(meth):
        axes[1].bar(x + (i - 1) * 0.27, [grp.get((s, l, m), np.nan) for s, l in cats], width=0.25, color=cmm[m], label=m)
    axes[1].set_xticks(x); axes[1].set_xticklabels([f"{s[:4]}\n{'1-64' if l.startswith('s') else '65-192'}" for s, l in cats], fontsize=5)
    axes[1].set_ylim(0.7, 1.0); axes[1].set_yticks([0.7, 0.75, 0.8, 0.85, 0.9]); axes[1].set_ylabel("E[1 - TV] (teacher bank)"); axes[1].grid(axis="x", visible=False)
    axes[1].set_title("(b) expected acceptance by\nworkload and response position"); axes[1].legend(ncol=3)
    sm = pd.read_csv(R / "tables/task3_spec_main.csv")
    pos = pd.DataFrame(read_jsonl(R / "task3/align/positions.jsonl.gz"))
    exp = pos[pos.bank == "teacher"].groupby("draft").exp_accept.mean()  # token-weighted over bank positions
    drafts = ["original", "off_s6304", "off_s6305", "on_s6304", "on_s6305"]
    cd = color_map(drafts)
    for g, mk in ((2, "o"), (4, "s")):
        for dr in drafts:
            q = sm[(sm.draft == dr) & (sm.gamma == g)]
            axes[2].scatter(exp[dr], q.accept_examined.iloc[0], color=cd[dr], marker=mk, s=14,
                            label=dr.replace("_s", " s") if g == 2 else None)
    lo, hi = 0.855, 0.895
    axes[2].plot([lo, hi], [lo, hi], color=INK2, lw=0.7, ls="--", label="y = x")
    axes[2].set_xlim(lo, hi); axes[2].set_ylim(lo, hi)
    axes[2].set_xlabel("expected E[1 - TV] (teacher bank)"); axes[2].set_ylabel("measured accepted / examined")
    axes[2].set_title("(c) predicted vs measured\nacceptance (circle $\\gamma$=2, square 4)"); axes[2].legend(fontsize=4.8, loc="upper left")
    fig.tight_layout(pad=0.3); save(fig, "report", "t3_distill_align")


if __name__ == "__main__":
    task1(); task2(); task3()

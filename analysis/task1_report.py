#!/usr/bin/env python
"""Task 1 tables, figures, paired bootstrap intervals and diagnostics from results/task1/."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pa4.common.utils import (bootstrap_mean_diff, bootstrap_ppl_diff, read_json, read_jsonl, results_root,
                              write_json)
from plotting import CAT, SEQ, color_map, plt, save, write_table

R = results_root() / "task1"
C = R / "conditions"
PROJ = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ORDER = ["fp16", "p1_s25", "p1_s50", "p1_s75", "p2_s25", "p2_s50", "p2_s75", "p3_s50", "p3_s75", "p2_24_s50",
         "q1_b8", "q1_b4", "q1_b3", "q2_b8", "q2_b4", "q2_b3", "q3_b4", "q3_b3"]


def load(name):
    d = C / name
    if not (d / "metrics.json").exists():
        return None
    m = read_json(d / "metrics.json")
    m["_ppl"] = {k: read_json(d / f"ppl_{k}.json") for k in ("wikitext2", "c4val")}
    m["_hs"] = read_jsonl(d / "hellaswag.jsonl")
    return m


def main():
    M = {n: load(n) for n in ORDER}
    M = {k: v for k, v in M.items() if v is not None}
    if "fp16" not in M:
        print("no baseline yet"); return
    base = M["fp16"]
    rows = []
    for n, m in M.items():
        q, st, lat = m["quality"], m["storage"], m["latency"]
        hs_full = q["hellaswag"]["n"] == 2000
        bhs = {s["order"]: s for s in base["_hs"]}
        hs = m["_hs"][:512]
        d_accn = 100 * np.mean([(s["pred_norm"] == s["gold"]) - (bhs[s["order"]]["pred_norm"] == s["gold"]) for s in hs])
        is_quant = n.startswith("q")
        rows.append(dict(
            condition=n,
            # pruning sparsity; for quantized models zeros are weights that merely round to 0 (reported separately)
            sparsity=np.nan if is_quant else m["achieved_sparsity"],
            frac_dequantized_to_zero=m["achieved_sparsity"] if is_quant else np.nan,
            bits_per_weight=st.get("nominal_bits_per_eligible_weight_packed", st["nominal_bits_per_eligible_weight_sparse16"] if n.startswith("p") else 16.0),
            wiki_ppl=q["wikitext2"]["ppl"], d_wiki_ppl=q["wikitext2"]["ppl"] - base["quality"]["wikitext2"]["ppl"],
            c4_ppl=q["c4val"]["ppl"], d_c4_ppl=q["c4val"]["ppl"] - base["quality"]["c4val"]["ppl"],
            wiki_nll=q["wikitext2"]["mean_nll"], c4_nll=q["c4val"]["mean_nll"], N_wiki=q["wikitext2"]["N"], N_c4=q["c4val"]["N"],
            hs_acc=q["hellaswag"]["acc"], hs_acc_norm=q["hellaswag"]["acc_norm"], hs_n=q["hellaswag"]["n"],
            hs512_acc_norm=q["hellaswag_diag512"]["acc_norm"], d_hs512_acc_norm_pp=d_accn,
            dev_nll=m["dev"]["mean_nll"], calib_err=m.get("calib_out_err_rel_mean"),
            dense_fp16_MB=st["dense_fp16_total_bytes"] / 1e6,
            sparse16_MB=np.nan if is_quant else st["sparse16_total_bytes"] / 1e6,
            packed_MB=st.get("packed_total_bytes", np.nan) / 1e6 if st.get("packed_total_bytes") else np.nan,
            compress_s=m.get("compression", {}).get("compression_seconds_total", 0.0),
            prefill512_ms=1000 * lat["prefill_512"]["mean"], prefill2048_ms=1000 * lat["prefill_2048"]["mean"],
            decode64_ms_per_tok=lat["decode_64"]["per_token_ms"], peak_GB=lat["prefill_2048"]["peak_allocated_bytes"] / 1e9,
        ))
    df = pd.DataFrame(rows)
    write_table(df, "task1_all_conditions", caption="Task 1: all conditions (paired deltas vs FP16).")
    df.to_json(R / "summary_table.json", orient="records", indent=1)

    # ---------------- paired bootstraps ----------------
    boots = {}

    def ppl_boot(a, b, suite):
        A, B = M[a]["_ppl"][suite], M[b]["_ppl"][suite]
        return bootstrap_ppl_diff(A["nll"], A["ntok"], B["nll"], B["ntok"])

    def hs_boot(a, b, key="pred_norm"):
        """Paired over every HellaSwag ID scored for both conditions (2,000 if both are full, else the first 512)."""
        A = {s["order"]: s for s in M[a]["_hs"]}
        B = {s["order"]: s for s in M[b]["_hs"]}
        common = sorted(set(A) & set(B))
        return bootstrap_mean_diff([float(A[o][key] == A[o]["gold"]) for o in common],
                                   [float(B[o][key] == B[o]["gold"]) for o in common])

    pairs = [("p3_s50", "p2_s50"), ("p3_s75", "p2_s75"), ("q3_b4", "q2_b4"), ("q3_b3", "q2_b3"),
             ("p2_s50", "p1_s50"), ("q2_b4", "q1_b4"), ("q2_b3", "q1_b3"), ("p2_24_s50", "p2_s50")]
    for a, b in pairs:
        if a in M and b in M:
            boots[f"{a}-{b}"] = dict(wikitext2_ppl=ppl_boot(a, b, "wikitext2"), c4val_ppl=ppl_boot(a, b, "c4val"),
                                     hellaswag_acc_norm=hs_boot(a, b), hellaswag_acc=hs_boot(a, b, "pred"),
                                     hellaswag_n=len(set(s["order"] for s in M[a]["_hs"]) & set(s["order"] for s in M[b]["_hs"])))
    write_json(R / "bootstrap_pairs.json", boots)
    brow = []
    for k, v in boots.items():
        brow.append(dict(pair=k, hellaswag_n=v["hellaswag_n"], **{f"{m}_{s}": v[m][s] for m in ("wikitext2_ppl", "c4val_ppl", "hellaswag_acc_norm")
                                    for s in ("point", "ci_low", "ci_high")}))
    if brow:
        write_table(pd.DataFrame(brow), "task1_bootstrap", caption="Paired 95% bootstrap CIs (2000 resamples).")

    # ---------------- quality vs resource ----------------
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))
    cm = color_map(["P1 magnitude", "P2 Wanda", "P3 adaptive", "2:4 Wanda"])
    ax = axes[0]
    for lab, pre in (("P1 magnitude", "p1_s"), ("P2 Wanda", "p2_s"), ("P3 adaptive", "p3_s")):
        sub = df[df.condition.str.startswith(pre)].sort_values("sparsity")
        if len(sub):
            ax.plot(sub.sparsity, sub.wiki_ppl, marker="o", color=cm[lab], label=lab)
    sub = df[df.condition == "p2_24_s50"]
    if len(sub):
        ax.scatter(sub.sparsity, sub.wiki_ppl, marker="s", color=cm["2:4 Wanda"], label="2:4 Wanda", zorder=3)
    ax.axhline(base["quality"]["wikitext2"]["ppl"], color="#52514e", lw=1, ls="--", label="FP16")
    ax.set_yscale("log"); ax.set_xlabel("eligible-weight sparsity"); ax.set_ylabel("WikiText-2 PPL (log)")
    ax.set_title("Pruning"); ax.legend()
    ax = axes[1]
    cq = color_map(["Q1 RTN", "Q2 AWQ-style", "Q3 mixed"])
    for lab, pre in (("Q1 RTN", "q1_"), ("Q2 AWQ-style", "q2_"), ("Q3 mixed", "q3_")):
        sub = df[df.condition.str.startswith(pre)].sort_values("packed_MB")
        if len(sub):
            ax.plot(sub.packed_MB, sub.wiki_ppl, marker="o", color=cq[lab], label=lab)
    ax.axhline(base["quality"]["wikitext2"]["ppl"], color="#52514e", lw=1, ls="--", label="FP16")
    ax.set_yscale("log"); ax.set_xlabel("total packed model bytes (MB, incl. metadata + FP16 params)")
    ax.set_ylabel("WikiText-2 PPL (log)"); ax.set_title("Weight quantization"); ax.legend()
    fig.tight_layout(); save(fig, "task1", "quality_vs_resource")

    # ---------------- sensitivity & allocation maps ----------------
    pr_path = R / "sensitivity" / "probe.json"
    if pr_path.exists():
        pr = read_json(pr_path)
        units = pr["units"]
        nl = 1 + max(u["layer"] for u in units.values())

        def grid(fn):
            g = np.full((nl, 7), np.nan)
            for u in units.values():
                g[u["layer"], PROJ.index(u["proj"])] = fn(u)
            return g

        i50 = pr["prune_grid"].index(0.5)
        maps = [("Fisher pruning cost @50% (log10)", grid(lambda u: np.log10(u["prune_fisher"][i50] + 1e-30))),
                ("Fisher quant cost @3 bit (log10)", grid(lambda u: np.log10(u["quant"]["3"]["fisher"] + 1e-30))),
                ("Calib. output err @3 bit (log10, rel)", grid(lambda u: np.log10(u["quant"]["3"]["out_err"] / u["out_energy"] + 1e-30)))]
        fig, axes = plt.subplots(1, 4, figsize=(9, 3.2), gridspec_kw=dict(width_ratios=[1, 1, 1, 0.8]))
        for ax, (t, g) in zip(axes[:3], maps):
            im = ax.imshow(g, aspect="auto", cmap=SEQ, origin="lower")
            ax.set_xticks(range(7)); ax.set_xticklabels([p.split("_")[0] for p in PROJ], rotation=90)
            ax.set_ylabel("block"); ax.set_title(t, fontsize=7); ax.grid(False)
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            fig.colorbar(im, ax=ax, fraction=0.05)
        ax = axes[3]
        for j, M_ in enumerate(pr["owl_M"]):
            ax.plot([pr["blocks"][str(i)]["owl"][str(M_)] for i in range(nl)], range(nl), color=CAT[j], label=f"M={M_}")
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlabel("OWL outlier ratio"); ax.set_ylabel("block"); ax.legend(); ax.set_title("OWL statistic", fontsize=7)
        fig.tight_layout(); save(fig, "task1", "sensitivity_maps")
        # magnitude vs Wanda overlap
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        cs = color_map(["0.25", "0.5", "0.75"])
        for s in ("0.25", "0.5", "0.75"):
            ax.plot(range(7), [np.mean([u["mag_wanda_jaccard"][s] for u in units.values() if u["proj"] == p]) for p in PROJ],
                    marker="o", color=cs[s], label=f"s={s}")
        ax.set_xticks(range(7)); ax.set_xticklabels([p.split("_")[0] for p in PROJ])
        ax.set_ylabel("Jaccard(pruned sets)"); ax.set_title("Magnitude vs Wanda masks"); ax.legend()
        fig.tight_layout(); save(fig, "task1", "mag_wanda_overlap")

    # final allocation maps
    al = R / "allocations"
    fig, axes = plt.subplots(1, 4, figsize=(9, 3.2))
    for ax, f, key, t in zip(axes, ["p3_s50", "p3_s75", "q3_b4", "q3_b3"],
                             ["unit_sparsity", "unit_sparsity", "bits", "bits"],
                             ["P3 sparsity @50%", "P3 sparsity @75%", "Q3 bits (INT4 budget)", "Q3 bits (INT3 budget)"]):
        p = al / f"{f}_selected.json"
        if not p.exists():
            ax.axis("off"); continue
        a = read_json(p)
        vals = a[key]
        nl = 1 + max(int(u.split(".")[1]) for u in vals)
        g = np.zeros((nl, 7))
        for u, v in vals.items():
            g[int(u.split(".")[1]), PROJ.index(u.split(".")[2])] = v
        im = ax.imshow(g, aspect="auto", cmap=SEQ, origin="lower")
        ax.set_xticks(range(7)); ax.set_xticklabels([p_.split("_")[0] for p_ in PROJ], rotation=90)
        ax.set_title(f"{t}\n[{a['design']}]", fontsize=7); ax.grid(False)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True)); ax.set_ylabel("block")
        fig.colorbar(im, ax=ax, fraction=0.05)
    fig.tight_layout(); save(fig, "task1", "allocation_maps")

    # matched-budget checks
    checks = {}
    for s in ("50", "75"):
        if f"p3_s{s}" in M and f"p2_s{s}" in M:
            checks[f"s{s}"] = dict(p3_zeros=M[f"p3_s{s}"]["zeros"], p2_zeros=M[f"p2_s{s}"]["zeros"],
                                   p1_zeros=M.get(f"p1_s{s}", {}).get("zeros"),
                                   equal=M[f"p3_s{s}"]["zeros"] == M[f"p2_s{s}"]["zeros"])
    for b in ("4", "3"):
        if f"q3_b{b}" in M and f"q2_b{b}" in M:
            q3, q2 = M[f"q3_b{b}"]["storage"]["packed_total_bytes"], M[f"q2_b{b}"]["storage"]["packed_total_bytes"]
            checks[f"b{b}"] = dict(q3_total_bytes=q3, q2_total_bytes=q2, within_budget=q3 <= q2, slack_bytes=q2 - q3)
    write_json(R / "matched_budget_checks.json", checks)

    # calibration criterion vs held-out quality (all conditions + dev candidates)
    pts = [(r.dev_nll, r.calib_err, np.log(r.wiki_ppl), np.log(r.c4_ppl), r.hs512_acc_norm, r.condition)
           for r in df.itertuples() if r.condition != "fp16" and r.calib_err is not None]
    diag = {}
    if len(pts) > 3:
        a = np.array([p[:5] for p in pts], float)
        diag["spearman_dev_nll_vs_log_wiki_ppl"] = spearmanr(a[:, 0], a[:, 2]).correlation
        diag["spearman_dev_nll_vs_hs_acc_norm"] = spearmanr(a[:, 0], a[:, 4]).correlation
        diag["spearman_calib_err_vs_log_wiki_ppl"] = spearmanr(a[:, 1], a[:, 2]).correlation
        diag["spearman_calib_err_vs_log_c4_ppl"] = spearmanr(a[:, 1], a[:, 3]).correlation
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5))
        fam = color_map(["P1", "P2", "P3", "2:4", "Q1", "Q2", "Q3"])

        def famof(n):
            return "2:4" if "24" in n else n[:2].upper()

        for ax, xi, xl in ((axes[0], 1, "mean rel. layer-output error (calibration)"), (axes[1], 0, "dev mean NLL (32 C4 seqs)")):
            for f in fam:
                sel = [p for p in pts if famof(p[5]) == f]
                if sel:
                    ax.scatter([p[xi] for p in sel], [np.exp(p[2]) for p in sel], color=fam[f], label=f, s=14)
            ax.set_xlabel(xl); ax.set_ylabel("WikiText-2 PPL"); ax.set_yscale("log")
            if xi == 1:
                ax.set_xscale("log")
        axes[0].legend(ncol=2)
        fig.tight_layout(); save(fig, "task1", "calib_vs_heldout")
    sel = al / "selection_summary.json"
    if sel.exists():
        diag["selection"] = read_json(sel)
    write_json(R / "calib_vs_heldout.json", diag)

    # qualitative HellaSwag cases: first 4 diagnostic IDs + first 4 where FP16 is right and P2@50% wrong
    qual = []
    if "p2_s50" in M:
        b = {s["order"]: s for s in base["_hs"]}
        p2 = {s["order"]: s for s in M["p2_s50"]["_hs"]}
        flips = [o for o in sorted(p2) if b[o]["pred_norm"] == b[o]["gold"] and p2[o]["pred_norm"] != p2[o]["gold"]][:4]
        from pa4.common import data as D
        try:
            hs_rows = {r["order"]: r for r in D.hellaswag()}
        except FileNotFoundError:
            hs_rows = {}
        for o in [0, 1, 2, 3] + flips:
            if o not in hs_rows:
                continue
            ex = hs_rows[o]["example"]
            entry = dict(order=o, selection="fixed first-4" if o < 4 else "FP16 correct, P2@50% wrong",
                         ctx=ex["ctx"], endings=ex["endings"], gold=int(ex["label"]), configs={})
            for n, m in M.items():
                s = {x["order"]: x for x in m["_hs"]}.get(o)
                if s:
                    entry["configs"][n] = dict(ll=[round(v, 2) for v in s["ll"]], pred_norm=s["pred_norm"])
            qual.append(entry)
    cont = {n: m["quality"].get("continuations") for n, m in M.items()}
    write_json(R / "qualitative.json", dict(hellaswag=qual, continuations=cont))
    print(df.to_string())


if __name__ == "__main__":
    main()

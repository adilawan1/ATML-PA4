#!/usr/bin/env python
"""Task 3 tables/figures: sampler checks, distillation, prefix-bank alignment, speculative execution, costs."""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from pa4.common.utils import bootstrap_mean_diff, paired_bootstrap, read_json, read_jsonl, results_root, write_json
from plotting import color_map, plt, save, write_table

R = results_root() / "task3"
DRAFTS = ["original", "off_s6304", "off_s6305", "on_s6304", "on_s6305"]


def sampler():
    f = R / "checks" / "sampler_checks.json"
    if not f.exists():
        return
    c = read_json(f)
    rows = [dict(test="Markov gamma=2 exact", chi2=c["markov_gamma2"]["exact"]["chi2"], p=c["markov_gamma2"]["exact"]["p_value"]),
            dict(test="Markov gamma=2 residual:=p (neg. control)", chi2=c["markov_gamma2"]["p"]["chi2"], p=c["markov_gamma2"]["p"]["p_value"]),
            dict(test="Markov gamma=1 exact (20k)", chi2=c["markov_gamma1"]["exact"]["chi2"], p=c["markov_gamma1"]["exact"]["p_value"]),
            dict(test="Markov+EOS gamma=3 exact", chi2=c["markov_eos_gamma3"]["chi2"], p=c["markov_eos_gamma3"]["p_value"])]
    write_table(pd.DataFrame(rows), "task3_sampler_checks")
    tab = pd.DataFrame(c["markov_gamma2"]["exact"]["table"])
    tab["empirical_wrong_residual"] = [r["empirical"] for r in c["markov_gamma2"]["p"]["table"]]
    write_table(tab, "task3_markov_joint_table", caption="Exact joint law vs empirical (100k sequences).")


def distillation():
    d = R / "distill"
    res = {n: read_json(d / n / "result.json") for n in DRAFTS[1:] if (d / n / "result.json").exists()}
    if not res:
        return
    cm = color_map(list(res))
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for n, r in res.items():
        x = [l["scored_tokens"] for l in r["logs"]]
        y = pd.Series([l["kl_per_token"] for l in r["logs"]]).rolling(10, min_periods=1).mean()
        ax.plot(x, y, color=cm[n], label=n, ls="-" if n.startswith("on") else "--")
    ax.set_xlabel("teacher-scored response tokens"); ax.set_ylabel("forward KL / token (10-update mean)")
    ax.set_title("Training KL on each method's own training prefixes\n(teacher samples vs student samples: not comparable across methods)", fontsize=7)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2)); fig.tight_layout(); save(fig, "task3", "distill_curves")
    pc = read_json(d / "teacher_pool_cost.json") if (d / "teacher_pool_cost.json").exists() else {}
    rows = []
    for n, r in res.items():
        m = read_json(d / n / "merge.json") if (d / n / "merge.json").exists() else {}
        c = r["costs"]
        total = r["wall_s"] + (pc.get("seconds", 0) if n.startswith("off") else 0) + m.get("merge_seconds", 0)
        rows.append(dict(run=n, updates=r["updates"], skipped=r["skipped_updates"], adapter_params=r["n_adapter_params"],
                         math_tokens=r["tokens"]["math"], short_tokens=r["tokens"]["short"], presentations=r["n_presentations"],
                         final_kl=np.mean([l["kl_per_token"] for l in r["logs"][-10:]]),
                         teacher_pool_s=pc.get("seconds", 0) if n.startswith("off") else 0,
                         student_rollout_s=c["student_rollout_s"], teacher_scoring_s=c["teacher_scoring_s"],
                         student_fwd_bwd_s=c["student_fwd_bwd_s"], optimizer_s=c["optimizer_s"], train_wall_s=r["wall_s"],
                         merge_s=m.get("merge_seconds"), merge_max_abs_logit_err=m.get("max_abs_logit_err"),
                         total_cost_s=total, peak_GB=(r["peak_alloc_bytes"] or 0) / 1e9,
                         mean_math_rollout_len=np.mean(r["rollout_lengths"]["math"]) if r["rollout_lengths"]["math"] else None))
    df = pd.DataFrame(rows)
    write_table(df, "task3_distill_costs")
    return df


def alignment():
    f = R / "align" / "positions.jsonl.gz"
    if not f.exists():
        return
    df = pd.DataFrame(read_jsonl(f))
    df["length"] = np.where(df.position <= 64, "short(1-64)", "long(65-192)")
    met = ["fkl", "rkl", "tv", "exp_accept", "greedy_agree"]
    t = df.groupby(["bank", "split", "length", "draft"])[met].mean().reset_index()
    write_table(t, "task3_alignment_full", caption="Token-weighted means over identical prefix positions.")
    # main table: per bank, averaged over seeds for trained drafts, with seed dispersion
    df["method"] = df.draft.str.replace(r"_s\d+", "", regex=True)
    seedm = df.groupby(["bank", "method", "draft"])[met].mean().reset_index()
    main = seedm.groupby(["bank", "method"])[met].agg(["mean", "std"]).reset_index()
    main.columns = ["_".join([c for c in col if c]) for col in main.columns]
    write_table(main, "task3_alignment_main")
    pb = df[df.length.str.startswith("short")].groupby(["pos_bin", "method"])["exp_accept"].mean().unstack()
    write_table(pb.reset_index(), "task3_alignment_by_position")
    # consistency: teacher bank vs exact speculative bank (same target-prefix law)
    cons = {}
    for b in ("teacher", "exact_spec_original_gamma4"):
        s = df[(df.bank == b) & (df.draft == "original")]
        cons[b] = dict(n_positions=len(s), mean_target_logp_of_sampled=float(s.target_logp_sampled.dropna().mean()),
                       exp_accept_original=float(s.exp_accept.mean()), fkl_original=float(s.fkl.mean()))
    a = df[(df.bank == "teacher") & (df.draft == "original")].groupby("prompt_id").exp_accept.mean()
    b = df[(df.bank == "exact_spec_original_gamma4") & (df.draft == "original")].groupby("prompt_id").exp_accept.mean()
    common = a.index.intersection(b.index)
    if len(common):
        cons["prompt_paired_diff_exp_accept"] = bootstrap_mean_diff(b[common].values, a[common].values)
    write_json(R / "align" / "consistency.json", cons)
    # figure: expected acceptance by bank and method
    banks = list(dict.fromkeys(df.bank))
    methods = ["original", "off", "on"]
    cm = color_map(methods)
    fig, ax = plt.subplots(figsize=(7, 2.6))
    x = np.arange(len(banks))
    for i, m in enumerate(methods):
        v = [seedm[(seedm.bank == bk) & (seedm.method == m)].exp_accept.mean() for bk in banks]
        ax.bar(x + (i - 1) * 0.27, v, width=0.25, color=cm[m], label=m)
    ax.set_xticks(x); ax.set_xticklabels([b.replace("own_", "own:") for b in banks], rotation=20, fontsize=6)
    ax.set_ylabel("E[1-TV] (expected 1-token acceptance)"); ax.set_ylim(0, 1); ax.legend()
    fig.tight_layout(); save(fig, "task3", "alignment_banks")
    # aligned disagreements (largest TV on teacher bank, original vs on-policy seed 6304)
    s = df[(df.bank == "teacher") & (df.draft.isin(["original", "on_s6304"]))]
    piv = s.pivot_table(index=["prompt_id", "position"], columns="draft", values="tv").dropna()
    piv["delta"] = piv["original"] - piv["on_s6304"]
    write_table(piv.sort_values("delta").head(5).reset_index().rename_axis(None, axis=1), "task3_prefix_disagreements")
    return df


def costmodel(gamma, c, v):
    def S(a):
        return (1 + sum(a ** j for j in range(1, gamma + 1))) / (gamma * c + v)

    f = lambda a: 1 + sum(a ** j for j in range(1, gamma + 1)) - (gamma * c + v)
    be = brentq(f, 0, 1) if f(0) < 0 < f(1) else (0.0 if f(0) >= 0 else None)
    return S, be


def speculative(align_df=None):
    sd = R / "spec"
    files = sorted(sd.glob("records_*.jsonl"))
    if not files:
        return
    recs = []
    for f in files:
        recs += read_jsonl(f)
    sc = read_json(sd / "step_costs.json") if (sd / "step_costs.json").exists() else None
    rows, boot = [], {}
    by = defaultdict(list)
    for r in recs:
        by[(r["tag"], r["gamma"])].append(r)
    for (tag, g), rs in sorted(by.items()):
        sp = [r["spec"] for r in rs]
        tg = [r["target"] for r in rs]
        acc = np.array([s["accepted"] for s in sp]); exam = np.array([s["examined"] for s in sp])
        prop = np.array([s["proposed"] for s in sp]); rounds = np.array([s["rounds"] for s in sp])
        nout = np.array([s["n_out"] for s in sp])
        # per-prompt paired throughput ratio, averaged over the two repetitions first
        pp = defaultdict(list)
        for r in rs:
            pp[r["prompt_id"]].append((r["spec"]["tokens_per_s"], r["target"]["tokens_per_s"]))
        ratio = np.array([np.mean([a / b for a, b in v]) for v in pp.values()])
        boot[f"{tag}_g{g}_throughput_ratio"] = paired_bootstrap(lambda idx: float(ratio[idx].mean()), len(ratio))
        # decode-only per-token latency ratio (target / speculative), excluding the one-off prefill: robust to the
        # output-length mismatch between the independent speculative and target-only samples of a prompt
        dd = defaultdict(list)
        for r in rs:
            dd[r["prompt_id"]].append((r["target"]["t_decode"] / r["target"]["n_out"]) /
                                      ((r["spec"]["t_total"] - r["spec"]["t_prefill"]) / r["spec"]["n_out"]))
        dratio = np.array([np.mean(v) for v in dd.values()])
        boot[f"{tag}_g{g}_decode_speedup"] = paired_bootstrap(lambda idx: float(dratio[idx].mean()), len(dratio))
        acc_pp = np.array([np.mean([x["spec"]["accepted"] / max(x["spec"]["examined"], 1) for x in rs if x["prompt_id"] == p]) for p in pp])
        boot[f"{tag}_g{g}_accept_examined"] = paired_bootstrap(lambda idx: float(acc_pp[idx].mean()), len(acc_pp))
        dec = {k: np.mean([s[k] for s in sp]) * 1000 for k in ("t_prefill", "t_draft", "t_verify", "t_accept", "t_rollback", "t_host_other", "t_total")}
        row = dict(draft=tag, gamma=g, n=len(rs), accept_examined=acc.sum() / exam.sum(), accept_proposed=acc.sum() / prop.sum(),
                   mean_accepted_prefix=acc.sum() / rounds.sum(), target_passes_per_token=sum(s["target_calls"] for s in sp) / nout.sum(),
                   draft_passes_per_token=sum(s["draft_calls"] for s in sp) / nout.sum(),
                   spec_tok_s=np.mean([s["tokens_per_s"] for s in sp]), target_tok_s=np.mean([t["tokens_per_s"] for t in tg]),
                   throughput_ratio=ratio.mean(), ratio_ci=(boot[f"{tag}_g{g}_throughput_ratio"]["ci_low"], boot[f"{tag}_g{g}_throughput_ratio"]["ci_high"]),
                   decode_speedup=dratio.mean(), decode_speedup_ci_low=boot[f"{tag}_g{g}_decode_speedup"]["ci_low"],
                   decode_speedup_ci_high=boot[f"{tag}_g{g}_decode_speedup"]["ci_high"],
                   eos_rate=np.mean([s["eos"] for s in sp]), truncated_rate=np.mean([s["truncated"] for s in sp]),
                   peak_spec_GB=np.mean([s["peak_alloc_bytes"] for s in sp]) / 1e9, peak_target_GB=np.mean([t["peak_alloc_bytes"] for t in tg]) / 1e9,
                   **{f"{k}_ms": v for k, v in dec.items()})
        # measured cost-model terms from the per-round timing decomposition
        n_rounds = rounds.sum()
        t_step_target = np.sum([t["t_decode"] for t in tg]) / sum(t["target_calls"] for t in tg)
        c = (np.sum([s["t_draft"] for s in sp]) / max(sum(s["draft_calls"] for s in sp), 1)) / t_step_target
        v = (np.sum([s["t_verify"] for s in sp]) / n_rounds) / t_step_target
        a = row["accept_examined"]
        S, be = costmodel(g, c, v)
        row.update(c=c, v_gamma=v, predicted_S=S(a), measured_S=row["spec_tok_s"] / row["target_tok_s"], break_even_alpha=be)
        rows.append(row)
        # by workload split / output-length bins
    df = pd.DataFrame(rows)
    write_table(df, "task3_spec_main", caption="Speculative decoding (EOS-aware, cap 64, 120 prompts x 2 reps).")
    # paired comparisons between drafts over prompts (repetitions averaged within prompt first)
    def per_prompt(tag, g, fn):
        d = defaultdict(list)
        for r in by.get((tag, g), []):
            d[r["prompt_id"]].append(fn(r))
        return {k: float(np.mean(v)) for k, v in d.items()}

    ratio_fn = lambda r: r["spec"]["tokens_per_s"] / r["target"]["tokens_per_s"]
    acc_fn = lambda r: r["spec"]["accepted"] / max(r["spec"]["examined"], 1)
    comps = []
    for g in sorted({k[1] for k in by}):
        pairs = [(d, "original") for d in DRAFTS[1:]] + [(f"on_s{s_}", f"off_s{s_}") for s_ in (6304, 6305)]
        for a, b in pairs:
            for metric, fn in (("throughput_ratio", ratio_fn), ("accept_examined", acc_fn)):
                A, B = per_prompt(a, g, fn), per_prompt(b, g, fn)
                common = sorted(set(A) & set(B))
                if len(common) < 2:
                    continue
                bt = bootstrap_mean_diff([A[k] for k in common], [B[k] for k in common], 2000)
                boot[f"{a}-{b}_g{g}_{metric}"] = bt
                comps.append(dict(gamma=g, comparison=f"{a} - {b}", metric=metric, n_prompts=len(common),
                                  diff=bt["point"], ci_low=bt["ci_low"], ci_high=bt["ci_high"]))
    if comps:
        write_table(pd.DataFrame(comps), "task3_spec_paired_comparisons",
                    caption="Paired 95% bootstrap CIs over prompts (2000 resamples); training seeds reported separately.")
    write_json(sd / "bootstrap.json", boot)
    wl = []
    for (tag, g), rs in sorted(by.items()):
        for split in ("math", "familiar", "shifted"):
            s = [r for r in rs if r["split"] == split]
            if not s:
                continue
            wl.append(dict(draft=tag, gamma=g, split=split, n=len(s),
                           accept_examined=sum(r["spec"]["accepted"] for r in s) / max(sum(r["spec"]["examined"] for r in s), 1),
                           target_passes_per_token=sum(r["spec"]["target_calls"] for r in s) / sum(r["spec"]["n_out"] for r in s),
                           throughput_ratio=np.mean([r["speedup"] for r in s]),
                           mean_out_len=np.mean([r["spec"]["n_out"] for r in s])))
        for lo, hi in ((1, 8), (9, 32), (33, 64)):
            s = [r for r in rs if lo <= r["spec"]["n_out"] <= hi]
            if s:
                wl.append(dict(draft=tag, gamma=g, split=f"len[{lo},{hi}]", n=len(s),
                               accept_examined=sum(r["spec"]["accepted"] for r in s) / max(sum(r["spec"]["examined"] for r in s), 1),
                               target_passes_per_token=sum(r["spec"]["target_calls"] for r in s) / sum(r["spec"]["n_out"] for r in s),
                               throughput_ratio=np.mean([r["speedup"] for r in s])))
    write_table(pd.DataFrame(wl), "task3_spec_workload")
    # expected (bank) vs measured acceptance
    if align_df is not None:
        exp = align_df[align_df.bank == "teacher"].groupby("draft").exp_accept.mean()
        sub = df[df.draft.isin(DRAFTS)]
        cm = color_map(DRAFTS)
        fig, ax = plt.subplots(figsize=(3.6, 2.6))
        for g, mk in ((2, "o"), (4, "s")):
            for d in DRAFTS:
                q = sub[(sub.draft == d) & (sub.gamma == g)]
                if len(q) and d in exp:
                    ax.scatter(exp[d], q.accept_examined.iloc[0], color=cm[d], marker=mk, label=f"{d} γ={g}")
        xs = [exp[d] for d in DRAFTS if d in exp]
        ys = sub.accept_examined.tolist()
        lo, hi = min(xs + ys) - 0.01, max(xs + ys) + 0.01
        ax.plot([lo, hi], [lo, hi], color="#52514e", lw=0.8, ls="--", label="y = x")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("expected acceptance E[1-TV] (teacher bank)"); ax.set_ylabel("measured accepted/examined")
        ax.legend(fontsize=5); fig.tight_layout(); save(fig, "task3", "expected_vs_measured_acceptance")
    # time decomposition figure
    fig, ax = plt.subplots(figsize=(6, 2.6))
    comp = ["t_prefill_ms", "t_draft_ms", "t_verify_ms", "t_accept_ms", "t_rollback_ms", "t_host_other_ms"]
    cm = color_map(comp)
    lab = [f"{r.draft} γ{r.gamma}" for r in df.itertuples()]
    bottom = np.zeros(len(df))
    for c_ in comp:
        ax.bar(range(len(df)), df[c_], bottom=bottom, color=cm[c_], label=c_[2:-3], width=0.7,
               edgecolor="white", linewidth=1)
        bottom += df[c_].values
    ax.set_xticks(range(len(df))); ax.set_xticklabels(lab, rotation=60, fontsize=5); ax.set_ylabel("ms per request (cap 64)")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=6, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.0)); fig.tight_layout(); save(fig, "task3", "spec_time_decomposition")
    return df


def standalone_and_breakeven(spec_df, dist_df):
    sd = R / "standalone"
    rows = []
    for f in sorted(sd.glob("*.jsonl")):
        rs = read_jsonl(f)
        row = dict(model=f.stem)
        for split in ("math", "familiar", "shifted"):
            s = [r for r in rs if r["split"] == split]
            row[f"{split}_acc"] = np.mean([r["correct"] for r in s])
            if split == "math":
                row["math_extractable"] = np.mean([r["extractable"] for r in s])
                row["math_truncated"] = np.mean([r["truncated"] for r in s])
            else:
                row[f"{split}_format_ok"] = np.mean([r["format_ok"] for r in s])
                # lenient: the gold answer appears as a whole word anywhere in the response (separates knowing the
                # answer from complying with the "return only ..." format; exact match requires both)
                row[f"{split}_answer_present"] = np.mean([bool(re.search(r"(?<![\w])" + re.escape(r["gold"]) + r"(?![\w])", r["text"]))
                                                          for r in s])
        rows.append(row)
    if rows:
        write_table(pd.DataFrame(rows), "task3_standalone_accuracy")
    if spec_df is None or dist_df is None:
        return
    be = []
    for r in spec_df.itertuples():
        if r.draft not in DRAFTS[1:]:
            continue
        save_ms = (1000 * 64 / r.target_tok_s) - (1000 * 64 / r.spec_tok_s)  # per 64-token request, measured
        cost = float(dist_df[dist_df.run == r.draft].total_cost_s.iloc[0]) * 1000
        orig = spec_df[(spec_df.draft == "original") & (spec_df.gamma == r.gamma)]
        save_vs_orig = (1000 * 64 / orig.spec_tok_s.iloc[0]) - (1000 * 64 / r.spec_tok_s) if len(orig) else np.nan
        be.append(dict(draft=r.draft, gamma=r.gamma, saving_ms_per_req_vs_target=save_ms,
                       saving_ms_per_req_vs_original_draft=save_vs_orig, distill_cost_ms=cost,
                       break_even_requests_vs_original=cost / save_vs_orig if save_vs_orig and save_vs_orig > 0 else None))
    write_table(pd.DataFrame(be), "task3_deployment_break_even")


def continuations():
    """Matched-prompt continuations (same prompt, gamma=4, rep 0) for three fixed prompts: first math, first
    familiar, first shifted ID. Decoded only if the tokenizer can be loaded."""
    files = sorted((R / "spec").glob("records_*.jsonl"))
    if not files:
        return
    recs = [r for f in files for r in read_jsonl(f) if r["gamma"] == 4 and r["rep"] == 0 and r["tag"] in DRAFTS]
    chosen = []
    for split in ("math", "familiar", "shifted"):
        ids = sorted({r["prompt_id"] for r in recs if r["split"] == split})
        if ids:
            chosen.append(ids[0])
    try:
        from transformers import AutoTokenizer

        from pa4.common.utils import model_revision
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", revision=model_revision("Qwen/Qwen2.5-1.5B-Instruct"))
        dec = lambda ids: tok.decode(ids, skip_special_tokens=False)
    except Exception:
        dec = lambda ids: ids
    out = []
    for pid in chosen:
        for r in recs:
            if r["prompt_id"] == pid:
                out.append(dict(prompt_id=pid, split=r["split"], draft=r["tag"], gamma=4,
                                speculative_output=dec(r["spec"]["out_ids"]), target_only_output=dec(r["target"]["out_ids"]),
                                accepted=r["spec"]["accepted"], examined=r["spec"]["examined"], eos=r["spec"]["eos"],
                                note="both are independent samples from the target distribution (same seed, different sampler)"))
    write_json(R / "spec" / "continuations.json", dict(selection="first ID of each split, gamma=4, repetition 0", rows=out))


def main():
    sampler()
    continuations()
    dist_df = distillation()
    align_df = alignment()
    spec_df = speculative(align_df)
    standalone_and_breakeven(spec_df, dist_df)


if __name__ == "__main__":
    main()

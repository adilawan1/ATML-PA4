#!/usr/bin/env python
"""Task 1 driver: pruning (P1/P2/P3, 2:4) and weight quantization (Q1/Q2/Q3) of Qwen2.5-1.5B.

Pipeline (each step is resumable; finished conditions are skipped unless --force):
  python scripts/task1.py baseline              # FP16 reference, enumeration tests, resources
  python scripts/task1.py uniform               # P1, P2 (25/50/75%), 2:4 Wanda, Q1, Q2 (8/4/3 bit)
  python scripts/task1.py sensitivity           # Fisher + dense probe pass (calibration set only)
  python scripts/task1.py candidates            # P3/Q3 candidate allocations (fixed before any test result)
  python scripts/task1.py select                # choose design per budget on the 32-seq C4 dev set
  python scripts/task1.py adaptive              # final P3 (50/75%) and Q3 (INT4/INT3 budgets)
  python scripts/task1.py all                   # everything above in order
  python scripts/task1.py run --cond p2_s50     # a single condition
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from pa4.common import data as D
from pa4.common.utils import (REF_SEED, Timer, environment_record, out_dir, read_json, set_seed, write_json,
                              write_jsonl)
from pa4.task1 import allocate as A
from pa4.task1.awq import quantize_model, uniform_bits
from pa4.task1.calib import calib_tensor
from pa4.task1.evaluate import (HELLASWAG_PROMPT, hellaswag_scores, hellaswag_summary, ppl_summary, window_nll)
from pa4.task1.model_utils import (BASE_MODEL, WeightSnapshot, count_nonzeros, eligible_linears, load_model,
                                   parameter_account, per_unit_sparsity, verify_enumeration)
from pa4.task1.prune import prune_model, uniform_row_counts
from pa4.task1.quant import packed_bytes
from pa4.task1.resources import latency_report, storage_report

FULL_HS = {"fp16", "p1_s50", "p2_s50", "p3_s50", "q1_b4", "q1_b3", "q2_b4", "q2_b3", "q3_b4", "q3_b3"}
UNIFORM = ["p1_s25", "p1_s50", "p1_s75", "p2_s25", "p2_s50", "p2_s75", "p2_24_s50",
           "q1_b8", "q1_b4", "q1_b3", "q2_b8", "q2_b4", "q2_b3"]
ADAPTIVE = ["p3_s50", "p3_s75", "q3_b4", "q3_b3"]
QUAL_PROMPTS = 2  # fixed: first 64 tokens of the first two WikiText-2 windows


class Ctx:
    def __init__(self, args):
        self.args = args
        set_seed(REF_SEED)
        torch.backends.cuda.matmul.allow_tf32 = True
        self.model, self.tok, self.rev = load_model(BASE_MODEL)
        self.snap = WeightSnapshot(self.model)
        self.calib_rows = D.c4_calibration()
        self.dev_rows = D.c4_development()
        self.calib = calib_tensor(self.calib_rows)
        self.windows = {k: D.eval_windows(k) for k in ("wikitext2", "c4val")}
        self.hs = D.hellaswag()
        self.root = out_dir("task1")
        self.env_extra = dict(model=BASE_MODEL, model_revision=self.rev, tokenizer=f"{BASE_MODEL}@{self.rev}",
                              precision="float16", attn_implementation=self.model.config._attn_implementation)
        self._save_selection()

    def _save_selection(self):
        p = self.root / "data_selection.json"
        if p.exists():
            return
        cal = {r["source_row"] for r in self.calib_rows}
        dev = {r["source_row"] for r in self.dev_rows}
        assert not (cal & dev), "calibration/development overlap"
        write_json(p, dict(
            model=BASE_MODEL, model_revision=self.rev, seed=REF_SEED,
            calibration_source_rows=[r["source_row"] for r in self.calib_rows],
            development_source_rows=[r["source_row"] for r in self.dev_rows],
            calibration_dev_disjoint=True,
            eval_windows={k: [dict(source_row=w["source_row"], token_offset=w["token_offset"], scored=w["scored_tokens"])
                              for w in v] for k, v in self.windows.items()},
            hellaswag_orders=[r["order"] for r in self.hs], hellaswag_source_rows=[r["source_row_index"] for r in self.hs],
            hellaswag_diagnostic_subset="first 512 of the supplied 2,000 IDs", hellaswag_format=HELLASWAG_PROMPT,
            env=environment_record(self.env_extra)))

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, name: str, d: Path) -> dict:
        m = self.model
        q = {}
        for k, wins in self.windows.items():
            r = window_nll(m, wins, batch_size=self.args.ppl_bs, pad_id=self.tok.pad_token_id or 0)
            write_json(d / f"ppl_{k}.json", r)
            q[k] = ppl_summary(r)
        hs_rows = self.hs if name in FULL_HS else self.hs[:512]
        sc = hellaswag_scores(m, self.tok, hs_rows, batch_size=self.args.hs_bs)
        write_jsonl(d / "hellaswag.jsonl", sc)
        q["hellaswag"] = hellaswag_summary(sc)
        q["hellaswag_diag512"] = hellaswag_summary(sc[:512])
        q["continuations"] = self.continuations()
        return q

    @torch.no_grad()
    def continuations(self):
        out = []
        dev = next(self.model.parameters()).device
        for w in self.windows["wikitext2"][:QUAL_PROMPTS]:
            ids = torch.tensor([w["token_ids"][:64]], device=dev)
            g = self.model.generate(ids, max_new_tokens=32, do_sample=False, pad_token_id=self.tok.eos_token_id)
            out.append(dict(source_row=w["source_row"], prompt=self.tok.decode(ids[0]),
                            continuation=self.tok.decode(g[0, 64:])))
        return out

    @torch.no_grad()
    def dev_nll(self) -> dict:
        wins = [dict(token_ids=r["token_ids"], score_target_offsets=[1, 512], scored_tokens=511) for r in self.dev_rows]
        r = window_nll(self.model, wins, batch_size=self.args.ppl_bs)
        return dict(**ppl_summary(r), per_seq=r["nll"])

    # ------------------------------------------------------------------ conditions
    def run(self, name: str, force: bool = False, dev_only: bool = False, alloc=None) -> dict:
        d = out_dir("task1", "dev_candidates" if dev_only else "conditions", name)
        if (d / "metrics.json").exists() and not force:
            print(f"[skip] {name}")
            return read_json(d / "metrics.json")
        print(f"=== {name} ===", flush=True)
        self.snap.restore(self.model)
        set_seed(REF_SEED)
        torch.cuda.empty_cache()
        info, quant_bytes, params = self.compress(name, alloc)
        nnz, tot = count_nonzeros(self.model)
        rec = dict(condition=name, params=params, achieved_nonzero_fraction=nnz / tot, achieved_sparsity=1 - nnz / tot,
                   n_eligible=tot, zeros=tot - nnz)
        rec["compression"] = {k: v for k, v in info.items() if k not in ("units", "per_block", "groups", "score_summary")}
        units = info.get("units", {})
        if units:
            rel = [u["out_err_rel"] for u in units.values()]
            rec["calib_out_err_rel_mean"] = float(np.mean(rel))
            rec["calib_out_err_rel_median"] = float(np.median(rel))
            write_json(d / "units.json", units)
        if "per_block" in info:
            write_json(d / "blocks.json", info["per_block"])
        if "groups" in info:
            write_json(d / "awq_groups.json", info["groups"])
        if "score_summary" in info:
            write_json(d / "score_summary.json", info["score_summary"])
        rec["dev"] = self.dev_nll()
        if not dev_only:
            rec["quality"] = self.evaluate(name, d)
            rec["storage"] = storage_report(self.model, quant_bytes)
            rec["latency"] = latency_report(self.model, reps=self.args.reps)
        rec["env"] = environment_record(self.env_extra)
        write_json(d / "metrics.json", rec)
        return rec

    def compress(self, name: str, alloc=None):
        m = self.model
        elig = eligible_linears(m)
        shapes = {u: tuple(l.weight.shape) for u, l in elig.items()}
        bs = self.args.calib_bs
        if name == "fp16":
            return {}, None, {}
        with Timer() as t:
            if name.startswith(("p1_", "p2_")):
                method = "magnitude" if name.startswith("p1") else "wanda"
                s = int(name.split("_s")[-1]) / 100
                structure = "2:4" if "_24_" in name else "unstructured"
                rc = {u: uniform_row_counts(shapes[u], s) for u in elig}
                info = prune_model(m, self.calib, method, rc, structure, bs, keep_scores=(method == "wanda"))
                params = dict(method=method, sparsity=s, structure=structure, per="output row")
                qb = None
            elif name.startswith("p3_") or name.startswith("devp3_"):
                s = int(name.split("_s")[-1].split("_")[0]) / 100
                alloc = alloc or read_json(self.root / "allocations" / f"p3_s{int(s*100)}_selected.json")
                rc = {u: torch.tensor(v, dtype=torch.long) for u, v in alloc["row_counts"].items()}
                info = prune_model(m, self.calib, "wanda", rc, "unstructured", bs)
                params = dict(method="wanda", sparsity=s, allocation=alloc["design"], unit=alloc["unit"])
                qb = None
            elif name.startswith(("q1_", "q2_")):
                b = int(name.split("_b")[-1])
                method = "rtn" if name.startswith("q1") else "awq"
                info = quantize_model(m, self.calib, uniform_bits(m, b), method, batch_size=bs)
                params = dict(method=method, bits=b, group=128, scale="fp16", zero="uint8", rounding="half-to-even")
                qb = info["eligible_packed_bytes"]
            elif name.startswith("q3_") or name.startswith("devq3_"):
                b = int(name.split("_b")[-1].split("_")[0])
                alloc = alloc or read_json(self.root / "allocations" / f"q3_b{b}_selected.json")
                info = quantize_model(m, self.calib, alloc["bits"], "awq", batch_size=bs)
                params = dict(method="awq", budget=f"Q2 uniform INT{b}", allocation=alloc["design"], group=128)
                qb = info["eligible_packed_bytes"]
            else:
                raise ValueError(name)
        info["compression_seconds_total"] = t.elapsed
        return info, qb, params


# ---------------------------------------------------------------------- commands


def cmd_baseline(ctx: Ctx, force=False):
    d = out_dir("task1", "conditions", "fp16")
    write_json(d / "enumeration_check.json", verify_enumeration(ctx.model))
    ctx.run("fp16", force)


def cmd_uniform(ctx: Ctx, force=False):
    for c in UNIFORM:
        ctx.run(c, force)


def cmd_sensitivity(ctx: Ctx, force=False):
    from pa4.task1.sensitivity import fisher_diagonal, probe

    d = out_dir("task1", "sensitivity")
    if (d / "probe.json").exists() and not force:
        print("[skip] sensitivity")
        return
    ctx.snap.restore(ctx.model)
    fis = fisher_diagonal(ctx.model, ctx.calib)
    print(f"Fisher: used={fis['used']} skipped={fis['skipped']} {fis['seconds']:.0f}s", flush=True)
    pr = probe(ctx.model, ctx.calib, fis["F"], batch_size=ctx.args.calib_bs, partial_path=d / "probe_partial.json")
    pr["fisher"] = dict(used=fis["used"], skipped=fis["skipped"], seconds=fis["seconds"], loss_scale=fis["loss_scale"])
    del fis
    gc.collect(); torch.cuda.empty_cache()
    write_json(d / "probe.json", pr)
    ctx.snap.restore(ctx.model)


def _q2_budget_bytes(ctx: Ctx, b: int) -> int:
    elig = eligible_linears(ctx.model)
    tot = sum(packed_bytes(tuple(l.weight.shape), b)["total"] for l in elig.values())
    tot += sum(2 * ctx.model.model.layers[i].self_attn.o_proj.in_features for i in range(len(ctx.model.model.layers)))
    return tot


def cmd_candidates(ctx: Ctx, force=False):
    """Build every P3/Q3 candidate allocation from calibration probes only (no evaluation results used)."""
    pr = read_json(out_dir("task1", "sensitivity") / "probe.json")
    units = pr["units"]
    d = out_dir("task1", "allocations")
    elig = eligible_linears(ctx.model)
    shapes = {u: tuple(l.weight.shape) for u, l in elig.items()}
    N = sum(r * c for r, c in shapes.values())
    n_layers = len(ctx.model.model.layers)
    cands = []
    for s in (0.50, 0.75):
        Z = int(round(s * N))
        # design A: Fisher-weighted Wanda pruning cost + greedy knapsack, unit = projection
        us, meta = A.fisher_prune_allocation(units, s, N)
        rc = A.exact_row_counts(us, shapes, Z)
        cands.append(dict(name=f"p3_s{int(s*100)}_fisher", target=s, design="fisher_knapsack", unit="projection",
                          unit_sparsity={u: float(rc[u].sum()) / (shapes[u][0] * shapes[u][1]) for u in rc},
                          row_counts={u: v.tolist() for u, v in rc.items()}, meta=meta))
        # design B: OWL outlier-ratio allocation, unit = block, M=5, lambda in {0.04, 0.08}
        D_ = [pr["blocks"][str(i)]["owl"]["5"] for i in range(n_layers)]
        sizes = np.array([pr["blocks"][str(i)]["numel"] for i in range(n_layers)])
        for lam in (0.04, 0.08):
            sl = A.rescale_clip(A.owl_sparsities(D_, s, lam), sizes, s)
            us = {u: float(sl[int(u.split(".")[1])]) for u in shapes}
            rc = A.exact_row_counts(us, shapes, Z)
            cands.append(dict(name=f"p3_s{int(s*100)}_owl_l{lam}", target=s, design=f"owl_M5_lambda{lam}", unit="block",
                              block_sparsity=sl.tolist(),
                              unit_sparsity={u: float(rc[u].sum()) / (shapes[u][0] * shapes[u][1]) for u in rc},
                              row_counts={u: v.tolist() for u, v in rc.items()}, meta=dict(target_zeros=Z, M=5, lam=lam)))
    for b in (4, 3):
        budget = _q2_budget_bytes(ctx, b)
        oscale = sum(2 * ctx.model.model.layers[i].self_attn.o_proj.in_features for i in range(n_layers))
        for key in ("fisher", "out_err"):
            bits, meta = A.quant_allocation(units, budget - oscale, key)
            meta.update(q2_budget_eligible_bytes=budget, o_proj_scale_bytes=oscale,
                        allocated_eligible_bytes=meta["bytes"] + oscale)
            assert meta["allocated_eligible_bytes"] <= budget
            hist = {str(k): sum(1 for v in bits.values() if v == k) for k in (2, 3, 4, 8)}
            cands.append(dict(name=f"q3_b{b}_{key}", budget_bits=b, design=f"{key}_knapsack", unit="projection",
                              bits=bits, bit_histogram=hist, meta=meta))
    for c in cands:
        write_json(d / f"{c['name']}.json", c)
    print("candidates:", [c["name"] for c in cands])


def cmd_select(ctx: Ctx, force=False):
    """Choose one design per budget by mean NLL on the 32-sequence C4 development set (never test data)."""
    d = out_dir("task1", "allocations")
    summary = {}
    for target in ("p3_s50", "p3_s75", "q3_b4", "q3_b3"):
        best = None
        rows = []
        for f in sorted(d.glob(f"{target}_*.json")):
            if f.name.endswith("_selected.json"):
                continue
            alloc = read_json(f)
            prefix = "dev" + target  # dev-only run name
            name = f"{prefix}_{alloc['name'].split(target + '_')[1]}"
            rec = ctx.run(name, force, dev_only=True, alloc=alloc)
            rows.append(dict(candidate=alloc["name"], dev_mean_nll=rec["dev"]["mean_nll"], dev_ppl=rec["dev"]["ppl"],
                             calib_out_err_rel_mean=rec.get("calib_out_err_rel_mean")))
            if best is None or rec["dev"]["mean_nll"] < best[1]:
                best = (alloc, rec["dev"]["mean_nll"])
        sel = dict(best[0])
        sel["selection"] = dict(criterion="lowest mean NLL on 32-seq C4 train development manifest", candidates=rows)
        write_json(d / f"{target}_selected.json", sel)
        summary[target] = rows
        print(target, "->", sel["name"])
    write_json(d / "selection_summary.json", summary)


def cmd_adaptive(ctx: Ctx, force=False):
    for c in ADAPTIVE:
        ctx.run(c, force)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["baseline", "uniform", "sensitivity", "candidates", "select", "adaptive", "all", "run"])
    ap.add_argument("--cond")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--calib-bs", type=int, default=8)
    ap.add_argument("--ppl-bs", type=int, default=4)
    ap.add_argument("--hs-bs", type=int, default=16)
    args = ap.parse_args()
    ctx = Ctx(args)
    if args.cmd == "run":
        ctx.run(args.cond, args.force)
        return
    steps = dict(baseline=cmd_baseline, uniform=cmd_uniform, sensitivity=cmd_sensitivity, candidates=cmd_candidates,
                 select=cmd_select, adaptive=cmd_adaptive)
    order = list(steps) if args.cmd == "all" else [args.cmd]
    for s in order:
        steps[s](ctx, args.force)


if __name__ == "__main__":
    main()

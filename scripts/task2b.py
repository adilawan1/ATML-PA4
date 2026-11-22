#!/usr/bin/env python
"""Task 2B: MQAR training/evaluation for every required mechanism x seed, plus resource sweeps.

  python scripts/task2b.py verify      # generator invariants on frozen test/dev + training samples
  python scripts/task2b.py train       # B1 + B2 runs (resumable), test evaluation, test-time controls
  python scripts/task2b.py b3          # B3 intervention -- only after inspecting B1/B2 and committing the prediction
  python scripts/task2b.py sweep       # forward-only inference sweep T in {256,1024,4096,8192}
  python scripts/task2b.py costs       # theoretical pair counts / state bytes at T=256
  python scripts/task2b.py all
B3 intervention (prediction stated in configs/task2.yaml BEFORE evaluation): gated hybrid of an 8-token
local softmax window and a 16-feature ELU+1 linear memory, K=16 mixed, state bytes <= ELU+1 comparator.
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from pa4.common import data as D
from pa4.common.utils import ckpt_root, environment_record, out_dir, read_json, time_fn, write_json
from pa4.task2.mqar import D_MODEL, MQARModel, evaluate, n_params, theoretical_costs
from pa4.task2.train_mqar import SEEDS, train_run

RUNS = [  # (name, mechanism, K, training regime, group)
    ("dense_K4_mixed", "dense", 4, "mixed", "B1"), ("dense_K8_mixed", "dense", 8, "mixed", "B1"),
    ("dense_K16_mixed", "dense", 16, "mixed", "B1"), ("dense_K16_near", "dense", 16, "near", "B1"),
    ("window_K4_near", "window", 4, "near", "B1"), ("window_K8_near", "window", 8, "near", "B1"),
    ("window_K16_near", "window", 16, "near", "B1"), ("window_K16_mixed", "window", 16, "mixed", "B1"),
    ("linear_K4_mixed", "linear", 4, "mixed", "B2"), ("linear_K8_mixed", "linear", 8, "mixed", "B2"),
    ("linear_K16_mixed", "linear", 16, "mixed", "B2"),
    ("dilated_K4_mixed", "dilated", 4, "mixed", "B2"), ("dilated_K16_mixed", "dilated", 16, "mixed", "B2"),
    ("hybrid_K16_mixed", "hybrid", 16, "mixed", "B3"),
]


def cmd_verify():
    g = D.mqar_generator()
    rep = {}
    for split in ("test", "development"):
        rows = D.mqar(split)
        for r in rows:
            g.validate(r)
        rep[split] = len(rows)
    for k in (4, 8, 16):
        for reg in ("near", "mixed", "far"):
            for i in range(200):
                g.validate(g.make_example("train", k, reg, i))
    rep["train_samples_checked"] = 9 * 200
    rep["far_outside_window_receptive_field"] = "2 layers x (W-1) = 126 < 128 <= far distance"
    write_json(out_dir("task2", "mqar") / "generator_checks.json", rep)
    print(rep)


def cmd_train(force=False, only=None, groups=("B1", "B2")):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_rows = D.mqar("development")
    test_rows = D.mqar("test")
    od = out_dir("task2", "mqar", "runs")
    for name, kind, k, regime, grp in RUNS:
        if grp not in groups or (only and name not in only):
            continue
        for seed in SEEDS:
            f = od / f"{name}_s{seed}.json"
            if f.exists() and not force:
                print("[skip]", f.name); continue
            print(f"=== {name} seed {seed} ===", flush=True)
            r = train_run(kind, k, regime, seed, dev_rows, dev)
            model = r.pop("model")
            res = dict(name=name, mechanism=kind, train_K=k, train_regime=regime, group=grp, seed=seed, **r)
            res["test"] = evaluate(model, test_rows, dev)
            if kind == "dense":  # isolation controls imposed at test time only
                model.set_test_override("window")
                res["test_window64_at_test"] = evaluate(model, test_rows, dev)
                model.set_test_override("topk")
                res["test_topk64_at_test"] = evaluate(model, test_rows, dev)
                model.set_test_override(None)
            res["theory"] = theoretical_costs(kind)
            res["env"] = environment_record(dict(precision="float32", attention=kind, data_stream="course mqar_generator split=train"))
            p = ckpt_root() / "mqar" / f"{name}_s{seed}.pt"
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), p)
            res["checkpoint"] = str(p.relative_to(ckpt_root()))
            write_json(f, res)
            del model
            gc.collect()


@torch.no_grad()
def cmd_sweep(reps=10):
    dev = torch.device("cuda")
    rows = []
    Bsz = 4
    for T in (256, 1024, 4096, 8192):
        g = torch.Generator().manual_seed(6304 + T)
        x = torch.randn(Bsz, T, D_MODEL, generator=g).to(dev)
        for kind in ("dense", "window", "dilated", "linear", "hybrid", "topk"):
            torch.manual_seed(6304)
            m = MQARModel("dense" if kind == "topk" else kind, max_len=8192).to(dev).eval()
            if kind == "topk":
                m.set_test_override("topk")
            attn = m.blocks[0].attn
            rec = dict(T=T, mechanism=kind, B=Bsz, H=4, d_head=32, unit="ms", reps=reps,
                       implementation={"dense": "SDPA is_causal", "window": "SDPA + dense bool mask (T x T)",
                                       "dilated": "SDPA + dense bool mask (4 x T x T)",
                                       "linear": "prefix-sum, materialises [B,H,T,32,32]",
                                       "hybrid": "dense bool mask + prefix-sum [B,H,T,16,32]",
                                       "topk": "dense scores + topk (materialises T x T)"}[kind])
            try:
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                t = time_fn(lambda: attn(x), reps, 3)
                rec.update(ms=1000 * t["mean"], ms_std=1000 * t["std"],
                           peak_alloc_bytes=torch.cuda.max_memory_allocated() - base, status="ok")
            except RuntimeError as e:
                rec["status"] = "OOM" if "out of memory" in str(e).lower() else str(e)[:200]
            attn._mask_cache.clear()
            del m
            gc.collect(); torch.cuda.empty_cache()
            print(rec["T"], kind, rec.get("ms"), rec["status"], flush=True)
            rows.append(rec)
    write_json(out_dir("task2", "mqar") / "inference_sweep.json", dict(rows=rows, env=environment_record(dict(precision="float32"))))


def cmd_costs():
    out = {k: theoretical_costs(k) for k in ("dense", "window", "dilated", "linear", "hybrid", "topk")}
    for k in out:
        out[k]["n_params"] = n_params(MQARModel("dense" if k == "topk" else k))
    write_json(out_dir("task2", "mqar") / "theoretical_costs_T256.json", out)
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["verify", "train", "b3", "sweep", "costs", "all"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if a.cmd in ("verify", "all"):
        cmd_verify()
    if a.cmd in ("costs", "all"):
        cmd_costs()
    if a.cmd in ("train", "all"):
        cmd_train(a.force, a.only)  # B1 + B2 only
    if a.cmd == "b3":
        # run only after inspecting B1/B2 and committing the prediction in configs/task2.yaml
        cmd_train(a.force, a.only, groups=("B3",))
    if a.cmd in ("sweep", "all"):
        cmd_sweep()


if __name__ == "__main__":
    main()

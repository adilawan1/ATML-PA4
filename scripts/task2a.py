#!/usr/bin/env python
"""Task 2A: dense vs tiled online-softmax vs PyTorch SDPA (B=2, H=4, d=64, L in 256..8192, FP32/FP16).

Outputs results/task2/attention/{bench.jsonl, tiles.json, edge_cases.json, backends.json}.
Forward-only timings use torch.no_grad(); forward+backward timings back-propagate a fixed upstream
gradient. 10 timed repetitions after 3 warm-up calls, CUDA-synchronized. Errors are against the dense FP32
reference (or, where that does not fit in memory, the FP32 tiled implementation, flagged as such).
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from pa4.common.utils import JsonlAppender, environment_record, out_dir, read_json, time_fn, write_json, write_jsonl
from pa4.task2.attention import SDPA_BACKENDS, dense_attention, detect_sdpa_kernel, sdpa_attention, tiled_attention

B, H, D = 2, 4, 64
LENGTHS = [256, 512, 1024, 2048, 4096, 8192]


def make_qkv(L, dtype, dev, seed=6304):
    g = torch.Generator(device="cpu").manual_seed(seed + L)
    q, k, v, gout = (torch.randn(B, H, L, D, generator=g) for _ in range(4))
    return [t.to(dev, dtype) for t in (q, k, v)], gout.to(dev, dtype)


def impls():
    d = {"dense": dense_attention, "tiled": lambda q, k, v: tiled_attention(q, k, v, 128, 128),
         "sdpa_default": sdpa_attention}
    for b in SDPA_BACKENDS:
        d[f"sdpa_{b.lower()}"] = (lambda bb: (lambda q, k, v: sdpa_attention(q, k, v, bb)))(b)
    return d


def is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def fwd_bwd(fn, q, k, v, g):
    qq, kk, vv = (t.detach().requires_grad_(True) for t in (q, k, v))
    out = fn(qq, kk, vv)
    out.backward(g)
    return out.detach(), (qq.grad, kk.grad, vv.grad)


def errs(out, grads, ref):
    ro, rg = ref
    e = dict(out_max_abs=float((out.float() - ro).abs().max()),
             out_rel_fro=float((out.float() - ro).norm() / ro.norm()))
    if grads is not None:
        for n, a, b in zip("qkv", grads, rg):
            e[f"d{n}_max_abs"] = float((a.float() - b).abs().max())
            e[f"d{n}_rel_fro"] = float((a.float() - b).norm() / b.norm())
    return e


def reference(L, dev):
    (q, k, v), g = make_qkv(L, torch.float32, dev)
    for name, fn in (("dense_fp32", dense_attention), ("tiled_fp32_substitute", lambda a, b, c: tiled_attention(a, b, c, 256, 256))):
        try:
            out, grads = fwd_bwd(fn, q, k, v, g)
            return name, (out.float(), [x.float() for x in grads])
        except RuntimeError as e:
            if not is_oom(e):
                raise
            gc.collect(); torch.cuda.empty_cache()
    raise RuntimeError("no reference fits")


def bench(args):
    dev = torch.device("cuda")
    od = out_dir("task2", "attention")
    log = JsonlAppender(od / "bench_rows.jsonl", key_fn=lambda r: (r["impl"], r["L"], r["dtype"]))
    rows = log.records
    bk_file = od / "backends_partial.json"
    backends = read_json(bk_file) if bk_file.exists() else {}
    names = list(impls())
    for L in LENGTHS:
        if all((n, L, dt) in log.done for n in names for dt in ("float32", "float16")) and (L != 1024 or len(backends) == 2):
            continue
        ref_name, ref = reference(L, dev)
        for dtype in (torch.float32, torch.float16):
            (q, k, v), g = make_qkv(L, dtype, dev)
            if L == 1024 and str(dtype) not in backends:
                backends[str(dtype)] = detect_sdpa_kernel(q, k, v)
                write_json(bk_file, backends)
            for name, fn in impls().items():
                if (name, L, str(dtype).replace("torch.", "")) in log.done:
                    continue
                rec = dict(impl=name, L=L, dtype=str(dtype).replace("torch.", ""), B=B, H=H, d=D, reference=ref_name,
                           reps=args.reps, warmup=3, unit="ms")
                if name == "tiled" and L > args.tiled_max_len:
                    rec["status"] = f"skipped (> {args.tiled_max_len})"
                    log.append(rec); continue
                try:
                    with torch.no_grad():
                        out = fn(q, k, v)
                    rec.update(errs(out, None, ref))
                    del out
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    base = torch.cuda.memory_allocated()

                    def f_only():
                        with torch.no_grad():
                            fn(q, k, v)

                    t = time_fn(f_only, args.reps, 3)
                    rec["fwd_ms"] = 1000 * t["mean"]; rec["fwd_ms_std"] = 1000 * t["std"]
                    rec["fwd_peak_alloc_bytes"] = torch.cuda.max_memory_allocated() - base
                    rec["fwd_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
                    out, grads = fwd_bwd(fn, q, k, v, g)
                    rec.update(errs(out, grads, ref))
                    del out, grads
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    t = time_fn(lambda: fwd_bwd(fn, q, k, v, g), args.reps, 3)
                    rec["fwdbwd_ms"] = 1000 * t["mean"]; rec["fwdbwd_ms_std"] = 1000 * t["std"]
                    rec["fwdbwd_peak_alloc_bytes"] = torch.cuda.max_memory_allocated() - base
                    rec["fwdbwd_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
                    rec["status"] = "ok"
                except RuntimeError as e:
                    rec["status"] = "OOM" if is_oom(e) else f"unsupported: {str(e).splitlines()[0][:200]}"
                gc.collect(); torch.cuda.empty_cache()
                print(rec["impl"], L, rec["dtype"], rec["status"], f"{rec.get('fwd_ms', float('nan')):.2f}ms", flush=True)
                log.append(rec)
        del ref
        gc.collect(); torch.cuda.empty_cache()
    write_jsonl(od / "bench.jsonl", rows)
    write_json(od / "backends.json", dict(default_dispatch_L1024=backends, env=environment_record(dict(note="dtype per row; backend per row"))))


def tiles_and_edges():
    dev = torch.device("cuda")
    od = out_dir("task2", "attention")
    L = 1024
    ref_name, ref = reference(L, dev)
    tiles = []
    for dtype in (torch.float32, torch.float16):
        (q, k, v), g = make_qkv(L, dtype, dev)
        for bq, bk in [(32, 32), (64, 64), (128, 128), (256, 256), (64, 256), (256, 64), (1024, 1024)]:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            with torch.no_grad():
                out = tiled_attention(q, k, v, bq, bk)
            peak = torch.cuda.max_memory_allocated() - base
            o2, gr = fwd_bwd(lambda a, b, c: tiled_attention(a, b, c, bq, bk), q, k, v, g)
            tiles.append(dict(dtype=str(dtype), bq=bq, bk=bk, fwd_peak_alloc_bytes=peak, **errs(o2, gr, ref),
                              score_tile_bytes=4 * B * H * bq * bk, full_score_bytes=4 * B * H * L * L))
    write_json(od / "tiles.json", dict(L=L, reference=ref_name, rows=tiles))
    # edge cases
    edges = {}
    (q, k, v), _ = make_qkv(256, torch.float32, dev)
    for name, fn in impls().items():
        try:
            o1 = fn(q, k, v)
            k2, v2 = k.clone(), v.clone()
            k2[:, :, 200:] += 100.0
            v2[:, :, 200:] -= 50.0
            o2 = fn(q, k2, v2)
            caus = float((o1[:, :, :200] - o2[:, :, :200]).abs().max())
            ext = {}
            for dtype, mult in ((torch.float32, 1e3), (torch.float16, 30.0)):
                qq, kk, vv = (t.to(dtype) for t in (q * mult, k, v))
                ref_e = dense_attention(qq.double(), kk.double(), vv.double())
                oe = fn(qq, kk, vv)
                ext[str(dtype)] = dict(multiplier=mult, finite=bool(torch.isfinite(oe).all()),
                                       max_abs_err_vs_fp64=float((oe.double() - ref_e).abs().max()))
            edges[name] = dict(causality_future_perturbation_max_change=caus, extreme_scores=ext)
        except RuntimeError as e:
            edges[name] = dict(status=str(e).splitlines()[0][:200])
    m = torch.ones(256, 256, dtype=torch.bool, device=dev)
    m[10] = False
    od_ = dense_attention(q, k, v, causal=True, mask=m)
    import torch.nn.functional as F
    sd = F.scaled_dot_product_attention(q, k, v, attn_mask=torch.tril(m))
    edges["all_masked_row"] = dict(dense_row_output_abs_max=float(od_[:, :, 10].abs().max()),
                                   dense_finite=bool(torch.isfinite(od_).all()),
                                   sdpa_row_finite=bool(torch.isfinite(sd[:, :, 10]).all()),
                                   note="dense defines a fully-masked row as 0; SDPA behaviour recorded")
    write_json(od / "edge_cases.json", edges)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--tiled-max-len", type=int, default=8192)
    ap.add_argument("--only", choices=["bench", "tiles"], default=None)
    a = ap.parse_args()
    od = out_dir("task2", "attention")
    if a.only in (None, "tiles") and not ((od / "tiles.json").exists() and (od / "edge_cases.json").exists()):
        tiles_and_edges()
    if a.only in (None, "bench"):
        bench(a)


if __name__ == "__main__":
    main()

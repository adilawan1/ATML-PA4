#!/usr/bin/env python
"""Print what is finished / partially done for every task (reads $PA4_RESULTS and $PA4_CKPT on Drive).

Use after a Colab disconnect to see where each step will resume. Nothing here modifies results.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pa4.common.utils import ckpt_root, results_root

R, CK = results_root(), ckpt_root()


def count_lines(p: Path) -> int:
    if not p.exists():
        return 0
    with open(p, "rb") as f:
        return sum(1 for l in f if l.strip())


def mark(ok):
    return "done   " if ok else "pending"


print(f"results: {R}\ncheckpoints: {CK}\n")
print("Task 1")
for c in ["fp16", "p1_s25", "p1_s50", "p1_s75", "p2_s25", "p2_s50", "p2_s75", "p2_24_s50", "q1_b8", "q1_b4", "q1_b3",
          "q2_b8", "q2_b4", "q2_b3"]:
    print(f"  {mark((R / 'task1/conditions' / c / 'metrics.json').exists())} {c}")
sens = R / "task1/sensitivity"
part = sens / "probe_partial.json"
print(f"  {mark((sens / 'probe.json').exists())} sensitivity" + ("" if (sens / "probe.json").exists() or not part.exists()
      else f" (partial file present: resumes at the next unfinished block)"))
print(f"  {mark(any((R / 'task1/allocations').glob('*_fisher.json')))} candidates")
for c in ["p3_s50", "p3_s75", "q3_b4", "q3_b3"]:
    print(f"  {mark((R / 'task1/allocations' / f'{c}_selected.json').exists())} select {c}   "
          f"{mark((R / 'task1/conditions' / c / 'metrics.json').exists())} final {c}")

print("\nTask 2")
print(f"  2A bench rows: {count_lines(R / 'task2/attention/bench_rows.jsonl')} / 84"
      f"   tiles: {mark((R / 'task2/attention/tiles.json').exists())}")
runs = list((R / "task2/mqar/runs").glob("*.json")) if (R / "task2/mqar/runs").exists() else []
print(f"  2B MQAR runs: {len(runs)} / 42   sweep: {mark((R / 'task2/mqar/inference_sweep.json').exists())}")
print(f"  2C KV prompts: {count_lines(R / 'task2/kv/cases.jsonl')} / 180   C5 rows: {count_lines(R / 'task2/kv/c5_bench_rows.jsonl')} / 16")

print("\nTask 3")
print(f"  {mark((R / 'task3/checks/sampler_checks.json').exists())} check")
pool_part = CK / "task3/teacher_pool.partial.json"
print(f"  {mark((CK / 'task3/teacher_pool.json').exists())} teacher pool" +
      ("" if (CK / "task3/teacher_pool.json").exists() or not pool_part.exists() else " (partial present)"))
for m in ("off", "on"):
    for s in (6304, 6305):
        n = f"{m}_s{s}"
        ck = CK / "task3/distill_ckpt" / f"{n}.pt"
        extra = ""
        if not (R / "task3/distill" / n / "result.json").exists() and ck.exists():
            import torch

            st = torch.load(ck, map_location="cpu", weights_only=False)
            extra = f" (checkpoint at update {st['update']}, {st['scored_total']} / 40000 tokens)"
        print(f"  {mark((R / 'task3/distill' / n / 'result.json').exists())} train {n}{extra}   "
              f"{mark((R / 'task3/distill' / n / 'merge.json').exists())} merge")
for f in ["exact_spec_original_gamma4"] + [f"own_{m}_s{s}" for m in ("off", "on") for s in (6304, 6305)]:
    done = (R / "task3/banks" / f"{f}.jsonl.gz").exists()
    part_n = count_lines(R / "task3/banks" / f"{f}.partial.jsonl")
    print(f"  {mark(done)} bank {f}" + ("" if done else f" ({part_n}/120 prompts)"))
print(f"  {mark((R / 'task3/align/positions.jsonl.gz').exists())} align "
      f"({count_lines(R / 'task3/align/positions.partial.jsonl')} groups so far)")
for d in ["original", "original_fixed64", "off_s6304", "off_s6305", "on_s6304", "on_s6305", "on_s6304_unmerged"]:
    tot = {"original_fixed64": 96, "on_s6304_unmerged": 20}.get(d, 480)
    print(f"  spec {d}: {count_lines(R / 'task3/spec' / f'records_{d}.jsonl')} / {tot} records")
for m in ["teacher", "original", "off_s6304", "off_s6305", "on_s6304", "on_s6305"]:
    print(f"  standalone {m}: {count_lines(R / 'task3/standalone' / f'{m}.jsonl')} / 120")

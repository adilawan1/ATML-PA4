# results/

Filled by the drivers (on Colab: `$PA4_RESULTS` on Drive, synced here by `notebooks/99_sync_results_to_repo.ipynb`).

* `task1/conditions/<cond>/` metrics.json (quality, storage, latency, env), ppl_*.json (per-window NLL), hellaswag.jsonl
  (per-example log-likelihoods), units.json (per-projection achieved sparsity/bits/bytes/calibration error)
* `task1/sensitivity/probe.json`, `task1/allocations/*.json` (all candidates + selected), `task1/dev_candidates/`
* `task2/attention/`, `task2/mqar/runs/*.json` (curves, test metrics per seed), `task2/kv/cases.jsonl` (per prompt × policy)
* `task3/checks/`, `task3/distill/<run>/`, `task3/banks/`, `task3/align/positions.jsonl.gz`, `task3/spec/records_*.jsonl`,
  `task3/standalone/`
* `tables/*.csv|.tex`, `figures/task*/` — produced by `analysis/task*_report.py`

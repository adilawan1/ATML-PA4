# ATML PA4 — Compression, Efficient Attention & KV Memory, Speculative Decoding

Code, configurations and machine-readable results for EE-5102/CS-6304 Programming Assignment 4 (Fall 2026).
All experiments are written for **Google Colab (Pro) GPUs** (A100 or L4 recommended; T4 works but is slower, and the
largest dense-attention shapes may OOM, which is then recorded as a result).

| Task | What is implemented | Entry point |
|---|---|---|
| 1 Model compression (Qwen2.5-1.5B) | magnitude (P1) and Wanda (P2) pruning, Fisher-knapsack / OWL nonuniform pruning (P3), 2:4 diagnostic, group-wise RTN (Q1), AWQ-style scaling (Q2), mixed precision (Q3), packed-byte accounting, latency/memory | `scripts/task1.py` |
| 2A Exact attention | dense, tiled online-softmax (with recomputation backward), SDPA backends | `scripts/task2a.py` |
| 2B Associative recall | dense / sliding-window / dilated / ELU+1 linear / gated hybrid (B3) MQAR Transformers, test-time mask controls | `scripts/task2b.py` |
| 2C KV memory (Qwen2.5-1.5B-Instruct) | sink+recent and question-aware FP16 eviction, token-wise vs channel-K/token-V INT4, FP16-select + INT2 hybrid at a matched INT4 byte budget; real bit-packing benchmark | `scripts/task2c.py` |
| 3 Speculative decoding | exact speculative sampler with cache rollback, χ² Markov tests, off/on-policy LoRA distillation (0.5B ← 1.5B), prefix-bank alignment, speculative benchmarks, cost model | `scripts/task3.py` |

## Repository layout

```
pa4/common/      seeding, CUDA timing, IO, environment records, paired bootstrap, locked-data loaders
pa4/task1/       model_utils (eligible projections, tied params, snapshot/restore), calib (block-sequential runner),
                 prune, quant (quantizer + bit packing + byte models), awq, sensitivity (Fisher + probes), allocate,
                 evaluate (PPL windows, HellaSwag), resources
pa4/task2/       attention (2A), mqar + train_mqar (2B), kv + kv_pack (2C)
pa4/task3/       spec (exact sampler, runners, toy χ² tests), distill, align (banks + metrics)
scripts/         task drivers (resumable), get_data.sh, record_env.py
analysis/        task{1,2,3}_report.py -> results/tables/*.csv|.tex and results/figures/**/*.pdf|.png
configs/         fixed hyper-parameters of every task (kept in sync with code by tests/test_configs.py)
notebooks/       Colab notebooks: 00 setup+tests, task1, task2, task3, 99 sync results to the repo
tests/           CPU unit tests on tiny random Qwen2 models + a synthetic-results smoke test for analysis
report/          NeurIPS-format report skeleton (main.tex) wired to the generated tables/figures
results/         machine-readable per-condition / per-example outputs (committed after the runs)
```

## Environment

* `transformers==4.57.1`, `huggingface_hub==0.36.2` (pinned by the handout), `peft==0.17.1`; see `requirements.txt`.
* PyTorch: Colab's preinstalled CUDA build. Every result file embeds an environment record (torch/transformers
  versions, GPU name and capability, CUDA/cuDNN, git commits of this repo and of the data repo, and the locked Hugging
  Face revisions of all models/datasets from `data/local_source_revisions.json`). `python scripts/record_env.py` writes
  `results/env.json`.
* Common seed 6304 (MQAR additionally 6305/6306; distillation 6304/6305).

## Data

Nothing from the datasets is committed here. `scripts/get_data.sh` clones the course data repository
[`atml-pa4-datasets`](https://github.com/AbDu11aHHH/atml-pa4-datasets) at commit `6a0a9ec9`, then

```bash
python $PA4_DATA/scripts/setup_data.py --public        # C4/WikiText-2/HellaSwag/GSM8K locked selections
python $PA4_DATA/scripts/setup_data.py --prefix-banks  # course teacher / original-draft prefix banks (GPU)
python $PA4_DATA/scripts/check_setup.py
```

The first `--public` run resolves and locks the Hugging Face commits in `data/local_source_revisions.json`; keep that
file (the notebooks place the data repo on Google Drive) and report those revisions. All models are loaded at the
locked revisions. Selected IDs, windows, prompt formats and seeds are re-saved with the results
(`results/task1/data_selection.json`, `results/task2/kv/setup.json`, `results/task3/distill/*/presentations.jsonl.gz`).

## Reproduction (Colab)

1. Push this repository to a **public** GitHub repo and set `REPO_URL` in the first cell of each notebook.
2. `notebooks/00_setup_data_and_tests.ipynb` — data preparation + unit tests (once).
3. `notebooks/task1_compression.ipynb`, `task2_attention_kv.ipynb`, `task3_speculative_distillation.ipynb` — each cell is
   one resumable step; outputs go to `$PA4_RESULTS` (Drive), checkpoints/adapters to `$PA4_CKPT` (Drive).
4. `notebooks/99_sync_results_to_repo.ipynb` — copy the results tree into `results/` and push.
5. Locally or in Colab: `python analysis/task{1,2,3}_report.py`, then compile `report/main.tex`.

Equivalent shell commands (any CUDA machine):

```bash
pip install -r requirements.txt && bash scripts/get_data.sh
export PA4_DATA=$PWD/external/atml-pa4-datasets PA4_RESULTS=$PWD/results PA4_CKPT=$PWD/checkpoints
python -m pytest -q tests
python scripts/task1.py all
python scripts/task2a.py && python scripts/task2b.py all && python scripts/task2c.py all
python scripts/task3.py all
for t in 1 2 3; do python analysis/task${t}_report.py; done
```

Every number in the report traces to a file under `results/` produced by these commands.

## Mapping from the handout to the code

### Task 1
| Step | Implementation |
|---|---|
| 1 Baseline & protocol | `task1.py baseline`; `verify_enumeration` asserts 28×7 eligible matrices, expected parameter count, tied embedding counted once; per-window NLL and per-example HellaSwag log-likelihoods saved |
| 2 P1 magnitude | `prune.row_prune_mask` on \|W\|, per output row, s ∈ {.25,.5,.75}; weights restored from a CPU snapshot (checksum-verified) before every condition |
| 3 P2 Wanda | score \|W\|·‖X_j‖₂ from block-sequential calibration inputs (Wanda protocol); score quantiles kept |
| 4 P3 nonuniform | sensitivity: empirical-Fisher OBD cost ½ΣF·W² of the Wanda-pruned set per projection (alt.: OWL outlier ratio per block); rule: greedy multiple-choice knapsack over a 0.025 sparsity grid in [0.10, 0.85] (alt.: OWL λ∈{0.04,0.08}); exact total-zero matching by ±1 weight per row; selection on the 32-sequence C4 dev set |
| 5 Q1 | `quant.quantize_groupwise` (asymmetric, G=128, FP16 scale, uint8 zero-point, round-half-even) + real bit packing; bytes counted from packed tensors |
| 6 Q2 | `awq.quantize_model(method="awq")`: α grid {0,…,1}, scales folded into RMSNorm gains and up_proj rows; o_proj scales serialized as FP16 (GQA makes the v→o fold invalid); folding equivalence check per block |
| 7 Q3 | per-unit costs at b∈{2,3,4,8} (Fisher-weighted ΔW² or calibration output error) → knapsack on exact packed bytes ≤ Q2 INT4 / INT3 budgets; same AWQ quantizer |
| 8 2:4 | `prune.nm_mask` with Wanda scores (quality/representation study only) |
| 9 Trade-offs | `analysis/task1_report.py`: quality vs sparsity / packed bytes, sensitivity & allocation maps, matched-budget checks, calibration-vs-held-out Spearman, paired bootstraps (P3–P2, Q3–Q2, …), qualitative HellaSwag cases |

### Task 2
| Step | Implementation |
|---|---|
| A1–A3 | `attention.dense_attention`, `TiledAttention` (online softmax forward, FlashAttention-2-style recomputation backward storing only O and LSE — a Python reference, **not** a fused kernel), SDPA with forced backends and profiler-detected default kernel; errors vs FP32 dense (or FP32 tiled where dense OOMs, flagged) |
| B1–B2 | `mqar.MQARModel` (only the attention operator changes), protocol in `train_mqar.py`; dense/window/dilated masks are explicit dense boolean masks (correctness implementation, not sparse kernels); linear attention materialises prefix tensors (reported) |
| B3 | gated 8-token local window + 16-feature ELU+1 memory; prediction and state-matching recorded in `configs/task2.yaml` before evaluation |
| C1 | split-prefill vs full-forward logits, rebuilt 100 %-retention cache vs `generate`, finiteness, absolute RoPE positions preserved (`position_ids`) with consistent physical cache length |
| C2–C4 | `kv.compress` / `kv.run_policy`; budgets from the handout formula; eviction keeps the largest legal number of FP16 positions incl. uint16 index bytes |
| C5 | `kv_pack` real INT4/INT2 packing (serialized bytes asserted equal to the formula), pack/unpack and decode-step timings at batch 1/8 × 2k/4k |

### Task 3
| Step | Implementation |
|---|---|
| 1 Sampler | `spec.speculative_generate` (one target verification per round, residual sampling, bonus token, EOS, cap truncation, cache crop); χ² on 100k length-3 sequences + wrong-residual negative control, EOS variant, γ=1 variant; real-model cache/split-prefill checks |
| 2 Baseline | `task3.py baseline` (EOS-aware cap 64, γ∈{2,4}, 2 reps, counterbalanced adjacent target-only, per-round timing decomposition, isolated step costs) |
| 3–4 Distillation | `distill.distill` off/on-policy with identical schedule, quota, updates per 256 scored tokens, GradScaler, LoRA r=8 |
| 5 Alignment | `task3.py banks/align`: course teacher & original-draft banks, our exact-speculative bank, each trained student's own rollouts (all drafts evaluated on every bank) |
| 6–7 Execution & workload | merged FP16 drafts (merge error reported), unmerged-adapter timing, workload/length breakdowns, cost model Ŝ_γ, break-even α, standalone accuracy, amortised distillation break-even |

## Design decisions and deviations (also stated in the report)

* **Block-sequential calibration** for Wanda, AWQ and error measurement: block *l* sees outputs of already compressed
  blocks. Layer-output error = ‖(W − Ŵ)X‖²/‖WX‖² on the 128×512 calibration tokens.
* **AWQ-style, not full AWQ**: scale search + folding reproduced; weight-clipping search, the module-level objective and
  fused W4A16 kernels are not. All quantized models execute as dense FP16 (fake-quantized); packed sizes are serialized
  estimates from real packing, latency is dense FP16 latency.
* **Sparse storage** uses the specified model (1 mask bit/weight, byte-aligned per matrix, + 16 bits/nonzero); no sparse
  kernel is used, so measured latency/memory equals dense FP16.
* **HellaSwag**: lm-eval-harness v1.0 preprocessing and character-length normalisation, zero-shot, Qwen base tokenizer
  without BOS.
* **KV question-aware scoring** is a simplified SnapKV-inspired fixed policy (per-layer selection shared by both KV
  heads, pooled attention mass of the question suffix), not the published head-specific algorithm. Quantized caches
  are stored dequantized for ordinary SDPA, so allocated memory ≠ packed bytes; both are reported.
* **B3 state matching**: hybrid state per layer = 4,160 FP32 values vs 4,224 for the ELU+1 comparator (−1.5 %); exact
  equality is impossible with integer feature dimension and window length, so the hybrid is kept *at or below* the
  comparator.
* **Standalone student accuracy** uses greedy decoding with caps 512 (math) / 32 (short form).
* **On-policy rollouts** are generated by a seeded manual sampler (temperature 1, top-p 1, no top-k) so every
  presentation has a recorded seed.

## Attribution of reused external code / algorithms

No source files were copied from external repositories. The following are re-implementations of published algorithms
or conventions, credited here and in the report:
* Wanda score — Sun et al. (2024), github.com/locuslab/wanda (MIT).
* AWQ pseudo-quantizer conventions (scale clamp 1e-5, zero-point clamp) and scale-search form — Lin et al. (2024),
  github.com/mit-han-lab/llm-awq (MIT).
* OWL layerwise sparsity formula — Yin et al. (2024), github.com/luuyin/OWL (MIT).
* HellaSwag preprocessing — EleutherAI lm-evaluation-harness (MIT), reproduced in `pa4/task1/evaluate.py`.
* FlashAttention-2 forward/backward recurrences — Dao et al. (2022, 2023), re-implemented in PyTorch.
* `apply_rotary_pos_emb` and `repeat_kv` are imported from Hugging Face Transformers (Apache-2.0); LoRA via PEFT.
* MQAR examples come from the course generator (`scripts/mqar_generator.py` in the data repository, imported, not copied).

## AI usage

Coding assistance from an LLM was used while writing this repository, as permitted by the assignment. The author has
reviewed, run and is responsible for every submitted line, and all experiments, results and analysis are their own.

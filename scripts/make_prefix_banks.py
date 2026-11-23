#!/usr/bin/env python
"""Generate the course teacher / original-draft prefix banks (drop-in for `setup_data.py --prefix-banks`).

Why: the course script passes `generator=` to `model.generate()`, which transformers==4.57.1 (pinned by the
handout) rejects (`model_kwargs are not used by the model: ['generator']`). This script reproduces the course
function exactly -- same prompts and order (60 GSM8K test IDs + 60 short-form test IDs), same chat formatting
with the target tokenizer, same per-prompt seeds 6304 + 10007*j (+123 for the draft), same 192-token cap, same
position sampler `random.Random('PA4-banks-v1:6304:{pid}:{which}')`, same record fields and the same output file
data/prefix_banks/prefix_banks_locked.jsonl.gz written with the course `save_lines` -- except that each response
is drawn with our seeded ancestral sampler (temperature 1, top-p 1, no top-k, CUDA torch.Generator seeded per
prompt, stop at EOS) instead of the generate() call that cannot accept the generator.

Resumable: each prompt's response is fsynced to <data>/data/prefix_banks/prefix_banks.partial.jsonl.
"""
from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from pa4.common.utils import JsonlAppender, data_root, write_json
from pa4.task3.distill import _manual_sample, eos_ids

SCHEME = [(1, 16, 8), (17, 32, 8), (33, 64, 16), (65, 192, 16)]


def course_module():
    spec = importlib.util.spec_from_file_location("course_setup_data", data_root() / "scripts" / "setup_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def course_prompts(S, rev):
    from datasets import load_dataset

    gsm = load_dataset(S.REPOS["gsm8k"][1], "main", split="test", revision=rev["gsm8k"])
    ms = S.read(S.MAN / "gsm8k_test_ids.jsonl")
    prompts = [(f'GSM8K-{r["order"]:04d}', gsm[r["row_index"]]["question"], "math") for r in ms]
    prompts.extend((x["id"], x["input"], "short") for x in S.read(S.ROOT / "data/short_form/short_test.jsonl"))
    return prompts


def positions_for(pid, which, length):
    rng = random.Random(f"PA4-banks-v1:6304:{pid}:{which}")
    pos = []
    for lo, hi, n in SCHEME:
        cand = list(range(lo, min(hi, length) + 1))
        pos.extend(sorted(rng.sample(cand, min(n, len(cand)))))
    return pos


@torch.no_grad()
def build(S, rev, tok, load_model, prompts, partial_path, device="cuda"):
    eos = eos_ids(tok)
    with JsonlAppender(partial_path, key_fn=lambda r: (r["bank"], r["prompt_id"])) as log:
        for which, key in [("teacher", "model_target"), ("original_draft", "model_draft")]:
            todo = [(j, p) for j, p in enumerate(prompts) if (which, p[0]) not in log.done]
            if not todo:
                continue
            print("Loading", which, flush=True)
            model = load_model(key)
            for j, (pid, prompt, domain) in todo:
                ids = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                              add_generation_prompt=True, return_tensors="pt").to(device)
                seed = 6304 + 10007 * j + (0 if which == "teacher" else 123)
                gen = torch.Generator(device=device).manual_seed(seed)
                seq = _manual_sample(model, ids, 192, gen, eos)[0, ids.shape[1]:].tolist()
                log.append(dict(bank=which, prompt_id=pid, j=j, domain=domain, prompt=prompt, seq=seq, seed=seed))
                if (j + 1) % 20 == 0:
                    print(which, j + 1, "/", len(prompts), flush=True)
            del model
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        responses = {(r["bank"], r["prompt_id"]): r for r in log.records}
    output = []
    for which in ("teacher", "original_draft"):
        for j, (pid, prompt, domain) in enumerate(prompts):
            r = responses[(which, pid)]
            seq = r["seq"]
            for pos in positions_for(pid, which, len(seq)):
                output.append(dict(prompt_id=pid, domain=domain, bank=which, position=pos, prompt=prompt,
                                   response_prefix_token_ids=seq[: pos - 1], source_length=len(seq),
                                   source_generation_seed=r["seed"]))
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild even if the bank file exists")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("needs a CUDA GPU (T4 is fine)")
    S = course_module()
    rev = S.revisions(False)
    dest = S.BANK / "prefix_banks_locked.jsonl.gz"
    if dest.exists() and not a.force:
        print("Prefix banks already exist:", dest); return
    S.BANK.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(S.REPOS["model_target"][1], revision=rev["model_target"])
    dtok = AutoTokenizer.from_pretrained(S.REPOS["model_draft"][1], revision=rev["model_draft"])
    for s in ["Example 123", "Hello there", "A complicated example with 𝜋"]:
        if tok.encode(s) != dtok.encode(s):
            raise RuntimeError("Target/draft tokenizers differ")
    prompts = course_prompts(S, rev)
    assert len(prompts) == 120, len(prompts)

    def load_model(key):
        return AutoModelForCausalLM.from_pretrained(S.REPOS[key][1], revision=rev[key], torch_dtype=torch.float16,
                                                    attn_implementation="sdpa").to("cuda").eval()

    out = build(S, rev, tok, load_model, prompts, S.BANK / "prefix_banks.partial.jsonl")
    S.save_lines(dest, out)
    write_json(S.BANK / "prefix_banks_generation_note.json", dict(
        generator="scripts/make_prefix_banks.py (ATML-PA4 repo)",
        reason="course setup_data.py passes generator= to generate(), rejected by transformers==4.57.1",
        sampler="ancestral, temperature 1, top-p 1, no top-k, per-prompt CUDA torch.Generator(seed), stop at EOS, cap 192",
        eos_ids=eos_ids(tok), revisions={k: rev[k] for k in ("model_target", "model_draft")}, n_records=len(out)))
    print(f"Wrote {dest} ({len(out)} records). Students generate speculative-policy prefixes with their own sampler.")


if __name__ == "__main__":
    main()

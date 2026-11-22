"""CPU tests for Task 3: exact sampler (toy + tiny HF models), cache rollback, distillation loop, banks."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pa4.task3 import align as AL
from pa4.task3 import distill as DS
from pa4.task3.spec import HFRunner, speculative_generate, target_only_generate, toy_eos_test, toy_markov_test
from tiny_models import tiny_qwen


class StubTok:
    eos_token_id = 1

    def convert_tokens_to_ids(self, t):
        return {"<|im_end|>": 1, "<|endoftext|>": 2}[t]


class SamplerTests(unittest.TestCase):
    def test_toy_chi2(self):
        r = toy_markov_test(20_000, 3, 2)
        self.assertTrue(r["setup"]["pass_exact"])
        self.assertTrue(r["setup"]["rejects_negative_control"])
        self.assertGreater(r["exact"]["bonus"], 0)
        self.assertGreater(r["exact"]["corrections"], 0)
        e = toy_eos_test(20_000)
        self.assertTrue(e["pass_exact"])

    def test_hf_cache_rollback(self):
        t = tiny_qwen(seed=1, vocab=300)
        d = tiny_qwen(seed=2, vocab=300)
        tr, dr = HFRunner(t), HFRunner(d)
        prompt = list(range(10, 30))
        for gamma in (0, 1, 2, 4):
            gen = torch.Generator().manual_seed(gamma)
            out, st = speculative_generate(tr, dr, prompt, gamma, 25, gen, eos_ids=[])
            self.assertEqual(len(out), 25)
            self.assertTrue(st.truncated)
            seq = prompt + out
            cached = tr.feed(seq[tr.n_cached:])[-1]
            full = torch.softmax(t(torch.tensor([seq])).logits[0, -1], -1)
            self.assertLess(float((cached - full).abs().max()), 1e-5)
            # draft cache is a consistent prefix as well
            dc = dr.feed(seq[dr.n_cached:])[-1]
            dfull = torch.softmax(d(torch.tensor([seq])).logits[0, -1], -1)
            self.assertLess(float((dc - dfull).abs().max()), 1e-5)
            self.assertEqual(st.proposed, sum(st.proposed_per_round))

    def test_identical_draft_accepts_everything(self):
        t = tiny_qwen(seed=1, vocab=300)
        tr, dr = HFRunner(t), HFRunner(t)
        out, st = speculative_generate(tr, dr, list(range(5, 15)), 4, 21, torch.Generator().manual_seed(0))
        self.assertEqual(st.accepted, st.proposed)
        self.assertEqual(st.corrections, 0)

    def test_eos_stops(self):
        t = tiny_qwen(seed=1, vocab=300)
        d = tiny_qwen(seed=2, vocab=300)
        tr, dr = HFRunner(t), HFRunner(d)
        hits = 0
        for s in range(30):
            out, st = speculative_generate(tr, dr, [3, 4, 5], 4, 40, torch.Generator().manual_seed(s),
                                           eos_ids=list(range(0, 150)))
            if st.eos:
                hits += 1
                self.assertLess(out[-1], 150)
                self.assertTrue(all(x >= 150 for x in out[:-1]))
        self.assertGreater(hits, 0)
        out, info = target_only_generate(tr, [3, 4, 5], 10, torch.Generator().manual_seed(0), eos_ids=[])
        self.assertEqual(len(out), 10)


class DistillTests(unittest.TestCase):
    def test_distill_loop(self):
        teacher = tiny_qwen(seed=1, vocab=300)
        student = tiny_qwen(seed=2, vocab=300)
        old_q, old_t = dict(DS.QUOTA), DS.TOKENS_PER_UPDATE
        DS.QUOTA.update(math=60, short=20)
        DS.TOKENS_PER_UPDATE = 16
        DS.CAP.update(math=12, short=4)
        try:
            prompts = {f"m{i}": dict(ids=list(range(10 + i, 20 + i)), category="math") for i in range(5)}
            prompts.update({f"s{i}": dict(ids=list(range(30 + i, 36 + i)), category="short") for i in range(3)})
            perms = DS.build_schedule([k for k in prompts if k[0] == "m"], [k for k in prompts if k[0] == "s"])
            pool = DS.teacher_pool(teacher, StubTok(), prompts)["pool"]
            for method in ("off", "on"):
                r = DS.distill(method, 6304, teacher, tiny_qwen(seed=2, vocab=300), StubTok(), prompts, perms, pool,
                               log=lambda *a: None)
                res = r["result"]
                self.assertEqual(res["tokens"], dict(math=60, short=20))
                self.assertEqual(res["logs"][-1]["scored_tokens"], 80)
                self.assertEqual(res["updates"], 5)
                self.assertEqual(sum(p["scored_tokens"] for p in r["presentations"]), 80)
                self.assertGreater(res["n_adapter_params"], 0)
        finally:
            DS.QUOTA.update(old_q)
            DS.TOKENS_PER_UPDATE = old_t
            DS.CAP.update(math=192, short=32)

    def test_kl(self):
        a = torch.randn(4, 50)
        self.assertLess(float(DS.token_kl(a, a).abs().max()), 1e-6)
        self.assertTrue(bool((DS.token_kl(a, torch.randn(4, 50)) > 0).all()))


class BankTests(unittest.TestCase):
    def test_positions_and_metrics(self):
        pos = AL.sample_positions("GSM8K-0000", "teacher", 100)
        self.assertEqual(len(pos), 8 + 8 + 16 + 16)
        self.assertTrue(all(1 <= p <= 100 for p in pos))
        recs = AL.bank_records("x", "math", "q", "own", list(range(50)), {})
        g = AL.group_bank(recs)
        e = g[("own", "x")]
        self.assertEqual(len(e["longest"]), max(e["positions"]) - 1)
        lp = torch.log_softmax(torch.randn(3, 20), -1)
        m = AL.pair_metrics(lp, lp)
        self.assertAlmostEqual(m["exp_accept"][0], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()

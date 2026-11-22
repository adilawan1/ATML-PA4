"""CPU tests for Task 1 components (quantizer, packing, pruning, AWQ folding, allocation, PPL scoring)."""
import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pa4.task1 import allocate as A
from pa4.task1.awq import quantize_model, uniform_bits
from pa4.task1.evaluate import hs_preprocess, ppl_summary, window_nll
from pa4.task1.model_utils import WeightSnapshot, count_nonzeros, eligible_linears, parameter_account, verify_enumeration
from pa4.task1.prune import nm_mask, prune_model, row_prune_mask, uniform_row_counts
from pa4.task1.quant import pack_bits, packed_bytes, quantize_and_pack, quantize_groupwise, sparse16_bytes, unpack_bits
from pa4.task1.resources import storage_report
from pa4.task1.sensitivity import fisher_diagonal, probe
from tiny_models import tiny_qwen


class QuantTests(unittest.TestCase):
    def test_pack_roundtrip(self):
        for b in (2, 3, 4, 8):
            codes = torch.randint(0, 2 ** b, (5, 128 + 3), dtype=torch.uint8)
            p = pack_bits(codes, b)
            self.assertEqual(p.shape[1], math.ceil(codes.shape[1] * b / 8))
            self.assertTrue(torch.equal(unpack_bits(p, b, codes.shape[1]), codes))

    def test_quantizer_properties(self):
        w = torch.randn(16, 256)
        for b in (2, 3, 4, 8):
            deq, q, s, z = quantize_groupwise(w, b)
            self.assertLessEqual(int(q.max()), 2 ** b - 1)
            err = (deq - w).abs().view(16, 2, 128).amax(-1)
            self.assertTrue(bool((err <= s.float() + 1e-4).all()))  # <= one step (rounded zero-point at range ends)
            _, rec = quantize_and_pack(w, b, verify=True)
            self.assertEqual(rec["total"], packed_bytes((16, 256), b)["total"])
        self.assertEqual(packed_bytes((1, 128), 4)["total"], 64 + 3)

    def test_sparse_model(self):
        self.assertEqual(sparse16_bytes(16, 8), 2 + 16)


class PruneTests(unittest.TestCase):
    def test_row_mask_counts(self):
        s = torch.rand(7, 64)
        k = torch.arange(7)
        m = row_prune_mask(s, k)
        self.assertTrue(torch.equal(m.sum(1), k))
        # pruned entries are the smallest of each row
        for r in range(7):
            if k[r] > 0:
                self.assertLessEqual(float(s[r][m[r]].max()), float(s[r][~m[r]].min()))

    def test_24(self):
        m = nm_mask(torch.rand(3, 16))
        self.assertTrue(bool((m.view(3, 4, 4).sum(-1) == 2).all()))


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.m = tiny_qwen()
        self.ids = torch.randint(0, 500, (8, 64))

    def test_enumeration_and_tied(self):
        rec = verify_enumeration(self.m)
        self.assertEqual(rec["n_eligible_matrices"], 14)
        self.assertTrue(rec["tied_embeddings"])
        acc = parameter_account(self.m)
        self.assertEqual(acc.n_total_naive - acc.n_total_unique, self.m.config.vocab_size * self.m.config.hidden_size)

    def test_prune_and_restore(self):
        snap = WeightSnapshot(self.m)
        shapes = {u: tuple(l.weight.shape) for u, l in eligible_linears(self.m).items()}
        for method in ("magnitude", "wanda"):
            snap.restore(self.m)
            rc = {u: uniform_row_counts(shapes[u], 0.5) for u in shapes}
            info = prune_model(self.m, self.ids, method, rc, batch_size=4)
            nnz, tot = count_nonzeros(self.m)
            self.assertEqual(tot - nnz, tot // 2)
            self.assertIn("block_output_rel_err", info["per_block"][0])
        snap.restore(self.m)
        self.assertEqual(count_nonzeros(self.m)[0], count_nonzeros(self.m)[1])
        info = prune_model(self.m, self.ids, "wanda", None, structure="2:4", batch_size=4)
        nnz, tot = count_nonzeros(self.m)
        self.assertEqual(tot - nnz, tot // 2)

    def test_awq_fold_and_bytes(self):
        snap = WeightSnapshot(self.m)
        ref = self.m(self.ids).logits
        info = quantize_model(self.m, self.ids, uniform_bits(self.m, 8), "awq", batch_size=4)
        self.assertTrue(all(c["rel_err"] < 1e-5 for c in info["fold_checks"]), info["fold_checks"])
        out = self.m(self.ids).logits
        self.assertLess(float((out - ref).abs().max()), 0.05)  # INT8 is near lossless
        exp = sum(packed_bytes(tuple(l.weight.shape), 8)["total"] for l in eligible_linears(self.m).values())
        exp += 2 * 2 * 128  # o_proj FP16 scales, 2 layers
        self.assertEqual(info["eligible_packed_bytes"], exp)
        snap.restore(self.m)
        r3 = quantize_model(self.m, self.ids, uniform_bits(self.m, 3), "rtn", batch_size=4)
        snap.restore(self.m)
        a3 = quantize_model(self.m, self.ids, uniform_bits(self.m, 3), "awq", batch_size=4)
        # alpha grid contains 0 -> AWQ calibration objective can never be worse than RTN per group
        for g in a3["groups"].values():
            self.assertLessEqual(g["loss"], g["loss_rtn"] + 1e-9)
        rep = storage_report(self.m, a3["eligible_packed_bytes"])
        self.assertGreater(rep["dense_fp16_total_bytes"], rep["packed_total_bytes"])

    def test_sensitivity_allocation(self):
        fis = fisher_diagonal(self.m, self.ids[:4])
        self.assertEqual(fis["used"], 4)
        pr = probe(self.m, self.ids, fis["F"], batch_size=4)
        units = pr["units"]
        shapes = {u: tuple(l.weight.shape) for u, l in eligible_linears(self.m).items()}
        N = sum(r * c for r, c in shapes.values())
        for s in (0.5, 0.75):
            us, meta = A.fisher_prune_allocation(units, s, N)
            rc = A.exact_row_counts(us, shapes, int(round(s * N)))
            self.assertEqual(sum(int(v.sum()) for v in rc.values()), int(round(s * N)))
            for u, v in rc.items():
                frac = float(v.sum()) / (shapes[u][0] * shapes[u][1])
                self.assertGreaterEqual(frac, 0.10 - 1.0 / shapes[u][1] - 1e-9)
                self.assertLessEqual(frac, 0.85 + 1.0 / shapes[u][1] + 1e-9)
            D_ = [pr["blocks"][i]["owl"]["5"] for i in range(2)]
            sl = A.rescale_clip(A.owl_sparsities(D_, s, 0.08), [1, 1], s)
            self.assertAlmostEqual(float(sl.mean()), s, places=6)
        budget = sum(packed_bytes(shapes[u], 4)["total"] for u in shapes)
        bits, meta = A.quant_allocation(units, budget, "fisher")
        self.assertLessEqual(meta["bytes"], budget)
        self.assertTrue(set(bits.values()) <= {2, 3, 4, 8})

    def test_window_nll(self):
        wins = [dict(token_ids=self.ids[i].tolist(), score_target_offsets=[1, 64], scored_tokens=63) for i in range(3)]
        r = window_nll(self.m, wins, batch_size=2)
        full = torch.nn.functional.cross_entropy(self.m(self.ids[:3]).logits[:, :-1].reshape(-1, 512),
                                                 self.ids[:3, 1:].reshape(-1), reduction="sum")
        self.assertAlmostEqual(sum(r["nll"]), float(full), places=2)
        self.assertEqual(ppl_summary(r)["N"], 189)

    def test_hs_preprocess(self):
        self.assertEqual(hs_preprocess(" A [title] b [step] c  d "), "A. b c d")


if __name__ == "__main__":
    unittest.main()

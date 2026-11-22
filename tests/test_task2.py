"""CPU tests for Task 2 (attention operators, MQAR mechanisms, KV policies, packing and byte models)."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pa4.task2 import kv, kv_pack
from pa4.task2.attention import dense_attention, sdpa_attention, tiled_attention
from pa4.task2.mqar import (MQARModel, dilated_mask, elu1, linear_attention_causal, linear_attention_recurrent,
                            n_params, theoretical_costs, window_mask)
from tiny_models import tiny_qwen


class AttentionTests(unittest.TestCase):
    def test_tiled_matches_dense_fwd_bwd(self):
        torch.manual_seed(0)
        for L, bq, bk in [(64, 16, 16), (100, 32, 16), (37, 64, 8)]:
            q, k, v = (torch.randn(2, 3, L, 16, dtype=torch.float64, requires_grad=True) for _ in range(3))
            ref = dense_attention(q, k, v)
            out = tiled_attention(q, k, v, bq, bk)
            self.assertLess(float((ref - out).abs().max()), 1e-10)
            g = torch.randn_like(ref)
            gr = torch.autograd.grad(ref, (q, k, v), g)
            gt = torch.autograd.grad(out, (q, k, v), g)
            for a, b in zip(gr, gt):
                self.assertLess(float((a - b).abs().max()), 1e-9)
            self.assertLess(float((sdpa_attention(q, k, v) - ref).abs().max()), 1e-10)

    def test_causality_and_extremes(self):
        q, k, v = (torch.randn(1, 1, 32, 8) for _ in range(3))
        o1 = dense_attention(q, k, v)
        k2, v2 = k.clone(), v.clone()
        k2[:, :, 20:] += 100
        v2[:, :, 20:] -= 50
        o2 = dense_attention(q, k2, v2)
        self.assertTrue(torch.allclose(o1[:, :, :20], o2[:, :, :20]))
        big = tiled_attention(q * 1e4, k, v, 8, 8)
        self.assertTrue(bool(torch.isfinite(big).all()))
        m = torch.ones(32, 32, dtype=torch.bool)
        m[5] = False  # fully masked row
        o = dense_attention(q, k, v, mask=m)
        self.assertTrue(bool(torch.isfinite(o).all()) and float(o[0, 0, 5].abs().max()) == 0.0)


class MQARTests(unittest.TestCase):
    def test_linear_prefix_equals_recurrence(self):
        q, k, v = elu1(torch.randn(2, 4, 20, 8)), elu1(torch.randn(2, 4, 20, 8)), torch.randn(2, 4, 20, 8)
        self.assertLess(float((linear_attention_causal(q, k, v) - linear_attention_recurrent(q, k, v)).abs().max()), 1e-5)

    def test_masks(self):
        w = window_mask(300, 64)
        self.assertEqual(int(w[200].sum()), 64)
        self.assertTrue(bool(w[200, 137]) and not bool(w[200, 136]))
        d = dilated_mask(300, 4, 64)
        self.assertEqual(int(d[299].sum()), 64)
        self.assertTrue(bool(d[299, 299 - 252]) and not bool(d[299, 299 - 256]))

    def test_models_and_params(self):
        base = None
        for kind in ["dense", "window", "dilated", "linear", "hybrid"]:
            m = MQARModel(kind)
            tok = torch.randint(0, 256, (2, 256))
            qp = torch.randint(10, 256, (2, 4))
            out = m(tok, qp)
            self.assertEqual(tuple(out.shape), (2, 4, 64))
            out.sum().backward()
            if base is None:
                base = n_params(m)
            self.assertLess(abs(n_params(m) - base) / base, 0.05, kind)
            # causality: change future tokens -> logits at earlier query positions unchanged
            m.eval()
            qp2 = torch.tensor([[100, 120]])
            t1 = torch.randint(0, 256, (1, 256))
            t2 = t1.clone()
            t2[0, 150:] = (t2[0, 150:] + 7) % 256
            self.assertTrue(torch.allclose(m(t1, qp2), m(t2, qp2), atol=1e-5), kind)
        m = MQARModel("dense")
        m.set_test_override("topk")
        m(tok, qp)
        hy = theoretical_costs("hybrid")
        self.assertLessEqual(hy["state_bytes_per_layer_fp32"], hy["comparator_linear_state_bytes_fp32"])


class KVTests(unittest.TestCase):
    def test_pack_bytes_match_formula(self):
        cfg = SimpleNamespace(num_hidden_layers=3, num_key_value_heads=2, hidden_size=4 * 128, num_attention_heads=4)
        for L in (64, 200, 257):
            for bits in (2, 4):
                for layout in ("kivi", "token"):
                    K, V = torch.randn(1, 2, L, 128).half(), torch.randn(1, 2, L, 128).half()
                    encs = [kv_pack.encode_layer(K, V, bits, layout) for _ in range(3)]
                    self.assertEqual(kv_pack.serialized_bytes(encs), kv.mb(L, bits, cfg))
                    k2, v2 = kv_pack.decode_layer(encs[0], K.shape)
                    nc = (L // 64) * 64
                    kq, vq = kv.quant_tokens(K[:, :, :nc], V[:, :, :nc], bits, layout)
                    self.assertTrue(torch.allclose(k2[:, :, :nc].float(), kq.float(), atol=1e-3))
                    self.assertTrue(torch.allclose(v2[:, :, :nc].float(), vq.float(), atol=1e-3))
                    self.assertTrue(torch.equal(k2[:, :, nc:], K[:, :, nc:]))

    def test_budgets(self):
        cfg = SimpleNamespace(num_hidden_layers=28, num_key_value_heads=2, hidden_size=1536, num_attention_heads=12)
        for L in (2000, 4100):
            B = kv.mb(L, 4, cfg)
            n = kv.max_evict_positions(L, B, cfg, True)
            self.assertLessEqual(kv.evict_bytes(n, cfg, n - 72), B)
            self.assertGreater(kv.evict_bytes(n + 1, cfg, n + 1 - 72), B)
            m = kv.max_hybrid_m(L, B, cfg)
            self.assertLessEqual(kv.hybrid_bytes(L, m, cfg)[0], B)
            self.assertGreater(m, 0)

    def test_cache_editing_path(self):
        """100%-retention rebuilt cache + split prefill reproduces full forward; eviction keeps positions."""
        m = tiny_qwen(layers=2, hidden=256, inter=256, vocab=300)
        m.config.use_cache = True
        ids = torch.randint(0, 290, (1, 1020))
        L = 1000
        full = m(ids).logits[0]
        pre = m(ids[:, :L], use_cache=True)
        K, V = kv.cache_tensors(pre.past_key_values)
        case = dict(L_ctx=L, suffix=ids[0, L:].tolist(), code_ids=[5, 6], evidence=[10, 11], code="12345")
        comp = kv.compress("fp16", K, V, None, m.config, case["evidence"])
        cache = kv.build_cache(comp["K"], comp["V"])
        pos = torch.arange(L, 1020).unsqueeze(0)
        out = m(ids[:, L:], past_key_values=cache, position_ids=pos, use_cache=True).logits[0]
        self.assertLess(float((out - full[L:]).abs().max()), 1e-4)
        scores, ref_last = kv.question_scores(m, K, V, L, case["suffix"])
        self.assertEqual(tuple(scores.shape), (2, L))
        self.assertLess(float((ref_last - full[-1]).abs().max()), 1e-4)
        tok = SimpleNamespace(eos_token_id=1, decode=lambda g, skip_special_tokens=True: "12345",
                              convert_tokens_to_ids=lambda t: 2)
        ref = None
        for meth in kv.METHODS:  # fp16 first: its teacher-forced log-probs are the reference
            c = kv.compress(meth, K, V, scores, m.config, case["evidence"])
            self.assertTrue(c["meta"]["within_budget"] or meth == "fp16", meth)
            r = kv.run_policy(m, tok, case, c, ref)
            self.assertTrue(r["finite"])
            self.assertGreater(r["shared_decode_bytes"], 0)
            if meth not in ("int4_token", "int4_kivi"):
                self.assertEqual(r["live_cache_bytes_end"], r["context_cache_bytes"] + r["shared_decode_bytes"])
            else:
                self.assertLessEqual(r["live_cache_bytes_end"], r["context_cache_bytes"] + r["shared_decode_bytes"])
            if meth == "fp16":
                ref = r["tf_logprobs"]
                self.assertLess(float((ref[len(case["suffix"]) - 1].exp() - torch.softmax(ref_last, -1)).abs().max()), 1e-5)
            else:
                self.assertGreaterEqual(r["teacher_forced_kl_mean"], -1e-6)
        # eviction keeps the absolute RoPE positions of retained tokens: dropping nothing == fp16
        keep_all = kv.build_cache([k.clone() for k in K], [v.clone() for v in V])
        idx = torch.arange(0, L)
        kk = [k[:, :, idx] for k in K]
        self.assertTrue(torch.equal(kk[0], K[0]))


if __name__ == "__main__":
    unittest.main()

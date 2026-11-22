"""Configs are a record of the constants used in code; keep them in sync."""
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pa4.task1 import awq, quant
from pa4.task2 import kv, train_mqar
from pa4.task3 import distill


class ConfigSync(unittest.TestCase):
    def test_sync(self):
        c1 = yaml.safe_load(open(ROOT / "configs/task1.yaml"))
        self.assertEqual(c1["quantization"]["group"], quant.GROUP)
        self.assertEqual(c1["quantization"]["awq_alpha_grid"], awq.ALPHA_GRID)
        c2 = yaml.safe_load(open(ROOT / "configs/task2.yaml"))
        t = c2["B_mqar"]["train"]
        self.assertEqual((t["peak_lr"], t["updates"], t["warmup"], t["batch"], t["final_lr_frac"], t["seeds"], t["eval_at"]),
                         (train_mqar.PEAK_LR, train_mqar.STEPS, train_mqar.WARMUP, train_mqar.BATCH, train_mqar.FINAL_FRAC,
                          train_mqar.SEEDS, train_mqar.EVAL_AT))
        self.assertEqual((c2["C_kv"]["safeguards"]["sink"], c2["C_kv"]["safeguards"]["recent"]), (kv.SINK, kv.RECENT))
        c3 = yaml.safe_load(open(ROOT / "configs/task3.yaml"))["distillation"]
        self.assertEqual(c3["quota"], distill.QUOTA)
        self.assertEqual(c3["caps"], distill.CAP)
        self.assertEqual((c3["optimizer"]["lr"], c3["update_every_scored_tokens"]), (distill.LR, distill.TOKENS_PER_UPDATE))
        self.assertEqual(c3["lora"]["r"], distill.LORA["r"])


if __name__ == "__main__":
    unittest.main()

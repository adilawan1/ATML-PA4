#!/usr/bin/env python
"""Write results/env.json (torch/transformers/CUDA/GPU, repo + data commits, HF revisions)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pa4.common.utils import environment_record, results_root, write_json

rec = environment_record()
write_json(results_root() / "env.json", rec)
print(rec)

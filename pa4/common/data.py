"""Loaders for the locked course data (atml-pa4-datasets). No data is committed to this repo."""
from __future__ import annotations

from pathlib import Path
from typing import List

from .utils import data_root, read_jsonl


def _p(*parts) -> Path:
    p = data_root().joinpath("data", *parts)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run `bash scripts/get_data.sh` and `python external/atml-pa4-datasets/scripts/setup_data.py --public`.")
    return p


# Task 1 ----------------------------------------------------------------------
def c4_calibration() -> List[dict]:
    rows = read_jsonl(_p("public_locked", "c4_calibration_locked.jsonl.gz"))
    assert len(rows) == 128 and all(len(r["token_ids"]) == 512 for r in rows)
    return rows


def c4_development() -> List[dict]:
    rows = read_jsonl(_p("public_locked", "c4_development_locked.jsonl.gz"))
    assert len(rows) == 32 and all(len(r["token_ids"]) == 512 for r in rows)
    return rows


def eval_windows(name: str) -> List[dict]:
    fn = {"wikitext2": "wikitext2_test_windows_locked.jsonl.gz", "c4val": "c4_validation_windows_locked.jsonl.gz"}[name]
    rows = read_jsonl(_p("public_locked", fn))
    assert sum(r["scored_tokens"] for r in rows) == 50000
    return rows


def hellaswag() -> List[dict]:
    rows = read_jsonl(_p("public_locked", "hellaswag_val_locked.jsonl.gz"))
    rows.sort(key=lambda r: r["order"])
    assert len(rows) == 2000
    return rows


# Task 2 ----------------------------------------------------------------------
def mqar(split: str) -> List[dict]:
    fn = {"test": "mqar_test.jsonl.gz", "development": "mqar_development.jsonl.gz"}[split]
    return read_jsonl(_p("mqar", fn))


def kv_cases() -> List[dict]:
    return read_jsonl(_p("kv_retrieval", "kv_retrieval_cases.jsonl.gz"))


# Task 3 ----------------------------------------------------------------------
def gsm8k(split: str) -> List[dict]:
    rows = read_jsonl(_p("public_locked", f"gsm8k_{split}_locked.jsonl.gz"))
    rows.sort(key=lambda r: r["order"])
    return rows


def short_form(split: str) -> List[dict]:
    return read_jsonl(_p("short_form", f"short_{split}.jsonl"))


def prefix_banks() -> List[dict]:
    return read_jsonl(_p("prefix_banks", "prefix_banks_locked.jsonl.gz"))


def mqar_generator():
    import importlib.util

    spec = importlib.util.spec_from_file_location("mqar_generator", data_root() / "scripts" / "mqar_generator.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

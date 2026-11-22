"""Shared helpers: paths, seeding, JSON/JSONL IO, CUDA timing, environment records, bootstrap."""
from __future__ import annotations

import contextlib
import gzip
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
REF_SEED = 6304

# ----------------------------------------------------------------------------- paths


def data_root() -> Path:
    """Location of the cloned course data repository (atml-pa4-datasets)."""
    p = Path(os.environ.get("PA4_DATA", REPO_ROOT / "external" / "atml-pa4-datasets"))
    return p


def results_root() -> Path:
    p = Path(os.environ.get("PA4_RESULTS", REPO_ROOT / "results"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def ckpt_root() -> Path:
    """Large artefacts (adapters, merged drafts, MQAR checkpoints). Not committed."""
    p = Path(os.environ.get("PA4_CKPT", REPO_ROOT / "checkpoints"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def out_dir(*parts: str) -> Path:
    p = results_root().joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------------------------------------------------------- io


def read_jsonl(path) -> List[dict]:
    path = Path(path)
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _atomic_open(path: Path, binary_gz: bool):
    """Write to a sibling temp file; _finish() fsyncs and renames it over `path` (a crash never leaves a torn file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    return tmp


def _finish(tmp: Path, path: Path) -> None:
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_jsonl(path, rows: Iterable[dict]) -> None:
    path = Path(path)
    tmp = _atomic_open(path, path.suffix == ".gz")
    op = gzip.open if path.suffix == ".gz" else open
    with op(tmp, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, default=_json_default) + "\n")
    _finish(tmp, path)


class JsonlAppender:
    """Append-only JSONL log that survives disconnects: every record is flushed + fsynced, and on open a torn
    trailing line (runtime killed mid-write) is truncated away. `done` holds keys of records already present."""

    def __init__(self, path, key_fn):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: List[dict] = []
        if self.path.exists():
            good = 0
            with open(self.path, "rb") as f:
                data = f.read()
            for line in data.splitlines(keepends=True):
                try:
                    if line.strip():
                        self.records.append(json.loads(line))
                    good += len(line)
                except json.JSONDecodeError:
                    break
            if good < len(data):
                with open(self.path, "r+b") as f:
                    f.truncate(good)
        self.key_fn = key_fn
        self.done = {key_fn(r) for r in self.records}
        self.fh = open(self.path, "a", encoding="utf-8")

    def append(self, rec: dict) -> None:
        self.fh.write(json.dumps(rec, default=_json_default) + "\n")
        self.fh.flush()
        os.fsync(self.fh.fileno())
        self.records.append(rec)
        self.done.add(self.key_fn(rec))

    def close(self):
        self.fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    try:
        import torch

        if isinstance(o, torch.Tensor):
            return o.detach().cpu().tolist()
        if isinstance(o, torch.dtype):
            return str(o)
    except ImportError:
        pass
    return str(o)


def write_json(path, obj) -> None:
    path = Path(path)
    tmp = _atomic_open(path, False)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)
    _finish(tmp, path)


def atomic_torch_save(obj, path) -> None:
    import torch

    path = Path(path)
    tmp = _atomic_open(path, False)
    torch.save(obj, tmp)
    _finish(tmp, path)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- seeding


def set_seed(seed: int = REF_SEED, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ----------------------------------------------------------------------------- timing


def cuda_sync():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


class Timer:
    """Wall-clock timer that synchronizes CUDA before start and stop."""

    def __init__(self, sync: bool = True):
        self.sync = sync
        self.elapsed = 0.0

    def __enter__(self):
        if self.sync:
            cuda_sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.sync:
            cuda_sync()
        self.elapsed = time.perf_counter() - self.t0


def time_fn(fn: Callable, reps: int = 5, warmup: int = 2) -> Dict[str, float]:
    """Warm up, then time `reps` CUDA-synchronized calls. Returns seconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        with Timer() as t:
            fn()
        times.append(t.elapsed)
    arr = np.asarray(times)
    return dict(
        unit="s", reps=reps, warmup=warmup, mean=float(arr.mean()), std=float(arr.std(ddof=1) if reps > 1 else 0.0),
        median=float(np.median(arr)), min=float(arr.min()), max=float(arr.max()), all=arr.tolist(),
    )


@contextlib.contextmanager
def peak_memory():
    """Context manager yielding a dict filled with peak allocated / reserved bytes (CUDA)."""
    import torch

    rec = {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    try:
        yield rec
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            rec["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            rec["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
            rec["baseline_allocated_bytes"] = int(base)
            rec["peak_minus_baseline_bytes"] = int(torch.cuda.max_memory_allocated() - base)


# ----------------------------------------------------------------------------- environment


def environment_record(extra: Optional[dict] = None) -> dict:
    rec = dict(
        python=sys.version, platform=platform.platform(), time=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        argv=sys.argv,
    )
    try:
        rec["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        rec["git_commit"] = None
    try:
        rec["data_repo_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=data_root(), text=True).strip()
    except Exception:
        rec["data_repo_commit"] = None
    lock = data_root() / "data" / "local_source_revisions.json"
    if lock.exists():
        rec["source_revisions"] = read_json(lock)
    for mod in ["torch", "transformers", "huggingface_hub", "datasets", "peft", "numpy", "scipy"]:
        try:
            rec[mod] = __import__(mod).__version__
        except Exception:
            rec[mod] = None
    try:
        import torch

        rec["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            rec["cuda_device"] = torch.cuda.get_device_name(0)
            rec["cuda_capability"] = list(torch.cuda.get_device_capability(0))
            rec["cuda_version"] = torch.version.cuda
            rec["cudnn_version"] = torch.backends.cudnn.version()
            rec["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
            # SDPA backends PyTorch may dispatch to (models load with attn_implementation="sdpa")
            rec["sdpa_backends_enabled"] = dict(
                flash=torch.backends.cuda.flash_sdp_enabled(), mem_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
                math=torch.backends.cuda.math_sdp_enabled(),
                cudnn=getattr(torch.backends.cuda, "cudnn_sdp_enabled", lambda: None)())
            rec["tf32_matmul"] = torch.backends.cuda.matmul.allow_tf32
    except Exception:
        pass
    if extra:
        rec.update(extra)
    return rec


def model_revision(repo_id: str) -> Optional[str]:
    """Pinned revision from the data repo's local lock (falls back to None = latest)."""
    lock = data_root() / "data" / "local_source_revisions.json"
    if not lock.exists():
        return None
    obj = read_json(lock)
    for name, rid in obj.get("repos", {}).items():
        if rid == repo_id:
            return obj["revisions"][name]
    return None


# ----------------------------------------------------------------------------- statistics


def paired_bootstrap(
    stat_fn: Callable[[np.ndarray], float], n_items: int, n_resamples: int = 2000, seed: int = REF_SEED,
) -> Dict[str, float]:
    """Generic bootstrap: stat_fn receives an index array (resample of range(n_items))."""
    rng = np.random.default_rng(seed)
    point = stat_fn(np.arange(n_items))
    vals = np.empty(n_resamples)
    for b in range(n_resamples):
        vals[b] = stat_fn(rng.integers(0, n_items, n_items))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return dict(point=float(point), ci_low=float(lo), ci_high=float(hi), n_items=n_items, n_resamples=n_resamples,
                se=float(vals.std(ddof=1)))


def bootstrap_mean_diff(a: Sequence[float], b: Sequence[float], n_resamples: int = 2000, seed: int = REF_SEED):
    """Paired bootstrap CI of mean(a - b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    return paired_bootstrap(lambda idx: float(d[idx].mean()), len(d), n_resamples, seed)


def bootstrap_ppl_diff(nll_a, tok_a, nll_b, tok_b, n_resamples: int = 2000, seed: int = REF_SEED):
    """Paired bootstrap over evaluation windows of PPL(a) - PPL(b): pooled token NLL per resample, then exp."""
    nll_a, nll_b = np.asarray(nll_a, float), np.asarray(nll_b, float)
    tok_a, tok_b = np.asarray(tok_a, float), np.asarray(tok_b, float)
    assert np.array_equal(tok_a, tok_b), "paired windows must have identical scored tokens"

    def stat(idx):
        t = tok_a[idx].sum()
        return math.exp(nll_a[idx].sum() / t) - math.exp(nll_b[idx].sum() / t)

    return paired_bootstrap(stat, len(nll_a), n_resamples, seed)


def human_bytes(n: float) -> str:
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TiB"

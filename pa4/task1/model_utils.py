"""Model loading, eligible-projection enumeration, weight snapshots and parameter/storage accounting."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from ..common.utils import model_revision

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
PROJ_NAMES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# projections that share the same input activation (AWQ groups / Wanda statistics)
INPUT_GROUPS = OrderedDict(
    attn_in=["q_proj", "k_proj", "v_proj"],
    o_in=["o_proj"],
    mlp_in=["gate_proj", "up_proj"],
    down_in=["down_proj"],
)
GROUP_OF = {p: g for g, ps in INPUT_GROUPS.items() for p in ps}


def load_model(name: str = BASE_MODEL, device: str = "cuda", dtype=torch.float16):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rev = model_revision(name)
    tok = AutoTokenizer.from_pretrained(name, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(name, revision=rev, torch_dtype=dtype, attn_implementation="sdpa")
    model = model.to(device).eval()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok, rev


def get_layers(model) -> nn.ModuleList:
    return model.model.layers


def block_linears(layer) -> "OrderedDict[str, nn.Linear]":
    a, m = layer.self_attn, layer.mlp
    return OrderedDict(q_proj=a.q_proj, k_proj=a.k_proj, v_proj=a.v_proj, o_proj=a.o_proj,
                       gate_proj=m.gate_proj, up_proj=m.up_proj, down_proj=m.down_proj)


def eligible_linears(model) -> "OrderedDict[str, nn.Linear]":
    out = OrderedDict()
    for i, layer in enumerate(get_layers(model)):
        for p, lin in block_linears(layer).items():
            out[f"layers.{i}.{p}"] = lin
    return out


def unit_name(i: int, proj: str) -> str:
    return f"layers.{i}.{proj}"


# ----------------------------------------------------------------------------- snapshots


class WeightSnapshot:
    """CPU copy of every eligible weight + every norm weight (AWQ folds into norms / up_proj rows).

    Restoring from this snapshot gives an independent copy of the unchanged FP16 checkpoint for every
    compression condition. A checksum is kept to verify the restore.
    """

    def __init__(self, model):
        self.tensors: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if self._tracked(name):
                self.tensors[name] = p.detach().to("cpu", copy=True)
        self.checksum = self._checksum(model)

    @staticmethod
    def _tracked(name: str) -> bool:
        return ("layers." in name) and (name.endswith(".weight"))

    @staticmethod
    def _checksum(model) -> float:
        s = 0.0
        for name, p in model.named_parameters():
            if WeightSnapshot._tracked(name):
                s += float(p.detach().float().abs().sum().item())
        return s

    @torch.no_grad()
    def restore(self, model) -> None:
        params = dict(model.named_parameters())
        for name, t in self.tensors.items():
            params[name].data.copy_(t.to(params[name].device, non_blocking=True))
        cs = self._checksum(model)
        assert abs(cs - self.checksum) <= 1e-6 * max(1.0, abs(self.checksum)), (cs, self.checksum)


# ----------------------------------------------------------------------------- accounting


@dataclass
class ParamAccount:
    n_eligible: int
    n_other: int          # unique non-eligible params (tied embedding counted once)
    n_total_unique: int
    n_total_naive: int    # sum over named parameters, tied params counted twice
    tied_embeddings: bool
    n_eligible_matrices: int


def parameter_account(model) -> ParamAccount:
    elig = eligible_linears(model)
    elig_ptrs = {lin.weight.data_ptr() for lin in elig.values()}
    seen = set()
    n_elig = n_other = 0
    for _, p in model.named_parameters(remove_duplicate=False):
        ptr = p.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        if ptr in elig_ptrs:
            n_elig += p.numel()
        else:
            n_other += p.numel()
    naive = sum(p.numel() for _, p in model.named_parameters(remove_duplicate=False))
    tied = model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
    return ParamAccount(n_elig, n_other, n_elig + n_other, naive, bool(tied), len(elig))


def verify_enumeration(model) -> dict:
    """Unit checks requested by Task 1 step 1."""
    cfg = model.config
    elig = eligible_linears(model)
    acc = parameter_account(model)
    n_layers = cfg.num_hidden_layers
    assert len(elig) == 7 * n_layers, len(elig)
    expected = 0
    h, kv = cfg.hidden_size, cfg.num_key_value_heads * (cfg.hidden_size // cfg.num_attention_heads)
    inter = cfg.intermediate_size
    expected = n_layers * (h * h + 2 * h * kv + h * h + 3 * h * inter)
    assert acc.n_eligible == expected, (acc.n_eligible, expected)
    # biases/norms/embeddings are not eligible
    for name, lin in elig.items():
        assert isinstance(lin, nn.Linear)
    return dict(
        n_eligible_matrices=len(elig), n_eligible_weights=acc.n_eligible, n_other_unique=acc.n_other,
        n_total_unique=acc.n_total_unique, n_total_naive=acc.n_total_naive, tied_embeddings=acc.tied_embeddings,
        tied_param_counted_once=acc.n_total_unique < acc.n_total_naive if acc.tied_embeddings else True,
        shapes={k: list(v.weight.shape) for k, v in list(elig.items())[:7]},
    )


def count_nonzeros(model) -> Tuple[int, int]:
    nnz = tot = 0
    for lin in eligible_linears(model).values():
        nnz += int((lin.weight != 0).sum().item())
        tot += lin.weight.numel()
    return nnz, tot


def per_unit_sparsity(model) -> Dict[str, float]:
    return {k: float((v.weight == 0).float().mean().item()) for k, v in eligible_linears(model).items()}

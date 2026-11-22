"""Uniform RTN weight quantization (Q1) and AWQ-style activation-aware scaling (Q2/Q3).

What is reproduced from AWQ (Lin et al., 2024):
  * per-input-channel scales s = mean|x|^alpha (normalised by sqrt(max*min)), alpha searched on a grid
    using calibration activations only;
  * the scaled weight W*diag(s) is quantized with the *same* group-wise quantizer as Q1 and the inverse
    scale is folded into the preceding operator where algebraically valid:
      attn_in (q,k,v)   <- input_layernorm weight  (RMSNorm gain / s)
      mlp_in (gate,up)  <- post_attention_layernorm weight
      down_in (down)    <- rows of up_proj  (act(gate) * up is linear in up's output)
      o_in (o_proj)     : NOT foldable into v_proj under GQA (v has 256 outputs, o_proj has 1536
                          inputs), so the FP16 compensating scales are serialized and counted.
  * block-sequential calibration: block l is calibrated on outputs of the already-quantized blocks.
What is NOT reproduced: AWQ's weight-clipping search, its exact module-output objective (we minimise the
summed per-projection output error ||(W - Q(W s)/s) X||^2 of the group), and the fused W4A16 kernels.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from .calib import call_layer, run_sequential
from .model_utils import INPUT_GROUPS, block_linears, unit_name
from .quant import GROUP, fake_quant, quantize_and_pack

ALPHA_GRID = [round(0.1 * i, 1) for i in range(11)]  # 0.0 ... 1.0 (alpha = 0 reproduces RTN)


@torch.no_grad()
def _err_only(w_ref: torch.Tensor, w_eff: torch.Tensor, X: torch.Tensor, chunk: int = 16384) -> float:
    dW = (w_ref.float() - w_eff.float()).to(X.dtype)
    num = 0.0
    for i in range(0, X.shape[0], chunk):
        a = X[i:i + chunk] @ dW.t()
        if not torch.isfinite(a).all():
            a = X[i:i + chunk].float() @ dW.float().t()
        num += a.float().pow(2).sum().item()
    return num


def _make_scale(absmean: torch.Tensor, alpha: float) -> torch.Tensor:
    s = absmean.float().clamp(min=1e-4).pow(alpha)
    s = s / (s.max() * s.min()).sqrt()
    return s


@torch.no_grad()
def search_scale(weights: Dict[str, torch.Tensor], bits: Dict[str, int], X: torch.Tensor, absmean: torch.Tensor,
                 grid: Sequence[float] = ALPHA_GRID, group: int = GROUP) -> dict:
    """Grid search alpha minimising sum_p ||(W_p - Q_b(W_p s)/s) X^T||^2 over projections sharing input X."""
    losses = []
    best = None
    for a in grid:
        s = _make_scale(absmean, a)
        tot = 0.0
        for p, w in weights.items():
            w_eff = fake_quant((w.float() * s).to(w.dtype), bits[p], group).float() / s
            tot += _err_only(w, w_eff, X)
        losses.append(tot)
        if best is None or tot < best[1]:
            best = (a, tot, s)
    return dict(alpha=best[0], loss=best[1], losses=losses, grid=list(grid), scale=best[2])


@torch.no_grad()
def quantize_model(
    model,
    calib_ids: torch.Tensor,
    bits: Dict[str, int],
    method: str = "awq",
    grid: Sequence[float] = ALPHA_GRID,
    batch_size: int = 8,
    group: int = GROUP,
    verify_pack_units: int = 2,
) -> dict:
    """Quantize all eligible projections in place (weights replaced by their dequantized FP16 values).

    method='rtn' -> Q1 (no scaling). method='awq' -> Q2/Q3 (bits may differ per unit).
    Returns per-unit bits/bytes/output errors, per-group alpha search records, folding checks and timing."""
    assert method in ("rtn", "awq")
    units: Dict[str, dict] = {}
    groups: Dict[str, dict] = {}
    fold_checks: List[dict] = []
    extra_bytes = dict(o_proj_fp16_scales=0)
    verified = [0]

    def block_fn(li, layer, inputs, ctx):
        lins = block_linears(layer)
        orig = {p: lin.weight.data.clone() for p, lin in lins.items()}
        b = {p: bits[unit_name(li, p)] for p in lins}
        col_s = {p: None for p in lins}
        row_s = {p: None for p in lins}
        if method == "awq":
            # 1) search scales on the unmodified block (calibration activations only)
            for g, projs in INPUT_GROUPS.items():
                r = search_scale({p: orig[p] for p in projs}, b, inputs[g]["X"], inputs[g]["absmean"], grid, group)
                groups[f"layers.{li}.{g}"] = dict(alpha=r["alpha"], loss=r["loss"], losses=r["losses"],
                                                  loss_rtn=r["losses"][list(grid).index(0.0)] if 0.0 in grid else None)
                for p in projs:
                    col_s[p] = r["scale"]
            # 2) fold inverse scales where algebraically valid; check full-precision equivalence
            h0 = ctx["hs"][0]
            ref = call_layer(layer, h0, ctx["kwargs"]).float()
            s_attn, s_mlp, s_down = col_s["q_proj"], col_s["gate_proj"], col_s["down_proj"]
            dt = orig["q_proj"].dtype
            ndt = layer.input_layernorm.weight.dtype
            layer.input_layernorm.weight.data = (layer.input_layernorm.weight.float() / s_attn).to(ndt)
            for p in ("q_proj", "k_proj", "v_proj"):
                lins[p].weight.data = (orig[p].float() * s_attn).to(dt)
            layer.post_attention_layernorm.weight.data = (layer.post_attention_layernorm.weight.float() / s_mlp).to(ndt)
            lins["gate_proj"].weight.data = (orig["gate_proj"].float() * s_mlp).to(dt)
            lins["up_proj"].weight.data = (orig["up_proj"].float() * s_mlp / s_down.unsqueeze(1)).to(dt)
            row_s["up_proj"] = s_down
            lins["down_proj"].weight.data = (orig["down_proj"].float() * s_down).to(dt)
            out = call_layer(layer, h0, ctx["kwargs"]).float()
            fold_checks.append(dict(layer=li, rel_err=float(((out - ref).norm() / ref.norm()).item()),
                                    max_abs_err=float((out - ref).abs().max().item())))
            extra_bytes["o_proj_fp16_scales"] += 2 * lins["o_proj"].in_features
        # 3) quantize every projection with its assigned bit width
        for p, lin in lins.items():
            u = unit_name(li, p)
            if method == "awq" and p == "o_proj":
                stored = (orig[p].float() * col_s[p]).to(orig[p].dtype)
            else:
                stored = lin.weight.data
            do_verify = verified[0] < verify_pack_units
            deq, rec = quantize_and_pack(stored, b[p], group, verify=do_verify)
            verified[0] += int(do_verify)
            # effective weight in original coordinates (for error measurement)
            eff = deq.float()
            if col_s[p] is not None:
                eff = eff / col_s[p]
            if row_s[p] is not None:
                eff = eff * row_s[p].unsqueeze(1)
            if method == "awq" and p == "o_proj":
                lin.weight.data = eff.to(orig[p].dtype)  # runtime x/s emulated by folding into the dequantized weight
            else:
                lin.weight.data = deq
            X = inputs[_group_of(p)]["X"]
            err = _err_only(orig[p], eff, X)
            den = _err_only(orig[p], torch.zeros_like(orig[p]), X)
            units[u] = dict(bits=b[p], bytes_payload=rec["payload"], bytes_metadata=rec["metadata"],
                            bytes_total=rec["total"], out_err_abs=err, out_err_rel=err / max(den, 1e-30))
        return {}

    res = run_sequential(model, calib_ids, block_fn, batch_size=batch_size, collect_outputs_error=True)
    res.update(units=units, groups=groups, fold_checks=fold_checks, extra_bytes=extra_bytes, method=method,
               grid=list(grid), packing_roundtrip_verified_units=verified[0])
    res["eligible_packed_bytes"] = sum(u["bytes_total"] for u in units.values()) + extra_bytes["o_proj_fp16_scales"]
    return res


def _group_of(p: str) -> str:
    for g, ps in INPUT_GROUPS.items():
        if p in ps:
            return g
    raise KeyError(p)


def uniform_bits(model, b: int) -> Dict[str, int]:
    from .model_utils import eligible_linears

    return {u: b for u in eligible_linears(model)}

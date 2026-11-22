"""Real packed KV-cache encoding/decoding (C5 systems benchmark) for INT4/INT2 with 64-element groups.

Layout per layer for a cache K,V of shape [B, H, L, d]:
  complete tokens nc = 64*floor(L/64) are quantized; the trailing L-nc tokens stay FP16;
  'kivi' : K channel-wise (groups of 64 tokens per channel), V token-wise (groups of 64 features);
  'token': K and V token-wise.
  codes are packed 8/b per byte along the group axis (LSB first); scale and offset are FP16 per group.
serialized_bytes() sums the actual tensor sizes and must equal kv.mb(L, b, cfg) per layer stack.
"""
from __future__ import annotations

from typing import Dict, List

import torch

GROUP = 64


def _pack(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """codes uint8 [..., n] with n % (8/bits) == 0 -> uint8 [..., n*bits/8]."""
    per = 8 // bits
    c = codes.view(*codes.shape[:-1], codes.shape[-1] // per, per).to(torch.int32)
    shifts = (torch.arange(per, device=codes.device, dtype=torch.int32) * bits)
    return (c << shifts).sum(-1).to(torch.uint8)


def _unpack(packed: torch.Tensor, bits: int) -> torch.Tensor:
    per = 8 // bits
    shifts = (torch.arange(per, device=packed.device, dtype=torch.uint8) * bits)
    mask = (1 << bits) - 1
    return ((packed.unsqueeze(-1) >> shifts) & mask).view(*packed.shape[:-1], packed.shape[-1] * per)


def _quant_last(x: torch.Tensor, bits: int):
    """Groups along the last dim (size 64). Returns packed codes, fp16 scale, fp16 offset."""
    xf = x.float()
    mn = xf.amin(-1, keepdim=True)
    mx = xf.amax(-1, keepdim=True)
    qmax = 2 ** bits - 1
    scale = ((mx - mn).clamp(min=1e-6) / qmax).half()
    off = mn.half()
    q = torch.clamp(torch.round((xf - off.float()) / scale.float()), 0, qmax).to(torch.uint8)
    return _pack(q, bits), scale, off


def _dequant_last(packed, scale, off, bits):
    q = _unpack(packed, bits).float()
    return (q * scale.float() + off.float()).half()


@torch.no_grad()
def encode_layer(K: torch.Tensor, V: torch.Tensor, bits: int, layout: str) -> Dict[str, torch.Tensor]:
    B, H, L, d = K.shape
    nc = (L // GROUP) * GROUP
    out = dict(meta=torch.tensor([L, nc, bits, int(layout == "kivi")]))
    Kc, Vc = K[:, :, :nc], V[:, :, :nc]
    if layout == "kivi":
        # [B,H,nc,d] -> [B,H,d,nc/64,64] : groups of 64 tokens along each channel
        kg = Kc.view(B, H, nc // GROUP, GROUP, d).permute(0, 1, 4, 2, 3).contiguous()
    else:
        kg = Kc.view(B, H, nc, d // GROUP, GROUP)
    out["k_codes"], out["k_scale"], out["k_off"] = _quant_last(kg, bits)
    vg = Vc.view(B, H, nc, d // GROUP, GROUP)
    out["v_codes"], out["v_scale"], out["v_off"] = _quant_last(vg, bits)
    out["k_res"] = K[:, :, nc:].contiguous()
    out["v_res"] = V[:, :, nc:].contiguous()
    return out


@torch.no_grad()
def decode_layer(enc: Dict[str, torch.Tensor], shape) -> tuple:
    B, H, L, d = shape
    L_, nc, bits, kivi = [int(x) for x in enc["meta"]]
    k = _dequant_last(enc["k_codes"], enc["k_scale"], enc["k_off"], bits)
    if kivi:
        k = k.permute(0, 1, 3, 4, 2).reshape(B, H, nc, d)
    else:
        k = k.view(B, H, nc, d)
    v = _dequant_last(enc["v_codes"], enc["v_scale"], enc["v_off"], bits).view(B, H, nc, d)
    return torch.cat([k, enc["k_res"]], 2), torch.cat([v, enc["v_res"]], 2)


def serialized_bytes(encs: List[Dict[str, torch.Tensor]]) -> int:
    """Payload + metadata + FP16 residual (the small per-layer header tensor is excluded)."""
    tot = 0
    for e in encs:
        for k, t in e.items():
            if k != "meta":
                tot += t.numel() * t.element_size()
    return tot

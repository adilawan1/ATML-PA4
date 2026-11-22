"""Group-wise asymmetric weight-only quantizer, bit packing and storage accounting.

Conventions (fixed for Q1/Q2/Q3):
  * groups of G=128 consecutive input columns per output row;
  * per group: wmin/wmax over the group (no forced inclusion of zero, as in AWQ's pseudo-quantizer),
    s = max(wmax - wmin, 1e-5) / (2^b - 1) stored as FP16, z = clamp(round(-wmin / s), 0, 2^b - 1)
    stored as one uint8 byte;
  * q = clamp(round(w / s) + z, 0, 2^b - 1) with torch.round (round-half-to-even);
  * dequantization ŵ = s * (q - z) using the FP16-rounded scale (what a kernel would read);
  * if in_features is not a multiple of G the row is zero-padded to the next multiple (padding is
    stored and counted). Qwen2.5-1.5B widths (1536, 8960) are multiples of 128, so no padding occurs;
  * packing: every row is packed as a little-endian bitstream of b-bit codes into ceil(cols*b/8) bytes.
Metadata per group = 2 bytes (FP16 scale) + 1 byte (uint8 zero-point) = 3 bytes.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

GROUP = 128
META_BYTES_PER_GROUP = 3  # fp16 scale + uint8 zero-point


def _pad_cols(w: torch.Tensor, group: int) -> Tuple[torch.Tensor, int]:
    cols = w.shape[1]
    pad = (-cols) % group
    if pad:
        w = torch.nn.functional.pad(w, (0, pad))
    return w, pad


@torch.no_grad()
def quantize_groupwise(w: torch.Tensor, bits: int, group: int = GROUP):
    """Return (dequantized weight in w.dtype, codes uint8 [rows, cols_padded], scale fp16, zero uint8)."""
    assert 1 <= bits <= 8
    rows, cols = w.shape
    W, pad = _pad_cols(w.float(), group)
    ng = W.shape[1] // group
    Wg = W.view(rows, ng, group)
    wmax = Wg.amax(dim=-1, keepdim=True)
    wmin = Wg.amin(dim=-1, keepdim=True)
    qmax = 2 ** bits - 1
    scale = ((wmax - wmin).clamp(min=1e-5) / qmax).half().float()  # FP16-rounded scale
    zero = torch.clamp(torch.round(-wmin / scale), 0, qmax)
    q = torch.clamp(torch.round(Wg / scale) + zero, 0, qmax)
    deq = ((q - zero) * scale).view(rows, -1)[:, :cols]
    return (deq.to(w.dtype), q.view(rows, -1).to(torch.uint8), scale.view(rows, ng).half(),
            zero.view(rows, ng).to(torch.uint8))


@torch.no_grad()
def fake_quant(w: torch.Tensor, bits: int, group: int = GROUP) -> torch.Tensor:
    return quantize_groupwise(w, bits, group)[0]


# ----------------------------------------------------------------------------- packing


@torch.no_grad()
def pack_bits(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack uint8 codes [rows, cols] (values < 2^bits) into uint8 [rows, ceil(cols*bits/8)] (LSB-first bitstream)."""
    rows, cols = codes.shape
    if bits == 8:
        return codes.contiguous().clone()
    shifts = torch.arange(bits, device=codes.device, dtype=torch.uint8)
    bitplanes = ((codes.unsqueeze(-1) >> shifts) & 1).view(rows, cols * bits)  # LSB first per code
    nbytes = math.ceil(cols * bits / 8)
    padb = nbytes * 8 - cols * bits
    if padb:
        bitplanes = torch.nn.functional.pad(bitplanes, (0, padb))
    bitplanes = bitplanes.view(rows, nbytes, 8).to(torch.int32)
    weights = (1 << torch.arange(8, device=codes.device, dtype=torch.int32))
    return (bitplanes * weights).sum(-1).to(torch.uint8)


@torch.no_grad()
def unpack_bits(packed: torch.Tensor, bits: int, cols: int) -> torch.Tensor:
    rows, nbytes = packed.shape
    if bits == 8:
        return packed[:, :cols].clone()
    shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
    bitstream = ((packed.unsqueeze(-1) >> shifts) & 1).view(rows, nbytes * 8)[:, : cols * bits]
    bitstream = bitstream.view(rows, cols, bits).to(torch.int32)
    weights = (1 << torch.arange(bits, device=packed.device, dtype=torch.int32))
    return (bitstream * weights).sum(-1).to(torch.uint8)


def packed_bytes(shape: Tuple[int, int], bits: int, group: int = GROUP) -> Dict[str, int]:
    """Analytic serialized size of one quantized matrix (must equal the actual packed tensors)."""
    rows, cols = shape
    cols_p = cols + ((-cols) % group)
    payload = rows * math.ceil(cols_p * bits / 8)
    meta = rows * (cols_p // group) * META_BYTES_PER_GROUP
    return dict(payload=payload, metadata=meta, total=payload + meta, padding_cols=cols_p - cols)


@torch.no_grad()
def quantize_and_pack(w: torch.Tensor, bits: int, group: int = GROUP, verify: bool = False):
    """Quantize, pack, and return (dequantized weight, byte record). Optionally verify the round trip."""
    deq, codes, scale, zero = quantize_groupwise(w, bits, group)
    packed = pack_bits(codes, bits)
    rec = dict(payload=packed.numel(), metadata=scale.numel() * 2 + zero.numel() * 1)
    rec["total"] = rec["payload"] + rec["metadata"]
    ana = packed_bytes(tuple(w.shape), bits, group)
    assert rec["total"] == ana["total"], (rec, ana)
    if verify:
        un = unpack_bits(packed, bits, codes.shape[1])
        assert torch.equal(un, codes), "pack/unpack round trip failed"
        ng = scale.shape[1]
        rebuilt = ((un.float().view(w.shape[0], ng, group) - zero.float().unsqueeze(-1)) * scale.float().unsqueeze(-1))
        rebuilt = rebuilt.view(w.shape[0], -1)[:, : w.shape[1]].to(w.dtype)
        assert torch.equal(rebuilt, deq), "dequantization from packed codes differs"
    return deq, rec


# ----------------------------------------------------------------------------- storage models


def fp16_bytes(n_params: int) -> int:
    return 2 * n_params


def sparse16_bytes(numel: int, nnz: int) -> int:
    """Specified sparse model: 1 mask bit per eligible weight (byte-padded per matrix) + 16 bits per nonzero."""
    return math.ceil(numel / 8) + 2 * nnz

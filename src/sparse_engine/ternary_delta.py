"""Ternary delta coding, and what it is actually worth.

WHY THE OBVIOUS FRAMING IS WRONG
    Compressing cross-machine updates cannot speed up the chosen two-node
    design, because that design already gives every expert a single owner, so
    expert gradients never cross the link at all. Communication is 0.5 of 23.3
    days; removing it entirely is a 1.02x speedup. Optimising it is optimising
    a solved problem.

WHAT IT IS ACTUALLY FOR
    The reason experts are owned outright is that sharing them was too
    expensive: data-parallel training needs every expert gradient reduced
    across a 0.0135 GB/s link, which measured at 4.4 GB/step and 40 days of
    communication. Sharding dodged that, but at a cost -- each token can only
    route among its node's 35,000 experts instead of all 70,000.

    So delta compression is not a speed optimisation. It is the thing that
    decides whether the ARCHITECTURE can be data-parallel, which restores the
    full routing pool. The bar is concrete:

        budget    ~0.054 GB/step   (to keep comm under ~2% as today)
        required  4.4 / 0.054      = 81x compression, minimum

    Below 81x, sharding stands and the pool stays halved. Above it, both
    machines can hold the whole table and every token sees every expert.

THE CODING SCHEME
    Ternary weights change state rarely: a latent value must cross a
    quantisation threshold to flip. Between syncs most weights are unchanged,
    so the delta is a sparse list of (position, new_state).

    Positions are encoded as GAPS rather than absolute indices, with a
    variable-length byte code (7 payload bits, 1 continuation bit) plus 2 bits
    of state. A fixed 6-bit gap field -- as one proposal suggested -- overflows
    almost always: at a 0.1% flip rate the mean gap is ~1000, far beyond the
    63 a 6-bit field can hold, so "1 byte per change" is not achievable at the
    sparsity that makes the scheme worth using.
"""
from __future__ import annotations

import torch

# States are packed 2 bits each: 0 -> 00, +1 -> 01, -1 -> 10.
STATE_BITS = 2
RAW_BITS_PER_WEIGHT = STATE_BITS


def quantize(latent: torch.Tensor, threshold: float) -> torch.Tensor:
    """Ternary quantisation: -1, 0, +1 as int8."""
    return (torch.sign(latent) * (latent.abs() > threshold)).to(torch.int8)


def encode_delta(before: torch.Tensor, after: torch.Tensor) -> bytearray:
    """Gap-coded (position, state) pairs for every weight that flipped.

    Byte layout, little-endian gap first:
        continuation | 7 gap bits      repeated while continuation set
        final byte   | 7 gap bits
        one byte     | 2 state bits
    """
    if before.shape != after.shape:
        raise ValueError("shapes must match")
    flat_b = before.reshape(-1)
    flat_a = after.reshape(-1)
    idx = (flat_b != flat_a).nonzero(as_tuple=True)[0]
    out = bytearray()
    prev = -1
    for i in idx.tolist():
        gap = i - prev - 1
        prev = i
        while gap >= 0x80:
            out.append(0x80 | (gap & 0x7F))
            gap >>= 7
        out.append(gap & 0x7F)
        v = int(flat_a[i])
        out.append(0b01 if v > 0 else (0b10 if v < 0 else 0b00))
    return out


def decode_delta(before: torch.Tensor, blob: bytes) -> torch.Tensor:
    """Exact inverse of encode_delta. Round-trip fidelity is not optional:
    a lossy sync silently desynchronises the two machines."""
    out = before.clone().reshape(-1)
    pos = -1
    i = 0
    n = len(blob)
    while i < n:
        gap = 0
        shift = 0
        while True:
            b = blob[i]
            i += 1
            gap |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        pos = pos + gap + 1
        code = blob[i]
        i += 1
        out[pos] = 1 if code == 0b01 else (-1 if code == 0b10 else 0)
    return out.reshape(before.shape)


def compression_ratio(n_weights: int, blob_len: int) -> float:
    """Against the packed 2-bit representation, not against fp32.

    Comparing to fp32 would inflate the number 16x for free and is the
    flattering denominator; the design already stores ternary.
    """
    raw_bytes = n_weights * RAW_BITS_PER_WEIGHT / 8
    return raw_bytes / max(1, blob_len)

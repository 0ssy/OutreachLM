"""Tests for ternary delta coding.

Round-trip fidelity is the load-bearing property: a lossy sync desynchronises
the two machines silently, and the divergence would surface much later as a
quality problem with no obvious cause.
"""
import pytest
import torch

from src.sparse_engine.ternary_delta import (
    compression_ratio,
    decode_delta,
    encode_delta,
    quantize,
)


def _states(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return quantize(torch.randn(n, generator=g), 0.5)


def test_round_trip_is_exact_for_random_deltas():
    for seed in range(5):
        before = _states(4096, seed)
        after = _states(4096, seed + 100)
        blob = encode_delta(before, after)
        assert torch.equal(decode_delta(before, bytes(blob)), after)


def test_round_trip_handles_no_change_and_total_change():
    before = _states(1024)
    assert len(encode_delta(before, before.clone())) == 0
    assert torch.equal(decode_delta(before, b""), before)

    flipped = ((before.int() + 1) % 3).to(torch.int8) - 1
    flipped = torch.where(flipped == before, before + 1, flipped)
    flipped = flipped.clamp(-1, 1).to(torch.int8)
    blob = encode_delta(before, flipped)
    assert torch.equal(decode_delta(before, bytes(blob)), flipped)


def test_round_trip_survives_very_large_gaps():
    """Gaps far beyond a single byte must cascade correctly. A fixed 6-bit
    field cannot represent these at all, which is why the encoding is
    variable-length."""
    n = 1 << 20
    before = torch.zeros(n, dtype=torch.int8)
    after = before.clone()
    after[0] = 1
    after[n - 1] = -1
    blob = encode_delta(before, after)
    assert torch.equal(decode_delta(before, bytes(blob)), after)


def test_shape_is_preserved():
    before = _states(64).reshape(8, 8)
    after = _states(64, 7).reshape(8, 8)
    blob = encode_delta(before, after)
    out = decode_delta(before, bytes(blob))
    assert out.shape == (8, 8)
    assert torch.equal(out, after)


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError):
        encode_delta(_states(16), _states(32))


def test_compression_is_measured_against_the_packed_form():
    """Comparing against fp32 would inflate every ratio 16x for free; the
    design already stores ternary, so that is the flattering denominator."""
    n = 1 << 16
    assert compression_ratio(n, n * 2 // 8) == pytest.approx(1.0)


def test_sparse_deltas_compress_and_dense_ones_do_not():
    n = 1 << 16
    g = torch.Generator().manual_seed(3)
    before = _states(n)

    sparse = before.clone()
    idx = torch.randperm(n, generator=g)[: n // 1000]
    sparse[idx] = -before[idx]
    r_sparse = compression_ratio(n, len(encode_delta(before, sparse)))

    dense = -before
    r_dense = compression_ratio(n, len(encode_delta(before, dense)))

    assert r_sparse > 20.0
    assert r_dense < 1.0        # gap-coding a dense change is a pessimisation

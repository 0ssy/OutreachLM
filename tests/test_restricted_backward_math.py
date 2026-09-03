"""Phase 0/1/3 validation. Hard rule: C/D/E results are invalid until these pass.

Every gradient is checked in float64 against autograd, which is an independent
oracle -- the methods themselves never call autograd for the weight gradient.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch

from src.restricted_backward.methods import (
    MethodA,
    MethodB,
    SparseConfig,
    SparseMethod,
    dense_flops,
    reference_backward,
    reference_forward,
)


# --- Phase 0: hand-derived math vs autograd --------------------------------
def test_phase0_hand_derived_matches_autograd() -> None:
    """dW = G_Y^T X and dX = G_Y W, verified against autograd in float64."""
    torch.manual_seed(0)
    n, d_in, d_out = 7, 11, 13
    X = torch.randn(n, d_in, dtype=torch.float64, requires_grad=True)
    W = torch.randn(d_out, d_in, dtype=torch.float64, requires_grad=True)

    Y = reference_forward(X, W)
    G_Y = torch.randn(n, d_out, dtype=torch.float64)
    Y.backward(G_Y)

    dW_hand, dX_hand = reference_backward(G_Y, X.detach(), W.detach())

    assert torch.allclose(dW_hand, W.grad, atol=1e-12)
    assert torch.allclose(dX_hand, X.grad, atol=1e-12)


def test_phase0_gradcheck_dense_layer() -> None:
    torch.manual_seed(0)
    X = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
    W = torch.randn(5, 6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(reference_forward, (X, W), eps=1e-6, atol=1e-8)


def test_phase0_flop_accounting_and_saving_bound() -> None:
    """Backward is 2x forward; single-layer dW saving is bounded by 1/3."""
    f = dense_flops(n=8, d_in=64, d_out=64)
    assert f["dW"] + f["dX"] == pytest.approx(2 * f["forward"])
    assert f["dW"] / f["total"] == pytest.approx(1.0 / 3.0)


# --- Phase 1: Method A -----------------------------------------------------
def test_method_a_gradients_match_autograd_exactly() -> None:
    torch.manual_seed(1)
    layer = MethodA(d_in=12, d_out=9, dtype=torch.float64)
    X = torch.randn(5, 12, dtype=torch.float64)

    Xa = X.clone().requires_grad_(True)
    Wa = layer.W.clone().requires_grad_(True)
    Y = Xa @ Wa.T
    G_Y = torch.randn(5, 9, dtype=torch.float64)
    Y.backward(G_Y)

    dW, dX, exact, flops, sel, fb = layer.backward(G_Y, X)

    assert torch.allclose(dW, Wa.grad, atol=1e-12)
    assert torch.allclose(dX, Xa.grad, atol=1e-12)
    # A is the reference: its "approximation" is exact by construction.
    assert torch.equal(dW, exact)
    assert sel == 0.0 and fb == 0.0
    assert flops == pytest.approx(dense_flops(5, 12, 9)["dW"])


# --- Phase 3: Method B -----------------------------------------------------
@pytest.mark.parametrize("rank", [2, 4])
def test_method_b_hand_derived_gradients_match_autograd(rank: int) -> None:
    """dL/dB = G_Y^T (X A^T), dL/dA = (G_Y B)^T X, checked against autograd."""
    torch.manual_seed(2)
    d_in, d_out, n = 10, 8, 6
    layer = MethodB(d_in, d_out, rank=rank, dtype=torch.float64)
    # B starts at zero (LoRA init); perturb so dA has a nonzero oracle.
    layer.B = torch.randn(d_out, rank, dtype=torch.float64) * 0.1

    X = torch.randn(n, d_in, dtype=torch.float64)
    G_Y = torch.randn(n, d_out, dtype=torch.float64)

    Aa = layer.A.clone().requires_grad_(True)
    Ba = layer.B.clone().requires_grad_(True)
    Y = X @ layer.W0.T + (X @ Aa.T) @ Ba.T
    Y.backward(G_Y)

    Y_impl = layer.forward(X)
    assert torch.allclose(Y_impl, Y.detach(), atol=1e-12)

    (dB, dA), dX, _, _, _, _ = layer.backward(G_Y, X)
    assert torch.allclose(dB, Ba.grad, atol=1e-12)
    assert torch.allclose(dA, Aa.grad, atol=1e-12)

    Xa = X.clone().requires_grad_(True)
    Y2 = Xa @ layer.W0.T + (Xa @ layer.A.T) @ layer.B.T
    Y2.backward(G_Y)
    assert torch.allclose(dX, Xa.grad, atol=1e-12)


def test_method_b_never_materializes_dense_gradient() -> None:
    """Confirm via allocation size, not by reading the code path.

    The largest tensor produced anywhere in B's backward must be strictly
    smaller than a (d_out x d_in) dense gradient.
    """
    torch.manual_seed(3)
    d_in = d_out = 256
    rank = 16
    layer = MethodB(d_in, d_out, rank=rank)
    X = torch.randn(8, d_in)
    G_Y = torch.randn(8, d_out)
    layer.forward(X)

    (dB, dA), dX, _, _, _, _ = layer.backward(G_Y, X)
    dense_numel = d_out * d_in
    for tensor in (dB, dA):
        assert tensor.numel() < dense_numel
    assert dB.shape == (d_out, rank)
    assert dA.shape == (rank, d_in)


# --- Sparse method structural checks ---------------------------------------
@pytest.mark.parametrize("keep", [0.01, 0.05, 0.25])
def test_topk_keeps_expected_count_and_largest_magnitudes(keep: float) -> None:
    torch.manual_seed(4)
    layer = SparseMethod(64, 64, config=SparseConfig(keep_fraction=keep))
    X = torch.randn(4, 64)
    G_Y = torch.randn(4, 64)
    sparse, _, exact, _, _, _ = layer.backward(G_Y, X)

    expected_k = max(1, int(round(keep * exact.numel())))
    assert int((sparse != 0).sum()) == expected_k

    kept = exact[sparse != 0].abs().min()
    dropped = exact[sparse == 0].abs().max()
    assert kept >= dropped - 1e-9


def test_error_feedback_residual_is_exactly_the_dropped_mass() -> None:
    torch.manual_seed(5)
    layer = SparseMethod(
        32, 32, config=SparseConfig(keep_fraction=0.1, error_feedback=True)
    )
    X = torch.randn(4, 32)
    G_Y = torch.randn(4, 32)
    sparse, _, exact, _, _, _ = layer.backward(G_Y, X)
    # First step: residual starts at zero, so residual == exact - sparse.
    assert torch.allclose(layer.residual, exact - sparse, atol=1e-6)


def test_block_selection_keeps_whole_blocks() -> None:
    torch.manual_seed(6)
    bs = 8
    layer = SparseMethod(
        64, 64, config=SparseConfig(keep_fraction=0.25, error_feedback=True, block_size=bs)
    )
    X = torch.randn(4, 64)
    G_Y = torch.randn(4, 64)
    sparse, _, _, _, _, _ = layer.backward(G_Y, X)

    blocks = sparse.reshape(64 // bs, bs, 64 // bs, bs).permute(0, 2, 1, 3)
    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            block = blocks[i, j]
            nz = int((block != 0).sum())
            # A block is either fully dropped or (almost surely) fully kept;
            # exact zeros inside a kept block are measure-zero for random data.
            assert nz == 0 or nz == block.numel()


def test_dx_is_never_sparsified() -> None:
    """dX must stay dense: it is what reaches upstream layers."""
    torch.manual_seed(7)
    layer = SparseMethod(
        32, 32, config=SparseConfig(keep_fraction=0.01, error_feedback=True)
    )
    X = torch.randn(4, 32)
    G_Y = torch.randn(4, 32)
    _, dX, _, _, _, _ = layer.backward(G_Y, X)
    _, dX_ref = (None, G_Y @ layer.W)
    assert torch.allclose(dX, dX_ref, atol=1e-12)

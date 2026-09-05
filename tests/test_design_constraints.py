"""Tests for the pre-implementation design constraints.

These verify the checker itself classifies known-good and known-bad operators
correctly, so it can be trusted as a gate for future mechanisms.
"""
import torch

from src.restricted_backward.design_constraints import (
    admits_error_feedback,
    gaussian_sketch,
    orthonormal_sketch,
    residual_expansion_factor,
    topk_rows,
)
from src.restricted_backward.true_restricted import MethodF
from src.restricted_backward.unbiased_sketch import (
    SketchWithFeedback,
    UnbiasedSketch,
)

N, D = 32, 128


def test_topk_is_contractive_and_admits_feedback():
    """Method G's compressor. Orthogonal projection, norm exactly 1."""
    for keep in (0.05, 0.25, 0.5):
        f = residual_expansion_factor(topk_rows(keep), N, D)
        assert f <= 1.0 + 1e-5, (keep, f)
        assert admits_error_feedback(topk_rows(keep), N, D)


def test_gaussian_sketch_is_expansive_and_is_rejected():
    """Method M's FIRST version. The checker must catch this before code."""
    for s in (2, 8, 16):
        f = residual_expansion_factor(gaussian_sketch(s), N, D)
        assert f > 1.0, (s, f)
        assert not admits_error_feedback(gaussian_sketch(s), N, D)


def test_orthonormal_sketch_is_contractive_and_admits_feedback():
    """Method M's SHIPPED version. Biased per step, safe under feedback."""
    for s in (2, 8, 16):
        assert admits_error_feedback(orthonormal_sketch(s), N, D)


def test_unbiasedness_and_contractivity_are_in_conflict():
    """The corollary, verified rather than asserted rhetorically.

    Gaussian P is unbiased (E[P P^T] = I) but expansive; orthonormal Q is
    contractive but biased (E[Q Q^T] = (s/n) I != I). No operator here is both.
    """
    s = 8
    g = torch.Generator().manual_seed(5)
    v = torch.randn(N, D, generator=g)

    gauss, ortho = gaussian_sketch(s, seed=1), orthonormal_sketch(s, seed=1)
    acc_g = torch.zeros_like(v)
    acc_o = torch.zeros_like(v)
    trials = 400
    for _ in range(trials):
        acc_g += gauss(v)
        acc_o += ortho(v)
    bias_g = float((acc_g / trials - v).norm() / v.norm())
    bias_o = float((acc_o / trials - v).norm() / v.norm())

    assert bias_g < 0.2                                    # unbiased
    assert bias_o > 0.5                                    # biased
    assert residual_expansion_factor(gauss, N, D) > 1.0     # expansive
    assert residual_expansion_factor(ortho, N, D) <= 1.0 + 1e-5


def test_shipped_methods_satisfy_the_constraint_they_rely_on():
    """Ties the abstract rule to the concrete classes that use feedback."""
    m_g = MethodF(D, D, keep_fraction=0.05, error_feedback=True, seed=0)

    def g_compress(v):
        k = max(1, int(round(0.05 * v.shape[1])))
        idx = torch.topk(v.norm(dim=0), k, sorted=False).indices
        out = torch.zeros_like(v)
        out[:, idx] = v[:, idx]
        return out

    assert admits_error_feedback(g_compress, N, D)

    m_m = SketchWithFeedback(D, D, sketch_dim=8, n_params=16, seed=0)
    assert admits_error_feedback(lambda v: (lambda Q: Q @ (Q.T @ v))(
        m_m._draw(v.shape[0])), N, D)


def test_plain_L_is_unbiased_and_therefore_must_not_use_feedback():
    """L is correct precisely because it has NO bank. If someone later adds
    error feedback to it without switching the sketch, this documents why
    that would diverge."""
    m = UnbiasedSketch(D, D, sketch_dim=8, seed=0)
    assert not hasattr(m, "_R")
    f = residual_expansion_factor(
        lambda v: (lambda P: P @ (P.T @ v))(m._draw(v.shape[0])), N, D
    )
    assert f > 1.0

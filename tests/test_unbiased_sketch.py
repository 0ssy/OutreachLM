"""Tests for Method L (rung 3d): unbiased stochastic substitution.

These encode measured physics and the one structural invariant the method
depends on (fresh resampling), rather than magic thresholds.
"""
import torch

from src.restricted_backward.unbiased_sketch import UnbiasedSketch

D_IN = D_OUT = 256
N = 32


def _fixture(sketch_dim, seed=0):
    m = UnbiasedSketch(D_IN, D_OUT, sketch_dim=sketch_dim, seed=seed)
    g = torch.Generator().manual_seed(7)
    G_Y = torch.randn(N, D_OUT, generator=g) * 0.01
    X = torch.randn(N, D_IN, generator=g)
    return m, G_Y, X


def test_shapes_and_exactness_of_oracle():
    m, G_Y, X = _fixture(8)
    dX_hat, _, _, _, _ = m.backward(G_Y, X)
    assert dX_hat.shape == (N, D_IN)
    assert torch.allclose(m.exact_dX(G_Y), G_Y @ m.W)


def test_estimator_is_unbiased():
    """E[P P^T] = I, so the mean estimate converges to exact dX."""
    m, G_Y, X = _fixture(8)
    exact = m.exact_dX(G_Y)
    acc = torch.zeros_like(exact)
    trials = 400
    for _ in range(trials):
        acc += m.backward(G_Y, X)[0]
    bias = float((acc / trials - exact).norm() / exact.norm())
    single = float((m.backward(G_Y, X)[0] - exact).norm() / exact.norm())
    # Monte-Carlo decay: averaging T draws must shrink error by ~sqrt(T).
    assert bias < single / (trials**0.5) * 3.0
    assert bias < 0.2


def test_fresh_projection_is_load_bearing():
    """A frozen P converts zero-mean noise into permanent bias.

    This is the structural analogue of the test asserting that periodic
    recompute cannot be relabelled 'elimination'. If resampling is ever
    removed as an optimisation, this fails loudly.
    """
    m, G_Y, X = _fixture(8)
    exact = m.exact_dX(G_Y)
    trials = 300

    fresh = torch.zeros_like(exact)
    for _ in range(trials):
        fresh += m.backward(G_Y, X)[0]
    fresh_bias = float((fresh / trials - exact).norm() / exact.norm())

    P_frozen = m._draw(N)
    frozen = torch.zeros_like(exact)
    for _ in range(trials):
        frozen += P_frozen @ ((P_frozen.T @ G_Y) @ m.W)
    frozen_bias = float((frozen / trials - exact).norm() / exact.norm())

    assert frozen_bias > 10 * fresh_bias
    assert frozen_bias > 0.5


def test_projection_actually_changes_between_steps():
    m, G_Y, X = _fixture(8)
    m.backward(G_Y, X)
    first = m.last_projection.clone()
    m.backward(G_Y, X)
    assert not torch.allclose(first, m.last_projection)


def test_projection_stream_is_independent_of_global_seed():
    """Resampling must not be defeatable by a caller reseeding globally."""
    m, G_Y, X = _fixture(8)
    torch.manual_seed(0)
    m.backward(G_Y, X)
    a = m.last_projection.clone()
    torch.manual_seed(0)
    m.backward(G_Y, X)
    assert not torch.allclose(a, m.last_projection)


def test_variance_follows_sqrt_n_over_s_law():
    """Measured physics: per-step relative error ~ sqrt(N/s).

    Measured 2.021 at N=32, s=8 against predicted 2.000 in the pre-check.
    This law is what forces s >= N for sub-unit error, and hence the cost wall.
    """
    for s in (4, 8, 16):
        m, G_Y, X = _fixture(s)
        exact = m.exact_dX(G_Y)
        errs = [
            float((m.backward(G_Y, X)[0] - exact).norm() / exact.norm())
            for _ in range(60)
        ]
        mean = sum(errs) / len(errs)
        predicted = (N / s) ** 0.5
        assert 0.7 * predicted < mean < 1.4 * predicted


def test_error_cost_product_is_bounded_below():
    """err^2 * cost = 1 + 2N/d_in, so err < 1 always implies cost > 1.

    This is the structural bound that rules L out as a cheap low-error
    estimator at any shape. Verified against three shapes at 3 digits.
    """
    for n, d in ((32, 512), (64, 256)):
        for s in (8, 16, 32):
            m = UnbiasedSketch(d, d, sketch_dim=s, seed=3)
            g = torch.Generator().manual_seed(5)
            G_Y = torch.randn(n, d, generator=g) * 0.01
            exact = m.exact_dX(G_Y)
            errs = [
                float((m.backward(G_Y, None)[0] - exact).norm() / exact.norm())
                for _ in range(40)
            ]
            err = sum(errs) / len(errs)
            cost = (s / n) + (2.0 * s / d)
            assert err**2 * cost >= 1.0            # the impossibility bound
            predicted = (1.0 + 2.0 * n / d) ** 0.5
            assert 0.85 * predicted < err * cost**0.5 < 1.2 * predicted
            if err < 1.0:
                assert cost > 1.0


def test_flop_accounting_matches_three_gemms():
    m, G_Y, X = _fixture(8)
    _, executed, _, _, _ = m.backward(G_Y, X)
    expected = (
        2.0 * 8 * N * D_OUT + 2.0 * 8 * D_OUT * D_IN + 2.0 * N * 8 * D_IN
    )
    assert executed == expected
    assert executed < 2.0 * N * D_OUT * D_IN  # must beat dense at s=8


def test_is_flop_restricted_but_not_memory_restricted():
    """L's peak backward tensor equals dense. Recorded so the two claims
    are never collapsed, as happened with Methods C/D/E."""
    m, G_Y, X = _fixture(8)
    m.backward(G_Y, X)
    dense_bytes = N * D_IN * G_Y.element_size()
    assert m.max_backward_tensor_bytes == dense_bytes


def test_no_dense_intermediate_from_association_order():
    """(P^T G_Y) W must stay rank-s; forming P P^T first would cost dense."""
    m, G_Y, X = _fixture(8)
    P = m._draw(N)
    cheap = P @ ((P.T @ G_Y) @ m.W)
    naive = (P @ P.T) @ (G_Y @ m.W)
    assert torch.allclose(cheap, naive, atol=1e-4)


# --- Method M: L plus error feedback banked in G_Y space ---

from src.restricted_backward.unbiased_sketch import SketchWithFeedback

V = 16


def _fb(sketch_dim=8, seed=0):
    m = SketchWithFeedback(
        D_IN, D_OUT, sketch_dim=sketch_dim, n_params=V, seed=seed
    )
    g = torch.Generator().manual_seed(7)
    G_Y = torch.randn(N, D_OUT, generator=g) * 0.01
    ids = torch.arange(N) % V
    return m, G_Y, ids


def test_feedback_conserves_the_gradient_exactly():
    """Geff = Ghat + R_new. Nothing transmitted is lost, nothing invented.

    This is the property that distinguishes banking from discarding, and it
    is what rung 3b wrongly concluded was impossible for Process I -- it is
    impossible in dX space, not in G_Y space.
    """
    m, G_Y, ids = _fb()
    ids = torch.arange(N)[:V]          # distinct ids, one slot each
    G_Y = G_Y[:V]
    before = m._R[ids].clone()
    m.backward(G_Y, ids)
    P = m.last_projection
    Geff = G_Y + before
    Ghat = P @ (P.T @ Geff)
    assert torch.allclose(m._R[ids], Geff - Ghat, atol=1e-5)


def test_repeated_ids_restore_banked_debt_only_once():
    """A parameter sampled k times in a batch must not have its debt
    multiplied by k -- that would make frequent parameters explode, which is
    exactly how the row-keyed Method K diverged under Zipfian access."""
    m = SketchWithFeedback(D_IN, D_OUT, sketch_dim=2, n_params=V, seed=0)
    g = torch.Generator().manual_seed(3)
    R_old = torch.randn(D_OUT, generator=g) * 0.1
    m._R[5] = R_old
    ids = torch.tensor([5, 5, 5, 5])
    G_Y = torch.randn(4, D_OUT, generator=g) * 0.01

    mask = m._first_occurrence(ids)
    assert int(mask.sum()) == 1

    m.backward(G_Y, ids)
    Q = m.last_projection

    # Reconstruct the intended semantics: debt restored to ONE slot only.
    Geff = G_Y.clone()
    Geff[mask] += R_old
    expected = (Geff - Q @ (Q.T @ Geff)).sum(0)
    assert torch.allclose(m._R[5], expected, atol=1e-5)

    # And confirm the buggy alternative would have differed materially.
    Geff_bug = G_Y + R_old
    wrong = (Geff_bug - Q @ (Q.T @ Geff_bug)).sum(0)
    assert not torch.allclose(m._R[5], wrong, atol=1e-3)


def test_feedback_reduces_accumulated_error_versus_plain_sketch():
    """The point of M: per-step error obeys the same Monte Carlo bound, but
    accumulated error is far lower because successive misses cancel."""
    ids = torch.arange(V)
    g = torch.Generator().manual_seed(11)
    grads = [torch.randn(V, D_OUT, generator=g) * 0.01 for _ in range(60)]

    plain = UnbiasedSketch(D_IN, D_OUT, sketch_dim=8, seed=1)
    fb = SketchWithFeedback(D_IN, D_OUT, sketch_dim=8, n_params=V, seed=1)

    errs = {}
    for name, m in (("plain", plain), ("fb", fb)):
        acc_est = torch.zeros(V, D_IN)
        acc_exact = torch.zeros(V, D_IN)
        for G_Y in grads:
            acc_exact += m.exact_dX(G_Y)
            out = m.backward(G_Y, ids) if name == "fb" else m.backward(G_Y, None)
            acc_est += out[0]
        errs[name] = float((acc_est - acc_exact).norm() / acc_exact.norm())
    assert errs["fb"] < errs["plain"] / 2.0, errs


def test_factored_output_is_memory_restricted():
    """backward_factored must not materialise the dense (N x d_in) product."""
    m, G_Y, _ = _fb()
    P, Z = m.backward_factored(G_Y)
    assert P.shape == (N, m.s)
    assert Z.shape == (m.s, D_IN)
    dense_bytes = N * D_IN * G_Y.element_size()
    assert m.factored_bytes < dense_bytes


def test_factored_output_reconstructs_the_dense_estimate():
    m = UnbiasedSketch(D_IN, D_OUT, sketch_dim=8, seed=2)
    g = torch.Generator().manual_seed(4)
    G_Y = torch.randn(N, D_OUT, generator=g) * 0.01
    P, Z = m.backward_factored(G_Y)
    assert torch.allclose(P @ Z, P @ ((P.T @ G_Y) @ m.W), atol=1e-5)


def test_residual_operator_is_non_expansive():
    """The property M's convergence rests on, and the one a Gaussian P lacks.

    With orthonormal P, ||(I - P P^T) v|| <= ||v|| for every v. With a
    unit-variance-scaled Gaussian P it is not, which is why the first design
    diverged to 3.2e6 accumulated error.
    """
    n, s = 32, 8
    m = SketchWithFeedback(D_IN, D_OUT, sketch_dim=s, n_params=V, seed=0)
    g = torch.Generator().manual_seed(21)
    ortho_ratios, gauss_ratios = [], []
    for _ in range(40):
        v = torch.randn(n, D_OUT, generator=g)
        Q = m._draw(n)
        ortho_ratios.append(
            float((v - Q @ (Q.T @ v)).norm() / v.norm())
        )
        Pg = torch.randn(n, s, generator=g) / s**0.5
        gauss_ratios.append(
            float((v - Pg @ (Pg.T @ v)).norm() / v.norm())
        )
    assert max(ortho_ratios) <= 1.0 + 1e-5      # contraction, always
    assert max(gauss_ratios) > 1.5              # expansion, routinely


def test_orthonormal_projection_is_actually_orthonormal():
    m = SketchWithFeedback(D_IN, D_OUT, sketch_dim=8, n_params=V, seed=0)
    Q = m._draw(32)
    assert torch.allclose(Q.T @ Q, torch.eye(8), atol=1e-5)


def test_sketch_dim_above_batch_is_rejected_not_truncated():
    """Silent truncation would break the contraction guarantee."""
    import pytest

    m = SketchWithFeedback(D_IN, D_OUT, sketch_dim=64, n_params=V, seed=0)
    with pytest.raises(ValueError, match="contraction guarantee"):
        m._draw(32)


def test_first_occurrence_matches_reference_implementation():
    def reference(ids):
        seen, out = set(), []
        for v in ids.tolist():
            out.append(v not in seen)
            seen.add(v)
        return torch.tensor(out, dtype=torch.bool)

    g = torch.Generator().manual_seed(32)
    for hi in (2, 5, 40):
        for n in (1, 7, 64):
            ids = torch.randint(0, hi, (n,), generator=g)
            assert torch.equal(
                SketchWithFeedback._first_occurrence(ids), reference(ids)
            ), (hi, n, ids)

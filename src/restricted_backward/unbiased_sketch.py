"""Method L -- Process I via unbiased stochastic substitution. Rung 3d.

Every prior Process I attempt SELECTED a subset of input channels and zeroed the
rest. That is a deterministic, biased operation, and rung 3b established why it
cannot work here: dX column norms have max/median = 1.43 (flat spectrum), so
top-5% captures only 8% of the energy and *any* selector inherits an error floor
of sqrt(1-0.08) = 0.959. The m=1 exact-oracle run confirmed ranking quality was
never the bottleneck.

Method L abandons selection entirely:

    dX_hat = P (P^T G_Y) W,     P: N x s,  P_ij ~ N(0, 1/s),  RESAMPLED EACH STEP

E[P P^T] = I_N by construction, so E[dX_hat] = G_Y W exactly. Nothing is zeroed,
so there is no floor for the flat spectrum to apply to. The error is zero-mean
variance, which averages out across steps the way minibatch noise does -- this is
what replaces error feedback, and it is why no banking is required.

WHY FRESH RESAMPLING IS LOAD-BEARING
    A frozen or reused P does not merely weaken the estimator, it converts the
    error from zero-mean noise into a fixed non-random bias, silently restoring
    the exact failure mode this method exists to escape. Measured: with P fixed,
    bias after 200 draws is 1.9133 and does not decay. With fresh P, 0.1412 and
    decaying as 1/sqrt(T). `test_fresh_projection_is_load_bearing` asserts this.

MEASURED COST WALL -- STATED UP FRONT, NOT IMPLIED AWAY
    P compresses the BATCH dimension N, which is already small. Relative error
    follows sqrt(N/s) (measured 2.021 at s=8, N=32; predicted 2.000), so driving
    per-step error below 1.0 requires s >= N, where cost is 1.06x dense.

        s      per-step rel err     cost
        2          3.979           0.07x
        8          2.021           0.27x
        16         1.427           0.53x
        32         1.018           1.06x

    This does not match Process O's economics and must not be reported as if it
    does. Method K delivers 0.2368 cumulative error at 0.05x.

THE BOUND IS THE STANDARD MONTE CARLO LAW, NOT A PROPERTY OF THIS SYSTEM
    This is the ordinary sketch-estimator tradeoff (error ~ 1/sqrt(s),
    cost ~ s, so error^2 * cost is constant) appearing in this parameterisation:

        cost = s/N + 2s/d_in    and    err = sqrt(N/s)
        =>  err^2 * cost = 1 + 2N/d_in    >= 1     for every shape.

    Verified to three digits at three shapes (predicted / measured):
        N=32,  d=1024 -> 1.031 / 1.02-1.05
        N=128, d=512  -> 1.225 / 1.23
        N=512, d=256  -> 2.236 / 2.25

    Consequence: err < 1 REQUIRES cost > 1. Halving error costs 4x. Because
    this is the generic Monte Carlo law rather than a quirk of this estimator,
    no cleverness in constructing P escapes it -- a better sketch, a structured
    or orthogonalised P, a different distribution all obey the same bound. Only
    a fundamentally non-Monte-Carlo mechanism could.
    `test_error_cost_product_is_bounded_below` asserts it.

WHY L IS STILL USEFUL DESPITE THAT BOUND
    Per-step accuracy is not what SGD needs. In the discriminating task below,
    L(s=16) ran at per-step error 1.43 -- worse than useless by the per-step
    metric -- and still converged to 0.0026 against dense's 0.0003, because the
    error is zero-mean and unstructured. Per-step error is the wrong yardstick;
    it is the error's STRUCTURE, not its magnitude, that determines whether
    optimisation survives it.

SCOPE RELATIVE TO METHOD K -- L IS THE GENERAL MECHANISM, K IS A SPECIAL CASE
    Rung 3d proposed an access-density split. Rung 3e falsified it, along with
    two successor hypotheses, and the correction matters: K is valid only under
    FULL-BATCH access (every unit of the restricted axis present every step).
    Under anything else K suffers debt misrouting (repairable by re-keying the
    bank) and repayment mis-scheduling (not repairable, and divergent under
    skew). L has no bank and no schedule, so neither failure mode exists.

    Measured with W frozen so the loss reflects dX fidelity only, 2 seeds,
    best-of-lr:

        pattern    density  regular   dense    K       L(s=16)
        full       dense    yes       0.0000   0.0000  0.0005
        cyclic     sparse   yes       0.0000   0.1514  0.0005
        uniform    sparse   no        0.0000   0.1631  0.0007
        zipf       dense    no        0.0122   0.1763  0.0534

    L is essentially INVARIANT to access pattern; K is not. Note cyclic has an
    exact period of 16 and K still fails there, so the axis is not regularity
    either. Prefer L by default; K only where full-batch access is guaranteed.

CATEGORY -- L IS FLOP-RESTRICTED BUT NOT MEMORY-RESTRICTED
    L still produces a dense (N x d_in) dX estimate every step, so its peak
    backward tensor equals dense. Method K produces only (N x |group|). This is
    the same distinction that separated Methods C/D/E from F/G, and it is
    recorded here so the two are never collapsed into one "restricted" claim.
"""
from __future__ import annotations

import time as _time

import torch

from src.restricted_backward.methods import reference_forward


class UnbiasedSketch:
    """Process I with no selection: an unbiased noisy estimate of every column."""

    name = "L"

    def __init__(
        self,
        d_in: int,
        d_out: int,
        *,
        sketch_dim: int,
        dtype=torch.float32,
        seed: int = 0,
    ):
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(d_out, d_in, generator=g, dtype=dtype) / d_in**0.5
        self.d_in, self.d_out = d_in, d_out
        self.s = sketch_dim
        self.dtype = dtype

        # Dedicated stream so resampling cannot be silently disabled by a caller
        # reseeding the global RNG.
        self._rng = torch.Generator().manual_seed(seed + 991)
        self._step = 0
        self._last_P: torch.Tensor | None = None
        self.max_backward_tensor_bytes = 0
        self.aux_bytes = 0

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return reference_forward(X, self.W)

    def _draw(self, n: int) -> torch.Tensor:
        """Fresh projection. E[P P^T] = I_n requires variance 1/s per entry."""
        return torch.randn(
            n, self.s, generator=self._rng, dtype=self.dtype
        ) / self.s**0.5

    def backward(self, G_Y: torch.Tensor, X: torch.Tensor):
        n = G_Y.shape[0]
        elem = G_Y.element_size()

        _f0 = _time.perf_counter()
        P = self._draw(n)
        self._last_P = P
        self.aux_bytes = P.numel() * elem
        feedback_s = _time.perf_counter() - _f0

        _g0 = _time.perf_counter()
        # Association order matters: (P^T G_Y) W keeps every intermediate at
        # rank s. Computing (P P^T) first would materialise an N x N and then
        # cost the full dense matvec, eliminating the saving entirely.
        compressed = P.T @ G_Y                       # (s, d_out)
        projected = compressed @ self.W              # (s, d_in)
        dX_hat = P @ projected                       # (N, d_in)
        gemm_s = _time.perf_counter() - _g0

        self._step += 1
        self.max_backward_tensor_bytes = max(
            self.max_backward_tensor_bytes, dX_hat.numel() * elem
        )
        executed = (
            2.0 * self.s * n * self.d_out
            + 2.0 * self.s * self.d_out * self.d_in
            + 2.0 * n * self.s * self.d_in
        )
        return dX_hat, executed, 0.0, feedback_s, gemm_s

    @property
    def last_projection(self) -> torch.Tensor | None:
        """Exposed so tests can assert P actually changes between steps."""
        return self._last_P

    def exact_dX(self, G_Y: torch.Tensor) -> torch.Tensor:
        """Oracle for error measurement only. Never in a timed path."""
        return G_Y @ self.W

    def backward_factored(self, G_Y: torch.Tensor):
        """dX as rank-s factors (P, Z) with dX = P @ Z, instead of the product.

        Fixes L's memory category. `backward` materialises a dense
        (N x d_in) estimate, so its peak backward tensor equals dense -- the
        same post-hoc-compression trap that disqualified Methods C/D/E. The
        factors are (N x s) and (s x d_in), so peak memory is
        O(s(N + d_in)) rather than O(N*d_in), and a consumer that itself
        contracts dX (any downstream matmul) never needs the product at all.
        """
        n = G_Y.shape[0]
        P = self._draw(n)
        self._last_P = P
        Z = (P.T @ G_Y) @ self.W
        self.factored_bytes = (P.numel() + Z.numel()) * G_Y.element_size()
        return P, Z

    def apply_dense_dW(self, dW: torch.Tensor, lr: float) -> None:
        self.W -= lr * dW

    def param_bytes(self) -> int:
        return self.W.numel() * self.W.element_size()


class SketchWithFeedback(UnbiasedSketch):
    """Method M -- error feedback banked in G_Y space, with a CONTRACTIVE sketch.

    Rung 3b concluded Process I error feedback was impossible because
    dX[:,j] = G_Y W[:,j] is a dense contraction, so per-column debt costs the
    full matvec. That is true of debt in dX SPACE. It is not true in G_Y SPACE:

        Geff  = G_Y + R[ids]             restore what previous sketches missed
        Ghat  = P (P^T Geff)             what this sketch transmits
        R[ids] = Geff - Ghat             bank the remainder, exactly
        dX̂    = Ghat W                  same estimator, better input

    The missed component is (N x d_out) -- computable in 2*N*s*d_out, about
    N/d_in ~ 3% of the dominant cost. Same principle that made Method G work on
    the output side: bank the shared upstream tensor, not the per-unit result.

    WHY P IS ORTHONORMAL HERE AND GAUSSIAN IN THE PARENT CLASS
        Error feedback converges only if the residual operator is
        NON-EXPANSIVE. Method G is safe because top-k is an orthogonal
        projection (norm exactly 1). A Gaussian P scaled for unbiasedness is
        not: E[P P^T] = I forces the operator to overshoot, so ||I - P P^T|| > 1
        whenever s < N and the bank amplifies its own contents every step.
        Measured on the first attempt: accumulated error 3.2e6 against plain
        L's 1.42 -- divergence, not degradation.

        Unbiasedness and contractivity are in direct conflict, so M takes the
        same trade Method G takes. P has ORTHONORMAL columns, making P P^T an
        orthogonal projector and I - P P^T its complement, with
        ||R||^2 = ||Geff||^2 - ||Ghat||^2 <= ||Geff||^2 exactly. Each step is
        then BIASED (it transmits only a random s-dimensional subspace) and the
        feedback loop repays that bias in full, which is precisely how top-k
        plus error feedback behaves. Asserted by
        `test_residual_operator_is_non_expansive`.

        Consequence: M is NOT an unbiased per-step estimator and must never be
        described as one. Its guarantee is cumulative, not instantaneous.

    Fresh resampling stays load-bearing for a different reason than in L: with
    a fixed P the retained subspace never rotates, so the orthogonal complement
    is banked forever and never transmitted.

    SCOPE. The bank is keyed by parameter id, so this applies where the
    restricted axis has a persistent identity. For activation gradients there
    is no such identity and plain `UnbiasedSketch` is the correct choice --
    feedback across steps is not merely expensive there but undefined.
    """

    name = "M"

    def __init__(self, d_in: int, d_out: int, *, sketch_dim: int,
                 n_params: int, dtype=torch.float32, seed: int = 0):
        super().__init__(d_in, d_out, sketch_dim=sketch_dim,
                         dtype=dtype, seed=seed)
        self.n_params = n_params
        self._R = torch.zeros(n_params, d_out, dtype=dtype)
        self.aux_bytes = self._R.numel() * self._R.element_size()

    def _draw(self, n: int) -> torch.Tensor:
        """Orthonormal columns, so I - P P^T is an orthogonal projector.

        Requires s <= n; a subspace of dimension s cannot be embedded in R^n
        otherwise, and silently truncating would break the contraction bound
        that this method's convergence rests on.
        """
        if self.s > n:
            raise ValueError(
                f"sketch_dim={self.s} exceeds batch dimension {n}; the "
                "contraction guarantee requires an s-dimensional subspace "
                "of R^n to exist."
            )
        A = torch.randn(n, self.s, generator=self._rng, dtype=self.dtype)
        Q, _ = torch.linalg.qr(A)
        return Q

    @staticmethod
    def _first_occurrence(ids: torch.Tensor) -> torch.Tensor:
        """Mask selecting one slot per distinct id.

        Needed because a repeated id must have its banked residual restored
        ONCE, not once per occurrence -- otherwise a frequently-sampled
        parameter has its debt multiplied by its own frequency, which is the
        mechanism that made row-keyed Method K diverge under Zipfian access.

        Vectorised: a Python loop here cost ~0.5 ms/step, which erased M's
        entire FLOP advantage (0.594x FLOPs but 1.03x wall-clock).
        """
        order = torch.argsort(ids, stable=True)
        srt = ids[order]
        first_sorted = torch.ones(srt.shape[0], dtype=torch.bool)
        first_sorted[1:] = srt[1:] != srt[:-1]
        mask = torch.zeros(ids.shape[0], dtype=torch.bool)
        mask[order] = first_sorted
        return mask

    def backward(self, G_Y: torch.Tensor, ids: torch.Tensor):
        n = G_Y.shape[0]
        elem = G_Y.element_size()

        _f0 = _time.perf_counter()
        Geff = G_Y.clone()
        first = self._first_occurrence(ids)
        Geff[first] += self._R[ids[first]]
        P = self._draw(n)
        self._last_P = P
        feedback_s = _time.perf_counter() - _f0

        _g0 = _time.perf_counter()
        compressed = P.T @ Geff                       # (s, d_out)
        dX_hat = P @ (compressed @ self.W)            # (N, d_in)
        gemm_s = _time.perf_counter() - _g0

        _f1 = _time.perf_counter()
        Ghat = P @ compressed                         # (N, d_out), cheap
        self._R[ids] = 0.0
        self._R.index_add_(0, ids, Geff - Ghat)       # exact remainder
        feedback_s += _time.perf_counter() - _f1

        self._step += 1
        self.max_backward_tensor_bytes = max(
            self.max_backward_tensor_bytes, dX_hat.numel() * elem
        )
        executed = (
            2.0 * self.s * n * self.d_out                # P^T Geff
            + 2.0 * self.s * self.d_out * self.d_in      # (.) W
            + 2.0 * n * self.s * self.d_in               # P (.)
            + 2.0 * n * self.s * self.d_out              # P compressed
        )
        return dX_hat, executed, 0.0, feedback_s, gemm_s

    @property
    def residual_norm(self) -> float:
        return float(self._R.norm())

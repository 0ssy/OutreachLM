"""Method K2 -- Process I with both of Method K's defects actually fixed.

Rung 3e established that Method K (`scheduled_accumulation.py`) fails under any
access pattern other than full-batch, for two independent reasons:

    1. MISROUTING     bank is (N x d_out), keyed by batch slot, so debt banked
                      for the parameter in slot i is repaid to whatever
                      parameter occupies slot i later.
    2. MIS-SCHEDULING repayment advances one group per STEP, while a
                      parameter's gradient arrives on its own TOUCH cadence.
                      Diverged under Zipfian access (1.18e29).

Both are fixed here, and neither fix is a threshold or a clamp.

FIX 1 -- KEY THE BANK BY PARAMETER
    The bank is (n_params x d_out). Debt is indexed by the parameter it was
    incurred for, so it cannot be misrouted regardless of batch composition,
    ordering, duplication or skew.

FIX 2 -- ADVANCE THE SCHEDULE IN TOUCH TIME, NOT STEP TIME
    Each parameter carries its own touch counter c_r. On its c-th touch it
    receives group (c mod G). Repayment cadence is therefore identical to
    gradient-arrival cadence by construction, for every parameter
    independently. A parameter touched once per 1000 steps and one touched
    every step are both fully repaid every G touches.

    Naively this needs the sum over each parameter's last G touches -- a
    sliding window, i.e. G snapshots per parameter, O(V*G*d_out). That is 84 MB
    at V=512, G=20, d_out=1024 and scales with vocabulary.

    Instead use a TWO-PHASE CYCLE. Parameter r accumulates into T_cur[r] during
    its current cycle of G touches; T_prev[r] holds the completed previous
    cycle's total. Group (c mod G) is repaid from T_prev[r]. Over one full
    cycle every group is visited exactly once, so every banked gradient is
    delivered to every input channel exactly once. Nothing is dropped and
    nothing is double-counted.

    Memory falls to 2 * n_params * d_out, and delay is bounded by 2G TOUCHES
    (not steps) for every parameter regardless of its frequency.

EXACTNESS
    Every gradient is delivered exactly once to every input channel, with a
    touch-time delay in [G, 2G). Asserted by
    `test_every_gradient_delivered_exactly_once`.

WARMUP
    A parameter's first cycle has T_prev = 0, so its first G touches deliver
    nothing. This is a bounded, one-off startup cost per parameter, reported by
    `warmup_touches`, not silently absorbed.

COST NOTE -- MEASURED, AND IT IS BAD
Because parameters cycle independently, rows in a batch generally want
different column groups, so the single restricted GEMM becomes one GEMM per
distinct group present. FLOP count is unchanged (sum of 2*n_g*d_out*k over
buckets is exactly 2*N*d_out*k) but the work splits into many small GEMMs.
Measured at D=512, N=32, keep=0.05: 0.05x the FLOPs of dense and 3.9x the
WALL-CLOCK.

THIS IS NOT THE SAME FAILURE AS METHODS C/D/E, despite the similar
symptom. Distinguishing them matters because the locus argument depends on
the difference:

    C/D/E   SUNK DENSE COST. The full dense dW was materialised and then
            compressed, so the dense GEMM was already paid for and the
            selection scan was pure addition. Restriction was in the wrong
            PLACE -- downstream of the computation it claimed to eliminate.
    K2      PER-PARAMETER BOOKKEEPING AND GEMM FRAGMENTATION. Restriction
            is in the right place (no dense dX is ever built, peak tensor
            is 0.003 MB vs dense's 0.066 MB), and the FLOPs really are
            0.05x. The loss comes from independent per-parameter cycles
            shattering one large GEMM into up to G small ones, each too
            small to reach useful CPU throughput, plus per-step index
            bookkeeping.

Same symptom (FLOP savings do not survive contact with a CPU), different
cause (there: wrong locus; here: right locus, wrong granularity of work).

WHAT IS FIXED HERE, AND WHAT IS NOT
Fixed: misrouting, mis-scheduling, and the cycle-boundary skip that made
repeated ids in one batch (Zipfian access) diverge to 1e29.

NOT fixed, because it is the mechanism rather than a defect: deferral
itself. Each channel waits G touches, so the delivered gradient was
computed 1-2 cycles ago at a stale point. Measured on uniform access,
W frozen, parameter recovery ||E - E*||/||E*||:

    keep   G    deferral      recovery
    0.05   20   20-40 touches  0.3767
    0.10   11   11-22          0.1644
    0.25    4    4-8           0.0720
    0.50    2    2-4           0.0716
    1.00    1    1-2           0.0717
    (dense 0.0293; Method L at s=16 reaches 0.0724)

Error falls monotonically as G -> 1 and converges exactly onto L's value,
confirming the residue is deferral and nothing else. At the sparsity that
makes the method worth using (keep=0.05) that costs 5x L's error; at the
keep where accuracy matches L, the FLOP saving is only 4x and the
wall-clock penalty remains. See PROTOCOL_TWO_SIDED.md, rung 3f.
"""
from __future__ import annotations

import time as _time

import torch

from src.restricted_backward.methods import reference_forward


class ParameterKeyedAccumulation:
    """Process I with parameter-keyed banking and touch-time repayment."""

    name = "K2"

    def __init__(
        self,
        d_in: int,
        d_out: int,
        *,
        keep_fraction: float,
        n_params: int,
        dtype=torch.float32,
        seed: int = 0,
    ):
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(d_out, d_in, generator=g, dtype=dtype) / d_in**0.5
        self.d_in, self.d_out, self.n_params = d_in, d_out, n_params
        self.k = max(1, int(round(keep_fraction * d_in)))
        self.n_groups = max(1, -(-d_in // self.k))

        perm = torch.randperm(d_in, generator=g)
        self.groups = [perm[i::self.n_groups] for i in range(self.n_groups)]

        # Keyed by PARAMETER, not batch slot. Two phases, not G snapshots.
        self._cur = torch.zeros(n_params, d_out, dtype=torch.float64)
        self._prev = torch.zeros(n_params, d_out, dtype=torch.float64)
        self._touches = torch.zeros(n_params, dtype=torch.long)

        self.max_backward_tensor_bytes = 0
        self.bucket_count = 0
        self.aux_bytes = 2 * n_params * d_out * 8

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return reference_forward(X, self.W)

    @property
    def warmup_touches(self) -> int:
        """Touches before a given parameter starts receiving repayment."""
        return self.n_groups

    @property
    def visit_period(self) -> int:
        """Every input channel of a parameter is repaid every G TOUCHES."""
        return self.n_groups

    def backward(self, G_Y: torch.Tensor, ids: torch.Tensor):
        """Restricted dX for the rows named by `ids`.

        Returns (buckets, executed_flops, _, feedback_s, gemm_s) where buckets
        is a list of (row_positions, col_indices, values).

        Repeated ids within one batch are SEPARATE SEQUENTIAL TOUCHES,
        processed in occurrence-rank order. Treating them as a single touch
        lets a parameter's counter jump straight over a cycle boundary
        (19 -> 24 with G=21) without rolling, which freezes _prev and lets _cur
        grow without bound -- exactly how the first version of this class
        diverged under Zipfian access. Within one rank sub-step every parameter
        appears at most once, so the single-touch logic is exactly correct.
        """
        elem = G_Y.element_size()
        feedback_s = 0.0
        gemm_s = 0.0
        executed = 0.0
        buckets = []

        ranks = self._occurrence_ranks(ids)
        for j in range(int(ranks.max()) + 1):
            sel = (ranks == j).nonzero(as_tuple=True)[0]
            ids_j = ids[sel]

            _f0 = _time.perf_counter()
            starting = ids_j[(self._touches[ids_j] % self.n_groups == 0)
                             & (self._touches[ids_j] > 0)]
            if starting.numel():
                self._prev[starting] = self._cur[starting]
                self._cur[starting] = 0.0

            self._cur.index_add_(0, ids_j, G_Y[sel].double())
            slot_groups = self._touches[ids_j] % self.n_groups
            debt = self._prev[ids_j].to(G_Y.dtype)
            feedback_s += _time.perf_counter() - _f0

            _g0 = _time.perf_counter()
            for gi in torch.unique(slot_groups).tolist():
                local = (slot_groups == gi).nonzero(as_tuple=True)[0]
                cols = self.groups[gi]
                vals = debt[local] @ self.W[:, cols]
                buckets.append((sel[local], cols, vals))
                executed += 2.0 * local.numel() * self.d_out * cols.numel()
                self.max_backward_tensor_bytes = max(
                    self.max_backward_tensor_bytes, vals.numel() * elem
                )
            gemm_s += _time.perf_counter() - _g0

            self._touches.index_add_(0, ids_j, torch.ones_like(ids_j))

        self.bucket_count = len(buckets)
        return buckets, executed, 0.0, feedback_s, gemm_s

    @staticmethod
    def _occurrence_ranks(ids: torch.Tensor) -> torch.Tensor:
        """Rank of each slot among slots sharing its id (0 for the first).

        Vectorised for the same reason as SketchWithFeedback._first_occurrence:
        per-step Python loops dominate wall-clock at these tensor sizes.
        """
        order = torch.argsort(ids, stable=True)
        srt = ids[order]
        pos = torch.arange(srt.shape[0])
        # Index at which each run of equal ids begins.
        starts = torch.zeros(srt.shape[0], dtype=torch.long)
        new_run = torch.ones(srt.shape[0], dtype=torch.bool)
        new_run[1:] = srt[1:] != srt[:-1]
        starts[new_run] = pos[new_run]
        starts = torch.cummax(starts, dim=0).values
        out = torch.zeros(ids.shape[0], dtype=torch.long)
        out[order] = pos - starts
        return out

    @staticmethod
    def densify(buckets, shape, dtype=torch.float32) -> torch.Tensor:
        """Dense view for measurement and for callers that need one."""
        out = torch.zeros(shape, dtype=dtype)
        for rows, cols, vals in buckets:
            out[rows.unsqueeze(1), cols.unsqueeze(0)] = vals
        return out

    def exact_dX(self, G_Y: torch.Tensor) -> torch.Tensor:
        """Oracle for error measurement only. Never in a timed path."""
        return G_Y @ self.W

    def apply_dense_dW(self, dW: torch.Tensor, lr: float) -> None:
        self.W -= lr * dW

    def param_bytes(self) -> int:
        return self.W.numel() * self.W.element_size()

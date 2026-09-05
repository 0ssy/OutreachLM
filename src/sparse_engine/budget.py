"""Corrected compute and memory accounting for a 70B-capacity sparse engine.

WHY THIS FILE EXISTS
    An earlier review of the 70B-on-a-laptop plan concluded "102,185 years"
    and declared it impossible. That review contained an accounting error and
    the conclusion does not follow from it.

    It used  compute = 6 * P_total * T.  For a sparsely-activated model the
    correct form is  compute = 6 * P_active * T,  where P_active is the
    parameters that actually participate per token. This is standard practice
    for mixture-of-experts models and is how Switch Transformer and Mixtral
    report their training cost. It also assumed T = 1e12 tokens, which the
    plan never specified.

    With active-parameter accounting and a token budget matched to the active
    size, the compute requirement lands in days, not millennia.

WHAT SURVIVED THE MEASUREMENTS AND WHAT DID NOT
    Survived, and is load-bearing:
      * Batch-token fusion (claim 2). Streaming weights once per step and
        applying them to a large token block is what makes the I/O affordable.
        This is the single most important idea in the plan.
      * Block sparsity (claim 1), when the active set is a CONTIGUOUS EXPERT
        TILE rather than scattered scalars. Measured: expert tiles run at
        137-158 GFLOP/s, ABOVE the 130.5 GFLOP/s large-GEMM figure, because a
        small tile stays resident in cache. Routing gather costs 1-4%.

    Did not survive, and the design must not depend on it:
      * "Zero-cycle conditional branching" over inactive weights. Measured:
        masking a dense matrix and multiplying anyway runs at 0.5x dense --
        SLOWER. The speedup comes from gathering active tiles into contiguous
        memory and running a smaller dense GEMM, not from skipping.
      * Unstructured 0.1% sparsity. Measured 201x fewer FLOPs but only ~12
        GFLOP/s -- the active submatrix is too small to use the machine. Hence
        the EXPERT_MIN_PARAMS floor below.
      * Integer arithmetic being cheaper. Measured: int32 matmul is 0.02x
        fp32, i.e. 41x SLOWER, because this CPU (Zen 3) has no VNNI and there
        is no optimised integer GEMM path. Ternary's real benefit is MEMORY:
        2 bits against 32 is 16x less traffic, and this workload is I/O bound.
        Compute in fp32; store in ternary.
      * A 1.25 GB tile resident in L3. Measured L3 is 16 MB.
      * "4 tokens per 512-bit register." This CPU is AVX2: 256-bit registers,
        no AVX-512, no VNNI (torch reports capability AVX2, Family 25 Model
        80). Register-blocking across tokens is also already what the GEMM
        kernel does. The available headroom came from routing BLOCK SIZE --
        145.6 GFLOP/s at 128 tokens/expert against 173.0 at 512 -- and is
        already taken in SPARSE_GFLOPS below.
      * Gradient-magnitude expert pruning as a free speedup. Two problems.
        It starves the pruned pool: a pruned expert emits no gradient, so its
        magnitude estimate never refreshes, so it is never reselected --
        simulated at 95.8% of experts never updated, and this project already
        measured the same mechanism at row granularity in rung 3e. Adding
        random re-probing fixes coverage (34.9% never updated) but spends the
        compute the pruning was meant to save. More fundamentally, active
        parameters ARE the model: compute is 6 * P_active * T, so cutting
        70 -> 20 active experts cuts P_active from 70M to 20M. That is a
        smaller model, not a faster one.
      * Bitwise delta compression "eliminating network delays". Cross-node
        communication is already 1.9% of the two-node budget under disjoint
        expert ownership, because experts have a single owner and their
        gradients never cross the link. Removing it entirely is a 1.02x
        speedup, and the ternary merge rules resolve a conflict this design
        does not have.

WHAT THIS DOES NOT CLAIM
    A 70B-total / 70M-active model is not a 70B dense model. Published MoE
    runs use far lower sparsity ratios -- Mixtral is 47B/13B (3.6x), Switch
    Transformer explored up to roughly 100x with measurable degradation. At
    1000x this design is well outside demonstrated territory, and the honest
    expectation is quality somewhere between a 70M and a few-hundred-M dense
    model, with the 70B table acting as addressable memory rather than as
    effective capacity. That is an empirical question this accounting cannot
    settle; it is a risk to the QUALITY of the result, not to its FEASIBILITY,
    and it is the reason the routing quality experiment must come early.

MEASURED CONSTANTS -- all from this machine, none assumed.
"""
from __future__ import annotations

from dataclasses import dataclass

# src/sparse_engine/bench_expert_shape.py and bench_expert_tuned.py
# 173 GFLOP/s sustained at the tuned point: a 2048x512 expert tile (1.05M
# params) fed 512 tokens, min of 9 trials. Throughput is shape-sensitive --
# 145.6 at 128 tokens, 173.0 at 512, 148.0 at 2048 -- so the routing block
# size is a real tuning parameter, not an implementation detail.
# A single-trial sweep suggested 190; that did not survive repetition.
SPARSE_GFLOPS = 173.0
TOKENS_PER_EXPERT_OPTIMAL = 512
DENSE_GFLOPS = 130.5           # large dense GEMM
ROUTING_OVERHEAD = 0.04        # gather cost as a fraction of expert GEMM

# src/sparse_engine/bench_stream_bandwidth.py
DISK_READ_GBPS = 4.10
DISK_WRITE_GBPS = 0.26         # 16x slower than read -- shapes the design
RAM_BW_GBPS = 10.6

# src/sparse_engine/bench_link.py -- cross-laptop WiFi, 144.4 Mbps negotiated.
# Bounded rather than directly measured (no listener on the far machine);
# 0.0135 GB/s is the OPTIMISTIC end of 35-75% link efficiency, so using it
# makes every two-node estimate below generous rather than flattering.
LINK_GBPS = 0.0135
LINK_RTT_MS = 0.80

# src/sparse_engine/run_delta_probe.py -- ternary delta coding against the
# PACKED 2-BIT form (not fp32, which would inflate the figure 16x for free).
# 12.1x at a 256-step sync interval; flips accumulate sublinearly, so longer
# intervals cost less per step even though each delta is larger.
DELTA_COMPRESSION = 12.1
DELTA_SYNC_DEFAULT = 256

# Platform (per node; the second laptop is identical)
RAM_GB = 15.4
DISK_FREE_GB = 119.5
L3_MB = 16.0
CORES = 6

# Below this an expert tile stops filling the vector units: the 0.1%
# unstructured case measured 12 GFLOP/s against 150 for a proper tile.
EXPERT_MIN_PARAMS = 512 * 1024

SECONDS_PER_DAY = 86400.0


@dataclass
class Config:
    """One candidate engine configuration.

    `nodes` and `strategy` describe how work is split across the two
    laptops. The link between them measures ~0.0135 GB/s against 4.10 GB/s
    local disk, so the strategy is chosen by that ratio, not by preference.
    """

    total_params: float
    active_params_per_token: float
    tokens: float
    expert_params: float
    tokens_per_step: int
    latent_bytes: int = 1          # int8 latent master
    weight_bits: int = 2           # ternary, packed
    latent_update_every: int = 16  # expert-level gradient accumulation
    nodes: int = 1
    strategy: str = "expert_sharded"
    trunk_params: float = 2e8      # embeddings + attention + router
    trunk_sync_every: int = 16
    delta_sync_every: int = DELTA_SYNC_DEFAULT
    d_model: int = 2048
    layers: int = 8

    # ---------------------------------------------------------------- sizes
    @property
    def n_experts(self) -> float:
        return self.total_params / self.expert_params

    @property
    def experts_per_token(self) -> float:
        return self.active_params_per_token / self.expert_params

    @property
    def weight_gb(self) -> float:
        return self.total_params * self.weight_bits / 8 / 1e9

    @property
    def latent_gb(self) -> float:
        return self.total_params * self.latent_bytes / 1e9

    @property
    def shard_fraction(self) -> float:
        """Fraction of the expert table one node stores."""
        if self.nodes > 1 and self.strategy in ("expert_sharded",
                                                "expert_parallel"):
            return 1.0 / self.nodes
        return 1.0

    @property
    def storage_per_node_gb(self) -> float:
        return (self.weight_gb + self.latent_gb) * self.shard_fraction

    @property
    def steps(self) -> float:
        """Steps are shared: with N nodes each step covers N token blocks."""
        return self.tokens / (self.tokens_per_step * self.nodes)

    # ------------------------------------------------------------- compute
    @property
    def compute_flops(self) -> float:
        """6 * P_active * T -- the corrected form."""
        return 6.0 * self.active_params_per_token * self.tokens

    @property
    def compute_seconds(self) -> float:
        eff = SPARSE_GFLOPS * 1e9 / (1.0 + ROUTING_OVERHEAD)
        return self.compute_flops / eff / self.nodes

    # ------------------------------------------------------------------ io
    @property
    def experts_touched_per_step(self) -> float:
        """A large token block routes to essentially every local expert, so
        the shard is streamed once per step. This is why batch-token fusion
        matters: the cost is per STEP, not per token."""
        local = self.n_experts * self.shard_fraction
        return min(local, self.tokens_per_step * self.experts_per_token)

    @property
    def io_seconds(self) -> float:
        local = max(self.n_experts * self.shard_fraction, 1.0)
        frac = self.experts_touched_per_step / local
        w = self.weight_gb * self.shard_fraction
        lat = self.latent_gb * self.shard_fraction
        read = (w + lat) * frac / DISK_READ_GBPS
        write = lat * frac / DISK_WRITE_GBPS / self.latent_update_every
        return (read + write) * self.steps

    # ---------------------------------------------------------------- comm
    @property
    def comm_gb_per_step(self) -> float:
        if self.nodes < 2:
            return 0.0
        if self.strategy == "expert_sharded":
            # Only the replicated trunk is synchronised, and only every
            # trunk_sync_every steps. Experts are owned outright, so their
            # gradients never cross the link.
            return (self.trunk_params * 4 / 1e9) / self.trunk_sync_every
        if self.strategy == "data_parallel":
            # Every node holds every expert, so every expert gradient must be
            # reduced across the link.
            return (self.latent_gb) / self.latent_update_every
        if self.strategy == "data_parallel_delta":
            # Same topology, but the synchronised object is a ternary DELTA
            # rather than a latent gradient. Measured flip rates give 12.1x
            # compression of the packed 2-bit form at a 256-step interval,
            # and flips accumulate sublinearly so longer intervals are
            # cheaper per step: 0.00567 GB/step against 0.02790 at 16 steps.
            return (self.weight_gb / DELTA_COMPRESSION
                    / self.delta_sync_every)
        if self.strategy == "expert_parallel":
            # All-to-all: each token's hidden vector travels to a remote
            # expert and the result returns, at every layer.
            remote = 1.0 - 1.0 / self.nodes
            return (self.tokens_per_step * self.experts_per_token * remote
                    * self.d_model * 4 * 2 * self.layers) / 1e9
        raise ValueError(self.strategy)

    @property
    def comm_seconds(self) -> float:
        return self.comm_gb_per_step / LINK_GBPS * self.steps

    @property
    def total_seconds(self) -> float:
        return self.compute_seconds + self.io_seconds + self.comm_seconds

    @property
    def days(self) -> float:
        return self.total_seconds / SECONDS_PER_DAY

    # ------------------------------------------------------------ feasible
    def working_set_gb(self) -> float:
        """RAM needed at any instant: activations plus the live expert tiles.

        Only the experts being applied right now are resident; the rest are on
        disk. Streaming is expert-parallel, so a bounded slice is live.
        """
        live_experts = min(self.n_experts, 64.0)
        tiles = live_experts * self.expert_params * (
            self.weight_bits / 8 + 4.0        # ternary + fp32 unpacked
        ) / 1e9
        acts = self.tokens_per_step * 2048 * 4 * 3 / 1e9
        return tiles + acts

    def problems(self, max_days: float = 90.0) -> list[str]:
        out = []
        if self.expert_params < EXPERT_MIN_PARAMS:
            out.append(
                f"expert too small ({self.expert_params / 1024:.0f}K params); "
                f"below {EXPERT_MIN_PARAMS / 1024:.0f}K the tile drops to "
                f"~12 GFLOP/s"
            )
        if self.storage_per_node_gb > DISK_FREE_GB:
            out.append(
                f"storage {self.storage_per_node_gb:.1f} GB/node exceeds "
                f"{DISK_FREE_GB} GB free"
            )
        ws = self.working_set_gb()
        if ws > RAM_GB * 0.7:
            out.append(f"working set {ws:.1f} GB too close to {RAM_GB} GB RAM")
        if self.tokens < 15.0 * self.active_params_per_token:
            out.append(
                f"token budget {self.tokens:.2e} is under-trained for "
                f"{self.active_params_per_token:.2e} active params "
                f"(Chinchilla wants ~20x)"
            )
        if self.comm_seconds > 0.25 * self.compute_seconds:
            out.append(
                f"link-bound: {self.comm_seconds / SECONDS_PER_DAY:.1f} days "
                f"of communication against "
                f"{self.compute_seconds / SECONDS_PER_DAY:.1f} days of compute"
            )
        if self.days > max_days:
            out.append(f"runtime {self.days:.0f} days exceeds {max_days:.0f}")
        return out

    def feasible(self, max_days: float = 90.0) -> bool:
        return not self.problems(max_days)


def report(name: str, c: Config) -> None:
    print(f"\n=== {name} ===")
    print(f"  nodes / strategy    {c.nodes} x  {c.strategy}")
    print(f"  total params        {c.total_params:.2e}")
    print(f"  active per token    {c.active_params_per_token:.2e} "
          f"({c.active_params_per_token / c.total_params:.2%})")
    print(f"  experts             {c.n_experts:,.0f} x "
          f"{c.expert_params / 1e6:.1f}M, {c.experts_per_token:.0f}/token")
    print(f"  tokens              {c.tokens:.2e} "
          f"({c.tokens / c.active_params_per_token:.0f}x active params)")
    print(f"  storage/node        {c.storage_per_node_gb:.1f} GB "
          f"of {DISK_FREE_GB} free")
    print(f"  RAM working set     {c.working_set_gb():.2f} GB")
    print(f"  compute             {c.compute_seconds / SECONDS_PER_DAY:.1f} d")
    print(f"  streaming I/O       {c.io_seconds / SECONDS_PER_DAY:.1f} d")
    print(f"  cross-node comm     {c.comm_seconds / SECONDS_PER_DAY:.1f} d "
          f"({c.comm_gb_per_step * 1000:.1f} MB/step)")
    print(f"  TOTAL               {c.days:.1f} days")
    probs = c.problems()
    print(f"  verdict             {'FEASIBLE' if not probs else 'BLOCKED'}")
    for p in probs:
        print(f"      - {p}")


BASE = dict(
    total_params=70e9, active_params_per_token=70e6, tokens=1.4e9,
    tokens_per_step=65536,
)


def main() -> None:
    print("70B-capacity sparse engine: TWO-NODE accounting")
    print(f"link measured at {LINK_GBPS} GB/s against {DISK_READ_GBPS} GB/s "
          f"local disk -- a {DISK_READ_GBPS / LINK_GBPS:.0f}x gap that")
    print("selects the parallelism strategy rather than merely tuning it.")

    single = Config(expert_params=1e6, nodes=1, **BASE)
    report("ONE NODE (previous result, for reference)", single)

    report("TWO NODES, expert-parallel all-to-all", Config(
        expert_params=35e6, nodes=2, strategy="expert_parallel", **BASE,
    ))

    report("TWO NODES, data-parallel with gradient sync", Config(
        expert_params=1e6, nodes=2, strategy="data_parallel", **BASE,
    ))

    best = Config(expert_params=1e6, nodes=2, strategy="expert_sharded",
                  **BASE)
    report("TWO NODES, disjoint experts + trunk sync", best)

    dpd = Config(expert_params=1e6, nodes=2, strategy="data_parallel_delta",
                 **BASE)
    report("TWO NODES, data-parallel with ternary delta sync", dpd)

    print(f"\n  speedup vs one node   {single.days / best.days:.2f}x "
          f"({single.days:.1f} -> {best.days:.1f} days)")
    print(f"  storage per node      {single.storage_per_node_gb:.1f} GB -> "
          f"{best.storage_per_node_gb:.1f} GB")
    print(f"\n  sharded {best.days:.1f} d but each token routes among "
          f"{best.n_experts / 2:,.0f} experts;")
    print(f"  delta-sync {dpd.days:.1f} d with all {dpd.n_experts:,.0f} "
          f"reachable.")
    print("  Measured routing cost of sharding at a converged horizon:")
    print("  1.52x worse loss at 512 experts, 1.12x at 2048.")


if __name__ == "__main__":
    main()

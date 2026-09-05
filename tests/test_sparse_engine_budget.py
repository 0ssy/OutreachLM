"""Tests for the corrected 70B sparse-engine accounting.

These assert against MEASURED hardware limits, so a configuration that would
not physically run cannot silently be declared feasible.
"""
import pytest

from src.sparse_engine.budget import (
    DISK_FREE_GB,
    DISK_READ_GBPS,
    DISK_WRITE_GBPS,
    EXPERT_MIN_PARAMS,
    LINK_GBPS,
    RAM_GB,
    Config,
)

TARGET = Config(
    total_params=70e9,
    active_params_per_token=70e6,
    tokens=1.4e9,
    expert_params=1e6,
    tokens_per_step=65536,
)


def test_the_70b_target_configuration_is_feasible():
    """The claim the whole design exists to support."""
    assert TARGET.feasible(max_days=90), TARGET.problems()
    assert TARGET.days < 90


def test_compute_uses_active_not_total_parameters():
    """The error that produced the '102,185 years' verdict.

    6 * P_total * T and 6 * P_active * T differ by the sparsity ratio, here
    1000x. Using the wrong one is the difference between millennia and weeks.
    """
    dense = Config(
        total_params=70e9, active_params_per_token=70e9, tokens=1.4e9,
        expert_params=70e9, tokens_per_step=65536,
    )
    assert dense.compute_flops / TARGET.compute_flops == pytest.approx(1000.0)
    assert TARGET.compute_flops == pytest.approx(6 * 70e6 * 1.4e9)


def test_token_budget_must_match_active_size_not_total():
    """The second half of the error: 1e12 tokens is sized for a 50B-active
    model, not a 70M-active one, and inflates cost by another ~700x."""
    over = Config(
        total_params=70e9, active_params_per_token=70e6, tokens=1e12,
        expert_params=1e6, tokens_per_step=65536,
    )
    assert not over.feasible(max_days=90)
    assert any("runtime" in p for p in over.problems(90))
    assert over.days / TARGET.days > 100


def test_storage_fits_measured_free_space():
    assert TARGET.weight_gb + TARGET.latent_gb < DISK_FREE_GB
    fp32_latent = Config(
        total_params=70e9, active_params_per_token=70e6, tokens=1.4e9,
        expert_params=1e6, tokens_per_step=65536, latent_bytes=4,
    )
    assert not fp32_latent.feasible()
    assert any("storage" in p for p in fp32_latent.problems())


def test_working_set_fits_ram():
    """Weights live on disk; only live expert tiles and activations are
    resident. If this ever exceeds RAM the design has silently become
    a load-everything design."""
    assert TARGET.working_set_gb() < RAM_GB * 0.7


def test_expert_tiles_must_be_large_enough_to_use_the_machine():
    """Unstructured 0.1% sparsity measured 12 GFLOP/s against 150 for a
    proper tile. The floor encodes that measurement."""
    tiny = Config(
        total_params=70e9, active_params_per_token=70e6, tokens=1.4e9,
        expert_params=EXPERT_MIN_PARAMS / 4, tokens_per_step=65536,
    )
    assert not tiny.feasible()
    assert any("expert too small" in p for p in tiny.problems())


def test_batch_token_fusion_is_load_bearing():
    """Claim 2 from the original plan, and the one that survived measurement.

    Weight streaming is charged per STEP. Shrinking the token block raises the
    step count without reducing the bytes moved, so I/O grows in proportion.
    """
    small = Config(
        total_params=70e9, active_params_per_token=70e6, tokens=1.4e9,
        expert_params=1e6, tokens_per_step=1024,
    )
    assert small.io_seconds > 10 * TARGET.io_seconds
    assert not small.feasible(max_days=90)


def test_write_bandwidth_asymmetry_is_respected():
    """Disk writes measured 0.26 GB/s against 4.10 GB/s reads. Latent
    write-back must be amortised or it dominates the step."""
    assert DISK_WRITE_GBPS < DISK_READ_GBPS / 10
    every_step = Config(
        total_params=70e9, active_params_per_token=70e6, tokens=1.4e9,
        expert_params=1e6, tokens_per_step=65536, latent_update_every=1,
    )
    # Not the full 16x, because reads are unaffected -- but writing every
    # step still multiplies total I/O by 7.6x on measured bandwidths.
    assert every_step.io_seconds / TARGET.io_seconds > 7.0
    assert not every_step.feasible(max_days=90)


def test_chinchilla_gate_rejects_undertrained_configs():
    starved = Config(
        total_params=70e9, active_params_per_token=70e6, tokens=1e8,
        expert_params=1e6, tokens_per_step=65536,
    )
    assert any("under-trained" in p for p in starved.problems())


# --------------------------- two-node strategies ---------------------------

TWO_NODE_BASE = dict(
    total_params=70e9, active_params_per_token=70e6, tokens=1.4e9,
    tokens_per_step=65536, nodes=2,
)

SHARDED = Config(expert_params=1e6, strategy="expert_sharded", **TWO_NODE_BASE)


def test_the_link_is_the_binding_constraint_not_compute():
    """0.0135 GB/s against 4.10 GB/s local disk. Every two-node decision
    follows from this ratio."""
    assert LINK_GBPS < DISK_READ_GBPS / 100


def test_expert_sharding_is_the_only_viable_two_node_strategy():
    assert SHARDED.feasible(max_days=90), SHARDED.problems()
    for bad in ("expert_parallel", "data_parallel"):
        cfg = Config(
            expert_params=35e6 if bad == "expert_parallel" else 1e6,
            strategy=bad, **TWO_NODE_BASE,
        )
        assert not cfg.feasible(max_days=90), bad
        assert any("link-bound" in p for p in cfg.problems()), bad


def test_all_to_all_is_slower_than_a_single_machine():
    """Adding a second laptop under the wrong strategy makes it worse, which
    is the whole reason to measure the link before designing."""
    single = Config(expert_params=1e6, nodes=1, total_params=70e9,
                    active_params_per_token=70e6, tokens=1.4e9,
                    tokens_per_step=65536)
    a2a = Config(expert_params=35e6, strategy="expert_parallel",
                 **TWO_NODE_BASE)
    assert a2a.days > single.days


def test_two_node_speedup_is_superlinear_because_the_shard_halves():
    """Compute halves and I/O quarters: each node streams half the table, and
    two nodes cover the token budget in half the steps."""
    single = Config(expert_params=1e6, nodes=1, total_params=70e9,
                    active_params_per_token=70e6, tokens=1.4e9,
                    tokens_per_step=65536)
    assert single.days / SHARDED.days > 2.0
    assert SHARDED.io_seconds < single.io_seconds / 3
    assert SHARDED.compute_seconds == pytest.approx(
        single.compute_seconds / 2
    )


def test_sharding_halves_storage_per_node():
    single = Config(expert_params=1e6, nodes=1, total_params=70e9,
                    active_params_per_token=70e6, tokens=1.4e9,
                    tokens_per_step=65536)
    assert SHARDED.storage_per_node_gb == pytest.approx(
        single.storage_per_node_gb / 2
    )
    assert SHARDED.storage_per_node_gb < DISK_FREE_GB / 2


def test_trunk_sync_interval_matters_but_does_not_dominate():
    """Even syncing the trunk every step stays affordable; it is the expert
    gradients that cannot cross the link, not the trunk."""
    every_step = Config(expert_params=1e6, strategy="expert_sharded",
                        trunk_sync_every=1, **TWO_NODE_BASE)
    assert every_step.comm_seconds == pytest.approx(
        SHARDED.comm_seconds * 16, rel=1e-6
    )
    assert every_step.comm_seconds < every_step.compute_seconds


# ----------------- claimed accelerations, checked not assumed ---------------

def test_routing_block_size_is_the_real_vectorization_win():
    """Throughput is shape-sensitive: 145.6 GFLOP/s at 128 tokens/expert
    against 173.0 at 512. That win comes from routing block size and is
    already banked in SPARSE_GFLOPS -- not from AVX-512, which this CPU
    (AVX2, Zen 3) does not have."""
    from src.sparse_engine.budget import (
        SPARSE_GFLOPS,
        TOKENS_PER_EXPERT_OPTIMAL,
    )

    assert SPARSE_GFLOPS == 173.0
    assert TOKENS_PER_EXPERT_OPTIMAL == 512


def test_pruning_active_experts_shrinks_the_model_not_the_clock():
    """Compute is 6 * P_active * T. Cutting active experts cuts P_active,
    which is the model. The token budget must shrink with it, so the result
    is a smaller model trained on less data -- not the same model faster."""
    full = Config(expert_params=1e6, nodes=2, strategy="expert_sharded",
                  total_params=70e9, active_params_per_token=70e6,
                  tokens=1.4e9, tokens_per_step=65536)
    pruned = Config(expert_params=1e6, nodes=2, strategy="expert_sharded",
                    total_params=70e9, active_params_per_token=20e6,
                    tokens=1.4e9, tokens_per_step=65536)
    assert pruned.compute_seconds < full.compute_seconds
    assert pruned.active_params_per_token < full.active_params_per_token
    # At the Chinchilla-matched token budget for its smaller active size the
    # saving is real but it is a different, smaller model.
    matched = Config(expert_params=1e6, nodes=2, strategy="expert_sharded",
                     total_params=70e9, active_params_per_token=20e6,
                     tokens=4e8, tokens_per_step=65536)
    assert matched.feasible(max_days=90)


def test_removing_all_cross_node_traffic_barely_helps():
    """Comm is already ~2% of the two-node budget, so delta compression
    cannot deliver a large speedup -- it optimises a solved problem."""
    base = Config(expert_params=1e6, nodes=2, strategy="expert_sharded",
                  total_params=70e9, active_params_per_token=70e6,
                  tokens=1.4e9, tokens_per_step=65536)
    free_comm = base.total_seconds - base.comm_seconds
    assert base.comm_seconds / base.total_seconds < 0.03
    assert base.total_seconds / free_comm < 1.05


def test_delta_sync_makes_data_parallel_viable():
    """Claim 3, rescued. Plain data-parallel is link-bound; the same topology
    with ternary delta sync is not, which restores the full routing pool."""
    plain = Config(expert_params=1e6, strategy="data_parallel",
                   **TWO_NODE_BASE)
    delta = Config(expert_params=1e6, strategy="data_parallel_delta",
                   **TWO_NODE_BASE)
    assert not plain.feasible(max_days=90)
    assert delta.feasible(max_days=90), delta.problems()
    assert delta.comm_gb_per_step < plain.comm_gb_per_step / 100


def test_longer_delta_sync_intervals_cost_less_per_step():
    """Flips accumulate sublinearly (0.231% over 16 steps, 0.894% over 256),
    so amortising over a longer interval is strictly cheaper. The original
    81x bar ignored this and was the wrong test."""
    short = Config(expert_params=1e6, strategy="data_parallel_delta",
                   delta_sync_every=16, **TWO_NODE_BASE)
    long = Config(expert_params=1e6, strategy="data_parallel_delta",
                  delta_sync_every=256, **TWO_NODE_BASE)
    assert long.comm_gb_per_step < short.comm_gb_per_step


def test_data_parallel_keeps_the_full_pool_but_not_the_storage_saving():
    """The honest trade: sharding is faster and halves storage; delta-sync
    keeps every expert reachable at ~2 days and 43.7 GB/node more."""
    sharded = Config(expert_params=1e6, strategy="expert_sharded",
                     **TWO_NODE_BASE)
    delta = Config(expert_params=1e6, strategy="data_parallel_delta",
                   **TWO_NODE_BASE)
    assert delta.days > sharded.days
    assert delta.storage_per_node_gb > sharded.storage_per_node_gb
    assert delta.storage_per_node_gb < DISK_FREE_GB
    assert delta.shard_fraction == 1.0

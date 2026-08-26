from outreachlm.architecture_scaling import ScalingSpec, run_architecture_scaling_experiments
from outreachlm.model_config import DenseTransformerConfig


def test_architecture_scaling_experiments_emit_scaling_curve_metrics():
    specs = [
        ScalingSpec(
            name="D-small",
            config=DenseTransformerConfig(
                vocab_size=64,
                context_length=16,
                embedding_dim=32,
                num_layers=1,
                num_heads=4,
                ffn_dim=64,
            ),
            steps=2,
            batch_size=2,
        ),
        ScalingSpec(
            name="D-medium",
            config=DenseTransformerConfig(
                vocab_size=64,
                context_length=16,
                embedding_dim=64,
                num_layers=2,
                num_heads=4,
                ffn_dim=128,
            ),
            steps=2,
            batch_size=2,
        ),
        ScalingSpec(
            name="D-large",
            config=DenseTransformerConfig(
                vocab_size=64,
                context_length=16,
                embedding_dim=96,
                num_layers=3,
                num_heads=6,
                ffn_dim=192,
            ),
            steps=2,
            batch_size=2,
        ),
    ]
    results = run_architecture_scaling_experiments(specs, seed=0)
    assert [item["name"] for item in results] == ["D-small", "D-medium", "D-large"]
    assert results[0]["parameter_count"] < results[1]["parameter_count"] < results[2]["parameter_count"]
    assert all(item["tokens_per_second"] > 0 for item in results)
    assert all(item["flops_per_step"] > 0 for item in results)


def test_moe_scaling_experiments_report_active_params_and_utilization():
    specs = [
        ScalingSpec(
            name="moe-2x1",
            config=DenseTransformerConfig(
                vocab_size=64,
                context_length=16,
                embedding_dim=32,
                num_layers=1,
                num_heads=4,
                ffn_dim=64,
                moe_enabled=True,
                num_experts=2,
                top_k=1,
                load_balancing_weight=0.01,
            ),
            steps=2,
            batch_size=2,
        ),
        ScalingSpec(
            name="moe-4x2",
            config=DenseTransformerConfig(
                vocab_size=64,
                context_length=16,
                embedding_dim=32,
                num_layers=1,
                num_heads=4,
                ffn_dim=64,
                moe_enabled=True,
                num_experts=4,
                top_k=2,
                load_balancing_weight=0.01,
            ),
            steps=2,
            batch_size=2,
        ),
    ]
    results = run_architecture_scaling_experiments(specs, seed=3)
    assert results[0]["active_parameters_per_token"] > 0
    assert results[1]["active_parameters_per_token"] > 0
    assert isinstance(results[0]["expert_utilization"], list)
    assert 0.0 <= results[0]["overflow_ratio"] <= 1.0

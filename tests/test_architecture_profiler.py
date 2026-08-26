from outreachlm.architecture_profiler import profile_architecture
from outreachlm.model_config import DenseTransformerConfig


def test_architecture_profiler_reports_positive_metrics():
    cfg = DenseTransformerConfig(
        vocab_size=128,
        context_length=64,
        embedding_dim=64,
        num_layers=2,
        num_heads=4,
        ffn_dim=128,
    )
    profile = profile_architecture(
        cfg,
        per_device_batch_size=2,
        gradient_accumulation_steps=2,
        world_size=2,
    )
    payload = profile.to_dict()
    assert payload["total_parameters"] > 0
    assert payload["embedding_parameters"] > 0
    assert payload["attention_parameters"] > 0
    assert payload["ffn_parameters"] > 0
    assert payload["normalization_parameters"] > 0
    assert payload["parameter_memory_bytes"] > 0
    assert payload["flops_per_token"] > 0
    assert payload["flops_per_step"] > 0
    assert payload["active_parameters_per_token"] > 0


def test_architecture_profiler_scales_up_with_larger_model():
    small = DenseTransformerConfig(
        vocab_size=128,
        context_length=64,
        embedding_dim=64,
        num_layers=2,
        num_heads=4,
        ffn_dim=128,
    )
    large = DenseTransformerConfig(
        vocab_size=128,
        context_length=128,
        embedding_dim=128,
        num_layers=4,
        num_heads=8,
        ffn_dim=256,
    )
    small_profile = profile_architecture(small)
    large_profile = profile_architecture(large)
    assert large_profile.total_parameters > small_profile.total_parameters
    assert large_profile.flops_per_step > small_profile.flops_per_step


def test_architecture_profiler_moe_active_parameters_less_than_total():
    cfg = DenseTransformerConfig(
        vocab_size=128,
        context_length=64,
        embedding_dim=64,
        num_layers=2,
        num_heads=4,
        ffn_dim=128,
        moe_enabled=True,
        num_experts=8,
        top_k=2,
        expert_ffn_dim=128,
    )
    profile = profile_architecture(cfg)
    assert profile.expert_parameters > 0
    assert profile.router_parameters > 0
    assert profile.active_parameters_per_token < profile.total_parameters

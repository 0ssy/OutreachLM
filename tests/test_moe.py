import torch

from outreachlm.model_config import DenseTransformerConfig
from outreachlm.moe import ExpertFFN, MoELayer, TopKRouter


def test_expert_parameters_are_independent() -> None:
    cfg = DenseTransformerConfig(
        vocab_size=64,
        context_length=16,
        embedding_dim=32,
        num_layers=1,
        num_heads=4,
        ffn_dim=64,
        moe_enabled=True,
        num_experts=2,
        top_k=1,
    )
    layer = MoELayer(cfg)
    assert layer.experts[0] is not layer.experts[1]
    assert layer.experts[0].w1.weight.data_ptr() != layer.experts[1].w1.weight.data_ptr()


def test_router_topk_shapes_and_determinism() -> None:
    torch.manual_seed(42)
    cfg = DenseTransformerConfig(
        vocab_size=64,
        context_length=16,
        embedding_dim=32,
        num_layers=1,
        num_heads=4,
        ffn_dim=64,
        moe_enabled=True,
        num_experts=4,
        top_k=2,
    )
    router = TopKRouter(cfg)
    x = torch.randn(5, cfg.embedding_dim)
    out1 = router(x)
    out2 = router(x)
    assert out1.logits.shape == (5, 4)
    assert out1.topk_indices.shape == (5, 2)
    assert out1.topk_weights.shape == (5, 2)
    assert torch.allclose(out1.probabilities, out2.probabilities, atol=1e-7, rtol=1e-7)


def test_moe_dispatch_sparse_and_shape() -> None:
    torch.manual_seed(7)
    cfg = DenseTransformerConfig(
        vocab_size=64,
        context_length=8,
        embedding_dim=32,
        num_layers=1,
        num_heads=4,
        ffn_dim=64,
        moe_enabled=True,
        num_experts=4,
        top_k=2,
        capacity_factor=0.5,
        moe_fallback="drop",
    )
    moe = MoELayer(cfg)
    x = torch.randn(2, 8, 32)
    y, aux, stats = moe(x)
    assert y.shape == x.shape
    assert aux.item() >= 0.0
    assert stats.tokens_routed == 2 * 8 * 2
    assert stats.tokens_accepted <= stats.tokens_routed
    assert stats.tokens_overflowed >= 0
    assert sum(stats.expert_routed) == stats.tokens_routed
    assert stats.routing_entropy_mean >= 0.0
    assert 0.0 <= stats.expert_balance_score <= 1.0


def test_moe_gradients_flow() -> None:
    torch.manual_seed(9)
    cfg = DenseTransformerConfig(
        vocab_size=64,
        context_length=8,
        embedding_dim=32,
        num_layers=1,
        num_heads=4,
        ffn_dim=64,
        moe_enabled=True,
        num_experts=4,
        top_k=2,
        load_balancing_weight=0.1,
    )
    moe = MoELayer(cfg)
    x = torch.randn(2, 8, 32, requires_grad=True)
    out, aux, _ = moe(x)
    loss = out.pow(2).mean() + (0.1 * aux)
    loss.backward()
    grads = [parameter.grad for parameter in moe.parameters() if parameter.requires_grad]
    assert any(grad is not None for grad in grads)


def test_capacity_overflow_fallback_dense_reduces_drops() -> None:
    torch.manual_seed(5)
    base = DenseTransformerConfig(
        vocab_size=64,
        context_length=8,
        embedding_dim=32,
        num_layers=1,
        num_heads=4,
        ffn_dim=64,
        moe_enabled=True,
        num_experts=2,
        top_k=2,
        capacity_factor=0.25,
    )
    drop_layer = MoELayer(base)
    dense_layer = MoELayer(DenseTransformerConfig.from_dict({**base.to_dict(), "moe_fallback": "dense"}))
    x = torch.randn(2, 8, 32)
    _, _, drop_stats = drop_layer(x)
    _, _, dense_stats = dense_layer(x)
    assert dense_stats.tokens_dropped <= drop_stats.tokens_dropped

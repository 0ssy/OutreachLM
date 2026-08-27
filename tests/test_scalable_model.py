import torch

from outreachlm.model_config import DenseTransformerConfig
from outreachlm.scalable_model import ScalableTransformerModel
from outreachlm.v4_model import OutreachV4Model


def test_scalable_model_forward_and_backward():
    cfg = DenseTransformerConfig(
        vocab_size=64,
        context_length=32,
        embedding_dim=64,
        num_layers=2,
        num_heads=4,
        ffn_dim=128,
    )
    model = ScalableTransformerModel(cfg)
    inputs = torch.randint(0, cfg.vocab_size, (2, 32))
    targets = torch.randint(0, cfg.vocab_size, (2, 32))
    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
    loss.backward()
    assert logits.shape == (2, 32, cfg.vocab_size)


def test_scalable_model_supports_ffn_variants():
    for variant in ("swiglu", "standard", "gated"):
        cfg = DenseTransformerConfig(
            vocab_size=32,
            context_length=16,
            embedding_dim=32,
            num_layers=1,
            num_heads=4,
            ffn_dim=64,
            ffn_variant=variant,
        )
        model = ScalableTransformerModel(cfg)
        logits = model(torch.randint(0, cfg.vocab_size, (1, 16)))
        assert logits.shape == (1, 16, cfg.vocab_size)


def test_scalable_model_context_scaling_configs_construct():
    for context_length in (256, 512, 1024, 2048, 4096, 8192):
        cfg = DenseTransformerConfig(
            vocab_size=64,
            context_length=context_length,
            embedding_dim=32,
            num_layers=1,
            num_heads=4,
            ffn_dim=64,
        )
        model = ScalableTransformerModel(cfg)
        logits = model(torch.randint(0, cfg.vocab_size, (1, 8)))
        assert logits.shape == (1, 8, cfg.vocab_size)


def test_v4_compatibility_state_dict_and_output_equivalence():
    v4 = OutreachV4Model(
        vocab_size=128,
        context_length=64,
        embedding_dim=64,
        num_layers=2,
        num_heads=4,
        ffn_dim=128,
    )
    cfg = DenseTransformerConfig(
        vocab_size=128,
        context_length=64,
        embedding_dim=64,
        num_layers=2,
        num_heads=4,
        ffn_dim=128,
        normalization="rmsnorm",
        positional_encoding="rope",
        ffn_variant="swiglu",
        attention_dropout=0.0,
        dropout=0.0,
        use_bias=True,
        tie_embeddings=True,
    )
    scalable = ScalableTransformerModel(cfg)
    scalable.load_state_dict(v4.state_dict(), strict=True)

    inputs = torch.randint(0, cfg.vocab_size, (2, 32))
    v4_logits = v4(inputs)
    scalable_logits = scalable(inputs)
    assert torch.allclose(v4_logits, scalable_logits, atol=1e-6, rtol=1e-6)


def test_scalable_model_moe_mode_produces_router_stats_and_combined_loss():
    cfg = DenseTransformerConfig(
        vocab_size=64,
        context_length=16,
        embedding_dim=32,
        num_layers=2,
        num_heads=4,
        ffn_dim=64,
        moe_enabled=True,
        num_experts=4,
        top_k=2,
        load_balancing_weight=0.2,
    )
    model = ScalableTransformerModel(cfg)
    inputs = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(inputs)
    language_loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, cfg.vocab_size),
        targets.reshape(-1),
    )
    total_loss = model.combine_with_moe_loss(language_loss)
    assert total_loss.item() >= language_loss.item()
    assert len(model.last_moe_stats) == cfg.num_layers


def test_scalable_model_supports_grouped_query_attention():
    cfg = DenseTransformerConfig(
        vocab_size=64,
        context_length=16,
        embedding_dim=64,
        num_layers=2,
        num_heads=8,
        kv_heads=2,
        ffn_dim=128,
    )
    model = ScalableTransformerModel(cfg)
    logits = model(torch.randint(0, cfg.vocab_size, (2, 16)))
    assert logits.shape == (2, 16, cfg.vocab_size)


def test_scalable_model_supports_multi_query_attention():
    cfg = DenseTransformerConfig(
        vocab_size=64,
        context_length=16,
        embedding_dim=64,
        num_layers=2,
        num_heads=8,
        kv_heads=1,
        ffn_dim=128,
    )
    model = ScalableTransformerModel(cfg)
    logits = model(torch.randint(0, cfg.vocab_size, (2, 16)))
    assert logits.shape == (2, 16, cfg.vocab_size)

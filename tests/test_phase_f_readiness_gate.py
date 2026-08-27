import torch
import torch.nn as nn

from outreachlm.architecture_profiler import profile_architecture
from outreachlm.architecture_scaling import ScalingSpec, run_architecture_scaling_experiments
from outreachlm.checkpoint import load_distributed_checkpoint, save_distributed_checkpoint
from outreachlm.data_pipeline import ResumableShardedBatchSource
from outreachlm.datasets import LanguageModelDataset
from outreachlm.evaluation_profiles import FrontierEvaluationProfile
from outreachlm.frontier_evaluation import summarize_frontier_step
from outreachlm.model_config import DenseTransformerConfig
from outreachlm.runtime import SingleDeviceRuntime
from outreachlm.scalable_model import ScalableTransformerModel


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int = 32, embedding_dim: int = 16) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.output(self.embedding(input_ids))


def test_phase_f_readiness_gate_contracts_hold(tmp_path):
    dense_cfg = DenseTransformerConfig(
        vocab_size=64,
        context_length=16,
        embedding_dim=64,
        num_layers=2,
        num_heads=8,
        kv_heads=8,
        ffn_dim=128,
    )
    moe_cfg = DenseTransformerConfig(
        vocab_size=64,
        context_length=16,
        embedding_dim=64,
        num_layers=2,
        num_heads=8,
        kv_heads=2,
        ffn_dim=128,
        moe_enabled=True,
        num_experts=4,
        top_k=2,
        load_balancing_weight=0.01,
    )
    dense_model = ScalableTransformerModel(dense_cfg)
    moe_model = ScalableTransformerModel(moe_cfg)
    dense_logits = dense_model(torch.randint(0, dense_cfg.vocab_size, (2, 16)))
    moe_logits = moe_model(torch.randint(0, moe_cfg.vocab_size, (2, 16)))
    assert dense_logits.shape == (2, 16, dense_cfg.vocab_size)
    assert moe_logits.shape == (2, 16, moe_cfg.vocab_size)

    dense_profile = profile_architecture(dense_cfg)
    moe_profile = profile_architecture(moe_cfg)
    assert dense_profile.total_parameters > 0
    assert moe_profile.active_parameters_per_token < moe_profile.total_parameters

    scaling = run_architecture_scaling_experiments(
        [
            ScalingSpec(name="dense", config=dense_cfg, steps=2, batch_size=2),
            ScalingSpec(name="moe", config=moe_cfg, steps=2, batch_size=2),
        ],
        seed=3,
    )
    assert scaling[0]["tokens_per_second"] > 0
    assert scaling[1]["routing_entropy_mean"] >= 0.0

    stream = ResumableShardedBatchSource(
        LanguageModelDataset(token_ids=(torch.arange(0, 260, dtype=torch.long) % 32), context_length=8),
        batch_size=4,
        rank=0,
        world_size=1,
        sequence_packing=2,
        shuffle=True,
        seed=11,
    )
    stream.next_batch()
    stream_state = stream.state_dict()
    assert "position" in stream_state and "epoch" in stream_state

    model = TinyLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_distributed_checkpoint(
        tmp_path / "checkpoint",
        model,
        optimizer,
        step=1,
        train_loss=0.0,
        best_validation_loss=0.0,
        config={"phase": "f10"},
        trainer_state={"data_position": stream_state},
    )
    loaded = load_distributed_checkpoint(
        tmp_path / "checkpoint",
        model,
        optimizer,
        torch.device("cpu"),
    )
    assert loaded["trainer_state"]["data_position"]["position"] >= 0

    eval_profile = FrontierEvaluationProfile()
    summary = summarize_frontier_step(
        {
            "throughput": {"tokens_per_second": 1.0, "samples_per_second": 1.0},
            "memory": {"communication_time_seconds": 0.0},
            "moe": {"overflow_ratio": 0.0, "expert_utilization": []},
        },
        validation_loss=1.0,
    )
    assert eval_profile.compute_perplexity is True
    assert summary["quality"]["perplexity"] is not None
    runtime = SingleDeviceRuntime("cpu")
    assert runtime.info.is_distributed is False

from outreachlm.frontier_evaluation import summarize_frontier_step


def test_frontier_evaluation_summary_includes_quality_system_and_moe_sections():
    summary = summarize_frontier_step(
        {
            "throughput": {
                "tokens_per_second": 100.0,
                "samples_per_second": 10.0,
                "global_tokens_per_second": 200.0,
                "global_samples_per_second": 20.0,
                "time_per_data_seconds": 0.01,
                "time_per_optimizer_seconds": 0.02,
            },
            "memory": {
                "communication_time_seconds": 0.5,
                "estimated_training_state_bytes": 1024,
            },
            "moe": {
                "load_balancing_loss": 0.1,
                "expert_utilization": [0.5, 0.5],
                "routing_entropy_mean": 0.69,
                "tokens_dropped": 1,
                "tokens_overflowed": 2,
                "overflow_ratio": 0.1,
                "expert_balance_score": 0.95,
            },
        },
        validation_loss=2.0,
        teacher_accuracy=0.4,
        free_rollout_match=0.25,
        divergence_position=41,
        recovery_rate=0.6,
    )

    assert summary["quality"]["validation_loss"] == 2.0
    assert summary["quality"]["perplexity"] is not None
    assert summary["systems"]["tokens_per_second"] == 100.0
    assert summary["systems"]["communication_time_seconds"] == 0.5
    assert summary["moe"]["overflow_ratio"] == 0.1

import pytest

from outreachlm.evaluation_profiles import EvaluationProfile, FrontierEvaluationProfile


def test_evaluation_profile_defaults_match_current_scripts():
    profile = EvaluationProfile()
    assert profile.prompt_length == 40
    assert profile.eval_length == 80
    assert profile.position_start == 40
    assert profile.position_end == 52
    assert profile.sample_count == 4096
    assert profile.sample_seed == 42
    assert profile.sample_batch_size == 256
    assert profile.fallback_topk == 5
    assert profile.heldout_slices == 4
    assert profile.output_topk == 5
    assert profile.hidden_transition_start == 38
    assert profile.hidden_transition_end == 45
    assert profile.output_sensitivity_start == 39
    assert profile.output_sensitivity_end == 43


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt_length": 0},
        {"eval_length": 0},
        {"eval_length": 40, "prompt_length": 40},
        {"position_start": 0},
        {"position_start": 45, "position_end": 44},
        {"sample_count": 0},
        {"sample_seed": -1},
        {"sample_batch_size": 0},
        {"fallback_topk": 0},
        {"heldout_slices": 0},
        {"output_topk": 0},
        {"hidden_transition_start": 0},
        {"hidden_transition_start": 46, "hidden_transition_end": 45},
        {"output_sensitivity_start": 0},
        {"output_sensitivity_start": 44, "output_sensitivity_end": 43},
        {"hidden_transition_end": 80},
        {"output_sensitivity_end": 80},
    ],
)
def test_evaluation_profile_validation(payload):
    kwargs = EvaluationProfile().to_dict()
    kwargs.update(payload)
    with pytest.raises(ValueError):
        EvaluationProfile(**kwargs)


def test_evaluation_profile_round_trip_dict():
    original = EvaluationProfile(
        prompt_length=48,
        eval_length=96,
        position_start=41,
        position_end=60,
        sample_count=1024,
        sample_seed=7,
        sample_batch_size=128,
        fallback_topk=10,
        heldout_slices=5,
        output_topk=8,
        hidden_transition_start=40,
        hidden_transition_end=50,
        output_sensitivity_start=42,
        output_sensitivity_end=47,
    )
    restored = EvaluationProfile.from_dict(original.to_dict())
    assert restored == original


def test_frontier_evaluation_profile_round_trip_dict():
    original = FrontierEvaluationProfile(
        validation_batches=16,
        compute_perplexity=True,
        compute_teacher_accuracy=True,
        compute_free_rollout=True,
        compute_divergence=True,
        compute_recovery=True,
        long_context_eval_length=2048,
        compute_tokens_per_second=True,
        compute_samples_per_second=True,
        compute_memory=True,
        compute_communication_overhead=True,
        compute_checkpoint_time=True,
        compute_data_loading_time=True,
        compute_moe_metrics=True,
    )
    restored = FrontierEvaluationProfile.from_dict(original.to_dict())
    assert restored == original


@pytest.mark.parametrize(
    "payload",
    [
        {"validation_batches": 0},
        {"long_context_eval_length": 0},
    ],
)
def test_frontier_evaluation_profile_validation(payload):
    kwargs = FrontierEvaluationProfile().to_dict()
    kwargs.update(payload)
    with pytest.raises(ValueError):
        FrontierEvaluationProfile(**kwargs)

import torch
import torch.nn.functional as F

from outreachlm.loss_plans import (
    LossPlan,
    LossTerm,
    PostErrorLossTerm,
    RecoveryLossTerm,
    RolloutCalibrationLossTerm,
    TeacherLossTerm,
)


def _constant_loss(name: str, value: float, weight: float = 1.0, enabled: bool = True) -> LossTerm:
    return LossTerm(
        name=name,
        weight=weight,
        enabled=enabled,
        loss_fn=lambda _ctx: torch.tensor(value, dtype=torch.float32),
    )


def test_teacher_only_plan_works():
    plan = LossPlan(
        [
            TeacherLossTerm(
                loss_fn=lambda _ctx: torch.tensor(3.0, dtype=torch.float32),
            )
        ]
    )
    result = plan.compute({})
    assert torch.isclose(result.total_loss, torch.tensor(3.0))
    assert "teacher" in result.term_losses


def test_weighted_terms_combine_correctly():
    plan = LossPlan(
        [
            TeacherLossTerm(loss_fn=lambda _ctx: torch.tensor(2.0), weight=1.0),
            RecoveryLossTerm(loss_fn=lambda _ctx: torch.tensor(4.0), weight=0.5),
        ]
    )
    result = plan.compute({})
    assert torch.isclose(result.total_loss, torch.tensor(4.0))


def test_disabled_terms_do_not_contribute():
    plan = LossPlan(
        [
            _constant_loss("a", 1.0, weight=1.0, enabled=False),
            _constant_loss("b", 2.0, weight=2.0, enabled=True),
        ]
    )
    result = plan.compute({})
    assert torch.isclose(result.total_loss, torch.tensor(4.0))
    assert "a" not in result.term_losses
    assert "b" in result.term_losses


def test_multiple_terms_compose_correctly():
    plan = LossPlan(
        [
            TeacherLossTerm(loss_fn=lambda _ctx: torch.tensor(1.0), weight=1.0),
            RecoveryLossTerm(loss_fn=lambda _ctx: torch.tensor(2.0), weight=2.0),
            PostErrorLossTerm(loss_fn=lambda _ctx: torch.tensor(3.0), weight=0.5),
        ]
    )
    result = plan.compute({})
    assert torch.isclose(result.total_loss, torch.tensor(6.5))


def test_diagnostics_include_individual_losses():
    plan = LossPlan(
        [
            TeacherLossTerm(loss_fn=lambda _ctx: torch.tensor(1.0), weight=1.0),
            RecoveryLossTerm(loss_fn=lambda _ctx: torch.tensor(2.0), weight=2.0),
        ]
    )
    result = plan.compute({})
    assert set(result.term_losses.keys()) == {"teacher", "recovery"}
    assert set(result.weighted_term_losses.keys()) == {"teacher", "recovery"}


def test_zero_weight_term_contributes_zero():
    plan = LossPlan(
        [
            TeacherLossTerm(loss_fn=lambda _ctx: torch.tensor(5.0), weight=1.0),
            RecoveryLossTerm(loss_fn=lambda _ctx: torch.tensor(100.0), weight=0.0),
        ]
    )
    result = plan.compute({})
    assert torch.isclose(result.total_loss, torch.tensor(5.0))
    assert torch.isclose(result.weighted_term_losses["recovery"], torch.tensor(0.0))


def test_gradients_flow_through_composed_loss():
    x = torch.tensor(2.0, requires_grad=True)
    plan = LossPlan(
        [
            TeacherLossTerm(loss_fn=lambda _ctx: x.pow(2), weight=1.0),
            RecoveryLossTerm(loss_fn=lambda _ctx: x * 3.0, weight=0.5),
        ]
    )
    result = plan.compute({})
    result.total_loss.backward()
    assert x.grad is not None
    assert x.grad.abs().item() > 0


def test_rollout_calibration_term_plugs_in_without_trainer_changes():
    plan = LossPlan(
        [
            TeacherLossTerm(loss_fn=lambda _ctx: torch.tensor(1.0), weight=1.0),
            RolloutCalibrationLossTerm(
                loss_fn=lambda _ctx: torch.tensor(0.25),
                weight=2.0,
            ),
        ]
    )
    result = plan.compute({})
    assert "rollout_calib" in result.term_losses
    assert torch.isclose(result.total_loss, torch.tensor(1.5))


def test_regression_v4_weighted_objective_math():
    teacher_logits = torch.tensor(
        [[[2.0, 0.0], [0.5, 1.0]]],
        requires_grad=True,
    )
    recovery_logits = torch.tensor(
        [[[1.8, 0.2], [0.4, 1.1]]],
        requires_grad=True,
    )
    post_error_logits = torch.tensor(
        [[[1.5, 0.5], [0.6, 0.9]]],
        requires_grad=True,
    )
    rollout_calib_loss = torch.tensor(0.3, requires_grad=True)
    targets = torch.tensor([[0, 1]])

    context = {
        "teacher_logits": teacher_logits,
        "recovery_logits": recovery_logits,
        "post_error_logits": post_error_logits,
        "targets": targets,
        "rollout_calib_loss": rollout_calib_loss,
    }

    plan = LossPlan(
        [
            TeacherLossTerm(
                weight=1.0,
                loss_fn=lambda ctx: F.cross_entropy(
                    ctx["teacher_logits"].reshape(-1, 2),
                    ctx["targets"].reshape(-1),
                ),
            ),
            RecoveryLossTerm(
                weight=2.0,
                loss_fn=lambda ctx: F.cross_entropy(
                    ctx["recovery_logits"].reshape(-1, 2),
                    ctx["targets"].reshape(-1),
                ),
            ),
            PostErrorLossTerm(
                weight=0.5,
                loss_fn=lambda ctx: F.cross_entropy(
                    ctx["post_error_logits"].reshape(-1, 2),
                    ctx["targets"].reshape(-1),
                ),
            ),
            RolloutCalibrationLossTerm(
                weight=1.5,
                loss_fn=lambda ctx: ctx["rollout_calib_loss"],
            ),
        ]
    )

    result = plan.compute(context)

    teacher = F.cross_entropy(teacher_logits.reshape(-1, 2), targets.reshape(-1))
    recovery = F.cross_entropy(recovery_logits.reshape(-1, 2), targets.reshape(-1))
    post_error = F.cross_entropy(post_error_logits.reshape(-1, 2), targets.reshape(-1))
    expected = teacher + (2.0 * recovery) + (0.5 * post_error) + (1.5 * rollout_calib_loss)

    assert torch.isclose(result.total_loss, expected)
    result.total_loss.backward()
    assert teacher_logits.grad is not None
    assert recovery_logits.grad is not None
    assert post_error_logits.grad is not None

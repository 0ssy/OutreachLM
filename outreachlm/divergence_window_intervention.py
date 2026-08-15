import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from outreachlm.architecture_capacity_pilot import (
    metric_row,
    metrics_snapshot,
)
from outreachlm.architecture_capacity_continuation import load_model_from_artifact
from outreachlm.generate import load_model_and_tokenizer
from outreachlm.objective_intervention_experiment import (
    build_frequency_balanced_weights,
)
from outreachlm.train import (
    CORPUS_PATH,
    VALIDATION_SPLIT,
    build_model_artifact,
    calculate_loss,
    get_learning_rate,
    get_random_batch,
    load_corpus,
    split_corpus,
)


def build_recovery_mixed_inputs(
    input_ids,
    teacher_logits,
    recovery_start_index,
):
    if recovery_start_index < 1:
        raise ValueError("recovery_start_index must be >= 1.")
    sequence_length = input_ids.shape[1]
    if recovery_start_index >= sequence_length:
        return input_ids.clone()

    predicted_next = torch.argmax(teacher_logits[:, :-1, :], dim=-1)
    mixed = input_ids.clone()
    mixed[:, recovery_start_index:] = predicted_next[:, recovery_start_index - 1 :]
    return mixed


def run_divergence_window_training(
    model,
    tokenizer,
    training_token_ids,
    validation_text,
    steps,
    batch_size,
    learning_rate,
    warmup_steps,
    min_learning_rate_ratio,
    label_smoothing,
    recovery_start_index,
    recovery_loss_weight,
    seed,
):
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )
    class_weights, _ = build_frequency_balanced_weights(
        training_token_ids=training_token_ids,
        vocab_size=tokenizer.vocab_size,
        device=device,
    )
    loss_function = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing,
    )

    losses = []
    teacher_losses = []
    recovery_losses = []
    checkpoints = []

    model.train()
    for step in range(1, steps + 1):
        input_ids, target_ids = get_random_batch(
            training_token_ids,
            model.context_length,
            batch_size,
            device,
        )

        teacher_logits = model(input_ids)
        if isinstance(teacher_logits, tuple):
            teacher_logits = teacher_logits[0]
        teacher_loss = calculate_loss(
            teacher_logits,
            target_ids,
            loss_function,
        )

        mixed_input_ids = build_recovery_mixed_inputs(
            input_ids=input_ids,
            teacher_logits=teacher_logits.detach(),
            recovery_start_index=recovery_start_index,
        )

        recovery_logits = model(mixed_input_ids)
        if isinstance(recovery_logits, tuple):
            recovery_logits = recovery_logits[0]

        if recovery_start_index >= target_ids.shape[1]:
            recovery_loss = torch.zeros((), device=device, dtype=teacher_loss.dtype)
        else:
            recovery_loss = calculate_loss(
                recovery_logits[:, recovery_start_index:, :],
                target_ids[:, recovery_start_index:],
                loss_function,
            )

        total_loss = teacher_loss + recovery_loss_weight * recovery_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()

        current_lr = get_learning_rate(
            step=step - 1,
            max_steps=steps,
            base_learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            min_learning_rate_ratio=min_learning_rate_ratio,
        )
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.step()

        losses.append(float(total_loss.item()))
        teacher_losses.append(float(teacher_loss.item()))
        recovery_losses.append(float(recovery_loss.item()))

        if step % max(steps // 4, 1) == 0 or step == steps:
            model.eval()
            checkpoints.append(
                {
                    "step": step,
                    "learning_rate": float(current_lr),
                    "metrics": metrics_snapshot(
                        model,
                        tokenizer,
                        validation_text,
                    ),
                }
            )
            model.train()

    model.eval()
    return {
        "model": model,
        "loss_summary": {
            "first_total_loss": float(losses[0]) if losses else None,
            "last_total_loss": float(losses[-1]) if losses else None,
            "mean_total_loss": float(sum(losses) / len(losses)) if losses else None,
            "first_teacher_loss": float(teacher_losses[0]) if teacher_losses else None,
            "last_teacher_loss": float(teacher_losses[-1]) if teacher_losses else None,
            "first_recovery_loss": float(recovery_losses[0]) if recovery_losses else None,
            "last_recovery_loss": float(recovery_losses[-1]) if recovery_losses else None,
        },
        "checkpoint_metrics": checkpoints,
        "final_metrics": metrics_snapshot(
            model,
            tokenizer,
            validation_text,
        ),
    }


def render_report(results):
    lines = []
    lines.append("OUTREACHLM DIVERGENCE-WINDOW INTERVENTION")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {results['timestamp']}")
    lines.append("")
    cfg = results["config"]
    lines.append(
        "CONFIG | steps={steps} batch={batch} lr={lr} warmup={warmup} "
        "label_smoothing={ls} recovery_start={rs} recovery_weight={rw}".format(
            steps=cfg["steps"],
            batch=cfg["batch_size"],
            lr=cfg["learning_rate"],
            warmup=cfg["warmup_steps"],
            ls=cfg["label_smoothing"],
            rs=cfg["recovery_start_index"],
            rw=cfg["recovery_loss_weight"],
        )
    )
    lines.append("")
    lines.append("METRIC COMPARISON")
    lines.append("-" * 80)
    lines.append(
        "condition,teacher_top1,free_match,prompt_logit_cosine,rollout_mean_entropy,"
        "first_repeated_bigram_step,first_repeated_trigram_step,first_free_divergence"
    )
    for condition in [
        "baseline_v1",
        "v2_before_intervention",
        "v2_after_intervention",
    ]:
        row = results["summary_table"][condition]
        lines.append(
            "{c},{t:.6f},{f:.6f},{p:.6f},{e:.6f},{b},{g},{d}".format(
                c=condition,
                t=row["teacher_top1"],
                f=row["free_match"],
                p=row["prompt_logit_cosine"],
                e=row["rollout_mean_entropy"],
                b=row["first_repeated_bigram_step"],
                g=row["first_repeated_trigram_step"],
                d=row["first_free_divergence"],
            )
        )

    lines.append("")
    ls = results["intervention"]["loss_summary"]
    lines.append(
        "Intervention total loss first/last/mean: "
        f"{ls['first_total_loss']:.6f} / {ls['last_total_loss']:.6f} / {ls['mean_total_loss']:.6f}"
    )
    lines.append(
        "Teacher loss first->last: "
        f"{ls['first_teacher_loss']:.6f} -> {ls['last_teacher_loss']:.6f}"
    )
    lines.append(
        "Recovery loss first->last: "
        f"{ls['first_recovery_loss']:.6f} -> {ls['last_recovery_loss']:.6f}"
    )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a single divergence-window intervention from V2 checkpoint "
            "to target teacher/free mismatch near position 41."
        ),
    )
    parser.add_argument(
        "--resume-artifact",
        type=Path,
        default=Path("experiments/architecture-v2-continuation-20260815-133859.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--min-learning-rate-ratio", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--recovery-start-index", type=int, default=40)
    parser.add_argument("--recovery-loss-weight", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    baseline_model, tokenizer = load_model_and_tokenizer()
    baseline_model.eval()

    v2_model, v2_model_config = load_model_from_artifact(args.resume_artifact)
    v2_model.eval()

    corpus_text = load_corpus(CORPUS_PATH)
    training_text, validation_text = split_corpus(
        corpus_text,
        VALIDATION_SPLIT,
    )
    training_token_ids = torch.tensor(
        tokenizer.encode(training_text),
        dtype=torch.long,
    )

    baseline_metrics = metrics_snapshot(
        baseline_model,
        tokenizer,
        validation_text,
    )
    v2_before_metrics = metrics_snapshot(
        v2_model,
        tokenizer,
        validation_text,
    )

    intervention = run_divergence_window_training(
        model=v2_model,
        tokenizer=tokenizer,
        training_token_ids=training_token_ids,
        validation_text=validation_text,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        min_learning_rate_ratio=args.min_learning_rate_ratio,
        label_smoothing=args.label_smoothing,
        recovery_start_index=args.recovery_start_index,
        recovery_loss_weight=args.recovery_loss_weight,
        seed=args.seed,
    )

    summary_table = {
        "baseline_v1": metric_row(baseline_metrics),
        "v2_before_intervention": metric_row(v2_before_metrics),
        "v2_after_intervention": metric_row(intervention["final_metrics"]),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    artifact_path = args.output_dir / f"v2-divergence-intervention-{stamp}.pt"
    model_artifact = build_model_artifact(
        model=intervention["model"],
        tokenizer=tokenizer,
        context_length=v2_model_config["context_length"],
        embedding_dim=v2_model_config["embedding_dim"],
        num_layers=v2_model_config.get("num_layers", 1),
        num_heads=v2_model_config.get("num_heads", 4),
        training_config={
            "seed": args.seed,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "min_learning_rate_ratio": args.min_learning_rate_ratio,
            "label_smoothing": args.label_smoothing,
            "recovery_start_index": args.recovery_start_index,
            "recovery_loss_weight": args.recovery_loss_weight,
            "resume_artifact": str(args.resume_artifact),
            "objective": "balanced_ce_plus_label_smoothing_with_recovery_loss",
        },
    )
    torch.save(model_artifact, artifact_path)

    intervention.pop("model", None)
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "resume_artifact": str(args.resume_artifact.resolve()),
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "min_learning_rate_ratio": args.min_learning_rate_ratio,
            "label_smoothing": args.label_smoothing,
            "recovery_start_index": args.recovery_start_index,
            "recovery_loss_weight": args.recovery_loss_weight,
            "seed": args.seed,
        },
        "baseline_v1": baseline_metrics,
        "v2_before_intervention": v2_before_metrics,
        "intervention": intervention,
        "summary_table": summary_table,
        "artifact_path": str(artifact_path.resolve()),
    }

    json_path = args.output_dir / f"v2-divergence-intervention-{stamp}.json"
    txt_path = args.output_dir / f"v2-divergence-intervention-{stamp}.txt"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(render_report(results))

    print(str(json_path.resolve()))
    print(str(txt_path.resolve()))
    print(str(artifact_path.resolve()))


if __name__ == "__main__":
    main()

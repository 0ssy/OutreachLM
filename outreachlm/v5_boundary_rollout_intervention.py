import argparse
import copy
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from outreachlm.architecture_capacity_continuation import load_model_from_artifact
from outreachlm.architecture_capacity_pilot import metric_row, metrics_snapshot
from outreachlm.divergence_window_intervention import build_recovery_mixed_inputs
from outreachlm.generate import TOKENIZER_PATH, load_tokenizer_artifact, upgrade_legacy_tokenizer_artifact
from outreachlm.objective_intervention_experiment import build_frequency_balanced_weights
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


def choose_forced_wrong_token(logits_at_boundary, gold_input_token):
    top2 = torch.topk(logits_at_boundary, k=2, dim=-1).indices
    top1 = top2[:, 0]
    top2_choice = top2[:, 1]
    forced_wrong = torch.where(top1 != gold_input_token, top1, top2_choice)
    return forced_wrong


def boundary_aware_rollout_loss(
    model,
    loss_function,
    input_ids,
    target_ids,
    teacher_logits,
    boundary_start_index,
    boundary_end_index,
    forced_error_index,
):
    sequence_length = input_ids.shape[1]
    if boundary_start_index < 0:
        raise ValueError("boundary_start_index must be >= 0.")
    if boundary_end_index < boundary_start_index:
        raise ValueError("boundary_end_index must be >= boundary_start_index.")
    if forced_error_index < 1:
        raise ValueError("forced_error_index must be >= 1.")
    if forced_error_index >= sequence_length:
        raise ValueError("forced_error_index must be < sequence length.")

    last_index = min(boundary_end_index, target_ids.shape[1] - 1)
    if boundary_start_index > last_index:
        return torch.zeros((), device=input_ids.device, dtype=teacher_logits.dtype)

    boundary_logits_forced = teacher_logits[:, forced_error_index - 1, :].detach()
    gold_input_token = input_ids[:, forced_error_index]
    forced_wrong = choose_forced_wrong_token(boundary_logits_forced, gold_input_token)

    free_tokens = input_ids[:, : forced_error_index + 1].clone()
    free_tokens[:, forced_error_index] = forced_wrong

    per_position_losses = []
    for target_index in range(forced_error_index, last_index + 1):
        logits = model(free_tokens)
        if isinstance(logits, tuple):
            logits = logits[0]
        step_logits = logits[:, -1, :]

        if target_index >= boundary_start_index:
            gold_target = target_ids[:, target_index]
            per_position_losses.append(loss_function(step_logits, gold_target))

        if target_index < last_index:
            next_token = torch.argmax(step_logits.detach(), dim=-1, keepdim=True)
            free_tokens = torch.cat([free_tokens, next_token], dim=1)

    if not per_position_losses:
        return torch.zeros((), device=input_ids.device, dtype=teacher_logits.dtype)
    return torch.stack(per_position_losses).mean()


def run_v5_intervention(
    base_model,
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
    boundary_loss_weight,
    boundary_start_index,
    boundary_end_index,
    forced_error_index,
    seed,
):
    torch.manual_seed(seed)
    model = copy.deepcopy(base_model)
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

    total_losses = []
    teacher_losses = []
    recovery_losses = []
    boundary_losses = []
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

        boundary_loss = boundary_aware_rollout_loss(
            model=model,
            loss_function=loss_function,
            input_ids=input_ids,
            target_ids=target_ids,
            teacher_logits=teacher_logits,
            boundary_start_index=boundary_start_index,
            boundary_end_index=boundary_end_index,
            forced_error_index=forced_error_index,
        )

        total_loss = (
            teacher_loss
            + (recovery_loss_weight * recovery_loss)
            + (boundary_loss_weight * boundary_loss)
        )

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

        total_losses.append(float(total_loss.item()))
        teacher_losses.append(float(teacher_loss.item()))
        recovery_losses.append(float(recovery_loss.item()))
        boundary_losses.append(float(boundary_loss.item()))

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
            "first_total_loss": float(total_losses[0]) if total_losses else None,
            "last_total_loss": float(total_losses[-1]) if total_losses else None,
            "mean_total_loss": float(sum(total_losses) / len(total_losses))
            if total_losses
            else None,
            "first_teacher_loss": float(teacher_losses[0]) if teacher_losses else None,
            "last_teacher_loss": float(teacher_losses[-1]) if teacher_losses else None,
            "first_recovery_loss": float(recovery_losses[0]) if recovery_losses else None,
            "last_recovery_loss": float(recovery_losses[-1]) if recovery_losses else None,
            "first_boundary_loss": float(boundary_losses[0]) if boundary_losses else None,
            "last_boundary_loss": float(boundary_losses[-1]) if boundary_losses else None,
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
    lines.append("OUTREACHLM V5 BOUNDARY-AWARE ROLLOUT INTERVENTION")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {results['timestamp']}")
    lines.append("")
    cfg = results["config"]
    lines.append(
        "CONFIG | steps={steps} batch={batch} lr={lr} warmup={warmup} "
        "label_smoothing={ls} recovery_start={rs} recovery_weight={rw} "
        "boundary_loss_weight={bw} boundary_window={bs}-{be} forced_error_index={fe}".format(
            steps=cfg["steps"],
            batch=cfg["batch_size"],
            lr=cfg["learning_rate"],
            warmup=cfg["warmup_steps"],
            ls=cfg["label_smoothing"],
            rs=cfg["recovery_start_index"],
            rw=cfg["recovery_loss_weight"],
            bw=cfg["boundary_loss_weight"],
            bs=cfg["boundary_start_index"],
            be=cfg["boundary_end_index"],
            fe=cfg["forced_error_index"],
        )
    )
    lines.append("")
    lines.append("METRIC COMPARISON")
    lines.append("-" * 80)
    lines.append(
        "condition,teacher_top1,free_match,prompt_logit_cosine,rollout_mean_entropy,"
        "first_repeated_bigram_step,first_repeated_trigram_step,first_free_divergence"
    )
    for condition in ["v2_before_intervention", "v5_after_intervention"]:
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
        "Total loss first/last/mean: "
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
    lines.append(
        "Boundary rollout loss first->last: "
        f"{ls['first_boundary_loss']:.6f} -> {ls['last_boundary_loss']:.6f}"
    )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "V5: V2 + recovery + boundary-aware rollout loss around position-41 region."
        )
    )
    parser.add_argument(
        "--resume-artifact",
        type=Path,
        default=Path("experiments/v2-divergence-intervention-20260816-113809.pt"),
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
    parser.add_argument("--recovery-loss-weight", type=float, default=2.0)
    parser.add_argument("--boundary-loss-weight", type=float, default=1.0)
    parser.add_argument("--boundary-start-index", type=int, default=40)
    parser.add_argument("--boundary-end-index", type=int, default=43)
    parser.add_argument("--forced-error-index", type=int, default=40)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = load_tokenizer_artifact(TOKENIZER_PATH)
    if tokenizer is None:
        tokenizer = upgrade_legacy_tokenizer_artifact(TOKENIZER_PATH)

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

    v2_before_metrics = metrics_snapshot(
        v2_model,
        tokenizer,
        validation_text,
    )
    intervention = run_v5_intervention(
        base_model=v2_model,
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
        boundary_loss_weight=args.boundary_loss_weight,
        boundary_start_index=args.boundary_start_index,
        boundary_end_index=args.boundary_end_index,
        forced_error_index=args.forced_error_index,
        seed=args.seed,
    )

    summary_table = {
        "v2_before_intervention": metric_row(v2_before_metrics),
        "v5_after_intervention": metric_row(intervention["final_metrics"]),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    artifact_path = args.output_dir / f"v5-boundary-rollout-intervention-{stamp}.pt"
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
            "boundary_loss_weight": args.boundary_loss_weight,
            "boundary_start_index": args.boundary_start_index,
            "boundary_end_index": args.boundary_end_index,
            "forced_error_index": args.forced_error_index,
            "resume_artifact": str(args.resume_artifact),
            "objective": "balanced_ce_ls_recovery_plus_boundary_rollout",
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
            "boundary_loss_weight": args.boundary_loss_weight,
            "boundary_start_index": args.boundary_start_index,
            "boundary_end_index": args.boundary_end_index,
            "forced_error_index": args.forced_error_index,
            "seed": args.seed,
        },
        "v2_before_intervention": v2_before_metrics,
        "intervention": intervention,
        "summary_table": summary_table,
        "artifact_path": str(artifact_path.resolve()),
    }

    json_path = args.output_dir / f"v5-boundary-rollout-intervention-{stamp}.json"
    txt_path = args.output_dir / f"v5-boundary-rollout-intervention-{stamp}.txt"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(render_report(results))

    print(str(json_path.resolve()))
    print(str(txt_path.resolve()))
    print(str(artifact_path.resolve()))


if __name__ == "__main__":
    main()

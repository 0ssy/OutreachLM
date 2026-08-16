import argparse
import copy
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from outreachlm.architecture_capacity_pilot import (
    metric_row,
    metrics_snapshot,
)
from outreachlm.architecture_capacity_continuation import load_model_from_artifact
from outreachlm.divergence_trajectory_mapping_analysis import (
    systematic_boundary_mapping,
)
from outreachlm.divergence_window_intervention import (
    build_recovery_mixed_inputs,
)
from outreachlm.generate import (
    TOKENIZER_PATH,
    load_tokenizer_artifact,
    upgrade_legacy_tokenizer_artifact,
)
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


def boundary_consistency_kl_loss(
    teacher_logits,
    free_logits,
    teacher_input_ids,
    free_input_ids,
    boundary_indices,
):
    total = torch.zeros((), device=teacher_logits.device)
    count = 0

    for index in boundary_indices:
        if index >= teacher_logits.shape[1]:
            continue

        # Logits at index i predict target token at position i+1.
        # Compare contexts used to produce this step.
        context_teacher = teacher_input_ids[:, : index + 1]
        context_free = free_input_ids[:, : index + 1]
        context_diff = (context_teacher != context_free).any(dim=1)

        if not context_diff.any():
            continue

        t = teacher_logits[context_diff, index, :].detach()
        f = free_logits[context_diff, index, :]
        t_probs = torch.softmax(t, dim=-1)
        f_log_probs = torch.log_softmax(f, dim=-1)
        kl = F.kl_div(
            f_log_probs,
            t_probs,
            reduction="batchmean",
        )
        total = total + kl
        count += 1

    if count == 0:
        return total
    return total / count


def run_test3_intervention(
    base_model,
    tokenizer,
    training_token_ids,
    validation_text,
    validation_token_ids,
    steps,
    batch_size,
    learning_rate,
    warmup_steps,
    min_learning_rate_ratio,
    label_smoothing,
    recovery_start_index,
    recovery_loss_weight,
    boundary_indices,
    consistency_weight,
    systematic_sample_count,
    systematic_batch_size,
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
    consistency_losses = []
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
        free_logits = model(mixed_input_ids)
        if isinstance(free_logits, tuple):
            free_logits = free_logits[0]

        if recovery_start_index >= target_ids.shape[1]:
            recovery_loss = torch.zeros((), device=device, dtype=teacher_loss.dtype)
        else:
            recovery_loss = calculate_loss(
                free_logits[:, recovery_start_index:, :],
                target_ids[:, recovery_start_index:],
                loss_function,
            )

        consistency_loss = boundary_consistency_kl_loss(
            teacher_logits=teacher_logits,
            free_logits=free_logits,
            teacher_input_ids=input_ids,
            free_input_ids=mixed_input_ids,
            boundary_indices=boundary_indices,
        )

        total_loss = (
            teacher_loss
            + (recovery_loss_weight * recovery_loss)
            + (consistency_weight * consistency_loss)
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
        consistency_losses.append(float(consistency_loss.item()))

        if step % max(steps // 4, 1) == 0 or step == steps:
            model.eval()
            checkpoint_metrics = metrics_snapshot(
                model,
                tokenizer,
                validation_text,
            )
            checkpoint_boundary = systematic_boundary_mapping(
                model=model,
                tokenizer=tokenizer,
                validation_token_ids=validation_token_ids,
                prompt_length=40,
                eval_length=80,
                sample_count=1024,
                seed=seed,
                batch_size=systematic_batch_size,
            )
            checkpoints.append(
                {
                    "step": step,
                    "learning_rate": float(current_lr),
                    "metrics": checkpoint_metrics,
                    "boundary_mapping": checkpoint_boundary,
                }
            )
            model.train()

    model.eval()
    final_metrics = metrics_snapshot(
        model,
        tokenizer,
        validation_text,
    )
    final_boundary = systematic_boundary_mapping(
        model=model,
        tokenizer=tokenizer,
        validation_token_ids=validation_token_ids,
        prompt_length=40,
        eval_length=80,
        sample_count=systematic_sample_count,
        seed=seed,
        batch_size=systematic_batch_size,
    )

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
            "first_consistency_loss": float(consistency_losses[0]) if consistency_losses else None,
            "last_consistency_loss": float(consistency_losses[-1]) if consistency_losses else None,
        },
        "checkpoint_metrics": checkpoints,
        "final_metrics": final_metrics,
        "final_boundary_mapping": final_boundary,
    }


def render_report(results):
    lines = []
    lines.append("OUTREACHLM TEST 3 — BOUNDARY CONSISTENCY INTERVENTION")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {results['timestamp']}")
    lines.append("")
    cfg = results["config"]
    lines.append(
        "CONFIG | steps={steps} batch={batch} lr={lr} warmup={warmup} ls={ls} "
        "recovery_start={rs} recovery_weight={rw} consistency_weight={cw} "
        "boundary_indices={bi}".format(
            steps=cfg["steps"],
            batch=cfg["batch_size"],
            lr=cfg["learning_rate"],
            warmup=cfg["warmup_steps"],
            ls=cfg["label_smoothing"],
            rs=cfg["recovery_start_index"],
            rw=cfg["recovery_loss_weight"],
            cw=cfg["consistency_weight"],
            bi=cfg["boundary_indices"],
        )
    )
    lines.append("")
    lines.append("DECISION METRICS")
    lines.append("-" * 80)
    lines.append(
        "condition,teacher_top1,free_match,prompt_logit_cosine,rollout_mean_entropy,"
        "first_repeated_bigram_step,first_repeated_trigram_step,first_free_divergence"
    )
    for condition in ["leader_before", "after_test3"]:
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
    b = results["boundary_before"]
    a = results["run"]["final_boundary_mapping"]
    lines.append("BOUNDARY MAPPING (systematic)")
    lines.append("-" * 80)
    lines.append(
        "pos41 free match (all/same/diff): "
        f"{b['position_41_free_match_rate']:.6f}/"
        f"{b['position_41_free_match_rate_when_context_same']:.6f}/"
        f"{b['position_41_free_match_rate_when_context_diff']:.6f} -> "
        f"{a['position_41_free_match_rate']:.6f}/"
        f"{a['position_41_free_match_rate_when_context_same']:.6f}/"
        f"{a['position_41_free_match_rate_when_context_diff']:.6f}"
    )
    lines.append(
        "logit cos (context diff): "
        f"{b['logit_cos_teacher_vs_free_mean_when_context_diff']:.6f} -> "
        f"{a['logit_cos_teacher_vs_free_mean_when_context_diff']:.6f}"
    )
    lines.append(
        "KL_sym (context diff): "
        f"{b['kl_symmetric_mean_when_context_diff']:.6f} -> "
        f"{a['kl_symmetric_mean_when_context_diff']:.6f}"
    )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Test 3: controlled intervention targeting output-mapping drift at divergence boundary."
        ),
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
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument("--boundary-indices", type=str, default="40,41,42")
    parser.add_argument("--systematic-sample-count", type=int, default=4096)
    parser.add_argument("--systematic-batch-size", type=int, default=256)
    return parser.parse_args()


def parse_indices(raw):
    values = []
    for part in raw.split(","):
        p = part.strip()
        if p:
            values.append(int(p))
    if not values:
        raise ValueError("boundary-indices cannot be empty.")
    return values


def main():
    args = parse_args()
    boundary_indices = parse_indices(args.boundary_indices)
    torch.manual_seed(args.seed)

    tokenizer = load_tokenizer_artifact(TOKENIZER_PATH)
    if tokenizer is None:
        tokenizer = upgrade_legacy_tokenizer_artifact(TOKENIZER_PATH)

    model, model_config = load_model_from_artifact(args.resume_artifact)
    model.eval()

    text = load_corpus(CORPUS_PATH)
    training_text, validation_text = split_corpus(text, VALIDATION_SPLIT)
    training_token_ids = torch.tensor(tokenizer.encode(training_text), dtype=torch.long)
    validation_token_ids = torch.tensor(tokenizer.encode(validation_text), dtype=torch.long)

    leader_metrics = metrics_snapshot(
        model,
        tokenizer,
        validation_text,
    )
    boundary_before = systematic_boundary_mapping(
        model=model,
        tokenizer=tokenizer,
        validation_token_ids=validation_token_ids,
        prompt_length=40,
        eval_length=80,
        sample_count=args.systematic_sample_count,
        seed=args.seed,
        batch_size=args.systematic_batch_size,
    )

    run = run_test3_intervention(
        base_model=model,
        tokenizer=tokenizer,
        training_token_ids=training_token_ids,
        validation_text=validation_text,
        validation_token_ids=validation_token_ids,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        min_learning_rate_ratio=args.min_learning_rate_ratio,
        label_smoothing=args.label_smoothing,
        recovery_start_index=args.recovery_start_index,
        recovery_loss_weight=args.recovery_loss_weight,
        boundary_indices=boundary_indices,
        consistency_weight=args.consistency_weight,
        systematic_sample_count=args.systematic_sample_count,
        systematic_batch_size=args.systematic_batch_size,
        seed=args.seed,
    )

    summary_table = {
        "leader_before": metric_row(leader_metrics),
        "after_test3": metric_row(run["final_metrics"]),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    artifact_path = args.output_dir / f"v2-test3-boundary-consistency-{stamp}.pt"
    artifact = build_model_artifact(
        model=run["model"],
        tokenizer=tokenizer,
        context_length=model_config["context_length"],
        embedding_dim=model_config["embedding_dim"],
        num_layers=model_config.get("num_layers", 1),
        num_heads=model_config.get("num_heads", 4),
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
            "consistency_weight": args.consistency_weight,
            "boundary_indices": boundary_indices,
            "resume_artifact": str(args.resume_artifact),
            "objective": "balanced_ce_ls_recovery_plus_boundary_consistency",
        },
    )
    torch.save(artifact, artifact_path)

    run.pop("model", None)
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
            "consistency_weight": args.consistency_weight,
            "boundary_indices": boundary_indices,
            "systematic_sample_count": args.systematic_sample_count,
            "systematic_batch_size": args.systematic_batch_size,
            "seed": args.seed,
        },
        "leader_before": leader_metrics,
        "boundary_before": boundary_before,
        "run": run,
        "summary_table": summary_table,
        "artifact_path": str(artifact_path.resolve()),
    }

    json_path = args.output_dir / f"test3-boundary-consistency-{stamp}.json"
    txt_path = args.output_dir / f"test3-boundary-consistency-{stamp}.txt"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(render_report(results))

    print(str(json_path.resolve()))
    print(str(txt_path.resolve()))
    print(str(artifact_path.resolve()))


if __name__ == "__main__":
    main()

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from outreachlm.generate import load_model_and_tokenizer
from outreachlm.objective_intervention_experiment import (
    build_frequency_balanced_weights,
    prompt_logit_alignment,
    rollout_metrics,
    teacher_forcing_vs_free_running,
)
from outreachlm.train import (
    CORPUS_PATH,
    VALIDATION_SPLIT,
    calculate_loss,
    get_learning_rate,
    get_random_batch,
    load_corpus,
    split_corpus,
)


def scheduled_sampling_rate(
    step,
    total_steps,
    min_rate=0.0,
    max_rate=0.4,
):
    if total_steps <= 1:
        return max_rate
    progress = (step - 1) / (total_steps - 1)
    return min_rate + (max_rate - min_rate) * progress


def build_mixed_inputs_with_scheduled_sampling(
    model,
    input_ids,
    sample_rate,
):
    if sample_rate <= 0.0:
        return input_ids

    with torch.no_grad():
        teacher_logits = model(input_ids)
        if isinstance(teacher_logits, tuple):
            teacher_logits = teacher_logits[0]
        predicted_next = torch.argmax(
            teacher_logits[:, :-1, :],
            dim=-1,
        )

    mixed = input_ids.clone()
    mask = (
        torch.rand(
            predicted_next.shape,
            device=input_ids.device,
        )
        < sample_rate
    )
    mixed[:, 1:] = torch.where(
        mask,
        predicted_next,
        input_ids[:, 1:],
    )
    return mixed


def metrics_snapshot(
    model,
    tokenizer,
    validation_text,
):
    return {
        "teacher_free": teacher_forcing_vs_free_running(
            model,
            tokenizer,
            validation_text,
        ),
        "logit_alignment": prompt_logit_alignment(
            model,
            tokenizer,
        ),
        "rollout": rollout_metrics(
            model,
            tokenizer,
            "OutreachLM is",
            max_new_tokens=60,
        ),
    }


def run_balanced_ls_control(
    base_model,
    training_token_ids,
    tokenizer,
    validation_text,
    steps,
    batch_size,
    learning_rate,
    warmup_steps,
    min_learning_rate_ratio,
    label_smoothing,
    seed,
):
    torch.manual_seed(seed)
    model = copy.deepcopy(base_model)
    model.train()
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
    for step in range(1, steps + 1):
        input_ids, target_ids = get_random_batch(
            training_token_ids,
            model.context_length,
            batch_size,
            device,
        )
        logits = model(input_ids)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = calculate_loss(
            logits,
            target_ids,
            loss_function,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
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
        losses.append(float(loss.item()))

    model.eval()
    return {
        "model": model,
        "loss_summary": {
            "first_loss": float(losses[0]) if losses else None,
            "last_loss": float(losses[-1]) if losses else None,
            "mean_loss": float(sum(losses) / len(losses)) if losses else None,
            "min_loss": float(min(losses)) if losses else None,
            "max_loss": float(max(losses)) if losses else None,
        },
        "metrics": metrics_snapshot(model, tokenizer, validation_text),
    }


def run_balanced_ls_with_scheduled_sampling(
    base_model,
    training_token_ids,
    tokenizer,
    validation_text,
    steps,
    batch_size,
    learning_rate,
    warmup_steps,
    min_learning_rate_ratio,
    label_smoothing,
    ss_min_rate,
    ss_max_rate,
    seed,
):
    torch.manual_seed(seed)
    model = copy.deepcopy(base_model)
    model.train()
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
    applied_rates = []
    checkpoint_metrics = []

    for step in range(1, steps + 1):
        input_ids, target_ids = get_random_batch(
            training_token_ids,
            model.context_length,
            batch_size,
            device,
        )
        current_rate = scheduled_sampling_rate(
            step=step,
            total_steps=steps,
            min_rate=ss_min_rate,
            max_rate=ss_max_rate,
        )
        mixed_input_ids = build_mixed_inputs_with_scheduled_sampling(
            model=model,
            input_ids=input_ids,
            sample_rate=current_rate,
        )

        logits = model(mixed_input_ids)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = calculate_loss(
            logits,
            target_ids,
            loss_function,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
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

        losses.append(float(loss.item()))
        applied_rates.append(float(current_rate))

        if step % max(steps // 4, 1) == 0 or step == steps:
            model.eval()
            checkpoint_metrics.append(
                {
                    "step": step,
                    "scheduled_sampling_rate": float(current_rate),
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
            "first_loss": float(losses[0]) if losses else None,
            "last_loss": float(losses[-1]) if losses else None,
            "mean_loss": float(sum(losses) / len(losses)) if losses else None,
            "min_loss": float(min(losses)) if losses else None,
            "max_loss": float(max(losses)) if losses else None,
        },
        "schedule_summary": {
            "min_applied_rate": float(min(applied_rates)) if applied_rates else None,
            "max_applied_rate": float(max(applied_rates)) if applied_rates else None,
            "mean_applied_rate": float(sum(applied_rates) / len(applied_rates))
            if applied_rates
            else None,
        },
        "checkpoint_metrics": checkpoint_metrics,
        "metrics": metrics_snapshot(model, tokenizer, validation_text),
    }


def metric_row(metrics):
    tf = metrics["teacher_free"]
    align = metrics["logit_alignment"]
    roll = metrics["rollout"]
    return {
        "teacher_top1": tf["teacher_top1_accuracy"],
        "free_match": tf["free_match_rate_against_target"],
        "prompt_logit_cosine": align["offdiag_logit_cosine_mean"],
        "rollout_mean_entropy": roll["mean_entropy"],
        "first_repeated_bigram_step": roll["first_repeated_bigram_step"],
        "first_repeated_trigram_step": roll["first_repeated_trigram_step"],
        "first_free_divergence": tf["free_first_divergence_position"],
    }


def render_report(results):
    lines = []
    lines.append("OUTREACHLM SCHEDULED SAMPLING INTERVENTION")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {results['timestamp']}")
    lines.append("")
    lines.append(
        "CONFIG | steps={steps} batch={batch} lr={lr} warmup={warmup} "
        "label_smoothing={ls} ss_rate={ss_min}->{ss_max}".format(
            steps=results["config"]["steps"],
            batch=results["config"]["batch_size"],
            lr=results["config"]["learning_rate"],
            warmup=results["config"]["warmup_steps"],
            ls=results["config"]["label_smoothing"],
            ss_min=results["config"]["scheduled_sampling_min_rate"],
            ss_max=results["config"]["scheduled_sampling_max_rate"],
        )
    )
    lines.append("")

    lines.append("METRIC COMPARISON")
    lines.append("-" * 80)
    header = (
        "condition,teacher_top1,free_match,prompt_logit_cosine,"
        "rollout_mean_entropy,first_repeated_bigram_step,"
        "first_repeated_trigram_step,first_free_divergence"
    )
    lines.append(header)
    for condition in [
        "baseline",
        "balanced_ls_control",
        "balanced_ls_scheduled_sampling",
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
    lines.append("SCHEDULE SUMMARY")
    lines.append("-" * 80)
    ss = results["balanced_ls_scheduled_sampling"]["schedule_summary"]
    lines.append(
        f"min/max/mean applied rate: "
        f"{ss['min_applied_rate']:.6f} / {ss['max_applied_rate']:.6f} / {ss['mean_applied_rate']:.6f}"
    )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run one controlled scheduled-sampling intervention with fixed architecture."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--min-learning-rate-ratio", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--scheduled-sampling-min-rate", type=float, default=0.0)
    parser.add_argument("--scheduled-sampling-max-rate", type=float, default=0.4)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    base_model, tokenizer = load_model_and_tokenizer()
    base_model.eval()

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
        base_model,
        tokenizer,
        validation_text,
    )

    balanced_control = run_balanced_ls_control(
        base_model=base_model,
        training_token_ids=training_token_ids,
        tokenizer=tokenizer,
        validation_text=validation_text,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        min_learning_rate_ratio=args.min_learning_rate_ratio,
        label_smoothing=args.label_smoothing,
        seed=args.seed,
    )

    scheduled = run_balanced_ls_with_scheduled_sampling(
        base_model=base_model,
        training_token_ids=training_token_ids,
        tokenizer=tokenizer,
        validation_text=validation_text,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        min_learning_rate_ratio=args.min_learning_rate_ratio,
        label_smoothing=args.label_smoothing,
        ss_min_rate=args.scheduled_sampling_min_rate,
        ss_max_rate=args.scheduled_sampling_max_rate,
        seed=args.seed,
    )

    summary_table = {
        "baseline": metric_row(baseline_metrics),
        "balanced_ls_control": metric_row(balanced_control["metrics"]),
        "balanced_ls_scheduled_sampling": metric_row(scheduled["metrics"]),
    }

    # Keep model objects out of JSON payload.
    balanced_control.pop("model", None)
    scheduled.pop("model", None)

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "seed": args.seed,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "min_learning_rate_ratio": args.min_learning_rate_ratio,
            "label_smoothing": args.label_smoothing,
            "scheduled_sampling_min_rate": args.scheduled_sampling_min_rate,
            "scheduled_sampling_max_rate": args.scheduled_sampling_max_rate,
        },
        "baseline": baseline_metrics,
        "balanced_ls_control": balanced_control,
        "balanced_ls_scheduled_sampling": scheduled,
        "summary_table": summary_table,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = args.output_dir / f"scheduled-sampling-intervention-{stamp}.json"
    txt_path = args.output_dir / f"scheduled-sampling-intervention-{stamp}.txt"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(render_report(results))

    print(str(json_path.resolve()))
    print(str(txt_path.resolve()))


if __name__ == "__main__":
    main()

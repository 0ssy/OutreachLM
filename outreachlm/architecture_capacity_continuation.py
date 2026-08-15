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
from outreachlm.generate import load_model_and_tokenizer
from outreachlm.objective_intervention_experiment import (
    build_frequency_balanced_weights,
)
from outreachlm.train import (
    CORPUS_PATH,
    VALIDATION_SPLIT,
    build_model_artifact,
    calculate_loss,
    create_model,
    get_learning_rate,
    get_random_batch,
    load_corpus,
    split_corpus,
)


def load_model_from_artifact(artifact_path):
    artifact = torch.load(
        artifact_path,
        map_location="cpu",
        weights_only=False,
    )
    model_config = artifact["model_config"]
    model = create_model(
        vocab_size=model_config["vocab_size"],
        context_length=model_config["context_length"],
        embedding_dim=model_config["embedding_dim"],
        num_layers=model_config.get("num_layers", 1),
        num_heads=model_config.get("num_heads", 4),
    )
    model.load_state_dict(artifact["model_state_dict"])
    return model, model_config


def continue_v2_training(
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
    checkpoints = []

    model.train()
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
            "first_loss": float(losses[0]) if losses else None,
            "last_loss": float(losses[-1]) if losses else None,
            "mean_loss": float(sum(losses) / len(losses)) if losses else None,
            "min_loss": float(min(losses)) if losses else None,
            "max_loss": float(max(losses)) if losses else None,
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
    lines.append("OUTREACHLM V2 CAPACITY CONTINUATION (3000 STEPS)")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {results['timestamp']}")
    lines.append("")
    cfg = results["config"]
    lines.append(
        "CONFIG | continue_steps={steps} batch={batch} lr={lr} warmup={warmup} "
        "label_smoothing={ls} min_lr_ratio={min_lr}".format(
            steps=cfg["continuation_steps"],
            batch=cfg["batch_size"],
            lr=cfg["learning_rate"],
            warmup=cfg["warmup_steps"],
            ls=cfg["label_smoothing"],
            min_lr=cfg["min_learning_rate_ratio"],
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
        "baseline",
        "v2_before_continuation",
        "v2_after_continuation",
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
    ls = results["v2_continuation"]["loss_summary"]
    lines.append(
        "V2 continuation loss first/last/mean: "
        f"{ls['first_loss']:.6f} / {ls['last_loss']:.6f} / {ls['mean_loss']:.6f}"
    )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continue V2 pilot training for 3000 steps with same objective.",
    )
    parser.add_argument(
        "--resume-artifact",
        type=Path,
        default=Path("experiments/architecture-v2-pilot-20260815-133251.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--continuation-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--min-learning-rate-ratio", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
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

    v2_continuation = continue_v2_training(
        model=v2_model,
        tokenizer=tokenizer,
        training_token_ids=training_token_ids,
        validation_text=validation_text,
        steps=args.continuation_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        min_learning_rate_ratio=args.min_learning_rate_ratio,
        label_smoothing=args.label_smoothing,
        seed=args.seed,
    )

    summary_table = {
        "baseline": metric_row(baseline_metrics),
        "v2_before_continuation": metric_row(v2_before_metrics),
        "v2_after_continuation": metric_row(v2_continuation["final_metrics"]),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    continued_artifact_path = (
        args.output_dir / f"architecture-v2-continuation-{stamp}.pt"
    )
    continued_artifact = build_model_artifact(
        model=v2_continuation["model"],
        tokenizer=tokenizer,
        context_length=v2_model_config["context_length"],
        embedding_dim=v2_model_config["embedding_dim"],
        num_layers=v2_model_config.get("num_layers", 1),
        num_heads=v2_model_config.get("num_heads", 4),
        training_config={
            "seed": args.seed,
            "continuation_steps": args.continuation_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "min_learning_rate_ratio": args.min_learning_rate_ratio,
            "label_smoothing": args.label_smoothing,
            "objective": "balanced_ce_plus_label_smoothing",
            "resume_artifact": str(args.resume_artifact),
        },
    )
    torch.save(continued_artifact, continued_artifact_path)

    v2_continuation.pop("model", None)

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "resume_artifact": str(args.resume_artifact.resolve()),
            "continuation_steps": args.continuation_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "min_learning_rate_ratio": args.min_learning_rate_ratio,
            "label_smoothing": args.label_smoothing,
            "seed": args.seed,
        },
        "baseline": baseline_metrics,
        "v2_before_continuation": v2_before_metrics,
        "v2_continuation": v2_continuation,
        "summary_table": summary_table,
        "continued_artifact_path": str(continued_artifact_path.resolve()),
    }

    json_path = args.output_dir / f"architecture-v2-continuation-{stamp}.json"
    txt_path = args.output_dir / f"architecture-v2-continuation-{stamp}.txt"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(render_report(results))

    print(str(json_path.resolve()))
    print(str(txt_path.resolve()))
    print(str(continued_artifact_path.resolve()))


if __name__ == "__main__":
    main()

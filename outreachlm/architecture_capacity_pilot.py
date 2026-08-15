import argparse
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
    build_model_artifact,
    calculate_loss,
    create_model,
    get_learning_rate,
    get_random_batch,
    load_corpus,
    split_corpus,
)


def metrics_snapshot(model, tokenizer, validation_text):
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


def run_v2_capacity_pilot(
    tokenizer,
    training_token_ids,
    validation_text,
    steps,
    batch_size,
    learning_rate,
    warmup_steps,
    min_learning_rate_ratio,
    label_smoothing,
    num_layers,
    num_heads,
    embedding_dim,
    context_length,
    seed,
):
    torch.manual_seed(seed)

    model = create_model(
        vocab_size=tokenizer.vocab_size,
        context_length=context_length,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
    )
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
            context_length,
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
    final_metrics = metrics_snapshot(
        model,
        tokenizer,
        validation_text,
    )

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
        "final_metrics": final_metrics,
    }


def render_report(results):
    lines = []
    lines.append("OUTREACHLM ARCHITECTURE CAPACITY PILOT (V2)")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {results['timestamp']}")
    lines.append("")
    cfg = results["config"]
    lines.append(
        "CONFIG | steps={steps} batch={batch} lr={lr} warmup={warmup} "
        "label_smoothing={ls} | arch: ctx={ctx} emb={emb} layers={layers} heads={heads}".format(
            steps=cfg["steps"],
            batch=cfg["batch_size"],
            lr=cfg["learning_rate"],
            warmup=cfg["warmup_steps"],
            ls=cfg["label_smoothing"],
            ctx=cfg["context_length"],
            emb=cfg["embedding_dim"],
            layers=cfg["num_layers"],
            heads=cfg["num_heads"],
        )
    )
    lines.append("")
    lines.append("METRIC COMPARISON")
    lines.append("-" * 80)
    lines.append(
        "condition,teacher_top1,free_match,prompt_logit_cosine,rollout_mean_entropy,"
        "first_repeated_bigram_step,first_repeated_trigram_step,first_free_divergence"
    )
    for condition in ["baseline", "architecture_v2_pilot"]:
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
    ls = results["architecture_v2_pilot"]["loss_summary"]
    lines.append(
        "V2 loss first/last/mean: "
        f"{ls['first_loss']:.6f} / {ls['last_loss']:.6f} / {ls['mean_loss']:.6f}"
    )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a fixed-objective V2 architecture capacity pilot.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--min-learning-rate-ratio", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    baseline_model, tokenizer = load_model_and_tokenizer()
    baseline_model.eval()

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

    v2_result = run_v2_capacity_pilot(
        tokenizer=tokenizer,
        training_token_ids=training_token_ids,
        validation_text=validation_text,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        min_learning_rate_ratio=args.min_learning_rate_ratio,
        label_smoothing=args.label_smoothing,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        embedding_dim=args.embedding_dim,
        context_length=args.context_length,
        seed=args.seed,
    )

    summary_table = {
        "baseline": metric_row(baseline_metrics),
        "architecture_v2_pilot": metric_row(v2_result["final_metrics"]),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    artifact_path = args.output_dir / f"architecture-v2-pilot-{stamp}.pt"
    model_artifact = build_model_artifact(
        model=v2_result["model"],
        tokenizer=tokenizer,
        context_length=args.context_length,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        training_config={
            "seed": args.seed,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "min_learning_rate_ratio": args.min_learning_rate_ratio,
            "label_smoothing": args.label_smoothing,
            "objective": "balanced_ce_plus_label_smoothing",
        },
    )
    torch.save(model_artifact, artifact_path)

    v2_result.pop("model", None)
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
            "context_length": args.context_length,
            "embedding_dim": args.embedding_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
        },
        "baseline": baseline_metrics,
        "architecture_v2_pilot": v2_result,
        "summary_table": summary_table,
        "model_artifact_path": str(artifact_path.resolve()),
    }

    json_path = args.output_dir / f"architecture-v2-pilot-{stamp}.json"
    txt_path = args.output_dir / f"architecture-v2-pilot-{stamp}.txt"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(render_report(results))

    print(str(json_path.resolve()))
    print(str(txt_path.resolve()))
    print(str(artifact_path.resolve()))


if __name__ == "__main__":
    main()

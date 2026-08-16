import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from outreachlm.architecture_capacity_continuation import load_model_from_artifact
from outreachlm.architecture_capacity_pilot import metric_row, metrics_snapshot
from outreachlm.divergence_window_intervention import build_recovery_mixed_inputs
from outreachlm.generate import (
    TOKENIZER_PATH,
    load_tokenizer_artifact,
    upgrade_legacy_tokenizer_artifact,
)
from outreachlm.objective_intervention_experiment import build_frequency_balanced_weights
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


def train_v3_from_scratch(
    tokenizer,
    training_token_ids,
    validation_text,
    output_dir,
    steps,
    checkpoint_interval,
    batch_size,
    learning_rate,
    warmup_steps,
    min_learning_rate_ratio,
    label_smoothing,
    recovery_start_index,
    recovery_loss_weight,
    embedding_dim,
    num_layers,
    num_heads,
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

    output_dir.mkdir(parents=True, exist_ok=True)
    losses_total = []
    losses_teacher = []
    losses_recovery = []
    checkpoint_records = []

    model.train()
    for step in range(1, steps + 1):
        input_ids, target_ids = get_random_batch(
            training_token_ids,
            context_length,
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

        total_loss = teacher_loss + (recovery_loss_weight * recovery_loss)

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

        losses_total.append(float(total_loss.item()))
        losses_teacher.append(float(teacher_loss.item()))
        losses_recovery.append(float(recovery_loss.item()))

        if step % checkpoint_interval == 0 or step == steps:
            model.eval()
            checkpoint_metrics = metrics_snapshot(
                model,
                tokenizer,
                validation_text,
            )
            checkpoint_record = {
                "step": step,
                "learning_rate": float(current_lr),
                "metrics": checkpoint_metrics,
                "loss_snapshot": {
                    "total": float(total_loss.item()),
                    "teacher": float(teacher_loss.item()),
                    "recovery": float(recovery_loss.item()),
                },
            }
            checkpoint_records.append(checkpoint_record)

            checkpoint_artifact = build_model_artifact(
                model=model,
                tokenizer=tokenizer,
                context_length=context_length,
                embedding_dim=embedding_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                training_config={
                    "seed": seed,
                    "steps_completed": step,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "warmup_steps": warmup_steps,
                    "min_learning_rate_ratio": min_learning_rate_ratio,
                    "label_smoothing": label_smoothing,
                    "recovery_start_index": recovery_start_index,
                    "recovery_loss_weight": recovery_loss_weight,
                    "objective": "balanced_ce_ls_plus_recovery",
                    "arch": {
                        "context_length": context_length,
                        "embedding_dim": embedding_dim,
                        "num_layers": num_layers,
                        "num_heads": num_heads,
                        "vocab_size": tokenizer.vocab_size,
                    },
                },
            )
            checkpoint_path = output_dir / f"v3-checkpoint-step-{step:05d}.pt"
            torch.save(checkpoint_artifact, checkpoint_path)
            checkpoint_record["checkpoint_path"] = str(checkpoint_path.resolve())
            model.train()

    model.eval()
    final_metrics = metrics_snapshot(
        model,
        tokenizer,
        validation_text,
    )
    final_artifact = build_model_artifact(
        model=model,
        tokenizer=tokenizer,
        context_length=context_length,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        training_config={
            "seed": seed,
            "steps_completed": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "min_learning_rate_ratio": min_learning_rate_ratio,
            "label_smoothing": label_smoothing,
            "recovery_start_index": recovery_start_index,
            "recovery_loss_weight": recovery_loss_weight,
            "objective": "balanced_ce_ls_plus_recovery",
            "arch": {
                "context_length": context_length,
                "embedding_dim": embedding_dim,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "vocab_size": tokenizer.vocab_size,
            },
        },
    )

    return {
        "model": model,
        "final_artifact": final_artifact,
        "loss_summary": {
            "first_total_loss": float(losses_total[0]) if losses_total else None,
            "last_total_loss": float(losses_total[-1]) if losses_total else None,
            "mean_total_loss": float(sum(losses_total) / len(losses_total))
            if losses_total
            else None,
            "first_teacher_loss": float(losses_teacher[0]) if losses_teacher else None,
            "last_teacher_loss": float(losses_teacher[-1]) if losses_teacher else None,
            "first_recovery_loss": float(losses_recovery[0]) if losses_recovery else None,
            "last_recovery_loss": float(losses_recovery[-1]) if losses_recovery else None,
        },
        "checkpoint_records": checkpoint_records,
        "final_metrics": final_metrics,
    }


def render_report(results):
    lines = []
    lines.append("OUTREACHLM V3 TRAINING + LEADER COMPARISON")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {results['timestamp']}")
    lines.append("")
    cfg = results["config"]
    lines.append(
        "CONFIG | steps={steps} ckpt_int={ckpt} batch={batch} lr={lr} warmup={warmup} "
        "min_lr_ratio={min_lr} ls={ls} recovery_w={rw}".format(
            steps=cfg["steps"],
            ckpt=cfg["checkpoint_interval"],
            batch=cfg["batch_size"],
            lr=cfg["learning_rate"],
            warmup=cfg["warmup_steps"],
            min_lr=cfg["min_learning_rate_ratio"],
            ls=cfg["label_smoothing"],
            rw=cfg["recovery_loss_weight"],
        )
    )
    lines.append(
        "ARCH | vocab={vocab} context={ctx} emb={emb} layers={layers} heads={heads}".format(
            vocab=cfg["vocab_size"],
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
    for condition in ["leader", "v3"]:
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
    ls = results["v3_training"]["loss_summary"]
    lines.append(
        "V3 loss first/last/mean: "
        f"{ls['first_total_loss']:.6f} / {ls['last_total_loss']:.6f} / {ls['mean_total_loss']:.6f}"
    )
    lines.append("Checkpoint metric snapshots:")
    for rec in results["v3_training"]["checkpoint_records"]:
        m = rec["metrics"]["teacher_free"]
        lines.append(
            f"  step={rec['step']:5d} "
            f"teacher_top1={m['teacher_top1_accuracy']:.6f} "
            f"free_match={m['free_match_rate_against_target']:.6f} "
            f"path={rec['checkpoint_path']}"
        )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train V3 from scratch and compare against current leader."
    )
    parser.add_argument(
        "--leader-artifact",
        type=Path,
        default=Path("experiments/v2-divergence-intervention-20260816-113809.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=4500)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--min-learning-rate-ratio", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--recovery-start-index", type=int, default=40)
    parser.add_argument("--recovery-loss-weight", type=float, default=2.0)

    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = load_tokenizer_artifact(TOKENIZER_PATH)
    if tokenizer is None:
        tokenizer = upgrade_legacy_tokenizer_artifact(TOKENIZER_PATH)
    if tokenizer.vocab_size != 490:
        raise RuntimeError(f"Expected vocab size 490, got {tokenizer.vocab_size}")

    text = load_corpus(CORPUS_PATH)
    training_text, validation_text = split_corpus(text, VALIDATION_SPLIT)
    training_token_ids = torch.tensor(tokenizer.encode(training_text), dtype=torch.long)

    leader_model, _ = load_model_from_artifact(args.leader_artifact)
    leader_model.eval()
    leader_metrics = metrics_snapshot(
        leader_model,
        tokenizer,
        validation_text,
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_dir / f"v3-training-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    v3_training = train_v3_from_scratch(
        tokenizer=tokenizer,
        training_token_ids=training_token_ids,
        validation_text=validation_text,
        output_dir=run_dir,
        steps=args.steps,
        checkpoint_interval=args.checkpoint_interval,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        min_learning_rate_ratio=args.min_learning_rate_ratio,
        label_smoothing=args.label_smoothing,
        recovery_start_index=args.recovery_start_index,
        recovery_loss_weight=args.recovery_loss_weight,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        context_length=args.context_length,
        seed=args.seed,
    )

    final_model_path = run_dir / "v3-final.pt"
    torch.save(v3_training["final_artifact"], final_model_path)

    summary_table = {
        "leader": metric_row(leader_metrics),
        "v3": metric_row(v3_training["final_metrics"]),
    }

    v3_training.pop("model", None)
    v3_training.pop("final_artifact", None)
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "leader_artifact": str(args.leader_artifact.resolve()),
            "steps": args.steps,
            "checkpoint_interval": args.checkpoint_interval,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "min_learning_rate_ratio": args.min_learning_rate_ratio,
            "label_smoothing": args.label_smoothing,
            "recovery_start_index": args.recovery_start_index,
            "recovery_loss_weight": args.recovery_loss_weight,
            "context_length": args.context_length,
            "embedding_dim": args.embedding_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "vocab_size": tokenizer.vocab_size,
            "seed": args.seed,
        },
        "leader_metrics": leader_metrics,
        "v3_training": v3_training,
        "summary_table": summary_table,
        "run_dir": str(run_dir.resolve()),
        "final_model_path": str(final_model_path.resolve()),
    }

    json_path = run_dir / "v3-train-and-compare.json"
    txt_path = run_dir / "v3-train-and-compare.txt"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(render_report(results))

    print(str(json_path.resolve()))
    print(str(txt_path.resolve()))
    print(str(final_model_path.resolve()))


if __name__ == "__main__":
    main()

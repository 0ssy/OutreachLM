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
    get_learning_rate,
    get_random_batch,
    load_corpus,
    split_corpus,
)


def render_report(results):
    lines = []
    lines.append("OUTREACHLM V3 CONTINUATION TO 4500 + LEADER COMPARISON")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {results['timestamp']}")
    lines.append("")
    cfg = results["config"]
    lines.append(
        "CONFIG | resume_step={resume} continue_steps={cont} target_total={target} "
        "batch={batch} lr={lr} warmup={warmup} min_lr={minlr} ls={ls} recovery_w={rw}".format(
            resume=cfg["resume_step"],
            cont=cfg["continuation_steps"],
            target=cfg["target_total_steps"],
            batch=cfg["batch_size"],
            lr=cfg["learning_rate"],
            warmup=cfg["warmup_steps"],
            minlr=cfg["min_learning_rate_ratio"],
            ls=cfg["label_smoothing"],
            rw=cfg["recovery_loss_weight"],
        )
    )
    lines.append("")
    lines.append(
        "condition,teacher_top1,free_match,prompt_logit_cosine,rollout_mean_entropy,"
        "first_repeated_bigram_step,first_repeated_trigram_step,first_free_divergence"
    )
    for condition in ["leader", "v3_1500", "v3_3000", "v3_4500"]:
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
    ls = results["continuation"]["loss_summary"]
    lines.append(
        "Continuation total loss first/last/mean: "
        f"{ls['first_total_loss']:.6f} / {ls['last_total_loss']:.6f} / {ls['mean_total_loss']:.6f}"
    )
    lines.append("Saved checkpoints:")
    for item in results["saved_checkpoints"]:
        lines.append(f"  step={item['step']}: {item['path']}")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continue V3 from 1500 to 4500 and compare against leader."
    )
    parser.add_argument(
        "--leader-artifact",
        type=Path,
        default=Path("experiments/v2-divergence-intervention-20260816-113809.pt"),
    )
    parser.add_argument(
        "--resume-artifact",
        type=Path,
        default=Path("experiments/v3-training-20260816-123145/v3-checkpoint-step-01500.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-step", type=int, default=1500)
    parser.add_argument("--continuation-steps", type=int, default=3000)
    parser.add_argument("--target-total-steps", type=int, default=4500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--min-learning-rate-ratio", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--recovery-start-index", type=int, default=40)
    parser.add_argument("--recovery-loss-weight", type=float, default=2.0)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.resume_step + args.continuation_steps != args.target_total_steps:
        raise ValueError(
            "resume_step + continuation_steps must equal target_total_steps."
        )

    tokenizer = load_tokenizer_artifact(TOKENIZER_PATH)
    if tokenizer is None:
        tokenizer = upgrade_legacy_tokenizer_artifact(TOKENIZER_PATH)

    text = load_corpus(CORPUS_PATH)
    training_text, validation_text = split_corpus(text, VALIDATION_SPLIT)
    training_token_ids = torch.tensor(tokenizer.encode(training_text), dtype=torch.long)

    leader_model, _ = load_model_from_artifact(args.leader_artifact)
    leader_model.eval()
    leader_metrics = metrics_snapshot(leader_model, tokenizer, validation_text)

    v3_model, model_config = load_model_from_artifact(args.resume_artifact)
    v3_model.eval()
    v3_1500_metrics = metrics_snapshot(v3_model, tokenizer, validation_text)

    device = next(v3_model.parameters()).device
    optimizer = torch.optim.AdamW(v3_model.parameters(), lr=args.learning_rate)
    class_weights, _ = build_frequency_balanced_weights(
        training_token_ids=training_token_ids,
        vocab_size=tokenizer.vocab_size,
        device=device,
    )
    loss_function = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )

    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_dir / f"v3-continue-{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    losses_total = []
    losses_teacher = []
    losses_recovery = []
    saved_checkpoints = []
    v3_3000_metrics = None
    v3_4500_metrics = None

    v3_model.train()
    for local_step in range(1, args.continuation_steps + 1):
        global_step = args.resume_step + local_step

        input_ids, target_ids = get_random_batch(
            training_token_ids,
            v3_model.context_length,
            args.batch_size,
            device,
        )

        teacher_logits = v3_model(input_ids)
        if isinstance(teacher_logits, tuple):
            teacher_logits = teacher_logits[0]
        teacher_loss = calculate_loss(teacher_logits, target_ids, loss_function)

        mixed_input_ids = build_recovery_mixed_inputs(
            input_ids=input_ids,
            teacher_logits=teacher_logits.detach(),
            recovery_start_index=args.recovery_start_index,
        )
        recovery_logits = v3_model(mixed_input_ids)
        if isinstance(recovery_logits, tuple):
            recovery_logits = recovery_logits[0]
        if args.recovery_start_index >= target_ids.shape[1]:
            recovery_loss = torch.zeros((), device=device, dtype=teacher_loss.dtype)
        else:
            recovery_loss = calculate_loss(
                recovery_logits[:, args.recovery_start_index:, :],
                target_ids[:, args.recovery_start_index:],
                loss_function,
            )

        total_loss = teacher_loss + args.recovery_loss_weight * recovery_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        lr = get_learning_rate(
            step=local_step - 1,
            max_steps=args.continuation_steps,
            base_learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            min_learning_rate_ratio=args.min_learning_rate_ratio,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()

        losses_total.append(float(total_loss.item()))
        losses_teacher.append(float(teacher_loss.item()))
        losses_recovery.append(float(recovery_loss.item()))

        if global_step in (3000, 4500):
            v3_model.eval()
            metrics = metrics_snapshot(v3_model, tokenizer, validation_text)
            artifact = build_model_artifact(
                model=v3_model,
                tokenizer=tokenizer,
                context_length=model_config["context_length"],
                embedding_dim=model_config["embedding_dim"],
                num_layers=model_config.get("num_layers", 1),
                num_heads=model_config.get("num_heads", 4),
                training_config={
                    "seed": args.seed,
                    "step": global_step,
                    "resume_artifact": str(args.resume_artifact),
                    "objective": "balanced_ce_ls_plus_recovery",
                    "label_smoothing": args.label_smoothing,
                    "recovery_start_index": args.recovery_start_index,
                    "recovery_loss_weight": args.recovery_loss_weight,
                    "learning_rate": args.learning_rate,
                    "warmup_steps": args.warmup_steps,
                    "min_learning_rate_ratio": args.min_learning_rate_ratio,
                    "batch_size": args.batch_size,
                },
            )
            checkpoint_path = run_dir / f"v3-step-{global_step}.pt"
            torch.save(artifact, checkpoint_path)
            saved_checkpoints.append(
                {
                    "step": global_step,
                    "path": str(checkpoint_path.resolve()),
                }
            )
            if global_step == 3000:
                v3_3000_metrics = metrics
            else:
                v3_4500_metrics = metrics
            v3_model.train()

    if v3_3000_metrics is None or v3_4500_metrics is None:
        raise RuntimeError("Expected metrics/checkpoints for steps 3000 and 4500.")

    summary_table = {
        "leader": metric_row(leader_metrics),
        "v3_1500": metric_row(v3_1500_metrics),
        "v3_3000": metric_row(v3_3000_metrics),
        "v3_4500": metric_row(v3_4500_metrics),
    }

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "leader_artifact": str(args.leader_artifact.resolve()),
            "resume_artifact": str(args.resume_artifact.resolve()),
            "seed": args.seed,
            "resume_step": args.resume_step,
            "continuation_steps": args.continuation_steps,
            "target_total_steps": args.target_total_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "min_learning_rate_ratio": args.min_learning_rate_ratio,
            "label_smoothing": args.label_smoothing,
            "recovery_start_index": args.recovery_start_index,
            "recovery_loss_weight": args.recovery_loss_weight,
        },
        "leader_metrics": leader_metrics,
        "v3_1500_metrics": v3_1500_metrics,
        "v3_3000_metrics": v3_3000_metrics,
        "v3_4500_metrics": v3_4500_metrics,
        "continuation": {
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
        },
        "summary_table": summary_table,
        "saved_checkpoints": saved_checkpoints,
        "run_dir": str(run_dir.resolve()),
    }

    json_path = run_dir / "v3-continue-and-compare.json"
    txt_path = run_dir / "v3-continue-and-compare.txt"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(render_report(results))

    print(str(json_path.resolve()))
    print(str(txt_path.resolve()))
    for item in saved_checkpoints:
        print(item["path"])


if __name__ == "__main__":
    main()

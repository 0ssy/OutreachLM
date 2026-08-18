import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from outreachlm.architecture_capacity_pilot import metric_row, metrics_snapshot
from outreachlm.divergence_window_intervention import build_recovery_mixed_inputs
from outreachlm.generate import (
    TOKENIZER_PATH,
    load_tokenizer_artifact,
)
from outreachlm.tokenizer import CharacterTokenizer
from outreachlm.train import (
    CORPUS_PATH,
    VALIDATION_SPLIT,
    calculate_loss,
    get_learning_rate,
    get_random_batch,
    load_corpus,
    split_corpus,
)
from outreachlm.v4_model import OutreachV4Model


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def save_tokenizer_config(tokenizer, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "vocab_size": tokenizer.vocab_size,
        "tokens": tokenizer.tokens,
        "pad_token": tokenizer.pad_token,
        "unk_token": tokenizer.unk_token,
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def load_or_build_tokenizer(training_text):
    tokenizer = load_tokenizer_artifact(TOKENIZER_PATH)
    if tokenizer is not None:
        return tokenizer
    return CharacterTokenizer(training_text)


def build_frequency_balanced_weights(token_ids, vocab_size, device):
    counts = torch.bincount(token_ids, minlength=vocab_size).to(torch.float32)
    total = counts.sum().clamp(min=1.0)
    probs = counts / total

    weights = torch.zeros_like(probs)
    observed = probs > 0
    weights[observed] = torch.pow(probs[observed], -0.5)
    observed_mean = weights[observed].mean().clamp(min=1e-12)
    weights[observed] = weights[observed] / observed_mean
    weights = torch.clamp(weights, min=0.25, max=4.0)
    return weights.to(device)


def create_v4_model(vocab_size, context_length, embedding_dim, num_layers, num_heads, ffn_dim):
    model = OutreachV4Model(
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
    )
    return model.to(DEVICE)


def build_v4_artifact(
    model,
    tokenizer,
    context_length,
    embedding_dim,
    num_layers,
    num_heads,
    ffn_dim,
    training_config,
):
    return {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "vocab_size": tokenizer.vocab_size,
            "context_length": context_length,
            "embedding_dim": embedding_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "ffn_dim": ffn_dim,
            "model_type": "outreachlm_v4",
        },
        "training_config": training_config,
        "tokenizer_config": {
            "tokens": tokenizer.tokens,
            "pad_token": tokenizer.pad_token,
            "unk_token": tokenizer.unk_token,
        },
    }


def smoke_test(model):
    model.train()
    input_ids = torch.randint(
        low=0,
        high=model.vocab_size,
        size=(2, 32),
        device=next(model.parameters()).device,
    )
    target_ids = torch.randint(
        low=0,
        high=model.vocab_size,
        size=(2, 32),
        device=next(model.parameters()).device,
    )
    loss_function = nn.CrossEntropyLoss()
    logits = model(input_ids)
    loss = calculate_loss(logits, target_ids, loss_function)
    loss.backward()
    gradients_finite = True
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            gradients_finite = False
            break
    return {
        "forward_shape": tuple(logits.shape),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "gradients_finite": gradients_finite,
        "parameter_count": int(model.parameter_count),
    }


def format_duration(seconds):
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def build_post_error_rollout_inputs(
    model,
    input_ids,
    teacher_logits,
    post_error_start_index,
    rollout_steps,
):
    if post_error_start_index < 1:
        raise ValueError("--post-error-start-index must be >= 1.")

    sequence_length = input_ids.shape[1]
    if post_error_start_index >= sequence_length:
        return input_ids.clone()

    mixed = input_ids.clone()
    with torch.no_grad():
        seed_logits = teacher_logits[:, post_error_start_index - 1, :]
        topk_count = min(2, seed_logits.shape[-1])
        topk_tokens = torch.topk(seed_logits, k=topk_count, dim=-1).indices
        top1_tokens = topk_tokens[:, 0]
        gold_tokens = input_ids[:, post_error_start_index]
        if topk_count > 1:
            seed_tokens = torch.where(
                top1_tokens == gold_tokens,
                topk_tokens[:, 1],
                top1_tokens,
            )
        else:
            seed_tokens = top1_tokens
        mixed[:, post_error_start_index] = seed_tokens

        if rollout_steps > 1:
            max_rollout_position = min(
                sequence_length - 1,
                post_error_start_index + rollout_steps - 1,
            )
            for token_position in range(
                post_error_start_index + 1,
                max_rollout_position + 1,
            ):
                prefix_logits = model(mixed[:, :token_position])
                next_token = torch.argmax(prefix_logits[:, -1, :], dim=-1)
                mixed[:, token_position] = next_token

    return mixed


def post_error_recovery_loss(
    model,
    loss_function,
    input_ids,
    target_ids,
    teacher_logits,
    post_error_start_index,
    rollout_steps,
    loss_window,
):
    if post_error_start_index >= target_ids.shape[1]:
        return torch.zeros((), device=target_ids.device, dtype=teacher_logits.dtype)

    mixed = build_post_error_rollout_inputs(
        model=model,
        input_ids=input_ids,
        teacher_logits=teacher_logits,
        post_error_start_index=post_error_start_index,
        rollout_steps=rollout_steps,
    )
    logits = model(mixed)

    start = post_error_start_index
    if loss_window <= 0:
        end = target_ids.shape[1]
    else:
        end = min(target_ids.shape[1], start + loss_window)

    return calculate_loss(
        logits=logits[:, start:end, :],
        target_ids=target_ids[:, start:end],
        loss_function=loss_function,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train OutreachLM V4 from scratch."
    )
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments") / "v4-training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=4500)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--min-learning-rate-ratio", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--recovery-start-index", type=int, default=40)
    parser.add_argument("--recovery-loss-weight", type=float, default=2.0)
    parser.add_argument("--post-error-loss-weight", type=float, default=0.0)
    parser.add_argument("--post-error-start-index", type=int, default=40)
    parser.add_argument("--post-error-rollout-steps", type=int, default=8)
    parser.add_argument("--post-error-loss-window", type=int, default=32)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=684)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--best-min-delta", type=float, default=1e-6)

    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.eval_interval <= 0:
        raise ValueError("--eval-interval must be positive.")
    if args.checkpoint_interval <= 0:
        raise ValueError("--checkpoint-interval must be positive.")
    if args.log_interval <= 0:
        raise ValueError("--log-interval must be positive.")
    if args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience must be zero or positive.")
    if args.post_error_loss_weight < 0:
        raise ValueError("--post-error-loss-weight must be zero or positive.")
    if args.post_error_start_index < 1:
        raise ValueError("--post-error-start-index must be >= 1.")
    if args.post_error_rollout_steps < 1:
        raise ValueError("--post-error-rollout-steps must be >= 1.")
    if args.post_error_loss_window == 0:
        raise ValueError("--post-error-loss-window must be > 0, or negative for full suffix.")

    torch.manual_seed(args.seed)

    text = load_corpus(args.corpus)
    training_text, validation_text = split_corpus(text, VALIDATION_SPLIT)

    tokenizer = load_or_build_tokenizer(training_text)
    if tokenizer.vocab_size != 490:
        raise RuntimeError(
            f"Expected tokenizer vocab size 490, got {tokenizer.vocab_size}."
        )

    model = create_v4_model(
        vocab_size=tokenizer.vocab_size,
        context_length=args.context_length,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
    )

    if args.smoke_test:
        result = smoke_test(model)
        print(f"forward shape: {result['forward_shape']}")
        print(f"loss finite:   {result['loss_finite']}")
        print(f"gradients:     {result['gradients_finite']}")
        print(f"parameters:    {result['parameter_count']}")
        return

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "timestamp": datetime.now().isoformat(),
        "seed": args.seed,
        "corpus": str(args.corpus),
        "steps": args.steps,
        "eval_interval": args.eval_interval,
        "checkpoint_interval": args.checkpoint_interval,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "min_learning_rate_ratio": args.min_learning_rate_ratio,
        "label_smoothing": args.label_smoothing,
        "recovery_start_index": args.recovery_start_index,
        "recovery_loss_weight": args.recovery_loss_weight,
        "post_error_loss_weight": args.post_error_loss_weight,
        "post_error_start_index": args.post_error_start_index,
        "post_error_rollout_steps": args.post_error_rollout_steps,
        "post_error_loss_window": args.post_error_loss_window,
        "grad_clip": args.grad_clip,
        "context_length": args.context_length,
        "embedding_dim": args.embedding_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "ffn_dim": args.ffn_dim,
        "early_stop_patience": args.early_stop_patience,
        "best_min_delta": args.best_min_delta,
        "vocab_size": tokenizer.vocab_size,
        "device": str(DEVICE),
    }

    with open(output_dir / "v4_config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    save_tokenizer_config(
        tokenizer=tokenizer,
        path=output_dir / "tokenizer.json",
    )

    training_token_ids = torch.tensor(
        tokenizer.encode(training_text),
        dtype=torch.long
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )
    class_weights = build_frequency_balanced_weights(
        token_ids=training_token_ids,
        vocab_size=tokenizer.vocab_size,
        device=DEVICE,
    )
    loss_function = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )

    loss_total_values = []
    loss_teacher_values = []
    loss_recovery_values = []
    loss_post_error_values = []
    checkpoints = []
    best_rollout_path = output_dir / "v4-best-rollout.pt"
    best_rollout_step = None
    best_rollout_free_match = float("-inf")
    best_rollout_metrics_row = None
    degradation_streak = 0
    stop_reason = None

    start_time = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        input_ids, target_ids = get_random_batch(
            token_ids=training_token_ids,
            context_length=args.context_length,
            batch_size=args.batch_size,
            device=DEVICE,
        )

        teacher_logits = model(input_ids)
        teacher_loss = calculate_loss(
            logits=teacher_logits,
            target_ids=target_ids,
            loss_function=loss_function,
        )

        mixed_input_ids = build_recovery_mixed_inputs(
            input_ids=input_ids,
            teacher_logits=teacher_logits.detach(),
            recovery_start_index=args.recovery_start_index,
        )
        recovery_logits = model(mixed_input_ids)
        if args.recovery_start_index >= target_ids.shape[1]:
            recovery_loss = torch.zeros(
                (),
                device=DEVICE,
                dtype=teacher_loss.dtype,
            )
        else:
            recovery_loss = calculate_loss(
                logits=recovery_logits[:, args.recovery_start_index:, :],
                target_ids=target_ids[:, args.recovery_start_index:],
                loss_function=loss_function,
            )

        if args.post_error_loss_weight > 0:
            post_error_loss = post_error_recovery_loss(
                model=model,
                loss_function=loss_function,
                input_ids=input_ids,
                target_ids=target_ids,
                teacher_logits=teacher_logits.detach(),
                post_error_start_index=args.post_error_start_index,
                rollout_steps=args.post_error_rollout_steps,
                loss_window=args.post_error_loss_window,
            )
        else:
            post_error_loss = torch.zeros(
                (),
                device=DEVICE,
                dtype=teacher_loss.dtype,
            )

        total_loss = (
            teacher_loss
            + (args.recovery_loss_weight * recovery_loss)
            + (args.post_error_loss_weight * post_error_loss)
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=args.grad_clip,
        )

        current_lr = get_learning_rate(
            step=step - 1,
            max_steps=args.steps,
            base_learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            min_learning_rate_ratio=args.min_learning_rate_ratio,
        )
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.step()

        loss_total_values.append(float(total_loss.item()))
        loss_teacher_values.append(float(teacher_loss.item()))
        loss_recovery_values.append(float(recovery_loss.item()))
        loss_post_error_values.append(float(post_error_loss.item()))

        if step % args.log_interval == 0 or step == 1 or step == args.steps:
            elapsed = max(1e-6, time.time() - start_time)
            steps_per_second = step / elapsed
            remaining_steps = args.steps - step
            eta_seconds = remaining_steps / max(steps_per_second, 1e-6)
            print(
                "step "
                f"{step}/{args.steps} "
                f"lr={current_lr:.6f} "
                f"total={float(total_loss.item()):.4f} "
                f"teacher={float(teacher_loss.item()):.4f} "
                f"recovery={float(recovery_loss.item()):.4f} "
                f"post_error={float(post_error_loss.item()):.4f} "
                f"eta={format_duration(eta_seconds)}",
                flush=True,
            )

        should_evaluate = (step % args.eval_interval == 0) or (step == args.steps)
        if should_evaluate:
            model.eval()
            snapshot = metrics_snapshot(
                model=model,
                tokenizer=tokenizer,
                validation_text=validation_text,
            )
            snapshot_row = metric_row(snapshot)
            free_match = float(snapshot_row["free_match"])
            improved = free_match > (best_rollout_free_match + args.best_min_delta)
            if improved:
                best_rollout_free_match = free_match
                best_rollout_step = step
                best_rollout_metrics_row = snapshot_row
                degradation_streak = 0
            elif free_match < (best_rollout_free_match - args.best_min_delta):
                degradation_streak += 1
            else:
                degradation_streak = 0

            print(
                f"eval step {step}: "
                f"free_match={snapshot_row['free_match']:.4f} "
                f"teacher_top1={snapshot_row['teacher_top1']:.4f} "
                f"first_divergence={snapshot_row['first_free_divergence']} "
                f"first_repeat_tri={snapshot_row['first_repeated_trigram_step']}",
                flush=True,
            )
            artifact = build_v4_artifact(
                model=model,
                tokenizer=tokenizer,
                context_length=args.context_length,
                embedding_dim=args.embedding_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                ffn_dim=args.ffn_dim,
                training_config=config,
            )
            checkpoint_path = None
            if step % args.checkpoint_interval == 0 or step == args.steps:
                checkpoint_path = output_dir / f"v4-checkpoint-step-{step:05d}.pt"
                torch.save(artifact, checkpoint_path)

            if improved:
                torch.save(artifact, best_rollout_path)
                print(
                    f"new best rollout checkpoint at step {step}: "
                    f"free_match={snapshot_row['free_match']:.4f}",
                    flush=True,
                )

            checkpoints.append(
                {
                    "step": step,
                    "learning_rate": current_lr,
                    "total_loss": float(total_loss.item()),
                    "teacher_loss": float(teacher_loss.item()),
                    "recovery_loss": float(recovery_loss.item()),
                    "post_error_loss": float(post_error_loss.item()),
                    "checkpoint_path": str(checkpoint_path.resolve()) if checkpoint_path is not None else None,
                    "best_rollout_checkpoint_updated": improved,
                    "degradation_streak": degradation_streak,
                    "metrics": snapshot,
                    "metrics_row": snapshot_row,
                }
            )
            if (
                args.early_stop_patience > 0
                and degradation_streak >= args.early_stop_patience
                and step < args.steps
            ):
                stop_reason = (
                    "free_match degraded for "
                    f"{degradation_streak} evaluations after best step {best_rollout_step}"
                )
                print(
                    "early stop triggered: "
                    f"{stop_reason}",
                    flush=True,
                )
                break
            model.train()

    if best_rollout_step is None:
        raise RuntimeError("No evaluation completed; best rollout checkpoint was not created.")

    final_model_path = output_dir / "v4-final.pt"
    shutil.copyfile(best_rollout_path, final_model_path)

    summary = {
        "loss_summary": {
            "first_total_loss": float(loss_total_values[0]) if loss_total_values else None,
            "last_total_loss": float(loss_total_values[-1]) if loss_total_values else None,
            "mean_total_loss": float(sum(loss_total_values) / len(loss_total_values))
            if loss_total_values
            else None,
            "first_teacher_loss": float(loss_teacher_values[0]) if loss_teacher_values else None,
            "last_teacher_loss": float(loss_teacher_values[-1]) if loss_teacher_values else None,
            "first_recovery_loss": float(loss_recovery_values[0]) if loss_recovery_values else None,
            "last_recovery_loss": float(loss_recovery_values[-1]) if loss_recovery_values else None,
            "first_post_error_loss": float(loss_post_error_values[0]) if loss_post_error_values else None,
            "last_post_error_loss": float(loss_post_error_values[-1]) if loss_post_error_values else None,
        },
        "selected_final_model": {
            "source": "best_rollout_checkpoint",
            "step": best_rollout_step,
            "free_match": best_rollout_free_match,
            "metrics_row": best_rollout_metrics_row,
            "best_rollout_checkpoint_path": str(best_rollout_path.resolve()),
            "final_model_path": str(final_model_path.resolve()),
        },
        "stopped_early": stop_reason is not None,
        "stop_reason": stop_reason,
        "checkpoints": checkpoints,
    }

    with open(output_dir / "v4_training_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"V4 checkpoints saved in: {output_dir}")
    print(
        f"Selected best rollout checkpoint: step {best_rollout_step} "
        f"(free_match={best_rollout_free_match:.4f})"
    )
    print(f"Final model: {final_model_path.resolve()}")


if __name__ == "__main__":
    main()

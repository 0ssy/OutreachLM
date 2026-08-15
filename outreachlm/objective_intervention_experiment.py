import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from outreachlm.generate import load_model_and_tokenizer
from outreachlm.train import (
    CORPUS_PATH,
    VALIDATION_SPLIT,
    calculate_loss,
    get_learning_rate,
    get_random_batch,
    load_corpus,
    split_corpus,
)


PROMPTS = [
    "Machine",
    "Machine learning",
    "Machine learning allows",
    "Machine learning allows computers",
    "Machine learning allows computers to",
    "The purpose of a transformer",
    "A computer system can",
    "OutreachLM is",
]


def token_text(tokenizer, token_id):
    token = tokenizer.id_to_token.get(token_id, "")
    return token.replace("\n", "\\n")


def safe_kl(p, q):
    p_safe = p.clamp(min=1e-12)
    q_safe = q.clamp(min=1e-12)
    return float(torch.sum(p_safe * torch.log(p_safe / q_safe)).item())


def encode_for_model(model, tokenizer, prompt):
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        token_ids = [tokenizer.token_to_id[tokenizer.unk_token]]
    context = token_ids[-model.context_length :]
    return torch.tensor(
        [context],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )


def prompt_logit_alignment(model, tokenizer):
    prompt_logits = {}
    with torch.no_grad():
        for prompt in PROMPTS:
            input_ids = encode_for_model(model, tokenizer, prompt)
            logits = model(input_ids)
            if isinstance(logits, tuple):
                logits = logits[0]
            prompt_logits[prompt] = logits[0, -1, :].detach().cpu()

    rows = []
    offdiag = []
    for left in PROMPTS:
        for right in PROMPTS:
            logits_left = prompt_logits[left]
            logits_right = prompt_logits[right]
            cosine = float(
                torch.cosine_similarity(
                    logits_left.unsqueeze(0),
                    logits_right.unsqueeze(0),
                    dim=-1,
                ).item()
            )
            probs_left = torch.softmax(logits_left, dim=-1)
            probs_right = torch.softmax(logits_right, dim=-1)
            kl_sym = 0.5 * (safe_kl(probs_left, probs_right) + safe_kl(probs_right, probs_left))
            rows.append(
                {
                    "prompt_a": left,
                    "prompt_b": right,
                    "logit_cosine": cosine,
                    "kl_symmetric": kl_sym,
                }
            )
            if left != right:
                offdiag.append(cosine)

    return {
        "pairwise": rows,
        "offdiag_logit_cosine_mean": float(sum(offdiag) / len(offdiag)),
        "offdiag_logit_cosine_min": float(min(offdiag)),
        "offdiag_logit_cosine_max": float(max(offdiag)),
    }


def select_validation_sequence(validation_text):
    candidates = [
        "outreachlm is",
        "machine learning",
        "a computer system",
        "the purpose of a transformer",
    ]
    lower = validation_text.lower()
    for candidate in candidates:
        index = lower.find(candidate)
        if index >= 0:
            start = max(0, index - 20)
            end = min(len(validation_text), index + 180)
            return validation_text[start:end], candidate
    return validation_text[:200], "fallback-start-of-validation"


def teacher_forcing_vs_free_running(model, tokenizer, validation_text):
    sequence_text, source = select_validation_sequence(validation_text)
    sequence_tokens = tokenizer.encode(sequence_text)
    if len(sequence_tokens) < 40:
        raise RuntimeError("Validation sequence too short for teacher/free analysis.")

    eval_length = min(120, len(sequence_tokens))
    prompt_length = min(40, eval_length // 2)
    prompt_tokens = sequence_tokens[:prompt_length]
    eval_tokens = sequence_tokens[:eval_length]

    teacher_rows = []
    for position in range(prompt_length, eval_length):
        context_tokens = eval_tokens[:position]
        gold_token = eval_tokens[position]
        input_ids = torch.tensor(
            [context_tokens[-model.context_length :]],
            dtype=torch.long,
            device=next(model.parameters()).device,
        )
        with torch.no_grad():
            logits = model(input_ids)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.softmax(logits[0, -1, :], dim=-1)
        top1 = int(torch.argmax(probs).item())
        teacher_rows.append(
            {
                "position": position,
                "gold_token_id": int(gold_token),
                "top1_token_id": top1,
                "gold_probability": float(probs[gold_token].item()),
            }
        )

    generated = list(prompt_tokens)
    free_rows = []
    for position in range(prompt_length, eval_length):
        input_ids = torch.tensor(
            [generated[-model.context_length :]],
            dtype=torch.long,
            device=next(model.parameters()).device,
        )
        with torch.no_grad():
            logits = model(input_ids)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.softmax(logits[0, -1, :], dim=-1)
        predicted = int(torch.argmax(probs).item())
        generated.append(predicted)
        free_rows.append(
            {
                "position": position,
                "predicted_token_id": predicted,
                "gold_token_id": int(eval_tokens[position]),
                "matches_gold": predicted == int(eval_tokens[position]),
            }
        )

    teacher_top1_correct = sum(
        1 for row in teacher_rows if row["top1_token_id"] == row["gold_token_id"]
    )
    teacher_accuracy = teacher_top1_correct / len(teacher_rows) if teacher_rows else 0.0
    teacher_avg_gold_probability = (
        sum(row["gold_probability"] for row in teacher_rows) / len(teacher_rows)
        if teacher_rows
        else 0.0
    )
    free_match_count = sum(1 for row in free_rows if row["matches_gold"])
    free_match_rate = free_match_count / len(free_rows) if free_rows else 0.0
    first_divergence = None
    for row in free_rows:
        if not row["matches_gold"]:
            first_divergence = row["position"]
            break

    return {
        "sequence_source": source,
        "prompt_text": tokenizer.decode(prompt_tokens),
        "teacher_top1_accuracy": float(teacher_accuracy),
        "teacher_average_gold_probability": float(teacher_avg_gold_probability),
        "free_match_rate_against_target": float(free_match_rate),
        "free_first_divergence_position": first_divergence,
        "target_continuation": tokenizer.decode(eval_tokens[prompt_length:eval_length]),
        "free_continuation": tokenizer.decode(generated[prompt_length:eval_length]),
    }


def first_repeat_step(sequence, n):
    seen = set()
    for idx in range(len(sequence) - n + 1):
        gram = tuple(sequence[idx : idx + n])
        if gram in seen:
            return idx + n
        seen.add(gram)
    return None


def rollout_metrics(model, tokenizer, prompt, max_new_tokens):
    generated = tokenizer.encode(prompt)
    if not generated:
        generated = [tokenizer.token_to_id[tokenizer.unk_token]]

    entropies = []
    for _ in range(max_new_tokens):
        input_ids = torch.tensor(
            [generated[-model.context_length :]],
            dtype=torch.long,
            device=next(model.parameters()).device,
        )
        with torch.no_grad():
            logits = model(input_ids)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.softmax(logits[0, -1, :], dim=-1)
        entropy = float(
            -torch.sum(probs.clamp(min=1e-12) * torch.log(probs.clamp(min=1e-12))).item()
        )
        next_token = int(torch.argmax(probs).item())
        generated.append(next_token)
        entropies.append(entropy)

    return {
        "prompt": prompt,
        "mean_entropy": float(sum(entropies) / len(entropies)) if entropies else 0.0,
        "first_repeated_bigram_step": first_repeat_step(generated, 2),
        "first_repeated_trigram_step": first_repeat_step(generated, 3),
        "generated_text": tokenizer.decode(generated),
    }


def build_frequency_balanced_weights(training_token_ids, vocab_size, device):
    counts = torch.bincount(training_token_ids, minlength=vocab_size).to(torch.float32)
    total = counts.sum().clamp(min=1.0)
    probs = counts / total

    weights = torch.zeros_like(probs)
    observed = probs > 0
    weights[observed] = torch.pow(probs[observed], -0.5)
    observed_mean = weights[observed].mean().clamp(min=1e-12)
    weights[observed] = weights[observed] / observed_mean
    weights = torch.clamp(weights, min=0.25, max=4.0)

    return weights.to(device), probs


def train_with_objective_intervention(
    model,
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
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    class_weights, empirical_probs = build_frequency_balanced_weights(
        training_token_ids=training_token_ids,
        vocab_size=tokenizer.vocab_size,
        device=device,
    )

    loss_function = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing,
    )

    loss_trace = []
    checkpoint_metrics = []

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

        loss = calculate_loss(logits, target_ids, loss_function)
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

        loss_trace.append(float(loss.item()))

        if step % max(steps // 4, 1) == 0 or step == steps:
            model.eval()
            checkpoint_metrics.append(
                {
                    "step": step,
                    "learning_rate": current_lr,
                    "teacher_free": teacher_forcing_vs_free_running(
                        model, tokenizer, validation_text
                    ),
                    "logit_alignment": prompt_logit_alignment(model, tokenizer),
                }
            )
            model.train()

    model.eval()
    final = {
        "teacher_free": teacher_forcing_vs_free_running(model, tokenizer, validation_text),
        "logit_alignment": prompt_logit_alignment(model, tokenizer),
        "rollout": rollout_metrics(model, tokenizer, "OutreachLM is", max_new_tokens=60),
    }

    return {
        "loss_summary": {
            "first_loss": float(loss_trace[0]) if loss_trace else None,
            "last_loss": float(loss_trace[-1]) if loss_trace else None,
            "mean_loss": float(sum(loss_trace) / len(loss_trace)) if loss_trace else None,
            "min_loss": float(min(loss_trace)) if loss_trace else None,
            "max_loss": float(max(loss_trace)) if loss_trace else None,
        },
        "checkpoint_metrics": checkpoint_metrics,
        "final_metrics": final,
        "weight_stats": {
            "mean": float(class_weights.mean().item()),
            "std": float(class_weights.std(unbiased=False).item()),
            "min": float(class_weights.min().item()),
            "max": float(class_weights.max().item()),
        },
        "top20_empirical_probs": [
            {
                "token_id": int(token_id),
                "token": token_text(tokenizer, int(token_id)),
                "probability": float(prob),
            }
            for prob, token_id in zip(
                *[
                    t.tolist()
                    for t in torch.topk(
                        empirical_probs.cpu(), k=min(20, empirical_probs.numel())
                    )
                ]
            )
        ],
    }


def render_report(results):
    lines = []
    lines.append("OUTREACHLM OBJECTIVE INTERVENTION (ARCHITECTURE FIXED)")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {results['timestamp']}")
    lines.append("")

    baseline = results["baseline"]
    lines.append("BASELINE SNAPSHOT")
    lines.append("-" * 80)
    lines.append(
        f"Teacher top1: {baseline['teacher_free']['teacher_top1_accuracy']:.6f} | "
        f"Free match: {baseline['teacher_free']['free_match_rate_against_target']:.6f} | "
        f"Logit cosine mean: {baseline['logit_alignment']['offdiag_logit_cosine_mean']:.6f}"
    )
    lines.append("")

    intervention = results["intervention"]
    final = intervention["final_metrics"]
    lines.append("INTERVENTION (BALANCED CE + LABEL SMOOTHING)")
    lines.append("-" * 80)
    lines.append(
        f"Loss first/last: {intervention['loss_summary']['first_loss']:.6f} / "
        f"{intervention['loss_summary']['last_loss']:.6f}"
    )
    lines.append(
        f"Teacher top1: {final['teacher_free']['teacher_top1_accuracy']:.6f} | "
        f"Free match: {final['teacher_free']['free_match_rate_against_target']:.6f} | "
        f"Logit cosine mean: {final['logit_alignment']['offdiag_logit_cosine_mean']:.6f}"
    )
    lines.append(
        f"Rollout mean entropy: {final['rollout']['mean_entropy']:.6f} | "
        f"First repeated trigram step: {final['rollout']['first_repeated_trigram_step']}"
    )
    lines.append("")

    lines.append("DELTA (INTERVENTION - BASELINE)")
    lines.append("-" * 80)
    lines.append(
        f"Teacher top1 delta: "
        f"{final['teacher_free']['teacher_top1_accuracy'] - baseline['teacher_free']['teacher_top1_accuracy']:+.6f}"
    )
    lines.append(
        f"Free match delta: "
        f"{final['teacher_free']['free_match_rate_against_target'] - baseline['teacher_free']['free_match_rate_against_target']:+.6f}"
    )
    lines.append(
        f"Logit cosine mean delta: "
        f"{final['logit_alignment']['offdiag_logit_cosine_mean'] - baseline['logit_alignment']['offdiag_logit_cosine_mean']:+.6f}"
    )

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one fixed-architecture training/objective intervention experiment.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--min-learning-rate-ratio", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer()
    text = load_corpus(CORPUS_PATH)
    training_text, validation_text = split_corpus(text, VALIDATION_SPLIT)
    training_token_ids = torch.tensor(tokenizer.encode(training_text), dtype=torch.long)

    model.eval()
    baseline = {
        "teacher_free": teacher_forcing_vs_free_running(model, tokenizer, validation_text),
        "logit_alignment": prompt_logit_alignment(model, tokenizer),
        "rollout": rollout_metrics(model, tokenizer, "OutreachLM is", max_new_tokens=60),
    }

    intervention_result = train_with_objective_intervention(
        model=model,
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
        },
        "baseline": baseline,
        "intervention": intervention_result,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = args.output_dir / f"objective-intervention-{stamp}.json"
    txt_path = args.output_dir / f"objective-intervention-{stamp}.txt"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(render_report(results))

    print(str(json_path.resolve()))
    print(str(txt_path.resolve()))


if __name__ == "__main__":
    main()

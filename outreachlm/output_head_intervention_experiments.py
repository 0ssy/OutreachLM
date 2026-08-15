import argparse
import copy
import json
import math
from datetime import datetime
from difflib import SequenceMatcher
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
    input_ids = torch.tensor(
        [context],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
    return input_ids


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
    for a in PROMPTS:
        for b in PROMPTS:
            logits_a = prompt_logits[a]
            logits_b = prompt_logits[b]
            cosine = float(
                torch.cosine_similarity(
                    logits_a.unsqueeze(0),
                    logits_b.unsqueeze(0),
                    dim=-1,
                ).item()
            )
            probs_a = torch.softmax(logits_a, dim=-1)
            probs_b = torch.softmax(logits_b, dim=-1)
            kl_sym = 0.5 * (safe_kl(probs_a, probs_b) + safe_kl(probs_b, probs_a))
            rows.append(
                {
                    "prompt_a": a,
                    "prompt_b": b,
                    "logit_cosine": cosine,
                    "kl_symmetric": kl_sym,
                }
            )
            if a != b:
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


def teacher_forced_vs_free_metrics(model, tokenizer, validation_text):
    sequence_text, source = select_validation_sequence(validation_text)
    sequence_tokens = tokenizer.encode(sequence_text)
    if len(sequence_tokens) < 40:
        raise RuntimeError("Validation sequence too short for teacher/free metrics.")

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
            logits = logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
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
    teacher_avg_gold_prob = (
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
        "teacher_average_gold_probability": float(teacher_avg_gold_prob),
        "free_match_rate_against_target": float(free_match_rate),
        "free_first_divergence_position": first_divergence,
        "target_continuation": tokenizer.decode(eval_tokens[prompt_length:eval_length]),
        "free_continuation": tokenizer.decode(generated[prompt_length:eval_length]),
    }


def rollout_entropy_metrics(model, tokenizer, prompt, max_new_tokens):
    generated = tokenizer.encode(prompt)
    if not generated:
        generated = [tokenizer.token_to_id[tokenizer.unk_token]]

    entropies = []
    token_ids = []
    token_probs = []
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
        next_id = int(torch.argmax(probs).item())
        generated.append(next_id)
        entropies.append(entropy)
        token_ids.append(next_id)
        token_probs.append(float(probs[next_id].item()))

    first_repeat_bigram = first_repeat_step(generated, 2)
    first_repeat_trigram = first_repeat_step(generated, 3)
    return {
        "prompt": prompt,
        "mean_entropy": float(sum(entropies) / len(entropies)) if entropies else 0.0,
        "min_entropy": float(min(entropies)) if entropies else 0.0,
        "max_entropy": float(max(entropies)) if entropies else 0.0,
        "first_repeated_bigram_step": first_repeat_bigram,
        "first_repeated_trigram_step": first_repeat_trigram,
        "generated_text": tokenizer.decode(generated),
        "first_15_tokens": [
            {
                "token_id": int(token_id),
                "token": token_text(tokenizer, int(token_id)),
                "probability": prob,
            }
            for token_id, prob in zip(token_ids[:15], token_probs[:15])
        ],
    }


def first_repeat_step(sequence, n):
    seen = set()
    for idx in range(len(sequence) - n + 1):
        gram = tuple(sequence[idx : idx + n])
        if gram in seen:
            return idx + n
        seen.add(gram)
    return None


def apply_intervention_center_output_head(model):
    with torch.no_grad():
        weight = model.output_head.weight.data
        bias = model.output_head.bias.data
        weight.sub_(weight.mean(dim=0, keepdim=True))
        bias.sub_(bias.mean())


def apply_intervention_reinit_output_head(model, empirical_probs):
    with torch.no_grad():
        nn.init.xavier_uniform_(model.output_head.weight)
        model.output_head.bias.copy_(torch.log(empirical_probs.clamp(min=1e-12)))


def evaluate_intervention(name, model, tokenizer, validation_text):
    alignment = prompt_logit_alignment(model, tokenizer)
    teacher_free = teacher_forced_vs_free_metrics(model, tokenizer, validation_text)
    entropy = rollout_entropy_metrics(model, tokenizer, "OutreachLM is", max_new_tokens=60)
    return {
        "name": name,
        "prompt_logit_alignment": alignment,
        "teacher_free": teacher_free,
        "rollout_entropy": entropy,
    }


def choose_intervention_candidate(baseline_result, candidate_results):
    baseline_cos = baseline_result["prompt_logit_alignment"]["offdiag_logit_cosine_mean"]
    viable = []
    for result in candidate_results:
        candidate_cos = result["prompt_logit_alignment"]["offdiag_logit_cosine_mean"]
        candidate_acc = result["teacher_free"]["teacher_top1_accuracy"]
        if candidate_cos <= baseline_cos - 0.01:
            viable.append((candidate_acc, result))
    if viable:
        viable.sort(key=lambda item: item[0], reverse=True)
        return viable[0][1]

    scored = []
    for result in candidate_results:
        candidate_cos = result["prompt_logit_alignment"]["offdiag_logit_cosine_mean"]
        candidate_acc = result["teacher_free"]["teacher_top1_accuracy"]
        score = candidate_acc - 0.5 * candidate_cos
        scored.append((score, result))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def run_training_intervention(
    model,
    training_token_ids,
    validation_text,
    tokenizer,
    steps,
    batch_size,
    learning_rate,
    warmup_steps,
    min_learning_rate_ratio,
    seed,
):
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    device = next(model.parameters()).device

    loss_trace = []
    eval_trace = []

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
            metrics = evaluate_intervention(
                name=f"step-{step}",
                model=model,
                tokenizer=tokenizer,
                validation_text=validation_text,
            )
            metrics["step"] = step
            metrics["learning_rate"] = current_lr
            eval_trace.append(metrics)
            model.train()

    model.eval()
    final_metrics = evaluate_intervention(
        name="trained-candidate",
        model=model,
        tokenizer=tokenizer,
        validation_text=validation_text,
    )
    return {
        "loss_trace_summary": {
            "first_loss": float(loss_trace[0]) if loss_trace else None,
            "last_loss": float(loss_trace[-1]) if loss_trace else None,
            "mean_loss": float(sum(loss_trace) / len(loss_trace)) if loss_trace else None,
            "min_loss": float(min(loss_trace)) if loss_trace else None,
            "max_loss": float(max(loss_trace)) if loss_trace else None,
        },
        "eval_trace": eval_trace,
        "final_metrics": final_metrics,
    }


def generate_tokens(model, tokenizer, prompt, max_new_tokens, mode, temperature, seed):
    device = next(model.parameters()).device
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        token_ids = [tokenizer.token_to_id[tokenizer.unk_token]]

    generated = list(token_ids)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    for _ in range(max_new_tokens):
        input_ids = torch.tensor(
            [generated[-model.context_length :]],
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            logits = model(input_ids)
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits[0, -1, :]

        if mode == "greedy":
            next_id = int(torch.argmax(logits).item())
        else:
            scaled = logits / max(temperature, 1e-6)
            probs = torch.softmax(scaled, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1, generator=generator).item())

        generated.append(next_id)
    return generated


def repetition_ratio(sequence, n):
    if len(sequence) < n:
        return 0.0
    grams = [tuple(sequence[i : i + n]) for i in range(len(sequence) - n + 1)]
    unique_count = len(set(grams))
    repeat_count = len(grams) - unique_count
    return float(repeat_count / len(grams))


def context_sensitivity_score(texts):
    if len(texts) < 2:
        return 0.0
    similarities = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            similarities.append(SequenceMatcher(a=texts[i], b=texts[j]).ratio())
    return float(sum(similarities) / len(similarities))


def controlled_generation_compare(
    baseline_model,
    corrected_model,
    tokenizer,
    max_new_tokens,
    sampling_temperature,
    seed,
):
    outputs = {}
    for model_name, model in [("baseline", baseline_model), ("corrected", corrected_model)]:
        outputs[model_name] = {}
        for mode in ["greedy", "sampling"]:
            mode_rows = []
            continuations = []
            for index, prompt in enumerate(PROMPTS):
                generated = generate_tokens(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    mode=mode,
                    temperature=sampling_temperature,
                    seed=seed + index,
                )
                prompt_ids = tokenizer.encode(prompt)
                continuation_ids = generated[len(prompt_ids) :]
                continuation_text = tokenizer.decode(continuation_ids)
                continuations.append(continuation_text)
                mode_rows.append(
                    {
                        "prompt": prompt,
                        "continuation_text": continuation_text,
                        "bigram_repeat_ratio": repetition_ratio(continuation_ids, 2),
                        "trigram_repeat_ratio": repetition_ratio(continuation_ids, 3),
                    }
                )

            unique_count = len(set(continuations))
            outputs[model_name][mode] = {
                "rows": mode_rows,
                "summary": {
                    "unique_continuations": unique_count,
                    "context_similarity_mean": context_sensitivity_score(continuations),
                    "mean_bigram_repeat_ratio": float(
                        sum(row["bigram_repeat_ratio"] for row in mode_rows) / len(mode_rows)
                    ),
                    "mean_trigram_repeat_ratio": float(
                        sum(row["trigram_repeat_ratio"] for row in mode_rows) / len(mode_rows)
                    ),
                },
            }
    return outputs


def build_model_artifact(model, tokenizer, training_config):
    return {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "vocab_size": tokenizer.vocab_size,
            "context_length": model.context_length,
            "embedding_dim": model.embedding_dim,
            "num_layers": model.num_layers,
            "num_heads": model.num_heads,
        },
        "training_config": training_config,
        "tokenizer_config": {
            "tokens": tokenizer.tokens,
            "pad_token": tokenizer.pad_token,
            "unk_token": tokenizer.unk_token,
        },
    }


def render_report(results):
    lines = []
    lines.append("OUTREACHLM EXPERIMENTS 8-10")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {results['timestamp']}")
    lines.append("")

    exp8 = results["experiment_8"]
    lines.append("EXPERIMENT 8 - OUTPUT HEAD INTERVENTION")
    lines.append("-" * 80)
    lines.append(
        f"Baseline offdiag logit cosine mean: "
        f"{exp8['baseline']['prompt_logit_alignment']['offdiag_logit_cosine_mean']:.6f}"
    )
    for candidate in exp8["candidates"]:
        lines.append(
            f"{candidate['name']}: cos_mean="
            f"{candidate['prompt_logit_alignment']['offdiag_logit_cosine_mean']:.6f}, "
            f"teacher_top1={candidate['teacher_free']['teacher_top1_accuracy']:.6f}, "
            f"free_match={candidate['teacher_free']['free_match_rate_against_target']:.6f}"
        )
    lines.append(f"Selected candidate: {exp8['selected_candidate_name']}")
    lines.append("")

    exp9 = results["experiment_9"]
    final9 = exp9["training_result"]["final_metrics"]
    lines.append("EXPERIMENT 9 - TRAINING INTERVENTION")
    lines.append("-" * 80)
    lines.append(
        f"Steps: {exp9['config']['train_steps']} | LR: {exp9['config']['learning_rate']}"
    )
    lines.append(
        f"Loss first/last: {exp9['training_result']['loss_trace_summary']['first_loss']:.6f} / "
        f"{exp9['training_result']['loss_trace_summary']['last_loss']:.6f}"
    )
    lines.append(
        f"Final teacher_top1: {final9['teacher_free']['teacher_top1_accuracy']:.6f} | "
        f"free_match: {final9['teacher_free']['free_match_rate_against_target']:.6f}"
    )
    lines.append(
        f"Final prompt logit cosine mean: "
        f"{final9['prompt_logit_alignment']['offdiag_logit_cosine_mean']:.6f}"
    )
    lines.append(
        f"Final rollout mean entropy: "
        f"{final9['rollout_entropy']['mean_entropy']:.6f}"
    )
    lines.append("")

    lines.append("EXPERIMENT 10 - CONTROLLED GENERATION")
    lines.append("-" * 80)
    for model_name, mode_data in results["experiment_10"]["comparison"].items():
        lines.append(model_name.upper())
        for mode in ["greedy", "sampling"]:
            summary = mode_data[mode]["summary"]
            lines.append(
                f"  {mode}: unique={summary['unique_continuations']}, "
                f"context_similarity={summary['context_similarity_mean']:.6f}, "
                f"repeat2={summary['mean_bigram_repeat_ratio']:.6f}, "
                f"repeat3={summary['mean_trigram_repeat_ratio']:.6f}"
            )
        lines.append("")

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run intervention experiments 8-10 for OutreachLM.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--min-learning-rate-ratio", type=float, default=0.1)
    parser.add_argument("--generation-tokens", type=int, default=80)
    parser.add_argument("--sampling-temperature", type=float, default=0.9)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    baseline_model, tokenizer = load_model_and_tokenizer()
    baseline_model.eval()

    corpus_text = load_corpus(CORPUS_PATH)
    training_text, validation_text = split_corpus(corpus_text, VALIDATION_SPLIT)
    training_token_ids = torch.tensor(tokenizer.encode(training_text), dtype=torch.long)
    vocab_size = tokenizer.vocab_size

    counts = torch.bincount(training_token_ids, minlength=vocab_size).to(torch.float32)
    empirical_probs = counts / counts.sum().clamp(min=1.0)
    empirical_probs = empirical_probs.to(next(baseline_model.parameters()).device)

    baseline_result = evaluate_intervention(
        name="baseline",
        model=baseline_model,
        tokenizer=tokenizer,
        validation_text=validation_text,
    )

    centered_model = copy.deepcopy(baseline_model)
    apply_intervention_center_output_head(centered_model)
    centered_result = evaluate_intervention(
        name="centered_output_head",
        model=centered_model,
        tokenizer=tokenizer,
        validation_text=validation_text,
    )

    reinit_model = copy.deepcopy(baseline_model)
    apply_intervention_reinit_output_head(reinit_model, empirical_probs)
    reinit_result = evaluate_intervention(
        name="reinitialized_output_head",
        model=reinit_model,
        tokenizer=tokenizer,
        validation_text=validation_text,
    )

    selected = choose_intervention_candidate(
        baseline_result=baseline_result,
        candidate_results=[centered_result, reinit_result],
    )
    if selected["name"] == "centered_output_head":
        selected_model = centered_model
    else:
        selected_model = reinit_model

    training_result = run_training_intervention(
        model=selected_model,
        training_token_ids=training_token_ids,
        validation_text=validation_text,
        tokenizer=tokenizer,
        steps=args.train_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        min_learning_rate_ratio=args.min_learning_rate_ratio,
        seed=args.seed,
    )

    generation_comparison = controlled_generation_compare(
        baseline_model=baseline_model,
        corrected_model=selected_model,
        tokenizer=tokenizer,
        max_new_tokens=args.generation_tokens,
        sampling_temperature=args.sampling_temperature,
        seed=args.seed,
    )

    training_config = {
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "min_learning_rate_ratio": args.min_learning_rate_ratio,
        "train_steps": args.train_steps,
        "batch_size": args.batch_size,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    corrected_model_path = args.output_dir / f"corrected-model-{stamp}.pt"
    corrected_artifact = build_model_artifact(
        model=selected_model,
        tokenizer=tokenizer,
        training_config=training_config,
    )
    torch.save(corrected_artifact, corrected_model_path)

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "seed": args.seed,
            "train_steps": args.train_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "min_learning_rate_ratio": args.min_learning_rate_ratio,
            "generation_tokens": args.generation_tokens,
            "sampling_temperature": args.sampling_temperature,
            "prompts": PROMPTS,
        },
        "experiment_8": {
            "baseline": baseline_result,
            "candidates": [centered_result, reinit_result],
            "selected_candidate_name": selected["name"],
        },
        "experiment_9": {
            "config": training_config,
            "training_result": training_result,
            "corrected_model_path": str(corrected_model_path.resolve()),
        },
        "experiment_10": {
            "comparison": generation_comparison,
        },
    }

    json_path = args.output_dir / f"interventions-8-10-{stamp}.json"
    txt_path = args.output_dir / f"interventions-8-10-{stamp}.txt"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(render_report(results))

    print(str(json_path.resolve()))
    print(str(txt_path.resolve()))
    print(str(corrected_model_path.resolve()))


if __name__ == "__main__":
    main()

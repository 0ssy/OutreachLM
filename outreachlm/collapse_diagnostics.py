import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import torch

from outreachlm.generate import load_model_and_tokenizer
from outreachlm.train import (
    CORPUS_PATH,
    VALIDATION_SPLIT,
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
    token = token.replace("\n", "\\n")
    return token


def encode_for_model(model, tokenizer, prompt):
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        token_ids = [tokenizer.token_to_id[tokenizer.unk_token]]

    context = token_ids[-model.context_length :]
    input_ids = torch.tensor(
        [context],
        dtype=torch.long,
        device=next(model.parameters()).device
    )
    return input_ids, token_ids


def forward_hidden_logits(model, input_ids):
    capture = {}

    def hook_fn(_, __, output):
        capture["final_norm"] = output.detach()

    handle = model.final_norm.register_forward_hook(hook_fn)
    with torch.no_grad():
        logits = model(input_ids)
        if isinstance(logits, tuple):
            logits = logits[0]
    handle.remove()

    hidden_last = capture["final_norm"][0, -1, :].detach()
    logits_last = logits[0, -1, :].detach()
    return hidden_last, logits_last


def safe_kl(p, q):
    p_safe = p.clamp(min=1e-12)
    q_safe = q.clamp(min=1e-12)
    return torch.sum(p_safe * torch.log(p_safe / q_safe)).item()


def topk_probs(tokenizer, probs, k):
    values, indices = torch.topk(probs, k=k)
    rows = []
    for prob, token_id in zip(values.tolist(), indices.tolist()):
        rows.append({
            "token_id": int(token_id),
            "token": token_text(tokenizer, int(token_id)),
            "probability": float(prob),
        })
    return rows


def experiment_1(model, tokenizer):
    prompt_data = {}

    for prompt in PROMPTS:
        input_ids, _ = encode_for_model(model, tokenizer, prompt)
        hidden, logits = forward_hidden_logits(model, input_ids)
        probs = torch.softmax(logits, dim=-1)

        prompt_data[prompt] = {
            "hidden": hidden.cpu(),
            "logits": logits.cpu(),
            "probs": probs.cpu(),
            "top10": topk_probs(tokenizer, probs.cpu(), 10),
        }

    pairwise = []
    for prompt_a in PROMPTS:
        for prompt_b in PROMPTS:
            h_a = prompt_data[prompt_a]["hidden"]
            h_b = prompt_data[prompt_b]["hidden"]
            l_a = prompt_data[prompt_a]["logits"]
            l_b = prompt_data[prompt_b]["logits"]
            p_a = prompt_data[prompt_a]["probs"]
            p_b = prompt_data[prompt_b]["probs"]

            hidden_cos = torch.cosine_similarity(
                h_a.unsqueeze(0),
                h_b.unsqueeze(0),
                dim=-1
            ).item()

            logit_cos = torch.cosine_similarity(
                l_a.unsqueeze(0),
                l_b.unsqueeze(0),
                dim=-1
            ).item()

            kl_ab = safe_kl(p_a, p_b)
            kl_ba = safe_kl(p_b, p_a)
            kl_sym = 0.5 * (kl_ab + kl_ba)

            pairwise.append({
                "prompt_a": prompt_a,
                "prompt_b": prompt_b,
                "hidden_cosine": hidden_cos,
                "logit_cosine": logit_cos,
                "kl_pq": kl_ab,
                "kl_qp": kl_ba,
                "kl_symmetric": kl_sym,
            })

    return {
        "prompts": PROMPTS,
        "per_prompt_top10": {
            prompt: prompt_data[prompt]["top10"]
            for prompt in PROMPTS
        },
        "pairwise": pairwise,
    }


def first_repeat_step(sequence, n):
    seen = set()
    for idx in range(len(sequence) - n + 1):
        gram = tuple(sequence[idx : idx + n])
        if gram in seen:
            return idx + n
        seen.add(gram)
    return None


def experiment_2(model, tokenizer, max_new_tokens):
    prompt = "OutreachLM is"
    _, generated_tokens = encode_for_model(
        model,
        tokenizer,
        prompt
    )

    rows = []
    previous_hidden = None

    for step in range(1, max_new_tokens + 1):
        input_ids = torch.tensor(
            [generated_tokens[-model.context_length :]],
            dtype=torch.long,
            device=next(model.parameters()).device
        )
        hidden, logits = forward_hidden_logits(model, input_ids)
        probs = torch.softmax(logits, dim=-1)

        entropy = -torch.sum(
            probs.clamp(min=1e-12)
            * torch.log(probs.clamp(min=1e-12))
        ).item()

        next_token_id = int(torch.argmax(probs).item())
        next_prob = float(probs[next_token_id].item())
        hidden_norm = float(torch.norm(hidden).item())

        if previous_hidden is None:
            hidden_cos = None
        else:
            hidden_cos = float(torch.cosine_similarity(
                hidden.unsqueeze(0),
                previous_hidden.unsqueeze(0),
                dim=-1
            ).item())

        rows.append({
            "step": step,
            "token_id": next_token_id,
            "token": token_text(tokenizer, next_token_id),
            "probability": next_prob,
            "entropy": entropy,
            "hidden_norm": hidden_norm,
            "hidden_cosine_to_previous": hidden_cos,
            "top_5_tokens": topk_probs(tokenizer, probs.cpu(), 5),
        })

        previous_hidden = hidden
        generated_tokens.append(next_token_id)

    repeat_bigram = first_repeat_step(generated_tokens, 2)
    repeat_trigram = first_repeat_step(generated_tokens, 3)

    generated_text = tokenizer.decode(generated_tokens)

    return {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "generated_text": generated_text,
        "first_repeated_bigram_step": repeat_bigram,
        "first_repeated_trigram_step": repeat_trigram,
        "rows": rows,
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


def token_rank(probs, token_id):
    sorted_indices = torch.argsort(probs, descending=True)
    rank = (sorted_indices == token_id).nonzero(as_tuple=True)[0]
    return int(rank.item()) + 1


def experiment_3(model, tokenizer, validation_text):
    sequence_text, sequence_source = select_validation_sequence(validation_text)
    sequence_tokens = tokenizer.encode(sequence_text)

    if len(sequence_tokens) < 40:
        raise RuntimeError("Validation sequence too short for experiment 3.")

    eval_length = min(120, len(sequence_tokens))
    prompt_length = min(40, eval_length // 2)
    prompt_tokens = sequence_tokens[:prompt_length]
    eval_tokens = sequence_tokens[:eval_length]

    teacher_rows = []
    for i in range(prompt_length, eval_length):
        context_tokens = eval_tokens[:i]
        gold_token = eval_tokens[i]
        input_ids = torch.tensor(
            [context_tokens[-model.context_length :]],
            dtype=torch.long,
            device=next(model.parameters()).device
        )
        _, logits = forward_hidden_logits(model, input_ids)
        probs = torch.softmax(logits, dim=-1)

        top1 = int(torch.argmax(probs).item())
        row = {
            "position": i,
            "gold_token_id": int(gold_token),
            "gold_token": token_text(tokenizer, int(gold_token)),
            "top1_token_id": top1,
            "top1_token": token_text(tokenizer, top1),
            "top1_probability": float(probs[top1].item()),
            "gold_probability": float(probs[gold_token].item()),
            "gold_rank": token_rank(probs, gold_token),
        }
        teacher_rows.append(row)

    generated = list(prompt_tokens)
    free_rows = []

    for i in range(prompt_length, eval_length):
        input_ids = torch.tensor(
            [generated[-model.context_length :]],
            dtype=torch.long,
            device=next(model.parameters()).device
        )
        _, logits = forward_hidden_logits(model, input_ids)
        probs = torch.softmax(logits, dim=-1)

        predicted = int(torch.argmax(probs).item())
        generated.append(predicted)

        free_rows.append({
            "position": i,
            "predicted_token_id": predicted,
            "predicted_token": token_text(tokenizer, predicted),
            "predicted_probability": float(probs[predicted].item()),
            "gold_token_id": int(eval_tokens[i]),
            "gold_token": token_text(tokenizer, int(eval_tokens[i])),
            "matches_gold": predicted == int(eval_tokens[i]),
        })

    teacher_top1_correct = sum(
        1 for row in teacher_rows
        if row["top1_token_id"] == row["gold_token_id"]
    )

    teacher_accuracy = (
        teacher_top1_correct / len(teacher_rows)
        if teacher_rows else 0.0
    )

    teacher_avg_gold_prob = (
        sum(row["gold_probability"] for row in teacher_rows) / len(teacher_rows)
        if teacher_rows else 0.0
    )

    free_match_count = sum(
        1 for row in free_rows if row["matches_gold"]
    )
    free_match_rate = (
        free_match_count / len(free_rows)
        if free_rows else 0.0
    )

    first_divergence = None
    for row in free_rows:
        if not row["matches_gold"]:
            first_divergence = row["position"]
            break

    target_continuation = tokenizer.decode(
        eval_tokens[prompt_length:eval_length]
    )
    free_continuation = tokenizer.decode(
        generated[prompt_length:eval_length]
    )

    return {
        "sequence_source": sequence_source,
        "sequence_text": sequence_text,
        "eval_length": eval_length,
        "prompt_length": prompt_length,
        "prompt_text": tokenizer.decode(prompt_tokens),
        "target_continuation_text": target_continuation,
        "free_running_continuation_text": free_continuation,
        "teacher_forcing": {
            "top1_accuracy": teacher_accuracy,
            "average_gold_probability": teacher_avg_gold_prob,
            "rows": teacher_rows,
        },
        "free_running": {
            "match_rate_against_target": free_match_rate,
            "first_divergence_position": first_divergence,
            "rows": free_rows,
        },
    }


def render_report(results):
    lines = []
    lines.append("OUTREACHLM COLLAPSE DIAGNOSTICS")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {results['timestamp']}")
    lines.append("")

    lines.append("EXPERIMENT 1: HIDDEN VS LOGIT COLLAPSE")
    lines.append("-" * 80)
    probe_pairs = [
        ("Machine learning allows", "A computer system can"),
        ("Machine learning allows computers to", "The purpose of a transformer"),
        ("OutreachLM is", "Machine"),
    ]
    pair_lookup = {
        (row["prompt_a"], row["prompt_b"]): row
        for row in results["experiment_1"]["pairwise"]
    }
    for a, b in probe_pairs:
        row = pair_lookup[(a, b)]
        lines.append(f"{a!r} vs {b!r}")
        lines.append(f"  hidden_cosine: {row['hidden_cosine']:.6f}")
        lines.append(f"  logit_cosine:  {row['logit_cosine']:.6f}")
        lines.append(f"  kl_symmetric:  {row['kl_symmetric']:.6f}")
    lines.append("")

    lines.append("EXPERIMENT 2: ATTRACTOR TRANSITION")
    lines.append("-" * 80)
    exp2 = results["experiment_2"]
    lines.append(f"Prompt: {exp2['prompt']}")
    lines.append(f"First repeated bigram step: {exp2['first_repeated_bigram_step']}")
    lines.append(f"First repeated trigram step: {exp2['first_repeated_trigram_step']}")
    lines.append("First 12 steps:")
    for row in exp2["rows"][:12]:
        lines.append(
            f"  step={row['step']:2d} token={row['token']!r:8s} "
            f"p={row['probability']:.6f} H={row['entropy']:.6f} "
            f"||h||={row['hidden_norm']:.6f} "
            f"cos(h_t,h_t-1)={row['hidden_cosine_to_previous']}"
        )
    lines.append("")

    lines.append("EXPERIMENT 3: TEACHER FORCING VS FREE RUNNING")
    lines.append("-" * 80)
    exp3 = results["experiment_3"]
    lines.append(f"Sequence source: {exp3['sequence_source']}")
    lines.append(f"Prompt text: {exp3['prompt_text']!r}")
    lines.append(
        f"Teacher forcing top1 accuracy: "
        f"{exp3['teacher_forcing']['top1_accuracy']:.6f}"
    )
    lines.append(
        f"Teacher forcing avg gold probability: "
        f"{exp3['teacher_forcing']['average_gold_probability']:.6f}"
    )
    lines.append(
        f"Free-running match rate vs target: "
        f"{exp3['free_running']['match_rate_against_target']:.6f}"
    )
    lines.append(
        f"First divergence position: "
        f"{exp3['free_running']['first_divergence_position']}"
    )
    lines.append("")
    lines.append("Target continuation:")
    lines.append(exp3["target_continuation_text"])
    lines.append("")
    lines.append("Free-running continuation:")
    lines.append(exp3["free_running_continuation_text"])

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run collapse diagnostics: representation/logit similarity, "
            "attractor transition, and teacher-forcing vs free-running."
        )
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=60
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments")
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer()

    full_text = load_corpus(CORPUS_PATH)
    _, validation_text = split_corpus(
        full_text,
        VALIDATION_SPLIT
    )

    exp1 = experiment_1(
        model=model,
        tokenizer=tokenizer
    )

    exp2 = experiment_2(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens
    )

    exp3 = experiment_3(
        model=model,
        tokenizer=tokenizer,
        validation_text=validation_text
    )

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "prompts": PROMPTS,
        },
        "experiment_1": exp1,
        "experiment_2": exp2,
        "experiment_3": exp3,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = args.output_dir / f"collapse-diagnostics-{stamp}.json"
    txt_path = args.output_dir / f"collapse-diagnostics-{stamp}.txt"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    report = render_report(results)
    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(report)

    print(str(json_path.resolve()))
    print(str(txt_path.resolve()))


if __name__ == "__main__":
    main()

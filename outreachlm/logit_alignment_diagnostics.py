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


def token_label(tokenizer, token_id):
    token = tokenizer.id_to_token.get(int(token_id), "")
    return token.replace("\n", "\\n")


def safe_kl(p, q):
    p_safe = p.clamp(min=1e-12)
    q_safe = q.clamp(min=1e-12)
    return float(torch.sum(p_safe * torch.log(p_safe / q_safe)).item())


def entropy_from_probs(probabilities):
    p = probabilities.clamp(min=1e-12)
    return float(-(p * torch.log(p)).sum().item())


def build_context_batch(token_ids, positions, context_length):
    contexts = []
    for pos in positions:
        start = pos - context_length
        contexts.append(token_ids[start:pos])
    return torch.tensor(contexts, dtype=torch.long)


def collect_validation_logits(
    model,
    tokenizer,
    validation_text,
    sample_count,
    seed,
    batch_size,
):
    token_ids = tokenizer.encode(validation_text)
    if len(token_ids) <= model.context_length + 1:
        raise RuntimeError("Validation token stream too short for diagnostics.")

    start = model.context_length
    stop = len(token_ids) - 1
    positions = list(range(start, stop))
    if not positions:
        raise RuntimeError("No valid positions available for context extraction.")

    actual_count = min(sample_count, len(positions))
    generator = torch.Generator()
    generator.manual_seed(seed)
    permutation = torch.randperm(len(positions), generator=generator).tolist()
    sampled_positions = sorted(positions[i] for i in permutation[:actual_count])

    context_batch_cpu = build_context_batch(token_ids, sampled_positions, model.context_length)
    logits_rows = []
    model_device = next(model.parameters()).device
    with torch.no_grad():
        for start_idx in range(0, actual_count, batch_size):
            end_idx = min(start_idx + batch_size, actual_count)
            input_ids = context_batch_cpu[start_idx:end_idx].to(model_device)
            logits = model(input_ids)
            logits_rows.append(logits[:, -1, :].detach().cpu())
    logits_matrix = torch.cat(logits_rows, dim=0)

    return {
        "positions": sampled_positions,
        "token_ids": token_ids,
        "logits_matrix": logits_matrix,
    }


def summarize_spectrum(matrix):
    singular_values = torch.linalg.svdvals(matrix)
    squared = singular_values * singular_values
    variance_total = float(squared.sum().item())
    if variance_total <= 0:
        explained = torch.zeros_like(squared)
    else:
        explained = squared / variance_total

    def value_at(index):
        if index - 1 < explained.numel():
            return float(explained[index - 1].item())
        return 0.0

    return {
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "singular_values_top20": [float(v) for v in singular_values[:20].tolist()],
        "explained_variance_ratio_top20": [float(v) for v in explained[:20].tolist()],
        "pc1_explained_variance": value_at(1),
        "pc2_explained_variance": value_at(2),
        "pc5_explained_variance": value_at(5),
        "pc10_explained_variance": value_at(10),
        "cumulative_pc1": value_at(1),
        "cumulative_pc2": value_at(1) + value_at(2),
        "cumulative_pc5": float(explained[:5].sum().item()),
        "cumulative_pc10": float(explained[:10].sum().item()),
    }


def analyze_dominant_direction(logits_matrix):
    mean_logits = logits_matrix.mean(dim=0)
    centered = logits_matrix - mean_logits.unsqueeze(0)

    raw_spectrum = summarize_spectrum(logits_matrix)
    centered_spectrum = summarize_spectrum(centered)

    mean_norm = torch.norm(mean_logits).clamp(min=1e-12)
    row_norms = torch.norm(logits_matrix, dim=1).clamp(min=1e-12)
    cosine_to_mean = torch.sum(logits_matrix * mean_logits.unsqueeze(0), dim=1) / (row_norms * mean_norm)

    return {
        "sample_count": int(logits_matrix.shape[0]),
        "vocab_size": int(logits_matrix.shape[1]),
        "raw_svd": raw_spectrum,
        "centered_svd": centered_spectrum,
        "cosine_to_mean_direction_stats": {
            "mean": float(cosine_to_mean.mean().item()),
            "std": float(cosine_to_mean.std(unbiased=False).item()),
            "min": float(cosine_to_mean.min().item()),
            "max": float(cosine_to_mean.max().item()),
        },
        "mean_logits_norm": float(mean_norm.item()),
        "mean_logits": mean_logits,
    }


def prompt_logits(model, tokenizer, prompt):
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        token_ids = [tokenizer.token_to_id[tokenizer.unk_token]]
    context = token_ids[-model.context_length :]
    input_ids = torch.tensor([context], dtype=torch.long, device=next(model.parameters()).device)
    with torch.no_grad():
        logits = model(input_ids)[0, -1, :].detach().cpu()
    return logits


def cosine(a, b):
    denom = torch.norm(a) * torch.norm(b)
    if float(denom.item()) == 0.0:
        return 0.0
    return float(torch.dot(a, b).item() / denom.item())


def prompt_centering_diagnostics(model, tokenizer, mean_logits):
    prompt_vectors = {prompt: prompt_logits(model, tokenizer, prompt) for prompt in PROMPTS}

    rows = []
    raw_cosines = []
    centered_cosines = []

    for prompt_a in PROMPTS:
        for prompt_b in PROMPTS:
            logits_a = prompt_vectors[prompt_a]
            logits_b = prompt_vectors[prompt_b]
            centered_a = logits_a - mean_logits
            centered_b = logits_b - mean_logits
            raw_cos = cosine(logits_a, logits_b)
            centered_cos = cosine(centered_a, centered_b)
            rows.append(
                {
                    "prompt_a": prompt_a,
                    "prompt_b": prompt_b,
                    "raw_logit_cosine": raw_cos,
                    "centered_logit_cosine": centered_cos,
                }
            )
            if prompt_a != prompt_b:
                raw_cosines.append(raw_cos)
                centered_cosines.append(centered_cos)

    return {
        "pairwise": rows,
        "offdiag_raw_cosine_stats": {
            "mean": float(sum(raw_cosines) / len(raw_cosines)),
            "min": float(min(raw_cosines)),
            "max": float(max(raw_cosines)),
        },
        "offdiag_centered_cosine_stats": {
            "mean": float(sum(centered_cosines) / len(centered_cosines)),
            "min": float(min(centered_cosines)),
            "max": float(max(centered_cosines)),
        },
    }


def mean_direction_top_tokens(tokenizer, mean_logits, count):
    probs = torch.softmax(mean_logits, dim=-1)
    values, indices = torch.topk(mean_logits, k=min(count, mean_logits.numel()))
    rows = []
    for value, token_id in zip(values.tolist(), indices.tolist()):
        rows.append(
            {
                "token_id": int(token_id),
                "token": token_label(tokenizer, int(token_id)),
                "mean_logit": float(value),
                "probability": float(probs[int(token_id)].item()),
            }
        )
    return rows


def select_validation_sequence(validation_text):
    candidates = [
        "outreachlm is a",
        "outreachlm is",
        "machine learning",
        "a computer system",
        "the purpose of a transformer",
    ]
    lower = validation_text.lower()
    for candidate in candidates:
        idx = lower.find(candidate)
        if idx >= 0:
            start = max(0, idx - 30)
            end = min(len(validation_text), idx + 280)
            return validation_text[start:end], candidate
    return validation_text[:280], "fallback-start-of-validation"


def next_token_distribution(model, token_history):
    context = token_history[-model.context_length :]
    input_ids = torch.tensor([context], dtype=torch.long, device=next(model.parameters()).device)
    with torch.no_grad():
        logits = model(input_ids)[0, -1, :].detach().cpu()
    return logits, torch.softmax(logits, dim=-1)


def teacher_vs_free_distribution_drift(model, tokenizer, validation_text):
    sequence_text, source = select_validation_sequence(validation_text)
    sequence_tokens = tokenizer.encode(sequence_text)
    if len(sequence_tokens) < 60:
        raise RuntimeError("Validation sequence too short for teacher/free drift analysis.")

    eval_len = min(140, len(sequence_tokens))
    prompt_len = min(40, eval_len // 2)
    eval_tokens = sequence_tokens[:eval_len]
    prompt_tokens = eval_tokens[:prompt_len]

    free_tokens = list(prompt_tokens)
    rows = []
    first_token_mismatch = None
    first_top1_disagreement = None

    for position in range(prompt_len, eval_len):
        gold_token = eval_tokens[position]

        teacher_logits, teacher_probs = next_token_distribution(model, eval_tokens[:position])
        free_logits, free_probs = next_token_distribution(model, free_tokens)

        teacher_top = int(torch.argmax(teacher_probs).item())
        free_top = int(torch.argmax(free_probs).item())
        generated_token = free_top
        free_tokens.append(generated_token)

        if generated_token != int(gold_token) and first_token_mismatch is None:
            first_token_mismatch = position
        if teacher_top != free_top and first_top1_disagreement is None:
            first_top1_disagreement = position

        kl_t_to_f = safe_kl(teacher_probs, free_probs)
        kl_f_to_t = safe_kl(free_probs, teacher_probs)
        kl_sym = 0.5 * (kl_t_to_f + kl_f_to_t)
        total_variation = 0.5 * float(torch.sum(torch.abs(teacher_probs - free_probs)).item())

        rows.append(
            {
                "position": int(position),
                "gold_token_id": int(gold_token),
                "gold_token": token_label(tokenizer, int(gold_token)),
                "teacher_top1_id": teacher_top,
                "teacher_top1_token": token_label(tokenizer, teacher_top),
                "free_top1_id": free_top,
                "free_top1_token": token_label(tokenizer, free_top),
                "teacher_gold_probability": float(teacher_probs[int(gold_token)].item()),
                "free_gold_probability": float(free_probs[int(gold_token)].item()),
                "teacher_entropy": entropy_from_probs(teacher_probs),
                "free_entropy": entropy_from_probs(free_probs),
                "kl_teacher_to_free": kl_t_to_f,
                "kl_free_to_teacher": kl_f_to_t,
                "kl_symmetric": kl_sym,
                "total_variation_distance": total_variation,
                "generated_matches_gold": generated_token == int(gold_token),
                "teacher_top1_matches_gold": teacher_top == int(gold_token),
                "free_top1_matches_gold": free_top == int(gold_token),
            }
        )

    kl_values = [row["kl_symmetric"] for row in rows]
    tv_values = [row["total_variation_distance"] for row in rows]
    first_kl_above_05 = next((row["position"] for row in rows if row["kl_symmetric"] > 0.5), None)
    first_kl_above_10 = next((row["position"] for row in rows if row["kl_symmetric"] > 1.0), None)

    teacher_correct = sum(1 for row in rows if row["teacher_top1_matches_gold"])
    free_correct = sum(1 for row in rows if row["free_top1_matches_gold"])
    generated_match = sum(1 for row in rows if row["generated_matches_gold"])
    total = len(rows)

    return {
        "sequence_source": source,
        "sequence_text": sequence_text,
        "prompt_text": tokenizer.decode(prompt_tokens),
        "eval_length": int(eval_len),
        "prompt_length": int(prompt_len),
        "rows": rows,
        "summary": {
            "teacher_top1_accuracy": float(teacher_correct / total if total else 0.0),
            "free_top1_accuracy": float(free_correct / total if total else 0.0),
            "generated_match_rate": float(generated_match / total if total else 0.0),
            "first_token_mismatch_position": first_token_mismatch,
            "first_top1_disagreement_position": first_top1_disagreement,
            "first_kl_symmetric_above_0_5": first_kl_above_05,
            "first_kl_symmetric_above_1_0": first_kl_above_10,
            "mean_kl_symmetric": float(sum(kl_values) / len(kl_values) if kl_values else 0.0),
            "max_kl_symmetric": float(max(kl_values) if kl_values else 0.0),
            "mean_total_variation_distance": float(sum(tv_values) / len(tv_values) if tv_values else 0.0),
            "max_total_variation_distance": float(max(tv_values) if tv_values else 0.0),
        },
    }


def render_report(results):
    lines = []
    lines.append("OUTREACHLM LOGIT-ALIGNMENT AND TRAJECTORY DIAGNOSTICS")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {results['timestamp']}")
    lines.append("")

    dominant = results["dominant_logit_direction"]
    raw = dominant["raw_svd"]
    centered = dominant["centered_svd"]
    lines.append("EXPERIMENT 4) DOMINANT LOGIT DIRECTION (VALIDATION CONTEXTS)")
    lines.append("-" * 80)
    lines.append(f"Samples x vocab: {raw['shape']}")
    lines.append(
        "Raw explained variance PC1/PC2/PC5/PC10: "
        f"{raw['pc1_explained_variance']:.6f} / "
        f"{raw['pc2_explained_variance']:.6f} / "
        f"{raw['pc5_explained_variance']:.6f} / "
        f"{raw['pc10_explained_variance']:.6f}"
    )
    lines.append(
        "Raw cumulative PC1/PC2/PC5/PC10: "
        f"{raw['cumulative_pc1']:.6f} / "
        f"{raw['cumulative_pc2']:.6f} / "
        f"{raw['cumulative_pc5']:.6f} / "
        f"{raw['cumulative_pc10']:.6f}"
    )
    lines.append(
        "Centered explained variance PC1/PC2/PC5/PC10: "
        f"{centered['pc1_explained_variance']:.6f} / "
        f"{centered['pc2_explained_variance']:.6f} / "
        f"{centered['pc5_explained_variance']:.6f} / "
        f"{centered['pc10_explained_variance']:.6f}"
    )
    lines.append(
        "Centered cumulative PC1/PC2/PC5/PC10: "
        f"{centered['cumulative_pc1']:.6f} / "
        f"{centered['cumulative_pc2']:.6f} / "
        f"{centered['cumulative_pc5']:.6f} / "
        f"{centered['cumulative_pc10']:.6f}"
    )
    cos_stats = dominant["cosine_to_mean_direction_stats"]
    lines.append(
        "Cosine(logits_i, mean_logits) mean/std/min/max: "
        f"{cos_stats['mean']:.6f} / {cos_stats['std']:.6f} / "
        f"{cos_stats['min']:.6f} / {cos_stats['max']:.6f}"
    )
    lines.append("")

    centering = results["mean_centering"]
    lines.append("EXPERIMENT 5) RAW VS MEAN-CENTERED LOGIT COSINES")
    lines.append("-" * 80)
    raw_stats = centering["offdiag_raw_cosine_stats"]
    centered_stats = centering["offdiag_centered_cosine_stats"]
    lines.append(
        "Prompt-pair offdiag cosine mean/min/max (raw): "
        f"{raw_stats['mean']:.6f} / {raw_stats['min']:.6f} / {raw_stats['max']:.6f}"
    )
    lines.append(
        "Prompt-pair offdiag cosine mean/min/max (centered): "
        f"{centered_stats['mean']:.6f} / {centered_stats['min']:.6f} / {centered_stats['max']:.6f}"
    )
    sample_pairs = [
        ("Machine learning allows", "A computer system can"),
        ("Machine learning allows computers to", "The purpose of a transformer"),
        ("OutreachLM is", "Machine"),
    ]
    pair_lookup = {(row["prompt_a"], row["prompt_b"]): row for row in centering["pairwise"]}
    for left, right in sample_pairs:
        row = pair_lookup[(left, right)]
        lines.append(f"{left!r} vs {right!r}")
        lines.append(
            f"  raw cosine: {row['raw_logit_cosine']:.6f} | centered cosine: {row['centered_logit_cosine']:.6f}"
        )
    lines.append("")

    lines.append("EXPERIMENT 6) TOP TOKENS OF MEAN LOGIT DIRECTION")
    lines.append("-" * 80)
    for row in results["mean_logit_top_tokens"][:30]:
        lines.append(
            f"id={row['token_id']:>3} token={row['token']!r} "
            f"mean_logit={row['mean_logit']:.6f} prob={row['probability']:.6f}"
        )
    lines.append("")

    drift = results["teacher_vs_free_drift"]
    summary = drift["summary"]
    lines.append("EXPERIMENT 7) TEACHER-FORCED VS FREE-RUN DISTRIBUTION DRIFT")
    lines.append("-" * 80)
    lines.append(f"Sequence source: {drift['sequence_source']}")
    lines.append(f"Prompt text: {drift['prompt_text']!r}")
    lines.append(
        "teacher_top1/free_top1/generated-match: "
        f"{summary['teacher_top1_accuracy']:.6f} / "
        f"{summary['free_top1_accuracy']:.6f} / "
        f"{summary['generated_match_rate']:.6f}"
    )
    lines.append(
        "first mismatch / first top1 disagreement: "
        f"{summary['first_token_mismatch_position']} / "
        f"{summary['first_top1_disagreement_position']}"
    )
    lines.append(
        "first KL_sym >0.5 / >1.0: "
        f"{summary['first_kl_symmetric_above_0_5']} / "
        f"{summary['first_kl_symmetric_above_1_0']}"
    )
    lines.append(
        "mean/max KL_sym: "
        f"{summary['mean_kl_symmetric']:.6f} / {summary['max_kl_symmetric']:.6f}"
    )
    lines.append(
        "mean/max TV distance: "
        f"{summary['mean_total_variation_distance']:.6f} / "
        f"{summary['max_total_variation_distance']:.6f}"
    )

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose dominant logit directions, centered-vs-raw similarities, "
            "and teacher-vs-free trajectory drift."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-count", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--top-tokens", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer()
    full_text = load_corpus(CORPUS_PATH)
    _, validation_text = split_corpus(full_text, VALIDATION_SPLIT)

    collected = collect_validation_logits(
        model=model,
        tokenizer=tokenizer,
        validation_text=validation_text,
        sample_count=args.sample_count,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    logits_matrix = collected["logits_matrix"]

    dominant = analyze_dominant_direction(logits_matrix)
    mean_logits = dominant.pop("mean_logits")
    centering = prompt_centering_diagnostics(model, tokenizer, mean_logits)
    top_tokens = mean_direction_top_tokens(tokenizer, mean_logits, args.top_tokens)
    drift = teacher_vs_free_distribution_drift(model, tokenizer, validation_text)

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "seed": args.seed,
            "sample_count_requested": args.sample_count,
            "sample_count_used": int(logits_matrix.shape[0]),
            "batch_size": args.batch_size,
            "prompts": PROMPTS,
        },
        "dominant_logit_direction": dominant,
        "mean_centering": centering,
        "mean_logit_top_tokens": top_tokens,
        "teacher_vs_free_drift": drift,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = args.output_dir / f"logit-alignment-diagnostics-{stamp}.json"
    txt_path = args.output_dir / f"logit-alignment-diagnostics-{stamp}.txt"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    report = render_report(results)
    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(report)

    print(str(json_path.resolve()))
    print(str(txt_path.resolve()))


if __name__ == "__main__":
    main()

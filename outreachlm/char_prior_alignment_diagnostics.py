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


def token_label(tokenizer, token_id):
    token = tokenizer.id_to_token.get(int(token_id), "")
    return token.replace("\n", "\\n")


def rank_with_ties(values):
    sorted_values, sorted_indices = torch.sort(values)
    ranks = torch.empty_like(values, dtype=torch.float32)
    index = 0
    length = sorted_values.numel()

    while index < length:
        next_index = index + 1
        while next_index < length and sorted_values[next_index] == sorted_values[index]:
            next_index += 1

        avg_rank = ((index + 1) + next_index) / 2.0
        ranks[sorted_indices[index:next_index]] = avg_rank
        index = next_index

    return ranks


def pearson_correlation(x, y):
    x_mean = x.mean()
    y_mean = y.mean()
    x_centered = x - x_mean
    y_centered = y - y_mean
    denom = torch.sqrt(
        (x_centered * x_centered).sum() * (y_centered * y_centered).sum()
    ).clamp(min=1e-12)
    return float((x_centered * y_centered).sum().item() / denom.item())


def spearman_correlation(x, y):
    x_rank = rank_with_ties(x)
    y_rank = rank_with_ties(y)
    return pearson_correlation(x_rank, y_rank)


def collect_mean_logits(model, token_ids, sample_count, seed, batch_size):
    context_length = model.context_length
    start = context_length
    stop = len(token_ids) - 1
    positions = list(range(start, stop))
    if not positions:
        raise RuntimeError("No valid positions for mean-logit collection.")

    actual_count = min(sample_count, len(positions))
    generator = torch.Generator()
    generator.manual_seed(seed)
    permutation = torch.randperm(len(positions), generator=generator).tolist()
    sampled_positions = sorted(positions[i] for i in permutation[:actual_count])

    contexts = []
    for position in sampled_positions:
        contexts.append(token_ids[position - context_length:position])

    context_tensor = torch.tensor(contexts, dtype=torch.long)
    model_device = next(model.parameters()).device

    logits_rows = []
    probs_rows = []
    with torch.no_grad():
        for batch_start in range(0, actual_count, batch_size):
            batch_end = min(batch_start + batch_size, actual_count)
            input_ids = context_tensor[batch_start:batch_end].to(model_device)
            logits = model(input_ids)[:, -1, :].detach().cpu()
            probs = torch.softmax(logits, dim=-1)
            logits_rows.append(logits)
            probs_rows.append(probs)

    logits_matrix = torch.cat(logits_rows, dim=0)
    probs_matrix = torch.cat(probs_rows, dim=0)

    return {
        "sample_count": actual_count,
        "mean_logits": logits_matrix.mean(dim=0),
        "mean_probs": probs_matrix.mean(dim=0),
    }


def top_tokens(values, tokenizer, count):
    k = min(count, values.numel())
    top_values, top_ids = torch.topk(values, k=k)
    rows = []
    for value, token_id in zip(top_values.tolist(), top_ids.tolist()):
        rows.append(
            {
                "token_id": int(token_id),
                "token": token_label(tokenizer, int(token_id)),
                "value": float(value),
            }
        )
    return rows


def unigram_baseline_cross_entropy(validation_token_ids, empirical_probs):
    eps = 1e-12
    token_probs = empirical_probs[validation_token_ids].clamp(min=eps)
    cross_entropy = float((-torch.log(token_probs)).mean().item())
    perplexity = float(math.exp(cross_entropy))
    return cross_entropy, perplexity


def render_report(results):
    lines = []
    lines.append("OUTREACHLM CHARACTER PRIOR ALIGNMENT DIAGNOSTICS")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {results['timestamp']}")
    lines.append("")

    counts = results["token_count_summary"]
    lines.append("A) TRAINING TOKEN COVERAGE")
    lines.append("-" * 80)
    lines.append(f"Vocabulary size: {counts['vocab_size']}")
    lines.append(f"Observed in training stream: {counts['observed_token_count']}")
    lines.append(f"Unseen in training stream: {counts['unseen_token_count']}")
    lines.append(f"PAD count/prob: {counts['pad_count']} / {counts['pad_probability']:.12f}")
    lines.append(f"UNK count/prob: {counts['unk_count']} / {counts['unk_probability']:.12f}")
    lines.append("")

    corr = results["alignment"]
    lines.append("B) LOG EMPIRICAL PROBABILITY VS MEAN LOGITS")
    lines.append("-" * 80)
    lines.append(
        f"Pearson (all tokens): {corr['pearson_all_tokens']:.6f}"
    )
    lines.append(
        f"Spearman (all tokens): {corr['spearman_all_tokens']:.6f}"
    )
    lines.append(
        f"Pearson (observed tokens only): {corr['pearson_observed_only']:.6f}"
    )
    lines.append(
        f"Spearman (observed tokens only): {corr['spearman_observed_only']:.6f}"
    )
    lines.append("")

    lines.append("C) TOP 20 EMPIRICAL CHARACTERS")
    lines.append("-" * 80)
    for row in results["top20_empirical"]:
        lines.append(
            f"id={row['token_id']:>3} token={row['token']!r} prob={row['probability']:.6f} "
            f"count={row['count']}"
        )
    lines.append("")

    lines.append("D) TOP 20 MEAN-LOGIT CHARACTERS")
    lines.append("-" * 80)
    for row in results["top20_mean_logit"]:
        lines.append(
            f"id={row['token_id']:>3} token={row['token']!r} mean_logit={row['mean_logit']:.6f} "
            f"mean_prob={row['mean_probability']:.6f}"
        )
    lines.append("")

    pad_unk = results["pad_unk_prediction_stats"]
    lines.append("E) PAD / UNK PREDICTION MASS")
    lines.append("-" * 80)
    lines.append(
        f"PAD rank/prob(mean softmax): {pad_unk['pad_rank_mean_probs']} / {pad_unk['pad_probability_mean_probs']:.12f}"
    )
    lines.append(
        f"UNK rank/prob(mean softmax): {pad_unk['unk_rank_mean_probs']} / {pad_unk['unk_probability_mean_probs']:.12f}"
    )
    lines.append("")

    baseline = results["unigram_baseline"]
    lines.append("F) EMPIRICAL UNIGRAM BASELINE ON VALIDATION TOKENS")
    lines.append("-" * 80)
    lines.append(f"Cross-entropy (nats): {baseline['cross_entropy_nats']:.6f}")
    lines.append(f"Perplexity: {baseline['perplexity']:.6f}")

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Character-frequency vs output-logit alignment diagnostics for OutreachLM."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-count", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer()
    text = load_corpus(CORPUS_PATH)
    training_text, validation_text = split_corpus(text, VALIDATION_SPLIT)

    training_token_ids = torch.tensor(tokenizer.encode(training_text), dtype=torch.long)
    validation_token_ids = torch.tensor(tokenizer.encode(validation_text), dtype=torch.long)

    vocab_size = tokenizer.vocab_size
    counts = torch.bincount(training_token_ids, minlength=vocab_size).to(torch.float64)
    total = counts.sum().clamp(min=1.0)
    empirical_probs = (counts / total).to(torch.float32)
    log_empirical_probs = torch.log(empirical_probs.clamp(min=1e-12))

    logits_data = collect_mean_logits(
        model=model,
        token_ids=validation_token_ids.tolist(),
        sample_count=args.sample_count,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    mean_logits = logits_data["mean_logits"]
    mean_probs = logits_data["mean_probs"]

    observed_mask = counts > 0
    observed_emp = log_empirical_probs[observed_mask]
    observed_logits = mean_logits[observed_mask]

    pearson_all = pearson_correlation(log_empirical_probs, mean_logits)
    spearman_all = spearman_correlation(log_empirical_probs, mean_logits)
    pearson_observed = pearson_correlation(observed_emp, observed_logits)
    spearman_observed = spearman_correlation(observed_emp, observed_logits)

    top_emp_probs, top_emp_ids = torch.topk(empirical_probs, k=min(20, vocab_size))
    top20_empirical = []
    for prob, token_id in zip(top_emp_probs.tolist(), top_emp_ids.tolist()):
        top20_empirical.append(
            {
                "token_id": int(token_id),
                "token": token_label(tokenizer, int(token_id)),
                "probability": float(prob),
                "count": int(counts[int(token_id)].item()),
            }
        )

    top_logit_values, top_logit_ids = torch.topk(mean_logits, k=min(20, vocab_size))
    top20_mean_logit = []
    for value, token_id in zip(top_logit_values.tolist(), top_logit_ids.tolist()):
        token_id = int(token_id)
        top20_mean_logit.append(
            {
                "token_id": token_id,
                "token": token_label(tokenizer, token_id),
                "mean_logit": float(value),
                "mean_probability": float(mean_probs[token_id].item()),
            }
        )

    sorted_prob_ids = torch.argsort(mean_probs, descending=True)
    pad_id = tokenizer.token_to_id[tokenizer.pad_token]
    unk_id = tokenizer.token_to_id[tokenizer.unk_token]
    pad_rank = int((sorted_prob_ids == pad_id).nonzero(as_tuple=True)[0].item()) + 1
    unk_rank = int((sorted_prob_ids == unk_id).nonzero(as_tuple=True)[0].item()) + 1

    cross_entropy, perplexity = unigram_baseline_cross_entropy(
        validation_token_ids=validation_token_ids,
        empirical_probs=empirical_probs,
    )

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "seed": args.seed,
            "sample_count_requested": args.sample_count,
            "sample_count_used": logits_data["sample_count"],
            "batch_size": args.batch_size,
        },
        "token_count_summary": {
            "vocab_size": vocab_size,
            "observed_token_count": int(observed_mask.sum().item()),
            "unseen_token_count": int((~observed_mask).sum().item()),
            "pad_count": int(counts[pad_id].item()),
            "unk_count": int(counts[unk_id].item()),
            "pad_probability": float(empirical_probs[pad_id].item()),
            "unk_probability": float(empirical_probs[unk_id].item()),
        },
        "alignment": {
            "pearson_all_tokens": pearson_all,
            "spearman_all_tokens": spearman_all,
            "pearson_observed_only": pearson_observed,
            "spearman_observed_only": spearman_observed,
        },
        "top20_empirical": top20_empirical,
        "top20_mean_logit": top20_mean_logit,
        "pad_unk_prediction_stats": {
            "pad_rank_mean_probs": pad_rank,
            "pad_probability_mean_probs": float(mean_probs[pad_id].item()),
            "unk_rank_mean_probs": unk_rank,
            "unk_probability_mean_probs": float(mean_probs[unk_id].item()),
        },
        "unigram_baseline": {
            "cross_entropy_nats": cross_entropy,
            "perplexity": perplexity,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = args.output_dir / f"char-prior-alignment-{stamp}.json"
    txt_path = args.output_dir / f"char-prior-alignment-{stamp}.txt"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(render_report(results))

    print(str(json_path.resolve()))
    print(str(txt_path.resolve()))


if __name__ == "__main__":
    main()

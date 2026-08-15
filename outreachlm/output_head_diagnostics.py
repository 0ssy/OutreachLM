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
    return input_ids


def capture_states(model, input_ids):
    capture = {}

    def pre_hook(_, inputs):
        capture["pre_norm"] = inputs[0].detach()

    def post_hook(_, __, output):
        capture["post_norm"] = output.detach()

    h1 = model.final_norm.register_forward_pre_hook(pre_hook)
    h2 = model.final_norm.register_forward_hook(post_hook)
    with torch.no_grad():
        logits = model(input_ids)
        if isinstance(logits, tuple):
            logits = logits[0]
    h1.remove()
    h2.remove()

    pre_last = capture["pre_norm"][0, -1, :].detach()
    post_last = capture["post_norm"][0, -1, :].detach()
    logits_ln = logits[0, -1, :].detach()
    logits_no_ln = model.output_head(pre_last).detach()
    return pre_last, post_last, logits_ln, logits_no_ln


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
            "token": token_label(tokenizer, int(token_id)),
            "probability": float(prob),
        })
    return rows


def output_head_geometry(model, tokenizer):
    W = model.output_head.weight.detach().cpu()  # [V, D]
    vocab_size, dim = W.shape

    s = torch.linalg.svdvals(W)
    s_sum = float(s.sum().item())
    if s_sum <= 0:
        effective_rank = 0.0
    else:
        p = s / s.sum()
        entropy = float(-(p * torch.log(p.clamp(min=1e-12))).sum().item())
        effective_rank = float(math.exp(entropy))

    numerical_rank = int((s > 1e-6).sum().item())

    norms = torch.norm(W, dim=1)
    norm_stats = {
        "mean": float(norms.mean().item()),
        "std": float(norms.std(unbiased=False).item()),
        "min": float(norms.min().item()),
        "p25": float(torch.quantile(norms, 0.25).item()),
        "median": float(torch.quantile(norms, 0.5).item()),
        "p75": float(torch.quantile(norms, 0.75).item()),
        "max": float(norms.max().item()),
    }

    normalized = W / norms.unsqueeze(1).clamp(min=1e-12)
    cos = normalized @ normalized.T
    mask = ~torch.eye(vocab_size, dtype=torch.bool)
    offdiag = cos[mask]
    cosine_stats = {
        "mean": float(offdiag.mean().item()),
        "std": float(offdiag.std(unbiased=False).item()),
        "min": float(offdiag.min().item()),
        "max": float(offdiag.max().item()),
    }

    pairs = []
    triu = torch.triu_indices(vocab_size, vocab_size, offset=1)
    pair_vals = cos[triu[0], triu[1]]
    top_vals, top_idx = torch.topk(pair_vals, k=min(20, pair_vals.numel()))
    for value, idx in zip(top_vals.tolist(), top_idx.tolist()):
        i = int(triu[0, idx].item())
        j = int(triu[1, idx].item())
        pairs.append({
            "token_a_id": i,
            "token_a": token_label(tokenizer, i),
            "token_b_id": j,
            "token_b": token_label(tokenizer, j),
            "cosine": float(value),
        })

    return {
        "shape": [vocab_size, dim],
        "singular_values_top20": [float(x) for x in s[:20].tolist()],
        "numerical_rank_eps_1e-6": numerical_rank,
        "effective_rank_entropy": effective_rank,
        "norm_stats": norm_stats,
        "row_cosine_stats_offdiag": cosine_stats,
        "top_most_similar_row_pairs": pairs,
    }


def prompt_projection_diagnostics(model, tokenizer):
    per_prompt = {}
    for prompt in PROMPTS:
        input_ids = encode_for_model(model, tokenizer, prompt)
        pre_h, post_h, logits_ln, logits_no_ln = capture_states(model, input_ids)
        probs_ln = torch.softmax(logits_ln, dim=-1)
        probs_no_ln = torch.softmax(logits_no_ln, dim=-1)

        per_prompt[prompt] = {
            "pre_hidden": pre_h.cpu(),
            "post_hidden": post_h.cpu(),
            "logits_ln": logits_ln.cpu(),
            "logits_no_ln": logits_no_ln.cpu(),
            "probs_ln": probs_ln.cpu(),
            "probs_no_ln": probs_no_ln.cpu(),
            "top10_ln": topk_probs(tokenizer, probs_ln.cpu(), 10),
            "top10_no_ln": topk_probs(tokenizer, probs_no_ln.cpu(), 10),
        }

    pairwise = []
    for a in PROMPTS:
        for b in PROMPTS:
            a_data = per_prompt[a]
            b_data = per_prompt[b]

            delta_h = a_data["post_hidden"] - b_data["post_hidden"]
            delta_l_ln = a_data["logits_ln"] - b_data["logits_ln"]
            delta_l_no_ln = a_data["logits_no_ln"] - b_data["logits_no_ln"]

            norm_dh = float(torch.norm(delta_h).item())
            norm_dl_ln = float(torch.norm(delta_l_ln).item())
            norm_dl_no_ln = float(torch.norm(delta_l_no_ln).item())

            ratio_ln = norm_dl_ln / norm_dh if norm_dh > 0 else 0.0
            ratio_no_ln = norm_dl_no_ln / norm_dh if norm_dh > 0 else 0.0

            hidden_cos = float(torch.cosine_similarity(
                a_data["post_hidden"].unsqueeze(0),
                b_data["post_hidden"].unsqueeze(0),
                dim=-1
            ).item())

            logit_cos_ln = float(torch.cosine_similarity(
                a_data["logits_ln"].unsqueeze(0),
                b_data["logits_ln"].unsqueeze(0),
                dim=-1
            ).item())

            logit_cos_no_ln = float(torch.cosine_similarity(
                a_data["logits_no_ln"].unsqueeze(0),
                b_data["logits_no_ln"].unsqueeze(0),
                dim=-1
            ).item())

            kl_sym_ln = 0.5 * (
                safe_kl(a_data["probs_ln"], b_data["probs_ln"])
                + safe_kl(b_data["probs_ln"], a_data["probs_ln"])
            )

            kl_sym_no_ln = 0.5 * (
                safe_kl(a_data["probs_no_ln"], b_data["probs_no_ln"])
                + safe_kl(b_data["probs_no_ln"], a_data["probs_no_ln"])
            )

            pairwise.append({
                "prompt_a": a,
                "prompt_b": b,
                "hidden_cosine": hidden_cos,
                "logit_cosine_ln": logit_cos_ln,
                "logit_cosine_no_ln": logit_cos_no_ln,
                "kl_symmetric_ln": kl_sym_ln,
                "kl_symmetric_no_ln": kl_sym_no_ln,
                "delta_hidden_norm": norm_dh,
                "delta_logits_norm_ln": norm_dl_ln,
                "delta_logits_norm_no_ln": norm_dl_no_ln,
                "compression_ratio_ln": ratio_ln,
                "compression_ratio_no_ln": ratio_no_ln,
            })

    logits_ln_stack = torch.stack(
        [per_prompt[p]["logits_ln"] for p in PROMPTS],
        dim=0
    )
    logits_no_ln_stack = torch.stack(
        [per_prompt[p]["logits_no_ln"] for p in PROMPTS],
        dim=0
    )

    var_ln = torch.var(logits_ln_stack, dim=0, unbiased=False)
    var_no_ln = torch.var(logits_no_ln_stack, dim=0, unbiased=False)

    top_var_ln_values, top_var_ln_ids = torch.topk(var_ln, k=10)
    top_var_no_ln_values, top_var_no_ln_ids = torch.topk(var_no_ln, k=10)

    variance_summary = {
        "mean_variance_ln": float(var_ln.mean().item()),
        "mean_variance_no_ln": float(var_no_ln.mean().item()),
        "median_variance_ln": float(torch.quantile(var_ln, 0.5).item()),
        "median_variance_no_ln": float(torch.quantile(var_no_ln, 0.5).item()),
        "top10_tokens_by_variance_ln": [
            {
                "token_id": int(tid),
                "token": token_label(tokenizer, int(tid)),
                "variance": float(v),
            }
            for v, tid in zip(top_var_ln_values.tolist(), top_var_ln_ids.tolist())
        ],
        "top10_tokens_by_variance_no_ln": [
            {
                "token_id": int(tid),
                "token": token_label(tokenizer, int(tid)),
                "variance": float(v),
            }
            for v, tid in zip(top_var_no_ln_values.tolist(), top_var_no_ln_ids.tolist())
        ],
    }

    return {
        "prompts": PROMPTS,
        "per_prompt_top10_ln": {
            p: per_prompt[p]["top10_ln"] for p in PROMPTS
        },
        "per_prompt_top10_no_ln": {
            p: per_prompt[p]["top10_no_ln"] for p in PROMPTS
        },
        "pairwise": pairwise,
        "logit_variance_summary": variance_summary,
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
        idx = lower.find(candidate)
        if idx >= 0:
            start = max(0, idx - 20)
            end = min(len(validation_text), idx + 220)
            return validation_text[start:end], candidate
    return validation_text[:220], "fallback-start-of-validation"


def token_rank(probs, token_id):
    sorted_ids = torch.argsort(probs, descending=True)
    rank = (sorted_ids == token_id).nonzero(as_tuple=True)[0]
    return int(rank.item()) + 1


def teacher_forcing_ablation(model, tokenizer, validation_text):
    seq_text, source = select_validation_sequence(validation_text)
    seq_tokens = tokenizer.encode(seq_text)
    if len(seq_tokens) < 40:
        raise RuntimeError("Validation sequence too short.")

    eval_len = min(120, len(seq_tokens))
    prompt_len = min(40, eval_len // 2)
    eval_tokens = seq_tokens[:eval_len]
    prompt_tokens = eval_tokens[:prompt_len]

    rows = []
    for pos in range(prompt_len, eval_len):
        context = eval_tokens[:pos]
        gold = eval_tokens[pos]
        input_ids = torch.tensor(
            [context[-model.context_length :]],
            dtype=torch.long,
            device=next(model.parameters()).device
        )
        _, _, logits_ln, logits_no_ln = capture_states(model, input_ids)

        probs_ln = torch.softmax(logits_ln, dim=-1)
        probs_no_ln = torch.softmax(logits_no_ln, dim=-1)

        top_ln = int(torch.argmax(probs_ln).item())
        top_no_ln = int(torch.argmax(probs_no_ln).item())

        rows.append({
            "position": pos,
            "gold_token_id": int(gold),
            "gold_token": token_label(tokenizer, int(gold)),
            "ln_top1_token_id": top_ln,
            "ln_top1_token": token_label(tokenizer, top_ln),
            "ln_top1_probability": float(probs_ln[top_ln].item()),
            "ln_gold_probability": float(probs_ln[gold].item()),
            "ln_gold_rank": token_rank(probs_ln, gold),
            "ln_correct": top_ln == int(gold),
            "no_ln_top1_token_id": top_no_ln,
            "no_ln_top1_token": token_label(tokenizer, top_no_ln),
            "no_ln_top1_probability": float(probs_no_ln[top_no_ln].item()),
            "no_ln_gold_probability": float(probs_no_ln[gold].item()),
            "no_ln_gold_rank": token_rank(probs_no_ln, gold),
            "no_ln_correct": top_no_ln == int(gold),
        })

    ln_correct = sum(1 for r in rows if r["ln_correct"])
    no_ln_correct = sum(1 for r in rows if r["no_ln_correct"])
    total = len(rows)

    ln_acc = ln_correct / total if total else 0.0
    no_ln_acc = no_ln_correct / total if total else 0.0

    ln_avg_gold_prob = (
        sum(r["ln_gold_probability"] for r in rows) / total if total else 0.0
    )
    no_ln_avg_gold_prob = (
        sum(r["no_ln_gold_probability"] for r in rows) / total if total else 0.0
    )

    return {
        "sequence_source": source,
        "sequence_text": seq_text,
        "prompt_text": tokenizer.decode(prompt_tokens),
        "eval_length": eval_len,
        "prompt_length": prompt_len,
        "ln_top1_accuracy": ln_acc,
        "no_ln_top1_accuracy": no_ln_acc,
        "ln_average_gold_probability": ln_avg_gold_prob,
        "no_ln_average_gold_probability": no_ln_avg_gold_prob,
        "rows": rows,
    }


def render_report(results):
    lines = []
    lines.append("OUTREACHLM OUTPUT-HEAD COLLAPSE DIAGNOSTICS")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {results['timestamp']}")
    lines.append("")

    geom = results["output_head_geometry"]
    lines.append("A) OUTPUT HEAD GEOMETRY")
    lines.append("-" * 80)
    lines.append(f"W shape: {geom['shape']}")
    lines.append(f"Numerical rank (eps=1e-6): {geom['numerical_rank_eps_1e-6']}")
    lines.append(f"Effective rank: {geom['effective_rank_entropy']:.6f}")
    lines.append(f"Row cosine mean/std/min/max: "
                 f"{geom['row_cosine_stats_offdiag']['mean']:.6f} / "
                 f"{geom['row_cosine_stats_offdiag']['std']:.6f} / "
                 f"{geom['row_cosine_stats_offdiag']['min']:.6f} / "
                 f"{geom['row_cosine_stats_offdiag']['max']:.6f}")
    lines.append("")

    lines.append("B) PROMPT PROJECTION COMPRESSION (sample pairs)")
    lines.append("-" * 80)
    sample_pairs = [
        ("Machine learning allows", "A computer system can"),
        ("Machine learning allows computers to", "The purpose of a transformer"),
        ("OutreachLM is", "Machine"),
    ]
    pair_lookup = {
        (row["prompt_a"], row["prompt_b"]): row
        for row in results["prompt_projection"]["pairwise"]
    }
    for a, b in sample_pairs:
        row = pair_lookup[(a, b)]
        lines.append(f"{a!r} vs {b!r}")
        lines.append(f"  hidden_cosine: {row['hidden_cosine']:.6f}")
        lines.append(f"  logit_cosine_ln: {row['logit_cosine_ln']:.6f}")
        lines.append(f"  logit_cosine_no_ln: {row['logit_cosine_no_ln']:.6f}")
        lines.append(f"  KL_sym ln/no_ln: "
                     f"{row['kl_symmetric_ln']:.6f} / {row['kl_symmetric_no_ln']:.6f}")
        lines.append(f"  ||Δh||: {row['delta_hidden_norm']:.6f}")
        lines.append(f"  ||Δlogits|| ln/no_ln: "
                     f"{row['delta_logits_norm_ln']:.6f} / "
                     f"{row['delta_logits_norm_no_ln']:.6f}")
        lines.append(f"  compression ratio ln/no_ln: "
                     f"{row['compression_ratio_ln']:.6f} / "
                     f"{row['compression_ratio_no_ln']:.6f}")
    lines.append("")

    var_summary = results["prompt_projection"]["logit_variance_summary"]
    lines.append("C) LOGIT VARIANCE ACROSS PROMPTS")
    lines.append("-" * 80)
    lines.append(f"Mean variance ln/no_ln: "
                 f"{var_summary['mean_variance_ln']:.6f} / "
                 f"{var_summary['mean_variance_no_ln']:.6f}")
    lines.append(f"Median variance ln/no_ln: "
                 f"{var_summary['median_variance_ln']:.6f} / "
                 f"{var_summary['median_variance_no_ln']:.6f}")
    lines.append("")

    ablation = results["teacher_forcing_ablation"]
    lines.append("D) LAYERNORM A/B (INFERENCE-ONLY, NO RETRAIN)")
    lines.append("-" * 80)
    lines.append(f"Sequence source: {ablation['sequence_source']}")
    lines.append(f"Prompt text: {ablation['prompt_text']!r}")
    lines.append(f"Teacher forcing top1 accuracy ln/no_ln: "
                 f"{ablation['ln_top1_accuracy']:.6f} / {ablation['no_ln_top1_accuracy']:.6f}")
    lines.append(f"Average gold probability ln/no_ln: "
                 f"{ablation['ln_average_gold_probability']:.6f} / "
                 f"{ablation['no_ln_average_gold_probability']:.6f}")

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Output-head collapse diagnostics: projection geometry, "
            "hidden/logit compression, and LayerNorm A/B."
        )
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
    _, validation_text = split_corpus(full_text, VALIDATION_SPLIT)

    geometry = output_head_geometry(model, tokenizer)
    projection = prompt_projection_diagnostics(model, tokenizer)
    ablation = teacher_forcing_ablation(model, tokenizer, validation_text)

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "seed": args.seed,
            "prompts": PROMPTS,
        },
        "output_head_geometry": geometry,
        "prompt_projection": projection,
        "teacher_forcing_ablation": ablation,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = args.output_dir / f"output-head-diagnostics-{stamp}.json"
    txt_path = args.output_dir / f"output-head-diagnostics-{stamp}.txt"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    report = render_report(results)
    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(report)

    print(str(json_path.resolve()))
    print(str(txt_path.resolve()))


if __name__ == "__main__":
    main()

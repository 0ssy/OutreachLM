import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch

from outreachlm.architecture_capacity_continuation import load_model_from_artifact
from outreachlm.generate import (
    TOKENIZER_PATH,
    load_tokenizer_artifact,
    upgrade_legacy_tokenizer_artifact,
)
from outreachlm.train import (
    CORPUS_PATH,
    VALIDATION_SPLIT,
    load_corpus,
    split_corpus,
)


def token_text(tokenizer, token_id):
    token = tokenizer.id_to_token.get(int(token_id), "")
    return token.replace("\n", "\\n")


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


def batched_last_logits(model, contexts, batch_size):
    rows = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, contexts.shape[0], batch_size):
            end = min(start + batch_size, contexts.shape[0])
            input_ids = contexts[start:end].to(device)
            logits = model(input_ids)
            if isinstance(logits, tuple):
                logits = logits[0]
            rows.append(logits[:, -1, :].detach().cpu())
    return torch.cat(rows, dim=0)


def topk_rows(tokenizer, logits, probs, k=10):
    values, indices = torch.topk(probs, k=min(k, probs.numel()))
    rows = []
    for prob, token_id in zip(values.tolist(), indices.tolist()):
        token_id = int(token_id)
        rows.append(
            {
                "token_id": token_id,
                "token": token_text(tokenizer, token_id),
                "probability": float(prob),
                "logit": float(logits[token_id].item()),
            }
        )
    return rows


def single_case_position_41(model, tokenizer, validation_text):
    sequence_text, sequence_source = select_validation_sequence(validation_text)
    sequence_tokens = tokenizer.encode(sequence_text)
    eval_length = min(120, len(sequence_tokens))
    prompt_length = min(40, eval_length // 2)
    if eval_length < 42:
        raise RuntimeError("Validation sequence too short for position-41 analysis.")

    eval_tokens = sequence_tokens[:eval_length]
    prompt_tokens = eval_tokens[:prompt_length]

    device = next(model.parameters()).device
    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        prompt_logits = model(prompt_tensor)
        if isinstance(prompt_logits, tuple):
            prompt_logits = prompt_logits[0]
        logits_pos40 = prompt_logits[0, -1, :].detach().cpu()
        probs_pos40 = torch.softmax(logits_pos40, dim=-1)
        pred_pos40 = int(torch.argmax(probs_pos40).item())

    teacher_context_41 = torch.tensor([eval_tokens[:41]], dtype=torch.long)
    free_context_41 = torch.tensor([prompt_tokens + [pred_pos40]], dtype=torch.long)
    teacher_logits_41 = batched_last_logits(model, teacher_context_41, batch_size=1)[0]
    free_logits_41 = batched_last_logits(model, free_context_41, batch_size=1)[0]

    teacher_probs_41 = torch.softmax(teacher_logits_41, dim=-1)
    free_probs_41 = torch.softmax(free_logits_41, dim=-1)

    gold_41 = int(eval_tokens[41])
    teacher_top_41 = int(torch.argmax(teacher_probs_41).item())
    free_top_41 = int(torch.argmax(free_probs_41).item())

    return {
        "sequence_source": sequence_source,
        "sequence_text": sequence_text,
        "prompt_text": tokenizer.decode(prompt_tokens),
        "position_40": {
            "gold_token_id": int(eval_tokens[40]),
            "gold_token": token_text(tokenizer, int(eval_tokens[40])),
            "pred_token_id": pred_pos40,
            "pred_token": token_text(tokenizer, pred_pos40),
            "pred_probability": float(probs_pos40[pred_pos40].item()),
        },
        "position_41": {
            "gold_token_id": gold_41,
            "gold_token": token_text(tokenizer, gold_41),
            "teacher_top1_token_id": teacher_top_41,
            "teacher_top1_token": token_text(tokenizer, teacher_top_41),
            "teacher_top1_probability": float(teacher_probs_41[teacher_top_41].item()),
            "teacher_gold_probability": float(teacher_probs_41[gold_41].item()),
            "free_top1_token_id": free_top_41,
            "free_top1_token": token_text(tokenizer, free_top_41),
            "free_top1_probability": float(free_probs_41[free_top_41].item()),
            "free_gold_probability": float(free_probs_41[gold_41].item()),
            "teacher_top10": topk_rows(tokenizer, teacher_logits_41, teacher_probs_41, 10),
            "free_top10": topk_rows(tokenizer, free_logits_41, free_probs_41, 10),
            "free_minus_gold_logit_margin": float(
                free_logits_41[free_top_41].item() - free_logits_41[gold_41].item()
            ),
        },
    }


def systematic_position_41_analysis(
    model,
    tokenizer,
    validation_token_ids,
    prompt_length,
    eval_length,
    sample_count,
    seed,
    batch_size,
):
    if eval_length < prompt_length + 2:
        raise ValueError("eval_length must be >= prompt_length + 2")

    max_start = len(validation_token_ids) - eval_length
    if max_start <= 0:
        raise RuntimeError("Validation token stream too short for systematic analysis.")

    starts = torch.arange(0, max_start + 1)
    actual_count = min(sample_count, starts.numel())
    generator = torch.Generator()
    generator.manual_seed(seed)
    perm = torch.randperm(starts.numel(), generator=generator)[:actual_count]
    sampled_starts = starts[perm]

    windows = torch.stack(
        [validation_token_ids[start : start + eval_length] for start in sampled_starts.tolist()],
        dim=0,
    )
    prompts = windows[:, :prompt_length]
    gold_pos41 = windows[:, prompt_length + 1]

    prompt_logits_last = batched_last_logits(model, prompts, batch_size)
    pred_pos40 = torch.argmax(prompt_logits_last, dim=-1)

    teacher_context_41 = windows[:, : prompt_length + 1]
    free_context_41 = torch.cat([prompts, pred_pos40.unsqueeze(1)], dim=1)

    teacher_logits_41 = batched_last_logits(model, teacher_context_41, batch_size)
    free_logits_41 = batched_last_logits(model, free_context_41, batch_size)

    teacher_probs_41 = torch.softmax(teacher_logits_41, dim=-1)
    free_probs_41 = torch.softmax(free_logits_41, dim=-1)

    teacher_pred_41 = torch.argmax(teacher_probs_41, dim=-1)
    free_pred_41 = torch.argmax(free_probs_41, dim=-1)

    teacher_match = teacher_pred_41 == gold_pos41
    free_match = free_pred_41 == gold_pos41

    gold_prob_free = free_probs_41.gather(1, gold_pos41.unsqueeze(1)).squeeze(1)
    pred_prob_free = free_probs_41.gather(1, free_pred_41.unsqueeze(1)).squeeze(1)
    gold_logit_free = free_logits_41.gather(1, gold_pos41.unsqueeze(1)).squeeze(1)
    pred_logit_free = free_logits_41.gather(1, free_pred_41.unsqueeze(1)).squeeze(1)
    margin_free = pred_logit_free - gold_logit_free

    mismatch_mask = ~free_match
    mismatch_gold = gold_pos41[mismatch_mask]
    mismatch_pred = free_pred_41[mismatch_mask]

    gold_counter = Counter(int(x) for x in mismatch_gold.tolist())
    pred_counter = Counter(int(x) for x in mismatch_pred.tolist())
    pair_counter = Counter(
        (int(g), int(p))
        for g, p in zip(mismatch_gold.tolist(), mismatch_pred.tolist())
    )

    top_wrong_pred_token_id = None
    top_wrong_pred_count = 0
    if pred_counter:
        top_wrong_pred_token_id, top_wrong_pred_count = pred_counter.most_common(1)[0]

    return {
        "sample_count": int(actual_count),
        "prompt_length": int(prompt_length),
        "position_analyzed": int(prompt_length + 1),
        "teacher_match_rate_at_pos41": float(teacher_match.float().mean().item()),
        "free_match_rate_at_pos41": float(free_match.float().mean().item()),
        "free_mismatch_rate_at_pos41": float(mismatch_mask.float().mean().item()),
        "free_gold_probability_mean": float(gold_prob_free.mean().item()),
        "free_gold_probability_mean_when_correct": float(
            gold_prob_free[free_match].mean().item() if free_match.any() else 0.0
        ),
        "free_gold_probability_mean_when_wrong": float(
            gold_prob_free[mismatch_mask].mean().item() if mismatch_mask.any() else 0.0
        ),
        "free_logit_margin_pred_minus_gold_mean": float(margin_free.mean().item()),
        "free_logit_margin_pred_minus_gold_mean_when_wrong": float(
            margin_free[mismatch_mask].mean().item() if mismatch_mask.any() else 0.0
        ),
        "most_common_wrong_pred_token": {
            "token_id": int(top_wrong_pred_token_id) if top_wrong_pred_token_id is not None else None,
            "token": token_text(tokenizer, int(top_wrong_pred_token_id))
            if top_wrong_pred_token_id is not None
            else None,
            "count": int(top_wrong_pred_count),
            "fraction_of_mismatches": float(
                top_wrong_pred_count / mismatch_mask.sum().item()
                if mismatch_mask.sum().item() > 0
                else 0.0
            ),
        },
        "top_mismatch_gold_tokens": [
            {
                "token_id": token_id,
                "token": token_text(tokenizer, token_id),
                "count": int(count),
            }
            for token_id, count in gold_counter.most_common(15)
        ],
        "top_mismatch_pred_tokens": [
            {
                "token_id": token_id,
                "token": token_text(tokenizer, token_id),
                "count": int(count),
            }
            for token_id, count in pred_counter.most_common(15)
        ],
        "top_mismatch_pairs_gold_to_pred": [
            {
                "gold_token_id": gold_id,
                "gold_token": token_text(tokenizer, gold_id),
                "pred_token_id": pred_id,
                "pred_token": token_text(tokenizer, pred_id),
                "count": int(count),
            }
            for (gold_id, pred_id), count in pair_counter.most_common(20)
        ],
    }


def render_report(results):
    lines = []
    lines.append("OUTREACHLM POSITION-41 TOKEN DECISION ANALYSIS")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {results['timestamp']}")
    lines.append("")
    lines.append(f"Model artifact: {results['config']['model_artifact']}")
    lines.append("")

    case = results["single_case"]
    pos41 = case["position_41"]
    lines.append("SINGLE-CASE (canonical divergence sequence)")
    lines.append("-" * 80)
    lines.append(f"Sequence source: {case['sequence_source']}")
    lines.append(f"Prompt: {case['prompt_text']!r}")
    lines.append(
        f"Position 40 | gold={case['position_40']['gold_token']!r} "
        f"pred={case['position_40']['pred_token']!r} "
        f"p(pred)={case['position_40']['pred_probability']:.6f}"
    )
    lines.append(
        f"Position 41 | gold={pos41['gold_token']!r} | "
        f"teacher top1={pos41['teacher_top1_token']!r} p={pos41['teacher_top1_probability']:.6f} "
        f"(gold p={pos41['teacher_gold_probability']:.6f})"
    )
    lines.append(
        f"             free top1={pos41['free_top1_token']!r} p={pos41['free_top1_probability']:.6f} "
        f"(gold p={pos41['free_gold_probability']:.6f}) "
        f"logit_margin(pred-gold)={pos41['free_minus_gold_logit_margin']:.6f}"
    )
    lines.append("")

    sys = results["systematic"]
    lines.append("SYSTEMATIC ACROSS VALIDATION WINDOWS")
    lines.append("-" * 80)
    lines.append(
        f"Sample count: {sys['sample_count']} | analyzed position index: {sys['position_analyzed']}"
    )
    lines.append(
        f"Teacher match rate @pos41: {sys['teacher_match_rate_at_pos41']:.6f}"
    )
    lines.append(
        f"Free match rate @pos41:    {sys['free_match_rate_at_pos41']:.6f}"
    )
    lines.append(
        f"Free mismatch rate @pos41: {sys['free_mismatch_rate_at_pos41']:.6f}"
    )
    lines.append(
        f"Free gold prob mean (all/correct/wrong): "
        f"{sys['free_gold_probability_mean']:.6f} / "
        f"{sys['free_gold_probability_mean_when_correct']:.6f} / "
        f"{sys['free_gold_probability_mean_when_wrong']:.6f}"
    )
    lines.append(
        f"Free logit margin pred-gold mean (all/wrong): "
        f"{sys['free_logit_margin_pred_minus_gold_mean']:.6f} / "
        f"{sys['free_logit_margin_pred_minus_gold_mean_when_wrong']:.6f}"
    )
    m = sys["most_common_wrong_pred_token"]
    lines.append(
        f"Most common wrong predicted token: {m['token']!r} "
        f"(count={m['count']}, fraction_of_mismatches={m['fraction_of_mismatches']:.6f})"
    )
    lines.append("")
    lines.append("Top mismatch gold->pred pairs:")
    for row in sys["top_mismatch_pairs_gold_to_pred"][:10]:
        lines.append(
            f"  {row['gold_token']!r} -> {row['pred_token']!r} : {row['count']}"
        )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect token decision at position 41 for the current leader checkpoint, "
            "including systematic behavior across validation windows."
        )
    )
    parser.add_argument(
        "--model-artifact",
        type=Path,
        default=Path("experiments/v2-divergence-intervention-20260816-113809.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-count", type=int, default=4096)
    parser.add_argument("--prompt-length", type=int, default=40)
    parser.add_argument("--eval-length", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    model, _ = load_model_from_artifact(args.model_artifact)
    model.eval()

    tokenizer = load_tokenizer_artifact(TOKENIZER_PATH)
    if tokenizer is None:
        tokenizer = upgrade_legacy_tokenizer_artifact(TOKENIZER_PATH)

    text = load_corpus(CORPUS_PATH)
    _, validation_text = split_corpus(text, VALIDATION_SPLIT)
    validation_token_ids = torch.tensor(tokenizer.encode(validation_text), dtype=torch.long)

    single_case = single_case_position_41(
        model=model,
        tokenizer=tokenizer,
        validation_text=validation_text,
    )
    systematic = systematic_position_41_analysis(
        model=model,
        tokenizer=tokenizer,
        validation_token_ids=validation_token_ids,
        prompt_length=args.prompt_length,
        eval_length=args.eval_length,
        sample_count=args.sample_count,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model_artifact": str(args.model_artifact.resolve()),
            "sample_count": args.sample_count,
            "prompt_length": args.prompt_length,
            "eval_length": args.eval_length,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "single_case": single_case,
        "systematic": systematic,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = args.output_dir / f"position41-decision-{stamp}.json"
    txt_path = args.output_dir / f"position41-decision-{stamp}.txt"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(render_report(results))

    print(str(json_path.resolve()))
    print(str(txt_path.resolve()))


if __name__ == "__main__":
    main()

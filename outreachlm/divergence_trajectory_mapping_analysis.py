import argparse
import json
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


def safe_kl(p, q):
    p_safe = p.clamp(min=1e-12)
    q_safe = q.clamp(min=1e-12)
    return float(torch.sum(p_safe * torch.log(p_safe / q_safe)).item())


def batch_last_post_hidden_and_logits(model, input_ids):
    capture = {}

    def post_hook(_, __, output):
        capture["post_norm"] = output.detach()

    handle = model.final_norm.register_forward_hook(post_hook)
    with torch.no_grad():
        logits = model(input_ids)
        if isinstance(logits, tuple):
            logits = logits[0]
    handle.remove()

    post_hidden = capture["post_norm"][:, -1, :].detach().cpu()
    last_logits = logits[:, -1, :].detach().cpu()
    return post_hidden, last_logits


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


def canonical_trajectory_analysis(model, tokenizer, validation_text, prompt_length=40, eval_length=120):
    sequence_text, sequence_source = select_validation_sequence(validation_text)
    sequence_tokens = tokenizer.encode(sequence_text)
    eval_length = min(eval_length, len(sequence_tokens))
    if eval_length < prompt_length + 2:
        raise RuntimeError("Canonical sequence too short for trajectory analysis.")

    eval_tokens = sequence_tokens[:eval_length]
    generated = list(eval_tokens[:prompt_length])

    rows = []
    first_divergence = None

    for position in range(prompt_length, eval_length):
        teacher_context = torch.tensor(
            [eval_tokens[:position]],
            dtype=torch.long,
            device=next(model.parameters()).device,
        )
        free_context = torch.tensor(
            [generated],
            dtype=torch.long,
            device=next(model.parameters()).device,
        )

        teacher_hidden, teacher_logits = batch_last_post_hidden_and_logits(model, teacher_context)
        free_hidden, free_logits = batch_last_post_hidden_and_logits(model, free_context)

        teacher_hidden = teacher_hidden[0]
        free_hidden = free_hidden[0]
        teacher_logits = teacher_logits[0]
        free_logits = free_logits[0]

        teacher_probs = torch.softmax(teacher_logits, dim=-1)
        free_probs = torch.softmax(free_logits, dim=-1)

        teacher_pred = int(torch.argmax(teacher_probs).item())
        free_pred = int(torch.argmax(free_probs).item())
        gold = int(eval_tokens[position])

        generated.append(free_pred)

        match = free_pred == gold
        if not match and first_divergence is None:
            first_divergence = position

        context_equal = (eval_tokens[:position] == generated[:-1])
        hidden_cos = float(
            torch.cosine_similarity(
                teacher_hidden.unsqueeze(0),
                free_hidden.unsqueeze(0),
                dim=-1,
            ).item()
        )
        logit_cos = float(
            torch.cosine_similarity(
                teacher_logits.unsqueeze(0),
                free_logits.unsqueeze(0),
                dim=-1,
            ).item()
        )
        kl_sym = 0.5 * (
            safe_kl(teacher_probs, free_probs) + safe_kl(free_probs, teacher_probs)
        )
        delta_h = float(torch.norm(teacher_hidden - free_hidden).item())
        delta_l = float(torch.norm(teacher_logits - free_logits).item())
        compression_ratio = float(delta_l / delta_h) if delta_h > 0 else 0.0

        rows.append(
            {
                "position": position,
                "gold_token_id": gold,
                "gold_token": token_text(tokenizer, gold),
                "teacher_pred_token_id": teacher_pred,
                "teacher_pred_token": token_text(tokenizer, teacher_pred),
                "teacher_gold_probability": float(teacher_probs[gold].item()),
                "teacher_pred_probability": float(teacher_probs[teacher_pred].item()),
                "free_pred_token_id": free_pred,
                "free_pred_token": token_text(tokenizer, free_pred),
                "free_gold_probability": float(free_probs[gold].item()),
                "free_pred_probability": float(free_probs[free_pred].item()),
                "free_matches_gold": match,
                "contexts_equal_before_prediction": bool(context_equal),
                "hidden_cos_teacher_vs_free": hidden_cos,
                "logit_cos_teacher_vs_free": logit_cos,
                "kl_symmetric_teacher_vs_free": kl_sym,
                "delta_hidden_norm": delta_h,
                "delta_logit_norm": delta_l,
                "delta_compression_ratio": compression_ratio,
            }
        )

    if first_divergence is None:
        first_divergence = -1
    start = max(prompt_length, first_divergence - 4) if first_divergence >= 0 else prompt_length
    end = min(eval_length - 1, first_divergence + 8) if first_divergence >= 0 else min(eval_length - 1, prompt_length + 12)
    focus_rows = [row for row in rows if start <= row["position"] <= end]

    return {
        "sequence_source": sequence_source,
        "sequence_text": sequence_text,
        "prompt_text": tokenizer.decode(eval_tokens[:prompt_length]),
        "prompt_length": prompt_length,
        "eval_length": eval_length,
        "first_divergence_position": first_divergence,
        "focus_window_start": start,
        "focus_window_end": end,
        "focus_rows": focus_rows,
        "all_rows": rows,
    }


def systematic_boundary_mapping(
    model,
    tokenizer,
    validation_token_ids,
    prompt_length=40,
    eval_length=80,
    sample_count=4096,
    seed=42,
    batch_size=256,
):
    if eval_length < prompt_length + 2:
        raise ValueError("eval_length must be >= prompt_length + 2")

    max_start = len(validation_token_ids) - eval_length
    if max_start <= 0:
        raise RuntimeError("Validation token stream too short.")

    starts = torch.arange(0, max_start + 1)
    actual_count = min(sample_count, starts.numel())
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampled = starts[torch.randperm(starts.numel(), generator=generator)[:actual_count]]

    windows = torch.stack(
        [validation_token_ids[start : start + eval_length] for start in sampled.tolist()],
        dim=0,
    )
    prompts = windows[:, :prompt_length]
    gold_pos40 = windows[:, prompt_length]
    gold_pos41 = windows[:, prompt_length + 1]

    device = next(model.parameters()).device
    prompt_logits_last = []
    with torch.no_grad():
        for start in range(0, actual_count, batch_size):
            end = min(start + batch_size, actual_count)
            logits = model(prompts[start:end].to(device))
            if isinstance(logits, tuple):
                logits = logits[0]
            prompt_logits_last.append(logits[:, -1, :].detach().cpu())
    prompt_logits_last = torch.cat(prompt_logits_last, dim=0)
    pred_pos40 = torch.argmax(prompt_logits_last, dim=-1)

    teacher_ctx41 = windows[:, : prompt_length + 1]
    free_ctx41 = torch.cat([prompts, pred_pos40.unsqueeze(1)], dim=1)

    teacher_hidden41, teacher_logits41 = [], []
    free_hidden41, free_logits41 = [], []
    for start in range(0, actual_count, batch_size):
        end = min(start + batch_size, actual_count)
        th, tl = batch_last_post_hidden_and_logits(model, teacher_ctx41[start:end].to(device))
        fh, fl = batch_last_post_hidden_and_logits(model, free_ctx41[start:end].to(device))
        teacher_hidden41.append(th)
        teacher_logits41.append(tl)
        free_hidden41.append(fh)
        free_logits41.append(fl)
    teacher_hidden41 = torch.cat(teacher_hidden41, dim=0)
    teacher_logits41 = torch.cat(teacher_logits41, dim=0)
    free_hidden41 = torch.cat(free_hidden41, dim=0)
    free_logits41 = torch.cat(free_logits41, dim=0)

    teacher_probs41 = torch.softmax(teacher_logits41, dim=-1)
    free_probs41 = torch.softmax(free_logits41, dim=-1)
    teacher_pred41 = torch.argmax(teacher_probs41, dim=-1)
    free_pred41 = torch.argmax(free_probs41, dim=-1)

    teacher_match41 = teacher_pred41 == gold_pos41
    free_match41 = free_pred41 == gold_pos41
    context_diff41 = pred_pos40 != gold_pos40
    context_same41 = ~context_diff41

    hidden_cos = torch.cosine_similarity(teacher_hidden41, free_hidden41, dim=-1)
    logit_cos = torch.cosine_similarity(teacher_logits41, free_logits41, dim=-1)
    delta_h = torch.norm(teacher_hidden41 - free_hidden41, dim=-1)
    delta_l = torch.norm(teacher_logits41 - free_logits41, dim=-1)

    kl_values = []
    for i in range(actual_count):
        kl = 0.5 * (
            safe_kl(teacher_probs41[i], free_probs41[i])
            + safe_kl(free_probs41[i], teacher_probs41[i])
        )
        kl_values.append(kl)
    kl_values = torch.tensor(kl_values, dtype=torch.float32)

    def masked_mean(values, mask):
        if mask.any():
            return float(values[mask].mean().item())
        return 0.0

    def rate(mask):
        return float(mask.float().mean().item())

    return {
        "sample_count": int(actual_count),
        "prompt_length": int(prompt_length),
        "position_40_prediction_match_rate": rate(pred_pos40 == gold_pos40),
        "position_40_prediction_mismatch_rate": rate(context_diff41),
        "position_41_teacher_match_rate": rate(teacher_match41),
        "position_41_free_match_rate": rate(free_match41),
        "position_41_free_match_rate_when_context_same": masked_mean(
            free_match41.float(), context_same41
        ),
        "position_41_free_match_rate_when_context_diff": masked_mean(
            free_match41.float(), context_diff41
        ),
        "hidden_cos_teacher_vs_free_mean_all": float(hidden_cos.mean().item()),
        "hidden_cos_teacher_vs_free_mean_when_context_same": masked_mean(hidden_cos, context_same41),
        "hidden_cos_teacher_vs_free_mean_when_context_diff": masked_mean(hidden_cos, context_diff41),
        "logit_cos_teacher_vs_free_mean_all": float(logit_cos.mean().item()),
        "logit_cos_teacher_vs_free_mean_when_context_same": masked_mean(logit_cos, context_same41),
        "logit_cos_teacher_vs_free_mean_when_context_diff": masked_mean(logit_cos, context_diff41),
        "kl_symmetric_mean_all": float(kl_values.mean().item()),
        "kl_symmetric_mean_when_context_same": masked_mean(kl_values, context_same41),
        "kl_symmetric_mean_when_context_diff": masked_mean(kl_values, context_diff41),
        "delta_hidden_norm_mean_when_context_diff": masked_mean(delta_h, context_diff41),
        "delta_logit_norm_mean_when_context_diff": masked_mean(delta_l, context_diff41),
        "delta_ratio_mean_when_context_diff": float(
            (delta_l[context_diff41] / delta_h[context_diff41].clamp(min=1e-12)).mean().item()
        )
        if context_diff41.any()
        else 0.0,
    }


def render_report(results):
    lines = []
    lines.append("OUTREACHLM DIVERGENCE-BOUNDARY TRAJECTORY MAPPING")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {results['timestamp']}")
    lines.append("")
    lines.append(f"Model artifact: {results['config']['model_artifact']}")
    lines.append("")

    c = results["canonical"]
    lines.append("CANONICAL TRAJECTORY (teacher vs free)")
    lines.append("-" * 80)
    lines.append(f"Sequence source: {c['sequence_source']}")
    lines.append(f"Prompt: {c['prompt_text']!r}")
    lines.append(f"First divergence position: {c['first_divergence_position']}")
    lines.append(
        f"Focus window: {c['focus_window_start']}..{c['focus_window_end']}"
    )
    lines.append("")
    lines.append(
        "pos | gold | t_pred | f_pred | ctx_equal | h_cos | l_cos | KL | "
        "t_gold_p | f_gold_p"
    )
    for row in c["focus_rows"]:
        lines.append(
            "{pos:3d} | {gold!r:>4} | {tp!r:>6} | {fp!r:>6} | {ctx!s:>9} | "
            "{hc:.4f} | {lc:.4f} | {kl:.4f} | {tgp:.4f} | {fgp:.4f}".format(
                pos=row["position"],
                gold=row["gold_token"],
                tp=row["teacher_pred_token"],
                fp=row["free_pred_token"],
                ctx=row["contexts_equal_before_prediction"],
                hc=row["hidden_cos_teacher_vs_free"],
                lc=row["logit_cos_teacher_vs_free"],
                kl=row["kl_symmetric_teacher_vs_free"],
                tgp=row["teacher_gold_probability"],
                fgp=row["free_gold_probability"],
            )
        )
    lines.append("")

    s = results["systematic"]
    lines.append("SYSTEMATIC BOUNDARY MAPPING")
    lines.append("-" * 80)
    lines.append(f"Sample count: {s['sample_count']}")
    lines.append(
        "pos40 match/mismatch rate: "
        f"{s['position_40_prediction_match_rate']:.6f} / "
        f"{s['position_40_prediction_mismatch_rate']:.6f}"
    )
    lines.append(
        "pos41 teacher/free match: "
        f"{s['position_41_teacher_match_rate']:.6f} / "
        f"{s['position_41_free_match_rate']:.6f}"
    )
    lines.append(
        "pos41 free match when context same/diff: "
        f"{s['position_41_free_match_rate_when_context_same']:.6f} / "
        f"{s['position_41_free_match_rate_when_context_diff']:.6f}"
    )
    lines.append(
        "hidden cos teacher/free mean all/same/diff: "
        f"{s['hidden_cos_teacher_vs_free_mean_all']:.6f} / "
        f"{s['hidden_cos_teacher_vs_free_mean_when_context_same']:.6f} / "
        f"{s['hidden_cos_teacher_vs_free_mean_when_context_diff']:.6f}"
    )
    lines.append(
        "logit cos teacher/free mean all/same/diff: "
        f"{s['logit_cos_teacher_vs_free_mean_all']:.6f} / "
        f"{s['logit_cos_teacher_vs_free_mean_when_context_same']:.6f} / "
        f"{s['logit_cos_teacher_vs_free_mean_when_context_diff']:.6f}"
    )
    lines.append(
        "KL_sym mean all/same/diff: "
        f"{s['kl_symmetric_mean_all']:.6f} / "
        f"{s['kl_symmetric_mean_when_context_same']:.6f} / "
        f"{s['kl_symmetric_mean_when_context_diff']:.6f}"
    )
    lines.append(
        "delta norms when context diff | ||dh||, ||dlogit||, ratio: "
        f"{s['delta_hidden_norm_mean_when_context_diff']:.6f}, "
        f"{s['delta_logit_norm_mean_when_context_diff']:.6f}, "
        f"{s['delta_ratio_mean_when_context_diff']:.6f}"
    )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze hidden/logit mapping at the teacher->free divergence boundary."
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

    canonical = canonical_trajectory_analysis(
        model=model,
        tokenizer=tokenizer,
        validation_text=validation_text,
        prompt_length=args.prompt_length,
        eval_length=min(120, len(validation_text)),
    )
    systematic = systematic_boundary_mapping(
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
        "canonical": canonical,
        "systematic": systematic,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = args.output_dir / f"divergence-trajectory-mapping-{stamp}.json"
    txt_path = args.output_dir / f"divergence-trajectory-mapping-{stamp}.txt"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(render_report(results))

    print(str(json_path.resolve()))
    print(str(txt_path.resolve()))


if __name__ == "__main__":
    main()

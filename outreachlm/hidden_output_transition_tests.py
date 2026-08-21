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
from outreachlm.v4_generate import load_model_and_tokenizer as load_v4_model_and_tokenizer


def token_text(tokenizer, token_id):
    token = tokenizer.id_to_token.get(int(token_id), "")
    return token.replace("\n", "\\n")


def safe_kl(p, q):
    p_safe = p.clamp(min=1e-12)
    q_safe = q.clamp(min=1e-12)
    return float(torch.sum(p_safe * torch.log(p_safe / q_safe)).item())


def batch_last_hidden_logits(model, contexts):
    capture = {}

    def post_hook(_, __, output):
        capture["post_norm"] = output.detach()

    handle = model.final_norm.register_forward_hook(post_hook)
    with torch.no_grad():
        logits = model(contexts)
        if isinstance(logits, tuple):
            logits = logits[0]
    handle.remove()

    hidden = capture["post_norm"][:, -1, :].detach().cpu()
    logits = logits[:, -1, :].detach().cpu()
    return hidden, logits


def sample_windows(validation_token_ids, eval_length, sample_count, seed):
    max_start = len(validation_token_ids) - eval_length
    if max_start <= 0:
        raise RuntimeError("Validation token stream too short for window sampling.")

    starts = torch.arange(0, max_start + 1)
    actual_count = min(sample_count, starts.numel())
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampled = starts[torch.randperm(starts.numel(), generator=generator)[:actual_count]]
    windows = torch.stack(
        [validation_token_ids[start : start + eval_length] for start in sampled.tolist()],
        dim=0,
    )
    return windows, actual_count


def load_model_for_suite(artifact_path):
    artifact = torch.load(
        artifact_path,
        map_location="cpu",
        weights_only=False,
    )
    model_type = artifact.get("model_config", {}).get("model_type")
    if model_type == "outreachlm_v4":
        model, _ = load_v4_model_and_tokenizer(artifact_path, None)
        return model
    model, _ = load_model_from_artifact(artifact_path)
    return model


def collect_teacher_free_states(
    model,
    windows,
    prompt_length,
    position_start,
    position_end,
    batch_size,
):
    if position_end >= windows.shape[1]:
        raise ValueError("position_end exceeds eval_length.")

    device = next(model.parameters()).device
    num_samples = windows.shape[0]

    generated = windows[:, :prompt_length].clone()

    by_pos = {}

    for position in range(position_start, position_end + 1):
        teacher_context = windows[:, :position]
        if position <= generated.shape[1]:
            free_context = generated[:, :position]
        else:
            raise RuntimeError("Generated sequence shorter than requested position.")

        teacher_hidden_rows = []
        teacher_logits_rows = []
        free_hidden_rows = []
        free_logits_rows = []

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            teacher_batch = teacher_context[start:end].to(device)
            free_batch = free_context[start:end].to(device)
            th, tl = batch_last_hidden_logits(model, teacher_batch)
            fh, fl = batch_last_hidden_logits(model, free_batch)
            teacher_hidden_rows.append(th)
            teacher_logits_rows.append(tl)
            free_hidden_rows.append(fh)
            free_logits_rows.append(fl)

        teacher_hidden = torch.cat(teacher_hidden_rows, dim=0)
        teacher_logits = torch.cat(teacher_logits_rows, dim=0)
        free_hidden = torch.cat(free_hidden_rows, dim=0)
        free_logits = torch.cat(free_logits_rows, dim=0)
        teacher_probs = torch.softmax(teacher_logits, dim=-1)
        free_probs = torch.softmax(free_logits, dim=-1)
        teacher_pred = torch.argmax(teacher_probs, dim=-1)
        free_pred = torch.argmax(free_probs, dim=-1)
        gold = windows[:, position]
        context_same = (
            (teacher_context == free_context).all(dim=1)
            if teacher_context.shape[1] == free_context.shape[1]
            else torch.zeros(num_samples, dtype=torch.bool)
        )

        by_pos[position] = {
            "teacher_hidden": teacher_hidden,
            "free_hidden": free_hidden,
            "teacher_logits": teacher_logits,
            "free_logits": free_logits,
            "teacher_probs": teacher_probs,
            "free_probs": free_probs,
            "teacher_pred": teacher_pred,
            "free_pred": free_pred,
            "gold": gold,
            "context_same": context_same,
        }

        # advance free rollout with free predictions for this position
        if position >= prompt_length:
            generated = torch.cat([generated, free_pred.unsqueeze(1)], dim=1)

    return by_pos


def test1_hidden_transition(by_pos, position_start, position_end):
    rows = []
    divergence_onset_position = None

    for position in range(position_start, position_end + 1):
        data = by_pos[position]
        teacher_hidden = data["teacher_hidden"]
        free_hidden = data["free_hidden"]
        context_same = data["context_same"]
        context_diff = ~context_same

        hidden_cos = torch.cosine_similarity(teacher_hidden, free_hidden, dim=-1)
        hidden_distance = 1.0 - hidden_cos
        delta = teacher_hidden - free_hidden
        per_dim_abs = torch.abs(delta)
        per_dim_mean_abs = per_dim_abs.mean(dim=0)
        per_dim_std_abs = per_dim_abs.std(dim=0, unbiased=False)

        if divergence_onset_position is None:
            if float(hidden_distance.mean().item()) > 1e-6 or context_diff.any():
                divergence_onset_position = position

        top_vals, top_idx = torch.topk(
            per_dim_mean_abs,
            k=min(10, per_dim_mean_abs.numel()),
        )

        row = {
            "position": position,
            "context_same_rate": float(context_same.float().mean().item()),
            "context_diff_rate": float(context_diff.float().mean().item()),
            "hidden_cosine_mean": float(hidden_cos.mean().item()),
            "hidden_cosine_min": float(hidden_cos.min().item()),
            "hidden_cosine_max": float(hidden_cos.max().item()),
            "hidden_cosine_distance_mean": float(hidden_distance.mean().item()),
            "hidden_norm_teacher_mean": float(torch.norm(teacher_hidden, dim=-1).mean().item()),
            "hidden_norm_free_mean": float(torch.norm(free_hidden, dim=-1).mean().item()),
            "delta_hidden_norm_mean": float(torch.norm(delta, dim=-1).mean().item()),
            "per_dimension_mean_abs_divergence": [float(x) for x in per_dim_mean_abs.tolist()],
            "per_dimension_std_abs_divergence": [float(x) for x in per_dim_std_abs.tolist()],
            "top10_divergent_dimensions": [
                {
                    "dimension": int(dim),
                    "mean_abs_divergence": float(val),
                }
                for val, dim in zip(top_vals.tolist(), top_idx.tolist())
            ],
        }
        rows.append(row)

    return {
        "positions": [int(p) for p in range(position_start, position_end + 1)],
        "divergence_onset_position": int(divergence_onset_position)
        if divergence_onset_position is not None
        else None,
        "rows": rows,
    }


def topk_overlap_rate(teacher_logits, free_logits, k):
    t_idx = torch.topk(teacher_logits, k=k, dim=-1).indices
    f_idx = torch.topk(free_logits, k=k, dim=-1).indices
    overlaps = []
    for i in range(t_idx.shape[0]):
        ts = set(int(x) for x in t_idx[i].tolist())
        fs = set(int(x) for x in f_idx[i].tolist())
        overlaps.append(len(ts & fs) / k)
    return float(sum(overlaps) / len(overlaps))


def mean_masked(values, mask):
    if mask.any():
        return float(values[mask].mean().item())
    return 0.0


def test2_output_head_sensitivity(by_pos, position_start, position_end, topk):
    rows = []
    for position in range(position_start, position_end + 1):
        data = by_pos[position]
        t_h = data["teacher_hidden"]
        f_h = data["free_hidden"]
        t_l = data["teacher_logits"]
        f_l = data["free_logits"]
        t_p = data["teacher_probs"]
        f_p = data["free_probs"]
        t_pred = data["teacher_pred"]
        f_pred = data["free_pred"]
        gold = data["gold"]
        context_same = data["context_same"]
        context_diff = ~context_same

        idx = torch.arange(gold.shape[0])
        t_gold_logit = t_l[idx, gold]
        f_gold_logit = f_l[idx, gold]
        t_pred_logit = t_l[idx, t_pred]
        f_pred_logit = f_l[idx, f_pred]
        t_gold_prob = t_p[idx, gold]
        f_gold_prob = f_p[idx, gold]
        t_pred_prob = t_p[idx, t_pred]
        f_pred_prob = f_p[idx, f_pred]

        margin_t = t_pred_logit - t_gold_logit
        margin_f = f_pred_logit - f_gold_logit
        margin_gap = margin_f - margin_t
        argmax_change = t_pred != f_pred

        hidden_delta = torch.norm(t_h - f_h, dim=-1)
        logit_delta = torch.norm(t_l - f_l, dim=-1)
        ratio = logit_delta / hidden_delta.clamp(min=1e-12)
        logit_cos = torch.cosine_similarity(t_l, f_l, dim=-1)

        kl_list = []
        for i in range(gold.shape[0]):
            kl = 0.5 * (safe_kl(t_p[i], f_p[i]) + safe_kl(f_p[i], t_p[i]))
            kl_list.append(kl)
        kl = torch.tensor(kl_list, dtype=torch.float32)

        rows.append(
            {
                "position": position,
                "context_diff_rate": float(context_diff.float().mean().item()),
                "teacher_gold_logit_mean": float(t_gold_logit.mean().item()),
                "free_gold_logit_mean": float(f_gold_logit.mean().item()),
                "teacher_pred_logit_mean": float(t_pred_logit.mean().item()),
                "free_pred_logit_mean": float(f_pred_logit.mean().item()),
                "teacher_margin_pred_minus_gold_mean": float(margin_t.mean().item()),
                "free_margin_pred_minus_gold_mean": float(margin_f.mean().item()),
                "margin_shift_free_minus_teacher_mean": float(margin_gap.mean().item()),
                "teacher_gold_probability_mean": float(t_gold_prob.mean().item()),
                "free_gold_probability_mean": float(f_gold_prob.mean().item()),
                "teacher_pred_probability_mean": float(t_pred_prob.mean().item()),
                "free_pred_probability_mean": float(f_pred_prob.mean().item()),
                "topk_overlap_rate": topk_overlap_rate(t_l, f_l, topk),
                "argmax_change_rate": float(argmax_change.float().mean().item()),
                "logit_cosine_mean": float(logit_cos.mean().item()),
                "kl_symmetric_mean": float(kl.mean().item()),
                "hidden_delta_norm_mean": float(hidden_delta.mean().item()),
                "logit_delta_norm_mean": float(logit_delta.mean().item()),
                "delta_ratio_logit_over_hidden_mean": float(ratio.mean().item()),
                "free_gold_probability_mean_when_context_diff": mean_masked(
                    f_gold_prob, context_diff
                ),
                "free_gold_probability_mean_when_context_same": mean_masked(
                    f_gold_prob, context_same
                ),
                "top_wrong_pred_tokens_when_context_diff": top_wrong_pred_tokens(
                    free_pred=f_pred,
                    gold=gold,
                    context_diff=context_diff,
                ),
            }
        )
    return {
        "positions": [int(p) for p in range(position_start, position_end + 1)],
        "rows": rows,
    }


def top_wrong_pred_tokens(free_pred, gold, context_diff, limit=10):
    wrong = context_diff & (free_pred != gold)
    if not wrong.any():
        return []
    pred_wrong = free_pred[wrong]
    values, counts = pred_wrong.unique(return_counts=True)
    order = torch.argsort(counts, descending=True)
    rows = []
    total = counts.sum().item()
    for idx in order[:limit]:
        token_id = int(values[idx].item())
        count = int(counts[idx].item())
        rows.append(
            {
                "token_id": token_id,
                "count": count,
                "fraction_of_wrong_context_diff": float(count / total),
            }
        )
    return rows


def render_report(results, tokenizer):
    lines = []
    lines.append("OUTREACHLM TESTS 1-2: HIDDEN TRAJECTORY + OUTPUT SENSITIVITY")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {results['timestamp']}")
    lines.append("")
    lines.append(f"Model artifact: {results['config']['model_artifact']}")
    lines.append("")

    t1 = results["test_1_hidden_transition"]
    lines.append("TEST 1 — HIDDEN-STATE TRANSITION (positions 38-45)")
    lines.append("-" * 80)
    lines.append(f"Divergence onset position: {t1['divergence_onset_position']}")
    lines.append("pos | ctx_diff | h_cos_dist_mean | ||h_t|| | ||h_f|| | ||dh||")
    for row in t1["rows"]:
        lines.append(
            "{pos:3d} | {cd:.4f} | {hd:.4f} | {tn:.4f} | {fn:.4f} | {dn:.4f}".format(
                pos=row["position"],
                cd=row["context_diff_rate"],
                hd=row["hidden_cosine_distance_mean"],
                tn=row["hidden_norm_teacher_mean"],
                fn=row["hidden_norm_free_mean"],
                dn=row["delta_hidden_norm_mean"],
            )
        )
    lines.append("")

    t2 = results["test_2_output_sensitivity"]
    lines.append("TEST 2 — OUTPUT-HEAD SENSITIVITY (positions 39-43)")
    lines.append("-" * 80)
    lines.append(
        "pos | ctx_diff | t_gold_p | f_gold_p | margin_t | margin_f | "
        "topk_ov | argmax_change | logit_cos | KL | ratio"
    )
    for row in t2["rows"]:
        lines.append(
            "{pos:3d} | {cd:.4f} | {tgp:.4f} | {fgp:.4f} | {mt:.4f} | {mf:.4f} | "
            "{ov:.4f} | {ac:.4f} | {lc:.4f} | {kl:.4f} | {rr:.4f}".format(
                pos=row["position"],
                cd=row["context_diff_rate"],
                tgp=row["teacher_gold_probability_mean"],
                fgp=row["free_gold_probability_mean"],
                mt=row["teacher_margin_pred_minus_gold_mean"],
                mf=row["free_margin_pred_minus_gold_mean"],
                ov=row["topk_overlap_rate"],
                ac=row["argmax_change_rate"],
                lc=row["logit_cosine_mean"],
                kl=row["kl_symmetric_mean"],
                rr=row["delta_ratio_logit_over_hidden_mean"],
            )
        )
        if row["top_wrong_pred_tokens_when_context_diff"]:
            top = row["top_wrong_pred_tokens_when_context_diff"][0]
            lines.append(
                f"      top wrong pred token (ctx diff): "
                f"{token_text(tokenizer, top['token_id'])!r} "
                f"fraction={top['fraction_of_wrong_context_diff']:.4f}"
            )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Test 1 and Test 2 on the leader checkpoint."
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
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    model = load_model_for_suite(args.model_artifact)
    model.eval()

    tokenizer = load_tokenizer_artifact(TOKENIZER_PATH)
    if tokenizer is None:
        tokenizer = upgrade_legacy_tokenizer_artifact(TOKENIZER_PATH)

    text = load_corpus(CORPUS_PATH)
    _, validation_text = split_corpus(text, VALIDATION_SPLIT)
    validation_token_ids = torch.tensor(tokenizer.encode(validation_text), dtype=torch.long)

    windows, actual_count = sample_windows(
        validation_token_ids=validation_token_ids,
        eval_length=args.eval_length,
        sample_count=args.sample_count,
        seed=args.seed,
    )

    by_pos = collect_teacher_free_states(
        model=model,
        windows=windows,
        prompt_length=args.prompt_length,
        position_start=38,
        position_end=45,
        batch_size=args.batch_size,
    )

    test1 = test1_hidden_transition(
        by_pos=by_pos,
        position_start=38,
        position_end=45,
    )
    test2 = test2_output_head_sensitivity(
        by_pos=by_pos,
        position_start=39,
        position_end=43,
        topk=args.topk,
    )

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model_artifact": str(args.model_artifact.resolve()),
            "sample_count_requested": args.sample_count,
            "sample_count_used": actual_count,
            "prompt_length": args.prompt_length,
            "eval_length": args.eval_length,
            "batch_size": args.batch_size,
            "topk": args.topk,
            "seed": args.seed,
        },
        "test_1_hidden_transition": test1,
        "test_2_output_sensitivity": test2,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = args.output_dir / f"tests1-2-hidden-output-transition-{stamp}.json"
    txt_path = args.output_dir / f"tests1-2-hidden-output-transition-{stamp}.txt"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(render_report(results, tokenizer))

    print(str(json_path.resolve()))
    print(str(txt_path.resolve()))


if __name__ == "__main__":
    main()

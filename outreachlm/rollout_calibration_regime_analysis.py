import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch

from outreachlm.architecture_capacity_continuation import load_model_from_artifact
from outreachlm.architecture_capacity_pilot import metric_row, metrics_snapshot
from outreachlm.divergence_trajectory_mapping_analysis import canonical_trajectory_analysis
from outreachlm.generate import TOKENIZER_PATH, load_tokenizer_artifact, upgrade_legacy_tokenizer_artifact
from outreachlm.hidden_output_transition_tests import collect_teacher_free_states, sample_windows
from outreachlm.objective_intervention_experiment import teacher_forcing_vs_free_running
from outreachlm.train import CORPUS_PATH, VALIDATION_SPLIT, load_corpus, split_corpus
from outreachlm.v4_generate import load_model_and_tokenizer as load_v4_model_and_tokenizer


def parse_model(raw):
    if "=" not in raw:
        raise ValueError("Model entry must be in label=path format.")
    label, path = raw.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError("Model entry requires non-empty label and path.")
    return {"label": label, "artifact_path": Path(path)}


def token_text(tokenizer, token_id):
    token = tokenizer.id_to_token.get(int(token_id), "")
    return token.replace("\n", "\\n")


def post_divergence_recovery(canonical):
    first_div = canonical["first_divergence_position"]
    rows = canonical["all_rows"]
    if first_div is None or first_div < 0:
        return {
            "has_divergence": False,
            "first_divergence_position": first_div,
            "next_12_match_rate": 1.0,
            "tail_match_rate": 1.0,
            "tail_steps": 0,
        }

    tail_rows = [row for row in rows if row["position"] > first_div]
    window_rows = [row for row in tail_rows if row["position"] <= first_div + 12]
    if not tail_rows:
        return {
            "has_divergence": True,
            "first_divergence_position": first_div,
            "next_12_match_rate": 0.0,
            "tail_match_rate": 0.0,
            "tail_steps": 0,
        }

    next12 = sum(1 for row in window_rows if row["free_matches_gold"]) / len(window_rows)
    tail = sum(1 for row in tail_rows if row["free_matches_gold"]) / len(tail_rows)
    return {
        "has_divergence": True,
        "first_divergence_position": first_div,
        "next_12_match_rate": float(next12),
        "tail_match_rate": float(tail),
        "tail_steps": len(tail_rows),
    }


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


def entropy_from_probs(probabilities):
    clipped = probabilities.clamp(min=1e-12)
    return -torch.sum(clipped * torch.log(clipped), dim=-1)


def fallback_token_ids(by_pos, start_position, end_position, topk):
    counts = Counter()
    for position in range(start_position, end_position + 1):
        row = by_pos[position]
        free_pred = row["free_pred"]
        gold = row["gold"]
        context_diff = ~row["context_same"]
        wrong = context_diff & (free_pred != gold)
        if wrong.any():
            for token in free_pred[wrong].tolist():
                counts[int(token)] += 1
    top = counts.most_common(topk)
    return [token_id for token_id, _ in top]


def mean_masked(values, mask):
    if mask.any():
        return float(values[mask].mean().item())
    return 0.0


def build_position_rows(by_pos, fallback_ids, start_position, end_position):
    rows = []
    for position in range(start_position, end_position + 1):
        row = by_pos[position]
        teacher_probs = row["teacher_probs"]
        free_probs = row["free_probs"]
        teacher_logits = row["teacher_logits"]
        free_logits = row["free_logits"]
        gold = row["gold"]
        teacher_pred = row["teacher_pred"]
        free_pred = row["free_pred"]
        context_same = row["context_same"]
        context_diff = ~context_same

        teacher_gold_p = teacher_probs[torch.arange(gold.shape[0]), gold]
        free_gold_p = free_probs[torch.arange(gold.shape[0]), gold]
        teacher_pred_logit = teacher_logits[torch.arange(gold.shape[0]), teacher_pred]
        free_pred_logit = free_logits[torch.arange(gold.shape[0]), free_pred]
        teacher_gold_logit = teacher_logits[torch.arange(gold.shape[0]), gold]
        free_gold_logit = free_logits[torch.arange(gold.shape[0]), gold]
        teacher_margin = teacher_pred_logit - teacher_gold_logit
        free_margin = free_pred_logit - free_gold_logit
        hidden_delta = torch.norm(row["teacher_hidden"] - row["free_hidden"], dim=-1)
        logit_delta = torch.norm(teacher_logits - free_logits, dim=-1)
        argmax_change = (teacher_pred != free_pred).float()
        teacher_entropy = entropy_from_probs(teacher_probs)
        free_entropy = entropy_from_probs(free_probs)

        fallback_mass = torch.zeros_like(free_entropy)
        if fallback_ids:
            fallback_mass = free_probs[:, fallback_ids].sum(dim=-1)

        rows.append(
            {
                "position": int(position),
                "context_diff_rate": float(context_diff.float().mean().item()),
                "teacher_gold_probability_mean": float(teacher_gold_p.mean().item()),
                "free_gold_probability_mean": float(free_gold_p.mean().item()),
                "free_gold_probability_mean_when_context_diff": mean_masked(
                    free_gold_p, context_diff
                ),
                "teacher_margin_pred_minus_gold_mean": float(teacher_margin.mean().item()),
                "free_margin_pred_minus_gold_mean": float(free_margin.mean().item()),
                "margin_shift_free_minus_teacher_mean": float(
                    (free_margin - teacher_margin).mean().item()
                ),
                "teacher_entropy_mean": float(teacher_entropy.mean().item()),
                "free_entropy_mean": float(free_entropy.mean().item()),
                "entropy_shift_free_minus_teacher_mean": float(
                    (free_entropy - teacher_entropy).mean().item()
                ),
                "hidden_delta_norm_mean": float(hidden_delta.mean().item()),
                "logit_delta_norm_mean": float(logit_delta.mean().item()),
                "argmax_change_rate": float(argmax_change.mean().item()),
                "fallback_probability_mass_mean": float(fallback_mass.mean().item()),
                "fallback_probability_mass_mean_when_context_diff": mean_masked(
                    fallback_mass, context_diff
                ),
            }
        )
    return rows


def heldout_stability(model, tokenizer, validation_token_ids, slices):
    rows = []
    for slice_rec in slices:
        text = tokenizer.decode(
            validation_token_ids[
                slice_rec["start_token"] : slice_rec["end_token_exclusive"]
            ].tolist()
        )
        metric = teacher_forcing_vs_free_running(model, tokenizer, text)
        rows.append(
            {
                "slice_index": slice_rec["slice_index"],
                "free_match_rate_against_target": metric["free_match_rate_against_target"],
                "teacher_top1_accuracy": metric["teacher_top1_accuracy"],
                "first_divergence_position": metric["free_first_divergence_position"],
            }
        )
    free_values = [row["free_match_rate_against_target"] for row in rows]
    return {
        "rows": rows,
        "mean_free_match": float(sum(free_values) / len(free_values)) if free_values else 0.0,
        "min_free_match": float(min(free_values)) if free_values else 0.0,
        "max_free_match": float(max(free_values)) if free_values else 0.0,
        "free_match_range": float(max(free_values) - min(free_values)) if free_values else 0.0,
    }


def build_validation_slices(validation_token_ids, slice_count):
    total = int(validation_token_ids.shape[0])
    actual = max(1, min(slice_count, total))
    base = total // actual
    rem = total % actual
    slices = []
    start = 0
    for idx in range(actual):
        width = base + (1 if idx < rem else 0)
        end = start + width
        slices.append(
            {
                "slice_index": idx,
                "start_token": int(start),
                "end_token_exclusive": int(end),
            }
        )
        start = end
    return slices


def analyze_model(
    spec,
    tokenizer,
    validation_text,
    validation_token_ids,
    sample_seed,
    sample_count,
    sample_batch_size,
    position_start,
    position_end,
    fallback_topk,
    heldout_slices,
):
    model = load_model_for_suite(spec["artifact_path"])
    model.eval()

    eval_length = max(80, position_end + 1)
    windows, actual_count = sample_windows(
        validation_token_ids=validation_token_ids,
        eval_length=eval_length,
        sample_count=sample_count,
        seed=sample_seed,
    )
    by_pos = collect_teacher_free_states(
        model=model,
        windows=windows,
        prompt_length=40,
        position_start=position_start,
        position_end=position_end,
        batch_size=sample_batch_size,
    )

    fallback_ids = fallback_token_ids(
        by_pos=by_pos,
        start_position=max(42, position_start),
        end_position=position_end,
        topk=fallback_topk,
    )
    position_rows = build_position_rows(
        by_pos=by_pos,
        fallback_ids=fallback_ids,
        start_position=position_start,
        end_position=position_end,
    )

    suite = metrics_snapshot(model, tokenizer, validation_text)
    suite_row = metric_row(suite)
    canonical = canonical_trajectory_analysis(
        model=model,
        tokenizer=tokenizer,
        validation_text=validation_text,
        prompt_length=40,
        eval_length=80,
    )
    recovery = post_divergence_recovery(canonical)
    heldout = heldout_stability(
        model=model,
        tokenizer=tokenizer,
        validation_token_ids=validation_token_ids,
        slices=heldout_slices,
    )

    return {
        "label": spec["label"],
        "artifact_path": str(spec["artifact_path"].resolve()),
        "sample_count_used": int(actual_count),
        "suite_row": suite_row,
        "post_divergence_recovery": recovery,
        "fallback_tokens": [
            {"token_id": int(token_id), "token_text": token_text(tokenizer, token_id)}
            for token_id in fallback_ids
        ],
        "position_rows": position_rows,
        "heldout_stability": heldout,
    }


def map_rows(rows):
    return {row["position"]: row for row in rows}


def diff_row(candidate_row, leader_row, position):
    return {
        "position": int(position),
        "delta_free_gold_probability_mean": candidate_row["free_gold_probability_mean"]
        - leader_row["free_gold_probability_mean"],
        "delta_margin_shift_free_minus_teacher_mean": candidate_row[
            "margin_shift_free_minus_teacher_mean"
        ]
        - leader_row["margin_shift_free_minus_teacher_mean"],
        "delta_free_entropy_mean": candidate_row["free_entropy_mean"]
        - leader_row["free_entropy_mean"],
        "delta_hidden_delta_norm_mean": candidate_row["hidden_delta_norm_mean"]
        - leader_row["hidden_delta_norm_mean"],
        "delta_logit_delta_norm_mean": candidate_row["logit_delta_norm_mean"]
        - leader_row["logit_delta_norm_mean"],
        "delta_fallback_probability_mass_mean_when_context_diff": candidate_row[
            "fallback_probability_mass_mean_when_context_diff"
        ]
        - leader_row["fallback_probability_mass_mean_when_context_diff"],
        "delta_argmax_change_rate": candidate_row["argmax_change_rate"]
        - leader_row["argmax_change_rate"],
    }


def render_report(results):
    lines = []
    lines.append("OUTREACHLM ROLLOUT-CALIBRATION REGIME ANALYSIS")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {results['timestamp']}")
    lines.append("")
    cfg = results["config"]
    lines.append(f"Leader: {cfg['leader']['label']}={cfg['leader']['artifact_path']}")
    lines.append(
        "Candidates: "
        + ", ".join(f"{rec['label']}={rec['artifact_path']}" for rec in cfg["candidates"])
    )
    lines.append(
        f"Positions analyzed: {cfg['position_start']}..{cfg['position_end']} | sample_count={cfg['sample_count']} seed={cfg['sample_seed']}"
    )
    lines.append("")

    leader = results["leader"]
    lines.append("LEADER SNAPSHOT")
    lines.append("-" * 80)
    lines.append(
        "teacher_top1={t:.4f} free_match={f:.4f} first_div={d} post_div_next12={r:.4f}".format(
            t=leader["suite_row"]["teacher_top1"],
            f=leader["suite_row"]["free_match"],
            d=leader["suite_row"]["first_free_divergence"],
            r=leader["post_divergence_recovery"]["next_12_match_rate"],
        )
    )
    lines.append("")

    lines.append("CANDIDATE SNAPSHOT")
    lines.append("-" * 80)
    lines.append(
        "label,teacher_top1,free_match,first_div,post_div_next12,heldout_mean,heldout_range"
    )
    for rec in results["candidates"]:
        lines.append(
            "{l},{t:.6f},{f:.6f},{d},{r:.6f},{h:.6f},{hr:.6f}".format(
                l=rec["label"],
                t=rec["suite_row"]["teacher_top1"],
                f=rec["suite_row"]["free_match"],
                d=rec["suite_row"]["first_free_divergence"],
                r=rec["post_divergence_recovery"]["next_12_match_rate"],
                h=rec["heldout_stability"]["mean_free_match"],
                hr=rec["heldout_stability"]["free_match_range"],
            )
        )
    lines.append("")

    lines.append("POSITION-LEVEL DELTAS VS LEADER (candidate - leader)")
    lines.append("-" * 80)
    for comp in results["comparisons"]:
        lines.append(f"[{comp['candidate_label']}]")
        lines.append(
            "pos,delta_free_gold_p,delta_margin_shift,delta_free_entropy,"
            "delta_hidden_move,delta_logit_move,delta_fallback_mass_ctx_diff,delta_argmax_change"
        )
        for row in comp["position_deltas"]:
            lines.append(
                "{p},{a:.6f},{b:.6f},{c:.6f},{d:.6f},{e:.6f},{f:.6f},{g:.6f}".format(
                    p=row["position"],
                    a=row["delta_free_gold_probability_mean"],
                    b=row["delta_margin_shift_free_minus_teacher_mean"],
                    c=row["delta_free_entropy_mean"],
                    d=row["delta_hidden_delta_norm_mean"],
                    e=row["delta_logit_delta_norm_mean"],
                    f=row["delta_fallback_probability_mass_mean_when_context_diff"],
                    g=row["delta_argmax_change_rate"],
                )
            )
    lines.append("")

    lines.append("CROSS-SEED STABILITY (CANDIDATES)")
    lines.append("-" * 80)
    st = results["candidate_stability"]
    lines.append(
        "free_match mean/std/min/max = "
        f"{st['free_match_mean']:.6f}/{st['free_match_std']:.6f}/{st['free_match_min']:.6f}/{st['free_match_max']:.6f}"
    )
    lines.append(
        "post_div_next12 mean/std = "
        f"{st['post_div_next12_mean']:.6f}/{st['post_div_next12_std']:.6f}"
    )
    lines.append(
        "heldout_mean_free mean/std = "
        f"{st['heldout_mean_free_mean']:.6f}/{st['heldout_mean_free_std']:.6f}"
    )
    return "\n".join(lines)


def std(values):
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return float((sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze rollout-calibration regime: compare V2 leader vs failed models "
            "around position-41 transition and post-divergence behavior."
        )
    )
    parser.add_argument(
        "--leader",
        type=str,
        default="leader=experiments/v2-divergence-intervention-20260816-113809.pt",
        help="Leader model in label=path format.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate model(s) in label=path format. Repeat for multiple seeds/models.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--report-prefix", type=str, default="rollout-calibration-analysis")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--sample-count", type=int, default=4096)
    parser.add_argument("--sample-batch-size", type=int, default=256)
    parser.add_argument("--position-start", type=int, default=40)
    parser.add_argument("--position-end", type=int, default=52)
    parser.add_argument("--fallback-topk", type=int, default=5)
    parser.add_argument("--heldout-slices", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.position_start < 1:
        raise ValueError("position-start must be >= 1.")
    if args.position_end < args.position_start:
        raise ValueError("position-end must be >= position-start.")
    if args.sample_count < 1:
        raise ValueError("sample-count must be >= 1.")
    if args.heldout_slices < 1:
        raise ValueError("heldout-slices must be >= 1.")

    leader_spec = parse_model(args.leader)
    candidate_specs = [parse_model(raw) for raw in args.candidate]
    labels = [leader_spec["label"]] + [rec["label"] for rec in candidate_specs]
    if len(set(labels)) != len(labels):
        raise ValueError("All labels (leader + candidates) must be unique.")

    tokenizer = load_tokenizer_artifact(TOKENIZER_PATH)
    if tokenizer is None:
        tokenizer = upgrade_legacy_tokenizer_artifact(TOKENIZER_PATH)

    text = load_corpus(CORPUS_PATH)
    _, validation_text = split_corpus(text, VALIDATION_SPLIT)
    validation_token_ids = torch.tensor(tokenizer.encode(validation_text), dtype=torch.long)
    heldout_slices = build_validation_slices(
        validation_token_ids=validation_token_ids,
        slice_count=args.heldout_slices,
    )

    leader_analysis = analyze_model(
        spec=leader_spec,
        tokenizer=tokenizer,
        validation_text=validation_text,
        validation_token_ids=validation_token_ids,
        sample_seed=args.sample_seed,
        sample_count=args.sample_count,
        sample_batch_size=args.sample_batch_size,
        position_start=args.position_start,
        position_end=args.position_end,
        fallback_topk=args.fallback_topk,
        heldout_slices=heldout_slices,
    )
    candidate_analyses = [
        analyze_model(
            spec=spec,
            tokenizer=tokenizer,
            validation_text=validation_text,
            validation_token_ids=validation_token_ids,
            sample_seed=args.sample_seed,
            sample_count=args.sample_count,
            sample_batch_size=args.sample_batch_size,
            position_start=args.position_start,
            position_end=args.position_end,
            fallback_topk=args.fallback_topk,
            heldout_slices=heldout_slices,
        )
        for spec in candidate_specs
    ]

    leader_by_pos = map_rows(leader_analysis["position_rows"])
    comparisons = []
    for candidate in candidate_analyses:
        candidate_by_pos = map_rows(candidate["position_rows"])
        deltas = []
        for position in range(args.position_start, args.position_end + 1):
            deltas.append(
                diff_row(
                    candidate_row=candidate_by_pos[position],
                    leader_row=leader_by_pos[position],
                    position=position,
                )
            )
        comparisons.append(
            {
                "candidate_label": candidate["label"],
                "position_deltas": deltas,
            }
        )

    free_matches = [rec["suite_row"]["free_match"] for rec in candidate_analyses]
    post_recovery = [
        rec["post_divergence_recovery"]["next_12_match_rate"] for rec in candidate_analyses
    ]
    heldout_means = [rec["heldout_stability"]["mean_free_match"] for rec in candidate_analyses]
    candidate_stability = {
        "free_match_mean": float(sum(free_matches) / len(free_matches)) if free_matches else 0.0,
        "free_match_std": std(free_matches),
        "free_match_min": float(min(free_matches)) if free_matches else 0.0,
        "free_match_max": float(max(free_matches)) if free_matches else 0.0,
        "post_div_next12_mean": float(sum(post_recovery) / len(post_recovery))
        if post_recovery
        else 0.0,
        "post_div_next12_std": std(post_recovery),
        "heldout_mean_free_mean": float(sum(heldout_means) / len(heldout_means))
        if heldout_means
        else 0.0,
        "heldout_mean_free_std": std(heldout_means),
    }

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "leader": {
                "label": leader_spec["label"],
                "artifact_path": str(leader_spec["artifact_path"].resolve()),
            },
            "candidates": [
                {
                    "label": rec["label"],
                    "artifact_path": str(rec["artifact_path"].resolve()),
                }
                for rec in candidate_specs
            ],
            "sample_seed": args.sample_seed,
            "sample_count": args.sample_count,
            "sample_batch_size": args.sample_batch_size,
            "position_start": args.position_start,
            "position_end": args.position_end,
            "fallback_topk": args.fallback_topk,
            "heldout_slices": args.heldout_slices,
        },
        "leader": leader_analysis,
        "candidates": candidate_analyses,
        "comparisons": comparisons,
        "candidate_stability": candidate_stability,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = args.output_dir / f"{args.report_prefix}-{stamp}.json"
    txt_path = args.output_dir / f"{args.report_prefix}-{stamp}.txt"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(render_report(results))

    print(str(json_path.resolve()))
    print(str(txt_path.resolve()))


if __name__ == "__main__":
    main()

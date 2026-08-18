import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from outreachlm.architecture_capacity_continuation import load_model_from_artifact
from outreachlm.architecture_capacity_pilot import metric_row, metrics_snapshot
from outreachlm.divergence_trajectory_mapping_analysis import (
    canonical_trajectory_analysis,
    systematic_boundary_mapping,
)
from outreachlm.generate import (
    TOKENIZER_PATH,
    load_tokenizer_artifact,
    upgrade_legacy_tokenizer_artifact,
)
from outreachlm.hidden_output_transition_tests import (
    collect_teacher_free_states,
    sample_windows,
    test1_hidden_transition,
    test2_output_head_sensitivity,
)
from outreachlm.objective_intervention_experiment import teacher_forcing_vs_free_running
from outreachlm.train import CORPUS_PATH, VALIDATION_SPLIT, load_corpus, split_corpus
from outreachlm.v4_generate import load_model_and_tokenizer as load_v4_model_and_tokenizer


def parse_candidate(raw):
    if "=" not in raw:
        raise ValueError(
            "Each --candidate must be in 'label=path/to/artifact.pt' format."
        )
    label, path_raw = raw.split("=", 1)
    label = label.strip()
    path_raw = path_raw.strip()
    if not label:
        raise ValueError("Candidate label cannot be empty.")
    if not path_raw:
        raise ValueError("Candidate path cannot be empty.")
    return {
        "label": label,
        "artifact_path": Path(path_raw),
    }


def build_validation_slices(tokenizer, validation_token_ids, requested_slices):
    min_tokens_per_slice = 240
    total = int(validation_token_ids.shape[0])
    max_slices = max(1, total // min_tokens_per_slice)
    actual_slices = max(1, min(requested_slices, max_slices))

    base = total // actual_slices
    remainder = total % actual_slices
    slices = []
    start = 0
    for index in range(actual_slices):
        span = base + (1 if index < remainder else 0)
        end = start + span
        token_window = validation_token_ids[start:end]
        text = tokenizer.decode(token_window.tolist())
        slices.append(
            {
                "slice_index": index,
                "start_token": int(start),
                "end_token_exclusive": int(end),
                "token_count": int(span),
                "text": text,
            }
        )
        start = end
    return slices


def evaluate_heldout_slices(model, tokenizer, slices):
    rows = []
    for item in slices:
        metric = teacher_forcing_vs_free_running(model, tokenizer, item["text"])
        rows.append(
            {
                "slice_index": item["slice_index"],
                "start_token": item["start_token"],
                "end_token_exclusive": item["end_token_exclusive"],
                "token_count": item["token_count"],
                "sequence_source": metric["sequence_source"],
                "teacher_top1_accuracy": metric["teacher_top1_accuracy"],
                "free_match_rate_against_target": metric["free_match_rate_against_target"],
                "free_first_divergence_position": metric["free_first_divergence_position"],
            }
        )

    free_values = [row["free_match_rate_against_target"] for row in rows]
    teacher_values = [row["teacher_top1_accuracy"] for row in rows]

    return {
        "slice_count": len(rows),
        "rows": rows,
        "summary": {
            "mean_free_match": float(sum(free_values) / len(free_values))
            if free_values
            else 0.0,
            "min_free_match": float(min(free_values)) if free_values else 0.0,
            "max_free_match": float(max(free_values)) if free_values else 0.0,
            "mean_teacher_top1": float(sum(teacher_values) / len(teacher_values))
            if teacher_values
            else 0.0,
        },
    }


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


def row_for_position(rows, position):
    for row in rows:
        if row["position"] == position:
            return row
    return None


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


def evaluate_model_bundle(
    artifact_path,
    tokenizer,
    validation_text,
    validation_token_ids,
    heldout_slices,
    systematic_seed,
    systematic_sample_count,
    systematic_batch_size,
):
    model = load_model_for_suite(artifact_path)
    model.eval()

    suite_metrics = metrics_snapshot(model, tokenizer, validation_text)
    canonical = canonical_trajectory_analysis(
        model=model,
        tokenizer=tokenizer,
        validation_text=validation_text,
        prompt_length=40,
        eval_length=80,
    )
    systematic = systematic_boundary_mapping(
        model=model,
        tokenizer=tokenizer,
        validation_token_ids=validation_token_ids,
        prompt_length=40,
        eval_length=80,
        sample_count=systematic_sample_count,
        seed=systematic_seed,
        batch_size=systematic_batch_size,
    )

    windows, actual_count = sample_windows(
        validation_token_ids=validation_token_ids,
        eval_length=80,
        sample_count=systematic_sample_count,
        seed=systematic_seed,
    )
    by_pos = collect_teacher_free_states(
        model=model,
        windows=windows,
        prompt_length=40,
        position_start=38,
        position_end=45,
        batch_size=systematic_batch_size,
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
        topk=5,
    )
    heldout = evaluate_heldout_slices(
        model=model,
        tokenizer=tokenizer,
        slices=heldout_slices,
    )

    recovery = post_divergence_recovery(canonical)
    pos41_test2 = row_for_position(test2["rows"], 41)

    return {
        "artifact_path": str(artifact_path.resolve()),
        "suite_metrics": suite_metrics,
        "suite_row": metric_row(suite_metrics),
        "canonical_trajectory": canonical,
        "systematic_boundary": systematic,
        "tests_1_2": {
            "sample_count_used": actual_count,
            "test_1_hidden_transition": test1,
            "test_2_output_sensitivity": test2,
        },
        "heldout_slices": heldout,
        "position_41_summary": {
            "systematic_free_match_rate": systematic["position_41_free_match_rate"],
            "systematic_free_match_rate_when_context_diff": systematic[
                "position_41_free_match_rate_when_context_diff"
            ],
            "systematic_logit_cos_when_context_diff": systematic[
                "logit_cos_teacher_vs_free_mean_when_context_diff"
            ],
            "test2_context_diff_rate": pos41_test2["context_diff_rate"] if pos41_test2 else None,
            "test2_free_gold_probability_mean_when_context_diff": (
                pos41_test2["free_gold_probability_mean_when_context_diff"]
                if pos41_test2
                else None
            ),
        },
        "post_divergence_recovery": recovery,
    }


def gate_candidates(
    leader_bundle,
    candidate_bundles,
    min_seeds,
    required_margin,
    minimum_free_match,
    require_not_below_leader,
    require_divergence_or_recovery,
    minimum_post_divergence_next12_delta,
):
    leader_free = leader_bundle["suite_row"]["free_match"]
    leader_heldout = leader_bundle["heldout_slices"]["summary"]["mean_free_match"]
    leader_first_divergence = leader_bundle["suite_row"]["first_free_divergence"]
    leader_post_div_next12 = leader_bundle["post_divergence_recovery"]["next_12_match_rate"]

    candidate_rows = []
    seed_passes = []
    heldout_passes = []
    threshold_passes = []
    not_below_leader_passes = []
    divergence_or_recovery_passes = []
    for candidate in candidate_bundles:
        row = candidate["suite_row"]
        heldout_mean = candidate["heldout_slices"]["summary"]["mean_free_match"]
        free_match = row["free_match"]
        seed_pass = free_match > (leader_free + required_margin)
        heldout_pass = heldout_mean > (leader_heldout + required_margin)
        threshold_pass = (
            free_match > minimum_free_match
            if minimum_free_match is not None
            else True
        )
        not_below_leader_pass = free_match >= leader_free
        first_divergence = row["first_free_divergence"]
        post_div_next12 = candidate["post_divergence_recovery"]["next_12_match_rate"]
        divergence_or_recovery_pass = (
            (first_divergence is not None and first_divergence > leader_first_divergence)
            or (post_div_next12 >= (leader_post_div_next12 + minimum_post_divergence_next12_delta))
        )

        candidate_rows.append(
            {
                "label": candidate["label"],
                "artifact_path": candidate["artifact_path"],
                "teacher_top1": row["teacher_top1"],
                "free_match": free_match,
                "first_free_divergence": first_divergence,
                "heldout_mean_free_match": heldout_mean,
                "position_41_free_match_rate_when_context_diff": candidate[
                    "position_41_summary"
                ]["systematic_free_match_rate_when_context_diff"],
                "post_divergence_next12_match_rate": post_div_next12,
                "seed_pass": seed_pass,
                "heldout_pass": heldout_pass,
                "minimum_free_match_pass": threshold_pass,
                "not_below_leader_pass": not_below_leader_pass,
                "divergence_or_recovery_pass": divergence_or_recovery_pass,
            }
        )
        seed_passes.append(seed_pass)
        heldout_passes.append(heldout_pass)
        threshold_passes.append(threshold_pass)
        not_below_leader_passes.append(not_below_leader_pass)
        divergence_or_recovery_passes.append(divergence_or_recovery_pass)

    free_values = [row["free_match"] for row in candidate_rows]
    heldout_values = [row["heldout_mean_free_match"] for row in candidate_rows]
    post_div_values = [
        row["post_divergence_next12_match_rate"] for row in candidate_rows
    ]
    pos41_values = [
        row["position_41_free_match_rate_when_context_diff"] for row in candidate_rows
    ]

    seed_count = len(candidate_rows)
    minimum_seed_count_pass = seed_count >= min_seeds
    all_seed_pass = all(seed_passes) if seed_passes else False
    mean_free_match = float(sum(free_values) / len(free_values)) if free_values else 0.0
    mean_pass = mean_free_match > (leader_free + required_margin)
    all_heldout_pass = all(heldout_passes) if heldout_passes else False
    all_threshold_pass = all(threshold_passes) if threshold_passes else True
    all_not_below_leader_pass = (
        all(not_below_leader_passes) if not_below_leader_passes else True
    )
    all_divergence_or_recovery_pass = (
        all(divergence_or_recovery_passes) if divergence_or_recovery_passes else True
    )

    core_gate_pass = minimum_seed_count_pass and all_seed_pass and mean_pass
    promotion_pass = core_gate_pass and all_heldout_pass
    if minimum_free_match is not None:
        promotion_pass = promotion_pass and all_threshold_pass
    if require_not_below_leader:
        promotion_pass = promotion_pass and all_not_below_leader_pass
    if require_divergence_or_recovery:
        promotion_pass = promotion_pass and all_divergence_or_recovery_pass

    reasons = []
    if not minimum_seed_count_pass:
        reasons.append(
            f"Only {seed_count} candidate seed(s); protocol requires >= {min_seeds}."
        )
    if minimum_seed_count_pass and not all_seed_pass:
        failing = [row["label"] for row in candidate_rows if not row["seed_pass"]]
        reasons.append(
            "Not all seeds beat leader free-match threshold; failing seeds: "
            + ", ".join(failing)
        )
    if minimum_seed_count_pass and not mean_pass:
        reasons.append(
            "Mean free-match across seeds does not beat leader threshold."
        )
    if core_gate_pass and not all_heldout_pass:
        failing = [row["label"] for row in candidate_rows if not row["heldout_pass"]]
        reasons.append(
            "Held-out slice check failed for seed(s): " + ", ".join(failing)
        )
    if minimum_free_match is not None and not all_threshold_pass:
        failing = [
            row["label"] for row in candidate_rows if not row["minimum_free_match_pass"]
        ]
        reasons.append(
            f"Not all seeds beat minimum free-match threshold {minimum_free_match:.6f}; failing seeds: "
            + ", ".join(failing)
        )
    if require_not_below_leader and not all_not_below_leader_pass:
        failing = [
            row["label"] for row in candidate_rows if not row["not_below_leader_pass"]
        ]
        reasons.append(
            "Some seeds fell below leader free-match; failing seeds: " + ", ".join(failing)
        )
    if require_divergence_or_recovery and not all_divergence_or_recovery_pass:
        failing = [
            row["label"] for row in candidate_rows if not row["divergence_or_recovery_pass"]
        ]
        reasons.append(
            "Some seeds did not improve first divergence and did not improve post-divergence recovery; failing seeds: "
            + ", ".join(failing)
        )
    if not reasons:
        reasons.append("All promotion gate conditions passed.")

    return {
        "leader_free_match": leader_free,
        "leader_heldout_mean_free_match": leader_heldout,
        "leader_first_free_divergence": leader_first_divergence,
        "leader_post_divergence_next12_match_rate": leader_post_div_next12,
        "required_margin": required_margin,
        "minimum_free_match": minimum_free_match,
        "require_not_below_leader": require_not_below_leader,
        "require_divergence_or_recovery": require_divergence_or_recovery,
        "minimum_post_divergence_next12_delta": minimum_post_divergence_next12_delta,
        "candidate_seed_count": seed_count,
        "candidate_rows": candidate_rows,
        "aggregates": {
            "candidate_mean_free_match": mean_free_match,
            "candidate_min_free_match": float(min(free_values)) if free_values else 0.0,
            "candidate_max_free_match": float(max(free_values)) if free_values else 0.0,
            "candidate_mean_heldout_free_match": float(
                sum(heldout_values) / len(heldout_values)
            )
            if heldout_values
            else 0.0,
            "candidate_mean_post_divergence_next12_match_rate": float(
                sum(post_div_values) / len(post_div_values)
            )
            if post_div_values
            else 0.0,
            "candidate_mean_pos41_free_match_context_diff": float(
                sum(pos41_values) / len(pos41_values)
            )
            if pos41_values
            else 0.0,
        },
        "checks": {
            "minimum_seed_count_pass": minimum_seed_count_pass,
            "all_seed_free_match_pass": all_seed_pass,
            "mean_free_match_pass": mean_pass,
            "all_heldout_pass": all_heldout_pass,
            "all_minimum_free_match_pass": all_threshold_pass,
            "all_not_below_leader_pass": all_not_below_leader_pass,
            "all_divergence_or_recovery_pass": all_divergence_or_recovery_pass,
        },
        "promotion_pass": promotion_pass,
        "decision_reasons": reasons,
    }


def render_report(results):
    lines = []
    lines.append("OUTREACHLM LEADER GATING PROTOCOL REPORT")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {results['timestamp']}")
    lines.append("")
    cfg = results["config"]
    lines.append(f"Leader artifact: {cfg['leader_artifact']}")
    lines.append(
        "Candidates: "
        + ", ".join(
            f"{item['label']}={item['artifact_path']}" for item in cfg["candidates"]
        )
    )
    lines.append(
        f"Gate: min_seeds={cfg['min_seeds']} required_margin={cfg['required_free_match_margin']} "
        f"minimum_free_match={cfg['minimum_free_match']} require_not_below_leader={cfg['require_not_below_leader']} "
        f"require_divergence_or_recovery={cfg['require_divergence_or_recovery']}"
    )
    lines.append(
        f"Held-out slices={cfg['heldout_slices']} | boundary sample_count={cfg['systematic_sample_count']} seed={cfg['systematic_seed']}"
    )
    lines.append("")

    leader = results["leader"]
    leader_row = leader["suite_row"]
    lines.append("LEADER SNAPSHOT")
    lines.append("-" * 80)
    lines.append(
        "teacher_top1={t:.4f} free_match={f:.4f} first_div={d} heldout_mean_free={h:.4f}".format(
            t=leader_row["teacher_top1"],
            f=leader_row["free_match"],
            d=leader_row["first_free_divergence"],
            h=leader["heldout_slices"]["summary"]["mean_free_match"],
        )
    )
    lines.append(
        "leader post-divergence next12 match={r:.4f}".format(
            r=leader["post_divergence_recovery"]["next_12_match_rate"],
        )
    )
    lines.append("")

    lines.append("CANDIDATE SNAPSHOT")
    lines.append("-" * 80)
    lines.append(
        "label,teacher_top1,free_match,first_free_divergence,heldout_mean_free,"
        "pos41_free_match_ctx_diff,post_div_next12,seed_pass,heldout_pass,"
        "min_free_pass,not_below_leader_pass,divergence_or_recovery_pass"
    )
    for row in results["gate"]["candidate_rows"]:
        lines.append(
            "{label},{t:.6f},{f:.6f},{d},{h:.6f},{p:.6f},{r:.6f},{sp},{hp},{mfp},{nbl},{dor}".format(
                label=row["label"],
                t=row["teacher_top1"],
                f=row["free_match"],
                d=row["first_free_divergence"],
                h=row["heldout_mean_free_match"],
                p=row["position_41_free_match_rate_when_context_diff"],
                r=row["post_divergence_next12_match_rate"],
                sp=row["seed_pass"],
                hp=row["heldout_pass"],
                mfp=row["minimum_free_match_pass"],
                nbl=row["not_below_leader_pass"],
                dor=row["divergence_or_recovery_pass"],
            )
        )
    lines.append("")

    lines.append("GATE DECISION")
    lines.append("-" * 80)
    lines.append(f"promotion_pass={results['gate']['promotion_pass']}")
    for key, value in results["gate"]["checks"].items():
        lines.append(f"{key}={value}")
    agg = results["gate"]["aggregates"]
    lines.append(
        "candidate_mean_free_match={m:.6f} min={mn:.6f} max={mx:.6f}".format(
            m=agg["candidate_mean_free_match"],
            mn=agg["candidate_min_free_match"],
            mx=agg["candidate_max_free_match"],
        )
    )
    lines.append(
        "candidate_mean_heldout_free_match={h:.6f} "
        "candidate_mean_pos41_ctx_diff={p:.6f} "
        "candidate_mean_post_div_next12={r:.6f}".format(
            h=agg["candidate_mean_heldout_free_match"],
            p=agg["candidate_mean_pos41_free_match_context_diff"],
            r=agg["candidate_mean_post_divergence_next12_match_rate"],
        )
    )
    lines.append("reasons:")
    for reason in results["gate"]["decision_reasons"]:
        lines.append(f"- {reason}")

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run OutreachLM leader-gating protocol: multi-seed free-match gate + "
            "trajectory/position-41/post-divergence/generalization diagnostics."
        )
    )
    parser.add_argument(
        "--leader-artifact",
        type=Path,
        default=Path("experiments/v2-divergence-intervention-20260816-113809.pt"),
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate model in label=artifact.pt format. Repeat per seed.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--report-prefix", type=str, default="leader-gate")
    parser.add_argument("--min-seeds", type=int, default=2)
    parser.add_argument("--required-free-match-margin", type=float, default=0.0)
    parser.add_argument(
        "--minimum-free-match",
        type=float,
        default=None,
        help="Optional absolute per-seed floor for free_match (strictly greater-than).",
    )
    parser.add_argument(
        "--require-not-below-leader",
        action="store_true",
        help="Require every seed to have free_match >= leader free_match.",
    )
    parser.add_argument(
        "--require-divergence-or-recovery",
        action="store_true",
        help=(
            "Require every seed to either improve first divergence position or "
            "improve post-divergence next-12-step recovery."
        ),
    )
    parser.add_argument(
        "--minimum-post-divergence-next12-delta",
        type=float,
        default=0.0,
        help=(
            "Minimum required improvement over leader for post-divergence "
            "next-12-step match rate when divergence position is not improved."
        ),
    )
    parser.add_argument("--heldout-slices", type=int, default=4)
    parser.add_argument("--systematic-seed", type=int, default=42)
    parser.add_argument("--systematic-sample-count", type=int, default=4096)
    parser.add_argument("--systematic-batch-size", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.min_seeds < 1:
        raise ValueError("min-seeds must be >= 1.")
    if args.heldout_slices < 1:
        raise ValueError("heldout-slices must be >= 1.")

    candidate_specs = [parse_candidate(raw) for raw in args.candidate]
    labels = [item["label"] for item in candidate_specs]
    if len(set(labels)) != len(labels):
        raise ValueError("Candidate labels must be unique.")

    tokenizer = load_tokenizer_artifact(TOKENIZER_PATH)
    if tokenizer is None:
        tokenizer = upgrade_legacy_tokenizer_artifact(TOKENIZER_PATH)

    text = load_corpus(CORPUS_PATH)
    _, validation_text = split_corpus(text, VALIDATION_SPLIT)
    validation_token_ids = torch.tensor(tokenizer.encode(validation_text), dtype=torch.long)
    heldout_slices = build_validation_slices(
        tokenizer=tokenizer,
        validation_token_ids=validation_token_ids,
        requested_slices=args.heldout_slices,
    )

    leader_bundle = evaluate_model_bundle(
        artifact_path=args.leader_artifact,
        tokenizer=tokenizer,
        validation_text=validation_text,
        validation_token_ids=validation_token_ids,
        heldout_slices=heldout_slices,
        systematic_seed=args.systematic_seed,
        systematic_sample_count=args.systematic_sample_count,
        systematic_batch_size=args.systematic_batch_size,
    )

    candidate_bundles = []
    for spec in candidate_specs:
        bundle = evaluate_model_bundle(
            artifact_path=spec["artifact_path"],
            tokenizer=tokenizer,
            validation_text=validation_text,
            validation_token_ids=validation_token_ids,
            heldout_slices=heldout_slices,
            systematic_seed=args.systematic_seed,
            systematic_sample_count=args.systematic_sample_count,
            systematic_batch_size=args.systematic_batch_size,
        )
        bundle["label"] = spec["label"]
        candidate_bundles.append(bundle)

    gate = gate_candidates(
        leader_bundle=leader_bundle,
        candidate_bundles=candidate_bundles,
        min_seeds=args.min_seeds,
        required_margin=args.required_free_match_margin,
        minimum_free_match=args.minimum_free_match,
        require_not_below_leader=args.require_not_below_leader,
        require_divergence_or_recovery=args.require_divergence_or_recovery,
        minimum_post_divergence_next12_delta=args.minimum_post_divergence_next12_delta,
    )

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "leader_artifact": str(args.leader_artifact.resolve()),
            "candidates": [
                {
                    "label": item["label"],
                    "artifact_path": str(item["artifact_path"].resolve()),
                }
                for item in candidate_specs
            ],
            "min_seeds": args.min_seeds,
            "required_free_match_margin": args.required_free_match_margin,
            "minimum_free_match": args.minimum_free_match,
            "require_not_below_leader": args.require_not_below_leader,
            "require_divergence_or_recovery": args.require_divergence_or_recovery,
            "minimum_post_divergence_next12_delta": args.minimum_post_divergence_next12_delta,
            "heldout_slices": len(heldout_slices),
            "heldout_slice_boundaries": [
                {
                    "slice_index": item["slice_index"],
                    "start_token": item["start_token"],
                    "end_token_exclusive": item["end_token_exclusive"],
                    "token_count": item["token_count"],
                }
                for item in heldout_slices
            ],
            "systematic_seed": args.systematic_seed,
            "systematic_sample_count": args.systematic_sample_count,
            "systematic_batch_size": args.systematic_batch_size,
        },
        "leader": leader_bundle,
        "candidates": candidate_bundles,
        "gate": gate,
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

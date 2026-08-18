import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from outreachlm.architecture_capacity_continuation import load_model_from_artifact
from outreachlm.architecture_capacity_pilot import metric_row, metrics_snapshot
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


def render_report(results):
    lines = []
    lines.append("OUTREACHLM EVALUATION SUITE COMPARISON")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {results['timestamp']}")
    lines.append("")
    lines.append(f"Leader artifact: {results['config']['leader_artifact']}")
    lines.append(f"Candidate artifact: {results['config']['candidate_artifact']}")
    lines.append("")
    lines.append(
        "condition,teacher_top1,free_match,prompt_logit_cosine,rollout_mean_entropy,"
        "first_repeated_bigram_step,first_repeated_trigram_step,first_free_divergence"
    )
    for condition in ["leader", "candidate"]:
        row = results["summary_table"][condition]
        lines.append(
            "{c},{t:.6f},{f:.6f},{p:.6f},{e:.6f},{b},{g},{d}".format(
                c=condition,
                t=row["teacher_top1"],
                f=row["free_match"],
                p=row["prompt_logit_cosine"],
                e=row["rollout_mean_entropy"],
                b=row["first_repeated_bigram_step"],
                g=row["first_repeated_trigram_step"],
                d=row["first_free_divergence"],
            )
        )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two model artifacts on OutreachLM evaluation suite."
    )
    parser.add_argument(
        "--leader-artifact",
        type=Path,
        default=Path("experiments/v2-divergence-intervention-20260816-113809.pt"),
    )
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    return parser.parse_args()


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


def main():
    args = parse_args()

    tokenizer = load_tokenizer_artifact(TOKENIZER_PATH)
    if tokenizer is None:
        tokenizer = upgrade_legacy_tokenizer_artifact(TOKENIZER_PATH)

    text = load_corpus(CORPUS_PATH)
    _, validation_text = split_corpus(text, VALIDATION_SPLIT)

    leader_model = load_model_for_suite(args.leader_artifact)
    leader_model.eval()
    candidate_model = load_model_for_suite(args.candidate_artifact)
    candidate_model.eval()

    leader_metrics = metrics_snapshot(leader_model, tokenizer, validation_text)
    candidate_metrics = metrics_snapshot(candidate_model, tokenizer, validation_text)

    summary_table = {
        "leader": metric_row(leader_metrics),
        "candidate": metric_row(candidate_metrics),
    }

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "leader_artifact": str(args.leader_artifact.resolve()),
            "candidate_artifact": str(args.candidate_artifact.resolve()),
        },
        "leader_metrics": leader_metrics,
        "candidate_metrics": candidate_metrics,
        "summary_table": summary_table,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = args.output_dir / f"eval-compare-{stamp}.json"
    txt_path = args.output_dir / f"eval-compare-{stamp}.txt"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(render_report(results))

    print(str(json_path.resolve()))
    print(str(txt_path.resolve()))


if __name__ == "__main__":
    main()

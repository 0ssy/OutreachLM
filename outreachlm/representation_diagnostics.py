import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import torch

from outreachlm.generate import load_model_and_tokenizer


DEFAULT_CONTEXT_PROMPTS = [
    "Machine",
    "Machine learning",
    "Machine learning allows",
    "Machine learning allows computers",
]

DEFAULT_UNRELATED_PROMPTS = [
    "The purpose of a transformer",
    "A computer system can",
    "OutreachLM is",
]

DEFAULT_ENTROPY_PROMPTS = [
    "OutreachLM is",
    "The purpose of a transformer is",
    "Machine learning allows",
    "A computer system can",
]


def cosine_similarity(a, b):
    denom = (
        torch.norm(a) * torch.norm(b)
    ).item()
    if denom == 0:
        return 0.0
    return torch.dot(a, b).item() / denom


def encode_for_model(tokenizer, model, prompt):
    token_ids = tokenizer.encode(prompt)
    if len(token_ids) == 0:
        token_ids = [tokenizer.token_to_id[tokenizer.unk_token]]
    context = token_ids[-model.context_length:]
    input_ids = torch.tensor(
        [context],
        dtype=torch.long,
        device=next(model.parameters()).device
    )
    return input_ids


def representation_separation(
    model,
    tokenizer,
    context_prompts,
    unrelated_prompts
):
    prompt_vectors = {}

    for prompt in context_prompts + unrelated_prompts:
        input_ids = encode_for_model(
            tokenizer,
            model,
            prompt
        )

        hidden_capture = {}

        def hook_fn(_, __, output):
            hidden_capture["final_norm"] = output.detach().cpu()

        handle = model.final_norm.register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = model(input_ids)
        handle.remove()

        hidden = hidden_capture["final_norm"][0]
        last_vector = hidden[-1]
        mean_vector = hidden.mean(dim=0)

        prompt_vectors[prompt] = {
            "last": last_vector,
            "mean": mean_vector,
        }

    prompts = list(prompt_vectors.keys())
    matrix_last = []
    matrix_mean = []

    for p_i in prompts:
        row_last = []
        row_mean = []
        for p_j in prompts:
            row_last.append(
                cosine_similarity(
                    prompt_vectors[p_i]["last"],
                    prompt_vectors[p_j]["last"]
                )
            )
            row_mean.append(
                cosine_similarity(
                    prompt_vectors[p_i]["mean"],
                    prompt_vectors[p_j]["mean"]
                )
            )
        matrix_last.append(row_last)
        matrix_mean.append(row_mean)

    context_last = [
        prompt_vectors[p]["last"]
        for p in context_prompts
    ]
    unrelated_last = [
        prompt_vectors[p]["last"]
        for p in unrelated_prompts
    ]

    context_centroid = torch.stack(context_last).mean(dim=0)
    unrelated_centroid = torch.stack(unrelated_last).mean(dim=0)
    centroid_cosine = cosine_similarity(
        context_centroid,
        unrelated_centroid
    )

    return {
        "prompts": prompts,
        "context_prompts": context_prompts,
        "unrelated_prompts": unrelated_prompts,
        "cosine_matrix_last_token": matrix_last,
        "cosine_matrix_mean_token": matrix_mean,
        "centroid_cosine_context_vs_unrelated_last_token": centroid_cosine,
    }


def attention_diagnostics(
    model,
    tokenizer,
    prompts
):
    results = []

    for prompt in prompts:
        input_ids = encode_for_model(
            tokenizer,
            model,
            prompt
        )

        with torch.no_grad():
            logits, attention_history = model(
                input_ids,
                return_attention=True
            )

        per_layer = []
        for layer_idx, attn in enumerate(attention_history):
            # attn shape: [B, H, T, T]
            matrix = attn[0].detach().cpu()
            num_heads = matrix.shape[0]
            seq_len = matrix.shape[1]

            head_stats = []
            for head in range(num_heads):
                m = matrix[head]

                future_mass = 0.0
                total_mass = m.sum().item()
                for i in range(seq_len):
                    for j in range(i + 1, seq_len):
                        future_mass += m[i, j].item()

                safe = m.clamp(min=1e-12)
                entropy_per_pos = -torch.sum(
                    safe * torch.log(safe),
                    dim=-1
                )
                mean_entropy = entropy_per_pos.mean().item()

                diag = torch.diag(m).mean().item()

                prev_values = []
                for i in range(1, seq_len):
                    prev_values.append(m[i, i - 1].item())
                mean_prev = (
                    sum(prev_values) / len(prev_values)
                    if prev_values else 0.0
                )

                argmax_offsets = []
                for i in range(seq_len):
                    j = int(torch.argmax(m[i]).item())
                    argmax_offsets.append(i - j)

                head_stats.append({
                    "head": head,
                    "mean_entropy": mean_entropy,
                    "mean_self_attention": diag,
                    "mean_prev_attention": mean_prev,
                    "future_attention_mass": future_mass,
                    "future_attention_mass_ratio": (
                        future_mass / total_mass
                        if total_mass > 0 else 0.0
                    ),
                    "mean_argmax_offset": (
                        sum(argmax_offsets) / len(argmax_offsets)
                    ),
                })

            per_layer.append({
                "layer": layer_idx,
                "sequence_length": seq_len,
                "heads": head_stats,
            })

        results.append({
            "prompt": prompt,
            "layers": per_layer,
            "vocab_size": logits.shape[-1],
        })

    return results


def top5_from_logits(logits, tokenizer):
    probs = torch.softmax(logits, dim=-1)
    values, indices = torch.topk(probs, k=5)
    rows = []
    for prob, idx in zip(values.tolist(), indices.tolist()):
        rows.append({
            "token_id": int(idx),
            "char": tokenizer.decode([int(idx)]),
            "probability": float(prob),
        })
    return rows


def logit_entropy_and_repetition(
    model,
    tokenizer,
    prompts,
    max_new_tokens
):
    outputs = []

    for prompt in prompts:
        generated = list(tokenizer.encode(prompt))
        if not generated:
            generated = [tokenizer.token_to_id[tokenizer.unk_token]]

        steps = []
        repeated_bigram_first_step = None
        repeated_trigram_first_step = None
        seen_bigrams = set()
        seen_trigrams = set()

        for step_idx in range(max_new_tokens):
            input_ids = torch.tensor(
                [generated[-model.context_length:]],
                dtype=torch.long,
                device=next(model.parameters()).device
            )

            with torch.no_grad():
                logits = model(input_ids)
                if isinstance(logits, tuple):
                    logits = logits[0]
            next_logits = logits[0, -1, :]

            probs = torch.softmax(next_logits, dim=-1)
            safe = probs.clamp(min=1e-12)
            entropy = -torch.sum(
                safe * torch.log(safe)
            ).item()

            next_id = int(torch.argmax(probs).item())
            next_prob = float(probs[next_id].item())

            top5 = top5_from_logits(
                next_logits,
                tokenizer
            )

            generated.append(next_id)

            if len(generated) >= 2:
                bg = (generated[-2], generated[-1])
                if (
                    repeated_bigram_first_step is None
                    and bg in seen_bigrams
                ):
                    repeated_bigram_first_step = step_idx + 1
                seen_bigrams.add(bg)

            if len(generated) >= 3:
                tg = (
                    generated[-3],
                    generated[-2],
                    generated[-1]
                )
                if (
                    repeated_trigram_first_step is None
                    and tg in seen_trigrams
                ):
                    repeated_trigram_first_step = step_idx + 1
                seen_trigrams.add(tg)

            steps.append({
                "step": step_idx + 1,
                "argmax_token_id": next_id,
                "argmax_char": tokenizer.decode([next_id]),
                "argmax_probability": next_prob,
                "entropy": entropy,
                "top5": top5,
            })

        generated_text = tokenizer.decode(generated)

        tokens_as_chars = [
            tokenizer.decode([token])
            for token in generated
        ]

        outputs.append({
            "prompt": prompt,
            "generated_text": generated_text,
            "steps": steps,
            "repetition_dynamics": {
                "first_repeated_bigram_step": repeated_bigram_first_step,
                "first_repeated_trigram_step": repeated_trigram_first_step,
                "unique_bigrams": len(seen_bigrams),
                "unique_trigrams": len(seen_trigrams),
                "generated_token_count": len(generated),
            },
            "token_character_trace": tokens_as_chars,
        })

    return outputs


def render_text_report(results):
    lines = []
    lines.append("OUTREACHLM REPRESENTATION/ATTENTION/ENTROPY DIAGNOSTICS")
    lines.append("=" * 72)
    lines.append(f"Timestamp: {results['timestamp']}")
    lines.append("")

    rep = results["representation_separation"]
    lines.append("1) REPRESENTATION SEPARATION")
    lines.append(f"Context centroid cosine vs unrelated centroid: "
                 f"{rep['centroid_cosine_context_vs_unrelated_last_token']:.6f}")
    lines.append("Prompts:")
    for p in rep["prompts"]:
        lines.append(f"- {p}")
    lines.append("")

    lines.append("2) ATTENTION DIAGNOSTICS")
    for item in results["attention_diagnostics"]:
        lines.append(f"Prompt: {item['prompt']}")
        for layer in item["layers"]:
            lines.append(f"  Layer {layer['layer']}:")
            for head in layer["heads"]:
                lines.append(
                    f"    Head {head['head']} | "
                    f"entropy={head['mean_entropy']:.6f} | "
                    f"self={head['mean_self_attention']:.6f} | "
                    f"prev={head['mean_prev_attention']:.6f} | "
                    f"future_ratio={head['future_attention_mass_ratio']:.6f} | "
                    f"argmax_offset={head['mean_argmax_offset']:.3f}"
                )
        lines.append("")

    lines.append("3) LOGIT ENTROPY + REPETITION DYNAMICS")
    for item in results["entropy_and_repetition"]:
        lines.append(f"Prompt: {item['prompt']}")
        lines.append(f"Generated: {item['generated_text']}")
        r = item["repetition_dynamics"]
        lines.append(
            f"  first_repeated_bigram_step={r['first_repeated_bigram_step']} "
            f"first_repeated_trigram_step={r['first_repeated_trigram_step']} "
            f"unique_bigrams={r['unique_bigrams']} "
            f"unique_trigrams={r['unique_trigrams']}"
        )
        lines.append("  First 5 steps:")
        for row in item["steps"][:5]:
            lines.append(
                f"    step={row['step']} entropy={row['entropy']:.6f} "
                f"argmax={row['argmax_char']!r} p={row['argmax_probability']:.6f}"
            )
        lines.append("")

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run representation, attention, entropy, and repetition diagnostics "
            "for OutreachLM without training."
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

    representation = representation_separation(
        model=model,
        tokenizer=tokenizer,
        context_prompts=DEFAULT_CONTEXT_PROMPTS,
        unrelated_prompts=DEFAULT_UNRELATED_PROMPTS
    )

    attention = attention_diagnostics(
        model=model,
        tokenizer=tokenizer,
        prompts=DEFAULT_CONTEXT_PROMPTS + DEFAULT_UNRELATED_PROMPTS
    )

    entropy_rep = logit_entropy_and_repetition(
        model=model,
        tokenizer=tokenizer,
        prompts=DEFAULT_ENTROPY_PROMPTS,
        max_new_tokens=args.max_new_tokens
    )

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "representation_separation": representation,
        "attention_diagnostics": attention,
        "entropy_and_repetition": entropy_rep,
    }

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = args.output_dir / (
        f"representation-attention-entropy-{stamp}.json"
    )
    txt_path = args.output_dir / (
        f"representation-attention-entropy-{stamp}.txt"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2
        )

    text_report = render_text_report(results)
    with open(
        txt_path,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(text_report)

    print(str(json_path.resolve()))
    print(str(txt_path.resolve()))


if __name__ == "__main__":
    main()

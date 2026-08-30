from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from outreachlm.train import CORPUS_PATH, load_corpus
from outreachlm.phase_g_bridge import WordTokenizer

from src.phase_h_cache import PhaseHConfig
from src.phase_h_cache.tokenization.byte_tokenizer import ByteTokenizer
from src.phase_h_cache.tokenization.online_bpe import OnlineBPETokenizer


def _suite() -> dict[str, str]:
    corpus = load_corpus(CORPUS_PATH)
    baseline = " ".join(corpus.split()[:6000])
    technical = (
        "cache coherence memory bandwidth quantization residual divergence affine scheduling "
        "tokenization manifold reconstruction sparse distribution probability conservation "
    ) * 350
    oov = " ".join(f"novelterm_{idx}" for idx in range(4000))
    return {"baseline": baseline, "technical": technical, "oov": oov}


def _measure_profile(name: str, encoder, text_blocks: dict[str, str]) -> dict[str, Any]:
    encoded_counts: dict[str, int] = {}
    elapsed_total = 0.0
    total_chars = 0
    total_tokens = 0

    for block_name, block_text in text_blocks.items():
        t0 = time.perf_counter()
        token_ids = encoder(block_text)
        t1 = time.perf_counter()
        elapsed_total += t1 - t0
        encoded_counts[block_name] = len(token_ids)
        total_chars += len(block_text)
        total_tokens += len(token_ids)

    compression_ratio = total_chars / max(1, total_tokens)
    throughput = total_tokens / max(elapsed_total, 1e-12)
    return {
        "name": name,
        "encoded_token_counts": encoded_counts,
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "compression_ratio": compression_ratio,
        "throughput_tokens_per_second": throughput,
        "elapsed_seconds": elapsed_total,
    }


def run() -> dict[str, Any]:
    config = PhaseHConfig.load_default().raw["h2"]
    vocab_max = int(config["vocab_max"])
    merges_per_chunk = int(config["online_bpe_merges_per_chunk"])
    chunks = int(config["online_bpe_chunks"])
    loop_rounds = int(config["infinite_loop_rounds"])

    suite = _suite()

    # Profile A: frozen word tokenizer seam from Phase G.
    word_train_lines = [suite["baseline"], suite["technical"]]
    word_tokenizer = WordTokenizer.from_lines(word_train_lines)
    unk_id = word_tokenizer.unk_id

    def encode_word(text: str) -> list[int]:
        return word_tokenizer.encode(text, add_bos=False, add_eos=False)

    profile_a = _measure_profile("phase_g_word", encode_word, suite)
    profile_a["unk_fraction"] = (
        sum(1 for token_id in encode_word(suite["oov"]) if token_id == unk_id)
        / max(1, len(encode_word(suite["oov"])))
    )
    profile_a["vocabulary_size"] = word_tokenizer.vocab_size

    # Profile B: pure bytes.
    byte_tokenizer = ByteTokenizer()
    profile_b = _measure_profile("pure_bytes", byte_tokenizer.encode, suite)
    profile_b["vocabulary_size"] = byte_tokenizer.vocab_size

    # Profile C: online BPE.
    bpe = OnlineBPETokenizer(vocab_limit=vocab_max)
    training_stream = suite["baseline"] + " " + suite["technical"]
    chunk_size = max(1, len(training_stream) // max(1, chunks))
    for idx in range(chunks):
        start = idx * chunk_size
        end = len(training_stream) if idx == chunks - 1 else (idx + 1) * chunk_size
        bpe.learn_from_stream(training_stream[start:end], merge_steps=merges_per_chunk)
    profile_c = _measure_profile("online_bpe", bpe.encode, suite)

    stress_text = ("boundless ingestion stream adaptation " * 256).strip()
    for _ in range(loop_rounds):
        bpe.learn_from_stream(stress_text, merge_steps=8)
    profile_c["vocabulary_size"] = bpe.vocab_size
    profile_c["vocab_limit"] = vocab_max
    profile_c["vocab_bound_respected"] = bpe.vocab_size <= vocab_max

    gate_ratio = profile_c["compression_ratio"] / max(profile_b["compression_ratio"], 1e-12)
    gates = {
        "compression_ratio_vs_bytes_at_least_2_5x": gate_ratio >= 2.5,
        "vocabulary_bound_respected": bool(profile_c["vocab_bound_respected"]),
    }
    selected = "online_bpe" if all(gates.values()) else "bytes"

    return {
        "experiment_id": "h2_vocabulary_topology",
        "config": config,
        "profiles": {"A_phase_g_word": profile_a, "B_bytes": profile_b, "C_online_bpe": profile_c},
        "gate_metrics": {"online_bpe_to_bytes_compression_multiplier": gate_ratio},
        "hard_gates": gates,
        "selected_tokenizer_profile": selected,
        "final_vocabulary_size": int(profile_c["vocabulary_size"] if selected == "online_bpe" else profile_b["vocabulary_size"]),
        "average_compression_ratio": float(profile_c["compression_ratio"] if selected == "online_bpe" else profile_b["compression_ratio"]),
    }


def main() -> None:
    result = run()
    output_dir = Path("experiments") / "phase_h" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "h2_vocab.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

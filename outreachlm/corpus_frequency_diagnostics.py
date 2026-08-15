import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from outreachlm.train import (
    CORPUS_PATH,
    VALIDATION_SPLIT,
    load_corpus,
    split_corpus,
)


TARGET_PHRASES = [
    "the company",
    "and the company",
]


def word_tokens(text):
    return re.findall(r"[A-Za-z']+", text.lower())


def ngrams(tokens, n):
    return [
        tuple(tokens[i : i + n])
        for i in range(len(tokens) - n + 1)
    ]


def char_ngrams(text, n):
    return [
        text[i : i + n]
        for i in range(len(text) - n + 1)
    ]


def counter_to_rows(counter, limit):
    return [
        {"item": item, "count": count}
        for item, count in counter.most_common(limit)
    ]


def main():
    text = load_corpus(CORPUS_PATH)
    training_text, validation_text = split_corpus(
        text,
        VALIDATION_SPLIT
    )

    words = word_tokens(training_text)
    word_count = len(words)
    char_count = len(training_text)

    unigram_counter = Counter(words)
    bigram_counter = Counter(ngrams(words, 2))
    trigram_counter = Counter(ngrams(words, 3))
    char4_counter = Counter(char_ngrams(training_text.lower(), 4))
    char5_counter = Counter(char_ngrams(training_text.lower(), 5))

    phrase_stats = {}
    train_lower = training_text.lower()
    total_bigrams = max(word_count - 1, 1)
    total_trigrams = max(word_count - 2, 1)

    for phrase in TARGET_PHRASES:
        count = train_lower.count(phrase)
        tokens = phrase.split()
        if len(tokens) == 2:
            denom = total_bigrams
        elif len(tokens) == 3:
            denom = total_trigrams
        else:
            denom = max(word_count, 1)
        phrase_stats[phrase] = {
            "count": count,
            "per_million_ngrams": (count / denom) * 1_000_000,
        }

    results = {
        "timestamp": datetime.now().isoformat(),
        "split": {
            "train_characters": len(training_text),
            "validation_characters": len(validation_text),
            "train_words": word_count,
        },
        "phrase_stats": phrase_stats,
        "top_words": counter_to_rows(unigram_counter, 100),
        "top_word_bigrams": [
            {"item": " ".join(item), "count": count}
            for item, count in bigram_counter.most_common(100)
        ],
        "top_word_trigrams": [
            {"item": " ".join(item), "count": count}
            for item, count in trigram_counter.most_common(100)
        ],
        "top_char_4grams": counter_to_rows(char4_counter, 100),
        "top_char_5grams": counter_to_rows(char5_counter, 100),
        "meta": {
            "total_word_bigrams": total_bigrams,
            "total_word_trigrams": total_trigrams,
            "corpus_path": str(Path(CORPUS_PATH).resolve()),
            "character_count_train": char_count,
        },
    }

    out_dir = Path("experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    json_path = out_dir / f"corpus-frequency-{stamp}.json"
    txt_path = out_dir / f"corpus-frequency-{stamp}.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append("OUTREACHLM CORPUS FREQUENCY DIAGNOSTIC")
    lines.append("=" * 72)
    lines.append(f"Train characters: {len(training_text)}")
    lines.append(f"Train words: {word_count}")
    lines.append("")
    lines.append("Target phrase stats:")
    for phrase, stats in phrase_stats.items():
        lines.append(
            f"- {phrase!r}: count={stats['count']}, "
            f"per_million_ngrams={stats['per_million_ngrams']:.3f}"
        )
    lines.append("")
    lines.append("Top 20 words:")
    for row in results["top_words"][:20]:
        lines.append(f"- {row['item']}: {row['count']}")
    lines.append("")
    lines.append("Top 20 word bigrams:")
    for row in results["top_word_bigrams"][:20]:
        lines.append(f"- {row['item']}: {row['count']}")
    lines.append("")
    lines.append("Top 20 word trigrams:")
    for row in results["top_word_trigrams"][:20]:
        lines.append(f"- {row['item']}: {row['count']}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(str(json_path.resolve()))
    print(str(txt_path.resolve()))


if __name__ == "__main__":
    main()

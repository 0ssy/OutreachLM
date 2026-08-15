# OutreachLM — Corpus Frequency Diagnostic

## Why this diagnostic
After the generation-collapse findings, we needed to test whether repeated output (especially around “company” patterns) is strongly driven by corpus frequency or mostly by model-capacity dynamics.

## Script added
- [corpus_frequency_diagnostics.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/corpus_frequency_diagnostics.py)

It analyzes the training split and reports:
- top words
- top word bigrams
- top word trigrams
- top character 4-grams
- top character 5-grams
- explicit counts for:
  - `"the company"`
  - `"and the company"`

## Raw artifacts
- [corpus-frequency-20260815-124358.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/corpus-frequency-20260815-124358.json)
- [corpus-frequency-20260815-124358.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/corpus-frequency-20260815-124358.txt)

## Key results
Training split:
- characters: `3,002,707`
- words: very high-frequency function-word distribution

Target phrase stats:
- `"the company"`:
  - count: `82`
  - per million word-bigrams: `163.396`
- `"and the company"`:
  - count: `2`
  - per million word-trigrams: `3.985`

Top words (head):
- `the`, `and`, `to`, `of`, `a`, `in`, `you`, `is`, `for`, `that`, ...

Top bigrams (head):
- `of the`, `in the`, `to the`, `on the`, `you can`, `is a`, `it is`, `and the`, ...

Top trigrams (head):
- `one of the`, `as well as`, `a lot of`, `you want to`, `be able to`, ...

## Interpretation
Findings suggest:
- the corpus strongly reinforces generic connector phrases (`and`, `the`, `of the`, `and the`),
- but the exact phrase `"and the company"` is not overwhelmingly frequent.

So generation collapse likely reflects a combination of:
1. heavy high-frequency connector bias in data, and
2. limited model capacity/context causing attractor behavior in decoding.

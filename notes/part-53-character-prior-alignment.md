# OutreachLM — Character Prior Alignment

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

## Goal
Verify whether dominant logit direction mainly tracks empirical character priors, and quantify `<PAD>` / `<UNK>` behavior.

## Script and artifacts
Script:
- [char_prior_alignment_diagnostics.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/char_prior_alignment_diagnostics.py)

Raw outputs:
- [char-prior-alignment-20260815-131006.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/char-prior-alignment-20260815-131006.json)
- [char-prior-alignment-20260815-131006.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/char-prior-alignment-20260815-131006.txt)

## Results

### A) Training token coverage
- vocab size: `490`
- observed in training stream: `488`
- unseen in training stream: `2`
- `<PAD>` count/prob: `0 / 0.0`
- `<UNK>` count/prob: `0 / 0.0`

### B) Alignment: `log P_empirical(token)` vs mean model logits
- Pearson (all tokens): `0.879790`
- Spearman (all tokens): `0.781542`
- Pearson (observed-only): `0.899453`
- Spearman (observed-only): `0.780413`

### C) Top empirical characters (training stream)
Top entries:
- `' '` (`0.161551`)
- `'e'` (`0.093911`)
- `'t'` (`0.069039`)
- `'a'` (`0.062927`)
- `'o'` (`0.062576`)

### D) Top mean-logit characters
Top entries:
- `'i'`, `'e'`, `'a'`, `'t'`, `'s'`, `'o'`, `'r'`, `'l'`, `' '`, `'n'`, ...

This list is very close to broad character-frequency priors.

### E) `<PAD>` / `<UNK>` predicted probability
- `<PAD>` rank/prob(mean softmax): `460 / 7.264e-09`
- `<UNK>` rank/prob(mean softmax): `400 / 1.0097e-08`

Both are negligible in prediction mass.

### F) Empirical unigram baseline on validation tokens
- cross-entropy (nats): `3.126865`
- perplexity: `22.802374`

## Conclusion
The dominant output direction is strongly aligned with the corpus character prior (high correlation with empirical unigram log-probabilities), not a degenerate `<PAD>/<UNK>` artifact.  
This supports the current diagnosis: strong global character prior + insufficient conditional discrimination during free-running rollout.
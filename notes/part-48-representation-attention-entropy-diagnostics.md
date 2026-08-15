# OutreachLM — Representation, Attention, and Entropy Diagnostics

## Scope
Built and ran a no-training diagnostics pass covering:
1. context representation separation
2. attention behavior
3. logit entropy over generation
4. repetition dynamics onset

Script:
- [representation_diagnostics.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/representation_diagnostics.py)

Raw outputs:
- [representation-attention-entropy-20260815-123415.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/representation-attention-entropy-20260815-123415.json)
- [representation-attention-entropy-20260815-123415.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/representation-attention-entropy-20260815-123415.txt)

## Key findings

### 1) Representation separation
- Context-group vs unrelated-group centroid cosine (last-token state):
  - `0.5767`

Interpretation:
- representations are not collapsing to a single point
- but separation is moderate; semantic disentangling is still limited.

### 2) Attention behavior
- Future attention mass ratio is `0.0` across heads in sampled diagnostics.
  - causal masking is functioning.
- Head behaviors differ:
  - some heads show high previous-token focus
  - some heads show lower entropy (more peaked attention), others broader.

Interpretation:
- transformer is using prior tokens (not uniform attention)
- but usage appears strongly local in at least part of the head set.

### 3) Logit entropy profile
Across tested prompts (greedy rollout, 60 tokens):
- average entropy per step is around `1.69–1.72`
- minima around `0.35–0.37`
- maxima around `3.32–3.79`

Interpretation:
- model frequently enters low-entropy confident states.
- these confident states can correspond to repetitive continuations.

### 4) Repetition dynamics
First repeated n-gram onset (greedy):
- `OutreachLM is`: first repeated bigram at step `6`, trigram at step `7`
- other tested prompts: first repeated bigram around step `15`, trigram around step `19`

Interpretation:
- collapse into repeated local phrase patterns begins early and predictably.

## Conclusion
Diagnostics confirm:
- training learned stable local language statistics,
- causal attention path is valid,
- repetition collapse is tied to low-entropy attractor behavior under current model capacity/context.

This supports prioritizing architecture/context experiments next rather than blindly extending step count.

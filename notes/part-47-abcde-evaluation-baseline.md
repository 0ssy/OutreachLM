# OutreachLM — A/B/C/D/E Evaluation Baseline (No Training)

## Run scope
Executed full evaluation suite in one pass (no weight updates):
- A: deterministic (greedy/argmax)
- B: sampling comparison (`temp=0.3, 0.5, 0.8, 1.0`, `top_k=8`)
- C: context sensitivity prompts
- D: top-5 next-character probabilities
- E: automatic repetition/word-integrity metrics

Common generation settings:
- `max_new_tokens = 60`
- prompts from project plan

## Raw artifacts
Combined raw results saved to:
- [generation-eval-ABCDE-20260815-122918.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/generation-eval-ABCDE-20260815-122918.json)
- [generation-eval-ABCDE-20260815-122918.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/generation-eval-ABCDE-20260815-122918.txt)

## Key observations
1. **Greedy decoding collapses into repetitive high-probability continuations**  
   Frequent pattern: “and the company …”

2. **Sampling increases diversity but degrades lexical integrity as temperature rises**  
   `temp=0.8/1.0` shows more novelty and malformed words.

3. **Context sensitivity exists but remains shallow**  
   Next-character distributions clearly change with prompt length, but continuations still drift into repetitive local templates.

4. **Token-level probabilities are dominated by space and a few high-frequency letters**  
   Indicates strong local character modeling with weak semantic constraint.

## Metrics snapshot (overall, across A+B+C outputs)
- total words: `385`
- valid english-looking word rate: `0.927273`
- malformed word rate: `0.072727`
- repeated word rate: `0.026042`
- repeated 3-gram rate: `0.328982`
- average word length: `4.379221`
- sentence termination rate: `0.000000`

## Interpretation
This baseline confirms:
- optimization has learned useful local structure
- generation bottleneck is now model capacity/context/representation and decoding behavior, not a broken training loop.

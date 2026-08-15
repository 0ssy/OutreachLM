# OutreachLM — Output-Head Geometry and LayerNorm A/B

## Goal
Test whether contextual differences in hidden space are being compressed at vocabulary projection, and measure LayerNorm contribution at inference (no retrain).

## Script and artifacts
Script:
- [output_head_diagnostics.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/output_head_diagnostics.py)

Raw outputs:
- [output-head-diagnostics-20260815-125553.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/output-head-diagnostics-20260815-125553.json)
- [output-head-diagnostics-20260815-125553.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/output-head-diagnostics-20260815-125553.txt)

## A) Output-head geometry
- W shape: `[490, 64]`
- Numerical rank (`eps=1e-6`): `64`
- Effective rank (entropy): `33.8318`
- Row cosine (off-diagonal): mean/std/min/max = `0.6693 / 0.3689 / -0.6008 / 0.9897`

Interpretation:
- the head is full numerical rank, but output rows are strongly correlated on average,
- many token rows are near-collinear, so direction overlap in output space is high.

## B) Prompt projection compression
Selected pairs:

- `Machine learning allows` vs `A computer system can`
  - hidden cosine: `0.0808`
  - logit cosine LN/no-LN: `0.9918 / 0.9847`
  - symmetric KL LN/no-LN: `0.8336 / 1.2081`
  - `||Δh||`: `15.0421`
  - `||Δlogits||` LN/no-LN: `68.6025 / 74.6914`
  - ratio `||Δlogits|| / ||Δh||` LN/no-LN: `4.5607 / 4.9655`

- `Machine learning allows computers to` vs `The purpose of a transformer`
  - hidden cosine: `0.1174`
  - logit cosine LN/no-LN: `0.9925 / 0.9851`
  - symmetric KL LN/no-LN: `1.8386 / 2.3753`

- `OutreachLM is` vs `Machine`
  - hidden cosine: `0.4488`
  - logit cosine LN/no-LN: `0.9957 / 0.9923`
  - symmetric KL LN/no-LN: `0.9762 / 1.7774`

Across all non-identical prompt pairs:
- hidden cosine mean/min/max: `0.3819 / 0.0808 / 0.7061`
- logit cosine (LN) mean/min/max: `0.9944 / 0.9913 / 0.9975`
- logit cosine (no-LN) mean/min/max: `0.9896 / 0.9835 / 0.9963`

Interpretation:
- hidden representations are distinct,
- next-token logit vectors remain extremely aligned across prompts,
- context-specific separation weakens strongly at output distribution geometry.

## C) Logit variance across prompts
- Mean variance LN/no-LN: `4.8768 / 16.3818`
- Median variance LN/no-LN: `4.9861 / 14.9472`

Interpretation:
- removing final LayerNorm increases logit spread substantially,
- LayerNorm is likely damping amplitude differences at inference for this small model.

## D) Teacher forcing LayerNorm A/B (inference-only)
Sequence source: `fallback-start-of-validation`  
Prompt: `'the content on the Xbox 360 Marketplace,'`

- Top-1 accuracy LN/no-LN: `0.5125 / 0.4500`
- Average gold probability LN/no-LN: `0.3411 / 0.3914`

Interpretation:
- no-LN raises confidence on average but lowers top-1 accuracy,
- LN remains better calibrated for argmax correctness on this checkpoint.

## Conclusion
- The checkpoint still shows strong hidden-state distinction but highly aligned output directions across prompts.
- This supports the hypothesis that output-space geometry contributes to attractor behavior.
- LayerNorm removal does not provide an immediate inference-only fix; it widens spread but hurts top-1 accuracy.

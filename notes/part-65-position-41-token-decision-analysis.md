# OutreachLM — Position 41 Token Decision Analysis (Leader Checkpoint)

## Goal
Inspect the exact token decision at position 41 for the current leader checkpoint and determine whether the error pattern is systematic across validation windows.

## Checkpoint analyzed
- [v2-divergence-intervention-20260816-113809.pt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-divergence-intervention-20260816-113809.pt)

## Script and artifacts
Script:
- [position41_decision_analysis.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/position41_decision_analysis.py)

Outputs:
- [position41-decision-20260816-120239.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/position41-decision-20260816-120239.json)
- [position41-decision-20260816-120239.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/position41-decision-20260816-120239.txt)

## Canonical sequence result
Prompt (auto-selected fallback):
- `'the content on the Xbox 360 Marketplace,'`

At position 40:
- gold: `' '`
- predicted: `' '`
- p(pred): `0.825276`

At position 41:
- gold: `'n'`
- teacher top-1: `'a'` with p=`0.098477` (gold p=`0.013665`)
- free top-1: `'a'` with p=`0.098477` (gold p=`0.013665`)
- logit margin (pred - gold): `1.974987`

Observation:
- the position-41 error is already present under teacher-forced context in this canonical slice (teacher/free agree on the wrong token).

## Systematic validation analysis (4096 sampled windows)
- teacher match rate @ position 41: `0.400391`
- free match rate @ position 41: `0.246094`
- free mismatch rate @ position 41: `0.753906`
- free gold probability mean (all/correct/wrong):
  - `0.117065 / 0.323628 / 0.049638`
- free logit margin (pred - gold) mean:
  - all: `1.244871`
  - wrong only: `1.651228`

Most common wrong predicted token:
- `' '` with count `871`
- fraction of mismatches: `0.282060`

Top mismatch pairs:
- `'t' -> ' '`: `102`
- `'e' -> ' '`: `89`
- `'e' -> 'o'`: `87`
- `'s' -> ' '`: `85`
- `'i' -> ' '`: `70`

## Interpretation
- The position-41 failure is systematic and strongly biased toward high-frequency fallback tokens (especially space, then vowels like `o`).
- In the canonical slice, the first error is not caused by immediate teacher/free context mismatch (both choose the same wrong token); this points to a difficult local decision under current modeling capacity/training signal rather than only rollout drift at that exact step.

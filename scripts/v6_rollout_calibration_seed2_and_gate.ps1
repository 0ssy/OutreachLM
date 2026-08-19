# V6 rollout-calibration: seed2 training + two-seed gate

python -m outreachlm.train_v4 `
  --output-dir experiments/v6-training-rollout-calibration-seed1337 `
  --seed 1337 `
  --steps 1500 `
  --eval-interval 250 `
  --checkpoint-interval 500 `
  --batch-size 8 `
  --learning-rate 0.0005 `
  --warmup-steps 250 `
  --min-learning-rate-ratio 0.1 `
  --label-smoothing 0.05 `
  --recovery-start-index 40 `
  --recovery-loss-weight 2.0 `
  --post-error-loss-weight 0.0 `
  --rollout-calibration-loss-weight 1.0 `
  --rollout-calibration-forced-error-index 40 `
  --rollout-calibration-rollout-steps 8 `
  --rollout-calibration-start-index 41 `
  --rollout-calibration-end-index 43

python -m outreachlm.leader_gating_protocol `
  --leader-artifact experiments/v2-divergence-intervention-20260816-113809.pt `
  --candidate seed1=experiments/v6-training-rollout-calibration-seed1/v4-best-rollout.pt `
  --candidate seed1337=experiments/v6-training-rollout-calibration-seed1337/v4-best-rollout.pt `
  --min-seeds 2 `
  --required-free-match-margin 0.0 `
  --minimum-free-match 0.2 `
  --require-not-below-leader `
  --require-divergence-or-recovery `
  --minimum-post-divergence-next12-delta 0.0 `
  --output-dir experiments

# IGM-M3 HP3 Focused Review 1

Verdict: `CONDITIONAL_APPROVE_P0_AFTER_FINAL_TESTS`.

- HP3 reuses the hash-bound HP2 72/18/18 roles; 64 fit and 8 probe sequences
  are deterministic, stratified, disjoint, and probe/dev/eval payloads are
  unread at S0.
- Student is exact M0R1 encoder/FPN. Student input is genuinely masked in
  contiguous 4x4 stride-8 blocks while EMA targets see unmasked views.
- Reconstruction, hard-negative symmetric InfoNCE, symmetric stride-16
  expected geometry, and multiscale variance/covariance execute through the
  declared weights. Shuffled-H cycles homographies inside the identical
  four-sample optimizer batch.
- EMA parameters are excluded from optimizer. Export contains only exact
  M0R1 encoder keys; P0-only adapters/projector/predictor/scale embeddings are
  absent from inference.
- Probe code validates two 10k endpoint hashes and paired exposure streams
  before reading the eight train-role probe sequences; it aggregates metrics
  by sequence and uses all four viewpoint sequences.

Required before watchdog: final build tests, runtime-manifest parity, focused
review 2. Correct/shuffled 10k failures stop before probe or tiny.

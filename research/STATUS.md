# Project Status

Updated: 2026-09-30

## Best Accuracy-Improving Method

`E063-D0` is the best validated accuracy-improving method in the project.

- HPatches AUC@3/5/10: `58.05 / 67.05 / 76.86`.
- Historical untouched OfficialSLiM-O0: `55.10 / 64.68 / 74.00`.
- Delta: `+2.95 / +2.36 / +2.86 pp`.

E063-D0 is a DeformDesc feature-side method that retains the OfficialSLiM
matcher, fine stage, dual-softmax, quota, RRU, and evaluator. It is not a fully
independent matcher. Its improvement is concentrated in illumination; its
viewpoint AUC@5 is slightly below the historical baseline.

## Best Fully Independent Efficient Method

`IGM-M0R1` is the only independent method that completed the full matched-
scratch training and sequence-held-out evaluation funnel.

- AUC@3/5/10: `53.64 / 62.39 / 73.11`.
- Matched OfficialSLiM scratch: `65.96 / 76.09 / 85.73`.
- Parameter ratio: `0.1929`.
- Inference-P95 ratio: `0.1783`.

It demonstrates a substantial efficiency benefit but does not achieve accuracy
equivalence. The dominant gap is viewpoint generalization.

## Latest Experiment

`IGM-M3 GCRP` tested training-only geometry-consistent representation
pretraining with no additional inference module.

- Correct-H completed 10,000 steps with complete trainability and export
  integrity, but masked-loss ratio `0.4347184` missed the `<=0.40` gate.
- Shuffled-H contained four nonfinite total-loss rows and was invalid as the
  paired geometry control.
- Representation probe, matcher tiny, development, and evaluation remained
  unread.

Status: `inconclusive / S3_pretraining_trainability`.

## Current Boundary

No new experiment route is active in this portable snapshot. A future method
must be frozen under a new contract. The full validated numerical summary is in
`research/RESULTS_SUMMARY_IGM.md`.

# IGM Results Summary

This document is the portable summary of the independent image-matching work.
Raw result cards, checkpoints, trajectories, and host-bound runtime manifests
remain local scientific evidence and are intentionally not part of the initial
Git commit.

## Evaluation Protocol

- Dataset: HPatches sequences release.
- Isolation unit: complete sequence.
- Input: grayscale, shorter edge 480, aspect ratio preserved.
- Estimator: OpenCV homography RANSAC, 3 px at 480 scale, confidence 0.99999,
  10,000 iterations.
- Correspondences: confidence-sorted top 1000.
- Repeats: deterministic seeds 66-71.
- Aggregation: sequence-macro AUC per repeat, then repeat mean.
- Primary metric: sequence-macro AUC@5.
- Claim boundary: retrospective internal evidence; all HPatches sequences had
  prior project exposure.

## E063-D0 - Best Accuracy-Improving Method

E063-D0 inserts a DeformDesc feature branch before the unchanged OfficialSLiM
matcher stack. It was trained on MegaDepth and evaluated on HPatches; HPatches
did not select its checkpoint or training configuration.

| HPatches AUC | E063-D0 | OfficialSLiM-O0 | Delta |
|---|---:|---:|---:|
| AUC@3 | 58.05 | 55.10 | +2.95 pp |
| AUC@5 | 67.05 | 64.68 | +2.36 pp |
| AUC@10 | 76.86 | 74.00 | +2.86 pp |

Stratified AUC@5:

- illumination: `88.85` versus `81.18`, delta `+7.68 pp`;
- viewpoint: `46.96` versus `49.53`, delta `-2.57 pp`.

E063-D0 is the project's best validated accuracy-improving method, but it is a
hybrid feature-side method rather than a standalone matcher. The portable
DeformDesc core is under `tools/e063_deformdesc/`; reproducing the historical
endpoint also requires the external OfficialSLiM/ASCS lineage and MegaDepth.

## IGM-M0

M0 introduced a fully independent image matcher. Its training loss decreased,
but the original null probability and mutual decode were inconsistent: all
eight tiny pairs emitted zero matches. The exact recipe stopped at
`L2_training_recipe_rejected / S3_trainable`.

## IGM-M0R1

M0R1 replaced the assignment with bidirectional categorical distributions over
candidates plus a null dustbin. It passed tiny and completed the matched-scratch
formal comparison.

| Sequence-macro | IGM-M0R1 | OfficialSLiM scratch | Delta |
|---|---:|---:|---:|
| AUC@3 | 53.64 | 65.96 | -12.32 pp |
| AUC@5 | 62.39 | 76.09 | -13.70 pp |
| AUC@10 | 73.11 | 85.73 | -12.62 pp |

- AUC@5 seed deltas: `-17.16 pp`, `-10.25 pp`.
- Paired sequence-bootstrap 95% CI: `[-24.43,-3.67] pp`.
- Illumination AUC@5 delta: `+2.62 pp`.
- Viewpoint AUC@5 delta: `-30.03 pp`.
- Parameters: 1.133M versus 5.874M, ratio `0.1929`.
- Inference P95: about 71 ms versus 398 ms, ratio `0.1783`.
- Peak training memory: 3.79 GiB versus 6.72 GiB.

Conclusion: strong efficiency improvement, insufficient viewpoint accuracy.
The exact M0R1 recipe was rejected at L3/S5; independent matching remained
open.

## IGM-M1 VASI

VASI tested uncertainty-routed adaptive-span cross-view attention. The first
positive-control gate was unmatched to the parent, so a corrected paired test
was run with identical data, initialization, and optimization.

Viewpoint transfer medians:

- M0R1 parent: `2.008702 px`.
- VASI predicted route: `2.175814 px`.
- VASI GT route: `2.169403 px`.
- VASI permuted route: `2.169436 px`.

Route coverage was `1.0 / 1.0 / 0.0` for predicted/GT/permuted, but even the
GT route was worse than M0R1 and predicted routing did not separate from the
permutation control. The exact VASI integration stopped before tiny. No
held-out accuracy result exists.

## IGM-M3 GCRP

GCRP tested training-only geometry-consistent encoder/FPN pretraining with no
added inference module.

- Correct-H completed 10,000 steps with full gradient/update/EMA/reload/export
  integrity, but masked-loss ratio `0.4347184` missed the `<=0.40` gate.
- Shuffled-H had four nonfinite total-loss rows and was invalid as a paired
  representation control.
- The representation probe, matcher tiny, development, and evaluation remained
  unread.

Conclusion: `inconclusive / S3_pretraining_trainability`. The run establishes
trainability but neither benefit nor failure of geometry pretraining.

## Current Boundary

No method currently meets the accuracy-equivalence objective. M0R1 remains the
only independent method with a complete held-out evaluation and the strongest
validated efficiency result. VASI and GCRP provide scoped negative/inconclusive
mechanism evidence; neither supports a final method claim.

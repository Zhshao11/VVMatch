# P37-20260930-IGM-M3-GCRP-v1 Method Delta

Updated: 2026-09-30
State: `M3_GCRP_P0_INCONCLUSIVE_INTERPRETED__NO_NEW_ROUTE_ACTIVE`

## Terminal Evidence

Correct-H completed 10,000 steps with finite trajectory, every-tensor
gradient/update and exact EMA/reload/encoder export, but its masked-loss ratio
`0.4347184` missed the frozen `<=0.40` gate. Shuffled-H used the identical
exposure stream but had four nonfinite total-loss rows. The paired
representation probe was not admitted; its eight train-role sequences and
development/evaluation payloads remain unread. The exact P0 recipe is stopped
as `inconclusive`; no family-level or matching/pose conclusion follows.

The correct-H endpoint is not promoted as a near-pass: mask loss reached an
early minimum then rose, and late total/contrast losses did not converge toward
the gate. Shuffled-H is invalid, so correct-versus-shuffled attribution is
unavailable. Do not restart, extend, select an early checkpoint, or open the
probe from this experiment.

## Evidence Boundary

M0R1 is efficient but trails matched OfficialSLiM scratch by `13.70 pp` AUC@5,
with illumination `+2.62 pp` and viewpoint `-30.03 pp`. Exact VASI routing is
closed: matched parent/predicted/GT/permuted viewpoint medians were
`2.0087/2.1758/2.1694/2.1694 px`, so even GT routing had no action ceiling.

## Current Change

GCRP changes only the initialization of the independent M0R1 encoder/FPN.
During a 10k train-only pretraining phase it uses:

- masked cross-view EMA descriptor reconstruction;
- symmetric correspondence InfoNCE;
- soft expected-coordinate geometry;
- variance/covariance anti-collapse regularization.

Every pretraining-only module is deleted. Exported inference is exact M0R1.
The full matcher is then fine-tuned for 20k unchanged steps.

## Matched Comparison

Official scratch and M0R1 scratch receive 30k supervised steps. M3 receives
10k GCRP plus 20k supervised fine-tuning. All arms have equal two-image
exposure, FP32, effective batch, seeds, optimizer family, final-only rule, and
common evaluator. Reuse sealed HP2 roles as HP3.

The complete causal design and limitations are in
`research/PROMPT37_M3_GCRP_METHOD_DECISION_20260930.md`.

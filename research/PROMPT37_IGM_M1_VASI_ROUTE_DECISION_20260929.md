# Prompt37 IGM-M1 VASI Route Decision

Date: 2026-09-29
Decision: `CONTINUE_VASI_WITH_CORRECTED_MATCHED_POSITIVE_CONTROL`
Experiment: `P37-20260929-IGM-M1R1-HP2`

## Executive Decision

Do not switch methods yet. The completed M1 preflight is inconclusive because
its failed gate did not isolate the VASI mechanism. Continue the exact frozen
VASI architecture through one corrected, matched S2 experiment. Do not start
tiny, budget, formal, development, or evaluation unless this corrected control
passes.

If the corrected control shows no viewpoint-action ceiling over M0R1, stop
VASI and switch the next method-design cycle to geometry-consistent
representation pretraining (M3). Do not repair a failed VASI ceiling by adding
M2 feedback, widening the refiner, relaxing decode, or tuning thresholds.

## Background

The project objective is an independent visible-visible matcher, not an
OfficialSLiM backbone with gates or biases. The primary comparison trains each
architecture from scratch on sequence-disjoint HPatches roles under matched
data exposure and a common homography evaluator.

IGM-M0 used an independent multiscale encoder, dense pair interaction,
explicit null assignment, and bilateral refinement. It failed tiny because its
pair-mass-complement null was inconsistent with mutual decoding: loss fell, but
all eight pairs produced zero matches.

IGM-M0R1 replaced that interface with bidirectional categorical candidate-plus-
dustbin assignment. It passed tiny, resource, formal, and held-out sequence
evaluation. It was substantially more efficient than OfficialSLiM-HP-Scratch:

- parameter ratio: `0.1929`;
- inference P95 ratio: `0.1783`;
- mean P95 inference: about `71 ms` versus `398 ms`;
- formal usage: `10.20 GPU-hours`.

Its accuracy was not competitive overall:

- M0R1 AUC@3/5/10: `53.64 / 62.39 / 73.11`;
- Official scratch: `65.96 / 76.09 / 85.73`;
- AUC@5 delta: `-13.7003 pp`, 95% CI `[-24.4252,-3.6720]`;
- illumination AUC@5 delta: `+2.6244 pp`;
- viewpoint AUC@5 delta: `-30.0251 pp`.

Both methods supplied the evaluator's top 1000 correspondences. Pairwise error
increased sharply with viewpoint strength, so the primary bottleneck is wrong
long-range coarse correspondence rather than match count.

## VASI Design And Progress

VASI retains the independent M0R1 encoder widths, assignment, mutual decoder,
bilateral 9x9 refiner, cap, optimizer, and evaluator. It changes only cross-view
interaction:

1. stride-16 features predict bidirectional correspondence centers,
   uncertainty, and soft covisibility;
2. each stride-8 query samples a 6x6 opposite-image grid whose span expands
   with uncertainty;
3. exact local cross-attention aggregates the sampled tokens;
4. soft covisibility scales the message without a hard gate or crop.

HP2 S0 passed with frozen 72/18/18 sequence roles and zero sequence/file
intersection. The old unread development role was preserved; evaluation was
drawn from old train-only sequences and remains unread. Build R6 passed 12/12
checks and two focused reviews.

The final same-parameterization preflight established:

- inherited initial tensors exact: pass;
- GT route coverage: `1.0`;
- predicted route coverage: `1.0`;
- permuted route coverage: `0.0`;
- nonempty actions: pass;
- every-tensor gradients and updates: pass;
- exact reload: pass;
- GT-route median refined transfer: `1.7721 px`, failing the frozen `<=0.5 px`.

No tiny, budget, formal, development, endpoint evaluation, or HP2 evaluation
ran. Highest evidence is S2 and the status is inconclusive, not rejected.

## Why The Failed Gate Is Not Decision-Valid

The failed absolute transfer threshold mixes VASI routing with the unchanged
fine refiner. Routing itself passed its capacity and sensitivity controls.

More importantly, the parent and child positive-control metrics are not
matched:

- M0R1 trained on identity, translation, and scale at `64x64`; its reported
  `0.3916 px` is the pooled median over all final point errors. Identity errors
  near zero dominate that pooled statistic. Its final logged translation
  median was still about `2.81 px`.
- VASI trained on identity, translation, scale-rotation, and projective warps
  at `128x160`; its non-identity per-case medians were about `2.16-2.26 px`.

Thus `0.3916` versus `1.7721` is not a fair predecessor comparison. The
`<=0.5 px` gate cannot determine whether adaptive routing improves the exact
same viewpoint cases. It can only show that VASI plus the inherited refiner did
not meet an absolute subpixel target on a harder synthetic suite.

## Corrected M1R1 Test

Keep VASI source and semantics unchanged. Run fresh M0R1 and VASI students on
the exact same `128x160` identity, translation, scale-rotation, and projective
cases, paired seed, 600 steps, optimizer, sample order, and inherited tensor
initialization. Evaluate four paths:

- `M0R1-parent`;
- `VASI-predicted-route`;
- `VASI-GT-route`;
- `VASI-permuted-route`.

The primary unit is the pooled set of final transfer errors from only
scale-rotation and projective cases. Identity and translation remain safety
diagnostics and cannot dominate the primary statistic.

Admission requires all of the following:

1. GT route coverage `>=0.95`, predicted coverage `>=0.90`, and permuted
   coverage `<=0.50`.
2. `VASI-GT-route` viewpoint transfer median is at least 15% lower than
   `M0R1-parent`.
3. `VASI-predicted-route` viewpoint transfer median is at least 5% lower than
   `M0R1-parent`.
4. Predicted-route viewpoint transfer is at least `0.25 px` lower than the
   permuted-route control.
5. Every case emits nonempty matches; inherited tensors, gradients, updates,
   reload, masks, coordinates, M0R1 assignment/refiner semantics, and forbidden
   reads pass.

These relative gates directly test whether VASI improves viewpoint action
through its declared route, independent of the inherited refiner's absolute
subpixel ceiling.

## What The Corrected Test Can Reject

If the matched GT-route ceiling does not beat M0R1, VASI's exact adaptive-span
context has no demonstrated causal capacity on the intended synthetic
viewpoint transformations. Stop VASI before tiny. This rejects only the exact
VASI integration and does not reject adaptive span or covisibility broadly.

If GT route passes but predicted route fails, the interaction action is viable
but the exact route-learning objective/recipe fails. Stop this recipe; do not
run formal or hide the failure with GT routing.

If both pass, continue the already frozen HP2 tiny, resource, three-arm formal,
development gate, and conditional evaluation without changing VASI.

## Alternative If VASI Stops

The next route should be M3 geometry-consistent representation pretraining,
not M2 geometry feedback. M0R1 already learns illumination well but lacks
viewpoint-stable coarse features. Masked correspondence recovery,
transformation consistency, and viewpoint-structured pretraining act before
routing and directly target representation invariance. M2 feedback is lower
priority because a geometry reward cannot recover true matches absent from the
coarse representation.

Local literature motivating the decision includes ASpanFormer, CoMatch,
SceneGlue, and SGFormer. Online work checked through 2026-09-26 includes
LoRetta (`arXiv:2608.04106`) and RoMa-Omega (`arXiv:2609.09507`). These support
localization-before-refinement and geometry-grounded representation, but none
establishes novelty or efficacy for the exact VASI integration.

## Boundaries

- Reuse the frozen HP2 manifests; evaluation remains sealed.
- Do not change VASI architecture, parent assignment/refiner, optimizer, data
  roles, or evaluator in this correction.
- Do not add hard visibility gates, thresholds, top-k fallback, M2, or M3.
- Ordinary engineering and provenance errors are repaired autonomously.
- A passed corrected S2 authorizes the existing conditional funnel; it is not
  itself an accuracy or generalization result.


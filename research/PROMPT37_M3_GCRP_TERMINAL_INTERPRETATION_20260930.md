# Prompt37 M3 GCRP Terminal Interpretation

Date: 2026-09-30
Experiment: `P37-20260930-IGM-M3-HP3`
Decision: `INCONCLUSIVE_P0_STOP__NO_SAME_PROTOCOL_RESTART__NO_NEW_ROUTE_YET`

## Outcome

The exact GCRP P0 experiment validly stops before the representation probe.
The highest demonstrated evidence is S3 pretraining trainability. It does not
establish representation benefit, matcher benefit, viewpoint generalization,
or a family-level failure.

Correct-H completed 10,000 steps and 40,000 two-image exposures with a finite
trajectory, complete student gradient/update evidence, EMA outside the
optimizer, exact reload, and an inference-clean encoder/FPN export. Its frozen
endpoint masked-loss ratio was `0.4347184`, above the required `0.40`.

Shuffled-H completed the paired stream but contains four nonfinite total-loss
rows, first at step 3807. It is invalid for the correct-versus-shuffled causal
comparison.

## Why This Is Not A Near-Pass

The endpoint miss is numerically modest, but the trajectory does not support a
budget-limited interpretation:

- masked loss reached an early minimum near `0.104` at step 225, then rose;
- mean masked loss in the final 1000 steps was about `0.450`;
- the preceding 1000-step mean was about `0.459`, showing weak late movement
  rather than convergence toward the endpoint gate;
- total and contrastive loss in the final 1000 steps were worse than in the
  preceding 1000 steps;
- the protocol requires step 10,000, not the best historical checkpoint.

It is not defensible to extend the same trajectory, select an early checkpoint,
lower the gate, or relabel the result as passed.

## Why Geometry-Specific Benefit Is Unknown

The central question was whether correct geometry improves representation over
identical shuffled geometry. It was not answered:

1. shuffled-H is not a faithful full-trajectory control;
2. the eight train-role probe sequences remained unread;
3. no mutual-NN recall, cosine margin, or viewpoint probe result exists;
4. no exported encoder entered matcher tiny or downstream assignment;
5. no development/evaluation result exists.

The shuffled arm's reported mask ratio (`0.03096`) is nonselectable. Its
nonfinite trajectory and unexpectedly low mask term also warn that masked
reconstruction alone may admit shortcuts unrelated to correct geometry.

## Established Evidence

- The exact correct-H implementation is active, trainable, reloadable, and
  exports an inference-clean encoder/FPN.
- Its exact four-loss recipe misses the fixed P0 endpoint at 10,000 steps.
- The exact shuffled-H control is numerically invalid.
- Data isolation and fail-closed release worked: probe, matcher tiny,
  development, and evaluation were not opened.

## Unsupported Conclusions

- Geometry-consistent pretraining is not shown to help or hurt representation.
- The M0R1 viewpoint gap is not reduced or enlarged by M3.
- The correct-H encoder is not promoted and must not enter a post-hoc matcher.
- Masked correspondence learning, EMA targets, contrastive geometry, synthetic
  homography pretraining, and representation pretraining remain open broadly.
- There is no accuracy, latency, memory, pose, or held-out matching conclusion.

## Classification

`inconclusive / S3_pretraining_trainability`

This is narrower than `L2_training_recipe_rejected` because the paired causal
control is invalid and the representation probe never ran. It is not an
execution-invalid result: correct-H ran faithfully and its independent masked
endpoint failure is valid.

## Decision Boundary

- Do not restart the same P0 protocol.
- Do not continue from either P0 checkpoint.
- Do not repair shuffled-H and reuse the old correct-H endpoint post hoc.
- Do not open probe, matcher tiny, development, or evaluation.
- Do not infer that additional steps would pass.
- Do not launch a new route from this result in the current decision.

A future method decision must separately choose between a materially redesigned
pretext/control pair and a different representation family. It requires a new
frozen contract and is not an engineering retry.

## Evidence Paths

- `research/results/P37-20260930-IGM-M3-HP3/RESULT_CARD.md`
- `research/results/P37-20260930-IGM-M3-HP3/p0_correct_r1/complete_pretrain_trajectory.csv`
- `research/results/P37-20260930-IGM-M3-HP3/p0_shuffled_r1/complete_pretrain_trajectory.csv`
- `zzvr(ubuntu):/mnt/sda/zhushihao/moveExperiment/p37_igm_m3_hp3_20260930/postreview_terminal_r1`

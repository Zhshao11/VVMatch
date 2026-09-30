# Implementation And Training Audit Standard

Updated: 2026-07-21

## Purpose

An experiment can fail because the scientific idea is wrong, because the code does not implement the idea, because the intended branch is inactive, or because optimization never learns the intended function. These outcomes must not share one verdict.

FA-2B-D0 is the canonical warning: the fixed 50-epoch smoke failed, while the unchanged from-scratch 300-epoch diagnostic later recovered substantial teacher support. The 50-epoch gate remains a valid protocol failure, but it cannot support a broad architecture rejection.

## Functional Contract

Before implementation, define Function A in observable terms:

```text
input tensors and coordinate system
output tensors and valid range
where the output changes the main matching path
identity/no-op behavior
positive behavior on a synthetic or oracle example
fallback behavior
runtime and quota invariants
```

Code existence, parameter count, or a successful forward pass does not prove Function A is implemented.

## Implementation Fidelity Gate

The following evidence is required before a result can reject an implementation:

1. Unit contract: shapes, coordinate scales, masks, ranges, and fallback semantics pass.
2. Identity control: disabling A or setting A to identity reproduces the baseline at the declared tolerance.
3. Positive control: a synthetic, free-logit, GT, or teacher-replay input causes the intended output change.
4. Activation evidence: branch-use ratio and output delta show that A is exercised on real samples.
5. Gradient evidence: intended trainable parameters receive finite nonzero gradients.
6. Update evidence: optimizer parameter IDs include A and parameter deltas are nonzero after training.
7. Action evidence: the changed tensor reaches assignment, coordinates, confidence, or RRU as specified.
8. Baseline/evaluator parity: method differences are not caused by resize, RANSAC, repeat aggregation, GPU-cache mixing, or baseline checkpoint drift.

Failure of items 1, 2, 6, 7, or 8 normally makes the run `invalid`. Missing items 3, 4, or 5 normally makes it `inconclusive` rather than a scientific rejection.

## Training Adequacy Gate

Before concluding that a parameterized implementation lacks capacity or signal, record:

- tiny-set forced-overfit result;
- free-output or teacher-replay upper bound through the same selector;
- loss components and task metric over the complete training horizon;
- gradient norm and parameter-update norm;
- final-window slope or stability, not only the last point;
- optimizer steps and processed windows, not only epoch count;
- checkpoint rule fixed before evaluation;
- whether train and validation metrics are being distinguished.

Use these labels:

```text
adequate_converged
adequate_overfit_only
underfit
unstable
not_tested
unknown
```

`underfit`, `not_tested`, or `unknown` cannot support a method-family rejection.

## Verdict Levels

```text
L0 invalid:
  execution or evaluator is wrong; no hypothesis conclusion

L1 inconclusive implementation:
  code ran, but Function A activation, gradients, or positive control is missing

L2 training recipe rejected:
  implementation fidelity is established, but this optimizer/budget/loss did not learn it

L3 tested implementation rejected:
  fidelity and optimization are adequate; this concrete architecture/interface failed

L4 method-family evidence:
  multiple materially different faithful implementations fail under adequate training,
  while oracle/action-space evidence also limits the family
```

Most individual Prompt37 negative experiments should stop at L2 or L3. L4 requires an explicit synthesis record and must not be inferred from one runner.

## Required Audit Record

Each decision-changing rejection should have an entry in `implementation_audit.yaml` containing:

```yaml
experiment_id:
function_contract:
implementation_fidelity:
  identity_control:
  positive_control:
  activation_verified:
  gradients_verified:
  parameter_updates_verified:
  action_path_verified:
training_adequacy:
  label:
  tiny_overfit:
  convergence_evidence:
evaluation_fidelity:
  baseline_parity:
  evaluator_parity:
audit_verdict:
max_allowed_conclusion:
open_risks:
```

Unknown fields remain `unknown`; they are not silently treated as passed.

## Evidence Ladder

The name of a class or branch in source code is not evidence that the intended
function exists. Track each new Function A through these observable levels:

```text
S0 code_present:
  the module can be imported and called

S1 path_active:
  identity/no-op parity passes and real samples exercise the branch

S2 action_expressive:
  a synthetic, GT, teacher-replay, or free-output control produces the intended
  downstream assignment/coordinate change through the same action path

S3 trainable:
  intended parameters receive finite nonzero gradients, are owned by the
  optimizer, and measurably change after optimizer steps

S4 tiny_set_fit:
  the learned module recovers the positive control on a fixed tiny training set;
  the endpoint is converged or a fixed long-horizon stability audit is reported

S5 scene_generalization:
  a frozen checkpoint is positive on physically disjoint scenes under a
  predeclared checkpoint rule and matched controls

S6 pose_effect:
  the scene-isolated method improves paired official pose AUC under the frozen
  evaluator
```

A failure can only reject the tested rung and configuration. In particular:

- failure before S2 is an implementation failure, not a method result;
- failure before S4 is at most a training-recipe rejection;
- success at S4 is expression evidence, not generalization;
- no offline support metric, including L6 recovery, is a pose claim;
- one faithful implementation cannot by itself reject a broad method family.

## Prospective Logging Rule

For experiments created after 2026-07-21, archive explicit machine-readable
evidence for:

```text
identity output hashes or declared numeric tolerances
positive-control output deltas
branch-use counts
gradient norms for every new trainable parameter group
optimizer parameter IDs
initial-to-final parameter delta norms
optimizer steps and processed windows
complete task-metric trajectory and fixed checkpoint rule
```

Historical experiments may use clearly labeled inferred evidence. An inferred
gradient or update cannot silently become an explicit pass, and any negative
verdict must retain the narrower historical uncertainty in `open_risks`.

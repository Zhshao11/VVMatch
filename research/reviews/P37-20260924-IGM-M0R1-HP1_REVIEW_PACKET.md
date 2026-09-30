# P37-20260924-IGM-M0R1-HP1 Focused Review Packet

State: `READY_FOR_TWO_FOCUSED_REVIEWS`

## Frozen Inputs

- `research/proposals/P37-20260924-IGM-M0R1-v1/METHOD_DELTA.md`
- `research/proposals/P37-20260924-IGM-M0R1-v1/FUNCTION_CONTRACT.yaml`
- `research/proposals/P37-20260924-IGM-M0R1-v1/BUDGET_AND_GATES.yaml`
- `research/experiments/P37-20260924-IGM-M0R1-HP1.md`
- Exact source hashes:
  `research/results/P37-20260924-IGM-M0R1-HP1/SOURCE_SHA256.txt`

## Narrow Delta

- `tools/igm_m0r1/model.py`: class identity plus assignment construction and
  mutual decoder only. Source rows and target rows each softmax over candidates
  plus their existing linear-head null logit; confidence is geometric mean.
- `tools/igm_m0r1/losses.py`: coarse NLL consumes the exact categorical source
  and target distributions. Fine/cycle logic is inherited unchanged.
- `tools/prompt37_igm_m0r1_hp1_runner.py`: new arm ID, action/null logging,
  M0R1 paths, and the inherited frozen execution/evaluation provenance gates.
- `tools/prompt37_igm_m0r1_preflight.py`: parent parity, free-logit, and real
  student synthetic controls.
- `tools/test_prompt37_igm_m0r1.py` and
  `tools/test_prompt37_igm_m0r1_protocol.py`: assignment and inherited protocol
  checks.

Exact parent diffs:

- `research/results/P37-20260924-IGM-M0R1-HP1/PARENT_MODEL_DIFF.patch`
- `research/results/P37-20260924-IGM-M0R1-HP1/PARENT_LOSS_DIFF.patch`

## High-Risk Functions

Method review:

- `IndependentGeometryMatcherM0R1.categorical_assignment`
- `IndependentGeometryMatcherM0R1.mutual_matches`
- `_directional_coarse_loss`, `compute_igm_loss`
- free-logit and real-student positive controls

Split/provenance review:

- `implementation_bindings`, `run_training`, `validate_endpoint_freeze`
- `run_evaluation`, `run_aggregate`
- inherited S0 bundle and sample-stream binding

## Automated Evidence

- 11/11 final assignment/protocol tests pass:
  `research/results/P37-20260924-IGM-M0R1-HP1/UNIT_TESTS.txt`.
- Parent and rescue initialization have identical state keys and exact tensor
  bytes for every tensor:
  `research/results/P37-20260924-IGM-M0R1-HP1/parent_initialization_parity.json`.
- All free-logit identity, translation, repeated-texture, unmatched, and padded
  cases pass exact positive/null recovery without threshold or forced top-k:
  `research/results/P37-20260924-IGM-M0R1-HP1/free_logit_control.json`.
- Real student synthetic identity/translation/scale traverses the full model;
  aggregate median refined transfer is `0.391575 px`, all cases nonempty:
  `research/results/P37-20260924-IGM-M0R1-HP1/PREFLIGHT_SUMMARY.json`.
- Real HPatches 10-step smoke passes every-tensor gradients, updates, exact
  reload, S0 binding, and action logging:
  `research/results/P37-20260924-IGM-M0R1-HP1/S1_SMOKE_SUMMARY.json`.
- No development or evaluation payload was read.

## Review Questions

1. Does the exact A/B/null construction and mutual decoder implement the frozen
   categorical events without threshold, quota, fallback, or count loss?
2. Does coarse supervision distinguish trustworthy unmatched from excluded
   out-of-image/padded projections on both directions?
3. Is every parent module outside assignment/loss unchanged in state shape,
   initialization, and execution path?
4. Are action/null metrics sufficient to apply all tiny stop gates?
5. Do inherited split, endpoint-freeze, live-source/data-byte, and joint-result
   provenance gates remain intact under the new arm key?

Reviewers report findings before verdict. Approval binds the exact SHA file.

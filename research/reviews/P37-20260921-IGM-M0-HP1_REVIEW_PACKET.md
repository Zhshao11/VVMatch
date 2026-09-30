# P37-20260921-IGM-M0-HP1 Focused Review Packet

Review state: `READY_FOR_TWO_FOCUSED_REVIEWS`

## Frozen Inputs

- Method spec: `research/proposals/P37-20260921-IGM-v1/METHOD_SPEC.md`
- Function contract: `research/proposals/P37-20260921-IGM-v1/FUNCTION_CONTRACT.yaml`
- Budget/gates: `research/proposals/P37-20260921-IGM-v1/BUDGET_AND_GATES.yaml`
- Experiment sheet: `research/experiments/P37-20260921-IGM-M0-HP1.md`
- Exact SHA list: `research/results/P37-20260921-IGM-M0-HP1/S1_SOURCE_SHA256.txt`

## Changed Code

- `tools/igm_m0/model.py`: standalone encoder/FPN, two dense linear-attention
  interaction blocks, explicit-null assignment, mutual match extraction, and
  bilateral 9x9 fine refinement.
- `tools/igm_m0/losses.py`: homography positive-set/null supervision and
  symmetric fine/cycle losses.
- `tools/prompt37_igm_hp1_common.py`: neutral deterministic pair stream,
  augmentation, shared image geometry, homography updates, common padding, and
  coordinate restoration.
- `tools/official_slim_hp_scratch_adapter.py`: isolated clean public baseline
  construction from random initialization and HPatches homography supervision.
- `tools/prompt37_igm_hp1_runner.py`: common optimizer schedule, complete logs,
  checkpoint/reload, endpoint-freeze gate, and paired evaluator.
- `tools/prompt37_igm_hp1_s0.py`: E-051 allowlist recovery and frozen split.
- `tools/test_prompt37_igm_m0.py`: contract tests.

## Spec-To-Code Map

| Requirement | Implementation |
|---|---|
| independent raw-image matcher | `IndependentGeometryMatcherM0` |
| 32/64/128/192 encoder and 128/64 FPN | `IndependentPyramid` |
| two four-head ELU+1 interaction blocks | `PairInteractionBlock`, `LinearAttention` |
| fixed tau 0.1 and explicit null | `IndependentGeometryMatcherM0.assign` |
| mutual maxima, 2048 cap | `mutual_matches` |
| bilateral 9x9 stride-4 refinement | `refine`, `_sample_patches` |
| coarse/fine/cycle loss | `compute_igm_loss` |
| deterministic paired exposure | `deterministic_pair`, `load_pair` |
| scratch baseline | `OfficialScratchAdapter` |
| 20k final endpoint only | runner `--steps`, checkpoint and evaluation freeze |
| six paired evaluator repeats | `run_evaluation` |

## High-Risk Functions

First review:

- `tools/igm_m0/model.py`: `LinearAttention.forward`, `assign`,
  `mutual_matches`, `refine`, `_sample_patches`.
- `tools/igm_m0/losses.py`: `_directional_coarse_loss`, `compute_igm_loss`.
- `tools/official_slim_hp_scratch_adapter.py`: `_supervision`,
  `training_loss`, `infer`.

Second review:

- `tools/prompt37_igm_hp1_s0.py`: `split_sequences`, `main`.
- `tools/prompt37_igm_hp1_common.py`: `_transform_matrix`, `load_pair`,
  `warp_points`, `restore_points`.
- `tools/prompt37_igm_hp1_runner.py`: `run_training`, `run_evaluation`,
  `corner_error`, `auc`.

## Automated Evidence

- S0: `research/results/P37-20260921-IGM-M0-HP1/s0_config_audit/S0_AUDIT.json`.
  All 108, 52/56, 72/18/18, 720 directed-train, and zero-intersection gates pass.
- Unit tests: `research/results/P37-20260921-IGM-M0-HP1/S1_UNIT_TESTS.txt`.
  Four GPU tests pass: assignment row sum, group gradients, coordinate identity,
  target-permutation action change, and independent import/state keys.
- IGM real-batch smoke:
  `research/results/P37-20260921-IGM-M0-HP1/S1_IGM_SMOKE.json`.
  One four-sample FP32 optimizer step, all groups gradient/update, exact reload,
  1,132,994 parameters, 2.11 GiB peak.
- Official scratch real-batch smoke:
  `research/results/P37-20260921-IGM-M0-HP1/S1_OFFICIAL_SMOKE.json`.
  Same exposure/geometry/optimizer step, all groups gradient/update, exact
  reload, 5,873,675 parameters, 5.23 GiB peak.

## Known Residual Risk

- Tiny overfit, full synthetic control, and real-workload timing have not run;
  they are conditional post-review gates.
- Baseline retains its native loss parameterization as frozen. Its adapter
  expresses homography supervision through native coarse/fine/RRU tensors.
- The endpoint evaluator is newly implemented and requires the second focused
  review before any evaluation role payload is read.
- The clean baseline needs `PYTHONNOUSERSITE=1` in the JamMa CUDA environment
  to avoid unrelated user-site dependency shadowing.

Reviewers must report findings before verdict. An approval binds the exact SHA
list; any related fix requires a focused delta review.

# Prompt37 Independent Image Matching Research

This repository contains the reproducible source, frozen experiment contracts,
tests, and compact result evidence for an independent visible-visible image
matcher developed under the Prompt37 research protocol.

The repository does not include datasets, pretrained checkpoints, model
weights, raw trajectories, logs, remote mirrors, or full historical experiment
payloads. Those artifacts remain local and are referenced by hashes in result
cards and archive registries.

## Current Scientific Status

**Best accuracy-improving method:** `E063-D0`.

- HPatches AUC@3/5/10: `58.05 / 67.05 / 76.86`;
- historical untouched OfficialSLiM-O0: `55.10 / 64.68 / 74.00`;
- delta: `+2.95 / +2.36 / +2.86 pp`.

E063-D0 is a DeformDesc feature-side method evaluated through the unchanged
OfficialSLiM matcher/fine/dual-softmax/quota/RRU path. It is not a fully
independent matcher. Its gain is concentrated in illumination; viewpoint
AUC@5 is `46.96` versus OfficialSLiM's `49.53`.

**Best fully validated independent efficient method:** `IGM-M0R1`.

- sequence-macro HPatches AUC@3/5/10: `53.64 / 62.39 / 73.11`;
- matched OfficialSLiM scratch: `65.96 / 76.09 / 85.73`;
- AUC@5 delta: `-13.70 pp`;
- illumination delta: `+2.62 pp`;
- viewpoint delta: `-30.03 pp`;
- parameter ratio to OfficialSLiM: `0.1929`;
- inference-P95 ratio: `0.1783`.

The result demonstrates a strong efficiency advantage but not accuracy
equivalence. Later VASI routing and GCRP pretraining experiments stopped at
preregistered gates and did not produce held-out accuracy results.

**Latest experiment:** `IGM-M3 GCRP`, stopped as
`inconclusive / S3_pretraining_trainability` before representation probing.

See [research/CURRENT_STATE.md](research/CURRENT_STATE.md) and the current
[terminal interpretation](research/PROMPT37_M3_GCRP_TERMINAL_INTERPRETATION_20260930.md).
Portable numerical results are collected in
[research/RESULTS_SUMMARY_IGM.md](research/RESULTS_SUMMARY_IGM.md).

## Method Evolution

1. `IGM-M0`: independent encoder, pair interaction, explicit null assignment,
   and bilateral fine refinement. It failed tiny because training probability
   and mutual non-null decoding were inconsistent.
2. `IGM-M0R1`: replaced the assignment with bidirectional categorical
   candidate-plus-dustbin distributions. It passed tiny and formal evaluation,
   preserving high efficiency but failing viewpoint generalization.
3. `IGM-M1 VASI`: tested viewpoint-adaptive span interaction. A matched action
   ceiling showed no benefit even with GT routing, so the exact route stopped.
4. `IGM-M3 GCRP`: tested training-only geometry-consistent representation
   pretraining. Correct-H was trainable but missed its frozen P0 gate; the
   shuffled control became nonfinite, so the result is inconclusive.

The method is independent of OfficialSLiM, D0, and JCGA. OfficialSLiM is used
only as a separately instantiated scratch baseline in matched experiments.

## Repository Layout

- `tools/igm_m0*`: independent matcher implementations.
- `tools/e063_deformdesc/`: portable DeformDesc core extracted from E063-D0.
- `tools/igm_m1_vasi`: adaptive-span interaction experiment.
- `tools/igm_m3_gcrp`: training-only representation pretraining experiment.
- `tools/prompt37_igm_*.py`: data, training, evaluation, audit, and gate entry
  points.
- `tools/test_prompt37_igm*.py`: unit and protocol tests.
- `research/proposals/`: immutable method specifications and function
  contracts.
- `research/experiments/`: frozen experiment sheets and engineering handoffs.
- `research/RESULTS_SUMMARY_IGM.md`: portable validated outcome summary.
- `research/results/`: local frozen result cards and raw evidence; excluded from
  the initial portable commit because cards contain host-bound provenance.
- `research_archive/`: compact registries, audit standards, schemas, and tools;
  full artifact mirrors are ignored.

## Reproduction

Use Python 3.10 and install the pinned IGM dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements-igm.txt
```

Detailed data formats, commands, metrics, and validation order are documented
in [research/REPRODUCIBILITY_IGM.md](research/REPRODUCIBILITY_IGM.md).

OfficialSLiM scratch comparisons additionally require a clean checkout of the
OfficialSLiM source supplied through `--official-source`. No OfficialSLiM
checkpoint is loaded for the primary scratch comparison.

## Claim Boundary

All HPatches sequences had historical project exposure. Sequence-disjoint
roles provide controlled retrospective internal evidence, not a pristine paper
test. No `heldout1200`, protected/final data, SAR, or OSDataset data is included
or authorized by this repository.

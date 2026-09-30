# IGM Reproducibility Guide

## Scope

This guide reproduces the independent IGM method family and its gated HPatches
experiments. It does not recreate the full historical Prompt37 archive or ship
external datasets and baseline source code.

## Environment

Validated runtime:

- Python `3.10.20`
- PyTorch `2.8.0+cu128`
- NumPy `1.26.4`
- OpenCV `4.11.0`
- CUDA `12.8`

Install the portable dependencies from `requirements-igm.txt`. A compatible
PyTorch build may be selected for another CUDA platform, but record its version
in the result card.

## Dataset Input

The S0 tools expect the official HPatches sequences release ZIP. Each sequence
contains images `1` through `6` and homographies mapping image `1` to images
`2` through `6`.

Generate the original deterministic 72/18/18 roles:

```bash
python tools/prompt37_igm_hp1_s0.py \
  --zip /path/to/hpatches-sequences-release.zip \
  --output /path/to/hp1_s0
```

Generate the HP2 roles from the HP1 result, then freeze the HP3 pretraining
fit/probe split:

```bash
python tools/prompt37_igm_hp2_s0.py \
  --parent /path/to/hp1_s0 \
  --output /path/to/hp2_s0

python tools/prompt37_igm_hp3_s0.py \
  --hp2-s0 /path/to/hp2_s0 \
  --output /path/to/hp3_s0
```

Sequence identity is the isolation unit. Never place images or derived pairs
from one sequence in multiple roles. Preserve the generated manifests and
their SHA256 values.

## Core Method

`tools/igm_m0r1/` is the fully evaluated independent matcher:

```text
grayscale image pair
-> independent CNN/FPN
-> paired self/cross interaction
-> bidirectional categorical assignment with null dustbins
-> mutual non-null coarse matches
-> bilateral 9x9 fine refinement
-> confidence-sorted correspondences
```

Coordinates use pixel centers. Images are resized to a shorter edge of 480,
aspect ratio is preserved, and tensors are padded to a multiple of 16.

## E063-D0 Core Component

`tools/e063_deformdesc/` contains the portable feature-side module extracted
from the validated E063-D0 runner. It predicts bounded local sampling offsets
and reliability-weighted residuals from normalized descriptors.

Run its module tests with:

```bash
python -m unittest -v tools.test_e063_deformdesc
```

This module alone is not the complete historical E063-D0 pipeline. Exact E063
reproduction requires MegaDepth, its ASCS parent checkpoint, and a clean
OfficialSLiM source/checkpoint because the validated method retained the
OfficialSLiM matcher, fine stage, quota, RRU, and evaluator.

## Training Entry Points

The full M0R1 runner exposes `train`, `evaluate`, and `aggregate` subcommands:

```bash
python tools/prompt37_igm_m0r1_hp1_runner.py train --help
python tools/prompt37_igm_m0r1_hp1_runner.py evaluate --help
python tools/prompt37_igm_m0r1_hp1_runner.py aggregate --help
```

A typical training invocation supplies explicit data and provenance paths:

```bash
python tools/prompt37_igm_m0r1_hp1_runner.py train \
  --stage tiny \
  --arm igm_r1 \
  --seed 20260923 \
  --steps 1000 \
  --data-root /path/to/hpatches-sequences-release \
  --data-archive /path/to/hpatches-sequences-release.zip \
  --pair-manifest /path/to/directed_role_pairs.csv \
  --s0-audit /path/to/S0_AUDIT.json \
  --s0-hashes /path/to/MANIFEST_SHA256.json \
  --official-source /path/to/clean/OfficialSLiM \
  --approved-source-hashes /path/to/APPROVED_RUNTIME_SOURCE_SHA256.txt \
  --output /path/to/output
```

The source-hash manifest and launch guards are part of the scientific protocol,
not optional convenience files. Generate them from the exact reviewed source
set used for a run.

## GCRP Pretraining Entry Point

The current M3 runner implements `train` and `probe`. The terminal experiment
stopped before probe, so no representation-probe result should be fabricated.

```bash
python tools/prompt37_igm_m3_hp3_runner.py train --help
python tools/prompt37_igm_m3_hp3_runner.py probe --help
```

P0 training requires an arm (`correct` or `shuffled`), 10,000 fixed steps,
the HP3 pretrain-pair manifest, S0 hashes, and a reviewed runtime manifest.

## Tests

Run the independent family tests from the repository root:

```bash
python -m unittest -v \
  tools.test_e063_deformdesc \
  tools.test_prompt37_igm_m0 \
  tools.test_prompt37_igm_m0r1 \
  tools.test_prompt37_igm_m0r1_protocol \
  tools.test_prompt37_igm_m1_vasi \
  tools.test_prompt37_igm_m1_hp2_protocol \
  tools.test_prompt37_igm_m1r1_matched_s2 \
  tools.test_prompt37_igm_m3_gcrp
```

GPU positive controls and formal runs must use the same CUDA precision and
source hashes recorded by their frozen experiment sheet.

## Evaluation Metric

The common HPatches evaluator:

1. restores predicted correspondences to original image coordinates;
2. keeps the top 1000 by confidence;
3. estimates a homography with OpenCV RANSAC;
4. uses a 3-pixel threshold at shorter-edge 480, scaled to original target
   pixels, confidence `0.99999`, and `10000` iterations;
5. evaluates deterministic paired seeds `66` through `71`;
6. computes sequence-macro AUC@3/5/10 per repeat, then averages repeats;
7. never selects the best repeat.

Primary comparisons use equal-sequence macro AUC@5. Bootstrap confidence
intervals resample sequences with paired indices across methods and repeats.

## Reproduced Evidence

Terminal results are stored locally as `RESULT_CARD.md/json` files under
`research/results/`. They retain host-bound provenance and are intentionally
excluded from the initial portable commit together with raw logs, streams,
checkpoints, and trajectories. Portable validated outcomes are recorded in
`research/RESULTS_SUMMARY_IGM.md` without altering frozen evidence.

The currently supported conclusions are summarized in `README.md` and
`research/CURRENT_STATE.md`. Do not promote an intermediate checkpoint or
unread split beyond its recorded evidence rung.

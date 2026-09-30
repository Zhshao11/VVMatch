# P37-20260921-IGM-v1 Method Specification

Status: `FROZEN_FOR_ENGINEERING_AUDIT`
Experiment: `P37-20260921-IGM-M0-HP1`
Date: 2026-09-23

## Objective And Independence

IGM-M0 is a complete standalone visible-visible matcher trained from scratch on
HPatches. Its inference path starts from grayscale images and owns feature
extraction, pair interaction, partial assignment, and bilateral fine
refinement. It must not import, load, call, copy runtime tensors from, or wrap
OfficialSLiM, E063-D0, JCGA, their checkpoints, feature modules, assignment,
quota, RRU, or fine matcher.

OfficialSLiM appears only in a separate baseline process named
`OfficialSLiM-HP-Scratch`. The two methods share data roles, pair/augmentation
streams, image geometry, exposure counts, precision, optimizer-step schedule,
evaluation correspondences cap, and external homography evaluator. They do not
share code or parameters.

## IGM-M0 Architecture

Input is one-channel grayscale, shorter edge 480, aspect ratio preserved, and
right/bottom padded to a multiple of 16. Coordinates use pixel centers and are
restored to original HPatches coordinates before evaluation.

### Independent encoder and pyramid

A shared-weight two-image encoder is initialized from scratch:

```text
stem: 3x3 conv, stride 2, 1 -> 32, GroupNorm(8), GELU
stage2: two depthwise-separable residual blocks, stride 4, width 64
stage3: two depthwise-separable residual blocks, stride 8, width 128
stage4: two depthwise-separable residual blocks, stride 16, width 192
FPN: lateral 1x1 projections and bilinear top-down fusion
coarse output: stride 8, width 128
fine output: stride 4, width 64
```

Every residual block is depthwise 3x3, pointwise expansion two, GELU,
pointwise projection, and GroupNorm. There is no pretrained backbone.

### Dense pair interaction

Two identical interaction blocks operate on flattened stride-8 tokens. Each
block applies depthwise 3x3 positional mixing, per-image linear self-attention,
bidirectional linear cross-attention, and a width-128 FFN with expansion two.
Attention uses `phi(x) = elu(x) + 1`, four heads, pre-normalization, masks for
padding, and no token pruning, covisibility routing, early exit, sparse scan,
scene token, or GT input. This dense M0 is the frozen predecessor for the later
single-mechanism covisibility experiment.

### Partial assignment

Projected L2-normalized source and target states form pair logits
`S_ij = dot(Ws*u_i, Wt*v_j) / tau`, with `tau = 0.1` fixed. Independent source
and target matchability heads produce sigmoid values `a_i` and `b_j`.

```text
P_ij = a_i * b_j * softmax_j(S_ij) * softmax_i(S_ij)
Q_i,null = 1 - sum_j(P_ij)
```

All assignment probabilities are FP32 and must be finite with row sums one.
Inference keeps mutual row/column maxima with non-null selected on both sides.
Exact coordinate duplicates are removed by confidence and stable source/target
index. No learned or per-window quota is used. At most 2048 coarse matches are
sent to fine refinement; the common evaluator later caps final matches at 1000.

### Bilateral fine refinement

For each coarse match, bilinearly sample a 9x9 stride-4 patch from both fine
maps. A shared 1x1 projection maps both patches to width 64. Full normalized
patch correlation produces source-to-target and target-to-source conditional
distributions. Soft-argmax estimates offsets for both image coordinates;
neither side is held at the coarse center. The final confidence is coarse
assignment probability times the two fine peak probabilities. Refinement is
bounded to the 9x9 support and cannot read GT at inference.

## HPatches Supervision

For each directed train pair, the provided homography maps source pixel centers
to target coordinates; reverse pairs use the matrix inverse. A stride-8 source
token has a positive target set within one coarse-cell Euclidean radius of its
projected point. A valid projection with no positive discrete target is
unmatched; out-of-image and padded projections are excluded rather than called
unmatched.

```text
L_coarse = positive-set NLL + unmatched-null NLL
L_fine = symmetric Charbonnier transfer error / 4 pixels
L_cycle = symmetric forward/inverse refined-coordinate cycle error / 4 pixels
L_total = L_coarse + L_fine + 0.1 * L_cycle
```

Charbonnier epsilon is `1e-3`. Fine/cycle losses use only valid GT coarse
matches. No RANSAC, homography AUC, development metric, teacher model, proxy
gate, or post-hoc selected-ID edit participates in training.

## Baseline Contract

`OfficialSLiM-HP-Scratch` uses the clean public OfficialSLiM architecture but
loads no released or project checkpoint. All trainable tensors are freshly
initialized under the paired seed. It uses the same HPatches directed pairs,
augmentations, homography labels, image geometry, effective batch, optimizer
steps, precision, and schedule. Its native coarse/fine training objective may
express the common homography supervision through its own output interface;
loss terms and weights must be logged. The comparison is a matched full-method
scratch comparison, not a claim that the two architectures have identical
loss parameterizations.

Baseline source and runner are isolated from the IGM package. An automated
dependency scan must prove that the IGM import graph and state dict contain no
OfficialSLiM/D0/JCGA symbols, paths, or parameter keys.

## Training

Formal seeds are `20260923` and `20260924`. Both arms use AdamW, base learning
rate `2e-4`, weight decay `1e-4`, betas `[0.9, 0.999]`, global gradient clip
`1.0`, 500-step linear warmup, cosine decay to `0.05` of base LR, exactly
20,000 optimizer steps, FP32, and effective batch four. Gradient accumulation
may implement the effective batch but may not change sample order or optimizer
steps. Only step 20,000 is evaluated; no best checkpoint or dev-selected stop.

Train augmentation is identical and replayable across arms: independent
brightness, contrast, gamma, Gaussian noise, and blur; common resize/crop/flip
geometry with the homography updated analytically. No augmentation may use a
development/evaluation image or learned generator. Every pair, direction,
augmentation parameter, and RNG state is logged by deterministic sample ID.

## Evaluation

The common read-only evaluator uses final correspondences in original
coordinates, confidence-sorted top 1000, OpenCV `findHomography` RANSAC,
reprojection threshold 3 px at shorter-edge 480 scaled to original target
pixels, confidence `0.99999`, max iterations `10000`, and deterministic paired
seeds `[66, 67, 68, 69, 70, 71]`. Compute each repeat's sequence-macro
AUC@3/5/10, then average repeats. Never use best-of-repeat.

Primary metric is equal-sequence macro AUC@5 on the frozen evaluation role.
Report illumination/viewpoint strata, all per-sequence and per-repeat rows,
pair-weighted diagnostics, parameters, FLOPs, inference P50/P95, and peak
memory. Paired bootstrap uses sequence as the unit, 10,000 samples, seed
`20260923`, with common indices across methods/seeds/repeats.

## Interpretation

M0 is an independent base-method qualification, not the covisibility
innovation. A valid positive comparison can establish only retrospective
HPatches sequence-held-out benefit under the exact scratch protocol. All 108
sequences were historically consumed, so no pristine benchmark or paper-final
claim is allowed. A valid negative result rejects only this M0 architecture and
training recipe; it does not reject independent matchers or later M1-M3
mechanisms.


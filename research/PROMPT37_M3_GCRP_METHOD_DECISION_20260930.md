# Prompt37 M3 Geometry-Consistent Representation Pretraining Decision

Date: 2026-09-30
Method ID: `P37-20260930-IGM-M3-GCRP-v1`
Experiment ID: `P37-20260930-IGM-M3-HP3`
Decision: `FREEZE_M3_FOR_ENGINEERING_AUDIT`

## 1. Executive Decision

Stop the exact VASI route and switch to M3: Geometry-Consistent
Representation Pretraining (GCRP). The immediate problem is not the number of
matches, null assignment, or adaptive routing capacity. The lightweight M0R1
encoder does not produce viewpoint-stable coarse features. M3 changes how the
existing independent encoder/FPN is initialized, while preserving the entire
M0R1 inference architecture and its efficiency.

M3 uses only HP3 train-role images and known train homographies. It introduces
training-only EMA targets, masked feature reconstruction, correspondence
contrast, and soft geometric localization. All pretraining-only modules are
discarded before matcher fine-tuning and inference.

## 2. Research Objective

Build an independent matcher that reaches practical accuracy equivalence with
OfficialSLiM-HP-Scratch while retaining the demonstrated M0R1 efficiency:

- no OfficialSLiM, D0, JCGA, foundation-model, or pretrained matcher runtime;
- sequence-disjoint HPatches train/development/evaluation roles;
- primary AUC@5 within `-3 pp` of matched scratch OfficialSLiM;
- each seed and viewpoint stratum within `-5 pp`;
- parameter and inference-P95 ratios at most `0.25` of OfficialSLiM;
- no degradation hidden by best-of-repeat, threshold tuning, or test selection.

## 3. Progress To Date

### M0: independent baseline

M0 owned the path from raw images through encoder/FPN, pair interaction,
partial assignment, and bilateral refinement. It was independent of
OfficialSLiM but failed tiny: loss fell while all eight pairs emitted zero
matches. The exact M0 training/action recipe was rejected at L2/S3.

### M0R1: consistent dustbin assignment

M0R1 replaced the inconsistent pair-mass-complement null with bidirectional
categorical candidate-plus-dustbin assignment. It passed tiny and completed
matched-scratch sequence-held-out evaluation.

| Sequence-macro | M0R1 | Official scratch | Delta |
|---|---:|---:|---:|
| AUC@3 | 53.64 | 65.96 | -12.32 pp |
| AUC@5 | 62.39 | 76.09 | -13.70 pp |
| AUC@10 | 73.11 | 85.73 | -12.62 pp |

The paired AUC@5 95% CI was `[-24.43,-3.67] pp`. Illumination improved
`+2.62 pp`, while viewpoint regressed `-30.03 pp`.

M0R1 retained a strong efficiency result:

- 1.133M versus 5.874M parameters, ratio `0.1929`;
- about 71 ms versus 398 ms P95 inference, ratio `0.1783`;
- 3.79 versus 6.72 GiB peak training memory;
- 10.20 total formal GPU-hours for four trajectories.

### M1 VASI: adaptive route experiment

VASI introduced stride-16 probabilistic centers, uncertainty, soft
covisibility, and adaptive local exact attention. Its first absolute positive
control was not matched to M0R1 and was therefore inconclusive.

The corrected matched S2 used fresh paired M0R1/VASI on identical
identity/translation/scale-rotation/projective streams. Every integrity,
gradient, update, reload, assignment/refiner parity, and route coverage gate
passed. Viewpoint transfer medians were:

- M0R1 parent: `2.008702 px`;
- VASI predicted: `2.175814 px`;
- VASI GT route: `2.169403 px`;
- VASI permuted route: `2.169436 px`.

The GT route had no ceiling over M0R1, and predicted routing did not separate
from permutation. The exact VASI integration is rejected at L3 with no second
rescue. HP2 development/evaluation remained unread.

## 4. Current Bottleneck

The evidence localizes the bottleneck upstream of assignment and refinement:

1. M0R1 emits the evaluator's full 1000 correspondences, so count is not the
   limitation.
2. Illumination performance is already competitive, so generic photometric
   robustness is not the primary gap.
3. Viewpoint error worsens with transformation strength, reaching tens or
   hundreds of pixels on hard pairs.
4. VASI's GT route fully covers the true region but does not improve final
   correspondence error, so better routing cannot rescue weak features.
5. Geometry feedback is premature: a downstream reward cannot restore correct
   candidates if the coarse representation does not distinguish them.

The next action point is therefore the independent encoder/FPN representation
before pair interaction.

## 5. Archive Matching And Closed Routes

- E041 rejected one independent LWRD feature recipe under neutral feature
  diagnostics. It used descriptor/rank distillation and does not test masked
  cross-view geometry pretraining inside a full independent matcher.
- Prompt100 contains legacy DINO fusion, LoRA, and fine-refiner implementation
  memory under noncanonical evaluators. It cannot establish current-protocol
  benefit and should not be repeated.
- M0/M0R1 close only their exact scratch recipes. They explicitly leave M3
  geometry-consistent pretraining open.
- VASI closes its exact adaptive-span integration because even GT routing had
  no matched action ceiling. M3 does not reopen VASI.
- GeneralPruner-style geometry pretraining was planned in FA-9A but never ran
  because the upstream pose-label contract failed. That unexecuted plan is not
  evidence against M3.

## 6. Literature Evidence

### Local papers consulted

- GeneralPruner, *Scalable and Generalizable Correspondence Pruning via
  Geometry-Consistent Pre-training*: masked inlier reconstruction can learn
  transferable geometric representations without outlier interference. It
  operates on putative correspondence sets, not a raw-image matcher encoder.
- GIM, *Learning Generalizable Image Matcher From Internet Videos*: diverse
  pseudo-labeled video data and strong perspective augmentation improve matcher
  generalization, but require external matchers, robust fitting, and much larger
  data/compute than permitted here.
- OmniGlue: DINOv2-guided graph construction improves cross-domain matching but
  adds a foundation model at inference, conflicting with the efficiency goal.
- LiftFeat: training with pseudo surface-normal supervision can improve a
  lightweight descriptor, but depends on Depth Anything and retains a geometry
  branch at inference.

### Online update

Search date: 2026-09-30, arXiv sorted by submission date. RoMa-Omega
(`arXiv:2609.09507`) reports that feed-forward 3D reconstruction features are
strong matcher representations, but replacing M0R1 with a large 3D foundation
backbone would discard the measured efficiency advantage. A 2026-09-27 study
on geometric view synthesis (`arXiv:2609.33359`) shows that geometric synthetic
views can strongly help a self-supervised encoder while fixed-budget full
fine-tuning can hurt; this strengthens the case for a distinct pretraining
stage and matched total-budget controls. No newer paper found covers the exact
HPatches-only, training-only, zero-inference-overhead GCRP configuration.

### Provisional novelty gap

The defensible gap is not masked modeling, contrastive correspondence, EMA
teachers, or geometry augmentation individually. It is the exact combination:

> train-only masked cross-view geometry pretraining of the lightweight M0R1
> encoder/FPN, followed by removal of all pretraining heads and matched-budget
> full-matcher training under a sequence-isolated efficiency/non-inferiority
> contract.

Novelty remains provisional until broader paper review. The immediate purpose
is causal method recovery, not a final novelty claim.

## 7. M3 Method: GCRP

### 7.1 Data construction

Reuse the frozen HP2 roles as HP3 because neither HP2 development nor evaluation
was read. Pretraining reads only the 72 train-role sequences.

For a train pair `(I0,I1,H01)`, sample independent bounded projective transforms
`A0,A1` and strong independent photometric augmentations. The new known mapping
is:

```text
H' = A1 * H01 * inverse(A0)
```

Require at least 35% valid overlap. Sample rotation up to 30 degrees, log-uniform
scale in `[0.7,1.4]`, translation/crop, and bounded corner perspective. Preserve
pixel-center coordinates and masks exactly.

Within train-role only, freeze 64 pretraining-fit and 8 pretraining-probe
sequences, stratified 4 illumination/4 viewpoint for the probe. Probe sequences
are unread during pretraining and used only for the preregistered representation
gate. After the pretraining checkpoint is frozen, normal matcher fine-tuning may
use all 72 train sequences.

### 7.2 Architecture

The student is exactly the M0R1 encoder and FPN. An EMA copy of only those
modules processes unmasked weakly augmented views. Training-only components:

- a shared `128 -> 128` normalized projection head;
- a two-layer masked-token predictor;
- a small scale embedding for stride-4/8/16 features.

Mask 60% of valid stride-8 tokens in contiguous 4x4 blocks, independently in
both directions. Pretraining heads and the EMA copy are deleted after P0. Only
student encoder/FPN tensors enter M0R1 fine-tuning and inference.

### 7.3 Pretraining losses

For valid homography-corresponding locations:

```text
L_mask = 1 - cosine(predicted masked student token,
                    stop_grad(unmasked EMA target token))

L_corr = symmetric InfoNCE between corresponding stride-8 tokens,
         with same-image and opposite-image hard negatives outside 2 cells

L_geo  = symmetric expected-coordinate error of the stride-16 soft
         correlation distribution against H', normalized by image diagonal

L_vc   = VICReg-style per-dimension variance floor and off-diagonal covariance

L_pre  = L_mask + L_corr + 0.2 * L_geo + 0.05 * L_vc
```

EMA decay is `0.996`; InfoNCE temperature is `0.07`; clamp epsilon is `1e-8`.
No RANSAC, AUC, development/evaluation metric, external teacher, depth, pose,
or selected-ID feedback enters pretraining.

### 7.4 Fine-tuning and inference

Load only the GCRP student encoder/FPN into a fresh M0R1 model. Initialize pair
interaction, assignment, dustbins, and bilateral refiner from the exact paired
seed used by the scratch M0R1 control. Fine-tune every M0R1 parameter with the
unchanged M0R1 loss and decode.

Inference is byte-structurally the M0R1 architecture. No EMA teacher, mask,
projector, predictor, geometry head, or extra feature is callable.

## 8. Matched Controls And Budget

Three formal arms, two seeds:

1. `OfficialSLiM-HP-Scratch-30K`: 30k supervised steps.
2. `IGM-M0R1-Scratch-30K`: 30k supervised steps.
3. `IGM-M3-GCRP`: 10k GCRP steps plus 20k unchanged M0R1 fine-tuning.

All arms receive 30k optimizer steps, the same number of two-image exposures,
FP32, effective batch four, paired train sequence/pair stream, augmentation
seed ledger, optimizer family, final-step-only rule, and common evaluator.
Different objectives are the tested mechanism, not an uncontrolled budget
difference.

Pretraining S2 additionally compares correct-H GCRP to a paired shuffled-H
control with identical images, transforms, masks, steps, and architecture. The
shuffled control is diagnostic and is not promoted to formal.

Total six-trajectory credible forecast must fit 48 GPU-hours.

## 9. Funnel And Gates

### S0/S1

- reuse and hash-bind HP2 72/18/18 roles;
- freeze 64/8 pretrain-fit/probe inside train role;
- prove no dev/eval/pretrain-probe read during P0;
- prove no OfficialSLiM/D0/JCGA/foundation dependency;
- prove training-only heads are absent from inference state dict and call graph.

### S2 pretraining capacity

One seed, correct-H and shuffled-H, 10k steps:

- masked loss ratio `<=0.40`;
- every encoder/FPN and pretraining tensor receives finite nonzero gradients
  and updates; EMA changes only through EMA;
- no feature collapse: projected per-dimension standard deviation `>=0.05`;
- on the 8 train-role probe sequences, correct-H coarse mutual-NN recall@8px
  improves at least `+15 pp` over random initialization and `+10 pp` over
  shuffled-H;
- positive-minus-hard-negative cosine margin improves at least `+0.15` over
  random and `+0.10` over shuffled-H;
- all four viewpoint probe sequences improve over shuffled-H;
- checkpoint reload and deletion/absence of pretraining heads in the exported
  encoder pass.

This gate admits only representation initialization; it is not a matching
accuracy result.

### S3/S4 matcher expression

Load the exported encoder into M0R1 and run a 1500-step eight-pair tiny set with
four viewpoint pairs:

- loss ratio `<=0.25`;
- all eight pairs emit at least 20 matches;
- at least 7/8 pairs have at least 20 inliers at 3 px;
- all inherited/newly initialized groups receive gradients and updates;
- exact reload and inference absence of pretraining modules pass.

### Resource and formal

Time all three arms on real train-role workloads. If the six-trajectory credible
forecast exceeds 48 GPU-hours, stop without reducing the protocol. Otherwise
run two seeds per arm to fixed endpoints.

### Development admission

Read development only after all six endpoints are jointly frozen. M3 versus
fresh M0R1 must achieve:

- overall sequence-macro AUC@5 `>= +5 pp`;
- viewpoint AUC@5 `>= +10 pp`;
- both seed directions positive;
- illumination delta `>= -3 pp`.

Failure stops before HP3 evaluation.

### Evaluation success

Against fresh OfficialSLiM-HP-Scratch-30K:

- mean AUC@5 delta `>= -3 pp`;
- each seed delta `> -5 pp`;
- paired sequence-bootstrap 95% CI lower bound `> -5 pp`;
- viewpoint delta `>= -5 pp`;
- illumination delta `>= -3 pp`.

Against fresh M0R1-30K:

- mean AUC@5 gain `>= +8 pp`;
- viewpoint gain `>= +15 pp`;
- both seed directions positive.

Efficiency:

- inference parameter ratio to Official `<=0.25`;
- inference P95 ratio `<=0.25`;
- peak memory ratio `<=0.75`;
- exported M3 inference parameter keys exactly equal M0R1.

## 10. What This Experiment Can Reject

A valid pretraining-capacity failure rejects only the exact masking,
correspondence, geometry losses, transform distribution, optimizer, and 10k
recipe. It cannot reject geometry pretraining broadly.

A valid fine-tuning/generalization failure rejects the exact GCRP initialization
plus unchanged M0R1 20k recipe under HP3. It cannot reject external-data
pretraining, foundation features, depth supervision, or independent matchers.

It cannot establish a pristine benchmark result because all HPatches sequences
were historically consumed. It cannot establish fresh-scene pose utility.

## 11. Primary Risks

- synthetic/projective pretraining may specialize to planar homographies;
- the masked task may learn trivial source-to-target copying rather than useful
  descriptors;
- EMA targets may collapse without variance/covariance control;
- 72 HPatches train sequences may be too small for representation pretraining;
- extra training may improve optimization rather than representation, hence the
  matched 30k controls;
- stronger viewpoint augmentation may harm illumination performance;
- train-role probe selection can overfit the pretraining recipe, so it is used
  once under a frozen endpoint and never substitutes for development.

## 12. Forbidden Changes

- no DINO, VGGT, MASt3R, Depth Anything, external teacher, internet video, or
  MegaDepth training;
- no VASI, M2 geometry feedback, hard visibility gate, threshold/top-k rescue,
  learned RANSAC weight, or evaluator change;
- no development/evaluation read before its gate;
- no warm-start from M0R1 formal checkpoints;
- no deletion, overwrite, protocol reduction, best-checkpoint, or best-repeat.


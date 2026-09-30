# P37-20260926-IGM-M1-VASI-v1 Method Specification

Status: `FROZEN_FOR_ENGINEERING_AUDIT`
Experiment: `P37-20260926-IGM-M1-HP2`
Date: 2026-09-26
Name: `Viewpoint-Adaptive Span Interaction (VASI)`

## Motivation

IGM-M0R1 is efficient but its held-out viewpoint AUC@5 is `33.05` versus
OfficialSLiM-HP-Scratch `63.07`, while illumination is `+2.62 pp`. Evaluation
uses 1000 matches for both methods; the problem is not correspondence count.
Corner errors grow sharply with viewpoint strength, showing that wrong
long-range coarse assignments dominate and the unchanged 9x9 fine refiner
cannot recover them.

The next version changes one causal mechanism: cross-view context routing.
Encoder stages, FPN widths, M0R1 bidirectional dustbin assignment, mutual
decode, bilateral fine refinement, data supervision, optimizer, and evaluator
remain unchanged.

## VASI Module

### 1. Stride-16 guide interaction

Expose the already-computed encoder stage-4 features from both images. Project
width `192 -> 128`, apply one shared linear self-attention plus bidirectional
linear cross-attention guide block, and produce source-to-target and
target-to-source guide logits plus one null logit per query.

Each direction forms a categorical distribution over the opposite stride-16
grid plus null. From the non-null mass, compute the expected target coordinate
`mu=(mu_x,mu_y)`, diagonal standard deviation `sigma=(sigma_x,sigma_y)`, and
soft covisibility `c=1-p_null`. Empty opposite grids map to null one.

### 2. Adaptive-span exact cross-attention

For each stride-8 query, bilinearly upsample `mu`, `sigma`, and `c`. Construct a
`6 x 6` sampling grid centered at `mu`. Its half-span in each axis is

```text
r = clamp(2 + 1.5 * sigma, 2, 12)  # stride-8 cells
```

and sampling points cover `[-r_x,+r_x] x [-r_y,+r_y]`. Bilinearly sample the
opposite stride-8 contextual feature map at those 36 locations. Two VASI blocks
apply parent linear self-attention followed by exact four-head cross-attention
over the sampled tokens and the unchanged width-128 FFN.

The cross-attention logits receive a learned bias from normalized relative
offset `(sample_xy-mu)/r` through a shared `2 -> 32 -> 4` MLP. The returned
cross-view message is multiplied by soft covisibility `c`; there is no hard
visibility mask, crop, threshold, pruning, early exit, or selected-ID edit.

The same parameters and operations are used in both image directions. Padding
participates only through masks. Homography labels do not enter inference.

### 3. Unchanged assignment and refinement

The updated stride-8 tokens enter the exact M0R1 source/target categorical
candidate-plus-dustbin assignment. Mutual non-null argmax, confidence,
deduplication, 2048 pre-refinement cap, bilateral 9x9 stride-4 refinement, and
common top-1000 evaluator are byte-identical in semantics to M0R1.

## Training

The parent coarse/fine/cycle objective is unchanged. VASI adds only guide
supervision from the same train-role homography:

```text
L_route = bidirectional guide positive-set/null NLL
L_span  = bidirectional Gaussian NLL of valid projected stride-16 coordinates
L_total = L_parent + 0.2 * L_route + 0.05 * L_span
```

`L_span` is evaluated only for valid non-null projections. Standard deviation
is `softplus(raw_sigma)+0.25` and clamped to at most 12 stride-8 cells after
conversion. Unknown, padded, and out-of-image locations are excluded. No
RANSAC/homography AUC, teacher, development metric, or evaluation label enters
training.

## One-Mechanism Attribution

`IGM-M1-VASI` is compared with freshly trained `IGM-M0R1` on the identical HP2
stream. The contrast attributes the combined effect of uncertainty-routed,
soft-covisibility-weighted adaptive cross-view context. It does not separately
claim novelty or benefit for uncertainty, covisibility, positional bias, or
adaptive span.

M2 efficient geometry feedback and M3 representation pretraining are forbidden
in this experiment. OfficialSLiM is a separate matched-scratch baseline and is
not a runtime dependency.

## Literature And Novelty Boundary

Local sources consulted: ASpanFormer (uncertainty-driven attention span),
CoMatch (covisibility-guided token condensation and message suppression),
SceneGlue (explicit visibility), and SGFormer (shallow structural guidance).
Online arXiv search on 2026-09-26, sorted by submission date, found LoRetta
(`arXiv:2608.04106`) and RoMa-Omega (`arXiv:2609.09507`). LoRetta strengthened
the localization-before-registration recommendation; RoMa-Omega strengthened
the representation-pretraining alternative but would sacrifice the current
efficiency goal.

Adaptive span, covisibility, dustbins, and bilateral refinement are prior art.
Novelty is provisional and limited to this lightweight integration in an
independent scratch matcher under the exact efficiency/non-inferiority contract.
The immediate objective is empirical recovery, not a broad novelty claim.

## Expected Outcome And Limitations

VASI should improve viewpoint coarse localization while retaining most of
M0R1's efficiency. It may fail if the lightweight encoder lacks affine/viewpoint
invariance before routing, if guide distributions are overconfident, or if 36
samples cannot cover multimodal correspondence hypotheses. A valid failure
rejects only this VASI implementation/recipe and does not reject M2/M3 or
independent matchers.


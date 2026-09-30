# Prompt37 Independent HPatches Method Decision

Date: 2026-09-21
Status: `USER_REDIRECTED_METHOD_DESIGN__NO_EXPERIMENT_LAUNCHED`
Method ID: `P37-20260921-IGM-v1`
Working name: `Independent Geometry Matcher (IGM)`

## Corrected Evidence Record

E063-D0 was trained on MegaDepth train-only data. HPatches was evaluation-only
after endpoint freeze; it was not used to train D0. Untouched OfficialSLiM was
also pretrained outside HPatches. Therefore the existing HPatches numbers are
useful off-the-shelf/development observations, but they are not a controlled
HPatches-trained architecture comparison.

Under the consumed 108-sequence/540-pair HPatches480 protocol, D0 scored
`58.05 / 67.05 / 76.86` AUC@3/5/10 versus OfficialSLiM-O0
`55.10 / 64.68 / 74.00`. This is a historical internal gain only. It does not
answer whether an independent architecture is better under matched HPatches
training, and it cannot serve as a pristine final result.

## New Comparison Contract

The primary comparison will use HPatches for both training and evaluation, but
never the same sequences in both roles. Freeze a deterministic, illumination/
viewpoint-stratified split of the recovered 108 valid sequences into disjoint
train, development, and evaluation sequence sets. All images and derived pairs
from one sequence stay in one role. Model, checkpoint, threshold, resolution,
and method decisions may use train and development only; evaluation remains
unread until the endpoint is frozen.

Because all 108 sequences were historically consumed by earlier Prompt37
evaluation, the new evaluation split is a retrospective held-out comparison,
not a pristine benchmark or final paper test.

Primary baseline: `OfficialSLiM-HP-Scratch`, the OfficialSLiM architecture
trained from random initialization on exactly the same HPatches train split,
pair stream, geometric labels, image resolution, data exposure, precision,
and external homography estimator/evaluator as IGM. This comparison isolates
architecture under a common small-data regime. It must not load the published
MegaDepth checkpoint.

Secondary references, reported separately and never used as the primary
architecture control:

- untouched released `OfficialSLiM-O0`, an off-the-shelf transfer reference;
- historical `E063-D0`, which remains MegaDepth-trained and OfficialSLiM-
  dependent;
- an optional published-initialization fine-tune table only if both methods
  have a declared, comparable external initialization. It cannot be pooled
  with the scratch comparison.

## Independent Method Boundary

IGM must be a standalone image matcher. Its inference graph begins with raw
images and owns its encoder, feature pyramid, cross-view interaction,
assignment, confidence/null handling, and fine refinement. It may use generic
neural operators and generic non-matcher initialization, but it must not load,
call, wrap, freeze, or modify OfficialSLiM, D0, JCGA, their feature tensors,
fine matcher, dual-softmax, quota, RRU, or checkpoints.

All methods may feed their final correspondences to one common read-only
homography estimator and HPatches evaluator. The common evaluator is not part
of IGM and must be identical across methods.

The standalone base `IGM-M0` must be functional before any proposed mechanism
is attributed. Its minimum architecture is:

```text
raw image pair
-> independent shared-weight multiscale encoder and feature pyramid
-> independent coarse cross-view interaction
-> explicit partial assignment with matchability/null
-> bilateral fine/subpixel refinement
-> final correspondences
-> common external homography estimator/evaluator
```

This is not permission to reproduce SLiM under another name. Source and module
dependency audits must show no OfficialSLiM/D0/JCGA runtime dependency.

## One-Mechanism Method Versions

Each successor changes one causal mechanism relative to the preceding frozen
standalone version. A later version cannot absorb an unverified mechanism and
then attribute the combined result retrospectively.

### IGM-M1: Covisibility-Conditioned Sparse Context

Add an explicit cross-view covisibility prediction that controls which tokens
send contextual messages. It acts inside IGM's own interaction module, not as
a post-hoc gate on OfficialSLiM and not as a direct selected-ID edit. The
mechanism targets non-overlap distraction and redundant all-token interaction.

Closest local work: LightGlue matchability/adaptive pruning, JamMa joint linear
interaction, CoMatch covisibility-guided condensation, and SceneGlue explicit
visibility. The provisional gap is an independent HPatches-trained matcher
whose covisibility controls sparse cross-view context and is evaluated under a
strict matched scratch protocol. Novelty is not yet verified.

Minimum falsification question: does covisibility routing improve held-out
sequence matching while reducing context computation, beyond an equal-
parameter no-routing control? It cannot establish the value of later geometry
feedback or representation pretraining.

### IGM-M2: Action-Consistent Efficient Geometry Feedback

Add a training-only low-cost geometry-feedback estimator after IGM's own
assignment/conflict resolution. It must be calibrated against exact matched
homography utility on train-only data and must not enter inference. The target
is the measured JCGA lesson that exact sampled robust-estimator feedback can be
an order of magnitude slower than supervised steps.

The mechanism is distinct from historical feature-side proxy/gate routes only
if its gradient direction and helpful/harmful set ordering agree with exact
action-path geometry under held-out train sequences. A shuffled-teacher or
proxy-only success is insufficient.

Minimum falsification question: can the estimator preserve exact-feedback
gradient/order information at materially lower cost? Failure rejects only the
estimator, not IGM-M1 or geometry feedback as a family.

### IGM-M3: Geometry-Consistent Representation Pretraining

Add masked correspondence recovery, transformation consistency, and
covisibility-aware representation learning using HPatches training sequences
only. Fine-tuning and endpoint evaluation still use IGM's real assignment and
common homography metric. This targets the repeated local finding that strong
training/interface proxies do not guarantee cross-sequence geometry.

Closest local work includes GeneralPruner-style masked geometric pretraining;
the 2026-09-08 RoMa-Omega report further suggests that geometry-grounded 3D
representations can strengthen matchers. The provisional gap is a lightweight,
HPatches-only pretraining objective tied to IGM's assignment, not a large
foundation-backbone substitution.

Minimum falsification question: does pretraining improve sequence-held-out
geometry over the identical IGM-M2 architecture and training endpoint? It
cannot by itself establish broad cross-dataset generalization.

## Quality And Claim Boundary

- Freeze the sequence split and file hashes before reading evaluation metrics.
- Use structured dataset APIs and preserve sequence identity.
- Match training data, exposure, resolution, precision, and evaluator for the
  primary scratch comparison; report model size, FLOPs, latency, and memory.
- Require activation, gradient, update, positive-control, reload, and action-
  path evidence for each new module.
- Keep complete trajectories; a falling training loss is not an efficacy
  result, and training inadequacy cannot reject a method family.
- Never combine M1/M2/M3 in one attribution step.
- Existing HPatches results remain consumed historical evidence. This route
  produces a controlled internal comparison, not a pristine final claim.
- No SourceMux, SAR, OSDataset, heldout1200, Gate2/test/final/protected data,
  or artifact deletion/overwrite.

## Literature Review Record

Local library consulted: LightGlue, JamMa, CoMatch, SceneGlue, and the
`Prompt37_跨场景安全匹配_文献地图_20260731.md` synthesis. Online arXiv search date:
2026-09-21, sorted by submission date. New relevant results were RoMa-Omega
(`arXiv:2609.09507`) and co-occurrence-aware quadratic assignment
(`arXiv:2609.17905`). They support geometry-grounded representations and
structured assignment respectively, but do not establish novelty for IGM and
do not justify importing a large 3D backbone or NP-hard solver into the first
standalone version.

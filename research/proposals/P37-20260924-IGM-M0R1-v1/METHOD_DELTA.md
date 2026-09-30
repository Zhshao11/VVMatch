# P37-20260924-IGM-M0R1-v1 Method Delta

Date: 2026-09-24
Parent: `P37-20260921-IGM-v1 / IGM-M0`
Experiment: `P37-20260924-IGM-M0R1-HP1`
Status: `FROZEN_SINGLE_RESCUE`

## Causal Diagnosis

M0 optimized positive-set mass but decoded a match only when one dual-softmax
cell exceeded both `1 - row_pair_sum` and `1 - column_pair_sum`. On a dense
grid, dual-softmax cell mass and its row/column sums begin very small while the
constructed null approaches one. The supervised loss could fall without
making any discrete match admissible. The observed endpoint was exactly this
failure: loss ratio `0.229374`, complete gradients/updates/reload, but zero
matches on all eight tiny pairs.

## M0R1 Change: Consistent Bidirectional Dustbin Assignment

Keep the independent M0 encoder, FPN, two dense pair-interaction blocks,
bilateral 9x9 fine refinement, data, optimizer, and evaluator unchanged.
Replace only M0's assignment probability and loss interface.

For valid source token `i`, target token `j`, pair logit `S_ij`, source-null
logit `n_i`, and target-null logit `m_j`:

```text
A_i = softmax([S_i1 ... S_iN, n_i])
B_j = softmax([S_1j ... S_Mj, m_j])
C_ij = sqrt(A_ij * B_ji)
```

`n_i` and `m_j` are the existing source/target matchability linear heads
reinterpreted as null logits; no new head, threshold, quota, or post-hoc gate
is added. Invalid/padded candidates are masked before softmax. Rows with no
valid opposite-image token have null probability one.

Inference emits `(i,j)` iff `argmax(A_i)=j`, `argmax(B_j)=i`, and neither
categorical argmax is null. Confidence is `C_ij`. Stable exact-coordinate
deduplication, the 2048 pre-refinement cap, bilateral refinement, and common
top-1000 evaluator remain unchanged. There is no confidence threshold.

Training uses the same homography labels but the exact inference
parameterization:

```text
L_src = positive-set NLL over A, or source-null NLL for trustworthy unmatched
L_tgt = positive-set NLL over B, or target-null NLL for trustworthy unmatched
L_coarse = 0.5 * (L_src + L_tgt)
L_total = L_coarse + L_fine + 0.1 * L_cycle
```

Unknown, padded, and out-of-image projections remain excluded. No auxiliary
pair-mass proxy, match-count loss, teacher, RANSAC loss, or decode override is
allowed.

## Why This Is The Admitted Repair

This change directly aligns the optimized categorical event with the discrete
action and preserves explicit unmatched handling. Merely lowering a null
threshold, initializing a matchability bias, adding a count penalty, or
forcing top-k matches would hide the structural mismatch and is forbidden.

LightGlue and SuperGlue establish partial/dustbin assignment as prior art;
M0R1 does not claim novelty for dustbins. It is the corrected standalone base
needed before testing IGM-M1 covisibility, M2 geometry feedback, or M3
pretraining. None of M1-M3 is implemented in this rescue.

## What It Can Reject

A faithful tiny failure rejects the exact bidirectional-dustbin assignment,
M0 representation, unchanged fine refiner, losses, optimizer, and 1000-step
rescue recipe. It cannot reject independent matching, covisibility context,
geometry feedback, representation pretraining, or broad partial assignment.


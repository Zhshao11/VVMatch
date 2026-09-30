# P37-20260930-IGM-M3-GCRP-v1 Method Specification

Status: `FROZEN_FOR_ENGINEERING_AUDIT`
Experiment: `P37-20260930-IGM-M3-HP3`

The complete method decision, evidence, architecture, mathematical losses,
controls, gates, limitations, and claim boundary are frozen in:

`research/PROMPT37_M3_GCRP_METHOD_DECISION_20260930.md`

Normative implementation summary:

1. Reuse exact M0R1 inference architecture; do not use VASI.
2. P0 pretrains only M0R1 encoder/FPN for 10k steps on HP3 train-role pairs
   transformed by known composed homographies.
3. Use 60% contiguous token masking, EMA unmasked targets, masked descriptor
   reconstruction, symmetric InfoNCE, soft expected-coordinate geometry, and
   variance/covariance regularization.
4. Delete all P0-only modules and export only encoder/FPN tensors.
5. Initialize a fresh full M0R1 model with the exported encoder/FPN and paired-
   seed scratch initialization for every other tensor; fine-tune 20k steps.
6. Compare with fresh 30k-step M0R1 and OfficialSLiM scratch controls under
   identical HP3 exposure and final-only evaluation.

No external pretrained model or inference dependency is permitted.


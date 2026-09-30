# Focused Review 1

Verdict: `CONDITIONAL_APPROVE_PREFLIGHT_ONLY`; tiny is not yet approved.

Findings resolved in build_r3:

1. `P1` The draft sampled the opposite raw stride-8 state before that side's
   self-attention. It now computes source and target post-self snapshots first,
   then performs both cross directions from those snapshots, matching VASI.
2. `P1` Endpoint validation required four entries although HP2 has three arms
   by two seeds. It now requires the exact six-key lattice.
3. `P1` The draft HP2 protocol test imported the M0R1 runner and old seeds.
   It now tests the HP2 runner, 72/18/18 roles, new seeds, six endpoints and
   recomputed development gate.
4. `P1` GT/predicted/permuted route controls were not provenance blockers.
   The runner now requires a hash-closed passing route/student preflight before
   tiny, budget or formal; checkpoints and endpoint freeze bind its SHA.
5. `P2` S0 validation expected the old status string. It now accepts only the
   HP2-specific `S0_PASSED_EVALUATION_UNREAD` state.

The assignment, mutual decode, 2048 cap, bilateral refiner and stable dedup
match M0R1 under exact-initialization behavioral tests. No M2/M3, learned
geometry feedback, hard visibility gate/crop, pretrained matcher dependency,
or evaluation payload path is present in the admitted implementation.

Remaining gate: actual preflight measurements. Any failed frozen threshold
stops before tiny and is not a method-family conclusion.

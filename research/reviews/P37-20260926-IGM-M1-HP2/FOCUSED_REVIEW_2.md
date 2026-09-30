# Focused Review 2

Verdict: `STOP_BEFORE_TINY__POSITIVE_CONTROL_INCONCLUSIVE`.

Reviewed bytes:

- VASI model `3fbeab8c...d999c`
- VASI loss `9189fa62...5840`
- HP2 runner `f7b7792d...ad42`
- preflight `5eb12d70...1211`
- build_r6 tests `eac5c4aa...7042` (`12/12 PASS`)
- terminal preflight R3 card `8befe527...938`

Implementation fidelity findings:

- inherited encoder/FPN, affinity/null and refiner tensors initialize exactly
  against paired-seed M0R1;
- assignment, mutual non-null decode, 2048 cap, bilateral 9x9 refinement and
  stable dedup behavior match the parent controls;
- VASI applies shared stride-16 guide interaction and categorical null routes,
  then two blocks whose bidirectional exact sampled cross-attention consumes
  both post-self snapshots; padding masks only invalid samples;
- no M2/M3 component, geometry-reward training, hard visibility gate/crop,
  threshold/top-k rescue, pretrained matcher dependency, development read or
  evaluation read is present;
- HP2 S0 is hash-closed at 72/18/18 sequences with zero sequence/file role
  intersections and evaluation payload unread;
- endpoint validation now requires the exact six-arm/seed lattice and binds
  route preflight, tiny, source, data and evaluator hashes.

Measured controls:

- GT-route coverage `1.0`;
- predicted-route coverage `1.0`;
- permuted-route coverage `0.0`;
- all cases emit matches;
- every trainable tensor receives finite nonzero gradients and updates;
- checkpoint reload is exact;
- GT-route median refined transfer is `1.7721316 px`, failing the frozen
  `<=0.5 px` gate. Predicted-route median is `1.7734146 px`.

R1 was invalid for this conclusion because it did not send GT/permuted centers
through the actual downstream path. R2 corrected that path but used an
unfrozen, excessively strong control magnitude. R3 kept the method, threshold,
600-step optimizer and four transformation families unchanged while using the
parent-control magnitude; the same refined-transfer gate still failed.

Highest demonstrated rung is `S2_action_expressive`: routing changes the
intended action and is trainable, but the same-path positive control does not
recover the required refined correspondence accuracy. This is insufficient to
run the 1500-step tiny and insufficient for an architectural rejection. Mark
the experiment `inconclusive`; do not reject VASI, M1, M2/M3, adaptive span,
covisibility, or independent matchers broadly.

# IGM-M1 HP2 Review Packet

Review scope: frozen VASI implementation, HP2 role seal, training/evaluation
gates, and same-parameterization preflight. Frozen method/spec/sheet SHA256:
`2658431f...0675`, `f2b2b5...5612`, `4474e4...21e8`,
`a2f653...e216` respectively.

Changed/new runtime files:

- `tools/igm_m1_vasi/{model.py,losses.py,data.py,__init__.py}`
- `tools/prompt37_igm_m1_hp2_runner.py`
- `tools/prompt37_igm_m1_launch_guard.py`
- `tools/prompt37_igm_m1_vasi_preflight.py`
- `tools/prompt37_igm_hp2_s0.py`
- `tools/test_prompt37_igm_m1_vasi.py`
- `tools/test_prompt37_igm_m1_hp2_protocol.py`

High-risk mappings:

- stage-4 guide distribution/null/mu/sigma/covisibility: `model.py::guide_routes`
- post-self adaptive 6x6 sampled exact attention and offset bias:
  `AdaptiveSpanInteractionBlock`, `interact`
- unchanged M0R1 assignment/refiner/cap: `assign`, `categorical_assignment`,
  `mutual_matches`, `refine`, `stable_filter_and_deduplicate`
- parent plus route/span losses: `losses.py::compute_igm_loss`
- role seal/source/tiny/endpoint/development/evaluation gates:
  `prompt37_igm_m1_hp2_runner.py`
- actual student controls: `prompt37_igm_m1_vasi_preflight.py`

Automated result: zzvr build_r3, 12/12 tests PASS in
`zzvr(ubuntu):/mnt/sda/zhushihao/moveExperiment/p37_igm_m1_hp2_20260926/build_r3/UNIT_TESTS.txt`.
HP2 S0: 72/18/18 sequence roles, 360/90/90 native pairs, sequence/file
intersections empty; evaluation payload unread.

Unresolved before tiny: execute and hash-freeze the real same-parameterization
preflight; review its route coverages, permutation separation, refined error,
per-tensor gradients/updates and reload. No development/evaluation reads.

# Prompt37 Experiment Engineering Handoff

Date: 2026-09-30 (Asia/Shanghai)
Workspace: repository root
Current experiment: `P37-20260930-IGM-M3-HP3`
Disposition: `inconclusive`; frozen P0/S2 gate stopped the funnel before probe.

## Recent Conversation And Decision Context

The user authorized method and experiment engineering to prepare and run
in-scope experiments without routine approval, while keeping frozen scientific,
data, compute, host, and artifact boundaries. The visible-visible independent
matcher route compared IGM against untouched OfficialSLiM-HP-Scratch under
matched HPatches sequence-disjoint training. M0R1 was efficient but trailed
Official by 13.70 percentage points at AUC@5, with a 30.03-point viewpoint
deficit. The corrected VASI/M1 S2 control then failed its matched viewpoint
action ceiling and stopped before tiny. Method decision froze M3 GCRP as a
training-only encoder/FPN pretraining route, not an inference architecture
change.

HP3 reused the sealed HP2 72/18/18 sequence roles. S0 froze a 64-sequence
pretrain-fit and eight-sequence train-role probe split. Five unit tests and two
focused reviews passed. The correct-H and shuffled-H 10,000-step P0 jobs ran
on `zzvr(ubuntu)` with the same 40,000-exposure stream.

## Terminal Evidence

- Correct-H watchdog `20260930_005030_7b9b092c` completed 10,000 optimizer
  steps. Its full loss trajectory is finite; all 83 trainable tensors received
  gradients and updated. EMA updated outside the optimizer, checkpoint reload
  was exact, and encoder-only export matched the checkpoint. Masked-loss
  last20/first20 ratio `0.4347184026` fails the frozen `<=0.40` admission gate.
- Shuffled-H watchdog `20260930_005030_992675ed` completed 10,000 steps and
  the identical exposure stream, but independent trajectory validation found
  four nonfinite total-loss rows, first at step 3807. Runner-reported success
  does not make it a valid paired representation control.
- Paired exposure SHA-256:
  `bd846fdcc6c804af31e709d733fa048800f7f5909272bc87471a7597f98a2311`.
  Raw card and trajectory SHA-256 values are in the terminal JSON card.
- The correct-H gate failure independently stops the funnel. The shuffled-H
  defect further prevents the correct-versus-shuffled representation claim.
  The eight probe sequences and all development/evaluation payloads remain
  unread. No matcher tiny, budget, formal, evaluation, or pose run occurred.

The highest demonstrated evidence is S3 pretraining trainability. The exact
P0 recipe failed admission, but there is no valid broad rejection of GCRP,
geometry-consistent pretraining, independent matching, or pose utility.

## Evidence And Closure

- Frozen experiment sheet:
  `research/experiments/P37-20260930-IGM-M3-HP3.md`
- Frozen original engineering handoff:
  `research/experiments/P37-20260930-IGM-M3-HP3_ENGINEERING_HANDOFF.md`
- Local reviewed terminal cards:
  `research/results/P37-20260930-IGM-M3-HP3/RESULT_CARD.md` and `.json`
- Local raw mirrors: sibling `p0_correct_r1/` and `p0_shuffled_r1/` beneath
  the terminal result directory.
- Remote raw roots (preserve unchanged):
  `zzvr(ubuntu):/mnt/sda/zhushihao/moveExperiment/p37_igm_m3_hp3_20260930/p0_correct_r1`
  and
  `zzvr(ubuntu):/mnt/sda/zhushihao/moveExperiment/p37_igm_m3_hp3_20260930/p0_shuffled_r1`.
- Remote reviewed-card mirror (new root):
  `zzvr(ubuntu):/mnt/sda/zhushihao/moveExperiment/p37_igm_m3_hp3_20260930/postreview_terminal_r1`.
  MD SHA-256 `64d2f6f29928c3a8fb352bde2b346b66991c4af13646cf8ff5448c824a185c45`;
  JSON SHA-256 `8249602af971150e1a043cbc14edfeecfabcab3bf8b32d9861f3ca1251fe7ef9`.
  Both were checked byte-identical locally and remotely.
- `research_archive/implementation_audit.yaml`,
  `research_archive/experiment_registry.yaml`, and current research state,
  index, task, contract, method-delta, and decision-log files record the stop.
  `ruby research_archive/tools/validate_archive.rb` passed.
- The method-decision task `01a0c2d2-5feb-7fb2-b9bc-a7012d490ba7` was
  notified of the terminal evidence and claim boundary.

## Instructions For The New Engineering Task

Read `AGENTS.md`, `research/CURRENT_STATE.md`, `research/METHOD_DELTA.md`,
`research/INNOVATION_CONTRACT.yaml`, and the terminal card first. Verify that
the terminal evidence and archive remain coherent, then wait for a newly frozen
method decision before implementing another experiment. Do not restart either
P0 job, read the eight probe sequences or development/evaluation payloads,
launch matcher tiny/formal work, relax the `<=0.40` gate, or reinterpret the
shuffled-H nonfinite trajectory as valid. Any scientifically new route needs a
new versioned experiment root and the required archive/contract preflight.

Do not use SourceMux or `bl_tunnel`; do not delete, overwrite, move, or compress
raw artifacts. Every remote evidence path must state its execution host.

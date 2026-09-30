# P37-20260930-IGM-M3-HP3 Engineering Handoff

Owner: Prompt37 visible-visible experiment engineering Sol
Date: 2026-09-30

Execute the frozen M3 experiment in
`research/experiments/P37-20260930-IGM-M3-HP3.md`. Read the complete decision
and proposal directory named there. Work directly in the shared workspace; no
worktree.

Engineering owns implementation, automated checks, two focused reviews,
ordinary fixes, watchdog/queue handling, artifacts, audit/registry updates,
archive validation, and terminal result cards. Do not return code, test,
environment, provenance, or runtime errors to the user or method task unless
the only repair changes frozen method/data/budget/optimizer/evaluator semantics.

After gates pass, proceed autonomously through pretraining control, matcher
tiny, resource preflight, formal, development gate, and conditional evaluation.
Use zzvr first; all long or queued GPU work uses watchdog. Do not reduce arms,
seeds, steps, controls, or metrics. Every terminal stop writes RESULT_CARD.md
and RESULT_CARD.json with host-qualified remote paths and hashes.

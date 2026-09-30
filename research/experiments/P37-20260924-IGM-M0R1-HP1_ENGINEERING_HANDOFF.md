# P37-20260924-IGM-M0R1-HP1 Engineering Handoff

Owner: Prompt37 visible-visible experiment engineering
Date: 2026-09-24

Implement and execute the frozen single rescue in
`research/experiments/P37-20260924-IGM-M0R1-HP1.md`, using the proposal files in
`research/proposals/P37-20260924-IGM-M0R1-v1/`.

Engineering owns implementation, automated checks, focused reviews, fixes,
watchdog submission, queue/resource handling, artifact collection, audit
records, and terminal result cards. Fix ordinary code, test, environment,
provenance, and runtime errors autonomously without asking the user or method
task. Return only if a proposed fix would change the frozen assignment/loss/
data/evaluator semantics.

After all preflight/review gates pass, submit tiny directly. After tiny passes,
continue directly through budget preflight and conditional formal. Long or
queued GPU jobs must use watchdog and `zzvr` first. Do not reduce arms, seeds,
steps, or evaluation rules. Every terminal stop must include `RESULT_CARD.md`
and `RESULT_CARD.json`, host-qualified absolute remote paths, implementation
audit update when applicable, and archive validation.

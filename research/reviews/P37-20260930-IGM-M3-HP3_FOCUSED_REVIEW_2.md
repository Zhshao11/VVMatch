# IGM-M3 HP3 Focused Review 2

Verdict: `APPROVE_P0_CORRECT_AND_SHUFFLED_WATCHDOG`.

Final build_r3 tests pass `5/5`. Runtime manifest binds all method, proposal,
sheet, runner, GCRP, M0R1 and data-helper bytes. S0 binds the inherited HP2
roles and the 64-fit/8-probe partition; probe/dev/eval payloads are unread.

Correct-H and shuffled-H use identical deterministic fit rows, transforms,
masks, exposures, model initialization, optimizer, 10k steps and architecture.
Shuffled-H differs only by cycling the four homographies inside each optimizer
batch. Student input receives real contiguous token masking; EMA sees unmasked
views and is outside the optimizer. Mask reconstruction, symmetric hard-
negative InfoNCE, symmetric expected geometry and multiscale variance/
covariance are active. Export is exact M0R1 encoder/FPN state only.

Probe remains blocked until both step-10000 endpoints, reloads, exports and
exposure-stream hashes freeze. Any P0 integrity failure stops before probe; any
representation gate failure stops before matcher tiny.

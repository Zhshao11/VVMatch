# Git Change Inventory - 2026-09-30

## Repository State

- Git root: this workspace.
- Branch: `main`, unborn; there are no commits and no tracked files.
- Remote: none configured.
- `git diff` and `git diff --stat` are empty because every project file is
  currently untracked, not because the workspace is unchanged.
- `.git` contains about 1.8GB of unreachable/Codex checkpoint objects. The only
  ref is a Codex turn-diff tree containing two recovered-chat files. It is not
  usable project history and has not been deleted.

## Work Completed

### Core matching methods

- E063-D0 DeformDesc feature-side improvement, retained as the best validated
  accuracy-improving hybrid method.
- Independent CNN/FPN matcher (`IGM-M0`).
- Correct bidirectional candidate-plus-dustbin assignment (`IGM-M0R1`).
- Viewpoint-adaptive span interaction research branch (`IGM-M1 VASI`).
- Training-only geometry-consistent representation pretraining branch
  (`IGM-M3 GCRP`).

### Matching and filtering

- Explicit null assignment and mutual non-null decoding.
- Stable confidence ordering, duplicate removal, and correspondence caps.
- Bilateral source/target fine refinement.
- Padding, valid-region, pixel-center, and coordinate-restoration checks.

### Geometry and evaluation

- Known-homography label generation and synthetic transformation controls.
- Common OpenCV homography/RANSAC evaluator with fixed top-1000 inputs,
  deterministic repeats, sequence-macro AUC, paired bootstrap, and no
  best-of-repeat selection.
- Matched-scratch OfficialSLiM adapter. The clean OfficialSLiM source remains an
  external dependency; no checkpoint is used in the primary scratch control.

### Reproducibility and audit

- Deterministic HP1/HP2/HP3 sequence-role generators.
- Source, data, sample-stream, checkpoint, gradient/update, reload, endpoint,
  and forbidden-read audits.
- Unit/protocol tests and frozen method/function/budget contracts.
- Terminal result cards and scoped archive classifications.

## A. Recommended For Git

- `.gitignore`, `README.md`, `requirements-igm.txt`.
- `tools/igm_m0/`, `tools/igm_m0r1/`, `tools/igm_m1_vasi/`,
  `tools/igm_m3_gcrp/`.
- Portable E063 DeformDesc core and module test, plus the E063 method/function
  contract. The full host-bound historical runner remains in category B.
- `tools/prompt37_igm_*.py`, `tools/official_slim_hp_scratch_adapter.py`, and
  `tools/test_prompt37_igm*.py`.
- IGM method decisions, proposals, function contracts, budget files, experiment
  sheets, handoffs, final focused reviews, method/innovation summaries, and the portable
  `RESULTS_SUMMARY_IGM.md`.
- Archive audit standard, schemas/templates, validation tools, and current
  design summaries; exclude full artifact mirrors.
- This inventory and `research/REPRODUCIBILITY_IGM.md`.

## B. Can Be Committed After Additional Curation

- Legacy non-IGM `tools/prompt37_*` scripts: scientifically valuable but large,
  versioned inconsistently, and often contain host-specific paths. They should
  be grouped into a separate historical commit after path/config cleanup.
- Older JCGA/E-series experiment configs: useful provenance, but frozen runtime
  JSON/YAML files contain local absolute paths. Prefer sanitized examples plus
  immutable local originals outside the public commit.
- `gpu_keeper/`: source and tests are reusable, but zzvr configs/services and
  personal email settings must remain ignored. This belongs in a separate
  infrastructure commit, not the matcher-method commit.
- Full `research_archive/inventory/` and old design material: compact enough,
  but should be reviewed for local paths and relevance before inclusion.
- `research_archive/experiment_registry.yaml`, `implementation_audit.yaml`,
  and `legacy_family_registry.yaml`: scientifically important, but they contain
  historical local absolute paths. Keep them local until a path-preserving,
  schema-compatible portability migration is designed.
- Historical root Markdown files predating IGM should be placed under a clear
  `docs/history/` structure before being added.
- Raw terminal cards, `CURRENT_STATE.md`, `DECISION_LOG.md`,
  `EXPERIMENT_INDEX.md`, `TASKS.md`, `WORKFLOW_PROTOCOL.md`, and `AGENTS.md`
  contain host-local paths, missing local-card links, or broad
  operational history. Keep them local for the initial portable commit and
  publish sanitized derivatives instead of rewriting frozen evidence.

## C. Excluded From Git

- `.codex*` recovery, remote mirror, build, and review directories.
- `tmp/`, downloaded paper repositories, HPatches ZIP/images, cached tensors,
  and MegaDepth/other datasets.
- Checkpoints and binary state (`*.pt`, `*.ckpt`, `*.pth`, `*.npz`, etc.).
- Raw logs, `__pycache__`, `.pyc`, local virtual environments, and coverage
  caches.
- Large result payloads: exposure streams, full trajectories, raw pair exports,
  optimizer histories, per-pair tensor diagnostics, and remote artifact mirrors.
- `research_archive/experiments/` full mirrors (about 918MB).
- Host-specific GPU/watchdog configs, systemd service files, SMTP secret paths,
  and personal notification addresses.
- Frozen runtime manifests with machine-specific absolute paths unless later
  transformed into explicitly documented examples.

## `.gitignore` Changes

Added patterns for all categories above. The complete `research/results/`
tree stays local; portable outcomes live in `research/RESULTS_SUMMARY_IGM.md`.

## Reproducibility Status

The curated IGM source set contains:

- method implementations;
- deterministic split generators;
- train/evaluate/aggregate entry points;
- pinned validated dependency versions;
- input format and command documentation;
- fixed metric definition;
- tests and scientific gates;
- a portable, non-fabricated results summary.

Reproducing the OfficialSLiM scratch arm requires an external clean OfficialSLiM
source checkout supplied by path. Reproducing GPU numbers also requires a
compatible CUDA/PyTorch environment and the HPatches dataset, neither of which
is stored in Git.

## Proposed Commit Structure

Because this is an initial repository with no history, use multiple commits:

1. `chore: establish repository hygiene and reproducibility docs`
2. `feat: add independent image matching research pipeline`
3. `docs: record method decisions and validated experiment outcomes`

Do not push until a remote is configured and the user explicitly requests it.

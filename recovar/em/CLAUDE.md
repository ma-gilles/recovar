# EM / RELION parity contract

Correctness, GPU performance, then clarity. Complete RECOVAR cleanup **EM first,
GUI excluded**, before new-engine development. Root instructions also apply.

## Start and ownership

- Read [current EM status](../../docs/development/em_status.md) and the current
  work-package handoff. Load historical evidence only for the active question.
  Follow [the efficient workflow](../../docs/development/agent_workflow.md).
- State scope: structural, docs, diagnostic, algorithmic, performance or PR
  preparation. Establish one measurable question and its cheapest useful check.
- Before validation, record HEAD, branch, `git status --short --branch`,
  `git diff HEAD --stat`, diff SHA-256 and untracked inputs. Confirm required
  ancestors with `recovar.em.diagnostics.parity_provenance`. Names are not provenance.
- Preserve unrelated changes, pinned PR158 controls and every queued/running
  source snapshot. One writer per source/build; consult the status-linked board.
  The primary owns integration and claims. Parallel work requires authorization
  and [SUBAGENTS.md](SUBAGENTS.md); use bounded, disjoint scopes.
- EM APIs may change for clarity; migrate callers/tests/docs together and remove
  unused forwarding wrappers. Preserve scientific defaults, casts, reduction
  order, JIT boundaries, memory lifetime and non-EM APIs/formats during
  structural work. Keep numerical and runtime repairs separate.
- `recovar/em/` is the refinement implementation root. Keep responsibility
  package initializers free of execution imports. Import `refine_single_volume`
  from `recovar.em.refinement.iteration_loop`, K-class execution from `recovar.em.classification.k_class`
  and result assembly/types from `recovar.em.classification.k_class_results`. Helpers and
  diagnostics must not initialize schedulers, dense/local engines or sparse
  scoring. EM/VDAM APIs, CLIs and historical Python object names have no backward
  compatibility requirement; migrate maintained callers to current owners.
  Preserve the main heterogeneity pipeline APIs and saved formats.
  Earlier independent EM/covariance formulations belong in `reference/`;
  replay and score-audit modules belong in `diagnostics/`.

## Scientific rules — never waived by cleanup or cost reduction

- Intended production EM is **float32**. Double is a diagnostic reference, not a
  production parity remedy. Never enable double scoring/projection/M-step by
  default or claim completion from double-only runs. A policy change requires
  the user. Preserve deliberate existing higher-precision metadata, host and
  necessary numerical operations; no blanket narrowing or non-EM changes.
- Compare matched inputs/state/candidates. A gap shrinking in double alone does
  not prove roundoff: check serialization, casts, semantics and float32 bounds.
  Report effective precision for scoring, projection, accumulation and M-step
  separately; inherited F64/C128 stages are numerical work, not just metadata.
- Preserve canonical source sampler Euler angles as metadata and derive matrices
  for computation. Do not reconstruct known angles from rounded matrices during
  cleanup. Carry their identity through selection, permutation and class-prior
  layouts. Treat missing canonical metadata as an explicit API/design question;
  keep legacy conversion fallbacks separate until reviewed, not silently removed.
- Find the first divergent iteration/half/class/particle/pass/state field, then
  replay fixed state and candidates. Compare scores, posteriors, poses and
  accumulators. If fixed-state arithmetic agrees, move one state boundary earlier.
  Confirm RELION source/dump behavior, add a failing targeted regression, make
  the smallest repair, climb validation, and record negative findings as well.
  Follow the [investigation/capture procedure](../../docs/development/em_parity_runbook.md#investigation-loop)
  for deep parity work; its state inventory is required.
- GPU score/Pmax gaps around `1e-4` are normally arithmetic-level parity;
  investigate reproducible `1e-3` gaps or systematic drift. Require exact discrete
  choices outside the error band. A near-tie flip needs measured competing scores
  and margins; never dismiss it without evidence. Convergence iteration and
  finalization must match exactly unless strong evidence establishes otherwise.
- Only shellwise FSC, FSC-AUC and established FSC score/resolution summaries
  against GT and RELION are map-quality gates. Correlation is diagnostic only.
  K=4 requires Hungarian matching and per-class results; no averaging away a
  poor class. Missing measurements are not passes. Never widen tolerances, edit
  baselines or change milestone gates without an explicit user decision.
- Near-perfect RELION parity for K1 auto-refine and exactly K4 classification
  precedes speed qualification. RELION semantics are the default during closure.
  Strict oracle mode reproduces the pinned GUI workflow including firstiter_cc;
  later opt-in quality differences must be named, tested and GT-qualified.
  Major policies belong in typed configuration/CLI options, not env-only forks.
  Never label intentional differences strict parity or tune until outputs agree.
- Preserve the reviewed final-grid-correction default (off); the strict target
  specifies on. Resolve this discrepancy separately with explicit qualification.
  Preserve `run_halfset_em_iteration` reading state.Ft_y/Ft_CTF after finish_up_M_step.

## Validation and hardware

- Use pixi with clean Python environment and checkout/JAX import provenance.
  Use focused checks per edit and one review/publication per cohesive package.
  Read the [validation ladder](../../docs/development/em_parity_runbook.md#validation-ladder)
  before selecting tests; the full fast parity tier runs at most once per 3–4
  hours unless its path changed, it is being fixed, or final validation is due.
- EM-only work must not run repo-wide full/long suites or SPA/ET table extraction.
  Shared-code changes require their applicable checks within authorized scope.
  The EM long tier and multi-iteration/contention-sensitive GPU work are Slurm-only.
- **Leave physical local GPU0 free.** Immediately check nvidia-smi; use only idle
  GPUs1–3, at most three across agents. Restrict CUDA_VISIBLE_DEVICES by selected
  idle UUID before any GPU-capable process. In Slurm preserve assigned visibility.
- Before jobs read [environment and scratch procedures](../../docs/development/em_parity_runbook.md#environment-gpu-and-scratch).
  Set PYTHONNOUSERSITE=1, XLA_PYTHON_CLIENT_PREALLOCATE=false; unset contaminating
  Python/conda variables and use per-job runtime roots. Keep long-lived sources
  under CRYOEM/gilleslab/mg6942/em_dev; disposable outputs under
  CRYOEM/gilleslab/em_work/codex with SAFE_TO_DELETE. Preserve curated fixtures.
- Before RELION comparisons, captures or builds read the
  [oracle rules](../../docs/development/em_parity_runbook.md#relion-oracle-rules).
  The oracle is RELION 5.0.1 throughout; that section pins the source commit,
  the dump build and the reference binaries. Any comparison against it must use
  `--relion-particle-shuffle mt19937`, because the CLI default reproduces the
  older half-set ordering and silently breaks per-particle correspondence.
  Coordinate the shared RELION source/build; never rebuild pinned binaries or
  create another clone. Pin source, patched build, command, metadata, seed,
  subset/MPI layout and hardware. Restarted per-half captures fail closed unless
  the loaded noise is proved to match the target subset shellwise.
- Before quality/performance claims read
  [benchmark requirements](../../docs/development/em_parity_runbook.md#benchmark-design-and-reporting)
  and quantitative gates in `docs/math/em_parity_program.md`. Completion requires
  production-float32 K1 and exactly K4, each >=100k particles and >=256x256,
  matched inputs/seeds/maps/masks and same-GPU-class RECOVAR/RELION pairs. Close
  synthetic K1 trajectory first, then a characterized real-particle confirmation,
  then K4. Small, historical or double-only results cannot satisfy completion.

## Standing metric artifacts — read before running anything, update after

This programme records metrics in tracked artifacts, not only in run reports. A
result that is not written into one of these is not tracked, and on 2026-09-21
that is exactly how a fortnight of work came to be measured on a single 10k
subset while the real-data gate went unread. Find the artifact your change
touches, read its current value first, and write the new value back.

| Artifact | What it holds | Regenerate / check |
| --- | --- | --- |
| `docs/math/em_k1_realdata_science_equivalence_scorecard_v1.json` + `.md` | The EMPIAR real-data gate: per-dataset RECOVAR vs RELION resolution, curve RMSE, half and cross-engine FSC-AUC, with thresholds and pass/fail | `python scripts/summarize_em_k1_realdata_science_equivalence.py --check-markdown` (read-only, CPU, seconds) |
| [archived real-data evidence inventory](https://github.com/ma-gilles/recovar-experiments/blob/8b62d4e1389cb7c106de2436ac38df6e0b7ca172/snapshots/em_development_records_20260921/source/docs/benchmarks/em/diagnostics/k1-realdata-evidence-inventory-20260903.json) | The read-only audit behind it: per dataset the particle count, box, commit, Slurm jobs, sha256-pinned maps and the exact reproduction commands | read it; do not edit |
| `docs/math/em_relion_parity_scorecard_v1.json`, `em_k4_class_fsc_auc_scorecard_v1.json`, `vdam_relion_parity_scorecard_v1.json` | The synthetic fixed suites: K=1 34-case, K=4 per-iteration per-class, VDAM 12-case | `python scripts/summarize_*_scorecard.py --check docs/math/<file>.md` |
| all of the above at once | The consolidated panel a PR body carries | `python scripts/report_em_parity_progress.py --format markdown` |
| `tests/baselines/em_parity_completion_*.json` | The pinned completion references for K=1 100k/256, K=1 5k/128 and K=4 100k/256 | compared by the completion tests; never edit without an explicit user decision |
| `em_parity_quality_{fast,long}_ledger_*.json`, written beside each case's outputs | What the parity tiers actually measured in one run | `python scripts/extract_em_parity_tables.py --ledger-root <run> --tier fast\|long` |

The last row is the loop that closes: run a tier, collect its ledgers, extract
the table, put the table in the PR body. `scripts/extract_em_parity_tables.py`
exists for EM-scoped PRs precisely so that `scripts/extract_regression_tables.py`,
which serves the SPA/ET pipeline suite, is not used instead.

Two traps that hid this system for two weeks, both real, both worth checking
before concluding a tier is broken:

- **The fast tier needs the native RELION binding**, and it is not built in
  every checkout. Without it three of the seven cases fail at import and never
  reach a comparison. Point `RECOVAR_RELION_BIND_BUILD_DIR` at a built
  `_relion_bind_core*.so` before deciding the tier is red for numerical reasons.
  Ad-hoc harnesses supply their own, which is why they run when the tier does not.
- **Three K-class cases need a dispatch schedule captured from the same oracle
  run** and refuse without `--relion-dispatch-schedule`. That refusal is a
  missing fixture, not a parity failure.

End-to-end runs are the point of the real-data rows, and short tests do not
substitute for them. A completion or real-data claim needs the full run, its
ledger, and the artifact above updated, so the next session can see the number
move.

## Delivery

Keep current conclusion, evidence links, ownership and next action in EM status;
keep detailed histories in artifacts/linked notes. Update best_metrics only for
completion attempts. Each package has one concise reproducible receipt: source
and dirty identity, commands/environment, tests/job IDs, artifacts, outcomes and
open gates, git status and diff stat. No completion while required gates/jobs
remain open. Run `python scripts/check_agent_guides.py` after guide edits.
This file and CLAUDE.md must remain byte-for-byte identical.

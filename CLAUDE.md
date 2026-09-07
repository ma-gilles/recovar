# RECOVAR development contract

RECOVAR estimates conformational heterogeneity from cryo-EM and cryo-ET data.
Engineering priorities are correctness, GPU performance, then clarity.

## Start and resume

1. Establish the task, checkout, branch and current evidence before editing.
   User instructions and authorization persist across turns. Do not ask again
   for work already covered by the task.
2. Read the applicable scoped guides below. Historical experiment notes are
   evidence for their recorded source; their old next actions are not current
   instructions. Use the [codebase map](docs/development/codebase.md) to locate
   the workflow entry point and the modules that own its state and kernels.
3. Print `git rev-parse HEAD`, `git status --short --branch`,
   `git diff HEAD --stat`, and `git diff HEAD | sha256sum` before validation.
   Record untracked files used by a run. A worktree name is not provenance.
4. Choose one concrete change and its smallest useful check. Preserve unrelated
   work. Keep control checkouts and queued/running candidates immutable.

| Affected area | Read before working |
| --- | --- |
| Python source and numerical conventions | [recovar/CLAUDE.md](recovar/CLAUDE.md) |
| Tests, tolerances and baselines | [tests/CLAUDE.md](tests/CLAUDE.md) |
| EM and RELION parity | [recovar/em/AGENTS.md](recovar/em/AGENTS.md) |
| PPCA refinement | [recovar/em/ppca_refinement/AGENTS.md](recovar/em/ppca_refinement/AGENTS.md) |
| CUDA and FFI | [recovar/cuda/CLAUDE.md](recovar/cuda/CLAUDE.md) |
| GUI | [recovar/gui_v2/CLAUDE.md](recovar/gui_v2/CLAUDE.md) |
| Documentation | [docs/CLAUDE.md](docs/CLAUDE.md) |

## Implement and review

- Prefer small functions with explicit inputs, units, layouts and ownership.
  Use simple containers when they clarify state; avoid forwarding layers.
- Separate correctness repairs, performance changes and structural cleanup.
  Preserve numerical casts, reduction order, JIT boundaries, memory lifetime,
  serialized formats and scientific defaults during cleanup. Preserve non-EM
  public APIs. EM APIs may change when this simplifies the implementation;
  migrate affected callers, tests and documentation in the same change.
- Remove private dead code only after checking callers, dynamic registration,
  CLI entry points, tests, notebooks and serialized/imported names. Keep
  independent numerical references independent of production code.
- Validate assumptions early. Do not hide errors with fallback results, skipped
  checks or fabricated success. Resolve TODOs with evidence before removing them.
- Keep math documentation linked to implementing functions, and docstrings
  linked back to the documented formulation. Update both when behavior changes.
- Never widen a scientific tolerance or change `tests/baselines/` without an
  explicit user instruction. Missing measurements are not passing comparisons.
- Keep diffs focused. Do not reformat unrelated code or commit large datasets,
  checkpoints, binaries, generated run outputs or credentials.

## Environment and validation

Use the checkout's frozen pixi environment. Before Python imports, select CPU
or assigned GPU visibility and remove Python/conda contamination. Verify
RECOVAR imports from this checkout and JAX from its `.pixi/envs/default`.
Explicitly build and identify custom CUDA libraries before GPU qualification;
the current runtime loader can build missing libraries automatically.

Follow [CONTRIBUTING.md](CONTRIBUTING.md) for exact setup, validation and PR
requirements, [Della development](docs/development/della.md) for cluster resources
and paper-data paths, and [benchmark contracts](docs/development/benchmarks.md)
for reusable accuracy and performance evidence. Use Slurm for integration,
multi-iteration, long or contention-sensitive GPU work. Reserve local GPUs for
short checks following the user's device policy.

For EM-only work, use the scoped EM validation ladder. Shared pipeline or
repository-wide cleanup requires the applicable SPA/ET and downstream checks
as well. Existing authorization for that scope covers its necessary validation;
a genuinely new scientific objective requires a separate decision.

## Branches and delivery

Work on a feature branch (`codex/<task>` for Codex), never directly on `dev`.
Target `dev`, not the old public `main`. Preserve an explicitly pinned control.
Rebasing an implementation creates a new candidate that needs fresh validation.
Never force-push unless explicitly asked. Before pushing or opening a PR, follow
all applicable checks and table requirements in CONTRIBUTING.md and scoped guides.

Report the change, its reason, exact checks and job IDs, outcomes and unresolved
limitations, reproduction commands, artifact paths, `git status --short --branch`
and `git diff HEAD --stat`. Distinguish executed, quality-accepted and
performance-qualified results. Do not claim completion with required jobs pending.

This file and root `CLAUDE.md` must remain byte-for-byte identical. The same rule
applies to the EM AGENTS/CLAUDE pair. Check both with `cmp` after editing.

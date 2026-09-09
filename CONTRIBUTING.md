# Contributing to RECOVAR

Make one reviewable change at a time. Establish an unchanged control before
structural or numerical work, and keep baseline failures visible.

## Package map

`core/` provides Fourier, CTF, geometry and forward/adjoint primitives;
`data_io/` loads and indexes particles; `reconstruction/` estimates the mean
and noise; `heterogeneity/` estimates covariance, PCA coordinates and volumes;
`em/` contains refinement and classification; `output/` serializes and analyzes
results; `commands/` orchestrates CLI workflows; `cuda/` supplies optional native
kernels; `simulation/` generates fixtures; `gui_v2/` provides the web interface.

The covariance pipeline runs dataset loading → mean/noise reconstruction →
covariance → PCA → embedding → kernel regression → output. Consult the scoped
source guide for FFT frames, packed layouts and normalization before changing
those boundaries. PPCA latent dimension and classification K are different axes.
The [contributor codebase map](docs/development/codebase.md) identifies the
separate pipeline PPCA, pose-refinement, K-class, VDAM and earlier EM entry
points, along with their state, kernel and diagnostic dependencies.

## Reproducible environment

From the intended checkout, use pixi and the committed lockfile:

```bash
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
pixi install --frozen
pixi run install-recovar
pixi run python - <<'PYCODE'
from pathlib import Path
import recovar, jax
repo = Path.cwd().resolve()
assert Path(recovar.__file__).resolve().is_relative_to(repo)
assert Path(jax.__file__).resolve().is_relative_to(repo / '.pixi/envs/default')
print(recovar.__file__, jax.__file__, jax.__version__, jax.devices())
PYCODE
```

The empty GPU visibility above is for CPU setup. For GPU execution, start a
separate process with the assigned visibility and `JAX_PLATFORMS=cuda,cpu` before
imports. Never reuse a process that already initialized the wrong backend.
See [Della](docs/development/della.md) for physical-device and Slurm rules.

The optional fast-marching extension builds during installation when a compiler
is available. Custom CUDA requires a compatible local nvcc and this environment's
JAX FFI headers. Build explicitly into an exclusive run directory:

```bash
# Set RUN_ROOT to a new writable run directory before this command.
PIXI_PY="$(pixi run which python)"
PYTHON="$PIXI_PY" make -C recovar/cuda LIB="$RUN_ROOT/libcuda_backproject.so" all
export RECOVAR_CUDA_LIB="$RUN_ROOT/libcuda_backproject.so"
sha256sum "$RECOVAR_CUDA_LIB"
```

Record source, lock, compiler, headers, loaded library path and library hash.
Verify the library hash before loading, immediately after loading, and after
each paired run. The loader can rebuild a missing or stale library, including
an explicit `RECOVAR_CUDA_LIB` path. For qualification, place a freshly copied
binary in a dedicated directory and make both the file and directory read-only
before loading. Record the actual loaded path; successful import alone does not
prove that the intended extension ran. Never rebuild a shared library while
another process uses it. End-user pip installation is described in
[installation](docs/getting-started/installation.md).

## Validation during development

Read [test rules](tests/CLAUDE.md) before changing or selecting tests. Run the
smallest meaningful check first; advance after it passes. Use explicit backend
placement before pytest collection, which can initialize JAX. Give each run its
own `RECOVAR_JAX_CACHE_DIR` and `JAX_COMPILATION_CACHE_DIR`; RECOVAR can otherwise
reuse a shared cache despite `XDG_CACHE_HOME`. Record whether these caches began
empty or were warmed by a specified command.

| Scope | Starting check | Further qualification |
| --- | --- | --- |
| Pure helpers and reporting | `pixi run python -m pytest -v tests/unit/<affected_test>.py` on CPU | Real missing/invalid/duplicate input cases; affected callers |
| Dense/local EM | `pixi run test-em-fast-guard` | [EM ladder](recovar/em/AGENTS.md), including GPU and K1/K4 gates |
| Shared pipeline | Affected unit/integration tests | SPA, cryo-ET, outlier and downstream quality/performance under Slurm |
| GUI or docs | Applicable scoped checks | Build and relevant user workflow checks |

Use focused tests between edits. Group related changes into a frozen checkpoint
for broader CPU and applicable GPU checks; full long suites are publication or
milestone checks, not the default response to a small change. Reuse saved outputs
for report-only audits. Repeat a scientific run when the source, workload or an
unresolved failure requires it, and record that reason. The
[cleanup plan](docs/development/cleanup_plan.md) tracks the current boundaries
and qualification gaps.

`pixi run test-fast` selects the repository's unit tier; do not assume every
unit test is tiny or independent of external fixtures. Long and GPU tests run
under Slurm. Record selected versus executed counts, skips and process exit
status. Preserve full logs; a truncated console tail is not a result archive.

Some legacy EM tests write ledgers beside baselines, and performance helpers
can auto-save hardware entries. Isolate their result-writing paths before
qualification. Do not overwrite established baselines as a side effect of a
benchmark. An optional local fixture skip is not accepted qualification.

## Documentation environment

Documentation builds use a separate locked environment with no scientific or
GPU dependencies. The default environment remains unchanged.

```bash
pixi install -e docs --locked
pixi run -e docs docs-build
```

For a local preview, run `pixi run -e docs mkdocs serve`. API references are
collected statically from source; building docs must not require importing JAX
or native extensions. Keep user-site packages disabled when invoking Python
directly, and record the docs lockfile identity with build evidence.

## Before pushing or creating a PR

For shared/non-EM changes, including this codebase cleanup:

1. Fetch and rebase the implementation onto `origin/dev`. Keep the pinned
   control unchanged; record the rebased candidate separately.
2. Submit `./scripts/run_tests_parallel.sh long-test` and wait for its summary.
   All required unit, smoke, SPA, ET, outlier, downstream, trajectory, indices,
   stress and isolated-function checks must pass. Fix failures and rerun the
   affected qualification; do not push a failing candidate.
3. Run `pixi run python scripts/extract_regression_tables.py` on the completed
   results and put the quality and performance comparison tables in the PR.
   Include SPA and ET, hardware, baseline/current values and signed percent
   changes. Use ↑/↓ for increase/decrease and mark regressions over 10% as
   **REGRESSED**. Missing or incompatible hardware measurements are not “OK.”
4. Include exact source identities, test commands, Slurm IDs and linked logs.

EM-only changes follow [the EM contract](recovar/em/AGENTS.md), including its
scoped suites and completion evidence, instead of unrelated SPA/ET suites.
A change spanning both scopes requires both sets of applicable checks when
covered by the task; do not infer repeated permission requirements from scope.

Explicit user authorization may allow publishing a draft checkpoint before
these publication prerequisites are complete. Record that authorization and
all missing or failed checks in the PR; retain the requested integration base
and frozen controls. Draft publication does not waive merge, quality or
performance acceptance gates.

Start a PR description with the problem and resulting behavior, then evidence.
Do not confuse replay agreement with an autonomous trajectory, or per-class
quality with an average. See [benchmark contracts](docs/development/benchmarks.md).

## Code style

Ruff uses the 120-character line limit in `pyproject.toml`. Check the Python
files changed by the task, without formatting unrelated files. For uncommitted
work, pass those files explicitly to `pixi run python -m ruff check` and
`pixi run python -m ruff format --check`. The helper
`scripts/check_changed_python.sh --base REV --head REV` compares committed
revisions; it does not validate unstaged changes. Keep existing pre-commit hooks
and their scoped checks in sync with the pinned development toolchain.

## Instruction maintenance

Keep root AGENTS/CLAUDE and EM AGENTS/CLAUDE mirrors identical. Scoped loader
files, such as PPCA's AGENTS.md, intentionally link to their substantive guide.
Keep durable invariants in guides, a short active state/next check in the program
board, and dated experiments in linked archives. Preserve superseded evidence
with its original source and an explicit historical label.

```bash
cmp AGENTS.md CLAUDE.md
cmp recovar/em/AGENTS.md recovar/em/CLAUDE.md
pixi run python scripts/check_agent_guides.py
```

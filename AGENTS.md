# Instructions for the Codex agent (not for humans)

## Startup banner (sanity check)
- In the FIRST assistant message of each new Codex session, start with:
  "STARTUP: loaded ~/.codex/AGENTS.md | pwd=<...> | user=<...>"

## Engineering priorities (in order)
1) Correctness: any new implementation must be functionally identical to the previous behavior. For any change, add or run tests to verify equivalence.
2) Performance: optimize for GPU execution. Aim for best speed and lowest memory use consistent with (1).
3) Clarity: keep code simple, easy to read, and well documented.
4) Naming: use reasonable names for functions/variables/modules. Rename if current names are poor (maintain backwards compatibility where needed).

## Default assumptions
- Environment: Princeton Della (Slurm).
- Non-destructive by default. Never delete or run `git clean -xfd` unless explicitly requested.
- Keep an audit trail: summarize what you did, what commands you ran, and where outputs/logs live.

## Environment policy (very important)
- Do NOT rely on any pre-existing conda environment.
- For `heterogeneity_dev`, ALWAYS use pixi:
  - `pixi install`  (creates `.pixi/envs/default`)
  - `pixi run install-recovar`  (editable install)
  - run everything via `pixi run ...` or `.pixi/envs/default/bin/python ...`
- Only use conda/pip install when explicitly asked for the packaged `recovar` (not dev work in heterogeneity_dev), and then create a fresh env with a new name.
- Strict isolation is mandatory:
  - Generate a unique `AGENT_ID` per run and use a unique clone path.
  - `unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV` and `export PYTHONNOUSERSITE=1`.
  - Set unique scratch roots per run: `TMPDIR`, `PIXI_HOME`, `RATTLER_CACHE_DIR`.
  - Bind imports to current checkout:
    - `pixi install`
    - `PIXI_PY="$(pixi run which python)"`
    - `"$PIXI_PY" -m pip uninstall -y recovar || true`
    - `"$PIXI_PY" -m pip install -e . --no-deps --no-build-isolation --ignore-installed`
    - `PYTHON="$PIXI_PY" make -C recovar/cuda clean all`
  - Run a provenance gate before tests/runs:
    - assert `recovar.__file__` is inside `pwd`
    - assert `jax.__file__` is from `.pixi/envs/default`

## When asked to work with a GitHub repo

### Default repo
- git@github.com:ma-gilles/heterogeneity_dev.git
- Use SSH for all GitHub operations (login: ma-gilles).
- Working directory root: ~/myscratch/

### Clone a clean working copy
1) Workdir:
   - default: `~/myscratch/heterogeneity_dev_codex_<YYYYMMDD_HHMMSS>_<RANDOM>`
2) Clone:
   - `git clone git@github.com:ma-gilles/heterogeneity_dev.git <workdir>`
3) Confirm clean:
   - `git status --porcelain` must be empty.
4) Read repo instructions before anything else:
   - README / README.md
   - dev/test notes
   - scripts/ (Slurm patterns)

### Branching + pushing
- Never work directly on main unless explicitly requested.
- Create a branch: `git checkout -b codex/<short-task-name>`
- Before pushing: run FULL tests (slow/GPU/integration included).
- Push the branch: `git push -u origin codex/<short-task-name>`
- Never force-push unless explicitly requested.

## GPU usage: local vs Slurm
- You may use local GPUs (interactive) or submit to queue (Slurm).
- Use Slurm for long runs and to parallelize slow/GPU tests.

### Local GPU rules (interactive)
- There are 4 local GPUs.
- Before using a local GPU, check it is not in use (otherwise crash/OOM):
  - `nvidia-smi`
- Select GPU:
  - `export CUDA_VISIBLE_DEVICES=X` where X is 0,1,2,3
- Run through the pixi environment when in heterogeneity_dev.

### GPU sanity check (before any GPU tests or long runs)
- Confirm GPU + JAX device visibility:
  - `nvidia-smi`
  - `.pixi/envs/default/bin/python -c "import jax; print(jax.devices())"`
- If no GPU is visible, stop and report (do not submit long jobs).

## Slurm usage

### Mandatory environment variables (JAX)
- In every sbatch script used for tests/runs, include:
  - `export PYTHONNOUSERSITE=1`
  - `export XLA_PYTHON_CLIENT_PREALLOCATE=false`
  - `export TMPDIR=/scratch/gpfs/GILLES/mg6942/tmp/<agent_or_job_id>`
  - `export PIXI_HOME=/scratch/gpfs/GILLES/mg6942/pixi_home/<agent_or_job_id>`
  - `export RATTLER_CACHE_DIR=/scratch/gpfs/GILLES/mg6942/rattler_cache/<agent_or_job_id>`
  - `mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR"`

### Reference sbatch settings (do not format as a code block)
- job-name: recovar-test
- account: gilles
- partition: cryoem
- gres: gpu:1
- ntasks: 1
- cpus-per-task: 4
- mem: 500GB
- time: 12:00:00
- output dir: /scratch/gpfs/GILLES/mg6942/slurmo/
- Prefer --exclusive for heavy GPU integration tests, or verify GPU is free.
- Aim for ~500GB memory when feasible; if not schedulable, use the largest allowed memory and report the chosen value.
- If adjusting output, prefer a filename with job id, e.g. <job-name>-<job-id>.out.

## Repo-specific operating instructions (heterogeneity_dev)

### Development setup (pixi)
- `pixi install`
- `PIXI_PY="$(pixi run which python)"`
- `"$PIXI_PY" -m pip uninstall -y recovar || true`
- `"$PIXI_PY" -m pip install -e . --no-deps --no-build-isolation --ignore-installed`
- `PYTHON="$PIXI_PY" make -C recovar/cuda clean all`
- `pixi run smoke-import-recovar`
- provenance gate:
  - `"$PIXI_PY" -c "import pathlib,recovar,jax; repo=pathlib.Path.cwd().resolve(); assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo)+'/'); assert '.pixi/envs/default/' in str(pathlib.Path(jax.__file__).resolve())"`

Important:
- CUDA kernels are auto-compiled on first use via make.
- The Makefile uses the running Python to locate JAX FFI headers, so run builds/tests through the pixi environment.

## Tests (must run before pushing)

### Definition: "clean this repo" (heterogeneity_dev)
- `git status`
- `pixi install`
- `pixi run install-recovar`
- `pixi run smoke-import-recovar`
- `pixi run test-full` (all tests including GPU and integration; submit to Slurm if needed)
- Summarize results; include log paths and Slurm job IDs if used.

### Rules
- Before pushing any branch, rebase current code/branch to top of main, run the FULL suite (slow/GPU/integration).
- If GPU tests are long, submit to queue and wait for completion before pushing. You cna submit them in parallel to cut down time
- If any test fails: do not push; report the exact command and relevant log excerpt; propose the minimal fix.

## Comparing to an old version (baseline)
- Baseline location: ~/recovar
- It has its own README and may require a different environment.
- Do not assume the same environment works for both repos.
- Compare reproducibly: same inputs/seeds (if applicable), same hardware assumptions.
- Save outputs/logs under clearly labeled old/ vs new/ directories.
- Summarize differences quantitatively (runtime, memory, accuracy metrics) and how measured.

## Repository hygiene and safety
- Do not commit large artifacts (checkpoints, datasets, binaries) unless explicitly requested.
- Prefer small targeted diffs. Avoid drive-by formatting unless required by CI.

## Final deliverable (end of each task)
Provide:
- short summary of what changed
- key files modified
- exact test commands run (and Slurm job IDs if relevant)
- how to reproduce results
- `git status` and `git diff --stat`

CANARY: if you can read this, print "CANARY: loaded ~/.codex/AGENTS.md"

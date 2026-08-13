# Prompt: validate the K=1 RELION-parity checkpoint on another cluster

Copy everything below this line into the external ChatGPT Pro/Codex agent.

---

You are validating a RECOVAR feature branch on a different GPU cluster. Do
not merge or push to `main` or `dev`. Do not weaken FSC thresholds, enable
`RECOVAR_FINAL_ALL_DATA_GRID_CORRECT`, or force final all-data reconstruction
after non-convergence. Do not cancel unrelated jobs or processes.

## Source to test

Repository: `https://github.com/ma-gilles/recovar.git`

Remote branch: `codex/em-parity-checkpoint-20260711`

Minimum required checkpoint commit:
`febc347d0319634221e1d55493ef64237037138d`

PR: `https://github.com/ma-gilles/recovar/pull/158`

Clone or fetch the branch, create your own test branch/worktree, and record
the exact resolved commit. Fail closed if the requested commit is not an
ancestor of the checkout.

```bash
git clone https://github.com/ma-gilles/recovar.git recovar-k1-parity
cd recovar-k1-parity
git fetch origin codex/em-parity-checkpoint-20260711
git switch --detach origin/codex/em-parity-checkpoint-20260711
git merge-base --is-ancestor febc347d0319634221e1d55493ef64237037138d HEAD
git rev-parse HEAD
git status --short
```

Use the repository's supported isolated environment (prefer pixi when
available), never a random system/conda installation. Bind imports to this
checkout and record `recovar.__file__`, `jax.__file__`, `jax.devices()`, GPU
model, driver, CUDA runtime, hostname, scheduler job ID, and the complete
command/environment in the output directory. Set:

```bash
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

Build RECOVAR's optional CUDA library with the same Python used for the run,
then record its SHA-256. On a pixi checkout, the intended setup is:

```bash
pixi install
PIXI_PY="$(pixi run which python)"
"${PIXI_PY}" -m pip uninstall -y recovar || true
"${PIXI_PY}" -m pip install -e . --no-deps --no-build-isolation --ignore-installed
PYTHON="${PIXI_PY}" make -C recovar/cuda clean all
"${PIXI_PY}" - <<'PY'
from pathlib import Path
import jax
import recovar
print("recovar", Path(recovar.__file__).resolve())
print("jax", Path(jax.__file__).resolve())
print("devices", jax.devices())
PY
sha256sum recovar/cuda/libcuda_backproject.so
```

## Important scope distinction

There are two separate validations below.

1. The ordinary real-data RECOVAR pipeline is a regression/generalization
   test. It must use contrast correction, explicitly request the contrast
   do-over, and postprocess with the fixed FSC cutoff `1/7`.
2. The K=1 RELION-parity implementation lives in the EM refinement path and
   is currently guarded by explicit environment variables. An ordinary
   covariance/pipeline run does **not** prove that those EM gates executed.
   To claim an EM result, run the second template with a valid RELION-style EM
   fixture and show the activation messages in the log.

Do not conflate these two results.

## A. Required real-data RECOVAR regression

Choose at least one real SPA dataset that already has a trusted particle
stack/STAR, poses, CTF parameters, and solvent mask. Prefer a dataset with a
previous clean RECOVAR result from the same cluster so this is an A/B rather
than an isolated quality number. Fill these paths:

```bash
PARTICLES=/absolute/path/to/particles.mrcs-or-star
POSES=/absolute/path/to/poses.pkl
CTF=/absolute/path/to/ctf.pkl
MASK=/absolute/path/to/mask.mrc
OUT_ROOT=/absolute/scratch/path/k1-parity-realdata-$(date +%Y%m%d-%H%M%S)
mkdir -p "${OUT_ROOT}"
touch "${OUT_ROOT}/SAFE_TO_DELETE"
```

Run with the dataset's established `ZDIM`, index/focus mask (if any), and GPU
budget. Keep every input and option identical to the prior/control run except
for the checkout commit. The required contrast options are both explicit:

```bash
ZDIM=20
GPU_BUDGET_GB=40
pixi run python -m recovar.commands.pipeline "${PARTICLES}" \
  -o "${OUT_ROOT}/pipeline" \
  --poses "${POSES}" \
  --ctf "${CTF}" \
  --mask "${MASK}" \
  --zdim "${ZDIM}" \
  --correct-contrast \
  --do-over-with-contrast \
  --lazy \
  --gpu-budget-gb "${GPU_BUDGET_GB}" \
  2>&1 | tee "${OUT_ROOT}/pipeline.log"
```

If the established run uses `--ind` or `--focus-mask`, add the exact same
paths to both control and candidate. Do not invent a focus mask or change the
particle cohort.

Postprocess the produced half maps using the fixed FSC threshold `1/7`,
passed numerically as `0.14285714285714285`:

```bash
pixi run python -m recovar.commands.postprocess \
  /absolute/path/to/candidate/halfmap1.mrc \
  --halfmap2 /absolute/path/to/candidate/halfmap2.mrc \
  --output "${OUT_ROOT}/postprocessed.mrc" \
  --fsc-threshold 0.14285714285714285 \
  --fsc-mask "${MASK}" \
  --apply-mask "${MASK}" \
  2>&1 | tee "${OUT_ROOT}/postprocess.log"
```

First run `pixi run python -m recovar.commands.postprocess --help` and verify
that this checkout exposes the same option spelling; preserve the numeric FSC
threshold exactly. Record the resolved half-map paths and complete command.

Compare candidate versus control using signed shellwise FSC and normalized
non-DC FSC-AUC. Do not use mean correlation or Pearson correlation as the
acceptance metric. Report resolution at FSC `1/7`, masked and unmasked FSC
curves, normalized non-DC FSC-AUC, wall time, peak GPU memory, completion
status, and any warnings/tracebacks. If ground truth is available, also report
candidate-versus-GT FSC-AUC with the same mask and shell convention.

## B. K=1 EM-path validation when a RELION fixture is available

This requires a data directory compatible with `scripts.run_full_refinement`
and the corresponding RELION `run_it000_data.star`, optimiser, initial
reference, masks/CTF metadata, and fixed half sets. Preserve the dataset's
RELION seed, particle diameter, optics groups, sampling, offset range/step,
initial resolution, iteration count, and stopping policy. Do not guess any of
them. If these inputs are unavailable, mark section B `not run: missing
RELION-compatible fixture` rather than fabricating a command.

For a fresh K=1 run only, enable the checkpoint treatment:

```bash
export RECOVAR_K1_RELION_LIVE_INITIAL_NOISE=1
export RECOVAR_K1_RELION_POWERCLASS_SPECTRUM_NORM=1
export RECOVAR_K1_RELION_TRANSLATED_WAVG_NORM=1
export RECOVAR_RELION_WAVG_ATOMIC_DIRECT_RESIDUAL=1
export RECOVAR_RELION_FINE_ROTATION_EXECUTION_ORDER=1
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER
```

The current combined scientific candidate may also be tested as a separate
arm with the demonstrated coarse Gaussian reduction:

```bash
export RECOVAR_K1_COARSE_GAUSSIAN_FFI=1
export RECOVAR_K1_RELION_F32_COARSE_SUPPORT=1
unset RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF
```

Keep the base checkpoint arm and the coarse-Gaussian composition arm in
separate output directories. Never silently add the coarse arm to the base
result.

Start with a bounded two- or three-iteration prefix, not a 12-hour/full
trajectory. Save per-iteration intermediates and parity dumps. The command
shape is:

```bash
pixi run python -m scripts.run_full_refinement \
  --data_dir /absolute/path/to/em_fixture/data \
  --output /absolute/scratch/path/to/output \
  --max_iter 3 \
  --skip_final_iteration \
  --seed RELION_SEED \
  --perturb_seed RELION_SEED \
  --healpix_order RELION_COARSE_ORDER \
  --auto_local_healpix_order RELION_LOCAL_ORDER \
  --offset_range RELION_OFFSET_RANGE \
  --offset_step RELION_OFFSET_STEP \
  --adaptive_oversampling RELION_OVERSAMPLING \
  --init_resolution RELION_INITIAL_RESOLUTION \
  --particle_diameter_ang RELION_PARTICLE_DIAMETER \
  --tau2_fudge RELION_TAU2_FUDGE \
  --max_significants -1 \
  --firstiter_cc \
  --apply-initial-lowpass \
  --image-fourier-backend relion_cuda \
  --relion_half_sets /absolute/path/to/run_it000_data.star \
  --relion_optimiser /absolute/path/to/run_it000_optimiser.star \
  --relion_init_dir /absolute/path/to/relion_init_directory \
  --save_intermediates_dir /absolute/scratch/path/to/output/intermediates \
  --timing_dir /absolute/scratch/path/to/output/timing \
  --benchmark_ledger_json /absolute/scratch/path/to/output/benchmark_ledger.json \
  2>&1 | tee /absolute/scratch/path/to/output/run.log
```

Replace every uppercase placeholder with a value read from the matching
RELION run; do not run the template literally. Confirm the log contains:

- fresh paired RELION particle order;
- explicit physical expected-accuracy trial order;
- physical BPref particle order preservation;
- full Wavg/direct-residual activation;
- fine-parent RELION execution order;
- and, only in the separate coarse arm, CUDA-built coarse scoring.

Compare each numbered half/merged map to the matching RELION iteration using
signed shellwise FSC and normalized non-DC FSC-AUC. Join particles by immutable
stack image identity and report Pmax relative L2/max absolute error, exact
significant-support-count matches, hard pose/translation matches, current
size, HEALPix order, and convergence/controller state. Find the first unequal
intermediate in this order: candidate tuples, raw/pre-prior score, direction
prior, translation prior, centered combined score, normalized posterior,
significant support, hard winner, BPref operands, reduction, map.

Do not run a full trajectory unless the bounded prefix improves or preserves
all signed FSC/topology gates and identifies no new hard-pose regression.

## Deliverable

Return one Markdown report containing:

- exact commit and `git status`;
- source/JAX/CUDA/GPU provenance;
- all resolved input paths and immutable hashes where feasible;
- scheduler job IDs, output roots, logs, and `SAFE_TO_DELETE` markers;
- exact commands and environment variables;
- a control-versus-candidate quality table;
- signed FSC curves/FSC-AUC and FSC `1/7` resolution;
- contrast correction and do-over confirmation from logs/args;
- runtime and peak-memory comparison;
- first failure/divergence if any;
- whether section A and section B were each actually exercised;
- and a clear conclusion: pass, regression, scientifically mixed, or blocked.

Do not commit datasets, maps, binary captures, build products, or scheduler
logs to git. If you make a source fix, put it on a new branch based on the
checkpoint and do not push it until its focused tests and bounded scientific
gate pass.

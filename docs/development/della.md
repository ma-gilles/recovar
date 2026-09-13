# Development on Princeton Della

Use pixi, an isolated feature worktree and unique run roots. Keep long-lived
EM source checkouts under `/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/`;
the GILLES project filesystem is quota-constrained. Curated fixtures stay in
place. Never erase a fixture to make room for disposable outputs.

## GPU placement

There are four GPUs on the user's local development machine. Leave physical
GPU 0 free for other users. Short local checks may use only idle physical GPUs
1, 2 or 3, at most three in total across concurrent work. Check `nvidia-smi`
for existing processes first and set `CUDA_VISIBLE_DEVICES` to the selected
physical GPU UUID before any JAX import or pytest collection. Logical device 0
inside that restricted process refers to the selected physical GPU.

Slurm jobs use the scheduler's assigned visibility; do not replace it with the
local 1/2/3 rule. Use Slurm for GPU, integration, multi-iteration, long or
contention-sensitive work. Queue waits are expected. Compare timing pairs on
the same physical GPU sequentially, and record its model, UUID and driver.

For CPU setup and tests, set both `CUDA_VISIBLE_DEVICES=''` and
`JAX_PLATFORMS=cpu` before imports. GPU placement needs a fresh process with
`JAX_PLATFORMS=cuda,cpu`. Do not let automatic device selection occupy GPU 0 on the
local machine.

## Job environment and artifacts

Every sbatch script must remove Python/conda contamination and use its checkout's
pixi Python. Preserve the scheduler's GPU visibility and create private runtime
roots:

```bash
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/${SLURM_JOB_ID}/tmp
export PIXI_HOME=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/${SLURM_JOB_ID}/rattler_cache
export RECOVAR_JAX_CACHE_DIR=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/${SLURM_JOB_ID}/jax_cache
export JAX_COMPILATION_CACHE_DIR="$RECOVAR_JAX_CACHE_DIR"
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR" "$RECOVAR_JAX_CACHE_DIR"
```

Save bulky disposable runs under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/<dated-run-name>/`, or the other
scratch roots specified below, with a `SAFE_TO_DELETE` marker. Keep small
manifests, commands, checksums, measurements and logs in durable review records.
Freeze the source and input identities from submission until completion.

The shared RELION dump build is
`/scratch/gpfs/GILLES/mg6942/relion/build_patched/`. Do not clone another oracle
or edit/rebuild that shared resource without coordinating with its users.
Record the actual source, patch and binary identity; a directory name does not
identify the oracle. See [EM rules](../../recovar/em/AGENTS.md).

## RECOVAR Paper Dataset Runbook

When the user says "rerun all RECOVAR paper datasets" or asks for a
paper-dataset benchmark, do not ask where the data are or which masks to use.
Use this runbook, then verify each path exists before submitting Slurm jobs.

Canonical metadata/mask root:

```bash
RECOVAR_DATASETS_ROOT=/home/mg6942/mytigress/RECOVAR_datasets
```

This root contains `poses.pkl`, `ctf.pkl`, masks, focus masks, and indices. It
does not contain the particle stacks. Particle stacks are local under
`/tigress/CRYOEM/singerlab/mg6942` or
`/scratch/gpfs/AMITS/mg6942/cryodrgn_empiar`.

Default real paper-dataset inputs and run modes:

| Dataset | Particles | Metadata/masks | Focused/paper run | Solvent survey run |
| --- | --- | --- | --- |
| `10073` | `/tigress/CRYOEM/singerlab/mg6942/10073/recovar_data/particles.256.mrcs` | `$RECOVAR_DATASETS_ROOT/10073/{poses.pkl,ctf.pkl,mask.mrc,focus_mask.mrc,ind.pkl}` | `--zdim 4`, `--mask`, `--focus-mask`, `--ind`, `--correct-contrast` | `--zdim 20`, `--mask`, `--correct-contrast`; no focus mask or index |
| `10076` | `/tigress/CRYOEM/singerlab/mg6942/10076/particles.256.mrcs` | `$RECOVAR_DATASETS_ROOT/10076/{poses.pkl,ctf.pkl,mask.mrc}` | `--zdim 20`, `--mask`, `--correct-contrast`; no canonical focus mask or index | `--zdim 20`, `--mask`, `--correct-contrast`; no focus mask or index |
| `10180` | `/scratch/gpfs/AMITS/mg6942/cryodrgn_empiar/empiar10180/inputs/particles.256.mrcs` | `$RECOVAR_DATASETS_ROOT/10180/{poses.pkl,ctf.pkl,mask.mrc,focus_mask.mrc,filtered.ind.pkl}` | `--zdim 4`, `--mask`, `--focus-mask`, `--ind filtered.ind.pkl`, `--correct-contrast` | `--zdim 20`, `--mask`, `--correct-contrast`; no focus mask or index |
| `10345` | `/tigress/CRYOEM/singerlab/mg6942/10345/recovar_data/particles.256.mrcs` | `$RECOVAR_DATASETS_ROOT/10345/{poses.pkl,ctf.pkl,mask.mrc,focus_mask.mrc,ind.pkl}` | `--zdim 4`, `--mask`, `--focus-mask`, `--ind`, `--correct-contrast` | `--zdim 20`, `--mask`, `--correct-contrast`; no focus mask or index |

The standard workflow is: first run a zdim-20 solvent-mask-only survey on every
dataset, then use curated indices and focus masks for the focused rerun. In the
canonical focused rerun, 10076 remains zdim 20 with only the solvent mask
because this root has no canonical focus mask or curated index for it; the other
paper datasets use zdim 4 with focus mask and index.

For 10076 focus-mask ablations only, older benchmark scripts used
`/scratch/gpfs/AMITS/mg6942/cryodrgn_empiar/empiar10076/inputs/recovar_masks/mask_10076.mrc`.
Do not use that as the default paper input unless the user explicitly asks for
the focus-mask ablation.

Default command shape for a plain PPCA paper-data rerun:

```bash
pixi run python -m recovar.commands.pipeline "$PARTICLES" \
  -o "$OUT_DIR" \
  --poses "$POSES" --ctf "$CTF" \
  --mask "$MASK" \
  ${FOCUS_MASK:+--focus-mask "$FOCUS_MASK"} \
  ${IND:+--ind "$IND"} \
  --correct-contrast \
  --lazy --gpu-budget-gb 40 \
  --zdim "$ZDIM" \
  --use-ppca --ppca-zdim "$ZDIM" --ppca-em-iters 20
```

For easy apple-to-apple comparison with the current covariance approach, submit
both `covariance` and `ppca` by default for every dataset/run mode. Keep
particles, poses, CTF, mask, focus mask, index, zdim, contrast flag, GPU budget,
and analysis settings identical. The only pipeline difference should be that
PPCA adds `--use-ppca --ppca-zdim "$ZDIM" --ppca-em-iters 20`. For projected
covariance, add `--ppca-projected-covariance`, but plain PPCA is the current
first-line PPCA benchmark unless the user asks otherwise.

Run `recovar analyze` after pipeline completion with the same `ZDIM` used for
the pipeline: `--zdim "$ZDIM" --n-clusters 20 --n-trajectories 0
--skip-centers` for fast benchmark UMAP checks, or `--n-trajectories 2` for
full paper-style trajectory outputs.

For every focused zdim-4 run, also run conformational-density estimation after
`analyze`:

```bash
pixi run python -m recovar.commands.estimate_conformational_density "$OUT_DIR" \
  --pca_dim 4 --z_dim_used 4
```

This applies to the focused `10073`, `10180`, and `10345` runs. It does not
apply to the zdim-20 solvent surveys or to canonical `10076`, which is zdim 20.

Relevant local orchestration starting points:

- `scripts/recovar_paper_dataset_run.sbatch`
- `scripts/submit_recovar_paper_datasets.sh`
- `docs/guide/tutorial.md`
- older PPCA worktrees may also have `scripts/ppca_default_benchmark.py`,
  `scripts/realdata_cov_ppca_focus_compare.sbatch`, and
  `scripts/submit_realdata_cov_ppca_compare.sh`; if present, keep their inputs
  consistent with this runbook

PPCA methods require a checkout whose pipeline help includes `--use-ppca` (for
example the active PPCA worktree/branch). If the current checkout lacks that
flag, switch to a PPCA-capable checkout before submitting PPCA jobs. The paper
dataset runner fails fast when `METHOD != covariance` and `--use-ppca` is absent.

Before submitting, record the repo commit, exact command line, resolved input
paths, Slurm job IDs, output root, and key logs. Save bulky disposable outputs
under `/scratch/gpfs/GILLES/mg6942/` or overflow
`/scratch/gpfs/CRYOEM/gilleslab/` with a `SAFE_TO_DELETE` marker.


For PPCA comparisons, use at least 20 EM iterations. Shorter runs are smoke
checks, not benchmark-quality results. Use 30 for contrast, FSC-cutoff or
projected-covariance diagnostics when convergence is unclear. Do not interpret
an unconverged short run as a subspace or embedding-quality regression.

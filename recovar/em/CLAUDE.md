# EM Module Agent Guide

This is the compact operating guide for agents working in `recovar/em`.
Longer parity history and detailed run notes live in
`docs/math/relion_parity_agent_notes.md`.

## Startup

- `recovar/em/AGENTS.md` is a loader only; keep durable EM guidance here.
- Use the current checked-out task branch. Do not switch to historical branch
  names unless the user explicitly asks for a baseline comparison.
- Before implementation work, read the relevant code/tests plus:
  - `docs/math/relion_parity_current_status_2026_04_25.md`
  - `docs/math/relion_parity_roadmap_2026_04_27.md`
  - `docs/math/relion_parity_agent_notes.md` for detailed parity notes
- At task start, classify the work as docs-only, unit-test repair,
  algorithmic parity, performance-only, or PR preparation. Let that determine
  validation scope.

## Non-Negotiables

- Goal: perfect quality parity with RELION and near RELION speed parity for
  dense single-volume EM refinement.
- Do not solve parity by parameter tuning. Identify the RELION source behavior,
  metadata value, or dump-level mismatch first, then encode the same behavior
  in RECOVAR with a targeted test.
- Keep algorithmic parity changes separate from performance changes. Batching,
  caps, memory layout, and scheduling changes are performance-only until output
  equivalence is proven against the old path.
- Never widen test tolerances or edit baselines without explicit approval.
- Do not modify `heterogeneity.py`; it has separate owners.
- Preserve public/shared state contracts: `split_E_M_v2` reads `state.Ft_y`
  and `state.Ft_CTF` after `finish_up_M_step`.

## Numeric Contract

- For RELION accelerated GPU parity, `~1e-4` score/Pmax accuracy is usually
  arithmetic-level parity when pose agreement is exact.
- Escalate gaps at `1e-3` or larger, pose flips, or systematic multi-iteration
  drift.
- If unsure, rerun RECOVAR with float64 scoring (`JAX_ENABLE_X64=1` and the
  current float64 replay/refine option). When needed, obtain a RELION CPU,
  double, or `ACC_DOUBLE_PRECISION` dump for the same particle/candidate set.
- Do not chase bitwise equality against RELION GPU texture arithmetic. Do
  chase reproducible gaps beyond the `~1e-4` band with source-level and
  dump-level evidence.

## Evidence Policy

For deep parity work, dump enough state to identify first divergence:

- raw E-step scores and posterior probabilities
- every attempted pose in full-grid pass 1, adaptive/oversampled pass 2, and
  local search
- best poses after each pass, priors, masks, noise accumulators, `Ft_y`,
  `Ft_ctf`, maps, FSC, tau2, data-vs-prior, current size, and resolution state

For parity claims, report fixture, particle count, box size, commit/branch,
hardware, exact command, output directory, Slurm job ID if applicable, and
quantitative deltas.

## Environment

Use pixi, not a pre-existing conda environment. Run tests and scripts through
`pixi run ...` or `.pixi/envs/default/bin/python`.

For GPU tests, bind imports and rebuild CUDA FFI first:

```bash
pixi run install-recovar
PIXI_PY="$(pixi run which python)"
PYTHON="$PIXI_PY" make -C recovar/cuda clean all
```

Then run a provenance gate that checks:

- `recovar.__file__` is inside the checkout
- JAX imports from `.pixi/envs/default`
- `jax.devices()` sees the selected GPU
- `recovar.cuda_backproject.cuda_available()` is true

Set `RECOVAR_DISABLE_CUDA=1` only when intentionally testing the CPU/JAX
fallback.

## Testing

For normal EM iteration, use targeted EM tests and focused replay/dump jobs.
Do not run the full RECOVAR-wide suite unless the user asks, the change crosses
out of EM/refinement, or you are preparing a PR.

Default quick guard:

```bash
pixi run test-em-fast-guard
```

Focused tests commonly needed after dense/local helper refactors:

- `tests/unit/test_refine_relion_mode.py::test_tracked_local_engine_todo_ids_are_present`
- `tests/unit/test_refine_relion_mode.py::test_run_local_em_exact_windowed_path_computes_reconstruction_abs2_without_full_buffer`
- `tests/unit/test_sparse_pass2_bucketed_perf.py::test_bucketed_call_count_bounded_versus_perimage`
- touched dense profile/return-shape tests in `tests/unit/test_half_spectrum_em.py`

Use local GPUs only for short checks after `nvidia-smi` confirms an idle
device. Use Slurm for long, slow, GPU, integration, or PR-gating jobs. Keep
scratch roots under `/scratch/gpfs/GILLES/mg6942/`.

## RELION Conventions

RECOVAR and RELION use different real-space volume frames:

```python
vol_recovar = -np.transpose(vol_relion, (2, 1, 0))
```

Use `recovar.utils.helpers.load_relion_volume(path)` for RELION-produced MRCs
before FSC/comparison against RECOVAR output. Use `load_mrc` / `write_mrc` for
RECOVAR, cryoSPARC, and cryoDRGN-frame MRCs. Do not "fix" `R_to_relion` or
`R_from_relion`; they are intentionally paired with the volume convention.
`tests/unit/test_relion_volume_convention.py` pins this.

For RELION CLI parity, do not trust `relion_refine --help` defaults. Include
the GUI-equivalent flags from `pipeline_jobs.cpp::initialiseAutorefineJob()`:

```bash
--ctf --firstiter_cc --flatten_solvent --zero_mask \
--low_resol_join_halves 40 --norm --scale
```

RELION iter-1 `ave_Pmax = 1.0` with `--firstiter_cc` is a hard winner-take-all
cross-correlation artifact, not Bayesian inference. Do not add that path to
RECOVAR merely to match the iter-1 number.

## Active EM Pointers

- Dense homogeneous RELION-parity work lives under `dense_single_volume/`.
- New dense/refine work should target `dense_single_volume/em_engine.py` and
  helpers unless a shared utility really belongs elsewhere.
- `EMState` delegates to `dense_single_volume/` for the homogeneous path.
  `SGDState` and `HeterogeneousEMState` still call older functions directly.
- The old `core.py` / `m_step.py` path is still shared; change it only with
  targeted equivalence tests.

## Active Cleanup Ledger

Do not delete these TODOs or cleanup notes unless the implementation actually
removes the debt and the summary names the replacement.

- Apply the same cleanup pattern to `local_em_engine.py` / `local_big_jit.py`
  after the dense path is cleaned up; most dense EM readability TODOs have a
  local-engine analogue.
- Next local-engine cleanup should split `run_local_em_exact` into named
  bucket helpers with a small result/context object. Keep big-JIT and
  split/debug paths behaviorally identical while removing duplicated
  correction, projection, M-step, and noise setup blocks.
- Implement K-class EM for both dense and local engines after the single-class
  TODO cleanup is stable, with near RELION K-class end-to-end parity as the
  acceptance target.

## Useful Artifacts

- Current status: `docs/math/relion_parity_current_status_2026_04_25.md`
- Roadmap and issue map: `docs/math/relion_parity_roadmap_2026_04_27.md`
- Detailed agent notes: `docs/math/relion_parity_agent_notes.md`
- Algorithm note: `docs/math/relion_updateSSNR_algorithm_2026_04_25.md`
- Tiny parity fixture: `/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_tiny_parity/`
- RELION checkout: `/scratch/gpfs/GILLES/mg6942/relion`
- Patched RELION build: `/scratch/gpfs/GILLES/mg6942/relion/build_patched`

## Delivery

End every task with:

- what changed and key files modified
- exact test commands run, with Slurm job IDs/logs if applicable
- how to reproduce results
- `git status`
- `git diff --stat`

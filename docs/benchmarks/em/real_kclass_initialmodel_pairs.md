# Real-data K-class InitialModel pairs

This runbook launches a matched RELION/RECOVAR K-class InitialModel diagnostic
on one Slurm GPU. It is the bounded first real-data K>1 gate for the EM pull
request: every numbered iteration is compared after Hungarian class matching,
and the report preserves shellwise cross-engine FSC, FSC-AUC, hard-assignment
agreement, full class populations, class-collapse status, input hashes, source
trees, commands, environment, and separate per-engine wall/HBM/RSS.

This is not a gold-standard refinement benchmark. InitialModel emits one map
per class rather than independently refined half maps. A passing result means
that RECOVAR follows RELION's InitialModel trajectory and retains the same
class populations on the selected particles. It does not establish final
resolution, biological class validity, or seed-to-seed stability. Those claims
still require matched Class3D/refinement runs with shared initial maps, frozen
halves, a common mask, and multiple seeds.

## Frozen fixtures

The launcher accepts only the following existing 10,000-particle fixtures.
It validates the STAR and selected-index file before writing a job, and the
pair runner hashes the STAR and every referenced stack before either engine
runs.

| Dataset | Fixture | `particles.star` SHA-256 | `source_indices.npy` file SHA-256 |
| --- | --- | --- | --- |
| EMPIAR-10345 | `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_real10345_10k_fixture_v1_20260823/data` | `e5d9f77ff38d0e5137412892e7cc7591ba09265fb928b649cdeab58208a540f5` | `9f812a7bfd6bb9dd071786143a501c6803f6c05541faee36c7d6e07f0aa787a3` |
| EMPIAR-10076 | `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_10k_fixture_20260712/data` | `2560afeea6839dddbb38b47d26cdf8944535a799d1e6d3e1441535c96043998f` | `b58a6d11fb292a9ed9573ac75c6a0673f4a0e8c216f4dadf11a7f0537b0e2c9d` |

The fixture manifests are retained and hashed as lineage evidence. The 10345
fixture contains a balanced frozen half-set label, but InitialModel does not
consume that split as two independent refinements.

## Frozen default contract

The default pilot is K=4, C1, eight iterations, seed 0, `tau2_fudge=4`,
HEALPix order 1, oversampling 1, offset range/step 6/2 pixels, padding factor
1, 500-image batches, the exact `relion_cuda` image Fourier backend, and
5,000-rotation blocks. It audits iterations 1--8,
the numbered artifacts emitted after each completed native InitialModel update.
Iteration 0 remains available only as an explicit checkpoint for legacy or
frozen oracles that actually retain matching iteration-0 artifacts.
Every iteration requires minimum matched cross-engine FSC-AUC 0.999,
permutation-aware assignment agreement 0.995 wherever assignments exist,
identical particle identity and assigned/unassigned sets, and no class below
1% in either engine once assignments exist. Iteration 0 is normally unassigned
in both engines and is recorded as non-evaluable rather than as four collapsed
classes. Collapse is a separate scientific failure even when the two engines
reproduce the same collapse.

Fresh mode runs RELION followed by RECOVAR as separate processes on the same
physical GPU. This is the only mode that reports a RECOVAR/RELION runtime
ratio. Frozen mode reuses a validated immutable RELION directory and runs only
RECOVAR; it reports no runtime ratio and marks legacy missing HBM/RSS/source
fields explicitly.

## EMPIAR-10345 fresh pair

Run from a clean checkout containing these scripts. This first invocation is a
dry run: it validates the fixture/source/binaries and writes a sealed sbatch
script, but submits nothing.

```bash
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
PIXI_PY=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_pr158_k4_matrix_expanded_axes_20260901/.pixi/envs/default/bin/python
RUN_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10345_initialmodel_$(git rev-parse --short=10 HEAD)_20260901
"${PIXI_PY}" -m scripts.launch_em_real_kclass_initialmodel_slurm \
  --dataset 10345 \
  --output-root "${RUN_ROOT}" \
  --relion-refine /scratch/gpfs/GILLES/mg6942/relion_clean_f2c1a384/build_clean_pinned/bin/relion_refine \
  --relion-source-dir /scratch/gpfs/GILLES/mg6942/relion_clean_f2c1a384/src \
  --pixi-python "${PIXI_PY}"
```

Inspect `${RUN_ROOT}/submission_manifest.json` and
`${RUN_ROOT}/scripts/run_pair.sbatch`. Submit the already sealed script with
the exact command printed by the launcher:

```bash
sbatch --parsable "${RUN_ROOT}/scripts/run_pair.sbatch"
```

Alternatively, add `--submit` to the launcher command to validate, write, and
submit in one invocation. The generated job requests exactly one H100, one
task, eight CPUs, and 192 GB without `--exclusive`; it fails if Slurm's
requested and allocated TRES differ or if `OverSubscribe` is not `OK`.

## EMPIAR-10076 and frozen replay

Use the same command with `--dataset 10076` and a distinct output root. A
legacy frozen RELION oracle is available at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k4_real10076_10k_8bb910f3/pair/pair_report.json`:

```bash
"${PIXI_PY}" -m scripts.launch_em_real_kclass_initialmodel_slurm \
  --dataset 10076 \
  --output-root /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_initialmodel_frozen_$(git rev-parse --short=10 HEAD)_20260901 \
  --reference-pair-report /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k4_real10076_10k_8bb910f3/pair/pair_report.json \
  --relion-refine /scratch/gpfs/GILLES/mg6942/relion_clean_f2c1a384/build_clean_pinned/bin/relion_refine \
  --relion-source-dir /scratch/gpfs/GILLES/mg6942/relion_clean_f2c1a384/src \
  --pixi-python "${PIXI_PY}"
```

The legacy run at commit `8bb910f38c16c2d10f96015c233d668115ac4745`
passed its old trajectory gate: minimum matched FSC-AUC was
0.9999987694 and minimum assignment agreement was 0.9988095. RELION took
71.65 s and RECOVAR took 705.67 s on job 12796578. Both engines also converged
to a class distribution near `[0.4104, 0.0010, 0.3983, 0.1903]`; therefore the
second class collapsed under the current 1% policy. This legacy result proves
cross-engine reproduction of that collapse, not a scientifically stable K=4
partition. It predates separate HBM/RSS capture and does not record a
cryptographic RELION binary-to-source build attestation, so it is not a
registry-ready performance result.

## Outputs and admission

Every disposable outer run root and pair root contains `SAFE_TO_DELETE`.
The principal outputs under `<run-root>/pair/` are:

- `pair_report.json`: complete pair provenance, commands, resources, summary,
  and artifact hashes;
- `trajectory_audit.json`: per-iteration pairwise FSC-AUC matrix, Hungarian
  permutation, matched scores, assignment agreement, populations, and pass;
- `trajectory_shellwise_fsc.npz`: every candidate/reference class FSC curve;
- `relion/` and `recovar/`: exact command, combined log, one-second GPU monitor,
  GNU-time MaxRSS record, maps, and STAR artifacts.

A completed run is not checked into the benchmark registry automatically.
After the Slurm job finishes, copy only compact JSON/NPZ evidence into a schema
record, preserve the absolute run root and hashes, and run
`scripts/validate_em_benchmark_registry.py`. Do not admit an InitialModel-only
record as evidence of final real-data resolution; retain that limitation and
schedule the matched multi-seed half-map refinement described above.

## Code and focused tests

- Pair runner: `scripts/run_em_real_kclass_initialmodel_pair.py`
- Slurm launcher: `scripts/launch_em_real_kclass_initialmodel_slurm.py`
- Permutation/FSC/population audit: `scripts/audit_em_real_kclass_initialmodel.py`
- Focused tests: `tests/unit/initial_model/test_audit_em_real_kclass_initialmodel.py`,
  `tests/unit/initial_model/test_run_em_real_kclass_initialmodel_pair.py`, and
  `tests/unit/initial_model/test_launch_em_real_kclass_initialmodel_slurm.py`

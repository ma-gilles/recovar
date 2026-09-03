# Current-source C1 K=4 three-seed qualification

This report seals the C1 Ribosembly K=4 baseline rerun against current PR
source.  Its machine-readable campaign record is
`campaigns/k4-ribosembly-three-seed-9006957c6-h100.json`.  The external run
root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k4_case2_current_3seed_c1_10k128_9006957c6_20260902T2245ET`
and carries a `SAFE_TO_DELETE` marker.

RECOVAR ran from clean isolated commit
`9006957c6625963ee2efe7ba90f014fa2eac7955`, tree
`c398711136faa0154a3e3892e295a2f875268f50`.  RELION ran from commit
`d476e6f6a4f1f37627c06ace5227fc374c0c2b05`, tree
`1633d228e89d91ede8ad0996e727ec6ab1bc96ee`, with the frozen dispatch
instrumentation diff and executable recorded in the campaign ledger.  Every
paired case used one physical H100 for both engines, sequentially.

## Result

All three frozen seeds pass both the formal trajectory gate and the signed-GT
science gate.  The v2 aggregate classifies the campaign `TRAJECTORY_EXACT`:

- 3/3 replicates pass;
- 72/72 numbered/final class cells pass;
- minimum direct RECOVAR--RELION FSC-AUC is `0.9975014219`;
- minimum signed RECOVAR-minus-RELION GT FSC-AUC delta is
  `-0.0001863652`;
- minimum hard-class agreement is `0.9951`;
- every numbered and final Hungarian permutation is identity; and
- all controller-topology and class-population audits pass, with no collapsed
  class.

Here, `TRAJECTORY_EXACT` is the registry's thresholded scientific trajectory
classification; it does not mean byte-identical floating-point maps.  The
verdict uses FSC/FSC-AUC, signed GT quality, assignments, controller topology,
and occupancy rather than map correlation.

The independently rigid-aligned endpoint evaluator reports:

| Seed | RECOVAR mean GT FSC-AUC | RELION mean GT FSC-AUC | Delta | Final assignment agreement |
| ---: | ---: | ---: | ---: | ---: |
| 41001 | 0.2343172814 | 0.2343004550 | +0.0000168264 | 0.9951 |
| 41002 | 0.2394394522 | 0.2392850819 | +0.0001543703 | 0.9952 |
| 41003 | 0.2381818367 | 0.2381468141 | +0.0000350226 | 0.9959 |

The cross-seed endpoint means are `0.2373128568` for RECOVAR and
`0.2372441170` for RELION, a delta of `+0.0000687398`.  The trajectory auditor
also evaluates the frozen numbered/final maps under its own alignment and
shell convention; those per-case values are retained separately in the JSON
ledger and should not be interchanged with the independently aligned endpoint
means above.

All runs reached the five-iteration cap without satisfying convergence and
correctly recorded `final_all_data_ran=false`.  The final RECOVAR maps are
exactly the last numbered regularized half-map averages.  This is therefore
iteration-cap trajectory evidence, not post-convergence final-all-data
evidence.

## Performance

| Seed / job | RECOVAR / RELION wall (s) | RECOVAR / RELION peak HBM (MiB) | Slurm elapsed (s) |
| --- | ---: | ---: | ---: |
| 41001 / `13369457` | 1279 / 121 | 17087 / 79577 | 1467 |
| 41002 / `13369458` | 1271 / 124 | 17087 / 79579 | 1516 |
| 41003 / `13369459` | 1232 / 122 | 17089 / 79577 | 1433 |

Median engine walls are 1271 seconds for RECOVAR and 122 seconds for RELION.
RECOVAR is 10.10--10.57 times slower on this small five-iteration workload,
but its maximum sampled HBM is about 21.5% of RELION's.  HBM is a lower-bound
peak sampled every five seconds, not an allocator-exact maximum.

Setup job `13369456`, case jobs `13369457`--`13369459`, summary job
`13369460`, trajectory/topology audit jobs `13370363`--`13370365`, and strict
aggregate job `13370366` all completed `0:0`, were nonexclusive, and had exact
requested/allocated resources.  The audit jobs ran on CPU; the paired
refinements ran on H100s.

## Evidence and reproduction

The authoritative aggregate is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k4_case2_current_3seed_c1_10k128_9006957c6_20260902T2245ET/em_kclass_multiseed_strict_summary.json`,
SHA-256
`2c13a2bd90610b65e901398478802c33518b98903b4ef01175c6ee25a56a6a7d`.
Its Markdown rendering has SHA-256
`fd430b79bf0414dc79ee74b87cd89c7bebc134470c9a197953cc45f9315d811d`.
The case table and submission environment have SHA-256
`2c09a6f7eb6be50284c592fe246182098e0767d496f6d5de52efbdeabc84c678`
and `2e2a0a62cc3bf2bd61938a604c7e8c0789f4feffeab7b9e9536e149d94068d2d`.

The run root contains immutable setup, case, summary, and audit launchers as
historical records.  They hard-code the sealed run root and **must not be
resubmitted in place**.  The campaign JSON retains those original commands,
dependencies, input hashes, engine artifacts, resource records, and audit
reports for provenance.

Generate new launchers against a fresh, non-existing run root to repeat the
three-seed slice.  For example, after replacing the timestamped value below
with a new absolute path:

```bash
cd /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/source_k4_case2_current_3seed_9006957c6_20260902T2238ET/checkout
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export EM_KCLASS_MATRIX_PIXI_PY=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_pr158_k4_origin_docs_8cbebdecc_20260902/.pixi/envs/default/bin/python3.11
export RELION_SRC_DIR=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717/source/src
export EM_KCLASS_MATRIX_RELION_REFINE_MPI=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717/build/bin/relion_refine_mpi
export RELION_MODULE=relion/5.0.1/gcc-11.5.0-gpu
export RELION_MPI_RANKS=3
export KCLASS_IMAGE_BATCH_SIZE=50
export KCLASS_ROTATION_BLOCK_SIZE=2000
export EM_KCLASS_MATRIX_GT_ALIGN_REFINE_ORDERS=3
export SBATCH_ACCOUNT=gilles
export SBATCH_PARTITION=cryoem
export SBATCH_CONSTRAINT=h100
NEW_RUN_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k4_case2_repeat_YYYYMMDDTHHMMSSZ
"$EM_KCLASS_MATRIX_PIXI_PY" scripts/run_em_kclass_robustness_matrix_slurm.py \
  --scratch-dir "$NEW_RUN_ROOT" \
  --case 2 \
  --three-seed-suite
```

`NEW_RUN_ROOT` must not already exist; the generator creates it, adds the
`SAFE_TO_DELETE` marker, writes root-specific launchers, submits setup first,
submits the three cases with the setup dependency, and submits the summary with
the case dependencies.  Preserve the generated `submission.env` and
`case_table.tsv` with the new evidence.

The original trajectory audits were submitted with `sbatch --wrap` as jobs
`13370363`--`13370365`.  The checked reproduction launcher was materialized
after completion from those jobs' exact Slurm `SubmitLine`; `audit_jobs.tsv`
binds the original job IDs and the campaign ledger resolves their original
stdout/stderr paths.

The fail-closed campaign record was regenerated with:

```bash
cd /scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_pr158_k4_origin_docs_8cbebdecc_20260902
pixi run python scripts/seal_em_kclass_campaign.py \
  --run-root /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k4_case2_current_3seed_c1_10k128_9006957c6_20260902T2245ET \
  --n-classes 4 \
  --campaign-id k4-ribosembly-three-seed-9006957c6-h100 \
  --recorded-at 2026-09-03T03:11:00+00:00 \
  --output docs/benchmarks/em/campaigns/k4-ribosembly-three-seed-9006957c6-h100.json
```

No final-all-data override was enabled.  In particular,
`RECOVAR_FINAL_ALL_DATA_GRID_CORRECT` and
`RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER` were unset.

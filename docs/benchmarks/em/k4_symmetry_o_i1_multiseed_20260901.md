# O/I1 synthetic K=4 three-seed validation

This document accompanies the machine-validated campaign records
`campaigns/k4-o-three-seed-22efd8065-h100.json` and
`campaigns/k4-i1-three-seed-22efd8065-h100.json`. The sealed run root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k4_symmetry_oi_fixed_22efd8065_20260901`.
All six paired runs used RECOVAR commit
`22efd8065ed97f29f4bcdbb0eb79dcfb33df923d`, one physical NVIDIA H100,
and the same dispatch-instrumented RELION executable. RELION and RECOVAR ran
serially on the same GPU within each case.

## Scope and gate

The campaigns independently generate three 5,000-particle Ribosembly
fixtures for each of octahedral (`O`) and icosahedral (`I1`) symmetry. The
generator, RELION, and RECOVAR receive the same canonical symmetry label. All
fixtures use a 128-pixel box, white-noise level 1, uniform poses and class
weights, batch size 50, rotation block size 2,000, and a five-iteration cap.

Every iteration compares the average of RECOVAR's two regularized class half
maps with the matching RELION Class3D full map. Classes are matched afresh by
Hungarian maximization of unmasked normalized FSC-AUC. A campaign passes only
if, for every one of its 60 numbered iteration/class cells:

- direct RECOVAR-to-RELION FSC-AUC is at least 0.995;
- signed RECOVAR-minus-RELION GT FSC-AUC is at least -0.002; and
- available class-assignment agreement is at least 0.99.

The audit also requires complete five-iteration map topology, complete K=4
assignments, and exact identity between each RECOVAR `final_class` map and its
last numbered half-map average. None of the six jobs converged before the cap,
so no final-all-data reconstruction was forced.

## Frozen results

All six trajectories pass. `Min direct`, `min GT delta`, and `min agreement`
are minima over all five numbered iterations; `final mean delta` is averaged
over the four final Hungarian-matched class pairs. HBM is the five-second
engine-specific monitor peak in MiB.

| Symmetry | Seed | Paired job | Audit job | Cells | Min direct | Min GT delta | Min agreement | Final mean delta | RECOVAR / RELION wall (s) | RECOVAR / RELION HBM (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| O | 41001 | 13303554 | 13304773 | 20/20 | 0.999843004 | -1.185e-4 | 99.76% | +4.406e-5 | 482 / 64 | 18091 / 79579 |
| O | 41002 | 13303555 | 13304774 | 20/20 | 0.999783812 | -1.838e-4 | 99.86% | +4.537e-5 | 440 / 67 | 18087 / 79577 |
| O | 41003 | 13303556 | 13304775 | 20/20 | 0.999693008 | -1.581e-4 | 99.88% | -6.038e-5 | 467 / 67 | 18095 / 79581 |
| I1 | 41001 | 13303557 | 13304776 | 20/20 | 0.999993311 | -7.854e-6 | 100.00% | -3.219e-6 | 391 / 69 | 18083 / 79579 |
| I1 | 41002 | 13303558 | 13304777 | 20/20 | 0.999786869 | -1.094e-4 | 99.86% | +3.780e-5 | 397 / 69 | 18083 / 79577 |
| I1 | 41003 | 13303559 | 13304778 | 20/20 | 0.999998989 | -1.013e-5 | 100.00% | -2.842e-6 | 394 / 68 | 18083 / 79583 |

The final hard class populations are clear in both engines for every seed;
no class is below the frozen 1% threshold. Across all 24 final matched class
pairs, the independent rigid-alignment endpoint evaluator reports identical
RECOVAR and RELION GT FSC=0.143 resolutions. The O resolutions span
20.92--45.33 A and the I1 resolutions span 18.76--34.00 A. The exact
per-class values, populations, pairings, signed FSC-AUC values, shellwise
curves, source hashes, and performance records are retained in the two JSON
scorecards and their referenced sealed artifacts.

These campaigns also exercise the fixed non-C1 local-search boundary. Every
run reaches iteration 2 through the adaptive K-class coarse projector and
sparse compact-pair planner, then completes iteration 5. This is the path that
the earlier O/I1 attempt could not enter when it was routed to an unsupported
dense non-C1 implementation.

## Performance interpretation

The median engine wall times are 467 s RECOVAR versus 67 s RELION for O and
394 s versus 69 s for I1. The corresponding median per-seed wall ratios are
6.97x and 5.75x. RECOVAR's largest observed HBM peak is 18,095 MiB; RELION's
is 79,583 MiB. These are matched-H100 measurements but not claims that the
engines perform identical work internally. In particular, RELION's monitor
observes its near-full-device allocation, whereas RECOVAR uses non-preallocated
JAX memory. The scorecards retain the raw paired wall markers, monitor CSVs,
Slurm elapsed time, MaxRSS, exact one-GPU allocations, nodes, and GPU UUIDs.

## Reproduce

The campaign was generated with the following environment and launcher. The
runner writes immutable per-case launchers containing the full RECOVAR and
RELION commands.

```bash
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export EM_KCLASS_MATRIX_PIXI_PY=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_pr158_realdata_science_equiv_20260830/.pixi/envs/default/bin/python3.11
export RELION_SRC_DIR=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717/source/src
export EM_KCLASS_MATRIX_RELION_REFINE_MPI=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717/build/bin/relion_refine_mpi
export RELION_MODULE=relion/5.0.0/gcc-11.5.0
export SBATCH_ACCOUNT=gilles
export SBATCH_PARTITION=cryoem
export SBATCH_CONSTRAINT=h100
export EM_KCLASS_MATRIX_SETUP_PARTITION=cpu
export EM_KCLASS_MATRIX_SUMMARY_PARTITION=cpu
export EM_KCLASS_MATRIX_EXCLUSIVE=0
export RELION_MPI_RANKS=3
export KCLASS_IMAGE_BATCH_SIZE=50
export KCLASS_ROTATION_BLOCK_SIZE=2000
export EM_KCLASS_MATRIX_GT_ALIGN_REFINE_ORDERS=3
/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_pr158_realdata_science_equiv_20260830/.pixi/envs/default/bin/python3.11 \
  scripts/run_em_kclass_robustness_matrix_slurm.py \
  --scratch-dir /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k4_symmetry_oi_fixed_22efd8065_20260901 \
  --three-seed-suite --case 33 --case 34
```

This submitted setup job `13303553`, paired jobs `13303554`--`13303559`, and
summary job `13303560`. Each paired job requested and received exactly one
H100; O requested/received 24 CPUs and 256 GiB, while I1 requested/received 24
CPUs and 320 GiB. The independent CPU trajectory audits are jobs
`13304773`--`13304778`; each requested and received 4 CPUs and 32 GiB and
exited `0:0`.

Validate the compact records normally with:

```bash
pixi run python scripts/validate_em_benchmark_registry.py
pixi run pytest tests/unit/test_validate_em_benchmark_registry.py
```

On Della, add `--verify-files` to rehash every referenced external input and
artifact. Missing or changed files fail closed.

## Limitations

- These are synthetic, five-iteration, 5,000-particle tests of one molecular
  family; they do not replace converged 100k/256 or real-data K=4 validation.
- RELION Class3D emits full class maps rather than independent half maps, so
  masked and unmasked RELION half-map FSC are outside this campaign's scope.
- `TRAJECTORY_EXACT` is the registry name for passing every frozen numerical
  gate. It does not mean that map bytes or all floating-point state are equal.
- The fixtures test correct use of a shared O or I1 label; they do not test
  wrong-symmetry detection or ab-initio symmetry discovery.
- HBM peaks are five-second `nvidia-smi` lower bounds, not allocator-exact
  maxima.

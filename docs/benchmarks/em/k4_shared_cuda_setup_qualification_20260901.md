# K=4 shared-CUDA setup qualification

This qualification checks the orchestration introduced by RECOVAR commit
`23a980366c653162e582adc44ca6f1a9e4b66af3`: the K-class matrix launcher builds
the custom CUDA library and external RELION binding once in a dependency setup
job, seals both binaries, and makes every case job verify and reuse them. It is
an orchestration and one-iteration smoke qualification, not a replacement for
the five-iteration K=4 trajectory records.

The first attempt, setup job `13305300`, requested a CPU-only node and failed
before any test because that node exposed neither `nvcc` nor `nvidia-smi`. No
scientific result was produced. Commit `23a980366` changes the setup defaults
to the case partition and constraint with `gpu:1`, rejects a setup GRES that
does not contain a GPU, and records the resolved setup GRES in
`submission.env`.

## Successful end-to-end result

The corrected run is sealed under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k4_latest_setupqual_23a980366_20260901`
and contains a `SAFE_TO_DELETE` marker. All requested and allocated resources
matched exactly; no job was exclusive.

| Stage | Job | State | Elapsed | Requested = allocated | MaxRSS |
| --- | ---: | --- | ---: | --- | ---: |
| Shared setup | 13305541 | COMPLETED `0:0` | 272 s | `cpu=8,mem=64G,gres/gpu=1,node=1` | 2858 MiB |
| Case 21, one iteration | 13305542 | COMPLETED `0:0` | 250 s | `cpu=24,mem=192G,gres/gpu=1,node=1` | 4132 MiB |
| Summary | 13305543 | COMPLETED `0:0` | 23 s | `cpu=2,mem=64G,node=1` | 60 MiB |

The setup ran on one NVIDIA H100 80GB HBM3 and produced:

- `libcuda_backproject.so`, SHA-256
  `9a9a9a6f3b218aa245c2845f615ef8156e33be8c93c342c6da7ced2c082c8899`;
- `_relion_bind_core.cpython-311-x86_64-linux-gnu.so`, SHA-256
  `6cbdb9719b96a8e5f81f163c773b2c4804d8134e31487ed907f7d6c331f86176`.

The case log contains checksum verification for both sealed binaries and no
`nvcc` invocation or binding build. It then completed the 3,000-particle,
box-128, K=4 case-21 endpoint. RECOVAR and RELION mean GT FSC-AUC were
`0.13578052979492225` and `0.13578049259079011`, respectively, a signed
RECOVAR-minus-RELION difference of `3.720413213614826e-08`. RECOVAR used
8,845 MiB peak HBM and 104 s wall time; RELION used 79,585 MiB peak HBM and
11 s wall time. These are one-iteration smoke measurements and should not be
used as converged performance numbers.

## Reproduce from source

The generated scripts in the sealed run root are immutable evidence and must
not be resubmitted in place. To reproduce the one-iteration case from the
qualified source, choose a new empty scratch root and invoke the source
launcher. The launcher writes the setup, case, and summary jobs with the same
dependency chain, creates the `SAFE_TO_DELETE` marker, and records the resolved
environment in its new run root:

```bash
cd /scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_pr158_em_evidence_integration_20260901
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export SBATCH_PARTITION=cryoem
export SBATCH_ACCOUNT=gilles
export SBATCH_CONSTRAINT=h100
export RELION_MODULE=relion/5.0.0/gcc-11.5.0
export RELION_SRC_DIR=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717/source/src
export EM_KCLASS_MATRIX_RELION_REFINE_MPI=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/relion_k4_100k_dispatchv2_20260717/build/bin/relion_refine_mpi
export EM_KCLASS_MATRIX_PIXI_PY=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_pr158_em_evidence_integration_20260901/.pixi/envs/default/bin/python
export RELION_MPI_RANKS=3
export KCLASS_IMAGE_BATCH_SIZE=50
export KCLASS_ROTATION_BLOCK_SIZE=2000
export EM_KCLASS_MATRIX_GT_ALIGN_REFINE_ORDERS=3
/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_pr158_em_evidence_integration_20260901/.pixi/envs/default/bin/python \
  scripts/run_em_kclass_robustness_matrix_slurm.py \
  --scratch-dir /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k4_shared_cuda_setup_reproduction_20260901 \
  --case 21 \
  --max-iter-override 1 \
  --time-limit-override 00:45:00
```

Use a different new absolute `--scratch-dir` for each repetition. The command
submits exactly one H100 GPU for setup and one H100 GPU for the case; it never
requests an exclusive node. After submission, verify requested and allocated
resources from the recorded job IDs with `scontrol show job` before accepting
the run.

The setup and case launcher SHA-256 digests are, respectively,
`11022ed0aaf5331624a61c42bb197c288c94d3d2a72fc761536453b23aeb90e8`
and
`8084222bccb8338bf6b87abc43f9c3d2754b72037e18419bd79e44bf4a17c422`.
The generated summary JSON has SHA-256
`e3d4f48bdd2acf70f335a4f0e9e987bdeabfbc4e68eb1d5a7530e14f6d4ed8ea`.

The normal source-level regression gate is:

```bash
pixi run pytest tests/unit/test_run_em_kclass_robustness_matrix_slurm.py
pixi run ruff check scripts/run_em_kclass_robustness_matrix_slurm.py tests/unit/test_run_em_kclass_robustness_matrix_slurm.py
```

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

## Reproduce the sealed scripts

The generated scripts are immutable evidence. To repeat the same dependency
chain in a fresh copy of the sealed root, submit the setup first, make the case
depend on successful setup, and make the summary depend on the case reaching a
terminal state:

```bash
setup_job=$(sbatch --parsable /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k4_latest_setupqual_23a980366_20260901/jobs/em_kclass_matrix_setup.sh)
case_job=$(sbatch --parsable --dependency=afterok:${setup_job} /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k4_latest_setupqual_23a980366_20260901/jobs/em_kclass_matrix_21_ribo_k4_3k_g128_white_noise0p2_uniform.sh)
sbatch --dependency=afterany:${case_job} /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k4_latest_setupqual_23a980366_20260901/jobs/em_kclass_matrix_summary.sh
```

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

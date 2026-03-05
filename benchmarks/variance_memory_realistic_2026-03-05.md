# Variance Memory Benchmark (Realistic Local Dataset)

Date: 2026-03-05  
Branch: `fix/variance-memory-noise-half`  
Environment: `conda run -n recovar_dev_1`, `TMPDIR=/home/mg6942/tmp_jax_bench`

## Dataset

- Particles: `/home/mg6942/recovar/outputs/proper_em_results/20260115_003317/sim/particles.128.mrcs`
- Poses: `/home/mg6942/recovar/outputs/proper_em_results/20260115_003317/sim/poses.pkl`
- CTF: `/home/mg6942/recovar/outputs/proper_em_results/20260115_003317/sim/ctf.pkl`
- Box size: `128x128`
- Total images in stack: `1000`

## Commands

Old baseline (repo: `/home/mg6942/recovar`):

```bash
TMPDIR=/home/mg6942/tmp_jax_bench conda run -n recovar_dev_1 \
  python -m recovar.commands.pipeline \
  /home/mg6942/recovar/outputs/proper_em_results/20260115_003317/sim/particles.128.mrcs \
  --poses /home/mg6942/recovar/outputs/proper_em_results/20260115_003317/sim/poses.pkl \
  --ctf /home/mg6942/recovar/outputs/proper_em_results/20260115_003317/sim/ctf.pkl \
  --mask none \
  -o /home/mg6942/recovar/outputs/proper_em_results/20260115_003317/bench_old_20260305_tmpdir
```

Patched branch (repo: `/home/mg6942/heterogeneity_dev-1_memfix`):

```bash
TMPDIR=/home/mg6942/tmp_jax_bench conda run -n recovar_dev_1 \
  python -m recovar.commands.pipeline \
  /home/mg6942/recovar/outputs/proper_em_results/20260115_003317/sim/particles.128.mrcs \
  --poses /home/mg6942/recovar/outputs/proper_em_results/20260115_003317/sim/poses.pkl \
  --ctf /home/mg6942/recovar/outputs/proper_em_results/20260115_003317/sim/ctf.pkl \
  --mask none \
  -o /home/mg6942/heterogeneity_dev-1_memfix/bench_new_20260305_tmpdir
```

Same pair with `--n-images 500`:

- Old output: `/home/mg6942/recovar/outputs/proper_em_results/20260115_003317/bench_old_20260305_n500`
- New output: `/home/mg6942/heterogeneity_dev-1_memfix/bench_new_20260305_n500`

## Results

### Full stack (`n_images=1000`)

- Old baseline: failed with OOM during noise->variance path. Runtime error reported attempted allocation `17.82 GiB`.
- Patched branch: also failed with OOM, but earlier in noise estimation, attempted allocation `7.87 GiB`.
- In both runs, `run.log`-reported peak before failure was still low (`peak:1`), because failure happened during a large deferred allocation.

### Reduced stack (`n_images=500`)

- Old baseline: failed with OOM in variance path, attempted allocation `9.14 GiB`.
- Patched branch: reached and completed `compute_variance` passes:
  - `compute_variance: time to compute variance: 6.3s`
  - subsequent quick recompute passes `0.2s`
  - `report_memory_device` reached `peak:4`
- Patched run was manually stopped after variance and early covariance setup to avoid a long PCA tail.

## Relevant Logs

- Old 1000: `/home/mg6942/recovar/outputs/proper_em_results/20260115_003317/bench_old_20260305_tmpdir/run.log`
- New 1000: `/home/mg6942/heterogeneity_dev-1_memfix/bench_new_20260305_tmpdir/run.log`
- Old 500: `/home/mg6942/recovar/outputs/proper_em_results/20260115_003317/bench_old_20260305_n500/run.log`
- New 500: `/home/mg6942/heterogeneity_dev-1_memfix/bench_new_20260305_n500/run.log`

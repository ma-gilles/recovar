# cryoSPARC 3DVA ↔ recovar adapter

Tools for running cryoSPARC 3DVA on recovar-simulated datasets and scoring
the output through the same pipeline that scores recovar's covariance / PPCA
methods, so 3DVA can be compared head-to-head against cov, ppca-100,
ppca-projcov-iter, etc.

## Convention translation

The adapter in `adapt_3dva_to_result.py` converts cryoSPARC 3DVA outputs into
a recovar-style `result/` directory. Key conversions (verified against a cov
baseline on Ribosembly σ²=1):

- **Eigenvolumes**: cryoSPARC ships the `q` components as real MRC maps with
  arbitrary scale. The adapter does a real QR on the flattened components and
  rescales so each column's Fourier-Hermitian norm is 1 (equivalent to real
  L2 norm `1/sqrt(N)`) — the convention pipeline scoring expects.
- **Latent coordinates**: `z_pipeline = sqrt(N) * R @ z_cryosparc`, where `R`
  is the upper-triangular factor from the QR above.
- **Mean**: use the consensus mean the 3DVA job was seeded with (from the
  `cov` baseline's `output/volumes/mean.mrc`), NOT the `_map.mrc` cryoSPARC
  re-reconstructs — they are at different absolute scales.
- **Optional `c_scale`**: cryoSPARC's internal z units are not the same as
  recovar's Fourier-projection units. If a GT pack is passed, the adapter
  fits a single global scalar by OLS against `alpha_gt` so `embed_metric` is
  comparable. Set `gt_pack=None` to skip (will give meaningful `pc_metric`
  and `cluster_metric` but low `embed_metric`).

## Scripts

- `adapt_3dva_to_result.py` — core adapter. Importable or runnable as
  `python adapt_3dva_to_result.py <staged_dir> <cov_result_dir> <out_result_dir> <var_uid>`.
- `launch_3dva_sweep.py` — stages inputs to `/tigress/...` (cryoSPARC
  rejects symlinks to `/scratch`) and queues `import_particles` +
  `import_volumes` + `var_3D` jobs for a sweep of (dataset, SNR) cases
  via `cryosparc-tools`. Site-specific — edit `STAGING_ROOT`,
  `SWEEP_ROOT`, project/workspace IDs, and license for your install.
- `stage_and_adapt_all.py` — once `var_3D` jobs complete, copies outputs
  out of the cryoSPARC project dir, calls `adapt` on each, and scores via
  `ppca_refit_subspace_em.score_one_result`. Reads job status from
  `job.json` on disk (no cryosparc-tools dependency in the scoring env).

## Example comparison (16 cases, apples-to-apples sweep)

3DVA is competitive with `cov` at σ²=1 and matches `ppca` on `embed_metric`
at σ²=10 on some datasets, but collapses at σ²≤0.1 while recovar's
cov / ppca methods still extract meaningful structure. Full numbers in the
PR description.

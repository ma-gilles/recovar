# Deconvolved Kernel Regression Test Protocol

This note is for development experiments that compare standard latent kernel
regression against the one-dimensional deconvolved kernel regression path.
It is intentionally practical: generate a synthetic spike run, create an
apple-to-apple noisy latent coordinate, run both estimators on the same data,
and measure against known ground truth.

## Code Paths

Use the live implementation rather than reimplementing the kernel:

- Deconvolved weights:
  `recovar.heterogeneity.deconvolved_kernel_regression.deconvolution_weights_1d_many`
- Deconvolved candidate volumes:
  `recovar.heterogeneity.deconvolved_kernel_regression.estimate_deconvolved_kernel_volumes`
- Standard candidate volumes:
  `recovar.heterogeneity.kernel_regression_reconstruction.estimate_standard_kernel_volumes`
- `compute_state` dispatch:
  `recovar.heterogeneity.heterogeneity_volume.make_volumes_kernel_estimate_local`

For deconvolution, `latent_precision` means inverse latent noise variance:

```python
latent_noise_variance = 1.0 / latent_precision
sigma_ref = median(sqrt(latent_noise_variance))
h_grid = lambda_grid * sigma_ref
```

Do not invert `latent_precision` twice.

## Data Generation

Use a synthetic spike run where every particle has a known state assignment and
every state has a known GT volume. The current 100k reference experiment used:

```bash
pixi run python recovar/commands/spike_walkthrough.py \
  --output-dir /scratch/gpfs/GILLES/mg6942/runs/spike_grid256_box320_noise10_n100k_opt34_YYYYMMDD \
  --pdb-dir /home/mg6942/myscratch/spike_pdb_motion \
  --render-bfactor 100 \
  --render-voxel-size 1.25 \
  --grid-size 256 \
  --n-images 100000 \
  --noise-level 10.0 \
  --compute-state-save-all-estimates
```

Useful outputs:

- particle/state labels: `03_dataset/state_assignment.npy`
- GT target state volume: `04_ground_truth/gt_vol0032.mrc`
- source pipeline: `06_pipeline/`
- source latent precision: `06_pipeline/model/zdim_1/latent_precision_noreg.npy`

## Artificial Noisy GT Embedding

To isolate the kernel-regression question from embedding/model mismatch, create
a copy of the source pipeline and replace its zdim-1 latent coordinate with a
known true coordinate plus Gaussian noise:

1. Load GT PC0 scores from the active GT volumes, indexed by
   `03_dataset/state_assignment.npy`.
2. Affine-scale those true GT scores to the same range as the source oracle
   embedding.
3. Add Gaussian noise with
   `sigma_noise = mean(sqrt(1 / latent_precision_noreg))`.
4. Set `latent_precision_noreg` to the corresponding constant precision
   `1 / sigma_noise**2`.
5. Use the true scaled coordinate of the target state as the target latent point.

Record a metadata JSON with the source pipeline, target state, scaling slope and
intercept, noise seed, noise sigma, and target latent point. This makes the run
reproducible and confirms that both standard and deconvolved modes use identical
data.

## Candidate Runs

Run standard and deconvolved `compute_state` on the same copied pipeline and the
same target latent point.

Standard:

```bash
pixi run python -m recovar.commands.compute_state \
  /path/to/pipeline_gtpc0_scaled_meansigma_seedYYYYMMDD \
  -o /path/to/output_standard \
  --latent-points /path/to/target_latent_point_true_state32.txt \
  --zdim1 \
  --save-all-estimates \
  --kernel-regression-mode standard
```

Deconvolved, broad lambda grid:

```bash
pixi run python -m recovar.commands.compute_state \
  /path/to/pipeline_gtpc0_scaled_meansigma_seedYYYYMMDD \
  -o /path/to/output_deconvolved_lam0p2_20 \
  --latent-points /path/to/target_latent_point_true_state32.txt \
  --zdim1 \
  --save-all-estimates \
  --kernel-regression-mode deconvolved \
  --deconv-lambda-grid 0.2,0.21970823,0.24135853,0.26514227,0.2912697,0.31997174,0.35150212,0.38613955,0.42419018,0.46599036,0.51190958,0.56235374,0.61776872,0.67864435,0.74551874,0.81898301,0.89968653,0.98834267,1.0857351,1.1927247,1.3102571,1.4393713,1.5812086,1.7370227,1.908191,2.0962263,2.3027908,2.5297104,2.778991,3.0528359,3.3536659,3.6841399,4.0471793,4.445993,4.8841062,5.3653916,5.8941034,6.4749151,7.1129606,7.8138799,8.5838685,9.4297327,10.358949,11.379732,12.501104,13.732977,15.08624,16.572855,18.205964,20
```

Long GPU runs should go through Slurm. Keep `PYTHONNOUSERSITE=1`,
`XLA_PYTHON_CLIENT_PREALLOCATE=false`, unique scratch `TMPDIR`, `PIXI_HOME`, and
`RATTLER_CACHE_DIR`, and log `recovar.__file__`, `jax.__file__`, `jax.devices()`,
the git commit, command, and output directory.

## Masks

For local comparisons, use one mask for all GT metrics. The current experiment
uses the same local ball as the CV subvolume:

```text
subvolume: 1712
center: [15, -11, 54]
radius: 13 voxels
mask: local_cv_mask_subvolume1712_plus2z_center_15_-11_54_radius13.mrc
```

Do not mix masks between FSC and error plots unless the plot title says so.

## Primary Metrics

For this comparison, CV is diagnostic only. The primary metrics are true
candidate quality against GT.

For candidate volume `V_h`, target GT volume `V_gt`, and mask `M`:

```text
FSC_h(shell) =
  FSC_shell( FFT(M * V_h), FFT(M * V_gt) )

relative_error_h(shell) =
  sum_shell |FFT(M * (V_h - V_gt))|^2
  / max(sum_shell |FFT(M * V_gt)|^2, eps)
```

Both curves use the full volume Fourier shell axis:

```text
spatial_frequency(shell) = shell / (box_size_voxels * voxel_size_A)
```

For the 256-voxel, 1.25 A/voxel spike runs, the frequency range is
`0..0.39375 1/A`.

Plot all candidate curves when tuning the grid:

- standard candidates: all 50 `estimates_filt/*.mrc`
- deconvolved candidates: all 50 `estimates_filt/*.mrc`
- mark the candidate with best median relative error over a fixed shell range,
  e.g. `0 < frequency <= 0.35 1/A`

## Kernel Shape Diagnostics

Plot real-space latent kernels over the observed latent-difference range.

For the standard path, `heterogeneity_bins` are precision-distance thresholds.
In the 1D diagnostic plot, use:

```python
u = sqrt(precision_ref * diff**2 / (2 * bin_value))
K_std = 0.75 * (1 - u**2) * (abs(u) < 1)
```

For the deconvolved path, use the live code:

```python
h = lambda_value * sigma_ref
K_dec = epanechnikov_deconvolution_kernel_1d(diff / h, lambda_value)
```

Always include normalized shape plots in addition to raw plots. Small lambda
values can have very large signed weights, so raw plots alone can hide the
usable part of the grid.

## Toy Sanity Check

Before interpreting cryo-EM results, run a scalar errors-in-variables toy:

1. Draw `x_true` on the same latent scale as the spike run.
2. Set `y = sin(x_true / sigma_ref) + y_noise`.
3. Observe `x_obs = x_true + Normal(0, sigma_ref)`.
4. Estimate `E[y | x]` at target grid points using:
   - standard Epanechnikov weights on noisy `x_obs`
   - deconvolved weights from `deconvolution_weights_1d_many`
5. Solve the scalar weighted normal equation:
   `estimate = sum(weight * y_obs) / sum(weight)`.

Expected qualitative result: deconvolution should reduce x-noise bias for a
moderate lambda range, while too-small lambda values can become unstable because
signed weights nearly cancel.

## What To Report

For each run, report:

- git commit and command line
- dataset root, pipeline root, target GT volume, target latent point
- mask path and mask definition
- standard grid and deconvolved lambda grid
- all-candidate FSC-vs-GT plot
- all-candidate relative-error-vs-GT plot
- real-space kernel raw and normalized plots
- best candidate by median true relative error over the selected shell range
- whether any tiny-lambda candidates are numerically unstable

Do not use local CV error as the main success criterion for this experiment.
Use CV only to understand how `compute_state` would choose an estimator.

## Report Generator

The spike walkthrough now attempts to generate the comparison report by default.
It writes `<output_dir>/08_kernel_report` when it can resolve both a standard
and a deconvolved `compute_state` output. If the current run only produced one
mode, pass the paired output explicitly:

```bash
pixi run python -m recovar.commands.spike_walkthrough \
  --output-dir /path/to/spike_run \
  --compute-state-kernel-regression-mode deconvolved \
  --kernel-report-standard-dir /path/to/standard_compute_state \
  --kernel-report-deconvolved-dir /path/to/deconvolved_compute_state \
  --kernel-report-mask /path/to/local_cv_mask.mrc
```

The report can also be regenerated directly:

```bash
recovar spike_kernel_report \
  --standard-root /path/to/standard_compute_state \
  --deconvolved-root /path/to/deconvolved_compute_state \
  --pipeline-root /path/to/pipeline \
  --target-point /path/to/target_latent_point.txt \
  --target-volume /path/to/gt_vol0032.mrc \
  --mask /path/to/local_cv_mask.mrc \
  --out-dir /path/to/08_kernel_report
```

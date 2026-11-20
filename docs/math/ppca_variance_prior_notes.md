# PPCA Variance Prior Notes

This note clarifies the different quantities produced by the variance-estimation code and which of them have the right semantics for a PPCA prior.

## Goal

For PPCA with basis size `q`, we want a per-voxel, per-component prior of the form

`W_prior[k, j] ~= (1 / q) * E[ sum_j |W_j(k)|^2 ]`

or, after shell averaging,

`W_prior[k, j] ~= d_shell(s(k)) / q`

where `d_shell[s]` is a shell-averaged estimate of the total Fourier signal variance on shell `s`.

That means the natural target quantity is **signal power**, not a reconstruction regularization parameter.

## Quantities Returned By `compute_variance()`

Let `h in {0,1}` index the two half-sets.

The variance kernel accumulates, per voxel `k`,

- `signal_h[k]`: backprojected residual-squared numerator
- `ctf_w_h[k]`: backprojected CTF/noise weight denominator

The half-map variance estimate is

`d_h[k] = signal_h[k] / ctf_w_h[k]`

and the direct combined estimate is

`d_comb[k] = (d_0[k] + d_1[k]) / 2`

This `combined` quantity is the direct estimate of the per-voxel Fourier signal variance. If we want a PPCA prior, the most direct shell candidate is

`d_shell[s] = average_{k in shell s} d_comb[k]`

possibly followed by clipping and smoothing.

## What `compute_fsc_prior_gpu_v2()` Returns

`compute_fsc_prior_gpu_v2()` does **not** return raw shell variance. It returns a RELION-style shell regularization `tau`.

For shell FSC

`FSC[s] = corr(d_0, d_1 on shell s)`

it forms

`SNR[s] = FSC[s] / (1 - FSC[s])`

and then computes

`prior_avg[s] = SNR[s] * sum_{k in s} bot[k] / sum_{k in s} top[k]`

When no previous prior is supplied, the code reduces to

- `top[k] = 1`
- `bot[k] = 1 / lhs[k]`

so

`prior_avg[s] = SNR[s] * average_{k in s}(1 / lhs[k])`

This is a conservative reconstruction prior `tau`, not the shell signal variance `d_shell[s]`.

Consequences:

- `tau` can be orders of magnitude smaller than `combined`
- large `lhs` drives `tau` down
- poor half-set agreement drives `tau` down
- using `tau` as a PPCA variance prior is a semantic mismatch

## Effect Of `substract_shell_mean`

There are two FSC-based `tau` variants:

- `prior_total_signal`: `substract_shell_mean=False`
- `prior_shell_subtracted`: `substract_shell_mean=True`

The shell-subtracted version removes the radial shell mean before FSC, so it measures only within-shell fluctuations. This makes it even less appropriate as a PPCA prior on total shell signal power.

The non-subtracted version is closer in spirit to total signal power, but it is still a RELION `tau`, not `d_shell[s]`.

## What The Old Note Code Actually Used

The old note code in `/home/mg6942/PPCA-EM-Notes/debug_variance_prior.py` did **not** use `variance_prior` as the PPCA candidate. It used

`average_over_shells(var_est["combined"])`

with an all-ones mask.

That is the key semantic difference.

The old backup code also differed in two ways:

- `compute_variance()` used `substract_shell_mean=False` when forming `variance_prior`
- `compute_variance()` prefiltered the mean when `disc_type == "cubic"`

Only the first difference is relevant here. The current variance kernel uses `linear_interp` internally, so the missing cubic-prefilter branch is not a live bug in the current variance-estimation path.

## Most Reasonable PPCA Prior Candidate

For PPCA, the most reasonable one-shot prior candidate is the shell-averaged
regularized combined estimate, with a high-resolution fallback:

1. Estimate the mean.
2. Compute `variance["combined"]` with `use_regularization=True`.
3. Shell-average `variance["combined"]`.
4. Keep the reliable shells unchanged.
5. Repair the unreliable high-frequency tail with a `|mean|^2 * median_ratio`
   fallback.
6. Clip to a small positive floor.
7. Broadcast that shell curve back to all voxels and divide by `q`.

In symbols,

`d_raw_shell[s] = average_{k in s} d_comb_reg[k]`

`d_ppca_shell[s] = d_raw_shell[s]` on reliable shells

`d_ppca_shell[s] = |mean|^2_shell[s] * median_ratio` on the unreliable tail

`W_prior[k, j] = d_ppca_shell[s(k)] / q`

This used to be what the cleaned `data_once` path did in `bench_mstep.py`.

Current provisional bench decision:

- `data_once` now aliases to the legacy `hybrid_shell` prior
- the initial `commands/pipeline.py --use-ppca` integration also uses the same `hybrid_shell` prior
- this is a temporary fallback, not the final intended formulation
- we still need to tighten up the unified estimated-prior path

This still has caveats:

- with unknown contrast, `combined` is inflated unless contrast is already corrected or mostly estimated
- moving masks can distort the estimate
- an all-ones mask is still the cleanest diagnostic setting

So for contrast-enabled PPCA, the intended strategy is still:

- estimate contrast first, at least approximately
- then estimate the shell variance prior from the contrast-corrected residual path
- use the repaired shell-averaged `combined`, not the FSC-derived `variance_prior`

## Old-Code Check

I checked the linked old code paths.

Relevant differences that were present there and are not part of the current mainline path:

- old debug script used an all-ones mask for the prior diagnostic
- old debug script used shell-averaged `combined`, not `variance_prior`
- old backup `compute_variance()` used `substract_shell_mean=False`
- old backup carried a Gaussian/hybrid shell heuristic as an experimental fallback

I did **not** find a missing correctness fix in the old code beyond those semantic differences. In particular, I did not find another active cubic-prefilter bug in the current variance-estimation path because it is using `linear_interp`, not cubic slicing.

## Recommendation

Do not use `variance_prior`, `prior_total_signal`, or `prior_shell_subtracted` as the PPCA prior candidate.

Longer-term, use shell-averaged `variance["combined"]` instead, preferably:

- from the regularized `combined` estimate
- with a tail-only `|mean|^2` fallback
- with an all-ones mask for diagnostics
- after contrast has been estimated well enough that the variance scale is meaningful

Short-term provisional fallback for bench work:

- use `hybrid_shell`
- keep `combined_reg_raw` and `combined_noreg_raw` as diagnostics
- treat this as a stopgap until the unified estimated-prior path is tightened up

## Code References

- `recovar/heterogeneity/covariance_estimation.py:compute_variance`
- `recovar/reconstruction/regularization.py:compute_fsc_prior_gpu_v2`
- `bench_mstep.py:estimate_data_prior`
- `recovar/ppca/prior_estimation.py`
- `/home/mg6942/PPCA-EM-Notes/debug_variance_prior.py`
- `/home/mg6942/PPCA-EM-Notes/recovar_backup/recovar/covariance_estimation.py:compute_variance`
- `/home/mg6942/PPCA-EM-Notes/recovar_backup/recovar/ppca/whitening_test.py:compute_hybrid_prior_shells`

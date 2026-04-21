# `ave_Pmax` Is A Fragile Parity Metric

Date: 2026-04-21

For multi-iteration RELION parity work, do **not** use `ave_Pmax` as the
only health signal.

`ave_Pmax` is still worth logging because RELION uses it in control flow, but
it is highly sensitive to tiny changes in the score landscape. Small score or
prior shifts can move a particle from `Pmax=0.95` to `0.55` without meaning a
large change in the underlying reconstruction quality. That makes `ave_Pmax`
an excellent tripwire and a poor standalone verdict.

## What To Track Alongside `ave_Pmax`

For every parity iteration, save and compare at least:

- `current_size`
- estimated resolution / FSC crossing shell
- `tau2` shell values
- `sigma2_noise` shell values
- `data_vs_prior` / SSNR shell values
- `sigma_offset`
- `nr_significant_samples` or effective local-search support
- per-particle `Pmax` distribution summaries, not just the mean:
  - median
  - 10th / 90th percentiles
  - count of large outliers vs RELION
- angle-refinement metrics that are less coarse than `ave_Pmax`:
  - per-particle angular error to RELION best pose: mean / median / p90 / max
  - per-particle view-direction error to RELION best pose: mean / median / p90 / max
  - per-particle in-plane / psi-like error to RELION best pose: mean / median / p90 / max
  - per-particle translation error to RELION best pose: mean / median / p90 / max
  - threshold fractions, e.g. fraction within `2°`, `5°`, `10°`, `20°`
  - `smallest_change_angles`
  - `smallest_change_offsets`
  - `acc_rot`
  - `frac_changed`
- half-map or replay-volume agreement metrics:
  - FSC
  - volume correlation
  - shell-wise radial power comparisons
- ground-truth metrics when the data are simulated:
  - per-particle RECOVAR-vs-GT angular error: mean / median / p90 / max
  - per-particle RELION-vs-GT angular error: mean / median / p90 / max
  - split RECOVAR-vs-GT and RELION-vs-GT view-direction / in-plane errors
  - final half-map / merged-map FSC vs `reference_gt.mrc`
  - final half-map / merged-map correlation vs `reference_gt.mrc`
  - per-iteration regularized / unregularized / RELION map FSC vs GT
  - FSC=0.5 and FSC=0.143 crossing shells/resolutions

## Practical Rule

Interpret `ave_Pmax` together with the coupled state that drives the next
iteration:

- if `ave_Pmax` moves but `current_size`, `tau2`, `sigma2_noise`, FSC, and
  shell power all stay aligned, treat it as a sensitive posterior-shape
  warning, not immediate proof of a broken reconstruction
- if `ave_Pmax` moves together with `tau2`, `sigma2_noise`, `current_size`, or
  FSC, treat it as a real trajectory divergence

## Current Implication For This Branch

When debugging the hp4 local-search transition, the minimum comparison table
should include:

- `ave_Pmax`
- `current_size`
- resolution
- `sigma_offset`
- local-search support size
- angular-error quantiles to RELION best poses
- view-direction-error quantiles to RELION best poses
- in-plane / psi-error quantiles to RELION best poses
- translation-error quantiles to RELION best poses
- threshold fractions within `2°`, `5°`, `10°`, `20°`
- `smallest_change_angles`
- `acc_rot`
- first few shells of `tau2`
- first few shells of `sigma2_noise`

This branch should keep reporting `ave_Pmax`, but no future parity claim
should be based on `ave_Pmax` alone.

## Fairness Rule For GT Comparisons

When using simulated data with a known `reference_gt.mrc`, GT metrics are more
useful than RECOVAR-vs-RELION map correlation, but they are only meaningful if
the particle counts match.

- If RECOVAR is run on a subsampled replay but RELION is compared using the
  full dataset, the GT gap is confounded by the `N_particles` mismatch.
- In practice, `sigma2` and `tau2` can shift by roughly the half-set particle
  count ratio, so a subsampled replay may look artificially worse than RELION
  even when the implementation is correct for that subset.
- For any GT-vs-RELION claim, use either:
  - the full dataset on both sides, or
  - matched subsets on both sides

## Why Separate View Direction And In-Plane Error

At hp4 local search, full SO(3) angle can look worse than the actual selector
support problem. The local-search support factorizes into view direction and
in-plane angle, so parity debugging should track those separately:

- view-direction error isolates whether the correct HEALPix neighborhood is
  being searched
- in-plane error isolates whether the psi support inside that neighborhood is
  aligned
- full rotation error is still useful, but it should be interpreted only
  alongside those two split metrics

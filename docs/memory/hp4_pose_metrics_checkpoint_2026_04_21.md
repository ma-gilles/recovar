# HP4 Pose Metrics Checkpoint

Date: 2026-04-21

This note records the first checkpoint after switching the local-search
selector to use metadata derived from the actual scored grid and after fixing
`get_local_rotation_grid_fast()` to use `max(sigma_rot, sigma_psi)` for the
direction cone.

## Main Takeaway

The hp4 local-search pose path improved substantially on the tiny replay
benchmark even before re-running the full 5k dataset.

The old 64-particle hp4 run at
`_agent_scratch/multi_iter_tiny_hp4_pertgrid_v3_metrics/pose_comparison_iter003.npz`
had:

- angular error mean `29.37°`
- angular error median `11.25°`
- angular error p90 `98.46°`
- angular error p95 `146.01°`
- angular error p99 `175.11°`
- translation error mean `0.741 px`

The new 64-particle hp4 run at
`_agent_scratch/multi_iter_tiny_hp4_sub064_v2_posemetrics/pose_comparison_iter003.npz`
had:

- angular error mean `5.16°`
- angular error median `4.52°`
- angular error p90 `8.63°`
- view-direction error mean `3.49°`
- in-plane error mean `3.14°`
- translation error mean `0.636 px`

The new 256-particle hp4 run at
`_agent_scratch/multi_iter_tiny_hp4_sub256_v1/pose_comparison_iter003.npz`
had:

- `ave_Pmax` gap only `-0.0093` at hp4
- final half-map correlation vs RELION `~0.935`
- angular error mean `2.93°`
- angular error median `3.75°`
- angular error p90 `5.62°`
- view-direction error mean `2.32°`
- in-plane error mean `1.42°`
- translation error mean `0.547 px`

## Interpretation

- The hp4 failure is no longer best described as a gross pose-collapse.
- Splitting pose error into full rotation, view direction, and in-plane angle
  is materially more informative than relying on `ave_Pmax`.
- The selector/path changes improved the local-search geometry enough that the
  next branch-level question is full-scale stability on the original 5k run,
  not whether hp4 is still catastrophically wrong on tiny subsets.

## Jobs And Runs Started From This Checkpoint

- small full-loop benchmark:
  - Slurm job `7211695`
  - 500 images, noise `0.1`
- full 5k hp4 replay:
  - Slurm job `7211897`
  - `scripts/run_multi_iter_parity.py --iter 3 --max_iter 4 --max_healpix_order 4`

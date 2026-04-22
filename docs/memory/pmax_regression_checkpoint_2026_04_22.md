## hp4 Pmax regression checkpoint

Branch state for parallel debugging on `claude/relion-parity-flag-audit`.

### Goal

Debug the matched-stage hp4 `ave_Pmax` regression on the 5k replay benchmark.

Reference comparison stage:
- start from RELION iter 5
- run 1 RECOVAR hp4 iteration
- compare against RELION iter 6

### Old-good matched-stage baseline

From:
- `_agent_scratch/multi_iter_full_hp4_v4_childfix_gt/pmax_comparison_iter002.npz`
- `_agent_scratch/multi_iter_full_hp4_v4_childfix_gt/gt_comparison_iter002.npz`

Metrics:
- RECOVAR `ave_Pmax = 0.8486816373`
- RELION `ave_Pmax = 0.9289474272`
- gap `= -0.0802657899`
- merged FSC vs GT @ `0.5`: `37`
- half-map FSC @ `0.143`: `41`
- merged corr vs GT: `0.9580670316`

### Current default matched-stage result

From:
- `_agent_scratch/local_profile_full5k_it006_v1/pmax_comparison_iter000.npz`
- `_agent_scratch/local_profile_full5k_it006_v1/gt_comparison_iter000.npz`

Metrics:
- RECOVAR `ave_Pmax = 0.8279855920`
- RELION `ave_Pmax = 0.9289474272`
- gap `= -0.1009618352`
- merged FSC vs GT @ `0.5`: `37`
- half-map FSC @ `0.143`: `41`
- merged corr vs GT: `0.9559876947`

### What has been ruled out

The following runs are numerically identical for the final matched-stage outputs:

1. Current default path
- output: `_agent_scratch/local_profile_full5k_it006_v1`
- toggles: profile `on`, projection reuse `on`, fused windowed adjoint `on`

2. All new toggles disabled
- output: `_agent_scratch/local_profile_full5k_it006_control_offoffoff_v1`
- toggles: profile `off`, projection reuse `off`, fused windowed adjoint `off`

3. Exact per-image local search
- output: `_agent_scratch/local_profile_full5k_it006_single_image_groups_on_v1`
- toggles: all above `off`, plus `single_image_groups=on`

Observed result:
- all three give RECOVAR `ave_Pmax = 0.8279855920`
- all three keep the same matched-stage map/pose quality:
  - merged FSC @ `0.5`: `37`
  - half-map FSC @ `0.143`: `41`
  - merged corr vs GT: `0.9559876947`

Implications:
- the latest profiling / projection-reuse / fused-adjoint tranche is innocent
- grouped local batching is also innocent

### Current active isolate

Run in progress:
- output: `_agent_scratch/local_profile_full5k_it006_single_image_no_bucket_v1`
- command toggles:
  - `local_search_profile=off`
  - `local_search_projection_reuse=off`
  - `local_search_fused_windowed_adjoint=off`
  - `local_search_force_single_image_groups=on`
  - `local_search_bucket_rotation_blocks=off`

Purpose:
- isolate whether the `f93cb15e` local-rotation bucketing / padding path changed posterior sharpness

### Remaining suspect set if `single_image_no_bucket` still lands at `0.8279855920`

At that point the main remaining suspects are the `sampling.py` local-grid convention changes introduced in `f93cb15e`:
- viewing direction row vs column
- RELION NEST pixel ordering in `rotation_indices_to_matrices`
- custom local-grid metadata / psi extraction path

### Benchmarks to preserve

Use these as the fixed regression checkpoints for future changes:

1. Old-good hp4 matched stage
- `_agent_scratch/multi_iter_full_hp4_v4_childfix_gt`

2. Current default hp4 matched stage
- `_agent_scratch/local_profile_full5k_it006_v1`

3. Current control with latest toggles disabled
- `_agent_scratch/local_profile_full5k_it006_control_offoffoff_v1`

4. Current exact per-image control
- `_agent_scratch/local_profile_full5k_it006_single_image_groups_on_v1`

For each, record:
- RECOVAR `ave_Pmax`
- RELION `ave_Pmax`
- merged FSC vs GT @ `0.5`
- half-map FSC @ `0.143`
- merged corr vs GT

### Code added in this checkpoint

Debug flags wired through `refine_single_volume` and `run_multi_iter_parity.py`:
- `local_search_return_profile`
- `local_search_reuse_pass1_projections`
- `local_search_fused_windowed_adjoint`
- `local_search_force_single_image_groups`
- `local_search_bucket_rotation_blocks`

These are for regression isolation only and are not yet claimed as final interface.

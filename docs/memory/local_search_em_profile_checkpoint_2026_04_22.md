# Local Search EM Profile Checkpoint: 2026-04-22

## Scope

This checkpoint lands the first staged speed-work tranche after the
grouped-local exact path critique:

- phase-split timing inside `run_em_v2`
- pass-1 to pass-2 projection reuse inside the current grouped exact path
- fused windowed adjoint path that avoids the explicit scatter-back timing
  bucket in the local-search replay path
- benchmark bootstrap hardening in
  `scripts/run_relion_parity_benchmark_slurm.sh`

The goal of this checkpoint is not to claim RELION speed parity. It is
to make the grouped exact path measurable enough to decide the next
structural optimization step.

## Code Changes

- `recovar/em/dense_single_volume/types.py`
  - added `EMProfileStats`
- `recovar/em/dense_single_volume/engine_v2.py`
  - added phase timing and work-shape reporting
  - added `reuse_pass1_projections`
  - added `fused_windowed_adjoint`
- `recovar/em/dense_single_volume/refine.py`
  - grouped local-search path now requests EM profiles
  - saves `itXXX_halfY_local_profile.npz` when intermediates are enabled
  - records union/padding waste metrics per grouped local-search iteration
- `scripts/run_relion_parity_benchmark_slurm.sh`
  - switched bootstrap to `pixi run install-recovar`
- `tests/unit/test_half_spectrum_em.py`
  - regression test proving projection reuse preserves outputs while
    reducing projection-block calls

## Smoke Run

Command:

`CUDA_VISIBLE_DEVICES=1 pixi run python scripts/run_multi_iter_parity.py --relion_dir /scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/relion_ref_os0 --data_star /scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/particles.star --iter 5 --max_iter 1 --skip_final_iteration --max_particles 64 --output_dir _agent_scratch/local_profile_smoke_64_v1`

Wall time:

- `WALL 52.54`

Saved profile artifacts:

- `_agent_scratch/local_profile_smoke_64_v1/intermediates/it000_half1_local_profile.npz`
- `_agent_scratch/local_profile_smoke_64_v1/intermediates/it000_half2_local_profile.npz`

## Key Profile Signals From The 64-Particle hp4 Smoke Run

Half 1:

- `n_chunks = 16`
- `em_time_s = 14.50`
- `em_pass1_projection_s = 0.226`
- `em_pass2_projection_s = 0.0`
- `em_window_scatter_s = 0.0`
- `em_adjoint_y_s = 2.232`
- `em_adjoint_ctf_s = 0.755`
- `em_reused_pass1_projections = 16`
- `union_waste_fraction = 0.4977`
- `padded_waste_fraction = 0.5421`

Half 2:

- `n_chunks = 16`
- `em_time_s = 5.44`
- `em_pass1_projection_s = 0.053`
- `em_pass2_projection_s = 0.0`
- `em_window_scatter_s = 0.0`
- `em_adjoint_y_s = 1.843`
- `em_adjoint_ctf_s = 0.387`
- `em_reused_pass1_projections = 16`
- `union_waste_fraction = 0.5000`
- `padded_waste_fraction = 0.5417`

Interpretation:

- projection reuse is active in every grouped local-search chunk
- the fused windowed adjoint path removes the explicit scatter timing
  bucket from this path
- grouped local search still wastes roughly half of the dense
  image-rotation grid through union and padding effects even on this
  small replay

## Validation Run For This Checkpoint

- `pixi run python -m py_compile recovar/em/dense_single_volume/types.py recovar/em/dense_single_volume/engine_v2.py recovar/em/dense_single_volume/refine.py tests/unit/test_half_spectrum_em.py`
- `pixi run pytest tests/unit/test_half_spectrum_em.py -k "profile_and_projection_reuse or multiple_rotation_blocks or multiple_image_batches" -v`
- `pixi run pytest tests/unit/test_refine_relion_mode.py -k "local_search" -v`
- `bash -n scripts/run_relion_parity_benchmark_slurm.sh`

## Long-Run Tracking

The resubmitted long jobs remain live and are tracked separately in
`docs/memory/long_run_resubmissions_2026_04_22.md`.

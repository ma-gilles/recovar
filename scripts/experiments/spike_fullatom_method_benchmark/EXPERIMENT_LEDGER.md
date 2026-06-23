# Experiment Ledger

## 2026-06-07: Uniform Noise=3 State-50 Available Movie And Halfmap FSC

Scripts added:

- `scripts/experiments/spike_fullatom_method_benchmark/refresh_uniform_noise3_state50_available_movies.py`
- `scripts/experiments/spike_fullatom_method_benchmark/compare_uniform_noise3_state50_halfmap_vs_mapmodel.py`
- `scripts/experiments/spike_fullatom_method_benchmark/UNIFORM_NOISE3_STATE50_MOVIE_AND_HALFMAP_FSC.md`

Dataset:

- `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531`
- `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise3_b80_20260531`

Completed outputs:

- Movie output root:
  `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/state50_uniform_noise3_available_movies_20260607`
- Non-moving movie output root:
  `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/state50_uniform_noise3_available_notmoving_movies_20260607`
- Halfmap comparison output root:
  `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/state50_halfmap_vs_mapmodel_fsc_20260607`
- Non-moving halfmap comparison output root:
  `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/state50_halfmap_vs_mapmodel_notmoving_fsc_20260607`

Notes:

- RECOVAR and 3DFlex include `10k, 30k, 100k, 300k, 1M, 3M`.
- cryoDRGN includes `10k, 30k, 100k, 300k, 1M`; no decoded 3M state-50 output was present.
- The first movie run failed because `module load chimerax` has no default; the script now uses `chimerax/1.9`.
- The first halfmap comparison was stopped because the brute-force 3DFlex all-state search was too slow. The script now precomputes the shared GT trajectory FFTs once.

Commands:

```bash
.pixi/envs/default/bin/python -m py_compile \
  scripts/experiments/spike_fullatom_method_benchmark/refresh_uniform_noise3_state50_available_movies.py \
  scripts/experiments/spike_fullatom_method_benchmark/compare_uniform_noise3_state50_halfmap_vs_mapmodel.py

.pixi/envs/default/bin/python \
  scripts/experiments/spike_fullatom_method_benchmark/refresh_uniform_noise3_state50_available_movies.py \
  --output-dir /scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/state50_uniform_noise3_available_movies_20260607

.pixi/envs/default/bin/python \
  scripts/experiments/spike_fullatom_method_benchmark/refresh_uniform_noise3_state50_available_movies.py \
  --skip-render \
  --output-dir /scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/state50_uniform_noise3_available_movies_20260607

.pixi/envs/default/bin/python \
  scripts/experiments/spike_fullatom_method_benchmark/compare_uniform_noise3_state50_halfmap_vs_mapmodel.py \
  --output-dir /scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/state50_halfmap_vs_mapmodel_fsc_20260607

.pixi/envs/default/bin/python \
  scripts/experiments/spike_fullatom_method_benchmark/refresh_uniform_noise3_state50_available_movies.py \
  --mask-mode notmoving \
  --output-dir /scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/state50_uniform_noise3_available_notmoving_movies_20260607

.pixi/envs/default/bin/python \
  scripts/experiments/spike_fullatom_method_benchmark/compare_uniform_noise3_state50_halfmap_vs_mapmodel.py \
  --mask-mode notmoving \
  --output-dir /scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/state50_halfmap_vs_mapmodel_notmoving_fsc_20260607
```

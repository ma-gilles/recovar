# Uniform Noise=3 State-50 Movie And Halfmap FSC

Date: 2026-06-07

This note records the reproducible state-50 uniform noise=3 refresh that combines
volume renderings with broad-mask FSC curves, and the halfmap-vs-map-to-model FSC
comparison.

## Dataset

- Sweep root:
  `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531`
- Source GT/data root:
  `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise3_b80_20260531`
- Noise level: `3`
- B factor: `80`
- State: `50`
- Dataset sizes: `10k, 30k, 100k, 300k, 1M, 3M`

## Masks And View

- FSC/render mask:
  `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_direct_volume_shell_metrics_20260523/full_gt_vols_plus_masks_20260524/masks/broad_mask.mrc`
- ChimeraX view:
  `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/chimerax_state50_method_progression_moving_mask_zoomed_view_20260601/zoomed_moving_view_extracted.json`
- Rendering contour level: `0.013`
- GT overlay opacity: `0.3`
- Fixed method colors:
  - RECOVAR: `#1b9e77`
  - cryoDRGN: `#d95f02`
  - 3DFlex: `#7570b3`
  - GT: `#8f8f8f`

## Volume Sources

RECOVAR uses filtered `compute_state` state-50 maps:

`/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/recovar/nXXXXXXXX/compute_state_zdim4_noreg_focus/state0050/state000.mrc`

cryoDRGN uses decoded mean-label maps:

`/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/nXXXXXXXX/evaluation/cryodrgn/zdim1/decoded_volumes/labels_mean_z_epoch019/`

The 10k/30k/100k/300k cryoDRGN state-50 outputs are `gt_label_002.mrc` because
those decode jobs only emitted the three-state grid. The 1M output is
`gt_label_050.mrc`. No decoded 3M cryoDRGN state-50 output exists in this sweep
tree, so the 3M frame omits cryoDRGN rather than drawing an unavailable panel.

3DFlex uses generated mean-latent state-50 maps:

- `10k`: `/projects/CRYOEM/singerlab/mg6942/CS-testres/J504/J504_series_000/J504_series_000_frame_002.mrc`
- `30k`: `/projects/CRYOEM/singerlab/mg6942/CS-testres/J506/J506_series_000/J506_series_000_frame_002.mrc`
- `100k`: `/projects/CRYOEM/singerlab/mg6942/CS-testres/J518/J518_series_000/J518_series_000_frame_002.mrc`
- `300k`: `/projects/CRYOEM/singerlab/mg6942/CS-testres/J526/J526_series_000/J526_series_000_frame_002.mrc`
- `1M`: `/projects/CRYOEM/singerlab/mg6942/CS-testres/J528/J528_series_000/J528_series_000_frame_002.mrc`
- `3M`: `/projects/CRYOEM/singerlab/mg6942/CS-testres/J544/J544_series_000/J544_series_000_frame_002.mrc`

## Movie Refresh Command

```bash
AGENT_ID="codex_uniform_noise3_compose_$(date +%Y%m%d_%H%M%S)_$RANDOM"
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/$AGENT_ID"
export PIXI_HOME="/scratch/gpfs/GILLES/mg6942/pixi_home/$AGENT_ID"
export RATTLER_CACHE_DIR="/scratch/gpfs/GILLES/mg6942/rattler_cache/$AGENT_ID"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR"

.pixi/envs/default/bin/python \
  scripts/experiments/spike_fullatom_method_benchmark/refresh_uniform_noise3_state50_available_movies.py \
  --output-dir /scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/state50_uniform_noise3_available_movies_20260607
```

The first attempt failed because `module load chimerax` has no default version
on Della. The script now defaults to `chimerax/1.9`.

Primary outputs:

- GIF:
  `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/state50_uniform_noise3_available_movies_20260607/frames/state50_uniform_noise3_available_broad_volume_plus_fsc.gif`
- MP4:
  `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/state50_uniform_noise3_available_movies_20260607/frames/state50_uniform_noise3_available_broad_volume_plus_fsc.mp4`
- Final frame:
  `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/state50_uniform_noise3_available_movies_20260607/state50_uniform_noise3_available_broad_volume_plus_fsc_final.png`
- FSC summary:
  `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/state50_uniform_noise3_available_movies_20260607/state50_uniform_noise3_available_broad_fsc_summary.csv`
- All-method same-axes FSC plot:
  `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/state50_uniform_noise3_available_movies_20260607/state50_uniform_noise3_available_broad_fsc_all_methods_same_axes.png`
- Audit:
  `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/state50_uniform_noise3_available_movies_20260607/state50_uniform_noise3_available_broad_movie_audit.json`

Non-moving-mask version:

```bash
.pixi/envs/default/bin/python \
  scripts/experiments/spike_fullatom_method_benchmark/refresh_uniform_noise3_state50_available_movies.py \
  --mask-mode notmoving \
  --output-dir /scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/state50_uniform_noise3_available_notmoving_movies_20260607
```

Primary non-moving outputs:

- GIF:
  `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/state50_uniform_noise3_available_notmoving_movies_20260607/frames/state50_uniform_noise3_available_notmoving_volume_plus_fsc.gif`
- MP4:
  `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/state50_uniform_noise3_available_notmoving_movies_20260607/frames/state50_uniform_noise3_available_notmoving_volume_plus_fsc.mp4`
- All-method same-axes FSC plot:
  `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/state50_uniform_noise3_available_notmoving_movies_20260607/state50_uniform_noise3_available_notmoving_fsc_all_methods_same_axes.png`
- FSC summary:
  `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/state50_uniform_noise3_available_notmoving_movies_20260607/state50_uniform_noise3_available_notmoving_fsc_summary.csv`

## Halfmap Comparison

For a halfmap FSC `h`, the script records both Rosenthal-Henderson conversion
conventions:

- power/squared convention: `2*h/(1+h)`
- correlation convention: `sqrt(max(0, 2*h/(1+h)))`

RECOVAR:

- halfmap FSC: `state000_half1_unfil.mrc` vs `state000_half2_unfil.mrc`
- map-to-model FSC: `state000.mrc` and `state000_unfil.mrc` vs GT state 50

3DFlex:

- halfmap FSC: `Jxxx_flex_map_half_A.mrc` vs `Jxxx_flex_map_half_B.mrc`
- map-to-model FSC: the highres full flex map `Jxxx_flex_map.mrc` compared
  against all 100 GT states, selecting the best state by broad-mask FSC AUC
- generated state-50 mean-latent map is also scored against GT state 50 for
  context

The 3DFlex best-GT search precomputes the GT trajectory FFTs once from the 10k
source run because the GT state trajectory is shared across the size sweep.

Command:

```bash
AGENT_ID="codex_halfmap_noise3_fast_$(date +%Y%m%d_%H%M%S)_$RANDOM"
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/$AGENT_ID"
export PIXI_HOME="/scratch/gpfs/GILLES/mg6942/pixi_home/$AGENT_ID"
export RATTLER_CACHE_DIR="/scratch/gpfs/GILLES/mg6942/rattler_cache/$AGENT_ID"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR"

.pixi/envs/default/bin/python \
  scripts/experiments/spike_fullatom_method_benchmark/compare_uniform_noise3_state50_halfmap_vs_mapmodel.py \
  --output-dir /scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/state50_halfmap_vs_mapmodel_fsc_20260607
```

Non-moving halfmap command:

```bash
.pixi/envs/default/bin/python \
  scripts/experiments/spike_fullatom_method_benchmark/compare_uniform_noise3_state50_halfmap_vs_mapmodel.py \
  --mask-mode notmoving \
  --output-dir /scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/state50_halfmap_vs_mapmodel_notmoving_fsc_20260607
```

Primary outputs:

- Summary CSV:
  `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/state50_halfmap_vs_mapmodel_fsc_20260607/uniform_noise3_state50_halfmap_vs_mapmodel_summary.csv`
- RECOVAR curves:
  `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/state50_halfmap_vs_mapmodel_fsc_20260607/uniform_noise3_state50_recovar_halfmap_vs_mapmodel_curves.png`
- 3DFlex curves:
  `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/state50_halfmap_vs_mapmodel_fsc_20260607/uniform_noise3_state50_3dflex_halfmap_vs_mapmodel_curves.png`
- Resolution summary:
  `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/state50_halfmap_vs_mapmodel_fsc_20260607/uniform_noise3_state50_halfmap_vs_mapmodel_resolution_summary.png`
- Audit:
  `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/state50_halfmap_vs_mapmodel_fsc_20260607/uniform_noise3_state50_halfmap_vs_mapmodel_audit.json`
- Non-moving halfmap root:
  `/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531/state50_halfmap_vs_mapmodel_notmoving_fsc_20260607`

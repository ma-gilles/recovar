# ChimeraX Rendering Pipeline

This renders RECOVAR/cryoDRGN/3DFlex spike maps reproducibly from a CSV
manifest.  The manifest records every map path, method, state, image count,
noise level, color, and contour level.

## Build A Manifest

```bash
python scripts/experiments/spike_fullatom_method_benchmark/build_chimerax_render_manifest.py \
  --output /scratch/gpfs/CRYOEM/gilleslab/tmp/spike_method_renderings_20260531/manifest_current.csv \
  --states 0,25,50 \
  --methods recovar,cryodrgn,3dflex,ground_truth \
  --level-percentile 70 \
  --level-sigma 1.5
```

The contour level is computed per map as:

```text
max(positive_voxel_percentile_70, mean + 1.5 * std)
```

This avoids tiny positive background values producing nearly transparent
surfaces.  The chosen level, mean, and std are saved in the manifest.

## Render With ChimeraX

Use `chimerax/1.9` on Della.  `chimerax/1.11` currently fails at startup on
this cluster because its bundled Python wants a newer OpenSSL than the runtime
library provides.

Small test:

```bash
module purge
module load chimerax/1.9
chimerax --nogui --offscreen --script \
  "scripts/experiments/spike_fullatom_method_benchmark/render_chimerax_manifest.py \
   --manifest /scratch/gpfs/CRYOEM/gilleslab/tmp/spike_method_renderings_20260531/manifest_current.csv \
   --output-dir /scratch/gpfs/CRYOEM/gilleslab/tmp/spike_method_renderings_20260531/renders \
   --noise-levels 1 \
   --methods recovar,ground_truth \
   --states 50 \
   --n-images 10000,30000 \
   --width 1600 --height 1200"
```

Slurm full-current render:

```bash
sbatch scripts/experiments/spike_fullatom_method_benchmark/submit_chimerax_render_manifest.sbatch \
  /scratch/gpfs/CRYOEM/gilleslab/tmp/spike_method_renderings_20260531/manifest_current.csv \
  /scratch/gpfs/CRYOEM/gilleslab/tmp/spike_method_renderings_20260531/renders_current \
  --width 1600 --height 1200
```

## Notes

- Rendering is manifest-driven; rerun the manifest builder after more sweep
  jobs finish.
- Filenames include noise level, collection/root, method, number of images,
  state, and role, so different sweeps do not overwrite each other.
- The default view uses a fixed yaw/pitch and perspective camera.  ChimeraX
  1.9 offscreen crashes on `camera ortho` in this environment; pass
  `--orthographic` only if using an install where that works.
- The PNGs can be assembled into movies using sorted filenames or a simple
  frame manifest grouped by `(noise_level, method, state)`.

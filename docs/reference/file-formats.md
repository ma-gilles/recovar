# File Formats

## Input formats

### RELION STAR (`.star`)

RELION 3.0+ STAR files with optics groups. Must contain:

- **Optics table**: `_rlnImagePixelSize`, `_rlnImageSize`, `_rlnVoltage`, `_rlnSphericalAberration`, `_rlnAmplitudeContrast`
- **Particles table**: `_rlnImageName` (format: `index@path.mrcs`)
- **Poses** (optional for auto-extraction): `_rlnAngleRot`, `_rlnAngleTilt`, `_rlnAnglePsi`, `_rlnOriginXAngst`, `_rlnOriginYAngst`
- **CTF** (optional for auto-extraction): `_rlnDefocusU`, `_rlnDefocusV`, `_rlnDefocusAngle`

### cryoSPARC (`.cs`)

NumPy structured array (`.npy` format with `.cs` extension). Must contain:

- **Images**: `blob/path`, `blob/idx`, `blob/shape`, `blob/psize_A`
- **Poses** (for auto-extraction): `alignments3D/pose` (Rodrigues vectors), `alignments3D/shift`
- **CTF** (for auto-extraction): `ctf/df1_A`, `ctf/df2_A`, `ctf/df_angle_rad`, `ctf/accel_kv`, `ctf/cs_mm`, `ctf/amp_contrast`

### MRC stack (`.mrcs`)

Standard MRC2014 format image stack. Requires separate `--poses` and `--ctf` pickle files.

### Pickle files (`.pkl`)

Legacy format for poses and CTF parameters:

- **Poses**: Tuple of `(rotations, translations)` where rotations is `(N, 3, 3)` and translations is `(N, 2)` in fractional units
- **CTF**: Array of shape `(N, 9)` with columns `[D, Apix, DFU, DFV, DFANG, VOLT, CS, W, PHASE_SHIFT]`

## Output formats

### Pipeline output

```
output/
  job.json                      # Job metadata (version, timing, parameters)
  command.txt                   # Command line used
  run.log                       # Full log
  README.txt                    # Human-readable output summary
  downsampled/                  # Cached downsampled data (if --downsample)
    particles.128.mrcs
    particles.128.star
  model/                        # Internal model
    params.pkl
    zdim_4/                     # Per-zdim embeddings
      latent_coords.npy
    zdim_10/
      latent_coords.npy
  output/
    volumes/
      mean.mrc                  # Mean reconstruction
      mean_filt.mrc             # Filtered mean
      mean_half1_unfil.mrc      # Half-map 1
      mean_half2_unfil.mrc      # Half-map 2
      mask.mrc                  # Mask used
      dilated_mask.mrc          # Dilated mask
    plots/                      # Diagnostic plots (eigenvalues, FSC, etc.)
```

### Analysis output

```
output/analysis_10/
  job.json                      # Job metadata
  command.txt                   # Command used
  run.log                       # Full log
  README.txt                    # Output summary
  plots/                        # All plots
    contrast_histogram.png
    PCA/                        # PC scatter plots with k-means
    umap/                       # UMAP embeddings
    density/                    # Density plots (if provided)
    density_sliced/             # Sliced density plots
  data/                         # Non-volume data
    kmeans_result.pkl           # K-means labels and centers
    trajectory_endpoints.pkl    # Trajectory endpoint indices
  kmeans/                       # K-means cluster center volumes
    center000.mrc               # Volume at cluster center 0
    center001.mrc               # Volume at cluster center 1
    center000_half1_unfil.mrc   # Half-map for FSC
    centers.txt                 # Center coordinates (np.loadtxt)
    diagnostics/center000/      # Per-volume diagnostics
  traj000/                      # Trajectory 0 volumes
    state000.mrc
    state001.mrc
    diagnostics/state000/       # Per-volume diagnostics
```

### Density output

```
density/
  job.json                      # Job metadata
  command.txt                   # Command used
  run.log                       # Full log
  plots/                        # Density plots
    all_densities.png           # Visualization of all densities
    Lcurve.png                  # L-curve for alpha selection
  data/                         # Density data
    deconv_density_0.pkl        # Density at alpha[0]
    deconv_density_1.pkl        # Density at alpha[1]
    ...
    deconv_density_knee.pkl     # Optimal density (L-curve knee)
```

### Volume files (`.mrc`)

All output volumes are in MRC2014 format with correct voxel size in the header. Open with:

- **UCSF ChimeraX** (recommended)
- **UCSF Chimera**
- **PyMOL**
- **EMAN2**

### Index files (`.pkl`)

Particle index files are Python pickle files containing a 1D NumPy integer array. Load with:

```python
import pickle
with open("indices.pkl", "rb") as f:
    indices = pickle.load(f)
```

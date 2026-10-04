# recovar.simulation

Synthetic dataset generation and forward-model image simulation for testing
and validation.

## Explicit simulation configuration

For a dataset-independent simulation, provide a JSON configuration. There is
no implicit volume, experimental CTF table, noise template, physical pixel size,
signal calibration, dose or tilt interval. Missing scientific settings fail
before image generation; input errors never fall back to a benchmark dataset.

```bash
# Validate inputs and print all resolved settings without writing outputs.
recovar make_test_dataset ./new_simulation \
    --simulation-config /path/to/simulation.json \
    --output-format relion5 --dry-run

# Generate using the same explicit configuration.
recovar make_test_dataset ./new_simulation \
    --simulation-config /path/to/simulation.json \
    --output-format relion5
```

Use `--output-format recovar` with the **same config** to change only the export
format, not the simulated noise, signal or poses. The CLI writes beneath
`<output_dir>/test_dataset/`; the Python API below uses the output folder directly.
`simulation_config.json` records the resolved scientific and computational
settings. The new path refuses to overwrite existing outputs.

Start from [the unfilled JSON template](../../examples/simulation_config.template.json).
Replace the `null` values and volume path with your choices; the template is
intentionally not a ready-to-run scientific preset. Paths in the JSON are
relative to that JSON file, not the shell's working directory. Unknown keys
are rejected to catch misspelled settings. Do not combine `--simulation-config`
with legacy flags such as `--noise-level`, `--n-images`, `--seed` or `--tilt-series`:
put those settings in the JSON instead.

### Scientific inputs

| JSON key | Meaning |
|---|---|
| `volumes` | Explicit list of ground-truth MRC volumes. |
| `grid_size` | Even output image size, no larger than the input volume grid. |
| `n_particles`, `n_tilts` | Physical particle count and views per particle; total images is their product. |
| `angle_per_tilt_deg` | Dose-symmetric acquisition angles: 0, +step, −step, +2step, −2step, … |
| `dose_per_tilt` | e⁻/Å² per image; zero explicitly disables dose attenuation. |
| `signal_scale` | Multiply source volume Fourier amplitudes by this factor, with no automatic SNR calibration. |
| `seed` | Explicit random seed for poses, weighted class draws, CTF sampling and Gaussian noise. |
| `volume_weights` | Optional probabilities for sampled particle assignments. |
| `volume_assignments_file` | Optional exact, ordered particle-to-volume indices; mutually exclusive with `volume_weights`. |
| `poses` | `{"mode":"uniform"}` or `{"mode":"file","file":"rotations.npy"}`. |
| `noise` | Explicit `none`, `white`, `radial` or `radial_per_tilt` model, as described below. |
| `ctf` | Explicit constant parameters or a user-supplied sampling table. |
| `tilt_amplitude_weighting` | Explicit `none` or `cosine`; separate from dose attenuation. |

Physical pixel size is inferred from the **input MRC headers**, then adjusted
for Fourier downsampling. Volumes must have matching even cubic dimensions and
positive isotropic voxel sizes. File poses must be an `(n_particles, 3, 3)`
NumPy array of RECOVAR base rotation matrices. Standard generated geometry uses zero shifts,
linear interpolation, no contrast jitter and no outlier particles; these fixed
model choices are printed in the resolved configuration. It does not reuse
another experiment's microscope parameters or calibration factor.

`volume_weights` is optional: the generic default is equal probability per
input volume, not fixed proportions from a particular dataset. Supplied weights
must sum to one. Assignments are sampled per particle, so counts need not exactly
equal their expectations.

For an exact mixture and particle order, instead provide
`"volume_assignments_file":"assignments.npy"`. The file must be a numeric NumPy
`.npy` array with integer dtype and shape `(n_particles,)`. Each value is a
zero-based index into `volumes`; every tilt of that particle inherits the same
index. Negative, out-of-range, floating-point and image-level assignments are
rejected. This option and `volume_weights` are mutually exclusive. The resolved
configuration records the absolute source path and per-volume counts, while
`simulation_info.pkl` records the exact particle- and image-level assignments.

Computational defaults are `batch_size=32` and `noise_rng_batch_size=128`,
recorded explicitly in the resolved configuration.

### Noise

- `{"model":"none"}`: no added noise.
- `{"model":"white","std":...}`: Gaussian noise with the requested real-space
  image standard deviation, in the same intensity units as the scaled signal.
- `{"model":"radial","spectrum_file":"noise.npy","variance_scale":...}`:
  Gaussian noise shaped by your supplied radial Fourier power spectrum. The file
  must contain `grid_size/2 - 1` finite, nonnegative shell powers in the low-level
  RECOVAR simulator's FFT convention. `variance_scale` multiplies powers, not
  amplitudes: doubling noise amplitude requires multiplying power by four. This
  model uses one supplied spectrum for all images.
- `{"model":"radial_per_tilt","spectrum_file":"noise_by_tilt.npy","variance_scale":...}`:
  use a separate radial spectrum at each acquisition rank. The file must have
  shape `(n_tilts, grid_size/2 - 1)` with finite, nonnegative Fourier powers.
  Row 0 applies to each particle's first acquired image, row 1 to its second,
  and so on—not to the later angle-sorted RELION5 frame order. Each particle
  gets independent noise, with the same spectrum for a given acquisition rank.
  The seed is split by acquisition rank using NumPy `SeedSequence.spawn`;
  the resolved configuration and `simulation_info.pkl` record those seeds.

There is no automatic `noise_10076.pkl`, `/50000` factor, target-SNR fit or
per-species rescaling in this path. Equal noise does not imply equal SNR for
structures with different signal powers.

### CTF

For `ctf.mode="constant"`, explicitly supply all of:

- `defocus_u`, `defocus_v` in Å; `defocus_angle_deg` in degrees.
- `voltage_kv` in kV; `cs_mm` in mm; `amplitude_contrast` in [0, 1].
- `phase_shift_deg` in degrees; `bfactor` in Å².

Alternatively use `{"mode":"sample","file":"ctf.npy"}`. The file must be an
`(N, 9)` array with columns `DFU, DFV, DFANG, VOLT, CS, W, PHASE_SHIFT, BFACTOR,
CONTRAST` in those units. Rows are drawn independently per tilt image. Microscope
voltage, Cs and amplitude contrast must be constant across the table. This is
not a legacy `ctf.pkl` table containing box size and pixel size columns.

The configured path directly constructs the forward-model dose columns using
midpoint cumulative exposure `(acquisition_index + 0.5) * dose_per_tilt`; it
does not use the legacy generator's appended-dose code. Nonzero-dose simulation
supports 200 or 300 kV; native RELION5 export currently requires 300 kV for
nonzero dose. RECOVAR's critical-exposure hard cutoff is absent from native
RELION's exponential dose model; the native export manifest records this
limitation. Do not assume an exact high-frequency forward-model match.

Configured RECOVAR output rejects nonzero physical B-factors because its
existing grouped STAR reader cannot preserve that envelope. Use `bfactor=0`
or RELION5 output; the old reader is not modified. CTF-premultiplied simulations
are not supported by this new configured path.

### Ordered per-image metadata replay

To reuse explicit acquisition-ordered metadata instead of drawing poses or CTFs,
use the following alternatives (RECOVAR output only):

```json
{
  "poses": {
    "mode": "per_image_file",
    "rotations_file": "rotations_by_particle.npy",
    "translations_file": "translations_by_particle.npy"
  },
  "angle_per_tilt_deg": null,
  "ctf": {"mode": "per_image_file", "file": "ctf_by_particle.npy"},
  "dose_per_tilt": null,
  "tilt_amplitude_weighting": "none"
}
```

This is a fragment to insert into the complete configuration, not a complete
simulation preset. Rotation arrays have shape `(n_particles, n_tilts, 3, 3)`;
translation arrays have shape `(n_particles, n_tilts, 2)` and are in image
pixels. Both use RECOVAR's internal conventions. CTF arrays have shape
`(n_particles, n_tilts, 11)` with ordered columns `DFU, DFV, DFANG, VOLT, CS, W,
PHASE_SHIFT, BFACTOR, CONTRAST, DOSE, TILT_ANGLE`. All fields are used in supplied
order, without random sampling or replacement by a midpoint-dose schedule.
Doses must be nonnegative and nondecreasing within each particle. The explicit
`null` settings prevent competing sources of geometry or exposure information.

Because unchanged RECOVAR STAR readers cannot preserve a physical B-factor or
a separate CTF tilt-angle amplitude term, both `BFACTOR` and `TILT_ANGLE` must be
zero for this mode. Contrast factors already present in the file are preserved;
there is no extra cosine weighting. Native RELION5 export currently requires the
standard synthetic geometry and does not accept per-image replay inputs.
The original 11-column CTF arrays are retained in `simulation_info.pkl`, and
their B-factor values are not replaced by WARP dose bookkeeping in the STAR.

Noise is still freshly generated from the explicitly selected model and seed.
Replaying signal metadata does not promise bitwise reproduction of an older
run's noise draws. Particle class assignments are newly sampled unless an exact
`volume_assignments_file` is supplied.

### Outputs and Python API

Native RELION5 output includes `particles.star`, `tomograms.star`,
`optimisation_set.star`, `tilt_series/*.star`, per-particle float32 stacks,
`simulation_info.pkl`, frame/particle index mappings, and `relion5_export.json`.
Use the optimisation set as RELION5's tomography refinement/classification input.
These are already extracted particle images, not raw movies or full-field
tilt-series data. Native paths are absolute; update them if moving the export.

```python
from recovar.simulation.configured_simulation import (
    load_simulation_config,
    run_configured_simulation,
)

config = load_simulation_config("simulation.json")
images, info = run_configured_simulation("./output", config, output_format="relion5")
```

Returned arrays retain RECOVAR's sign and acquisition order. Native export
negates images and reorders frames together with their metadata; it does not
resimulate noise. Both configured output formats write float32 stacks.

## Legacy RECOVAR compatibility

Without `--simulation-config`, `make_test_dataset` retains its original RECOVAR
test defaults and generation/writing code. No existing experiment is modified.
`--output-format relion5` without an explicit config now errors instead of
silently inheriting those benchmark defaults. The legacy
`simulator.generate_synthetic_dataset` likewise directs native-output callers
to `run_configured_simulation`.

Configured `simulation_info.pkl` records explicit input paths and settings, not
the legacy volume-prefix schema. It is not currently supported by
`synthetic_dataset.load_heterogeneous_reconstruction` or the legacy
`run_test_all_metrics` ground-truth loader.

## simulator

::: recovar.simulation.simulator
    options:
      members_order: source

## synthetic_dataset

::: recovar.simulation.synthetic_dataset
    options:
      members_order: source

## simulate_scattering_potential

::: recovar.simulation.simulate_scattering_potential
    options:
      members_order: source

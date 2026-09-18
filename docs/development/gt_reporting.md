# Rigid ground-truth reporting

The optional rigid fitter helps compare maps whose coordinate frames differ by
rotation, translation or handedness. It is a CPU reporting tool; it does not
change EM execution. Existing `--gt_align` behavior and output schemas remain
unchanged unless `--gt_align_rigid` is selected.

For a trajectory or paired comparison, fit one explicitly chosen reference and
apply that same transform to every map. Independently fitting every checkpoint
can conceal frame changes. Select the fit reference and controls before examining
quality results; preserve the resulting transform with the run evidence.

## Fit once and reuse

Fit the input labeled `native` to GT and apply its transform to both inputs:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu pixi run python scripts/evaluate_ab_initio_gt.py \
  --volume native.mrc --label native \
  --volume candidate.mrc --label candidate \
  --gt_volume gt.mrc --volume_frame relion --gt_frame recovar \
  --gt_align --gt_align_rigid --gt_align_fit_label native \
  --gt_align_transform_output frame.json --output_json metrics.json
```

Apply the frozen transform to another matching map without fitting or building
a rotation grid:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu pixi run python scripts/evaluate_ab_initio_gt.py \
  --volume checkpoint.mrc --label checkpoint \
  --gt_volume gt.mrc --volume_frame relion --gt_frame recovar \
  --gt_align --gt_align_rigid --gt_align_transform_json frame.json \
  --output_json checkpoint_metrics.json --output_npz checkpoint_curves.npz
```

With neither common-transform option, rigid mode fits each input independently
and labels that behavior in the receipt. `--gt_align_allow_sign` is incompatible:
rigid mode fixes contrast sign at +1. Legacy discrete refinement orders and sigma
are inactive in this mode. Saved-transform application uses the saved controls,
not newly supplied fitting options.

## Geometry and evidence

The forward transform in loaded RECOVAR array coordinates is
`y = c + R M (x − c) + t`, with `c = (shape − 1)/2`. `R` is a proper rotation;
`M` optionally reflects array axis 0. Translation `t` uses full-resolution voxels
in array-axis order 0,1,2; it is not an implicit Cartesian XYZ vector. The caller
also reports translations in Angstrom. RELION-frame loading uses the existing
RECOVAR conversion before fitting or application.

`RigidVolumeTransform` copies geometry into immutable tuples. Its versioned JSON
records rotation, translation, handedness, interpolation, cubic shape, isotropic
voxel size and the hash of loaded GT values. Application rejects incompatible
shape, voxel size or GT identity. Its identity covers the transform contents;
the fit receipt and reference metadata are separate reporting evidence.

The fitter lowpasses on the original grid, uses linear FFT correlation for
translation seeds, then optimizes rotation/translation on fixed reference sample
coordinates with centered normalization recomputed for each trial. Final maps
are resampled once from the original input. New fitting calculations use float64
on CPU; this does not change production EM precision.

The receipt exposes the coarse grid identity, fitting controls, optimizer status,
seed fallback and boundary flags. A finite-grid/local optimizer can miss the
correct transform even when it reports success. Clipping, interpolation and weak
signal also limit registration. Fit correlation is an optimization diagnostic;
FSC/FSC-AUC and the established scientific gates determine map quality. This tool
adds no quality threshold and does not make a run scientifically accepted.

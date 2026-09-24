# Atomic-model volume transform in the simulator

A Coulomb potential computed from an atomic model in vacuum lacks the
contribution of the displaced solvent, so its low-frequency contrast is too
high compared with a molecule embedded in vitreous ice. A model without
B-factors also lacks the high-frequency falloff of real data. The simulator can
optionally apply the solvent-contrast approximation of Henderson & McMullan
(2013), [doi:10.1093/jmicro/dfs094](https://doi.org/10.1093/jmicro/dfs094),
followed by a B-factor. Together they are the **EM-development preset**, a
better approximation of real data for EM/VDAM development.

This option is for input volumes generated from atomic models without solvent
correction or B-factors. Do not use it for experimental reconstructions or
volumes that are already solvent-corrected. It is off by default in recovar's
simulator and is never inferred from the input files. It is an approximate
model, not a universal empirical correction.

## Model

For each input volume $V$ the simulator projects the effective volume $T(V)$:

$$
\mathcal{F}[T(V)](\mathbf q) = H(\mathbf q)\,
\exp\!\left(-\frac{B_\text{atomic}\,|\mathbf q|^2}{4}\right)\mathcal{F}[V](\mathbf q),
\qquad
H(\mathbf q) = 1 - a\,\exp\!\left(-\frac{B\,|\mathbf q|^2}{4}\right),
$$

with defaults $a = 0.8$, $B = 2000$ Å² and $B_\text{atomic} = 100$ Å². The
B-factor term uses RELION's CTF convention $\exp(-B s^2/4)$ and the simulator's
existing `get_B_factor_scaling`; $B_\text{atomic} = 0$ disables it. $\mathbf q$ is the 3D spatial
frequency in cycles/Å: on recovar's centered DFT grid
(`fourier_transform_utils.get_dft3`), the integer frequency index $\mathbf k$
gives $\mathbf q = \mathbf k / (N\,\Delta)$ with $N$ the grid size and
$\Delta$ the simulation voxel size. $H(0) = 1 - a$ (0.2 at the defaults) and
$H \to 1$ at high frequency. Validation requires $0 \le a \le 1$,
$B \ge 0$ and $B_\text{atomic} \ge 0$.

Implementation: [`solvent_contrast_filter`](../../recovar/simulation/solvent_contrast.py)
and [`apply_solvent_contrast`](../../recovar/simulation/solvent_contrast.py).

Check the input maps before accepting the default $B_\text{atomic}$. recovar's
bundled `recovar/assets/vol000{0,1,2}.mrc` are 5nrl maps (64³, 8.5 Å) that
were already multiplied by $\exp(-100\,|\mathbf q|^2/4)$ when they were made
(`make_trajectories.ipynb`); a shell-amplitude fit against freshly generated
5nrl conformations gives $B \approx 113$ Å². Volumes from
`generate_trajectory_volumes` also carry its `Bfactor`. For such inputs pass
`atomic_bfactor=0` (`--atomic-bfactor 0`) to add only the solvent term; relax's
fixture scripts do this automatically.

## Where it is applied

[`generate_synthetic_dataset`](../../recovar/simulation/simulator.py)
applies the operator when the preset is on:

- Python: `generate_synthetic_dataset(..., **solvent_contrast.EM_DEVELOPMENT_PRESET)`,
  which is `atomic_solvent_correction=True` with the defaults spelled out;
  override with `solvent_contrast_a`, `solvent_contrast_B`, `atomic_bfactor`.
- CLI: `--atomic-solvent-correction [--solvent-contrast-a A] [--solvent-contrast-b B]
  [--atomic-bfactor B_ATOMIC]` on `recovar make_test_dataset` and
  `run_test_all_metrics`; `make_spike_datasets.main(**kwargs)` forwards the same keywords.

The order is:

1. load and resample the input volumes to `grid_size` (unchanged);
2. compute the global `scale_vol` from the **uncorrected** volumes;
3. project $T(V \cdot \texttt{scale\_vol})$; the outlier volume, if any, is also filtered.

Images keep the normal CTF and noise mechanisms; completed noisy images are
never filtered. Volumes are not renormalized after filtering and negative
values are not clamped. The noise variance is set from the noise model before
filtering, so at a fixed noise level the correction lowers the SNR. The later
image-power normalization (the 10-image probe, or RELION-style background
normalization) rescales signal and noise variance by the same factor, so it
changes the image scale but keeps that SNR. The final `scale_vol` records it.

## Metadata and exactly-once application

`simulation_info.pkl` gains the key `atomic_solvent_correction`:

| Field | Meaning |
| --- | --- |
| `enabled` | `False` when the option is off (the only field then) |
| `model`, `model_version` | `"henderson_mcmullan_2013"`, `2` for the combined transform; version-1 records (no B-factor term) still load with $B_\text{atomic} = 0$ |
| `a`, `B`, `B_atomic`, `units` | actual parameters; units `dimensionless`, `angstrom^2`, `angstrom^2`, `q` in `cycles/angstrom` |
| `voxel_size`, `grid_size`, `volume_shape` | grid on which $H$ is evaluated (Å, voxels) |
| `fourier_convention`, `formula`, `reference`, `applied_to` | human-readable provenance |
| `applied_to_outlier_volume` | whether the outlier volume was filtered |
| `ground_truth_representation` | `"uncorrected_inputs"`: `volumes_path_root` points at the original files and the loader applies $T$; `"corrected_effective"`: the referenced volumes are already $T(V)$ and are not filtered again |

The input files are never modified.
[`load_ground_truth_volumes`](../../recovar/simulation/synthetic_dataset.py)
reads the record through
[`record_from_simulation_info`](../../recovar/simulation/solvent_contrast.py)
and rebuilds exactly the array handed to the projector,
$T(V \cdot \texttt{scale\_vol})$. relax's fixture-preparation scripts write their
`reference_gt*.mrc` maps through `load_heterogeneous_reconstruction`, and relax
enables the preset by default in those scripts (`--no-atomic-solvent-correction`
opts out). Datasets without the key, or with
`enabled: False`, load the legacy uncorrected truth. An enabled record with an
unknown model or version, missing fields, invalid parameters, an unknown
representation, or a grid that disagrees with `simulation_info` raises
`ValueError` instead of returning wrong truth.

## Ground truth for PCA and evaluation

`load_heterogeneous_reconstruction` builds `HeterogeneousVolumeDistribution`
from these effective volumes. Its mean, covariance and PCs are therefore those
of the corrected ensemble, with the same state weights:

$$
\mu_\text{true} = T(\mu), \qquad \Sigma_\text{true} = T\,\Sigma\,T^*.
$$

The PCs are recomputed from the centered corrected ensemble. They are generally
not the individually filtered old PCs. Equivalently, from a low-rank
$\Sigma = U\Lambda U^*$, the SVD $T(U)\Lambda^{1/2} = U_\text{new} S W^*$ gives
eigenvalues $\operatorname{diag}(S)^2$. The existing Fourier radial mask is
diagonal, like $H$, so the two commute. Nothing is cached across loads, so the
uncorrected statistics cannot leak in.

The analysis side needs no flag. Do not add $H$ to the PPCA or pipeline
observation operator, do not filter observed images on load, and do not
inverse-filter estimates: the unknown being reconstructed is already the
corrected volume.

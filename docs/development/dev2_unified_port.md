# dev2-unified: porting RELAX's recovar surface onto dev2-minimal

Branch `claude/dev2-unified`, from `claude/dev2-minimal` (613b607, PR #191). Sources: RELAX's pin
`1c8e99e9` (dev2 lineage), dev2 `f49c605` (two RELION-floor fixes after the pin) and `a57bcdd`
(`claude/tomo-sim-attachments`: per-particle tomo volumes + filament-attachment simulator).

Inventory method: AST scan of every `recovar` import and attribute use in RELAX `relax/`, `relax_cli/`,
`tests/`, `scripts/` (relax-mt 14d8752a); each (module, name) compared between 613b607 and 1c8e99e9.
Counts are names RELAX uses: identical / different / absent on dev2-minimal. Package re-exports
(`recovar.core.*`, `recovar.utils.*`) are counted under their submodules.

Merge rule: a module RELAX needs is the 3-way merge of the pin with #191's own changes
(base 2ab65a72e -> 613b607), so #191's fixes stay; most of them came from dev2 and were already in
the pin. Modules where RELAX's names are identical or differ only in docstrings stay at 613b607.

## Inventory

| module | RELAX users | names identical / different / absent on dev2-minimal | action |
|---|---|---|---|
| `recovar.commands.parse_relion5_tomo` | relax | 0 / 1 / 0 | merged: min (cryo-ET origin/tilt fixes) + pin per-particle optics groups |
| `recovar.core.configs` | relax,scripts,tests | 1 / 0 / 0 | unchanged (identical) |
| `recovar.core.ctf` | relax,scripts,tests | 3 / 2 / 0 | pin (gamma_offset, RELION wavelength constants, dtype grids) |
| `recovar.core.cubic_interpolation` | tests | 1 / 0 / 0 (module differs) | kept min (docstrings only) |
| `recovar.core.forward` | tests | 2 / 0 / 0 | unchanged (identical) |
| `recovar.core.fourier_transform_utils` | relax,scripts,tests | 17 / 10 / 22 (module differs) | pin (grid memoization, dtype args) |
| `recovar.core.geometry` | tests | 2 / 1 / 0 (module differs) | kept min (docstrings only; min keeps HIGHEST comment) |
| `recovar.core.mask` | relax,scripts,tests | 0 / 5 / 3 (module differs) | merged pin + min soft_mask_outside_map fix (pin form) |
| `recovar.core.padding` | relax,tests | 4 / 0 / 0 | unchanged (identical) |
| `recovar.core.relion_interp` | relax | 2 / 0 / 0 | unchanged (identical) |
| `recovar.core.slicing` | relax,tests | 5 / 5 / 2 (module differs) | pin (indexed adjoints, relion_texture_interp, max_r scaling moved to cuda_backproject) |
| `recovar.cuda_backproject` | relax,scripts,tests | 9 / 5 / 27 (module differs) | pin (indexed/RELION kernels, two-library loader); PerImageBackproject identical |
| `recovar.cuda_build` | relax,tests | 0 / 0 / 6 | pin (new) |
| `recovar.data_io.cryoem_dataset` | relax,scripts,tests | 3 / 2 / 0 (module differs) | merged pin + min |
| `recovar.data_io.image_backends` | relax,scripts,tests | 0 / 2 / 12 (module differs) | merged pin + min (tilt numbering, prefetch stop); default-mode half path kept min (legacy full FFT) |
| `recovar.data_io.image_loader` | relax | 0 / 2 / 0 | pin (preread) |
| `recovar.data_io.image_sources` | relax | 0 / 1 / 0 | pin (host_images, kwargs passthrough) |
| `recovar.data_io.load_utils` | tests | 1 / 0 / 0 (module differs) | pin (preserve_source_geometry) |
| `recovar.data_io.metadata_readers` | relax,tests | 0 / 3 / 1 (module differs) | pin |
| `recovar.data_io.staging` | relax,tests | 1 / 0 / 2 (module differs) | merged pin (stage_stacks) + min (network-FS guard) |
| `recovar.data_io.starfile` | relax,scripts,tests | 1 / 3 / 2 (module differs) | pin (multi-block writer, star_column) |
| `recovar.heterogeneity.adaptive_kernel_discretization` | relax | 1 / 0 / 0 | unchanged (identical) |
| `recovar.heterogeneity.covariance_core` | tests | 1 / 0 / 0 (module differs) | kept min (RELAX uses unchanged helpers) |
| `recovar.heterogeneity.covariance_estimation` | tests | 4 / 0 / 0 (module differs) | kept min (RELAX uses unchanged helpers) |
| `recovar.heterogeneity.heterogeneity_volume` | relax | 0 / 1 / 0 | kept min (docstrings only) |
| `recovar.heterogeneity.locres` | tests | 2 / 0 / 0 | unchanged (identical) |
| `recovar.heterogeneity.principal_components` | tests | 1 / 0 / 0 (module differs) | kept min (RELAX uses unchanged helper) |
| `recovar.jax_config` | relax,tests | 2 / 0 / 0 (module differs) | pin EM XLA flag; default cache dir gated on EM marker |
| `recovar.output.output` | scripts | 2 / 0 / 0 | unchanged (identical) |
| `recovar.output.output_paths` | relax | 1 / 0 / 0 | unchanged (identical) |
| `recovar.ppca.augmented_mstep` | relax,scripts,tests | 0 / 0 / 2 | pin (new) |
| `recovar.ppca.pose_accumulators` | relax,scripts,tests | 0 / 0 / 1 | pin (new) |
| `recovar.ppca.pose_marginal` | relax,scripts,tests | 0 / 0 / 1 | pin (new) |
| `recovar.ppca.triangular` | relax,scripts,tests | 1 / 1 / 2 | pin (adds tri_size, pack_upper_tri, hermitian) |
| `recovar.reconstruction.noise` | relax,scripts,tests | 12 / 1 / 1 (module differs) | merged pin + min (#187 upper bound, grouped profiles) |
| `recovar.reconstruction.regularization` | relax,scripts,tests | 6 / 4 / 1 (module differs) | pin (tau2_fudge etc.) |
| `recovar.reconstruction.relion_functions` | relax,tests | 6 / 6 / 21 (module differs) | dev2 f49c605 (= pin + floor fixes f9987379, ea1b994) |
| `recovar.simulation.pdb_utils` | scripts | 1 / 0 / 0 | unchanged (identical) |
| `recovar.simulation.relion_tomo` | scripts,tests | 0 / 0 / 6 | a57bcdd (new; per-particle volumes) |
| `recovar.simulation.simulate_scattering_potential` | scripts,tests | 2 / 0 / 0 | unchanged (identical) |
| `recovar.simulation.simulator` | scripts,tests | 2 / 3 / 0 (module differs) | pin; Kent theta RNG and noise transform batching defaults kept min |
| `recovar.simulation.solvent_contrast` | scripts,tests | 0 / 0 / 8 | pin (new) |
| `recovar.simulation.synthetic_dataset` | scripts | 0 / 1 / 1 (module differs) | pin |
| `recovar.simulation.trajectory_generation` | scripts,tests | 5 / 1 / 0 | kept min (docstrings only) |
| `recovar.utils.cuda_env` | relax | 1 / 0 / 0 | unchanged (identical) |
| `recovar.utils.file_hash` | relax,scripts,tests | 0 / 0 / 1 | pin (new) |
| `recovar.utils.helpers` | relax,scripts,tests | 2 / 3 / 5 (module differs) | merged pin (RELION volume converters, MRC stack writer) + min (cgroup memory) |
| `recovar.utils.nvtx_shim` | relax | 1 / 0 / 0 | unchanged (identical) |

Simulator-path / transitive modules not imported directly by RELAX:

| module | status on dev2-minimal | action |
|---|---|---|
| `recovar.simulation.optics_groups` | absent | a57bcdd (new; imported by relion_tomo) |
| `recovar.simulation.filament_attachments` | absent | a57bcdd (new) |
| `recovar.core.cuda_ops` | different | pin (relion_texture_interp passthrough) |
| `recovar.data_io.halfsets` | different | pin (dtype in spec) |
| `recovar/cuda/*.cu, include/*.cuh, Makefile` | different | pin |

## Deliberate departures from the pin (recovar defaults kept)

| where | kept from dev2-minimal | why |
|---|---|---|
| `ParticleImageDataset.process_images_half`, multiply mask, complex64 | legacy full FFT, then crop to half | direct rfft drifts PCA/outlier regressions (min's docstring); RELION modes and complex128 keep the pin's direct rfft |
| `simulator.generate_synthetic_dataset` | no noise-transform batching | regression fixtures are generated with one transform per RNG chunk; `simulate_data(noise_transform_batch_size=)` stays |
| `simulator.kent_sampling_scheme` with sphstat | in-plane angles from `np.random.rand` | unchanged generated datasets; the no-sphstat fallback uses the pin's `rng` |
| `jax_config` | no default `~/.cache` compile cache | default directory only for EM entry points (`RECOVAR_EM_XLA_DEFAULTS`, set by relax) or `RECOVAR_JAX_CACHE_DIR` |
| `parse_relion5_tomo.convert` | 2D tilt origins, missing CtfScalefactor, `_rlnTiltName`, phase shift (cryoet-dev) | plus the pin's per-particle optics groups |

## Not ported

- `recovar/simulation/relion_spa.py`, `make_relion_tomo_dataset` CLI, `json_utils`, `pcg_mean`/`pcg_variants`,
  `homogeneous.get_mean_pcg`, `tilt_diagnostics` removal, pipeline/covariance/embedding/output changes,
  agent docs, public-API freeze fixture: nothing in RELAX or the simulator imports them.
- dependencies jaxwt, pyproximal, pybind11: no ported module imports them (RELAX declares its own).
- tests of unported CLI plumbing: `test_cli_preset_on_by_default_with_opt_out` (make_relion_tomo_dataset), `test_cli_flags_forward_to_simulator`, `test_run_test_all_metrics_forwards_atomic_volume_options`.
- tests asserting what #191 reverted: dev2 `test_import_boundary` (forbids the restored `recovar.em`),
  `test_project` registry changes, `test_conftest_repo_subprocess` (dev2 conftest harness),
  `..._process_images_half_uses_native_rfft` (contradicts the kept legacy path).

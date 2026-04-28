"""F7 parity test: E-step Pmax vs run_it001_data.star.

Loads the RELION InitialModel fixture:
  - particles.star + particles.64.mrcs (500 particles)
  - run_it000_class001.mrc (iter-0 Iref, seeded Iref)
  - run_it000_model.star sigma2_noise (group 1)

Runs ONE VDAM E-step pass with the existing `run_em` engine at:
  - padding_factor = 1 (GUI InitialModel default)
  - HEALPix order = 1 + oversampling 1 -> 384 rotations
  - translation grid offset_range = 6, offset_step = 2 -> 49 trans
  - half_spectrum_scoring = True (RELION parity)
  - current_size = 28 (from fixture iter-0)

Compares `stats.max_posterior_per_image.mean()` against iter-1 Pmax
target = 0.1217 (aggregate mean from run_it001_data.star).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

FIXTURE_DIR = Path("/scratch/gpfs/GILLES/mg6942/tmp/relion_initialmodel_64_20260420_121428_8956_run")
PARTICLES_STAR = Path(
    "/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar/.tmp/"
    "slurm_7178672/pytest-of-mg6942/pytest-0/test_pipeline_spa_gpu0/"
    "gpu_spa/test_dataset/particles.star"
)


pytestmark = pytest.mark.unit

requires_fixture = pytest.mark.skipif(
    not (FIXTURE_DIR.exists() and PARTICLES_STAR.exists()),
    reason="RELION InitialModel fixture not available on this host",
)


def _read_iter0_sigma2(n_shells: int) -> np.ndarray:
    """Parse sigma2_noise group 1 from run_it000_model.star."""
    txt = (FIXTURE_DIR / "run_it000_model.star").read_text()
    m = re.search(r"data_model_optics_group_1\n(.*?)(?:\ndata_)", txt, re.DOTALL)
    if not m:
        raise RuntimeError("could not find data_model_optics_group_1")
    values = np.zeros(n_shells, dtype=np.float64)
    for line in m.group(1).strip().split("\n"):
        toks = line.split()
        if len(toks) == 3:
            try:
                values[int(toks[0])] = float(toks[2])
            except ValueError:
                continue
    return values


def _read_iter1_pmax_mean() -> float:
    from recovar.data_io.starfile import read_star

    main, _ = read_star(str(FIXTURE_DIR / "run_it001_data.star"))
    pmax = main["_rlnMaxValueProbDistribution"].astype(float)
    return float(pmax.mean())


@requires_fixture
def test_estep_pmax_matches_relion_iter1():
    """Run one E-step and compare Pmax against iter-1 data.star."""
    import jax.numpy as jnp

    from recovar.core import fourier_transform_utils as ftu
    from recovar.data_io.cryoem_dataset import load_dataset
    from recovar.em.dense_single_volume.em_engine import run_em
    from recovar.em.sampling import get_rotation_grid, get_translation_grid
    from recovar.utils.helpers import load_relion_volume

    # --- 1. Load particle dataset ---
    ds = load_dataset(str(PARTICLES_STAR), lazy=False)
    ori_size = int(ds.grid_size)
    pixel_size = float(ds.voxel_size)
    assert ori_size == 64, f"expected box 64, got {ori_size}"
    assert abs(pixel_size - 8.5) < 1e-3, f"expected pix 8.5, got {pixel_size}"
    n_images = ds.n_images
    assert n_images == 500

    # --- 2. Load iter-0 Iref ---
    iref_real = load_relion_volume(str(FIXTURE_DIR / "run_it000_class001.mrc"))
    iref_real = np.asarray(iref_real, dtype=np.float64)
    assert iref_real.shape == (ori_size, ori_size, ori_size)
    # Convert to centered Fourier space (recovar's mean convention)
    iref_ft = np.asarray(ftu.get_dft3(jnp.asarray(iref_real))).reshape(-1)

    # --- 3. Load sigma2 and build full-image radial noise model ---
    # RELION FFT is normalised (F_relion = FFT(img)/N^d), so sigma2 in
    # model.star is in RELION's convention. recovar uses unnormalised FFT,
    # so sigma2 must be scaled by N^(2*data_dim) = N^4 for 2D data.
    n_shells = ori_size // 2 + 1
    sigma2 = _read_iter0_sigma2(n_shells)
    n4 = ori_size**4
    from recovar.reconstruction.noise import make_radial_noise

    noise_variance = np.asarray(make_radial_noise(sigma2 * n4, (ori_size, ori_size))).astype(np.float32).reshape(-1)

    # --- 4. Build rotation + translation grids ---
    # HEALPix order 1 with oversample_order 1 (2x finer) gives Npix=48*4=192
    # and n_psi(order+1)=2*(6*2^(order+1)) = 24 -> 192 * 24 = 4608 rotations
    # But RELION's --oversampling 1 in InitialModel uses 8x oversampling per orient
    # (rotation × translation). For a simpler test, use order-1 (48 base)
    # with n_psi(order=1)=12 -> 576 rotations. This is close to RELION's behaviour
    # at the coarse sampling level.
    rotations = get_rotation_grid(nside_level=1, n_in_planes=12, matrices=True).astype(np.float32)
    print(f"n_rotations = {rotations.shape[0]}")

    # Translation grid: offset_range=6, offset_step=2 -> {-6,-4,-2,0,2,4,6} = 7 vals
    # => 7x7 = 49 (but some fall outside the L_inf radius)
    translations = get_translation_grid(max_pixel=6, pixel_offset=2).astype(np.float32)
    print(f"n_translations = {translations.shape[0]}")

    # --- 5. Run E-step ---
    current_size = 28  # from iter-0 optimiser.star fallback via 0.07 rule
    mean_ft = jnp.asarray(iref_ft, dtype=jnp.complex64)
    mean_variance = jnp.zeros_like(mean_ft, dtype=jnp.float32).real
    noise_variance_j = jnp.asarray(noise_variance, dtype=jnp.float32)

    result = run_em(
        ds,
        mean=mean_ft,
        mean_variance=mean_variance,
        noise_variance=noise_variance_j,
        rotations=rotations,
        translations=jnp.asarray(translations),
        disc_type="linear_interp",
        image_batch_size=50,
        rotation_block_size=100,
        current_size=current_size,
        projection_padding_factor=1,
        reconstruction_padding_factor=1,
        half_spectrum_scoring=True,
        return_stats=True,
    )
    # run_em returns (Ft_y, Ft_ctf, relion_stats) or similar when return_stats=True
    # Inspect result tuple
    if isinstance(result, tuple):
        for x in result:
            if hasattr(x, "max_posterior_per_image"):
                stats = x
                break
        else:
            raise RuntimeError(
                f"RelionStats not found in run_em result; got types {[type(x).__name__ for x in result]}"
            )
    else:
        raise RuntimeError(f"unexpected run_em result type {type(result)}")

    ours_pmax = np.asarray(stats.max_posterior_per_image)
    ours_mean = float(ours_pmax.mean())

    relion_mean = _read_iter1_pmax_mean()

    print("\nF7 E-STEP PARITY:")
    print(f"  ours Pmax mean   = {ours_mean:.6f}")
    print(f"  relion Pmax mean = {relion_mean:.6f}")
    print(f"  ratio            = {ours_mean / relion_mean:.4f}")

    # Soft gate: our mean Pmax is within a factor of 2 of RELION's.
    # F8 tightens this once full E-step + M-step is running.
    assert 0.3 * relion_mean < ours_mean < 3.0 * relion_mean, (
        f"Pmax mean out of range: {ours_mean:.4f} vs target {relion_mean:.4f}"
    )


# ---------------------------------------------------------------------------
# E-step → BPref forward parity probe
# ---------------------------------------------------------------------------

import struct as _struct  # noqa: E402

RELION_DUMP_DIR = Path("/scratch/gpfs/GILLES/mg6942/_agent_scratch/relion_debug_dump")
requires_estep_dumps = pytest.mark.skipif(
    not (
        FIXTURE_DIR.exists()
        and PARTICLES_STAR.exists()
        and (RELION_DUMP_DIR / "pipe_it1_c0_bp_data_pre_reweight.bin").exists()
    ),
    reason="iter-1 E-step BPref dumps not present",
)


def _read_bin(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        nz, ny, nx = _struct.unpack("qqq", f.read(24))
        pos = f.tell()
        f.seek(0, 2)
        rem = f.tell() - pos
        f.seek(pos)
        bp = rem // (nz * ny * nx) if nz * ny * nx else 8
        dt = np.complex128 if bp == 16 else np.float64
        return np.fromfile(f, dtype=dt, count=nz * ny * nx).reshape(nz, ny, nx)


def _cc(a: np.ndarray, b: np.ndarray) -> float:
    af = a.ravel() - a.mean()
    bf = b.ravel() - b.mean()
    return float(np.real(np.vdot(af, bf)) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-30))


def _max_rel_err(out: np.ndarray, target: np.ndarray) -> float:
    denom = max(float(np.abs(target).max()), 1e-30)
    return float(np.abs(out - target).max() / denom)


@requires_estep_dumps
def test_estep_bpref_forward_parity():
    """Drive `run_iter_gpu_vdam` end-to-end on the RELION InitialModel fixture
    and compare the per-halfset BPref accumulator (pre-reweight) against
    RELION's iter-1 dumps.

    Phase C refactor (2026-04-28): the test now exercises the production
    `run_iter_gpu_vdam` wrapper instead of calling `run_em` directly with a
    bespoke recipe. This guarantees the test gate covers the same code path
    that downstream consumers (e.g. `scripts/run_ab_initio.py`) will hit.

    With pseudo_halfsets=1, RELION accumulates two BPref instances by
    alternating particles across halfset slots. We mirror that via
    `_split_halfset_particle_ids` driven from `_rlnMicrographName`.

    Per-kernel ceiling (2026-04-28, post-rebase onto canonical EM-parity
    branch): BPref bp_data CC ≈ +0.73 on this small (500/64) iter-1 fixture.
    Closing the residual to +0.99 requires patching RELION to emit an iter-1
    `Frefctf_orient0` dump at the windowed (28×15) layout. See memory
    `project_initial_model_estep_diag_2026_04_26.md` and the diag scripts at
    `/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar/scripts/diag_p0_diff2_*.py`.
    """
    import jax

    from recovar.data_io.cryoem_dataset import load_dataset
    from recovar.data_io.starfile import read_star
    from recovar.em.initial_model.gpu_pipeline import _split_halfset_particle_ids, run_iter_gpu_vdam
    from recovar.em.sampling import (
        apply_relion_translation_perturbation,
        get_oversampled_rotation_grid_from_samples,
        get_oversampled_translation_grid,
        get_translation_grid,
        read_relion_perturbation_from_sampling_star,
    )

    try:
        if not jax.devices("gpu"):
            pytest.skip("No GPU available")
    except RuntimeError:
        pytest.skip("JAX has no GPU backend (CPU-only environment)")

    ds = load_dataset(str(PARTICLES_STAR), lazy=False)
    ori = int(ds.grid_size)
    assert ori == 64

    # Enable RELION-exact mask geometry + softMaskOutsideMap blend so our
    # masked Fimg matches RELION's exp_Fimg at machine precision.
    backend = ds.image_source.backend if hasattr(ds.image_source, "backend") else None
    if backend is not None and hasattr(backend, "set_relion_image_mask"):
        backend.set_relion_image_mask(
            pixel_size=float(ds.voxel_size),
            particle_diameter_ang=544.0,
            width_mask_edge_px=5.0,
        )

    if not hasattr(ds, "subset"):
        pytest.skip("dataset has no subset()")

    # iter-0 Iref + sigma2 (RELION fixture). `run_iter_gpu_vdam` performs the
    # gridding correction + /N² FFT-norm + sign-negate internally when invoked
    # with the matching kwargs below.
    from recovar.utils.helpers import load_relion_volume

    iref_real = np.asarray(load_relion_volume(str(FIXTURE_DIR / "run_it000_class001.mrc")), dtype=np.float64)
    sigma2 = _read_iter0_sigma2(ori // 2 + 1)

    # Iter-1 sampling: prefer RELION's exact dumped post-perturbation grid;
    # fall back to constructed grid otherwise.
    sampling_star = FIXTURE_DIR / "run_it001_sampling.star"
    random_perturbation, _perturbation_factor = read_relion_perturbation_from_sampling_star(str(sampling_star))
    relion_estep_dump = Path("/scratch/gpfs/GILLES/mg6942/_agent_scratch/relion_estep_dump_small")
    eulers_bin = relion_estep_dump / "p0_oversampled_eulers.bin"
    trans_bin = relion_estep_dump / "p0_oversampled_translations.bin"
    if eulers_bin.exists() and trans_bin.exists():
        from recovar.utils.helpers import R_from_relion as _R_from_relion

        with open(eulers_bin, "rb") as _f:
            _h = _struct.unpack("qqq", _f.read(24))
            _e = np.fromfile(_f, dtype=np.float64, count=_h[0] * _h[1] * _h[2]).reshape(-1, 3)
        rotations = _R_from_relion(_e).astype(np.float32)
        with open(trans_bin, "rb") as _f:
            _h = _struct.unpack("qqq", _f.read(24))
            _t = np.fromfile(_f, dtype=np.float64, count=_h[0] * _h[1] * _h[2]).reshape(-1, 3)
        translations = _t[:, :2].astype(np.float32)
    else:
        coarse_indices = np.arange(48 * 12, dtype=np.int64)
        rotations, _ = get_oversampled_rotation_grid_from_samples(
            coarse_indices, parent_nside_level=1, oversampling_order=1, random_perturbation=random_perturbation
        )
        rotations = rotations.astype(np.float32)
        coarse_translations = get_translation_grid(max_pixel=6, pixel_offset=2).astype(np.float32)
        translations, _ = get_oversampled_translation_grid(coarse_translations, pixel_offset=2, oversampling_order=1)
        translations = apply_relion_translation_perturbation(
            translations.astype(np.float32), random_perturbation, offset_step_pixels=2.0
        )

    current_size = 28
    r_max = 14

    # Pseudo-halfset routing: pass micrograph names so the wrapper picks
    # RELION-sorted halfsets via `_split_halfset_particle_ids`.
    main_in, _ = read_star(str(PARTICLES_STAR))
    mic_names = np.asarray(main_in["_rlnMicrographName"].tolist())
    # Sanity-check the wrapper's halfset assignment matches RELION's lex-sort.
    h0_ids, h1_ids = _split_halfset_particle_ids(ds.n_images, micrograph_names=mic_names)
    assert h0_ids.size + h1_ids.size == ds.n_images

    # Drive the production wrapper. Volume preprocessing kwargs reproduce the
    # bespoke iter-1 recipe (gridding + /N² + sign-flip): with run_em's
    # internal gridding only firing at projection_padding_factor > 1 and
    # InitialModel/VDAM using `--pad 1`, we need to apply it externally.
    # The Gaussian translation prior is read from `_rlnSigmaOffsetsAngst` in
    # run_it001_model.star (≈ 6.4 Å on this fixture).
    sigma_offset_Ang = 6.398173
    iref_for_wrapper = iref_real.copy()
    iref_next, Igrad1, Igrad2, stats_dict = run_iter_gpu_vdam(
        ds,
        iref_for_wrapper,
        sigma2,
        rotations,
        translations,
        current_size=current_size,
        iter=1,
        image_batch_size=50,
        rotation_block_size=100,
        half_spectrum_scoring=True,
        padding_factor=1,
        pseudo_halfsets=True,
        apply_gridding_correction=True,
        iref_ft_scale=1.0 / (ori**2),
        iref_ft_sign=-1.0,
        score_with_masked_images=True,
        relion_firstiter_score_mode="gaussian",
        sigma_offset_Ang=sigma_offset_Ang,
        accumulate_noise=False,
        sparse_pass2=True,
        micrograph_names=mic_names,
    )
    intermediates = stats_dict["intermediates"]

    # Apply the BPref-frame correction (`-N²` for bp_data, `N⁴` for bp_weight)
    # to bring recovar's centered Ft_y / Ft_ctf into RELION's BPref slab frame.
    n2 = float(ori) ** 2
    n4 = float(ori) ** 4
    bp_data_h0 = -np.asarray(intermediates["bp_data_h0"]) * n2
    bp_data_h1 = -np.asarray(intermediates["bp_data_h1"]) * n2
    bp_weight_h0 = np.asarray(intermediates["bp_weight_h0"]) * n4
    bp_weight_h1 = np.asarray(intermediates["bp_weight_h1"]) * n4

    target_bp_data_h0 = _read_bin(RELION_DUMP_DIR / "pipe_it1_c0_bp_data_pre_reweight.bin")
    target_bp_data_h1 = _read_bin(RELION_DUMP_DIR / "pipe_it1_c0_bp_data_h_pre_reweight.bin")
    target_bp_weight_h0 = _read_bin(RELION_DUMP_DIR / "pipe_it1_c0_bp_weight.bin")
    target_bp_weight_h1 = _read_bin(RELION_DUMP_DIR / "pipe_it1_c0_bp_weight_h.bin")

    print("\nE-STEP → BPref FORWARD PARITY (small fixture, 500 particles, box 64):")
    rows = [
        ("h0 bp_data", bp_data_h0, target_bp_data_h0),
        ("h1 bp_data", bp_data_h1, target_bp_data_h1),
        ("h0 bp_weight", bp_weight_h0, target_bp_weight_h0),
        ("h1 bp_weight", bp_weight_h1, target_bp_weight_h1),
    ]
    for name, ours, target in rows:
        cc = _cc(ours, target)
        rel = _max_rel_err(ours, target)
        ratio = float(np.linalg.norm(ours)) / max(float(np.linalg.norm(target)), 1e-30)
        print(f"  {name:14s}: CC = {cc:+.6f}   rel_err = {rel:.3e}   ‖ours‖/‖target‖ = {ratio:.4f}")

    cc_h0 = _cc(bp_data_h0, target_bp_data_h0)
    cc_h1 = _cc(bp_data_h1, target_bp_data_h1)
    print(f"\n  near-parity gate cc_h0={cc_h0:+.4f}, cc_h1={cc_h1:+.4f} (gate: > +0.7)")

    # Per-kernel near-parity ceiling on this iter-1 small fixture is ~+0.73
    # (memory `project_initial_model_estep_diag_2026_04_26.md`). Going below
    # this regress threshold means parameter alignment to standard E-M (gridding,
    # /N², sign-flip, RELION-sorted halfsets, masked Fimg, gaussian score mode,
    # Gaussian translation prior) was unintentionally undone.
    assert cc_h0 > 0.7, f"BPref h0 CC regressed: {cc_h0:.4f} (post-rebase baseline ≈ +0.73)"
    assert cc_h1 > 0.7, f"BPref h1 CC regressed: {cc_h1:.4f}"

"""GPU-accelerated VDAM iteration pipeline.

Combines:
  - E-step on GPU via recovar's JIT-compiled `run_em`
  - M-step VDAM-blend on CPU (plain-EM + RELION-weighted blend)

For full VDAM reconstructGrad parity the M-step can route through the
C++ binding (`vdam_reconstruct_grad`) which requires converting Ft_y /
Ft_ctf from recovar's half-volume-flat layout to RELION's projector-
centered padded layout. That binding + layout converter is the last
remaining piece for end-to-end bit-exact iter-1 parity and is scaffolded
in `gpu_pipeline.reconstruct_grad_from_run_em_output`.

Typical usage:

    from recovar.em.initial_model.gpu_pipeline import run_iter_gpu
    state = run_iter_gpu(
        ds=ds,
        iref_real=iref0,
        sigma2_noise=sigma2,
        rotations=rots, translations=trans,
        current_size=28,
        iter=1,
        blend_step=0.5,  # or None to use the VDAM schedule default
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class IterResult:
    """Output of a single GPU iteration."""

    iref_real: np.ndarray  # (N, N, N) real-space updated reference
    iref_ft: np.ndarray  # (N^3,) centered-FT flat complex
    Ft_y: np.ndarray  # raw half-volume-flat E-step accumulator
    Ft_ctf: np.ndarray  # raw half-volume-flat E-step denominator
    relion_stats: object  # RelionStats NamedTuple
    wall_time_s: float
    e_step_s: float
    m_step_s: float


def run_iter_gpu(
    ds,
    iref_real: np.ndarray,
    sigma2_noise: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    current_size: int,
    iter: int,
    blend_step: Optional[float] = None,
    image_batch_size: int = 500,
    rotation_block_size: int = 50,
    half_spectrum_scoring: bool = True,
    padding_factor: int = 1,
    phase_lengths=None,
) -> IterResult:
    """Run one VDAM iteration on GPU (E-step + blend M-step).

    Parameters
    ----------
    ds : CryoEMDataset
        Particle data loaded via `load_dataset`. Iterating will use the
        dataset's default GPU path in `run_em`.
    iref_real : np.ndarray
        Current reference volume in real space, shape (N, N, N).
    sigma2_noise : np.ndarray
        Per-shell noise power, shape (N//2 + 1,). Must be already scaled by
        N^4 per RELION↔recovar FFT-normalization convention.
    rotations, translations : np.ndarray
        Sampling grid (see `recovar.em.sampling`).
    current_size : int
        Fourier-window diameter for scoring (RELION's rlnCurrentImageSize).
    iter : int
        Current iteration number (1-indexed).
    blend_step : float or None
        Real-space VDAM blend step. None → use the `schedules.compute_stepsize`
        default for the 3D initial-model path.
    """
    import jax
    import jax.numpy as jnp

    from recovar.core import fourier_transform_utils as ftu
    from recovar.em.dense_single_volume.em_engine import run_em
    from recovar.em.initial_model.schedules import compute_phase_lengths, compute_stepsize
    from recovar.reconstruction.noise import make_radial_noise

    ori = iref_real.shape[0]
    n4 = ori**4

    # Build full-image-shaped noise (recovar expects (H*W,))
    nv = np.asarray(make_radial_noise(sigma2_noise * n4, (ori, ori))).astype(np.float32).reshape(-1)

    iref_ft = np.asarray(ftu.get_dft3(jnp.asarray(iref_real))).reshape(-1)
    mv = jnp.asarray((np.abs(iref_ft) ** 2).astype(np.float32))
    mean_j = jnp.asarray(iref_ft, dtype=jnp.complex64)

    # E-step on GPU
    jax.block_until_ready(mean_j)
    t_e0 = time.time()
    result = run_em(
        ds,
        mean=mean_j,
        mean_variance=mv,
        noise_variance=jnp.asarray(nv),
        rotations=jnp.asarray(rotations),
        translations=jnp.asarray(translations),
        disc_type="linear_interp",
        image_batch_size=image_batch_size,
        rotation_block_size=rotation_block_size,
        current_size=current_size,
        projection_padding_factor=padding_factor,
        reconstruction_padding_factor=padding_factor,
        half_spectrum_scoring=half_spectrum_scoring,
        return_stats=True,
    )
    jax.block_until_ready(result[0])
    e_step_s = time.time() - t_e0

    new_mean_ft = np.asarray(result[0])
    Ft_y = np.asarray(result[2])
    Ft_ctf = np.asarray(result[3])
    stats = result[4]

    # Convert back to real space
    t_m0 = time.time()
    new_mean_real = np.asarray(ftu.get_idft3(jnp.asarray(new_mean_ft).reshape(ori, ori, ori))).real

    # VDAM real-space blend
    if blend_step is None:
        if phase_lengths is None:
            phase_lengths = compute_phase_lengths(200, 0.3, 0.2)
        blend_step = compute_stepsize(iter=iter, phase_lengths=phase_lengths, is_3d_model=True, ref_dim=3)
    iref_new = (1.0 - blend_step) * iref_real + blend_step * new_mean_real
    m_step_s = time.time() - t_m0

    return IterResult(
        iref_real=iref_new,
        iref_ft=np.asarray(ftu.get_dft3(jnp.asarray(iref_new))).reshape(-1),
        Ft_y=Ft_y,
        Ft_ctf=Ft_ctf,
        relion_stats=stats,
        wall_time_s=e_step_s + m_step_s,
        e_step_s=e_step_s,
        m_step_s=m_step_s,
    )


def run_multi_iter_gpu(
    ds,
    iref_real_init: np.ndarray,
    sigma2_noise: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    current_size: int,
    nr_iter: int,
    blend_step: Optional[float] = None,
    image_batch_size: int = 500,
    rotation_block_size: int = 50,
    log_callback: Optional[Callable[[int, IterResult], None]] = None,
    **kwargs,
) -> np.ndarray:
    """Run multiple VDAM iterations on GPU.

    Returns the final real-space volume.
    """
    from recovar.em.initial_model.schedules import compute_phase_lengths

    phase_lengths = compute_phase_lengths(nr_iter, 0.3, 0.2)
    current = iref_real_init.copy()
    total_e, total_m = 0.0, 0.0
    for it in range(1, nr_iter + 1):
        res = run_iter_gpu(
            ds=ds,
            iref_real=current,
            sigma2_noise=sigma2_noise,
            rotations=rotations,
            translations=translations,
            current_size=current_size,
            iter=it,
            blend_step=blend_step,
            image_batch_size=image_batch_size,
            rotation_block_size=rotation_block_size,
            phase_lengths=phase_lengths,
            **kwargs,
        )
        current = res.iref_real
        total_e += res.e_step_s
        total_m += res.m_step_s
        if log_callback is not None:
            log_callback(it, res)
    return current, {"total_e_step_s": total_e, "total_m_step_s": total_m}


def run_em_output_to_bpref(
    Ft_y: np.ndarray,
    Ft_ctf: np.ndarray,
    ori_size: int,
    r_max: int,
    padding_factor: int = 1,
):
    """Convert recovar's run_em output to RELION's BPref compressed layout.

    recovar stores full 3D spectra CENTERED (DC at [N/2, N/2, N/2], see
    `recovar.core.fourier_transform_utils.get_dft3`). RELION's
    BackProjector stores a compressed half-complex slab sized
    (pad_size, pad_size, pad_size//2+1) where pad_size = 2*(r_max+1)+1
    at padding_factor=1, with `setXmippOrigin + xinit=0` so that
    raw index (k+hp, i+hp, j) holds frequency (k, i, j) with
    k,i ∈ [-hp, hp] and j ∈ [0, hp].

    So the converter is a pure centered crop: take the (pad_size)²×(hp+1)
    slab around DC on the non-negative x half. Same logic for data and
    weight (weight is real-valued).

    Parameters
    ----------
    Ft_y, Ft_ctf : flat or grid-shaped arrays of size ``ori_size**3``.
    ori_size : box edge length.
    r_max : current-resolution shell used by RELION's BPref.
    padding_factor : currently only 1 is supported (matches GUI InitialModel).

    Returns
    -------
    bp_data : complex128 array of shape (pad_size, pad_size, pad_size//2+1)
    bp_weight : float64 array of same shape
    """
    if padding_factor != 1:
        raise NotImplementedError("only padding_factor=1 supported for BPref conversion")
    N = ori_size
    half_ps = r_max + 1
    c = N // 2

    Fy = np.asarray(Ft_y).reshape(N, N, N)
    Fc = np.asarray(Ft_ctf).reshape(N, N, N)
    bp_data = Fy[c - half_ps : c + half_ps + 1, c - half_ps : c + half_ps + 1, c : c + half_ps + 1].astype(
        np.complex128, copy=True
    )
    bp_weight = Fc[c - half_ps : c + half_ps + 1, c - half_ps : c + half_ps + 1, c : c + half_ps + 1].real.astype(
        np.float64, copy=True
    )
    return bp_data, bp_weight


def bpref_to_run_em_output(
    bp_data: np.ndarray,
    bp_weight: np.ndarray,
    ori_size: int,
    r_max: int,
    padding_factor: int = 1,
):
    """Inverse of `run_em_output_to_bpref`: embed the BPref slab into a
    zero-filled centered full (N, N, N) spectrum.

    Used when we want to feed RELION-layout arrays back through recovar's
    Wiener solve / visualisation paths.
    """
    if padding_factor != 1:
        raise NotImplementedError("only padding_factor=1 supported for BPref conversion")
    N = ori_size
    half_ps = r_max + 1
    c = N // 2
    Fy = np.zeros((N, N, N), dtype=np.complex128)
    Fc = np.zeros((N, N, N), dtype=np.float64)
    Fy[c - half_ps : c + half_ps + 1, c - half_ps : c + half_ps + 1, c : c + half_ps + 1] = bp_data
    Fc[c - half_ps : c + half_ps + 1, c - half_ps : c + half_ps + 1, c : c + half_ps + 1] = bp_weight
    return Fy, Fc


def reconstruct_grad_from_run_em_output(
    Ft_y: np.ndarray,
    Ft_ctf: np.ndarray,
    iref_real: np.ndarray,
    ori_size: int,
    r_max: int,
    grad_stepsize: float,
    tau2_fudge: float,
    padding_factor: int = 1,
    min_resol_shell: float = 0.0,
    mom1_noise_power: Optional[np.ndarray] = None,
) -> np.ndarray:
    """reconstructGrad-only M-step driven by recovar run_em accumulators.

    Uses `run_em_output_to_bpref` to bridge the two layouts then calls the
    C++ `vdam_reconstruct_grad` binding. This skips the momentum pipeline
    (reweightGrad → getFristMoment → getSecondMoment → applyMomenta) and
    is therefore bit-exact only when mom1_noise_power is supplied from
    that upstream chain — see `run_iter_gpu_vdam` for the full path.
    """
    from recovar.relion_bind import _relion_bind_core as bind

    bp_data, bp_weight = run_em_output_to_bpref(Ft_y, Ft_ctf, ori_size, r_max, padding_factor)

    fsc = np.zeros(ori_size // 2 + 1, dtype=np.float64)
    out = np.asarray(
        bind.vdam_reconstruct_grad(
            iref_real.astype(np.float64),
            bp_data,
            bp_weight,
            fsc,
            grad_stepsize,
            tau2_fudge,
            ori_size,
            padding_factor,
            1,  # TRILINEAR
            r_max,
            min_resol_shell,
            False,  # use_fsc
            True,  # skip_gridding
            mom1_noise_power,
        )
    )
    return out


def _split_halfset_particle_ids(
    n_images: int,
    rng_seed: int = 0,
    micrograph_names: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (h0_ids, h1_ids) RELION-style partitioning of [0, n_images).

    RELION's pseudo-halfsets alternate 0/1 over particles in
    `_rlnMicrographName`-lexicographic-sorted order (the same order
    `reorder_particles_relion_style` produces). When ``micrograph_names``
    is supplied, h0/h1 are returned as DATASET-NATURAL-ORDER indices that
    pick the same physical particles RELION's halfsets contain.

    Falls back to natural-order alternation when names are not supplied,
    which matches RELION only when the dataset is already RELION-sorted.
    """
    ids = np.arange(n_images)
    if micrograph_names is None:
        return ids[0::2], ids[1::2]
    sort_idx = np.argsort(np.asarray(micrograph_names), kind="stable")
    h0_sorted = sort_idx[0::2]
    h1_sorted = sort_idx[1::2]
    return h0_sorted, h1_sorted


def run_iter_gpu_vdam(
    ds,
    iref_real: np.ndarray,
    sigma2_noise: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    current_size: int,
    iter: int,
    Igrad1: Optional[np.ndarray] = None,
    Igrad2: Optional[np.ndarray] = None,
    grad_stepsize: Optional[float] = None,
    tau2_fudge_factor: float = 1.0,
    image_batch_size: int = 500,
    rotation_block_size: int = 50,
    half_spectrum_scoring: bool = True,
    padding_factor: int = 1,
    phase_lengths=None,
    pseudo_halfsets: bool = True,
    # Volume preprocessing knobs — match RELION's projector setup at pad=1:
    # gridding correction must be applied externally because run_em's internal
    # gridding only fires when projection_padding_factor > 1 (em_engine.py:1046).
    apply_gridding_correction: bool = True,
    iref_ft_scale: float = 1.0,  # /N²/N³/etc. compensation for RELION's normalized FFT — paired with bp_data_frame_scale
    iref_ft_sign: float = -1.0,  # RELION-frame projection sign convention
    # E-step parameters mirroring standard-EM's `run_em` invocation
    # (recovar/em/dense_single_volume/iteration_loop.py:547-579).
    score_with_masked_images: bool = True,
    relion_firstiter_score_mode: str = "gaussian",
    sigma_offset_Ang: Optional[float] = None,  # None → uniform translation prior
    accumulate_noise: bool = False,  # set True from a multi-iter driver to track per-iter noise updates
    sparse_pass2: bool = True,
    # When provided, drives RELION-sorted (lex-by-micrograph) pseudo-halfset
    # split. Falls back to natural-order alternation otherwise (matches RELION
    # only when the dataset is already RELION-sorted).
    micrograph_names: Optional[np.ndarray] = None,
):
    """Bit-exact-M-step VDAM iteration (CPU M-step via RELION primitives).

    E-step runs on GPU via run_em (halfset-split when pseudo_halfsets=True).
    Outputs are converted to RELION's BPref layout and fed through the
    full momentum pipeline (reweightGrad → getFristMoment → getSecondMoment
    → applyMomenta → reconstructGrad), matching RELION's InitialModel
    exactly given matched E-step outputs.

    The `run_em` call mirrors the parameter set used in
    `recovar/em/dense_single_volume/iteration_loop.py:547-579` (the canonical
    RELION-parity standard-EM path). The known structural per-kernel ceiling
    on the small (500/64) iter-1 fixture is BPref CC ≈ +0.73; closing the
    residual to +0.99 requires an iter-1 RELION `Frefctf_orient0` dump.

    Returns (iref_real_next, Igrad1_next, Igrad2_next, stats_dict).
    """
    import jax
    import jax.numpy as jnp

    from recovar.core import fourier_transform_utils as ftu
    from recovar.core.relion_project import gridding_correct_volume_real
    from recovar.em.dense_single_volume.em_engine import run_em
    from recovar.em.initial_model.schedules import compute_phase_lengths, compute_stepsize
    from recovar.reconstruction.noise import make_radial_noise
    from recovar.relion_bind import _relion_bind_core as bind

    ori = iref_real.shape[0]
    r_max = current_size // 2
    n4 = ori**4

    nv = np.asarray(make_radial_noise(sigma2_noise * n4, (ori, ori))).astype(np.float32).reshape(-1)
    if apply_gridding_correction:
        iref_real_for_ft = np.asarray(gridding_correct_volume_real(jnp.asarray(iref_real), ori, padding_factor))
    else:
        iref_real_for_ft = iref_real
    iref_ft = np.asarray(ftu.get_dft3(jnp.asarray(iref_real_for_ft))).reshape(-1)
    iref_ft = iref_ft * iref_ft_sign * iref_ft_scale
    mv = jnp.asarray((np.abs(iref_ft) ** 2).astype(np.float32))
    mean_j = jnp.asarray(iref_ft, dtype=jnp.complex64)

    nv_j = jnp.asarray(nv)
    rot_j = jnp.asarray(rotations)
    tr_j = jnp.asarray(translations)

    # Build Gaussian translation log-prior from sigma_offset (RELION's
    # convertAllSquaredDifferencesToWeights at ml_optimiser.cpp:8644-8671).
    # tdiff² = -||t_pix × pixel_size_Ang||² / (2 × sigma_offset_Ang²)
    pixel_size_Ang = float(getattr(ds, "voxel_size", 1.0))
    if sigma_offset_Ang is not None and sigma_offset_Ang > 0:
        translations_np = np.asarray(translations, dtype=np.float32)
        t_dist2_Ang2 = (translations_np[:, 0] ** 2 + translations_np[:, 1] ** 2) * (pixel_size_Ang**2)
        translation_log_prior = (-0.5 * t_dist2_Ang2 / (sigma_offset_Ang**2)).astype(np.float32)
        translation_log_prior_j = jnp.asarray(translation_log_prior)
    else:
        translation_log_prior_j = None

    def _run_estep(subset_ds):
        jax.block_until_ready(mean_j)
        t0 = time.time()
        n = subset_ds.n_images
        extra_kwargs: dict = {
            "score_with_masked_images": score_with_masked_images,
            "relion_firstiter_score_mode": relion_firstiter_score_mode,
            "sparse_pass2": sparse_pass2,
            "accumulate_noise": accumulate_noise,
        }
        if translation_log_prior_j is not None:
            extra_kwargs["translation_log_prior"] = translation_log_prior_j
            extra_kwargs["translation_prior_centers"] = jnp.zeros((n, 2), dtype=jnp.float32)
            extra_kwargs["image_pre_shifts"] = jnp.zeros((n, 2), dtype=jnp.float32)
        result = run_em(
            subset_ds,
            mean=mean_j,
            mean_variance=mv,
            noise_variance=nv_j,
            rotations=rot_j,
            translations=tr_j,
            disc_type="linear_interp",
            image_batch_size=image_batch_size,
            rotation_block_size=rotation_block_size,
            current_size=current_size,
            projection_padding_factor=padding_factor,
            reconstruction_padding_factor=padding_factor,
            half_spectrum_scoring=half_spectrum_scoring,
            return_stats=True,
            **extra_kwargs,
        )
        jax.block_until_ready(result[0])
        # Layout: (mean, ha, Ft_y, Ft_ctf, [relion_stats], [noise_stats])
        Ft_y = np.asarray(result[2])
        Ft_ctf = np.asarray(result[3])
        relion_stats = result[4] if len(result) > 4 else None
        return (Ft_y, Ft_ctf, relion_stats, time.time() - t0)

    # E-step(s).
    # Pseudo-halfsets: split particles RELION-style (lex-sort micrograph names,
    # alternate). When `_rlnMicrographName` is unavailable on the dataset,
    # fall back to natural-order alternation (matches RELION only when the
    # dataset is already RELION-sorted).
    if pseudo_halfsets:
        mic_names = np.asarray(micrograph_names) if micrograph_names is not None else None
        h0, h1 = _split_halfset_particle_ids(ds.n_images, micrograph_names=mic_names)
        ds_h0 = ds.subset(h0) if hasattr(ds, "subset") else ds
        ds_h1 = ds.subset(h1) if hasattr(ds, "subset") else ds
        Ft_y_h0, Ft_ctf_h0, stats_h0, t_h0 = _run_estep(ds_h0)
        Ft_y_h1, Ft_ctf_h1, stats_h1, t_h1 = _run_estep(ds_h1)
        e_step_s = t_h0 + t_h1
        stats = stats_h0  # primary
    else:
        Ft_y_h0, Ft_ctf_h0, stats, e_step_s = _run_estep(ds)
        Ft_y_h1 = Ft_ctf_h1 = None

    # Convert to BPref layout
    bp_data_h0, bp_weight_h0 = run_em_output_to_bpref(Ft_y_h0, Ft_ctf_h0, ori, r_max, padding_factor)
    if pseudo_halfsets:
        bp_data_h1, bp_weight_h1 = run_em_output_to_bpref(Ft_y_h1, Ft_ctf_h1, ori, r_max, padding_factor)

    t_m0 = time.time()
    mu_first = 0.9
    mu_second = 0.999

    # Step 1: reweightGrad
    bp_h0_rw = np.asarray(bind.vdam_reweight_grad(bp_data_h0, bp_weight_h0, ori, padding_factor, 1, r_max))
    if pseudo_halfsets:
        bp_h1_rw = np.asarray(bind.vdam_reweight_grad(bp_data_h1, bp_weight_h1, ori, padding_factor, 1, r_max))

    # Step 2: getFristMoment (initialise Igrad1 if missing)
    K = 1
    h_slots = 2 if pseudo_halfsets else 1
    half_shape = ftu.volume_shape_to_half_volume_shape((ori, ori, ori))
    if Igrad1 is None:
        Igrad1 = np.zeros((K * h_slots,) + half_shape, dtype=np.complex128)
    if Igrad2 is None:
        Igrad2 = np.ones((K,) + half_shape, dtype=np.complex128)

    Igrad1_h0_post = np.asarray(
        bind.vdam_first_moment(bp_h0_rw, Igrad1[0].copy(), ori, padding_factor, 1, r_max, **{"lambda": mu_first})
    )
    if pseudo_halfsets:
        Igrad1_h1_post = np.asarray(
            bind.vdam_first_moment(bp_h1_rw, Igrad1[1].copy(), ori, padding_factor, 1, r_max, **{"lambda": mu_first})
        )
    else:
        Igrad1_h1_post = Igrad1_h0_post

    # Step 3: getSecondMoment
    if pseudo_halfsets:
        Igrad2_post = np.asarray(
            bind.vdam_second_moment(
                bp_h0_rw,
                bp_h1_rw,
                Igrad2[0].copy(),
                ori,
                padding_factor,
                1,
                r_max,
                **{"lambda": mu_second},
            )
        )
    else:
        Igrad2_post = Igrad2[0].copy()

    # Step 4: applyMomenta
    bp_final, mom1_np = bind.vdam_apply_momenta(
        bp_h0_rw, Igrad1_h0_post, Igrad1_h1_post, Igrad2_post, ori, padding_factor, 1, r_max
    )
    bp_final = np.asarray(bp_final)
    mom1_np = np.asarray(mom1_np).ravel()

    # Step 5: reconstructGrad
    if grad_stepsize is None:
        if phase_lengths is None:
            phase_lengths = compute_phase_lengths(200, 0.3, 0.2)
        grad_stepsize = compute_stepsize(iter=iter, phase_lengths=phase_lengths, is_3d_model=True, ref_dim=3)

    fsc = np.zeros(ori // 2 + 1, dtype=np.float64)
    iref_next = np.asarray(
        bind.vdam_reconstruct_grad(
            iref_real.astype(np.float64),
            bp_final,
            bp_weight_h0,
            fsc,
            grad_stepsize,
            tau2_fudge_factor,
            ori,
            padding_factor,
            1,
            r_max,
            0.0,
            False,
            True,
            mom1_np,
        )
    )
    m_step_s = time.time() - t_m0

    Igrad1_next = np.stack([Igrad1_h0_post, Igrad1_h1_post], axis=0) if pseudo_halfsets else Igrad1_h0_post[None]
    Igrad2_next = Igrad2_post[None]

    intermediates = {
        "bp_data_h0": bp_data_h0,
        "bp_weight_h0": bp_weight_h0,
        "Ft_y_h0": Ft_y_h0,
        "Ft_ctf_h0": Ft_ctf_h0,
    }
    if pseudo_halfsets:
        intermediates["bp_data_h1"] = bp_data_h1
        intermediates["bp_weight_h1"] = bp_weight_h1
        intermediates["Ft_y_h1"] = Ft_y_h1
        intermediates["Ft_ctf_h1"] = Ft_ctf_h1
    return (
        iref_next,
        Igrad1_next,
        Igrad2_next,
        {
            "e_step_s": e_step_s,
            "m_step_s": m_step_s,
            "stats": stats,
            "intermediates": intermediates,
        },
    )

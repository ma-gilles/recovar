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
    at padding_factor=1.  Its half axis corresponds to recovar's first
    frequency axis, while the public BPref array order is the transpose
    (axis2, axis1, axis0_half).

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
    bp_slice = (
        slice(c, c + half_ps + 1),
        slice(c - half_ps, c + half_ps + 1),
        slice(c - half_ps, c + half_ps + 1),
    )
    bp_data = np.transpose(Fy[bp_slice], (2, 1, 0)).astype(np.complex128, copy=True)
    bp_weight = np.transpose(Fc[bp_slice], (2, 1, 0)).real.astype(np.float64, copy=True)
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
    bp_slice = (
        slice(c, c + half_ps + 1),
        slice(c - half_ps, c + half_ps + 1),
        slice(c - half_ps, c + half_ps + 1),
    )
    Fy[bp_slice] = np.transpose(bp_data, (2, 1, 0))
    Fc[bp_slice] = np.transpose(bp_weight, (2, 1, 0))
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


def relion_sorted_particle_ids_to_dataset_ids(
    n_images: int,
    *,
    micrograph_names: Optional[np.ndarray] = None,
    sorted_particle_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Map RELION ``sorted_idx`` values onto RECOVAR dataset row ids.

    RELION randomizes particles after first putting the data into its
    "natural" order, which for this SPA path is the stable lexicographic
    micrograph-name order.  The dumped ``sorted_idx`` values are positions in
    that RELION-natural order, not necessarily the original STAR row ids.
    """

    ids = np.arange(n_images, dtype=np.int64)
    if sorted_particle_ids is None:
        order = ids
    else:
        order = np.asarray(sorted_particle_ids, dtype=np.int64).reshape(-1)
        if order.size != n_images:
            raise ValueError(f"sorted_particle_ids size {order.size} != n_images {n_images}")
        if not np.array_equal(np.sort(order), ids):
            raise ValueError("sorted_particle_ids must be a permutation of [0, n_images)")

    if micrograph_names is None:
        return order

    natural_order = np.argsort(np.asarray(micrograph_names).astype(str), kind="stable")
    if natural_order.size != n_images:
        raise ValueError(f"micrograph_names size {natural_order.size} != n_images {n_images}")
    return natural_order[order]


def _split_halfset_particle_ids(
    n_images: int,
    rng_seed: int = 0,
    micrograph_names: Optional[np.ndarray] = None,
    sorted_particle_ids: Optional[np.ndarray] = None,
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
    if sorted_particle_ids is not None:
        order = relion_sorted_particle_ids_to_dataset_ids(
            n_images,
            micrograph_names=micrograph_names,
            sorted_particle_ids=sorted_particle_ids,
        )
        return order[0::2], order[1::2]
    ids = np.arange(n_images)
    if micrograph_names is None:
        return ids[0::2], ids[1::2]
    sort_idx = np.argsort(np.asarray(micrograph_names), kind="stable")
    h0_sorted = sort_idx[0::2]
    h1_sorted = sort_idx[1::2]
    return h0_sorted, h1_sorted


def _infer_translation_step(translations: np.ndarray) -> float:
    """Infer the parent-grid translation step from a RELION translation table."""

    vals = np.unique(np.asarray(translations, dtype=np.float32).reshape(-1))
    diffs = np.diff(np.sort(vals))
    diffs = diffs[diffs > 1e-6]
    return float(diffs.min()) if diffs.size else 1.0


def _relion_unperturbed_translation_prior_grid(
    scored_translations: np.ndarray,
    *,
    random_perturbation: float,
    translation_step: Optional[float],
) -> np.ndarray:
    """Return RELION parent offsets used for Gaussian translation priors.

    RELION perturbs the translation samples before scoring, but evaluates the
    Gaussian offset prior on the unperturbed parent grid.  If the perturbation
    value is not passed through, the first-iteration translation grid is
    symmetric, so the uniform perturbation is recoverable from its centroid.
    """

    prior_grid = np.asarray(scored_translations, dtype=np.float32).copy()
    if prior_grid.size == 0:
        return prior_grid
    if translation_step is not None and abs(float(random_perturbation)) > 1e-12:
        shift = np.float32(float(random_perturbation) * float(translation_step))
        return prior_grid - shift
    center = np.mean(prior_grid, axis=0, dtype=np.float64).astype(np.float32)
    if float(np.linalg.norm(center)) > 1e-6:
        return prior_grid - center[None, :]
    return prior_grid


def _infer_relion_adaptive_grids(
    rotations: np.ndarray,
    translations: np.ndarray,
    *,
    nside_level: Optional[int],
    oversampling_order: int,
    random_perturbation: float,
    translation_step: Optional[float],
    rotation_index_order: str = "relion",
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Return coarse grids for RELION adaptive pass-1/pass-2 replay.

    ``run_iter_gpu_vdam`` historically accepted the dense oversampled grid
    directly.  RELION adaptive replay needs the coarse grid because pass 1
    selects significant coarse samples and pass 2 regenerates the child poses.
    This helper keeps the old dense-grid call sites usable by inferring the
    parent coarse grid from the child-grid sizes/order.
    """

    from recovar.em.sampling import (
        apply_relion_rotation_perturbation,
        get_relion_rotation_grid,
        relion_angular_sampling_deg,
        rotation_grid_size,
    )

    rotations_np = np.asarray(rotations, dtype=np.float32)
    translations_np = np.asarray(translations, dtype=np.float32)
    oversampling_order = int(oversampling_order)
    if oversampling_order < 0:
        raise ValueError(f"oversampling_order must be >= 0, got {oversampling_order}")

    rot_child_factor = 8**oversampling_order
    trans_child_factor = 4**oversampling_order

    if nside_level is None:
        inferred = None
        for order in range(0, 10):
            n_coarse = int(rotation_grid_size(order))
            if rotations_np.shape[0] in {n_coarse, n_coarse * rot_child_factor}:
                inferred = order
                break
        if inferred is None:
            raise ValueError(
                "Could not infer RELION adaptive nside_level from "
                f"{rotations_np.shape[0]} rotations and oversampling_order={oversampling_order}; "
                "pass adaptive_nside_level explicitly."
            )
        nside_level = inferred
    nside_level = int(nside_level)
    n_coarse_rot = int(rotation_grid_size(nside_level))

    if rotations_np.shape[0] == n_coarse_rot:
        coarse_rotations = rotations_np
    elif rotations_np.shape[0] == n_coarse_rot * rot_child_factor:
        coarse_rotations = get_relion_rotation_grid(
            nside_level,
            rotation_index_order=rotation_index_order,
        ).astype(np.float32)
        if abs(float(random_perturbation)) > 1e-12:
            coarse_rotations = apply_relion_rotation_perturbation(
                coarse_rotations,
                float(random_perturbation),
                relion_angular_sampling_deg(nside_level, adaptive_oversampling=0),
            ).astype(np.float32)
    else:
        raise ValueError(
            "RELION adaptive mode expected either coarse rotations "
            f"({n_coarse_rot}) or oversampled rotations ({n_coarse_rot * rot_child_factor}), "
            f"got {rotations_np.shape[0]}"
        )

    if trans_child_factor > 1 and translations_np.shape[0] % trans_child_factor == 0:
        # Dense call sites pass children in parent-major order.  Averaging each
        # child group recovers the perturbed coarse parent center because
        # RELION's translation perturbation is a constant additive shift.
        coarse_translations = translations_np.reshape(-1, trans_child_factor, translations_np.shape[-1]).mean(axis=1)
    else:
        coarse_translations = translations_np

    if translation_step is None:
        translation_step = _infer_translation_step(coarse_translations)

    return (
        np.asarray(coarse_rotations, dtype=np.float32),
        np.asarray(coarse_translations, dtype=np.float32),
        nside_level,
        float(translation_step),
    )


def _infer_relion_particle_diameter_Ang(ds) -> Optional[float]:
    """Best-effort extraction of RELION particle diameter from dataset metadata."""

    backend = getattr(getattr(ds, "image_source", None), "backend", None)
    params = getattr(backend, "_relion_image_mask_params", None)
    if params is not None and len(params) >= 2:
        particle_diameter = params[1]
        if particle_diameter is not None and float(particle_diameter) > 0:
            return float(particle_diameter)

    for attr in ("particle_diameter_Ang", "particle_diameter_ang", "particle_diameter_angstrom"):
        value = getattr(ds, attr, None)
        if value is not None and float(value) > 0:
            return float(value)
    return None


def _relion_adaptive_coarse_current_size(
    ds,
    *,
    current_size: int,
    nside_level: int,
    ori_size: int,
    particle_diameter_Ang: Optional[float] = None,
    max_coarse_size: Optional[int] = None,
) -> int:
    """Compute RELION's pass-1 coarse image size for adaptive oversampling."""

    del ds, nside_level, ori_size, particle_diameter_Ang
    coarse_size = int(current_size)
    if max_coarse_size is not None and int(max_coarse_size) > 0:
        coarse_size = min(int(coarse_size), int(max_coarse_size))
    # RELION InitialModel keeps exp_current_image_size fixed across the coarse
    # and oversampled passes. The dumped iter-1 fixture reports current_size=28
    # for both pass0/over0 and pass1/over1; shrinking pass 1 here changes the
    # significant-pose threshold and expands the pass-2 support incorrectly.
    return int(coarse_size)


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
    iref_ft_sign: float = 1.0,  # Raw RELION InitialModel scoring frame; ctf carries the sign flip.
    # E-step parameters mirroring standard-EM's `run_em` invocation
    # (recovar/em/dense_single_volume/iteration_loop.py:547-579).
    score_with_masked_images: bool = True,
    relion_firstiter_score_mode: str = "gaussian",
    sigma_offset_Ang: Optional[float] = None,  # None → uniform translation prior
    accumulate_noise: bool = False,  # set True from a multi-iter driver to track per-iter noise updates
    sparse_pass2: bool = True,
    relion_half_volume_mstep: bool = True,
    estep_mode: str = "relion_adaptive",
    adaptive_nside_level: Optional[int] = None,
    adaptive_oversampling_order: int = 1,
    adaptive_random_perturbation: float = 0.0,
    adaptive_translation_step: Optional[float] = None,
    adaptive_fraction: float = 0.999,
    adaptive_max_significants: int = -1,
    adaptive_projection_relion_texture_interp: bool = True,
    adaptive_projection_force_jax: bool = False,
    adaptive_use_float64_scoring: bool = False,
    adaptive_rotation_index_order: str = "relion",
    adaptive_square_window: bool = False,
    adaptive_coarse_current_size: Optional[int] = None,
    adaptive_particle_diameter_Ang: Optional[float] = None,
    adaptive_max_coarse_size: Optional[int] = None,
    # When provided, drives RELION-sorted (lex-by-micrograph) pseudo-halfset
    # split. Falls back to natural-order alternation otherwise (matches RELION
    # only when the dataset is already RELION-sorted).
    micrograph_names: Optional[np.ndarray] = None,
    sorted_particle_ids: Optional[np.ndarray] = None,
):
    """Bit-exact-M-step VDAM iteration (CPU M-step via RELION primitives).

    E-step runs on GPU via run_em (halfset-split when pseudo_halfsets=True).
    Outputs are converted to RELION's BPref layout and fed through the
    full momentum pipeline (reweightGrad → getFristMoment → getSecondMoment
    → applyMomenta → reconstructGrad), matching RELION's InitialModel
    exactly given matched E-step outputs.

    The `run_em` call mirrors the parameter set used in
    `recovar/em/dense_single_volume/iteration_loop.py:547-579` (the canonical
    RELION-parity standard-EM path). ``relion_half_volume_mstep`` keeps this
    InitialModel path on RELION's native BackProjector convention by
    accumulating M-step sufficient statistics in a packed half-volume with
    fold/conjugation semantics, then expanding back to recovar's public
    full-volume accumulator contract before BPref conversion.

    Returns (iref_real_next, Igrad1_next, Igrad2_next, stats_dict).
    """
    import jax
    import jax.numpy as jnp

    from recovar.core import fourier_transform_utils as ftu
    from recovar.core.relion_project import gridding_correct_volume_real
    from recovar.em.dense_single_volume.em_engine import run_em
    from recovar.em.dense_single_volume.helpers.significance import _compute_significance_batched
    from recovar.em.dense_single_volume.iteration_loop import _run_sparse_pass2_local_search_iteration
    from recovar.em.initial_model.schedules import compute_phase_lengths, compute_stepsize
    from recovar.reconstruction.noise import make_radial_noise
    from recovar.relion_bind import _relion_bind_core as bind

    ori = iref_real.shape[0]
    r_max = current_size // 2
    n4 = ori**4
    if bool(adaptive_use_float64_scoring) and not bool(jax.config.read("jax_enable_x64")):
        raise RuntimeError(
            "adaptive_use_float64_scoring=True requires JAX_ENABLE_X64=1 before process startup; "
            "otherwise JAX silently truncates complex128/float64 arrays to float32."
        )

    estep_mode = str(estep_mode)
    if estep_mode not in {"dense", "relion_adaptive"}:
        raise ValueError(f"estep_mode must be 'dense' or 'relion_adaptive', got {estep_mode!r}")

    # RELION adaptive InitialModel consumes sigma2 in RELION's Minvsigma2
    # frame. Its particle/reference Fourier arrays are normalized by N^2
    # instead. The dense recovar path keeps the historical N^4-scaled
    # convention used by the standard EM engine.
    noise_scale = 1.0 if estep_mode == "relion_adaptive" else n4
    nv = np.asarray(make_radial_noise(sigma2_noise * noise_scale, (ori, ori))).astype(np.float32).reshape(-1)
    if apply_gridding_correction:
        iref_real_for_ft = np.asarray(gridding_correct_volume_real(jnp.asarray(iref_real), ori, padding_factor))
    else:
        iref_real_for_ft = iref_real
    iref_ft = np.asarray(ftu.get_dft3(jnp.asarray(iref_real_for_ft))).reshape(-1)
    iref_ft = iref_ft * iref_ft_sign * iref_ft_scale
    mv = jnp.asarray((np.abs(iref_ft) ** 2).astype(np.float32))
    mean_dtype = jnp.complex128 if bool(adaptive_use_float64_scoring) else jnp.complex64
    mean_j = jnp.asarray(iref_ft, dtype=mean_dtype)

    nv_j = jnp.asarray(nv)
    rot_j = jnp.asarray(rotations)
    tr_j = jnp.asarray(translations)
    relion_projector_half_j = None
    relion_projector_r_max = None
    adaptive_grids = None
    adaptive_translation_log_prior_j = None
    if estep_mode == "relion_adaptive":
        from recovar.utils.helpers import recovar_volume_to_relion

        # RELION InitialModel scores against Projector::data (PPref), not
        # recovar's centered full Fourier grid. The binding returns PPref/N;
        # multiplying by ori reproduces RELION's dumped PPref/Fref amplitudes.
        iref_relion_frame_for_projector = recovar_volume_to_relion(np.asarray(iref_real, dtype=np.float64))
        ppref_tuple = bind.compute_fourier_transform_map(
            iref_relion_frame_for_projector,
            ori_size=int(ori),
            padding_factor=int(padding_factor),
            current_size=int(current_size),
            do_gridding=bool(apply_gridding_correction),
        )
        relion_projector_half = np.asarray(ppref_tuple[0], dtype=np.complex128) * float(ori)
        relion_projector_r_max = int(ppref_tuple[4])
        relion_projector_half_j = jnp.asarray(relion_projector_half, dtype=mean_dtype)

        adaptive_grids = _infer_relion_adaptive_grids(
            rotations,
            translations,
            nside_level=adaptive_nside_level,
            oversampling_order=adaptive_oversampling_order,
            random_perturbation=adaptive_random_perturbation,
            translation_step=adaptive_translation_step,
            rotation_index_order=adaptive_rotation_index_order,
        )
        coarse_rotations, coarse_translations, adaptive_nside_level, adaptive_translation_step = adaptive_grids
        coarse_rot_j = jnp.asarray(coarse_rotations)
        coarse_tr_j = jnp.asarray(coarse_translations)
        if adaptive_coarse_current_size is None:
            adaptive_pass1_current_size = _relion_adaptive_coarse_current_size(
                ds,
                current_size=int(current_size),
                nside_level=int(adaptive_nside_level),
                ori_size=int(ori),
                particle_diameter_Ang=adaptive_particle_diameter_Ang,
                max_coarse_size=adaptive_max_coarse_size,
            )
        else:
            adaptive_pass1_current_size = int(adaptive_coarse_current_size)
    else:
        coarse_rotations = coarse_translations = coarse_rot_j = coarse_tr_j = None
        adaptive_pass1_current_size = None

    # Build Gaussian translation log-prior from sigma_offset (RELION's
    # convertAllSquaredDifferencesToWeights at ml_optimiser.cpp:9134-9259).
    # Dense EM paths store translations in pixels. RELION InitialModel stores
    # sampling translations in Angstroms for the offset prior, then still
    # multiplies that term by pixel_size² inside
    # convertAllSquaredDifferencesToWeights(). Replaying from pixel-space phase
    # shifts therefore requires a pixel_size⁴ factor for the prior exponent.
    pixel_size_Ang = float(getattr(ds, "voxel_size", 1.0))
    if sigma_offset_Ang is not None and sigma_offset_Ang > 0:
        translations_np = np.asarray(translations, dtype=np.float32)
        t_dist2_relion_prior = (translations_np[:, 0] ** 2 + translations_np[:, 1] ** 2) * (pixel_size_Ang**4)
        translation_log_prior = (-0.5 * t_dist2_relion_prior / (sigma_offset_Ang**2)).astype(np.float32)
        translation_log_prior_j = jnp.asarray(translation_log_prior)
        if estep_mode == "relion_adaptive":
            coarse_trans_np = np.asarray(coarse_translations, dtype=np.float32)
            coarse_prior_trans_np = _relion_unperturbed_translation_prior_grid(
                coarse_trans_np,
                random_perturbation=float(adaptive_random_perturbation),
                translation_step=adaptive_translation_step,
            )
            coarse_t_dist2_relion_prior = (
                coarse_prior_trans_np[:, 0] ** 2 + coarse_prior_trans_np[:, 1] ** 2
            ) * (pixel_size_Ang**4)
            adaptive_translation_log_prior = (
                -0.5 * coarse_t_dist2_relion_prior / (sigma_offset_Ang**2)
            ).astype(np.float32)
            adaptive_translation_log_prior_j = jnp.asarray(adaptive_translation_log_prior)
    else:
        translation_log_prior_j = None

    def _run_estep(subset_ds):
        jax.block_until_ready(mean_j)
        t0 = time.time()
        n = subset_ds.n_images
        image_pre_shifts = jnp.zeros((n, 2), dtype=jnp.float32)
        translation_prior_centers = jnp.zeros((n, 2), dtype=jnp.float32)
        relion_image_corrections = np.full(n, -1.0 / (float(ori) ** 2), dtype=np.float32)
        if estep_mode == "relion_adaptive":
            sig_result = _compute_significance_batched(
                subset_ds,
                mean_j,
                nv_j,
                coarse_rotations,
                coarse_translations,
                "linear_interp",
                adaptive_fraction=float(adaptive_fraction),
                max_significants=int(adaptive_max_significants),
                image_batch_size=image_batch_size,
                rotation_block_size=rotation_block_size,
                current_size=int(adaptive_pass1_current_size),
                score_with_masked_images=score_with_masked_images,
                return_significant_sample_indices=True,
                rotation_log_prior=None,
                translation_log_prior=adaptive_translation_log_prior_j,
                image_corrections=relion_image_corrections,
                image_pre_shifts=image_pre_shifts,
                half_spectrum_scoring=half_spectrum_scoring,
                projection_padding_factor=padding_factor,
                do_gridding_correction=False,
                square_window=bool(adaptive_square_window),
                use_float64_scoring=bool(adaptive_use_float64_scoring),
                projection_force_jax=bool(adaptive_projection_force_jax),
                relion_projector_half=relion_projector_half_j,
                relion_projector_r_max=relion_projector_r_max,
                return_full_stats=True,
            )
            (
                _sig_rot_any,
                n_sig,
                _coarse_ha,
                significant_sample_indices,
                coarse_stats,
            ) = sig_result
            pass2_result = _run_sparse_pass2_local_search_iteration(
                subset_ds,
                mean_j,
                mv,
                nv_j,
                coarse_translations,
                significant_sample_indices,
                int(adaptive_nside_level),
                "linear_interp",
                oversampling_order=int(adaptive_oversampling_order),
                current_size=current_size,
                translation_step=float(adaptive_translation_step),
                rotation_log_prior=None,
                translation_log_prior=adaptive_translation_log_prior_j,
                score_with_masked_images=score_with_masked_images,
                return_stats=True,
                accumulate_noise=accumulate_noise,
                half_spectrum_scoring=half_spectrum_scoring,
                projection_relion_texture_interp=bool(adaptive_projection_relion_texture_interp),
                projection_force_jax=bool(adaptive_projection_force_jax),
                relion_projector_half=relion_projector_half_j,
                relion_projector_r_max=relion_projector_r_max,
                projection_padding_factor=padding_factor,
                reconstruction_padding_factor=padding_factor,
                image_corrections=relion_image_corrections,
                scale_corrections=None,
                image_pre_shifts=image_pre_shifts,
                mstep_subtract_ctf_projection=True,
                mstep_relion_x_half=bool(relion_half_volume_mstep),
                use_float64_scoring=bool(adaptive_use_float64_scoring),
                do_gridding_correction=False,
                square_window=bool(adaptive_square_window),
                random_perturbation=float(adaptive_random_perturbation),
                rotation_index_order=str(adaptive_rotation_index_order),
                image_batch_size=image_batch_size,
                rotation_block_size=rotation_block_size,
                adaptive_fraction=float(adaptive_fraction),
                translation_prior_centers=(
                    translation_prior_centers if adaptive_translation_log_prior_j is not None else None
                ),
            )
            Ft_y = np.asarray(pass2_result[0])
            Ft_ctf = np.asarray(pass2_result[1])
            relion_stats = pass2_result[6] if len(pass2_result) > 6 else None
            jax.block_until_ready(pass2_result[0])
            meta = {
                "mode": "relion_adaptive",
                "nside_level": int(adaptive_nside_level),
                "oversampling_order": int(adaptive_oversampling_order),
                "translation_step": float(adaptive_translation_step),
                "random_perturbation": float(adaptive_random_perturbation),
                "pass1_current_size": int(adaptive_pass1_current_size or current_size),
                "rotation_index_order": str(adaptive_rotation_index_order),
                "square_window": bool(adaptive_square_window),
                "projection_force_jax": bool(adaptive_projection_force_jax),
                "use_float64_scoring": bool(adaptive_use_float64_scoring),
                "n_sig_min": int(np.min(n_sig)) if len(n_sig) else 0,
                "n_sig_max": int(np.max(n_sig)) if len(n_sig) else 0,
                "n_sig_mean": float(np.mean(n_sig)) if len(n_sig) else 0.0,
                "coarse_max_posterior_mean": (
                    float(np.mean(np.asarray(coarse_stats["max_posterior_per_image"])))
                    if coarse_stats is not None and "max_posterior_per_image" in coarse_stats
                    else None
                ),
            }
            return (Ft_y, Ft_ctf, relion_stats, time.time() - t0, meta)

        extra_kwargs: dict = {
            "score_with_masked_images": score_with_masked_images,
            "relion_firstiter_score_mode": relion_firstiter_score_mode,
            "sparse_pass2": sparse_pass2,
            "accumulate_noise": accumulate_noise,
        }
        if translation_log_prior_j is not None:
            extra_kwargs["translation_log_prior"] = translation_log_prior_j
            extra_kwargs["translation_prior_centers"] = translation_prior_centers
            extra_kwargs["image_pre_shifts"] = image_pre_shifts
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
            relion_half_volume_mstep=relion_half_volume_mstep,
            **extra_kwargs,
        )
        jax.block_until_ready(result[0])
        # Layout: (mean, ha, Ft_y, Ft_ctf, [relion_stats], [noise_stats])
        Ft_y = np.asarray(result[2])
        Ft_ctf = np.asarray(result[3])
        relion_stats = result[4] if len(result) > 4 else None
        return (Ft_y, Ft_ctf, relion_stats, time.time() - t0, {"mode": "dense"})

    # E-step(s).
    # Pseudo-halfsets: split particles RELION-style (lex-sort micrograph names,
    # alternate). When `_rlnMicrographName` is unavailable on the dataset,
    # fall back to natural-order alternation (matches RELION only when the
    # dataset is already RELION-sorted).
    if pseudo_halfsets:
        mic_names = np.asarray(micrograph_names) if micrograph_names is not None else None
        h0, h1 = _split_halfset_particle_ids(
            ds.n_images,
            micrograph_names=mic_names,
            sorted_particle_ids=sorted_particle_ids,
        )
        ds_h0 = ds.subset(h0) if hasattr(ds, "subset") else ds
        ds_h1 = ds.subset(h1) if hasattr(ds, "subset") else ds
        Ft_y_h0, Ft_ctf_h0, stats_h0, t_h0, meta_h0 = _run_estep(ds_h0)
        Ft_y_h1, Ft_ctf_h1, stats_h1, t_h1, meta_h1 = _run_estep(ds_h1)
        e_step_s = t_h0 + t_h1
        stats = stats_h0  # primary
    else:
        estep_ds = ds
        if sorted_particle_ids is not None:
            order = relion_sorted_particle_ids_to_dataset_ids(
                ds.n_images,
                micrograph_names=(np.asarray(micrograph_names) if micrograph_names is not None else None),
                sorted_particle_ids=sorted_particle_ids,
            )
            estep_ds = ds.subset(order) if hasattr(ds, "subset") else ds
        Ft_y_h0, Ft_ctf_h0, stats, e_step_s, meta_h0 = _run_estep(estep_ds)
        Ft_y_h1 = Ft_ctf_h1 = None
        meta_h1 = None

    # Convert recovar's centered-flat (Ft_y, Ft_ctf) accumulator to RELION's
    # BPref slab layout. Dense mode uses recovar's historical unnormalized
    # image + N^4 noise frame and therefore needs (-N^2, N^4) conversion.
    # RELION-adaptive mode already scores/accumulates in RELION's normalized
    # image + direct-Minvsigma2 frame, including RELION's signed CTF/image
    # convention, so no additional bp_data sign conversion remains.
    n2_frame = float(ori) ** 2
    n4_frame = float(ori) ** 4
    bp_data_frame_scale = 1.0 if estep_mode == "relion_adaptive" else -n2_frame
    bp_weight_frame_scale = 1.0 if estep_mode == "relion_adaptive" else n4_frame
    bp_data_h0, bp_weight_h0 = run_em_output_to_bpref(Ft_y_h0, Ft_ctf_h0, ori, r_max, padding_factor)
    bp_data_h0 = bp_data_h0 * bp_data_frame_scale
    bp_weight_h0 = bp_weight_h0 * bp_weight_frame_scale
    if pseudo_halfsets:
        bp_data_h1, bp_weight_h1 = run_em_output_to_bpref(Ft_y_h1, Ft_ctf_h1, ori, r_max, padding_factor)
        bp_data_h1 = bp_data_h1 * bp_data_frame_scale
        bp_weight_h1 = bp_weight_h1 * bp_weight_frame_scale

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
        # RELION inits Igrad2 to Complex(1, 1) per voxel (ml_model.cpp:1683
        # MlModel::reset_class). With μ_second=0.999 EMA the init persists
        # for hundreds of iters. `np.ones(complex128) = 1+0i` would diverge.
        Igrad2 = np.full((K,) + half_shape, 1.0 + 1.0j, dtype=np.complex128)

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
    # Convert iref_real (recovar-frame) → RELION-native frame for the M-step
    # chain, which expects the raw RELION-mrc layout (axes (2,1,0), no negate).
    # The +0.9999 pinned chain test feeds RELION's iref_before.bin directly
    # without any frame conversion, so we must do the same here.
    from recovar.utils.helpers import recovar_volume_to_relion, relion_volume_to_recovar

    iref_relion_frame = recovar_volume_to_relion(np.asarray(iref_real, dtype=np.float64))
    iref_next_relion_frame = np.asarray(
        bind.vdam_reconstruct_grad(
            iref_relion_frame,
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
    # Convert back to recovar-frame for downstream callers (so the next
    # iter's iref input chains naturally).
    iref_next = relion_volume_to_recovar(iref_next_relion_frame)
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
            "estep_mode": estep_mode,
            "estep_meta_h0": meta_h0,
            "estep_meta_h1": meta_h1,
            "intermediates": intermediates,
        },
    )

"""Mean-volume reconstruction and K-class helpers for the iteration loop.

Per-iteration aggregator + reconstruction helpers extracted verbatim from
``iteration_loop.py``. None of these wrap symbols that pytest monkeypatches
at ``iteration_loop.<name>``; all dependencies are imported directly.
"""

from __future__ import annotations

import logging
import os
import time

import jax.numpy as jnp
import numpy as np

from recovar.core import fourier_transform_utils, mask
from recovar.em.dense.scoring_policy import _dense_global_scoring_dtype
from recovar.em.helpers.orientation_priors import (
    class_weights_from_direction_prior,
    collapse_rotation_posterior_to_direction_prior,
    make_relion_direction_log_prior,
)
from recovar.em.helpers.resolution import shell_index_to_resolution_angstrom
from recovar.reconstruction import regularization

logger = logging.getLogger(__name__)

WIDTH_FMASK_EDGE: float = 2.0  # ml_optimiser.h:91


def prepare_initial_mean_variance(
    initial_mean_variance, *, use_per_half_mean_variance, k_class_enabled, log
):
    """Initialize shared/per-half tau2 using the existing reduction and dtype policy.

    The controller retains the original JAX array throughout refinement.
    Shared priors alias one array; the opt-in half priors keep separate values.
    """
    if use_per_half_mean_variance:
        if k_class_enabled:
            raise ValueError("per-half scoring tau2 is supported only for K=1")
        if initial_mean_variance.ndim != 2 or initial_mean_variance.shape[0] != 2:
            raise ValueError(
                "per-half scoring tau2 requires init_mean_variance with leading half axis 2"
            )
        mean_variance_per_half = [
            jnp.asarray(initial_mean_variance[0]),
            jnp.asarray(initial_mean_variance[1]),
        ]
        mean_variance = jnp.asarray(
            0.5
            * (
                mean_variance_per_half[0].astype(jnp.float64)
                + mean_variance_per_half[1].astype(jnp.float64)
            ),
            dtype=_dense_global_scoring_dtype(),
        )
        log.info("Initialized exact per-half K=1 tau2 priors")
    else:
        mean_variance = initial_mean_variance
        mean_variance_per_half = [mean_variance, mean_variance]
    return mean_variance, mean_variance_per_half


def _mean_variance_for_scoring_half(mean_variance_per_half, half_index):
    """Select the exact half-owned K=1 tau2 prior passed to the scorer."""

    if len(mean_variance_per_half) != 2 or int(half_index) not in (0, 1):
        raise ValueError("per-half scoring tau2 requires exactly two halves and index 0 or 1")
    return mean_variance_per_half[int(half_index)]

def _updated_mean_variance_per_half(
    shared_mean_variance,
    updated_mean_variance_per_half,
    *,
    use_per_half_mean_variance,
):
    """Keep historical K=1 scoring on shared tau2 unless explicitly enabled."""

    if use_per_half_mean_variance:
        if len(updated_mean_variance_per_half) != 2:
            raise ValueError("per-half scoring tau2 update requires exactly two halves")
        return [
            jnp.asarray(updated_mean_variance_per_half[0]),
            jnp.asarray(updated_mean_variance_per_half[1]),
        ]
    return [shared_mean_variance, shared_mean_variance]


def _normalize_class_log_priors(n_classes: int, class_log_priors=None) -> np.ndarray:
    """Return normalized log priors for the class axis."""

    if n_classes < 1:
        raise ValueError(f"n_classes must be >= 1, got {n_classes}")
    if class_log_priors is None:
        return np.full(n_classes, -np.log(float(n_classes)), dtype=np.float64)
    log_priors = np.asarray(class_log_priors, dtype=np.float64)
    if log_priors.shape != (n_classes,):
        raise ValueError(f"class_log_priors must have shape ({n_classes},), got {log_priors.shape}")
    if not np.all(np.isfinite(log_priors)):
        raise ValueError("class_log_priors must be finite")
    max_log_prior = float(np.max(log_priors))
    log_norm = max_log_prior + float(np.log(np.sum(np.exp(log_priors - max_log_prior))))
    return log_priors - log_norm


def _initialize_class_log_priors(n_classes: int, init_class_log_priors=None, init_direction_prior=None) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized log priors for the class axis and class weights, defaulting to uniform."""
    class_log_priors = _normalize_class_log_priors(n_classes, init_class_log_priors)
    class_weights = np.exp(class_log_priors)
    if n_classes > 1 and init_class_log_priors is None and init_direction_prior is not None:
        inferred_class_weights = class_weights_from_direction_prior(init_direction_prior, n_classes)
        if inferred_class_weights is not None:
            if np.any(inferred_class_weights <= 0.0):
                raise ValueError("RELION direction-prior row sums imply a zero-probability class")
            class_weights = inferred_class_weights
            class_log_priors = np.log(class_weights)
    return class_log_priors, class_weights


def _normalize_initial_means(init_volume, n_classes: int):
    """Normalize initial references to the refine loop's half/class layout."""

    def _as_class_array(value):
        arr = jnp.asarray(value)
        if n_classes == 1:
            if arr.ndim == 1:
                return arr
            if arr.ndim == 2 and int(arr.shape[0]) == 1:
                return arr[0]
        else:
            if arr.ndim == 1:
                return jnp.tile(arr[None, :], (n_classes, 1))
            if arr.ndim == 2 and int(arr.shape[0]) == n_classes:
                return arr
        raise ValueError(
            "init_volume must be a flat reference, a per-class reference array, "
            "or a pair of per-half references compatible with n_classes="
            f"{n_classes}; got shape {tuple(arr.shape)}",
        )

    if isinstance(init_volume, (list, tuple)) and len(init_volume) == 2:
        return [_as_class_array(init_volume[0]), _as_class_array(init_volume[1])]

    arr = jnp.asarray(init_volume)
    if n_classes == 1 and arr.ndim == 2 and int(arr.shape[0]) == 2:
        return [arr[0], arr[1]]
    if n_classes > 1 and arr.ndim == 3 and int(arr.shape[0]) == 2 and int(arr.shape[1]) == n_classes:
        return [arr[0], arr[1]]
    shared = _as_class_array(arr)
    return [jnp.array(shared), jnp.array(shared)]


def _class_weights_from_posterior(class_posterior_per_half, n_classes: int, previous_weights: np.ndarray) -> np.ndarray:
    """Normalize class posterior sums across both half-sets."""

    counts = np.zeros(n_classes, dtype=np.float64)
    for posterior in class_posterior_per_half:
        if posterior is not None:
            counts += np.asarray(posterior, dtype=np.float64)
    total = float(np.sum(counts))
    if total <= 0.0:
        return np.asarray(previous_weights, dtype=np.float64)
    weights = np.maximum(counts / total, 1e-12)
    return weights / float(np.sum(weights))


def _combined_class_direction_prior_from_halves(
    class_rotation_posterior_per_half, n_classes: int, healpix_order: int, *, dtype: np.dtype = np.float32
):
    """Collapse Class3D rotation posterior sums after undoing RECOVAR's half split.

    RELION Class3D has a single ``mymodel.pdf_direction[class]`` updated from
    ``wsum_model.pdf_direction[class]`` over all particles.  RECOVAR's two
    E-step halves are only a parallelization artifact for K>1, so combine their
    per-class rotation posterior sums before forming the next iteration's
    direction prior.
    """

    combined_priors = []
    for class_idx in range(n_classes):
        combined = None
        for per_half in class_rotation_posterior_per_half:
            if per_half is None:
                continue
            per_class = np.asarray(per_half[class_idx], dtype=np.float64)
            combined = per_class if combined is None else combined + per_class
        if combined is None:
            return None
        combined_priors.append(
            collapse_rotation_posterior_to_direction_prior(combined, healpix_order, dtype=dtype)
        )
    return np.stack(combined_priors, axis=0)


def _previous_resolution_angstrom_for_half_join(
    pixel_resolutions, current_resolution, *, grid_size: int, voxel_size: float
):
    """Previous-iteration resolution in Å that caps the low-resolution half join.

    The last recorded shell resolution wins when the history has one; a
    non-positive recorded shell leaves the join uncapped. Without history, a
    finite state resolution is used. Mirrors RELION's
    ``XMIPP_MAX(low_resol_join_halves, 1./mymodel.current_resolution)``.
    """

    if pixel_resolutions:
        previous_shell = pixel_resolutions[-1]
        if previous_shell > 0:
            return shell_index_to_resolution_angstrom(previous_shell, grid_size, voxel_size)
        return None
    if np.isfinite(float(current_resolution)):
        return float(current_resolution)
    return None


def join_half_accumulators_at_low_resolution(
    Ft_y_0,
    Ft_y_1,
    Ft_ctf_0,
    Ft_ctf_1,
    *,
    accumulator_volume_shape,
    grid_size: int,
    voxel_size: float,
    low_resol_join_halves_angstrom: float,
    pixel_resolutions,
    current_resolution,
    padding_factor,
):
    """Apply RELION's ``--low_resol_join_halves`` to K=1 half accumulators before the Wiener solve.

    Averaging the low-resolution shells of the two half accumulators forces
    both half-maps to share their low-frequency content, as in
    ``MlOptimiserMpi::joinTwoHalvesAtLowResolution``; without it the half-map
    FSC drifts below 1 at SNR-poor low shells and current-size growth lags.
    The join radius is capped by the previous iteration's resolution so shells
    beyond the map's actual resolution are never joined. The regular and final
    all-data passes call this with their own accumulators. Returns the four
    joined arrays in input order; refinement state is not mutated.
    """

    previous_resolution_angstrom = _previous_resolution_angstrom_for_half_join(
        pixel_resolutions,
        current_resolution,
        grid_size=grid_size,
        voxel_size=voxel_size,
    )
    return regularization.join_halves_at_low_resolution(
        Ft_y_0,
        Ft_y_1,
        Ft_ctf_0,
        Ft_ctf_1,
        accumulator_volume_shape,
        voxel_size,
        grid_size,
        low_resol_join_halves_angstrom,
        current_resolution_angstrom=previous_resolution_angstrom,
        padding_factor=padding_factor,
    )


_CLASS_TAU2_DETAIL_KEYS = (
    "prior_shells",
    "sigma2_shells",
    "avg_weight_shells",
    "shell_sum",
    "shell_count",
    "fsc_shells",
    "ssnr_shells",
)


def _class_tau2_from_iref_power_spectrum(
    iref_fourier, volume_shape, *, padding_factor, current_size: int, frame_scale: float
):
    """Class3D tau2 for one class from its previous ``Iref`` power spectrum.

    Dense RECOVAR accumulators live in the historical unnormalised image frame
    (RELION BPref weight = ``Ft_ctf * N^4``), so the RELION-frame tau2 is
    scaled by ``frame_scale`` before the Wiener solve. Returns the
    RECOVAR-frame tau2 volume, the RELION-frame radial shells and the
    RECOVAR-frame radial shells, all in the RELION result dtype.
    """

    mean_signal_variance_relion, details = regularization.compute_relion_tau2_from_iref_power_spectrum(
        iref_fourier,
        volume_shape,
        padding_factor=padding_factor,
        current_size=current_size,
        return_details=True,
    )
    mean_signal_variance = mean_signal_variance_relion * jnp.asarray(
        frame_scale,
        dtype=mean_signal_variance_relion.dtype,
    )
    tau2_shells_relion_frame = jnp.asarray(
        details["tau2_shells"],
        dtype=mean_signal_variance.dtype,
    )
    tau2_shells_recovar_frame = tau2_shells_relion_frame * jnp.asarray(
        frame_scale,
        dtype=mean_signal_variance.dtype,
    )
    return mean_signal_variance, tau2_shells_relion_frame, tau2_shells_recovar_frame


def _class_tau2_update_details(
    Ft_ctf_class,
    tau2_shells_recovar_frame,
    shell_stats,
    volume_shape,
    *,
    padding_factor,
    tau2_fudge,
    current_size: int,
    full_half_axis,
    accumulator_volume_shape,
):
    """Data-vs-prior and the host tau2 detail record for one class.

    ``shell_stats`` are the round-shell weight statistics of ``Ft_ctf_class``.
    The record uses the K=1 per-half key layout with ``fsc_shells`` set to
    ``None``. Returns ``(data_vs_prior, details)``.
    """

    data_vs_prior = regularization.compute_data_vs_prior(
        Ft_ctf_class,
        tau2_shells_recovar_frame,
        volume_shape,
        padding_factor=padding_factor,
        tau2_fudge=tau2_fudge,
        current_size=current_size,
        full_half_axis=full_half_axis,
        accumulator_volume_shape=accumulator_volume_shape,
    )
    details = {
        "prior_shells": np.asarray(tau2_shells_recovar_frame, dtype=np.float64),
        "sigma2_shells": np.asarray(
            jnp.where(
                shell_stats["avg_weight_shells"] > 0,
                1.0 / (padding_factor**3 * shell_stats["avg_weight_shells"]),
                0.0,
            ),
            dtype=np.float64,
        ),
        "avg_weight_shells": np.asarray(shell_stats["avg_weight_shells"], dtype=np.float64),
        "shell_sum": np.asarray(shell_stats["shell_sum"], dtype=np.float64),
        "shell_count": np.asarray(shell_stats["shell_count"], dtype=np.float64),
        "fsc_shells": None,
        "ssnr_shells": np.asarray(data_vs_prior, dtype=np.float64),
    }
    return data_vs_prior, details


def _stack_class_tau2_update_details(details_per_class):
    """Stack per-class tau2 detail records along a leading class axis.

    ``fsc_shells`` stays ``None``: Class3D has no half-set FSC.
    """

    return {
        key: None if key == "fsc_shells" else np.stack([details[key] for details in details_per_class], axis=0)
        for key in _CLASS_TAU2_DETAIL_KEYS
    }


def update_learned_direction_priors(
    *,
    rotation_posterior_per_half,
    class_rotation_posterior_per_half,
    global_direction_prior_per_half,
    global_direction_prior_order_per_half,
    class_direction_prior_per_half,
    class_direction_prior_order_per_half,
    n_classes: int,
    use_local: bool,
    k1_direction_prior_order: int,
    k1_direction_prior_size: int,
    current_healpix_order: int,
    exhaustive_grid_size: int,
    n_effective_rotations: int,
    dtype,
    log,
) -> None:
    """Learn the next iteration's direction priors from this iteration's posteriors.

    Mutates the four caller-owned per-half lists in place. K=1 collapses each
    half's rotation posterior at ``k1_direction_prior_order`` when both halves
    report posteriors of ``k1_direction_prior_size`` rotations; a half whose
    collapsed prior cannot form a RELION log prior is skipped with a warning
    on ``log``. K-class combines both halves' per-class posteriors on the
    exhaustive grid only for global scoring whose scorer grid has
    ``exhaustive_grid_size`` rotations, and stores an independent copy per
    half. The caller supplies the grid sizes so its sampling policy stays the
    single source of grid geometry.
    """

    if n_classes <= 1 and all(
        np.asarray(rot_sum).shape[0] == k1_direction_prior_size for rot_sum in rotation_posterior_per_half
    ):
        for k in range(2):
            direction_prior_k = collapse_rotation_posterior_to_direction_prior(
                np.asarray(rotation_posterior_per_half[k], dtype=np.float64),
                k1_direction_prior_order,
                dtype=dtype,
            )
            try:
                make_relion_direction_log_prior(direction_prior_k, k1_direction_prior_order)
            except ValueError as exc:
                log.warning(
                    "Skipping K=1 direction prior update for half-%d at healpix_order=%d: %s",
                    k + 1,
                    k1_direction_prior_order,
                    exc,
                )
                continue
            global_direction_prior_per_half[k] = direction_prior_k
            global_direction_prior_order_per_half[k] = k1_direction_prior_order
    elif (
        not use_local
        and n_classes > 1
        and n_effective_rotations == exhaustive_grid_size
        and all(rot_sum is not None for rot_sum in class_rotation_posterior_per_half)
    ):
        combined_class_direction_prior = _combined_class_direction_prior_from_halves(
            class_rotation_posterior_per_half,
            n_classes,
            current_healpix_order,
            dtype=dtype,
        )
        for k in range(2):
            class_direction_prior_per_half[k] = combined_class_direction_prior.copy()
            class_direction_prior_order_per_half[k] = current_healpix_order


def _merged_mean_from_halves(means, class_weights=None):
    merged = (means[0] + means[1]) / 2
    if class_weights is None:
        return merged, None
    class_weights_jax = jnp.asarray(class_weights, dtype=merged.real.dtype)
    return jnp.sum(class_weights_jax[:, None] * merged, axis=0), merged


def _reconstruct_volume_eager(
    Ft_ctf,
    Ft_y,
    vol_shape,
    padding_factor,
    tau,
    tau2_fudge,
    projection_padding_factor,
    use_spherical_mask=True,
    grid_correct=True,
    minres_map=0,
    current_size=None,
    return_real_space=False,
    accumulator_volume_shape=None,
    tau_is_1d=False,
    preserve_output_precision=False,
    relion_filter_scale=None,
):
    """Eager RELION-style reconstruction from full or half Fourier accumulators.

    This keeps the reconstruction boundary out of a single monolithic JIT while
    letting the local exact path keep its accumulators in packed half-volume
    layout until the final iDFT boundary.
    """
    from recovar.reconstruction import relion_functions

    return relion_functions.post_process_from_filter_v2(
        Ft_ctf,
        Ft_y,
        vol_shape,
        padding_factor,
        tau=tau,
        kernel="triangular",
        use_spherical_mask=use_spherical_mask,
        grid_correct=grid_correct,
        gridding_correct="radial",
        kernel_width=1,
        tau2_fudge=tau2_fudge,
        gridding_padding_factor=projection_padding_factor,
        minres_map=minres_map,
        current_size=current_size,
        return_real_space=return_real_space,
        accumulator_volume_shape=accumulator_volume_shape,
        tau_is_1d=tau_is_1d,
        preserve_output_precision=preserve_output_precision,
        relion_filter_scale=relion_filter_scale,
    )


def initial_low_pass_filter_references(
    Iref: np.ndarray,
    *,
    ori_size: int,
    pixel_size: float,
    ini_high_ang: float,
    filter_edgewidth: float = WIDTH_FMASK_EDGE,
) -> np.ndarray:
    """``initialLowPassFilterReferences`` (ml_optimiser.cpp:3336): cosine-taper from r=radius outward to r=radius_p."""
    edge_width = float(filter_edgewidth)
    radius = ori_size * pixel_size / ini_high_ang - edge_width / 2.0
    radius_p = radius + edge_width
    N = Iref.shape[1]
    kz = np.fft.fftfreq(N, d=1.0) * N
    kx = np.arange(N // 2 + 1, dtype=np.float64)
    r = np.sqrt(kz[:, None, None] ** 2 + kz[None, :, None] ** 2 + kx[None, None, :] ** 2)
    mask = np.zeros_like(r)
    mask[r < radius] = 1.0
    edge = (r >= radius) & (r <= radius_p)
    if edge_width > 0:
        mask[edge] = 0.5 - 0.5 * np.cos(np.pi * (radius_p - r[edge]) / edge_width)

    out = np.zeros_like(Iref)
    for k in range(Iref.shape[0]):
        vol = Iref[k]
        F = np.fft.rfftn(vol, axes=(0, 1, 2), norm=None) / vol.size
        out[k] = np.fft.irfftn(F * mask * vol.size, s=vol.shape, axes=(0, 1, 2), norm=None)
    return out


def _apply_relion_initial_lowpass_filter(
    volume_ft_flat, volume_shape, voxel_size, ini_high_angstrom, filter_edgewidth=5
):
    """Apply RELION's ``initialLowPassFilterReferences`` to a full Fourier volume."""
    if ini_high_angstrom is None or float(ini_high_angstrom) <= 0.0:
        return volume_ft_flat
    original = jnp.asarray(volume_ft_flat).reshape(volume_shape)
    volume_real = np.real(np.asarray(fourier_transform_utils.get_idft3(original))).astype(
        np.float64,
        copy=False,
    )
    filtered_real = initial_low_pass_filter_references(
        volume_real[None, ...],
        ori_size=int(volume_shape[0]),
        pixel_size=float(voxel_size),
        ini_high_ang=float(ini_high_angstrom),
        filter_edgewidth=float(filter_edgewidth),
    )[0]
    filtered_ft = fourier_transform_utils.get_dft3(jnp.asarray(filtered_real))
    return filtered_ft.astype(original.dtype).reshape(-1)


def _align_fourier_volume_sign_to_reference(volume_ft_flat, reference_ft_flat, volume_shape):
    """Keep reconstructed volumes on the same real-space sign branch as the reference."""
    if reference_ft_flat is None:
        return volume_ft_flat, False
    vol_real = np.asarray(
        fourier_transform_utils.get_idft3(jnp.asarray(volume_ft_flat).reshape(volume_shape)),
        dtype=np.float64,
    ).reshape(-1)
    ref_real = np.asarray(
        fourier_transform_utils.get_idft3(jnp.asarray(reference_ft_flat).reshape(volume_shape)),
        dtype=np.float64,
    ).reshape(-1)
    vol_centered = vol_real - float(np.mean(vol_real))
    ref_centered = ref_real - float(np.mean(ref_real))
    overlap = float(np.dot(ref_centered, vol_centered))
    if overlap < 0.0:
        return -volume_ft_flat, True
    return volume_ft_flat, False


def _reconstruct_and_postprocess_means(
    means,
    *,
    Ft_y_0,
    Ft_y_1,
    Ft_ctf_0,
    Ft_ctf_1,
    Ft_y_combined,
    Ft_ctf_combined,
    mean_signal_variance,
    mean_signal_variance_shells,
    mean_signal_variance_per_half,
    n_classes: int,
    cs,
    iteration: int,
    grid_size: int,
    cryo,
    volume_shape,
    tau2_fudge: float,
    padding_factor: int,
    projection_padding_factor: int,
    relion_minres_map: int,
    particle_diameter_ang,
    relion_firstiter_cc_this_iter: bool,
    relion_firstiter_ini_high_angstrom,
    relion_width_mask_edge: int,
    relion_fmask_edge: int,
    accumulator_volume_shape=None,
) -> None:
    """Run one iteration's regularized reconstruction + post-processing.

    Mutates ``means`` in place. Performs Wiener reconstruction (per-class for
    K>1, per-half for K=1), optional pre-mask debug dump, RELION solvent
    flatten, and iter-1 firstiter_cc low-pass filter.

    ``relion_width_mask_edge`` is the real-space mask edge (RELION's
    ``--maskedge`` = 5). ``relion_fmask_edge`` is the Fourier mask edge for
    the iter-1 ``ini_high`` low-pass filter (RELION's ``WIDTH_FMASK_EDGE`` = 2).
    Mixing the two produces a softer Fourier filter than RELION applies.
    """

    _t_recon = time.time()
    cs_int = int(cs) if cs is not None else None
    if n_classes > 1:
        shared_class_maps = []
        for class_idx in range(n_classes):
            logger.info(
                "Class3D reconstruction start: iter=%d class=%d/%d current_size=%s",
                iteration + 1,
                class_idx + 1,
                n_classes,
                cs_int,
            )
            class_map = _reconstruct_volume_eager(
                Ft_ctf_combined[class_idx],
                Ft_y_combined[class_idx],
                volume_shape,
                padding_factor,
                tau=(
                    mean_signal_variance_shells[class_idx]
                    if mean_signal_variance_shells is not None
                    else mean_signal_variance[class_idx]
                ),
                tau2_fudge=tau2_fudge,
                projection_padding_factor=projection_padding_factor,
                minres_map=relion_minres_map,
                current_size=cs_int,
                accumulator_volume_shape=accumulator_volume_shape,
                tau_is_1d=mean_signal_variance_shells is not None,
            ).reshape(-1)
            shared_class_maps.append(class_map)
            logger.info(
                "Class3D reconstruction done: iter=%d class=%d/%d elapsed=%.1fs",
                iteration + 1,
                class_idx + 1,
                n_classes,
                time.time() - _t_recon,
            )
        shared_classes = jnp.stack(shared_class_maps, axis=0)
        logger.info(
            "Class3D reconstruction stack complete: iter=%d classes=%d elapsed=%.1fs",
            iteration + 1,
            n_classes,
            time.time() - _t_recon,
        )
        means[0] = shared_classes
        means[1] = shared_classes
    else:
        for k in range(2):
            Ft_y_k_local = Ft_y_0 if k == 0 else Ft_y_1
            Ft_ctf_k_local = Ft_ctf_0 if k == 0 else Ft_ctf_1
            # This RELION build uses double RFLOAT in BackProjector::reconstruct.
            # Keep the stored/controller tau2 state compact, but promote the
            # reconstruction operand so 1 / (padding_factor**3 * tau2) is not
            # rounded in float32 before it enters the Wiener denominator.
            reconstruction_tau = jnp.asarray(
                mean_signal_variance_per_half[k],
                dtype=jnp.float64,
            )
            means[k] = _reconstruct_volume_eager(
                Ft_ctf_k_local,
                Ft_y_k_local,
                volume_shape,
                padding_factor,
                tau=reconstruction_tau,
                tau2_fudge=tau2_fudge,
                projection_padding_factor=projection_padding_factor,
                minres_map=relion_minres_map,
                current_size=cs_int,
                accumulator_volume_shape=accumulator_volume_shape,
                preserve_output_precision=True,
                relion_filter_scale=float(volume_shape[0] ** 4),
            ).reshape(-1)

    for k in range(2):
        # Diagnostic: dump pre-mask Wiener output when env var set.
        _premask_dump = os.environ.get("RECOVAR_PREMASK_DUMP_DIR")
        if _premask_dump:
            from recovar.em.diagnostics.reconstruction import write_premask_mean

            write_premask_mean(
                means[k], output_dir=_premask_dump, half_index=k, iteration=iteration,
                current_size=cs, grid_size=grid_size, voxel_size=cryo.voxel_size,
                volume_shape=volume_shape, n_classes=n_classes,
            )

        # RELION filters Iref inside maximizationOtherParameters, then calls
        # solventFlatten from the outer iteration loop.  These operations do
        # not commute: masking in real space after the Fourier low-pass adds a
        # small, deterministic high-shell tail.
        if relion_firstiter_cc_this_iter:
            if n_classes > 1:
                means[k] = jnp.stack(
                    [
                        _apply_relion_initial_lowpass_filter(
                            means[k][class_idx],
                            volume_shape,
                            cryo.voxel_size,
                            relion_firstiter_ini_high_angstrom,
                            filter_edgewidth=relion_fmask_edge,
                        )
                        for class_idx in range(n_classes)
                    ],
                    axis=0,
                )
            else:
                means[k] = _apply_relion_initial_lowpass_filter(
                    means[k],
                    volume_shape,
                    cryo.voxel_size,
                    relion_firstiter_ini_high_angstrom,
                    filter_edgewidth=relion_fmask_edge,
                )
        if particle_diameter_ang is not None and particle_diameter_ang > 0:
            flatten_radius = particle_diameter_ang / (2.0 * cryo.voxel_size)
            solvent_mask = mask.raised_cosine_mask(
                volume_shape,
                radius=flatten_radius,
                radius_p=flatten_radius + relion_width_mask_edge,
                offset=jnp.zeros(3),
                dtype=(means[k].real.dtype if n_classes <= 1 else means[k][0].real.dtype),
            )
            if n_classes > 1:
                flattened_classes = []
                for class_idx in range(n_classes):
                    vol_real = fourier_transform_utils.get_idft3(means[k][class_idx].reshape(volume_shape))
                    flattened_classes.append(
                        fourier_transform_utils.get_dft3(vol_real * solvent_mask).reshape(-1),
                    )
                means[k] = jnp.stack(flattened_classes, axis=0)
            else:
                vol_real = fourier_transform_utils.get_idft3(means[k].reshape(volume_shape))
                means[k] = fourier_transform_utils.get_dft3(vol_real * solvent_mask).reshape(-1)
    if relion_firstiter_cc_this_iter and relion_firstiter_ini_high_angstrom is not None:
        logger.info(
            "RELION iter-1 CC emulation: reapplying ini_high low-pass filter at %.2f A",
            float(relion_firstiter_ini_high_angstrom),
        )
    logger.info("Regularized reconstruction (2 halves + flatten): %.1fs", time.time() - _t_recon)


# ---------------------------------------------------------------------------
# Unregularized half-map reconstruction + sign alignment
# ---------------------------------------------------------------------------


def compute_unregularized_halfmaps_and_align_signs(
    *,
    means: list,
    previous_means: list,
    Ft_y_per_half: tuple,
    Ft_ctf_per_half: tuple,
    Ft_y_combined,
    Ft_ctf_combined,
    volume_shape,
    n_classes: int,
    tau2_fudge: float,
    padding_factor: int,
    projection_padding_factor: int,
    minres_map: int,
    need_unreg_means: bool,
    accumulator_volume_shape=None,
) -> list:
    """Reconstruct unregularized half-maps (only when diagnostics need them)
    and sign-align the regularized means against the previous-iter reference.

    Mutates the caller-owned ``means`` list in place and returns the two
    unregularized maps (or ``[None, None]`` when diagnostics are disabled).

    For K-class refinement both halves share the same Iref-derived
    prior, so the unregularized accumulator is the combined Ft_y/Ft_ctf
    rather than the per-half pair; the K=1 path reconstructs from each
    half's own accumulators.

    Sign alignment uses ``_align_fourier_volume_sign_to_reference`` against
    the previous iteration's means; in the K-class case both half-slots
    end up pointing at the same shared K-stack.
    """

    _t_unreg = time.time()
    if need_unreg_means:
        if n_classes > 1:
            unreg_shared = jnp.stack(
                [
                    _reconstruct_volume_eager(
                        Ft_ctf_combined[class_idx],
                        Ft_y_combined[class_idx],
                        volume_shape,
                        padding_factor,
                        tau=None,
                        tau2_fudge=tau2_fudge,
                        projection_padding_factor=projection_padding_factor,
                        minres_map=minres_map,
                        accumulator_volume_shape=accumulator_volume_shape,
                    ).reshape(-1)
                    for class_idx in range(n_classes)
                ],
                axis=0,
            )
            unreg_means: list = [unreg_shared, unreg_shared]
        else:
            unreg_means = [
                _reconstruct_volume_eager(
                    Ft_ctf_half,
                    Ft_y_half,
                    volume_shape,
                    padding_factor,
                    tau=None,
                    tau2_fudge=tau2_fudge,
                    projection_padding_factor=projection_padding_factor,
                    minres_map=minres_map,
                    accumulator_volume_shape=accumulator_volume_shape,
                )
                for Ft_ctf_half, Ft_y_half in zip(Ft_ctf_per_half, Ft_y_per_half)
            ]
    else:
        unreg_means = [None, None]

    if n_classes > 1:
        aligned_classes = []
        unreg_classes = [] if unreg_means[0] is not None else None
        for class_idx in range(n_classes):
            aligned_class, sign_flipped = _align_fourier_volume_sign_to_reference(
                means[0][class_idx],
                previous_means[0][class_idx],
                volume_shape,
            )
            aligned_classes.append(aligned_class)
            if unreg_classes is not None:
                unreg_classes.append(-unreg_means[0][class_idx] if sign_flipped else unreg_means[0][class_idx])
            if sign_flipped:
                logger.info("Aligned shared class-%d volume sign to the previous reference", class_idx + 1)
        shared_aligned = jnp.stack(aligned_classes, axis=0)
        means[0] = shared_aligned
        means[1] = shared_aligned
        if unreg_classes is not None:
            shared_unreg = jnp.stack(unreg_classes, axis=0)
            unreg_means = [shared_unreg, shared_unreg]
    else:
        for k in range(2):
            means[k], sign_flipped = _align_fourier_volume_sign_to_reference(
                means[k],
                previous_means[k],
                volume_shape,
            )
            if sign_flipped and unreg_means[k] is not None:
                unreg_means[k] = -unreg_means[k]
            if sign_flipped:
                logger.info("Aligned half-%d volume sign to the previous reference", k + 1)
    logger.info(
        "Unregularized reconstruction (2 halves): %.1fs%s",
        time.time() - _t_unreg,
        "" if need_unreg_means else " (skipped; diagnostics disabled)",
    )
    return unreg_means

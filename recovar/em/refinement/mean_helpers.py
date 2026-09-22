"""Mean-volume reconstruction and K-class helpers for the iteration loop.

Per-iteration aggregator + reconstruction helpers extracted verbatim from
``iteration_loop.py``. None of these wrap symbols that pytest monkeypatches
at ``iteration_loop.<name>``; all dependencies are imported directly.
"""

from __future__ import annotations

import functools
import gc
import logging
import math
import os
import time

import jax
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

_LARGE_IRFFT_TRANSFORM_SIZE_LIMIT = np.iinfo(np.int32).max

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


def _snapshot_and_release_previous_k1_means(means):
    """Copy both K1 references to host before releasing their active buffers."""
    if len(means) != 2:
        raise ValueError(f"K=1 refinement requires exactly two half maps, got {len(means)}")
    previous_means = [
        np.asarray(mean).copy() if mean is not None else None for mean in means
    ]
    for half_index in range(2):
        means[half_index] = None
    gc.collect()
    return previous_means


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
    class_rotation_posterior_per_half, n_classes: int, healpix_order: int, *, dtype: np.dtype = np.float32, symmetry: str = "C1"
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
            collapse_rotation_posterior_to_direction_prior(combined, healpix_order, dtype=dtype, **({"symmetry": symmetry} if symmetry != "C1" else {}))
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
    preserve_inputs=True,
    return_retained_first_numerator=False,
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
        **({"preserve_inputs": False} if not preserve_inputs else {}),
        **({"return_retained_first_numerator": True} if return_retained_first_numerator else {}),
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
    symmetry: str = "C1",
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
                **({"symmetry": symmetry} if symmetry != "C1" else {}),
            )
            try:
                make_relion_direction_log_prior(direction_prior_k, k1_direction_prior_order, **({"symmetry": symmetry} if symmetry != "C1" else {}))
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
            **({"symmetry": symmetry} if symmetry != "C1" else {}),
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
    retained_device_numerator=None,
):
    """Eager RELION-style reconstruction from full or half Fourier accumulators.

    This keeps the reconstruction boundary out of a single monolithic JIT while
    letting the local exact path keep its accumulators in packed half-volume
    layout until the final iDFT boundary.
    """
    from recovar.reconstruction import relion_functions

    Ft_ctf, Ft_y = _pack_compact_full_accumulators_for_large_relion_ifft(
        Ft_ctf,
        Ft_y,
        vol_shape,
        padding_factor,
        accumulator_volume_shape,
        relion_functions,
    )
    postprocess_args = (Ft_ctf, Ft_y, vol_shape, padding_factor)
    postprocess_kwargs = dict(
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
        # EM transform precision follows its accumulation precision, not the
        # deliberate RFLOAT denominator/gridding operands. Complex128 inputs
        # retain the diagnostic transform path.
        fft_compute_dtype=jnp.result_type(Ft_y.dtype, jnp.complex64),
        relion_filter_scale=relion_filter_scale,
    )
    host_stage_large_ifft = _should_host_stage_large_relion_ifft(
        Ft_ctf,
        Ft_y,
        vol_shape,
        padding_factor,
        accumulator_volume_shape,
        relion_functions,
    )
    if retained_device_numerator is not None and not host_stage_large_ifft:
        raise ValueError(
            "A retained device numerator is only valid for the large host-staged "
            "RELION reconstruction path"
        )
    if not host_stage_large_ifft:
        result = relion_functions.post_process_from_filter_v2(
            *postprocess_args,
            **postprocess_kwargs,
        )
        reconstruction_shape = relion_functions._relion_reconstruction_padded_shape(
            vol_shape,
            padding_factor,
        )
        if _large_irfft_requires_explicit_normalization(reconstruction_shape):
            # XLA's built-in ``norm='backward'`` normalization overflows its
            # signed-int32 transform-size product at 1600^3 and silently omits
            # the reciprocal.  Physically large device accumulators take this
            # monolithic branch to avoid overlapping another box-scale device
            # buffer, so apply the reciprocal to the completed result in a
            # separate donating executable.
            transform_size = math.prod(reconstruction_shape)
            logger.info(
                "RELION large inverse-FFT normalization boundary: "
                "reconstruction_shape=%s transform_size=%d "
                "implementation=jax_monolithic_dynamic_scale",
                reconstruction_shape,
                transform_size,
            )
            inverse_transform_scale = jnp.asarray(
                np.float32(1.0 / float(transform_size)),
            )
            result = _normalize_large_irfft_result_donate(
                result,
                inverse_transform_scale,
            )
        return result

    accumulator_shape = (
        tuple(3 * [int(vol_shape[0]) * int(padding_factor)])
        if accumulator_volume_shape is None
        else tuple(int(s) for s in accumulator_volume_shape)
    )
    reconstruction_shape = relion_functions._relion_reconstruction_padded_shape(
        vol_shape,
        padding_factor,
    )
    packed_half_bytes = int(
        np.prod(fourier_transform_utils.volume_shape_to_half_volume_shape(reconstruction_shape))
        * np.dtype(np.complex64).itemsize
    )
    logger.info(
        "RELION split pre-IFFT host boundary: accumulator_shape=%s "
        "reconstruction_shape=%s packed_half_bytes=%d",
        accumulator_shape,
        reconstruction_shape,
        packed_half_bytes,
    )
    if accumulator_shape[0] > reconstruction_shape[0]:
        # The original Stage-A executable combined denominator
        # regularization and complex division.  Although donation aliases its
        # numerator to the output, the regularization needs several
        # box-scale float32 temporaries.  Keep the complex64 numerator on the
        # host while those temporaries are live, then stage it only for the
        # zero-temporary donating divide.
        stage_a_filter = jnp.asarray(Ft_ctf)
        stage_a_filter.block_until_ready()
        if np.dtype(stage_a_filter.dtype) != np.dtype(np.float32):
            raise TypeError(
                "Large RELION Stage A requires a float32 filter for exact "
                f"donation, got {stage_a_filter.dtype}"
            )
        logger.info(
            "RELION Stage A staging host filter for donating regularization: "
            "shape=%s dtype=%s",
            tuple(stage_a_filter.shape),
            stage_a_filter.dtype,
        )
        regularized_filter_device = (
            relion_functions._regularize_large_relion_half_filter_donate_ctf(
                stage_a_filter,
                tau,
                vol_shape,
                padding_factor,
                tau2_fudge,
                minres_map,
                current_size,
                accumulator_shape,
                tau_is_1d,
                relion_filter_scale,
            )
        )
        regularized_filter_device.block_until_ready()
        filter_input_donated = _device_array_is_deleted(stage_a_filter)
        if filter_input_donated is not True:
            _delete_device_array(regularized_filter_device)
            _delete_device_array(stage_a_filter)
            raise RuntimeError(
                "Large RELION Stage-A regularization did not donate its float32 filter input"
            )
        if np.dtype(regularized_filter_device.dtype) != np.dtype(np.float32):
            _delete_device_array(regularized_filter_device)
            raise TypeError(
                "Large RELION Stage-A regularization must return float32, got "
                f"{regularized_filter_device.dtype}"
            )
        logger.info(
            "RELION Stage A regularization complete: filter_input_donated=%s",
            filter_input_donated,
        )

        stage_a_numerator = None
        stage_a_numerator_source = "device"
        if retained_device_numerator is not None:
            if tuple(retained_device_numerator.shape) != tuple(Ft_y.shape):
                raise ValueError(
                    "Retained device numerator shape does not match the host numerator: "
                    f"{tuple(retained_device_numerator.shape)} != {tuple(Ft_y.shape)}"
                )
            if np.dtype(retained_device_numerator.dtype) != np.dtype(Ft_y.dtype):
                raise ValueError(
                    "Retained device numerator dtype does not match the host numerator: "
                    f"{retained_device_numerator.dtype} != {Ft_y.dtype}"
                )
            stage_a_numerator = retained_device_numerator
            stage_a_numerator_source = "retained_join"
            logger.info(
                "RELION Stage A reusing retained half-0 device numerator: shape=%s dtype=%s",
                tuple(retained_device_numerator.shape),
                retained_device_numerator.dtype,
            )
        else:
            # A NumPy argument passed directly to a donate_argnums JIT is first
            # staged by dispatch, but that transient input cannot be donated.
            # Materialise an explicit JAX array so half 2 can alias its 15-GiB
            # Stage-A output into the staged numerator just as half 1 aliases
            # the retained low-resolution-join buffer.
            stage_a_numerator = jnp.asarray(Ft_y)
            stage_a_numerator.block_until_ready()
            if isinstance(Ft_y, np.ndarray):
                stage_a_numerator_source = "staged_numpy"
                logger.info(
                    "RELION Stage A staging host numerator for donation: shape=%s dtype=%s",
                    tuple(stage_a_numerator.shape),
                    stage_a_numerator.dtype,
                )
        wiener_half_device = relion_functions._divide_large_relion_half_numerator_donate_numerator(
            stage_a_numerator,
            regularized_filter_device,
            padding_factor,
            current_size,
            accumulator_shape,
        )
        wiener_half_device.block_until_ready()
        wiener_half_host = np.asarray(jax.device_get(wiener_half_device)).reshape(
            fourier_transform_utils.volume_shape_to_half_volume_shape(accumulator_shape),
        )
        _delete_device_array(wiener_half_device)
        _delete_device_array(stage_a_numerator)
        _delete_device_array(regularized_filter_device)
        _delete_device_array(stage_a_filter)
        logger.info(
            "RELION Stage A released donated device numerator after host transfer: "
            "source=%s output_deleted=%s numerator_deleted=%s filter_deleted=%s",
            stage_a_numerator_source,
            _device_array_is_deleted(wiener_half_device),
            _device_array_is_deleted(stage_a_numerator),
            _device_array_is_deleted(regularized_filter_device),
        )
        del wiener_half_device
        del stage_a_numerator
        del regularized_filter_device
        del stage_a_filter
        gc.collect()

        fftw_half_host = _crop_relion_wiener_half_to_fftw_host(
            wiener_half_host,
            accumulator_shape,
            reconstruction_shape,
            relion_functions,
        )
        del wiener_half_host
        gc.collect()
    else:
        if retained_device_numerator is not None:
            raise ValueError(
                "A retained device numerator is only valid for the crop branch of the "
                "large host-staged RELION reconstruction path"
            )
        fftw_half_device = relion_functions.post_process_from_filter_v2(
            *postprocess_args,
            **postprocess_kwargs,
            input_half_volume=True,
            return_fftw_half_before_ifft=True,
        )
        fftw_half_device.block_until_ready()
        fftw_half_host = np.asarray(jax.device_get(fftw_half_device))
        del fftw_half_device
        gc.collect()

    explicit_irfft_normalization = _large_irfft_requires_explicit_normalization(
        reconstruction_shape,
    )
    host_irfft = _large_relion_host_irfft_enabled(reconstruction_shape)
    if host_irfft:
        workers = _relion_host_fft_workers()
        logger.info(
            "RELION padded inverse FFT using host scipy.fft: reconstruction_shape=%s "
            "output_shape=%s input_bytes=%d workers=%d",
            reconstruction_shape,
            tuple(int(size) for size in vol_shape),
            int(fftw_half_host.nbytes),
            workers,
        )
        unpadded_real_host = _host_irfft_and_center_crop(
            fftw_half_host,
            reconstruction_shape,
            vol_shape,
            workers=workers,
        )
        del fftw_half_host
        gc.collect()
        result = relion_functions._finish_large_relion_postprocess_from_unpadded_real(
            unpadded_real_host,
            vol_shape,
            padding_factor,
            kernel="triangular",
            use_spherical_mask=use_spherical_mask,
            grid_correct=grid_correct,
            gridding_correct="radial",
            kernel_width=1,
            return_real_space=return_real_space,
            gridding_padding_factor=projection_padding_factor,
        )
    else:
        result = relion_functions._finish_large_relion_postprocess_from_fftw_half(
            fftw_half_host,
            vol_shape,
            padding_factor,
            kernel="triangular",
            use_spherical_mask=use_spherical_mask,
            grid_correct=grid_correct,
            gridding_correct="radial",
            kernel_width=1,
            return_real_space=return_real_space,
            gridding_padding_factor=projection_padding_factor,
        )
    if explicit_irfft_normalization:
        transform_size = math.prod(reconstruction_shape)
        logger.info(
            "RELION large inverse-FFT normalization boundary: "
            "reconstruction_shape=%s transform_size=%d implementation=%s",
            reconstruction_shape,
            transform_size,
            "scipy_host_backward" if host_irfft else "jax_dynamic_scale",
        )
    if explicit_irfft_normalization and not host_irfft:
        # XLA's built-in ``norm='backward'`` normalization overflows its
        # signed-int32 transform-size product at 1600^3 and silently omits the
        # reciprocal. Apply that reciprocal in a separate executable after
        # the affected inverse-FFT executable has completed.
        # Donation keeps this correction memory-neutral for box-scale maps.
        inverse_transform_scale = jnp.asarray(
            np.float32(1.0 / float(transform_size)),
        )
        result = _normalize_large_irfft_result_donate(
            result,
            inverse_transform_scale,
        )
    return result


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
    mean_signal_variance_shells_per_half=None,
    retained_Ft_y_0_device=None,
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
    if n_classes > 1 and retained_Ft_y_0_device is not None:
        raise ValueError("The retained half-0 numerator path is only valid for K=1")
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
        if mean_signal_variance_shells_per_half is not None and len(mean_signal_variance_shells_per_half) != 2:
            raise ValueError("K=1 reconstruction tau2 shells require exactly two halves")
        for k in range(2):
            Ft_y_k_local = Ft_y_0 if k == 0 else Ft_y_1
            Ft_ctf_k_local = Ft_ctf_0 if k == 0 else Ft_ctf_1
            # This RELION build uses double RFLOAT in BackProjector::reconstruct.
            # Keep the stored/controller tau2 state compact, but promote the
            # reconstruction operand so 1 / (padding_factor**3 * tau2) is not
            # rounded in float32 before it enters the Wiener denominator.
            reconstruction_tau_source = (
                mean_signal_variance_shells_per_half[k]
                if mean_signal_variance_shells_per_half is not None
                else mean_signal_variance_per_half[k]
            )
            reconstruction_tau = jnp.asarray(
                reconstruction_tau_source,
                dtype=jnp.float64,
            )
            reconstructed = _reconstruct_volume_eager(
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
                tau_is_1d=mean_signal_variance_shells_per_half is not None,
                preserve_output_precision=True,
                relion_filter_scale=float(volume_shape[0] ** 4),
                **({"retained_device_numerator": retained_Ft_y_0_device} if k == 0 and retained_Ft_y_0_device is not None else {}),
            ).reshape(-1)
            means[k] = _finish_host_staged_reconstruction(
                reconstructed, Ft_ctf_k_local, Ft_y_k_local,
            )
            if k == 0 and retained_Ft_y_0_device is not None:
                retained_Ft_y_0_device = None
                gc.collect()

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
            flatten_radius = (
                float(particle_diameter_ang) / (2.0 * float(cryo.voxel_size))
                if n_classes == 1 else particle_diameter_ang / (2.0 * cryo.voxel_size)
            )
            solvent_mask = _make_relion_solvent_mask(
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
                means[k] = _apply_relion_solvent_flatten_k1(
                    means[k], solvent_mask, volume_shape, half_index=k,
                )
                if _large_relion_solvent_mask_uses_compiled_builder(volume_shape):
                    solvent_mask = None
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
    and apply the legacy K=1 sign-continuity check.

    Mutates the caller-owned ``means`` list in place and returns the two
    unregularized maps (or ``[None, None]`` when diagnostics are disabled).

    For K-class refinement both halves share the same Iref-derived
    prior, so the unregularized accumulator is the combined Ft_y/Ft_ctf
    rather than the per-half pair; the K=1 path reconstructs from each
    half's own accumulators.

    K-class maps preserve the sign fixed by the image/CTF convention; both
    half-slots share that K-stack. K=1 retains sign alignment against its
    previous reference. See docs/math/relion_refinement_algorithm.md for the
    reconstruction convention.
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
        # The image/CTF convention fixes K-class reconstruction signs.
        # Weak overlap with a previous reference must not negate a class.
        means[1] = means[0]
        if unreg_means[0] is not None:
            unreg_means[1] = unreg_means[0]
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


def _large_irfft_requires_explicit_normalization(volume_shape) -> bool:
    """Return whether XLA's inverse-FFT normalization exceeds int32 range."""

    return math.prod(int(size) for size in volume_shape) > _LARGE_IRFFT_TRANSFORM_SIZE_LIMIT


def _large_relion_host_irfft_enabled(volume_shape) -> bool:
    """Return whether a padded RELION inverse FFT should execute on the host."""

    mode = os.environ.get("RECOVAR_RELION_HOST_IRFFT", "auto").strip().lower()
    if mode in {"0", "false", "no", "off", "never"}:
        return False
    if mode in {"1", "true", "yes", "on", "always"}:
        return True
    if mode != "auto":
        logger.warning(
            "Unrecognised RECOVAR_RELION_HOST_IRFFT=%r; using auto",
            mode,
        )
    return _large_irfft_requires_explicit_normalization(volume_shape)


def _relion_host_fft_workers() -> int:
    configured = os.environ.get("RECOVAR_RELION_HOST_FFT_WORKERS")
    if configured is None:
        configured = os.environ.get("SLURM_CPUS_PER_TASK", "1")
    try:
        workers = int(configured)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid RELION host FFT worker count %r; using one worker",
            configured,
        )
        return 1
    return max(1, workers)


def _host_irfft_and_center_crop(
    fftw_half,
    reconstruction_shape,
    output_shape,
    *,
    workers=None,
):
    """Run a normalized c64-to-f32 inverse FFT and retain only its center crop.

    ``fftw_half`` is already in raw FFTW order.  The crop indices combine
    ``ifftshift`` with RELION's spatial unpadding so the host never allocates a
    second reconstruction-sized real volume merely to shift it.
    """

    from scipy import fft as scipy_fft

    reconstruction_shape = tuple(int(size) for size in reconstruction_shape)
    output_shape = tuple(int(size) for size in output_shape)
    if len(reconstruction_shape) != 3 or len(output_shape) != 3:
        raise ValueError(
            "RELION host inverse FFT requires three-dimensional shapes, got "
            f"reconstruction={reconstruction_shape} output={output_shape}"
        )
    if any(output > reconstruction for output, reconstruction in zip(output_shape, reconstruction_shape)):
        raise ValueError(
            "RELION host inverse FFT crop cannot exceed its reconstruction: "
            f"reconstruction={reconstruction_shape} output={output_shape}"
        )

    expected_half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(
        reconstruction_shape,
    )
    fftw_half = np.asarray(fftw_half, dtype=np.complex64, order="C").reshape(
        expected_half_shape,
    )
    workers = _relion_host_fft_workers() if workers is None else max(1, int(workers))
    real_raw = scipy_fft.irfftn(
        fftw_half,
        s=reconstruction_shape,
        axes=(-3, -2, -1),
        norm="backward",
        overwrite_x=True,
        workers=workers,
    )
    if real_raw.dtype != np.float32:
        raise TypeError(f"RELION host inverse FFT returned {real_raw.dtype}, expected float32")

    raw_indices = []
    for reconstruction, output in zip(reconstruction_shape, output_shape):
        padding_width = reconstruction - output
        pad_before = padding_width // 2
        centered_indices = np.arange(pad_before, pad_before + output, dtype=np.intp)
        # ``np.fft.ifftshift`` takes centered output index ``i`` from raw
        # input index ``i + floor(N / 2)``.  Keep the explicit floor because
        # the distinction matters for odd reconstruction sizes.
        raw_indices.append((centered_indices + reconstruction // 2) % reconstruction)
    cropped = np.asarray(
        real_raw[np.ix_(*raw_indices)],
        dtype=np.float32,
        order="C",
    )
    del real_raw
    return cropped


@functools.partial(jax.jit, donate_argnums=(0,))
def _normalize_large_irfft_result_donate(result, inverse_transform_scale):
    """Normalize a large raw inverse FFT in a separate donating executable."""

    return result * inverse_transform_scale


def _should_host_stage_large_relion_ifft(
    Ft_ctf,
    Ft_y,
    vol_shape,
    padding_factor,
    accumulator_volume_shape,
    relion_functions,
):
    """Return whether eager reconstruction should cross the padded-iFFT host boundary."""

    accumulator_shape = (
        tuple(3 * [int(vol_shape[0]) * int(padding_factor)])
        if accumulator_volume_shape is None
        else tuple(int(s) for s in accumulator_volume_shape)
    )
    reconstruction_shape = relion_functions._relion_reconstruction_padded_shape(
        vol_shape,
        padding_factor,
    )
    if accumulator_shape == reconstruction_shape:
        return False
    half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(accumulator_shape)
    half_size = int(np.prod(half_shape))

    def _is_packed_half(array):
        return tuple(array.shape) == half_shape or (array.ndim == 1 and int(array.size) == half_size)

    # The split is governed by the inverse-FFT grid, not the current-size
    # accumulator. Early box-scale iterations can have a compact accumulator
    # but still pad to a 1600^3 transform whose built-in normalization
    # overflows XLA's signed-int32 transform-size product. Compact device
    # accumulators can safely form and host-stage that one padded boundary.
    # Large accumulators still require the existing earlier host offload so
    # their storage cannot overlap the padded inverse-FFT workspace.
    accumulator_is_large = relion_functions._large_grid_postprocess_is_physically_large(
        int(np.prod(accumulator_shape)),
    )
    inputs_are_host = isinstance(Ft_ctf, np.ndarray) and isinstance(Ft_y, np.ndarray)
    if not _is_packed_half(Ft_ctf) or not _is_packed_half(Ft_y):
        return False
    if accumulator_is_large and not inputs_are_host:
        return False
    return relion_functions._large_grid_postprocess_single_precision_enabled(
        int(np.prod(reconstruction_shape)),
    )


def _pack_compact_full_accumulators_for_large_relion_ifft(
    Ft_ctf,
    Ft_y,
    vol_shape,
    padding_factor,
    accumulator_volume_shape,
    relion_functions,
):
    """Losslessly repack compact full accumulators before a giant padded iFFT.

    The RELION x-half M-step keeps the historical RECOVAR full-volume public
    contract on normal-sized accumulator grids. At large box sizes an early
    iteration can therefore reach reconstruction with compact full Hermitian
    arrays even though its padded inverse-FFT grid is giant. Repack only this
    compact/full case so it can use the packed pre-iFFT host boundary without
    changing the public M-step contract or the large-accumulator offload path.
    """

    accumulator_shape = (
        tuple(3 * [int(vol_shape[0]) * int(padding_factor)])
        if accumulator_volume_shape is None
        else tuple(int(s) for s in accumulator_volume_shape)
    )
    reconstruction_shape = relion_functions._relion_reconstruction_padded_shape(
        vol_shape,
        padding_factor,
    )
    if accumulator_shape == reconstruction_shape:
        return Ft_ctf, Ft_y

    accumulator_voxels = int(np.prod(accumulator_shape))
    reconstruction_voxels = int(np.prod(reconstruction_shape))
    # The accumulator decision is physical: forcing single precision on a
    # compact grid must not misclassify it as too large to repack.  The
    # reconstruction still needs the single-precision large-grid path because
    # this boundary stages a complex64 packed half-volume.
    if relion_functions._large_grid_postprocess_is_physically_large(
        accumulator_voxels,
    ) or not relion_functions._large_grid_postprocess_single_precision_enabled(
        reconstruction_voxels,
    ):
        return Ft_ctf, Ft_y

    def _is_full(array):
        return tuple(array.shape) == accumulator_shape or (
            array.ndim == 1 and int(array.size) == accumulator_voxels
        )

    if not _is_full(Ft_ctf) or not _is_full(Ft_y):
        return Ft_ctf, Ft_y

    logger.info(
        "RELION giant-iFFT compact full-to-half repack: accumulator_shape=%s "
        "reconstruction_shape=%s",
        accumulator_shape,
        reconstruction_shape,
    )
    return (
        fourier_transform_utils.full_volume_to_half_volume(
            Ft_ctf,
            accumulator_shape,
        ).reshape(-1),
        fourier_transform_utils.full_volume_to_half_volume(
            Ft_y,
            accumulator_shape,
        ).reshape(-1),
    )


def _crop_relion_wiener_half_to_fftw_host(
    wiener_half,
    accumulator_shape,
    reconstruction_shape,
    relion_functions,
):
    """Crop a centered packed half-volume into raw FFTW order on the host."""

    accumulator_shape = tuple(int(s) for s in accumulator_shape)
    reconstruction_shape = tuple(int(s) for s in reconstruction_shape)
    accumulator_half_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(
        accumulator_shape,
    )
    wiener_half = np.asarray(wiener_half).reshape(accumulator_half_shape)
    centered_axis_idx = relion_functions._relion_centered_axis_take_indices(
        accumulator_shape[0],
        reconstruction_shape[0],
    )
    raw_axis_idx = np.fft.ifftshift(centered_axis_idx)
    col_idx = np.arange(reconstruction_shape[-1] // 2 + 1, dtype=np.int32)
    return wiener_half[np.ix_(raw_axis_idx, raw_axis_idx, col_idx)]


def _delete_device_array(value):
    """Release a completed JAX buffer even when another dead handle survives."""

    delete = getattr(value, "delete", None)
    if callable(delete):
        try:
            delete()
        except RuntimeError:
            # Donation invalidates the input handle when the output aliases it.
            pass


def _device_array_is_deleted(value):
    """Return JAX's deletion state when the device-array API exposes it."""

    is_deleted = getattr(value, "is_deleted", None)
    return bool(is_deleted()) if callable(is_deleted) else None


def _finish_host_staged_reconstruction(result, *accumulators):
    """Finish a host-staged reconstruction before dispatching the next half.

    Large RELION accumulators are moved to NumPy before reconstruction so the
    device only needs one half's inputs and FFT workspace at a time.  JAX
    dispatch is asynchronous, so wait here before the loop starts the other
    half; otherwise both padded FFT workspaces can overlap despite the host
    staging boundary.
    """

    if any(isinstance(accumulator, np.ndarray) for accumulator in accumulators):
        result.block_until_ready()
        gc.collect()
    return result


_LARGE_RELION_SOLVENT_MASK_COORDINATE_BYTES_LIMIT = 2 * 1024**3

def _relion_solvent_mask_unfused_coordinate_bytes(volume_shape) -> int:
    """Estimate the promoted coordinate stack used by ``raised_cosine_mask``."""

    return math.prod(int(size) for size in volume_shape) * 3 * np.dtype(np.float64).itemsize


def _large_relion_solvent_mask_uses_compiled_builder(volume_shape) -> bool:
    """Return whether the unfused solvent-mask coordinate stack is too large."""

    return (
        _relion_solvent_mask_unfused_coordinate_bytes(volume_shape)
        > _LARGE_RELION_SOLVENT_MASK_COORDINATE_BYTES_LIMIT
    )


@functools.cache
def _compiled_relion_solvent_mask(volume_shape, *, dtype=None):
    """Cache a shape-specialized builder, without retaining a mask array."""

    volume_shape = tuple(int(size) for size in volume_shape)

    @jax.jit
    def build(radius, radius_p, offset):
        return mask.raised_cosine_mask(
            volume_shape,
            radius=radius,
            radius_p=radius_p,
            offset=offset,
            dtype=dtype,
        )

    return build


def _make_relion_solvent_mask(volume_shape, *, radius, radius_p, offset, dtype=None):
    """Build a RELION solvent mask without materializing a giant coordinate stack."""

    volume_shape = tuple(int(size) for size in volume_shape)
    if not _large_relion_solvent_mask_uses_compiled_builder(volume_shape):
        return mask.raised_cosine_mask(
            volume_shape,
            radius=radius,
            radius_p=radius_p,
            offset=offset,
            dtype=dtype,
        )

    estimated_bytes = _relion_solvent_mask_unfused_coordinate_bytes(volume_shape)
    logger.info(
        "RELION box-scale solvent mask fused construction: shape=%s "
        "estimated_unfused_coordinate_bytes=%d",
        volume_shape,
        estimated_bytes,
    )
    solvent_mask = _compiled_relion_solvent_mask(
        volume_shape, **({"dtype": dtype} if dtype is not None else {}),
    )(
        radius,
        radius_p,
        offset,
    )
    solvent_mask.block_until_ready()
    logger.info(
        "RELION box-scale solvent mask ready: shape=%s dtype=%s",
        volume_shape,
        solvent_mask.dtype,
    )
    return solvent_mask


def _apply_relion_solvent_flatten_k1(
    volume_ft_flat,
    solvent_mask,
    volume_shape,
    *,
    half_index,
):
    """Apply the K=1 solvent mask and host-stage box-scale FFT results."""

    vol_real = fourier_transform_utils.get_idft3(volume_ft_flat.reshape(volume_shape))
    flattened = fourier_transform_utils.get_dft3(vol_real * solvent_mask).reshape(-1)
    if not _large_relion_solvent_mask_uses_compiled_builder(volume_shape):
        return flattened

    # JAX dispatch is asynchronous.  At box 800, even after the first half's
    # real-space volume and mask are released, retaining its complex128 FFT
    # output leaves too little room for the second normalized FFT allocation.
    # Finish the exact existing FFT, copy its completed bits to host memory,
    # and release all three device buffers before starting the next half.  The
    # transform arithmetic, normalization, dtype, and flattened shape remain
    # unchanged; only the storage owner crosses the device/host boundary.
    flattened.block_until_ready()
    flattened_host = np.array(jax.device_get(flattened), copy=True, order="C")
    _delete_device_array(flattened)
    _delete_device_array(vol_real)
    _delete_device_array(solvent_mask)
    output_device_deleted = _device_array_is_deleted(flattened)
    vol_real_deleted = _device_array_is_deleted(vol_real)
    solvent_mask_deleted = _device_array_is_deleted(solvent_mask)
    del flattened, vol_real, solvent_mask
    gc.collect()
    logger.info(
        "RELION box-scale solvent flatten lifecycle: half=%d shape=%s "
        "output_ready=True output_host=True output_device_deleted=%s "
        "vol_real_deleted=%s solvent_mask_deleted=%s output_dtype=%s "
        "output_c_contiguous=%s",
        int(half_index) + 1,
        tuple(int(size) for size in volume_shape),
        output_device_deleted,
        vol_real_deleted,
        solvent_mask_deleted,
        flattened_host.dtype,
        bool(flattened_host.flags.c_contiguous),
    )
    return flattened_host

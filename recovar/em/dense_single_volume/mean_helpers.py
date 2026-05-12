"""Mean-volume / noise-variance / K-class helpers for the iteration loop.

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
from recovar.em.dense_single_volume.helpers.orientation_priors import (
    collapse_rotation_posterior_to_direction_prior,
)
from recovar.em.dense_single_volume.helpers.types import make_noise_stats

logger = logging.getLogger(__name__)


def _normalize_noise_variance_per_half(init_noise_variance, n_halves=2):
    """Return a list of per-half flattened noise-variance arrays.

    RELION stores and updates ``sigma2_noise`` separately for each half-model.
    Legacy RECOVAR callers pass one shared image-shaped array; keep that path
    by duplicating the shared vector.
    """
    if n_halves <= 0:
        raise ValueError(f"n_halves must be positive, got {n_halves}")

    if isinstance(init_noise_variance, (list, tuple)):
        if len(init_noise_variance) != n_halves:
            raise ValueError(
                f"Expected {n_halves} per-half noise arrays, got {len(init_noise_variance)}",
            )
        per_half = [jnp.asarray(noise_k).reshape(-1) for noise_k in init_noise_variance]
    else:
        noise_arr = jnp.asarray(init_noise_variance)
        if noise_arr.ndim == 1:
            shared = noise_arr.reshape(-1)
            per_half = [jnp.array(shared) for _ in range(n_halves)]
        elif noise_arr.ndim == 2 and noise_arr.shape[0] == n_halves:
            per_half = [jnp.asarray(noise_arr[k]).reshape(-1) for k in range(n_halves)]
        else:
            raise ValueError(
                "init_noise_variance must be a flat shared array or a "
                f"({n_halves}, image_size) per-half array; got shape {tuple(noise_arr.shape)}",
            )

    sizes = [int(noise_k.size) for noise_k in per_half]
    if len(set(sizes)) != 1:
        raise ValueError(f"Per-half noise arrays must have the same size; got {sizes}")
    return per_half


def _mean_noise_variance(noise_variance_per_half):
    """Average per-half image noise for diagnostics and compatibility outputs."""
    return jnp.mean(
        jnp.stack([jnp.asarray(noise_k).reshape(-1) for noise_k in noise_variance_per_half], axis=0),
        axis=0,
    )


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


def _combined_noise_stats(noise_stats_per_half):
    """Sum half-set noise sufficient statistics before RELION Class3D normalization."""

    stats = [stats_k for stats_k in noise_stats_per_half if stats_k is not None]
    if not stats:
        return None
    wsum_sigma2_noise = np.sum(
        [np.asarray(stats_k.wsum_sigma2_noise, dtype=np.float64) for stats_k in stats],
        axis=0,
    )
    wsum_img_power = np.sum(
        [np.asarray(stats_k.wsum_img_power, dtype=np.float64) for stats_k in stats],
        axis=0,
    )
    wsum_sigma2_offset = float(sum(float(stats_k.wsum_sigma2_offset) for stats_k in stats))
    sumw = float(sum(float(stats_k.sumw) for stats_k in stats))

    if any(stats_k.wsum_noise_a2 is not None for stats_k in stats):
        wsum_noise_a2 = np.sum(
            [
                np.zeros_like(wsum_sigma2_noise)
                if stats_k.wsum_noise_a2 is None
                else np.asarray(stats_k.wsum_noise_a2, dtype=np.float64)
                for stats_k in stats
            ],
            axis=0,
        )
    else:
        wsum_noise_a2 = None

    if any(stats_k.wsum_noise_xa is not None for stats_k in stats):
        wsum_noise_xa = np.sum(
            [
                np.zeros_like(wsum_sigma2_noise)
                if stats_k.wsum_noise_xa is None
                else np.asarray(stats_k.wsum_noise_xa, dtype=np.float64)
                for stats_k in stats
            ],
            axis=0,
        )
    else:
        wsum_noise_xa = None

    return make_noise_stats(
        wsum_sigma2_noise=wsum_sigma2_noise,
        wsum_img_power=wsum_img_power,
        wsum_sigma2_offset=wsum_sigma2_offset,
        sumw=sumw,
        wsum_noise_a2=wsum_noise_a2,
        wsum_noise_xa=wsum_noise_xa,
        array_dtype=jnp.float64,
    )


def _combined_class_direction_prior_from_halves(class_rotation_posterior_per_half, n_classes: int, healpix_order: int):
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
        combined_priors.append(collapse_rotation_posterior_to_direction_prior(combined, healpix_order))
    return np.stack(combined_priors, axis=0)


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
    )


def _apply_relion_initial_lowpass_filter(
    volume_ft_flat, volume_shape, voxel_size, ini_high_angstrom, filter_edgewidth=5
):
    """Apply RELION's ``initialLowPassFilterReferences`` to a full Fourier volume."""
    if ini_high_angstrom is None or float(ini_high_angstrom) <= 0.0:
        return volume_ft_flat
    from recovar.heterogeneity import locres

    filtered = locres.low_pass_filter_map(
        jnp.asarray(volume_ft_flat).reshape(volume_shape),
        volume_shape[0],
        float(ini_high_angstrom),
        float(voxel_size),
        int(filter_edgewidth),
        do_highpass_instead=False,
        volume_shape=volume_shape,
    )
    return filtered.reshape(-1)


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
    mean_signal_variance_per_half,
    n_classes: int,
    k_class_enabled: bool,
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
) -> None:
    """Run one iteration's regularized reconstruction + post-processing.

    Mutates ``means`` in place. Performs Wiener reconstruction (per-class for
    K>1, per-half for K=1), optional pre-mask debug dump, RELION solvent
    flatten, and iter-1 firstiter_cc low-pass filter.
    """

    _t_recon = time.time()
    cs_int = int(cs) if cs is not None else None
    if k_class_enabled:
        shared_classes = jnp.stack(
            [
                _reconstruct_volume_eager(
                    Ft_ctf_combined[class_idx],
                    Ft_y_combined[class_idx],
                    volume_shape,
                    padding_factor,
                    tau=mean_signal_variance[class_idx],
                    tau2_fudge=tau2_fudge,
                    projection_padding_factor=projection_padding_factor,
                    minres_map=relion_minres_map,
                    current_size=cs_int,
                ).reshape(-1)
                for class_idx in range(n_classes)
            ],
            axis=0,
        )
        means[0] = shared_classes
        means[1] = shared_classes
    else:
        for k in range(2):
            Ft_y_k_local = Ft_y_0 if k == 0 else Ft_y_1
            Ft_ctf_k_local = Ft_ctf_0 if k == 0 else Ft_ctf_1
            means[k] = _reconstruct_volume_eager(
                Ft_ctf_k_local,
                Ft_y_k_local,
                volume_shape,
                padding_factor,
                tau=mean_signal_variance_per_half[k],
                tau2_fudge=tau2_fudge,
                projection_padding_factor=projection_padding_factor,
                minres_map=relion_minres_map,
                current_size=cs_int,
            ).reshape(-1)

    for k in range(2):
        # Diagnostic: dump pre-mask Wiener output when env var set.
        _premask_dump = os.environ.get("RECOVAR_PREMASK_DUMP_DIR")
        if _premask_dump:
            import pathlib

            pathlib.Path(_premask_dump).mkdir(parents=True, exist_ok=True)
            np.savez(
                pathlib.Path(_premask_dump) / f"recovar_premask_it{iteration + 1:03d}_half{k + 1}.npz",
                iteration=np.int32(iteration + 1),
                half=np.int32(k + 1),
                current_size=np.int32(cs),
                grid_size=np.int32(grid_size),
                voxel_size=np.float32(cryo.voxel_size),
                volume_shape=np.asarray(volume_shape, dtype=np.int32),
                means_premask=np.asarray(means[k], dtype=np.complex64),
            )

        # RELION's solventFlatten (ml_optimiser.cpp:5469): mask the
        # reconstructed reference outside particle_diameter to remove
        # solvent noise before the next E-step's projections.
        if particle_diameter_ang is not None and particle_diameter_ang > 0:
            flatten_radius = particle_diameter_ang / (2.0 * cryo.voxel_size)
            solvent_mask = mask.raised_cosine_mask(
                volume_shape,
                radius=flatten_radius,
                radius_p=flatten_radius + relion_width_mask_edge,
                offset=jnp.zeros(3),
            )
            if k_class_enabled:
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
        if relion_firstiter_cc_this_iter:
            if k_class_enabled:
                means[k] = jnp.stack(
                    [
                        _apply_relion_initial_lowpass_filter(
                            means[k][class_idx],
                            volume_shape,
                            cryo.voxel_size,
                            relion_firstiter_ini_high_angstrom,
                            filter_edgewidth=relion_width_mask_edge,
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
                    filter_edgewidth=relion_width_mask_edge,
                )
    if relion_firstiter_cc_this_iter and relion_firstiter_ini_high_angstrom is not None:
        logger.info(
            "RELION iter-1 CC emulation: reapplying ini_high low-pass filter at %.2f A",
            float(relion_firstiter_ini_high_angstrom),
        )
    logger.info("Regularized reconstruction (2 halves + flatten): %.1fs", time.time() - _t_recon)


# ---------------------------------------------------------------------------
# Main refinement loop
# ---------------------------------------------------------------------------

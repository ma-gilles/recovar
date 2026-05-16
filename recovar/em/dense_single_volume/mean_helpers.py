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
    relion_fmask_edge: int,
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
    if relion_firstiter_cc_this_iter and relion_firstiter_ini_high_angstrom is not None:
        logger.info(
            "RELION iter-1 CC emulation: reapplying ini_high low-pass filter at %.2f A",
            float(relion_firstiter_ini_high_angstrom),
        )
    logger.info("Regularized reconstruction (2 halves + flatten): %.1fs", time.time() - _t_recon)


# ---------------------------------------------------------------------------
# C1 (RELION-parity) sigma_offset update from posterior moments
# ---------------------------------------------------------------------------
from dataclasses import dataclass as _dataclass  # noqa: E402  -- inline import


@_dataclass
class SigmaOffsetUpdateResult:
    """Posterior-weighted ``sigma_offset`` update result.

    ``per_class_sigma_offset_angstrom`` is the K-class diagnostic, currently
    logged only (the cross-class aggregate ``current_sigma_offset_angstrom``
    is what feeds the next iteration's translation prior). Both halves of
    the dual cross-class / hard-assignment fallback formula are folded
    into ``current_sigma_offset_angstrom`` before return.
    """

    current_sigma_offset_angstrom: float
    per_class_sigma_offset_angstrom: np.ndarray | None


def update_c1_sigma_offset_from_posterior(
    *,
    noise_stats_per_half,
    noise_stats_per_half_per_class,
    current_sigma_offset_angstrom: float,
    n_classes: int,
    k_class_enabled: bool,
    state_fallback_offsets_angstrom: float,
) -> SigmaOffsetUpdateResult:
    """RELION C1 posterior-weighted ``sigma_offset`` update.

    Prefer RELION's posterior-weighted sufficient statistic:

        sigma2_offset_new = wsum_sigma2_offset / (2 * sum_weight)

    for 2D single-particle data. Fall back to the older hard-assignment
    proxy only when a path does not propagate the full posterior moment
    (``sigma2_offset_wsum == 0``). Both halves of the M-step contribute.
    """

    sigma2_offset_wsum = 0.0
    sigma2_offset_sumw = 0.0
    for stats_k in noise_stats_per_half:
        if stats_k is None:
            continue
        sigma2_offset_wsum += float(getattr(stats_k, "wsum_sigma2_offset", 0.0))
        sigma2_offset_sumw += float(getattr(stats_k, "sumw", 0.0))
    # D.2: per-class sigma_offset diagnostic. RELION Class3D maintains K
    # independent sigma_offset values; recovar currently uses the
    # cross-class aggregate. Compute and log per-class sigmas to gauge
    # whether the per-class refactor is justified for this fixture.
    per_class_sigma_offset = None
    if k_class_enabled:
        per_class_w = np.zeros(n_classes, dtype=np.float64)
        per_class_n = np.zeros(n_classes, dtype=np.float64)
        for half_per_class in noise_stats_per_half_per_class:
            if half_per_class is None:
                continue
            for c, stats_c in enumerate(half_per_class):
                if stats_c is None:
                    continue
                per_class_w[c] += float(getattr(stats_c, "wsum_sigma2_offset", 0.0))
                per_class_n[c] += float(getattr(stats_c, "sumw", 0.0))
        min_sigma2 = 2.0
        per_class_sigma_offset = np.full(n_classes, current_sigma_offset_angstrom, dtype=np.float64)
        for c in range(n_classes):
            if per_class_w[c] > 0.0 and per_class_n[c] > 0.0:
                s2 = max(per_class_w[c] / (2.0 * per_class_n[c]), min_sigma2)
                per_class_sigma_offset[c] = float(np.sqrt(s2))
        logger.info(
            "C1: per-class sigma_offset = [%s] (cross-class aggregate %.3f Å)",
            ", ".join(f"{s:.3f}" for s in per_class_sigma_offset),
            float(np.sqrt(max(sigma2_offset_wsum / max(2.0 * sigma2_offset_sumw, 1e-30), min_sigma2)))
            if sigma2_offset_wsum > 0
            else current_sigma_offset_angstrom,
        )
    if sigma2_offset_wsum > 0.0 and sigma2_offset_sumw > 0.0:
        min_sigma2_angstrom2 = 2.0
        sigma2_offset_angstrom2 = max(
            sigma2_offset_wsum / (2.0 * sigma2_offset_sumw),
            min_sigma2_angstrom2,
        )
        current_sigma_offset_angstrom = float(np.sqrt(sigma2_offset_angstrom2))
        logger.info(
            "C1: sigma_offset updated %.3f Å from posterior variance (clamp sigma^2 >= %.3f Å^2)",
            current_sigma_offset_angstrom,
            min_sigma2_angstrom2,
        )
    else:
        new_sigma_offset_angstrom = state_fallback_offsets_angstrom
        if np.isfinite(new_sigma_offset_angstrom) and new_sigma_offset_angstrom > 0:
            min_sigma_angstrom = float(np.sqrt(2.0))  # RELION min_sigma2_offset = 2 Å²
            current_sigma_offset_angstrom = max(
                float(new_sigma_offset_angstrom),
                min_sigma_angstrom,
            )
            logger.info(
                "C1 fallback: sigma_offset updated %.3f Å from hard assignments (clamp >= %.3f Å)",
                current_sigma_offset_angstrom,
                min_sigma_angstrom,
            )
    return SigmaOffsetUpdateResult(
        current_sigma_offset_angstrom=current_sigma_offset_angstrom,
        per_class_sigma_offset_angstrom=per_class_sigma_offset,
    )


# ---------------------------------------------------------------------------
# Unregularized half-map reconstruction + sign alignment
# ---------------------------------------------------------------------------


@_dataclass
class UnregularizedHalfmapResult:
    """Unregularized half-maps + sign-flip telemetry.

    ``unregularized_means`` is a 2-element list, one per half. For K-class
    refinement both halves point at the same shared K-stack (RELION's
    Class3D shares one mean across halves).

    ``aligned_means`` is the input ``means`` argument after sign alignment
    against ``previous_means`` (passed in so the caller can pick it up;
    the helper also mutates ``means`` in place for convenience).
    """

    unregularized_means: list
    aligned_means: list
    any_sign_flipped: bool


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
    k_class_enabled: bool,
    tau2_fudge: float,
    padding_factor: int,
    projection_padding_factor: int,
    minres_map: int,
    need_unreg_means: bool,
) -> UnregularizedHalfmapResult:
    """Reconstruct unregularized half-maps (only when diagnostics need them)
    and sign-align the regularized means against the previous-iter reference.

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
        if k_class_enabled:
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
                )
                for Ft_ctf_half, Ft_y_half in zip(Ft_ctf_per_half, Ft_y_per_half)
            ]
    else:
        unreg_means = [None, None]

    any_sign_flipped = False
    if k_class_enabled:
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
                any_sign_flipped = True
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
                any_sign_flipped = True
                logger.info("Aligned half-%d volume sign to the previous reference", k + 1)
    logger.info(
        "Unregularized reconstruction (2 halves): %.1fs%s",
        time.time() - _t_unreg,
        "" if need_unreg_means else " (skipped; diagnostics disabled)",
    )
    return UnregularizedHalfmapResult(
        unregularized_means=unreg_means,
        aligned_means=means,
        any_sign_flipped=any_sign_flipped,
    )


# ---------------------------------------------------------------------------
# RELION posterior-weighted noise update
# ---------------------------------------------------------------------------


@_dataclass
class NoiseUpdateResult:
    """Posterior-weighted noise-variance update output.

    All four arrays are normalized to RELION conventions:
    - ``noise_from_res`` / ``noise_from_res_per_half`` are per-shell
      sigma2_noise (1D arrays of length ``n_shells``).
    - ``noise_variance_per_half`` is the same data unrolled to a flat
      ``ravel(make_radial_noise(...))`` representation for the engine.
    - ``noise_variance`` is the mean of the two halves' radial.
    - ``previous_noise_radial[_per_half]`` carry the per-shell values
      forward to the next iteration's update.
    """

    noise_from_res: np.ndarray
    noise_from_res_per_half: list
    noise_variance_per_half: list
    noise_variance: object
    previous_noise_radial: object
    previous_noise_radial_per_half: list


def update_posterior_noise_variance(
    *,
    noise_stats_per_half,
    noise_variance_per_half: list,
    previous_noise_radial_per_half: list,
    previous_noise_radial,
    cryo,
    k_class_enabled: bool,
    relion_firstiter_cc_this_iter: bool,
    iteration: int,
    cs: int,
    maybe_dump_noise_update_debug=None,
) -> NoiseUpdateResult:
    """RELION-style posterior-weighted noise update.

    Sums the ``wsum_sigma2_noise``/``wsum_img_power`` accumulators from
    both half-sets and normalizes via RELION's M-step formula. K-class
    refinement shares one sigma2_noise across classes (Class3D ordering);
    K=1 keeps independent per-half sigma2_noise.

    When ``relion_firstiter_cc_this_iter`` is true, keeps the previous
    sigma2_noise (matching RELION's iter-1 CC emulation, which skips the
    first-iter noise update).
    """

    from recovar.reconstruction import noise

    if noise_stats_per_half[0] is None or noise_stats_per_half[1] is None:
        raise RuntimeError(
            "RELION mode expected per-half NoiseStats from the EM engine; "
            "ensure accumulate_noise=True is plumbed through pass 2.",
        )

    if relion_firstiter_cc_this_iter:
        noise_from_res_per_half = [np.asarray(noise_k, dtype=np.float64) for noise_k in previous_noise_radial_per_half]
        noise_from_res = np.mean(np.stack(noise_from_res_per_half, axis=0), axis=0)
        logger.info(
            "RELION iter-1 CC emulation: keeping previous sigma2_noise (skip first-iter noise update)",
        )
        return NoiseUpdateResult(
            noise_from_res=noise_from_res,
            noise_from_res_per_half=noise_from_res_per_half,
            noise_variance_per_half=noise_variance_per_half,
            noise_variance=_mean_noise_variance(noise_variance_per_half),
            previous_noise_radial=previous_noise_radial,
            previous_noise_radial_per_half=previous_noise_radial_per_half,
        )

    if k_class_enabled:
        combined_noise_stats = _combined_noise_stats(noise_stats_per_half)
        if combined_noise_stats is None:
            raise RuntimeError("K-class noise update expected at least one NoiseStats object")
        noise_shared = noise.normalize_wsum_to_sigma2_noise(
            np.asarray(combined_noise_stats.wsum_sigma2_noise, dtype=np.float64),
            np.asarray(combined_noise_stats.wsum_img_power, dtype=np.float64),
            combined_noise_stats.sumw,
            cryo.image_shape,
        )
        noise_from_res = np.asarray(noise_shared, dtype=np.float64)
        noise_from_res_per_half = [noise_from_res.copy(), noise_from_res.copy()]
        noise_variance_shared = jnp.asarray(
            noise.make_radial_noise(noise_shared, cryo.image_shape),
        ).reshape(-1)
        noise_variance_per_half = [noise_variance_shared, noise_variance_shared]
    else:
        noise_from_res_per_half = []
        for k_noise, stats_k in enumerate(noise_stats_per_half):
            noise_k = noise.normalize_wsum_to_sigma2_noise(
                np.asarray(stats_k.wsum_sigma2_noise, dtype=np.float64),
                np.asarray(stats_k.wsum_img_power, dtype=np.float64),
                stats_k.sumw,
                cryo.image_shape,
            )
            noise_from_res_per_half.append(np.asarray(noise_k, dtype=np.float64))
            noise_variance_per_half[k_noise] = jnp.asarray(
                noise.make_radial_noise(noise_k, cryo.image_shape),
            ).reshape(-1)
        noise_from_res = np.mean(np.stack(noise_from_res_per_half, axis=0), axis=0)

    # Log per-shell noise comparison (first 10 shells) for convergence diagnostics.
    old_noise_radial = previous_noise_radial
    n_log = min(10, len(noise_from_res), len(old_noise_radial))
    logger.info(
        "Noise update per shell (first %d): old=[%s] new=[%s]",
        n_log,
        ", ".join(f"{float(x):.3e}" for x in old_noise_radial[:n_log]),
        ", ".join(f"{float(x):.3e}" for x in noise_from_res[:n_log]),
    )
    if maybe_dump_noise_update_debug is not None:
        maybe_dump_noise_update_debug(
            iteration=iteration,
            current_size=cs,
            image_shape=cryo.image_shape,
            noise_stats_per_half=noise_stats_per_half,
            previous_noise_radial_per_half=previous_noise_radial_per_half,
            noise_from_res_per_half=noise_from_res_per_half,
            noise_from_res=noise_from_res,
        )

    new_previous_noise_radial = jnp.asarray(noise_from_res, dtype=jnp.float32)
    noise_variance = _mean_noise_variance(noise_variance_per_half)
    return NoiseUpdateResult(
        noise_from_res=noise_from_res,
        noise_from_res_per_half=noise_from_res_per_half,
        noise_variance_per_half=noise_variance_per_half,
        noise_variance=noise_variance,
        previous_noise_radial=new_previous_noise_radial,
        previous_noise_radial_per_half=noise_from_res_per_half,
    )


# ---------------------------------------------------------------------------
# Main refinement loop
# ---------------------------------------------------------------------------

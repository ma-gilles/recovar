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


def _relion_optimizer_average_pmax(max_posterior_per_half, normalization_mass_per_half=None):
    """Return RELION's optimizer Pmax scalar and its normalization mass.

    In split-half refinement, RELION computes this independently per half,
    then broadcasts half 1's scalar for shared scheduling. Class3D divides
    half 1's Pmax sum by that half's retained M-step posterior mass
    (``sum(wsum_model.pdf_class)``). The concatenated array is still returned
    for per-particle diagnostics.
    """

    per_half = [np.asarray(pmax, dtype=np.float32).reshape(-1) for pmax in max_posterior_per_half]
    if not per_half:
        raise ValueError("RELION average Pmax requires at least one half-set")
    combined = np.concatenate(per_half, axis=0)
    numerator = float(np.sum(per_half[0], dtype=np.float64))
    if normalization_mass_per_half is None:
        denominator = float(per_half[0].size)
    else:
        half1_mass = normalization_mass_per_half[0]
        if half1_mass is None:
            raise ValueError("RELION average Pmax requires half-1 M-step posterior mass")
        denominator = float(np.asarray(half1_mass, dtype=np.float64))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError(f"RELION average-Pmax denominator must be finite and positive, got {denominator}")
    average = numerator / denominator
    if not np.isfinite(average) or average < 0.0 or average > 1.0 + 1e-6:
        raise ValueError(f"RELION average Pmax must be a probability, got {average}")
    return combined, float(average), denominator


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

    def _sum_optional_field(name: str, like):
        values = [getattr(stats_k, name, None) for stats_k in stats]
        if all(value is None for value in values):
            return None
        return np.sum(
            [
                np.zeros_like(like, dtype=np.float64) if value is None else np.asarray(value, dtype=np.float64)
                for value in values
            ],
            axis=0,
        )

    wsum_noise_a2 = _sum_optional_field("wsum_noise_a2", wsum_sigma2_noise)
    wsum_noise_xa = _sum_optional_field("wsum_noise_xa", wsum_sigma2_noise)
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
    return_real_space=False,
    accumulator_volume_shape=None,
    tau_is_1d=False,
    preserve_output_precision=False,
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
    )


def _apply_relion_initial_lowpass_filter(
    volume_ft_flat, volume_shape, voxel_size, ini_high_angstrom, filter_edgewidth=5
):
    """Apply RELION's ``initialLowPassFilterReferences`` to a full Fourier volume."""
    if ini_high_angstrom is None or float(ini_high_angstrom) <= 0.0:
        return volume_ft_flat
    from recovar.em.initial_model.bootstrap_iref import initial_low_pass_filter_references

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
    if k_class_enabled:
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
            ).reshape(-1)

    for k in range(2):
        # Diagnostic: dump pre-mask Wiener output when env var set.
        _premask_dump = os.environ.get("RECOVAR_PREMASK_DUMP_DIR")
        if _premask_dump:
            import pathlib

            pathlib.Path(_premask_dump).mkdir(parents=True, exist_ok=True)
            _preserve_premask_dtype = os.environ.get(
                "RECOVAR_PREMASK_DUMP_PRESERVE_DTYPE", ""
            ).strip().lower() not in {"", "0", "false", "no", "off"}
            _premask_fourier = np.asarray(means[k])
            if k_class_enabled:
                _premask_real = np.stack(
                    [
                        np.asarray(
                            fourier_transform_utils.get_idft3(
                                means[k][class_idx].reshape(volume_shape)
                            )
                        ).real
                        for class_idx in range(n_classes)
                    ],
                    axis=0,
                )
            else:
                _premask_real = np.asarray(
                    fourier_transform_utils.get_idft3(means[k].reshape(volume_shape))
                ).real
            np.savez(
                pathlib.Path(_premask_dump) / f"recovar_premask_it{iteration + 1:03d}_half{k + 1}.npz",
                iteration=np.int32(iteration + 1),
                half=np.int32(k + 1),
                current_size=np.int32(cs),
                grid_size=np.int32(grid_size),
                voxel_size=np.float32(cryo.voxel_size),
                volume_shape=np.asarray(volume_shape, dtype=np.int32),
                means_premask=(
                    _premask_fourier
                    if _preserve_premask_dtype
                    else np.asarray(_premask_fourier, dtype=np.complex64)
                ),
                means_premask_real=(
                    _premask_real
                    if _preserve_premask_dtype
                    else np.asarray(_premask_real, dtype=np.float32)
                ),
                dump_preserve_dtype=np.int32(int(_preserve_premask_dtype)),
            )

        # RELION filters Iref inside maximizationOtherParameters, then calls
        # solventFlatten from the outer iteration loop.  These operations do
        # not commute: masking in real space after the Fourier low-pass adds a
        # small, deterministic high-shell tail.
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

    RELION gold-standard refinement has one model per half-set, so each half
    updates and consumes its own sigma offset. The scalar value remains the
    mean of the pair for backward-compatible telemetry only.
    """

    current_sigma_offset_angstrom: float
    current_sigma_offset_angstrom_per_half: list[float]
    per_class_sigma_offset_angstrom: np.ndarray | None

    @property
    def per_half_sigma_offset_angstrom(self):
        """Backward-compatible alias for pre-PR157 callers."""

        return np.asarray(self.current_sigma_offset_angstrom_per_half, dtype=np.float64)


def _sigma_offset_from_moment(
    wsum: float,
    sumw: float,
    *,
    current_sigma_offset_angstrom: float,
    state_fallback_offsets_angstrom: float,
) -> float:
    min_sigma2_angstrom2 = 2.0
    if wsum > 0.0 and sumw > 0.0:
        return float(np.sqrt(max(wsum / (2.0 * sumw), min_sigma2_angstrom2)))
    if np.isfinite(state_fallback_offsets_angstrom) and state_fallback_offsets_angstrom > 0.0:
        return max(float(state_fallback_offsets_angstrom), float(np.sqrt(min_sigma2_angstrom2)))
    return float(current_sigma_offset_angstrom)


def update_c1_sigma_offset_from_posterior(
    *,
    noise_stats_per_half,
    noise_stats_per_half_per_class,
    current_sigma_offset_angstrom: float | None = None,
    current_sigma_offset_angstrom_per_half=None,
    n_classes: int,
    k_class_enabled: bool,
    state_fallback_offsets_angstrom: float,
) -> SigmaOffsetUpdateResult:
    """RELION C1 posterior-weighted ``sigma_offset`` update per half-set.

    Prefer RELION's posterior-weighted sufficient statistic:

        sigma2_offset_new = wsum_sigma2_offset / (2 * sum_weight)

    for 2D single-particle data. A half without a propagated posterior moment
    uses the hard-assignment fallback independently; pooling the other half's
    posterior into it would not match RELION's gold-standard models.
    """

    if current_sigma_offset_angstrom_per_half is None:
        if current_sigma_offset_angstrom is None:
            raise ValueError("a scalar or per-half current sigma offset is required")
        current_per_half = np.full(2, float(current_sigma_offset_angstrom), dtype=np.float64)
    else:
        current_per_half = np.asarray(current_sigma_offset_angstrom_per_half, dtype=np.float64).reshape(-1)
        if current_per_half.size != 2 or not np.all(np.isfinite(current_per_half)):
            raise ValueError("current_sigma_offset_angstrom_per_half must contain two finite values")
    per_half_values = []
    pooled_wsum = 0.0
    pooled_sumw = 0.0
    for half_idx, stats_k in enumerate(noise_stats_per_half):
        if stats_k is None:
            per_half_values.append(
                _sigma_offset_from_moment(
                    0.0,
                    0.0,
                    current_sigma_offset_angstrom=float(current_per_half[half_idx]),
                    state_fallback_offsets_angstrom=state_fallback_offsets_angstrom,
                )
            )
            continue
        wsum_k = float(getattr(stats_k, "wsum_sigma2_offset", 0.0))
        sumw_k = float(getattr(stats_k, "sumw", 0.0))
        pooled_wsum += wsum_k
        pooled_sumw += sumw_k
        per_half_values.append(
            _sigma_offset_from_moment(
                wsum_k,
                sumw_k,
                current_sigma_offset_angstrom=float(current_per_half[half_idx]),
                state_fallback_offsets_angstrom=state_fallback_offsets_angstrom,
            )
        )
    if len(per_half_values) != 2:
        raise ValueError(f"noise_stats_per_half must contain two halves, got {len(per_half_values)}")
    per_half_sigma_offset = np.asarray(per_half_values, dtype=np.float64)
    if k_class_enabled:
        shared_sigma_offset = _sigma_offset_from_moment(
            pooled_wsum,
            pooled_sumw,
            current_sigma_offset_angstrom=float(np.mean(current_per_half)),
            state_fallback_offsets_angstrom=state_fallback_offsets_angstrom,
        )
        per_half_sigma_offset[:] = shared_sigma_offset
    current_sigma_offset_angstrom = float(np.mean(per_half_sigma_offset))
    # D.2: per-class sigma_offset diagnostic. RELION Class3D maintains one
    # shared sigma2_offset in model_general; per-class values here are logged
    # only to help diagnose skewed class posteriors without changing the live
    # shared translation prior.
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
            current_sigma_offset_angstrom,
        )
    logger.info(
        "C1: sigma_offset updated per half [%.3f, %.3f] Å (mean %.3f Å)",
        per_half_sigma_offset[0],
        per_half_sigma_offset[1],
        current_sigma_offset_angstrom,
    )
    return SigmaOffsetUpdateResult(
        current_sigma_offset_angstrom=current_sigma_offset_angstrom,
        current_sigma_offset_angstrom_per_half=per_half_sigma_offset.tolist(),
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
    accumulator_volume_shape=None,
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
# RELION norm/scale correction update
# ---------------------------------------------------------------------------


@_dataclass
class NormScaleCorrectionUpdateResult:
    """Native RELION norm/scale correction state for the next iteration."""

    norm_corrections_per_half: list
    avg_norm_correction_per_half: list
    group_scale_corrections_per_half: list
    image_corrections_per_half: list
    scale_corrections_per_half: list
    zero_norm_residual_counts: list


def _half_list_or_none(values, *, n_halves: int, name: str):
    if values is None:
        return [None] * n_halves
    if not isinstance(values, (list, tuple)) or len(values) != n_halves:
        raise ValueError(f"{name} must be a {n_halves}-element list/tuple or None")
    return list(values)


def _derive_group_scale_from_image_scale(scale_per_image, group_ids, n_groups):
    group_scale = np.ones(n_groups, dtype=np.float64)
    if scale_per_image is None:
        return group_scale
    sums = np.bincount(group_ids, weights=np.asarray(scale_per_image, dtype=np.float64), minlength=n_groups)
    counts = np.bincount(group_ids, minlength=n_groups)
    present = counts > 0
    group_scale[present] = sums[present] / counts[present]
    return group_scale


def update_relion_norm_scale_corrections(
    *,
    noise_stats_per_half,
    image_corrections_per_half=None,
    scale_corrections_per_half=None,
    group_ids_per_half=None,
    group_count_per_half=None,
    group_scale_corrections_per_half=None,
    avg_norm_correction_per_half=None,
    relion_firstiter_cc_this_iter: bool = False,
    do_norm_correction: bool = True,
    do_scale_correction: bool = True,
    scale_relaxation_mu: float = 0.0,
    eps: float = 1e-30,
) -> NormScaleCorrectionUpdateResult:
    """Update RELION-style per-image norm and per-group scale corrections.

    The existing scoring paths consume two per-image arrays:
    ``image_corrections = (avg_norm / normcorr) * scale[group_id]`` and
    ``scale_corrections = scale[group_id]``.  This helper updates the native
    RELION state behind those arrays from M-step sufficient statistics.

    Scale sufficient statistics are expected to match RELION's
    ``wsum_model.wsum_signal_product`` and ``wsum_model.wsum_reference_power``:
    the collection site must divide XA by the old scale and AA by its square
    before accumulation.

    RELION's average norm-correction numerator contains one updated
    ``normcorr`` per particle, without posterior weighting.  Its denominator
    is ``wsum_model.pdf_class.sum()``, i.e. the retained significant-support
    posterior mass.  That mass is normally slightly smaller than the particle
    count, so using a conventional arithmetic mean introduces a systematic
    normalization drift.
    """

    stats_per_half = _half_list_or_none(noise_stats_per_half, n_halves=2, name="noise_stats_per_half")
    image_corr_in = _half_list_or_none(image_corrections_per_half, n_halves=2, name="image_corrections_per_half")
    scale_corr_in = _half_list_or_none(scale_corrections_per_half, n_halves=2, name="scale_corrections_per_half")
    group_ids_in = _half_list_or_none(group_ids_per_half, n_halves=2, name="group_ids_per_half")
    group_count_in = _half_list_or_none(group_count_per_half, n_halves=2, name="group_count_per_half")
    group_scale_in = _half_list_or_none(
        group_scale_corrections_per_half,
        n_halves=2,
        name="group_scale_corrections_per_half",
    )
    avg_norm_in = (
        [1.0, 1.0]
        if avg_norm_correction_per_half is None
        else list(avg_norm_correction_per_half)
        if isinstance(avg_norm_correction_per_half, (list, tuple)) and len(avg_norm_correction_per_half) == 2
        else None
    )
    if avg_norm_in is None:
        raise ValueError("avg_norm_correction_per_half must be a 2-element list/tuple or None")
    if not (0.0 <= float(scale_relaxation_mu) <= 1.0):
        raise ValueError(f"scale_relaxation_mu must be in [0, 1], got {scale_relaxation_mu}")

    out_norm = []
    out_avg_norm = []
    out_group_scale = []
    out_image_corr = []
    out_scale_corr = []
    out_zero_norm_counts = []

    for half_idx, stats in enumerate(stats_per_half):
        if stats is None:
            raise RuntimeError("RELION norm/scale update expected per-half NoiseStats")

        norm_stats = getattr(stats, "wsum_norm_correction", None)
        if norm_stats is None:
            if image_corr_in[half_idx] is None and scale_corr_in[half_idx] is None and group_ids_in[half_idx] is None:
                raise RuntimeError("Cannot infer image count without norm stats or correction arrays")
            n_images = int(
                len(
                    next(
                        value
                        for value in (image_corr_in[half_idx], scale_corr_in[half_idx], group_ids_in[half_idx])
                        if value is not None
                    )
                )
            )
        else:
            n_images = int(np.asarray(norm_stats).reshape(-1).shape[0])

        group_ids = (
            np.zeros(n_images, dtype=np.int64)
            if group_ids_in[half_idx] is None
            else np.asarray(group_ids_in[half_idx], dtype=np.int64).reshape(-1)
        )
        if group_ids.shape != (n_images,):
            raise ValueError(
                f"group_ids_per_half[{half_idx}] has shape {group_ids.shape}, expected ({n_images},)",
            )
        if group_ids.size and int(np.min(group_ids)) < 0:
            raise ValueError("group IDs must be non-negative")

        scale_per_image_in = (
            None
            if scale_corr_in[half_idx] is None
            else np.asarray(scale_corr_in[half_idx], dtype=np.float64).reshape(-1)
        )
        image_corr = (
            None
            if image_corr_in[half_idx] is None
            else np.asarray(image_corr_in[half_idx], dtype=np.float64).reshape(-1)
        )
        if scale_per_image_in is not None and scale_per_image_in.shape != (n_images,):
            raise ValueError(
                f"scale_corrections_per_half[{half_idx}] has shape {scale_per_image_in.shape}, expected ({n_images},)",
            )
        if image_corr is not None and image_corr.shape != (n_images,):
            raise ValueError(
                f"image_corrections_per_half[{half_idx}] has shape {image_corr.shape}, expected ({n_images},)",
            )

        n_groups_from_ids = int(np.max(group_ids)) + 1 if group_ids.size else 1
        explicit_group_count = 0
        if group_count_in[half_idx] is not None:
            explicit_group_count = int(group_count_in[half_idx])
            if (
                explicit_group_count < 0
                or not np.isfinite(float(group_count_in[half_idx]))
                or float(group_count_in[half_idx]) != float(explicit_group_count)
            ):
                raise ValueError(
                    f"group_count_per_half[{half_idx}] must be a non-negative integer, "
                    f"got {group_count_in[half_idx]!r}"
                )
        required_group_count = max(explicit_group_count, n_groups_from_ids)
        if group_scale_in[half_idx] is None:
            n_groups = required_group_count
            group_scale_old = _derive_group_scale_from_image_scale(scale_per_image_in, group_ids, n_groups)
        else:
            group_scale_old = np.asarray(group_scale_in[half_idx], dtype=np.float64).reshape(-1)
            n_groups = int(group_scale_old.shape[0])
            if n_groups < required_group_count:
                raise ValueError(
                    f"group_scale_corrections_per_half[{half_idx}] has {n_groups} groups, "
                    f"but group IDs / explicit count require {required_group_count}",
                )
        if np.any(group_scale_old <= 0.0):
            raise ValueError("group scale corrections must be positive")

        scale_per_image_old = group_scale_old[group_ids]
        if scale_per_image_in is not None:
            scale_per_image_old = scale_per_image_in
        if image_corr is None:
            image_corr = scale_per_image_old.copy()
        if np.any(image_corr <= 0.0):
            raise ValueError("image corrections must be positive")

        image_norm_factor = image_corr / np.maximum(scale_per_image_old, eps)
        avg_norm_old = float(avg_norm_in[half_idx])
        if avg_norm_old <= 0.0:
            raise ValueError("avg_norm corrections must be positive")

        if do_norm_correction and norm_stats is not None:
            norm_residual = np.asarray(norm_stats, dtype=np.float64).reshape(-1)
            if norm_residual.shape != (n_images,):
                raise ValueError(
                    f"wsum_norm_correction for half {half_idx} has shape {norm_residual.shape}, "
                    f"expected ({n_images},)",
                )
            if not np.all(np.isfinite(norm_residual)):
                raise ValueError("wsum_norm_correction must be finite")
            if np.any(norm_residual < -1e-12):
                raise ValueError("wsum_norm_correction must be non-negative")
            old_norm_over_avg = scale_per_image_old / image_corr
            valid_norm = norm_residual > eps
            zero_norm_count = int(n_images - np.count_nonzero(valid_norm))
            normcorr_from_stats = old_norm_over_avg * np.sqrt(np.maximum(2.0 * norm_residual, 0.0))
            retained_sum_weight = float(getattr(stats, "sumw", 0.0))
            if retained_sum_weight > 0.0:
                target_avg_norm = float(np.sum(normcorr_from_stats[valid_norm]) / retained_sum_weight)
            elif np.any(valid_norm):
                target_avg_norm = float(np.mean(normcorr_from_stats[valid_norm]))
            else:
                target_avg_norm = avg_norm_old
            avg_norm_new = float(scale_relaxation_mu) * avg_norm_old + (1.0 - float(scale_relaxation_mu)) * target_avg_norm
            image_norm_factor_new = image_norm_factor.copy()
            image_norm_factor_new[valid_norm] = avg_norm_new / np.maximum(normcorr_from_stats[valid_norm], eps)
            image_norm_factor = image_norm_factor_new
            normcorr_new = avg_norm_new / np.maximum(image_norm_factor, eps)
        else:
            normcorr_new = avg_norm_old / np.maximum(image_norm_factor, eps)
            avg_norm_new = avg_norm_old
            zero_norm_count = 0

        scale_xa = getattr(stats, "wsum_scale_correction_xa", None)
        scale_aa = getattr(stats, "wsum_scale_correction_aa", None)
        if (
            do_scale_correction
            and not relion_firstiter_cc_this_iter
            and scale_xa is not None
            and scale_aa is not None
        ):
            xa = np.asarray(scale_xa, dtype=np.float64).reshape(-1)
            aa = np.asarray(scale_aa, dtype=np.float64).reshape(-1)
            if xa.shape != (n_groups,) or aa.shape != (n_groups,):
                raise ValueError(
                    f"scale stats for half {half_idx} have shapes {xa.shape}/{aa.shape}, expected ({n_groups},)",
                )
            scale_target = np.ones_like(xa, dtype=np.float64)
            np.divide(xa, aa, out=scale_target, where=aa > 0.0)
            scale_new = float(scale_relaxation_mu) * group_scale_old + (1.0 - float(scale_relaxation_mu)) * scale_target
            sorted_scale = np.sort(scale_new)
            median = float(sorted_scale[n_groups // 2])
            if np.isfinite(median) and median > 0.0:
                scale_new = np.clip(scale_new, median / 5.0, 5.0 * median)
            counts = np.bincount(group_ids, minlength=n_groups).astype(np.float64)
            count_sum = float(np.sum(counts))
            if count_sum > 0.0:
                avg_scale = float(np.sum(counts * scale_new) / count_sum)
                if avg_scale > 0.0 and np.isfinite(avg_scale):
                    scale_new = scale_new / avg_scale
        else:
            scale_new = group_scale_old.copy()

        scale_per_image_new = scale_new[group_ids]
        image_corr_new = image_norm_factor * scale_per_image_new

        out_norm.append(jnp.asarray(normcorr_new, dtype=jnp.float32))
        out_avg_norm.append(avg_norm_new)
        out_group_scale.append(jnp.asarray(scale_new, dtype=jnp.float32))
        out_scale_corr.append(jnp.asarray(scale_per_image_new, dtype=jnp.float32))
        out_image_corr.append(jnp.asarray(image_corr_new, dtype=jnp.float32))
        out_zero_norm_counts.append(zero_norm_count)

    return NormScaleCorrectionUpdateResult(
        norm_corrections_per_half=out_norm,
        avg_norm_correction_per_half=out_avg_norm,
        group_scale_corrections_per_half=out_group_scale,
        image_corrections_per_half=out_image_corr,
        scale_corrections_per_half=out_scale_corr,
        zero_norm_residual_counts=out_zero_norm_counts,
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

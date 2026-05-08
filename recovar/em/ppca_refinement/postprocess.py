"""Post-solve PPCA volume heuristics.

This module intentionally lives outside the augmented M-step. RELION's K=1
reference update applies masking/background-fill/grid-correction heuristics
after the Wiener solve and then uses the processed reference for scoring. For
PPCA this is not yet the final mathematical strategy: the long-term target is
a masked/preconditioned PCG objective like the non-refinement PPCA path. Until
that exists, the default heuristic is explicit and diagnostic-heavy:

* mean: RELION-style soft mask with background fill, optional grid correction;
* W columns: same soft support but zero-filled outside the mask, optional grid
  correction.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp

from recovar.core import fourier_transform_utils as ftu


PPCA_POSTPROCESS_HEURISTIC_WARNING = "heuristic_post_solve_mask_grid_correction_not_masked_pcg_objective"


class PPCAPostprocessResult(NamedTuple):
    mu_half: jax.Array
    W_half: jax.Array
    diagnostics: dict


def _validate_half_volume(vec, volume_shape, name: str):
    half_shape = ftu.volume_shape_to_half_volume_shape(tuple(volume_shape))
    half_size = int(np.prod(half_shape))
    arr = jnp.asarray(vec)
    if arr.shape != (half_size,):
        raise ValueError(f"{name} shape {arr.shape} != ({half_size},)")
    return arr.reshape(half_shape)


def _half_stack_to_real(stack, volume_shape):
    half_shape = ftu.volume_shape_to_half_volume_shape(tuple(volume_shape))
    stack = jnp.asarray(stack)
    return ftu.get_idft3_real(stack.reshape((stack.shape[0],) + half_shape), volume_shape=volume_shape)


def _real_stack_to_half(stack):
    stack = jnp.asarray(stack).real
    return ftu.get_dft3_real(stack).reshape((stack.shape[0], -1))


def _half_volume_frequency_mask(volume_shape, max_r: float, *, dtype):
    coords = ftu.get_k_coordinate_of_each_pixel_3d_real(tuple(volume_shape), voxel_size=1, scaled=False)
    r2 = jnp.sum(jnp.asarray(coords, dtype=jnp.float32) ** 2, axis=-1)
    return (r2 <= float(max_r) ** 2).astype(dtype)


def _half_radial_shell_labels(volume_shape) -> np.ndarray:
    labels = np.asarray(
        ftu.get_grid_of_radial_distances_real(tuple(volume_shape), scaled=False, frequency_shift=0),
        dtype=np.int32,
    ).reshape(-1)
    half_size = int(np.prod(ftu.volume_shape_to_half_volume_shape(tuple(volume_shape))))
    if labels.size != half_size:
        raise AssertionError(f"radial shell labels have {labels.size} entries, expected {half_size}")
    return labels


def _cap_W_shell_power(W_before, W_after, volume_shape):
    """Do not let real-space W masking invent new per-shell covariance power."""

    W_before = jnp.asarray(W_before)
    W_after = jnp.asarray(W_after)
    if W_after.shape[1] == 0:
        return W_after, {
            "postprocess_cap_W_shell_power": True,
            "postprocess_W_shell_power_scale_min": 1.0,
            "postprocess_W_shell_power_scale_mean": 1.0,
        }

    labels_np = _half_radial_shell_labels(volume_shape)
    labels = jnp.asarray(labels_np, dtype=jnp.int32)
    shell_count = int(labels_np.max(initial=0)) + 1
    counts = jnp.bincount(labels, length=shell_count).astype(W_after.real.dtype)
    safe_counts = jnp.where(counts > 0, counts, 1.0)

    before_power = jnp.sum(jnp.abs(W_before) ** 2, axis=1).real.astype(W_after.real.dtype)
    after_power = jnp.sum(jnp.abs(W_after) ** 2, axis=1).real.astype(W_after.real.dtype)
    before_shell = jnp.bincount(labels, weights=before_power, length=shell_count) / safe_counts
    after_shell = jnp.bincount(labels, weights=after_power, length=shell_count) / safe_counts

    eps = jnp.asarray(1.0e-30, dtype=W_after.real.dtype)
    shell_scale = jnp.where(after_shell > before_shell, jnp.sqrt(before_shell / jnp.maximum(after_shell, eps)), 1.0)
    W_capped = W_after * shell_scale[labels, None].astype(W_after.dtype)

    capped_power = jnp.sum(jnp.abs(W_capped) ** 2, axis=1).real.astype(W_after.real.dtype)
    capped_shell = jnp.bincount(labels, weights=capped_power, length=shell_count) / safe_counts
    diagnostics = {
        "postprocess_cap_W_shell_power": True,
        "postprocess_W_shell_power_scale_min": float(jnp.min(shell_scale)) if shell_count else 1.0,
        "postprocess_W_shell_power_scale_mean": float(jnp.mean(shell_scale)) if shell_count else 1.0,
        "postprocess_W_shell_power_input_sum": float(jnp.sum(before_shell * counts)),
        "postprocess_W_shell_power_pre_cap_sum": float(jnp.sum(after_shell * counts)),
        "postprocess_W_shell_power_output_sum": float(jnp.sum(capped_shell * counts)),
    }
    return W_capped, diagnostics


def _soft_mask_and_background_weights(volume_shape, *, mask_radius_px: float | None, cosine_width_px: float, dtype):
    radius = -1 if mask_radius_px is None else float(mask_radius_px)
    if radius < 0:
        radius = np.max(np.array(volume_shape) // 2)
    cosine_width = float(cosine_width_px)
    radius_p = radius + cosine_width

    volume_coords = ftu.get_k_coordinate_of_each_pixel(volume_shape, voxel_size=1, scaled=False).reshape(
        list(volume_shape) + [len(list(volume_shape))]
    )
    r = jnp.linalg.norm(volume_coords, axis=-1)
    mask1 = r < radius
    mask2 = (r >= radius) & (r <= radius_p)
    mask3 = r > radius_p
    raised_cos = 0.5 + 0.5 * jnp.cos(jnp.pi * (radius_p - r) / cosine_width)

    soft_mask = jnp.zeros(volume_shape, dtype=dtype)
    soft_mask = jnp.where(mask1, 1, soft_mask)
    soft_mask = jnp.where(mask2, 1 - raised_cos, soft_mask)
    background_weight = jnp.zeros_like(soft_mask)
    background_weight = jnp.where(mask3, 1, background_weight)
    background_weight = jnp.where(mask2, raised_cos, background_weight)
    return soft_mask.astype(dtype), background_weight.astype(dtype)


def _gridding_kernel(volume_shape, *, gridding_padding_factor: float, gridding_order: int, gridding_correct: str, dtype):
    pixels = ftu.get_k_coordinate_of_each_pixel(volume_shape, 1, scaled=False).astype(jnp.float64)
    if gridding_correct == "radial":
        r = jnp.sqrt(jnp.sum(pixels**2, axis=-1))
        safe_rval = jnp.where(r > 0, r / (int(volume_shape[0]) * float(gridding_padding_factor)), 1.0)
        sinc = jnp.where(r > 0, jnp.sin(jnp.pi * safe_rval) / (jnp.pi * safe_rval), 1.0)
        if int(gridding_order) == 0:
            kernel = sinc
        elif int(gridding_order) == 1:
            kernel = sinc**2
        else:
            raise ValueError("gridding_order must be 0 or 1")
    else:
        pixels_rescaled = pixels / (int(volume_shape[0]) * float(gridding_padding_factor))

        def sinc(ar):
            return jnp.where(jnp.abs(ar) < 1e-8, 1.0, jnp.sin(jnp.pi * ar) / (jnp.pi * ar))

        if int(gridding_order) == 0:
            kernel_fn = sinc
        elif int(gridding_order) == 1:
            kernel_fn = lambda x: sinc(x) ** 2
        else:
            raise ValueError("gridding_order must be 0 or 1")
        kernel = kernel_fn(pixels_rescaled[:, 0]) * kernel_fn(pixels_rescaled[:, 1]) * kernel_fn(
            pixels_rescaled[:, 2]
        )
    return kernel.reshape(volume_shape).astype(dtype)


def postprocess_ppca_half_volumes(
    mu_half,
    W_half,
    volume_shape,
    *,
    strategy: str = "mean_and_w_mask",
    mask_radius_px: float | None = None,
    cosine_width_px: float = 3.0,
    grid_correct: bool = True,
    gridding_padding_factor: float = 1.0,
    gridding_order: int = 1,
    gridding_correct: str = "radial",
    bandlimit_max_r: float | None = None,
    cap_W_shell_power: bool = True,
) -> PPCAPostprocessResult:
    """Apply explicit post-solve PPCA reference heuristics.

    ``strategy`` choices:

    ``none``
        Return raw M-step outputs unchanged.
    ``mean_only``
        Apply RELION-style background-fill mask/grid correction to the mean;
        leave W unchanged.
    ``mean_and_w_mask``
        Default. Apply the mean postprocess above and soft-mask every W column
        with zero background outside the mask.
    """

    strategy = str(strategy)
    if strategy not in {"none", "mean_only", "mean_and_w_mask"}:
        raise ValueError(
            "postprocess strategy must be 'none', 'mean_only', or 'mean_and_w_mask', "
            f"got {strategy!r}"
        )
    if gridding_correct not in {"radial", "square"}:
        raise ValueError(f"gridding_correct must be 'radial' or 'square', got {gridding_correct!r}")

    volume_shape = tuple(int(s) for s in volume_shape)
    mu_half = jnp.asarray(mu_half)
    W_half = jnp.asarray(W_half)
    if W_half.ndim != 2:
        raise ValueError(f"W_half must have shape [n_frequency, q], got {W_half.shape}")
    _validate_half_volume(mu_half, volume_shape, "mu_half")
    if W_half.shape[0] != mu_half.shape[0]:
        raise ValueError(f"W_half frequency dimension {W_half.shape[0]} != mu_half size {mu_half.shape[0]}")

    applies_W_shell_cap = bool(strategy == "mean_and_w_mask" and cap_W_shell_power and W_half.shape[1] > 0)
    diagnostics = {
        "postprocess_strategy": strategy,
        "postprocess_warning": PPCA_POSTPROCESS_HEURISTIC_WARNING,
        "postprocess_mask_radius_px": None if mask_radius_px is None else float(mask_radius_px),
        "postprocess_cosine_width_px": float(cosine_width_px),
        "postprocess_grid_correct": bool(grid_correct),
        "postprocess_gridding_padding_factor": float(gridding_padding_factor),
        "postprocess_gridding_order": int(gridding_order),
        "postprocess_gridding_correct": str(gridding_correct),
        "postprocess_bandlimit_max_r": None if bandlimit_max_r is None else float(bandlimit_max_r),
        "postprocess_cap_W_shell_power": applies_W_shell_cap,
    }
    if strategy == "none":
        return PPCAPostprocessResult(mu_half=mu_half, W_half=W_half, diagnostics=diagnostics)

    soft_mask, background_weight = _soft_mask_and_background_weights(
        volume_shape,
        mask_radius_px=mask_radius_px,
        cosine_width_px=float(cosine_width_px),
        dtype=mu_half.real.dtype,
    )
    diagnostics["postprocess_mask_mean"] = float(jnp.mean(soft_mask))

    if strategy == "mean_only" or W_half.shape[1] == 0:
        half_stack = mu_half[None, :]
    else:
        half_stack = jnp.concatenate([mu_half[None, :], jnp.swapaxes(W_half, 0, 1)], axis=0)
    real_stack = _half_stack_to_real(half_stack, volume_shape)

    bg_weight_sum = jnp.sum(background_weight)
    safe_bg_weight_sum = jnp.where(bg_weight_sum > 0.0, bg_weight_sum, 1.0)
    mean_background = jnp.sum(real_stack[0] * background_weight) / safe_bg_weight_sum
    processed_stack = real_stack.at[0].set(soft_mask * real_stack[0] + background_weight * mean_background)
    if strategy == "mean_and_w_mask" and W_half.shape[1] > 0:
        processed_stack = processed_stack.at[1:].set(soft_mask[None, :, :, :] * real_stack[1:])

    if grid_correct:
        kernel = _gridding_kernel(
            volume_shape,
            gridding_padding_factor=float(gridding_padding_factor),
            gridding_order=int(gridding_order),
            gridding_correct=str(gridding_correct),
            dtype=processed_stack.dtype,
        )
        processed_stack = processed_stack / kernel[None, :, :, :]

    half_out = _real_stack_to_half(processed_stack)
    if bandlimit_max_r is not None:
        bandlimit_mask = _half_volume_frequency_mask(
            volume_shape,
            float(bandlimit_max_r),
            dtype=half_out.real.dtype,
        )
        half_out = half_out * bandlimit_mask[None, :]
        diagnostics["postprocess_bandlimit_fraction"] = float(jnp.mean(bandlimit_mask))
    mu_out = half_out[0].astype(mu_half.dtype)
    W_out = jnp.swapaxes(half_out[1:].astype(W_half.dtype), 0, 1) if strategy == "mean_and_w_mask" else W_half
    if strategy == "mean_and_w_mask" and cap_W_shell_power:
        W_out, cap_diagnostics = _cap_W_shell_power(W_half, W_out, volume_shape)
        diagnostics.update(cap_diagnostics)
    return PPCAPostprocessResult(mu_half=mu_out, W_half=W_out, diagnostics=diagnostics)

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

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from recovar.core import fourier_transform_utils as ftu

PPCA_POSTPROCESS_HEURISTIC_WARNING = "heuristic_post_solve_mask_grid_correction_not_masked_pcg_objective"


@dataclass(frozen=True)
class PostprocessConfig:
    """Bundle of post-M-step heuristic parameters.

    Wraps the 8 kwargs previously sprawling across every EM-iteration entry
    point. ``strategy="none"`` disables postprocess entirely; the other fields
    are only consulted when the strategy enables masking / grid correction.

    ``external_mask_volume`` (optional) replaces the built-in spherical soft
    mask with an arbitrary real-space mask of shape ``volume_shape``. This is
    how the pipeline's dilated solvent mask is plumbed in: pass it through and
    the mean is masked with background fill outside the mask, while W columns
    are zeroed outside (when strategy is ``mean_and_w_mask``). Background-fill
    behavior follows from the mask's binary support; soft transitions can be
    encoded directly in ``external_mask_volume`` if desired.
    """

    strategy: str = "mean_and_w_mask"
    mask_radius_px: float | None = None
    cosine_width_px: float = 3.0
    grid_correct: bool = True
    gridding_padding_factor: float = 1.0
    gridding_order: int = 1
    gridding_correct: str = "radial"
    bandlimit_max_r: float | None = None
    external_mask_volume: np.ndarray | None = None


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


def _gridding_kernel(
    volume_shape, *, gridding_padding_factor: float, gridding_order: int, gridding_correct: str, dtype
):
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
        kernel = kernel_fn(pixels_rescaled[:, 0]) * kernel_fn(pixels_rescaled[:, 1]) * kernel_fn(pixels_rescaled[:, 2])
    return kernel.reshape(volume_shape).astype(dtype)


def postprocess_ppca_half_volumes(
    mu_half,
    W_half,
    volume_shape,
    *,
    config: PostprocessConfig | None = None,
) -> PPCAPostprocessResult:
    cfg = config if config is not None else PostprocessConfig()
    strategy = cfg.strategy
    mask_radius_px = cfg.mask_radius_px
    cosine_width_px = cfg.cosine_width_px
    grid_correct = cfg.grid_correct
    gridding_padding_factor = cfg.gridding_padding_factor
    gridding_order = cfg.gridding_order
    gridding_correct = cfg.gridding_correct
    bandlimit_max_r = cfg.bandlimit_max_r
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
    ``w_only_mask``
        Soft-mask every W column with zero background outside the mask; leave
        the mean unchanged. Use this when the mean should pass through the
        solver/postprocess without any mask or background-fill (e.g., to
        preserve image-likelihood-fitted DC and solvent contributions in mu)
        but W must be confined to the structure region.
    """

    strategy = str(strategy)
    if strategy not in {"none", "mean_only", "mean_and_w_mask", "w_only_mask"}:
        raise ValueError(
            "postprocess strategy must be 'none', 'mean_only', 'mean_and_w_mask', or 'w_only_mask', "
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
    }
    if strategy == "none":
        return PPCAPostprocessResult(mu_half=mu_half, W_half=W_half, diagnostics=diagnostics)

    external_mask = cfg.external_mask_volume
    if external_mask is not None:
        ext = jnp.asarray(external_mask, dtype=mu_half.real.dtype)
        if tuple(ext.shape) != volume_shape:
            raise ValueError(
                f"external_mask_volume shape {ext.shape} != volume_shape {volume_shape}"
            )
        # Binary or pre-softened mask; bg = 1 - mask, so the mean's background
        # fill blends from inside (mask=1) to outside (mask=0) seamlessly with
        # any soft transition encoded in the external mask itself.
        soft_mask = ext
        background_weight = (1.0 - ext).astype(mu_half.real.dtype)
        diagnostics["postprocess_mask_source"] = "external_mask_volume"
    else:
        soft_mask, background_weight = _soft_mask_and_background_weights(
            volume_shape,
            mask_radius_px=mask_radius_px,
            cosine_width_px=float(cosine_width_px),
            dtype=mu_half.real.dtype,
        )
        diagnostics["postprocess_mask_source"] = "soft_radius"
    diagnostics["postprocess_mask_mean"] = float(jnp.mean(soft_mask))

    # Decide which volumes to put through the masking/grid-correction stack.
    # The stack order is [mu, W_0, ..., W_{q-1}]; rows we don't touch are
    # simply not included so they bypass mask + grid correction entirely.
    apply_to_mu = strategy in {"mean_only", "mean_and_w_mask"}
    apply_to_W = strategy in {"mean_and_w_mask", "w_only_mask"} and W_half.shape[1] > 0
    if not apply_to_mu and not apply_to_W:
        return PPCAPostprocessResult(mu_half=mu_half, W_half=W_half, diagnostics=diagnostics)

    stack_parts = []
    if apply_to_mu:
        stack_parts.append(mu_half[None, :])
    if apply_to_W:
        stack_parts.append(jnp.swapaxes(W_half, 0, 1))
    half_stack = jnp.concatenate(stack_parts, axis=0)
    real_stack = _half_stack_to_real(half_stack, volume_shape)

    cursor = 0
    if apply_to_mu:
        bg_weight_sum = jnp.sum(background_weight)
        safe_bg_weight_sum = jnp.where(bg_weight_sum > 0.0, bg_weight_sum, 1.0)
        mean_background = jnp.sum(real_stack[cursor] * background_weight) / safe_bg_weight_sum
        real_stack = real_stack.at[cursor].set(
            soft_mask * real_stack[cursor] + background_weight * mean_background
        )
        cursor += 1
    if apply_to_W:
        n_W = int(W_half.shape[1])
        real_stack = real_stack.at[cursor : cursor + n_W].set(
            soft_mask[None, :, :, :] * real_stack[cursor : cursor + n_W]
        )

    if grid_correct:
        kernel = _gridding_kernel(
            volume_shape,
            gridding_padding_factor=float(gridding_padding_factor),
            gridding_order=int(gridding_order),
            gridding_correct=str(gridding_correct),
            dtype=real_stack.dtype,
        )
        real_stack = real_stack / kernel[None, :, :, :]

    half_out = _real_stack_to_half(real_stack)
    if bandlimit_max_r is not None:
        bandlimit_mask = _half_volume_frequency_mask(
            volume_shape,
            float(bandlimit_max_r),
            dtype=half_out.real.dtype,
        )
        half_out = half_out * bandlimit_mask[None, :]
        diagnostics["postprocess_bandlimit_fraction"] = float(jnp.mean(bandlimit_mask))

    cursor = 0
    if apply_to_mu:
        mu_out = half_out[cursor].astype(mu_half.dtype)
        cursor += 1
    else:
        mu_out = mu_half
    if apply_to_W:
        n_W = int(W_half.shape[1])
        W_out = jnp.swapaxes(half_out[cursor : cursor + n_W].astype(W_half.dtype), 0, 1)
    else:
        W_out = W_half
    return PPCAPostprocessResult(mu_half=mu_out, W_half=W_out, diagnostics=diagnostics)

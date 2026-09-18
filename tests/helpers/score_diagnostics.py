"""Test-only score diagnostics, retained without changing their arithmetic.

The float64 JAX lane replay is diagnostic, not a production precision policy.
Repeat/scale summaries cannot qualify numerical noise or scientific acceptance
by themselves. Runtime scoring does not import this module.
"""

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def _relion_coarse_normalized_cc_rescore_f64(
    shifted_candidates,
    score_weight_candidates,
    projection_candidates,
    half_weights,
    fftw_order,
):
    """Rescore coarse candidates with RELION double-ACC lane arithmetic."""

    shifted = jnp.asarray(shifted_candidates, dtype=jnp.complex128)[..., fftw_order]
    score_weight = jnp.asarray(score_weight_candidates, dtype=jnp.float64)[..., fftw_order]
    projection = jnp.asarray(projection_candidates, dtype=jnp.complex128)[..., fftw_order]
    weights = jnp.asarray(half_weights, dtype=jnp.float64)[fftw_order]
    numerator_pixels = jnp.real(jnp.conj(shifted) * projection) * weights
    norm_pixels = score_weight * (jnp.abs(projection) ** 2) * weights

    def reduce_lanes(values):
        n_pixels = int(values.shape[-1])
        n_passes = (n_pixels + 127) // 128
        padded = jnp.pad(
            values,
            [(0, 0)] * (values.ndim - 1) + [(0, n_passes * 128 - n_pixels)],
        )
        passes = padded.reshape(values.shape[:-1] + (n_passes, 128))
        lanes = jnp.zeros(values.shape[:-1] + (128,), dtype=jnp.float64)
        for pass_index in range(n_passes):
            lanes = lanes + passes[..., pass_index, :]
        for stride in (64, 32, 16, 8, 4, 2, 1):
            lanes = lanes[..., :stride] + lanes[..., stride : 2 * stride]
        return lanes[..., 0]

    numerator = reduce_lanes(numerator_pixels)
    norm = reduce_lanes(norm_pixels)
    contribution = numerator / (
        jnp.asarray(128.0, dtype=jnp.float64)
        * jnp.sqrt(jnp.maximum(norm, jnp.asarray(1e-30, dtype=jnp.float64)))
    )

    def add_once(_, accumulated):
        return accumulated + contribution

    return jax.lax.fori_loop(0, 128, add_once, jnp.zeros_like(contribution))


def _coarse_gaussian_repeat_spread_diagnostics(score_delta_repeats):
    """Report repeat-to-repeat drift without defining an acceptance epsilon.

    ``score_delta_repeats`` has layout
    ``[repeat,image,class,rotation,translation]`` and should be assembled from
    immutable same-hardware paired captures.  The caller must compare this raw
    spread with native/direct repeats, scale panels, discrete outcomes, and
    final quality; this helper cannot promote the GEMM path by itself.
    """

    deltas = np.asarray(score_delta_repeats, dtype=np.float64)
    if deltas.ndim != 5 or deltas.shape[0] < 2:
        raise ValueError(
            "coarse GEMM repeat diagnostics require at least two "
            "[repeat,image,class,rotation,translation] score-delta surfaces",
        )
    flattened = deltas.reshape(deltas.shape[0], deltas.shape[1], -1)
    signed_mean_per_repeat_image = np.mean(flattened, axis=2)
    max_abs_per_repeat_image = np.max(np.abs(flattened), axis=2)
    return {
        "repeat_count": np.asarray(deltas.shape[0], dtype=np.int64),
        "signed_mean_delta_per_repeat_image": signed_mean_per_repeat_image,
        "max_abs_delta_per_repeat_image": max_abs_per_repeat_image,
        "signed_mean_repeat_spread_per_image": np.ptp(
            signed_mean_per_repeat_image,
            axis=0,
        ),
        "max_abs_repeat_spread_per_image": np.ptp(
            max_abs_per_repeat_image,
            axis=0,
        ),
        "elementwise_delta_repeat_spread": np.ptp(deltas, axis=0),
        "qualification_status": np.asarray(
            "NO_GO_raw_repeat_spread_requires_native_scale_discrete_quality_runtime_context"
        ),
    }


def _coarse_gaussian_scale_panel_diagnostics(
    operand_scales,
    score_deltas_by_scale,
    *,
    precision_bits: int,
):
    """Classify strictly growing multiscale cancellation drift as NO-GO.

    This is an ordering test over raw observations, not an epsilon threshold.
    A non-growing panel remains unqualified until same-hardware repeats,
    discrete outcomes, final quality, and clean runtime all pass.
    """

    scales = np.asarray(operand_scales, dtype=np.float64).reshape(-1)
    deltas = np.asarray(score_deltas_by_scale, dtype=np.float64)
    if scales.size < 2 or deltas.ndim != 5 or deltas.shape[0] != scales.size:
        raise ValueError(
            "coarse GEMM scale diagnostics require matching scale and "
            "[scale,image,class,rotation,translation] arrays with at least two scales",
        )
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0) or np.any(np.diff(scales) <= 0.0):
        raise ValueError("coarse GEMM operand scales must be finite, positive, and increasing")
    if int(precision_bits) not in (32, 64):
        raise ValueError("coarse GEMM precision_bits must be 32 or 64")
    flattened = deltas.reshape(scales.size, -1)
    max_abs_by_scale = np.max(np.abs(flattened), axis=1)
    signed_mean_by_scale = np.mean(flattened, axis=1)
    growing_steps = max_abs_by_scale[1:] > max_abs_by_scale[:-1]
    scale_amplified = bool(np.any(growing_steps))
    return {
        "operand_scales": scales,
        "precision_bits": np.asarray(precision_bits, dtype=np.int64),
        "signed_mean_delta_by_scale": signed_mean_by_scale,
        "max_abs_delta_by_scale": max_abs_by_scale,
        "scale_growth_steps": growing_steps,
        "scale_amplified": np.asarray(scale_amplified),
        "qualification_status": np.asarray(
            (
                "NO_GO_scale-amplified_drift"
                if scale_amplified
                else "NO_GO_unqualified_non-growing_scale_panel"
            )
        ),
    }

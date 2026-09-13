"""Per-half noise initialization, sufficient statistics and posterior updates.

Keep shell profiles, expanded pixel variances and their aliasing explicit.
The controller and replay diagnostics import this owner directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass as _dataclass

import jax.numpy as jnp
import numpy as np

from recovar.em.helpers.types import make_noise_stats

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



def _noise_radial_history(noise_variance_per_half, image_shape, *, dtype):
    """Build fresh per-half shell profiles and their mean in the requested dtype.

    Pixel-array normalization and noise estimation remain with the caller.
    Preserve the float64 host shell reduction before the final JAX cast.
    """
    from recovar.em.relion.relion_metadata import _radial_profile_from_noise_variance

    per_half = [_radial_profile_from_noise_variance(noise_k, image_shape) for noise_k in noise_variance_per_half]
    mean = jnp.asarray(np.mean(np.stack(per_half, axis=0), axis=0), dtype=dtype)
    return per_half, mean



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



@_dataclass
class NoiseUpdateResult:
    """Noise-update values in both shell and image-pixel layouts.

    ``noise_from_res`` is the mean per-shell sigma2_noise profile;
    ``noise_from_res_per_half`` contains its two per-half profiles.
    These are NumPy float64 arrays with ``n_shells`` entries per profile.

    ``noise_variance_per_half`` contains flattened image-pixel arrays obtained
    by expanding the radial noise model. ``noise_variance`` is their elementwise
    mean in that same pixel layout, not a shell profile. The returned
    ``previous_noise_radial`` and ``previous_noise_radial_per_half`` carry shell
    profiles for the next update and its diagnostics.

    Ownership is path-dependent. Ordinary K1 updates replace entries in the
    supplied pixel-array list. K-class updates return a new two-entry list whose
    entries refer to the same expanded shared-noise array. The first-iteration
    CC path preserves the input pixel list and previous radial-history objects.
    Otherwise the returned per-half radial history is the same list as
    ``noise_from_res_per_half``. Do not assume these outputs are independent
    copies or change their aliasing during structural cleanup.
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

    new_previous_noise_radial = jnp.asarray(noise_from_res)
    noise_variance = _mean_noise_variance(noise_variance_per_half)
    return NoiseUpdateResult(
        noise_from_res=noise_from_res,
        noise_from_res_per_half=noise_from_res_per_half,
        noise_variance_per_half=noise_variance_per_half,
        noise_variance=noise_variance,
        previous_noise_radial=new_previous_noise_radial,
        previous_noise_radial_per_half=noise_from_res_per_half,
    )

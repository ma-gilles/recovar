"""cryoSPARC-style upper bound on per-image pose score for BnB pruning.

The cryoSPARC supplement (Punjani et al. 2017, Suppl Note 2) derives a
probabilistic *lower* bound on the alignment error E(r,t) over all poses
(Eq 22), based on:
  * exact low-frequency squared error A_L(r,t),
  * total high-frequency image power B_1 (pose-independent, drops out for
    pose pruning),
  * the maximum CTF-modulated high-frequency model-slice power
    P^max_{i,H}(L) := max_r 1/2 sum_{|l|>L} h_l (C_il^2/sigma_il^2) |Y_l(r)|^2,
  * a 4-sigma noise correction tau * sqrt(P^max_{i,H}(L)).

In RECOVAR's score convention (s = -E + image-only constant), this becomes
an *upper* bound on the per-image, per-pose score:

    U_iL(r,t) = s_iL(r,t) + Delta_iH(L)
    Delta_iH(L) = P^max_{i,H}(L) + tau * sqrt(P^max_{i,H}(L))

The image-only term sum_{|l|>L} 1/(2 sigma_l^2) |X_l|^2 cancels in the
per-image softmax over poses and is omitted.

This module computes Delta_iH for two bound modes:

* ``cryosparc_score_upper_correction`` — the supplement's probabilistic 4
  sigma form (default; matches the published cryoSPARC algorithm).
* ``cauchy_score_upper_correction`` — a deterministic Cauchy-Schwarz upper
  bound on the high-frequency score increment, useful for unit tests and
  no-false-pruning debug runs.

Both functions take ``P^max_{i,H}`` as an input; ``P^max_{i,H}`` itself is
computed by ``compute_high_model_pmax_per_image`` which delegates to
``helpers.projection.compute_projections_block`` (no fork of the projector).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers.projection import (
    compute_projections_block,
)


def compute_high_model_pmax_per_image(
    mean: jnp.ndarray,
    rotations_for_bound: np.ndarray,
    ctf2_over_nv_half: jnp.ndarray,
    half_weights: jnp.ndarray,
    high_indices: jnp.ndarray,
    *,
    image_shape: tuple[int, int],
    volume_shape: tuple[int, int, int],
    disc_type: str,
    rotation_block_size: int = 5000,
) -> jnp.ndarray:
    """Per-image P^max_{i,H} = 1/2 max_r sum_{l in H} h_l (C^2/sigma^2) |Y_l(r)|^2.

    Suppl Eq 19/22 generalized to colored noise. The maximum is over the
    supplied rotation cover; for paper-faithful operation pass the current
    BnB stage rotation grid. ``Y_l(r)`` is the projection of the current
    structure ``mean`` at rotation ``r``.

    Inputs are aligned with RECOVAR's existing scoring kernel:
        - ``mean``: (volume_size,) centered Fourier volume.
        - ``ctf2_over_nv_half``: (n_images, n_half), per-pixel C^2/sigma^2,
          *without* the Hermitian half-spectrum weight ``h_l`` (which is
          applied separately).
        - ``half_weights``: (n_half,), the Hermitian weights from
          ``helpers.half_spectrum.make_half_image_weights``.
        - ``high_indices``: (n_high,) packed-half indices into the high band.

    Returns
    -------
    pmax : jnp.ndarray, shape (n_images,), float
        ``P^max_{i,H}`` per image.
    """
    rotations_for_bound = np.asarray(rotations_for_bound, dtype=np.float32)
    n_rot = int(rotations_for_bound.shape[0])
    if n_rot == 0:
        n_images = int(ctf2_over_nv_half.shape[0])
        return jnp.zeros((n_images,), dtype=ctf2_over_nv_half.dtype)

    high_indices = jnp.asarray(high_indices, dtype=jnp.int32)
    weights_high = jnp.asarray(half_weights)[high_indices]
    ctf2_high = jnp.asarray(ctf2_over_nv_half)[:, high_indices]
    weighted_ctf2 = ctf2_high * weights_high[None, :]

    n_images = int(ctf2_high.shape[0])
    pmax = jnp.full((n_images,), 0.0, dtype=ctf2_high.dtype)

    block = max(1, int(rotation_block_size))
    for r0 in range(0, n_rot, block):
        r1 = min(r0 + block, n_rot)
        rot_block = rotations_for_bound[r0:r1]
        _, proj_abs2_half = compute_projections_block(
            mean,
            rot_block,
            image_shape,
            volume_shape,
            disc_type,
            return_abs2=True,
        )
        proj_abs2_high = proj_abs2_half[:, high_indices]
        # power[i, r] = sum_l weighted_ctf2[i,l] * proj_abs2_high[r,l]
        power = jnp.matmul(weighted_ctf2, proj_abs2_high.T)
        block_max = jnp.max(power, axis=1)
        pmax = jnp.maximum(pmax, block_max)

    return 0.5 * pmax


def cryosparc_score_upper_correction(
    pmax_per_image: jnp.ndarray,
    *,
    tau_sigma: float = 4.0,
) -> jnp.ndarray:
    """Suppl Eq 22 score-space correction Delta_iH = P^max + tau * sqrt(P^max).

    Default ``tau_sigma=4.0`` matches the cryoSPARC paper's 0.999936
    probability. The image-only term cancels in the per-image softmax and is
    excluded.

    Combined with low-frequency score s_iL(r,t):
        s_i(r,t) <= s_iL(r,t) + Delta_iH(L)
    """
    pmax_safe = jnp.maximum(pmax_per_image, 0.0)
    return pmax_safe + float(tau_sigma) * jnp.sqrt(pmax_safe)


def cauchy_score_upper_correction(
    image_high_power_per_image: jnp.ndarray,
    pmax_per_image: jnp.ndarray,
) -> jnp.ndarray:
    """Deterministic Cauchy upper bound on the high-frequency score increment.

    Derivation: in score space, s_H(q) = -E_H(q) + image_high_power, where
        E_H(q) = sum_{l>L} 1/(2 sigma^2) |C Y - S X|^2.

    Expanding and bounding with Cauchy-Schwarz on the cross term gives
        s_H(q) <= -1/2 |b|^2 + |a| * |b|,
    where
        |a|^2 = sum h_l |X_l|^2 / sigma_l^2 = 2 * image_high_power,
        |b|^2 = sum h_l C_l^2 |Y_l(r)|^2 / sigma_l^2 <= 2 * P^max_{i,H}.

    Maximize over |b| with the constraint |b|^2 <= 2 P^max:
        s_H(q) <= image_high_power                  if image_high_power <= P^max
        s_H(q) <= -P^max + 2 sqrt(P^max * image_high_power)
                                                    if image_high_power >  P^max

    This bound is *deterministic* — no probabilistic 4-sigma assumption — so
    it can never be violated for any pose. Looser than the cryoSPARC bound
    but useful for no-false-pruning unit tests.
    """
    img_safe = jnp.maximum(image_high_power_per_image, 0.0)
    pmax_safe = jnp.maximum(pmax_per_image, 0.0)
    case_low = img_safe
    case_high = -pmax_safe + 2.0 * jnp.sqrt(pmax_safe * img_safe)
    return jnp.where(img_safe <= pmax_safe, case_low, case_high)


def compute_image_high_power_per_image(
    image_power_over_sigma2_high: jnp.ndarray,
    half_weights_high: jnp.ndarray,
) -> jnp.ndarray:
    """Per-image high-frequency image power 1/2 sum_l h_l |X_il|^2 / sigma_il^2.

    This is the |a|^2 / 2 quantity in the deterministic Cauchy bound. The
    cryoSPARC probabilistic bound (``cryosparc_score_upper_correction``)
    does NOT depend on this — only the Cauchy variant does. We expose this
    function so callers can compute the per-image |a|^2 once per stage.

    Parameters
    ----------
    image_power_over_sigma2_high : (n_images, n_high) real
        Per-pixel ``|X_il|^2 / sigma_il^2`` on the high-frequency packed-half
        slice. CTF^2 is NOT included; the cryoSPARC bound uses image power
        ``1/(2 sigma^2) |X|^2``, not ``1/(2 sigma^2) C^2 |X|^2``.
    half_weights_high : (n_high,) real
        Hermitian half-spectrum weights restricted to the high band.

    Returns
    -------
    image_high_power_per_image : (n_images,) real
    """
    weighted = image_power_over_sigma2_high * half_weights_high[None, :]
    return 0.5 * jnp.sum(weighted, axis=1)

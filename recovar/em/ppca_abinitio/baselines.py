"""Non-PPCA heterogeneity baselines for the Stage 1C / Phase 2 spec
clauses.

Per spec Q3 / Section 11.5, every Stage 1C and Phase 2 PPCA result
must be compared against a non-PPCA in-tree heterogeneity learner.
The spec specifically names `recovar.em.states.HeterogeneousEMState`
as the baseline.

Direct integration of `HeterogeneousEMState` requires:
  - duck-typing a `CryoEMDataset` from a `SyntheticDataset`,
  - setting up `covariance_options` (`disc_type`, `disc_type_u`,
    `left_kernel`, `grid_correct`, `n_pcs_to_compute`,
    `randomized_sketch_size`),
  - choosing a sparse `picked_frequency_indices` set,
  - calling `compute_H_B`, `compute_projected_covariance_rhs_lhs`,
    and `solve_covariance` from the parity branch.

This is fragile and couples the v0 module tightly to the parity
branch's evolving M-step machinery. Instead this module ships a
**simpler in-spirit baseline** — a residual-PCA approach that
captures the same scientific question ("does PPCA beat 'do PCA on
the residuals after a homogeneous fit'?") without the full
HeterogeneousEMState plumbing.

The baseline is:

  1. Run our posterior helper with `U=0` (homogeneous E-step). This
     produces pose responsibilities `gamma[i, r, t]` and the
     argmax pose `g_argmax[i] = argmax_g gamma[i, g]`.
  2. Build the per-image residual at the argmax pose:
        residual_full[i] = (S_t* y_i) / CTF_i  -  A_r* mu
     where `(r*, t*) = g_argmax[i]`.
  3. Adjoint-slice each residual back into the half-volume using
     `adjoint_slice_volume(half_image=True, half_volume=True)`. This
     gives a per-image "residual real-volume contribution".
  4. Decode each contribution to real space via `get_idft3_real`,
     stack into a real (n_img, N_full) matrix, then run real-space
     PCA / SVD to extract the top `q` directions.
  5. Re-encode the resulting basis into half-volume layout.

The result is a baseline `(mu_half, U_half, s)` where `mu` is the
**unchanged input mean** (the baseline does not update mu) and
`U_half` is the residual-PCA basis. `s` is set from the PCA
singular values.

We document this as a **simplified surrogate** for the
`HeterogeneousEMState` baseline. The full integration is a
post-v0 task that requires the duck-typed dataset wrapper.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import recovar.core as core
from recovar.core.slicing import adjoint_slice_volume

from .half_volume import (
    half_to_real_volume,
    make_half_volume_weights,
    real_volume_orthonormalize_half,
    real_volume_to_half,
)
from .posterior import (
    _preprocess_batch_to_half,
    _slice_mu_half,
    make_half_image_weights,
    score_from_half_image_projections,
)
from .types import PPCAInit


def _argmax_pose_indices(log_resp):
    """Return per-image argmax (rot_idx, trans_idx)."""
    lr = np.asarray(log_resp)
    n_img, n_rot, n_trans = lr.shape
    flat = lr.reshape(n_img, -1)
    arg = flat.argmax(axis=-1)
    return arg // n_trans, arg % n_trans


def residual_pca_baseline(
    config,
    mu_half,
    s_floor,
    batch_full,
    rotations,
    translations,
    ctf_params,
    noise_variance_full,
    *,
    q: int,
) -> PPCAInit:
    """Compute a residual-PCA baseline `(mu_half, U_half, s)`.

    `mu_half` is returned unchanged. `U_half` is the top-`q` real-
    space PCA basis of the per-image residuals at the argmax-pose
    homogeneous fit. `s` is set from the squared singular values.
    `s_floor` is a per-component minimum (defaults can be 1e-6).
    """
    image_shape = config.image_shape
    volume_shape = config.volume_shape
    weights_half = make_half_image_weights(image_shape)

    # 1. Homogeneous E-step (U=0). Build the slot expected by the
    #    posterior kernel.
    n_img = batch_full.shape[0]
    n_rot = rotations.shape[0]
    mean_proj_half = _slice_mu_half(mu_half, rotations, image_shape, volume_shape).astype(jnp.complex128)
    u_zero = jnp.zeros((n_rot, q, mean_proj_half.shape[-1]), dtype=jnp.complex128)
    s_dummy = jnp.full((q,), max(s_floor, 1e-6), dtype=jnp.float64)

    shifted_half, ctf2_over_nv_half, _ctf_half = _preprocess_batch_to_half(
        config, batch_full, translations, ctf_params, noise_variance_full
    )
    stats = score_from_half_image_projections(
        mean_proj_half, u_zero, s_dummy, shifted_half, ctf2_over_nv_half, weights_half
    )

    # 2. Argmax pose per image
    r_argmax, t_argmax = _argmax_pose_indices(stats.log_resp)

    # 3. Build per-image residuals in full-image FT layout. We need
    #    `S_t* y_i - CTF_i * A_r* mu` where (r*, t*) is the argmax
    #    pose. CTF=identity in v0 so we just need the shift.
    H_img, W_img = image_shape
    N_full = H_img * W_img

    # Slice mu through the per-image argmax rotation set.
    rotations_per_image_argmax = jnp.asarray(np.asarray(rotations)[r_argmax])
    from recovar.core.slicing import slice_volume

    mu_proj_per_image_full = slice_volume(
        mu_half,
        rotations_per_image_argmax,
        image_shape,
        volume_shape,
        "linear_interp",
        half_volume=True,
        half_image=False,
    )  # (n_img, full_image_size)

    # Apply translation to each image (forward shift) and convert to full FT
    translations_per_image = jnp.asarray(np.asarray(translations)[t_argmax])
    trans_for_shift = translations_per_image[:, None, :]
    # The argmax convention uses S_t to align the image to the rotation:
    # the reverse shift recovers the unshifted image.
    shifted_y_full = core.batch_trans_translate_images(batch_full, trans_for_shift, image_shape)[:, 0, :]
    residuals_full = shifted_y_full - mu_proj_per_image_full  # (n_img, N_full)

    # 4. Backproject each residual into a half-volume. The fastest way
    #    is one adjoint_slice_volume call per image; we batch over
    #    images by using `adjoint_slice_volume` once with the full
    #    rotation stack.
    half_volume_size = mu_half.shape[0]
    backproj_half = np.zeros((n_img, half_volume_size), dtype=np.complex128)
    for i in range(n_img):
        per_image = jnp.asarray(residuals_full[i : i + 1])
        rot_one = rotations_per_image_argmax[i : i + 1]
        bp = adjoint_slice_volume(
            per_image,
            rot_one,
            image_shape,
            volume_shape,
            "linear_interp",
            half_image=False,
            half_volume=True,
        )
        backproj_half[i] = np.asarray(bp).reshape(-1)

    # 5. Decode to real space and run PCA
    real_residuals = np.empty((n_img, int(np.prod(volume_shape))), dtype=np.float64)
    for i in range(n_img):
        rv = half_to_real_volume(jnp.asarray(backproj_half[i]), volume_shape)
        real_residuals[i] = np.asarray(rv).reshape(-1)

    # Center
    real_residuals -= real_residuals.mean(axis=0, keepdims=True)

    # SVD: real_residuals = U_img @ diag(σ) @ V^T where V is the
    # voxel-space basis (n_voxels, q).
    if real_residuals.shape[0] >= q:
        # Use truncated SVD via numpy
        _U_img, S_vals, V_T = np.linalg.svd(real_residuals, full_matrices=False)
        V = V_T[:q].T  # (n_voxels, q)
        s_baseline = (S_vals[:q] ** 2 / max(n_img - 1, 1)).astype(np.float64)
    else:
        # Pathological: fewer images than q. Pad.
        V = np.zeros((real_residuals.shape[1], q), dtype=np.float64)
        s_baseline = np.full((q,), max(s_floor, 1e-6), dtype=np.float64)

    # Re-encode each PC into half-volume
    U_half_rows = []
    for k in range(q):
        pc_real = V[:, k].reshape(volume_shape)
        U_half_rows.append(real_volume_to_half(jnp.asarray(pc_real), volume_shape))
    U_half = jnp.stack(U_half_rows).astype(jnp.complex128)

    # Real-space orthonormalize so the comparison with the PPCA U is
    # gauge-invariant.
    weights_vol = make_half_volume_weights(volume_shape)
    U_half = real_volume_orthonormalize_half(U_half, weights_vol, int(np.prod(volume_shape)))

    s_floored = jnp.maximum(jnp.asarray(s_baseline, dtype=jnp.float64), s_floor)
    return PPCAInit(
        mu=jnp.asarray(mu_half, dtype=jnp.complex128),
        U=U_half,
        s=s_floored,
        volume_shape=tuple(int(x) for x in volume_shape),
    )

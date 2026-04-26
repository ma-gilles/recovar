"""Mean update for the PPCA-ab-initio v0 loop.

Per spec Section 8.2:

- **v0 (debugging ablation)** — homogeneous backprojection of `y_i`
  weighted by PPCA-shaped responsibilities. Not a Stage 1B graduation
  gate.
- **v1 (graduation gate)** — *residualized* backprojection of
  `y_i^res(g) = y_i - CTF_i · A_g · U · m_{i,g}` weighted by the
  same responsibilities.

Both share the same Wiener-filter solve at the end:

    mu_next[k] = Ft_y[k] / (Ft_ctf[k] + tau)

For v0 we accumulate everything in **half-volume** layout via
`adjoint_slice_volume(half_image=True, half_volume=True)`. Half-image
inner products require the rfft Hermitian weighting; the
backprojection itself does not (it is the adjoint of the slice
operator, which is geometrically correct in either layout).

The accumulator math
--------------------

The full sum is

    Ft_y[k_voxel]   = sum_i sum_r sum_t γ_{i,r,t} · A_r^* (CTF_i · S_t y_i)
    Ft_ctf[k_voxel] = sum_i sum_r sum_t γ_{i,r,t} · A_r^* (CTF_i^2)

where `A_r^*` is `adjoint_slice_volume(..., R_r)`. We factor the inner
sums by collapsing `(image, translation)` into a per-rotation image
before the backprojection, then call `adjoint_slice_volume` once on
the full rotation stack:

    per_r_image_y[r, k_pix] = sum_i CTF_i[k_pix] · sum_t γ_{i,r,t} · S_t y_i[k_pix]
    per_r_image_ctf[r, k_pix] = sum_i CTF_i^2[k_pix] · sum_t γ_{i,r,t}

For the residualized variant, we additionally subtract a bias term
that captures the latent contribution:

    bias[k_voxel] = sum_r A_r^*( sum_k u_proj_half[r, k] *
                                  sum_i CTF_i^2 *
                                  sum_t γ_{i,r,t} · m_{i,r,t,k} )

so that `Ft_y_res = Ft_y - bias`.

The half-image phase shifts `S_t y_i` are applied in the **full image**
(per spec Section 4.6 / engine_v2 lines 196-214) and then converted to
half-image. CTF and noise weighting are applied in half-image.

For v0 the accumulator is a Python loop over the (rotation, translation)
grid. Real workloads will JIT-compile a single-block kernel; v0
prioritizes correctness over speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

import recovar.core as core
import recovar.core.fourier_transform_utils as ftu
from recovar.core.slicing import adjoint_slice_volume

from .posterior import (
    _preprocess_batch_to_half,
    _slice_mu_half,
    _slice_U_half,
    make_half_image_weights,
    score_from_half_image_projections,
)
from .types import PPCAInit

# ---------------------------------------------------------------------------
# Per-rotation image construction
# ---------------------------------------------------------------------------


def _per_rotation_residual_image(
    config,
    batch_full,
    translations,
    ctf_params,
    noise_variance_full,
    log_resp,
    *,
    residual_subtraction_half=None,
):
    """Build, for each rotation r, the half-image we will backproject:

        per_r[r, k_pix] = sum_i CTF_i[k_pix] · sum_t γ_{i,r,t} · S_t y_i[k_pix]
                          - residual_subtraction_half[r, k_pix]    (if provided)

    `residual_subtraction_half` is the half-image of `CTF_i² · A_r U m`
    summed over `(i, t)` for the residualized variant. It is computed
    by `_per_rotation_bias_image` and subtracted here.

    Returns
    -------
    per_r : (n_rot, half_image_size) complex128
    """
    n_img = batch_full.shape[0]
    n_rot = log_resp.shape[1]
    n_trans = log_resp.shape[2]
    image_shape = config.image_shape

    CTF_full = config.compute_ctf(ctf_params, half_image=False)  # (n_img, full)
    processed = config.process_fn(batch_full, apply_image_mask=False)
    ctf_weighted_full = processed * CTF_full / noise_variance_full  # (n_img, full)
    # NOTE: we do NOT divide by noise_variance here because the
    # accumulator math wants `CTF · y`, not `CTF · y / σ²`. The
    # `1/σ²` cancels with `Ft_ctf` (which would also need it). For
    # the basic Wiener filter `mu = Ft_y / (Ft_ctf + tau)` to make
    # sense per-voxel, both Ft_y and Ft_ctf must use consistent
    # units. We use σ²-weighted units throughout (matches the
    # production M_with_precompute path).

    # Apply phase shifts in full image
    shifted = core.batch_trans_translate_images(
        ctf_weighted_full,
        jnp.repeat(translations[None], n_img, axis=0),
        image_shape,
    )  # (n_img, n_trans, full)
    shifted_half = ftu.full_image_to_half_image(shifted.reshape(n_img * n_trans, -1), image_shape).reshape(
        n_img, n_trans, -1
    )

    gamma = jnp.exp(log_resp)  # (n_img, n_rot, n_trans)
    # Single fused contraction over (i, t) → (r, k). Avoids the
    # (n_img, n_rot, half_image) intermediate that OOM'd at vol=64
    # (Phase 4 finding). XLA plans this as one streaming reduction.
    per_r = jnp.einsum("irt,itk->rk", gamma, shifted_half).astype(jnp.complex128)

    if residual_subtraction_half is not None:
        per_r = per_r - residual_subtraction_half
    return per_r


def _per_rotation_ctf_image(
    config,
    ctf_params,
    noise_variance_full,
    log_resp,
):
    """Build, for each rotation r, the per-pixel `(CTF² / σ²)` weight
    summed over images and translations weighted by responsibilities:

        per_r_ctf[r, k_pix] = sum_i CTF_i²[k_pix] · sum_t γ_{i,r,t}

    Note: this is real-valued (CTF² is real) and lives in half-image.
    """
    n_img = ctf_params.shape[0]
    image_shape = config.image_shape

    CTF_full = config.compute_ctf(ctf_params, half_image=False)  # (n_img, full)
    ctf2_full = (CTF_full**2) / noise_variance_full  # (n_img, full)
    ctf2_half = ftu.full_image_to_half_image(ctf2_full, image_shape)  # (n_img, half)

    gamma = jnp.exp(log_resp)  # (n_img, n_rot, n_trans)
    gamma_per_ir = jnp.sum(gamma, axis=-1)  # (n_img, n_rot)
    # per_r_ctf[r, k] = sum_i gamma_per_ir[i, r] * ctf2_half[i, k]
    per_r_ctf = jnp.einsum("ir,ik->rk", gamma_per_ir, ctf2_half)
    return per_r_ctf


def _per_rotation_bias_image(
    config,
    ctf_params,
    noise_variance_full,
    log_resp,
    post_mean,
    u_proj_half,
):
    """Build the per-rotation half-image to subtract from `per_r`
    when residualizing:

        bias_image[r, k_pix] = sum_k u_proj_half[r, k, k_pix] *
                                sum_i CTF_i²[k_pix] / σ²[k_pix] *
                                sum_t γ_{i,r,t} · m_{i,r,t,k}

    Returns
    -------
    bias_half : (n_rot, half_image_size) complex128
    """
    n_img = ctf_params.shape[0]
    n_rot = log_resp.shape[1]
    q = post_mean.shape[-1]
    image_shape = config.image_shape

    CTF_full = config.compute_ctf(ctf_params, half_image=False)
    ctf2_full = (CTF_full**2) / noise_variance_full
    ctf2_half = ftu.full_image_to_half_image(ctf2_full, image_shape)  # (n_img, half)

    gamma = jnp.exp(log_resp)  # (n_img, n_rot, n_trans)
    # m_weighted is small for typical q (n_img × n_rot × q): kept
    # as the original two-step contraction; fusing here gave XLA
    # a worse contraction path at vol=64 (Phase 8.1 finding).
    m_weighted = jnp.einsum("irt,irtk->irk", gamma, post_mean)  # (n_img, n_rot, q)
    inner_per_r = jnp.einsum("ip,irk->rkp", ctf2_half, m_weighted)  # (n_rot, q, half)
    bias = jnp.sum(u_proj_half * inner_per_r, axis=1)  # (n_rot, half)
    return bias.astype(jnp.complex128)


# ---------------------------------------------------------------------------
# Public mean-update entry points
# ---------------------------------------------------------------------------


@dataclass
class MeanUpdateResult:
    """One mean-update output."""

    mu_half: jnp.ndarray  # (half_volume_size,) complex128
    Ft_y_half: jnp.ndarray  # accumulator (for diagnostics)
    Ft_ctf_half: jnp.ndarray  # accumulator (for diagnostics)


def _solve_wiener(Ft_y_half, Ft_ctf_half, tau, *, min_tau_frac: float = 0.05):
    """mu_next = Ft_y / (Ft_ctf + tau_eff). Element-wise per voxel.

    `tau_eff = max(tau, min_tau_frac * mean(Ft_ctf))`. The data-adaptive
    floor prevents the Wiener filter from blowing up at voxels with
    no pose coverage (where `Ft_ctf` is zero or near-zero) — these
    voxels collapse to ~0, which is the right behaviour for an
    uninformative prior.

    `min_tau_frac=0.05` means the smallest effective regularization
    is 5% of the average data coverage. v0 default; tunable per
    experiment.
    """
    Ft_ctf_real = Ft_ctf_half.real if jnp.iscomplexobj(Ft_ctf_half) else Ft_ctf_half
    floor = float(min_tau_frac) * float(jnp.mean(Ft_ctf_real))
    tau_eff = jnp.maximum(jnp.asarray(tau, dtype=jnp.float64), floor)
    return Ft_y_half / (Ft_ctf_half + tau_eff)


def _backproject_to_half_volume(per_r_half, rotations, image_shape, volume_shape):
    """One adjoint_slice_volume call on the full rotation stack.

    Uses nearest discretization to match the forward model in
    `posterior.py` (and the simulator). With nearest, A_g is binary,
    so the back-projection scatters each pixel into exactly one voxel
    — this is what makes the M-step's diagonal-block solve exact.
    """
    return adjoint_slice_volume(
        per_r_half,
        rotations,
        image_shape,
        volume_shape,
        "nearest",
        half_image=True,
        half_volume=True,
    )


def _accumulate_mu_for_batch(
    config,
    init,
    mean_proj_half,
    u_proj_half,
    weights_half,
    batch_b,
    ctf_b,
    translations,
    noise_variance_full,
    *,
    residualize: bool,
):
    """Per-image-batch contribution to (per_r, per_r_ctf) accumulators.

    Returns shapes (n_rot, half_image_size) that sum trivially across
    batches. Same math as the unbatched path; just sliced to a chunk.
    """
    shifted_half_b, ctf2_over_nv_half_b, _ = _preprocess_batch_to_half(
        config, batch_b, translations, ctf_b, noise_variance_full
    )
    stats_b = score_from_half_image_projections(
        mean_proj_half,
        u_proj_half,
        init.s,
        shifted_half_b,
        ctf2_over_nv_half_b,
        weights_half,
    )
    bias_b = None
    if residualize:
        bias_b = _per_rotation_bias_image(
            config,
            ctf_b,
            noise_variance_full,
            stats_b.log_resp,
            stats_b.post_mean,
            u_proj_half,
        )
    per_r_b = _per_rotation_residual_image(
        config,
        batch_b,
        translations,
        ctf_b,
        noise_variance_full,
        stats_b.log_resp,
        residual_subtraction_half=bias_b,
    )
    per_r_ctf_b = _per_rotation_ctf_image(
        config,
        ctf_b,
        noise_variance_full,
        stats_b.log_resp,
    )
    return per_r_b, per_r_ctf_b


def _stream_or_full_mu_update(
    config,
    init,
    batch_full,
    rotations,
    translations,
    ctf_params,
    noise_variance_full,
    *,
    tau,
    residualize,
    image_batch_size,
):
    """Common driver: image-batched accumulation when image_batch_size
    is set; otherwise the original single-shot code path."""
    weights_half = make_half_image_weights(config.image_shape)
    mean_proj_half = _slice_mu_half(init.mu, rotations, config.image_shape, config.volume_shape).astype(jnp.complex128)
    u_proj_half = _slice_U_half(init.U, rotations, config.image_shape, config.volume_shape).astype(jnp.complex128)

    n_img_total = batch_full.shape[0]
    n_rot = rotations.shape[0]
    half_image_size = mean_proj_half.shape[-1]
    use_streaming = image_batch_size is not None and image_batch_size < n_img_total

    if use_streaming:
        per_r = jnp.zeros((n_rot, half_image_size), dtype=jnp.complex128)
        per_r_ctf = jnp.zeros((n_rot, half_image_size), dtype=jnp.float64)
        for i_start in range(0, n_img_total, image_batch_size):
            i_end = min(i_start + image_batch_size, n_img_total)
            per_r_b, per_r_ctf_b = _accumulate_mu_for_batch(
                config,
                init,
                mean_proj_half,
                u_proj_half,
                weights_half,
                batch_full[i_start:i_end],
                ctf_params[i_start:i_end],
                translations,
                noise_variance_full,
                residualize=residualize,
            )
            per_r = per_r + per_r_b
            per_r_ctf = per_r_ctf + per_r_ctf_b
    else:
        per_r_b, per_r_ctf_b = _accumulate_mu_for_batch(
            config,
            init,
            mean_proj_half,
            u_proj_half,
            weights_half,
            batch_full,
            ctf_params,
            translations,
            noise_variance_full,
            residualize=residualize,
        )
        per_r = per_r_b
        per_r_ctf = per_r_ctf_b

    Ft_y_half = _backproject_to_half_volume(per_r, rotations, config.image_shape, config.volume_shape)
    Ft_ctf_half = _backproject_to_half_volume(per_r_ctf, rotations, config.image_shape, config.volume_shape)
    mu_next = _solve_wiener(Ft_y_half, Ft_ctf_half, tau).astype(jnp.complex128)
    return MeanUpdateResult(
        mu_half=mu_next,
        Ft_y_half=Ft_y_half,
        Ft_ctf_half=Ft_ctf_half,
    )


def update_mu_homogeneous(
    config,
    init: PPCAInit,
    batch_full,
    rotations,
    translations,
    ctf_params,
    noise_variance_full,
    *,
    tau: float = 1e-3,
    image_batch_size: int | None = None,
) -> MeanUpdateResult:
    """v0 homogeneous mean update; image-batched when image_batch_size is set."""
    return _stream_or_full_mu_update(
        config,
        init,
        batch_full,
        rotations,
        translations,
        ctf_params,
        noise_variance_full,
        tau=tau,
        residualize=False,
        image_batch_size=image_batch_size,
    )


def update_mu_residualized(
    config,
    init: PPCAInit,
    batch_full,
    rotations,
    translations,
    ctf_params,
    noise_variance_full,
    *,
    tau: float = 1e-3,
    image_batch_size: int | None = None,
) -> MeanUpdateResult:
    """v1 graduation-gate residualized mean update; image-batched when set."""
    return _stream_or_full_mu_update(
        config,
        init,
        batch_full,
        rotations,
        translations,
        ctf_params,
        noise_variance_full,
        tau=tau,
        residualize=True,
        image_batch_size=image_batch_size,
    )

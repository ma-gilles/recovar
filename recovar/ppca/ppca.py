"""Direct PPCA implementation ported onto the current recovar APIs."""

import functools
import logging

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import recovar.core.fourier_transform_utils as ftu
from recovar import core, utils
from recovar.core import linalg
from recovar.heterogeneity import covariance_estimation
from recovar.output import metrics

logger = logging.getLogger(__name__)


def _normalize_experiment_datasets(experiment_datasets):
    if hasattr(experiment_datasets, "materialize_halfset_datasets"):
        return experiment_datasets, list(experiment_datasets.materialize_halfset_datasets())
    if isinstance(experiment_datasets, (list, tuple)):
        datasets = list(experiment_datasets)
        if not datasets:
            raise ValueError("experiment_datasets cannot be empty")
        full_dataset = None
        if len(datasets) == 1 and hasattr(datasets[0], "materialize_halfset_datasets"):
            full_dataset = datasets[0]
            datasets = list(full_dataset.materialize_halfset_datasets())
        return full_dataset, datasets
    raise TypeError("experiment_datasets must be a CryoEMDataset or a sequence of CryoEMDataset objects")


def _iter_processed_batches(experiment_dataset, batch_size):
    for (
        batch,
        rotation_matrices,
        translations,
        ctf_params,
        _noise_variance,
        _particle_indices,
        image_indices,
    ) in experiment_dataset.iter_batches(
        batch_size,
        by_image=not getattr(experiment_dataset, "tilt_series_flag", False),
    ):
        yield (
            experiment_dataset.process_images(batch, apply_image_mask=False),
            ctf_params,
            rotation_matrices,
            translations,
            image_indices,
        )


def _forward_model_from_map(
    volume,
    ctf_params,
    rotation_matrices,
    image_shape,
    volume_shape,
    voxel_size,
    ctf_evaluator,
    disc_type,
    skip_ctf=False,
):
    slices = core.slice_volume(
        volume,
        rotation_matrices,
        image_shape,
        volume_shape,
        disc_type,
    )
    if not skip_ctf:
        slices = slices * ctf_evaluator(ctf_params, image_shape, voxel_size)
    return slices


def _prepare_mean_estimate_for_slicing(mean_estimate, mean_estimate_raw, volume_shape, disc_type_mean):
    """Return the mean representation expected by ``slice_volume``.

    ``slice_volume(..., disc_type="cubic")`` expects cubic B-spline
    coefficients, not raw Fourier samples. Callers can either pass raw Fourier
    data via ``mean_estimate_raw`` or pass precomputed coefficients directly via
    ``mean_estimate``.
    """
    if disc_type_mean != "cubic":
        return mean_estimate
    if mean_estimate_raw is None:
        return mean_estimate
    return core.precompute_cubic_coefficients(mean_estimate_raw, volume_shape)


def compute_Cz_from_second_moments(second_moment_zs):
    """
    Compute the empirical posterior covariance Ĉ_z = (1/N) Σ_n E[z_n z_n^T | y_n].

    Args:
        second_moment_zs: Array of shape (N, q, q) containing E[z_n z_n^T | y_n] for each sample
                          where E[z_n z_n^T | y_n] = Σ_n + μ_n μ_n^T

    Returns:
        C_z: The empirical posterior covariance of shape (q, q)
    """
    return jnp.mean(second_moment_zs, axis=0)


batch_over_vol_slice_volume = jax.vmap(core.slice_volume, in_axes=(1, None, None, None, None), out_axes=1)


def check_imaginary_part(x, image_shape, name, skip_ft=False):
    if not skip_ft:
        if len(image_shape) == 2:
            y = ftu.get_idft2(x.reshape(-1, *image_shape))
        else:
            y = ftu.get_idft3(x.reshape(-1, *image_shape))
    else:
        y = x
    imag_norm = np.linalg.norm(y.imag)
    ratio = np.inf if imag_norm == 0 else np.linalg.norm(y.real) / imag_norm
    print("imaginary part ratio", name, ratio)
    return ratio


batch_over_vol_adjoint_slice_volume = jax.vmap(
    core.adjoint_slice_volume, in_axes=(-1, None, None, None, None), out_axes=-1
)


def _tri_size(q):
    """Number of upper-triangular entries (including diagonal) in a q×q matrix."""
    return (q * (q + 1)) // 2


_half_slice_volume = functools.partial(core.slice_volume, half_volume=True, half_image=True)
batch_over_vol_slice_volume_half = jax.vmap(
    _half_slice_volume, in_axes=(1, None, None, None, None), out_axes=1
)

_half_adjoint_slice_volume = functools.partial(core.adjoint_slice_volume, half_image=True, half_volume=True)
batch_over_vol_adjoint_slice_volume_half = jax.vmap(
    _half_adjoint_slice_volume, in_axes=(-1, None, None, None, None), out_axes=-1
)


def _e_step_half_inner(
    images_half, mean, W_half, CTF_params, rotation_matrices, translations,
    voxel_size, noise_variance_half,
    image_shape, volume_shape, ctf_evaluator, compute_ll,
    disc_type_mean, disc_type,
    z_prior_precision_diag,  # (q,) array of 1/var per dim. Use jnp.ones(q) for identity prior.
):
    """JIT'd E-step core: computes sufficient stats and c=1 posterior.

    Returns sufficient statistics (H, g, h, t, nu, y_norm_sq) plus
    noise-whitened images/mean/CTF for backprojection, and the standard
    (c=1) posterior moments.

    Contrast dispatch happens OUTSIDE JIT in E_M_step_batch_half.
    """
    basis_size = W_half.shape[1]

    w_1d = linalg.half_spectrum_last_axis_weights(image_shape[1])
    rfft_w = jnp.tile(w_1d, (image_shape[0], 1)).reshape(-1)

    images_half = core.translate_images(images_half, translations, image_shape, half_image=True) / jnp.sqrt(
        noise_variance_half
    )
    CTF_half = ctf_evaluator(CTF_params, image_shape, voxel_size, half_image=True) / jnp.sqrt(noise_variance_half)

    projected_mean_half = core.slice_volume(
        mean, rotation_matrices, image_shape, volume_shape, disc_type_mean, half_image=True
    )
    projected_mean_half = (
        projected_mean_half
        * ctf_evaluator(CTF_params, image_shape, voxel_size, half_image=True)
        / jnp.sqrt(noise_variance_half)
    )

    ctf_squared_half = CTF_half**2

    PW_half = batch_over_vol_slice_volume_half(W_half, rotation_matrices, image_shape, volume_shape, disc_type)
    PW_half *= CTF_half[:, None, :]

    PW_w = jnp.conj(PW_half) * rfft_w[None, None, :]

    # Sufficient statistics for latent posterior
    H = (PW_w @ PW_half.transpose(0, 2, 1)).real
    g = (PW_w @ images_half[..., None]).real.squeeze(-1)
    h = (PW_w @ projected_mean_half[..., None]).real.squeeze(-1)
    t = jnp.sum(rfft_w * jnp.real(jnp.conj(images_half) * projected_mean_half), axis=-1)
    nu = jnp.sum(rfft_w * jnp.real(jnp.conj(projected_mean_half) * projected_mean_half), axis=-1)
    y_norm_sq = jnp.sum(rfft_w * jnp.real(jnp.conj(images_half) * images_half), axis=-1)

    # Standard c=1 posterior (always computed — used for LL and as fallback).
    # Caller always supplies z_prior_precision_diag (a (q,) array). For the
    # default identity prior z ~ N(0, I), pass jnp.ones(q). For a calibrated
    # prior z ~ N(0, diag(eig)), pass 1/eig.
    M_n = H + jnp.diag(z_prior_precision_diag)
    b_n = (g - h)[..., None]
    M_n_inv = jax.numpy.linalg.pinv(M_n, hermitian=True)
    expected_zs = (M_n_inv @ b_n).squeeze(-1)
    second_moment_zs = M_n_inv + linalg.broadcast_outer(expected_zs, jnp.conj(expected_zs))

    ll_sum = jnp.array(0.0, dtype=images_half.dtype)
    if compute_ll:
        u = b_n.squeeze(-1)
        quad = jnp.real(jnp.sum(jnp.conj(u) * (M_n_inv @ u[..., None]).squeeze(-1), axis=-1))
        r2 = jnp.sum(rfft_w * jnp.real(jnp.conj(images_half) * images_half), axis=-1)
        centered_half = images_half - projected_mean_half
        r2 = jnp.sum(rfft_w * jnp.real(jnp.conj(centered_half) * centered_half), axis=-1)
        L = jnp.linalg.cholesky(M_n)
        logdetM = 2.0 * jnp.sum(jnp.log(jnp.real(jnp.diagonal(L, axis1=1, axis2=2))), axis=-1)
        d_n = np.prod(image_shape)
        ll_per_image = -0.5 * (d_n * jnp.log(2.0 * jnp.pi) + r2 - quad + logdetM)
        ll_sum = jnp.sum(ll_per_image)

    return (expected_zs, second_moment_zs, ctf_squared_half,
            images_half, projected_mean_half, CTF_half, ll_sum,
            H, g, h, t, nu, y_norm_sq)


def E_M_step_batch_half(
    images_half,
    lhs_summed,
    rhs_summed,
    mean,
    W_half,
    CTF_params,
    rotation_matrices,
    translations,
    image_shape,
    volume_shape,
    grid_size,
    voxel_size,
    noise_variance_half,
    ctf_evaluator,
    compute_ll,
    disc_type_mean="cubic",
    disc_type="linear_interp",
    compute_stats=True,
    contrast_mode="none",
    contrast_grid=None,
    contrast_weights=None,
    eigenvalues=None,
    contrast_mean=1.0,
    contrast_variance=np.inf,
):
    """Half-spectrum, upper-triangular-LHS variant of :func:`E_M_step_batch`.

    E-step is JIT'd. LHS backprojection uses the fused CUDA kernel.
    When contrast_mode != "none", uses contrast-weighted moments:
      LHS += CTF² ⊗ E[c²zz^T],  RHS += CTF·y·E[cz]^T - CTF·Aμ·E[c²z]^T
    """
    basis_size = W_half.shape[1]
    tri_i, tri_j = np.triu_indices(basis_size)
    tri_sz = len(tri_i)

    from recovar.ppca import contrast_posterior

    # --- JIT'd E-step: sufficient stats + c=1 posterior ---
    # When `eigenvalues` is provided AND contrast_mode == "none", use it as the
    # c=1 prior variance per component (z_k ~ N(0, eigenvalues[k])). Otherwise
    # default to the identity prior (jnp.ones).
    if eigenvalues is not None and contrast_mode == "none":
        z_prior_precision_diag = 1.0 / jnp.maximum(jnp.asarray(eigenvalues, dtype=W_half.real.dtype), 1e-12)
    else:
        z_prior_precision_diag = jnp.ones(basis_size, dtype=W_half.real.dtype)
    (expected_zs, second_moment_zs, ctf_squared_half,
     images_half_w, projected_mean_half_w, CTF_half, ll_sum,
     H, g, h, t, nu, y_norm_sq) = _e_step_half_inner(
        images_half, mean, W_half, CTF_params, rotation_matrices,
        translations, voxel_size, noise_variance_half, image_shape,
        volume_shape, ctf_evaluator, compute_ll, disc_type_mean, disc_type,
        z_prior_precision_diag,
    )

    # --- Contrast dispatch (outside JIT) ---
    if contrast_mode == "none":
        mean_cz = expected_zs
        mean_c2z = expected_zs
        second_moment_czz = second_moment_zs
        mean_c = jnp.ones(expected_zs.shape[0])
    else:
        if eigenvalues is None:
            eigenvalues = jnp.ones(basis_size)
        result = contrast_posterior.solve_latent_posterior(
            H=H, g=g, h=h, t=t, nu=nu, y_norm_sq=y_norm_sq,
            lambdas=eigenvalues,
            contrast_mode=contrast_mode,
            contrast_nodes=contrast_grid,
            contrast_weights=contrast_weights,
            contrast_mean=contrast_mean,
            contrast_variance=contrast_variance,
        )
        expected_zs = result.mean_z
        mean_cz = result.mean_cz
        mean_c2z = result.mean_c2z
        second_moment_czz = result.second_moment_czz
        mean_c = result.mean_c

    # --- backprojection ---
    if compute_stats:
        half_volume_size = lhs_summed.shape[0]
        # LHS uses E[c²zz^T] (= E[zz^T] when c=1)
        second_moment_tri = second_moment_czz[:, tri_i, tri_j]
        ctf_squared_full = ftu.half_image_to_full_image(ctf_squared_half, image_shape)
        _max_r = image_shape[0] // 2 - 1

        n_images = ctf_squared_full.shape[0]
        real_dtype = lhs_summed.dtype
        from recovar.cuda_backproject import per_image_backproject

        ctf2_bp = per_image_backproject(
            jnp.zeros((half_volume_size, n_images), dtype=real_dtype),
            ctf_squared_full.real.astype(real_dtype),
            jnp.asarray(rotation_matrices),
            image_shape,
            volume_shape,
            max_r=_max_r,
        )  # (half_vol, n_images)

        ## TODO: memory-inefficient GEMM — see original comment
        lhs_summed = lhs_summed + (ctf2_bp @ second_moment_tri.real.astype(real_dtype))
        # except Exception:
        #     # Fallback: standard per-column backprojection (slower but works on CPU)
        #     tri_sz = second_moment_tri.shape[1]
        #     for k in range(tri_sz):
        #         w = ctf_squared_full.real.astype(real_dtype) * second_moment_tri[:, k : k + 1].real.astype(real_dtype)
        #         bp_col = core.batch_adjoint_slice_volume(
        #             w[:, :, None].transpose(2, 0, 1),
        #             rotation_matrices,
        #             image_shape,
        #             volume_shape,
        #             disc_type,
        #             half_image=False,
        #             half_volume=True,
        #         )
        #         lhs_summed = lhs_summed.at[:, k].add(bp_col.reshape(-1).real)

        # The W-update always uses the basis adjoint P_W*, so both RHS terms
        # must backproject through the basis interpolation disc_type. For c=1
        # this collapses to the original centered residual CTF · (y - Aμ) · E[z]^T.
        rhs_residual = (
            images_half_w[..., None] * jnp.conj(mean_cz)[:, None, :]
            - projected_mean_half_w[..., None] * jnp.conj(mean_c2z)[:, None, :]
        )
        before_rhs = (CTF_half[..., None] * rhs_residual).transpose(2, 0, 1)
        bp_rhs = core.batch_adjoint_slice_volume(
            before_rhs,
            rotation_matrices,
            image_shape,
            volume_shape,
            disc_type,
            half_image=True,
            half_volume=True,
        )
        rhs_summed = rhs_summed + bp_rhs.T

    ll_per_image = jnp.zeros((0,), dtype=images_half.dtype)
    return lhs_summed, rhs_summed, expected_zs, second_moment_czz, ll_sum, ll_per_image, mean_c


def unpack_tri_to_full(lhs_tri, basis_size):
    """Unpack upper-triangular ``(…, tri_size)`` to symmetric ``(…, q, q)``.

    Useful for converting the output of :func:`E_M_step_batch_half` back to the
    full matrix format expected by downstream solvers.
    """
    tri_i, tri_j = np.triu_indices(basis_size)
    shape = lhs_tri.shape[:-1] + (basis_size, basis_size)
    out = jnp.zeros(shape, dtype=lhs_tri.dtype)
    out = out.at[..., tri_i, tri_j].set(lhs_tri)
    out = out.at[..., tri_j, tri_i].set(lhs_tri)
    return out


batch1_symmetrize_ft_volume = jax.vmap(utils.symmetrize_ft_volume, in_axes=(1, None), out_axes=1)


# ═════════════════════════════════════════════════════════════════════════
# Masked PCG hard M-step solver (K-internalised, support-restricted)
#
# Solves   min_V  (KV)^T A (KV) + V^T Λ V   subject to  V ⊂ supp(mask)
# in real-space reduced coordinates V = E Z where E gathers the mask
# support, with circulant Fourier preconditioner. K = sinc² is the
# trilinear gridding kernel; baking it into the operator means the
# returned W is already mask-zeroed AND deconvolved — no post-EM cleanup
# needed.
#
# Adapted from bench_mstep.py (the masked-mstep benchmark). See
# docs/math/masked_mstep.md for the derivation.
# ═════════════════════════════════════════════════════════════════════════

_MSTEP_PC_BATCH = 4
_MSTEP_CHUNK = 100_000


def _compute_gridding_kernel(volume_shape):
    """K(x) = sinc²(x/D), the inverse of trilinear interpolation in Fourier."""
    D = volume_shape[0]
    c = np.arange(D, dtype=np.float32) - D / 2
    n = c / D

    def sinc(a):
        s = np.where(np.abs(a) < 1e-8, 1.0, a)
        return np.where(np.abs(a) < 1e-8, 1.0, np.sin(np.pi * s) / (np.pi * s))

    g = sinc(n) ** 2
    return jnp.array((g[:, None, None] * g[None, :, None] * g[None, None, :]).astype(np.float32))


def _mstep_batched_rfft(V, vs):
    """(q, D, D, D) real → (half_vol, q) complex."""
    q = V.shape[0]
    hvs = ftu.get_real_fft_packed_shape(vs)
    hv = int(np.prod(hvs))
    parts = []
    for j0 in range(0, q, _MSTEP_PC_BATCH):
        j1 = min(j0 + _MSTEP_PC_BATCH, q)
        parts.append(ftu.get_dft3_real(V[j0:j1]).reshape(j1 - j0, hv).T)
    return jnp.concatenate(parts, axis=1)


def _mstep_batched_irfft(W_h, vs, q):
    """(half_vol, q) complex → (q, D, D, D) real."""
    hvs = ftu.get_real_fft_packed_shape(vs)
    parts = []
    for j0 in range(0, q, _MSTEP_PC_BATCH):
        j1 = min(j0 + _MSTEP_PC_BATCH, q)
        parts.append(ftu.get_idft3_real(W_h[:, j0:j1].T.reshape(j1 - j0, *hvs), vs))
    return jnp.concatenate(parts, axis=0)


def _mstep_A_mul_fourier(W_h, lhs_tri, q, unpack_fn):
    """A(ξ) · W(ξ), chunked. ``lhs_tri`` is upper-tri-packed (hv, tri_sz)."""
    hv = W_h.shape[0]
    out = jnp.zeros_like(W_h)
    for i0 in range(0, hv, _MSTEP_CHUNK):
        i1 = min(i0 + _MSTEP_CHUNK, hv)
        L = unpack_fn(lhs_tri[i0:i1], q)
        if L.ndim == 2:
            L = L[:, :, None]
        out = out.at[i0:i1].set(jnp.einsum("vij,vj->vi", L, W_h[i0:i1]))
    return out


def _mstep_AL_solve_fourier(W_h, lhs_tri, reg_diag, q, unpack_fn, lhs_scale=1.0):
    """(scale·A(ξ) + Λ(ξ))^{-1} · W(ξ), chunked — used by the preconditioner."""
    hv = W_h.shape[0]
    parts = []
    for i0 in range(0, hv, _MSTEP_CHUNK):
        i1 = min(i0 + _MSTEP_CHUNK, hv)
        L = unpack_fn(lhs_tri[i0:i1], q)
        if L.ndim == 2:
            L = L[:, :, None]
        if lhs_scale != 1.0:
            L = lhs_scale * L
        D = L.at[:, jnp.arange(q), jnp.arange(q)].add(reg_diag[i0:i1])
        parts.append(jnp.linalg.solve(D, W_h[i0:i1, :, None])[..., 0])
    return jnp.concatenate(parts, axis=0)


def _mstep_apply_fourier_op(V_real, lhs_tri, reg_diag, q, vs, unpack_fn, G):
    """Real-space matvec   K · iFFT[(A + Λ) · FFT[K · V]]."""
    KV = G[None] * V_real
    KV_h = _mstep_batched_rfft(KV, vs)
    AKV_h = _mstep_A_mul_fourier(KV_h, lhs_tri, q, unpack_fn)
    result_h = AKV_h + reg_diag * KV_h
    result = _mstep_batched_irfft(result_h, vs, q)
    return G[None] * result


def _mstep_cg(matvec, b, x0, maxiter, tol, precond):
    """Preconditioned CG, real-flat vectors. Returns ``x`` at convergence/maxiter."""
    x = x0
    r = b - matvec(x)
    z = precond(r) if precond is not None else r
    p = z
    rz = float(jnp.sum(r * z))
    b2 = max(float(jnp.sum(b * b)), 1e-30)
    for _ in range(maxiter):
        Ap = matvec(p)
        pAp = float(jnp.sum(p * Ap))
        if pAp < 1e-30:
            break
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rr = float(jnp.sum(r * r))
        if jnp.sqrt(rr / b2) < tol:
            break
        z = precond(r) if precond is not None else r
        rz_new = float(jnp.sum(r * z))
        p = z + (rz_new / max(abs(rz), 1e-30)) * p
        rz = rz_new
    return x


def _pcg_hard_mstep(lhs_tri, rhs_h, reg_diag, mask, vs, q, unpack_fn,
                    W0_real=None, maxiter=20, tol=1e-4):
    """Masked, K-internalised PCG hard M-step.

    Solves the masked PPCA M-step in support-restricted coordinates Z, with
    the gridding kernel K = sinc² baked into the operator. The returned W
    is real-space (q, D, D, D), already mask-zeroed and K-deconvolved —
    callers feed it straight into rfft (no post-mask, no post-gridding).
    """
    G = _compute_gridding_kernel(vs)
    sup = jnp.where(jnp.asarray(mask).ravel() > 0.5)[0]
    n_sup = sup.shape[0]
    vol = int(np.prod(vs))
    N = vol
    k_eff_sq = float(jnp.sum(G ** 2)) / N

    def scatter(Z_flat):
        Z = Z_flat.reshape(n_sup, q)
        f = jnp.zeros((q, vol), dtype=jnp.float32)
        return f.at[:, sup].set(Z.T).reshape(q, *vs)

    def gather(V):
        return V.reshape(q, vol)[:, sup].T.ravel()

    def matvec(Z_flat):
        V = scatter(Z_flat)
        return gather(_mstep_apply_fourier_op(V, lhs_tri, reg_diag, q, vs, unpack_fn, G))

    # RHS = E^T K iFFT[d]
    d_real = _mstep_batched_irfft(rhs_h, vs, q)
    rhs_flat = gather(G[None] * d_real).astype(jnp.float32)

    # Circulant preconditioner: E^T iFFT[(k_eff² A + Λ)^{-1} FFT[E Z]]
    prec_reg = k_eff_sq * reg_diag

    def precond(Z_flat):
        V = scatter(Z_flat)
        V_h = _mstep_batched_rfft(V, vs)
        S_h = _mstep_AL_solve_fourier(V_h, lhs_tri, prec_reg, q, unpack_fn, lhs_scale=k_eff_sq)
        return gather(_mstep_batched_irfft(S_h, vs, q))

    # Initial guess: warmstart from previous iter (in support coords) or
    # per-voxel Wiener (with K^{-1}) gathered to the support.
    if W0_real is not None:
        x0 = gather(jnp.asarray(W0_real, dtype=jnp.float32).reshape(q, *vs))
    else:
        W0_h = _mstep_AL_solve_fourier(rhs_h, lhs_tri, reg_diag, q, unpack_fn)
        W0_r = _mstep_batched_irfft(W0_h, vs, q) / jnp.maximum(G[None], 0.01)
        x0 = gather(W0_r).astype(jnp.float32)

    Z_flat = _mstep_cg(matvec, rhs_flat, x0, maxiter, tol, precond)
    return scatter(Z_flat).astype(jnp.float32)


def _iter_processed_batches_half(experiment_dataset, batch_size):
    """Like _iter_processed_batches but yields half-spectrum images and noise."""
    for (
        batch,
        rotation_matrices,
        translations,
        ctf_params,
        _noise_variance,
        _particle_indices,
        image_indices,
    ) in experiment_dataset.iter_batches(
        batch_size,
        by_image=not getattr(experiment_dataset, "tilt_series_flag", False),
    ):
        yield (
            experiment_dataset.process_images_half(batch, apply_image_mask=False),
            ctf_params,
            rotation_matrices,
            translations,
            image_indices,
        )


def EM_step_half(
    experiment_datasets,
    mean_estimate,
    W_estimate,
    batch_size,
    W_prior,
    disc_type_mean="cubic",
    disc_type="linear_interp",
    recompute_ll=False,
    mean_estimate_raw=None,
    volume_mask=None,
    pcg_maxiter=20,
    W_prev_real=None,
    contrast_mode="none",
    contrast_grid=None,
    contrast_weights=None,
    eigenvalues=None,
    contrast_mean=1.0,
    contrast_variance=np.inf,
    return_mean_c=False,
):
    """Half-spectrum EM step for L2-regularized PPCA.

    Accumulates ``lhs`` (half_vol, tri_sz) and ``rhs`` (half_vol, q) in
    half-volume / upper-triangular format, then runs the K-internalised
    masked PCG hard solver to produce the new W. The returned W is in
    half-Fourier shape, already mask-zeroed and gridding-corrected — the
    outer EM loop does not need any post-step cleanup.
    """
    full_dataset, dataset_list = _normalize_experiment_datasets(experiment_datasets)
    ref = full_dataset if full_dataset is not None else dataset_list[0]
    basis_size = W_estimate.shape[-1]
    volume_shape = ref.volume_shape
    half_volume_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
    half_volume_size = int(np.prod(half_volume_shape))
    tri_sz = _tri_size(basis_size)

    # W_estimate can be either (volume_size, q) full Fourier or (half_vol, q) half-volume.
    # Detect and convert if needed.
    if W_estimate.shape[0] == half_volume_size:
        W_half = W_estimate  # already half-volume
    else:
        W_half = ftu.full_volume_to_half_volume(W_estimate.T, volume_shape).T

    mean_for_slicing = _prepare_mean_estimate_for_slicing(
        mean_estimate,
        mean_estimate_raw,
        volume_shape,
        disc_type_mean,
    )

    # Half-volume accumulators
    lhs_summed = jnp.zeros((half_volume_size, tri_sz), dtype=ref.dtype_real)
    rhs_summed = jnp.zeros((half_volume_size, basis_size), dtype=ref.dtype)

    ll_sum = jnp.array(0.0, dtype=ref.dtype)
    expected_zs = []
    second_moment_zs = []
    mean_cs = []

    for experiment_dataset in dataset_list:
        for batch_half, ctf_params, rotation_matrices, translations, batch_image_ind in _iter_processed_batches_half(
            experiment_dataset, batch_size
        ):
            noise_variance_half = experiment_dataset.noise.get_half(batch_image_ind)
            lhs_summed, rhs_summed, ez_batch, smz_batch, ll_batch, _, _mc = E_M_step_batch_half(
                batch_half,
                lhs_summed,
                rhs_summed,
                mean_for_slicing,
                W_half,
                ctf_params,
                rotation_matrices,
                translations,
                experiment_dataset.image_shape,
                experiment_dataset.volume_shape,
                experiment_dataset.grid_size,
                experiment_dataset.voxel_size,
                noise_variance_half,
                experiment_dataset.ctf_evaluator,
                compute_ll=True,
                disc_type_mean=disc_type_mean,
                disc_type=disc_type,
                compute_stats=True,
                contrast_mode=contrast_mode,
                contrast_grid=contrast_grid,
                contrast_weights=contrast_weights,
                eigenvalues=eigenvalues,
                contrast_mean=contrast_mean,
                contrast_variance=contrast_variance,
            )
            expected_zs.append(np.array(ez_batch))
            second_moment_zs.append(np.array(smz_batch))
            mean_cs.append(np.array(_mc))
            ll_sum += ll_batch

    expected_zs = np.concatenate(expected_zs, axis=0)
    second_moment_zs = np.concatenate(second_moment_zs, axis=0)
    mean_cs = np.concatenate(mean_cs, axis=0)
    expected_zs_mean = np.mean(expected_zs, axis=0)
    expected_zs_var = np.var(expected_zs, axis=0)

    # ------------------------------------------------------------------
    # M-step solve: K-internalised PCG hard solver in support-restricted
    # real-space coordinates. The output W_real is already mask-zeroed and
    # K-deconvolved — callers feed it straight into rfft. No post-EM
    # mask projection or gridding correction is needed.
    # ------------------------------------------------------------------
    W_prior_half = ftu.full_volume_to_half_volume(W_prior.T, volume_shape).T
    reg_half = 1 / (W_prior_half + 1e-16)
    mask = jnp.array(volume_mask).reshape(volume_shape) if volume_mask is not None \
        else jnp.ones(volume_shape, dtype=jnp.float32)

    W0 = None
    if W_prev_real is not None:
        W0 = jnp.array(W_prev_real.T.reshape(basis_size, *volume_shape))

    W_real = _pcg_hard_mstep(
        lhs_summed,
        rhs_summed,
        reg_half,
        mask,
        volume_shape,
        basis_size,
        unpack_tri_to_full,
        W0_real=W0,
        maxiter=pcg_maxiter,
        tol=1e-4,
    )
    # (q, D, D, D) real → (half_vol, q) half-Fourier
    W = ftu.get_dft3_real(W_real).reshape(basis_size, -1).T

    if jnp.any(jnp.isnan(W)):
        logger.error("EM_step_half: NaN in W after M-step")
        raise ValueError("NaN in W after M-step")

    # ------------------------------------------------------------------
    # Log-likelihood (uses E-step LL with OLD W; recompute_ll is handled in EM loop)
    # ------------------------------------------------------------------
    W_prior_half = (
        ftu.full_volume_to_half_volume(W_prior.T, volume_shape).T if W_prior.shape[0] != W.shape[0] else W_prior
    )
    ll_prior = jnp.linalg.norm(W / jnp.sqrt(W_prior_half + 1e-16)) ** 2
    neg_ll_total = float(-ll_sum.real + ll_prior.real)
    neg_ll_data = float(-ll_sum.real)
    neg_ll_prior = float(ll_prior.real)

    if jnp.isnan(W).any():
        logger.error("EM_step_half produced NaN in W")
        raise ValueError("EM_step_half produced NaN in W")

    if return_mean_c:
        return (
            W,
            expected_zs,
            second_moment_zs,
            expected_zs_mean,
            expected_zs_var,
            neg_ll_total,
            neg_ll_data,
            neg_ll_prior,
            mean_cs,
        )
    return W, expected_zs, second_moment_zs, expected_zs_mean, expected_zs_var, neg_ll_total, neg_ll_data, neg_ll_prior


# @functools.partial(jax.jit, static_argnums = [5])

def batch_vec(x):
    return x.swapaxes(-1, -2).reshape(-1, x.shape[-1] ** 2)


def batch_unvec(x):
    n = np.sqrt(x.shape[-1]).astype(int)
    return x.reshape(-1, n, n).swapaxes(-1, -2)


def _orthonormalize_W_to_basis(W_half, volume_shape):
    """Orthonormalise a half-Fourier loading matrix in real space.

    Returns ``(U_real, s, Vt)`` where ``U_real`` has shape
    ``(q, *volume_shape)`` and is orthonormal in the PPCA Fourier convention
    (i.e. real-space norm = 1/√vol_size, Fourier-space norm = 1), and
    ``s`` are the corresponding eigenvalues (squared singular values in
    the Fourier convention). Used by :func:`EM` to take a clean basis
    snapshot for projected-covariance refinement.
    """
    vs = volume_shape
    half_vs = ftu.volume_shape_to_half_volume_shape(vs)
    q = W_half.shape[1]
    vol_size = int(np.prod(vs))

    # half-Fourier rfft → real space, all q columns at once
    W_half_arr = np.asarray(W_half).T.reshape(q, *half_vs)
    W_real_vol = np.asarray(ftu.get_idft3_real(W_half_arr, vs))
    W_real = W_real_vol.reshape(q, vol_size).T.astype(np.float32)

    U_flat, S_real, Vt = np.linalg.svd(W_real, full_matrices=False)
    s = (S_real ** 2 * vol_size).astype(np.float32)
    U_flat = U_flat / np.sqrt(vol_size)
    U_real = U_flat.T.reshape(q, *vs).astype(np.float32)
    return U_real, s, Vt


def EM(
    experiment_dataset,
    mean_estimate,
    W_initial,
    W_prior,
    EM_iter=20,
    disc_type_mean="cubic",
    disc_type="linear_interp",
    disc_type_u="linear_interp",
    return_iteration_data=False,
    return_posterior_info=False,
    recompute_ll=False,
    volume_mask=None,
    dilated_volume_mask=None,
    pcg_maxiter=20,
    contrast_mode="none",
    contrast_grid=None,
    contrast_mean=1.0,
    contrast_variance=np.inf,
    projcov_every=0,
    projcov_start=0,
    gpu_memory_to_use=40,
):
    """Run EM for L2-regularized PPCA.

    Args:
        experiment_dataset: List of cryo-EM datasets.
        mean_estimate: Mean volume estimate (Fourier).
        W_initial: Initial loading matrix.
        W_prior: Prior variance for L2 regularization. Larger → less regularization.
        EM_iter: Number of EM iterations.
        disc_type_mean: Interpolation type for mean projection ('cubic' or 'linear_interp').
        return_iteration_data: If True, return per-iteration diagnostics.
        return_posterior_info: If True, also return ``mean_c`` from the final E-step.
        recompute_ll: If True, recompute data log-likelihood using updated W each iter.
        volume_mask: Real-space support mask used by the M-step PCG solver and the
            post-EM mask projection.
        dilated_volume_mask: Mask passed to ``pca_by_projected_covariance``.
        pcg_maxiter: Max CG iterations in the M-step PCG solve.
        soft_penalty_lam: If >0, use soft mask penalty λ||(1-mask)*W||² in PCG M-step.

        projcov_every: int, default 0. Run projected covariance after every
            Nth EM iteration. ``0`` = never (plain PPCA). ``1`` = every iter
            (interleaved). ``EM_iter`` = once at the end (single-shot refinement).
        projcov_start: int, default 0. First EM iter (0-indexed) at which a
            projcov pass is allowed.
        gpu_memory_to_use: Memory budget hint for projcov.

    Returns:
        U: Principal components (SVD of W in half-Fourier).
        S: Singular values squared.
        W: Final loading matrix in half-Fourier.
        expected_zs, second_moment_zs: Final posterior moments.
        iteration_data (optional), posterior_info (optional).
    """
    # Initialize
    # import jax.random as jr
    # matrix_key, vector_key = jr.split(jr.PRNGKey(0))
    # W = jr.normal(matrix_key, (experiment_dataset.volume_size, basis_size), dtype = experiment_dataset.dtype_real)
    # W = linalg.batch_dft3(W, experiment_dataset.volume_shape, basis_size)
    # eigenvalue = np.ones(basis_size)
    full_dataset, dataset_list = _normalize_experiment_datasets(experiment_dataset)
    reference_dataset = full_dataset if full_dataset is not None else dataset_list[0]
    if volume_mask is None:
        volume_mask = np.ones(reference_dataset.volume_shape)
    if dilated_volume_mask is None:
        dilated_volume_mask = volume_mask
    basis_size = W_initial.shape[-1]

    if projcov_every > 0:
        # Lazy import to avoid pulling heterogeneity into the cold path.
        from recovar.heterogeneity import principal_components

    if contrast_grid is None:
        contrast_grid = np.ones([1])
    # Larger batches amortize per-batch overhead (kernel launches, backprojection).
    # 200 is optimal for 256³/20 PCs on 80 GB GPU (~37 GB peak, 4.6 ms/img).
    # Smaller grids use less memory so the same batch size is safe.
    batch_size = 200
    W = W_initial

    # Keep the raw Fourier mean here. EM_step/EM_step_half precompute cubic
    # spline coefficients once per step, outside the per-batch loops.
    mean_estimate_raw = mean_estimate

    # PCG warmstart state
    _W_prev_real = None

    # Initialize table for collecting iteration data
    iteration_data = []
    posterior_info = None

    # Print table header
    print("\n" + "=" * 130)
    print("EM ALGORITHM CONVERGENCE TABLE")
    print("=" * 130)
    header = f"{'Iter':>4} | {'Neg_LL_Total':>12} | {'Neg_LL_Data':>12} | {'Neg_LL_Prior':>12} | {'Exp_ZS_Mean':>12} | {'Exp_ZS_Var':>12}"
    print(header)
    print("-" * len(header))

    vs = reference_dataset.volume_shape
    half_vs = ftu.volume_shape_to_half_volume_shape(vs)

    for iter_i in range(EM_iter):
        # Contrast warmup: skip contrast for first few iters when random W
        # makes H >> I and causes contrast to collapse to the grid boundary.
        contrast_warmup = 3
        _contrast_mode = contrast_mode if iter_i >= contrast_warmup else "none"

        (
            W,
            expected_zs,
            second_moment_zs,
            expected_zs_mean,
            expected_zs_var,
            neg_ll_total,
            neg_ll_data,
            neg_ll_prior,
            mean_c,
        ) = EM_step_half(
            experiment_dataset,
            mean_estimate,
            W,
            batch_size,
            W_prior,
            disc_type_mean=disc_type_mean,
            disc_type=disc_type,
            recompute_ll=recompute_ll,
            mean_estimate_raw=mean_estimate_raw,
            volume_mask=volume_mask,
            pcg_maxiter=pcg_maxiter,
            W_prev_real=_W_prev_real if iter_i > 0 else None,
            contrast_mode=_contrast_mode,
            contrast_grid=jnp.array(contrast_grid) if contrast_grid is not None else None,
            eigenvalues=None,  # Λ=I (z~N(0,I), W absorbs scale)
            contrast_mean=contrast_mean,
            contrast_variance=contrast_variance,
            return_mean_c=True,
        )
        posterior_info = {"mean_c": np.asarray(mean_c, dtype=np.float32)}

        # The K-internalised PCG hard solver inside EM_step_half returns W
        # in half-Fourier shape, already mask-zeroed and K-deconvolved. No
        # post-step mask projection or gridding correction needed.
        # Cache the real-space form for next iter's CG warmstart.
        W_real_for_warmstart = ftu.get_idft3_real(W.T.reshape(basis_size, *half_vs), vs)
        _W_prev_real = np.asarray(W_real_for_warmstart.reshape(basis_size, -1).T)

        # ── Optional projected-covariance refinement of the spectrum ────────
        # Every Nth iter (starting from projcov_start) we orthonormalise W → U,
        # run pca_by_projected_covariance to get a calibrated spectrum in
        # span(U), and rebake the calibration directly into W as
        # ``W ← U_refined · √projcov_s``. The next E-step then uses this
        # calibrated W and runs unconstrained.
        if (
            projcov_every > 0
            and iter_i >= projcov_start
            and (iter_i - projcov_start) % projcov_every == 0
        ):
            U_real, s_em, _ = _orthonormalize_W_to_basis(W, vs)
            q_loc = U_real.shape[0]
            vol_size = int(np.prod(vs))
            basis_fourier = (
                np.asarray(ftu.get_dft3(U_real)).reshape(q_loc, vol_size).T.astype(np.complex64)
            )
            refined_u, projcov_s = principal_components.pca_by_projected_covariance(
                reference_dataset,
                basis_fourier,
                mean_estimate_raw,
                dilated_volume_mask,
                disc_type=disc_type,
                disc_type_u=disc_type_u,
                gpu_memory_to_use=gpu_memory_to_use,
                use_mask=True,
                n_pcs_to_compute=q_loc,
            )
            logger.info(
                "  iter %d projcov: s_em=[%.2e,%.2e,..] s_pc=[%.2e,%.2e,..]",
                iter_i + 1,
                float(s_em[0]),
                float(s_em[1]) if len(s_em) > 1 else 0.0,
                float(projcov_s[0]),
                float(projcov_s[1]) if len(projcov_s) > 1 else 0.0,
            )
            # Bake the calibrated spectrum into W (W^T W = diag(projcov_s)),
            # then convert back to half-Fourier for the next E-step.
            W_full = (refined_u * np.sqrt(projcov_s)[None, :]).astype(np.complex64)
            W_full_grid = W_full.T.reshape(q_loc, *vs)
            W_half_grid = ftu.full_volume_to_half_volume(W_full_grid, vs)
            W = jnp.array(np.asarray(W_half_grid).reshape(q_loc, -1).T)

        # Recompute LL with the FINAL W (after mask + gridding) for fair comparison
        if recompute_ll:
            ll_sum = jnp.array(0.0, dtype=jnp.complex64)
            _hv = int(np.prod(ftu.volume_shape_to_half_volume_shape(vs)))
            _tsz = _tri_size(basis_size)
            ll_sum_r = jnp.array(0.0, dtype=jnp.complex64)
            mean_for_slicing = _prepare_mean_estimate_for_slicing(
                mean_estimate,
                mean_estimate_raw,
                reference_dataset.volume_shape,
                disc_type_mean,
            )
            _, _ds_list = _normalize_experiment_datasets(experiment_dataset)
            for _ds in _ds_list:
                for _bh, _cp, _rm, _tr, _bi in _iter_processed_batches_half(_ds, batch_size):
                    _nvh = _ds.noise.get_half(_bi)
                    _, _, _, _, _llb, _, _ = E_M_step_batch_half(
                        _bh,
                        jnp.zeros((_hv, _tsz), dtype=jnp.float32),
                        jnp.zeros((_hv, basis_size), dtype=jnp.complex64),
                        mean_for_slicing,
                        W,
                        _cp,
                        _rm,
                        _tr,
                        _ds.image_shape,
                        _ds.volume_shape,
                        _ds.grid_size,
                        _ds.voxel_size,
                        _nvh,
                        _ds.ctf_evaluator,
                        compute_ll=True,
                        disc_type_mean=disc_type_mean,
                        disc_type=disc_type,
                        compute_stats=False,
                    )
                    ll_sum_r += _llb
            _Wph = ftu.full_volume_to_half_volume(W_prior.T, vs).T if W_prior.shape[0] != W.shape[0] else W_prior
            neg_ll_data = float(-ll_sum_r.real)
            neg_ll_prior = float(jnp.linalg.norm(W / jnp.sqrt(_Wph + 1e-16)) ** 2)
            neg_ll_total = neg_ll_data + neg_ll_prior

        logger.info(f"Done with EM step {iter_i}")

        # Collect iteration data
        C_z = compute_Cz_from_second_moments(second_moment_zs)
        N_z = expected_zs.shape[0]
        E_mean_outer = (
            (expected_zs.T @ np.conj(expected_zs)).real / N_z
            if expected_zs.dtype in (np.complex64, np.complex128)
            else (expected_zs.T @ expected_zs) / N_z
        )
        E_mean_outer = np.asarray(E_mean_outer)
        q = W.shape[1]
        I_q = jnp.eye(q, dtype=W.dtype)
        iter_info = {
            "Iteration": iter_i,
            "Neg_LL_Total": float(neg_ll_total),
            "Neg_LL_Data": float(neg_ll_data),
            "Neg_LL_Prior": float(neg_ll_prior),
            "Expected_ZS_Mean": float(np.mean(expected_zs_mean)),
            "Expected_ZS_Var": float(np.mean(expected_zs_var)),
            "W_norm": float(jnp.linalg.norm(W)),
            "trace_Cz": float(jnp.trace(C_z)),
            "constraint_violation": float(jnp.linalg.norm(C_z - I_q)),
            "trace_E_mean_outer": float(np.trace(E_mean_outer)),
            "norm_E_mean_outer_minus_I": float(np.linalg.norm(E_mean_outer - np.eye(q))),
        }
        iteration_data.append(iter_info)

        row = f"{iter_i:>4} | {neg_ll_total:12.6e} | {neg_ll_data:12.6e} | {neg_ll_prior:12.6e} | {np.mean(expected_zs_mean):12.6e} | {np.mean(expected_zs_var):12.6e}"
        print(row)

    # Print final summary
    print("=" * 130)
    print("EM ALGORITHM COMPLETED")
    print("=" * 130)

    # Return W in its natural half-Fourier (half_vol, q) shape. Consumers that
    # need full-Fourier convert at their own boundary via
    # ftu.half_volume_to_full_volume(W.T, vol_shape).T. SVD is taken on the
    # half-Fourier W; U inherits the same shape.
    U, S, _ = jnp.linalg.svd(W, full_matrices=False)
    if return_iteration_data and return_posterior_info:
        return U, S**2, W, expected_zs, second_moment_zs, iteration_data, posterior_info
    if return_iteration_data:
        return U, S**2, W, expected_zs, second_moment_zs, iteration_data
    if return_posterior_info:
        return U, S**2, W, expected_zs, second_moment_zs, posterior_info
    return U, S**2, W, expected_zs, second_moment_zs



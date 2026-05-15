"""Noise-aware one-dimensional local polynomial Fourier regression."""

import logging
import math

import jax
import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar import utils
from recovar.core import mask
from recovar.core.forward import forward_model
from recovar.core.geometry import translate_images
from recovar.cuda_backproject import custom_cuda_requested
from recovar.heterogeneity import kernel_regression_reconstruction as kernel_recon
from recovar.reconstruction import noise as noise_mod
from recovar.reconstruction import relion_functions

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_POLY_DEGREE = 3
DEFAULT_LOCAL_POLY_BANDWIDTH_MULTIPLIERS = np.asarray(
    [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0],
    dtype=np.float32,
)
MAX_LOCAL_POLY_DEGREE = 8
_LOCAL_POLY_EPS = 1e-12
LOCAL_POLY_BASIS_OPTIONS = ("monomial", "legendre", "weighted_cholesky")
LOCAL_POLY_POL_REG_TYPES = ("none", "coeff", "deriv1", "deriv2")
DEFAULT_LOCAL_POLY_BASIS = "monomial"
DEFAULT_LOCAL_POLY_BASIS_QUANTILE = 0.995
DEFAULT_LOCAL_POLY_CHOLESKY_JITTER = 1e-6
DEFAULT_LOCAL_POLY_MOMENT_QUADRATURE = 9


def _as_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: _as_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(val) for val in value]
    return value


def _validate_local_poly_basis(basis):
    basis = str(basis)
    if basis not in LOCAL_POLY_BASIS_OPTIONS:
        raise ValueError(f"local_poly basis must be one of {LOCAL_POLY_BASIS_OPTIONS}, got {basis!r}")
    return basis


def _validate_pol_reg_type(pol_reg_type):
    pol_reg_type = str(pol_reg_type)
    if pol_reg_type not in LOCAL_POLY_POL_REG_TYPES:
        raise ValueError(
            f"local_poly polynomial regularization type must be one of {LOCAL_POLY_POL_REG_TYPES}, "
            f"got {pol_reg_type!r}"
        )
    return pol_reg_type


def _coerce_positive_1d_array(values, name):
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        out = arr
    elif arr.ndim == 2 and arr.shape[1] == 1:
        out = arr[:, 0]
    elif arr.ndim == 3 and arr.shape[1:] == (1, 1):
        out = arr[:, 0, 0]
    else:
        raise NotImplementedError(f"local_poly only supports 1D {name}; got shape {arr.shape}")
    if out.size == 0 or not np.all(np.isfinite(out)) or np.any(out <= 0):
        raise ValueError(f"{name} must contain finite positive values")
    return out.astype(np.float32, copy=False)


def coerce_1d_latent_differences(latent_differences):
    """Return latent differences as a flat 1D float32 array."""
    arr = np.asarray(latent_differences, dtype=np.float32)
    if arr.ndim == 1:
        out = arr
    elif arr.ndim == 2 and arr.shape[1] == 1:
        out = arr[:, 0]
    else:
        raise NotImplementedError(f"local_poly only supports zdim=1; got shape {arr.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError("latent_differences must be finite")
    return out.astype(np.float32, copy=False)


def coerce_1d_latent_coords(zs):
    """Return 1D latent coordinates as a flat float32 array."""
    return coerce_1d_latent_differences(zs)


def coerce_1d_latent_precision(latent_precision):
    """Return 1D latent precision as finite positive float32 values."""
    return _coerce_positive_1d_array(latent_precision, "latent_precision")


def _coerce_bandwidth_multipliers(multipliers):
    if multipliers is None:
        return DEFAULT_LOCAL_POLY_BANDWIDTH_MULTIPLIERS.copy()
    arr = np.asarray(multipliers, dtype=np.float32).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        raise ValueError(f"local_poly bandwidth multipliers must be finite positive values, got {multipliers}")
    return arr


def local_poly_bandwidth_grid_info_1d(
    latent_diff,
    latent_precision,
    n_min_particles,
    multipliers=None,
):
    """Return ``(multipliers, h_grid, sigma_ref, h_min, r_min)`` for one target."""
    latent_diff = coerce_1d_latent_differences(latent_diff).astype(np.float64)
    latent_precision = coerce_1d_latent_precision(latent_precision).astype(np.float64)
    if latent_diff.shape != latent_precision.shape:
        raise ValueError(
            "latent_diff and latent_precision must have the same flattened shape, "
            f"got {latent_diff.shape} and {latent_precision.shape}"
        )
    multipliers = _coerce_bandwidth_multipliers(multipliers)
    latent_std = np.sqrt(1.0 / latent_precision)
    sigma_ref = float(np.median(latent_std))
    if not np.isfinite(sigma_ref) or sigma_ref <= 0:
        raise ValueError(f"Invalid local_poly sigma_ref={sigma_ref}")

    n_images = latent_diff.size
    if n_images == 0:
        raise ValueError("No latent points for local_poly bandwidth selection")
    if n_min_particles is None:
        n_min_particles = 1
    closest_idx = max(0, min(int(n_min_particles), n_images) - 1)
    r_min = float(np.partition(np.abs(latent_diff), closest_idx)[closest_idx])
    h_min = max(1.25 * sigma_ref, r_min, _LOCAL_POLY_EPS)
    h_grid = h_min * multipliers.astype(np.float64)
    return multipliers, h_grid.astype(np.float32), sigma_ref, float(h_min), r_min


def local_poly_bandwidth_grid_1d(
    latent_diff,
    latent_precision,
    n_min_particles,
    multipliers=None,
):
    """Return the positive local-polynomial bandwidth grid for one target."""
    return local_poly_bandwidth_grid_info_1d(
        latent_diff,
        latent_precision,
        n_min_particles,
        multipliers=multipliers,
    )[1]


def _gaussian_raw_moments(mean, variance, max_order):
    moments = [np.ones_like(mean, dtype=np.float64)]
    if max_order == 0:
        return moments
    moments.append(mean.astype(np.float64, copy=False))
    for order in range(2, max_order + 1):
        moments.append(mean * moments[order - 1] + (order - 1) * variance * moments[order - 2])
    return moments


def gaussian_window_polynomial_moments_1d(
    latent_diff,
    latent_precision,
    h,
    degree,
    poly_scale=None,
):
    """Closed-form posterior-window moments for 1D local polynomial regression.

    Returns ``(m, M)`` with shapes ``(n_images, degree + 1)`` and
    ``(n_images, degree + 1, degree + 1)``.
    """
    latent_diff = coerce_1d_latent_differences(latent_diff).astype(np.float64)
    latent_precision = coerce_1d_latent_precision(latent_precision).astype(np.float64)
    if latent_diff.shape != latent_precision.shape:
        raise ValueError(
            "latent_diff and latent_precision must have the same flattened shape, "
            f"got {latent_diff.shape} and {latent_precision.shape}"
        )
    degree = int(degree)
    if degree < 0 or degree > MAX_LOCAL_POLY_DEGREE:
        raise ValueError(f"local_poly degree must be between 0 and {MAX_LOCAL_POLY_DEGREE}, got {degree}")
    h = float(h)
    if not np.isfinite(h) or h <= 0:
        raise ValueError(f"h must be finite and positive, got {h}")
    if poly_scale is None:
        poly_scale = h
    poly_scale = float(poly_scale)
    if not np.isfinite(poly_scale) or poly_scale <= 0:
        raise ValueError(f"poly_scale must be finite and positive, got {poly_scale}")

    variance = 1.0 / latent_precision
    h2 = h * h
    denom = h2 + variance
    alpha = h / np.sqrt(denom) * np.exp(-0.5 * latent_diff**2 / denom)
    mu = latent_diff * h2 / denom
    tau2 = variance * h2 / denom

    t_mean = mu / poly_scale
    t_var = tau2 / (poly_scale * poly_scale)
    raw_moments = _gaussian_raw_moments(t_mean, t_var, 2 * degree)
    factorials = np.asarray([math.factorial(idx) for idx in range(degree + 1)], dtype=np.float64)

    m = np.empty((latent_diff.size, degree + 1), dtype=np.float64)
    M = np.empty((latent_diff.size, degree + 1, degree + 1), dtype=np.float64)
    for r in range(degree + 1):
        m[:, r] = alpha * raw_moments[r] / factorials[r]
        for s in range(degree + 1):
            M[:, r, s] = alpha * raw_moments[r + s] / (factorials[r] * factorials[s])
    return m.astype(np.float32), M.astype(np.float32)


def hermite_quadrature_1d(n_quadrature):
    """Return nodes and normalized weights for E[f(N(0, 1))]."""
    n_quadrature = int(n_quadrature)
    if n_quadrature <= 0:
        raise ValueError(f"n_quadrature must be positive, got {n_quadrature}")
    nodes, weights = np.polynomial.hermite.hermgauss(n_quadrature)
    return nodes.astype(np.float32), (weights / np.sqrt(np.pi)).astype(np.float32)


def _posterior_window_alpha_t_quadrature(
    latent_diff,
    latent_precision,
    h,
    n_quadrature,
    poly_scale=None,
):
    latent_diff = coerce_1d_latent_differences(latent_diff).astype(np.float64)
    latent_precision = coerce_1d_latent_precision(latent_precision).astype(np.float64)
    if latent_diff.shape != latent_precision.shape:
        raise ValueError(
            "latent_diff and latent_precision must have the same flattened shape, "
            f"got {latent_diff.shape} and {latent_precision.shape}"
        )
    h = float(h)
    if not np.isfinite(h) or h <= 0:
        raise ValueError(f"h must be finite and positive, got {h}")
    if poly_scale is None:
        poly_scale = h
    poly_scale = float(poly_scale)
    if not np.isfinite(poly_scale) or poly_scale <= 0:
        raise ValueError(f"poly_scale must be finite and positive, got {poly_scale}")

    variance = 1.0 / latent_precision
    h2 = h * h
    denom = h2 + variance
    alpha = h / np.sqrt(denom) * np.exp(-0.5 * latent_diff**2 / denom)
    mu = latent_diff * h2 / denom
    tau2 = variance * h2 / denom

    nodes, quad_weights = hermite_quadrature_1d(n_quadrature)
    xdiff = mu[:, None] + np.sqrt(np.maximum(2.0 * tau2, 0.0))[:, None] * nodes[None, :]
    t = xdiff / poly_scale
    return alpha.astype(np.float32), t.astype(np.float32), quad_weights


def _monomial_feature_stack(t, degree):
    factorials = np.asarray([math.factorial(idx) for idx in range(degree + 1)], dtype=np.float64)
    return np.stack([t**r / factorials[r] for r in range(degree + 1)], axis=-1).astype(np.float32)


def _monomial_derivative_stack(t, degree, derivative_order):
    t = np.asarray(t, dtype=np.float64)
    derivative_order = int(derivative_order)
    out = []
    for r in range(degree + 1):
        if r < derivative_order:
            out.append(np.zeros_like(t, dtype=np.float64))
        else:
            out.append(t ** (r - derivative_order) / math.factorial(r - derivative_order))
    return np.stack(out, axis=-1).astype(np.float32)


def _legendre_feature_stack(t, degree, scale, derivative_order=0):
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Legendre basis scale must be finite and positive, got {scale}")
    u = np.asarray(t, dtype=np.float64) / scale
    out = []
    for r in range(degree + 1):
        coeff = np.zeros(r + 1, dtype=np.float64)
        coeff[-1] = 1.0
        if derivative_order:
            coeff = np.polynomial.legendre.legder(coeff, m=derivative_order)
            if coeff.size == 0:
                out.append(np.zeros_like(u, dtype=np.float64))
                continue
        values = np.polynomial.legendre.legval(u, coeff)
        if derivative_order:
            values = values / (scale**derivative_order)
        out.append(values)
    return np.stack(out, axis=-1).astype(np.float32)


def _weighted_quantile(values, weights, quantile):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.shape != weights.shape:
        raise ValueError(f"values and weights must have matching shapes, got {values.shape} and {weights.shape}")
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(finite):
        return float(np.nanmax(np.abs(values))) if values.size else 1.0
    values = values[finite]
    weights = weights[finite]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    total = float(cdf[-1])
    if total <= 0:
        return float(values[-1])
    q = float(np.clip(quantile, 0.0, 1.0))
    return float(values[min(np.searchsorted(cdf, q * total, side="left"), values.size - 1)])


def _basis_features_from_spec(t, degree, basis_spec, derivative_order=0):
    basis = basis_spec["basis"]
    if basis == "monomial":
        return _monomial_derivative_stack(t, degree, derivative_order) if derivative_order else _monomial_feature_stack(t, degree)
    if basis == "legendre":
        return _legendre_feature_stack(t, degree, basis_spec["basis_scale"], derivative_order=derivative_order)
    if basis == "weighted_cholesky":
        raw = (
            _monomial_derivative_stack(t, degree, derivative_order)
            if derivative_order
            else _monomial_feature_stack(t, degree)
        )
        return np.einsum("...r,rs->...s", raw, basis_spec["basis_transform"], optimize=True).astype(np.float32)
    raise ValueError(f"Unknown local_poly basis {basis!r}")


def _basis_gram_condition(features, weights, denom):
    gram = np.einsum("bq,bqr,bqs->rs", weights, features, features, optimize=True) / max(float(denom), _LOCAL_POLY_EPS)
    gram = 0.5 * (gram + gram.T)
    return gram.astype(np.float64), float(np.linalg.cond(gram))


def local_polynomial_basis_spec_1d(
    latent_diff,
    latent_precision,
    h,
    degree,
    *,
    n_quadrature=DEFAULT_LOCAL_POLY_MOMENT_QUADRATURE,
    basis=DEFAULT_LOCAL_POLY_BASIS,
    poly_scale=None,
    basis_quantile=DEFAULT_LOCAL_POLY_BASIS_QUANTILE,
    cholesky_jitter=DEFAULT_LOCAL_POLY_CHOLESKY_JITTER,
):
    """Build a basis specification for one local polynomial bandwidth."""
    degree = int(degree)
    if degree < 0 or degree > MAX_LOCAL_POLY_DEGREE:
        raise ValueError(f"local_poly degree must be between 0 and {MAX_LOCAL_POLY_DEGREE}, got {degree}")
    basis = _validate_local_poly_basis(basis)
    alpha, t, quad_weights = _posterior_window_alpha_t_quadrature(
        latent_diff,
        latent_precision,
        h,
        n_quadrature,
        poly_scale=poly_scale,
    )
    weights = alpha[:, None].astype(np.float64) * quad_weights[None, :].astype(np.float64)
    denom = float(np.sum(alpha))
    n_features = degree + 1
    raw_features = _monomial_feature_stack(t, degree)
    raw_gram, raw_cond = _basis_gram_condition(raw_features, weights, denom)

    basis_scale = 1.0
    basis_transform = np.eye(n_features, dtype=np.float64)
    cholesky_jitter_used = 0.0
    if basis == "legendre":
        basis_scale = max(
            _weighted_quantile(np.abs(t), weights, basis_quantile),
            _LOCAL_POLY_EPS,
        )
    elif basis == "weighted_cholesky":
        trace_scale = float(np.trace(raw_gram)) / max(n_features, 1)
        if not np.isfinite(trace_scale) or trace_scale <= 0:
            trace_scale = 1.0
        cholesky_jitter_used = float(cholesky_jitter) * trace_scale
        eye = np.eye(n_features, dtype=np.float64)
        for attempt in range(8):
            try:
                lower = np.linalg.cholesky(raw_gram + cholesky_jitter_used * (10.0**attempt) * eye)
                cholesky_jitter_used *= 10.0**attempt
                break
            except np.linalg.LinAlgError:
                lower = None
        if lower is None:
            raise np.linalg.LinAlgError("Could not Cholesky-factor local polynomial weighted Gram matrix")
        upper = lower.T
        basis_transform = np.linalg.inv(upper)

    features = _basis_features_from_spec(
        t,
        degree,
        {
            "basis": basis,
            "basis_scale": basis_scale,
            "basis_transform": basis_transform,
        },
    )
    basis_gram, basis_cond = _basis_gram_condition(features, weights, denom)
    deriv1 = _basis_features_from_spec(
        t,
        degree,
        {
            "basis": basis,
            "basis_scale": basis_scale,
            "basis_transform": basis_transform,
        },
        derivative_order=1,
    )
    deriv2 = _basis_features_from_spec(
        t,
        degree,
        {
            "basis": basis,
            "basis_scale": basis_scale,
            "basis_transform": basis_transform,
        },
        derivative_order=2,
    )
    deriv1_gram, deriv1_cond = _basis_gram_condition(deriv1, weights, denom)
    deriv2_gram, deriv2_cond = _basis_gram_condition(deriv2, weights, denom)
    t0 = np.zeros((1,), dtype=np.float32)
    target_eval = _basis_features_from_spec(
        t0,
        degree,
        {
            "basis": basis,
            "basis_scale": basis_scale,
            "basis_transform": basis_transform,
        },
    )[0]
    return {
        "basis": basis,
        "degree": int(degree),
        "h": float(h),
        "poly_scale": float(h if poly_scale is None else poly_scale),
        "basis_scale": float(basis_scale),
        "basis_transform": basis_transform.astype(np.float32),
        "target_eval": target_eval.astype(np.float32),
        "basis_gram": basis_gram.astype(np.float32),
        "deriv1_gram": deriv1_gram.astype(np.float32),
        "deriv2_gram": deriv2_gram.astype(np.float32),
        "basis_info": {
            "basis": basis,
            "h": float(h),
            "poly_scale": float(h if poly_scale is None else poly_scale),
            "basis_scale": float(basis_scale),
            "basis_quantile": float(basis_quantile),
            "cholesky_jitter_requested": float(cholesky_jitter),
            "cholesky_jitter_used": float(cholesky_jitter_used),
            "raw_gram_condition": float(raw_cond),
            "basis_gram_condition": float(basis_cond),
            "deriv1_gram_condition": float(deriv1_cond),
            "deriv2_gram_condition": float(deriv2_cond),
            "target_eval": target_eval.astype(np.float64).tolist(),
        },
    }


def local_polynomial_basis_specs_1d(
    latent_diff,
    latent_precision,
    h_grid,
    degree,
    *,
    n_quadrature=DEFAULT_LOCAL_POLY_MOMENT_QUADRATURE,
    basis=DEFAULT_LOCAL_POLY_BASIS,
    basis_quantile=DEFAULT_LOCAL_POLY_BASIS_QUANTILE,
    cholesky_jitter=DEFAULT_LOCAL_POLY_CHOLESKY_JITTER,
):
    return [
        local_polynomial_basis_spec_1d(
            latent_diff,
            latent_precision,
            float(h),
            degree,
            n_quadrature=n_quadrature,
            basis=basis,
            poly_scale=float(h),
            basis_quantile=basis_quantile,
            cholesky_jitter=cholesky_jitter,
        )
        for h in np.asarray(h_grid, dtype=np.float32).reshape(-1)
    ]


def gaussian_window_polynomial_quadrature_1d(
    latent_diff,
    latent_precision,
    h,
    degree,
    n_quadrature,
    poly_scale=None,
    basis_spec=None,
):
    """Quadrature features for the product of the latent posterior and window.

    Returns ``(alpha, phi, quad_weights)`` where ``alpha`` is the scalar product
    normalizer, ``phi`` has shape ``(n_images, n_quadrature, degree + 1)``, and
    ``quad_weights`` are normalized Gauss-Hermite weights.
    """
    degree = int(degree)
    if degree < 0 or degree > MAX_LOCAL_POLY_DEGREE:
        raise ValueError(f"local_poly degree must be between 0 and {MAX_LOCAL_POLY_DEGREE}, got {degree}")
    alpha, t, quad_weights = _posterior_window_alpha_t_quadrature(
        latent_diff,
        latent_precision,
        h,
        n_quadrature,
        poly_scale=poly_scale,
    )
    if basis_spec is None:
        phi = _monomial_feature_stack(t, degree)
    else:
        phi = _basis_features_from_spec(t, degree, basis_spec)
    return alpha.astype(np.float32), phi.astype(np.float32), quad_weights


def local_polynomial_regularization_matrix(
    basis_spec,
    *,
    pol_reg_type="none",
    pol_reg_eta=0.0,
    pol_reg_power=2.0,
):
    pol_reg_type = _validate_pol_reg_type(pol_reg_type)
    eta = float(pol_reg_eta)
    if not np.isfinite(eta) or eta < 0:
        raise ValueError(f"local_poly polynomial regularization eta must be finite and nonnegative, got {eta}")
    power = float(pol_reg_power)
    if not np.isfinite(power):
        raise ValueError(f"local_poly polynomial regularization power must be finite, got {power}")
    n_features = int(basis_spec["degree"]) + 1
    if pol_reg_type == "none" or eta == 0:
        return np.eye(n_features, dtype=np.float32)
    if pol_reg_type == "coeff":
        orders = np.arange(n_features, dtype=np.float32)
        diag = np.ones(n_features, dtype=np.float32)
        diag[1:] = 1.0 + eta * np.power(orders[1:], power)
        return np.diag(diag).astype(np.float32)
    if pol_reg_type == "deriv1":
        return (np.eye(n_features, dtype=np.float32) + eta * np.asarray(basis_spec["deriv1_gram"], dtype=np.float32))
    if pol_reg_type == "deriv2":
        return (np.eye(n_features, dtype=np.float32) + eta * np.asarray(basis_spec["deriv2_gram"], dtype=np.float32))
    raise ValueError(f"Unknown polynomial regularization type {pol_reg_type!r}")


def local_polynomial_regularization_matrices(
    basis_specs,
    *,
    pol_reg_type="none",
    pol_reg_eta=0.0,
    pol_reg_power=2.0,
):
    return np.stack(
        [
            local_polynomial_regularization_matrix(
                spec,
                pol_reg_type=pol_reg_type,
                pol_reg_eta=pol_reg_eta,
                pol_reg_power=pol_reg_power,
            )
            for spec in basis_specs
        ],
        axis=0,
    ).astype(np.float32)


def evaluate_local_polynomial_target_coefficients(theta_coeffs, target_eval_all=None):
    """Evaluate Fourier coefficients at the target latent coordinate."""
    theta_coeffs = np.asarray(theta_coeffs)
    if theta_coeffs.ndim != 3:
        raise ValueError(
            "theta_coeffs must have shape (n_bandwidths, degree+1, half_volume_size); "
            f"got {theta_coeffs.shape}"
        )
    if target_eval_all is None:
        return theta_coeffs[:, 0]
    target_eval_all = np.asarray(target_eval_all, dtype=theta_coeffs.real.dtype)
    if target_eval_all.ndim == 1:
        target_eval_all = np.broadcast_to(target_eval_all[None, :], theta_coeffs.shape[:2])
    if target_eval_all.shape != theta_coeffs.shape[:2]:
        raise ValueError(f"target_eval_all shape {target_eval_all.shape} does not match {theta_coeffs.shape[:2]}")
    return np.einsum("br,brv->bv", target_eval_all, theta_coeffs, optimize=True)


def _expand_tilt_latent_array_to_images(experiment_dataset, values, name):
    values = np.asarray(values)
    if values.shape[0] == experiment_dataset.n_images:
        return values
    if (
        getattr(experiment_dataset, "tilt_series_flag", False)
        and hasattr(experiment_dataset, "tilt_particles")
        and values.shape[0] != experiment_dataset.n_images
    ):
        per_image = np.empty(experiment_dataset.n_images, dtype=values.dtype)
        for p_idx, tilt_inds in enumerate(experiment_dataset.tilt_particles):
            per_image[tilt_inds] = values[p_idx]
        return per_image
    raise ValueError(f"{name} length {values.shape[0]} does not match dataset n_images={experiment_dataset.n_images}")


def _auto_local_poly_bandwidth_batch_size(n_bandwidths, degree, half_volume_size, complex_dtype, real_dtype):
    n_features = int(degree) + 1
    per_bandwidth_gb = (
        half_volume_size
        * (
            n_features * np.dtype(complex_dtype).itemsize
            + n_features * n_features * np.dtype(real_dtype).itemsize
        )
        / (1024**3)
    )
    if per_bandwidth_gb <= 0:
        return 1
    target_gb = max(1.0, 0.20 * float(utils.get_gpu_memory_total()))
    return max(1, min(int(n_bandwidths), 8, int(target_gb / per_bandwidth_gb)))


def _local_poly_batch_size(experiment_dataset, lhs_all, rhs_all, half_volume_size):
    accum_gb = utils.get_size_in_gb(rhs_all) + utils.get_size_in_gb(lhs_all)
    avail_gb = kernel_recon._effective_heterogeneity_memory_budget(max(1.0, utils.get_gpu_memory_total() - accum_gb))
    batch_size = int(utils.get_image_batch_size(experiment_dataset.grid_size, avail_gb))
    if custom_cuda_requested() and jax.default_backend() == "gpu":
        bytes_per_image = half_volume_size * (
            np.dtype(experiment_dataset.dtype).itemsize + np.dtype(experiment_dataset.dtype_real).itemsize
        )
        target_bytes = utils.get_gpu_memory_total() * (1024**3) * 0.45
        memory_limited = max(1, int(target_bytes // max(1, bytes_per_image)))
        batch_size = max(1, min(batch_size, memory_limited, 256))
    return batch_size


def _local_poly_weight_sets(
    latent_diff,
    latent_precision,
    h_group,
    degree,
    *,
    basis_specs=None,
    basis_spec_offset=0,
    moment_quadrature=DEFAULT_LOCAL_POLY_MOMENT_QUADRATURE,
):
    n_features = int(degree) + 1
    rhs_rows = []
    lhs_rows = []
    for local_idx, h in enumerate(h_group):
        basis_spec = None if basis_specs is None else basis_specs[basis_spec_offset + local_idx]
        if basis_spec is None or basis_spec["basis"] == "monomial":
            m, M = gaussian_window_polynomial_moments_1d(
                latent_diff,
                latent_precision,
                h=float(h),
                degree=degree,
                poly_scale=float(h),
            )
        else:
            alpha, phi, quad_weights = gaussian_window_polynomial_quadrature_1d(
                latent_diff,
                latent_precision,
                h=float(h),
                degree=degree,
                n_quadrature=moment_quadrature,
                poly_scale=float(h),
                basis_spec=basis_spec,
            )
            weighted = alpha[:, None] * quad_weights[None, :]
            m = np.einsum("bq,bqr->br", weighted, phi, optimize=True)
            M = np.einsum("bq,bqr,bqs->brs", weighted, phi, phi, optimize=True)
        rhs_rows.extend([m[:, r] for r in range(n_features)])
        lhs_rows.extend([M[:, r, s] for r in range(n_features) for s in range(n_features)])
    return np.asarray(rhs_rows, dtype=np.float32), np.asarray(lhs_rows, dtype=np.float32)


def _local_poly_upsampled_shape_and_valid_half(experiment_dataset, upsampling_factor, half_volume_size):
    vol_upsample = kernel_recon._postprocess_upsampling_factor(upsampling_factor)
    upsampled_volume_shape = tuple(3 * [experiment_dataset.volume_shape[0] * vol_upsample])
    expected_half_size = int(
        np.prod(fourier_transform_utils.volume_shape_to_half_volume_shape(upsampled_volume_shape))
    )
    if half_volume_size != expected_half_size:
        raise ValueError(f"half_volume_size {half_volume_size} does not match expected {expected_half_size}")

    valid_full = (
        mask.get_radial_mask(upsampled_volume_shape, radius=upsampled_volume_shape[0] // 2 - 1)
        .reshape(-1)
        .astype(np.float32)
    )
    valid_half = np.asarray(
        fourier_transform_utils.full_volume_to_half_volume(jnp.asarray(valid_full), upsampled_volume_shape)
    ).reshape(-1).real.astype(np.float32)
    return vol_upsample, upsampled_volume_shape, valid_half


def _sample_normal_equation_condition(lhs, rho, pol_reg_matrix, max_samples=4096):
    half_volume_size = int(lhs.shape[-1])
    if half_volume_size == 0:
        return {
            "normal_eq_condition_median": np.nan,
            "normal_eq_condition_p95": np.nan,
            "normal_eq_condition_max": np.nan,
            "normal_eq_condition_sample_count": 0,
        }
    max_samples = int(max(1, max_samples))
    if half_volume_size <= max_samples:
        indices = np.arange(half_volume_size, dtype=np.int64)
    else:
        indices = np.linspace(0, half_volume_size - 1, max_samples, dtype=np.int64)
    gram = np.moveaxis(lhs[:, :, indices], -1, 0).astype(np.float64, copy=True)
    gram = 0.5 * (gram + np.swapaxes(gram, 1, 2))
    gram += rho[indices, None, None].astype(np.float64) * np.asarray(pol_reg_matrix, dtype=np.float64)[None, :, :]
    cond = np.linalg.cond(gram)
    cond = cond[np.isfinite(cond)]
    if cond.size == 0:
        return {
            "normal_eq_condition_median": np.inf,
            "normal_eq_condition_p95": np.inf,
            "normal_eq_condition_max": np.inf,
            "normal_eq_condition_sample_count": int(indices.size),
        }
    return {
        "normal_eq_condition_median": float(np.median(cond)),
        "normal_eq_condition_p95": float(np.percentile(cond, 95)),
        "normal_eq_condition_max": float(np.max(cond)),
        "normal_eq_condition_sample_count": int(indices.size),
    }


def solve_local_polynomial_fourier_coefficients(
    lhs_all,
    rhs_all,
    experiment_dataset,
    *,
    tau=None,
    pol_reg_matrices=None,
    upsampling_factor=None,
    solve_chunk_size=262144,
    return_diagnostics=False,
):
    """Solve per-voxel polynomial normal equations and return all theta_r."""
    lhs_all = np.asarray(lhs_all, dtype=np.float32)
    rhs_all = np.asarray(rhs_all)
    if lhs_all.ndim != 4 or rhs_all.ndim != 3:
        raise ValueError(
            "lhs_all must have shape (n_bandwidths, degree+1, degree+1, half_volume_size) "
            f"and rhs_all (n_bandwidths, degree+1, half_volume_size); got {lhs_all.shape}, {rhs_all.shape}"
        )
    if (
        lhs_all.shape[0] != rhs_all.shape[0]
        or lhs_all.shape[1] != lhs_all.shape[2]
        or lhs_all.shape[1] != rhs_all.shape[1]
    ):
        raise ValueError(f"Incompatible local_poly lhs/rhs shapes: {lhs_all.shape} and {rhs_all.shape}")

    n_bandwidths, n_features, _, half_volume_size = lhs_all.shape
    if pol_reg_matrices is None:
        pol_reg_matrices = np.broadcast_to(
            np.eye(n_features, dtype=np.float32)[None, :, :],
            (n_bandwidths, n_features, n_features),
        )
    pol_reg_matrices = np.asarray(pol_reg_matrices, dtype=np.float32)
    if pol_reg_matrices.shape != (n_bandwidths, n_features, n_features):
        raise ValueError(
            "pol_reg_matrices must have shape (n_bandwidths, degree+1, degree+1), "
            f"got {pol_reg_matrices.shape}"
        )
    vol_upsample, upsampled_volume_shape, valid_half = _local_poly_upsampled_shape_and_valid_half(
        experiment_dataset,
        upsampling_factor,
        half_volume_size,
    )

    coeffs = np.zeros((n_bandwidths, n_features, half_volume_size), dtype=rhs_all.dtype)
    diagnostics = []
    solve_chunk_size = int(max(1, solve_chunk_size))
    for bw_idx in range(n_bandwidths):
        lhs = lhs_all[bw_idx]
        rhs = rhs_all[bw_idx]
        pol_reg_matrix = pol_reg_matrices[bw_idx]
        reg_filter = np.asarray(
            relion_functions.adjust_regularization_relion_style(
                jnp.asarray(lhs[0, 0]),
                upsampled_volume_shape,
                tau=None if tau is None else jnp.asarray(tau),
                padding_factor=vol_upsample,
                max_res_shell=None,
                half_volume=True,
            )
        ).reshape(-1).astype(np.float32)
        rho = np.maximum(reg_filter - lhs[0, 0], 0.0).astype(np.float32)
        if return_diagnostics:
            diagnostics.append(
                {
                    "bandwidth_index": int(bw_idx),
                    **_sample_normal_equation_condition(lhs, rho, pol_reg_matrix),
                }
            )
        for start in range(0, half_volume_size, solve_chunk_size):
            stop = min(start + solve_chunk_size, half_volume_size)
            gram = np.moveaxis(lhs[:, :, start:stop], -1, 0).astype(np.float32, copy=True)
            gram = 0.5 * (gram + np.swapaxes(gram, 1, 2))
            gram += rho[start:stop, None, None] * pol_reg_matrix[None, :, :]
            rhs_chunk = np.moveaxis(rhs[:, start:stop], -1, 0)
            theta = np.linalg.solve(gram, rhs_chunk[..., None])[..., 0]
            coeffs[bw_idx, :, start:stop] = np.moveaxis(theta, 0, -1)
        coeffs[bw_idx] *= valid_half.astype(coeffs.real.dtype)[None, :]
    if return_diagnostics:
        return coeffs, diagnostics
    return coeffs


def solve_local_polynomial_fourier_system(
    lhs_all,
    rhs_all,
    experiment_dataset,
    *,
    tau=None,
    target_eval_all=None,
    pol_reg_matrices=None,
    grid_correct=True,
    disc_type="linear_interp",
    use_spherical_mask=True,
    upsampling_factor=None,
    return_real_space=False,
    solve_chunk_size=262144,
    return_diagnostics=False,
):
    """Solve per-voxel polynomial normal equations and post-process target estimate."""
    lhs_all = np.asarray(lhs_all, dtype=np.float32)
    rhs_all = np.asarray(rhs_all)
    if lhs_all.ndim != 4 or rhs_all.ndim != 3:
        raise ValueError(
            "lhs_all must have shape (n_bandwidths, degree+1, degree+1, half_volume_size) "
            f"and rhs_all (n_bandwidths, degree+1, half_volume_size); got {lhs_all.shape}, {rhs_all.shape}"
        )
    n_bandwidths, _, _, half_volume_size = lhs_all.shape
    kernel_type = "triangular" if disc_type == "linear_interp" else "square"
    vol_upsample, _, _ = _local_poly_upsampled_shape_and_valid_half(
        experiment_dataset,
        upsampling_factor,
        half_volume_size,
    )
    solved = solve_local_polynomial_fourier_coefficients(
        lhs_all,
        rhs_all,
        experiment_dataset,
        tau=tau,
        pol_reg_matrices=pol_reg_matrices,
        upsampling_factor=upsampling_factor,
        solve_chunk_size=solve_chunk_size,
        return_diagnostics=return_diagnostics,
    )
    if return_diagnostics:
        coeffs, diagnostics = solved
    else:
        coeffs = solved
        diagnostics = None
    target_coeffs = evaluate_local_polynomial_target_coefficients(coeffs, target_eval_all=target_eval_all)
    estimates = []
    for bw_idx in range(n_bandwidths):
        estimates.append(
            relion_functions.post_process_predivided_fourier_volume(
                jnp.asarray(target_coeffs[bw_idx]),
                experiment_dataset.volume_shape,
                vol_upsample,
                kernel=kernel_type,
                use_spherical_mask=use_spherical_mask,
                grid_correct=grid_correct,
                gridding_correct="square",
                kernel_width=1,
                return_real_space=return_real_space,
                input_half_volume=True,
            ).reshape(-1)
        )
    estimates = np.asarray(jnp.stack(estimates, axis=0))
    if return_diagnostics:
        return estimates, diagnostics
    return estimates


def postprocess_local_polynomial_fourier_coefficients(
    theta_coeffs,
    experiment_dataset,
    *,
    target_eval_all=None,
    grid_correct=True,
    disc_type="linear_interp",
    use_spherical_mask=True,
    upsampling_factor=None,
    return_real_space=False,
):
    """Post-process already solved local-polynomial coefficients using theta_0."""
    theta_coeffs = np.asarray(theta_coeffs)
    if theta_coeffs.ndim != 3:
        raise ValueError(
            "theta_coeffs must have shape (n_bandwidths, degree+1, half_volume_size); "
            f"got {theta_coeffs.shape}"
        )
    half_volume_size = int(theta_coeffs.shape[-1])
    kernel_type = "triangular" if disc_type == "linear_interp" else "square"
    vol_upsample, _, _ = _local_poly_upsampled_shape_and_valid_half(
        experiment_dataset,
        upsampling_factor,
        half_volume_size,
    )
    target_coeffs = evaluate_local_polynomial_target_coefficients(theta_coeffs, target_eval_all=target_eval_all)
    estimates = []
    for bw_idx in range(theta_coeffs.shape[0]):
        estimates.append(
            relion_functions.post_process_predivided_fourier_volume(
                jnp.asarray(target_coeffs[bw_idx]),
                experiment_dataset.volume_shape,
                vol_upsample,
                kernel=kernel_type,
                use_spherical_mask=use_spherical_mask,
                grid_correct=grid_correct,
                gridding_correct="square",
                kernel_width=1,
                return_real_space=return_real_space,
                input_half_volume=True,
            ).reshape(-1)
        )
    return np.asarray(jnp.stack(estimates, axis=0))


def estimate_local_polynomial_volumes(
    experiment_dataset,
    latent_differences,
    latent_precision,
    h_grid,
    *,
    degree=DEFAULT_LOCAL_POLY_DEGREE,
    batch_size=None,
    tau=None,
    grid_correct=True,
    disc_type="linear_interp",
    use_spherical_mask=True,
    return_lhs_rhs=False,
    upsampling_factor=None,
    return_real_space=False,
    use_fast_rfft=False,
    bandwidth_batch_size=None,
    basis=DEFAULT_LOCAL_POLY_BASIS,
    basis_quantile=DEFAULT_LOCAL_POLY_BASIS_QUANTILE,
    cholesky_jitter=DEFAULT_LOCAL_POLY_CHOLESKY_JITTER,
    moment_quadrature=DEFAULT_LOCAL_POLY_MOMENT_QUADRATURE,
    pol_reg_type="none",
    pol_reg_eta=0.0,
    pol_reg_power=2.0,
    return_diagnostics=False,
):
    """Estimate local-polynomial candidate volumes for one halfset."""
    latent_differences = coerce_1d_latent_differences(latent_differences)
    latent_precision = coerce_1d_latent_precision(latent_precision)
    latent_differences = _expand_tilt_latent_array_to_images(
        experiment_dataset, latent_differences, "latent_differences"
    )
    latent_precision = _expand_tilt_latent_array_to_images(experiment_dataset, latent_precision, "latent_precision")
    if latent_differences.shape != latent_precision.shape:
        raise ValueError(
            "latent_differences and latent_precision must have the same flattened shape, "
            f"got {latent_differences.shape} and {latent_precision.shape}"
        )
    degree = int(degree)
    if degree < 0 or degree > MAX_LOCAL_POLY_DEGREE:
        raise ValueError(f"local_poly degree must be between 0 and {MAX_LOCAL_POLY_DEGREE}, got {degree}")
    h_grid = np.asarray(h_grid, dtype=np.float32).reshape(-1)
    if h_grid.size == 0 or not np.all(np.isfinite(h_grid)) or np.any(h_grid <= 0):
        raise ValueError(f"h_grid must contain finite positive values, got {h_grid}")
    basis = _validate_local_poly_basis(basis)
    pol_reg_type = _validate_pol_reg_type(pol_reg_type)

    n_bandwidths = h_grid.size
    n_features = degree + 1
    half_volume_size = kernel_recon._candidate_half_volume_size(experiment_dataset, upsampling_factor)
    rhs_all = np.zeros((n_bandwidths, n_features, half_volume_size), dtype=experiment_dataset.dtype)
    lhs_all = np.zeros((n_bandwidths, n_features, n_features, half_volume_size), dtype=experiment_dataset.dtype_real)
    basis_specs = local_polynomial_basis_specs_1d(
        latent_differences,
        latent_precision,
        h_grid,
        degree,
        n_quadrature=moment_quadrature,
        basis=basis,
        basis_quantile=basis_quantile,
        cholesky_jitter=cholesky_jitter,
    )
    target_eval_all = np.stack([spec["target_eval"] for spec in basis_specs], axis=0).astype(np.float32)
    pol_reg_matrices = local_polynomial_regularization_matrices(
        basis_specs,
        pol_reg_type=pol_reg_type,
        pol_reg_eta=pol_reg_eta,
        pol_reg_power=pol_reg_power,
    )

    if bandwidth_batch_size is None:
        bandwidth_batch_size = _auto_local_poly_bandwidth_batch_size(
            n_bandwidths,
            degree,
            half_volume_size,
            experiment_dataset.dtype,
            experiment_dataset.dtype_real,
        )
    bandwidth_batch_size = int(max(1, min(n_bandwidths, bandwidth_batch_size)))
    if batch_size is None:
        batch_size = _local_poly_batch_size(experiment_dataset, lhs_all, rhs_all, half_volume_size)

    logger.info("batch size in local_poly heterogeneity kernel: %s", batch_size)
    logger.info("bandwidth batch size in local_poly heterogeneity kernel: %s", bandwidth_batch_size)
    logger.info(
        "local_poly degree=%s basis=%s pol_reg_type=%s pol_reg_eta=%s h_grid=%s",
        degree,
        basis,
        pol_reg_type,
        pol_reg_eta,
        h_grid,
    )

    config = kernel_recon._reconstruction_config(experiment_dataset, disc_type, upsampling_factor)
    n_rhs_sets = n_features
    n_lhs_sets = n_features * n_features
    for h_start in range(0, n_bandwidths, bandwidth_batch_size):
        h_stop = min(h_start + bandwidth_batch_size, n_bandwidths)
        h_group = h_grid[h_start:h_stop]
        n_h_group = h_group.size
        n_weight_sets = n_h_group * (n_rhs_sets + n_lhs_sets)
        Ft_y_acc = jnp.zeros((n_h_group * n_rhs_sets, half_volume_size), dtype=experiment_dataset.dtype)
        Ft_ctf_acc = jnp.zeros((n_h_group * n_lhs_sets, half_volume_size), dtype=experiment_dataset.dtype_real)
        raw_batches = experiment_dataset.iter_batches(
            batch_size,
            noise_model=experiment_dataset.noise,
            noise_half=False,
        )
        for (
            raw_images,
            rotation_matrices,
            translations,
            ctf_params,
            noise_variance,
            _particle_indices,
            image_indices,
        ) in raw_batches:
            image_indices = np.asarray(image_indices, dtype=np.int32)
            rhs_weights, lhs_weights = _local_poly_weight_sets(
                latent_differences[image_indices],
                latent_precision[image_indices],
                h_group,
                degree,
                basis_specs=basis_specs,
                basis_spec_offset=h_start,
                moment_quadrature=moment_quadrature,
            )
            current_batch_size, images, rotation_matrices, translations, ctf_params, noise_variance = (
                kernel_recon._prepare_half_image_batch(
                    experiment_dataset,
                    raw_images,
                    rotation_matrices,
                    translations,
                    ctf_params,
                    noise_variance,
                    batch_size=batch_size,
                    use_fast_rfft=use_fast_rfft,
                )
            )
            image_weights = np.concatenate([rhs_weights, lhs_weights], axis=0)
            if image_weights.shape[0] != n_weight_sets:
                raise RuntimeError(
                    f"Unexpected local_poly weight-set count {image_weights.shape[0]} != {n_weight_sets}"
                )
            image_weights = kernel_recon._pad_image_weight_matrix_for_fixed_batch(
                image_weights,
                current_batch_size=current_batch_size,
                target_batch_size=batch_size,
            )
            Ft_all_y, Ft_all_ctf = kernel_recon.backproject_weight_sets_from_fft(
                config,
                images,
                ctf_params,
                rotation_matrices,
                translations,
                noise_variance,
                image_weights,
                Ft_y=jnp.concatenate([Ft_y_acc, jnp.zeros_like(Ft_ctf_acc, dtype=Ft_y_acc.dtype)], axis=0),
                Ft_ctf=jnp.concatenate([jnp.zeros_like(Ft_y_acc, dtype=Ft_ctf_acc.dtype), Ft_ctf_acc], axis=0),
            )
            Ft_y_acc = Ft_all_y[: n_h_group * n_rhs_sets]
            Ft_ctf_acc = Ft_all_ctf[n_h_group * n_rhs_sets :]

        rhs_all[h_start:h_stop] = np.asarray(Ft_y_acc).reshape(n_h_group, n_features, half_volume_size)
        lhs_all[h_start:h_stop] = np.asarray(Ft_ctf_acc).reshape(
            n_h_group,
            n_features,
            n_features,
            half_volume_size,
        )

    solved = solve_local_polynomial_fourier_system(
        lhs_all,
        rhs_all,
        experiment_dataset,
        tau=tau,
        target_eval_all=target_eval_all,
        pol_reg_matrices=pol_reg_matrices,
        grid_correct=grid_correct,
        disc_type=disc_type,
        use_spherical_mask=use_spherical_mask,
        upsampling_factor=upsampling_factor,
        return_real_space=return_real_space,
        return_diagnostics=return_diagnostics,
    )
    if return_diagnostics:
        estimates, solve_diagnostics = solved
    else:
        estimates = solved
        solve_diagnostics = None
    if return_lhs_rhs:
        outputs = [estimates, np.asarray(lhs_all), np.asarray(rhs_all)]
        if return_diagnostics:
            outputs.append(
                {
                    "basis": basis,
                    "basis_info": [_as_jsonable(spec["basis_info"]) for spec in basis_specs],
                    "pol_reg_type": pol_reg_type,
                    "pol_reg_eta": float(pol_reg_eta),
                    "pol_reg_power": float(pol_reg_power),
                    "target_eval": target_eval_all.tolist(),
                    "solve_diagnostics": solve_diagnostics,
                }
            )
        return tuple(outputs)
    if return_diagnostics:
        return estimates, {
            "basis": basis,
            "basis_info": [_as_jsonable(spec["basis_info"]) for spec in basis_specs],
            "pol_reg_type": pol_reg_type,
            "pol_reg_eta": float(pol_reg_eta),
            "pol_reg_power": float(pol_reg_power),
            "target_eval": target_eval_all.tolist(),
            "solve_diagnostics": solve_diagnostics,
        }
    return estimates


def _pad_quadrature_features_for_fixed_batch(phi, current_batch_size, target_batch_size):
    phi = np.asarray(phi, dtype=np.float32)
    if phi.ndim != 3:
        raise ValueError(f"phi must have shape (n_images, n_quadrature, n_features), got {phi.shape}")
    if phi.shape[0] != current_batch_size:
        raise ValueError(f"phi image axis must match batch size: {phi.shape[0]} != {current_batch_size}")
    if current_batch_size == target_batch_size:
        return phi
    padded = np.zeros((target_batch_size, *phi.shape[1:]), dtype=phi.dtype)
    padded[:current_batch_size] = phi
    return padded


def _em_quadrature_posteriors_for_prepared_batch(
    config,
    images,
    ctf_params,
    rotation_matrices,
    translations,
    noise_variance,
    theta_coeffs,
    phi_padded,
    quad_weights,
    current_batch_size,
    *,
    em_temperature=1.0,
    em_prior_mix=0.0,
):
    n_features = int(theta_coeffs.shape[0])
    batch_size = int(images.shape[0])
    theta_coeffs = jnp.asarray(theta_coeffs)
    phi = jnp.asarray(phi_padded, dtype=theta_coeffs.real.dtype)
    quad_weights = jnp.asarray(quad_weights, dtype=theta_coeffs.real.dtype)

    centered_images = translate_images(images, translations, config.image_shape, half_image=True)
    centered_flat = centered_images.reshape(batch_size, -1)
    noise_half = noise_mod.to_batched_half_pixel_noise(noise_variance, config.image_shape, batch_size=batch_size)
    if noise_half.shape[0] == 1 and batch_size != 1:
        noise_half = jnp.broadcast_to(noise_half, centered_images.shape)
    noise_flat = noise_half.reshape(batch_size, -1)

    projected_basis = []
    for feature_idx in range(n_features):
        projected_basis.append(
            forward_model(
                config,
                theta_coeffs[feature_idx],
                ctf_params,
                rotation_matrices,
                half_image=True,
                half_volume=True,
            ).reshape(batch_size, -1)
        )
    projected_basis = jnp.stack(projected_basis, axis=0)
    predicted = jnp.einsum("bqr,rbp->bqp", phi, projected_basis)
    residual = jnp.sum(jnp.abs(centered_flat[:, None, :] - predicted) ** 2 / noise_flat[:, None, :], axis=-1)
    em_temperature = float(em_temperature)
    if em_temperature <= 0 or not np.isfinite(em_temperature):
        raise ValueError(f"em_temperature must be finite and positive, got {em_temperature}")
    em_prior_mix = float(em_prior_mix)
    if em_prior_mix < 0 or em_prior_mix > 1 or not np.isfinite(em_prior_mix):
        raise ValueError(f"em_prior_mix must be in [0, 1], got {em_prior_mix}")

    quad_prior = quad_weights / jnp.maximum(jnp.sum(quad_weights), 1e-30)
    log_prob = jnp.log(jnp.maximum(quad_prior, 1e-30))[None, :] - 0.5 * residual / em_temperature
    gamma = jax.nn.softmax(log_prob, axis=1)
    if em_prior_mix > 0:
        gamma = (1.0 - em_prior_mix) * gamma + em_prior_mix * quad_prior[None, :]
    return np.asarray(gamma[:current_batch_size])


def _em_weight_sets(alpha, phi, gamma):
    alpha = np.asarray(alpha, dtype=np.float32).reshape(-1)
    phi = np.asarray(phi, dtype=np.float32)
    gamma = np.asarray(gamma, dtype=np.float32)
    if phi.ndim != 3 or gamma.ndim != 2 or phi.shape[:2] != gamma.shape or phi.shape[0] != alpha.size:
        raise ValueError(
            "Incompatible EM weights: "
            f"alpha {alpha.shape}, phi {phi.shape}, gamma {gamma.shape}"
        )
    weighted = alpha[:, None] * gamma
    rhs_weights = np.einsum("bq,bqr->rb", weighted, phi, optimize=True)
    lhs_weights = np.einsum("bq,bqr,bqs->rsb", weighted, phi, phi, optimize=True)
    return rhs_weights.astype(np.float32), lhs_weights.reshape(phi.shape[2] * phi.shape[2], phi.shape[0]).astype(np.float32)


def _accumulate_local_polynomial_em_normal_equations(
    experiment_dataset,
    latent_differences,
    latent_precision,
    h_grid,
    theta_coeffs,
    *,
    degree,
    n_quadrature,
    basis_specs,
    batch_size,
    disc_type,
    upsampling_factor,
    use_fast_rfft,
    em_temperature,
    em_prior_mix,
):
    n_bandwidths = h_grid.size
    n_features = degree + 1
    half_volume_size = kernel_recon._candidate_half_volume_size(experiment_dataset, upsampling_factor)
    rhs_all = np.zeros((n_bandwidths, n_features, half_volume_size), dtype=experiment_dataset.dtype)
    lhs_all = np.zeros((n_bandwidths, n_features, n_features, half_volume_size), dtype=experiment_dataset.dtype_real)
    config = kernel_recon._reconstruction_config(experiment_dataset, disc_type, upsampling_factor)
    diagnostics = []

    for h_idx, h in enumerate(h_grid):
        Ft_y_acc = jnp.zeros((n_features, half_volume_size), dtype=experiment_dataset.dtype)
        Ft_ctf_acc = jnp.zeros((n_features * n_features, half_volume_size), dtype=experiment_dataset.dtype_real)
        entropy_sum = 0.0
        max_gamma_sum = 0.0
        effective_nodes_sum = 0.0
        frac_gt_09_count = 0
        frac_gt_099_count = 0
        n_gamma = 0
        raw_batches = experiment_dataset.iter_batches(
            batch_size,
            noise_model=experiment_dataset.noise,
            noise_half=False,
        )
        for (
            raw_images,
            rotation_matrices,
            translations,
            ctf_params,
            noise_variance,
            _particle_indices,
            image_indices,
        ) in raw_batches:
            image_indices = np.asarray(image_indices, dtype=np.int32)
            current_batch_size, images, rotation_matrices, translations, ctf_params, noise_variance = (
                kernel_recon._prepare_half_image_batch(
                    experiment_dataset,
                    raw_images,
                    rotation_matrices,
                    translations,
                    ctf_params,
                    noise_variance,
                    batch_size=batch_size,
                    use_fast_rfft=use_fast_rfft,
                )
            )
            alpha, phi, quad_weights = gaussian_window_polynomial_quadrature_1d(
                latent_differences[image_indices],
                latent_precision[image_indices],
                h=float(h),
                degree=degree,
                n_quadrature=n_quadrature,
                poly_scale=float(h),
                basis_spec=basis_specs[h_idx],
            )
            phi_padded = _pad_quadrature_features_for_fixed_batch(phi, current_batch_size, batch_size)
            gamma = _em_quadrature_posteriors_for_prepared_batch(
                config,
                images,
                ctf_params,
                rotation_matrices,
                translations,
                noise_variance,
                theta_coeffs[h_idx],
                phi_padded,
                quad_weights,
                current_batch_size,
                em_temperature=em_temperature,
                em_prior_mix=em_prior_mix,
            )
            rhs_weights, lhs_weights = _em_weight_sets(alpha, phi, gamma)
            entropy = -np.sum(gamma * np.log(np.maximum(gamma, 1e-30)), axis=1)
            max_gamma = np.max(gamma, axis=1)
            entropy_sum += float(np.sum(entropy))
            effective_nodes_sum += float(np.sum(np.exp(entropy)))
            max_gamma_sum += float(np.sum(max_gamma))
            frac_gt_09_count += int(np.count_nonzero(max_gamma > 0.9))
            frac_gt_099_count += int(np.count_nonzero(max_gamma > 0.99))
            n_gamma += int(gamma.shape[0])

            image_weights = np.concatenate([rhs_weights, lhs_weights], axis=0)
            image_weights = kernel_recon._pad_image_weight_matrix_for_fixed_batch(
                image_weights,
                current_batch_size=current_batch_size,
                target_batch_size=batch_size,
            )
            Ft_all_y, Ft_all_ctf = kernel_recon.backproject_weight_sets_from_fft(
                config,
                images,
                ctf_params,
                rotation_matrices,
                translations,
                noise_variance,
                image_weights,
                Ft_y=jnp.concatenate([Ft_y_acc, jnp.zeros_like(Ft_ctf_acc, dtype=Ft_y_acc.dtype)], axis=0),
                Ft_ctf=jnp.concatenate([jnp.zeros_like(Ft_y_acc, dtype=Ft_ctf_acc.dtype), Ft_ctf_acc], axis=0),
            )
            Ft_y_acc = Ft_all_y[:n_features]
            Ft_ctf_acc = Ft_all_ctf[n_features:]

        rhs_all[h_idx] = np.asarray(Ft_y_acc).reshape(n_features, half_volume_size)
        lhs_all[h_idx] = np.asarray(Ft_ctf_acc).reshape(n_features, n_features, half_volume_size)
        diagnostics.append(
            {
                "h": float(h),
                "mean_gamma_entropy": float(entropy_sum / max(n_gamma, 1)),
                "effective_quadrature_nodes": float(effective_nodes_sum / max(n_gamma, 1)),
                "mean_max_gamma": float(max_gamma_sum / max(n_gamma, 1)),
                "fraction_max_gamma_gt_0p9": float(frac_gt_09_count / max(n_gamma, 1)),
                "fraction_max_gamma_gt_0p99": float(frac_gt_099_count / max(n_gamma, 1)),
            }
        )

    return lhs_all, rhs_all, diagnostics


def estimate_local_polynomial_volumes_em(
    experiment_dataset,
    latent_differences,
    latent_precision,
    h_grid,
    *,
    degree=DEFAULT_LOCAL_POLY_DEGREE,
    n_iterations=2,
    n_quadrature=5,
    batch_size=None,
    tau=None,
    grid_correct=True,
    disc_type="linear_interp",
    use_spherical_mask=True,
    return_lhs_rhs=False,
    return_diagnostics=False,
    upsampling_factor=None,
    return_real_space=False,
    use_fast_rfft=False,
    em_temperature=1.0,
    em_prior_mix=0.0,
    em_update_damping=1.0,
    basis=DEFAULT_LOCAL_POLY_BASIS,
    basis_quantile=DEFAULT_LOCAL_POLY_BASIS_QUANTILE,
    cholesky_jitter=DEFAULT_LOCAL_POLY_CHOLESKY_JITTER,
    pol_reg_type="none",
    pol_reg_eta=0.0,
    pol_reg_power=2.0,
):
    """Estimate local-polynomial volumes with EM quadrature over latent x."""
    latent_differences = coerce_1d_latent_differences(latent_differences)
    latent_precision = coerce_1d_latent_precision(latent_precision)
    latent_differences = _expand_tilt_latent_array_to_images(
        experiment_dataset, latent_differences, "latent_differences"
    )
    latent_precision = _expand_tilt_latent_array_to_images(experiment_dataset, latent_precision, "latent_precision")
    if latent_differences.shape != latent_precision.shape:
        raise ValueError(
            "latent_differences and latent_precision must have the same flattened shape, "
            f"got {latent_differences.shape} and {latent_precision.shape}"
        )
    degree = int(degree)
    if degree < 0 or degree > MAX_LOCAL_POLY_DEGREE:
        raise ValueError(f"local_poly degree must be between 0 and {MAX_LOCAL_POLY_DEGREE}, got {degree}")
    h_grid = np.asarray(h_grid, dtype=np.float32).reshape(-1)
    if h_grid.size == 0 or not np.all(np.isfinite(h_grid)) or np.any(h_grid <= 0):
        raise ValueError(f"h_grid must contain finite positive values, got {h_grid}")
    basis = _validate_local_poly_basis(basis)
    pol_reg_type = _validate_pol_reg_type(pol_reg_type)
    n_iterations = int(n_iterations)
    if n_iterations < 0:
        raise ValueError(f"n_iterations must be nonnegative, got {n_iterations}")
    em_temperature = float(em_temperature)
    if em_temperature <= 0 or not np.isfinite(em_temperature):
        raise ValueError(f"em_temperature must be finite and positive, got {em_temperature}")
    em_prior_mix = float(em_prior_mix)
    if em_prior_mix < 0 or em_prior_mix > 1 or not np.isfinite(em_prior_mix):
        raise ValueError(f"em_prior_mix must be in [0, 1], got {em_prior_mix}")
    em_update_damping = float(em_update_damping)
    if em_update_damping <= 0 or em_update_damping > 1 or not np.isfinite(em_update_damping):
        raise ValueError(f"em_update_damping must be in (0, 1], got {em_update_damping}")

    half_volume_size = kernel_recon._candidate_half_volume_size(experiment_dataset, upsampling_factor)
    if batch_size is None:
        lhs_probe = np.zeros((h_grid.size, degree + 1, degree + 1, half_volume_size), dtype=experiment_dataset.dtype_real)
        rhs_probe = np.zeros((h_grid.size, degree + 1, half_volume_size), dtype=experiment_dataset.dtype)
        batch_size = _local_poly_batch_size(experiment_dataset, lhs_probe, rhs_probe, half_volume_size)

    logger.info(
        "local_poly EM initialization: degree=%s n_iter=%s n_quad=%s basis=%s pol_reg_type=%s pol_reg_eta=%s "
        "temperature=%s prior_mix=%s update_damping=%s h_grid=%s",
        degree,
        n_iterations,
        n_quadrature,
        basis,
        pol_reg_type,
        pol_reg_eta,
        em_temperature,
        em_prior_mix,
        em_update_damping,
        h_grid,
    )
    basis_specs = local_polynomial_basis_specs_1d(
        latent_differences,
        latent_precision,
        h_grid,
        degree,
        n_quadrature=n_quadrature,
        basis=basis,
        basis_quantile=basis_quantile,
        cholesky_jitter=cholesky_jitter,
    )
    target_eval_all = np.stack([spec["target_eval"] for spec in basis_specs], axis=0).astype(np.float32)
    pol_reg_matrices = local_polynomial_regularization_matrices(
        basis_specs,
        pol_reg_type=pol_reg_type,
        pol_reg_eta=pol_reg_eta,
        pol_reg_power=pol_reg_power,
    )
    _, lhs_all, rhs_all, init_diagnostics = estimate_local_polynomial_volumes(
        experiment_dataset,
        latent_differences,
        latent_precision,
        h_grid,
        degree=degree,
        batch_size=batch_size,
        tau=tau,
        grid_correct=grid_correct,
        disc_type=disc_type,
        use_spherical_mask=use_spherical_mask,
        return_lhs_rhs=True,
        upsampling_factor=upsampling_factor,
        return_real_space=False,
        use_fast_rfft=use_fast_rfft,
        basis=basis,
        basis_quantile=basis_quantile,
        cholesky_jitter=cholesky_jitter,
        moment_quadrature=n_quadrature,
        pol_reg_type=pol_reg_type,
        pol_reg_eta=pol_reg_eta,
        pol_reg_power=pol_reg_power,
        return_diagnostics=True,
    )
    theta_coeffs, init_solve_diagnostics = solve_local_polynomial_fourier_coefficients(
        lhs_all,
        rhs_all,
        experiment_dataset,
        tau=tau,
        pol_reg_matrices=pol_reg_matrices,
        upsampling_factor=upsampling_factor,
        return_diagnostics=True,
    )

    diagnostics = [
        {
            "stage": "initial_moment_fit",
            "basis": basis,
            "basis_info": [_as_jsonable(spec["basis_info"]) for spec in basis_specs],
            "pol_reg_type": pol_reg_type,
            "pol_reg_eta": float(pol_reg_eta),
            "pol_reg_power": float(pol_reg_power),
            "target_eval": target_eval_all.tolist(),
            "solve_diagnostics": init_solve_diagnostics,
            "moment_diagnostics": init_diagnostics,
        }
    ]
    for iteration in range(n_iterations):
        logger.info("local_poly EM iteration %s/%s", iteration + 1, n_iterations)
        lhs_all, rhs_all, iteration_diag = _accumulate_local_polynomial_em_normal_equations(
            experiment_dataset,
            latent_differences,
            latent_precision,
            h_grid,
            theta_coeffs,
            degree=degree,
            n_quadrature=n_quadrature,
            basis_specs=basis_specs,
            batch_size=batch_size,
            disc_type=disc_type,
            upsampling_factor=upsampling_factor,
            use_fast_rfft=use_fast_rfft,
            em_temperature=em_temperature,
            em_prior_mix=em_prior_mix,
        )
        new_theta_coeffs, solve_diagnostics = solve_local_polynomial_fourier_coefficients(
            lhs_all,
            rhs_all,
            experiment_dataset,
            tau=tau,
            pol_reg_matrices=pol_reg_matrices,
            upsampling_factor=upsampling_factor,
            return_diagnostics=True,
        )
        theta_coeffs = (1.0 - em_update_damping) * theta_coeffs + em_update_damping * new_theta_coeffs
        diagnostics.append(
            {
                "stage": "em_iteration",
                "iteration": int(iteration + 1),
                "bandwidths": iteration_diag,
                "solve_diagnostics": solve_diagnostics,
            }
        )

    estimates = postprocess_local_polynomial_fourier_coefficients(
        theta_coeffs,
        experiment_dataset,
        target_eval_all=target_eval_all,
        grid_correct=grid_correct,
        disc_type=disc_type,
        use_spherical_mask=use_spherical_mask,
        upsampling_factor=upsampling_factor,
        return_real_space=return_real_space,
    )
    outputs = [estimates]
    if return_lhs_rhs:
        outputs.extend([np.asarray(lhs_all), np.asarray(rhs_all)])
    if return_diagnostics:
        outputs.append(diagnostics)
    if len(outputs) == 1:
        return outputs[0]
    return tuple(outputs)

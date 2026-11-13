"""Helpers for building PPCA variance priors.

See docs/math/ppca_variance_prior_notes.md.
"""

import logging

import jax.numpy as jnp
import numpy as np

from recovar.heterogeneity import covariance_estimation
from recovar.reconstruction import relion_functions
from recovar.reconstruction import regularization
from recovar.utils import batch_make_radial_image, make_radial_image

logger = logging.getLogger(__name__)


def shell_average_real(values, volume_shape):
    """Average a real per-voxel quantity over Fourier shells."""
    shell = regularization.average_over_shells(jnp.array(np.asarray(values)).reshape(-1), volume_shape)
    return np.asarray(shell).reshape(-1).astype(np.float32)


def make_radial_prior_from_shell_total(
    shell_total,
    npc,
    volume_shape,
    *,
    label=None,
    clip_negative=True,
    variance_floor=1e-8,
):
    """Broadcast a total shell variance curve into the per-PC PPCA prior."""
    shell_total = np.asarray(shell_total, dtype=np.float32).reshape(-1)
    radial_raw = shell_total / float(npc)
    radial_used = np.maximum(radial_raw, variance_floor) if clip_negative else radial_raw
    img = np.array(batch_make_radial_image(jnp.array(radial_used).reshape(1, -1), volume_shape, True))[0]
    W_prior = np.tile(img.reshape(-1, 1), (1, npc)).astype(np.float32)
    neg_frac = float(np.mean(radial_raw < 0))
    if label is not None:
        logger.info(
            "%s shell prior: median(radial)=%.2e neg_shell_frac=%.3f",
            label,
            float(np.median(radial_used)),
            neg_frac,
        )
    return {
        "W_prior": W_prior,
        "shell_total": shell_total,
        "radial_raw": radial_raw,
        "radial_used": radial_used,
        "negative_shell_fraction": neg_frac,
    }


def repair_shell_total_with_mean_sq(
    shell_total,
    mean_sq_shells,
    *,
    variance_floor=1e-8,
    meansq_threshold_frac=0.01,
):
    """Repair unreliable high-resolution shell totals with a |mean|^2 fallback.

    Reliable shells are positive finite shells whose mean-squared scaffold is
    still informative. Fallback is applied to the unreliable high-frequency
    suffix and to any non-positive / non-finite shells.
    """
    shell_total = np.asarray(shell_total, dtype=np.float32).reshape(-1)
    mean_sq_shells = np.asarray(mean_sq_shells, dtype=np.float32).reshape(-1)
    if shell_total.shape != mean_sq_shells.shape:
        raise ValueError("shell_total and mean_sq_shells must have the same shape")

    finite_positive_mean = mean_sq_shells[np.isfinite(mean_sq_shells) & (mean_sq_shells > variance_floor)]
    meansq_threshold = (
        float(np.median(finite_positive_mean)) * meansq_threshold_frac if finite_positive_mean.size > 0 else 0.0
    )

    reliable = np.isfinite(shell_total) & (shell_total > variance_floor) & (mean_sq_shells > meansq_threshold)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(mean_sq_shells > variance_floor, shell_total / mean_sq_shells, np.nan)
    median_ratio = float(np.nanmedian(ratios[reliable])) if np.any(reliable) else 1.0
    meansq_fallback = mean_sq_shells * median_ratio

    if np.any(reliable):
        last_reliable = int(np.max(np.where(reliable)[0]))
        tail_fallback_mask = np.arange(shell_total.size) > last_reliable
    else:
        last_reliable = -1
        tail_fallback_mask = np.ones(shell_total.size, dtype=bool)

    invalid_mask = ~np.isfinite(shell_total) | (shell_total <= variance_floor)
    fallback_mask = tail_fallback_mask | invalid_mask

    repaired = np.asarray(shell_total, dtype=np.float32).copy()
    repaired[fallback_mask] = meansq_fallback[fallback_mask]
    repaired = np.where(np.isfinite(repaired), repaired, meansq_fallback)
    repaired = np.maximum(repaired, variance_floor)

    return {
        "raw_shell_total": shell_total,
        "repaired_shell_total": repaired,
        "mean_sq_shells": mean_sq_shells,
        "reliable": reliable,
        "tail_fallback_mask": fallback_mask,
        "median_ratio": median_ratio,
        "meansq_fallback": meansq_fallback,
        "meansq_threshold": meansq_threshold,
        "last_reliable_shell": last_reliable,
    }


def make_estimated_prior_from_combined(
    combined_fourier_variance,
    mean_estimate,
    npc,
    volume_shape,
    *,
    label="DataCombinedReg",
    variance_floor=1e-8,
    meansq_threshold_frac=0.01,
):
    """Build a cleaned PPCA prior from shell-averaged ``variance['combined']``."""
    raw_shell_total = shell_average_real(combined_fourier_variance, volume_shape)
    mean_sq_shells = shell_average_real(np.abs(np.asarray(mean_estimate).reshape(-1)) ** 2, volume_shape)
    repaired = repair_shell_total_with_mean_sq(
        raw_shell_total,
        mean_sq_shells,
        variance_floor=variance_floor,
        meansq_threshold_frac=meansq_threshold_frac,
    )
    radial = make_radial_prior_from_shell_total(
        repaired["repaired_shell_total"],
        npc,
        volume_shape,
        label=label,
        clip_negative=True,
        variance_floor=variance_floor,
    )
    radial.update(repaired)
    return radial


def make_gt_prior_from_variance_total(fourier_variance_total, npc, volume_shape):
    """Build the GT PPCA prior from total per-voxel Fourier variance."""
    shell_total = shell_average_real(fourier_variance_total, volume_shape)
    return make_radial_prior_from_shell_total(
        shell_total,
        npc,
        volume_shape,
        label="GT",
        clip_negative=False,
    )


def estimate_gaussian_shell_prior_from_data(dataset, mean_estimate, npc, volume_shape, batch_size):
    """Estimate the legacy Gaussian shell PPCA prior from half-set data.

    This reproduces the shell-prior construction used in the PPCA-EM notes:
    compute RELION-style half estimates on an all-ones mask, shell-average
    them, and convert the shell mean into a radial PPCA prior.  This remains a
    diagnostic/provisional estimator rather than a final production prior.
    """
    volume_mask = np.ones(volume_shape, dtype=np.float32)
    ctf_w, signal = [], []
    for halfset_dataset in dataset.materialize_halfset_datasets():
        fw, sig, _noise_weight, _noise_signal = covariance_estimation.variance_relion_style_triangular_kernel(
            halfset_dataset,
            mean_estimate,
            batch_size,
            image_subset=None,
            volume_mask=volume_mask,
            disc_type="linear_interp",
        )
        ctf_w.append(
            np.asarray(
                relion_functions.adjust_regularization_relion_style(
                    fw, halfset_dataset.volume_shape
                )
            )
        )
        signal.append(np.asarray(sig))

    corrected = [
        covariance_estimation._safe_div(jnp.asarray(signal[idx]), jnp.asarray(ctf_w[idx]))
        for idx in range(2)
    ]
    lhs = (np.asarray(ctf_w[0]) + np.asarray(ctf_w[1])) / 2

    rhs_total = np.asarray(signal[0]).real + np.asarray(signal[1]).real
    lhs_total = np.asarray(ctf_w[0]).real + np.asarray(ctf_w[1]).real
    rhs_shell = np.asarray(
        regularization.sum_over_shells(jnp.array(rhs_total), volume_shape).real
    ).reshape(-1)
    lhs_shell = np.asarray(
        regularization.sum_over_shells(jnp.array(lhs_total), volume_shape).real
    ).reshape(-1)
    shell_mean = np.where(lhs_shell > 1e-20, rhs_shell / lhs_shell, 0.0)

    fsc = np.asarray(
        regularization.get_fsc_gpu(
            corrected[0],
            corrected[1],
            volume_shape,
            substract_shell_mean=True,
        ).real
    ).reshape(-1)
    fsc_raw = np.asarray(
        regularization.get_fsc_gpu(
            corrected[0],
            corrected[1],
            volume_shape,
            substract_shell_mean=False,
        ).real
    ).reshape(-1)
    fsc_clipped = np.clip(fsc, 0.01, 0.999)
    shell_var = shell_mean**2 * (1.0 - fsc_clipped) / fsc_clipped

    shell_mean_vol = np.asarray(make_radial_image(jnp.array(shell_mean), volume_shape)).reshape(-1).real
    shell_var_vol = np.asarray(make_radial_image(jnp.array(shell_var), volume_shape)).reshape(-1).real
    combined = np.asarray((corrected[0] + corrected[1]) / 2).real.reshape(-1)

    inv_shell_var_vol = np.where(shell_var_vol > 1e-20, 1.0 / shell_var_vol, 0.0)
    corrected_gp = []
    for idx in range(2):
        rhs_idx = np.asarray(signal[idx]).real.reshape(-1)
        lhs_idx = np.asarray(ctf_w[idx]).real.reshape(-1)
        corrected_gp.append(
            np.where(
                lhs_idx + inv_shell_var_vol > 1e-20,
                (rhs_idx + shell_mean_vol * inv_shell_var_vol)
                / (lhs_idx + inv_shell_var_vol),
                shell_mean_vol,
            )
        )
    combined_gp = (corrected_gp[0] + corrected_gp[1]) / 2

    radial = make_radial_prior_from_shell_total(
        shell_mean,
        npc,
        volume_shape,
        label="GaussianShell",
        clip_negative=True,
    )
    return {
        "W_prior": radial["W_prior"],
        "radial_raw": radial["radial_raw"],
        "radial_used": radial["radial_used"],
        "shell_mean": np.asarray(shell_mean),
        "shell_var": np.asarray(shell_var),
        "shell_mean_vol": shell_mean_vol,
        "shell_var_vol": shell_var_vol,
        "fsc": fsc,
        "fsc_raw": fsc_raw,
        "corrected0": np.asarray(corrected[0]).real.reshape(-1),
        "corrected1": np.asarray(corrected[1]).real.reshape(-1),
        "combined": combined,
        "corrected_gp0": corrected_gp[0],
        "corrected_gp1": corrected_gp[1],
        "combined_gp": combined_gp,
        "lhs": np.asarray(lhs).real.reshape(-1),
    }


def estimate_hybrid_shell_prior_from_data(dataset, mean_estimate, npc, volume_shape, batch_size):
    """Estimate the provisional hybrid-shell PPCA prior from data.

    This uses the legacy Gaussian shell estimate and repairs unreliable shells
    with a ``|mean|^2`` scaffold.  It is the current stopgap prior for PPCA
    pipeline integration and should be tightened up later.
    """
    gaussian = estimate_gaussian_shell_prior_from_data(
        dataset, mean_estimate, npc, volume_shape, batch_size
    )
    mean_sq_shells = shell_average_real(
        np.abs(np.asarray(mean_estimate).reshape(-1)) ** 2,
        volume_shape,
    )
    repaired = repair_shell_total_with_mean_sq(
        gaussian["shell_mean"],
        mean_sq_shells,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        var_over_meansq = np.where(
            repaired["mean_sq_shells"] > 1e-12,
            repaired["raw_shell_total"] / repaired["mean_sq_shells"],
            np.nan,
        )
    radial = make_radial_prior_from_shell_total(
        repaired["repaired_shell_total"],
        npc,
        volume_shape,
        label="HybridShell",
        clip_negative=True,
    )
    gaussian.update(
        {
            "W_prior": radial["W_prior"],
            "radial_raw": radial["radial_raw"],
            "radial_used": radial["radial_used"],
            "hybrid_prior_shells": np.asarray(repaired["repaired_shell_total"]),
            "mean_sq_shells": np.asarray(repaired["mean_sq_shells"]),
            "var_over_meansq": np.asarray(var_over_meansq),
            "reliable": np.asarray(repaired["reliable"]),
            "median_ratio": float(repaired["median_ratio"]),
            "meansq_fallback": np.asarray(repaired["meansq_fallback"]),
            "tail_fallback_mask": np.asarray(repaired["tail_fallback_mask"]),
            "meansq_threshold": float(repaired["meansq_threshold"]),
            "last_reliable_shell": int(repaired["last_reliable_shell"]),
        }
    )
    return gaussian

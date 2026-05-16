"""Shared diagnostics-dict construction for PPCA EM iterations.

Both ``dense_dataset.py``, ``local_dataset.py`` and ``dense_engine.py``
publish the same handful of summary keys in their result diagnostics. The
helpers here own that contract so adding a key only happens in one place.

The ``flavor``-specific extras (sparse_pass2 stats for dense, image-scale
range for both flavors, Fourier-window sizes for dense) are layered in via
keyword arguments.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from recovar.em.ppca_refinement.config import MeanRegularizationConfig

__all__ = [
    "build_iteration_diagnostics",
    "resolve_image_scale_range",
]


def _concat_or_empty(arrays, dtype=jnp.int32):
    if arrays:
        return jnp.concatenate(arrays)
    return jnp.zeros((0,), dtype=dtype)


def build_iteration_diagnostics(
    *,
    pmax_values,
    nsig_values,
    best_rotations,
    best_translations,
    top_rotations=None,
    top_rotation_ids=None,
    top_translations=None,
    top_log_scores=None,
    top_posteriors=None,
    log_likelihood: float,
    n_images: int,
    mean_reg: MeanRegularizationConfig,
    image_scale_min: float,
    image_scale_max: float,
    image_scale_corrections,
    extras: dict | None = None,
) -> dict:
    """Build the common diagnostic-dict payload for one EM iteration.

    Parameters
    ----------
    pmax_values, nsig_values, best_rotations, best_translations
        Lists of per-block JAX arrays accumulated during the E-step.
    log_likelihood, n_images
        Scalar scaffolds used to publish ``log_likelihood`` and ``logZ_mean``.
    mean_reg
        Already-resolved mean-regularization config; its three fields are
        published verbatim for parity-check provenance.
    image_scale_min, image_scale_max, image_scale_corrections
        ``image_scale_corrections`` is the original argument; the three
        related keys answer "did the iteration apply per-image scaling and
        what was the range".
    extras
        Optional dict of flavor-specific keys merged on top.
    """

    diagnostics = {
        "pmax_mean": float(jnp.mean(jnp.concatenate(pmax_values))) if pmax_values else float("nan"),
        "nsig_mean": float(jnp.mean(jnp.concatenate(nsig_values))) if nsig_values else float("nan"),
        "log_likelihood": float(log_likelihood),
        "logZ_mean": float(log_likelihood / n_images) if n_images else float("nan"),
        "max_posterior_per_image": _concat_or_empty(pmax_values, dtype=jnp.float32),
        "n_significant_per_image": _concat_or_empty(nsig_values),
        "best_rotation_idx": _concat_or_empty(best_rotations),
        "best_translation_idx": _concat_or_empty(best_translations),
        "mean_regularization_style": str(mean_reg.style),
        "mean_tau2_fudge": float(mean_reg.tau2_fudge),
        "mean_minres_map": int(mean_reg.minres_map),
        "uses_image_scale_corrections": bool(image_scale_corrections is not None),
        "image_scale_min": float(image_scale_min),
        "image_scale_max": float(image_scale_max),
    }
    if top_rotations is not None:
        diagnostics["top_rotation_idx"] = (
            jnp.concatenate(top_rotations, axis=0) if top_rotations else jnp.zeros((0, 1), dtype=jnp.int32)
        )
    if top_rotation_ids is not None:
        diagnostics["top_rotation_id"] = (
            jnp.concatenate(top_rotation_ids, axis=0) if top_rotation_ids else jnp.zeros((0, 1), dtype=jnp.int32)
        )
    if top_translations is not None:
        diagnostics["top_translation_idx"] = (
            jnp.concatenate(top_translations, axis=0) if top_translations else jnp.zeros((0, 1), dtype=jnp.int32)
        )
    if top_log_scores is not None:
        top_log_score = (
            jnp.concatenate(top_log_scores, axis=0) if top_log_scores else jnp.zeros((0, 1), dtype=jnp.float32)
        )
        diagnostics["top_log_score"] = top_log_score
        diagnostics["top_log_score_per_image"] = top_log_score
    if top_posteriors is not None:
        top_posterior = (
            jnp.concatenate(top_posteriors, axis=0) if top_posteriors else jnp.zeros((0, 1), dtype=jnp.float32)
        )
        diagnostics["top_posterior"] = top_posterior
        diagnostics["top_posterior_per_image"] = top_posterior
    if top_rotations is not None or top_rotation_ids is not None or top_translations is not None:
        top_width_source = diagnostics.get(
            "top_rotation_id",
            diagnostics.get("top_rotation_idx", jnp.zeros((0, 1), dtype=jnp.int32)),
        )
        diagnostics["top_p_poses"] = int(top_width_source.shape[-1])
        diagnostics["top_pose_count"] = int(top_width_source.shape[-1])
    if extras:
        diagnostics.update(extras)
    return diagnostics


def resolve_image_scale_range(image_scale_corrections, image_indices):
    """Return ``(min, max)`` of the per-image scale factors selected by ``image_indices``.

    When ``image_scale_corrections`` is ``None`` we report ``(1.0, 1.0)``,
    matching the iteration's "no scaling" semantics.
    """
    if image_scale_corrections is None:
        return 1.0, 1.0
    arr = np.asarray(image_scale_corrections, dtype=np.float32)
    selected = arr if image_indices is None else arr[np.asarray(image_indices, dtype=np.int64)]
    if selected.size == 0:
        return float("nan"), float("nan")
    return float(np.min(selected)), float(np.max(selected))

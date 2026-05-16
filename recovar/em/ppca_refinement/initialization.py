"""PPCA initialization from K-class or GT volume banks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

import recovar.core.fourier_transform_utils as ftu
from recovar.utils import helpers as utils


VolumeAligner = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class PPCAInitialization:
    """Initializer output in the same representation as the loaded volumes."""

    mu: np.ndarray
    W: np.ndarray
    weights: np.ndarray
    aligned_volumes: np.ndarray
    diagnostics: dict = field(default_factory=dict)


def _normalize_weights(weights, n_volumes: int) -> np.ndarray:
    if weights is None:
        return np.full(n_volumes, 1.0 / float(n_volumes), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (n_volumes,):
        raise ValueError(f"weights must have shape ({n_volumes},), got {weights.shape}")
    if np.any(weights < 0) or not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite and nonnegative")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("weights must have positive total mass")
    return weights / total


def _load_one_volume(volume, *, frame: str) -> np.ndarray:
    frame = str(frame)
    if frame not in {"recovar", "relion", "fourier"}:
        raise ValueError("frame must be one of 'recovar', 'relion', or 'fourier'")
    if isinstance(volume, (str, Path)):
        if frame == "relion":
            return np.asarray(utils.load_relion_volume(volume), dtype=np.float32)
        if frame == "recovar":
            return np.asarray(utils.load_mrc(volume), dtype=np.float32)
        raise ValueError("fourier-frame path loading is intentionally not guessed; pass arrays for Fourier volumes")
    arr = np.asarray(volume)
    if arr.ndim != 3:
        raise ValueError(f"each volume must be 3D, got {arr.shape}")
    if frame == "relion":
        return np.asarray(utils.relion_volume_to_recovar(arr), dtype=np.float32)
    return np.array(arr, dtype=np.complex64 if np.iscomplexobj(arr) else np.float32, copy=True)


def load_volume_stack(volumes, *, frame: str) -> np.ndarray:
    """Load or coerce a volume bank with exactly one declared frame conversion."""
    if isinstance(volumes, np.ndarray) and volumes.ndim == 4:
        return np.stack([_load_one_volume(vol, frame=frame) for vol in volumes], axis=0)
    if not isinstance(volumes, Iterable):
        raise ValueError("volumes must be a 4D array or an iterable of 3D arrays/paths")
    loaded = [_load_one_volume(vol, frame=frame) for vol in volumes]
    if not loaded:
        raise ValueError("at least one volume is required")
    shape = loaded[0].shape
    if any(vol.shape != shape for vol in loaded):
        raise ValueError("all volumes must have the same shape")
    return np.stack(loaded, axis=0)


def _align_volume_stack(volumes: np.ndarray, alignment_fn: VolumeAligner | None) -> tuple[np.ndarray, bool]:
    if alignment_fn is None:
        return np.array(volumes, copy=True), False
    reference = np.asarray(volumes[0])
    aligned = [reference]
    for moving in volumes[1:]:
        aligned_moving = np.asarray(alignment_fn(reference, np.asarray(moving)))
        if aligned_moving.shape != reference.shape:
            raise ValueError(f"alignment_fn returned shape {aligned_moving.shape}, expected {reference.shape}")
        aligned.append(aligned_moving)
    return np.stack(aligned, axis=0), True


def empirical_weighted_covariance(volumes: np.ndarray, weights) -> np.ndarray:
    """Return the probability-weighted centered covariance of flattened volumes."""
    volumes = np.asarray(volumes)
    weights = _normalize_weights(weights, int(volumes.shape[0]))
    flat = volumes.reshape(volumes.shape[0], -1)
    mean = np.sum(weights[:, None] * flat, axis=0)
    centered = flat - mean[None, :]
    return (centered * weights[:, None]).T @ np.conj(centered)


def covariance_from_loading_matrix(W: np.ndarray) -> np.ndarray:
    """Return ``W W*`` for loading volumes shaped ``[q, *volume_shape]``."""
    W = np.asarray(W)
    flat = W.reshape(W.shape[0], -1)
    return flat.T @ np.conj(flat)


def real_volume_to_centered_fourier(volume: np.ndarray) -> np.ndarray:
    """Convert a recovar-frame real-space volume to centered Fourier storage."""
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3D, got {volume.shape}")
    return np.asarray(ftu.get_dft3(volume), dtype=np.complex64)


def real_volume_to_centered_fourier_half(volume: np.ndarray) -> np.ndarray:
    """Convert a real-space volume to flattened RECOVAR half-Fourier storage."""
    centered = real_volume_to_centered_fourier(volume)
    return np.asarray(ftu.full_volume_to_half_volume(centered, centered.shape), dtype=np.complex64).reshape(-1)


def _half_volume_size(volume_shape) -> int:
    return int(np.prod(ftu.volume_shape_to_half_volume_shape(tuple(volume_shape))))


def _coerce_loading_columns_to_fourier_half(
    W: np.ndarray,
    *,
    volume_shape,
    volume_domain: str,
    q: int | None = None,
) -> np.ndarray:
    """Return loadings as ``[half_size, q]`` in centered half-Fourier storage."""
    volume_shape = tuple(int(x) for x in volume_shape)
    half_size = _half_volume_size(volume_shape)
    full_size = int(np.prod(volume_shape))
    arr = np.asarray(W)
    domain = str(volume_domain)
    if domain == "auto":
        if arr.ndim == 2 and arr.shape[0] == half_size:
            domain = "fourier_half"
        elif arr.ndim == 2 and arr.shape[1] == half_size:
            domain = "fourier_half"
        elif arr.ndim >= 2 and tuple(arr.shape[1:]) == volume_shape and np.iscomplexobj(arr):
            domain = "fourier_full"
        elif arr.ndim >= 2 and tuple(arr.shape[1:]) == volume_shape:
            domain = "real"
        else:
            raise ValueError("could not infer loading volume_domain")

    if domain == "fourier_half":
        if arr.ndim == 1:
            columns = arr.reshape(half_size, 1)
        elif arr.ndim == 2 and arr.shape[0] == half_size:
            columns = arr
        elif arr.ndim == 2 and arr.shape[1] == half_size:
            columns = arr.T
        else:
            raise ValueError(f"fourier_half W must be [half_size, q] or [q, half_size], got {arr.shape}")
        if q is not None:
            columns = columns[:, : int(q)]
        return np.asarray(columns, dtype=np.complex64)

    if domain not in {"real", "fourier_full"}:
        raise ValueError("volume_domain must be 'auto', 'real', 'fourier_full', or 'fourier_half'")
    if arr.ndim < 2 or tuple(arr.shape[1:]) != volume_shape:
        raise ValueError(f"{domain} W must be shaped [q, *volume_shape], got {arr.shape}")
    if q is not None:
        arr = arr[: int(q)]
    columns = []
    for loading in arr:
        if domain == "real":
            columns.append(real_volume_to_centered_fourier_half(loading))
        else:
            if loading.size != full_size:
                raise ValueError(f"fourier_full loading has {loading.size} elements, expected {full_size}")
            half = ftu.full_volume_to_half_volume(np.asarray(loading).reshape(volume_shape), volume_shape).reshape(-1)
            columns.append(np.asarray(half, dtype=np.complex64))
    if not columns:
        return np.zeros((half_size, 0), dtype=np.complex64)
    return np.stack(columns, axis=1)


def _box_normalization(volume_shape, *, box_size_power: float) -> float:
    """Return the explicit synthetic-prior DFT normalization factor."""
    volume_shape = tuple(int(x) for x in volume_shape)
    if len(set(volume_shape)) != 1:
        raise ValueError(f"box-size prior normalization expects a cubic volume, got {volume_shape}")
    return float(volume_shape[0]) ** float(box_size_power)


def _half_radial_shell_labels(volume_shape) -> np.ndarray:
    labels = np.asarray(
        ftu.get_grid_of_radial_distances_real(tuple(volume_shape), scaled=False, frequency_shift=0),
        dtype=np.int64,
    ).reshape(-1)
    expected_size = _half_volume_size(volume_shape)
    if labels.size != expected_size:
        raise AssertionError(f"radial shell labels have {labels.size} entries, expected {expected_size}")
    return labels


def _average_half_values_over_shells(values: np.ndarray, volume_shape) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    labels = _half_radial_shell_labels(volume_shape)
    if values.shape != labels.shape:
        raise ValueError(f"values shape {values.shape} does not match half-shell labels {labels.shape}")
    shell_count = int(labels.max(initial=0)) + 1
    counts = np.bincount(labels, minlength=shell_count).astype(np.float64)
    sums = np.bincount(labels, weights=values, minlength=shell_count).astype(np.float64)
    shell_mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return shell_mean[labels]


def loading_row_norm_variance_prior(
    W: np.ndarray,
    *,
    volume_shape,
    volume_domain: str = "auto",
    q: int | None = None,
    box_size_power: float = 0.0,
    scale: float = 1.0,
    floor: float = 0.0,
    shell_average: bool = True,
    divide_by_q_total: bool = False,
    q_total: int | None = None,
) -> np.ndarray:
    """Build a variance-like W prior from the GT loading row norm.

    The PPCA latent prior remains identity. W stores the eigenvalue/covariance
    scale, so the per-frequency GT variance is ``sum_k |W[xi, k]|^2`` in the
    same half-Fourier coordinates used by the dense E/M steps. By default this
    row-norm-squared curve is averaged over radial shells, matching the
    no-pose PPCA prior style and avoiding coefficient-wise prior spikes.

    By default the prior is the raw half-Fourier row norm, matching the scale
    of the augmented dense M-step unknowns. ``box_size_power`` is an explicit
    synthetic/debug knob for legacy experiments that divide DFT power by
    ``N**box_size_power``; it should not be confused with RELION InitialModel
    BPref stats.
    """
    columns = _coerce_loading_columns_to_fourier_half(W, volume_shape=volume_shape, volume_domain=volume_domain, q=q)
    row_norm = np.sum(np.abs(columns) ** 2, axis=1, dtype=np.float64)
    if shell_average:
        row_norm = _average_half_values_over_shells(row_norm, volume_shape)
    if divide_by_q_total and columns.shape[1] > 0:
        divisor = int(columns.shape[1]) if q_total is None else int(q_total)
        if divisor <= 0:
            raise ValueError("q_total must be positive when divide_by_q_total=True")
        row_norm = row_norm / float(divisor)
    prior = row_norm * float(scale) / _box_normalization(volume_shape, box_size_power=box_size_power)
    if floor > 0.0:
        prior = np.maximum(prior, float(floor))
    return np.repeat(prior[:, None], columns.shape[1], axis=1).astype(np.float32)


def pipeline_variance_W_prior(
    pipeline_variance_half: np.ndarray,
    *,
    q: int,
    divide_by_q: bool = True,
    floor: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    """Wrap a pipeline-saved per-half-voxel (or per-shell) signal variance into a W prior.

    The recovar pipeline saves per-Fourier-voxel signal-variance arrays in
    ``params.pkl['variance_est']`` (``'prior'`` and ``'combined'`` keys). When
    re-exported into a PPCA init NPZ by ``prepare_ppca_init_from_pipeline_output_v2.py``
    the array is already flattened in half-spectrum order and matches the EM
    refinement's per-voxel W prior layout. To produce the ``(half_size, q)``
    array the M-step expects, we repeat the same prior across q columns. By
    default the per-PC variance budget is the total signal variance divided
    by q so that ``sum_k |W[xi, k]|^2`` matches the pipeline's estimate.
    """
    arr = np.asarray(pipeline_variance_half, dtype=np.float64).reshape(-1)
    if divide_by_q and q > 0:
        arr = arr / float(q)
    arr = arr * float(scale)
    if floor > 0.0:
        arr = np.maximum(arr, float(floor))
    return np.repeat(arr.astype(np.float32)[:, None], int(q), axis=1)


def volume_power_variance_prior(
    volume: np.ndarray,
    *,
    volume_shape,
    volume_domain: str = "auto",
    box_size_power: float = 0.0,
    scale: float = 1.0,
    floor: float = 0.0,
    shell_average: bool = True,
) -> np.ndarray:
    """Build a variance-like mean prior from one volume's half-Fourier power."""
    volume_shape = tuple(int(x) for x in volume_shape)
    half_size = _half_volume_size(volume_shape)
    arr = np.asarray(volume)
    domain = str(volume_domain)
    if domain == "auto":
        if arr.size == half_size:
            domain = "fourier_half"
        elif arr.size == int(np.prod(volume_shape)) and np.iscomplexobj(arr):
            domain = "fourier_full"
        elif tuple(arr.shape) == volume_shape:
            domain = "real"
        else:
            raise ValueError("could not infer volume_domain")
    if domain == "fourier_half":
        if arr.size != half_size:
            raise ValueError(f"fourier_half volume has {arr.size} elements, expected {half_size}")
        half = np.asarray(arr.reshape(-1), dtype=np.complex64)
    elif domain == "fourier_full":
        half = np.asarray(ftu.full_volume_to_half_volume(arr.reshape(volume_shape), volume_shape), dtype=np.complex64).reshape(-1)
    elif domain == "real":
        half = real_volume_to_centered_fourier_half(arr)
    else:
        raise ValueError("volume_domain must be 'auto', 'real', 'fourier_full', or 'fourier_half'")
    prior = np.abs(half).astype(np.float64) ** 2
    if shell_average:
        prior = _average_half_values_over_shells(prior, volume_shape)
    prior = prior * float(scale) / _box_normalization(volume_shape, box_size_power=box_size_power)
    if floor > 0.0:
        prior = np.maximum(prior, float(floor))
    return prior.astype(np.float32)


def _weighted_mean_and_loading_pca(
    volumes: np.ndarray,
    *,
    q: int,
    weights,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if q < 0:
        raise ValueError("q must be nonnegative")
    weights = _normalize_weights(weights, int(volumes.shape[0]))
    work = np.asarray(volumes)
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != work.shape[1:]:
            raise ValueError(f"mask shape {mask.shape} != volume shape {work.shape[1:]}")
        work = work * mask[None, ...]
    flat = work.reshape(work.shape[0], -1)
    mu_flat = np.sum(weights[:, None] * flat, axis=0)
    centered = flat - mu_flat[None, :]
    weighted_centered = centered * np.sqrt(weights)[:, None]
    _u, singular_values, vh = np.linalg.svd(weighted_centered, full_matrices=False)
    q_eff = min(int(q), vh.shape[0])
    W_flat = np.zeros((int(q), flat.shape[1]), dtype=np.result_type(work.dtype, np.float32))
    if q_eff:
        W_flat[:q_eff] = singular_values[:q_eff, None] * vh[:q_eff]
    diagnostics = {
        "q_requested": int(q),
        "q_effective": int(q_eff),
        "singular_values": singular_values[:q_eff].astype(np.float64),
        "weighted_covariance_trace": float(np.real(np.sum(singular_values**2))),
        "latent_prior": "identity",
        "W_stores_covariance_scale": True,
    }
    return mu_flat.reshape(work.shape[1:]), W_flat.reshape((int(q),) + work.shape[1:]), weights, diagnostics


def initialize_ppca_from_kclass_volumes(
    volumes,
    *,
    q: int,
    class_weights=None,
    frame: str = "recovar",
    alignment_fn: VolumeAligner | None = None,
    mask: np.ndarray | None = None,
) -> PPCAInitialization:
    """Initialize PPCA from aligned K-class volumes.

    The returned loadings satisfy ``W W*`` equals the rank-``q`` weighted
    centered-volume covariance in the loaded representation. No eigenvalue
    scale is hidden in the latent prior.
    """
    loaded = load_volume_stack(volumes, frame=frame)
    aligned, alignment_applied = _align_volume_stack(loaded, alignment_fn)
    mu, W, weights, diagnostics = _weighted_mean_and_loading_pca(
        aligned,
        q=q,
        weights=class_weights,
        mask=mask,
    )
    diagnostics.update(
        {
            "source": "kclass",
            "input_frame": frame,
            "alignment_applied": alignment_applied,
            "n_input_volumes": int(aligned.shape[0]),
        }
    )
    return PPCAInitialization(mu=mu, W=W, weights=weights, aligned_volumes=aligned, diagnostics=diagnostics)


def initialize_ppca_from_gt_volumes(
    volumes,
    *,
    q: int,
    weights=None,
    frame: str,
    alignment_fn: VolumeAligner | None = None,
    mask: np.ndarray | None = None,
    amplitude_scale: float | None = None,
) -> PPCAInitialization:
    """Initialize PPCA from GT/evaluation volumes with explicit conventions.

    GT volumes are accepted only with an explicit frame. ``amplitude_scale`` is
    optional and applied uniformly before mean/PCA so tests can pin synthetic
    Fourier or real-space scaling without guessing.
    """
    if frame is None:
        raise ValueError("GT initialization requires an explicit frame")
    loaded = load_volume_stack(volumes, frame=frame)
    if amplitude_scale is not None:
        loaded = loaded * float(amplitude_scale)
    aligned, alignment_applied = _align_volume_stack(loaded, alignment_fn)
    mu, W, norm_weights, diagnostics = _weighted_mean_and_loading_pca(
        aligned,
        q=q,
        weights=weights,
        mask=mask,
    )
    diagnostics.update(
        {
            "source": "gt",
            "input_frame": frame,
            "alignment_applied": alignment_applied,
            "amplitude_scale": 1.0 if amplitude_scale is None else float(amplitude_scale),
            "n_input_volumes": int(aligned.shape[0]),
        }
    )
    return PPCAInitialization(mu=mu, W=W, weights=norm_weights, aligned_volumes=aligned, diagnostics=diagnostics)

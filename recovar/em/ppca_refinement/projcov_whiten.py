"""Post-M-step whitening helpers for the PPCA refinement.

Two whitening flavors:

* :func:`whiten_W_svd_post_mstep` — matches the recovar PPCA route's M-step
  (recovar/heterogeneity/ppca.py:91-94): given the closed-form Wiener W
  output, replace it with ``U @ diag(S)`` from ``svd(W)``. The discarded
  right factor ``Vt`` picks a canonical orthogonal-column orientation for W.
  Cheap (one SVD per iter), self-contained.

* :func:`whiten_W_via_projcov` — heavier variant that mirrors the covariance
  route's ``pca_by_projected_covariance`` whitening (used for the
  fixed-pose pipeline's ``u_rescaled``/``s_rescaled``): temporarily overrides
  the dataset's poses with the EM's current best poses, runs the pipeline's
  projected-covariance estimator with current W as basis, returns the
  re-eigendecomposed W. Requires GPU work proportional to the dataset.

The cheap SVD flavor is what the pipeline's ``recovar.heterogeneity.ppca.EM``
loop does internally — i.e. it IS the "whitening step the PPCA route always
runs". The full-projcov variant is the user's "hacky" suggestion if the SVD
flavor isn't enough.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Iterable

import jax.numpy as jnp
import numpy as np

from recovar.core import fourier_transform_utils as ftu

logger = logging.getLogger(__name__)


def whiten_W_svd_post_mstep(W_half_flat: np.ndarray) -> np.ndarray:
    """SVD-whiten the M-step W output (matches recovar.heterogeneity.ppca.M_step).

    ``W_half_flat`` is ``(half_size, q)`` complex (centered half-Fourier flat).
    Computes ``svd(W_half_flat, full_matrices=False)`` and returns ``U @ diag(S)``.
    The columns of the returned W are orthonormal in the half-Fourier inner
    product (up to the SVD's tolerance). Singular values are sorted descending.

    This is the cheap whitening the pipeline's recovar.heterogeneity.ppca.EM
    runs after every M-step. The discarded ``Vt`` factor is the per-iter
    orthogonal rotation that the Tipping-Bishop optimum is invariant under;
    discarding it gives a canonical column-orthogonal W and prevents the
    column basis from drifting iter-to-iter.
    """
    W = jnp.asarray(W_half_flat)
    if W.ndim != 2:
        raise ValueError(f"W_half_flat must be 2D (half_size, q); got shape {W.shape}")
    U, S, _ = jnp.linalg.svd(W, full_matrices=False)
    out = U @ jnp.diag(S)
    return np.asarray(out, dtype=W_half_flat.dtype)


@contextlib.contextmanager
def _override_dataset_poses(dataset, *, rotation_matrices=None, translations=None):
    """Temporarily replace the dataset's pose arrays.

    Restores the original arrays on exit. Uses the existing CryoEMDataset
    setters which validate dtypes against the metadata schema.
    """
    saved_rot = None
    saved_trans = None
    if rotation_matrices is not None:
        saved_rot = np.asarray(dataset.rotation_matrices, copy=True)
        dataset.rotation_matrices = np.asarray(rotation_matrices)
    if translations is not None:
        saved_trans = np.asarray(dataset.translations, copy=True)
        dataset.translations = np.asarray(translations)
    try:
        yield
    finally:
        if saved_rot is not None:
            dataset.rotation_matrices = saved_rot
        if saved_trans is not None:
            dataset.translations = saved_trans


def _half_flat_to_full_fourier_flat(half_flat: np.ndarray, volume_shape) -> np.ndarray:
    """Convert centered half-Fourier flat ``(half_size,)`` -> full Fourier flat ``(vol_size,)``."""
    half = jnp.asarray(half_flat).reshape(-1)
    full = ftu.half_volume_to_full_volume(half, tuple(volume_shape))
    return np.asarray(full).reshape(-1).astype(np.complex64)


def _full_fourier_flat_to_half_flat(full_flat: np.ndarray, volume_shape) -> np.ndarray:
    """Convert full Fourier flat ``(vol_size,)`` -> centered half-Fourier flat ``(half_size,)``."""
    full = jnp.asarray(full_flat).reshape(tuple(volume_shape))
    half = ftu.full_volume_to_half_volume(full, tuple(volume_shape))
    return np.asarray(half).reshape(-1).astype(np.complex64)


def whiten_W_via_projcov(
    dataset,
    mu_half_flat: np.ndarray,
    W_half_flat: np.ndarray,
    *,
    best_rotation_matrices: np.ndarray,
    best_translations: np.ndarray | None,
    volume_mask: np.ndarray,
    disc_type: str = "linear_interp",
    disc_type_u: str = "linear_interp",
    use_mask: bool = True,
    ignore_zero_frequency: bool = False,
    gpu_memory_to_use: int = 40,
    n_pcs_to_compute: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Whiten the EM's current ``(mu, W)`` via the pipeline's projcov estimator.

    Hacky post-M-step refinement step. Calls
    :func:`recovar.heterogeneity.principal_components.pca_by_projected_covariance`
    with the EM's current best poses overriding the dataset's stored poses.
    The returned ``W`` is re-orthogonalized and rescaled by the per-PC
    eigenvalues of the projected covariance — the same form the pipeline
    uses for ``u_rescaled * sqrt(s_rescaled)``.

    Returns
    -------
    W_whitened_half_flat : ``(half_size, q)`` complex64
        The whitened W in centered half-Fourier flat layout (same convention
        as the EM's ``W_half``).
    s_rescaled : ``(q,)`` float32
        Per-PC eigenvalues from the projected covariance solve.
    """
    from recovar.heterogeneity.principal_components import pca_by_projected_covariance

    volume_shape = tuple(int(x) for x in dataset.volume_shape)
    half_size = int(np.prod(ftu.volume_shape_to_half_volume_shape(volume_shape)))
    vol_size = int(np.prod(volume_shape))

    mu_half = np.asarray(mu_half_flat, dtype=np.complex64).reshape(-1)
    W_half = np.asarray(W_half_flat, dtype=np.complex64)
    if mu_half.shape != (half_size,):
        raise ValueError(f"mu_half_flat must have shape ({half_size},), got {mu_half.shape}")
    if W_half.ndim != 2 or W_half.shape[0] != half_size:
        raise ValueError(f"W_half_flat must have shape (half_size={half_size}, q), got {W_half.shape}")
    q = int(W_half.shape[1])
    if q == 0:
        return W_half.astype(np.complex64), np.zeros((0,), dtype=np.float32)

    # Convert mu (half-flat) -> full Fourier full-volume flat
    mu_full_flat = _half_flat_to_full_fourier_flat(mu_half, volume_shape)

    # Convert W (half_size, q) -> (vol_size, q) full Fourier basis
    basis_full = np.empty((vol_size, q), dtype=np.complex64)
    for k in range(q):
        basis_full[:, k] = _half_flat_to_full_fourier_flat(W_half[:, k], volume_shape)

    # Override dataset poses with EM best poses, then call pipeline projcov
    n_images = int(dataset.n_images)
    best_rot = np.asarray(best_rotation_matrices, dtype=np.float32)
    if best_rot.shape != (n_images, 3, 3):
        raise ValueError(
            f"best_rotation_matrices shape {best_rot.shape} != ({n_images}, 3, 3)"
        )
    if best_translations is None:
        best_t = np.zeros((n_images, 2), dtype=np.float32)
    else:
        best_t = np.asarray(best_translations, dtype=np.float32)
        if best_t.shape != (n_images, 2):
            raise ValueError(
                f"best_translations shape {best_t.shape} != ({n_images}, 2)"
            )

    logger.info(
        "Running pca_by_projected_covariance for post-M-step whitening (q=%d, n_images=%d, mask_voxels=%d)",
        q,
        n_images,
        int((np.asarray(volume_mask) > 0.5).sum()),
    )
    with _override_dataset_poses(dataset, rotation_matrices=best_rot, translations=best_t):
        u_new, s_new = pca_by_projected_covariance(
            dataset,
            basis_full,
            mu_full_flat,
            np.asarray(volume_mask, dtype=np.float32),
            disc_type=str(disc_type),
            disc_type_u=str(disc_type_u),
            gpu_memory_to_use=int(gpu_memory_to_use),
            use_mask=bool(use_mask),
            ignore_zero_frequency=bool(ignore_zero_frequency),
            n_pcs_to_compute=n_pcs_to_compute,
        )

    u_new = np.asarray(u_new, dtype=np.complex64)
    s_new = np.asarray(s_new, dtype=np.float32)
    if u_new.shape != (vol_size, q):
        raise ValueError(f"pca_by_projected_covariance returned u shape {u_new.shape}, expected ({vol_size},{q})")
    if s_new.shape != (q,):
        raise ValueError(f"pca_by_projected_covariance returned s shape {s_new.shape}, expected ({q},)")

    # W_whitened = u_new × sqrt(s_new) per PC column, in full Fourier full-volume flat
    sqrt_s = np.sqrt(np.maximum(s_new, 0.0)).astype(np.float32)
    W_whitened_full = u_new * sqrt_s[None, :].astype(u_new.dtype)

    # Convert back to half-Fourier flat
    W_whitened_half = np.empty((half_size, q), dtype=np.complex64)
    for k in range(q):
        W_whitened_half[:, k] = _full_fourier_flat_to_half_flat(W_whitened_full[:, k], volume_shape)
    return W_whitened_half, s_new

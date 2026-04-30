"""VDAM E-step adapter helpers.

The full E-step re-uses the dense-path kernels in
`recovar.em.dense_single_volume.em_engine`, but applies three
InitialModel-specific corrections before scoring:

  1. `padding_factor = 1` (not 2) on the projector / backprojector side.
  2. `Minvsigma2[0] = 0` — DC shell excluded from the likelihood to match
     RELION behaviour (avoids the amplitude leak documented in
     `project_relion_pf2_dc_exclusion_bug.md`).
  3. Hermitian-weight convention: RELION uses `w=1` for all half-complex
     pixels (including redundant conjugate pairs). We match that for
     parity; see `project_relion_hermitian_weight_debt.md`.

This module exposes the small RELION-convention helpers and a posterior
summary container used by tests/debugging. The production InitialModel
E-step wiring lives in `dense_adapter.py`, which calls the native dense
K-class engine jointly over classes and poses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Helper 1: Minvsigma2 with DC zeroed
# ---------------------------------------------------------------------------


def minvsigma2_with_dc_zero(sigma2_per_group: np.ndarray) -> np.ndarray:
    """Return `1 / sigma2_noise` with the DC shell set to 0.

    RELION sets `Minvsigma2[0] = 0` before applying it to the scoring
    kernel (see ml_optimiser.cpp's expectation-step preprocessing). This
    excludes the DC term from the likelihood.

    Output shape matches the input; caller is responsible for tiling into
    the 2D scoring map via shell indices.
    """
    sigma2 = np.asarray(sigma2_per_group, dtype=np.float64)
    if sigma2.ndim not in (1, 2):
        raise ValueError("sigma2 must be (n_shells,) or (G, n_shells)")
    # Guard against zeros in sigma2
    inv = np.zeros_like(sigma2)
    nz = sigma2 > 0
    inv[nz] = 1.0 / sigma2[nz]
    # Zero the DC shell (ires = 0)
    if sigma2.ndim == 1:
        inv[0] = 0.0
    else:
        inv[:, 0] = 0.0
    return inv


# ---------------------------------------------------------------------------
# Helper 2: Hermitian weights for half-complex scoring
# ---------------------------------------------------------------------------


def hermitian_weights_relion(ori_size: int) -> np.ndarray:
    """Return RELION's half-complex pixel weights (all ones).

    RELION iterates over the rfft half-complex layout
    `(ori_size, ori_size/2 + 1)` with weight = 1 for every pixel
    — not 2 for interior frequencies as a strict Hermitian count would
    require. We match this for parity.

    Full "mathematically correct" weights (2 for interior, 1 for
    DC/Nyquist columns) live in
    `recovar.em.dense_single_volume.helpers.fourier_window.make_half_image_weights`
    and are selectable post-parity.
    """
    if ori_size < 2:
        raise ValueError("ori_size must be >= 2")
    return np.ones((ori_size, ori_size // 2 + 1), dtype=np.float64)


# ---------------------------------------------------------------------------
# Helper 3: current-resolution Fourier crop
# ---------------------------------------------------------------------------


def fourier_crop_half(image_half: np.ndarray, current_size: int) -> np.ndarray:
    """Crop a half-complex 2D image to `current_size`.

    RELION `windowFourierTransform` drops all Fourier coefficients outside
    the shell radius `current_size/2`. For half-complex arrays the output
    is `(current_size, current_size/2 + 1)`.

    `image_half` must be shape `(ori_size, ori_size/2 + 1)`.

    Mirrors what `fftw.cpp::windowFourierTransform` does at the slice
    size specified by `current_size`, no padding_factor.
    """
    if image_half.ndim != 2:
        raise ValueError("image_half must be 2D (ori_size, ori_size/2+1)")
    ori_size = image_half.shape[0]
    if image_half.shape[1] != ori_size // 2 + 1:
        raise ValueError(f"image_half expected (N, N/2+1), got {image_half.shape}")
    if current_size > ori_size:
        raise ValueError(f"current_size={current_size} > ori_size={ori_size}")
    if current_size < 2 or current_size % 2:
        raise ValueError(f"current_size={current_size} must be even and >= 2")
    if current_size == ori_size:
        return np.ascontiguousarray(image_half)

    half_cs = current_size // 2
    out_y = current_size
    out_x = current_size // 2 + 1
    out = np.zeros((out_y, out_x), dtype=image_half.dtype)

    # RELION's `windowFourierTransform` keeps low y-rows from [0, half_cs]
    # and the mirrored negative-frequency rows from the tail. For the
    # half-complex x-direction (which only stores non-negative frequencies)
    # we just take the first `out_x` columns.
    out[:half_cs, :out_x] = image_half[:half_cs, :out_x]
    out[half_cs:, :out_x] = image_half[ori_size - (out_y - half_cs) :, :out_x]
    return out


# ---------------------------------------------------------------------------
# Posterior container
# ---------------------------------------------------------------------------


@dataclass
class VdamPosterior:
    """Output of one VDAM E-step batch.

    `weights` is the normalised posterior over `(image, class, rotation,
    translation)`, shape depends on sampling. `pmax` / `nr_significant`
    are per-image summaries that mirror
    `_rlnMaxValueProbDistribution` / `_rlnNrOfSignificantSamples` in
    RELION's `run_itNNN_data.star`.

    Arrays are kept as NumPy here because this container is used outside
    the dense-kernel JIT boundary for diagnostics.
    """

    # (N_img, K, n_rot, n_trans) for debugging; (N_img, M) where
    # M is the flattened hidden-variable dimension is also acceptable
    weights: np.ndarray

    # (N_img,) per-image max posterior
    pmax: np.ndarray

    # (N_img,) per-image count of non-negligible samples
    nr_significant: np.ndarray

    # (N_img,) per-image best class index
    best_class: np.ndarray

    # (N_img, 3) per-image best rotation (rot, tilt, psi in degrees)
    best_euler: np.ndarray

    # (N_img, 2) per-image best translation (x, y in pixels)
    best_trans: np.ndarray


def build_posterior_summary(
    weights: np.ndarray,
    significance_threshold: float = 1e-8,
) -> VdamPosterior:
    """Construct a `VdamPosterior` from a `(N, K, n_rot, n_trans)` weight
    tensor by extracting per-image Pmax, nr_significant, and argmax
    indices. The euler / trans outputs are index summaries; callers that
    need physical Euler angles or pixel offsets should convert them with
    the sampling grid they used for scoring.
    """
    if weights.ndim != 4:
        raise ValueError(f"weights must be 4D (N, K, n_rot, n_trans), got {weights.shape}")
    N = weights.shape[0]

    flat = weights.reshape(N, -1)
    pmax = flat.max(axis=1)
    nr_significant = (flat > significance_threshold).sum(axis=1)
    argmax = flat.argmax(axis=1)

    K, n_rot, n_trans = weights.shape[1:]
    best_class = argmax // (n_rot * n_trans)
    rest = argmax % (n_rot * n_trans)
    # Rotations and translations are opaque indices here; Phase 4 converts
    # them via the sampling grid into Euler triples + pixel offsets.
    best_rot_idx = rest // n_trans
    best_trans_idx = rest % n_trans

    return VdamPosterior(
        weights=weights,
        pmax=pmax,
        nr_significant=nr_significant,
        best_class=best_class,
        best_euler=np.column_stack([best_rot_idx.astype(np.float64), np.zeros(N), np.zeros(N)]),
        best_trans=np.column_stack([best_trans_idx.astype(np.float64), np.zeros(N)]),
    )

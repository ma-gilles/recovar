"""Ground-truth map metrics for InitialModel / ab-initio benchmarks.

Ab-initio volumes have an arbitrary global orientation, and cryo-EM maps also
have an unresolved handedness without extra information.  The helpers here keep
raw GT metrics separate from alignment-aware metrics so benchmark outputs can
report both without hiding which ambiguity was used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

DEFAULT_GT_ALIGN_HEALPIX_ORDER = 2
DEFAULT_GT_ALIGN_MAX_SHELL = 8
DEFAULT_GT_ALIGN_REFINE_ORDERS: tuple[int, ...] = (3, 4)
# Angular sigma (degrees) to keep around the coarse-best rotation when
# building each refinement-order grid. Generous enough to clear coarse-grid
# step error (HEALPix-2 step ≈ 15°) without exploding the refinement size.
DEFAULT_GT_ALIGN_REFINE_SIGMA_DEG: float = 30.0


@dataclass(frozen=True)
class VolumeAlignment:
    """Result of aligning a reconstructed volume to a GT reference."""

    aligned_volume: np.ndarray
    corr: float
    score: float
    rotation_index: int
    rotation_matrix: np.ndarray
    mirror_x: bool
    sign: int


def centered_correlation(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """Return centered real-space correlation for same-shaped volumes."""
    a = np.asarray(lhs, dtype=np.float64).ravel()
    b = np.asarray(rhs, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"Correlation inputs must have matching size, got {a.shape} and {b.shape}")
    a = a - float(a.mean())
    b = b - float(b.mean())
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def first_shell_below_threshold(fsc_values: np.ndarray, threshold: float) -> int:
    """Return the first FSC shell below ``threshold``, or -1 if none."""
    below = np.where(np.asarray(fsc_values, dtype=np.float64) < float(threshold))[0]
    return int(below[0]) if below.size else -1


def rotate_volume_about_center(volume: np.ndarray, rotation_matrix: np.ndarray, *, order: int = 1) -> np.ndarray:
    """Rotate ``volume`` around its geometric center with scipy interpolation.

    ``rotation_matrix`` is expressed in array-axis coordinates.  The search
    scans the same convention it applies, so callers do not need to interpret
    the matrix as a RELION or RECOVAR pose.
    """
    vol = np.asarray(volume, dtype=np.float64)
    if vol.ndim != 3 or len(set(vol.shape)) != 1:
        raise ValueError(f"Expected a cubic 3D volume, got shape {vol.shape}")
    matrix = np.asarray(rotation_matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation_matrix must have shape (3, 3), got {matrix.shape}")
    center = (np.asarray(vol.shape, dtype=np.float64) - 1.0) * 0.5
    inv_matrix = matrix.T
    offset = center - inv_matrix @ center
    return ndimage.affine_transform(
        vol,
        inv_matrix,
        offset=offset,
        output_shape=vol.shape,
        order=int(order),
        mode="constant",
        cval=0.0,
        prefilter=bool(order > 1),
    )


def alignment_score_box_size(volume_size: int, max_shell: int | None) -> int:
    """Return the compact box size needed to score shells up to ``max_shell``."""
    n = int(volume_size)
    if n <= 0:
        raise ValueError(f"volume_size must be positive, got {volume_size}")
    if max_shell is None or int(max_shell) <= 0:
        return n
    return min(n, max(9, 2 * int(max_shell) + 1))


def lowpass_volume_by_shell(
    volume: np.ndarray,
    max_shell: int | None,
    *,
    output_size: int | None = None,
) -> np.ndarray:
    """Low-pass a cubic volume by zeroing Fourier shells above ``max_shell``.

    If ``output_size`` is smaller than the input box, the low-frequency Fourier
    cube is cropped before inverse FFT.  Alignment scoring only uses low shells,
    so this preserves the information being scored while avoiding thousands of
    full-box interpolations during high-order rotation searches.
    """
    vol = np.asarray(volume, dtype=np.float64)
    if vol.ndim != 3 or len(set(vol.shape)) != 1:
        raise ValueError(f"Expected a cubic 3D volume, got shape {vol.shape}")
    if max_shell is None:
        return vol.copy()
    shell_limit = int(max_shell)
    if shell_limit <= 0:
        return vol.copy()

    n = int(vol.shape[0])
    out_n = n if output_size is None else int(output_size)
    if out_n <= 0 or out_n > n:
        raise ValueError(f"output_size must be in [1, {n}], got {output_size}")
    if out_n != n and out_n % 2 == 0:
        raise ValueError(f"cropped output_size must be odd, got {out_n}")

    ft = np.fft.fftn(vol)
    if out_n != n:
        shifted = np.fft.fftshift(ft)
        half = out_n // 2
        center = n // 2
        cropped = shifted[
            center - half : center + half + 1,
            center - half : center + half + 1,
            center - half : center + half + 1,
        ]
        ft = np.fft.ifftshift(cropped)

    freqs = np.fft.fftfreq(out_n) * out_n
    z, y, x = np.meshgrid(freqs, freqs, freqs, indexing="ij")
    shells = np.rint(np.sqrt(x * x + y * y + z * z)).astype(np.int32)
    ft[shells > shell_limit] = 0.0
    return np.real(np.fft.ifftn(ft))


def relion_alignment_rotations(healpix_order: int) -> np.ndarray:
    """Return the RELION/RECOVAR SO(3) search grid for GT alignment."""
    from recovar.em.sampling import get_rotation_grid_at_order

    return np.asarray(get_rotation_grid_at_order(int(healpix_order), matrices=True), dtype=np.float64)


def _angular_distance_deg(R_ref: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    """Return geodesic SO(3) angular distance from ``R_ref`` to each rotation, in degrees."""
    # angle θ satisfies trace(R_ref.T @ R) = 1 + 2 cos θ
    traces = np.einsum("ij,kij->k", R_ref, rotations)
    cos_theta = np.clip(0.5 * (traces - 1.0), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def _scan_grid(
    vol_lowpass: np.ndarray,
    ref_lowpass: np.ndarray,
    rotations: np.ndarray,
    *,
    mirror_options,
    sign_options,
    interpolation_order: int,
):
    """Inner search loop: pick best (mirror, sign, rot) by centered correlation."""
    best_score = -np.inf
    best_corr = -np.inf
    best_idx = 0
    best_mirror = False
    best_sign = 1
    for mirror_x in mirror_options:
        base = vol_lowpass[::-1, :, :] if mirror_x else vol_lowpass
        for idx, rotation in enumerate(rotations):
            candidate = rotate_volume_about_center(base, rotation, order=interpolation_order)
            corr = centered_correlation(candidate, ref_lowpass)
            for sign in sign_options:
                score = sign * corr
                if score > best_score:
                    best_score = float(score)
                    best_corr = float(corr)
                    best_idx = int(idx)
                    best_mirror = bool(mirror_x)
                    best_sign = int(sign)
    return best_score, best_corr, best_idx, best_mirror, best_sign


def align_volume_to_reference(
    volume: np.ndarray,
    reference: np.ndarray,
    rotations: np.ndarray,
    *,
    score_max_shell: int = 8,
    allow_mirror: bool = True,
    allow_sign: bool = False,
    interpolation_order: int = 1,
    refine_orders: tuple[int, ...] | None = None,
    refine_sigma_deg: float = DEFAULT_GT_ALIGN_REFINE_SIGMA_DEG,
) -> VolumeAlignment:
    """Align ``volume`` to ``reference`` by coarse rotation search + local refinement.

    Mirrors the EM convention: a coarse HEALPix grid picks the best
    orientation, then successive finer-order grids are searched LOCALLY
    around that pick (within ``refine_sigma_deg`` of the current best).
    The search score is centered correlation between low-pass filtered
    volumes.  ``corr`` returned is the centered correlation of the full
    aligned map against the full reference.

    If ``allow_mirror`` is true, an x-axis mirror is tested at the coarse
    stage to cover the cryo-EM handedness ambiguity; the chosen mirror is
    then locked through local refinement.  ``allow_sign`` is off by
    default because density sign is not a physical ab-initio pose
    ambiguity; enable it only to diagnose contrast convention issues.

    Pass ``refine_orders=None`` (or an empty tuple) to disable local
    refinement and reproduce the legacy coarse-only behavior.
    """
    vol = np.asarray(volume, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if vol.shape != ref.shape:
        raise ValueError(f"Volume and reference shapes must match, got {vol.shape} and {ref.shape}")
    if vol.ndim != 3 or len(set(vol.shape)) != 1:
        raise ValueError(f"Expected cubic 3D volumes, got shape {vol.shape}")

    rot_grid = np.asarray(rotations, dtype=np.float64)
    if rot_grid.ndim != 3 or rot_grid.shape[1:] != (3, 3):
        raise ValueError(f"rotations must have shape (n, 3, 3), got {rot_grid.shape}")
    if rot_grid.shape[0] == 0:
        raise ValueError("rotations must contain at least one matrix")

    score_box_size = alignment_score_box_size(vol.shape[0], score_max_shell)
    ref_score = lowpass_volume_by_shell(ref, score_max_shell, output_size=score_box_size)
    vol_score = lowpass_volume_by_shell(vol, score_max_shell, output_size=score_box_size)

    mirror_options = (False, True) if allow_mirror else (False,)
    sign_options = (1, -1) if allow_sign else (1,)

    # Coarse pass over the user-supplied grid.
    best_score, best_corr, best_idx, best_mirror, best_sign = _scan_grid(
        vol_score,
        ref_score,
        rot_grid,
        mirror_options=mirror_options,
        sign_options=sign_options,
        interpolation_order=interpolation_order,
    )
    best_rotation_matrix = np.asarray(rot_grid[best_idx], dtype=np.float64)

    # Local refinement: lock mirror/sign, build finer grids, keep rotations
    # within ``refine_sigma_deg`` of the current best, and pick the best.
    refine_orders = tuple(refine_orders or ())
    if refine_orders:
        for order in refine_orders:
            fine = relion_alignment_rotations(int(order))
            angles = _angular_distance_deg(best_rotation_matrix, fine)
            local_mask = angles <= float(refine_sigma_deg)
            if not local_mask.any():
                continue
            local_rotations = fine[local_mask]
            r_score, r_corr, r_idx_local, _, _ = _scan_grid(
                vol_score,
                ref_score,
                local_rotations,
                mirror_options=(best_mirror,),
                sign_options=(best_sign,),
                interpolation_order=interpolation_order,
            )
            if r_score > best_score:
                best_score = r_score
                best_corr = r_corr
                best_rotation_matrix = np.asarray(local_rotations[r_idx_local], dtype=np.float64)

    full_base = vol[::-1, :, :] if best_mirror else vol
    aligned = rotate_volume_about_center(full_base, best_rotation_matrix, order=interpolation_order)
    if best_sign < 0:
        aligned = -aligned
    full_corr = centered_correlation(aligned, ref)

    # Map best_rotation_matrix back to a coarse-grid index when possible.
    coarse_dist = _angular_distance_deg(best_rotation_matrix, rot_grid)
    rotation_index = int(np.argmin(coarse_dist))

    return VolumeAlignment(
        aligned_volume=np.asarray(aligned, dtype=np.float64),
        corr=float(full_corr),
        score=float(best_score),
        rotation_index=rotation_index,
        rotation_matrix=best_rotation_matrix,
        mirror_x=bool(best_mirror),
        sign=int(best_sign),
    )

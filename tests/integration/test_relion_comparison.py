"""Comparison helpers for RELION reference data.

These are utility functions (NOT pytest test functions) that will be called
by integration tests in later phases of the RELION-parity plan.

Each function compares a specific aspect of our EM output against RELION's
reference output extracted by scripts/extract_relion_reference.py.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_relion_reference(reference_dir: str, iteration: int) -> dict:
    """Load a single iteration's extracted RELION reference data.

    Parameters
    ----------
    reference_dir : str
        Directory containing iteration_NNN.npz files.
    iteration : int
        Iteration number.

    Returns
    -------
    dict
        Keys include: current_resolution, current_image_size, sigma2_noise,
        reference_sigma2, tau2, fsc, euler_angles, origins,
        nr_significant_samples, etc.
    """
    path = Path(reference_dir) / f"iteration_{iteration:03d}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"RELION reference not found: {path}. "
            f"Run scripts/extract_relion_reference.py first."
        )
    data = dict(np.load(str(path), allow_pickle=True))
    # Convert 0-d arrays back to scalars
    for key in ("current_resolution", "current_image_size", "pixel_size",
                "original_image_size", "n_optics_groups"):
        if key in data and data[key].ndim == 0:
            data[key] = data[key].item()
    return data


def compare_hard_assignments(
    our_assignments: np.ndarray,
    relion_angles: np.ndarray,
    rotation_grid: np.ndarray,
    tolerance_deg: float = 7.5,
) -> dict:
    """Compare best rotation per image against RELION's Euler angles.

    Parameters
    ----------
    our_assignments : (n_images,) int
        Indices into rotation_grid for our best rotation per image.
    relion_angles : (n_images, 3) float
        RELION Euler angles (Rot, Tilt, Psi) in degrees, ZYZ convention.
    rotation_grid : (n_rotations, 3, 3) float
        Our rotation matrices.
    tolerance_deg : float
        Angular distance threshold in degrees. Default 7.5 corresponds to
        one HEALPix step at order 3 (the standard coarse sampling).

    Returns
    -------
    dict
        Keys:
        - fraction_matching: float, fraction of images where angular distance
          is within tolerance_deg.
        - angular_distances_deg: (n_images,) array of angular distances.
        - median_distance_deg: float, median angular distance.
        - mean_distance_deg: float, mean angular distance.
    """
    from scipy.spatial.transform import Rotation

    n_images = len(our_assignments)
    assert relion_angles.shape == (n_images, 3), (
        f"Shape mismatch: our_assignments has {n_images} images, "
        f"relion_angles has {relion_angles.shape[0]}"
    )

    # Convert RELION Euler angles (Rot, Tilt, Psi) to rotation matrices.
    # RELION uses intrinsic ZYZ convention.
    relion_rots = Rotation.from_euler(
        "ZYZ", relion_angles, degrees=True
    ).as_matrix()  # (n_images, 3, 3)

    # Get our rotation matrices for the assigned rotations
    our_rots = rotation_grid[our_assignments]  # (n_images, 3, 3)

    # Compute angular distance between each pair of rotation matrices.
    # For rotation matrices R1, R2: angular distance = arccos((trace(R1^T R2) - 1) / 2)
    # Clip to handle numerical precision issues.
    product = np.einsum("nij,nij->n", our_rots, relion_rots)  # trace(R1^T R2) per image
    cos_angle = np.clip((product - 1.0) / 2.0, -1.0, 1.0)
    angular_distances_rad = np.arccos(cos_angle)
    angular_distances_deg = np.degrees(angular_distances_rad)

    fraction_matching = float(np.mean(angular_distances_deg <= tolerance_deg))

    result = {
        "fraction_matching": fraction_matching,
        "angular_distances_deg": angular_distances_deg,
        "median_distance_deg": float(np.median(angular_distances_deg)),
        "mean_distance_deg": float(np.mean(angular_distances_deg)),
    }

    logger.info(
        "Hard assignment comparison: %.1f%% within %.1f deg "
        "(median=%.2f deg, mean=%.2f deg)",
        fraction_matching * 100, tolerance_deg,
        result["median_distance_deg"], result["mean_distance_deg"],
    )
    return result


def compare_volumes(
    our_volume: np.ndarray,
    relion_mrc_path: str,
    volume_shape: tuple[int, int, int],
) -> dict:
    """Compute FSC between our volume (in Fourier space) and RELION's MRC file.

    Parameters
    ----------
    our_volume : (N,) complex array
        Our reconstructed volume in flattened Fourier space.
    relion_mrc_path : str
        Path to RELION's MRC volume file.
    volume_shape : tuple of 3 ints
        Shape of the 3D volume.

    Returns
    -------
    dict
        Keys:
        - fsc: (n_shells,) array of FSC values per shell.
        - fsc_0143_shell: int or None, shell where FSC drops below 0.143.
        - fsc_05_shell: int or None, shell where FSC drops below 0.5.
    """
    import mrcfile

    # Load RELION volume from MRC
    with mrcfile.open(relion_mrc_path, mode="r") as mrc:
        relion_vol_real = np.array(mrc.data, dtype=np.float32)

    assert relion_vol_real.shape == volume_shape, (
        f"Volume shape mismatch: expected {volume_shape}, "
        f"got {relion_vol_real.shape}"
    )

    # FFT to Fourier space (matching recovar convention)
    relion_vol_ft = np.fft.fftn(
        np.fft.ifftshift(relion_vol_real)
    ).reshape(-1).astype(np.complex64)

    # Compute FSC using recovar's GPU implementation
    try:
        import jax.numpy as jnp
        from recovar.reconstruction.regularization import get_fsc_gpu

        fsc = np.array(get_fsc_gpu(
            jnp.array(our_volume),
            jnp.array(relion_vol_ft),
            volume_shape,
        ))
    except ImportError:
        # Fallback: numpy-based FSC
        logger.warning("JAX not available, using numpy FSC (slower)")
        fsc = _fsc_numpy(our_volume, relion_vol_ft, volume_shape)

    # Find resolution shells
    fsc_0143_shell = _find_threshold_shell(fsc, 0.143)
    fsc_05_shell = _find_threshold_shell(fsc, 0.5)

    result = {
        "fsc": fsc,
        "fsc_0143_shell": fsc_0143_shell,
        "fsc_05_shell": fsc_05_shell,
    }

    logger.info(
        "Volume comparison: FSC=0.143 at shell %s, FSC=0.5 at shell %s",
        fsc_0143_shell, fsc_05_shell,
    )
    return result


def compare_resolution_trajectory(
    our_sizes: np.ndarray,
    relion_sizes: np.ndarray,
) -> dict:
    """Compare current_image_size sequences between our code and RELION.

    Parameters
    ----------
    our_sizes : (n_iterations,) int array
        Our current_image_size at each iteration.
    relion_sizes : (n_iterations,) int array
        RELION's rlnCurrentImageSize at each iteration.

    Returns
    -------
    dict
        Keys:
        - max_abs_diff: int, maximum absolute difference in image size.
        - per_iteration_diff: (n_iterations,) int array of differences.
        - matching: bool, True if all differences are 0.
    """
    n = min(len(our_sizes), len(relion_sizes))
    if len(our_sizes) != len(relion_sizes):
        logger.warning(
            "Trajectory length mismatch: ours=%d, relion=%d. "
            "Comparing first %d iterations.",
            len(our_sizes), len(relion_sizes), n,
        )

    ours = np.array(our_sizes[:n], dtype=np.int32)
    theirs = np.array(relion_sizes[:n], dtype=np.int32)
    diff = ours - theirs

    result = {
        "max_abs_diff": int(np.max(np.abs(diff))),
        "per_iteration_diff": diff,
        "matching": bool(np.all(diff == 0)),
    }

    logger.info(
        "Resolution trajectory: max_abs_diff=%d, matching=%s",
        result["max_abs_diff"], result["matching"],
    )
    return result


def compare_noise_spectra(
    our_noise: np.ndarray,
    relion_noise: np.ndarray,
    rtol: float = 0.1,
) -> dict:
    """Per-shell noise variance comparison.

    Parameters
    ----------
    our_noise : (n_shells,) float array
        Our per-shell noise variance estimate (sigma^2).
    relion_noise : (n_shells,) float array
        RELION's rlnSigma2Noise per shell.
    rtol : float
        Relative tolerance for "matching" threshold.

    Returns
    -------
    dict
        Keys:
        - max_relative_diff: float, maximum relative difference across shells.
        - per_shell_relative_diff: (n_shells,) array.
        - matching: bool, True if all shells are within rtol.
        - worst_shell: int, shell index with largest relative difference.
    """
    n = min(len(our_noise), len(relion_noise))
    ours = np.array(our_noise[:n], dtype=np.float64)
    theirs = np.array(relion_noise[:n], dtype=np.float64)

    # Relative difference, avoiding division by zero
    denom = np.maximum(np.abs(theirs), 1e-30)
    rel_diff = np.abs(ours - theirs) / denom

    # Exclude shells where both are near zero (no signal)
    both_small = (np.abs(ours) < 1e-20) & (np.abs(theirs) < 1e-20)
    rel_diff[both_small] = 0.0

    max_rel = float(np.max(rel_diff))
    worst_shell = int(np.argmax(rel_diff))

    result = {
        "max_relative_diff": max_rel,
        "per_shell_relative_diff": rel_diff,
        "matching": bool(max_rel <= rtol),
        "worst_shell": worst_shell,
    }

    logger.info(
        "Noise spectra comparison: max_rel_diff=%.4f at shell %d, "
        "matching(rtol=%.2f)=%s",
        max_rel, worst_shell, rtol, result["matching"],
    )
    return result


def compare_prior_spectra(
    our_prior: np.ndarray,
    relion_prior: np.ndarray,
    rtol: float = 0.1,
) -> dict:
    """Per-shell signal prior (tau^2 or reference_sigma2) comparison.

    Parameters
    ----------
    our_prior : (n_shells,) float array
        Our per-shell signal prior estimate.
    relion_prior : (n_shells,) float array
        RELION's rlnReferenceSigma2 or rlnReferenceTau2 per shell.
    rtol : float
        Relative tolerance for "matching" threshold.

    Returns
    -------
    dict
        Keys:
        - max_relative_diff: float, maximum relative difference across shells.
        - per_shell_relative_diff: (n_shells,) array.
        - matching: bool, True if all shells are within rtol.
        - worst_shell: int, shell index with largest relative difference.
    """
    n = min(len(our_prior), len(relion_prior))
    ours = np.array(our_prior[:n], dtype=np.float64)
    theirs = np.array(relion_prior[:n], dtype=np.float64)

    # Relative difference, avoiding division by zero
    denom = np.maximum(np.abs(theirs), 1e-30)
    rel_diff = np.abs(ours - theirs) / denom

    # Exclude shells where both are near zero (beyond current resolution)
    both_small = (np.abs(ours) < 1e-20) & (np.abs(theirs) < 1e-20)
    rel_diff[both_small] = 0.0

    max_rel = float(np.max(rel_diff))
    worst_shell = int(np.argmax(rel_diff))

    result = {
        "max_relative_diff": max_rel,
        "per_shell_relative_diff": rel_diff,
        "matching": bool(max_rel <= rtol),
        "worst_shell": worst_shell,
    }

    logger.info(
        "Prior spectra comparison: max_rel_diff=%.4f at shell %d, "
        "matching(rtol=%.2f)=%s",
        max_rel, worst_shell, rtol, result["matching"],
    )
    return result


def compare_significant_samples(
    our_counts: np.ndarray,
    relion_counts: np.ndarray,
) -> dict:
    """Compare per-image number of significant samples.

    Parameters
    ----------
    our_counts : (n_images,) int array
        Our rlnNrOfSignificantSamples equivalent.
    relion_counts : (n_images,) int array
        RELION's rlnNrOfSignificantSamples.

    Returns
    -------
    dict
        Keys:
        - median_ours: float
        - median_relion: float
        - correlation: float, Pearson correlation of log counts.
        - ks_statistic: float, KS test statistic between distributions.
    """
    from scipy import stats

    ours = np.array(our_counts, dtype=np.float64)
    theirs = np.array(relion_counts, dtype=np.float64)

    # Log-space correlation (significant samples span orders of magnitude)
    log_ours = np.log1p(ours)
    log_theirs = np.log1p(theirs)
    correlation = float(np.corrcoef(log_ours, log_theirs)[0, 1])

    # KS test
    ks_stat, ks_pval = stats.ks_2samp(ours, theirs)

    result = {
        "median_ours": float(np.median(ours)),
        "median_relion": float(np.median(theirs)),
        "correlation": correlation,
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pval),
    }

    logger.info(
        "Significant samples: median ours=%.0f, relion=%.0f, "
        "corr=%.3f, KS=%.3f",
        result["median_ours"], result["median_relion"],
        correlation, ks_stat,
    )
    return result


# --- Internal helpers ---


def _find_threshold_shell(fsc: np.ndarray, threshold: float):
    """Find first shell where FSC drops below threshold (after initially being above)."""
    above = fsc >= threshold
    if not np.any(above):
        return 0
    # Find first shell after initial above region where FSC drops below
    for i in range(1, len(fsc)):
        if above[i - 1] and not above[i]:
            return i
    return None  # Never drops below


def _fsc_numpy(vol1: np.ndarray, vol2: np.ndarray, volume_shape: tuple) -> np.ndarray:
    """Numpy fallback for FSC computation (no GPU needed)."""
    from recovar.core.fourier_transform_utils import get_grid_of_radial_distances

    radii = get_grid_of_radial_distances(volume_shape, scaled=False).astype(int).reshape(-1)
    max_r = volume_shape[0] // 2

    fsc = np.zeros(max_r + 1)
    for r in range(max_r + 1):
        mask = radii == r
        if not np.any(mask):
            continue
        a = vol1[mask]
        b = vol2[mask]
        num = np.real(np.sum(np.conj(a) * b))
        den = np.sqrt(np.sum(np.abs(a) ** 2) * np.sum(np.abs(b) ** 2))
        fsc[r] = num / den if den > 0 else 0.0

    return fsc

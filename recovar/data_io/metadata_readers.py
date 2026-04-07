"""
Extract poses and CTF parameters directly from RELION .star and cryoSPARC .cs files.

This module eliminates the need for cryoDRGN preprocessing. The output formats
match exactly what ``load_utils.load_ctf_params`` and ``load_utils.load_poses``
return, so downstream code (``load_cryodrgn_dataset``) can consume either source
interchangeably.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as SciPyRot

from recovar.data_io.starfile import StarFile
from recovar.utils import R_from_relion

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# STAR file parsing
# ---------------------------------------------------------------------------


def parse_poses_from_star(
    star_path: str,
    D: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract rotation matrices and translations from a RELION .star file.

    Args:
        star_path: Path to .star file.
        D: Target image dimension in pixels (used for translation normalisation).

    Returns:
        rotations: ``(N, 3, 3)`` rotation matrices (float64).
        translations: ``(N, 2)`` translations in **fractional** units (|val| <= 1).
            Multiply by *D* to obtain pixel offsets.
    """
    sf = StarFile.load(star_path)
    n = len(sf)

    # --- Euler angles → rotation matrices ---
    rot_col = sf.get_optics_values("_rlnAngleRot", dtype=np.float64)
    tilt_col = sf.get_optics_values("_rlnAngleTilt", dtype=np.float64)
    psi_col = sf.get_optics_values("_rlnAnglePsi", dtype=np.float64)

    if rot_col is None or tilt_col is None or psi_col is None:
        raise ValueError(
            "STAR file must contain _rlnAngleRot, _rlnAngleTilt, _rlnAnglePsi "
            "to extract poses. Provide a --poses pkl file instead."
        )

    euler = np.stack([rot_col, tilt_col, psi_col], axis=1)  # (N, 3)
    rotations = R_from_relion(euler, degrees=True)  # (N, 3, 3)

    # --- Translations ---
    # RELION 3.1: _rlnOriginXAngst / _rlnOriginYAngst (Angstroms)
    # RELION 3.0: _rlnOriginX / _rlnOriginY (pixels)
    tx = sf.get_optics_values("_rlnOriginXAngst", dtype=np.float64)
    ty = sf.get_optics_values("_rlnOriginYAngst", dtype=np.float64)

    if tx is not None and ty is not None:
        # Convert Angstroms → fractional via pixel size
        apix = sf.apix  # per-particle pixel sizes
        if apix is None:
            raise ValueError(
                "STAR file has _rlnOriginXAngst but no _rlnImagePixelSize. Cannot convert translations to pixel units."
            )
        apix = apix.astype(np.float64)
        trans_pixels = np.stack([tx / apix, ty / apix], axis=1)  # (N, 2)
    else:
        # Try RELION 3.0 pixel-unit columns
        tx = sf.get_optics_values("_rlnOriginX", dtype=np.float64)
        ty = sf.get_optics_values("_rlnOriginY", dtype=np.float64)
        if tx is not None and ty is not None:
            trans_pixels = np.stack([tx, ty], axis=1)
        else:
            logger.warning("No translation columns found in STAR file; assuming zero shifts.")
            trans_pixels = np.zeros((n, 2), dtype=np.float64)

    # Pixel → fractional (same convention as cryoDRGN pkl files)
    resolution = sf.resolution  # per-particle image sizes
    if resolution is not None:
        trans_fractional = trans_pixels / resolution.astype(np.float64).reshape(-1, 1)
    else:
        trans_fractional = trans_pixels / float(D)

    return rotations, trans_fractional


def parse_ctf_from_star(
    star_path: str,
    D: int,
) -> np.ndarray:
    """Extract CTF parameters from a RELION .star file.

    Args:
        star_path: Path to .star file.
        D: Target image dimension in pixels. Pixel size is adjusted
           for the ratio ``original_D / D``.

    Returns:
        ``(N, 8)`` array with columns
        ``[Apix, DFU, DFV, DFANG, VOLT, CS, W, PHASE_SHIFT]``.
        This matches the output format of ``load_utils.load_ctf_params``.
    """
    sf = StarFile.load(star_path)
    n = len(sf)

    # Required fields
    dfu = sf.get_optics_values("_rlnDefocusU", dtype=np.float64)
    dfv = sf.get_optics_values("_rlnDefocusV", dtype=np.float64)
    dfang = sf.get_optics_values("_rlnDefocusAngle", dtype=np.float64)
    volt = sf.get_optics_values("_rlnVoltage", dtype=np.float64)
    cs = sf.get_optics_values("_rlnSphericalAberration", dtype=np.float64)
    w = sf.get_optics_values("_rlnAmplitudeContrast", dtype=np.float64)

    for name, val in [
        ("_rlnDefocusU", dfu),
        ("_rlnDefocusV", dfv),
        ("_rlnDefocusAngle", dfang),
        ("_rlnVoltage", volt),
        ("_rlnSphericalAberration", cs),
        ("_rlnAmplitudeContrast", w),
    ]:
        if val is None:
            raise ValueError(f"STAR file missing required CTF field {name}. Provide a --ctf pkl file instead.")

    # Optional: phase shift (default 0)
    phase = sf.get_optics_values("_rlnPhaseShift", dtype=np.float64)
    if phase is None:
        phase = np.zeros(n, dtype=np.float64)

    # Pixel size adjustment
    orig_apix = sf.apix
    orig_D = sf.resolution
    if orig_apix is not None and orig_D is not None:
        orig_apix = orig_apix.astype(np.float64)
        orig_D = orig_D.astype(np.float64)
        new_apix = orig_D * orig_apix / float(D)
    elif orig_apix is not None:
        # Have Apix (e.g. from RELION 3.0 Magnification/DetectorPixelSize)
        # but no _rlnImageSize.  Assume the STAR pixel size describes the
        # images at their native resolution; if images are later downsampled,
        # the pipeline adjusts Apix separately.
        orig_apix = orig_apix.astype(np.float64)
        new_apix = orig_apix
        logger.info(
            "No _rlnImageSize in STAR; using Apix=%.4f from "
            "Magnification/DetectorPixelSize (assuming native resolution).",
            orig_apix[0],
        )
    else:
        logger.warning(
            "Could not determine pixel size from STAR file. "
            "Using Apix=1.0; this will be incorrect. "
            "Provide a --ctf pkl file or a RELION 3.1+ .star with optics table."
        )
        new_apix = np.ones(n, dtype=np.float64)

    # Assemble: [Apix, DFU, DFV, DFANG, VOLT, CS, W, PHASE_SHIFT]
    ctf = np.column_stack([new_apix, dfu, dfv, dfang, volt, cs, w, phase])

    logger.info(
        "Parsed CTF from STAR: %d particles, Apix=%.4f (first), DFU range=[%.0f, %.0f]",
        n,
        ctf[0, 0],
        dfu.min(),
        dfu.max(),
    )
    return ctf.astype(np.float64)


# ---------------------------------------------------------------------------
# cryoSPARC .cs file parsing
# ---------------------------------------------------------------------------


def _load_cs(cs_path: str) -> np.ndarray:
    """Load a cryoSPARC .cs file (NumPy structured array)."""
    data = np.load(cs_path)
    if data.dtype.names is None:
        raise ValueError(f"{cs_path} does not appear to be a cryoSPARC .cs file")
    return data


def parse_poses_from_cs(
    cs_path: str,
    D: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract rotation matrices and translations from a cryoSPARC .cs file.

    CryoSPARC stores rotations as 3-vector exponential maps (axis-angle / Rodrigues).
    Translations are in pixels.

    Args:
        cs_path: Path to .cs file.
        D: Target image dimension in pixels.

    Returns:
        rotations: ``(N, 3, 3)`` rotation matrices.
        translations: ``(N, 2)`` in fractional units.
    """
    data = _load_cs(cs_path)
    n = len(data)

    # --- Rotations (exponential map → rotation matrix) ---
    pose_key = None
    for key in ("alignments3D/pose", "alignments_class_0/pose"):
        if key in data.dtype.names:
            pose_key = key
            break

    if pose_key is None:
        raise ValueError("CS file has no alignments3D/pose field. Provide a --poses pkl file instead.")

    rotvecs = data[pose_key].astype(np.float64)  # (N, 3)
    rot_matrices = SciPyRot.from_rotvec(rotvecs).as_matrix()  # (N, 3, 3)
    # Transpose to match cryoDRGN convention
    rot_matrices = rot_matrices.transpose(0, 2, 1)

    # --- Translations ---
    shift_key = None
    for key in ("alignments3D/shift", "alignments_class_0/shift"):
        if key in data.dtype.names:
            shift_key = key
            break

    if shift_key is not None:
        # cryoSPARC stores alignments*/shift in pixel units. Keep the values
        # as-is and only convert to the fractional convention expected by recovar.
        trans_pixels = data[shift_key].astype(np.float64)
    else:
        logger.warning("No translation field found in CS file; assuming zero shifts.")
        trans_pixels = np.zeros((n, 2), dtype=np.float64)

    # Pixel → fractional
    trans_fractional = trans_pixels / float(D)

    return rot_matrices, trans_fractional


def parse_ctf_from_cs(
    cs_path: str,
    D: int,
) -> np.ndarray:
    """Extract CTF parameters from a cryoSPARC .cs file.

    Args:
        cs_path: Path to .cs file.
        D: Target image dimension in pixels.

    Returns:
        ``(N, 8)`` array: ``[Apix, DFU, DFV, DFANG, VOLT, CS, W, PHASE_SHIFT]``.
    """
    data = _load_cs(cs_path)
    n = len(data)

    # Required CTF fields
    required = {
        "ctf/df1_A": "DefocusU",
        "ctf/df2_A": "DefocusV",
        "ctf/df_angle_rad": "DefocusAngle",
        "ctf/accel_kv": "Voltage",
        "ctf/cs_mm": "SphericalAberration",
        "ctf/amp_contrast": "AmplitudeContrast",
    }
    for field, desc in required.items():
        if field not in data.dtype.names:
            raise ValueError(
                f"CS file missing required CTF field '{field}' ({desc}). Provide a --ctf pkl file instead."
            )

    dfu = data["ctf/df1_A"].astype(np.float64)
    dfv = data["ctf/df2_A"].astype(np.float64)
    dfang_rad = data["ctf/df_angle_rad"].astype(np.float64)
    volt = data["ctf/accel_kv"].astype(np.float64)
    cs = data["ctf/cs_mm"].astype(np.float64)
    w = data["ctf/amp_contrast"].astype(np.float64)

    # Convert radians → degrees
    dfang = np.degrees(dfang_rad)

    # Phase shift (optional, in radians)
    if "ctf/phase_shift_rad" in data.dtype.names:
        phase = np.degrees(data["ctf/phase_shift_rad"].astype(np.float64))
    else:
        phase = np.zeros(n, dtype=np.float64)

    # Pixel size
    if "blob/psize_A" in data.dtype.names:
        orig_apix = data["blob/psize_A"].astype(np.float64)
    else:
        logger.warning("CS file has no blob/psize_A; using Apix=1.0")
        orig_apix = np.ones(n, dtype=np.float64)

    if "blob/shape" in data.dtype.names:
        orig_D = data["blob/shape"][:, 0].astype(np.float64)
    else:
        orig_D = np.full(n, float(D), dtype=np.float64)

    new_apix = orig_D * orig_apix / float(D)

    ctf = np.column_stack([new_apix, dfu, dfv, dfang, volt, cs, w, phase])

    logger.info(
        "Parsed CTF from CS: %d particles, Apix=%.4f (first), DFU range=[%.0f, %.0f]",
        n,
        ctf[0, 0],
        dfu.min(),
        dfu.max(),
    )
    return ctf.astype(np.float64)


# ---------------------------------------------------------------------------
# Auto-dispatch helpers
# ---------------------------------------------------------------------------


def can_extract_poses(filepath: str) -> bool:
    """Return True if poses can be auto-extracted from this file type."""
    return filepath.lower().endswith((".star", ".cs"))


def auto_parse_poses(
    filepath: str,
    D: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Auto-extract poses from STAR or CS file based on extension."""
    lower = filepath.lower()
    if lower.endswith(".star"):
        return parse_poses_from_star(filepath, D)
    elif lower.endswith(".cs"):
        return parse_poses_from_cs(filepath, D)
    else:
        raise ValueError(f"Cannot auto-extract poses from '{filepath}'. Supported formats: .star, .cs")


def auto_parse_ctf(
    filepath: str,
    D: int,
) -> np.ndarray:
    """Auto-extract CTF parameters from STAR or CS file based on extension."""
    lower = filepath.lower()
    if lower.endswith(".star"):
        return parse_ctf_from_star(filepath, D)
    elif lower.endswith(".cs"):
        return parse_ctf_from_cs(filepath, D)
    else:
        raise ValueError(f"Cannot auto-extract CTF from '{filepath}'. Supported formats: .star, .cs")

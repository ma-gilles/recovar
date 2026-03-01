"""
Utilities for loading CTF parameters and pose information from pickle files.
Equivalent to cryodrgn/load
"""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np

from recovar import utils

logger = logging.getLogger(__name__)


def _normalize_pose_indices(ind: np.ndarray, n_total: int) -> np.ndarray:
    """Normalize pose-selection indices (integer ids or boolean mask)."""
    arr = np.asarray(ind)
    if arr.dtype == bool:
        if arr.ndim != 1:
            raise ValueError("Pose index boolean mask must be 1D")
        if arr.size != int(n_total):
            raise ValueError(
                f"Pose index boolean mask length {arr.size} must match number of poses {int(n_total)}"
            )
        return np.flatnonzero(arr).astype(np.int32, copy=False)

    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError("Pose indices must be 1D")
    if arr.dtype.kind not in ("i", "u"):
        raise TypeError("Pose indices must be integer or boolean mask")

    arr = arr.astype(np.int64, copy=False).reshape(-1)
    if arr.size > 0:
        if np.any(arr < 0):
            raise IndexError("Pose indices contain negative values")
        if np.any(arr >= int(n_total)):
            raise IndexError(f"Pose indices contain values >= number of poses ({int(n_total)})")
    return arr.astype(np.int32, copy=False)


def print_ctf_params(params: np.ndarray) -> None:
    """Log CTF parameters in a readable format.
    
    Args:
        params: Array of 9 CTF parameters
    """
    if len(params) != 9:
        raise ValueError(f"Expected 9 CTF parameters, got {len(params)}")
    
    param_names = [
        ("Image size (pix)", int(params[0])),   
        ("A/pix", params[1]),
        ("DefocusU (A)", params[2]),
        ("DefocusV (A)", params[3]),
        ("Dfang (deg)", params[4]),
        ("voltage (kV)", params[5]),
        ("cs (mm)", params[6]),
        ("w", params[7]),
        ("Phase shift (deg)", params[8])
    ]
    
    for name, value in param_names:
        logger.info("%18s: %s", name, value)


def load_ctf_params(D: int, ctf_params_pkl: str) -> np.ndarray:
    """Load and adjust CTF parameters for a given image size.
    
    Args:
        D: Target image dimension (must be even)
        ctf_params_pkl: Path to pickle file containing CTF parameters
        
    Returns:
        CTF parameters array with shape (N, 8), excluding image size column
    """
    if D % 2 != 0:
        raise ValueError(f"Image dimension D must be even, got {D}")
    
    # Load parameters from pickle
    ctf_params = np.asarray(utils.pickle_load(ctf_params_pkl))
    if ctf_params.ndim != 2:
        raise ValueError(f"CTF parameters must be a 2D array of shape (N, 9), got ndim={ctf_params.ndim}")
    if ctf_params.shape[0] == 0:
        raise ValueError("CTF parameters are empty")
    if ctf_params.shape[1] != 9:
        raise ValueError(f"Expected 9 CTF parameters per image, got {ctf_params.shape[1]}")
    if not np.issubdtype(ctf_params.dtype, np.number):
        raise ValueError("CTF parameters must be numeric")
    if not np.all(np.isfinite(ctf_params)):
        raise ValueError("CTF parameters contain non-finite values (NaN/Inf)")
    
    # Adjust pixel size based on original and target dimensions
    original_D = ctf_params[0, 0]
    original_Apix = ctf_params[0, 1]
    new_Apix = original_D * original_Apix / D
    
    # Update all rows with new dimensions
    ctf_params[:, 0] = D
    ctf_params[:, 1] = new_Apix
    
    # Log the first entry as example
    print_ctf_params(ctf_params[0])
    
    # Return without the image size column
    return ctf_params[:, 1:]


def load_poses(
    infile: Union[str, List[str]],
    Nimg: int,
    D: int,
    ind: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
    """Load pose information (rotations and translations) from pickle files.
    
    Args:
        infile: Path to pickle file(s). Can be:
                - Single file containing (rotations, translations) tuple
                - Single file containing rotations only
                - List of two files: [rotations_file, translations_file]
        Nimg: Expected number of images
        D: Image dimension in pixels
        ind: Optional index array to filter poses
        
    Returns:
        Tuple of (rotations, translations, D) where:
            - rotations: (Nimg, 3, 3) array of rotation matrices
            - translations: (Nimg, 2) array of translations in pixels, or None
            - D: Image dimension (passthrough)
    """
    # Handle file input
    if isinstance(infile, str):
        infile = [infile]
    
    if len(infile) not in (1, 2):
        raise ValueError(f"Expected 1 or 2 input files, got {len(infile)}")
    
    # Load data from pickle file(s)
    if len(infile) == 2:
        # Separate rotation and translation files
        rot_data = utils.pickle_load(infile[0])
        trans_data = utils.pickle_load(infile[1])
        poses = (rot_data, trans_data)
    else:
        # Single file - may contain tuple or just rotations
        poses = utils.pickle_load(infile[0])
        if not isinstance(poses, tuple):
            poses = (poses,)
    
    # Extract rotations
    rots = np.asarray(poses[0])
    if not np.issubdtype(rots.dtype, np.number):
        raise ValueError("Rotation array must be numeric")
    if not np.all(np.isfinite(rots)):
        raise ValueError("Rotation array contains non-finite values (NaN/Inf)")

    # Validate rotation shape
    expected_rot_shape = (Nimg, 3, 3)
    pose_ind = None
    if ind is not None:
        # Always honor explicit pose selection. This is critical for duplicate or
        # reordered index sets where len(ind) may still equal Nimg.
        pose_ind = _normalize_pose_indices(ind, n_total=len(rots))
        rots = rots[pose_ind]

    if rots.shape != expected_rot_shape:
        raise ValueError(f"Rotation array has shape {rots.shape}, expected {expected_rot_shape}")
    
    # Extract translations if available
    trans = None
    if len(poses) == 2:
        trans = np.asarray(poses[1])
        if not np.issubdtype(trans.dtype, np.number):
            raise ValueError("Translation array must be numeric")
        if not np.all(np.isfinite(trans)):
            raise ValueError("Translation array contains non-finite values (NaN/Inf)")

        if len(poses[0]) != len(trans):
            raise ValueError(
                f"Rotation/translation count mismatch: {len(poses[0])} rotations vs {len(trans)} translations"
            )

        # Apply the same pose subset as rotations to preserve row alignment.
        if pose_ind is not None:
            trans = trans[pose_ind]
        
        # Validate translation shape
        expected_trans_shape = (Nimg, 2)
        if trans.shape != expected_trans_shape:
            raise ValueError(
                f"Translation array has shape {trans.shape}, expected {expected_trans_shape}"
            )
        
        # Check that translations are in fractional units (0-1 range)
        if not np.all(np.abs(trans) <= 1):
            raise ValueError(
                "Translations must be in fractional units with |value| <= 1. "
                "Old pose format with pixel units is not supported."
            )
        
        # Convert from fractional units to pixels
        trans = trans * D
    else:
        logger.warning("No translations found in pose file")
    
    return rots, trans, D

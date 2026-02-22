"""
Utilities for loading CTF parameters and pose information from pickle files.
Equivalent to cryodrgn/load
"""

from recovar import utils
import logging
import numpy as np
from typing import Optional, Tuple, Union, List

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
        logger.info(f"{name:18s}: {value}")


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
    ctf_params = utils.pickle_load(ctf_params_pkl)
    
    if ctf_params.shape[1] != 9:
        raise ValueError(f"Expected 9 CTF parameters per image, got {ctf_params.shape[1]}")
    
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
    rots = poses[0]

    # Validate rotation shape
    expected_rot_shape = (Nimg, 3, 3)
    pose_ind = None
    if ind is not None and len(rots) > Nimg:
        pose_ind = _normalize_pose_indices(ind, n_total=len(rots))
        rots = rots[pose_ind]

    if rots.shape != expected_rot_shape:
        raise ValueError(f"Rotation array has shape {rots.shape}, expected {expected_rot_shape}")
    
    # Extract translations if available
    trans = None
    if len(poses) == 2:
        trans = poses[1]

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
        if not np.all(trans <= 1):
            raise ValueError(
                "Translations must be in fractional units (0-1 range). "
                "Old pose format with pixel units is not supported."
            )
        
        # Convert from fractional units to pixels
        trans = trans * D
    else:
        logger.warning("No translations found in pose file")
    
    return rots, trans, D

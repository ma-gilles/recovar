"""General-purpose utilities: I/O, array manipulation, timing."""

import functools
import logging
import os
import pickle
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import mrcfile
import more_itertools
import numpy as np
import pandas as pd
import psutil
import starfile

import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar import core

logger = logging.getLogger(__name__)
_module_logger = logger  # alias for use in functions where 'logger' is a parameter


@functools.partial(jax.jit, static_argnums=[1, 2])
def make_radial_image(average_image_PS, image_shape, extend_last_frequency=True):
    if extend_last_frequency:
        last_noise_band = average_image_PS[-1]
        average_image_PS = jnp.concatenate([average_image_PS, last_noise_band * jnp.ones_like(average_image_PS)])
    radial_distances = (
        fourier_transform_utils.get_grid_of_radial_distances(image_shape, scaled=False, frequency_shift=0)
        .astype(int)
        .reshape(-1)
    )
    prior = jnp.asarray(average_image_PS)[radial_distances]
    return prior


batch_make_radial_image = jax.vmap(make_radial_image, in_axes=(0, None, None))


@functools.partial(jax.jit, static_argnums=[1, 2])
def make_radial_image_half(average_image_PS, image_shape, extend_last_frequency=True):
    """Like :func:`make_radial_image` but outputs rfft-packed half-spectrum.

    Uses ``get_grid_of_radial_distances_real`` so the output has shape
    ``(H * (W//2+1),)`` instead of ``(H * W,)``.
    """
    if extend_last_frequency:
        last_noise_band = average_image_PS[-1]
        average_image_PS = jnp.concatenate([average_image_PS, last_noise_band * jnp.ones_like(average_image_PS)])
    radial_distances = (
        fourier_transform_utils.get_grid_of_radial_distances_real(image_shape, scaled=False, frequency_shift=0)
        .astype(int)
        .reshape(-1)
    )
    prior = jnp.asarray(average_image_PS)[radial_distances]
    return prior


batch_make_radial_image_half = jax.vmap(make_radial_image_half, in_axes=(0, None, None))


def index_batch_iter(n_units, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if n_units < 0:
        raise ValueError("n_units must be >= 0")
    return more_itertools.chunked(np.arange(n_units), batch_size)


def subset_batch_iter(subset_indices, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    return more_itertools.chunked(subset_indices, batch_size)


def subset_and_indices_batch_iter(subset_indices, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    return zip(
        subset_batch_iter(np.arange(len(subset_indices)), batch_size), subset_batch_iter(subset_indices, batch_size)
    )


def estimate_variance(u, s):
    var = np.sum(np.abs(u) ** 2 * s[..., None], axis=0)
    return var


# inner psutil function
def get_process_memory_used():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return int(mem_info.rss / 1e9)


GPU_MEMORY_LIMIT = None


def set_gpu_memory_limit(gb):
    """Set the GPU memory limit (in GB) used for batch-size calculations."""
    global GPU_MEMORY_LIMIT
    GPU_MEMORY_LIMIT = gb


def get_gpu_memory_total(device=0):
    if GPU_MEMORY_LIMIT is not None:
        return GPU_MEMORY_LIMIT
    if jax_has_gpu():
        mem_stats = jax.local_devices()[device].memory_stats()
        if mem_stats is not None and "bytes_limit" in mem_stats:
            return int(mem_stats["bytes_limit"] / 1e9)
        else:
            logger.warning(
                "get_gpu_memory_total: Could not read GPU memory_stats bytes_limit via JAX (memory_stats() returned None/empty). Falling back to 80GB."
            )
            return int(80)
    else:
        available_gb = int(psutil.virtual_memory().available / 1e9)
        # Use half of available RAM as a safety margin for CPU-only mode
        cpu_limit = max(1, available_gb // 2)
        logger.info(
            "GPU not found. Using %d GB (half of %d GB available RAM) for batching on CPU.", cpu_limit, available_gb
        )
        return cpu_limit


def get_gpu_memory_used(device=0):
    if jax_has_gpu():
        mem_stats = jax.local_devices()[device].memory_stats()
        if mem_stats is not None and "bytes_in_use" in mem_stats:
            return int(mem_stats["bytes_in_use"] / 1e9)
        else:
            logger.warning(
                "get_gpu_memory_used: Could not read GPU memory_stats bytes_in_use via JAX (memory_stats() returned None/empty). Returning 0."
            )
            return int(0)
    else:
        return int(0)


def get_peak_gpu_memory_used(device=0):
    if jax_has_gpu():
        mem_stats = jax.local_devices()[device].memory_stats()
        if mem_stats is not None and "peak_bytes_in_use" in mem_stats:
            return int(mem_stats["peak_bytes_in_use"] / 1e9)
        else:
            logger.warning(
                "get_peak_gpu_memory_used: Could not read GPU memory_stats peak_bytes_in_use via JAX (memory_stats() returned None/empty). Returning 0."
            )
            return int(0)
    else:
        return int(0)


def report_memory_device(device=0, logger=None):
    output_str = f"GPU mem in use:{get_gpu_memory_used(device)}; peak:{get_peak_gpu_memory_used(device)}; total available:{get_gpu_memory_total(device)}, process mem in use:{get_process_memory_used()}"
    _log = logger if logger is not None else _module_logger
    _log.info(output_str)


def get_size_in_gb(x):
    return x.size * x.itemsize / 1e9


def write_mrc(file, ar, voxel_size=None):
    """Write a real-space volume to MRC in the cryosparc/cryoDRGN axis convention.

    Use this paired with :func:`load_mrc` for round-trip safe MRC I/O.
    Do NOT use raw ``mrcfile.new(...).set_data(vol)`` — it omits the
    ``(2, 1, 0)`` transpose and produces files in the wrong axis order.
    """
    # This is to agree with the cryosparc/cryoDRGN convention
    if ar.ndim == 3 and np.isclose(ar.shape, ar.shape[0]).all():
        ar = np.transpose(ar, (2, 1, 0))

    with mrcfile.new(file, overwrite=True) as mrc:
        mrc.set_data(ar.real.astype(np.float32))
        if voxel_size is not None:
            mrc.voxel_size = voxel_size


def load_mrc(filepath, return_voxel_size=False):
    """Load a real-space volume from MRC in the cryosparc/cryoDRGN axis convention.

    Use this paired with :func:`write_mrc` for round-trip safe MRC I/O.
    Do NOT use raw ``mrcfile.open(...)`` — it skips the ``(2, 1, 0)``
    transpose, leaving the data in the wrong axis order, which silently
    corrupts downstream FFT / projection / FSC calculations.
    """
    with mrcfile.open(filepath) as mrc:
        data = mrc.data
        if return_voxel_size:
            voxel_size = mrc.voxel_size

    # This is to agree with the cryosparc/cryoDRGN convention
    if data.ndim == 3 and np.isclose(data.shape, data.shape[0]).all():
        data = np.transpose(data, (2, 1, 0))

    if return_voxel_size:
        return data, voxel_size

    return data


def relion_volume_to_recovar(vol_relion):
    """Convert a RELION-convention volume into recovar's convention.

    RELION and recovar use different 3D coordinate frames for real-space
    volumes::

        vol_recovar = -np.transpose(vol_relion, (2, 1, 0))

    i.e. negate AND swap X<->Z. The negation is paired with RELION's
    Euler convention (see :func:`R_to_relion` / :func:`R_from_relion`),
    which is why ``R_to_relion`` is correct as written despite the
    volume axis flip — they cancel out at the projection step.

    This helper exists so that comparison code that loads RELION's MRC
    outputs (e.g. via ``mrcfile.open``) can convert into recovar's
    internal frame for FSC / direct comparison. Verified CC=0.998 vs the
    recovar reconstruction on EMPIAR challenge1.

    Parameters
    ----------
    vol_relion : numpy.ndarray, shape (N, N, N), real
        RELION-convention real-space volume, as returned by raw
        ``mrcfile.open(relion_output.mrc).data``.

    Returns
    -------
    vol_recovar : numpy.ndarray, shape (N, N, N), real
        Same volume in recovar's convention. Suitable for comparison
        against ``utils.load_mrc(...)`` outputs.

    Notes
    -----
    See ``recovar/em/CLAUDE.md`` and ``~/CLAUDE.md`` for the full
    convention discussion. This helper was previously removed from
    helpers.py in commit 4703c634, leaving the docs stale and forcing
    every subsequent maintainer to rediscover the convention by hand.
    The unit test ``tests/unit/test_relion_volume_convention.py``
    pins the helper to make sure it can't be silently removed again.
    """
    return -np.transpose(vol_relion, (2, 1, 0))


def recovar_volume_to_relion(vol_recovar):
    """Convert a recovar-convention volume into RELION's convention.

    Inverse of :func:`relion_volume_to_recovar`. The function is its own
    inverse because the operation ``-T(2,1,0)`` is an involution.

    Use this when writing a recovar reconstruction into a file that
    will be loaded by RELION (or compared against a RELION reference
    that you only have in RELION-frame).
    """
    return -np.transpose(vol_recovar, (2, 1, 0))


def load_relion_volume(filepath, return_voxel_size=False):
    """Load a RELION-output MRC and return it in recovar's convention.

    This is the canonical way to load any volume produced by RELION
    (e.g. ``run_class001.mrc``, ``run_it010_half1_class001.mrc``,
    ``postprocess_masked.mrc``) for direct comparison against recovar
    outputs.

    Equivalent to::

        with mrcfile.open(filepath) as m:
            return relion_volume_to_recovar(m.data.copy())

    Do NOT use ``utils.load_mrc(...)`` for RELION files — that helper
    is designed for cryosparc/cryoDRGN axis order and will leave RELION
    volumes in the wrong frame.

    Parameters
    ----------
    filepath : str or Path
        Path to a RELION-produced .mrc file.
    return_voxel_size : bool
        If True, also return the voxel size from the MRC header.

    Returns
    -------
    vol : numpy.ndarray, shape (N, N, N), real
        Volume in recovar's convention.
    voxel_size : float, optional
    """
    with mrcfile.open(filepath) as mrc:
        data = np.array(mrc.data, dtype=np.float32, copy=True)
        if return_voxel_size:
            voxel_size = mrc.voxel_size
    vol = relion_volume_to_recovar(data)
    if return_voxel_size:
        return vol, voxel_size
    return vol


def symmetrize_ft_volume(vol, volume_shape):
    og_volume_shape = vol.shape
    vol = vol.reshape(volume_shape)
    vol = vol.at[1:, 1:, 1:].set(0.5 * (np.conj(np.flip(vol[1:, 1:, 1:])) + vol[1:, 1:, 1:]))
    return vol.reshape(og_volume_shape)


def get_all_dataset_indices(cryos):
    return np.concatenate([cryo.dataset_indices for cryo in cryos])


def get_inverse_dataset_indices(cryos):
    return np.argsort(np.concatenate([cryo.dataset_indices for cryo in cryos]))


def guess_grid_size_from_vol_size(vol_size):
    return np.round((vol_size) ** (1 / 3)).astype(int)


def guess_vol_shape_from_vol_size(vol_size):
    return tuple(3 * [guess_grid_size_from_vol_size(vol_size)])


def safe_batch_size(raw_size):
    """Clamp a raw batch size to at least 1 and convert to int."""
    return max(1, int(raw_size))


# Sometimes, memory can grow like O(vol_batch_size * image_batch_size)
def get_image_batch_size(grid_size, gpu_memory):
    """Calculate batch size for image processing.

    Args:
        grid_size: Size of the grid
        gpu_memory: Available GPU memory in GB

    Returns:
        Integer batch size with reasonable bounds
    """
    if grid_size < 1:
        raise ValueError("grid_size must be positive")
    if gpu_memory <= 0:
        raise ValueError("gpu_memory must be positive")

    # Use floating point arithmetic to avoid integer overflow
    # Convert to float before any calculations
    grid_size_f = float(grid_size)
    gpu_memory_f = float(gpu_memory)

    # Each image is grid_size^2 complex64 values (8 bytes each).
    # 2^18 / grid_size^2 targets ~2 GB per batch at gpu_memory=1.
    batch_size = (2.0**18.0) / (grid_size_f * grid_size_f) * gpu_memory_f

    # Add reasonable bounds
    max_batch_size = 2**20  # Reduced from 2**30 to be more conservative
    min_batch_size = 1

    # Clip to reasonable bounds
    batch_size = max(min_batch_size, min(max_batch_size, batch_size))

    return int(batch_size)


def get_vol_batch_size(grid_size, gpu_memory):
    """Calculate batch size for volume processing."""
    if grid_size < 1:
        raise ValueError("grid_size must be positive")
    if gpu_memory <= 0:
        raise ValueError("gpu_memory must be positive")

    # Use floating point arithmetic
    grid_size_f = float(grid_size)
    gpu_memory_f = float(gpu_memory)

    # Empirical formula: 25 volumes at 256^3 fit in ~38 GB.
    # Scales cubically with grid_size and linearly with gpu_memory.
    batch_size = 25.0 * (256.0 / grid_size_f) ** 3.0 * gpu_memory_f / 38.0

    # Add reasonable bounds
    max_batch_size = 2**20  # Reduced from 2**30
    min_batch_size = 1

    batch_size = max(min_batch_size, min(max_batch_size, batch_size))
    return int(batch_size)


def get_column_batch_size(grid_size, gpu_memory):
    """Calculate batch size for column processing."""
    if grid_size < 1:
        raise ValueError("grid_size must be positive")
    if gpu_memory <= 0:
        raise ValueError("gpu_memory must be positive")

    # Use floating point arithmetic
    grid_size_f = float(grid_size)
    gpu_memory_f = float(gpu_memory)

    # Empirical formula: 50 columns at 256^3 fit in ~38 GB.
    # Column processing uses ~half the memory per element as full volumes.
    batch_size = 50.0 * (256.0 / grid_size_f) ** 3.0 * gpu_memory_f / 38.0

    # Add reasonable bounds
    max_batch_size = 2**20  # Reduced from 2**30
    min_batch_size = 1

    batch_size = max(min_batch_size, min(max_batch_size, batch_size))
    return int(batch_size)


from collections import namedtuple

BatchSizes = namedtuple("BatchSizes", ["image", "volume", "column"])


def compute_batch_sizes(grid_size, gpu_memory):
    """Compute all batch sizes at once.

    Returns a ``BatchSizes(image, volume, column)`` namedtuple.
    """
    return BatchSizes(
        image=get_image_batch_size(grid_size, gpu_memory),
        volume=get_vol_batch_size(grid_size, gpu_memory),
        column=get_column_batch_size(grid_size, gpu_memory),
    )


def get_latent_density_batch_size(test_pts, zdim, gpu_memory):
    return np.max([int(gpu_memory / (3 * (get_size_in_gb(test_pts) * zdim**1))), 1])


def get_embedding_batch_size(basis, image_size, contrast_grid, zdim, gpu_memory):

    left_over_memory = gpu_memory - get_size_in_gb(basis)
    # assert left_over_memory > 0, "GPU memory too small?"

    # Per-image memory: image_size * max(zdim, 4) for projections + contrast_grid * zdim^2 for residuals,
    # each 8 bytes (complex64). The /20 safety factor accounts for JIT intermediates.
    batch_size = int(
        left_over_memory / ((image_size * np.max([zdim, 4]) + contrast_grid.size * zdim**2) * 8 / 1e9) / 20
    )

    if batch_size < 1:
        logger.warning("GPU may be too small for the default parameters. Trying anyway")
        return 1

    return batch_size


@dataclass
class AlgorithmOptions:
    """Typed container for pipeline algorithm options."""

    volume_mask_option: str
    zs_dim_to_test: list
    contrast: str
    ignore_zero_frequency: bool
    keep_intermediate: bool


def make_algorithm_options(args):
    return AlgorithmOptions(
        volume_mask_option=args.mask,
        zs_dim_to_test=args.zdim,
        contrast="contrast_qr" if args.correct_contrast else "none",
        ignore_zero_frequency=args.ignore_zero_frequency,
        keep_intermediate=args.keep_intermediate,
    )


def pickle_dump(object, file):
    with open(file, "wb") as f:
        pickle.dump(object, f)


def pickle_load(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def get_variances(covariance_cols, picked_frequencies=None):
    volume_shape = guess_vol_shape_from_vol_size(covariance_cols.shape[-1])

    variances = np.zeros(picked_frequencies.size, covariance_cols.dtype)
    for k in range(picked_frequencies.size):
        variances[k] = covariance_cols[picked_frequencies[k], k]

    return variances


def get_number_of_index_batch(n_images, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if n_images < 0:
        raise ValueError("n_images must be >= 0")
    return int(np.ceil(n_images / batch_size))


def get_batch_of_indices(n_images, batch_size, k):
    batch_st = int(k * batch_size)
    batch_end = int(np.min([(k + 1) * batch_size, n_images]))
    return batch_st, batch_end


def get_batch_of_indices_arange(n_images, batch_size, k):
    batch_st = int(k * batch_size)
    batch_end = int(np.min([(k + 1) * batch_size, n_images]))
    return np.arange(batch_st, batch_end)


def jax_has_gpu():
    try:
        _ = jax.device_put(jax.numpy.ones(1), device=jax.devices("gpu")[0])
        return True
    except Exception:
        return False


def dtype_to_real(rvs_dtype):
    return rvs_dtype.type(0).real.dtype


def write_starfile(
    CTF_params,
    rotation_matrices,
    translations,
    voxel_size,
    grid_size,
    particles_file,
    output_filename,
    halfset_indices=None,
    tilt_groups=None,
):

    # Stored like this in CTF_params
    # dfu (float or Bx1 tensor): DefocusU (Angstrom)
    # dfv (float or Bx1 tensor): DefocusV (Angstrom)
    # dfang (float or Bx1 tensor): DefocusAngle (degrees)
    # volt (float or Bx1 tensor): accelerating voltage (kV)
    # cs (float or Bx1 tensor): spherical aberration (mm)
    # w (float or Bx1 tensor): amplitude contrast ratio
    # phase_shift (float or Bx1 tensor): degrees
    # bfactor (float or Bx1 tensor): envelope fcn B-factor (Angstrom^2)
    grid_size = int(grid_size)

    keys = [
        "rlnOpticsGroup",
        "rlnOpticsGroupName",
        "rlnAmplitudeContrast",
        "rlnSphericalAberration",
        "rlnVoltage",
        "rlnImagePixelSize",
        "rlnImageSize",
        "rlnImageDimensionality",
    ]
    dtype = np.float64
    values = [
        1,
        "opticsGroup1",
        CTF_params[0, 5].astype(dtype),
        CTF_params[0, 4].astype(dtype),
        CTF_params[0, 3].astype(dtype),
        voxel_size,
        grid_size,
        2,
    ]
    optic_df = pd.DataFrame.from_dict({0: values}, orient="index", columns=keys)
    n_images = CTF_params.shape[0]
    image_names = [f"{k + 1}@{particles_file}" for k in range(n_images)]
    micrograph_names = [f"{k + 1}" for k in range(n_images)]
    optics_group = np.ones(n_images).astype(int)

    # _rlnGroupName #20

    keys = [
        "rlnImageName",
        "rlnMicrographName",
        "rlnDefocusU",
        "rlnDefocusV",
        "rlnDefocusAngle",
        "rlnPhaseShift",
        "rlnOpticsGroup",
        "rlnTiltName",
    ]

    # Give each image its own tilt name for compatibility with tilt series parsing
    tilt_names = [f"tilt_{k + 1}" for k in range(n_images)]
    values = [
        image_names,
        micrograph_names,
        CTF_params[:, 0].astype(dtype),
        CTF_params[:, 1].astype(dtype),
        CTF_params[:, 2].astype(dtype),
        CTF_params[:, 6].astype(dtype),
        optics_group,
        tilt_names,
    ]

    if rotation_matrices is not None:
        rotation_matrices = rotation_matrices.astype(np.float32)
        translations = translations.astype(np.float32)
        keys += [
            "rlnAngleRot",
            "rlnAngleTilt",
            "rlnAnglePsi",
            "rlnOriginXAngst",
            "rlnOriginYAngst",
        ]
        rots = R_to_relion(rotation_matrices)
        values += [rots[:, 0], rots[:, 1], rots[:, 2], translations[:, 0], translations[:, 1]]

    if tilt_groups is not None:
        keys += ["rlnGroupName"]
        group_names = [f"tilt_{group_n + 1:07d}" for i, group_n in enumerate(tilt_groups)]
        values += [group_names]
        # Also write contrast variation if using tilt groups?
        keys += ["rlnCtfScalefactor"]
        values += [CTF_params[:, core.CTFParamIndex.CONTRAST]]
        keys += ["rlnCtfBfactor"]
        values += [CTF_params[:, core.CTFParamIndex.BFACTOR]]
        keys += ["rlnMicrographPreExposure"]
        values += [CTF_params[:, core.CTFParamIndex.DOSE]]

    if halfset_indices is not None:
        keys += ["rlnRandomSubset"]
        values += [halfset_indices]

    d = dict(zip(keys, values))
    particles_df = pd.DataFrame(d)
    star_df = {"optics": optic_df, "particles": particles_df}
    starfile.write(star_df, output_filename)

    return


def R_to_relion(rot: np.ndarray, degrees: bool = True) -> np.ndarray:
    """Convert rotation matrices to RELION Euler angles.

    Converts rotation matrices to RELION's ZXZ Euler angle convention
    with appropriate coordinate frame transformations.

    Args:
        rot: Rotation matrix or matrices of shape (3,3) or (N,3,3)
        degrees: If True, return angles in degrees; otherwise radians

    Returns:
        Euler angles in RELION convention, shape (N,3)
    """
    from scipy.spatial.transform import Rotation as SciPyRot

    # Ensure batch dimension
    matrices = rot.reshape(1, 3, 3) if rot.shape == (3, 3) else rot

    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError(f"Expected shape (N,3,3), got {rot.shape}")

    # Create coordinate frame adjustment matrix
    # RELION uses different handedness conventions
    frame_adjust = np.array([[1, -1, 1], [-1, 1, -1], [1, -1, 1]], dtype=np.float64)

    # Apply frame adjustment
    adjusted_matrices = matrices * frame_adjust

    # Convert to ZXZ Euler angles
    scipy_rot = SciPyRot.from_matrix(adjusted_matrices)
    angles = scipy_rot.as_euler("zxz", degrees=True)

    # Apply RELION-specific angle offsets
    # Adjust first rotation (around Z)
    angles[:, 0] = angles[:, 0] - 90.0
    # Adjust third rotation (around Z)
    angles[:, 2] = angles[:, 2] + 90.0

    # Normalize to [-180, 180] range
    angles = (angles + 180.0) % 360.0 - 180.0

    # Convert to radians if requested
    if not degrees:
        angles = np.deg2rad(angles)

    return angles


def R_from_relion(euler_: np.ndarray, degrees: bool = True) -> np.ndarray:
    """Convert RELION Euler angles to rotation matrices.

    Converts RELION's ZXZ Euler angle convention to rotation matrices
    with appropriate coordinate frame transformations.

    Args:
        euler_: Euler angles of shape (3,) or (N,3) in RELION convention
        degrees: If True, input angles are in degrees; otherwise radians

    Returns:
        Rotation matrices of shape (N,3,3)
    """
    from scipy.spatial.transform import Rotation as SciPyRot

    # Work with a copy to avoid modifying input
    angles = euler_.copy()

    # Ensure batch dimension
    angles = angles.reshape(1, 3) if angles.shape == (3,) else angles

    # Reverse RELION-specific angle offsets
    # These are the inverse of the operations in R_to_relion
    angles[:, 0] = angles[:, 0] + 90.0
    angles[:, 2] = angles[:, 2] - 90.0

    # Convert to rotation matrices using ZXZ convention
    scipy_rot = SciPyRot.from_euler("zxz", angles, degrees=degrees)
    matrices = scipy_rot.as_matrix()

    # Create inverse coordinate frame adjustment matrix
    frame_adjust = np.array([[1, -1, 1], [-1, 1, -1], [1, -1, 1]], dtype=np.float64)

    # Apply inverse frame adjustment
    result = matrices * frame_adjust

    return result


def write_starfile_from_cryodrgn_format(
    ctf_path, pose_path, particles_file_path, output_filename, halfset_indices=None
):
    ctf = pickle_load(ctf_path)
    poses = pickle_load(pose_path)
    rots = poses[0]
    trans = poses[1]
    write_starfile(
        ctf[:, 2:], rots, trans, ctf[0, 1], ctf[0, 0], particles_file_path, output_filename, halfset_indices=None
    )


def downsample_vol_by_fourier_truncation(vol_input, target_grid_size):
    input_grid_size = int(vol_input.shape[0])
    target_grid_size = int(target_grid_size)
    if target_grid_size < 1:
        raise ValueError("target_grid_size must be positive")
    if target_grid_size > input_grid_size:
        raise ValueError("target_grid_size must be <= input grid size")

    X = fourier_transform_utils.get_dft3(vol_input)
    crop_start = (input_grid_size - target_grid_size) // 2
    crop_end = crop_start + target_grid_size
    X = X[crop_start:crop_end, crop_start:crop_end, crop_start:crop_end]
    X = np.array(X)
    X[0, :, :] = X[0, :, :].real
    X[:, 0, :] = X[:, 0, :].real
    X[:, :, 0] = X[:, :, 0].real
    return fourier_transform_utils.get_idft3(X)


class RobustStreamHandler(logging.StreamHandler):
    """StreamHandler that silently handles stale NFS/GPFS file handles."""

    def emit(self, record):
        try:
            super().emit(record)
        except OSError:
            pass


class RobustFileHandler(logging.FileHandler):
    """FileHandler that reopens the file on stale NFS/GPFS handles."""

    def emit(self, record):
        try:
            super().emit(record)
        except OSError:
            try:
                self.close()
                self._open()
                super().emit(record)
            except OSError:
                pass


def basic_config_logger(output_folder):
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
        level=logging.INFO,
        force=True,
        handlers=[RobustFileHandler(f"{output_folder}/run.log"), RobustStreamHandler()],
    )


class DuplicateFilter:
    def __init__(self):
        self.msgs = set()

    def filter(self, record):
        rv = record.msg not in self.msgs
        self.msgs.add(record.msg)
        return rv


# ── RELION-exact Euler angle conversion (from RELION src/euler.cpp) ──────


def relion_euler_to_matrix(rot_deg, tilt_deg, psi_deg):
    """RELION's exact Euler angles to rotation matrix (from src/euler.cpp)."""
    import numpy as np
    rot, tilt, psi = np.radians(rot_deg), np.radians(tilt_deg), np.radians(psi_deg)
    ca, sa = np.cos(rot), np.sin(rot)
    cb, sb = np.cos(tilt), np.sin(tilt)
    cg, sg = np.cos(psi), np.sin(psi)
    return np.array([
        [ca * cb * cg - sa * sg, ca * cb * sg + sa * cg, -ca * sb],
        [-sa * cb * cg - ca * sg, -sa * cb * sg + ca * cg, sa * sb],
        [sb * cg, sb * sg, cb],
    ])


def relion_matrix_to_euler(A):
    """RELION's exact rotation matrix to Euler angles (from src/euler.cpp)."""
    import numpy as np
    abs_sb = np.sqrt(A[0, 2] ** 2 + A[1, 2] ** 2)
    if abs_sb > 1e-6:
        psi = np.degrees(np.arctan2(A[2, 1], A[2, 0]))
        rot = np.degrees(np.arctan2(A[1, 2], -A[0, 2]))
        if np.abs(np.sin(np.radians(rot))) > 1e-6:
            tilt = np.degrees(np.arctan2(A[1, 2] / np.sin(np.radians(rot)), A[2, 2]))
        else:
            tilt = np.degrees(np.arctan2(-A[0, 2] / np.cos(np.radians(rot)), A[2, 2]))
    else:
        rot = 0.0
        tilt = np.degrees(np.arctan2(abs_sb, A[2, 2]))
        psi = np.degrees(np.arctan2(A[0, 1], A[0, 0]))
    return np.array([rot, tilt, psi])


def relion_volume_to_recovar(vol_relion):
    """Convert RELION-convention volume to recovar convention: negate + transpose(2,1,0)."""
    import numpy as np
    return -np.transpose(vol_relion, (2, 1, 0))


def recovar_volume_to_relion(vol_recovar):
    """Convert recovar-convention volume to RELION convention: negate + transpose(2,1,0)."""
    import numpy as np
    return -np.transpose(vol_recovar, (2, 1, 0))

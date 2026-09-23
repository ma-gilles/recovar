"""Approximate solvent-contrast correction for volumes computed from atomic models.

A Coulomb potential computed from an atomic model in vacuum lacks the
displaced-solvent contribution, so its low-frequency contrast is too high.
Henderson & McMullan (2013), doi:10.1093/jmicro/dfs094, approximate the
effect by the radial filter

    H(q) = 1 - a * exp(-B * |q|^2 / 4),

with ``q`` the 3D spatial frequency in cycles per angstrom. The simulator
projects the effective volume ``T(V)`` with ``FFT[T(V)] = H * FFT[V]`` and the
ground-truth loader applies the same operator from the record stored in
``simulation_info.pkl``. See ``docs/math/atomic_solvent_contrast.md`` for the
model, the metadata schema and the ground-truth PCA convention.
"""

import numbers

import numpy as np

from recovar import core

MODEL = "henderson_mcmullan_2013"
MODEL_VERSION = 1
REFERENCE = "Henderson & McMullan (2013) Microscopy 62:43-50, doi:10.1093/jmicro/dfs094"
DEFAULT_A = 0.8
DEFAULT_B = 2000.0

# ``simulation_info`` key holding the correction record.
METADATA_KEY = "atomic_solvent_correction"

# Values of the record's ``ground_truth_representation`` field.
UNCORRECTED_INPUTS = "uncorrected_inputs"
CORRECTED_EFFECTIVE = "corrected_effective"

_REQUIRED_ENABLED_FIELDS = (
    "model",
    "model_version",
    "a",
    "B",
    "units",
    "voxel_size",
    "grid_size",
    "volume_shape",
    "ground_truth_representation",
)


def validate_parameters(a, B):
    """Return ``(a, B)`` as floats after checking ``0 <= a <= 1`` and ``B >= 0``."""
    for name, value in (("a", a), ("B", B)):
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not np.isfinite(value):
            raise ValueError(f"atomic solvent correction parameter {name} must be a finite number, got {value!r}")
    a, B = float(a), float(B)
    if not 0.0 <= a <= 1.0:
        raise ValueError(f"atomic solvent correction amplitude a must lie in [0, 1], got {a}")
    if B < 0.0:
        raise ValueError(f"atomic solvent correction B must be non-negative (angstrom^2), got {B}")
    return a, B


def solvent_contrast_filter(volume_shape, voxel_size, a=DEFAULT_A, B=DEFAULT_B):
    """Flat ``H(q) = 1 - a exp(-B |q|^2 / 4)`` on recovar's centered Fourier grid.

    ``q`` is in cycles/angstrom: the integer frequency index divided by
    ``N * voxel_size``, the same convention as ``simulator.get_B_factor_scaling``.
    Returned in float64, flattened in the layout of ``ftu.get_dft3(...).reshape(-1)``.
    Formulation: ``docs/math/atomic_solvent_contrast.md``.
    """
    a, B = validate_parameters(a, B)
    if not voxel_size > 0:
        raise ValueError(f"voxel_size must be positive (angstrom), got {voxel_size}")
    volume_shape = tuple(int(n) for n in volume_shape)
    vol_idx = np.arange(np.prod(volume_shape))
    freqs = np.asarray(core.vec_indices_to_frequencies(vol_idx, volume_shape), dtype=np.float64)
    q = freqs / (np.asarray(volume_shape, dtype=np.float64) * float(voxel_size))
    q_norm_sq = np.sum(q**2, axis=-1)
    return 1.0 - a * np.exp(-B * q_norm_sq / 4.0)


def apply_solvent_contrast(volumes_ft, volume_shape, voxel_size, a=DEFAULT_A, B=DEFAULT_B):
    """Apply ``T`` to flat centered-Fourier volumes of shape ``(..., prod(volume_shape))``.

    No renormalization and no clamping: the attenuation is part of the model.
    The filter is cast to the volumes' real precision so a complex64 stack
    stays complex64.
    """
    volumes_ft = np.asarray(volumes_ft)
    filt = solvent_contrast_filter(volume_shape, voxel_size, a, B)
    if volumes_ft.shape[-1] != filt.size:
        raise ValueError(f"volume size {volumes_ft.shape[-1]} does not match volume_shape {tuple(volume_shape)}")
    real_dtype = np.finfo(volumes_ft.dtype).dtype if np.iscomplexobj(volumes_ft) else volumes_ft.dtype
    return volumes_ft * filt.astype(real_dtype)


def make_record(enabled, voxel_size=None, grid_size=None, a=DEFAULT_A, B=DEFAULT_B, applied_to_outlier_volume=False):
    """Build the ``simulation_info[METADATA_KEY]`` record written by the simulator.

    A disabled record only states ``enabled: False``. An enabled record carries
    everything needed to reproduce ``T`` and says that ``volumes_path_root``
    points at the uncorrected input volumes, so the loader applies ``T`` once.
    """
    if not enabled:
        return {"enabled": False}
    a, B = validate_parameters(a, B)
    grid_size = int(grid_size)
    return {
        "enabled": True,
        "model": MODEL,
        "model_version": MODEL_VERSION,
        "reference": REFERENCE,
        "formula": "H(q) = 1 - a * exp(-B * |q|^2 / 4)",
        "a": a,
        "B": B,
        "units": {"a": "dimensionless", "B": "angstrom^2", "q": "cycles/angstrom", "voxel_size": "angstrom"},
        "voxel_size": float(voxel_size),
        "grid_size": grid_size,
        "volume_shape": [grid_size] * 3,
        "fourier_convention": "recovar centered DFT (fourier_transform_utils.get_dft3); q = k / (N * voxel_size)",
        "applied_to": "clean volumes after resampling and global scale_vol, before projection",
        "applied_to_outlier_volume": bool(applied_to_outlier_volume),
        "ground_truth_representation": UNCORRECTED_INPUTS,
    }


def record_from_simulation_info(simulation_info):
    """Return the validated enabled record, or ``None`` when no correction applies.

    Datasets written before this option have no record and keep their legacy
    (uncorrected) truth. An enabled record with an unknown model or version,
    missing fields, or a grid that disagrees with ``simulation_info`` raises.
    """
    record = simulation_info.get(METADATA_KEY)
    if record is None:
        return None
    if not isinstance(record, dict) or "enabled" not in record:
        raise ValueError(f"simulation_info[{METADATA_KEY!r}] is malformed: {record!r}")
    if not record["enabled"]:
        return None
    missing = [field for field in _REQUIRED_ENABLED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"atomic solvent correction record is incomplete; missing {missing}")
    if record["model"] != MODEL or record["model_version"] != MODEL_VERSION:
        raise ValueError(
            f"unsupported atomic solvent correction model {record['model']!r} version {record['model_version']!r}; "
            f"this recovar supports {MODEL!r} version {MODEL_VERSION}"
        )
    validate_parameters(record["a"], record["B"])
    if record["ground_truth_representation"] not in (UNCORRECTED_INPUTS, CORRECTED_EFFECTIVE):
        raise ValueError(
            f"unknown ground_truth_representation {record['ground_truth_representation']!r} in the "
            "atomic solvent correction record"
        )
    if "grid_size" in simulation_info and int(record["grid_size"]) != int(simulation_info["grid_size"]):
        raise ValueError(
            f"atomic solvent correction grid_size {record['grid_size']} does not match "
            f"simulation grid_size {simulation_info['grid_size']}"
        )
    return record


def apply_record(volumes_ft, record):
    """Apply the operator described by an enabled record to flat Fourier volumes."""
    return apply_solvent_contrast(volumes_ft, record["volume_shape"], record["voxel_size"], record["a"], record["B"])

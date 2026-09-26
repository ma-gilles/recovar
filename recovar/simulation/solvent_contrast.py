"""Atomic-model volume transform: solvent contrast plus a B-factor.

A Coulomb potential computed from an atomic model in vacuum lacks the
displaced-solvent contribution, so its low-frequency contrast is too high, and
a model without B-factors has no high-frequency falloff. The simulator can
project the effective volume ``T(V)`` with

    FFT[T(V)](q) = H(q) * exp(-B_atomic * |q|^2 / 4) * FFT[V](q),
    H(q) = 1 - a * exp(-B * |q|^2 / 4),

``q`` the 3D spatial frequency in cycles per angstrom. ``H`` is the
Henderson & McMullan (2013) solvent-contrast approximation,
doi:10.1093/jmicro/dfs094; the B-factor term uses RELION's CTF convention
``exp(-B s^2 / 4)``. The ground-truth loader applies the same operator from the
record stored in ``simulation_info.pkl``. See
``docs/math/atomic_solvent_contrast.md`` for the model, the metadata schema and
the ground-truth PCA convention.
"""

import argparse
import numbers

import numpy as np

from recovar import core

MODEL = "henderson_mcmullan_2013"
# Version 1: H only. Version 2: H * exp(-B_atomic |q|^2 / 4).
MODEL_VERSION = 2
SUPPORTED_MODEL_VERSIONS = (1, 2)
REFERENCE = "Henderson & McMullan (2013) Microscopy 62:43-50, doi:10.1093/jmicro/dfs094"
DEFAULT_A = 0.8
DEFAULT_B = 2000.0
DEFAULT_ATOMIC_BFACTOR = 100.0

# ``simulation_info`` key holding the correction record.
METADATA_KEY = "atomic_solvent_correction"

# Values of the record's ``ground_truth_representation`` field.
UNCORRECTED_INPUTS = "uncorrected_inputs"
CORRECTED_EFFECTIVE = "corrected_effective"

# The EM/VDAM development preset: ``generate_synthetic_dataset(..., **EM_DEVELOPMENT_PRESET)``.
EM_DEVELOPMENT_PRESET = {
    "atomic_solvent_correction": True,
    "solvent_contrast_a": DEFAULT_A,
    "solvent_contrast_B": DEFAULT_B,
    "atomic_bfactor": DEFAULT_ATOMIC_BFACTOR,
}

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


def validate_parameters(a, B, atomic_bfactor=0.0):
    """Return ``(a, B, atomic_bfactor)`` as floats; ``0 <= a <= 1``, ``B >= 0``, ``atomic_bfactor >= 0``."""
    for name, value in (("a", a), ("B", B), ("atomic_bfactor", atomic_bfactor)):
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not np.isfinite(value):
            raise ValueError(f"atomic solvent correction parameter {name} must be a finite number, got {value!r}")
    a, B, atomic_bfactor = float(a), float(B), float(atomic_bfactor)
    if not 0.0 <= a <= 1.0:
        raise ValueError(f"atomic solvent correction amplitude a must lie in [0, 1], got {a}")
    if B < 0.0:
        raise ValueError(f"atomic solvent correction B must be non-negative (angstrom^2), got {B}")
    if atomic_bfactor < 0.0:
        raise ValueError(f"atomic_bfactor must be non-negative (angstrom^2), got {atomic_bfactor}")
    return a, B, atomic_bfactor


def solvent_contrast_filter(volume_shape, voxel_size, a=DEFAULT_A, B=DEFAULT_B, atomic_bfactor=DEFAULT_ATOMIC_BFACTOR):
    """Flat ``H(q) exp(-B_atomic |q|^2 / 4)`` on recovar's centered Fourier grid.

    ``H(q) = 1 - a exp(-B |q|^2 / 4)``. ``q`` is in cycles/angstrom: the integer
    frequency index divided by ``N * voxel_size``, the same convention as
    ``simulator.get_B_factor_scaling``. ``atomic_bfactor=0`` gives ``H`` alone.
    Returned in float64, flattened in the layout of ``ftu.get_dft3(...).reshape(-1)``.
    Formulation: ``docs/math/atomic_solvent_contrast.md``.
    """
    a, B, atomic_bfactor = validate_parameters(a, B, atomic_bfactor)
    if not voxel_size > 0:
        raise ValueError(f"voxel_size must be positive (angstrom), got {voxel_size}")
    volume_shape = tuple(int(n) for n in volume_shape)
    vol_idx = np.arange(np.prod(volume_shape))
    freqs = np.asarray(core.vec_indices_to_frequencies(vol_idx, volume_shape), dtype=np.float64)
    q = freqs / (np.asarray(volume_shape, dtype=np.float64) * float(voxel_size))
    q_norm_sq = np.sum(q**2, axis=-1)
    return (1.0 - a * np.exp(-B * q_norm_sq / 4.0)) * np.exp(-atomic_bfactor * q_norm_sq / 4.0)


def apply_solvent_contrast(
    volumes_ft, volume_shape, voxel_size, a=DEFAULT_A, B=DEFAULT_B, atomic_bfactor=DEFAULT_ATOMIC_BFACTOR
):
    """Apply ``T`` to flat centered-Fourier volumes of shape ``(..., prod(volume_shape))``.

    No renormalization and no clamping: the attenuation is part of the model.
    The filter is cast to the volumes' real precision so a complex64 stack
    stays complex64.
    """
    volumes_ft = np.asarray(volumes_ft)
    filt = solvent_contrast_filter(volume_shape, voxel_size, a, B, atomic_bfactor)
    if volumes_ft.shape[-1] != filt.size:
        raise ValueError(f"volume size {volumes_ft.shape[-1]} does not match volume_shape {tuple(volume_shape)}")
    real_dtype = np.finfo(volumes_ft.dtype).dtype if np.iscomplexobj(volumes_ft) else volumes_ft.dtype
    return volumes_ft * filt.astype(real_dtype)


def make_record(
    enabled,
    voxel_size=None,
    grid_size=None,
    a=DEFAULT_A,
    B=DEFAULT_B,
    atomic_bfactor=DEFAULT_ATOMIC_BFACTOR,
    applied_to_outlier_volume=False,
):
    """Build the ``simulation_info[METADATA_KEY]`` record written by the simulator.

    A disabled record only states ``enabled: False``. An enabled record carries
    everything needed to reproduce ``T`` and says that ``volumes_path_root``
    points at the uncorrected input volumes, so the loader applies ``T`` once.
    """
    if not enabled:
        return {"enabled": False}
    a, B, atomic_bfactor = validate_parameters(a, B, atomic_bfactor)
    grid_size = int(grid_size)
    return {
        "enabled": True,
        "model": MODEL,
        "model_version": MODEL_VERSION,
        "reference": REFERENCE,
        "formula": "T(q) = (1 - a * exp(-B * |q|^2 / 4)) * exp(-B_atomic * |q|^2 / 4)",
        "a": a,
        "B": B,
        "B_atomic": atomic_bfactor,
        "units": {
            "a": "dimensionless",
            "B": "angstrom^2",
            "B_atomic": "angstrom^2",
            "q": "cycles/angstrom",
            "voxel_size": "angstrom",
        },
        "voxel_size": float(voxel_size),
        "grid_size": grid_size,
        "volume_shape": [grid_size] * 3,
        "fourier_convention": "recovar centered DFT (fourier_transform_utils.get_dft3); q = k / (N * voxel_size)",
        "applied_to": "clean volumes after resampling and global scale_vol, before projection",
        "applied_to_outlier_volume": bool(applied_to_outlier_volume),
        "ground_truth_representation": UNCORRECTED_INPUTS,
    }


def record_from_options(
    atomic_solvent_correction,
    voxel_size,
    grid_size,
    solvent_contrast_a=None,
    solvent_contrast_B=None,
    atomic_bfactor=None,
    applied_to_outlier_volume=False,
):
    """Record for a simulator's keyword options; unset parameters take the preset defaults.

    Parameters other than the switch are only valid with ``atomic_solvent_correction=True``.
    """
    if not atomic_solvent_correction:
        if solvent_contrast_a is not None or solvent_contrast_B is not None or atomic_bfactor is not None:
            raise ValueError(
                "solvent_contrast_a/solvent_contrast_B/atomic_bfactor require atomic_solvent_correction=True"
            )
        return make_record(False)
    return make_record(
        True,
        voxel_size=voxel_size,
        grid_size=grid_size,
        a=DEFAULT_A if solvent_contrast_a is None else solvent_contrast_a,
        B=DEFAULT_B if solvent_contrast_B is None else solvent_contrast_B,
        atomic_bfactor=DEFAULT_ATOMIC_BFACTOR if atomic_bfactor is None else atomic_bfactor,
        applied_to_outlier_volume=applied_to_outlier_volume,
    )


def record_from_simulation_info(simulation_info):
    """Return the validated enabled record, or ``None`` when no correction applies.

    Datasets written before this option have no record and keep their legacy
    (uncorrected) truth. Version 1 records (no B-factor term) still load. An
    enabled record with an unknown model or version, missing fields, or a grid
    that disagrees with ``simulation_info`` raises.
    """
    record = simulation_info.get(METADATA_KEY)
    if record is None:
        return None
    if not isinstance(record, dict) or "enabled" not in record:
        raise ValueError(f"simulation_info[{METADATA_KEY!r}] is malformed: {record!r}")
    if not record["enabled"]:
        return None
    missing = [field for field in _REQUIRED_ENABLED_FIELDS if field not in record]
    if record.get("model_version") == 2 and "B_atomic" not in record:
        missing.append("B_atomic")
    if missing:
        raise ValueError(f"atomic solvent correction record is incomplete; missing {missing}")
    if record["model"] != MODEL or record["model_version"] not in SUPPORTED_MODEL_VERSIONS:
        raise ValueError(
            f"unsupported atomic solvent correction model {record['model']!r} version {record['model_version']!r}; "
            f"this recovar supports {MODEL!r} versions {SUPPORTED_MODEL_VERSIONS}"
        )
    validate_parameters(record["a"], record["B"], _atomic_bfactor(record))
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


def _atomic_bfactor(record):
    return record["B_atomic"] if record["model_version"] >= 2 else 0.0


def apply_record(volumes_ft, record):
    """Apply the operator described by an enabled record to flat Fourier volumes."""
    return apply_solvent_contrast(
        volumes_ft, record["volume_shape"], record["voxel_size"], record["a"], record["B"], _atomic_bfactor(record)
    )


def add_cli_arguments(parser, enabled_by_default=False):
    """Add the simulator's atomic-volume transform options to an argparse parser.

    With ``enabled_by_default=True`` (EM/VDAM development scripts) the preset is
    on and ``--no-atomic-solvent-correction`` turns it off for experimental or
    already-corrected maps; recovar's own commands keep it opt-in.
    """
    description = (
        "EM-development preset for volumes computed from atomic models without solvent or B-factors: "
        "multiply their Fourier transform by (1 - a exp(-B |q|^2/4)) exp(-B_atomic |q|^2/4), q in cycles/A, "
        f"with a={DEFAULT_A}, B={DEFAULT_B:g} A^2 (Henderson-McMullan 2013) and B_atomic={DEFAULT_ATOMIC_BFACTOR:g} "
        "A^2 unless overridden. Ground-truth loading applies it automatically."
    )
    if enabled_by_default:
        parser.add_argument(
            "--atomic-solvent-correction",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=description + " On by default; use --no-atomic-solvent-correction for experimental or "
            "already-corrected maps.",
        )
    else:
        parser.add_argument(
            "--atomic-solvent-correction",
            action="store_true",
            help=description + " Off by default; do not use for experimental or already-corrected maps.",
        )
    parser.add_argument(
        "--solvent-contrast-a",
        type=float,
        default=None,
        help=f"Solvent-contrast amplitude a in [0, 1] (default {DEFAULT_A}); requires --atomic-solvent-correction",
    )
    parser.add_argument(
        "--solvent-contrast-b",
        type=float,
        default=None,
        help=f"Solvent-contrast B in A^2, >= 0 (default {DEFAULT_B:g}); requires --atomic-solvent-correction",
    )
    parser.add_argument(
        "--atomic-bfactor",
        type=float,
        default=None,
        help=(
            f"B-factor B_atomic in A^2, >= 0 (default {DEFAULT_ATOMIC_BFACTOR:g}; 0 disables it); "
            "requires --atomic-solvent-correction"
        ),
    )


def kwargs_from_cli_args(args):
    """Map parsed :func:`add_cli_arguments` options to ``generate_synthetic_dataset`` keywords."""
    return {
        "atomic_solvent_correction": args.atomic_solvent_correction,
        "solvent_contrast_a": args.solvent_contrast_a,
        "solvent_contrast_B": args.solvent_contrast_b,
        "atomic_bfactor": args.atomic_bfactor,
    }

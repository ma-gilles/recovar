"""Validate named native RELION tomography motion files before subset reuse.

RELION 5's ``Trajectory::read`` matches named STAR blocks to particle names, but
uses row order for the legacy ``data_0``, ``data_1``, ... format. Reusing that
legacy file after selecting/reordering particles can silently assign the wrong
motion. This module deliberately refuses it rather than converting it.

Trajectory rows cover every frame of a tomogram, not only the visible images in
a compact extracted particle stack (``ParticleSet::checkTrajectoryLengths``).
No input file is modified and no trajectory is copied or recalculated.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import starfile

_SHIFT_COLUMNS = [f"rlnOrigin{axis}Angst" for axis in "XYZ"]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _frame_count(value, name):
    try:
        flags = np.asarray(ast.literal_eval(value) if isinstance(value, str) else value)
    except (ValueError, SyntaxError, TypeError) as exc:
        raise ValueError(f"Invalid native visibility vector for particle {name}") from exc
    _require(
        flags.ndim == 1 and len(flags) > 0 and np.isin(flags, [0, 1]).all(),
        f"Native visibility must be a nonempty binary vector for particle {name}",
    )
    _require(np.any(flags), f"Particle {name} has no visible native frames")
    return len(flags)


def validate_named_trajectories(path, selected_particles_frame) -> None:
    """Require named, finite XYZ motion for every selected physical particle.

    The source STAR must begin with a scalar ``data_general`` block containing
    an integer ``rlnParticleNumber`` equal to its number of named particle
    blocks. A block named exactly as each selected ``rlnTomoParticleName`` must
    contain all three Angstrom shift columns and one row per *full* native
    visibility-vector entry. Extra, unselected named particles are allowed.

    Legacy row-indexed files, duplicate/missing particle blocks, incomplete
    metadata, non-finite shifts and incompatible frame counts raise ValueError.
    """
    selected = selected_particles_frame
    _require(
        isinstance(selected, pd.DataFrame) and len(selected) > 0,
        "Trajectory validation requires a nonempty selected particle table",
    )
    for column in ("rlnTomoParticleName", "rlnTomoVisibleFrames"):
        _require(column in selected, f"Selected native particles lack {column}")
    names = selected["rlnTomoParticleName"]
    _require(names.notna().all(), "Selected trajectory particle IDs contain null values")
    names = names.astype(str)
    _require(
        not names.isin(["", "?", "."]).any() and names.is_unique,
        "Selected trajectory particle IDs must be nonempty and unique",
    )
    expected_frames = {name: _frame_count(value, name) for name, value in zip(names, selected["rlnTomoVisibleFrames"])}

    path = Path(path).resolve(strict=True)
    _require(path.is_file(), f"Trajectory source is not a file: {path}")
    # starfile stores blocks in a dict, which silently replaces duplicate keys.
    # Scan the native block headers first so duplicate particle IDs cannot be
    # hidden by parsing. Native trajectory payloads are numeric XYZ tables.
    with path.open() as handle:
        block_names = [line.strip()[5:] for line in handle if line.strip().startswith("data_")]
    _require(
        len(block_names) >= 2 and block_names[0] == "general",
        "Named trajectory STAR must begin with data_general and contain particle blocks",
    )
    _require(
        all(block_names) and len(set(block_names)) == len(block_names),
        "Duplicate or empty trajectory STAR block names are ambiguous",
    )
    _require(
        block_names[1] != "0",
        "Legacy row-indexed trajectories (data_0, data_1, ...) cannot be reused after particle subsetting; provide named trajectories",
    )

    blocks = starfile.read(path, always_dict=True)
    _require(list(blocks) == block_names, "Trajectory STAR block structure could not be preserved")
    general = blocks["general"]
    _require(
        isinstance(general, dict) and "rlnParticleNumber" in general,
        "Trajectory data_general requires scalar rlnParticleNumber",
    )
    count = general["rlnParticleNumber"]
    _require(
        isinstance(count, (int, float, np.integer, np.floating))
        and not isinstance(count, (bool, np.bool_))
        and np.isfinite(count)
        and float(count).is_integer(),
        "Trajectory rlnParticleNumber must be a finite integer",
    )
    _require(
        int(count) == len(block_names) - 1 and int(count) >= len(selected),
        "Trajectory rlnParticleNumber does not match the named blocks or selected particle count",
    )
    missing = sorted(set(names) - set(block_names[1:]))
    _require(not missing, f"Missing named trajectories for selected particles: {missing[:10]}")

    for name, frame_count in expected_frames.items():
        shifts = blocks[name]
        _require(isinstance(shifts, pd.DataFrame), f"Trajectory {name} must be an XYZ loop table")
        _require(
            all(column in shifts for column in _SHIFT_COLUMNS),
            f"Trajectory {name} requires rlnOriginXAngst/YAngst/ZAngst",
        )
        _require(
            len(shifts) == frame_count,
            f"Trajectory {name} has {len(shifts)} rows; expected all {frame_count} tomogram frames, not only visible images",
        )
        try:
            values = shifts[_SHIFT_COLUMNS].to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Trajectory {name} contains nonnumeric XYZ shifts") from exc
        _require(np.isfinite(values).all(), f"Trajectory {name} contains non-finite XYZ shifts")

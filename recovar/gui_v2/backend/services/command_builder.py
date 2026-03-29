"""Build CLI command lists for recovar job types.

Each builder takes validated parameters from the API and returns a
``list[str]`` suitable for ``subprocess`` or sbatch.  No user strings
are interpolated into shell commands — everything is passed as separate
argv elements.
"""

from __future__ import annotations

import os
import sys
from typing import Any


def _python() -> str:
    """Return the Python interpreter path."""
    return sys.executable


def _add_optional(cmd: list[str], flag: str, value: Any) -> None:
    """Append ``--flag value`` to *cmd* if value is truthy."""
    if value is not None and value != "" and value is not False:
        cmd.extend([flag, str(value)])


def _add_flag(cmd: list[str], flag: str, value: bool | None) -> None:
    """Append ``--flag`` to *cmd* if value is True."""
    if value:
        cmd.append(flag)


def build_pipeline_command(params: dict[str, Any]) -> list[str]:
    """Build ``recovar pipeline`` command from validated parameters.

    Required params: particles, mask
    Optional params: all other pipeline CLI flags
    """
    cmd = [_python(), "-m", "recovar", "pipeline"]

    # Positional: particles
    cmd.append(params["particles"])

    # Required
    _add_optional(cmd, "--mask", params.get("mask"))

    # Output
    _add_optional(cmd, "-o", params.get("outdir"))
    _add_optional(cmd, "--project", params.get("project"))

    # Zdim (list → comma-separated string)
    zdim = params.get("zdim")
    if zdim is not None:
        if isinstance(zdim, list):
            zdim = ",".join(str(z) for z in zdim)
        _add_optional(cmd, "--zdim", zdim)

    # Dataset loading
    _add_optional(cmd, "--poses", params.get("poses"))
    _add_optional(cmd, "--ctf", params.get("ctf"))
    _add_optional(cmd, "--ind", params.get("ind"))
    _add_optional(cmd, "--datadir", params.get("datadir"))
    _add_optional(cmd, "--strip-prefix", params.get("strip_prefix"))
    _add_optional(cmd, "--n-images", params.get("n_images"))
    _add_optional(cmd, "--halfsets", params.get("halfsets"))
    _add_optional(cmd, "--tilt-series-ctf", params.get("tilt_series_ctf"))

    # Processing flags
    _add_optional(cmd, "--downsample", params.get("downsample"))
    _add_flag(cmd, "--lazy", params.get("lazy"))
    _add_flag(cmd, "--correct-contrast", params.get("correct_contrast"))
    _add_optional(cmd, "--focus-mask", params.get("focus_mask"))
    _add_optional(cmd, "--Bfactor", params.get("Bfactor"))
    _add_optional(cmd, "--n-bins", params.get("n_bins"))

    return cmd


def build_analyze_command(params: dict[str, Any]) -> list[str]:
    """Build ``recovar analyze`` command."""
    cmd = [_python(), "-m", "recovar", "analyze"]

    # Positional: result_dir
    cmd.append(params["result_dir"])

    # Required
    _add_optional(cmd, "--zdim", params.get("zdim"))

    # Optional
    _add_optional(cmd, "-o", params.get("outdir"))
    _add_optional(cmd, "--n-clusters", params.get("n_clusters"))
    _add_optional(cmd, "--n-trajectories", params.get("n_trajectories"))
    _add_optional(cmd, "--n-vols-along-path", params.get("n_vols_along_path"))
    _add_flag(cmd, "--skip-umap", params.get("skip_umap"))

    return cmd


def build_compute_state_command(
    params: dict[str, Any],
    latent_points_file: str,
) -> list[str]:
    """Build ``recovar compute_state`` command.

    *latent_points_file* is the path to a text file containing the
    latent coordinate vector, written by the API before submission.
    """
    cmd = [_python(), "-m", "recovar", "compute_state"]

    cmd.append(params["result_dir"])

    _add_optional(cmd, "--zdim", params.get("zdim"))
    _add_optional(cmd, "-o", params.get("outdir"))
    cmd.extend(["--latent-points", latent_points_file])

    return cmd


def build_compute_trajectory_command(
    params: dict[str, Any],
    z_start_file: str,
    z_end_file: str,
) -> list[str]:
    """Build ``recovar compute_trajectory`` command."""
    cmd = [_python(), "-m", "recovar", "compute_trajectory"]

    cmd.append(params["result_dir"])

    _add_optional(cmd, "--zdim", params.get("zdim"))
    _add_optional(cmd, "-o", params.get("outdir"))
    cmd.extend(["--z_st", z_start_file])
    cmd.extend(["--z_end", z_end_file])
    _add_optional(cmd, "--n-vols-along-path", params.get("n_vols_along_path"))

    return cmd


# Map job type names to their command builders.
# ComputeState and ComputeTrajectory need special handling (coord files),
# so they are not included here.
COMMAND_BUILDERS: dict[str, Any] = {
    "Pipeline": build_pipeline_command,
    "Analyze": build_analyze_command,
}

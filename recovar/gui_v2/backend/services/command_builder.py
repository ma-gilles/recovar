"""Build CLI command lists for recovar job types.

Each builder takes validated parameters from the API and returns a
``list[str]`` suitable for ``subprocess`` or sbatch.  No user strings
are interpolated into shell commands — everything is passed as separate
argv elements.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any


def _recovar_cmd() -> list[str]:
    """Return the argv prefix that invokes the ``recovar`` CLI.

    The ``recovar`` package uses a ``[project.scripts]`` entry point
    (``recovar.command_line:main_commands``), not ``__main__.py``, so
    ``python -m recovar`` does not work.  We prefer the ``recovar``
    script installed alongside the running Python interpreter, then fall
    back to the one on ``PATH``, and as a last resort invoke the
    console-script entry point directly through the interpreter.

    The result is a list so it can be spread into an argv list — a
    multi-token string would be treated by ``subprocess`` as a single
    executable path containing spaces.
    """
    # Look for 'recovar' next to the current Python executable
    python_dir = Path(sys.executable).parent
    candidate = python_dir / "recovar"
    if candidate.is_file():
        return [str(candidate)]
    # Fallback: hope it's on PATH
    found = shutil.which("recovar")
    if found:
        return [found]
    # Last resort: invoke the console-script entry point through the
    # interpreter (there is no recovar/__main__.py, so ``-m recovar``
    # would not work).
    return [
        sys.executable,
        "-c",
        "from recovar.command_line import main_commands; main_commands()",
    ]


def _add_optional(cmd: list[str], flag: str, value: Any) -> None:
    """Append ``--flag value`` to *cmd* if value is truthy."""
    if value is not None and value != "" and value is not False:
        cmd.extend([flag, str(value)])


def _add_flag(cmd: list[str], flag: str, value: bool | None) -> None:
    """Append ``--flag`` to *cmd* if value is True."""
    if value:
        cmd.append(flag)


def _add_bool_optional(cmd: list[str], flag_base: str, value: bool | None) -> None:
    """Emit ``--flag`` for True, ``--no-flag`` for False, nothing for None.

    For argparse ``BooleanOptionalAction`` options (e.g. ``--do-over-with-contrast``)
    where the default is context-dependent, so we only pass a flag when the
    user has made an explicit choice.
    """
    if value is True:
        cmd.append(f"--{flag_base}")
    elif value is False:
        cmd.append(f"--no-{flag_base}")


def build_pipeline_command(params: dict[str, Any]) -> list[str]:
    """Build ``recovar pipeline`` command from validated parameters.

    Required params: particles, mask
    Optional params: all other pipeline CLI flags
    """
    cmd = [*_recovar_cmd(), "pipeline"]

    # Positional: particles
    cmd.append(params["particles"])

    # Required. ``recovar pipeline`` declares --mask as required=True, and
    # _add_optional drops empty strings, so guard explicitly here to surface
    # a clear error instead of a cryptic argparse "required" abort at launch.
    mask = params.get("mask")
    if not mask:
        raise ValueError("Mask is required for a pipeline job (--mask).")
    _add_optional(cmd, "--mask", mask)

    # Output
    _add_optional(cmd, "-o", params.get("outdir"))
    _add_optional(cmd, "--project", params.get("project"))
    _add_optional(cmd, "--output-name", params.get("output_name"))

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
    _add_flag(cmd, "--tilt-series", params.get("tilt_series"))
    _add_optional(cmd, "--tilt-series-ctf", params.get("tilt_series_ctf"))

    # Processing flags
    _add_optional(cmd, "--downsample", params.get("downsample"))
    _add_flag(cmd, "--lazy", params.get("lazy"))
    # --correct-contrast is a BooleanOptionalAction (--correct-contrast /
    # --no-correct-contrast), like do-over-with-contrast below, so use the
    # BooleanOptionalAction-aware helper to allow an explicit "off".
    _add_bool_optional(cmd, "correct-contrast", params.get("correct_contrast"))
    _add_bool_optional(cmd, "do-over-with-contrast", params.get("do_over_with_contrast"))
    _add_optional(cmd, "--focus-mask", params.get("focus_mask"))
    # NOTE: --Bfactor and --n-bins are NOT pipeline flags. They are defined
    # only in standard_downstream_args (analyze/compute_state/compute_trajectory),
    # so emitting them here would make argparse abort with "unrecognized
    # arguments" if a (e.g. cloned/prefilled) job carried those params.

    # Memory planning (matches recovar.utils.parser_args.add_memory_planning_args).
    # --adaptive-n-pcs is ON by default in recovar; --gpu-budget-gb + the low-memory
    # knobs let large box-256 runs fit a budget instead of OOMing.
    _add_optional(cmd, "--gpu-budget-gb", params.get("gpu_budget_gb"))
    _add_bool_optional(cmd, "adaptive-n-pcs", params.get("adaptive_n_pcs"))
    _add_flag(cmd, "--low-memory-option", params.get("low_memory_option"))
    _add_flag(cmd, "--very-low-memory-option", params.get("very_low_memory_option"))

    return cmd


def build_analyze_command(params: dict[str, Any]) -> list[str]:
    """Build ``recovar analyze`` command."""
    cmd = [*_recovar_cmd(), "analyze"]

    # Positional: result_dir
    cmd.append(params["result_dir"])

    # Required
    _add_optional(cmd, "--zdim", params.get("zdim"))

    # Optional
    _add_optional(cmd, "-o", params.get("outdir"))
    _add_optional(cmd, "--output-name", params.get("output_name"))
    _add_optional(cmd, "--n-clusters", params.get("n_clusters"))
    _add_optional(cmd, "--n-trajectories", params.get("n_trajectories"))
    _add_optional(cmd, "--n-vols-along-path", params.get("n_vols_along_path"))
    # Kernel-regression speed/quality knobs (see issue #14): lower n-bins and
    # maskrad-fraction trade resolution for a large speedup on the k-means
    # center volumes. The "Quick Analyze" button sets both to 10.
    _add_optional(cmd, "--n-bins", params.get("n_bins"))
    _add_optional(cmd, "--maskrad-fraction", params.get("maskrad_fraction"))
    _add_flag(cmd, "--skip-umap", params.get("skip_umap"))
    _add_flag(cmd, "--no-z-regularization", params.get("no_z_regularization"))

    return cmd


def build_compute_state_command(
    params: dict[str, Any],
    latent_points_file: str,
) -> list[str]:
    """Build ``recovar compute_state`` command.

    *latent_points_file* is the path to a text file containing the
    latent coordinate vector, written by the API before submission.
    """
    cmd = [*_recovar_cmd(), "compute_state"]

    cmd.append(params["result_dir"])

    # Note: compute_state does not accept --zdim; the dimensionality is read
    # from the latent_points file. For a 1-D embedding (zdim=1) the file holds
    # a single value, which np.loadtxt returns as a 0-d/scalar array; that case
    # is ambiguous, so compute_state requires the explicit --zdim1 flag.
    _add_optional(cmd, "-o", params.get("outdir"))
    cmd.extend(["--latent-points", latent_points_file])
    if params.get("zdim") == 1:
        cmd.append("--zdim1")

    return cmd


def build_compute_trajectory_command(
    params: dict[str, Any],
    z_start_file: str,
    z_end_file: str,
) -> list[str]:
    """Build ``recovar compute_trajectory`` command."""
    cmd = [*_recovar_cmd(), "compute_trajectory"]

    cmd.append(params["result_dir"])

    _add_optional(cmd, "-o", params.get("outdir"))
    _add_optional(cmd, "--zdim", params.get("zdim"))
    cmd.extend(["--z_st", z_start_file])
    cmd.extend(["--z_end", z_end_file])
    _add_optional(cmd, "--n-vols-along-path", params.get("n_vols_along_path"))
    _add_optional(cmd, "--density", params.get("density"))

    return cmd


def build_density_command(params: dict[str, Any]) -> list[str]:
    """Build ``recovar estimate_conformational_density`` command."""
    cmd = [*_recovar_cmd(), "estimate_conformational_density"]
    cmd.append(params["result_dir"])
    _add_optional(cmd, "--output_dir", params.get("outdir"))
    _add_optional(cmd, "--pca_dim", params.get("pca_dim"))
    _add_optional(cmd, "--z_dim_used", params.get("z_dim_used"))
    _add_optional(cmd, "--percentile_reject", params.get("percentile_reject"))
    _add_optional(cmd, "--num_disc_points", params.get("num_disc_points"))
    _add_optional(cmd, "--percentile_bound", params.get("percentile_bound"))
    return cmd


def build_stable_states_command(params: dict[str, Any]) -> list[str]:
    """Build ``recovar estimate_stable_states`` command."""
    cmd = [*_recovar_cmd(), "estimate_stable_states"]
    cmd.append(params["density"])
    _add_optional(cmd, "-o", params.get("outdir"))
    _add_optional(cmd, "--percent_top", params.get("percent_top"))
    _add_optional(cmd, "--n_local_maxs", params.get("n_local_maxs"))
    return cmd


def build_postprocess_command(params: dict[str, Any]) -> list[str]:
    """Build ``recovar postprocess`` command."""
    cmd = [*_recovar_cmd(), "postprocess"]
    cmd.append(params["input"])
    _add_optional(cmd, "--output", params.get("outdir"))
    _add_optional(cmd, "--halfmap2", params.get("halfmap2"))
    _add_optional(cmd, "--voxel-size", params.get("voxel_size"))
    _add_optional(cmd, "--B-factor", params.get("B_factor"))
    _add_optional(cmd, "--mask-radius", params.get("mask_radius"))
    _add_optional(cmd, "--fsc-mask", params.get("fsc_mask"))
    _add_optional(cmd, "--apply-mask", params.get("apply_mask"))
    _add_flag(cmd, "--batch", params.get("batch"))
    _add_flag(cmd, "--estimate-B-factor", params.get("estimate_B_factor"))
    _add_flag(cmd, "--local", params.get("local"))
    _add_optional(cmd, "--locres-sampling", params.get("locres_sampling"))
    _add_optional(cmd, "--locres-maskrad", params.get("locres_maskrad"))
    _add_optional(cmd, "--locres-edgwidth", params.get("locres_edgwidth"))
    return cmd


def build_downsample_command(params: dict[str, Any]) -> list[str]:
    """Build ``recovar downsample`` command."""
    cmd = [*_recovar_cmd(), "downsample"]
    cmd.append(params["particles"])
    cmd.extend(["-D", str(params["target_D"])])
    _add_optional(cmd, "-o", params.get("outdir"))
    _add_optional(cmd, "--datadir", params.get("datadir"))
    _add_optional(cmd, "--strip-prefix", params.get("strip_prefix"))
    _add_optional(cmd, "--batch-size", params.get("batch_size"))
    return cmd


# Map job type names to their command builders.
# ComputeState and ComputeTrajectory need special handling (coord files),
# so they are not included here.
COMMAND_BUILDERS: dict[str, Any] = {
    "Pipeline": build_pipeline_command,
    "Analyze": build_analyze_command,
    "Density": build_density_command,
    "StableStates": build_stable_states_command,
    "Postprocess": build_postprocess_command,
    "Downsample": build_downsample_command,
}

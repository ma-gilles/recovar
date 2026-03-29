"""Filesystem scanner for importing existing recovar pipeline outputs.

Walks a project directory (or arbitrary path) looking for completed jobs
based on the directory structure and metadata files defined in
``recovar.output.output_paths`` and ``recovar.project.registry``.

Imported jobs get status ``COMPLETED`` in the GUI database.  Jobs whose
metadata cannot be fully parsed get status ``COMPLETED`` with a
``"legacy_import": true`` flag in params.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Pattern for job directories: job_0001, job_0002, etc.
_JOB_DIR_RE = re.compile(r"^job_(\d{4,})$")


@dataclass
class ScannedJob:
    """A job discovered on the filesystem during a scan."""

    type: str  # e.g. "Pipeline", "Analyze"
    output_dir: str  # absolute path
    status: str = "completed"
    params: dict = field(default_factory=dict)
    created_at: datetime.datetime | None = None
    completed_at: datetime.datetime | None = None
    parent_job_dirs: list[str] = field(default_factory=list)
    legacy: bool = False


def _read_job_json(job_dir: str) -> dict | None:
    """Read job.json from a job directory, or None if missing/invalid."""
    path = os.path.join(job_dir, "job.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not parse %s: %s", path, exc)
        return None


def _read_metadata_json(job_dir: str) -> dict | None:
    """Read model/metadata.json from a pipeline job directory."""
    path = os.path.join(job_dir, "model", "metadata.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not parse %s: %s", path, exc)
        return None


def _extract_params_from_pkl(job_dir: str) -> dict:
    """Extract pipeline parameters from model/params.pkl for legacy imports.

    Reads the pickled params dict and extracts the ``input_args`` Namespace
    to recover the original command-line parameters (particles, mask, zdim, etc.).
    Also extracts volume_shape and voxel_size from the params dict itself.
    """
    pkl_path = os.path.join(job_dir, "model", "params.pkl")
    if not os.path.isfile(pkl_path):
        return {}

    try:
        import pickle
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception as exc:
        logger.warning("Could not load %s: %s", pkl_path, exc)
        return {}

    params: dict = {}

    # Extract input_args (argparse Namespace stored as numpy 0-d array)
    input_args = data.get("input_args")
    if input_args is not None:
        # May be wrapped in a numpy 0-d array
        if hasattr(input_args, "item"):
            input_args = input_args.item()
        if hasattr(input_args, "__dict__"):
            args_dict = vars(input_args)
            # Extract the most useful parameters
            _keys = [
                "particles", "outdir", "zdim", "mask", "poses", "ctf",
                "ind", "halfsets", "downsample", "lazy", "correct_contrast",
                "focus_mask", "n_images", "datadir", "tilt_series",
                "premultiplied_ctf",
            ]
            for k in _keys:
                if k in args_dict and args_dict[k] is not None:
                    v = args_dict[k]
                    # Convert numpy types to plain Python
                    if hasattr(v, "tolist"):
                        v = v.tolist()
                    params[k] = v

    # Extract top-level metadata
    if "volume_shape" in data:
        vs = data["volume_shape"]
        params["volume_shape"] = vs.tolist() if hasattr(vs, "tolist") else list(vs)
    if "voxel_size" in data:
        vx = data["voxel_size"]
        params["voxel_size"] = float(vx.item()) if hasattr(vx, "item") else float(vx)
    if "version" in data:
        params["recovar_version"] = str(data["version"])

    return params


def _is_pipeline_output(job_dir: str) -> bool:
    """Check if a directory looks like a pipeline output."""
    # Primary check: model/metadata.json
    if os.path.isfile(os.path.join(job_dir, "model", "metadata.json")):
        return True
    # Fallback: model/params.pkl (older versions)
    if os.path.isfile(os.path.join(job_dir, "model", "params.pkl")):
        return True
    return False


def _is_analyze_output(job_dir: str) -> bool:
    """Check if a directory looks like an analyze output.

    Analyze outputs live inside a pipeline directory as ``analysis_N/``
    subdirectories, but can also be standalone job directories.
    """
    # Check for kmeans data or plots subdirectory typical of analyze
    if os.path.isdir(os.path.join(job_dir, "data")) and os.path.isdir(
        os.path.join(job_dir, "plots")
    ):
        return True
    if os.path.isdir(os.path.join(job_dir, "kmeans")):
        return True
    return False


def _detect_job_type_from_dir(type_dir_name: str, job_dir: str) -> str | None:
    """Infer the job type from the parent directory name and contents."""
    # Try to match known directory names from the registry
    # Import here to avoid circular imports at module level
    try:
        from recovar.project.registry import JOB_TYPES

        for jt in JOB_TYPES.values():
            if jt.dir_name == type_dir_name:
                return jt.name
    except ImportError:
        pass

    # Fallback: heuristic detection
    known_dirs = {
        "Pipeline": "Pipeline",
        "Analyze": "Analyze",
        "ReconstructState": "ReconstructState",
        "ReconstructTrajectory": "ReconstructTrajectory",
        "Density": "Density",
        "StableStates": "StableStates",
        "JunkDetection": "JunkDetection",
        "OutlierDetection": "OutlierDetection",
        "Postprocess": "Postprocess",
        "Downsample": "Downsample",
        "ExtractSubset": "ExtractSubset",
        "PipelineWithOutliers": "PipelineWithOutliers",
        "ReconstructExternal": "ReconstructExternal",
    }
    return known_dirs.get(type_dir_name)


def _scan_job_dir(type_name: str, job_dir: str) -> ScannedJob:
    """Build a ScannedJob from a single job directory."""
    job_data = _read_job_json(job_dir)

    params: dict = {}
    created_at: datetime.datetime | None = None
    completed_at: datetime.datetime | None = None
    status = "completed"
    parent_dirs: list[str] = []
    legacy = False

    if job_data:
        params = job_data.get("parameters", {})
        status = job_data.get("status", "completed")

        timing = job_data.get("timing", {})
        if timing.get("started_at"):
            try:
                created_at = datetime.datetime.fromisoformat(timing["started_at"])
            except ValueError:
                pass
        if timing.get("completed_at"):
            try:
                completed_at = datetime.datetime.fromisoformat(timing["completed_at"])
            except ValueError:
                pass

        prov = job_data.get("provenance", {})
        if prov.get("pipeline_result_dir"):
            parent_dirs.append(prov["pipeline_result_dir"])
    else:
        # No job.json — legacy import
        legacy = True
        # Try to get a creation time from directory mtime
        try:
            mtime = os.path.getmtime(job_dir)
            created_at = datetime.datetime.fromtimestamp(mtime)
        except OSError:
            pass
        # Extract params from metadata.json or params.pkl
        meta = _read_metadata_json(job_dir)
        if meta:
            params = meta
        if type_name == "Pipeline":
            pkl_params = _extract_params_from_pkl(job_dir)
            if pkl_params:
                params.update(pkl_params)

    return ScannedJob(
        type=type_name,
        output_dir=os.path.abspath(job_dir),
        status=status,
        params=params,
        created_at=created_at,
        completed_at=completed_at,
        parent_job_dirs=parent_dirs,
        legacy=legacy,
    )


def scan_project_directory(project_dir: str) -> list[ScannedJob]:
    """Scan a project directory for existing job outputs.

    Looks for ``{TypeDir}/job_{NNNN}/`` subdirectories matching the
    recovar project convention.  Also detects ``analysis_N/``
    directories inside pipeline jobs.

    Parameters
    ----------
    project_dir : str
        Absolute path to the project root directory.

    Returns
    -------
    list[ScannedJob]
        Discovered jobs, sorted by creation time (oldest first).
    """
    project_dir = os.path.abspath(project_dir)
    results: list[ScannedJob] = []

    if not os.path.isdir(project_dir):
        logger.warning("Scan path does not exist: %s", project_dir)
        return results

    # Walk top-level directories looking for type dirs (Pipeline/, Analyze/, etc.)
    try:
        top_entries = sorted(os.listdir(project_dir))
    except OSError as exc:
        logger.error("Cannot list directory %s: %s", project_dir, exc)
        return results

    for type_dir_name in top_entries:
        type_dir = os.path.join(project_dir, type_dir_name)
        if not os.path.isdir(type_dir):
            continue

        # Check if this looks like a job-type directory
        job_type = _detect_job_type_from_dir(type_dir_name, type_dir)
        if job_type is None:
            continue

        # Scan for job_NNNN subdirectories
        try:
            job_entries = sorted(os.listdir(type_dir))
        except OSError:
            continue

        for job_name in job_entries:
            if not _JOB_DIR_RE.match(job_name):
                continue
            job_dir = os.path.join(type_dir, job_name)
            if not os.path.isdir(job_dir):
                continue

            scanned = _scan_job_dir(job_type, job_dir)
            results.append(scanned)

    # Sort by creation time (oldest first)
    results.sort(key=lambda s: s.created_at or datetime.datetime.min)
    return results


def scan_arbitrary_directory(scan_path: str) -> list[ScannedJob]:
    """Scan an arbitrary directory for recovar outputs.

    Unlike :func:`scan_project_directory`, this handles directories
    that may not follow the project convention.  It looks for:

    1. Direct pipeline outputs (``model/metadata.json`` or ``model/params.pkl``)
    2. Project-structured directories with ``{TypeDir}/job_NNNN/`` layout

    Parameters
    ----------
    scan_path : str
        Absolute path to scan.

    Returns
    -------
    list[ScannedJob]
    """
    scan_path = os.path.abspath(scan_path)

    # First check if this IS a pipeline output directly
    if _is_pipeline_output(scan_path):
        return [_scan_job_dir("Pipeline", scan_path)]

    # Check if it's an analyze output directly
    if _is_analyze_output(scan_path):
        return [_scan_job_dir("Analyze", scan_path)]

    # Otherwise treat it as a project directory
    return scan_project_directory(scan_path)

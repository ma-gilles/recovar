"""RecovarProject — project directory management with job tracking.

A project is a directory containing a ``project.json`` index and type-specific
subdirectories (``Pipeline/``, ``Analyze/``, etc.) where numbered job
directories live.

Concurrency safety: all project.json mutations are protected by an
``fcntl.flock``-based file lock, safe for parallel Slurm jobs.
"""

import datetime
import fcntl
import json
import logging
import os
import re
from contextlib import contextmanager
from typing import Any, Mapping, Optional

from recovar.project.registry import JOB_TYPES, get_job_type

logger = logging.getLogger(__name__)

PROJECT_FILE = "project.json"
LOCK_FILE = ".project.lock"
PROJECT_VERSION = "1.0"
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _get_recovar_version():
    try:
        from recovar import __version__

        return str(__version__)
    except Exception:
        return "unknown"


def find_project_root(start_dir: Optional[str] = None) -> Optional[str]:
    """Walk up from *start_dir* (default: cwd) looking for project.json.

    Returns the project root directory, or None if not found.
    """
    d = os.path.abspath(start_dir or os.getcwd())
    for _ in range(20):  # safety limit
        if os.path.isfile(os.path.join(d, PROJECT_FILE)):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def normalize_job_alias(value: str) -> str:
    """Normalize a user- or system-provided job label for CLI/UI use."""
    text = _NON_ALNUM_RE.sub("_", str(value).strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "job"


def uniquify_job_alias(alias: str, existing: set[str]) -> str:
    """Return *alias* or a numbered variant that is unique within *existing*."""
    if alias not in existing:
        return alias

    suffix = 2
    while True:
        candidate = f"{alias}_{suffix}"
        if candidate not in existing:
            return candidate
        suffix += 1


def _coerce_int(value: Any) -> int | None:
    if value in (None, "", False):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _basename_stem(path: Any) -> str | None:
    if not path:
        return None
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    alias = normalize_job_alias(stem)
    return alias if alias != "job" else None


def default_job_alias(command_name: str, params: Mapping[str, Any] | None = None) -> str:
    """Generate a readable default alias for a recovar job."""
    params = params or {}
    explicit = params.get("output_name")
    if explicit:
        return normalize_job_alias(str(explicit))

    command = str(command_name).strip().lower()

    if command == "pipeline":
        particle_stem = _basename_stem(params.get("particles"))
        downsample = _coerce_int(params.get("downsample"))
        if particle_stem and downsample:
            return f"{particle_stem}_d{downsample}"
        if particle_stem:
            return particle_stem
        return "pipeline"

    if command == "analyze":
        zdim = _coerce_int(params.get("zdim"))
        return f"embedding_k{zdim}" if zdim else "embedding"

    if command == "compute_state":
        zdim = _coerce_int(params.get("zdim"))
        return f"compute_state_k{zdim}" if zdim else "compute_state"

    if command == "compute_trajectory":
        zdim = _coerce_int(params.get("zdim"))
        return f"trajectory_k{zdim}" if zdim else "trajectory"

    if command == "estimate_conformational_density":
        zdim = _coerce_int(params.get("z_dim_used")) or _coerce_int(params.get("pca_dim"))
        return f"density_k{zdim}" if zdim else "density"

    if command == "estimate_stable_states":
        return "stable_states"

    if command == "junk_particle_detection":
        zdim = _coerce_int(params.get("zdim"))
        return f"junk_detection_k{zdim}" if zdim else "junk_detection"

    if command == "outlier_detection":
        return "outlier_detection"

    if command == "postprocess":
        input_stem = _basename_stem(params.get("input"))
        return f"postprocess_{input_stem}" if input_stem else "postprocess"

    if command == "downsample":
        target_d = _coerce_int(params.get("target_D"))
        particle_stem = _basename_stem(params.get("particles"))
        if particle_stem and target_d:
            return f"{particle_stem}_d{target_d}"
        if target_d:
            return f"downsample_d{target_d}"
        return "downsample"

    if command == "pipeline_with_outliers":
        return "pipeline_with_outliers"

    if command == "reconstruct_from_external_embedding":
        return "external_reconstruction"

    return normalize_job_alias(command.replace("-", "_"))


def infer_job_display_name(
    type_name: str,
    params: Mapping[str, Any] | None = None,
    output_dir: str | None = None,
    alias: str | None = None,
) -> str:
    """Return the best available human-readable label for a job."""
    if alias:
        return alias

    type_to_command = {
        "Pipeline": "pipeline",
        "Analyze": "analyze",
        "ComputeState": "compute_state",
        "ReconstructState": "compute_state",
        "ComputeTrajectory": "compute_trajectory",
        "ReconstructTrajectory": "compute_trajectory",
        "Density": "estimate_conformational_density",
        "StableStates": "estimate_stable_states",
        "JunkDetection": "junk_particle_detection",
        "OutlierDetection": "outlier_detection",
        "Postprocess": "postprocess",
        "Downsample": "downsample",
        "PipelineWithOutliers": "pipeline_with_outliers",
        "ReconstructExternal": "reconstruct_from_external_embedding",
    }
    command_name = type_to_command.get(type_name, type_name)
    display = default_job_alias(command_name, params)
    if display and display != "job":
        return display

    if output_dir:
        tail = os.path.basename(os.path.normpath(output_dir))
        if tail:
            return tail
    return normalize_job_alias(type_name)


class RecovarProject:
    """Manages a recovar project directory.

    Usage::

        proj = RecovarProject("/path/to/my_project")
        uid, job_dir = proj.allocate_job("compute_state")
        # job_dir = "/path/to/my_project/ComputeState/job_0001"
        proj.register_job_start(uid, args)
        # ... do work ...
        proj.register_job_complete(uid)

    Parameters
    ----------
    root : str
        Absolute path to the project directory.
    """

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self._project_file = os.path.join(self.root, PROJECT_FILE)
        self._lock_file = os.path.join(self.root, LOCK_FILE)

    @classmethod
    def init(cls, directory: str, name: Optional[str] = None) -> "RecovarProject":
        """Initialize a new project in *directory*.

        Creates the directory (if needed) and writes an empty ``project.json``.
        """
        directory = os.path.abspath(directory)
        os.makedirs(directory, exist_ok=True)

        proj_file = os.path.join(directory, PROJECT_FILE)
        if os.path.exists(proj_file):
            logger.info("Project already exists at %s", directory)
            return cls(directory)

        data = {
            "version": PROJECT_VERSION,
            "name": name or os.path.basename(directory),
            "created": datetime.datetime.now().isoformat(),
            "recovar_version": _get_recovar_version(),
            "counters": {},
            "jobs": [],
            "aliases": {},
        }
        with open(proj_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Initialized project at %s", directory)
        return cls(directory)

    @property
    def exists(self) -> bool:
        """Whether project.json exists."""
        return os.path.isfile(self._project_file)

    # ------------------------------------------------------------------
    # File locking
    # ------------------------------------------------------------------

    @contextmanager
    def _lock(self):
        """Acquire an exclusive file lock on .project.lock."""
        os.makedirs(self.root, exist_ok=True)
        fd = open(self._lock_file, "w")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            fd.close()

    def _read(self) -> dict:
        """Read project.json (must be called under lock)."""
        if not os.path.isfile(self._project_file):
            return {
                "version": PROJECT_VERSION,
                "name": os.path.basename(self.root),
                "created": datetime.datetime.now().isoformat(),
                "recovar_version": _get_recovar_version(),
                "counters": {},
                "jobs": [],
                "aliases": {},
            }
        with open(self._project_file) as f:
            return json.load(f)

    def _write(self, data: dict):
        """Write project.json atomically (must be called under lock)."""
        tmp = self._project_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self._project_file)

    # ------------------------------------------------------------------
    # Job allocation
    # ------------------------------------------------------------------

    def allocate_job(self, command_name: str) -> tuple:
        """Allocate the next job number for a command type.

        Returns ``(uid, job_dir)`` where *uid* is e.g. ``"ComputeState/job_0001"``
        and *job_dir* is the absolute path to the new job directory.
        """
        jt = get_job_type(command_name)
        if jt is None:
            raise ValueError(f"Unknown command: {command_name}")

        with self._lock():
            data = self._read()
            counters = data.setdefault("counters", {})
            current = counters.get(jt.name, 0)
            next_num = current + 1
            counters[jt.name] = next_num
            self._write(data)

        uid = f"{jt.dir_name}/job_{next_num:04d}"
        job_dir = os.path.join(self.root, uid)
        os.makedirs(job_dir, exist_ok=True)
        return uid, job_dir

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------

    def register_job_start(
        self,
        uid: str,
        command_name: str,
        command_line: str = "",
        parent_jobs: Optional[list] = None,
        alias: Optional[str] = None,
        description: str = "",
    ):
        """Add a job entry to project.json with status=running."""
        entry = {
            "uid": uid,
            "type": get_job_type(command_name).name if get_job_type(command_name) else command_name,
            "status": "running",
            "created": datetime.datetime.now().isoformat(),
            "completed": None,
            "parent_jobs": parent_jobs or [],
            "alias": None,
            "description": description,
        }
        with self._lock():
            data = self._read()
            jobs = [j for j in data.setdefault("jobs", []) if j.get("uid") != uid]
            aliases = {
                name: target
                for name, target in data.setdefault("aliases", {}).items()
                if target != uid
            }
            existing_aliases = {j.get("alias") for j in jobs if j.get("alias")}
            if alias:
                alias = uniquify_job_alias(normalize_job_alias(alias), existing_aliases)
                entry["alias"] = alias
                aliases[alias] = uid
            jobs.append(entry)
            data["jobs"] = jobs
            data["aliases"] = aliases
            self._write(data)

    def register_job_complete(self, uid: str, status: str = "completed"):
        """Update a job's status and completion time."""
        with self._lock():
            data = self._read()
            for job in data.get("jobs", []):
                if job.get("uid") == uid:
                    job["status"] = status
                    job["completed"] = datetime.datetime.now().isoformat()
                    break
            self._write(data)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_jobs(self, type_name: Optional[str] = None) -> list:
        """Return all job entries, optionally filtered by type."""
        with self._lock():
            data = self._read()
        jobs = data.get("jobs", [])
        if type_name:
            jobs = [j for j in jobs if j.get("type") == type_name]
        return jobs

    def find_latest_job(self, type_name: str, status: str = "completed") -> Optional[str]:
        """Find the most recent job of a given type with the given status.

        Returns the uid (e.g. ``"Pipeline/job_0002"``) or None.
        """
        jobs = self.list_jobs(type_name)
        candidates = [j for j in jobs if j.get("status") == status]
        if not candidates:
            return None
        # Sort by created timestamp (lexicographic ISO works)
        candidates.sort(key=lambda j: j.get("created", ""), reverse=True)
        return candidates[0]["uid"]

    def get_job_dir(self, uid: str) -> str:
        """Return the absolute path for a job uid."""
        return os.path.join(self.root, uid)

    def get_job_alias_map(self) -> dict[str, str]:
        """Return a mapping of job uid -> alias for jobs that define one."""
        with self._lock():
            data = self._read()
        return {job["uid"]: job["alias"] for job in data.get("jobs", []) if job.get("alias")}

    def infer_uid_from_job_dir(self, job_dir: str, expected_command: Optional[str] = None) -> Optional[str]:
        """Infer a project job uid from an on-disk job directory path."""
        abs_job_dir = os.path.abspath(job_dir)
        try:
            rel = os.path.relpath(abs_job_dir, self.root)
        except ValueError:
            return None
        if rel.startswith(".."):
            return None
        parts = rel.split(os.sep)
        if len(parts) < 2 or not parts[1].startswith("job_"):
            return None
        if expected_command is not None:
            jt = get_job_type(expected_command)
            if jt is not None and parts[0] != jt.dir_name:
                return None
        return f"{parts[0]}/{parts[1]}"

    def resolve_pipeline(self, result_dir_arg: Optional[str] = None) -> str:
        """Resolve a pipeline directory from a CLI argument.

        Resolution order:
        1. Absolute path → use directly
        2. Relative path (e.g. ``Pipeline/job_0001``) → resolve within project
        3. None → find latest completed Pipeline job
        """
        if result_dir_arg is not None:
            # Absolute path
            if os.path.isabs(result_dir_arg):
                return result_dir_arg
            # Relative within project
            candidate = os.path.join(self.root, result_dir_arg)
            if os.path.isdir(candidate):
                return candidate
            # Maybe just a job number like "job_0001"
            candidate2 = os.path.join(self.root, "Pipeline", result_dir_arg)
            if os.path.isdir(candidate2):
                return candidate2

            # Project alias (e.g. "ribosome_d128")
            with self._lock():
                aliases = self._read().get("aliases", {})
            alias_target = aliases.get(result_dir_arg)
            if alias_target is None and os.sep not in result_dir_arg:
                alias_target = aliases.get(normalize_job_alias(result_dir_arg))
            if alias_target:
                candidate3 = self.get_job_dir(alias_target)
                if os.path.isdir(candidate3):
                    return candidate3

            # Fall back to treating as absolute
            return os.path.abspath(result_dir_arg)

        # No argument — find latest pipeline
        uid = self.find_latest_job("Pipeline")
        if uid is None:
            raise ValueError("No completed Pipeline job found in project. Run 'recovar pipeline' first.")
        return self.get_job_dir(uid)

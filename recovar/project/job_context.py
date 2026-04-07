"""Context manager for project-aware job execution.

Wraps any command's ``main()`` to handle project detection, job allocation,
job.json lifecycle, and logging setup — all in one place.
"""

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

from recovar.output.job import JobDir
from recovar.project.project import RecovarProject, default_job_alias, find_project_root
from recovar.project.registry import get_job_type

logger = logging.getLogger(__name__)


@dataclass
class JobContext:
    """Result yielded by :func:`job_context`.

    Attributes
    ----------
    project : RecovarProject or None
        The project, if in project mode.
    output_dir : str
        Resolved output directory for this job.
    pipeline_dir : str or None
        Resolved pipeline input directory (for commands that need one).
    uid : str or None
        Job uid within the project (e.g. ``"ComputeState/job_0001"``).
    job : JobDir
        The underlying JobDir for this job.
    parent_jobs : list of str
        UIDs of parent jobs (for DAG tracking in project.json).
    """

    project: Optional[RecovarProject] = None
    output_dir: str = ""
    pipeline_dir: Optional[str] = None
    uid: Optional[str] = None
    job: Optional[JobDir] = None
    parent_jobs: list = field(default_factory=list)


def _detect_project(args) -> Optional[RecovarProject]:
    """Detect project from --project arg or auto-detection from cwd."""
    project_arg = getattr(args, "project", None)
    if project_arg is not None:
        root = os.path.abspath(project_arg)
        if not os.path.isfile(os.path.join(root, "project.json")):
            # Auto-init if --project points to an empty/new dir
            return RecovarProject.init(root)
        return RecovarProject(root)

    # Auto-detect from cwd
    root = find_project_root()
    if root is not None:
        return RecovarProject(root)

    return None


def _resolve_parent_uid(project: RecovarProject, result_dir: str) -> Optional[str]:
    """Try to determine the parent job uid from a result_dir path."""
    if project is None:
        return None
    # Check if result_dir is inside the project
    try:
        rel = os.path.relpath(result_dir, project.root)
    except ValueError:
        return None
    # Format: "TypeName/job_NNNN" or "TypeName/job_NNNN/..."
    parts = rel.split(os.sep)
    if len(parts) >= 2 and parts[1].startswith("job_"):
        return f"{parts[0]}/{parts[1]}"
    return None


@contextmanager
def job_context(args, command_name: str):
    """Context manager for project-aware job execution.

    Handles:
    - Project detection (from ``--project`` or auto-detection)
    - Job number allocation
    - Output directory resolution
    - Pipeline directory resolution (for downstream commands)
    - ``job.json`` start/complete lifecycle
    - Logging setup

    Usage::

        with job_context(args, "compute_state") as ctx:
            # ctx.output_dir is the resolved job directory
            # ctx.pipeline_dir is the resolved pipeline input
            compute_state(ctx.pipeline_dir, ctx.output_dir, ...)

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments. May have ``project``, ``outdir``,
        ``result_dir`` attributes.
    command_name : str
        The CLI command name (e.g. ``"compute_state"``).
    """
    jt = get_job_type(command_name)
    project = _detect_project(args)
    ctx = JobContext()
    ctx.project = project

    outdir = getattr(args, "outdir", None) or getattr(args, "output_dir", None) or getattr(args, "output", None)
    result_dir = getattr(args, "result_dir", None) or getattr(args, "recovar_result_dir", None)

    # --- Resolve output directory ---
    if project is not None and outdir is None:
        # Project mode: allocate a numbered job dir
        uid, job_dir = project.allocate_job(command_name)
        ctx.uid = uid
        ctx.output_dir = job_dir
    elif outdir is not None:
        # Explicit output dir (standalone or project override)
        ctx.output_dir = os.path.abspath(outdir)
        if project is not None:
            # Reuse a preallocated project job dir when one is provided (GUI path),
            # otherwise allocate a bookkeeping uid for this run.
            ctx.uid = project.infer_uid_from_job_dir(ctx.output_dir, expected_command=command_name)
            if ctx.uid is None:
                uid, _ = project.allocate_job(command_name)
                ctx.uid = uid
    elif result_dir is not None:
        # No project, no explicit outdir — auto-generate inside result_dir
        ctx.output_dir = os.path.join(os.path.abspath(result_dir), jt.dir_name if jt else command_name)
        os.makedirs(ctx.output_dir, exist_ok=True)
    else:
        raise ValueError(
            "No output directory specified. Use --project to enable auto-numbering, "
            "or -o to specify an output directory."
        )

    # --- Resolve pipeline directory ---
    if jt and jt.needs_pipeline:
        if project is not None:
            ctx.pipeline_dir = project.resolve_pipeline(result_dir)
        elif result_dir is not None:
            ctx.pipeline_dir = os.path.abspath(result_dir)
        else:
            raise ValueError(
                "No pipeline result directory specified. Pass result_dir explicitly, "
                "or use --project from within a project with a completed Pipeline job."
            )
    elif result_dir is not None:
        ctx.pipeline_dir = os.path.abspath(result_dir)

    # --- Track parent jobs ---
    if ctx.pipeline_dir and project is not None:
        parent_uid = _resolve_parent_uid(project, ctx.pipeline_dir)
        if parent_uid:
            ctx.parent_jobs = [parent_uid]

    # --- Create JobDir and start ---
    setattr(args, "_project_root", project.root if project is not None else None)
    setattr(args, "_project_uid", ctx.uid)
    setattr(args, "_resolved_pipeline_dir", ctx.pipeline_dir)

    job = JobDir(ctx.output_dir, command_name)
    ctx.job = job
    setattr(job, "_parent_result_dir", ctx.pipeline_dir)
    job.start(args)

    # --- Register in project ---
    if project is not None and ctx.uid is not None:
        import sys

        project.register_job_start(
            ctx.uid,
            command_name,
            command_line="python " + " ".join(sys.argv),
            parent_jobs=ctx.parent_jobs,
            alias=default_job_alias(command_name, vars(args)),
        )

    try:
        yield ctx
        job.complete()
        if project is not None and ctx.uid is not None:
            project.register_job_complete(ctx.uid, "completed")
    except Exception:
        job.complete(status="failed")
        if project is not None and ctx.uid is not None:
            project.register_job_complete(ctx.uid, "failed")
        raise

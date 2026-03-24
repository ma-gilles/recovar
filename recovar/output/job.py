"""Standardized job output directory management for RECOVAR commands.

Every RECOVAR command that produces output should use a :class:`JobDir` to get
consistent directory layout, provenance metadata (``job.json``), command-line
recording (``command.txt``), and logging (``run.log``).

Inspired by cryoSPARC's ``job.json``, Nextflow's provenance tracking, and
MLflow's structured metadata.

Usage::

    job = JobDir.create(
        outdir=args.outdir,
        command_name="compute_state",
        parent_result_dir=args.result_dir,
    )
    job.start(args)
    try:
        # ... run the command, writing outputs into job.root ...
        job.complete()
    except Exception:
        job.complete(status="failed")
        raise
"""

import datetime
import json
import logging
import os
import platform
import re
import subprocess
import sys

from recovar.utils.helpers import RobustFileHandler, RobustStreamHandler

logger = logging.getLogger(__name__)

_LOG_FMT = "%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s"


def _get_recovar_version():
    """Return the installed recovar version string."""
    try:
        from recovar import __version__
        return str(__version__)
    except Exception:
        return "unknown"


def _get_git_commit(repo_dir=None):
    """Return the short git commit hash, or None if unavailable."""
    if repo_dir is None:
        try:
            import recovar
            repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(recovar.__file__)))
        except Exception:
            return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir, capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _get_environment():
    """Collect environment metadata (hostname, SLURM, Python/JAX versions)."""
    env = {
        "hostname": platform.node(),
        "python_version": platform.python_version(),
    }

    # JAX version
    try:
        import jax
        env["jax_version"] = jax.__version__
    except Exception:
        pass

    # SLURM info
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        env["slurm_job_id"] = slurm_job_id
    slurm_job_name = os.environ.get("SLURM_JOB_NAME")
    if slurm_job_name:
        env["slurm_job_name"] = slurm_job_name

    # GPU info
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            gpus = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
            if gpus:
                env["gpu_devices"] = gpus
    except Exception:
        pass

    return env


def _args_to_dict(args):
    """Convert an argparse Namespace (or dict) to a JSON-serializable dict."""
    if args is None:
        return {}
    raw = vars(args) if hasattr(args, "__dict__") else dict(args)
    out = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = str(v)
    return out


def _collect_output_manifest(root_dir):
    """Scan the job directory and categorize output files."""
    volumes = []
    halfmaps = []
    plots = []
    diagnostics = []
    other = []

    for dirpath, _dirnames, filenames in os.walk(root_dir):
        reldir = os.path.relpath(dirpath, root_dir)
        for fname in sorted(filenames):
            if fname in ("job.json", "command.txt", "run.log"):
                continue
            relpath = os.path.join(reldir, fname) if reldir != "." else fname
            if fname.endswith(".mrc"):
                if "half" in fname and "unfil" in fname:
                    halfmaps.append(relpath)
                elif reldir.startswith("diagnostics"):
                    diagnostics.append(relpath)
                else:
                    volumes.append(relpath)
            elif fname.endswith(".png") or fname.endswith(".pdf"):
                plots.append(relpath)
            elif reldir.startswith("diagnostics"):
                diagnostics.append(relpath)
            else:
                other.append(relpath)

    manifest = {}
    if volumes:
        manifest["volumes"] = volumes
    if halfmaps:
        manifest["halfmaps"] = halfmaps
    if plots:
        manifest["plots"] = plots
    if diagnostics:
        manifest["diagnostics"] = diagnostics
    if other:
        manifest["other"] = other
    return manifest


class JobDir:
    """Standard output directory for any RECOVAR command.

    Provides consistent:
    - Directory structure
    - ``job.json`` provenance metadata (written at start, updated at completion)
    - ``command.txt`` recording the full command line
    - ``run.log`` with timestamped, NFS-robust logging

    Parameters
    ----------
    root_dir : str
        Absolute path to the job output directory.
    command_name : str
        Name of the RECOVAR command (e.g. ``"compute_state"``).
    """

    def __init__(self, root_dir, command_name):
        self.root = os.path.abspath(root_dir)
        self.command_name = command_name
        self._start_time = None

    @classmethod
    def create(cls, outdir, command_name, parent_result_dir=None,
               auto_number=False):
        """Create a JobDir, optionally with auto-numbered directory name.

        Parameters
        ----------
        outdir : str or None
            Explicit output directory path.  If *None* and *auto_number* is
            True, a numbered directory ``{command_name}_NNN/`` is created
            inside *parent_result_dir* (or the current directory).
        command_name : str
            RECOVAR command name (used as directory prefix for auto-numbering).
        parent_result_dir : str or None
            Parent pipeline result directory (recorded in provenance).
        auto_number : bool
            If True and *outdir* is None, auto-generate a numbered directory.

        Returns
        -------
        JobDir
        """
        if outdir is None and auto_number:
            parent = parent_result_dir if parent_result_dir else os.getcwd()
            outdir = cls._next_numbered_dir(parent, command_name)
            logger.info("Auto-numbered output directory: %s", outdir)

        if outdir is None:
            raise ValueError(
                "outdir is required (or use auto_number=True)"
            )

        job = cls(outdir, command_name)
        job._parent_result_dir = parent_result_dir
        return job

    @staticmethod
    def _next_numbered_dir(parent, prefix):
        """Find the next available ``{prefix}_NNN/`` directory under *parent*."""
        existing = []
        pattern = re.compile(rf"^{re.escape(prefix)}_(\d{{3,}})$")
        if os.path.isdir(parent):
            for name in os.listdir(parent):
                m = pattern.match(name)
                if m:
                    existing.append(int(m.group(1)))
        next_num = max(existing, default=0) + 1
        return os.path.join(parent, f"{prefix}_{next_num:03d}")

    # --- Standard paths ---

    @property
    def job_json(self):
        """Path to the job metadata file."""
        return os.path.join(self.root, "job.json")

    @property
    def command_txt(self):
        """Path to the command-line record file."""
        return os.path.join(self.root, "command.txt")

    @property
    def run_log(self):
        """Path to the log file."""
        return os.path.join(self.root, "run.log")

    @property
    def diagnostics_dir(self):
        """Path to the diagnostics subdirectory."""
        return os.path.join(self.root, "diagnostics")

    @property
    def plots_dir(self):
        """Path to the plots subdirectory."""
        return os.path.join(self.root, "plots")

    # --- Directory creation ---

    def ensure_dirs(self):
        """Create the job output directory (and diagnostics subdir)."""
        os.makedirs(self.root, exist_ok=True)

    # --- Lifecycle ---

    def start(self, args=None):
        """Initialize the job: create dirs, write command.txt, job.json, set up logging.

        Parameters
        ----------
        args : argparse.Namespace or dict, optional
            Command-line arguments to record in ``job.json``.
        """
        self._start_time = datetime.datetime.now()
        self.ensure_dirs()
        self._write_command_txt()
        self._write_job_json_start(args)
        self._setup_logging()
        logger.info("Job started: %s in %s", self.command_name, self.root)
        if args is not None:
            logger.info("Parameters: %s", args)

    def complete(self, status="completed"):
        """Finalize the job: update job.json with timing, status, output manifest.

        Parameters
        ----------
        status : str
            Final status (``"completed"`` or ``"failed"``).
        """
        end_time = datetime.datetime.now()
        duration = None
        if self._start_time is not None:
            duration = (end_time - self._start_time).total_seconds()

        # Read existing job.json
        try:
            with open(self.job_json) as f:
                job_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            job_data = {}

        job_data["status"] = status
        job_data.setdefault("timing", {})
        job_data["timing"]["completed_at"] = end_time.isoformat()
        if duration is not None:
            job_data["timing"]["duration_seconds"] = round(duration, 1)

        # Scan outputs
        job_data["outputs"] = _collect_output_manifest(self.root)

        try:
            with open(self.job_json, "w") as f:
                json.dump(job_data, f, indent=2)
        except (IOError, OSError) as e:
            logger.warning("Could not update job.json: %s", e)

        logger.info("Job %s: %s (%.1fs)", status, self.command_name,
                     duration if duration else 0)

    # --- Internal helpers ---

    def _write_command_txt(self):
        """Record the full command line."""
        try:
            with open(self.command_txt, "w") as f:
                f.write("python " + " ".join(sys.argv) + "\n")
        except (IOError, OSError) as e:
            logger.warning("Could not write command.txt: %s", e)

    def _write_job_json_start(self, args=None):
        """Write the initial job.json at job start."""
        parent_result_dir = getattr(self, "_parent_result_dir", None)

        job_data = {
            "recovar_version": _get_recovar_version(),
            "git_commit": _get_git_commit(),
            "command": self.command_name,
            "command_line": "python " + " ".join(sys.argv),
            "parameters": _args_to_dict(args),
            "provenance": {
                "pipeline_result_dir": parent_result_dir,
            },
            "timing": {
                "started_at": self._start_time.isoformat() if self._start_time else None,
            },
            "environment": _get_environment(),
            "status": "running",
        }

        try:
            with open(self.job_json, "w") as f:
                json.dump(job_data, f, indent=2)
        except (IOError, OSError) as e:
            logger.warning("Could not write job.json: %s", e)

    def _setup_logging(self):
        """Configure root logger to write to run.log and stderr."""
        logging.basicConfig(
            format=_LOG_FMT,
            level=logging.INFO,
            force=True,
            handlers=[
                RobustFileHandler(self.run_log),
                RobustStreamHandler(),
            ],
        )

"""Job management for the RECOVAR GUI.

Discovers existing pipeline outputs, tracks launched jobs, and interfaces
with SLURM for cluster job submission and monitoring.
"""

import json
import logging
import os
import pickle
import shutil
import subprocess
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _is_output_volume(filename: str) -> bool:
    """Return True if *filename* is a primary output MRC (not half/unfil/mask)."""
    return (
        filename.endswith(".mrc")
        and "_half" not in filename
        and "_unfil" not in filename
        and "_mask" not in filename
    )


def _list_volumes(directory: str) -> list[dict]:
    """List primary output MRC volumes in *directory* as dicts with path/name/display_name."""
    if not os.path.isdir(directory):
        return []
    return [
        {
            "path": os.path.join(directory, f),
            "name": f,
            "display_name": _vol_display_name(f),
        }
        for f in sorted(os.listdir(directory))
        if _is_output_volume(f)
    ]


def _list_images(directory: str) -> list[str]:
    """List image files (png/jpg/svg) in *directory* as absolute paths."""
    if not os.path.isdir(directory):
        return []
    return [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if f.endswith((".png", ".jpg", ".svg"))
    ]


def _load_json(path: str) -> Optional[dict]:
    """Load a JSON file, returning None on any error."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        logger.debug("Failed to load JSON %s: %s", path, e)
        return None


def _save_json(path: str, data: dict) -> bool:
    """Write *data* as JSON. Returns True on success."""
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except OSError as e:
        logger.warning("Failed to save JSON %s: %s", path, e)
        return False


def _has_output_volumes(directory: str) -> bool:
    """Return True if *directory* contains any primary output MRC files."""
    try:
        return any(
            _is_output_volume(f)
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
        )
    except OSError:
        return False


def _has_density_output(directory):
    """Return True if *directory* contains density estimation output."""
    return os.path.isfile(os.path.join(directory, "deconv_density_knee.pkl"))


def _has_stable_states_output(directory):
    """Return True if *directory* contains stable states output."""
    return os.path.isfile(os.path.join(directory, "stable_state_all_coords.txt"))


def _has_task_output(directory, task_type):
    """Return True if *directory* contains expected output for *task_type*."""
    if task_type == "density":
        return _has_density_output(directory)
    elif task_type == "stable_states":
        return _has_stable_states_output(directory)
    return _has_output_volumes(directory)


# ---------------------------------------------------------------------------
# Job model
# ---------------------------------------------------------------------------

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


@dataclass
class Job:
    id: str
    name: str
    output_dir: str
    status: str = STATUS_QUEUED
    created_at: float = 0.0
    slurm_job_id: Optional[str] = None
    pid: Optional[int] = None
    command: str = ""
    particles: str = ""
    mask: str = ""
    grid_size: Optional[int] = None
    downsample: Optional[int] = None
    n_images: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        # Filter to only known fields
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def created_str(self):
        if self.created_at:
            import datetime
            return datetime.datetime.fromtimestamp(self.created_at).strftime("%Y-%m-%d %H:%M")
        return ""

    @property
    def log_file(self):
        run_log = os.path.join(self.output_dir, "run.log")
        if os.path.isfile(run_log):
            return run_log
        # Try SLURM log pattern
        if self.slurm_job_id:
            for pattern in [
                os.path.join(self.output_dir, f"slurm-{self.slurm_job_id}.out"),
                os.path.join(os.path.dirname(self.output_dir), "logs", f"recovar-gui-{self.slurm_job_id}.out"),
            ]:
                if os.path.isfile(pattern):
                    return pattern
        return run_log  # may not exist yet

    @property
    def has_results(self):
        return (os.path.isfile(os.path.join(self.output_dir, "output", "mean.mrc")) or
                os.path.isfile(os.path.join(self.output_dir, "output", "volumes", "mean.mrc")) or
                os.path.isfile(os.path.join(self.output_dir, "model", "params.pkl")))

    @property
    def result_images(self):
        """List available result image files."""
        output_dir = os.path.join(self.output_dir, "output")
        images = _list_images(output_dir)
        if os.path.isdir(output_dir):
            for entry in sorted(os.listdir(output_dir)):
                sub = os.path.join(output_dir, entry)
                if os.path.isdir(sub) and (entry.startswith("analysis_") or entry == "plots"):
                    images.extend(_list_images(sub))
        # Also check top-level analysis dirs (analyze/, analysis_*/)
        if os.path.isdir(self.output_dir):
            for entry in sorted(os.listdir(self.output_dir)):
                if not (entry.startswith("analysis_") or entry == "analyze"):
                    continue
                analysis_dir = os.path.join(self.output_dir, entry)
                if not os.path.isdir(analysis_dir):
                    continue
                images.extend(_list_images(analysis_dir))
                for sub in ["umap", "PCA"]:
                    subdir = os.path.join(analysis_dir, sub)
                    if os.path.isdir(subdir):
                        images.extend(_list_images(subdir))
        return images

    @property
    def result_volumes(self):
        """List available MRC volume files from output directories."""
        vols = []
        seen = set()
        for subdir in ["output/volumes", "output"]:
            vol_dir = os.path.join(self.output_dir, subdir)
            if not os.path.isdir(vol_dir):
                continue
            for fname in sorted(os.listdir(vol_dir)):
                if fname.endswith(".mrc") and fname not in seen:
                    seen.add(fname)
                    vols.append(os.path.join(vol_dir, fname))
        return vols


# ---------------------------------------------------------------------------
# Compute task model (async volume/trajectory computations from GUI)
# ---------------------------------------------------------------------------

@dataclass
class ComputeTask:
    """Tracks an async volume or trajectory computation launched from the GUI."""
    id: str
    job_id: str
    task_type: str  # "volume" or "trajectory"
    status: str = STATUS_RUNNING
    output_dir: str = ""
    pid: Optional[int] = None
    slurm_job_id: Optional[str] = None
    created_at: float = 0.0
    error: Optional[str] = None
    label: str = ""

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Volume categorization
# ---------------------------------------------------------------------------

def _categorize_volume(filename):
    """Categorize a volume file by its filename pattern."""
    name = filename.lower()
    if name in ("mean.mrc", "mean_filt.mrc"):
        return "reconstruction"
    if "half" in name and "unfil" in name:
        return "halfmaps"
    if name.startswith("eigen_"):
        return "eigenvectors"
    if name.startswith("variance"):
        return "variance"
    if name in ("mask.mrc", "dilated_mask.mrc", "focus_mask.mrc", "complement_mask.mrc"):
        return "masks"
    if name.startswith("center"):
        return "kmeans"
    if name.startswith("state"):
        return "trajectory"
    return "other"


def _vol_display_name(filename):
    """Return a user-friendly display name for a volume file."""
    name = filename.replace(".mrc", "")
    names = {
        "mean": "Mean Volume",
        "mean_filt": "Filtered Mean",
        "mean_half1_unfil": "Half-map 1",
        "mean_half2_unfil": "Half-map 2",
        "mask": "Solvent Mask",
        "dilated_mask": "Dilated Mask",
        "focus_mask": "Focus Mask",
        "complement_mask": "Complement Mask",
    }
    if name in names:
        return names[name]
    if name.startswith("eigen_pos"):
        idx = name.replace("eigen_pos", "")
        return f"PC {int(idx)}"
    if name.startswith("eigen_neg"):
        idx = name.replace("eigen_neg", "")
        return f"PC {int(idx)} (neg, legacy)"
    if name.startswith("variance"):
        n = name.replace("variance", "")
        return f"Variance (top {n} PCs)"
    if name.startswith("center"):
        idx = name.split("_")[0].replace("center", "")
        if "half" in name:
            return name
        return f"K-means Center {int(idx)}"
    if name.startswith("state"):
        idx = name.split("_")[0].replace("state", "")
        if "half" in name:
            return name
        return f"State {int(idx)}"
    return name


# ---------------------------------------------------------------------------
# SLURM helpers
# ---------------------------------------------------------------------------

def _has_slurm():
    return shutil.which("sbatch") is not None


def _slurm_job_status(job_id: str) -> Optional[str]:
    """Query SLURM for job status. Returns status string or None."""
    try:
        result = subprocess.run(
            ["sacct", "-j", job_id, "--format=State", "--noheader", "-P"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            states = [s.strip() for s in result.stdout.strip().split("\n") if s.strip()]
            if states:
                state = states[0].upper()
                if "RUNNING" in state:
                    return STATUS_RUNNING
                if "COMPLETED" in state:
                    return STATUS_COMPLETED
                if any(kw in state for kw in ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_ME")):
                    return STATUS_FAILED
                if "PENDING" in state:
                    return STATUS_QUEUED
    except Exception:
        logger.debug("sacct query failed for job %s", job_id, exc_info=True)
    return None


def _slurm_submit(script_path: str) -> Optional[str]:
    """Submit a SLURM batch script. Returns job ID or None."""
    try:
        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            # Parse "Submitted batch job 12345"
            parts = result.stdout.strip().split()
            if len(parts) >= 4:
                return parts[-1]
    except Exception as e:
        logger.error("SLURM submit failed: %s", e)
    return None


# ---------------------------------------------------------------------------
# Job Manager
# ---------------------------------------------------------------------------

class JobManager:
    """Manages pipeline jobs: discovery, creation, monitoring."""

    def __init__(self, state_dir: Optional[str] = None):
        if state_dir is None:
            state_dir = os.path.expanduser("~/.recovar/gui")
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        self.jobs_file = os.path.join(state_dir, "jobs.json")
        self._jobs: dict[str, Job] = {}
        self._compute_tasks: dict[str, ComputeTask] = {}
        self._load()

    def _load(self):
        if os.path.isfile(self.jobs_file):
            try:
                with open(self.jobs_file) as f:
                    data = json.load(f)
                for jd in data.get("jobs", []):
                    job = Job.from_dict(jd)
                    self._jobs[job.id] = job
            except Exception as e:
                logger.warning("Failed to load jobs file: %s", e)

    def _save(self):
        _save_json(self.jobs_file, {"jobs": [j.to_dict() for j in self._jobs.values()]})

    def discover_jobs(self, scan_dirs: list[str]):
        """Scan directories for existing pipeline outputs.

        Recognises a pipeline output directory by the presence of any of:
        - ``metadata.json`` (top-level)
        - ``params.pkl`` (top-level, legacy)
        - ``model/params.pkl`` (current layout)
        """
        for scan_dir in scan_dirs:
            if not os.path.isdir(scan_dir):
                continue
            for entry in os.listdir(scan_dir):
                job_dir = os.path.join(scan_dir, entry)
                if not os.path.isdir(job_dir):
                    continue
                meta_file = os.path.join(job_dir, "metadata.json")
                # Check multiple possible locations for params.pkl
                is_pipeline_output = (
                    os.path.isfile(meta_file) or
                    os.path.isfile(os.path.join(job_dir, "params.pkl")) or
                    os.path.isfile(os.path.join(job_dir, "model", "params.pkl"))
                )
                if not is_pipeline_output:
                    continue
                job_id = f"discovered_{entry}"
                if job_id in self._jobs:
                    continue
                job = Job(
                    id=job_id,
                    name=entry,
                    output_dir=job_dir,
                    status=STATUS_COMPLETED,
                    created_at=os.path.getmtime(job_dir),
                )
                # Try to read metadata
                if os.path.isfile(meta_file):
                    try:
                        with open(meta_file) as f:
                            meta = json.load(f)
                        job.particles = meta.get("particles_file", "")
                        job.grid_size = meta.get("grid_size")
                        job.n_images = meta.get("n_images")
                        job.downsample = meta.get("downsample_applied")
                    except Exception:
                        pass
                # Try to get particle info from command.txt if no metadata
                if not job.particles:
                    cmd_file = os.path.join(job_dir, "command.txt")
                    if os.path.isfile(cmd_file):
                        try:
                            with open(cmd_file) as f:
                                job.command = f.read().strip()
                        except Exception:
                            pass
                self._jobs[job_id] = job
        self._save()
        self._recover_compute_tasks()

    def _recover_compute_tasks(self):
        """Re-discover compute tasks from gui_computed/ dirs on disk.

        This recovers tasks from previous GUI sessions so they appear in the
        task list even after a restart.
        """
        for job in self._jobs.values():
            computed_dir = os.path.join(job.output_dir, "gui_computed")
            if not os.path.isdir(computed_dir):
                continue
            for tdir_name in os.listdir(computed_dir):
                if tdir_name in self._compute_tasks:
                    continue  # already tracked in memory
                tdir = os.path.join(computed_dir, tdir_name)
                if not os.path.isdir(tdir):
                    continue
                # Determine task type from the directory name prefix
                if tdir_name.startswith("volume_"):
                    task_type = "volume"
                elif tdir_name.startswith("trajectory_"):
                    task_type = "trajectory"
                elif tdir_name.startswith("density_"):
                    task_type = "density"
                elif tdir_name.startswith("stable_states_"):
                    task_type = "stable_states"
                else:
                    continue
                # Must have a compute.sbatch or task_meta.json to be a real task
                # (e.g., "trajectory_*density" dirs are output subdirs, not tasks)
                if not (os.path.isfile(os.path.join(tdir, "compute.sbatch")) or
                        os.path.isfile(os.path.join(tdir, "task_meta.json"))):
                    continue
                has_output = _has_task_output(tdir, task_type)
                meta = _load_json(os.path.join(tdir, "task_meta.json")) or {}

                slurm_job_id = meta.get("slurm_job_id")
                label = meta.get("label", task_type)

                # Determine status
                # If we have a SLURM job ID (from task_meta.json), check it first
                # because the job might still be producing more output files
                if slurm_job_id:
                    slurm_status = _slurm_job_status(slurm_job_id)
                    if slurm_status in (STATUS_RUNNING, STATUS_QUEUED):
                        status = slurm_status
                    elif has_output:
                        status = STATUS_COMPLETED
                    else:
                        status = STATUS_FAILED
                elif has_output:
                    # Check if compute.log was modified recently (within 10 min)
                    # which would indicate the task is still running.
                    # GPU compute can go minutes between log writes.
                    log_path = os.path.join(tdir, "compute.log")
                    if os.path.isfile(log_path):
                        try:
                            log_age = time.time() - os.path.getmtime(log_path)
                            if log_age < 600:
                                status = STATUS_RUNNING
                            else:
                                status = STATUS_COMPLETED
                        except OSError:
                            status = STATUS_COMPLETED
                    else:
                        status = STATUS_COMPLETED
                else:
                    status = STATUS_FAILED

                # Fall back to generating label from files if no metadata
                if label == task_type:
                    if task_type == "volume":
                        lp = os.path.join(tdir, "latent_points.txt")
                        if os.path.isfile(lp):
                            try:

                                pts = np.loadtxt(lp)
                                coords = pts.flatten()[:3]
                                label = f"Volume at [{', '.join(f'{c:.2f}' for c in coords)}{'...' if len(pts.flatten()) > 3 else ''}]"
                            except Exception:
                                pass
                    elif task_type == "trajectory":
                        label = "Trajectory A\u2192B"
                    elif task_type == "density":
                        label = "Conformational Density"
                    elif task_type == "stable_states":
                        label = "Stable States"

                try:
                    created_at = meta.get("created_at") or os.path.getmtime(tdir)
                except OSError:
                    created_at = 0.0

                task = ComputeTask(
                    id=tdir_name,
                    job_id=meta.get("job_id", job.id),
                    task_type=task_type,
                    status=status,
                    output_dir=tdir,
                    slurm_job_id=slurm_job_id,
                    created_at=created_at,
                    label=label,
                )
                self._compute_tasks[tdir_name] = task
                logger.info("Recovered compute task: %s (%s)", tdir_name, status)

    def list_jobs(self) -> list[Job]:
        """Return all jobs sorted by creation time (newest first)."""
        self._refresh_statuses()
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def get_job(self, job_id: str) -> Optional[Job]:
        job = self._jobs.get(job_id)
        if job:
            self._refresh_job_status(job)
        return job

    def create_job(self, name: str, output_dir: str, command: str,
                   particles: str = "", mask: str = "",
                   downsample: Optional[int] = None,
                   use_slurm: bool = False,
                   slurm_partition: str = "cryoem",
                   slurm_account: str = "amits",
                   slurm_gpus: int = 1,
                   slurm_mem: str = "64G",
                   slurm_time: str = "4:00:00",
                   slurm_cpus: str = "8",
                   slurm_extra: str = "",
                   python_path: str = "python3") -> Job:
        """Create and launch a new pipeline job."""
        job_id = f"job_{int(time.time())}_{name.replace(' ', '_')}"
        job = Job(
            id=job_id,
            name=name,
            output_dir=output_dir,
            status=STATUS_QUEUED,
            created_at=time.time(),
            command=command,
            particles=particles,
            mask=mask,
            downsample=downsample,
        )

        os.makedirs(output_dir, exist_ok=True)

        if use_slurm and _has_slurm():
            # Write SLURM batch script
            script = os.path.join(self.state_dir, f"{job_id}.sbatch")
            log_dir = os.path.join(output_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)

            extra_sbatch = ""
            if slurm_extra:
                extra_sbatch = "\n".join(f"#SBATCH {flag.strip()}"
                                         for flag in slurm_extra.split() if flag.strip())
                extra_sbatch = "\n" + extra_sbatch

            gpu_line = f"\n#SBATCH --gres=gpu:{slurm_gpus}" if slurm_gpus > 0 else ""

            with open(script, "w") as f:
                f.write(f"""#!/bin/bash
#SBATCH --job-name=recovar-{name[:20]}
#SBATCH --output={log_dir}/slurm-%j.out
#SBATCH --error={log_dir}/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={slurm_cpus}
#SBATCH --mem={slurm_mem}
#SBATCH --time={slurm_time}
#SBATCH --partition={slurm_partition}{gpu_line}
#SBATCH --account={slurm_account}{extra_sbatch}

set -euo pipefail
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "Node: $(hostname)"
echo "Started at: $(date)"

{python_path} -m {command}

echo "Completed at: $(date)"
""")
            slurm_id = _slurm_submit(script)
            if slurm_id:
                job.slurm_job_id = slurm_id
                job.status = STATUS_QUEUED
                logger.info("Submitted SLURM job %s for %s", slurm_id, name)
            else:
                job.status = STATUS_FAILED
                job.error = "Failed to submit SLURM job"
        else:
            # Run directly as subprocess
            try:
                log_file = os.path.join(output_dir, "gui_run.log")
                with open(log_file, "w") as lf:
                    proc = subprocess.Popen(
                        [python_path, "-m"] + command.split(),
                        stdout=lf, stderr=subprocess.STDOUT,
                        env={**os.environ,
                             "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
                    )
                job.pid = proc.pid
                job.status = STATUS_RUNNING
                logger.info("Started local process PID %d for %s", proc.pid, name)
            except Exception as e:
                job.status = STATUS_FAILED
                job.error = str(e)

        self._jobs[job_id] = job
        self._save()
        return job

    def cancel_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        if job.slurm_job_id:
            try:
                subprocess.run(["scancel", job.slurm_job_id], timeout=10)
                job.status = STATUS_FAILED
                job.error = "Cancelled by user"
                self._save()
                return True
            except Exception:
                logger.warning("Failed to cancel SLURM job %s", job.slurm_job_id, exc_info=True)
        if job.pid:
            try:
                os.kill(job.pid, 15)  # SIGTERM
                job.status = STATUS_FAILED
                job.error = "Cancelled by user"
                self._save()
                return True
            except OSError:
                logger.warning("Failed to kill process %d", job.pid, exc_info=True)
        return False

    def delete_job(self, job_id: str) -> bool:
        """Remove job from tracking (does not delete output files)."""
        if job_id in self._jobs:
            del self._jobs[job_id]
            self._save()
            return True
        return False

    def get_log_content(self, job_id: str, n_lines: int = 200) -> str:
        job = self._jobs.get(job_id)
        if not job:
            return ""

        # Collect log content from all available log files
        # SLURM logs go LAST so they appear at the tail end (most relevant)
        log_sources = []
        slurm_logs = []

        # 1. Find SLURM logs (added to combined output last)
        if job.slurm_job_id:
            log_dir = os.path.join(job.output_dir, "logs")
            if os.path.isdir(log_dir):
                out_files = sorted(
                    [os.path.join(log_dir, f) for f in os.listdir(log_dir)
                     if f.endswith(".out")],
                    key=os.path.getmtime, reverse=True,
                )
                err_files = sorted(
                    [os.path.join(log_dir, f) for f in os.listdir(log_dir)
                     if f.endswith(".err") and os.path.getsize(os.path.join(log_dir, f)) > 0],
                    key=os.path.getmtime, reverse=True,
                )
                if out_files:
                    slurm_logs.append(out_files[0])
                if err_files:
                    slurm_logs.append(err_files[0])

        # 2. Standard log files (shown first, may be from older runs)
        for candidate in [
            os.path.join(job.output_dir, "run.log"),
            os.path.join(job.output_dir, "gui_run.log"),
        ]:
            if os.path.isfile(candidate) and candidate not in slurm_logs:
                log_sources.append(candidate)

        # 3. Add SLURM logs last (so errors appear at the end)
        log_sources.extend(slurm_logs)

        if not log_sources:
            return "No log file found yet."

        # Combine all sources
        combined = []
        for path in log_sources:
            try:
                with open(path) as f:
                    content = f.read()
                if content.strip():
                    label = os.path.basename(path)
                    combined.append(f"── {label} ──\n{content}")
            except OSError as e:
                logger.debug("Failed to read log %s: %s", label, e)

        if not combined:
            return "Log files are empty."

        full_text = "\n".join(combined)
        lines = full_text.split("\n")
        return "\n".join(lines[-n_lines:])

    def get_error_summary(self, job_id: str) -> str:
        """Extract a short error summary from logs (last traceback or error line)."""
        job = self._jobs.get(job_id)
        if not job:
            return ""
        if job.error and job.error != "Process exited":
            return job.error

        log_content = self.get_log_content(job_id, n_lines=500)
        if not log_content:
            return ""

        lines = log_content.split("\n")
        # Find last traceback
        last_tb_start = -1
        for i, line in enumerate(lines):
            if line.strip().startswith("Traceback"):
                last_tb_start = i
        if last_tb_start >= 0:
            # Extract just the error type and message (last line of traceback)
            for i in range(len(lines) - 1, last_tb_start, -1):
                line = lines[i].strip()
                if line and not line.startswith("File ") and not line.startswith("^"):
                    return line[:200]

        # Look for lines with ERROR or Error
        for line in reversed(lines):
            stripped = line.strip()
            if "Error:" in stripped or "ERROR" in stripped:
                return stripped[:200]

        return ""

    def get_job_params(self, job_id: str) -> dict:
        """Read job parameters from metadata.json or params.pkl."""
        job = self._jobs.get(job_id)
        if not job:
            return {}
        meta = _load_json(os.path.join(job.output_dir, "metadata.json"))
        return meta if meta is not None else {"command": job.command, "output_dir": job.output_dir}

    def _refresh_statuses(self):
        for job in self._jobs.values():
            if job.status in (STATUS_QUEUED, STATUS_RUNNING):
                self._refresh_job_status(job)
        self._save()

    def _refresh_job_status(self, job: Job):
        if job.status not in (STATUS_QUEUED, STATUS_RUNNING):
            return
        if job.slurm_job_id:
            new_status = _slurm_job_status(job.slurm_job_id)
            if new_status:
                old_status = job.status
                job.status = new_status
                # Extract error summary when job transitions to failed
                if new_status == STATUS_FAILED and old_status != STATUS_FAILED:
                    err = self.get_error_summary(job.id)
                    if err:
                        job.error = err
        elif job.pid:
            try:
                os.kill(job.pid, 0)  # Check if process is alive
            except OSError:
                # Process no longer running
                if job.has_results:
                    job.status = STATUS_COMPLETED
                else:
                    job.status = STATUS_FAILED
                    if not job.error:
                        err = self.get_error_summary(job.id)
                        job.error = err or "Process exited"


    # ── Analysis discovery ──────────────────────────────────────────

    def get_analysis_info(self, job_id: str) -> dict:
        """Discover available analysis results, volumes, and embeddings for a job."""
        job = self._jobs.get(job_id)
        if not job:
            return {"error": "Job not found"}

        info = {
            "has_model": False,
            "has_embeddings": False,
            "available_zdims": [],
            "volumes": [],
            "analyses": {},
            "computed": [],
        }

        model_dir = os.path.join(job.output_dir, "model")
        if os.path.isfile(os.path.join(model_dir, "params.pkl")):
            info["has_model"] = True

        # Check for embeddings and available zdims
        embeddings_path = os.path.join(model_dir, "embeddings.pkl")
        if os.path.isfile(embeddings_path):
            info["has_embeddings"] = True
            try:

                with open(embeddings_path, "rb") as f:
                    emb = pickle.load(f)
                for key in ["latent_coords", "zs"]:
                    if key in emb and isinstance(emb[key], dict):
                        zdims = []
                        for k in emb[key].keys():
                            try:
                                zdims.append(int(k))
                            except (ValueError, TypeError):
                                # Keys like '1_noreg' — extract numeric prefix
                                if isinstance(k, str) and "_" in k:
                                    try:
                                        zdims.append(int(k.split("_")[0]))
                                    except ValueError:
                                        pass
                        info["available_zdims"] = sorted(set(zdims))
                        break
            except Exception as e:
                logger.warning("Failed to read embeddings: %s", e)

        # Scan main volumes directory
        for subdir in ["output/volumes", "output"]:
            vol_dir = os.path.join(job.output_dir, subdir)
            if not os.path.isdir(vol_dir):
                continue
            seen_names = {v["name"] for v in info["volumes"]}
            for fname in sorted(os.listdir(vol_dir)):
                if fname.endswith(".mrc") and fname not in seen_names:
                    info["volumes"].append({
                        "path": os.path.join(vol_dir, fname),
                        "name": fname,
                        "display_name": _vol_display_name(fname),
                        "category": _categorize_volume(fname),
                    })

        # Scan analysis directories — check both top-level and output/
        # Recognises both "analysis_<zdim>" (standard) and bare "analyze" dirs
        analysis_search_dirs = [job.output_dir, os.path.join(job.output_dir, "output")]
        seen_analysis_names = set()
        for search_dir in analysis_search_dirs:
            if not os.path.isdir(search_dir):
                continue
            for entry in sorted(os.listdir(search_dir)):
                if not (entry.startswith("analysis_") or entry == "analyze"):
                    continue
                if entry in seen_analysis_names:
                    continue

                analysis_dir = os.path.join(search_dir, entry)
                if not os.path.isdir(analysis_dir):
                    continue

                # Verify it looks like an analysis dir (has kmeans or umap)
                has_markers = (
                    os.path.isfile(os.path.join(analysis_dir, "kmeans_result.pkl")) or
                    os.path.isdir(os.path.join(analysis_dir, "umap")) or
                    os.path.isdir(os.path.join(analysis_dir, "kmeans"))
                )
                if not has_markers:
                    continue

                seen_analysis_names.add(entry)

                # Parse zdim from name like "analysis_20" or "analysis_20_noreg"
                zdim_parsed = None
                if entry.startswith("analysis_"):
                    parts = entry.split("_")
                    for part in parts[1:]:
                        try:
                            zdim_parsed = int(part)
                            break
                        except ValueError:
                            continue

                # For bare "analyze" dirs, use the largest available zdim
                if zdim_parsed is None:
                    if info["available_zdims"]:
                        zdim_parsed = max(info["available_zdims"])
                    else:
                        zdim_parsed = 20  # reasonable default

                analysis = {
                    "name": entry,
                    "plots": [],
                    "kmeans_volumes": [],
                    "trajectories": [],
                }

                # K-means volumes in centers/ or kmeans/ subdirs
                for kmeans_name in ["centers", "kmeans"]:
                    analysis["kmeans_volumes"].extend(
                        _list_volumes(os.path.join(analysis_dir, kmeans_name))
                    )

                # Trajectories: traj* or path* subdirs
                for tentry in sorted(os.listdir(analysis_dir)):
                    if not (tentry.startswith("traj") or tentry.startswith("path")):
                        continue
                    traj_dir = os.path.join(analysis_dir, tentry)
                    if not os.path.isdir(traj_dir):
                        continue
                    traj_vols = _list_volumes(traj_dir) + _list_volumes(
                        os.path.join(traj_dir, "density")
                    )
                    traj_plots = _list_images(traj_dir)
                    if traj_vols or traj_plots:
                        analysis["trajectories"].append({
                            "name": tentry,
                            "volumes": traj_vols,
                            "plots": traj_plots,
                        })

                # Plots and images
                analysis["plots"] = _list_images(analysis_dir)
                for plot_subdir in ["umap", "PCA"]:
                    analysis["plots"].extend(_list_images(
                        os.path.join(analysis_dir, plot_subdir)
                    ))

                # Check for UMAP and k-means availability
                umap_embedding = os.path.join(analysis_dir, "umap", "umap_embedding.pkl")
                analysis["has_umap"] = os.path.isfile(umap_embedding)

                kmeans_result = os.path.join(analysis_dir, "kmeans_result.pkl")
                analysis["has_kmeans"] = os.path.isfile(kmeans_result)

                # Use zdim as key, but append suffix for multiple analyses at same zdim
                key = str(zdim_parsed)
                if key in info["analyses"]:
                    key = f"{zdim_parsed}_{entry}"
                info["analyses"][key] = analysis

        # Scan GUI-computed volumes
        computed_dir = os.path.join(job.output_dir, "gui_computed")
        if os.path.isdir(computed_dir):
            for tdir in sorted(os.listdir(computed_dir), reverse=True):
                task_dir = os.path.join(computed_dir, tdir)
                if not os.path.isdir(task_dir):
                    continue
                for vol in _list_volumes(task_dir):
                    vol["name"] = f"{tdir}/{vol['name']}"
                    info["computed"].append(vol)

        # Discover density estimation results
        # Check CLI-generated output in density/ subdir
        density_info = None
        cli_density_pkl = os.path.join(job.output_dir, "density", "deconv_density_knee.pkl")
        if os.path.isfile(cli_density_pkl):
            density_info = {
                "density_pkl": cli_density_pkl,
                "plots": _list_images(os.path.join(job.output_dir, "density")),
            }
        # Check GUI-generated output in gui_computed/density_*/
        if os.path.isdir(computed_dir):
            for tdir in sorted(os.listdir(computed_dir), reverse=True):
                if not tdir.startswith("density_"):
                    continue
                task_dir = os.path.join(computed_dir, tdir)
                knee_pkl = os.path.join(task_dir, "deconv_density_knee.pkl")
                if os.path.isfile(knee_pkl):
                    density_info = {
                        "density_pkl": knee_pkl,
                        "plots": _list_images(task_dir),
                    }
                    # Try to read pca_dim from task metadata
                    meta_path = os.path.join(task_dir, "task_meta.json")
                    if os.path.isfile(meta_path):
                        try:
                            meta = _load_json(meta_path)
                            pd = meta.get("params", {}).get("pca_dim")
                            if pd:
                                density_info["pca_dim"] = pd
                        except Exception:
                            pass
                    break  # use most recent
        if density_info:
            info["density"] = density_info

        # Discover stable states results from gui_computed/stable_states_*/
        if os.path.isdir(computed_dir):
            for tdir in sorted(os.listdir(computed_dir), reverse=True):
                if not tdir.startswith("stable_states_"):
                    continue
                task_dir = os.path.join(computed_dir, tdir)
                coords_file = os.path.join(task_dir, "stable_state_all_coords.txt")
                if os.path.isfile(coords_file):
                    try:
                        coords = np.loadtxt(coords_file)
                        if coords.ndim == 1:
                            coords = coords.reshape(1, -1)
                        info["stable_states"] = {
                            "coords": coords.tolist(),
                            "plots": (_list_images(task_dir) +
                                      _list_images(os.path.join(task_dir, "density"))),
                        }
                    except Exception:
                        pass
                    break  # use most recent

        return info

    def get_embedding_data(self, job_id: str, zdim: int,
                           max_points: int = 15000) -> Optional[dict]:
        """Load embedding coordinates for scatter plot visualization."""

        job = self._jobs.get(job_id)
        if not job:
            return None

        embeddings_path = os.path.join(job.output_dir, "model", "embeddings.pkl")
        if not os.path.isfile(embeddings_path):
            return None

        try:

            with open(embeddings_path, "rb") as f:
                emb = pickle.load(f)

            # Support both old and new embedding formats
            coords_dict = None
            for key in ["latent_coords", "zs"]:
                if key in emb and isinstance(emb[key], dict):
                    coords_dict = emb[key]
                    break
            if coords_dict is None:
                return None

            # Try exact key first, then integer, then string variants
            zs_arr = None
            for candidate in [zdim, str(zdim)]:
                if candidate in coords_dict:
                    zs_arr = coords_dict[candidate]
                    break
            if zs_arr is None:
                return None

            zs = np.asarray(zs_arr, dtype=np.float32)
            n_total = zs.shape[0]
            actual_zdim = zs.shape[1] if zs.ndim > 1 else 1

            # Subsample for performance
            if n_total > max_points:
                rng = np.random.RandomState(42)
                indices = rng.choice(n_total, max_points, replace=False)
                indices.sort()
                zs = zs[indices]

            result = {
                "n_total": n_total,
                "n_displayed": len(zs),
                "zdim": actual_zdim,
            }

            # Return each dimension as a list
            if zs.ndim == 1:
                result["dim0"] = zs.tolist()
            else:
                for i in range(min(actual_zdim, 10)):
                    result[f"dim{i}"] = zs[:, i].tolist()

            return result
        except Exception as e:
            logger.error("Failed to load embeddings: %s", e)
            return None

    def _find_analysis_dir(self, job: Job, zdim: int) -> Optional[str]:
        """Find the analysis directory for a given zdim.

        Checks for both ``analysis_<zdim>`` directories and bare ``analyze/``
        directories (which contain results for the largest available zdim).
        """
        for search_dir in [job.output_dir, os.path.join(job.output_dir, "output")]:
            if not os.path.isdir(search_dir):
                continue
            # Prefer exact analysis_<zdim> match
            for entry in sorted(os.listdir(search_dir)):
                if entry.startswith(f"analysis_{zdim}"):
                    path = os.path.join(search_dir, entry)
                    if os.path.isdir(path):
                        return path
        # Fall back to bare "analyze" dir (typically the largest zdim)
        for search_dir in [job.output_dir, os.path.join(job.output_dir, "output")]:
            analyze_dir = os.path.join(search_dir, "analyze")
            if os.path.isdir(analyze_dir) and (
                os.path.isfile(os.path.join(analyze_dir, "kmeans_result.pkl")) or
                os.path.isdir(os.path.join(analyze_dir, "umap"))
            ):
                return analyze_dir
        return None

    def get_umap_data(self, job_id: str, zdim: int,
                      max_points: int = 15000) -> Optional[dict]:
        """Load UMAP coordinates for scatter plot visualization."""

        job = self._jobs.get(job_id)
        if not job:
            return None

        analysis_dir = self._find_analysis_dir(job, zdim)
        if not analysis_dir:
            return None

        umap_path = os.path.join(analysis_dir, "umap", "umap_embedding.pkl")
        if not os.path.isfile(umap_path):
            return None

        try:

            with open(umap_path, "rb") as f:
                umap_raw = pickle.load(f)

            umap_emb = np.asarray(umap_raw, dtype=np.float32)
            if umap_emb.ndim == 1:
                return None
            n_total = umap_emb.shape[0]

            if n_total > max_points:
                rng = np.random.RandomState(42)
                indices = rng.choice(n_total, max_points, replace=False)
                indices.sort()
                umap_emb = umap_emb[indices]

            return {
                "n_total": n_total,
                "n_displayed": len(umap_emb),
                "umap0": umap_emb[:, 0].tolist(),
                "umap1": umap_emb[:, 1].tolist(),
            }
        except Exception as e:
            logger.error("Failed to load UMAP data: %s", e)
            return None

    def get_kmeans_data(self, job_id: str, zdim: int,
                        max_points: int = 15000) -> Optional[dict]:
        """Load k-means cluster labels for coloring the scatter plot."""

        job = self._jobs.get(job_id)
        if not job:
            return None

        analysis_dir = self._find_analysis_dir(job, zdim)
        if not analysis_dir:
            return None

        kmeans_path = os.path.join(analysis_dir, "kmeans_result.pkl")
        if not os.path.isfile(kmeans_path):
            return None

        try:

            with open(kmeans_path, "rb") as f:
                kmeans_raw = pickle.load(f)

            # kmeans_result.pkl may be a dict with 'labels' or just an array
            if isinstance(kmeans_raw, dict):
                labels = np.asarray(kmeans_raw.get("labels", kmeans_raw.get("cluster_labels")), dtype=np.int32)
            else:
                labels = np.asarray(kmeans_raw, dtype=np.int32)

            n_total = len(labels)
            n_clusters = int(labels.max()) + 1

            # Apply same subsampling as embeddings for consistency
            if n_total > max_points:
                rng = np.random.RandomState(42)
                indices = rng.choice(n_total, max_points, replace=False)
                indices.sort()
                labels = labels[indices]

            return {
                "labels": labels.tolist(),
                "n_clusters": n_clusters,
                "n_total": n_total,
                "n_displayed": len(labels),
            }
        except Exception as e:
            logger.error("Failed to load k-means data: %s", e)
            return None

    # ── Compute task management ──────────────────────────────────

    def submit_compute_task(self, job_id: str, task_type: str,
                            params: dict, python_path: str,
                            use_slurm: bool = False,
                            slurm_opts: Optional[dict] = None,
                            repo_root: Optional[str] = None) -> Optional[ComputeTask]:
        """Submit an async compute task (volume or trajectory)."""

        job = self._jobs.get(job_id)
        if not job:
            return None

        task_id = f"{task_type}_{int(time.time())}"
        output_dir = os.path.join(job.output_dir, "gui_computed", task_id)
        os.makedirs(output_dir, exist_ok=True)

        if task_type == "volume":
            coords = np.array(params["coords"], dtype=np.float32)
            if coords.ndim == 1:
                coords = coords.reshape(1, -1)
            np.savetxt(os.path.join(output_dir, "latent_points.txt"), coords)

            cmd = [
                python_path, "-m", "recovar.commands.compute_state",
                job.output_dir,
                "--latent-points", os.path.join(output_dir, "latent_points.txt"),
                "--outdir", output_dir,
                "--lazy",
            ]
        elif task_type == "trajectory":
            z_st = np.array(params["z_start"], dtype=np.float32)
            z_end = np.array(params["z_end"], dtype=np.float32)
            endpts = np.vstack([z_st, z_end])
            np.savetxt(os.path.join(output_dir, "endpoints.txt"), endpts)

            zdim = params.get("zdim", len(z_st))
            n_vols = params.get("n_vols", 6)
            cmd = [
                python_path, "-m", "recovar.commands.compute_trajectory",
                job.output_dir,
                "--outdir", output_dir,
                "--zdim", str(zdim),
                "--n-vols-along-path", str(n_vols),
                "--endpts", os.path.join(output_dir, "endpoints.txt"),
                "--lazy",
            ]
        elif task_type == "density":
            pca_dim = params.get("pca_dim", 4)
            cmd = [
                python_path, "-m",
                "recovar.commands.estimate_conformational_density",
                job.output_dir,
                "--output_dir", output_dir,
                "--pca_dim", str(pca_dim),
            ]
            z_dim_used = params.get("z_dim_used")
            if z_dim_used:
                cmd.extend(["--z_dim_used", str(z_dim_used)])
        elif task_type == "stable_states":
            density_pkl = params.get("density_pkl", "")
            percent_top = params.get("percent_top", 1)
            n_local_maxs = params.get("n_local_maxs", 3)
            cmd = [
                python_path, "-m",
                "recovar.commands.estimate_stable_states",
                density_pkl,
                "-o", output_dir,
                "--percent_top", str(percent_top),
                "--n_local_maxs", str(n_local_maxs),
            ]
        else:
            return None

        task = ComputeTask(
            id=task_id, job_id=job_id, task_type=task_type,
            output_dir=output_dir, created_at=time.time(),
            label=params.get("label", task_type),
        )
        # Stash selected params for metadata persistence
        if task_type == "density":
            task._params = {"pca_dim": params.get("pca_dim", 4)}

        env = {**os.environ,
               "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
        if repo_root:
            env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

        if use_slurm and _has_slurm():
            opts = slurm_opts or {}
            needs_gpu = task_type not in ("density", "stable_states")
            # Task-specific SLURM defaults: density needs lots of memory & time
            _slurm_defaults = {
                "density":       {"mem": "128G", "time": "2:00:00"},
                "stable_states": {"mem": "64G",  "time": "1:00:00"},
            }
            _td = _slurm_defaults.get(task_type, {"mem": "32G", "time": "1:00:00"})
            script_path = os.path.join(output_dir, "compute.sbatch")
            with open(script_path, "w") as f:
                f.write(f"""#!/bin/bash
#SBATCH --job-name=recovar-compute
#SBATCH --output={output_dir}/compute.log
#SBATCH --error={output_dir}/compute.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem={opts.get('mem', _td['mem'])}
#SBATCH --time={opts.get('time', _td['time'])}
#SBATCH --partition={opts.get('partition', 'cryoem')}
{"#SBATCH --gres=gpu:1" if needs_gpu else "# No GPU needed for " + task_type}
#SBATCH --account={opts.get('account', 'amits')}

set -euo pipefail
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
{f'export PYTHONPATH="{repo_root}:${{PYTHONPATH:-}}"' if repo_root else ''}

echo "PYTHONPATH=${{PYTHONPATH:-}}"
echo "which python: {python_path}"
{python_path} -c "import recovar; print('recovar from:', recovar.__file__); from recovar.cuda_backproject import cuda_available; print('CUDA available:', cuda_available())" || echo "Pre-check failed"

{' '.join(cmd)}
""")
            slurm_id = _slurm_submit(script_path)
            if slurm_id:
                task.slurm_job_id = slurm_id
                task.status = STATUS_QUEUED
                self._save_task_meta(task)  # persist SLURM job ID
            else:
                task.status = STATUS_FAILED
                task.error = "Failed to submit SLURM job"
        else:
            try:
                log_file = os.path.join(output_dir, "compute.log")
                with open(log_file, "w") as lf:
                    proc = subprocess.Popen(
                        cmd, stdout=lf, stderr=subprocess.STDOUT, env=env,
                    )
                task.pid = proc.pid
            except Exception as e:
                task.status = STATUS_FAILED
                task.error = str(e)

        self._compute_tasks[task_id] = task
        # Persist task metadata so it can be recovered after GUI restart
        self._save_task_meta(task)
        return task

    def _save_task_meta(self, task: ComputeTask):
        """Write task metadata to disk for recovery after restart."""
        meta = {
            "id": task.id, "job_id": task.job_id,
            "task_type": task.task_type, "label": task.label,
            "slurm_job_id": task.slurm_job_id,
            "created_at": task.created_at,
        }
        if hasattr(task, '_params'):
            meta["params"] = task._params
        _save_json(os.path.join(task.output_dir, "task_meta.json"), meta)

    def get_compute_task(self, task_id: str) -> Optional[dict]:
        """Get status of a compute task, including output volumes if done."""
        task = self._compute_tasks.get(task_id)
        if not task:
            return None

        # Re-check failed tasks: if output appeared since recovery, promote to completed
        if task.status == STATUS_FAILED and _has_task_output(task.output_dir, task.task_type):
            task.status = STATUS_COMPLETED
            task.error = None

        # Refresh status
        if task.status in (STATUS_QUEUED, STATUS_RUNNING):
            if task.slurm_job_id:
                new_status = _slurm_job_status(task.slurm_job_id)
                if new_status:
                    old = task.status
                    task.status = new_status
                    if new_status == STATUS_FAILED and old != STATUS_FAILED:
                        # Extract error from compute log
                        log_path = os.path.join(task.output_dir, "compute.log")
                        if os.path.isfile(log_path):
                            try:
                                with open(log_path) as lf:
                                    lines = lf.read().strip().splitlines()
                                # Last non-empty line is usually the error
                                task.error = lines[-1][:200] if lines else "Unknown error"
                            except Exception:
                                task.error = "Job failed (could not read log)"
                        else:
                            task.error = "Job failed"
            elif task.pid:
                try:
                    os.kill(task.pid, 0)
                except OSError:
                    # Process exited - check for output
                    if _has_task_output(task.output_dir, task.task_type):
                        task.status = STATUS_COMPLETED
                    else:
                        task.status = STATUS_FAILED
                        task.error = "No output generated"

        log_path = os.path.join(task.output_dir, "compute.log")
        result = {
            "id": task.id,
            "type": task.task_type,
            "status": task.status,
            "error": task.error,
            "label": task.label,
            "volumes": [],
            "plots": [],
            "log_path": log_path if os.path.isfile(log_path) else None,
        }

        if task.status == STATUS_COMPLETED:
            if task.task_type == "density":
                knee_pkl = os.path.join(task.output_dir, "deconv_density_knee.pkl")
                if os.path.isfile(knee_pkl):
                    result["density_pkl"] = knee_pkl
                # Include pca_dim from stored params or task_meta
                pca_dim = getattr(task, '_params', {}).get("pca_dim")
                if not pca_dim:
                    meta_path = os.path.join(task.output_dir, "task_meta.json")
                    if os.path.isfile(meta_path):
                        try:
                            meta = _load_json(meta_path)
                            pca_dim = meta.get("params", {}).get("pca_dim")
                        except Exception:
                            pass
                if pca_dim:
                    result["pca_dim"] = pca_dim
                result["plots"] = _list_images(task.output_dir)
            elif task.task_type == "stable_states":
                coords_file = os.path.join(task.output_dir, "stable_state_all_coords.txt")
                if os.path.isfile(coords_file):
                    try:
                        coords = np.loadtxt(coords_file)
                        if coords.ndim == 1:
                            coords = coords.reshape(1, -1)
                        result["stable_states"] = coords.tolist()
                    except Exception:
                        result["stable_states"] = []
                result["plots"] = (_list_images(task.output_dir) +
                                   _list_images(os.path.join(task.output_dir, "density")))
            else:
                result["volumes"] = _list_volumes(task.output_dir)

        return result

    def list_compute_tasks(self, job_id: str) -> list[dict]:
        """List all compute tasks for a job."""
        tasks = []
        for task in self._compute_tasks.values():
            if task.job_id == job_id:
                tasks.append(self.get_compute_task(task.id))
        return [t for t in tasks if t is not None]


# ---------------------------------------------------------------------------
# File browser helper
# ---------------------------------------------------------------------------

def browse_directory(path: str) -> dict:
    """List contents of a directory for the file browser."""
    path = os.path.expanduser(path)
    if not os.path.isdir(path):
        return {"error": f"Not a directory: {path}", "entries": [], "path": path}

    entries = []
    try:
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            is_dir = os.path.isdir(full)
            # Skip hidden files
            if name.startswith("."):
                continue
            entry = {
                "name": name,
                "path": full,
                "is_dir": is_dir,
                "size": "",
            }
            if not is_dir:
                try:
                    size = os.path.getsize(full)
                    if size < 1024:
                        entry["size"] = f"{size} B"
                    elif size < 1024 * 1024:
                        entry["size"] = f"{size / 1024:.1f} KB"
                    elif size < 1024 * 1024 * 1024:
                        entry["size"] = f"{size / (1024*1024):.1f} MB"
                    else:
                        entry["size"] = f"{size / (1024*1024*1024):.1f} GB"
                except OSError:
                    pass
                # Mark relevant file types
                ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
                entry["type"] = ext
            else:
                entry["type"] = "dir"
            entries.append(entry)
    except PermissionError:
        return {"error": f"Permission denied: {path}", "entries": [], "path": path}

    parent = os.path.dirname(path)
    return {
        "path": path,
        "parent": parent if parent != path else None,
        "entries": entries,
        "error": None,
    }

"""Job management for the RECOVAR GUI.

Discovers existing pipeline outputs, tracks launched jobs, and interfaces
with SLURM for cluster job submission and monitoring.
"""

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

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
        images = []
        output_dir = os.path.join(self.output_dir, "output")
        if not os.path.isdir(output_dir):
            return images
        for fname in sorted(os.listdir(output_dir)):
            if fname.endswith((".png", ".jpg", ".svg")):
                images.append(os.path.join(output_dir, fname))
        # Also check analysis subdirectories
        analysis_dir = os.path.join(output_dir, "analysis")
        if os.path.isdir(analysis_dir):
            for root, _, files in os.walk(analysis_dir):
                for fname in sorted(files):
                    if fname.endswith((".png", ".jpg", ".svg")):
                        images.append(os.path.join(root, fname))
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
                if "FAILED" in state or "CANCELLED" in state or "TIMEOUT" in state:
                    return STATUS_FAILED
                if "PENDING" in state:
                    return STATUS_QUEUED
    except Exception:
        pass
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
        data = {"jobs": [j.to_dict() for j in self._jobs.values()]}
        with open(self.jobs_file, "w") as f:
            json.dump(data, f, indent=2)

    def discover_jobs(self, scan_dirs: list[str]):
        """Scan directories for existing pipeline outputs."""
        for scan_dir in scan_dirs:
            if not os.path.isdir(scan_dir):
                continue
            for entry in os.listdir(scan_dir):
                job_dir = os.path.join(scan_dir, entry)
                meta_file = os.path.join(job_dir, "metadata.json")
                params_file = os.path.join(job_dir, "params.pkl")
                if os.path.isfile(meta_file) or os.path.isfile(params_file):
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
                    self._jobs[job_id] = job
        self._save()

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
            with open(script, "w") as f:
                f.write(f"""#!/bin/bash
#SBATCH --job-name=recovar-{name[:20]}
#SBATCH --output={log_dir}/slurm-%j.out
#SBATCH --error={log_dir}/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem={slurm_mem}
#SBATCH --time={slurm_time}
#SBATCH --partition={slurm_partition}
#SBATCH --gres=gpu:{slurm_gpus}
#SBATCH --account={slurm_account}

set -euo pipefail
export PYTHONNOUSERSITE=1
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
                        env={**os.environ, "PYTHONNOUSERSITE": "1",
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
                pass
        if job.pid:
            try:
                os.kill(job.pid, 15)  # SIGTERM
                job.status = STATUS_FAILED
                job.error = "Cancelled by user"
                self._save()
                return True
            except Exception:
                pass
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
        log_path = job.log_file
        # Also try gui_run.log
        if not os.path.isfile(log_path):
            log_path = os.path.join(job.output_dir, "gui_run.log")
        if not os.path.isfile(log_path):
            # Check for slurm logs in logs/
            log_dir = os.path.join(job.output_dir, "logs")
            if os.path.isdir(log_dir):
                logs = sorted(
                    [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith(".out")],
                    key=os.path.getmtime, reverse=True,
                )
                if logs:
                    log_path = logs[0]
        if not os.path.isfile(log_path):
            return "No log file found yet."
        try:
            with open(log_path) as f:
                lines = f.readlines()
            return "".join(lines[-n_lines:])
        except Exception as e:
            return f"Error reading log: {e}"

    def get_job_params(self, job_id: str) -> dict:
        """Read job parameters from metadata.json or params.pkl."""
        job = self._jobs.get(job_id)
        if not job:
            return {}
        meta_file = os.path.join(job.output_dir, "metadata.json")
        if os.path.isfile(meta_file):
            try:
                with open(meta_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"command": job.command, "output_dir": job.output_dir}

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
                job.status = new_status
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
                        job.error = "Process exited"


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
                import pickle
                with open(embeddings_path, "rb") as f:
                    emb = pickle.load(f)
                for key in ["latent_coords", "zs"]:
                    if key in emb and isinstance(emb[key], dict):
                        info["available_zdims"] = sorted(int(k) for k in emb[key].keys())
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

        # Scan analysis directories
        if os.path.isdir(job.output_dir):
            for entry in sorted(os.listdir(job.output_dir)):
                if not entry.startswith("analysis_"):
                    continue
                zdim_str = entry.split("_", 1)[1]
                try:
                    zdim = int(zdim_str)
                except ValueError:
                    continue
                analysis_dir = os.path.join(job.output_dir, entry)
                if not os.path.isdir(analysis_dir):
                    continue

                analysis = {"plots": [], "kmeans_volumes": [], "trajectories": []}

                # K-means volumes
                kmeans_dir = os.path.join(analysis_dir, "kmeans")
                if os.path.isdir(kmeans_dir):
                    for f in sorted(os.listdir(kmeans_dir)):
                        if f.endswith(".mrc") and "_half" not in f and "_unfil" not in f:
                            analysis["kmeans_volumes"].append({
                                "path": os.path.join(kmeans_dir, f),
                                "name": f,
                                "display_name": _vol_display_name(f),
                            })

                # Trajectories
                for tentry in sorted(os.listdir(analysis_dir)):
                    if not tentry.startswith("traj"):
                        continue
                    traj_dir = os.path.join(analysis_dir, tentry)
                    if not os.path.isdir(traj_dir):
                        continue
                    traj_vols = []
                    for f in sorted(os.listdir(traj_dir)):
                        if f.endswith(".mrc") and "_half" not in f and "_unfil" not in f:
                            traj_vols.append({
                                "path": os.path.join(traj_dir, f),
                                "name": f,
                                "display_name": _vol_display_name(f),
                            })
                    if traj_vols:
                        analysis["trajectories"].append({
                            "name": tentry,
                            "volumes": traj_vols,
                        })

                # Plots and images
                for root, _, files in os.walk(analysis_dir):
                    for f in sorted(files):
                        if f.endswith((".png", ".jpg", ".svg")):
                            analysis["plots"].append(os.path.join(root, f))

                info["analyses"][str(zdim)] = analysis

        # Scan GUI-computed volumes
        computed_dir = os.path.join(job.output_dir, "gui_computed")
        if os.path.isdir(computed_dir):
            for tdir in sorted(os.listdir(computed_dir), reverse=True):
                task_dir = os.path.join(computed_dir, tdir)
                if not os.path.isdir(task_dir):
                    continue
                for f in sorted(os.listdir(task_dir)):
                    if f.endswith(".mrc") and "_half" not in f and "_unfil" not in f:
                        info["computed"].append({
                            "path": os.path.join(task_dir, f),
                            "name": f"{tdir}/{f}",
                            "display_name": _vol_display_name(f),
                        })

        return info

    def get_embedding_data(self, job_id: str, zdim: int,
                           max_points: int = 15000) -> Optional[dict]:
        """Load embedding coordinates for scatter plot visualization."""
        import numpy as np
        job = self._jobs.get(job_id)
        if not job:
            return None

        embeddings_path = os.path.join(job.output_dir, "model", "embeddings.pkl")
        if not os.path.isfile(embeddings_path):
            return None

        try:
            import pickle
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

            if zdim not in coords_dict:
                return None

            zs = np.asarray(coords_dict[zdim], dtype=np.float32)
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

    # ── Compute task management ──────────────────────────────────

    def submit_compute_task(self, job_id: str, task_type: str,
                            params: dict, python_path: str,
                            use_slurm: bool = False,
                            slurm_opts: Optional[dict] = None) -> Optional[ComputeTask]:
        """Submit an async compute task (volume or trajectory)."""
        import numpy as np
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
                "--result-dir", job.output_dir,
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
                "--result-dir", job.output_dir,
                "--outdir", output_dir,
                "--zdim", str(zdim),
                "--n-vols-along-path", str(n_vols),
                "--endpts", os.path.join(output_dir, "endpoints.txt"),
                "--lazy",
            ]
        else:
            return None

        task = ComputeTask(
            id=task_id, job_id=job_id, task_type=task_type,
            output_dir=output_dir, created_at=time.time(),
            label=params.get("label", task_type),
        )

        env = {**os.environ, "PYTHONNOUSERSITE": "1",
               "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}

        if use_slurm and _has_slurm():
            opts = slurm_opts or {}
            script_path = os.path.join(output_dir, "compute.sbatch")
            with open(script_path, "w") as f:
                f.write(f"""#!/bin/bash
#SBATCH --job-name=recovar-compute
#SBATCH --output={output_dir}/compute.log
#SBATCH --error={output_dir}/compute.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem={opts.get('mem', '32G')}
#SBATCH --time={opts.get('time', '1:00:00')}
#SBATCH --partition={opts.get('partition', 'cryoem')}
#SBATCH --gres=gpu:1
#SBATCH --account={opts.get('account', 'amits')}

set -euo pipefail
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

{' '.join(cmd)}
""")
            slurm_id = _slurm_submit(script_path)
            if slurm_id:
                task.slurm_job_id = slurm_id
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
        return task

    def get_compute_task(self, task_id: str) -> Optional[dict]:
        """Get status of a compute task, including output volumes if done."""
        task = self._compute_tasks.get(task_id)
        if not task:
            return None

        # Refresh status
        if task.status == STATUS_RUNNING:
            if task.slurm_job_id:
                new_status = _slurm_job_status(task.slurm_job_id)
                if new_status:
                    task.status = new_status
            elif task.pid:
                try:
                    os.kill(task.pid, 0)
                except OSError:
                    # Process exited - check for output
                    has_output = any(
                        f.endswith(".mrc") and "_half" not in f and "_unfil" not in f
                        for f in os.listdir(task.output_dir)
                        if os.path.isfile(os.path.join(task.output_dir, f))
                    )
                    task.status = STATUS_COMPLETED if has_output else STATUS_FAILED
                    if not has_output:
                        task.error = "No output volumes generated"

        result = {
            "id": task.id,
            "type": task.task_type,
            "status": task.status,
            "error": task.error,
            "label": task.label,
            "volumes": [],
        }

        if task.status == STATUS_COMPLETED:
            for f in sorted(os.listdir(task.output_dir)):
                if f.endswith(".mrc") and "_half" not in f and "_unfil" not in f:
                    result["volumes"].append({
                        "path": os.path.join(task.output_dir, f),
                        "name": f,
                        "display_name": _vol_display_name(f),
                    })

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

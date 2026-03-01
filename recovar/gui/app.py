"""Flask application for the RECOVAR web GUI."""

import io
import json
import logging
import os

import numpy as np
from flask import (
    Flask, render_template, request, jsonify, redirect, url_for,
    send_file, Response,
)

from recovar.gui.job_manager import JobManager, browse_directory

logger = logging.getLogger(__name__)


def create_app(scan_dirs=None, state_dir=None, python_path=None):
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    )
    app.secret_key = os.urandom(24)

    if python_path is None:
        import sys
        python_path = sys.executable

    manager = JobManager(state_dir=state_dir)
    if scan_dirs:
        manager.discover_jobs(scan_dirs)

    # Store config on app
    app.config["PYTHON_PATH"] = python_path
    app.config["SCAN_DIRS"] = scan_dirs or []

    @app.context_processor
    def inject_recent_jobs():
        """Make recent jobs available to all templates (for sidebar)."""
        return {"recent_jobs": manager.list_jobs()[:10]}

    # ── Dashboard ──────────────────────────────────────────────────────
    @app.route("/")
    def dashboard():
        if scan_dirs:
            manager.discover_jobs(scan_dirs)
        jobs = manager.list_jobs()
        return render_template("dashboard.html", jobs=jobs, has_slurm=_has_slurm())

    # ── New Job ────────────────────────────────────────────────────────
    @app.route("/jobs/new")
    def new_job():
        return render_template("new_job.html", has_slurm=_has_slurm(),
                               python_path=python_path)

    @app.route("/jobs", methods=["POST"])
    def create_job():
        form = request.form
        name = form.get("name", "pipeline_run")
        output_dir = form.get("output_dir", "")
        particles = form.get("particles", "")
        mask = form.get("mask", "from_halfmaps")
        zdim = form.get("zdim", "1,2,4,10,20")
        downsample = form.get("downsample", "256")
        no_downsample = form.get("no_downsample") == "on"
        correct_contrast = form.get("correct_contrast") == "on"
        tilt_series = form.get("tilt_series") == "on"
        lazy = form.get("lazy") == "on"
        only_mean = form.get("only_mean") == "on"
        accept_cpu = form.get("accept_cpu") == "on"
        gpu_memory = form.get("gpu_memory", "")

        # Build command
        cmd_parts = [
            "recovar.commands.pipeline",
            particles,
            "-o", output_dir,
            "--mask", mask,
        ]
        if zdim:
            cmd_parts.extend(["--zdim", zdim])
        if no_downsample:
            cmd_parts.append("--no-downsample")
        elif downsample and downsample != "256":
            cmd_parts.extend(["--downsample", downsample])
        if correct_contrast:
            cmd_parts.append("--correct-contrast")
        if tilt_series:
            cmd_parts.append("--tilt-series")
        if lazy:
            cmd_parts.append("--lazy")
        if only_mean:
            cmd_parts.append("--only-mean")
        if accept_cpu:
            cmd_parts.append("--accept-cpu")
        if gpu_memory:
            cmd_parts.extend(["--gpu-gb", gpu_memory])

        command = " ".join(cmd_parts)

        use_slurm = form.get("execution") == "slurm"
        ds_val = None if no_downsample else (int(downsample) if downsample else None)

        job = manager.create_job(
            name=name,
            output_dir=output_dir,
            command=command,
            particles=particles,
            mask=mask,
            downsample=ds_val,
            use_slurm=use_slurm,
            slurm_partition=form.get("slurm_partition", "cryoem"),
            slurm_account=form.get("slurm_account", "amits"),
            slurm_gpus=int(form.get("slurm_gpus", "1")),
            slurm_mem=form.get("slurm_mem", "64G"),
            slurm_time=form.get("slurm_time", "4:00:00"),
            python_path=python_path,
        )
        return redirect(url_for("job_detail", job_id=job.id))

    # ── Job Detail ─────────────────────────────────────────────────────
    @app.route("/jobs/<job_id>")
    def job_detail(job_id):
        job = manager.get_job(job_id)
        if not job:
            return "Job not found", 404
        params = manager.get_job_params(job_id)
        return render_template("job_detail.html", job=job, params=params)

    @app.route("/jobs/<job_id>/cancel", methods=["POST"])
    def cancel_job(job_id):
        manager.cancel_job(job_id)
        return redirect(url_for("job_detail", job_id=job_id))

    @app.route("/jobs/<job_id>/delete", methods=["POST"])
    def delete_job(job_id):
        manager.delete_job(job_id)
        return redirect(url_for("dashboard"))

    # ── API: Logs (htmx partial) ───────────────────────────────────────
    @app.route("/api/jobs/<job_id>/logs")
    def api_job_logs(job_id):
        content = manager.get_log_content(job_id, n_lines=300)
        return f'<pre class="text-xs text-slate-300 font-mono whitespace-pre-wrap">{_escape(content)}</pre>'

    @app.route("/api/jobs/<job_id>/status")
    def api_job_status(job_id):
        job = manager.get_job(job_id)
        if not job:
            return ""
        color = _status_color(job.status)
        return f'<span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium {color}">' \
               f'<span class="w-2 h-2 rounded-full bg-current"></span>{job.status.upper()}</span>'

    # ── API: File Browser ──────────────────────────────────────────────
    @app.route("/api/browse")
    def api_browse():
        path = request.args.get("path", os.path.expanduser("~"))
        result = browse_directory(path)
        return jsonify(result)

    # ── API: Volume slice ──────────────────────────────────────────────
    @app.route("/api/volume/slice")
    def api_volume_slice():
        """Return a PNG slice of an MRC volume."""
        path = request.args.get("path", "")
        axis = int(request.args.get("axis", 2))
        idx = request.args.get("idx")

        if not os.path.isfile(path) or not path.endswith(".mrc"):
            return "Not found", 404

        try:
            import mrcfile
            with mrcfile.open(path, mode="r") as mrc:
                data = mrc.data
                if idx is None:
                    idx = data.shape[axis] // 2
                else:
                    idx = int(idx)
                idx = max(0, min(idx, data.shape[axis] - 1))
                slc = np.take(data, idx, axis=axis)

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=100)
            ax.imshow(slc, cmap="gray", origin="lower")
            ax.axis("off")
            fig.tight_layout(pad=0.1)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#1e293b")
            plt.close(fig)
            buf.seek(0)
            return send_file(buf, mimetype="image/png")
        except Exception as e:
            return str(e), 500

    @app.route("/api/volume/raw")
    def api_volume_raw():
        """Serve an MRC file directly for NGL viewer."""
        path = request.args.get("path", "")
        if not os.path.isfile(path) or not path.endswith(".mrc"):
            return "Not found", 404
        return send_file(path, mimetype="application/octet-stream",
                         download_name=os.path.basename(path))

    @app.route("/api/volume/info")
    def api_volume_info():
        """Return metadata about an MRC volume."""
        path = request.args.get("path", "")
        if not os.path.isfile(path) or not path.endswith(".mrc"):
            return jsonify({"error": "Not found"})
        try:
            import mrcfile
            with mrcfile.open(path, mode="r") as mrc:
                shape = list(mrc.data.shape)
                voxel_size = float(mrc.voxel_size.x)
                return jsonify({
                    "shape": shape,
                    "voxel_size": voxel_size,
                    "min": float(mrc.data.min()),
                    "max": float(mrc.data.max()),
                    "mean": float(mrc.data.mean()),
                })
        except Exception as e:
            return jsonify({"error": str(e)})

    # ── Serve result images ────────────────────────────────────────────
    @app.route("/api/image")
    def api_image():
        path = request.args.get("path", "")
        if not os.path.isfile(path):
            return "Not found", 404
        ext = path.rsplit(".", 1)[-1].lower()
        mimetypes = {"png": "image/png", "jpg": "image/jpeg",
                     "jpeg": "image/jpeg", "svg": "image/svg+xml"}
        return send_file(path, mimetype=mimetypes.get(ext, "image/png"))

    # ── API: Analysis Info ─────────────────────────────────────────────
    @app.route("/api/jobs/<job_id>/analysis")
    def api_analysis_info(job_id):
        """Return structured info about available analyses, volumes, zdims."""
        info = manager.get_analysis_info(job_id)
        return jsonify(info)

    # ── API: Embedding Data ───────────────────────────────────────────
    @app.route("/api/jobs/<job_id>/embeddings")
    def api_embeddings(job_id):
        """Return latent coordinates for scatter plot visualization."""
        zdim = request.args.get("zdim", type=int)
        max_points = request.args.get("max_points", 15000, type=int)
        if zdim is None:
            return jsonify({"error": "zdim parameter required"}), 400
        data = manager.get_embedding_data(job_id, zdim, max_points)
        if data is None:
            return jsonify({"error": "Embeddings not available"}), 404
        return jsonify(data)

    # ── API: Compute (volume / trajectory) ────────────────────────────
    @app.route("/api/jobs/<job_id>/compute", methods=["POST"])
    def api_compute(job_id):
        """Launch an async volume or trajectory computation."""
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400

        task_type = data.get("type")
        if task_type not in ("volume", "trajectory"):
            return jsonify({"error": "type must be 'volume' or 'trajectory'"}), 400

        use_slurm = data.get("use_slurm", _has_slurm())
        slurm_opts = data.get("slurm_opts")

        task = manager.submit_compute_task(
            job_id=job_id,
            task_type=task_type,
            params=data,
            python_path=python_path,
            use_slurm=use_slurm,
            slurm_opts=slurm_opts,
        )
        if task is None:
            return jsonify({"error": "Failed to create task"}), 500

        return jsonify({
            "task_id": task.id,
            "status": task.status,
            "error": task.error,
        })

    # ── API: Task Status ──────────────────────────────────────────────
    @app.route("/api/jobs/<job_id>/tasks/<task_id>")
    def api_task_status(job_id, task_id):
        """Check status of an async compute task."""
        result = manager.get_compute_task(task_id)
        if result is None:
            return jsonify({"error": "Task not found"}), 404
        return jsonify(result)

    @app.route("/api/jobs/<job_id>/tasks")
    def api_list_tasks(job_id):
        """List all compute tasks for a job."""
        tasks = manager.list_compute_tasks(job_id)
        return jsonify({"tasks": tasks})

    # ── System info ───────────────────────────────────────────────────
    @app.route("/api/system")
    def api_system_info():
        import shutil
        info = {"hostname": os.uname().nodename}
        # GPU info
        try:
            result = os.popen("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader").read()
            info["gpus"] = [line.strip() for line in result.strip().split("\n") if line.strip()]
        except Exception:
            info["gpus"] = []
        # Disk space
        try:
            usage = shutil.disk_usage(os.path.expanduser("~"))
            info["disk_free_gb"] = round(usage.free / (1024**3), 1)
            info["disk_total_gb"] = round(usage.total / (1024**3), 1)
        except Exception:
            info["disk_free_gb"] = 0
            info["disk_total_gb"] = 0
        info["has_slurm"] = _has_slurm()
        return jsonify(info)

    return app


def _has_slurm():
    import shutil
    return shutil.which("sbatch") is not None


def _status_color(status):
    return {
        "completed": "text-emerald-400 bg-emerald-400/10",
        "running": "text-sky-400 bg-sky-400/10",
        "queued": "text-amber-400 bg-amber-400/10",
        "failed": "text-red-400 bg-red-400/10",
    }.get(status, "text-slate-400 bg-slate-400/10")


def _escape(text):
    """HTML-escape text for safe embedding."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))

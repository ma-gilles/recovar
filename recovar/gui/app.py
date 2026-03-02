"""Flask application for the RECOVAR web GUI."""

import io
import logging
import os
import re
import shlex
import time

import numpy as np
from flask import (
    Flask, render_template, request, jsonify, redirect, url_for,
    send_file,
)

from recovar.gui.job_manager import JobManager, browse_directory

logger = logging.getLogger(__name__)


def create_app(scan_dirs=None, state_dir=None, python_path=None):
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
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
    # Store repo root so SLURM jobs get the correct PYTHONPATH
    app.config["REPO_ROOT"] = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        sort = request.args.get("sort", "newest")
        if sort == "oldest":
            jobs = list(reversed(jobs))
        elif sort == "name":
            jobs = sorted(jobs, key=lambda j: j.name.lower())
        elif sort == "status":
            status_order = {"running": 0, "queued": 1, "completed": 2, "failed": 3}
            jobs = sorted(jobs, key=lambda j: (status_order.get(j.status, 9), -j.created_at))
        return render_template("dashboard.html", jobs=jobs, has_slurm=_has_slurm(),
                               current_sort=sort)

    # ── New Job ────────────────────────────────────────────────────────
    @app.route("/jobs/new")
    def new_job():
        return render_template("new_job.html", has_slurm=_has_slurm(),
                               python_path=python_path, clone_from=None,
                               clone_params=None)

    @app.route("/jobs", methods=["POST"])
    def create_job():
        form = request.form
        job_type = form.get("job_type", "pipeline")
        name = form.get("name", f"{job_type}_run")
        output_dir = form.get("output_dir", "").strip()

        use_slurm = form.get("execution") == "slurm"
        slurm_partition = form.get("slurm_partition", "cryoem")
        slurm_account = form.get("slurm_account", "amits")
        slurm_gpus = int(form.get("slurm_gpus", "1"))
        slurm_mem = form.get("slurm_mem", "64G")
        slurm_time = form.get("slurm_time", "4:00:00")
        slurm_cpus = form.get("slurm_cpus", "8")
        slurm_extra = form.get("slurm_extra", "").strip()
        gpu_memory = form.get("gpu_memory", "")

        particles = ""
        mask = ""
        ds_val = None

        if job_type == "pipeline":
            particles = form.get("particles", "").strip()
            mask = form.get("mask", "from_halfmaps").strip()
            zdim = form.get("zdim", "1,2,4,10,20").strip()
            downsample = form.get("downsample", "256").strip()
            no_downsample = form.get("no_downsample") == "on"
            correct_contrast = form.get("correct_contrast") == "on"
            tilt_series = form.get("tilt_series") == "on"
            lazy = form.get("lazy") == "on"
            only_mean = form.get("only_mean") == "on"
            accept_cpu = form.get("accept_cpu") == "on"

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
            elif downsample:
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

            # Advanced options
            for opt_name, flag_name in [
                ("focus_mask", "--focus-mask"),
                ("mask_dilate_iter", "--mask-dilate-iter"),
                ("ind", "--ind"),
                ("halfsets", "--halfsets"),
                ("datadir", "--datadir"),
                ("n_images", "--n-images"),
                ("dose_per_tilt", "--dose-per-tilt"),
                ("angle_per_tilt", "--angle-per-tilt"),
                ("ntilts", "--ntilts"),
                ("tilt_series_ctf", "--tilt-series-ctf"),
            ]:
                val = form.get(opt_name, "").strip()
                if val:
                    cmd_parts.extend([flag_name, val])
            for opt_name, flag_name in [
                ("multi_gpu", "--multi-gpu"),
                ("low_memory", "--low-memory-option"),
                ("keep_intermediate", "--keep-intermediate"),
                ("ignore_zero_freq", "--ignore-zero-frequency"),
                ("keep_input_mask", "--keep-input-mask"),
                ("use_complement_mask", "--use-complement-mask"),
            ]:
                if form.get(opt_name) == "on":
                    cmd_parts.append(flag_name)

            ds_val = None if no_downsample else (int(downsample) if downsample else None)

        elif job_type == "analyze":
            result_dir = form.get("result_dir", "").strip()
            analyze_zdim = form.get("analyze_zdim", "20").strip()
            n_clusters = form.get("n_clusters", "20").strip()
            n_traj_vols = form.get("n_traj_vols", "6").strip()

            cmd_parts = [
                "recovar.commands.analyze",
                result_dir,
                "-o", output_dir,
                "--zdim", analyze_zdim,
                "--n-clusters", n_clusters,
                "--n-vols-along-path", n_traj_vols,
            ]
            if form.get("lazy") == "on":
                cmd_parts.append("--lazy")

        elif job_type == "compute_state":
            result_dir = form.get("result_dir", "").strip()

            cmd_parts = [
                "recovar.commands.compute_state",
                result_dir,
                "--outdir", output_dir,
                "--lazy",
            ]

        elif job_type == "compute_trajectory":
            result_dir = form.get("result_dir", "").strip()
            compute_zdim = form.get("compute_zdim", "4").strip()

            cmd_parts = [
                "recovar.commands.compute_trajectory",
                result_dir,
                "--outdir", output_dir,
                "--zdim", compute_zdim,
                "--lazy",
            ]

        else:
            cmd_parts = []

        # Append any extra CLI arguments the user typed in
        extra_args = form.get("extra_args", "").strip()
        if extra_args:
            cmd_parts.extend(shlex.split(extra_args))

        command = " ".join(cmd_parts)

        job = manager.create_job(
            name=name,
            output_dir=output_dir,
            command=command,
            particles=particles,
            mask=mask,
            downsample=ds_val,
            use_slurm=use_slurm,
            slurm_partition=slurm_partition,
            slurm_account=slurm_account,
            slurm_gpus=slurm_gpus,
            slurm_mem=slurm_mem,
            slurm_time=slurm_time,
            slurm_cpus=slurm_cpus,
            slurm_extra=slurm_extra,
            python_path=python_path,
        )
        return redirect(url_for("job_detail", job_id=job.id))

    @app.route("/jobs/<job_id>/clone")
    def clone_job(job_id):
        """Pre-fill a new job form from an existing job's parameters."""
        job = manager.get_job(job_id)
        if not job:
            return redirect(url_for("new_job"))
        # Parse command to extract all form params
        clone_params = _parse_command_params(job.command)
        # Override with stored fields (more reliable than parsed)
        if job.particles:
            clone_params['particles'] = job.particles
        if job.mask:
            clone_params['mask'] = job.mask
        if job.output_dir:
            clone_params['output_dir'] = job.output_dir
        # Smart name generation
        clone_params['name'] = _next_clone_name(job.name)
        # Read SLURM settings from sbatch script if available
        slurm_params = _parse_sbatch_params(manager.state_dir, job_id)
        clone_params.update(slurm_params)
        return render_template("new_job.html", has_slurm=_has_slurm(),
                               python_path=python_path, clone_from=job,
                               clone_params=clone_params)

    # ── Job Detail ─────────────────────────────────────────────────────
    @app.route("/jobs/<job_id>")
    def job_detail(job_id):
        job = manager.get_job(job_id)
        if not job:
            return "Job not found", 404
        # Lazily populate error for failed jobs
        if job.status == "failed" and not job.error:
            err = manager.get_error_summary(job_id)
            if err:
                job.error = err
                manager._save()
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
        # Strip ANSI escape codes
        content = re.sub(r'\x1b\[[0-9;]*m', '', content)
        # Highlight log lines by severity
        lines = content.split('\n')
        highlighted = []
        for line in lines:
            escaped = _escape(line)
            if any(kw in line for kw in ('ERROR', 'Traceback', 'Exception', 'FAILED')):
                highlighted.append(f'<span class="log-error">{escaped}</span>')
            elif any(kw in line for kw in ('WARNING', 'WARN', 'UserWarning')):
                highlighted.append(f'<span class="log-warning">{escaped}</span>')
            else:
                highlighted.append(escaped)
        html_content = '\n'.join(highlighted)
        return f'<pre class="text-xs text-slate-300 font-mono whitespace-pre-wrap">{html_content}</pre>'

    @app.route("/api/jobs/<job_id>/status")
    def api_job_status(job_id):
        job = manager.get_job(job_id)
        if not job:
            return ""
        # Lazily populate error summary for failed jobs
        if job.status == "failed" and not job.error:
            err = manager.get_error_summary(job_id)
            if err:
                job.error = err
                manager._save()
        color = _status_color(job.status)
        html = f'<span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium {color}">' \
               f'<span class="w-2 h-2 rounded-full bg-current"></span>{job.status.upper()}</span>'
        # Include error summary for failed jobs
        if job.status == "failed" and job.error:
            html += f'<div class="mt-2 text-xs text-red-400 bg-red-950/50 border border-red-500/20 rounded-lg px-3 py-2 font-mono">{_escape(job.error)}</div>'
        return html

    @app.route("/api/jobs/<job_id>/error")
    def api_job_error(job_id):
        """Return parsed error summary for a failed job."""
        summary = manager.get_error_summary(job_id)
        return jsonify({"error": summary})

    # ── API: File Browser ──────────────────────────────────────────────
    @app.route("/api/browse")
    def api_browse():
        raw_path = request.args.get("path", os.path.expanduser("~"))
        path = _safe_path(raw_path) or os.path.expanduser("~")
        result = browse_directory(path)
        return jsonify(result)

    # ── API: Validate STAR file ───────────────────────────────────────
    @app.route("/api/validate-star")
    def api_validate_star():
        """Quick validation of a STAR/CS file: check for poses, CTF, particle count."""
        path = _safe_path(request.args.get("path", ""))
        if not path or not os.path.isfile(path):
            return jsonify({"valid": False, "error": "File not found"})
        result = _validate_particles_file(path)
        return jsonify(result)

    # ── API: Volume slice ──────────────────────────────────────────────
    @app.route("/api/volume/slice")
    def api_volume_slice():
        """Return a PNG slice of an MRC volume."""
        path = _safe_path(request.args.get("path", ""))
        if not path or not os.path.isfile(path) or not path.endswith(".mrc"):
            return jsonify({"error": "Volume not found"}), 404

        axis = int(request.args.get("axis", 2))
        axis = max(0, min(axis, 2))
        idx = request.args.get("idx")
        if idx is not None:
            idx = int(idx)

        # Check cache first
        cached = _get_cached_slice(path, axis, idx)
        if cached is not None:
            return send_file(io.BytesIO(cached), mimetype="image/png")

        try:
            import mrcfile
            with mrcfile.open(path, mode="r") as mrc:
                data = mrc.data
                if idx is None:
                    idx = data.shape[axis] // 2
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
            buf_bytes = buf.read()
            _set_cached_slice(path, axis, idx, buf_bytes)
            return send_file(io.BytesIO(buf_bytes), mimetype="image/png")
        except Exception as e:
            logger.error("Volume slice error: %s", e)
            return jsonify({"error": "Failed to render slice"}), 500

    @app.route("/api/volume/raw")
    def api_volume_raw():
        """Serve an MRC file directly for NGL viewer."""
        path = _safe_path(request.args.get("path", ""))
        if not path or not os.path.isfile(path) or not path.endswith(".mrc"):
            return jsonify({"error": "Volume not found"}), 404
        return send_file(path, mimetype="application/octet-stream",
                         download_name=os.path.basename(path))

    @app.route("/api/volume/info")
    def api_volume_info():
        """Return metadata about an MRC volume."""
        path = _safe_path(request.args.get("path", ""))
        if not path or not os.path.isfile(path) or not path.endswith(".mrc"):
            return jsonify({"error": "Volume not found"}), 404
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
            logger.warning("Volume info error for %s: %s", path, e)
            return jsonify({"error": str(e)}), 500

    # ── Serve result images ────────────────────────────────────────────
    @app.route("/api/image")
    def api_image():
        path = _safe_path(request.args.get("path", ""))
        if not path or not os.path.isfile(path):
            return jsonify({"error": "Image not found"}), 404
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

    # ── API: UMAP Embeddings ─────────────────────────────────────
    @app.route("/api/jobs/<job_id>/embeddings/umap")
    def api_umap_embeddings(job_id):
        """Return UMAP coordinates for scatter plot visualization."""
        zdim = request.args.get("zdim", type=int)
        max_points = request.args.get("max_points", 15000, type=int)
        if zdim is None:
            return jsonify({"error": "zdim parameter required"}), 400
        data = manager.get_umap_data(job_id, zdim, max_points)
        if data is None:
            return jsonify({"error": "UMAP embeddings not available"}), 404
        return jsonify(data)

    # ── API: K-means Labels ───────────────────────────────────────
    @app.route("/api/jobs/<job_id>/embeddings/kmeans")
    def api_kmeans_labels(job_id):
        """Return k-means cluster labels for coloring the scatter plot."""
        zdim = request.args.get("zdim", type=int)
        max_points = request.args.get("max_points", 15000, type=int)
        if zdim is None:
            return jsonify({"error": "zdim parameter required"}), 400
        data = manager.get_kmeans_data(job_id, zdim, max_points)
        if data is None:
            return jsonify({"error": "K-means data not available"}), 404
        return jsonify(data)

    # ── API: Mask Export ──────────────────────────────────────────
    @app.route("/api/volume/mask", methods=["POST"])
    def api_volume_mask():
        """Create a binary mask from a volume at a given threshold."""
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400

        path = _safe_path(data.get("path", ""))
        threshold_sigma = float(data.get("threshold_sigma", 3.0))

        if not path or not os.path.isfile(path) or not path.endswith(".mrc"):
            return jsonify({"error": "Volume not found"}), 404

        try:
            import mrcfile
            from scipy.ndimage import binary_fill_holes, binary_dilation

            with mrcfile.open(path, mode="r") as mrc:
                vol = mrc.data.copy()
                voxel_size = float(mrc.voxel_size.x)

            threshold = vol.mean() + threshold_sigma * vol.std()
            mask = (vol >= threshold).astype(np.float32)

            # Fill holes and slight dilation for cleaner mask
            mask = binary_fill_holes(mask).astype(np.float32)
            mask = binary_dilation(mask, iterations=1).astype(np.float32)

            volume_fraction = float(mask.sum() / mask.size)

            # Save mask alongside the input volume
            mask_dir = os.path.dirname(path)
            base = os.path.splitext(os.path.basename(path))[0]
            output_path = data.get("output_path") or os.path.join(
                mask_dir, f"{base}_mask_{threshold_sigma:.1f}sigma.mrc"
            )

            with mrcfile.new(output_path, overwrite=True) as mrc_out:
                mrc_out.set_data(mask)
                mrc_out.voxel_size = voxel_size

            return jsonify({
                "path": output_path,
                "threshold": float(threshold),
                "threshold_sigma": threshold_sigma,
                "volume_fraction": volume_fraction,
            })
        except Exception as e:
            logger.error("Mask export failed: %s", e)
            return jsonify({"error": str(e)}), 500

    # ── API: Compute (volume / trajectory) ────────────────────────────
    @app.route("/api/jobs/<job_id>/compute", methods=["POST"])
    def api_compute(job_id):
        """Launch an async volume or trajectory computation."""
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400

        task_type = data.get("type")
        if task_type not in ("volume", "trajectory", "density", "stable_states"):
            return jsonify({"error": "type must be 'volume', 'trajectory', 'density', or 'stable_states'"}), 400

        use_slurm = data.get("use_slurm", _has_slurm())
        slurm_opts = data.get("slurm_opts")

        task = manager.submit_compute_task(
            job_id=job_id,
            task_type=task_type,
            params=data,
            python_path=python_path,
            use_slurm=use_slurm,
            slurm_opts=slurm_opts,
            repo_root=app.config.get("REPO_ROOT"),
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
            logger.debug("nvidia-smi query failed", exc_info=True)
            info["gpus"] = []
        # Disk space
        try:
            usage = shutil.disk_usage(os.path.expanduser("~"))
            info["disk_free_gb"] = round(usage.free / (1024**3), 1)
            info["disk_total_gb"] = round(usage.total / (1024**3), 1)
        except Exception:
            logger.debug("disk_usage query failed", exc_info=True)
            info["disk_free_gb"] = 0
            info["disk_total_gb"] = 0
        info["has_slurm"] = _has_slurm()
        return jsonify(info)

    @app.route("/debug/molstar")
    def debug_molstar():
        """Diagnostic page to test Mol* 3D viewer independently."""
        # Pick the first available volume for testing
        test_vol = request.args.get("path", "")
        if not test_vol:
            # Try to find any .mrc file from discovered jobs
            for job in manager._jobs.values():
                vols_dir = os.path.join(job.output_dir, "volumes")
                if os.path.isdir(vols_dir):
                    for f in sorted(os.listdir(vols_dir)):
                        if f.endswith(".mrc") and not f.startswith("eigen"):
                            test_vol = os.path.join(vols_dir, f)
                            break
                if test_vol:
                    break
        return render_template("debug_molstar.html", test_vol=test_vol)

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


def _safe_path(path: str):
    """Validate and resolve a path, preventing directory traversal."""
    if not path:
        return None
    resolved = os.path.realpath(path)
    if '..' in os.path.normpath(path).split(os.sep):
        return None
    return resolved


# Simple in-memory cache for volume slices with TTL
_slice_cache: dict[tuple, tuple[float, bytes]] = {}  # key -> (timestamp, bytes)
_SLICE_CACHE_MAX = 200
_SLICE_CACHE_TTL = 300  # seconds


def _get_cached_slice(path, axis, idx):
    try:
        key = (path, axis, idx, os.path.getmtime(path))
        entry = _slice_cache.get(key)
        if entry is None:
            return None
        ts, buf = entry
        if time.time() - ts > _SLICE_CACHE_TTL:
            del _slice_cache[key]
            return None
        return buf
    except OSError:
        return None


def _set_cached_slice(path, axis, idx, buf_bytes):
    try:
        key = (path, axis, idx, os.path.getmtime(path))
        if len(_slice_cache) >= _SLICE_CACHE_MAX:
            oldest_key = next(iter(_slice_cache))
            del _slice_cache[oldest_key]
        _slice_cache[key] = (time.time(), buf_bytes)
    except OSError:
        pass


def _next_clone_name(name: str) -> str:
    """Generate a smart incremented name for cloning/retrying a job.

    'my_run'        -> 'my_run_v2'
    'my_run_v2'     -> 'my_run_v3'
    'my_run_v12'    -> 'my_run_v13'
    'my_run_retry'  -> 'my_run_v2'
    'run_retry_retry' -> 'run_v2'
    """
    # Strip trailing _retry suffixes
    stripped = re.sub(r'(_retry)+$', '', name)
    # Check for existing _vN suffix
    m = re.match(r'^(.+)_v(\d+)$', stripped)
    if m:
        return f"{m.group(1)}_v{int(m.group(2)) + 1}"
    return f"{stripped}_v2"


def _parse_command_params(command: str) -> dict:
    """Parse a recovar CLI command string into a dict of form parameters."""
    params = {}
    if not command:
        return params
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()

    if not parts:
        return params

    # Detect job type from module name
    module = parts[0]
    if 'pipeline' in module:
        params['job_type'] = 'pipeline'
    elif 'analyze' in module:
        params['job_type'] = 'analyze'
    elif 'compute_state' in module:
        params['job_type'] = 'compute_state'
    elif 'compute_trajectory' in module:
        params['job_type'] = 'compute_trajectory'

    # Map CLI flags to form field names (these have dedicated form fields)
    value_flags = {
        '--mask': 'mask',
        '--zdim': 'zdim',
        '--downsample': 'downsample',
        '--gpu-gb': 'gpu_memory',
        '--gpu-memory': 'gpu_memory',
        '--focus-mask': 'focus_mask',
        '--mask-dilate-iter': 'mask_dilate_iter',
        '--ind': 'ind',
        '--halfsets': 'halfsets',
        '--datadir': 'datadir',
        '--n-images': 'n_images',
        '--dose-per-tilt': 'dose_per_tilt',
        '--angle-per-tilt': 'angle_per_tilt',
        '--ntilts': 'ntilts',
        '--tilt-series-ctf': 'tilt_series_ctf',
        '-o': 'output_dir',
        '--outdir': 'output_dir',
        '--result-dir': 'result_dir',
        '--n-clusters': 'n_clusters',
        '--n-traj-vols': 'n_traj_vols',
        # analyze
        '--zdim-for-analysis': 'analyze_zdim',
        '--compute-zdim': 'compute_zdim',
    }
    boolean_flags = {
        '--no-downsample': 'no_downsample',
        '--correct-contrast': 'correct_contrast',
        '--tilt-series': 'tilt_series',
        '--lazy': 'lazy',
        '--only-mean': 'only_mean',
        '--accept-cpu': 'accept_cpu',
        '--multi-gpu': 'multi_gpu',
        '--low-memory': 'low_memory',
        '--low-memory-option': 'low_memory',
        '--keep-intermediate': 'keep_intermediate',
        '--ignore-zero-freq': 'ignore_zero_freq',
        '--ignore-zero-frequency': 'ignore_zero_freq',
        '--keep-input-mask': 'keep_input_mask',
        '--use-complement-mask': 'use_complement_mask',
    }

    extra_parts = []  # CLI args not mapped to dedicated form fields

    i = 1  # skip module name
    # First positional arg after module is particles (for pipeline)
    if i < len(parts) and not parts[i].startswith('-'):
        if params.get('job_type') == 'pipeline':
            params['particles'] = parts[i]
        i += 1

    while i < len(parts):
        arg = parts[i]
        if arg in value_flags and i + 1 < len(parts):
            params[value_flags[arg]] = parts[i + 1]
            i += 2
        elif arg in boolean_flags:
            params[boolean_flags[arg]] = True
            i += 1
        elif arg.startswith('-') and i + 1 < len(parts) and not parts[i + 1].startswith('-'):
            # Unrecognized flag with a value -> extra args
            extra_parts.append(f"{arg} {shlex.quote(parts[i + 1])}")
            i += 2
        elif arg.startswith('-'):
            # Unrecognized boolean flag -> extra args
            extra_parts.append(arg)
            i += 1
        else:
            i += 1

    if extra_parts:
        params['extra_args'] = ' '.join(extra_parts)

    return params


def _parse_sbatch_params(state_dir: str, job_id: str) -> dict:
    """Parse SLURM parameters from a job's sbatch script."""
    params = {}
    script = os.path.join(state_dir, f"{job_id}.sbatch")
    if not os.path.isfile(script):
        return params
    try:
        with open(script) as f:
            content = f.read()
        sbatch_map = {
            '--partition=': 'slurm_partition',
            '--account=': 'slurm_account',
            '--mem=': 'slurm_mem',
            '--time=': 'slurm_time',
            '--cpus-per-task=': 'slurm_cpus',
        }
        for line in content.split('\n'):
            line = line.strip()
            if not line.startswith('#SBATCH'):
                continue
            directive = line[len('#SBATCH'):].strip()
            for prefix, key in sbatch_map.items():
                if directive.startswith(prefix):
                    params[key] = directive[len(prefix):]
            if directive.startswith('--gres=gpu:'):
                try:
                    params['slurm_gpus'] = directive.split(':')[-1]
                except (IndexError, ValueError):
                    pass
        # Collect extra sbatch flags we don't explicitly handle
        known_prefixes = list(sbatch_map.keys()) + [
            '--gres=', '--job-name=', '--output=', '--error=',
            '--nodes=', '--ntasks=',
        ]
        extra = []
        for line in content.split('\n'):
            line = line.strip()
            if not line.startswith('#SBATCH'):
                continue
            directive = line[len('#SBATCH'):].strip()
            if not any(directive.startswith(p) for p in known_prefixes):
                extra.append(directive)
        if extra:
            params['slurm_extra'] = ' '.join(extra)
        params['execution'] = 'slurm'
    except Exception:
        logger.debug("Failed to extract SLURM params from job script", exc_info=True)
    return params


def _validate_particles_file(path: str) -> dict:
    """Validate a particles file and return info about its contents."""
    result = {"valid": False, "path": path, "warnings": [], "info": {}}
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""

    if ext == "star":
        try:
            import starfile
            data = starfile.read(path, always_dict=True)

            # Find the particles table
            particles_df = None
            for key in ["particles", "data_particles", ""]:
                if key in data:
                    particles_df = data[key]
                    break
            if particles_df is None:
                # Try first available table
                for key, val in data.items():
                    if hasattr(val, "columns") and len(val) > 10:
                        particles_df = val
                        break

            if particles_df is None:
                result["error"] = "No particle data table found in STAR file"
                return result

            n_particles = len(particles_df)
            result["info"]["n_particles"] = n_particles
            cols = set(particles_df.columns)

            # Check for poses
            pose_cols = {"rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"}
            has_poses = pose_cols.issubset(cols)
            result["info"]["has_poses"] = has_poses
            if not has_poses:
                result["warnings"].append(
                    "Missing pose columns (rlnAngleRot/Tilt/Psi). "
                    "Pipeline will fail unless you provide a --poses PKL file."
                )

            # Check for CTF
            ctf_cols = {"rlnDefocusU", "rlnDefocusV"}
            has_ctf = ctf_cols.issubset(cols)
            result["info"]["has_ctf"] = has_ctf
            if not has_ctf:
                result["warnings"].append("No CTF columns found (rlnDefocusU/V)")

            # Check for image paths
            has_images = "rlnImageName" in cols
            result["info"]["has_images"] = has_images
            if not has_images:
                result["warnings"].append("No rlnImageName column — can't locate particle images")

            # Check for optics group
            has_optics = "optics" in data or "data_optics" in data
            result["info"]["has_optics"] = has_optics

            # Get pixel size if available
            if has_optics:
                optics_key = "optics" if "optics" in data else "data_optics"
                optics = data[optics_key]
                if hasattr(optics, "columns") and "rlnImagePixelSize" in optics.columns:
                    result["info"]["pixel_size"] = float(optics["rlnImagePixelSize"].iloc[0])
                if hasattr(optics, "columns") and "rlnImageSize" in optics.columns:
                    result["info"]["image_size"] = int(optics["rlnImageSize"].iloc[0])

            result["valid"] = True
            if result["warnings"]:
                result["valid"] = False  # Has critical warnings

        except Exception as e:
            result["error"] = f"Failed to parse STAR file: {e}"

    elif ext == "cs":
        try:
            import numpy as np
            cs = np.load(path)
            n_particles = len(cs)
            result["info"]["n_particles"] = n_particles
            fields = set(cs.dtype.names) if cs.dtype.names else set()

            has_poses = any("pose" in f.lower() for f in fields)
            has_ctf = any("ctf" in f.lower() for f in fields)
            result["info"]["has_poses"] = has_poses
            result["info"]["has_ctf"] = has_ctf

            if not has_poses:
                result["warnings"].append("No pose fields found in CS file")

            result["valid"] = True
        except Exception as e:
            result["error"] = f"Failed to parse CS file: {e}"

    elif ext in ("mrcs", "mrc"):
        result["valid"] = True
        result["info"]["format"] = "MRCS stack"
        result["warnings"].append("MRCS file selected — you'll also need poses and CTF (as PKL files or in a STAR file)")

    else:
        result["error"] = f"Unsupported file format: .{ext}"

    return result

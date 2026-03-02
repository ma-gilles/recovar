"""Tests for recovar.gui.job_manager module.

Covers helper functions, Job/ComputeTask dataclasses, and core JobManager
operations (persistence, discovery, analysis info, compute task lifecycle).
"""

import json
import os
import time

import pytest

from recovar.gui.job_manager import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    ComputeTask,
    Job,
    JobManager,
    _categorize_volume,
    _has_output_volumes,
    _is_output_volume,
    _list_images,
    _list_volumes,
    _load_json,
    _save_json,
    _vol_display_name,
)


# ---------------------------------------------------------------------------
# _is_output_volume
# ---------------------------------------------------------------------------

class TestIsOutputVolume:
    def test_accepts_plain_mrc(self):
        assert _is_output_volume("mean.mrc")

    def test_rejects_half_map(self):
        assert not _is_output_volume("mean_half1.mrc")
        assert not _is_output_volume("mean_half2_unfil.mrc")

    def test_rejects_unfiltered(self):
        assert not _is_output_volume("vol_unfil.mrc")

    def test_rejects_mask(self):
        assert not _is_output_volume("vol_mask.mrc")
        assert not _is_output_volume("dilated_mask.mrc")

    def test_rejects_non_mrc(self):
        assert not _is_output_volume("vol.txt")
        assert not _is_output_volume("vol.pkl")
        assert not _is_output_volume("")

    def test_accepts_eigenvector(self):
        assert _is_output_volume("eigen_pos0.mrc")

    def test_accepts_center(self):
        assert _is_output_volume("center0.mrc")

    def test_accepts_state(self):
        assert _is_output_volume("state0.mrc")


# ---------------------------------------------------------------------------
# _vol_display_name
# ---------------------------------------------------------------------------

class TestVolDisplayName:
    def test_known_names(self):
        assert _vol_display_name("mean.mrc") == "Mean Volume"
        assert _vol_display_name("mean_filt.mrc") == "Filtered Mean"
        assert _vol_display_name("mask.mrc") == "Solvent Mask"

    def test_eigenvector(self):
        assert _vol_display_name("eigen_pos0.mrc") == "PC 0"
        assert _vol_display_name("eigen_pos5.mrc") == "PC 5"

    def test_negative_eigenvector(self):
        assert "neg" in _vol_display_name("eigen_neg0.mrc").lower()

    def test_center(self):
        assert "Center" in _vol_display_name("center0.mrc")

    def test_state(self):
        assert "State" in _vol_display_name("state0.mrc")

    def test_variance(self):
        assert "Variance" in _vol_display_name("variance10.mrc")

    def test_unknown(self):
        # Unknown names return the stem
        assert _vol_display_name("custom_vol.mrc") == "custom_vol"


# ---------------------------------------------------------------------------
# _categorize_volume
# ---------------------------------------------------------------------------

class TestCategorizeVolume:
    def test_reconstruction(self):
        assert _categorize_volume("mean.mrc") == "reconstruction"
        assert _categorize_volume("mean_filt.mrc") == "reconstruction"

    def test_eigenvectors(self):
        assert _categorize_volume("eigen_pos0.mrc") == "eigenvectors"

    def test_variance(self):
        assert _categorize_volume("variance10.mrc") == "variance"

    def test_masks(self):
        assert _categorize_volume("mask.mrc") == "masks"
        assert _categorize_volume("dilated_mask.mrc") == "masks"

    def test_kmeans(self):
        assert _categorize_volume("center0.mrc") == "kmeans"

    def test_trajectory(self):
        assert _categorize_volume("state0.mrc") == "trajectory"

    def test_other(self):
        assert _categorize_volume("custom.mrc") == "other"


# ---------------------------------------------------------------------------
# _load_json / _save_json
# ---------------------------------------------------------------------------

class TestLoadSaveJson:
    def test_roundtrip(self, tmp_path):
        path = str(tmp_path / "data.json")
        data = {"key": "value", "nested": [1, 2, 3]}
        assert _save_json(path, data) is True
        loaded = _load_json(path)
        assert loaded == data

    def test_load_missing_returns_none(self, tmp_path):
        assert _load_json(str(tmp_path / "nonexistent.json")) is None

    def test_load_invalid_json_returns_none(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            f.write("not json{{{")
        assert _load_json(path) is None

    def test_save_to_invalid_path(self, tmp_path):
        assert _save_json(str(tmp_path / "no" / "such" / "dir" / "f.json"), {}) is False


# ---------------------------------------------------------------------------
# _list_volumes / _list_images / _has_output_volumes
# ---------------------------------------------------------------------------

class TestListVolumes:
    def test_empty_dir(self, tmp_path):
        assert _list_volumes(str(tmp_path)) == []

    def test_nonexistent_dir(self):
        assert _list_volumes("/no/such/dir") == []

    def test_filters_correctly(self, tmp_path):
        # Create test files
        for name in ["mean.mrc", "mean_half1.mrc", "mean_unfil.mrc",
                      "vol_mask.mrc", "center0.mrc", "notes.txt"]:
            (tmp_path / name).touch()
        vols = _list_volumes(str(tmp_path))
        names = [v["name"] for v in vols]
        assert "mean.mrc" in names
        assert "center0.mrc" in names
        assert "mean_half1.mrc" not in names
        assert "mean_unfil.mrc" not in names
        assert "vol_mask.mrc" not in names
        assert "notes.txt" not in names

    def test_returns_correct_structure(self, tmp_path):
        (tmp_path / "mean.mrc").touch()
        vols = _list_volumes(str(tmp_path))
        assert len(vols) == 1
        vol = vols[0]
        assert vol["path"] == str(tmp_path / "mean.mrc")
        assert vol["name"] == "mean.mrc"
        assert vol["display_name"] == "Mean Volume"

    def test_sorted_order(self, tmp_path):
        for name in ["center2.mrc", "center0.mrc", "center1.mrc"]:
            (tmp_path / name).touch()
        vols = _list_volumes(str(tmp_path))
        assert [v["name"] for v in vols] == ["center0.mrc", "center1.mrc", "center2.mrc"]


class TestListImages:
    def test_empty_dir(self, tmp_path):
        assert _list_images(str(tmp_path)) == []

    def test_nonexistent_dir(self):
        assert _list_images("/no/such/dir") == []

    def test_finds_images(self, tmp_path):
        for name in ["plot.png", "fig.jpg", "diagram.svg", "data.txt", "vol.mrc"]:
            (tmp_path / name).touch()
        imgs = _list_images(str(tmp_path))
        basenames = [os.path.basename(p) for p in imgs]
        assert "plot.png" in basenames
        assert "fig.jpg" in basenames
        assert "diagram.svg" in basenames
        assert "data.txt" not in basenames
        assert "vol.mrc" not in basenames


class TestHasOutputVolumes:
    def test_empty_dir(self, tmp_path):
        assert _has_output_volumes(str(tmp_path)) is False

    def test_nonexistent_dir(self):
        assert _has_output_volumes("/no/such/dir") is False

    def test_with_output_volumes(self, tmp_path):
        (tmp_path / "mean.mrc").touch()
        assert _has_output_volumes(str(tmp_path)) is True

    def test_only_half_maps(self, tmp_path):
        (tmp_path / "mean_half1.mrc").touch()
        assert _has_output_volumes(str(tmp_path)) is False


# ---------------------------------------------------------------------------
# Job dataclass
# ---------------------------------------------------------------------------

class TestJob:
    def test_roundtrip(self):
        job = Job(id="j1", name="test", output_dir="/tmp/out",
                  status=STATUS_COMPLETED, created_at=1000.0)
        d = job.to_dict()
        j2 = Job.from_dict(d)
        assert j2.id == "j1"
        assert j2.name == "test"
        assert j2.status == STATUS_COMPLETED

    def test_from_dict_ignores_unknown_fields(self):
        d = {"id": "j1", "name": "test", "output_dir": "/tmp", "extra_field": 42}
        job = Job.from_dict(d)
        assert job.id == "j1"
        assert not hasattr(job, "extra_field")

    def test_created_str(self):
        job = Job(id="j1", name="test", output_dir="/tmp", created_at=0.0)
        assert job.created_str == ""
        job.created_at = 1704067200.0  # 2024-01-01 00:00 UTC
        assert job.created_str  # non-empty string

    def test_default_status(self):
        job = Job(id="j1", name="test", output_dir="/tmp")
        assert job.status == STATUS_QUEUED


# ---------------------------------------------------------------------------
# ComputeTask dataclass
# ---------------------------------------------------------------------------

class TestComputeTask:
    def test_defaults(self):
        task = ComputeTask(id="t1", job_id="j1", task_type="volume",
                           status=STATUS_QUEUED, output_dir="/tmp")
        assert task.slurm_job_id is None
        assert task.pid is None
        assert task.error is None

    def test_to_dict(self):
        task = ComputeTask(id="t1", job_id="j1", task_type="volume",
                           status=STATUS_COMPLETED, output_dir="/tmp",
                           label="Test Volume")
        d = task.to_dict()
        assert d["id"] == "t1"
        assert d["label"] == "Test Volume"


# ---------------------------------------------------------------------------
# JobManager persistence
# ---------------------------------------------------------------------------

class TestJobManagerPersistence:
    def test_save_and_load(self, tmp_path):
        state_dir = str(tmp_path / "state")
        os.makedirs(state_dir)

        # Create a manager and add a job
        mgr = JobManager(state_dir=state_dir)
        job = Job(id="j1", name="test_job", output_dir=str(tmp_path / "output"),
                  status=STATUS_COMPLETED, created_at=time.time())
        mgr._jobs["j1"] = job
        mgr._save()

        # Load into a new manager
        mgr2 = JobManager(state_dir=state_dir)
        assert "j1" in mgr2._jobs
        assert mgr2._jobs["j1"].name == "test_job"
        assert mgr2._jobs["j1"].status == STATUS_COMPLETED

    def test_empty_state(self, tmp_path):
        state_dir = str(tmp_path / "state")
        os.makedirs(state_dir)
        mgr = JobManager(state_dir=state_dir)
        assert len(mgr._jobs) == 0


# ---------------------------------------------------------------------------
# JobManager.discover_jobs
# ---------------------------------------------------------------------------

class TestDiscoverJobs:
    def test_discovers_by_metadata_json(self, tmp_path):
        state_dir = str(tmp_path / "state")
        os.makedirs(state_dir)

        # Create a fake pipeline output
        job_dir = tmp_path / "scan" / "my_run"
        job_dir.mkdir(parents=True)
        with open(job_dir / "metadata.json", "w") as f:
            json.dump({"grid_size": 128}, f)

        mgr = JobManager(state_dir=state_dir)
        mgr.discover_jobs([str(tmp_path / "scan")])

        assert len(mgr._jobs) == 1
        job = list(mgr._jobs.values())[0]
        assert job.name == "my_run"
        assert job.status == STATUS_COMPLETED

    def test_discovers_by_params_pkl(self, tmp_path):
        state_dir = str(tmp_path / "state")
        os.makedirs(state_dir)

        job_dir = tmp_path / "scan" / "run2"
        (job_dir / "model").mkdir(parents=True)
        (job_dir / "model" / "params.pkl").touch()

        mgr = JobManager(state_dir=state_dir)
        mgr.discover_jobs([str(tmp_path / "scan")])

        assert len(mgr._jobs) == 1

    def test_skips_nonexistent_scan_dirs(self, tmp_path):
        state_dir = str(tmp_path / "state")
        os.makedirs(state_dir)
        mgr = JobManager(state_dir=state_dir)
        mgr.discover_jobs(["/no/such/dir"])
        assert len(mgr._jobs) == 0

    def test_does_not_duplicate_existing(self, tmp_path):
        state_dir = str(tmp_path / "state")
        os.makedirs(state_dir)

        job_dir = tmp_path / "scan" / "run1"
        job_dir.mkdir(parents=True)
        (job_dir / "metadata.json").write_text("{}")

        mgr = JobManager(state_dir=state_dir)
        mgr.discover_jobs([str(tmp_path / "scan")])
        count1 = len(mgr._jobs)
        mgr.discover_jobs([str(tmp_path / "scan")])
        assert len(mgr._jobs) == count1


# ---------------------------------------------------------------------------
# JobManager.get_analysis_info
# ---------------------------------------------------------------------------

class TestGetAnalysisInfo:
    def _make_job(self, tmp_path, state_dir):
        """Create a JobManager with a fake completed job."""
        os.makedirs(state_dir, exist_ok=True)
        output_dir = tmp_path / "output"

        # Create volumes directory
        vol_dir = output_dir / "output" / "volumes"
        vol_dir.mkdir(parents=True)
        for name in ["mean.mrc", "mean_half1.mrc", "eigen_pos0.mrc"]:
            (vol_dir / name).touch()

        # Create model directory
        model_dir = output_dir / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "params.pkl").touch()

        mgr = JobManager(state_dir=state_dir)
        job = Job(id="j1", name="test", output_dir=str(output_dir),
                  status=STATUS_COMPLETED, created_at=time.time())
        mgr._jobs["j1"] = job
        return mgr

    def test_finds_volumes(self, tmp_path):
        mgr = self._make_job(tmp_path, str(tmp_path / "state"))
        info = mgr.get_analysis_info("j1")
        vol_names = [v["name"] for v in info["volumes"]]
        assert "mean.mrc" in vol_names
        assert "eigen_pos0.mrc" in vol_names
        # Half maps included in volumes list (only _is_output_volume filters in analysis)
        assert info["has_model"] is True

    def test_finds_kmeans_volumes(self, tmp_path):
        mgr = self._make_job(tmp_path, str(tmp_path / "state"))
        job = mgr._jobs["j1"]

        # Create an analysis directory with kmeans
        analysis_dir = os.path.join(job.output_dir, "analysis_10")
        centers_dir = os.path.join(analysis_dir, "centers")
        kmeans_dir = os.path.join(analysis_dir, "kmeans")
        os.makedirs(centers_dir)
        os.makedirs(kmeans_dir)  # marker dir required by get_analysis_info
        for i in range(3):
            open(os.path.join(centers_dir, f"center{i}.mrc"), "w").close()

        info = mgr.get_analysis_info("j1")
        assert "10" in info["analyses"]
        assert len(info["analyses"]["10"]["kmeans_volumes"]) == 3

    def test_finds_trajectory_volumes(self, tmp_path):
        mgr = self._make_job(tmp_path, str(tmp_path / "state"))
        job = mgr._jobs["j1"]

        analysis_dir = os.path.join(job.output_dir, "analysis_5")
        traj_dir = os.path.join(analysis_dir, "traj0")
        umap_dir = os.path.join(analysis_dir, "umap")
        os.makedirs(traj_dir)
        os.makedirs(umap_dir)  # marker dir required by get_analysis_info
        for i in range(5):
            open(os.path.join(traj_dir, f"state{i}.mrc"), "w").close()

        info = mgr.get_analysis_info("j1")
        assert "5" in info["analyses"]
        trajs = info["analyses"]["5"]["trajectories"]
        assert len(trajs) == 1
        assert len(trajs[0]["volumes"]) == 5

    def test_finds_plots(self, tmp_path):
        mgr = self._make_job(tmp_path, str(tmp_path / "state"))
        job = mgr._jobs["j1"]

        analysis_dir = os.path.join(job.output_dir, "analysis_10")
        os.makedirs(analysis_dir)
        open(os.path.join(analysis_dir, "embedding.png"), "w").close()
        umap_dir = os.path.join(analysis_dir, "umap")
        os.makedirs(umap_dir)
        open(os.path.join(umap_dir, "umap_plot.png"), "w").close()

        info = mgr.get_analysis_info("j1")
        plots = info["analyses"]["10"]["plots"]
        assert len(plots) == 2

    def test_nonexistent_job(self, tmp_path):
        mgr = JobManager(state_dir=str(tmp_path))
        info = mgr.get_analysis_info("nonexistent")
        assert "error" in info

    def test_finds_computed_volumes(self, tmp_path):
        mgr = self._make_job(tmp_path, str(tmp_path / "state"))
        job = mgr._jobs["j1"]

        computed_dir = os.path.join(job.output_dir, "gui_computed", "volume_001")
        os.makedirs(computed_dir)
        open(os.path.join(computed_dir, "vol.mrc"), "w").close()

        info = mgr.get_analysis_info("j1")
        assert len(info["computed"]) == 1
        assert "volume_001/vol.mrc" == info["computed"][0]["name"]


# ---------------------------------------------------------------------------
# JobManager.get_job_params
# ---------------------------------------------------------------------------

class TestGetJobParams:
    def test_returns_metadata_json(self, tmp_path):
        state_dir = str(tmp_path / "state")
        os.makedirs(state_dir)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        with open(output_dir / "metadata.json", "w") as f:
            json.dump({"grid_size": 128, "zdim": 10}, f)

        mgr = JobManager(state_dir=state_dir)
        job = Job(id="j1", name="test", output_dir=str(output_dir))
        mgr._jobs["j1"] = job

        params = mgr.get_job_params("j1")
        assert params["grid_size"] == 128

    def test_fallback_to_command(self, tmp_path):
        state_dir = str(tmp_path / "state")
        os.makedirs(state_dir)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        mgr = JobManager(state_dir=state_dir)
        job = Job(id="j1", name="test", output_dir=str(output_dir),
                  command="recovar.commands.pipeline")
        mgr._jobs["j1"] = job

        params = mgr.get_job_params("j1")
        assert params["command"] == "recovar.commands.pipeline"

    def test_nonexistent_job(self, tmp_path):
        mgr = JobManager(state_dir=str(tmp_path))
        assert mgr.get_job_params("x") == {}


# ---------------------------------------------------------------------------
# JobManager._recover_compute_tasks
# ---------------------------------------------------------------------------

class TestRecoverComputeTasks:
    def test_recovers_volume_task_with_meta(self, tmp_path):
        state_dir = str(tmp_path / "state")
        os.makedirs(state_dir)

        output_dir = tmp_path / "output"
        task_dir = output_dir / "gui_computed" / "volume_abc"
        task_dir.mkdir(parents=True)

        # Create output and metadata
        (task_dir / "vol.mrc").touch()
        (task_dir / "compute.sbatch").touch()
        with open(task_dir / "task_meta.json", "w") as f:
            json.dump({
                "id": "volume_abc", "job_id": "j1",
                "task_type": "volume", "label": "Test Vol",
                "slurm_job_id": None, "created_at": 1000.0,
            }, f)

        mgr = JobManager(state_dir=state_dir)
        job = Job(id="j1", name="test", output_dir=str(output_dir),
                  status=STATUS_COMPLETED)
        mgr._jobs["j1"] = job
        mgr._recover_compute_tasks()

        assert "volume_abc" in mgr._compute_tasks
        task = mgr._compute_tasks["volume_abc"]
        assert task.status == STATUS_COMPLETED
        assert task.label == "Test Vol"

    def test_recovers_trajectory_task(self, tmp_path):
        state_dir = str(tmp_path / "state")
        os.makedirs(state_dir)

        output_dir = tmp_path / "output"
        task_dir = output_dir / "gui_computed" / "trajectory_xyz"
        task_dir.mkdir(parents=True)
        (task_dir / "state0.mrc").touch()
        (task_dir / "task_meta.json").touch()
        with open(task_dir / "task_meta.json", "w") as f:
            json.dump({"task_type": "trajectory"}, f)

        mgr = JobManager(state_dir=state_dir)
        job = Job(id="j1", name="test", output_dir=str(output_dir))
        mgr._jobs["j1"] = job
        mgr._recover_compute_tasks()

        assert "trajectory_xyz" in mgr._compute_tasks

    def test_skips_unknown_prefixes(self, tmp_path):
        state_dir = str(tmp_path / "state")
        os.makedirs(state_dir)

        output_dir = tmp_path / "output"
        task_dir = output_dir / "gui_computed" / "unknown_task"
        task_dir.mkdir(parents=True)
        (task_dir / "vol.mrc").touch()

        mgr = JobManager(state_dir=state_dir)
        job = Job(id="j1", name="test", output_dir=str(output_dir))
        mgr._jobs["j1"] = job
        mgr._recover_compute_tasks()

        assert "unknown_task" not in mgr._compute_tasks

    def test_marks_failed_when_no_output(self, tmp_path):
        state_dir = str(tmp_path / "state")
        os.makedirs(state_dir)

        output_dir = tmp_path / "output"
        task_dir = output_dir / "gui_computed" / "volume_empty"
        task_dir.mkdir(parents=True)
        (task_dir / "compute.sbatch").touch()
        # No .mrc files → should be marked failed

        mgr = JobManager(state_dir=state_dir)
        job = Job(id="j1", name="test", output_dir=str(output_dir))
        mgr._jobs["j1"] = job
        mgr._recover_compute_tasks()

        assert mgr._compute_tasks["volume_empty"].status == STATUS_FAILED


# ---------------------------------------------------------------------------
# JobManager.get_compute_task
# ---------------------------------------------------------------------------

class TestGetComputeTask:
    def test_returns_none_for_unknown(self, tmp_path):
        mgr = JobManager(state_dir=str(tmp_path))
        assert mgr.get_compute_task("x") is None

    def test_completed_task_lists_volumes(self, tmp_path):
        state_dir = str(tmp_path / "state")
        os.makedirs(state_dir)

        task_dir = tmp_path / "task_out"
        task_dir.mkdir()
        (task_dir / "vol.mrc").touch()

        mgr = JobManager(state_dir=state_dir)
        task = ComputeTask(
            id="t1", job_id="j1", task_type="volume",
            status=STATUS_COMPLETED, output_dir=str(task_dir),
            label="Test",
        )
        mgr._compute_tasks["t1"] = task

        result = mgr.get_compute_task("t1")
        assert result["status"] == STATUS_COMPLETED
        assert len(result["volumes"]) == 1
        assert result["volumes"][0]["name"] == "vol.mrc"

    def test_promotes_failed_to_completed_when_output_appears(self, tmp_path):
        state_dir = str(tmp_path / "state")
        os.makedirs(state_dir)

        task_dir = tmp_path / "task_out"
        task_dir.mkdir()

        mgr = JobManager(state_dir=state_dir)
        task = ComputeTask(
            id="t1", job_id="j1", task_type="volume",
            status=STATUS_FAILED, output_dir=str(task_dir),
            error="No output",
        )
        mgr._compute_tasks["t1"] = task

        # Initially failed
        result = mgr.get_compute_task("t1")
        assert result["status"] == STATUS_FAILED

        # Output appears
        (task_dir / "vol.mrc").touch()
        result = mgr.get_compute_task("t1")
        assert result["status"] == STATUS_COMPLETED
        assert result["error"] is None


# ---------------------------------------------------------------------------
# JobManager._save_task_meta
# ---------------------------------------------------------------------------

class TestSaveTaskMeta:
    def test_writes_metadata(self, tmp_path):
        state_dir = str(tmp_path / "state")
        os.makedirs(state_dir)
        task_dir = tmp_path / "task_out"
        task_dir.mkdir()

        mgr = JobManager(state_dir=state_dir)
        task = ComputeTask(
            id="t1", job_id="j1", task_type="volume",
            status=STATUS_COMPLETED, output_dir=str(task_dir),
            slurm_job_id="12345", label="My Volume",
            created_at=1000.0,
        )
        mgr._save_task_meta(task)

        meta = _load_json(str(task_dir / "task_meta.json"))
        assert meta is not None
        assert meta["id"] == "t1"
        assert meta["slurm_job_id"] == "12345"
        assert meta["label"] == "My Volume"


# ---------------------------------------------------------------------------
# JobManager.list_jobs / get_job / delete_job
# ---------------------------------------------------------------------------

class TestJobManagerCRUD:
    def _make_mgr(self, tmp_path):
        state_dir = str(tmp_path / "state")
        os.makedirs(state_dir)
        mgr = JobManager(state_dir=state_dir)
        for i in range(3):
            job = Job(id=f"j{i}", name=f"run_{i}",
                      output_dir=str(tmp_path / f"out_{i}"),
                      status=STATUS_COMPLETED,
                      created_at=1000.0 + i)
            mgr._jobs[job.id] = job
        return mgr

    def test_list_jobs_sorted_newest_first(self, tmp_path):
        mgr = self._make_mgr(tmp_path)
        jobs = mgr.list_jobs()
        assert jobs[0].id == "j2"  # newest
        assert jobs[-1].id == "j0"  # oldest

    def test_get_existing_job(self, tmp_path):
        mgr = self._make_mgr(tmp_path)
        job = mgr.get_job("j1")
        assert job is not None
        assert job.name == "run_1"

    def test_get_nonexistent_job(self, tmp_path):
        mgr = self._make_mgr(tmp_path)
        assert mgr.get_job("x") is None

    def test_delete_job(self, tmp_path):
        mgr = self._make_mgr(tmp_path)
        assert mgr.delete_job("j1") is True
        assert "j1" not in mgr._jobs
        assert mgr.delete_job("j1") is False


# ---------------------------------------------------------------------------
# JobManager.list_compute_tasks
# ---------------------------------------------------------------------------

class TestListComputeTasks:
    def test_filters_by_job_id(self, tmp_path):
        state_dir = str(tmp_path / "state")
        os.makedirs(state_dir)
        mgr = JobManager(state_dir=state_dir)

        for i, jid in enumerate(["j1", "j1", "j2"]):
            task = ComputeTask(
                id=f"t{i}", job_id=jid, task_type="volume",
                status=STATUS_COMPLETED, output_dir=str(tmp_path),
                created_at=1000.0 + i,
            )
            mgr._compute_tasks[task.id] = task

        tasks = mgr.list_compute_tasks("j1")
        assert len(tasks) == 2
        assert all(t["id"].startswith("t") for t in tasks)

    def test_empty_for_unknown_job(self, tmp_path):
        mgr = JobManager(state_dir=str(tmp_path))
        assert mgr.list_compute_tasks("x") == []

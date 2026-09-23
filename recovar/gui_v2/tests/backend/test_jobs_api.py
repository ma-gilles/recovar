"""Tests for the jobs API (Phase 1 Step 3).

Covers:
    - POST /api/jobs (submit)
    - GET /api/jobs/:id (detail)
    - POST /api/jobs/:id/cancel
    - POST /api/jobs/:id/reconcile
    - GET /api/jobs/:id/volumes
    - GET /api/jobs/:id/plots
    - GET /api/jobs/:id/suggested-next
    - Command builder unit tests
    - Startup reconciliation
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from recovar.gui_v2.backend.db import close_all
from recovar.gui_v2.backend.main import create_app
from recovar.gui_v2.backend.services.command_builder import (
    build_analyze_command,
    build_compute_state_command,
    build_compute_trajectory_command,
    build_density_command,
    build_downsample_command,
    build_pipeline_command,
    build_postprocess_command,
    build_stable_states_command,
)


@pytest.fixture
def app():
    return create_app()


@pytest_asyncio.fixture
async def client(app):
    # Clear shared state from previous tests
    from recovar.gui_v2.backend.api.project import _project_registry
    _project_registry.clear()
    from recovar.gui_v2.backend.api.jobs import _poll_tasks
    for t in _poll_tasks.values():
        t.cancel()
    _poll_tasks.clear()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    await close_all()


# ------------------------------------------------------------------
# Command builder unit tests
# ------------------------------------------------------------------


class TestCommandBuilders:
    def test_pipeline_command_minimal(self):
        cmd = build_pipeline_command({
            "particles": "/data/test.star",
            "mask": "from_halfmaps",
        })
        # cmd[0] is the recovar entry-point script (not python -m recovar)
        assert cmd[0].endswith("recovar") or "recovar" in cmd[0]
        assert "pipeline" in cmd
        assert "/data/test.star" in cmd
        assert "--mask" in cmd
        assert "from_halfmaps" in cmd

    def test_pipeline_command_full(self):
        cmd = build_pipeline_command({
            "particles": "/data/test.star",
            "mask": "/data/mask.mrc",
            "outdir": "/out",
            "zdim": [2, 4, 10],
            "downsample": 128,
            "lazy": True,
            "correct_contrast": True,
        })
        assert "--mask" in cmd
        assert "/data/mask.mrc" in cmd
        assert "-o" in cmd
        assert "/out" in cmd
        assert "--zdim" in cmd
        assert "2,4,10" in cmd
        assert "--downsample" in cmd
        assert "128" in cmd
        assert "--lazy" in cmd
        assert "--correct-contrast" in cmd

    def test_analyze_command(self):
        cmd = build_analyze_command({
            "result_dir": "/out/Pipeline/job_0001",
            "zdim": 4,
            "n_clusters": 20,
        })
        assert "analyze" in cmd
        assert "/out/Pipeline/job_0001" in cmd
        assert "--zdim" in cmd
        assert "--n-clusters" in cmd

    def test_compute_state_command(self):
        cmd = build_compute_state_command(
            {"result_dir": "/out/Pipeline/job_0001", "zdim": 4},
            "/tmp/coords.txt",
        )
        assert "compute_state" in cmd
        assert "--latent-points" in cmd
        assert "/tmp/coords.txt" in cmd

    def test_compute_trajectory_command(self):
        cmd = build_compute_trajectory_command(
            {"result_dir": "/out/Pipeline/job_0001", "zdim": 4, "n_vols_along_path": 6},
            "/tmp/start.txt",
            "/tmp/end.txt",
        )
        assert "compute_trajectory" in cmd
        assert "--z_st" in cmd
        assert "--z_end" in cmd
        assert "--n-vols-along-path" in cmd

    def test_density_command(self):
        cmd = build_density_command({
            "result_dir": "/out/Pipeline/job_0001",
            "outdir": "/out/density",
            "pca_dim": 3,
            "z_dim_used": 4,
        })
        assert "estimate_conformational_density" in cmd
        assert "/out/Pipeline/job_0001" in cmd
        assert "--output_dir" in cmd
        assert "--pca_dim" in cmd

    def test_stable_states_command(self):
        cmd = build_stable_states_command({
            "density": "/out/density/data/deconv_density_knee.pkl",
            "outdir": "/out/stable",
            "percent_top": 2.0,
            "n_local_maxs": 5,
        })
        assert "estimate_stable_states" in cmd
        assert "/out/density/data/deconv_density_knee.pkl" in cmd
        assert "-o" in cmd
        assert "--percent_top" in cmd
        assert "--n_local_maxs" in cmd

    def test_postprocess_command(self):
        cmd = build_postprocess_command({
            "input": "/data/halfmap1.mrc",
            "outdir": "/out/pp",
            "B_factor": -50.0,
            "batch": True,
            "estimate_B_factor": True,
            "local": True,
        })
        assert "postprocess" in cmd
        assert "/data/halfmap1.mrc" in cmd
        assert "--output" in cmd
        assert "--B-factor" in cmd
        assert "--batch" in cmd
        assert "--estimate-B-factor" in cmd
        assert "--local" in cmd

    def test_downsample_command(self):
        cmd = build_downsample_command({
            "particles": "/data/particles.star",
            "target_D": 128,
            "outdir": "/out/ds",
            "batch_size": 500,
        })
        assert "downsample" in cmd
        assert "/data/particles.star" in cmd
        assert "-D" in cmd
        assert "128" in cmd
        assert "--batch-size" in cmd
        assert "500" in cmd


# ------------------------------------------------------------------
# API endpoint tests (mock executor)
# ------------------------------------------------------------------


class TestJobsAPI:
    @pytest.mark.asyncio
    async def test_submit_and_get_job(self, client: AsyncClient, tmp_path: Path):
        """Submit a pipeline job, then retrieve it."""
        project_dir = str(tmp_path / "job_project")

        # Create project first
        resp = await client.post(
            "/api/projects",
            json={"path": project_dir, "name": "Jobs Test"},
        )
        project_id = resp.json()["id"]

        # Mock the executor
        mock_executor = AsyncMock()
        mock_executor.submit = AsyncMock(return_value="mock-handle-123")

        with patch("recovar.gui_v2.backend.api.jobs.get_executor", return_value=mock_executor):
            resp = await client.post(
                "/api/jobs",
                json={
                    "project_id": project_id,
                    "type": "Pipeline",
                    "params": {
                        "particles": "/data/test.star",
                        "mask": "from_halfmaps",
                    },
                },
            )

        assert resp.status_code == 201
        data = resp.json()
        assert data["type"] == "Pipeline"
        assert data["status"] == "running"
        assert data["handle"] == "mock-handle-123"
        job_id = data["id"]

        # GET the job
        resp = await client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200
        detail = resp.json()
        assert detail["id"] == job_id
        assert detail["type"] == "Pipeline"
        assert detail["output_dir"].endswith("job_0001")

    @pytest.mark.asyncio
    async def test_submit_unknown_type(self, client: AsyncClient, tmp_path: Path):
        project_dir = str(tmp_path / "bad_type")
        resp = await client.post(
            "/api/projects",
            json={"path": project_dir, "name": "Bad Type"},
        )
        project_id = resp.json()["id"]

        resp = await client.post(
            "/api/jobs",
            json={
                "project_id": project_id,
                "type": "InvalidType",
                "params": {},
            },
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_job_not_found(self, client: AsyncClient):
        resp = await client.get("/api/jobs/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_volumes_listing(self, client: AsyncClient, tmp_path: Path):
        """Create a job with MRC files and list them."""
        project_dir = str(tmp_path / "vol_project")
        resp = await client.post(
            "/api/projects",
            json={"path": project_dir, "name": "Volumes"},
        )
        project_id = resp.json()["id"]

        mock_executor = AsyncMock()
        mock_executor.submit = AsyncMock(return_value="handle")

        with patch("recovar.gui_v2.backend.api.jobs.get_executor", return_value=mock_executor):
            resp = await client.post(
                "/api/jobs",
                json={
                    "project_id": project_id,
                    "type": "Pipeline",
                    "params": {"particles": "/data/test.star", "mask": "sphere"},
                },
            )
        job_id = resp.json()["id"]

        # Get the output dir and create fake volumes
        job_resp = await client.get(f"/api/jobs/{job_id}")
        output_dir = job_resp.json()["output_dir"]
        vol_dir = os.path.join(output_dir, "output", "volumes")
        os.makedirs(vol_dir, exist_ok=True)
        for name in ("mean.mrc", "eigen_pos0000.mrc", "mask.mrc"):
            (Path(vol_dir) / name).write_bytes(b"\x00" * 100)

        # List volumes
        resp = await client.get(f"/api/jobs/{job_id}/volumes")
        assert resp.status_code == 200
        volumes = resp.json()
        assert len(volumes) == 3
        names = {v["name"] for v in volumes}
        assert "mean.mrc" in names
        categories = {v["category"] for v in volumes}
        assert "mean" in categories

    @pytest.mark.asyncio
    async def test_plots_listing(self, client: AsyncClient, tmp_path: Path):
        project_dir = str(tmp_path / "plot_project")
        resp = await client.post(
            "/api/projects",
            json={"path": project_dir, "name": "Plots"},
        )
        project_id = resp.json()["id"]

        mock_executor = AsyncMock()
        mock_executor.submit = AsyncMock(return_value="handle")

        with patch("recovar.gui_v2.backend.api.jobs.get_executor", return_value=mock_executor):
            resp = await client.post(
                "/api/jobs",
                json={
                    "project_id": project_id,
                    "type": "Pipeline",
                    "params": {"particles": "/data/test.star", "mask": "sphere"},
                },
            )
        job_id = resp.json()["id"]

        # Create fake plots
        job_resp = await client.get(f"/api/jobs/{job_id}")
        output_dir = job_resp.json()["output_dir"]
        plots_dir = os.path.join(output_dir, "output", "plots")
        os.makedirs(plots_dir, exist_ok=True)
        (Path(plots_dir) / "summary.png").write_bytes(b"\x89PNG")

        resp = await client.get(f"/api/jobs/{job_id}/plots")
        assert resp.status_code == 200
        assert len(resp.json()) == 1
        assert resp.json()[0]["name"] == "summary.png"

    @pytest.mark.asyncio
    async def test_suggested_next_for_pipeline(
        self, client: AsyncClient, tmp_path: Path
    ):
        """Completed Pipeline job suggests Analyze."""
        project_dir = str(tmp_path / "suggest_project")
        resp = await client.post(
            "/api/projects",
            json={"path": project_dir, "name": "Suggest"},
        )
        project_id = resp.json()["id"]

        mock_executor = AsyncMock()
        mock_executor.submit = AsyncMock(return_value="handle")

        with patch("recovar.gui_v2.backend.api.jobs.get_executor", return_value=mock_executor):
            resp = await client.post(
                "/api/jobs",
                json={
                    "project_id": project_id,
                    "type": "Pipeline",
                    "params": {"particles": "/data/test.star", "mask": "sphere"},
                },
            )
        job_id = resp.json()["id"]

        # Force the job to completed state
        from recovar.gui_v2.backend.config import get_db_path
        from recovar.gui_v2.backend.db import init_db
        from recovar.gui_v2.backend.models.job import Job
        from sqlalchemy import select

        db_path = get_db_path(project_dir)
        sf = await init_db(db_path)
        async with sf() as session:
            stmt = select(Job).where(Job.id == job_id)
            result = await session.execute(stmt)
            job = result.scalar_one()
            job.status = "completed"
            await session.commit()

        # Verify status update is visible via API
        check = await client.get(f"/api/jobs/{job_id}")
        assert check.json()["status"] == "completed"

        resp = await client.get(f"/api/jobs/{job_id}/suggested-next")
        assert resp.status_code == 200
        suggestions = resp.json()
        assert len(suggestions) >= 1
        types = [s["type"] for s in suggestions]
        assert "Analyze" in types
        assert "Density" in types

    @pytest.mark.asyncio
    async def test_submit_density_job(self, client: AsyncClient, tmp_path: Path):
        project_dir = str(tmp_path / "density_project")
        resp = await client.post(
            "/api/projects",
            json={"path": project_dir, "name": "Density Test"},
        )
        project_id = resp.json()["id"]
        mock_executor = AsyncMock()
        mock_executor.submit = AsyncMock(return_value="density-handle")
        with patch("recovar.gui_v2.backend.api.jobs.get_executor", return_value=mock_executor):
            resp = await client.post(
                "/api/jobs",
                json={
                    "project_id": project_id,
                    "type": "density",
                    "params": {"result_dir": "/data/pipeline_output", "pca_dim": 3},
                },
            )
        assert resp.status_code == 201
        assert resp.json()["type"] == "Density"

    @pytest.mark.asyncio
    async def test_submit_downsample_job(self, client: AsyncClient, tmp_path: Path):
        project_dir = str(tmp_path / "ds_project")
        resp = await client.post(
            "/api/projects",
            json={"path": project_dir, "name": "Downsample Test"},
        )
        project_id = resp.json()["id"]
        mock_executor = AsyncMock()
        mock_executor.submit = AsyncMock(return_value="ds-handle")
        with patch("recovar.gui_v2.backend.api.jobs.get_executor", return_value=mock_executor):
            resp = await client.post(
                "/api/jobs",
                json={
                    "project_id": project_id,
                    "type": "downsample",
                    "params": {"particles": "/data/particles.star", "target_D": 128},
                },
            )
        assert resp.status_code == 201
        assert resp.json()["type"] == "Downsample"

    @pytest.mark.asyncio
    async def test_submit_postprocess_job(self, client: AsyncClient, tmp_path: Path):
        project_dir = str(tmp_path / "pp_project")
        resp = await client.post(
            "/api/projects",
            json={"path": project_dir, "name": "Postprocess Test"},
        )
        project_id = resp.json()["id"]
        mock_executor = AsyncMock()
        mock_executor.submit = AsyncMock(return_value="pp-handle")
        with patch("recovar.gui_v2.backend.api.jobs.get_executor", return_value=mock_executor):
            resp = await client.post(
                "/api/jobs",
                json={
                    "project_id": project_id,
                    "type": "postprocess",
                    "params": {"input": "/data/halfmap1.mrc"},
                },
            )
        assert resp.status_code == 201
        assert resp.json()["type"] == "Postprocess"

    @pytest.mark.asyncio
    async def test_submit_stable_states_job(self, client: AsyncClient, tmp_path: Path):
        project_dir = str(tmp_path / "ss_project")
        resp = await client.post(
            "/api/projects",
            json={"path": project_dir, "name": "StableStates Test"},
        )
        project_id = resp.json()["id"]
        mock_executor = AsyncMock()
        mock_executor.submit = AsyncMock(return_value="ss-handle")
        with patch("recovar.gui_v2.backend.api.jobs.get_executor", return_value=mock_executor):
            resp = await client.post(
                "/api/jobs",
                json={
                    "project_id": project_id,
                    "type": "stable_states",
                    "params": {"density": "/data/density.pkl"},
                },
            )
        assert resp.status_code == 201
        assert resp.json()["type"] == "StableStates"


# ------------------------------------------------------------------
# Reconcile endpoint tests
# ------------------------------------------------------------------


class TestReconcileEndpoint:
    """Tests for POST /api/jobs/:id/reconcile."""

    async def _create_running_job(
        self, client: AsyncClient, tmp_path: Path
    ) -> tuple[str, str]:
        """Helper: create a project + submit a running job.  Returns (project_id, job_id)."""
        project_dir = str(tmp_path / "reconcile_project")
        resp = await client.post(
            "/api/projects",
            json={"path": project_dir, "name": "Reconcile Test"},
        )
        project_id = resp.json()["id"]

        mock_executor = AsyncMock()
        mock_executor.submit = AsyncMock(return_value="slurm-42")

        with patch(
            "recovar.gui_v2.backend.api.jobs.get_executor",
            return_value=mock_executor,
        ):
            resp = await client.post(
                "/api/jobs",
                json={
                    "project_id": project_id,
                    "type": "Pipeline",
                    "params": {
                        "particles": "/data/test.star",
                        "mask": "sphere",
                    },
                },
            )
        assert resp.status_code == 201
        job_id = resp.json()["id"]
        return project_id, job_id

    @pytest.mark.asyncio
    async def test_reconcile_running_to_completed(
        self, client: AsyncClient, tmp_path: Path
    ):
        """Job is running in DB but SLURM says COMPLETED -> should update."""
        _, job_id = await self._create_running_job(client, tmp_path)

        # Verify it's running
        resp = await client.get(f"/api/jobs/{job_id}")
        assert resp.json()["status"] == "running"

        # Mock executor.status to return COMPLETED
        from recovar.gui_v2.backend.services.executor import (
            JobStatus as ExecJobStatus,
        )

        mock_executor = AsyncMock()
        mock_executor.status = AsyncMock(return_value=ExecJobStatus.COMPLETED)

        with patch(
            "recovar.gui_v2.backend.api.jobs.get_executor",
            return_value=mock_executor,
        ):
            resp = await client.post(f"/api/jobs/{job_id}/reconcile")

        assert resp.status_code == 200
        data = resp.json()
        assert data["previous_status"] == "running"
        assert data["new_status"] == "completed"
        assert data["changed"] is True

        # Verify the DB was updated
        resp = await client.get(f"/api/jobs/{job_id}")
        assert resp.json()["status"] == "completed"

    @pytest.mark.asyncio
    async def test_reconcile_running_to_failed(
        self, client: AsyncClient, tmp_path: Path
    ):
        """Job is running in DB but SLURM says FAILED -> should update."""
        _, job_id = await self._create_running_job(client, tmp_path)

        from recovar.gui_v2.backend.services.executor import (
            JobStatus as ExecJobStatus,
        )

        mock_executor = AsyncMock()
        mock_executor.status = AsyncMock(return_value=ExecJobStatus.FAILED)

        with patch(
            "recovar.gui_v2.backend.api.jobs.get_executor",
            return_value=mock_executor,
        ):
            resp = await client.post(f"/api/jobs/{job_id}/reconcile")

        assert resp.status_code == 200
        data = resp.json()
        assert data["new_status"] == "failed"
        assert data["changed"] is True
        assert data["error"] is not None

    @pytest.mark.asyncio
    async def test_reconcile_still_running(
        self, client: AsyncClient, tmp_path: Path
    ):
        """Job is running in both DB and SLURM -> no change."""
        _, job_id = await self._create_running_job(client, tmp_path)

        from recovar.gui_v2.backend.services.executor import (
            JobStatus as ExecJobStatus,
        )

        mock_executor = AsyncMock()
        mock_executor.status = AsyncMock(return_value=ExecJobStatus.RUNNING)

        with patch(
            "recovar.gui_v2.backend.api.jobs.get_executor",
            return_value=mock_executor,
        ):
            resp = await client.post(f"/api/jobs/{job_id}/reconcile")

        assert resp.status_code == 200
        data = resp.json()
        assert data["changed"] is False
        assert data["new_status"] == "running"

    @pytest.mark.asyncio
    async def test_reconcile_already_terminal(
        self, client: AsyncClient, tmp_path: Path
    ):
        """Job already completed -> reconcile is a no-op."""
        _, job_id = await self._create_running_job(client, tmp_path)

        # Force the job to completed state
        from recovar.gui_v2.backend.config import get_db_path
        from recovar.gui_v2.backend.db import init_db
        from recovar.gui_v2.backend.models.job import Job
        from sqlalchemy import select

        project_dir = str(tmp_path / "reconcile_project")
        db_path = get_db_path(project_dir)
        sf = await init_db(db_path)
        async with sf() as session:
            stmt = select(Job).where(Job.id == job_id)
            result = await session.execute(stmt)
            job = result.scalar_one()
            job.status = "completed"
            await session.commit()

        resp = await client.post(f"/api/jobs/{job_id}/reconcile")
        assert resp.status_code == 200
        data = resp.json()
        assert data["changed"] is False
        assert data["new_status"] == "completed"

    @pytest.mark.asyncio
    async def test_reconcile_unknown_becomes_failed(
        self, client: AsyncClient, tmp_path: Path
    ):
        """SLURM returns UNKNOWN -> should mark as failed."""
        _, job_id = await self._create_running_job(client, tmp_path)

        from recovar.gui_v2.backend.services.executor import (
            JobStatus as ExecJobStatus,
        )

        mock_executor = AsyncMock()
        mock_executor.status = AsyncMock(return_value=ExecJobStatus.UNKNOWN)

        with patch(
            "recovar.gui_v2.backend.api.jobs.get_executor",
            return_value=mock_executor,
        ):
            resp = await client.post(f"/api/jobs/{job_id}/reconcile")

        assert resp.status_code == 200
        data = resp.json()
        assert data["new_status"] == "failed"
        assert data["changed"] is True
        assert "unknown" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_reconcile_not_found(self, client: AsyncClient):
        resp = await client.post("/api/jobs/nonexistent/reconcile")
        assert resp.status_code == 404


# ------------------------------------------------------------------
# Startup reconciliation tests
# ------------------------------------------------------------------


class TestStartupReconcile:
    """Tests for _reconcile_on_startup in main.py."""

    @pytest.mark.asyncio
    async def test_reconcile_on_startup_updates_completed_job(
        self, tmp_path: Path
    ):
        """Simulate a job that completed while server was down."""
        from recovar.gui_v2.backend.api.jobs import _poll_tasks
        from recovar.gui_v2.backend.api.project import _project_registry
        from recovar.gui_v2.backend.config import get_db_path
        from recovar.gui_v2.backend.db import init_db
        from recovar.gui_v2.backend.main import _reconcile_on_startup
        from recovar.gui_v2.backend.models.job import Job, JobStatus
        from recovar.gui_v2.backend.models.project import Project
        from recovar.gui_v2.backend.services.executor import (
            JobStatus as ExecJobStatus,
        )
        from sqlalchemy import select

        # Clear state
        _project_registry.clear()
        for t in _poll_tasks.values():
            t.cancel()
        _poll_tasks.clear()

        # Create project DB with a "running" job
        project_dir = str(tmp_path / "startup_project")
        os.makedirs(project_dir, exist_ok=True)
        db_path = get_db_path(project_dir)
        sf = await init_db(db_path)

        async with sf() as session:
            project = Project(name="Startup Test", path=project_dir)
            session.add(project)
            await session.flush()

            job = Job(
                project_id=project.id,
                type="Pipeline",
                status=JobStatus.RUNNING.value,
                output_dir=str(tmp_path / "startup_project" / "Pipeline" / "job_0001"),
                executor_handle="slurm-999",
                slurm_id="999",
            )
            session.add(job)
            await session.commit()
            job_id = job.id
            project_id = project.id

        # Register the project
        _project_registry[project_id] = project_dir

        # Mock executor to return COMPLETED
        mock_executor = AsyncMock()
        mock_executor.status = AsyncMock(return_value=ExecJobStatus.COMPLETED)

        with patch(
            "recovar.gui_v2.backend.api.jobs.get_executor",
            return_value=mock_executor,
        ):
            await _reconcile_on_startup()

        # Verify the job was updated
        async with sf() as session:
            stmt = select(Job).where(Job.id == job_id)
            result = await session.execute(stmt)
            job = result.scalar_one()
            assert job.status == "completed"
            assert job.completed_at is not None

        # Clean up
        _project_registry.clear()
        await close_all()

    @pytest.mark.asyncio
    async def test_reconcile_on_startup_restarts_poller_for_running(
        self, tmp_path: Path
    ):
        """Job that is still running should get a new background poller."""
        from recovar.gui_v2.backend.api.jobs import _poll_tasks
        from recovar.gui_v2.backend.api.project import _project_registry
        from recovar.gui_v2.backend.config import get_db_path
        from recovar.gui_v2.backend.db import init_db
        from recovar.gui_v2.backend.main import _reconcile_on_startup
        from recovar.gui_v2.backend.models.job import Job, JobStatus
        from recovar.gui_v2.backend.models.project import Project
        from recovar.gui_v2.backend.services.executor import (
            JobStatus as ExecJobStatus,
        )

        # Clear state
        _project_registry.clear()
        for t in _poll_tasks.values():
            t.cancel()
        _poll_tasks.clear()

        project_dir = str(tmp_path / "poller_project")
        os.makedirs(project_dir, exist_ok=True)
        db_path = get_db_path(project_dir)
        sf = await init_db(db_path)

        async with sf() as session:
            project = Project(name="Poller Test", path=project_dir)
            session.add(project)
            await session.flush()

            job = Job(
                project_id=project.id,
                type="Pipeline",
                status=JobStatus.RUNNING.value,
                output_dir=str(tmp_path / "poller_project" / "Pipeline" / "job_0001"),
                executor_handle="slurm-888",
                slurm_id="888",
            )
            session.add(job)
            await session.commit()
            job_id = job.id
            project_id = project.id

        _project_registry[project_id] = project_dir

        # Mock executor to return RUNNING (job is still active)
        mock_executor = AsyncMock()
        mock_executor.status = AsyncMock(return_value=ExecJobStatus.RUNNING)

        with patch(
            "recovar.gui_v2.backend.api.jobs.get_executor",
            return_value=mock_executor,
        ):
            # Also patch _poll_job_status so the spawned task doesn't actually loop
            with patch(
                "recovar.gui_v2.backend.api.jobs._poll_job_status",
                new_callable=AsyncMock,
            ) as mock_poll:
                await _reconcile_on_startup()

                # A poller should have been created
                assert job_id in _poll_tasks

        # Clean up
        for t in _poll_tasks.values():
            t.cancel()
        _poll_tasks.clear()
        _project_registry.clear()
        await close_all()

    @pytest.mark.asyncio
    async def test_reconcile_on_startup_no_projects(self):
        """No registered projects -> reconcile is a no-op."""
        from recovar.gui_v2.backend.api.project import _project_registry
        from recovar.gui_v2.backend.main import _reconcile_on_startup

        _project_registry.clear()
        # Should not raise
        await _reconcile_on_startup()

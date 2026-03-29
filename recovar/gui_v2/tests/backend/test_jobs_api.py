"""Tests for the jobs API (Phase 1 Step 3).

Covers:
    - POST /api/jobs (submit)
    - GET /api/jobs/:id (detail)
    - POST /api/jobs/:id/cancel
    - GET /api/jobs/:id/volumes
    - GET /api/jobs/:id/plots
    - GET /api/jobs/:id/suggested-next
    - Command builder unit tests
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
    build_pipeline_command,
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
        assert cmd[0] == sys.executable
        assert "-m" in cmd
        assert "recovar" in cmd
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
        assert suggestions[0]["type"] == "Analyze"

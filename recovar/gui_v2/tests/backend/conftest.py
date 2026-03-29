"""Shared pytest fixtures for backend tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from recovar.gui_v2.backend.db import close_all
from recovar.gui_v2.backend.main import create_app


@pytest.fixture
def app():
    """Create a fresh FastAPI app for each test."""
    return create_app()


@pytest_asyncio.fixture
async def client(app):
    """Async HTTP client wired to the test app."""
    # Clear shared state from previous tests
    from recovar.gui_v2.backend.api.project import _project_registry
    _project_registry.clear()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    # Clean up all DB connections after each test
    await close_all()


@pytest.fixture
def tmp_project_dir(tmp_path: Path) -> Path:
    """Create a temporary directory to use as a project root."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    return project_dir


@pytest.fixture
def tmp_project_with_pipeline(tmp_path: Path) -> Path:
    """Create a project directory with a fake pipeline output for scan tests.

    Layout::

        test_project/
        ├── project.json
        └── Pipeline/
            └── job_0001/
                ├── job.json
                ├── model/
                │   ├── metadata.json
                │   └── params.pkl   (empty marker)
                └── output/
                    └── volumes/
                        └── mean.mrc (empty marker)
    """
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()

    # project.json (CLI canonical)
    project_json = {
        "version": "1.0",
        "name": "test_project",
        "created": "2026-01-01T00:00:00",
        "recovar_version": "1.0.0",
        "counters": {"Pipeline": 1},
        "jobs": [
            {
                "uid": "Pipeline/job_0001",
                "type": "Pipeline",
                "status": "completed",
                "created": "2026-01-01T00:00:00",
                "completed": "2026-01-01T01:00:00",
                "parent_jobs": [],
                "alias": None,
                "description": "",
            }
        ],
    }
    (project_dir / "project.json").write_text(json.dumps(project_json, indent=2))

    # Pipeline/job_0001
    job_dir = project_dir / "Pipeline" / "job_0001"
    model_dir = job_dir / "model"
    volumes_dir = job_dir / "output" / "volumes"
    model_dir.mkdir(parents=True)
    volumes_dir.mkdir(parents=True)

    # job.json
    job_json = {
        "recovar_version": "1.0.0",
        "git_commit": "abc1234",
        "command": "pipeline",
        "command_line": "python -m recovar pipeline --particles test.star",
        "parameters": {
            "particles": "test.star",
            "zdim": [2, 4],
        },
        "provenance": {"pipeline_result_dir": None},
        "timing": {
            "started_at": "2026-01-01T00:00:00",
            "completed_at": "2026-01-01T01:00:00",
            "duration_seconds": 3600.0,
        },
        "environment": {"hostname": "test-node"},
        "status": "completed",
        "outputs": {
            "volumes": ["output/volumes/mean.mrc"],
        },
    }
    (job_dir / "job.json").write_text(json.dumps(job_json, indent=2))

    # metadata.json
    metadata = {
        "n_particles": 50000,
        "image_shape": [128, 128],
        "pixel_size": 1.5,
        "zdim_values": [2, 4],
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Marker files
    (model_dir / "params.pkl").touch()
    (volumes_dir / "mean.mrc").touch()

    return project_dir


@pytest.fixture
def tmp_project_with_legacy(tmp_path: Path) -> Path:
    """Create a project directory with a legacy pipeline output (no job.json).

    Layout::

        test_project/
        └── Pipeline/
            └── job_0001/
                └── model/
                    └── params.pkl  (no metadata.json, no job.json)
    """
    project_dir = tmp_path / "test_project"
    job_dir = project_dir / "Pipeline" / "job_0001" / "model"
    job_dir.mkdir(parents=True)
    (job_dir / "params.pkl").touch()
    return project_dir

"""Tests for the subset .star export endpoint.

Covers:
    - POST /api/subsets/:id/export-star (success)
    - POST /api/subsets/:id/export-star (error cases)
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from recovar.gui_v2.backend.db import close_all
from recovar.gui_v2.backend.main import create_app


@pytest.fixture
def app():
    return create_app()


@pytest_asyncio.fixture
async def client(app):
    from recovar.gui_v2.backend.api.project import _project_registry

    _project_registry.clear()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    await close_all()


def _write_test_star(path: str, n_particles: int = 100) -> None:
    """Write a minimal RELION 3.1 .star file for testing."""
    with open(path, "w") as f:
        f.write("# Test star file\n\n")
        f.write("data_optics\n\n")
        f.write("loop_\n")
        f.write("_rlnOpticsGroup\n")
        f.write("_rlnImagePixelSize\n")
        f.write("_rlnImageSize\n")
        f.write("1 1.5 128\n")
        f.write("\n\n")
        f.write("data_particles\n\n")
        f.write("loop_\n")
        f.write("_rlnImageName\n")
        f.write("_rlnDefocusU\n")
        f.write("_rlnDefocusV\n")
        for i in range(n_particles):
            f.write(f"{i+1}@particles.mrcs {10000 + i} {10000 + i}\n")


class TestStarExport:
    @pytest.mark.asyncio
    async def test_export_star_success(
        self, client: AsyncClient, tmp_path: Path
    ):
        """Create a subset, export as .star, verify filtered content."""
        project_dir = str(tmp_path / "star_export_project")

        resp = await client.post(
            "/api/projects",
            json={"path": project_dir, "name": "Star Export"},
        )
        assert resp.status_code == 201
        project_id = resp.json()["id"]

        star_path = str(tmp_path / "particles.star")
        _write_test_star(star_path, n_particles=50)

        indices = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
        resp = await client.post(
            "/api/subsets",
            json={
                "project_id": project_id,
                "name": "test_subset",
                "indices": indices,
            },
        )
        assert resp.status_code == 201
        subset_id = resp.json()["id"]

        resp = await client.post(
            f"/api/subsets/{subset_id}/export-star",
            json={"particles_star": star_path},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_particles"] == 10
        assert data["path"].endswith(".star")
        assert os.path.exists(data["path"])

        from recovar.data_io.starfile import read_star

        exported_df, exported_optics = read_star(data["path"])
        assert len(exported_df) == 10
        assert exported_optics is not None

    @pytest.mark.asyncio
    async def test_export_star_missing_file(
        self, client: AsyncClient, tmp_path: Path
    ):
        """Export with non-existent star file returns 400."""
        project_dir = str(tmp_path / "missing_star_project")
        resp = await client.post(
            "/api/projects",
            json={"path": project_dir, "name": "Missing Star"},
        )
        project_id = resp.json()["id"]

        resp = await client.post(
            "/api/subsets",
            json={
                "project_id": project_id,
                "name": "test_subset2",
                "indices": [0, 1, 2],
            },
        )
        subset_id = resp.json()["id"]

        resp = await client.post(
            f"/api/subsets/{subset_id}/export-star",
            json={"particles_star": "/nonexistent/particles.star"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_export_star_not_star_file(
        self, client: AsyncClient, tmp_path: Path
    ):
        """Export with a non-.star file returns 400."""
        project_dir = str(tmp_path / "not_star_project")
        resp = await client.post(
            "/api/projects",
            json={"path": project_dir, "name": "Not Star"},
        )
        project_id = resp.json()["id"]

        resp = await client.post(
            "/api/subsets",
            json={
                "project_id": project_id,
                "name": "test_subset3",
                "indices": [0, 1],
            },
        )
        subset_id = resp.json()["id"]

        cs_path = str(tmp_path / "particles.cs")
        Path(cs_path).touch()

        resp = await client.post(
            f"/api/subsets/{subset_id}/export-star",
            json={"particles_star": cs_path},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_export_star_index_out_of_range(
        self, client: AsyncClient, tmp_path: Path
    ):
        """Export with indices exceeding star file particle count returns 400."""
        project_dir = str(tmp_path / "oob_project")
        resp = await client.post(
            "/api/projects",
            json={"path": project_dir, "name": "OOB"},
        )
        project_id = resp.json()["id"]

        star_path = str(tmp_path / "small.star")
        _write_test_star(star_path, n_particles=10)

        resp = await client.post(
            "/api/subsets",
            json={
                "project_id": project_id,
                "name": "oob_subset",
                "indices": [0, 5, 99],
            },
        )
        subset_id = resp.json()["id"]

        resp = await client.post(
            f"/api/subsets/{subset_id}/export-star",
            json={"particles_star": star_path},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_export_star_subset_not_found(
        self, client: AsyncClient
    ):
        """Export for non-existent subset returns 404."""
        resp = await client.post(
            "/api/subsets/nonexistent-id/export-star",
            json={"particles_star": "/data/particles.star"},
        )
        assert resp.status_code == 404

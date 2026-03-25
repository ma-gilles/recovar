"""Tests for recovar.gui.app Flask routes.

Uses Flask's test client to verify API endpoints without running a real server.
"""

import json
import os
import re

import pytest

from recovar.gui.app import create_app, _next_clone_name, _safe_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app_with_job(tmp_path):
    """Create a Flask test app with one completed job."""
    state_dir = str(tmp_path / "state")
    os.makedirs(state_dir)

    # Create a fake pipeline output
    output_dir = tmp_path / "output" / "test_run"
    vol_dir = output_dir / "output" / "volumes"
    vol_dir.mkdir(parents=True)
    (vol_dir / "mean.mrc").write_bytes(b"\x00" * 100)

    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "params.pkl").touch()

    with open(output_dir / "metadata.json", "w") as f:
        json.dump({"grid_size": 128}, f)

    app = create_app(
        scan_dirs=[str(tmp_path / "output")],
        state_dir=state_dir,
    )
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app_with_job):
    return app_with_job.test_client()


def _get_job_id(client):
    """Extract the first real job ID from the dashboard HTML."""
    resp = client.get("/")
    # Find job links, skip /jobs/new
    for match in re.finditer(rb'href="/jobs/([^"]+)"', resp.data):
        jid = match.group(1).decode()
        if jid != "new":
            return jid
    return None


# ---------------------------------------------------------------------------
# _safe_path
# ---------------------------------------------------------------------------


class TestSafePath:
    def test_valid_path(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        assert _safe_path(str(f)) == str(f)

    def test_rejects_empty(self):
        assert _safe_path("") is None

    def test_rejects_relative(self):
        assert _safe_path("../etc/passwd") is None

    def test_resolves_absolute(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        result = _safe_path(str(f))
        assert result is not None
        assert os.path.isabs(result)


# ---------------------------------------------------------------------------
# _next_clone_name
# ---------------------------------------------------------------------------


class TestNextCloneName:
    def test_adds_v2(self):
        assert _next_clone_name("my_run") == "my_run_v2"

    def test_increments_version(self):
        assert _next_clone_name("my_run_v2") == "my_run_v3"
        assert _next_clone_name("my_run_v10") == "my_run_v11"

    def test_empty_string(self):
        result = _next_clone_name("")
        assert "v2" in result


# ---------------------------------------------------------------------------
# Dashboard & Job detail pages
# ---------------------------------------------------------------------------


class TestDashboard:
    def test_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_contains_job_name(self, client):
        resp = client.get("/")
        assert b"test_run" in resp.data


class TestJobDetail:
    def test_returns_200(self, client):
        job_id = _get_job_id(client)
        assert job_id is not None
        resp = client.get(f"/jobs/{job_id}")
        assert resp.status_code == 200

    def test_nonexistent_job_redirects(self, client):
        resp = client.get("/jobs/nonexistent", follow_redirects=False)
        assert resp.status_code in (302, 404)


# ---------------------------------------------------------------------------
# API: Logs (returns HTML)
# ---------------------------------------------------------------------------


class TestAPILogs:
    def test_returns_html(self, client):
        job_id = _get_job_id(client)
        resp = client.get(f"/api/jobs/{job_id}/logs")
        assert resp.status_code == 200
        assert b"<pre" in resp.data

    def test_nonexistent_job_returns_empty_html(self, client):
        resp = client.get("/api/jobs/nonexistent/logs")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# API: Status (returns HTML)
# ---------------------------------------------------------------------------


class TestAPIStatus:
    def test_returns_html(self, client):
        job_id = _get_job_id(client)
        resp = client.get(f"/api/jobs/{job_id}/status")
        assert resp.status_code == 200
        # Completed jobs should have a status badge
        assert b"COMPLETED" in resp.data

    def test_nonexistent_returns_empty(self, client):
        resp = client.get("/api/jobs/nonexistent/status")
        assert resp.status_code == 200
        assert resp.data == b""


# ---------------------------------------------------------------------------
# API: Analysis (returns JSON)
# ---------------------------------------------------------------------------


class TestAPIAnalysis:
    def test_get_analysis(self, client):
        job_id = _get_job_id(client)
        resp = client.get(f"/api/jobs/{job_id}/analysis")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "volumes" in data
        assert "has_model" in data
        assert data["has_model"] is True

    def test_nonexistent_job(self, client):
        resp = client.get("/api/jobs/nonexistent/analysis")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "error" in data


# ---------------------------------------------------------------------------
# API: System info (returns JSON)
# ---------------------------------------------------------------------------


class TestAPISystem:
    def test_returns_system_info(self, client):
        resp = client.get("/api/system")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "hostname" in data
        assert "has_slurm" in data
        assert "disk_free_gb" in data


# ---------------------------------------------------------------------------
# API: Browse (returns JSON)
# ---------------------------------------------------------------------------


class TestAPIBrowse:
    def test_browse_home(self, client):
        resp = client.get("/api/browse?path=~")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "entries" in data
        assert "path" in data

    def test_browse_invalid_path(self, client):
        resp = client.get("/api/browse?path=/no/such/dir")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "error" in data


# ---------------------------------------------------------------------------
# API: Volume raw (returns file or 404)
# ---------------------------------------------------------------------------


class TestAPIVolumeRaw:
    def test_missing_path(self, client):
        resp = client.get("/api/volume/raw")
        assert resp.status_code == 404

    def test_nonexistent_volume(self, client):
        resp = client.get("/api/volume/raw?path=/no/such/vol.mrc")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# API: Tasks (returns JSON)
# ---------------------------------------------------------------------------


class TestAPITasks:
    def test_list_tasks(self, client):
        job_id = _get_job_id(client)
        resp = client.get(f"/api/jobs/{job_id}/tasks")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "tasks" in data
        assert isinstance(data["tasks"], list)

    def test_get_nonexistent_task(self, client):
        job_id = _get_job_id(client)
        resp = client.get(f"/api/jobs/{job_id}/tasks/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Debug Molstar page
# ---------------------------------------------------------------------------


class TestDebugMolstar:
    def test_returns_200(self, client):
        resp = client.get("/debug/molstar")
        assert resp.status_code == 200
        assert b"Mol" in resp.data  # Page mentions Mol*

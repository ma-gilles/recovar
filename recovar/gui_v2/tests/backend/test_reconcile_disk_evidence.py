"""Tests for judging a handle-less job by the outputs it left on disk.

A job the GUI imported with "Scan for Jobs", or that was submitted outside
the GUI, has no executor handle. There is nothing to poll for such a job
after a server restart, so its output directory is the only evidence of
what happened — and recovar writes ``job.json`` as the last step of a run.

Covers:
    - disk_completion_time: present / absent / failed / stale job.json
    - reconcile_jobs: handle-less job that finished → completed, not failed
    - reconcile_jobs: handle-less job with nothing to show → still failed
    - reconcile_jobs: executor lost the job (UNKNOWN) but disk says finished
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from recovar.gui_v2.backend.services.executor import (
    Executor,
    JobStatus,
    disk_completion_time,
    reconcile_jobs,
)


def _write_job_json(
    job_dir: Path,
    status: str = "completed",
    completed_at: str | None = None,
) -> Path:
    job_dir.mkdir(parents=True, exist_ok=True)
    payload: dict = {"command": "analyze", "status": status}
    if completed_at:
        payload["timing"] = {"completed_at": completed_at}
    path = job_dir / "job.json"
    path.write_text(json.dumps(payload))
    return path


# ---------------------------------------------------------------- disk probe


class TestDiskCompletionTime:
    def test_none_without_working_dir(self):
        assert disk_completion_time(None) is None

    def test_none_when_job_json_missing(self, tmp_path: Path):
        (tmp_path / "kmeans").mkdir()
        assert disk_completion_time(str(tmp_path)) is None

    def test_prefers_recorded_completion_time(self, tmp_path: Path):
        _write_job_json(tmp_path, completed_at="2026-08-12T09:37:11.500000")
        assert disk_completion_time(str(tmp_path)) == datetime.datetime(
            2026, 8, 12, 9, 37, 11, 500000
        )

    def test_falls_back_to_file_mtime(self, tmp_path: Path):
        path = _write_job_json(tmp_path)  # no timing block
        os.utime(path, (1_760_000_000, 1_760_000_000))
        assert disk_completion_time(str(tmp_path)) == datetime.datetime.utcfromtimestamp(
            1_760_000_000
        )

    def test_none_when_the_run_recorded_failure(self, tmp_path: Path):
        _write_job_json(tmp_path, status="failed")
        assert disk_completion_time(str(tmp_path)) is None

    def test_none_when_job_json_predates_this_run(self, tmp_path: Path):
        """A job.json left by an earlier run in the same directory is ignored."""
        path = _write_job_json(tmp_path)
        old = datetime.datetime(2026, 8, 1, 12, 0, 0)
        stamp = old.replace(tzinfo=datetime.timezone.utc).timestamp()
        os.utime(path, (stamp, stamp))
        started_now = datetime.datetime(2026, 8, 12, 12, 0, 0)
        assert disk_completion_time(str(tmp_path), started_now) is None

    def test_unreadable_job_json_still_counts_as_finished(self, tmp_path: Path):
        (tmp_path / "job.json").write_text("{ not json")
        assert disk_completion_time(str(tmp_path)) is not None


# ------------------------------------------------------------------ reconcile


class TestReconcileWithoutHandle:
    @pytest.mark.asyncio
    async def test_finished_job_is_completed_not_failed(self, tmp_path: Path):
        _write_job_json(tmp_path, completed_at="2026-08-12T09:37:11")
        mock_executor = AsyncMock(spec=Executor)

        updates = await reconcile_jobs(
            mock_executor,
            [
                {
                    "id": "j1",
                    "handle": None,
                    "db_status": "running",
                    "working_dir": str(tmp_path),
                }
            ],
        )

        assert len(updates) == 1
        assert updates[0]["new_status"] == JobStatus.COMPLETED.value
        assert updates[0]["error"] is None
        assert updates[0]["completed_at"] == datetime.datetime(2026, 8, 12, 9, 37, 11)

    @pytest.mark.asyncio
    async def test_job_with_no_outputs_is_still_failed(self, tmp_path: Path):
        mock_executor = AsyncMock(spec=Executor)

        updates = await reconcile_jobs(
            mock_executor,
            [
                {
                    "id": "j1",
                    "handle": None,
                    "db_status": "running",
                    "working_dir": str(tmp_path),
                }
            ],
        )

        assert updates[0]["new_status"] == JobStatus.FAILED.value
        assert "no completed output" in updates[0]["error"]

    @pytest.mark.asyncio
    async def test_run_that_recorded_failure_stays_failed(self, tmp_path: Path):
        _write_job_json(tmp_path, status="failed")
        mock_executor = AsyncMock(spec=Executor)

        updates = await reconcile_jobs(
            mock_executor,
            [
                {
                    "id": "j1",
                    "handle": None,
                    "db_status": "running",
                    "working_dir": str(tmp_path),
                }
            ],
        )

        assert updates[0]["new_status"] == JobStatus.FAILED.value


class TestReconcileUnknownStatus:
    @pytest.mark.asyncio
    async def test_unknown_but_finished_on_disk_is_completed(self, tmp_path: Path):
        """SLURM purging its accounting record must not turn a finished job red."""
        _write_job_json(tmp_path, completed_at="2026-08-12T09:37:11")
        mock_executor = AsyncMock(spec=Executor)
        mock_executor.status.return_value = JobStatus.UNKNOWN
        mock_executor.log_path.return_value = Path("/tmp/some.log")

        updates = await reconcile_jobs(
            mock_executor,
            [
                {
                    "id": "j1",
                    "handle": "12345",
                    "db_status": "running",
                    "working_dir": str(tmp_path),
                }
            ],
        )

        assert updates[0]["new_status"] == JobStatus.COMPLETED.value
        assert updates[0]["error"] is None

    @pytest.mark.asyncio
    async def test_unknown_with_no_outputs_still_fails(self, tmp_path: Path):
        mock_executor = AsyncMock(spec=Executor)
        mock_executor.status.return_value = JobStatus.UNKNOWN
        mock_executor.log_path.return_value = Path("/tmp/some.log")

        updates = await reconcile_jobs(
            mock_executor,
            [
                {
                    "id": "j1",
                    "handle": "12345",
                    "db_status": "running",
                    "working_dir": str(tmp_path),
                }
            ],
        )

        assert updates[0]["new_status"] == JobStatus.FAILED.value
        assert "unknown after server restart" in updates[0]["error"]

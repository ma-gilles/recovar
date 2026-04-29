"""Tests for the executor abstraction (Phase 1 Step 2).

Covers:
    - LocalExecutor: submit, status, cancel, cleanup
    - SlurmExecutor: sbatch template rendering
    - reconcile_jobs: reconnect-after-restart logic
    - Mock-based SLURM status polling
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from recovar.gui_v2.backend.services.executor import (
    Executor,
    JobStatus,
    LocalExecutor,
    SlurmExecutor,
    _render_sbatch_script,
    reconcile_jobs,
)

# ------------------------------------------------------------------
# LocalExecutor
# ------------------------------------------------------------------


class TestLocalExecutor:
    @pytest.mark.asyncio
    async def test_submit_and_complete(self, tmp_path: Path):
        """Submit a simple command that exits 0 → COMPLETED."""
        executor = LocalExecutor()
        handle = await executor.submit(
            job_id="test-001",
            command=[sys.executable, "-c", "print('hello')"],
            env={"PYTHONNOUSERSITE": "1"},
            working_dir=str(tmp_path),
        )
        assert handle  # PID string

        # Wait for completion
        for _ in range(50):
            st = await executor.status(handle)
            if st != JobStatus.RUNNING:
                break
            await asyncio.sleep(0.1)

        assert await executor.status(handle) == JobStatus.COMPLETED

        # Log file should contain output
        log = await executor.log_path(handle)
        assert log is not None
        assert log.exists()
        assert "hello" in log.read_text()

        await executor.cleanup(handle)

    @pytest.mark.asyncio
    async def test_submit_failure(self, tmp_path: Path):
        """Submit a command that exits non-zero → FAILED."""
        executor = LocalExecutor()
        handle = await executor.submit(
            job_id="test-fail",
            command=[sys.executable, "-c", "raise SystemExit(1)"],
            env={},
            working_dir=str(tmp_path),
        )

        for _ in range(50):
            st = await executor.status(handle)
            if st != JobStatus.RUNNING:
                break
            await asyncio.sleep(0.1)

        assert await executor.status(handle) == JobStatus.FAILED
        await executor.cleanup(handle)

    @pytest.mark.asyncio
    async def test_cancel(self, tmp_path: Path):
        """Cancel a long-running job → CANCELLED."""
        executor = LocalExecutor()
        handle = await executor.submit(
            job_id="test-cancel",
            command=[sys.executable, "-c", "import time; time.sleep(300)"],
            env={},
            working_dir=str(tmp_path),
        )

        # Should be running
        await asyncio.sleep(0.3)
        assert await executor.status(handle) == JobStatus.RUNNING

        # Cancel (override the 10s grace period for test speed)
        entry = executor._processes.get(handle)
        assert entry is not None
        proc, _log, pgid = entry
        try:
            os.killpg(pgid, 15)  # SIGTERM
        except OSError:
            pass
        await asyncio.sleep(0.5)

        st = await executor.status(handle)
        assert st in (JobStatus.CANCELLED, JobStatus.FAILED)
        await executor.cleanup(handle)

    @pytest.mark.asyncio
    async def test_status_unknown_after_cleanup(self, tmp_path: Path):
        executor = LocalExecutor()
        assert await executor.status("nonexistent") == JobStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_log_path(self, tmp_path: Path):
        executor = LocalExecutor()
        handle = await executor.submit(
            job_id="test-log",
            command=[sys.executable, "-c", "print('log test')"],
            env={},
            working_dir=str(tmp_path),
        )
        log = await executor.log_path(handle)
        assert log is not None
        assert str(log).endswith("run.log")

        # Nonexistent handle
        assert await executor.log_path("nope") is None
        await executor.cleanup(handle)


# ------------------------------------------------------------------
# SlurmExecutor (mock-based — no real SLURM on test nodes)
# ------------------------------------------------------------------


class TestSlurmExecutor:
    @pytest.mark.asyncio
    async def test_submit_calls_sbatch(self, tmp_path: Path):
        """Submit renders script + calls sbatch --parsable."""
        executor = SlurmExecutor()

        async def mock_create_subprocess_exec(*args, **kwargs):
            m = AsyncMock()
            m.communicate = AsyncMock(return_value=(b"12345\n", b""))
            m.returncode = 0
            return m

        with patch("asyncio.create_subprocess_exec", side_effect=mock_create_subprocess_exec):
            handle = await executor.submit(
                job_id="slurm-test",
                command=["python", "-m", "recovar", "pipeline", "--particles", "test.star"],
                env={"FOO": "bar"},
                working_dir=str(tmp_path),
            )

        assert handle == "12345"
        # submit.sh should have been written
        assert (tmp_path / "submit.sh").exists()
        content = (tmp_path / "submit.sh").read_text()
        assert "recovar" in content
        assert "#SBATCH" in content

    @pytest.mark.asyncio
    async def test_submit_sbatch_failure(self, tmp_path: Path):
        executor = SlurmExecutor()

        async def mock_fail(*args, **kwargs):
            m = AsyncMock()
            m.communicate = AsyncMock(return_value=(b"", b"sbatch: error\n"))
            m.returncode = 1
            return m

        with patch("asyncio.create_subprocess_exec", side_effect=mock_fail):
            with pytest.raises(RuntimeError, match="sbatch failed"):
                await executor.submit(
                    job_id="fail",
                    command=["echo"],
                    env={},
                    working_dir=str(tmp_path),
                )

    @pytest.mark.asyncio
    async def test_status_squeue_running(self):
        """squeue returns RUNNING → JobStatus.RUNNING."""
        executor = SlurmExecutor()

        async def mock_exec(*args, **kwargs):
            m = AsyncMock()
            m.communicate = AsyncMock(return_value=(b"RUNNING\n", b""))
            m.returncode = 0
            return m

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            st = await executor.status("12345")
        assert st == JobStatus.RUNNING

    @pytest.mark.asyncio
    async def test_status_sacct_completed(self):
        """squeue empty, sacct COMPLETED → JobStatus.COMPLETED."""
        executor = SlurmExecutor()
        call_count = 0

        async def mock_exec(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            m = AsyncMock()
            if call_count == 1:
                # squeue returns empty (job not in queue)
                m.communicate = AsyncMock(return_value=(b"", b""))
            else:
                # sacct returns COMPLETED
                m.communicate = AsyncMock(return_value=(b"COMPLETED\n", b""))
            m.returncode = 0
            return m

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            st = await executor.status("12345")
        assert st == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_status_timeout(self):
        """squeue returns TIMEOUT → JobStatus.FAILED."""
        executor = SlurmExecutor()
        call_count = 0

        async def mock_exec(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            m = AsyncMock()
            if call_count == 1:
                m.communicate = AsyncMock(return_value=(b"", b""))
            else:
                m.communicate = AsyncMock(return_value=(b"TIMEOUT\n", b""))
            m.returncode = 0
            return m

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            st = await executor.status("12345")
        assert st == JobStatus.FAILED

    @pytest.mark.asyncio
    async def test_status_cancelled(self):
        executor = SlurmExecutor()
        call_count = 0

        async def mock_exec(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            m = AsyncMock()
            if call_count == 1:
                m.communicate = AsyncMock(return_value=(b"", b""))
            else:
                m.communicate = AsyncMock(return_value=(b"CANCELLED+\n", b""))
            m.returncode = 0
            return m

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            st = await executor.status("12345")
        assert st == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_calls_scancel(self):
        executor = SlurmExecutor()

        async def mock_exec(*args, **kwargs):
            m = AsyncMock()
            m.communicate = AsyncMock(return_value=(b"", b""))
            m.returncode = 0
            return m

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec) as mock:
            await executor.cancel("12345")

        # Check scancel was called
        assert any("scancel" in str(call) for call in mock.call_args_list)


# ------------------------------------------------------------------
# sbatch template
# ------------------------------------------------------------------


class TestSbatchTemplate:
    def test_default_template(self):
        script = _render_sbatch_script(
            job_name="test-job",
            command="python -m recovar pipeline --particles test.star",
            env_vars={"MY_VAR": "my_val"},
            output_path="/tmp/slurm-%j.out",
        )
        assert "#!/bin/bash" in script
        assert "#SBATCH --job-name=test-job" in script
        # No partition/account given → directives must be omitted (not blank).
        assert "#SBATCH --partition" not in script
        assert "#SBATCH --account" not in script
        assert "#SBATCH --gres=gpu:1" in script
        assert "export PYTHONNOUSERSITE=1" in script
        assert "export XLA_PYTHON_CLIENT_PREALLOCATE=false" in script
        assert "export MY_VAR=my_val" in script
        assert "python -m recovar pipeline --particles test.star" in script

    def test_no_site_specific_strings_in_default_render(self):
        """v1.0.0 invariant: default render must not leak any HARDCODED
        Princeton identifier. The interpreter path (sys.executable) is
        dynamic and reflects whatever filesystem the user installed into,
        so we mock it to a neutral path — otherwise this test would falsely
        fail on developers whose home dir happens to contain "GILLES" or
        "princeton" purely by accident of installation location."""
        with patch("recovar.gui_v2.backend.services.executor.sys") as mock_sys:
            mock_sys.executable = "/usr/local/bin/python"
            script = _render_sbatch_script(
                job_name="default",
                command="echo hi",
                env_vars={},
                output_path="/tmp/out.log",
            )
        lowered = script.lower()
        for needle in ("cryoem", "amits", "gilles", "princeton", "/scratch/gpfs"):
            assert needle not in lowered, (
                f"Default sbatch render leaks hardcoded site-specific string {needle!r}.\nRendered script:\n{script}"
            )

    def test_empty_partition_account_omits_directives(self):
        script = _render_sbatch_script(
            job_name="empty",
            command="true",
            env_vars={},
            output_path="/tmp/out.log",
            partition="",
            account="",
        )
        # Empty `--partition=` is a parse error on some SLURM versions; we
        # must omit the directive line entirely.
        assert "#SBATCH --partition=" not in script
        assert "#SBATCH --partition\n" not in script
        assert "#SBATCH --account=" not in script
        assert "#SBATCH --account\n" not in script

    def test_with_cache_dir(self):
        script = _render_sbatch_script(
            job_name="cache-test",
            command="echo hello",
            env_vars={},
            output_path="/tmp/out.log",
            cache_dir="/dev/shm",
        )
        # shlex.quote leaves shell-safe paths unquoted; either form is valid.
        assert "RECOVAR_CACHE_DIR=/dev/shm/recovar_cache_${SLURM_JOB_ID}" in script
        assert "trap _recovar_cleanup EXIT TERM INT" in script

    def test_tmpdir_block_prefers_slurm_provided(self):
        """SLURM_TMPDIR must be checked before falling back to mktemp; we
        never delete a scheduler-provided dir."""
        script = _render_sbatch_script(
            job_name="tmpdir",
            command="true",
            env_vars={},
            output_path="/tmp/out.log",
        )
        # Order: SLURM_TMPDIR first, then $TMPDIR, then mktemp last.
        idx_slurm = script.find("SLURM_TMPDIR")
        idx_mktemp = script.find("mktemp -d")
        assert idx_slurm != -1, "SLURM_TMPDIR not referenced in tmpdir block"
        assert idx_mktemp != -1, "mktemp fallback missing"
        assert idx_slurm < idx_mktemp, "SLURM_TMPDIR must be checked before mktemp"
        # Cleanup must guard on RECOVAR_CREATED_TMPDIR so we only rm dirs we made.
        assert "RECOVAR_CREATED_TMPDIR" in script

    def test_command_with_special_chars_is_quoted(self):
        """If a path contains spaces, the rendered script must still execute
        correctly. Caller is responsible for shlex.join — we assert that the
        rendered output preserves whatever the caller passed."""
        import shlex

        argv = ["recovar", "pipeline", "/path with spaces/foo.cs"]
        joined = shlex.join(argv)
        script = _render_sbatch_script(
            job_name="quoted",
            command=joined,
            env_vars={},
            output_path="/tmp/out.log",
        )
        assert joined in script
        assert "'/path with spaces/foo.cs'" in script

    def test_env_var_values_are_quoted(self):
        script = _render_sbatch_script(
            job_name="qenv",
            command="true",
            env_vars={"PATH_WITH_SPACES": "/a b/c", "SIMPLE": "ok"},
            output_path="/tmp/out.log",
        )
        assert "export PATH_WITH_SPACES='/a b/c'" in script
        assert "export SIMPLE=ok" in script

    def test_brace_format_safety_in_command(self):
        """Commands containing literal `${VAR}` and `${X:-default}` must not
        crash the renderer (.format) and must be preserved verbatim."""
        script = _render_sbatch_script(
            job_name="braces",
            command='echo "${HOME}" "${UNSET:-default}" awk \'{print $1}\'',
            env_vars={},
            output_path="/tmp/out.log",
        )
        assert "${HOME}" in script
        assert "${UNSET:-default}" in script
        assert "awk '{print $1}'" in script

    def test_custom_slurm_opts(self):
        script = _render_sbatch_script(
            job_name="custom",
            command="echo",
            env_vars={},
            output_path="/tmp/out.log",
            partition="gpu",
            account="myacct",
            gpus=2,
            cpus=8,
            memory="500G",
            time="24:00:00",
        )
        assert "#SBATCH --partition=gpu" in script
        assert "#SBATCH --account=myacct" in script
        assert "#SBATCH --gres=gpu:2" in script
        assert "#SBATCH --cpus-per-task=8" in script
        assert "#SBATCH --mem=500G" in script
        assert "#SBATCH --time=24:00:00" in script

    def test_raw_directives_with_short_flag(self):
        """`-p gpu` must render as `#SBATCH -p gpu`, not `#SBATCH ---p gpu`."""
        script = _render_sbatch_script(
            job_name="raw",
            command="true",
            env_vars={},
            output_path="/tmp/out.log",
            raw_directives="-p gpu\n--qos=high\nmail-user=me@example.com",
        )
        assert "#SBATCH -p gpu" in script
        assert "#SBATCH ---p gpu" not in script
        assert "#SBATCH --qos=high" in script
        assert "#SBATCH --mail-user=me@example.com" in script


# ------------------------------------------------------------------
# Reconnect / reconcile
# ------------------------------------------------------------------


class TestReconcileJobs:
    @pytest.mark.asyncio
    async def test_running_stays_running(self):
        """Job that's still RUNNING in executor → no update needed."""
        mock_executor = AsyncMock(spec=Executor)
        mock_executor.status.return_value = JobStatus.RUNNING

        updates = await reconcile_jobs(
            mock_executor,
            [
                {"id": "j1", "handle": "12345", "db_status": "running"},
            ],
        )
        assert len(updates) == 0

    @pytest.mark.asyncio
    async def test_completed_while_down(self):
        """Job completed while server was down → update to COMPLETED."""
        mock_executor = AsyncMock(spec=Executor)
        mock_executor.status.return_value = JobStatus.COMPLETED

        updates = await reconcile_jobs(
            mock_executor,
            [
                {"id": "j1", "handle": "12345", "db_status": "running"},
            ],
        )
        assert len(updates) == 1
        assert updates[0]["new_status"] == "completed"
        assert updates[0]["error"] is None

    @pytest.mark.asyncio
    async def test_failed_while_down(self):
        mock_executor = AsyncMock(spec=Executor)
        mock_executor.status.return_value = JobStatus.FAILED

        updates = await reconcile_jobs(
            mock_executor,
            [
                {"id": "j1", "handle": "12345", "db_status": "running"},
            ],
        )
        assert len(updates) == 1
        assert updates[0]["new_status"] == "failed"
        assert updates[0]["error"] is not None

    @pytest.mark.asyncio
    async def test_unknown_becomes_failed(self):
        """UNKNOWN status → mark as FAILED with descriptive error."""
        mock_executor = AsyncMock(spec=Executor)
        mock_executor.status.return_value = JobStatus.UNKNOWN
        mock_executor.log_path.return_value = Path("/tmp/some.log")

        updates = await reconcile_jobs(
            mock_executor,
            [
                {"id": "j1", "handle": "12345", "db_status": "running"},
            ],
        )
        assert len(updates) == 1
        assert updates[0]["new_status"] == "failed"
        assert "unknown after server restart" in updates[0]["error"]

    @pytest.mark.asyncio
    async def test_no_handle(self):
        """Job with no handle → FAILED with error."""
        mock_executor = AsyncMock(spec=Executor)

        updates = await reconcile_jobs(
            mock_executor,
            [
                {"id": "j1", "handle": None, "db_status": "running"},
            ],
        )
        assert len(updates) == 1
        assert updates[0]["new_status"] == "failed"

    @pytest.mark.asyncio
    async def test_multiple_jobs(self):
        """Reconcile multiple jobs at once."""
        mock_executor = AsyncMock(spec=Executor)
        mock_executor.status.side_effect = [
            JobStatus.COMPLETED,
            JobStatus.RUNNING,
            JobStatus.FAILED,
        ]

        updates = await reconcile_jobs(
            mock_executor,
            [
                {"id": "j1", "handle": "100", "db_status": "running"},
                {"id": "j2", "handle": "200", "db_status": "running"},
                {"id": "j3", "handle": "300", "db_status": "queued"},
            ],
        )
        # j1: running→completed = update
        # j2: running→running = no update
        # j3: queued→failed = update
        assert len(updates) == 2
        ids = {u["id"] for u in updates}
        assert "j1" in ids
        assert "j3" in ids

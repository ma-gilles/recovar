"""CPU batch sizing reads the job's memory limit, not only the node's RAM (recovar.utils.helpers)."""

import pytest

from recovar.utils import helpers

pytestmark = pytest.mark.unit

GB = 10**9


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_cgroup_v2_job_limit_on_an_ancestor_caps_the_node_ram(tmp_path):
    # Slurm: the job cgroup carries memory.max; the step and task cgroups below it say "max".
    root = tmp_path / "cg"
    job = root / "system.slice/slurmstepd.scope/job_1"
    _write(job / "memory.max", str(96 * GB))
    _write(job / "memory.current", str(6 * GB))
    step = job / "step_0/user/task_0"
    _write(step / "memory.max", "max")
    _write(step / "memory.current", str(5 * GB))
    _write(tmp_path / "proc_cgroup", "0::/system.slice/slurmstepd.scope/job_1/step_0/user/task_0\n")

    available = helpers.host_memory_available_bytes(
        cgroup_root=root, proc_cgroup=tmp_path / "proc_cgroup", environ={}, node_available=800 * GB
    )
    assert available == 90 * GB


def test_cgroup_v1_memory_controller(tmp_path):
    root = tmp_path / "cg"
    job = root / "memory/slurm/uid_1/job_2"
    _write(job / "memory.limit_in_bytes", str(64 * GB))
    _write(job / "memory.usage_in_bytes", str(4 * GB))
    _write(root / "memory/memory.limit_in_bytes", str(1 << 62))  # v1 "no limit"
    _write(root / "memory/memory.usage_in_bytes", str(100 * GB))
    _write(tmp_path / "proc_cgroup", "12:memory:/slurm/uid_1/job_2\n4:cpu,cpuacct:/slurm/uid_1/job_2\n")

    available = helpers.host_memory_available_bytes(
        cgroup_root=root, proc_cgroup=tmp_path / "proc_cgroup", environ={}, node_available=800 * GB
    )
    assert available == 60 * GB


def test_slurm_mem_per_node_without_a_readable_cgroup(tmp_path, monkeypatch):
    class _Process:
        def __init__(self, _pid):
            pass

        def memory_info(self):
            class _Info:
                rss = 2 * 1024 * 1024 * 1024

            return _Info()

    monkeypatch.setattr(helpers.psutil, "Process", _Process)
    available = helpers.host_memory_available_bytes(
        cgroup_root=tmp_path / "missing",
        proc_cgroup=tmp_path / "missing_proc",
        environ={"SLURM_MEM_PER_NODE": "98304"},
        node_available=800 * GB,
    )
    assert available == (98304 - 2048) * 1024 * 1024


def test_no_job_limit_uses_the_node_ram_and_the_node_ram_caps_a_loose_limit(tmp_path):
    missing = dict(cgroup_root=tmp_path / "missing", proc_cgroup=tmp_path / "missing_proc", environ={})
    assert helpers.host_memory_available_bytes(node_available=12 * GB, **missing) == 12 * GB
    root = tmp_path / "cg"
    _write(root / "job/memory.max", str(500 * GB))
    _write(root / "job/memory.current", "0")
    _write(tmp_path / "proc_cgroup", "0::/job\n")
    assert (
        helpers.host_memory_available_bytes(
            cgroup_root=root, proc_cgroup=tmp_path / "proc_cgroup", environ={}, node_available=12 * GB
        )
        == 12 * GB
    )


def test_cpu_batch_limit_is_half_the_job_limit(monkeypatch):
    monkeypatch.setattr(helpers, "GPU_MEMORY_LIMIT", None)
    monkeypatch.setattr(helpers, "jax_has_gpu", lambda: False)
    monkeypatch.setattr(helpers, "host_memory_available_bytes", lambda: 90 * GB)
    assert helpers.get_gpu_memory_total() == 45

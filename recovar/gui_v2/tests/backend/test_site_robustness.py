"""Site probing, preflight checks and measured defaults.

These cover the failures seen when the GUI met clusters it had no built-in
knowledge of: a CPU-only default partition, GPU nodes smaller than the default
memory request, data on a mount point outside a hard-coded list, a per-user
process cap on the login node, and a CUDA kernel compute nodes cannot read.
"""

from __future__ import annotations

import pytest

from recovar.gui_v2.backend.api import files
from recovar.gui_v2.backend.services import preflight, project_config, site
from recovar.gui_v2.backend.services.site import CudaKernel, Partition

# Default partition is CPU-only; GPU nodes in two partitions, one down.
SINFO_CPU_DEFAULT = """\
gen*|(null)|768000+|up
gen*|(null)|1024000|up
gpu|gpu:a100:4|1000000|up
gpu|gpu:h100:8|2000000+|up
preempt|gpu:a100:4|512000|down
"""

# Default partition has GPUs but nodes are smaller than the 400G built-in request.
SINFO_SMALL_NODES = """\
main*|gpu:rtx:4(S:0-1)|385000|up
bigmem|(null)|1500000|up
"""


_real_detection = site._detect_slurm_defaults.__wrapped__


def _use_partitions(monkeypatch, sinfo_output: str, accepts=lambda partition, memory, account="": None) -> None:
    """Fake this site's partitions; *accepts* plays the scheduler's dry-run verdict."""
    monkeypatch.setattr(site, "slurm_partitions", lambda: tuple(site.parse_sinfo(sinfo_output)))
    monkeypatch.setattr(
        site,
        "sbatch_test",
        lambda partition, memory, account="": (accepts(partition, memory, account), "scheduler says no"),
    )
    monkeypatch.setattr(site, "detected_slurm_defaults", _real_detection)


class TestSlurmDetection:
    def test_parse_merges_node_groups(self):
        parts = {p.name: p for p in site.parse_sinfo(SINFO_CPU_DEFAULT)}
        assert parts["gen"] == Partition("gen", is_default=True, gpus=False, up=True, max_node_memory_mb=0)
        assert parts["gpu"].gpus and parts["gpu"].max_node_memory_mb == 2000000
        assert not parts["preempt"].up

    def test_gpu_partition_chosen_when_default_has_none(self, monkeypatch):
        _use_partitions(monkeypatch, SINFO_CPU_DEFAULT)
        assert site.detected_slurm_defaults("400G") == {"partition": "gpu"}

    def test_memory_lowered_to_fit_largest_gpu_node(self, monkeypatch):
        _use_partitions(monkeypatch, SINFO_SMALL_NODES)
        detected = site.detected_slurm_defaults("400G")
        assert "partition" not in detected  # default partition already has GPUs
        assert detected["memory"] == "337G"  # 90% of 375 GiB

    def test_site_that_routes_gpu_jobs_itself_needs_no_partition(self, monkeypatch):
        _use_partitions(monkeypatch, SINFO_CPU_DEFAULT, accepts=lambda partition, memory, account="": True)
        assert site.detected_slurm_defaults("400G") == {}

    def test_partition_the_user_cannot_use_is_skipped(self, monkeypatch):
        sinfo = "cpu*|(null)|190000|up\nailab|gpu:h200:8|1500000|up\ncryoem|gpu:a100:4|1000000|up\n"
        _use_partitions(monkeypatch, sinfo, accepts=lambda partition, memory, account="": partition == "cryoem")
        assert site.detected_slurm_defaults("400G") == {"partition": "cryoem"}

    def test_account_is_found_when_the_site_requires_one(self, monkeypatch):
        """A user with several accounts must name one; jobs without it are refused."""
        _use_partitions(
            monkeypatch,
            SINFO_CPU_DEFAULT,
            accepts=lambda partition, memory, account="": account == "lab_b" and partition == "gpu",
        )
        monkeypatch.setattr(site, "slurm_accounts", lambda: ("lab_a", "lab_b"))
        assert site.detected_slurm_defaults("400G") == {"account": "lab_b", "partition": "gpu"}

    def test_nothing_detected_when_every_partition_is_rejected(self, monkeypatch):
        _use_partitions(monkeypatch, SINFO_CPU_DEFAULT, accepts=lambda partition, memory, account="": False)
        assert site.detected_slurm_defaults("400G") == {}

    def test_nothing_detected_without_slurm(self, monkeypatch):
        _use_partitions(monkeypatch, "")
        assert site.detected_slurm_defaults("400G") == {}

    def test_user_setting_beats_detected(self, monkeypatch, tmp_path):
        _use_partitions(monkeypatch, SINFO_CPU_DEFAULT)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        assert project_config.resolve_slurm_defaults()["partition"] == "gpu"
        project_config.save_user_slurm_defaults({"partition": "preempt"})
        layered = project_config.resolve_slurm_defaults_layered()
        assert layered["detected"] == {"partition": "gpu"}
        assert layered["effective"]["partition"] == "preempt"

    @pytest.mark.parametrize("spec, gb", [("400G", 400), ("64000M", 62.5), ("1T", 1024), ("2048", 2), ("16GB", 16)])
    def test_memory_units(self, spec, gb):
        assert site._memory_gb(spec) == gb


class TestProcessLimit:
    def test_no_cap_when_limit_is_ample(self):
        assert site.thread_budget(None, 128) is None
        assert site.thread_budget(100000, 128) is None

    def test_cap_on_restricted_login_node(self):
        assert site.thread_budget(256, 128) == 8
        assert site.thread_budget(64, 96) == 2

    def test_small_machine_under_default_ulimit_is_left_alone(self):
        assert site.thread_budget(4096, 16) is None


class TestFilesystems:
    MOUNTS = (
        ("/mnt/fuseceph", "fuse"),
        ("/run/user/1000/doc", "fuse.portal"),
        ("/mnt/ceph/users", "ceph"),
        ("/scratch/gpfs", "gpfs"),
        ("/mnt/ceph", "ceph"),
        ("/dev/shm", "tmpfs"),
        ("/mnt/home", "nfs4"),
        ("/hpc", "autofs"),
        ("/hpc/projects", "autofs"),
        ("/hpc/projects/group.a", "lustre"),
        ("/misc", "autofs"),
        ("/tmp", "xfs"),
        ("/", "xfs"),
    )

    @pytest.fixture(autouse=True)
    def _fake_mounts(self, monkeypatch):
        ordered = tuple(sorted(self.MOUNTS, key=lambda m: len(m[0]), reverse=True))
        monkeypatch.setattr(site, "_mounts", lambda: ordered)

    def test_shared_roots_are_outermost_network_mounts(self):
        # /hpc: storage is automounted on demand, so the automount root is the root.
        # /misc: an automount with nothing shared mounted under it is ignored.
        assert site.shared_filesystem_roots() == ["/hpc", "/mnt/ceph", "/mnt/fuseceph", "/mnt/home", "/scratch/gpfs"]

    def test_shared_vs_node_local(self):
        assert site.is_shared_path("/mnt/ceph/users/me/data.star") is True
        assert site.is_shared_path("/dev/shm/recovar/lib.so") is False
        assert site.is_shared_path("/tmp/x") is None  # local disk: cannot tell

    def test_browser_allows_measured_mounts(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        monkeypatch.setenv(files.ALLOWED_ROOTS_ENV, "/opt/lab_data")
        files.configure_allowed_roots(files.default_allowed_roots())
        files._check_path_allowed("/mnt/ceph/users/me/particles.star")
        files._check_path_allowed("/opt/lab_data/x")
        with pytest.raises(Exception) as exc:
            files._check_path_allowed("/etc/passwd")
        assert exc.value.status_code == 403
        assert "--allow-root" in exc.value.detail

    def test_configured_roots_must_be_a_list(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        (tmp_path / "recovar").mkdir()
        (tmp_path / "recovar" / "config.toml").write_text('[gui]\nallowed_roots = "/data"\n')
        with pytest.raises(ValueError, match="must be a list"):
            project_config.configured_allowed_roots()


class TestPreflight:
    @pytest.fixture
    def on_cluster(self, monkeypatch):
        monkeypatch.setattr(preflight.shutil, "which", lambda cmd: f"/usr/bin/{cmd}" if cmd == "sbatch" else None)

    def _kernel(self, monkeypatch, **kwargs):
        kwargs.setdefault("package_path", "/env/site-packages/recovar/cuda/libcuda_backproject.so")
        monkeypatch.setattr(site, "cuda_kernel", lambda: CudaKernel(**kwargs))

    def test_kernel_on_node_local_storage_is_an_error(self, monkeypatch, on_cluster):
        self._kernel(monkeypatch, path="/dev/shm/cache/libcuda_backproject.so")
        monkeypatch.setattr(site, "is_shared_path", lambda p: False)
        monkeypatch.setattr(site, "fs_type", lambda p: "tmpfs")
        check = preflight._check_cuda_kernel()
        assert check.level == preflight.ERROR
        assert check.fix == "recovar build_custom_cuda --output /env/site-packages/recovar/cuda/libcuda_backproject.so"

    def test_missing_kernel_without_compiler_says_how_to_get_one(self, monkeypatch, on_cluster):
        self._kernel(monkeypatch, searched=["/home/me/.cache/recovar/cuda/libcuda_backproject.so"])
        monkeypatch.setattr(site, "is_shared_path", lambda p: True)
        check = preflight._check_cuda_kernel()
        assert check.level == preflight.ERROR
        assert "module avail cuda" in check.fix and "--output" not in check.fix

    def test_built_kernel_passes(self, monkeypatch, on_cluster):
        self._kernel(monkeypatch, path="/home/me/.cache/recovar/cuda/libcuda_backproject.so")
        monkeypatch.setattr(site, "is_shared_path", lambda p: True)
        assert preflight._check_cuda_kernel().level == preflight.OK

    def test_cpu_only_partition_setting_is_an_error(self, monkeypatch, on_cluster, tmp_path):
        _use_partitions(monkeypatch, SINFO_CPU_DEFAULT)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        assert preflight._check_scheduler().level == preflight.OK  # detected `gpu`
        project_config.save_user_slurm_defaults({"partition": "gen"})
        check = preflight._check_scheduler()
        assert check.level == preflight.ERROR and "`gpu`" in check.fix

    def test_scheduler_refusal_is_reported_with_its_own_message(self, monkeypatch, on_cluster, tmp_path):
        _use_partitions(monkeypatch, SINFO_CPU_DEFAULT, accepts=lambda partition, memory, account="": False)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        check = preflight._check_scheduler()
        assert check.level == preflight.ERROR and "scheduler says no" in check.detail

    def test_self_routing_site_with_cpu_default_partition_passes(self, monkeypatch, on_cluster, tmp_path):
        _use_partitions(monkeypatch, SINFO_CPU_DEFAULT, accepts=lambda partition, memory, account="": True)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        assert preflight._check_scheduler().level == preflight.OK

    def test_uncapped_server_on_restricted_node_is_an_error(self, monkeypatch):
        monkeypatch.setattr(site, "process_limit", lambda: 256)
        monkeypatch.setattr(preflight.os, "cpu_count", lambda: 128)
        monkeypatch.delenv(preflight.CORE_CAP_ENV, raising=False)
        assert preflight._check_process_limit().level == preflight.ERROR
        monkeypatch.setenv(preflight.CORE_CAP_ENV, "8")
        assert preflight._check_process_limit().level == preflight.OK


@pytest.mark.asyncio
async def test_preflight_endpoint_and_project_root(client, tmp_project_dir, monkeypatch):
    monkeypatch.setattr(site, "cuda_kernel", lambda: CudaKernel(probe_error="not probed in tests"))
    resp = await client.get("/api/system/preflight")
    assert resp.status_code == 200
    assert {c["id"] for c in resp.json()["checks"]} >= {"process_limit", "scheduler", "cuda_kernel"}

    files.configure_allowed_roots([])
    resp = await client.post("/api/projects", json={"path": str(tmp_project_dir), "name": "p"})
    assert resp.status_code == 201
    resp = await client.get("/api/files/browse", params={"path": str(tmp_project_dir)})
    assert resp.status_code == 200  # a project's own directory is always browsable


@pytest.mark.asyncio
async def test_unwritable_project_path_is_a_client_error(client):
    resp = await client.post("/api/projects", json={"path": "/proc/recovar_nope/x", "name": "p"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_subset_rerun_reuses_pipeline_inputs_and_original_indices(client, tmp_project_dir):
    """Lasso subset -> pipeline form values: same inputs as the source pipeline,
    restricted by the selection, whose indices already address the full dataset
    (embeddings are stored in original-file order)."""
    import pickle

    import numpy as np

    from recovar.gui_v2.backend.config import get_db_path
    from recovar.gui_v2.backend.db import init_db
    from recovar.gui_v2.backend.models.job import Job

    project = (await client.post("/api/projects", json={"path": str(tmp_project_dir), "name": "p"})).json()
    source_ind = tmp_project_dir / "half.pkl"
    source_ind.write_bytes(pickle.dumps(np.arange(100, 200)))  # pipeline analyzed particles 100..199
    pipeline_dir = str(tmp_project_dir / "Pipeline" / "job_0001")
    pipeline_params = {
        "particles": "/data/particles.mrcs",
        "poses": "/data/poses.pkl",
        "ctf": "/data/ctf.pkl",
        "mask": "sphere",
        "zdim": "4",
        "ind": str(source_ind),
        "output_name": "first",
        "slurm_opts": {"memory": "1G"},
    }
    session_factory = await init_db(get_db_path(tmp_project_dir))
    async with session_factory() as session:
        pipeline = Job(
            project_id=project["id"],
            type="Pipeline",
            status="completed",
            params=pipeline_params,
            output_dir=pipeline_dir,
        )
        analyze = Job(
            project_id=project["id"],
            type="Analyze",
            status="completed",
            params={"result_dir": pipeline_dir},
            output_dir=str(tmp_project_dir / "Analyze" / "job_0001"),
        )
        session.add_all([pipeline, analyze])
        await session.commit()
        analyze_id = analyze.id

    subset = (
        await client.post(
            "/api/subsets",
            json={
                "project_id": project["id"],
                "name": "lasso",
                "source_job_id": analyze_id,
                "zdim": 4,
                "method": {"tool": "lasso"},
                "indices": [100, 105, 199],
            },
        )
    ).json()
    resp = await client.get(f"/api/subsets/{subset['id']}/pipeline-params")
    assert resp.status_code == 200, resp.text
    params = resp.json()["params"]
    assert {k: params[k] for k in ("particles", "poses", "ctf", "mask", "zdim")} == {
        "particles": "/data/particles.mrcs",
        "poses": "/data/poses.pkl",
        "ctf": "/data/ctf.pkl",
        "mask": "sphere",
        "zdim": "4",
    }
    assert "output_name" not in params and "slurm_opts" not in params
    with open(params["ind"], "rb") as f:
        assert list(pickle.load(f)) == [100, 105, 199]

    # A particle the source pipeline never analyzed cannot be part of a selection from it.
    stray = (
        await client.post(
            "/api/subsets",
            json={"project_id": project["id"], "name": "stray", "source_job_id": analyze_id, "indices": [5, 150]},
        )
    ).json()
    resp = await client.get(f"/api/subsets/{stray['id']}/pipeline-params")
    assert resp.status_code == 400 and "not analyzed" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_job_imported_while_running_is_not_declared_failed(tmp_path):
    """A job Scan imported mid-run has no executor handle. On restart it follows
    what the job itself recorded, instead of being marked failed."""
    import json

    from recovar.gui_v2.backend.services.executor import LocalExecutor, reconcile_jobs

    job = {"id": "j1", "handle": None, "db_status": "running", "working_dir": str(tmp_path)}
    (tmp_path / "job.json").write_text(json.dumps({"status": "running"}))
    assert await reconcile_jobs(LocalExecutor(), [job]) == []
    (tmp_path / "job.json").write_text(json.dumps({"status": "completed"}))
    assert await reconcile_jobs(LocalExecutor(), [job]) == [{"id": "j1", "new_status": "completed", "error": None}]


@pytest.mark.asyncio
async def test_slurm_options_and_dry_run_endpoints(client, monkeypatch):
    _use_partitions(monkeypatch, SINFO_CPU_DEFAULT, accepts=lambda partition, memory, account="": partition == "gpu")
    monkeypatch.setattr(site, "slurm_accounts", lambda: ("lab_a",))

    options = (await client.get("/api/system/slurm-options")).json()
    assert options["detected"] == {"partition": "gpu"}
    assert options["accounts"] == ["lab_a"]
    assert {p["name"] for p in options["partitions"] if p["gpus"]} == {"gpu", "preempt"}

    ok = (await client.post("/api/system/slurm-test", json={"partition": "gpu", "memory": "64G"})).json()
    assert ok["accepted"] is True
    bad = (await client.post("/api/system/slurm-test", json={"partition": "gen", "memory": "64G"})).json()
    assert bad == {"accepted": False, "message": "scheduler says no"}


def test_blanking_a_saved_setting_clears_it(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    project_config.save_user_slurm_defaults({"partition": "old", "account": "lab_a"})
    project_config.save_user_slurm_defaults({"partition": "", "account": "lab_a"})
    assert project_config.resolve_slurm_defaults_layered()["user"] == {"account": "lab_a"}


def test_available_zdims_from_per_zdim_model_directories(tmp_path):
    """Pipeline outputs store each embedding under model/zdim_<N>/."""
    from recovar.gui_v2.backend.api.jobs import _read_available_zdims

    for name in ("zdim_20", "zdim_4", "zdim_abc"):
        (tmp_path / "model" / name).mkdir(parents=True)
    assert _read_available_zdims(str(tmp_path)) == [4, 20]


def test_failed_job_reports_the_exception_from_its_log(tmp_path):
    from recovar.gui_v2.backend.api.jobs import _last_error_line

    (tmp_path / "slurm-123.out").write_text(
        "INFO starting\nTraceback (most recent call last):\n"
        '  File "analyze.py", line 174, in analyze\n'
        '    raise ValueError("Requested zdim was not found in embedding outputs.")\n'
        "ValueError: Requested zdim was not found in embedding outputs.\n"
    )
    assert _last_error_line(str(tmp_path), "123") == (
        "Job failed: ValueError: Requested zdim was not found in embedding outputs."
    )
    assert _last_error_line(str(tmp_path), "999") is None  # no log for that handle


@pytest.mark.asyncio
async def test_browser_walks_down_from_root_to_allowed_directories(client, tmp_path):
    """Directories above the allowed roots list only what leads to a root."""
    data = tmp_path / "lab" / "data"
    data.mkdir(parents=True)
    (tmp_path / "lab" / "private").mkdir()
    (data / "particles.star").write_text("x")
    files.configure_allowed_roots([str(data)])

    above = (await client.get("/api/files/browse", params={"path": str(tmp_path / "lab")})).json()
    assert [e["name"] for e in above] == ["data"]  # "private" is not exposed
    top = await client.get("/api/files/browse", params={"path": "/"})
    assert top.status_code == 200 and all(e["is_dir"] for e in top.json())
    inside = (await client.get("/api/files/browse", params={"path": str(data)})).json()
    assert [e["name"] for e in inside] == ["particles.star"]
    outside = await client.get("/api/files/browse", params={"path": str(tmp_path / "lab" / "private")})
    assert outside.status_code == 403


@pytest.mark.asyncio
async def test_missing_chart_data_is_empty_not_an_error(client, tmp_project_dir):
    from recovar.gui_v2.backend.config import get_db_path
    from recovar.gui_v2.backend.db import init_db
    from recovar.gui_v2.backend.models.job import Job

    project = (await client.post("/api/projects", json={"path": str(tmp_project_dir), "name": "p"})).json()
    session_factory = await init_db(get_db_path(tmp_project_dir))
    async with session_factory() as session:
        job = Job(
            project_id=project["id"], type="Pipeline", status="completed", params={}, output_dir=str(tmp_project_dir)
        )
        session.add(job)
        await session.commit()
        job_id = job.id
    resp = await client.get(f"/api/jobs/{job_id}/chart-data", params={"name": "fsc"})
    assert resp.status_code == 200 and resp.json()["traces"] == []


@pytest.mark.asyncio
async def test_subset_from_another_projects_job_is_a_client_error(client, tmp_project_dir):
    project = (await client.post("/api/projects", json={"path": str(tmp_project_dir), "name": "p"})).json()
    resp = await client.post(
        "/api/subsets",
        json={
            "project_id": project["id"],
            "name": "s",
            "source_job_id": "job-of-another-project",
            "indices": [1, 2],
        },
    )
    assert resp.status_code == 400 and "different project" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_removing_a_job_keeps_the_subsets_selected_from_it(client, tmp_project_dir):
    from recovar.gui_v2.backend.config import get_db_path
    from recovar.gui_v2.backend.db import init_db
    from recovar.gui_v2.backend.models.job import Job

    project = (await client.post("/api/projects", json={"path": str(tmp_project_dir), "name": "p"})).json()
    session_factory = await init_db(get_db_path(tmp_project_dir))
    async with session_factory() as session:
        job = Job(
            project_id=project["id"], type="Analyze", status="completed", params={}, output_dir=str(tmp_project_dir)
        )
        session.add(job)
        await session.commit()
        job_id = job.id
    created = await client.post(
        "/api/subsets",
        json={
            "project_id": project["id"],
            "name": "s",
            "source_job_id": job_id,
            "indices": [1, 2],
        },
    )
    assert created.status_code in (200, 201), created.text
    assert (await client.delete(f"/api/jobs/{job_id}")).status_code == 204
    subsets = (await client.get("/api/subsets", params={"project_id": project["id"]})).json()
    assert [s["name"] for s in subsets] == ["s"] and subsets[0]["source_job_id"] is None


@pytest.mark.asyncio
async def test_unreadable_directory_is_forbidden_not_a_server_error(client, tmp_path, monkeypatch):
    from recovar.gui_v2.backend.api import files

    locked = tmp_path / "locked"
    locked.mkdir()
    files.add_allowed_root(str(tmp_path))
    locked.chmod(0)
    try:
        resp = await client.get("/api/files/browse", params={"path": str(locked)})
    finally:
        locked.chmod(0o700)
    assert resp.status_code == 403 and "permission" in resp.json()["detail"]


def test_explorer_shows_only_analyzed_particles_with_their_original_indices(tmp_path):
    """--ind pipelines store NaN rows for unused particles; those are not in the dataset."""
    import pickle

    import numpy as np

    from recovar.gui_v2.backend.api.embeddings import _load_embeddings_sync

    analyzed = np.array([1, 4, 5, 8])
    latent = np.full((10, 2), np.nan, dtype=np.float32)
    latent[analyzed] = np.arange(8, dtype=np.float32).reshape(4, 2)
    (tmp_path / "model" / "zdim_2").mkdir(parents=True)
    np.save(tmp_path / "model" / "zdim_2" / "latent_coords.npy", latent)
    analysis = tmp_path / "analysis"
    (analysis / "plots" / "umap").mkdir(parents=True)
    (analysis / "data").mkdir()
    with open(analysis / "plots" / "umap" / "umap_embedding.pkl", "wb") as f:
        pickle.dump(np.arange(20, dtype=np.float32).reshape(10, 2), f)  # padded to stored size
    with open(analysis / "data" / "kmeans_result.pkl", "wb") as f:
        pickle.dump({"labels": np.array([0, 1, 0, 1]), "centers": np.zeros((2, 2))}, f)  # analyzed only

    out = _load_embeddings_sync(str(tmp_path), 2, analyze_dir=str(analysis))
    assert out["n_particles"] == out["n_particles_total"] == 4
    assert out["orig_indices"].tolist() == [1, 4, 5, 8]
    assert np.array_equal(out["pca_coords"], latent[analyzed])
    assert out["umap_coords"][:, 0].tolist() == [2.0, 8.0, 10.0, 16.0]
    assert out["kmeans_labels"].tolist() == [0, 1, 0, 1]

    sub = _load_embeddings_sync(str(tmp_path), 2, analyze_dir=str(analysis), max_particles=2)
    assert sub["subsampled"] and set(sub["orig_indices"].tolist()) <= {1, 4, 5, 8}


@pytest.mark.asyncio
async def test_scan_asks_the_scheduler_about_runs_that_never_recorded_finishing(client, tmp_project_dir, monkeypatch):
    """A killed run leaves ``"status": "running"`` in job.json for ever; import must not trust it."""
    import json
    import shutil
    import socket

    from recovar.gui_v2.backend.api import jobs
    from recovar.gui_v2.backend.services.executor import JobStatus  # what executors really return

    def run_dir(name, environment):
        d = tmp_project_dir / "Pipeline" / name
        (d / "model").mkdir(parents=True)
        (d / "job.json").write_text(
            json.dumps({"command": "pipeline", "status": "running", "environment": environment})
        )
        return d

    run_dir("job_0001", {"slurm_job_id": "111", "hostname": "node1"})  # scancelled
    run_dir("job_0002", {"slurm_job_id": "222", "hostname": "node1"})  # still running
    run_dir("job_0003", {"slurm_job_id": "333", "hostname": "node1"})  # scheduler forgot it
    dead = run_dir("job_0004", {"hostname": socket.gethostname()})  # local process is gone
    (dead / "run.pid").write_text("999999999")
    run_dir("job_0005", {"hostname": "some-other-machine"})  # cannot be checked from here

    class FakeSlurm:
        async def status(self, handle):
            return {"111": JobStatus.CANCELLED, "222": JobStatus.RUNNING}.get(handle, JobStatus.UNKNOWN)

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(jobs, "_get_slurm_executor", lambda: FakeSlurm())

    project = (await client.post("/api/projects", json={"path": str(tmp_project_dir), "name": "p"})).json()
    resp = await client.post(f"/api/projects/{project['id']}/scan", json={"scan_path": str(tmp_project_dir)})
    assert resp.status_code == 200, resp.text
    status = {j["output_dir"].rsplit("/", 1)[-1]: j["status"] for j in resp.json()["imported"]}
    assert status == {
        "job_0001": "cancelled",
        "job_0002": "running",
        "job_0003": "failed",
        "job_0004": "failed",
        "job_0005": "running",
    }
    detail = (await client.get(f"/api/projects/{project['id']}")).json()["jobs"]
    by_name = {j["output_dir"].rsplit("/", 1)[-1]: j for j in detail}
    running = (await client.get(f"/api/jobs/{by_name['job_0002']['id']}")).json()
    assert running["slurm_id"] == "222"
    interrupted = (await client.get(f"/api/jobs/{by_name['job_0003']['id']}")).json()
    assert "interrupted" in interrupted["error"]


def test_account_is_found_without_dry_running_every_partition(monkeypatch):
    """Each dry-run takes seconds: a site that needs an account must not cost one per partition first."""
    partitions = [site.Partition(name=f"gpu{i}", gpus=True, up=True, max_node_memory_mb=512_000) for i in range(12)]
    asked = []

    def accepts(partition, memory, account=""):
        asked.append((partition, account))
        return account == "lab"

    monkeypatch.setattr(site, "slurm_partitions", lambda: partitions)
    monkeypatch.setattr(site, "slurm_accounts", lambda: ("other", "lab"))
    monkeypatch.setattr(site, "sbatch_accepts", accepts)
    assert _real_detection("64G") == {"account": "lab"}
    assert asked == [("", ""), ("", "other"), ("", "lab")]


@pytest.mark.asyncio
async def test_job_cancelled_with_scancel_is_cancelled_not_unknown(monkeypatch):
    """sacct reports a user's scancel as ``CANCELLED by <uid>``."""
    from recovar.gui_v2.backend.services import executor

    class Proc:
        returncode = 0

        def __init__(self, out):
            self.out = out

        async def communicate(self):
            return self.out, b""

    async def fake_exec(program, *args, **kwargs):
        return Proc(b"" if program == "squeue" else b"CANCELLED by 230216\nCANCELLED\n")

    monkeypatch.setattr(executor.asyncio, "create_subprocess_exec", fake_exec)
    assert (await executor.SlurmExecutor().status("17997518")).value == "cancelled"


def test_preflight_is_not_recomputed_on_every_page_load(monkeypatch):
    from recovar.gui_v2.backend.services import preflight

    runs = []
    monkeypatch.setattr(preflight, "run_checks", lambda: runs.append(1) or [])
    preflight.invalidate()
    preflight.cached_checks()
    preflight.cached_checks()
    assert len(runs) == 1
    preflight.cached_checks(refresh=True)  # "Check again"
    preflight.invalidate()  # settings saved
    preflight.cached_checks()
    assert len(runs) == 3
    preflight.invalidate()


@pytest.mark.asyncio
async def test_settings_reject_values_sbatch_would_reject(client, tmp_project_dir):
    """A bad default must fail when it is saved, with a message, not later at every submission."""
    bad = await client.put(
        "/api/settings/slurm-defaults/project",
        json={"project_dir": str(tmp_project_dir), "memory": "/tmp/whatever", "time": "tomorrow"},
    )
    assert bad.status_code == 422
    messages = " ".join(d["msg"] for d in bad.json()["detail"])
    assert "not a valid SLURM memory" in messages and "not a valid SLURM time" in messages

    good = await client.put(
        "/api/settings/slurm-defaults/project",
        json={
            "project_dir": str(tmp_project_dir),
            "partition": "gpu,cryoem",
            "account": "my-lab",
            "memory": "64g",
            "time": "2-12:00:00",
        },
    )
    assert good.status_code == 200, good.text
    assert good.json()["project"]["time"] == "2-12:00:00"
    cleared = await client.put(
        "/api/settings/slurm-defaults/project", json={"project_dir": str(tmp_project_dir), "time": ""}
    )
    assert cleared.status_code == 200 and "time" not in cleared.json()["project"]

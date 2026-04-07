"""Unit tests for recovar.project (RecovarProject, registry, job_context)."""

import argparse
import json
import os

import pytest

from recovar.project.project import RecovarProject, default_job_alias, find_project_root
from recovar.project.registry import JOB_TYPES, get_job_type

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_all_commands_have_types(self):
        expected = {
            "pipeline",
            "analyze",
            "compute_state",
            "compute_trajectory",
            "estimate_conformational_density",
            "estimate_stable_states",
            "junk_particle_detection",
            "outlier_detection",
            "postprocess",
            "downsample",
            "extract_image_subset",
            "extract_image_subset_from_kmeans",
            "pipeline_with_outliers",
            "reconstruct_from_external_embedding",
        }
        assert expected.issubset(set(JOB_TYPES.keys()))

    def test_get_job_type(self):
        jt = get_job_type("pipeline")
        assert jt.name == "Pipeline"
        assert jt.dir_name == "Pipeline"
        assert jt.produces_volumes is True

    def test_get_job_type_unknown(self):
        assert get_job_type("nonexistent") is None

    def test_dir_names_are_camelcase(self):
        for jt in JOB_TYPES.values():
            assert jt.dir_name[0].isupper(), f"{jt.command} dir_name should be CamelCase"


# ---------------------------------------------------------------------------
# RecovarProject tests
# ---------------------------------------------------------------------------


class TestRecovarProject:
    def test_init_creates_project_json(self, tmp_path):
        proj = RecovarProject.init(str(tmp_path / "myproj"))
        assert os.path.isfile(os.path.join(proj.root, "project.json"))
        with open(os.path.join(proj.root, "project.json")) as f:
            data = json.load(f)
        assert data["version"] == "1.0"
        assert data["name"] == "myproj"

    def test_init_with_name(self, tmp_path):
        proj = RecovarProject.init(str(tmp_path / "p"), name="My Analysis")
        with open(os.path.join(proj.root, "project.json")) as f:
            data = json.load(f)
        assert data["name"] == "My Analysis"

    def test_init_idempotent(self, tmp_path):
        proj1 = RecovarProject.init(str(tmp_path / "p"))
        proj2 = RecovarProject.init(str(tmp_path / "p"))
        assert proj1.root == proj2.root

    def test_exists(self, tmp_path):
        proj = RecovarProject(str(tmp_path / "nope"))
        assert not proj.exists
        proj = RecovarProject.init(str(tmp_path / "yes"))
        assert proj.exists

    def test_allocate_job_first(self, tmp_path):
        proj = RecovarProject.init(str(tmp_path / "p"))
        uid, job_dir = proj.allocate_job("compute_state")
        assert uid == "ReconstructState/job_0001"
        assert os.path.isdir(job_dir)
        assert job_dir.endswith("ReconstructState/job_0001")

    def test_allocate_job_increments(self, tmp_path):
        proj = RecovarProject.init(str(tmp_path / "p"))
        uid1, _ = proj.allocate_job("compute_state")
        uid2, _ = proj.allocate_job("compute_state")
        uid3, _ = proj.allocate_job("compute_state")
        assert uid1 == "ReconstructState/job_0001"
        assert uid2 == "ReconstructState/job_0002"
        assert uid3 == "ReconstructState/job_0003"

    def test_allocate_job_per_type(self, tmp_path):
        proj = RecovarProject.init(str(tmp_path / "p"))
        uid_p, _ = proj.allocate_job("pipeline")
        uid_a, _ = proj.allocate_job("analyze")
        uid_p2, _ = proj.allocate_job("pipeline")
        assert uid_p == "Pipeline/job_0001"
        assert uid_a == "Analyze/job_0001"
        assert uid_p2 == "Pipeline/job_0002"

    def test_register_and_list_jobs(self, tmp_path):
        proj = RecovarProject.init(str(tmp_path / "p"))
        uid, _ = proj.allocate_job("pipeline")
        proj.register_job_start(uid, "pipeline", "recovar pipeline ...")
        jobs = proj.list_jobs()
        assert len(jobs) == 1
        assert jobs[0]["uid"] == uid
        assert jobs[0]["status"] == "running"

    def test_register_complete(self, tmp_path):
        proj = RecovarProject.init(str(tmp_path / "p"))
        uid, _ = proj.allocate_job("pipeline")
        proj.register_job_start(uid, "pipeline")
        proj.register_job_complete(uid, "completed")
        jobs = proj.list_jobs()
        assert jobs[0]["status"] == "completed"
        assert jobs[0]["completed"] is not None

    def test_find_latest_job(self, tmp_path):
        proj = RecovarProject.init(str(tmp_path / "p"))
        uid1, _ = proj.allocate_job("pipeline")
        proj.register_job_start(uid1, "pipeline")
        proj.register_job_complete(uid1, "completed")
        uid2, _ = proj.allocate_job("pipeline")
        proj.register_job_start(uid2, "pipeline")
        proj.register_job_complete(uid2, "completed")

        latest = proj.find_latest_job("Pipeline")
        assert latest == "Pipeline/job_0002"

    def test_find_latest_job_none(self, tmp_path):
        proj = RecovarProject.init(str(tmp_path / "p"))
        assert proj.find_latest_job("Pipeline") is None

    def test_resolve_pipeline_absolute(self, tmp_path):
        proj = RecovarProject.init(str(tmp_path / "p"))
        abs_path = str(tmp_path / "external_pipeline")
        os.makedirs(abs_path)
        resolved = proj.resolve_pipeline(abs_path)
        assert resolved == abs_path

    def test_resolve_pipeline_relative(self, tmp_path):
        proj = RecovarProject.init(str(tmp_path / "p"))
        uid, job_dir = proj.allocate_job("pipeline")
        resolved = proj.resolve_pipeline("Pipeline/job_0001")
        assert resolved == job_dir

    def test_resolve_pipeline_auto_latest(self, tmp_path):
        proj = RecovarProject.init(str(tmp_path / "p"))
        uid, job_dir = proj.allocate_job("pipeline")
        proj.register_job_start(uid, "pipeline")
        proj.register_job_complete(uid, "completed")
        resolved = proj.resolve_pipeline(None)
        assert resolved == job_dir

    def test_register_job_start_records_unique_alias(self, tmp_path):
        proj = RecovarProject.init(str(tmp_path / "p"))
        uid1, _ = proj.allocate_job("pipeline")
        proj.register_job_start(uid1, "pipeline", alias="ribosome")
        uid2, _ = proj.allocate_job("pipeline")
        proj.register_job_start(uid2, "pipeline", alias="ribosome")
        jobs = {job["uid"]: job for job in proj.list_jobs()}
        assert jobs[uid1]["alias"] == "ribosome"
        assert jobs[uid2]["alias"] == "ribosome_2"
        assert proj.get_job_alias_map() == {uid1: "ribosome", uid2: "ribosome_2"}

    def test_resolve_pipeline_alias(self, tmp_path):
        proj = RecovarProject.init(str(tmp_path / "p"))
        uid, job_dir = proj.allocate_job("pipeline")
        proj.register_job_start(uid, "pipeline", alias="ribosome_d128")
        proj.register_job_complete(uid, "completed")
        assert proj.resolve_pipeline("ribosome_d128") == job_dir

    def test_default_job_alias_prefers_explicit_output_name(self):
        assert default_job_alias("analyze", {"output_name": "My Embedding"}) == "my_embedding"

    def test_parent_jobs_tracking(self, tmp_path):
        proj = RecovarProject.init(str(tmp_path / "p"))
        uid_p, _ = proj.allocate_job("pipeline")
        proj.register_job_start(uid_p, "pipeline")
        proj.register_job_complete(uid_p, "completed")

        uid_a, _ = proj.allocate_job("analyze")
        proj.register_job_start(uid_a, "analyze", parent_jobs=[uid_p])
        jobs = proj.list_jobs("Analyze")
        assert jobs[0]["parent_jobs"] == [uid_p]


# ---------------------------------------------------------------------------
# find_project_root tests
# ---------------------------------------------------------------------------


class TestFindProjectRoot:
    def test_finds_in_current_dir(self, tmp_path):
        (tmp_path / "project.json").write_text("{}")
        result = find_project_root(str(tmp_path))
        assert result == str(tmp_path)

    def test_finds_in_parent(self, tmp_path):
        (tmp_path / "project.json").write_text("{}")
        subdir = tmp_path / "Pipeline" / "job_0001"
        subdir.mkdir(parents=True)
        result = find_project_root(str(subdir))
        assert result == str(tmp_path)

    def test_returns_none_when_not_found(self, tmp_path):
        result = find_project_root(str(tmp_path))
        assert result is None


# ---------------------------------------------------------------------------
# job_context tests
# ---------------------------------------------------------------------------


class TestJobContext:
    def test_project_mode_creates_job_dir(self, tmp_path):
        from recovar.project.job_context import job_context

        proj = RecovarProject.init(str(tmp_path / "p"))
        args = argparse.Namespace(
            project=str(tmp_path / "p"),
            outdir=None,
            result_dir=None,
        )
        with job_context(args, "pipeline") as ctx:
            assert ctx.uid == "Pipeline/job_0001"
            assert os.path.isdir(ctx.output_dir)
            assert ctx.project is not None

        # After context exits, job should be completed
        jobs = proj.list_jobs()
        assert any(j["uid"] == "Pipeline/job_0001" and j["status"] == "completed" for j in jobs)

    def test_standalone_mode_with_explicit_outdir(self, tmp_path):
        from recovar.project.job_context import job_context

        args = argparse.Namespace(
            project=None,
            outdir=str(tmp_path / "my_output"),
            result_dir=None,
        )
        with job_context(args, "pipeline") as ctx:
            assert ctx.output_dir == str(tmp_path / "my_output")
            assert ctx.project is None
            assert ctx.uid is None

    def test_failed_job_records_status(self, tmp_path):
        from recovar.project.job_context import job_context

        proj = RecovarProject.init(str(tmp_path / "p"))
        args = argparse.Namespace(
            project=str(tmp_path / "p"),
            outdir=None,
            result_dir=None,
        )
        with pytest.raises(RuntimeError):
            with job_context(args, "pipeline") as ctx:
                raise RuntimeError("test failure")

        jobs = proj.list_jobs()
        assert any(j["uid"] == "Pipeline/job_0001" and j["status"] == "failed" for j in jobs)

    def test_project_mode_resolves_latest_pipeline_when_result_dir_omitted(self, tmp_path):
        from recovar.project.job_context import job_context

        proj = RecovarProject.init(str(tmp_path / "p"))
        pipeline_uid, pipeline_dir = proj.allocate_job("pipeline")
        proj.register_job_start(pipeline_uid, "pipeline", alias="sample_pipeline")
        proj.register_job_complete(pipeline_uid, "completed")

        args = argparse.Namespace(
            project=str(tmp_path / "p"),
            outdir=None,
            output_dir=None,
            output=None,
            result_dir=None,
            recovar_result_dir=None,
            output_name="Embedding K10",
        )
        with job_context(args, "analyze") as ctx:
            assert ctx.pipeline_dir == pipeline_dir
            assert ctx.parent_jobs == [pipeline_uid]
            assert ctx.uid == "Analyze/job_0001"

        jobs = proj.list_jobs("Analyze")
        assert jobs[0]["alias"] == "embedding_k10"

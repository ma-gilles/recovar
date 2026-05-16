"""Tests for discovering jobs nested inside pipeline output directories.

``analyze`` and the other downstream commands write into the pipeline
directory they consume, so the scanner has to descend one level below the
project walk to find them.

Covers:
    - nested Analyze in a flat project layout (``round_1/Analyze/``)
    - nested Analyze in the canonical ``Pipeline/job_0001/`` layout
    - pipeline artifact directories are not mistaken for jobs
    - no duplicates when the project root is itself a pipeline output
    - ``scan_arbitrary_directory`` on a pipeline directory
"""

from __future__ import annotations

import json
from pathlib import Path

from recovar.gui_v2.backend.services.scanner import (
    scan_arbitrary_directory,
    scan_project_directory,
)


def _make_pipeline(pipeline_dir: Path) -> Path:
    """Write the minimum markers that identify a pipeline output."""
    (pipeline_dir / "model").mkdir(parents=True)
    (pipeline_dir / "model" / "metadata.json").write_text(json.dumps({"zdim": 10}))
    (pipeline_dir / "output" / "volumes").mkdir(parents=True)
    return pipeline_dir


def _make_analyze(analyze_dir: Path, status: str = "completed") -> Path:
    """Write the minimum markers that identify an analyze output."""
    analyze_dir.mkdir(parents=True)
    (analyze_dir / "kmeans").mkdir()
    (analyze_dir / "job.json").write_text(
        json.dumps({"command": "analyze", "status": status, "parameters": {"zdim": 10}})
    )
    return analyze_dir


def _by_type(jobs: list) -> dict[str, list]:
    out: dict[str, list] = {}
    for job in jobs:
        out.setdefault(job.type, []).append(job)
    return out


def test_finds_analyze_nested_in_flat_pipeline_dir(tmp_path: Path) -> None:
    """A flat project (``round_1/``, ``round_2/``) exposes its Analyze jobs."""
    project = tmp_path / "project"
    for round_name in ("round_1", "round_2"):
        pipeline = _make_pipeline(project / round_name)
        _make_analyze(pipeline / "Analyze")

    jobs = _by_type(scan_project_directory(str(project)))

    assert len(jobs["Pipeline"]) == 2
    assert len(jobs["Analyze"]) == 2
    analyze_dirs = sorted(job.output_dir for job in jobs["Analyze"])
    assert analyze_dirs == [
        str(project / "round_1" / "Analyze"),
        str(project / "round_2" / "Analyze"),
    ]


def test_nested_analyze_records_its_parent_pipeline(tmp_path: Path) -> None:
    project = tmp_path / "project"
    pipeline = _make_pipeline(project / "round_1")
    _make_analyze(pipeline / "Analyze")

    analyze = _by_type(scan_project_directory(str(project)))["Analyze"][0]

    assert analyze.parent_job_dirs == [str(pipeline)]


def test_finds_analyze_nested_in_canonical_job_dir(tmp_path: Path) -> None:
    """The ``Pipeline/job_0001/`` layout also gets its nested jobs."""
    project = tmp_path / "project"
    pipeline = _make_pipeline(project / "Pipeline" / "job_0001")
    _make_analyze(pipeline / "analysis_0")

    jobs = _by_type(scan_project_directory(str(project)))

    assert len(jobs["Pipeline"]) == 1
    assert [job.output_dir for job in jobs["Analyze"]] == [str(pipeline / "analysis_0")]


def test_pipeline_artifact_dirs_are_not_jobs(tmp_path: Path) -> None:
    """``data/`` + ``plots/`` under a pipeline are artifacts, not an Analyze job."""
    project = tmp_path / "project"
    pipeline = _make_pipeline(project / "round_1")
    (pipeline / "data").mkdir()
    (pipeline / "plots").mkdir()

    jobs = scan_project_directory(str(project))

    assert [job.type for job in jobs] == ["Pipeline"]


def test_no_duplicates_when_project_root_is_the_pipeline(tmp_path: Path) -> None:
    """The root-is-a-pipeline path and the top-level walk must not double-count."""
    project = _make_pipeline(tmp_path / "project")
    _make_analyze(project / "Analyze")

    jobs = scan_project_directory(str(project))

    output_dirs = [job.output_dir for job in jobs]
    assert len(output_dirs) == len(set(output_dirs))
    assert sorted(job.type for job in jobs) == ["Analyze", "Pipeline"]


def test_scan_arbitrary_directory_includes_nested_jobs(tmp_path: Path) -> None:
    """Pointing the scanner straight at a pipeline dir still finds its Analyze."""
    pipeline = _make_pipeline(tmp_path / "some_pipeline_output")
    _make_analyze(pipeline / "Analyze")

    jobs = _by_type(scan_arbitrary_directory(str(pipeline)))

    assert len(jobs["Pipeline"]) == 1
    assert [job.output_dir for job in jobs["Analyze"]] == [str(pipeline / "Analyze")]


def test_nested_analyze_keeps_its_reported_status(tmp_path: Path) -> None:
    """A still-running analyze is imported as running, not silently completed."""
    project = tmp_path / "project"
    pipeline = _make_pipeline(project / "round_2")
    _make_analyze(pipeline / "Analyze", status="running")

    analyze = _by_type(scan_project_directory(str(project)))["Analyze"][0]

    assert analyze.status == "running"

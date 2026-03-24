"""Unit tests for recovar.output.job.JobDir and output_paths.VolumeOutputPaths."""

import json
import os

import pytest

from recovar.output.job import (
    JobDir,
    _args_to_dict,
    _collect_output_manifest,
    _get_recovar_version,
)
from recovar.output.output_paths import VolumeOutputPaths, resolve_volume_diag_path


# ---------------------------------------------------------------------------
# VolumeOutputPaths tests
# ---------------------------------------------------------------------------

class TestVolumeOutputPaths:
    def test_primary_output_paths(self, tmp_path):
        vp = VolumeOutputPaths(str(tmp_path), "center", 0)
        assert vp.stem == "center000"
        assert vp.filtered == os.path.join(str(tmp_path), "center000.mrc")
        assert vp.filtered_noB == os.path.join(str(tmp_path), "center000_noB.mrc")
        assert vp.half1_unfil == os.path.join(str(tmp_path), "center000_half1_unfil.mrc")
        assert vp.half2_unfil == os.path.join(str(tmp_path), "center000_half2_unfil.mrc")
        assert vp.unfil == os.path.join(str(tmp_path), "center000_unfil.mrc")

    def test_diagnostics_paths(self, tmp_path):
        vp = VolumeOutputPaths(str(tmp_path), "state", 5)
        diag = os.path.join(str(tmp_path), "diagnostics", "state005")
        assert vp.diag_dir == diag
        assert vp.params == os.path.join(diag, "params.pkl")
        assert vp.split_choice == os.path.join(diag, "split_choice.pkl")
        assert vp.locres == os.path.join(diag, "local_resolution.mrc")
        assert vp.sampling == os.path.join(diag, "sampling.mrc")
        assert vp.heterogeneity_distances == os.path.join(diag, "heterogeneity_distances.txt")
        assert vp.latent_coords == os.path.join(diag, "latent_coords.txt")

    def test_debug_paths(self, tmp_path):
        vp = VolumeOutputPaths(str(tmp_path), "center", 2)
        diag = vp.diag_dir
        assert vp.filtered_smooth == os.path.join(diag, "filtered_smooth.mrc")
        assert vp.cv_half1_unfil == os.path.join(diag, "CV_estimates_half1_unfil.mrc")
        assert vp.cv_noise_half2 == os.path.join(diag, "CV_noise_half2.mrc")

    def test_estimates_dir(self, tmp_path):
        vp = VolumeOutputPaths(str(tmp_path), "state", 0)
        assert vp.estimates_dir(1) == os.path.join(vp.diag_dir, "estimates_half1_unfil")
        assert vp.estimates_dir(2) == os.path.join(vp.diag_dir, "estimates_half2_unfil")
        assert vp.estimates_dir(1, filtered=True) == os.path.join(vp.diag_dir, "estimates_filt")

    def test_ensure_dirs(self, tmp_path):
        vp = VolumeOutputPaths(str(tmp_path / "out"), "center", 0)
        vp.ensure_dirs()
        assert os.path.isdir(str(tmp_path / "out"))
        assert os.path.isdir(vp.diag_dir)

    def test_different_indices(self, tmp_path):
        vp0 = VolumeOutputPaths(str(tmp_path), "center", 0)
        vp19 = VolumeOutputPaths(str(tmp_path), "center", 19)
        assert vp0.stem == "center000"
        assert vp19.stem == "center019"
        assert vp0.filtered != vp19.filtered


# ---------------------------------------------------------------------------
# resolve_volume_diag_path tests
# ---------------------------------------------------------------------------

class TestResolveVolumeDiagPath:
    def test_new_layout(self, tmp_path):
        """Find file in diagnostics/{stem}/ layout."""
        diag = tmp_path / "diagnostics" / "center000"
        diag.mkdir(parents=True)
        (diag / "params.pkl").write_text("new")

        result = resolve_volume_diag_path(str(tmp_path), "params.pkl", "center", 0)
        assert result == str(diag / "params.pkl")

    def test_flat_fallback(self, tmp_path):
        """Fall back to flat layout when diagnostics/ doesn't exist."""
        (tmp_path / "params.pkl").write_text("old")

        result = resolve_volume_diag_path(str(tmp_path), "params.pkl", "center", 0)
        assert result == str(tmp_path / "params.pkl")

    def test_new_layout_preferred(self, tmp_path):
        """New layout is preferred when both exist."""
        (tmp_path / "params.pkl").write_text("old")
        diag = tmp_path / "diagnostics" / "center000"
        diag.mkdir(parents=True)
        (diag / "params.pkl").write_text("new")

        result = resolve_volume_diag_path(str(tmp_path), "params.pkl", "center", 0)
        assert "diagnostics" in result

    def test_no_prefix(self, tmp_path):
        """Without prefix/index, just use flat path."""
        (tmp_path / "params.pkl").write_text("data")
        result = resolve_volume_diag_path(str(tmp_path), "params.pkl")
        assert result == str(tmp_path / "params.pkl")


# ---------------------------------------------------------------------------
# JobDir tests
# ---------------------------------------------------------------------------

class TestJobDir:
    def test_basic_properties(self, tmp_path):
        job = JobDir(str(tmp_path / "myoutput"), "compute_state")
        assert job.command_name == "compute_state"
        assert job.job_json.endswith("job.json")
        assert job.command_txt.endswith("command.txt")
        assert job.run_log.endswith("run.log")

    def test_ensure_dirs(self, tmp_path):
        job = JobDir(str(tmp_path / "myoutput"), "compute_state")
        job.ensure_dirs()
        assert os.path.isdir(job.root)

    def test_start_and_complete(self, tmp_path):
        job = JobDir(str(tmp_path / "myoutput"), "compute_state")
        job._parent_result_dir = "/some/pipeline"

        # Simulate args
        import argparse
        args = argparse.Namespace(outdir=str(tmp_path / "myoutput"), n_bins=50)

        job.start(args)

        # Check files created
        assert os.path.isfile(job.job_json)
        assert os.path.isfile(job.command_txt)
        assert os.path.isfile(job.run_log)

        # Check job.json content at start
        with open(job.job_json) as f:
            data = json.load(f)
        assert data["command"] == "compute_state"
        assert data["status"] == "running"
        assert "started_at" in data["timing"]
        assert data["recovar_version"] == _get_recovar_version()
        assert data["parameters"]["n_bins"] == 50
        assert data["provenance"]["pipeline_result_dir"] == "/some/pipeline"
        assert "hostname" in data["environment"]

        # Write a fake volume to test output manifest
        with open(os.path.join(job.root, "state000.mrc"), "w") as f:
            f.write("fake")

        job.complete()

        with open(job.job_json) as f:
            data = json.load(f)
        assert data["status"] == "completed"
        assert "completed_at" in data["timing"]
        assert "duration_seconds" in data["timing"]
        assert "state000.mrc" in data["outputs"].get("volumes", [])

    def test_complete_failed(self, tmp_path):
        job = JobDir(str(tmp_path / "out"), "analyze")
        job.start()
        job.complete(status="failed")

        with open(job.job_json) as f:
            data = json.load(f)
        assert data["status"] == "failed"


# ---------------------------------------------------------------------------
# Auto-numbering tests
# ---------------------------------------------------------------------------

class TestAutoNumbering:
    def test_first_job(self, tmp_path):
        result = JobDir._next_numbered_dir(str(tmp_path), "compute_state")
        assert result == os.path.join(str(tmp_path), "compute_state_001")

    def test_sequential_numbering(self, tmp_path):
        (tmp_path / "compute_state_001").mkdir()
        (tmp_path / "compute_state_002").mkdir()
        result = JobDir._next_numbered_dir(str(tmp_path), "compute_state")
        assert result == os.path.join(str(tmp_path), "compute_state_003")

    def test_gap_in_numbering(self, tmp_path):
        """Uses max+1, doesn't fill gaps."""
        (tmp_path / "compute_state_001").mkdir()
        (tmp_path / "compute_state_005").mkdir()
        result = JobDir._next_numbered_dir(str(tmp_path), "compute_state")
        assert result == os.path.join(str(tmp_path), "compute_state_006")

    def test_ignores_other_prefixes(self, tmp_path):
        (tmp_path / "analyze_001").mkdir()
        (tmp_path / "compute_state_003").mkdir()
        result = JobDir._next_numbered_dir(str(tmp_path), "analyze")
        assert result == os.path.join(str(tmp_path), "analyze_002")

    def test_create_with_auto_number(self, tmp_path):
        job = JobDir.create(
            outdir=None,
            command_name="compute_state",
            parent_result_dir=str(tmp_path),
            auto_number=True,
        )
        assert job.root == os.path.join(str(tmp_path), "compute_state_001")

    def test_create_with_explicit_outdir(self, tmp_path):
        job = JobDir.create(
            outdir=str(tmp_path / "my_custom_dir"),
            command_name="compute_state",
        )
        assert job.root == str(tmp_path / "my_custom_dir")

    def test_create_no_outdir_no_auto(self):
        with pytest.raises(ValueError, match="outdir is required"):
            JobDir.create(outdir=None, command_name="test")


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_args_to_dict(self):
        import argparse
        args = argparse.Namespace(
            outdir="/path/to/out",
            n_bins=50,
            _private="hidden",
        )
        d = _args_to_dict(args)
        assert d["outdir"] == "/path/to/out"
        assert d["n_bins"] == 50
        assert "_private" not in d

    def test_args_to_dict_none(self):
        assert _args_to_dict(None) == {}

    def test_collect_output_manifest(self, tmp_path):
        # Create some test files
        (tmp_path / "state000.mrc").write_text("vol")
        (tmp_path / "state000_half1_unfil.mrc").write_text("hm")
        (tmp_path / "job.json").write_text("{}")  # should be excluded
        diag = tmp_path / "diagnostics" / "state000"
        diag.mkdir(parents=True)
        (diag / "params.pkl").write_text("p")
        plots = tmp_path / "plots"
        plots.mkdir()
        (plots / "fsc.png").write_text("img")

        manifest = _collect_output_manifest(str(tmp_path))
        assert "state000.mrc" in manifest["volumes"]
        assert "state000_half1_unfil.mrc" in manifest["halfmaps"]
        assert any("params.pkl" in p for p in manifest["diagnostics"])
        assert any("fsc.png" in p for p in manifest["plots"])
        # job.json should not appear
        for category in manifest.values():
            for path in category:
                assert "job.json" not in path

    def test_get_recovar_version(self):
        v = _get_recovar_version()
        assert v != "unknown"

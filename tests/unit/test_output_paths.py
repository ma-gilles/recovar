"""Tests for recovar.output.output_paths — path management for pipeline output."""

import os

import pytest

from recovar.output.output_paths import (
    AnalysisPaths,
    ResultPaths,
    eigenvector_filename,
    variance_filename,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def test_eigenvector_filename_zero_padded():
    assert eigenvector_filename(0) == "eigen_pos0000.mrc"
    assert eigenvector_filename(5) == "eigen_pos0005.mrc"
    assert eigenvector_filename(123) == "eigen_pos0123.mrc"


def test_variance_filename():
    assert variance_filename(4) == "variance4.mrc"
    assert variance_filename(20) == "variance20.mrc"


# ---------------------------------------------------------------------------
# ResultPaths
# ---------------------------------------------------------------------------


class TestResultPaths:

    def test_directory_properties(self):
        rp = ResultPaths("/out")
        assert rp.model_dir == os.path.join("/out", "model")
        assert rp.output_dir == os.path.join("/out", "output")
        assert rp.volumes_dir == os.path.join("/out", "output", "volumes")
        assert rp.plots_dir == os.path.join("/out", "output", "plots")

    def test_model_file_paths(self):
        rp = ResultPaths("/out")
        assert rp.params.endswith("model/params.pkl")
        assert rp.embeddings.endswith("model/embeddings.pkl")
        assert rp.covariance_cols.endswith("model/covariance_cols.pkl")
        assert rp.halfsets.endswith("model/halfsets.pkl")
        assert rp.metadata.endswith("model/metadata.json")

    def test_volume_file_paths(self):
        rp = ResultPaths("/out")
        assert rp.mean_volume.endswith("volumes/mean.mrc")
        assert rp.mean_filtered.endswith("volumes/mean_filt.mrc")
        assert rp.mask_volume.endswith("volumes/mask.mrc")

    def test_eigenvector_path(self):
        rp = ResultPaths("/out")
        path = rp.eigenvector(3)
        assert path.endswith("volumes/eigen_pos0003.mrc")

    def test_variance_path(self):
        rp = ResultPaths("/out")
        path = rp.variance(10)
        assert path.endswith("volumes/variance10.mrc")

    def test_analysis_dir(self):
        rp = ResultPaths("/out")
        assert rp.analysis_dir(20) == os.path.join("/out", "analysis_20")

    def test_command_txt_and_run_log(self):
        rp = ResultPaths("/out")
        assert rp.command_txt == os.path.join("/out", "command.txt")
        assert rp.run_log == os.path.join("/out", "run.log")

    def test_ensure_dirs(self, tmp_path):
        rp = ResultPaths(str(tmp_path / "pipeline_out"))
        rp.ensure_dirs()
        assert os.path.isdir(rp.model_dir)
        assert os.path.isdir(rp.volumes_dir)
        assert os.path.isdir(rp.plots_dir)

    def test_ensure_model_dir(self, tmp_path):
        rp = ResultPaths(str(tmp_path / "pipeline_out2"))
        rp.ensure_model_dir()
        assert os.path.isdir(rp.model_dir)
        assert not os.path.isdir(rp.volumes_dir)

    def test_ensure_volumes_dir(self, tmp_path):
        rp = ResultPaths(str(tmp_path / "pipeline_out3"))
        rp.ensure_volumes_dir()
        assert os.path.isdir(rp.volumes_dir)


# ---------------------------------------------------------------------------
# AnalysisPaths
# ---------------------------------------------------------------------------


class TestAnalysisPaths:

    def test_kmeans_dir(self):
        ap = AnalysisPaths("/analysis_4")
        assert ap.kmeans_dir == os.path.join("/analysis_4", "kmeans")

    def test_plots_dir(self):
        ap = AnalysisPaths("/analysis_4")
        assert ap.plots_dir == os.path.join("/analysis_4", "plots")

    def test_traj_dir(self):
        ap = AnalysisPaths("/analysis_4")
        assert ap.traj_dir(1) == os.path.join("/analysis_4", "traj001")
        assert ap.traj_dir(10) == os.path.join("/analysis_4", "traj010")

    def test_vol_stem(self):
        assert AnalysisPaths.vol_stem("center", 0) == "center000"
        assert AnalysisPaths.vol_stem("state", 5) == "state005"

    def test_vol_filename(self):
        assert AnalysisPaths.vol_filename("center", 1) == "center001.mrc"

    def test_halfmap_filename(self):
        assert AnalysisPaths.halfmap_filename("center", 1, 1) == "center001_half1_unfil.mrc"
        assert AnalysisPaths.halfmap_filename("center", 1, 2) == "center001_half2_unfil.mrc"

    def test_diagnostics_subdir(self):
        assert AnalysisPaths.diagnostics_subdir("center", 0) == os.path.join(
            "diagnostics", "center000"
        )

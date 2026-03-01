import os
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")

from recovar.commands import outlier_detection as outlier_cmd


pytestmark = pytest.mark.unit


class _FakePipelineOutput:
    def __init__(self, payload):
        self._payload = payload

    def get(self, key):
        return self._payload[key]


def test_outlier_detection_from_contrast_spa_returns_image_indices():
    payload = {
        "input_args": SimpleNamespace(tilt_series=False, shared_contrast_across_tilts=False, particles="particles.mrcs"),
        "contrasts": {4: np.array([0.05, 0.2, 4.0, 0.5], dtype=np.float32)},
        "halfsets": [np.array([10, 11], dtype=np.int32), np.array([12, 13], dtype=np.int32)],
        "particles_halfsets": [np.array([10, 11], dtype=np.int32), np.array([12, 13], dtype=np.int32)],
    }
    po = _FakePipelineOutput(payload)

    image_outliers, image_inliers, particle_outliers, particle_inliers = outlier_cmd.outlier_detection_from_contrast(
        po,
        zdim_key=4,
        low_contrast_threshold=0.1,
        high_contrast_threshold=3.5,
        output_dir=None,
    )

    np.testing.assert_array_equal(image_outliers, np.array([10, 12], dtype=np.int32))
    np.testing.assert_array_equal(image_inliers, np.array([11, 13], dtype=np.int32))
    assert particle_outliers is None
    assert particle_inliers is None


def test_outlier_detection_from_contrast_tilt_shared_contrast_returns_particle_indices():
    payload = {
        "input_args": SimpleNamespace(tilt_series=True, shared_contrast_across_tilts=True, particles="particles.star"),
        "contrasts": {4: np.array([0.05, 0.2, 4.0], dtype=np.float32)},
        "halfsets": [np.array([0, 1, 2], dtype=np.int32), np.array([3, 4, 5], dtype=np.int32)],
        "particles_halfsets": [np.array([30, 31], dtype=np.int32), np.array([32], dtype=np.int32)],
    }
    po = _FakePipelineOutput(payload)

    image_outliers, image_inliers, particle_outliers, particle_inliers = outlier_cmd.outlier_detection_from_contrast(
        po,
        zdim_key=4,
        low_contrast_threshold=0.1,
        high_contrast_threshold=3.5,
        output_dir=None,
    )

    # Current behavior for shared-contrast ET path: image outputs are particle-index arrays.
    np.testing.assert_array_equal(image_outliers, np.array([30, 32], dtype=np.int32))
    np.testing.assert_array_equal(image_inliers, np.array([31], dtype=np.int32))
    assert particle_outliers is None
    assert particle_inliers is None


def test_outlier_detection_main_combines_anomaly_and_contrast_for_spa(monkeypatch, tmp_path):
    outdir = tmp_path / "outlier_output"
    args = SimpleNamespace(
        pipeline_output_dir=str(tmp_path / "pipeline_out"),
        zdim_key=4,
        no_z_regularization=False,
        output_dir=str(outdir),
        save_pipeline_indices=False,
        output_format="both",
        low_contrast_threshold=0.1,
        high_contrast_threshold=3.5,
        max_contrast=4.0,
        particle_bad_fraction_threshold=0.7,
        micrograph_bad_fraction_threshold=0.7,
        use_junk_detection=False,
        junk_threshold=0.5,
        particles_per_cluster=None,
    )

    class _Parser:
        def parse_args(self):
            return args

    monkeypatch.setattr(outlier_cmd, "add_args", lambda _parser: _Parser())

    payload = {
        "latent_coords": {4: np.zeros((4, 2), dtype=np.float32)},
        "input_args": SimpleNamespace(tilt_series=False, particles="particles.mrcs", shared_contrast=False),
        "particles_halfsets": [np.array([10, 11], dtype=np.int32), np.array([12, 13], dtype=np.int32)],
        "halfsets": [np.array([10, 11], dtype=np.int32), np.array([12, 13], dtype=np.int32)],
        "contrasts": {4: np.ones(4, dtype=np.float32)},
    }
    monkeypatch.setattr(outlier_cmd.output, "PipelineOutput", lambda _p: _FakePipelineOutput(payload))

    def fake_plot(_zs, _orig_indices, folder):
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "inliers_consensus.pkl"), "wb") as f:
            pickle.dump(np.array([10, 12, 13], dtype=np.int32), f)
        with open(os.path.join(folder, "outliers_consensus.pkl"), "wb") as f:
            pickle.dump(np.array([11], dtype=np.int32), f)

    monkeypatch.setattr(outlier_cmd, "plot_anomaly_detection_results", fake_plot)
    monkeypatch.setattr(
        outlier_cmd,
        "outlier_detection_from_contrast",
        lambda *_args, **_kwargs: (
            np.array([12], dtype=np.int32),
            np.array([10, 11, 13], dtype=np.int32),
            np.array([13], dtype=np.int32),
            np.array([10, 11, 12], dtype=np.int32),
        ),
    )
    monkeypatch.setattr(outlier_cmd, "create_particle_outlier_visualization", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(outlier_cmd, "create_overlap_matrix_visualization", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(outlier_cmd, "create_outlier_visualizations", lambda *_args, **_kwargs: None)

    outlier_cmd.main()

    combined_dir = outdir / "combined_results"
    with open(combined_dir / "combined_image_outliers_4.pkl", "rb") as f:
        combined_image_outliers = pickle.load(f)
    with open(combined_dir / "combined_particle_outliers_4.pkl", "rb") as f:
        combined_particle_outliers = pickle.load(f)
    with open(combined_dir / "combined_image_inliers_4.pkl", "rb") as f:
        combined_image_inliers = pickle.load(f)

    np.testing.assert_array_equal(combined_image_outliers, np.array([11, 12], dtype=np.int64))
    np.testing.assert_array_equal(combined_particle_outliers, np.array([11, 13], dtype=np.int64))
    np.testing.assert_array_equal(combined_image_inliers, np.array([10, 13], dtype=np.int64))


def test_outlier_detection_main_tilt_maps_particle_outliers_to_images(monkeypatch, tmp_path):
    outdir = tmp_path / "outlier_output_tilt"
    args = SimpleNamespace(
        pipeline_output_dir=str(tmp_path / "pipeline_out"),
        zdim_key=4,
        no_z_regularization=False,
        output_dir=str(outdir),
        save_pipeline_indices=False,
        output_format="both",
        low_contrast_threshold=0.1,
        high_contrast_threshold=3.5,
        max_contrast=4.0,
        particle_bad_fraction_threshold=0.7,
        micrograph_bad_fraction_threshold=0.7,
        use_junk_detection=False,
        junk_threshold=0.5,
        particles_per_cluster=None,
    )

    class _Parser:
        def parse_args(self):
            return args

    monkeypatch.setattr(outlier_cmd, "add_args", lambda _parser: _Parser())

    payload = {
        "latent_coords": {4: np.zeros((3, 2), dtype=np.float32)},
        "input_args": SimpleNamespace(tilt_series=True, particles="particles.star", shared_contrast=False),
        "particles_halfsets": [np.array([100, 101], dtype=np.int32), np.array([102], dtype=np.int32)],
        "halfsets": [np.array([0, 1, 2], dtype=np.int32), np.array([3, 4, 5], dtype=np.int32)],
        "contrasts": None,
    }
    monkeypatch.setattr(outlier_cmd.output, "PipelineOutput", lambda _p: _FakePipelineOutput(payload))

    def fake_plot(_zs, _orig_indices, folder):
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "inliers_consensus.pkl"), "wb") as f:
            pickle.dump(np.array([101], dtype=np.int32), f)
        with open(os.path.join(folder, "outliers_consensus.pkl"), "wb") as f:
            pickle.dump(np.array([100, 102], dtype=np.int32), f)

    monkeypatch.setattr(outlier_cmd, "plot_anomaly_detection_results", fake_plot)
    monkeypatch.setattr(outlier_cmd, "create_particle_outlier_visualization", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(outlier_cmd, "create_overlap_matrix_visualization", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(outlier_cmd, "create_outlier_visualizations", lambda *_args, **_kwargs: None)

    map_calls = {"n": 0}

    def fake_map(particle_indices, image_subset, _starfile):
        map_calls["n"] += 1
        # Stable mapping for both anomaly and combined particle arrays.
        if np.array_equal(np.sort(np.asarray(particle_indices)), np.array([100, 102], dtype=np.int32)):
            return np.array([1, 5], dtype=np.int32)
        return np.array([], dtype=np.int32)

    monkeypatch.setattr(outlier_cmd, "map_particle_original_indexing_to_images_original_indexing", fake_map)

    outlier_cmd.main()

    combined_dir = outdir / "combined_results"
    with open(combined_dir / "combined_image_outliers_4.pkl", "rb") as f:
        combined_image_outliers = pickle.load(f)
    with open(combined_dir / "combined_particle_outliers_4.pkl", "rb") as f:
        combined_particle_outliers = pickle.load(f)

    np.testing.assert_array_equal(combined_image_outliers, np.array([1, 5], dtype=np.int64))
    np.testing.assert_array_equal(combined_particle_outliers, np.array([100, 102], dtype=np.int64))
    assert map_calls["n"] >= 2


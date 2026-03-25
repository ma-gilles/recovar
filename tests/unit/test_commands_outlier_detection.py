import os
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")

from recovar.commands import outlier_detection as outlier_cmd
from recovar.data_io._index_utils import TiltSeriesOriginalIndexMap


pytestmark = pytest.mark.unit


class _FakePipelineOutput:
    def __init__(self, payload, sorted_payload=None):
        self._payload = payload
        self._sorted_payload = payload if sorted_payload is None else sorted_payload

    def get(self, key):
        return self._sorted_payload[key]

    def get_embedding_keys(self, entry):
        values = self._payload.get(entry)
        if values is None:
            return []
        return list(values.keys())

    def get_embedding_component(self, entry, key):
        return self._sorted_payload[entry][key]

    def get_unsorted_embedding_component(self, entry, key):
        return self._payload[entry][key]


def test_outlier_detection_from_contrast_spa_returns_image_indices():
    payload = {
        "input_args": SimpleNamespace(tilt_series=False, shared_contrast_across_tilts=False, particles="particles.mrcs"),
        "contrasts": {4: np.array([0.05, 0.2, 4.0, 0.5], dtype=np.float32)},
        "halfsets": [np.array([0, 1], dtype=np.int32), np.array([2, 3], dtype=np.int32)],
        "particles_halfsets": [np.array([0, 1], dtype=np.int32), np.array([2, 3], dtype=np.int32)],
    }
    po = _FakePipelineOutput(payload)

    image_outliers, image_inliers, particle_outliers, particle_inliers = outlier_cmd.outlier_detection_from_contrast(
        po,
        zdim_key=4,
        low_contrast_threshold=0.1,
        high_contrast_threshold=3.5,
        output_dir=None,
    )

    np.testing.assert_array_equal(image_outliers, np.array([0, 2], dtype=np.int32))
    np.testing.assert_array_equal(image_inliers, np.array([1, 3], dtype=np.int32))
    assert particle_outliers is None
    assert particle_inliers is None


def test_outlier_detection_from_contrast_tilt_shared_contrast_returns_particle_indices():
    payload = {
        "input_args": SimpleNamespace(tilt_series=True, shared_contrast_across_tilts=True, particles="particles.star"),
        "contrasts": {4: np.array([0.05, 0.2, 4.0], dtype=np.float32)},
        "halfsets": [np.array([0, 1, 2], dtype=np.int32), np.array([3, 4, 5], dtype=np.int32)],
        "particles_halfsets": [np.array([0, 1], dtype=np.int32), np.array([2], dtype=np.int32)],
    }
    po = _FakePipelineOutput(payload)

    image_outliers, image_inliers, particle_outliers, particle_inliers = outlier_cmd.outlier_detection_from_contrast(
        po,
        zdim_key=4,
        low_contrast_threshold=0.1,
        high_contrast_threshold=3.5,
        output_dir=None,
    )

    np.testing.assert_array_equal(image_outliers, np.array([0, 2], dtype=np.int32))
    np.testing.assert_array_equal(image_inliers, np.array([1], dtype=np.int32))
    assert particle_outliers is None
    assert particle_inliers is None


def test_outlier_detection_from_contrast_tilt_uses_explicit_image_to_particle_map(monkeypatch):
    payload = {
        "input_args": SimpleNamespace(tilt_series=True, shared_contrast_across_tilts=False, particles="particles.star"),
        "contrasts": {4: np.array([0.05, 0.05, np.nan, 0.2, 0.2], dtype=np.float32)},
        "halfsets": [np.array([0, 1], dtype=np.int32), np.array([3, 4], dtype=np.int32)],
        "particles_halfsets": [np.array([0], dtype=np.int32), np.array([1], dtype=np.int32)],
    }
    sorted_payload = {
        **payload,
        "contrasts": {4: np.array([0.05, 0.05, 0.2, 0.2], dtype=np.float32)},
    }
    po = _FakePipelineOutput(payload, sorted_payload=sorted_payload)

    image_to_particle = np.array([0, 0, 1, 1, 1], dtype=np.int32)
    tilt_index_map = TiltSeriesOriginalIndexMap(
        particle_to_images=(
            np.array([0, 1], dtype=np.int32),
            np.array([2, 3, 4], dtype=np.int32),
        ),
        image_to_particle=image_to_particle,
    )

    class _FakeTiltIndexMap:
        @staticmethod
        def from_particles_file(_starfile):
            return tilt_index_map

    monkeypatch.setattr(outlier_cmd, "TiltSeriesOriginalIndexMap", _FakeTiltIndexMap)
    monkeypatch.setattr(
        outlier_cmd.image_backends.TiltSeriesDataset,
        "parse_micrograph_tilt_mapping",
        staticmethod(lambda _starfile: (None, None)),
    )

    image_outliers, image_inliers, particle_outliers, particle_inliers = outlier_cmd.outlier_detection_from_contrast(
        po,
        zdim_key=4,
        low_contrast_threshold=0.1,
        high_contrast_threshold=3.5,
        particle_bad_fraction_threshold=0.7,
        output_dir=None,
    )

    np.testing.assert_array_equal(image_outliers, np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(image_inliers, np.array([3, 4], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(particle_outliers), np.array([0], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(particle_inliers), np.array([1], dtype=np.int32))


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
        "particles_halfsets": [np.array([0, 1], dtype=np.int32), np.array([2, 3], dtype=np.int32)],
        "halfsets": [np.array([0, 1], dtype=np.int32), np.array([2, 3], dtype=np.int32)],
        "contrasts": {4: np.ones(4, dtype=np.float32)},
    }
    sorted_payload = {
        **payload,
        "latent_coords": {4: np.full((4, 2), -999.0, dtype=np.float32)},
        "contrasts": {4: np.full(4, -999.0, dtype=np.float32)},
    }
    monkeypatch.setattr(
        outlier_cmd.output,
        "PipelineOutput",
        lambda _p: _FakePipelineOutput(payload, sorted_payload=sorted_payload),
    )

    def fake_plot(_zs, _orig_indices, folder, **_kw):
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "inliers_consensus.pkl"), "wb") as f:
            pickle.dump(np.array([0, 2, 3], dtype=np.int32), f)
        with open(os.path.join(folder, "outliers_consensus.pkl"), "wb") as f:
            pickle.dump(np.array([1], dtype=np.int32), f)

    monkeypatch.setattr(outlier_cmd, "plot_anomaly_detection_results", fake_plot)
    monkeypatch.setattr(
        outlier_cmd,
        "outlier_detection_from_contrast",
        lambda *_args, **_kwargs: (
            np.array([2], dtype=np.int32),
            np.array([0, 1, 3], dtype=np.int32),
            np.array([3], dtype=np.int32),
            np.array([0, 1, 2], dtype=np.int32),
        ),
    )
    monkeypatch.setattr(outlier_cmd, "create_particle_outlier_visualization", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(outlier_cmd, "create_overlap_matrix_visualization", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(outlier_cmd, "create_outlier_visualizations", lambda *_args, **_kwargs: None)

    outlier_cmd.main()

    combined_dir = outdir / "data" / "combined_results"
    with open(combined_dir / "combined_image_outliers_4.pkl", "rb") as f:
        combined_image_outliers = pickle.load(f)
    with open(combined_dir / "combined_particle_outliers_4.pkl", "rb") as f:
        combined_particle_outliers = pickle.load(f)
    with open(combined_dir / "combined_image_inliers_4.pkl", "rb") as f:
        combined_image_inliers = pickle.load(f)

    np.testing.assert_array_equal(combined_image_outliers, np.array([1, 2], dtype=np.int64))
    np.testing.assert_array_equal(combined_particle_outliers, np.array([1, 3], dtype=np.int64))
    np.testing.assert_array_equal(combined_image_inliers, np.array([0, 3], dtype=np.int64))


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
        "particles_halfsets": [np.array([0, 1], dtype=np.int32), np.array([2], dtype=np.int32)],
        "halfsets": [np.array([0, 1, 2], dtype=np.int32), np.array([3, 4, 5], dtype=np.int32)],
        "contrasts": None,
    }
    sorted_payload = {
        **payload,
        "latent_coords": {4: np.full((3, 2), -999.0, dtype=np.float32)},
    }
    monkeypatch.setattr(
        outlier_cmd.output,
        "PipelineOutput",
        lambda _p: _FakePipelineOutput(payload, sorted_payload=sorted_payload),
    )

    def fake_plot(_zs, _orig_indices, folder, **_kw):
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "inliers_consensus.pkl"), "wb") as f:
            pickle.dump(np.array([1], dtype=np.int32), f)
        with open(os.path.join(folder, "outliers_consensus.pkl"), "wb") as f:
            pickle.dump(np.array([0, 2], dtype=np.int32), f)

    monkeypatch.setattr(outlier_cmd, "plot_anomaly_detection_results", fake_plot)
    monkeypatch.setattr(outlier_cmd, "create_particle_outlier_visualization", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(outlier_cmd, "create_overlap_matrix_visualization", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(outlier_cmd, "create_outlier_visualizations", lambda *_args, **_kwargs: None)

    map_calls = {"n": 0}

    def fake_map(particle_indices, image_subset, _starfile):
        map_calls["n"] += 1
        # Stable mapping for both anomaly and combined particle arrays.
        if np.array_equal(np.sort(np.asarray(particle_indices)), np.array([0, 2], dtype=np.int32)):
            return np.array([1, 5], dtype=np.int32)
        return np.array([], dtype=np.int32)

    monkeypatch.setattr(outlier_cmd, "map_particle_original_indexing_to_images_original_indexing", fake_map)

    outlier_cmd.main()

    combined_dir = outdir / "data" / "combined_results"
    with open(combined_dir / "combined_image_outliers_4.pkl", "rb") as f:
        combined_image_outliers = pickle.load(f)
    with open(combined_dir / "combined_particle_outliers_4.pkl", "rb") as f:
        combined_particle_outliers = pickle.load(f)

    np.testing.assert_array_equal(combined_image_outliers, np.array([1, 5], dtype=np.int64))
    np.testing.assert_array_equal(combined_particle_outliers, np.array([0, 2], dtype=np.int64))
    assert map_calls["n"] >= 2


def test_outlier_detection_main_uses_sorted_embedding_component(monkeypatch, tmp_path):
    outdir = tmp_path / "outlier_output_filtered"
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
        "latent_coords": {
            4: np.array(
                [[0.0, 0.0], [np.nan, np.nan], [1.0, 1.0], [2.0, 2.0]],
                dtype=np.float32,
            )
        },
        "input_args": SimpleNamespace(tilt_series=False, particles="particles.mrcs", shared_contrast=False),
        "particles_halfsets": [np.array([0, 2], dtype=np.int32), np.array([3], dtype=np.int32)],
        "halfsets": [np.array([0, 2], dtype=np.int32), np.array([3], dtype=np.int32)],
        "contrasts": {4: np.array([0.1, np.nan, 0.2, 0.3], dtype=np.float32)},
    }
    sorted_payload = {
        **payload,
        "latent_coords": {
            4: np.array(
                [[10.0, 10.0], [11.0, 11.0], [12.0, 12.0]],
                dtype=np.float32,
            )
        },
        "contrasts": {4: np.array([0.1, 0.2, 0.3], dtype=np.float32)},
    }
    monkeypatch.setattr(
        outlier_cmd.output,
        "PipelineOutput",
        lambda _p: _FakePipelineOutput(payload, sorted_payload=sorted_payload),
    )

    def fake_plot(zs, original_indices, folder, **_kw):
        assert zs.shape == (3, 2)
        np.testing.assert_array_equal(
            zs,
            np.array(
                [[10.0, 10.0], [11.0, 11.0], [12.0, 12.0]],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(original_indices, np.array([0, 2, 3], dtype=np.int32))
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "inliers_consensus.pkl"), "wb") as f:
            pickle.dump(np.array([0, 3], dtype=np.int32), f)
        with open(os.path.join(folder, "outliers_consensus.pkl"), "wb") as f:
            pickle.dump(np.array([2], dtype=np.int32), f)

    monkeypatch.setattr(outlier_cmd, "plot_anomaly_detection_results", fake_plot)
    monkeypatch.setattr(
        outlier_cmd,
        "outlier_detection_from_contrast",
        lambda *_args, **_kwargs: (
            np.array([], dtype=np.int32),
            np.array([0, 2, 3], dtype=np.int32),
            None,
            None,
        ),
    )
    monkeypatch.setattr(outlier_cmd, "create_particle_outlier_visualization", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(outlier_cmd, "create_overlap_matrix_visualization", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(outlier_cmd, "create_outlier_visualizations", lambda *_args, **_kwargs: None)

    outlier_cmd.main()


def test_create_outlier_visualizations_tilt_series_uses_image_length_contrast_axis(monkeypatch, tmp_path):
    payload = {
        "latent_coords": {
            4: np.array(
                [[0.0, 0.0], [np.nan, np.nan], [1.0, 0.5], [2.0, -0.5]],
                dtype=np.float32,
            )
        },
        "contrasts": {4: np.array([0.2, 0.3, np.nan, 0.5, 0.7, 0.9, 1.1], dtype=np.float32)},
        "particles_halfsets": [np.array([0, 2], dtype=np.int32), np.array([3], dtype=np.int32)],
        "halfsets": [np.array([0, 1, 3], dtype=np.int32), np.array([4, 5, 6], dtype=np.int32)],
        "input_args": SimpleNamespace(shared_contrast_across_tilts=False),
    }
    sorted_payload = {
        **payload,
        "latent_coords": {
            4: np.array(
                [[0.0, 0.0], [1.0, 0.5], [2.0, -0.5]],
                dtype=np.float32,
            )
        },
        "contrasts": {4: np.array([0.2, 0.3, 0.5, 0.7, 0.9, 1.1], dtype=np.float32)},
    }
    pipeline_output = _FakePipelineOutput(payload, sorted_payload=sorted_payload)

    tilt_index_map = TiltSeriesOriginalIndexMap(
        particle_to_images=(
            np.array([0, 1], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([3, 4], dtype=np.int32),
            np.array([5, 6], dtype=np.int32),
        ),
        image_to_particle=np.array([0, 0, 1, 2, 2, 3, 3], dtype=np.int32),
    )

    class _FakeTiltIndexMap:
        @staticmethod
        def from_particles_file(_starfile):
            return tilt_index_map

    class _FakeUmap:
        def __init__(self, embedding):
            self.embedding_ = embedding

    monkeypatch.setattr(outlier_cmd, "TiltSeriesOriginalIndexMap", _FakeTiltIndexMap)
    monkeypatch.setattr(
        outlier_cmd.output,
        "umap_latent_space",
        lambda zs: _FakeUmap(np.asarray(zs[:, :2], dtype=np.float32)),
    )

    outlier_cmd.create_outlier_visualizations(
        pipeline_output,
        all_particle_outliers=[np.array([0], dtype=np.int32)],
        method_names=["Anomaly Detection"],
        combined_particle_outliers=np.array([0], dtype=np.int32),
        combined_particle_inliers=np.array([2, 3], dtype=np.int32),
        output_dir=str(tmp_path),
        zdim_key=4,
        total_particles=3,
        is_tilt_series=True,
        starfile="particles.star",
        noreg=False,
        save_all_plots=True,
    )

    stats_path = tmp_path / "data" / "outlier_visualizations" / "combined_4_stats.txt"
    assert stats_path.exists()
    stats_text = stats_path.read_text()
    assert "Outlier contrast - Mean: 0.250" in stats_text
    assert "Inlier contrast - Mean: 0.800" in stats_text

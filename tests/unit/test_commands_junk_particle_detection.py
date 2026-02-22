import os
import pickle

import numpy as np
import pytest

pytest.importorskip("jax")

from recovar.commands import junk_particle_detection as junk_cmd


pytestmark = pytest.mark.unit


def test_compute_fsc_auc_uses_threshold_crossing(monkeypatch):
    monkeypatch.setattr(
        junk_cmd.fourier_transform_utils,
        "get_1d_frequency_grid",
        lambda *_args, **_kwargs: np.array([-1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32),
    )

    fsc_curve = np.array([1.0, 0.8, 0.6, 0.2, 0.1], dtype=np.float32)
    auc = junk_cmd.compute_fsc_auc(fsc_curve, grid_size=8, voxel_size=1.0, threshold=0.5)
    assert auc == pytest.approx(0.7, rel=1e-5)


def test_compute_fsc_auc_returns_zero_when_all_below_threshold(monkeypatch):
    monkeypatch.setattr(
        junk_cmd.fourier_transform_utils,
        "get_1d_frequency_grid",
        lambda *_args, **_kwargs: np.array([0.0, 1.0, 2.0], dtype=np.float32),
    )

    fsc_curve = np.array([0.1, 0.05, 0.02], dtype=np.float32)
    auc = junk_cmd.compute_fsc_auc(fsc_curve, grid_size=8, voxel_size=1.0, threshold=0.5)
    assert auc == 0.0


def test_map_clusters_to_particles_maps_masks_and_stats():
    cluster_indices = np.array([0, 1, 1, 2, 0], dtype=np.int32)
    junk_clusters = np.array([1], dtype=np.int32)

    junk_particles, good_particles, stats = junk_cmd.map_clusters_to_particles(
        junk_clusters=junk_clusters,
        cluster_indices=cluster_indices,
        output_folder="/tmp/unused",
        zdim_key=4,
        method="adaptive_threshold",
    )

    np.testing.assert_array_equal(junk_particles, np.array([1, 2], dtype=np.int64))
    np.testing.assert_array_equal(good_particles, np.array([0, 3, 4], dtype=np.int64))
    assert stats["total_particles"] == 5
    assert stats["junk_particles"] == 2
    assert stats["n_junk_clusters"] == 1
    assert stats["method"] == "adaptive_threshold"


def test_save_particle_classifications_writes_pipeline_pickles(tmp_path):
    junk_particles = np.array([0, 2], dtype=np.int32)
    good_particles = np.array([1, 3], dtype=np.int32)
    cluster_indices = np.array([0, 1, 1, 0], dtype=np.int32)
    original_indices = np.array([10, 11, 12, 13], dtype=np.int32)
    stats = {"junk_fraction": 0.5}

    junk_cmd.save_particle_classifications(
        junk_particles=junk_particles,
        good_particles=good_particles,
        particle_stats=stats,
        cluster_indices=cluster_indices,
        output_folder=str(tmp_path),
        zdim_key=4,
        method="consensus",
        original_indices=original_indices,
    )

    with open(tmp_path / "junk_indices_4.pkl", "rb") as f:
        saved_junk = pickle.load(f)
    with open(tmp_path / "good_indices_4.pkl", "rb") as f:
        saved_good = pickle.load(f)
    with open(tmp_path / "particle_classifications_4.pkl", "rb") as f:
        saved_results = pickle.load(f)

    np.testing.assert_array_equal(saved_junk, np.array([10, 12], dtype=np.int32))
    np.testing.assert_array_equal(saved_good, np.array([11, 13], dtype=np.int32))
    assert saved_results["classification_method"] == "consensus"
    assert saved_results["metadata"]["total_particles"] == 4


def test_detect_junk_clusters_flags_low_quality_cluster_and_returns_info(monkeypatch, tmp_path):
    monkeypatch.setattr(junk_cmd, "create_junk_detection_visualizations", lambda *_args, **_kwargs: None)

    fsc_scores = {
        0: {"halfmap_fsc": 0.9, "vs_mean_fsc": 0.95},
        1: {"halfmap_fsc": 0.85, "vs_mean_fsc": 0.9},
        2: {"halfmap_fsc": 0.1, "vs_mean_fsc": 0.15},  # clearly worst
        3: {"halfmap_fsc": 0.45, "vs_mean_fsc": 0.5},
    }
    fsc_auc_scores = {
        0: {"halfmap_auc": 0.8, "vs_mean_auc": 0.85},
        1: {"halfmap_auc": 0.75, "vs_mean_auc": 0.8},
        2: {"halfmap_auc": 0.05, "vs_mean_auc": 0.08},  # clearly worst
        3: {"halfmap_auc": 0.35, "vs_mean_auc": 0.4},
    }

    junk_clusters, junk_info = junk_cmd.detect_junk_clusters(
        fsc_scores=fsc_scores,
        fsc_auc_scores=fsc_auc_scores,
        output_folder=str(tmp_path),
        zdim_key=4,
        method="adaptive_threshold",
        percentile_threshold=25,
        std_threshold=2.0,
        min_junk_fraction=0.1,
        max_junk_fraction=0.8,
    )

    assert 2 in set(junk_clusters.tolist())
    assert "scores" in junk_info
    assert "methods_results" in junk_info
    assert junk_info["method"] == "adaptive_threshold"


def test_detect_junk_clusters_rejects_unknown_method(monkeypatch, tmp_path):
    monkeypatch.setattr(junk_cmd, "create_junk_detection_visualizations", lambda *_args, **_kwargs: None)
    fsc_scores = {0: {"halfmap_fsc": 0.5, "vs_mean_fsc": 0.5}}
    fsc_auc_scores = {0: {"halfmap_auc": 0.5, "vs_mean_auc": 0.5}}

    with pytest.raises(ValueError, match="Unknown method"):
        junk_cmd.detect_junk_clusters(
            fsc_scores=fsc_scores,
            fsc_auc_scores=fsc_auc_scores,
            output_folder=str(tmp_path),
            zdim_key=4,
            method="not_a_method",
        )


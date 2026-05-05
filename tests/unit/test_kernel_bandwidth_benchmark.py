"""Unit tests for the 1D kernel-bandwidth benchmark helpers."""

import argparse
from types import SimpleNamespace

import numpy as np
import pytest

from recovar.commands import benchmark_kernel_bandwidth_1d as benchmark_cmd
from recovar.heterogeneity import kernel_bandwidth_benchmark as kb

pytestmark = pytest.mark.unit


def test_project_volume_trajectory_exact_rank_one():
    mean = np.ones((2, 2, 2), dtype=np.float32)
    mode = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    volumes = np.stack([mean - 2.0 * mode, mean, mean + 2.0 * mode], axis=0)

    projected, meta = kb.project_volume_trajectory(volumes, 1)

    assert projected.shape == volumes.shape
    assert np.allclose(projected, volumes)
    assert meta["n_pcs"] == 1
    assert meta["scores"].shape[0] == volumes.shape[0]


def test_choose_bandwidth_bins_is_sorted_and_positive():
    bins = kb.choose_bandwidth_bins([np.array([0.0, 1.0, 2.0]), np.array([3.0, 4.0])], n_bandwidths=5, n_min_particles=1)

    assert bins.shape == (5,)
    assert np.all(bins > 0)
    assert np.all(np.diff(bins) >= 0)


def test_shellwise_oracle_and_cv_handles_zero_error():
    target = np.zeros((2, 2, 2), dtype=np.float32)
    estimates = [np.zeros((3, 2, 2, 2), dtype=np.float32), np.zeros((3, 2, 2, 2), dtype=np.float32)]
    cv = [np.zeros((2, 2, 2), dtype=np.float32), np.zeros((2, 2, 2), dtype=np.float32)]
    lhs = [np.ones((2, 2, 2), dtype=np.float32), np.ones((2, 2, 2), dtype=np.float32)]

    result = kb.compute_shellwise_oracle_and_cv(estimates, cv, lhs, target)

    assert result["oracle_error"].shape[0] == 3
    assert result["oracle_error"].shape == result["cv_score"].shape
    assert np.allclose(result["oracle_error"], 0.0)
    assert np.allclose(result["cv_score"], 0.0)
    assert np.all(result["oracle_choice"] == 0)
    assert np.all(result["cv_choice"] == 0)
    assert np.allclose(result["regret"], 0.0)


def test_benchmark_command_add_args_parses_expected_flags():
    parser = benchmark_cmd.add_args(argparse.ArgumentParser())
    default_args = parser.parse_args(["--output-dir", "/tmp/out"])
    args = parser.parse_args(
        [
            "--output-dir",
            "/tmp/out",
            "--trajectory-source",
            "toy",
            "--embedding-source",
            "gt-pc",
            "--pc-project",
            "1",
        ]
    )

    assert args.output_dir == "/tmp/out"
    assert args.trajectory_source == "toy"
    assert default_args.trajectory_source == "pdb5nrl"
    assert args.embedding_source == "gt-pc"
    assert args.pc_project == 1
    assert args.zdim == 1


def test_run_benchmark_smoke(monkeypatch, tmp_path):
    class _FakeHVD:
        volume_shape = (2, 2, 2)
        volumes = np.zeros((3, 2, 2, 2), dtype=np.float32)

    class _FakeDataset:
        volume_shape = (2, 2, 2)

        def split_halfset_array(self, arr, per_particle=False):
            arr = np.asarray(arr)
            return [arr[:2], arr[2:]]

        def get_halfset(self, half):
            return self

        def get_valid_frequency_indices(self):
            return np.ones(self.volume_shape, dtype=np.float32)

    def fake_write_dataset(*_args, **_kwargs):
        dataset_dir = tmp_path / "test_dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        sim_info = {
            "image_assignment": np.array([0, 1, 2, 0], dtype=np.int32),
            "volumes_path_root": str(tmp_path / "unused" / "vol"),
            "grid_size": 2,
        }
        return np.zeros((4, 2, 2, 2), dtype=np.float32), sim_info

    monkeypatch.setattr(benchmark_cmd, "_prepare_trajectory_volumes", lambda args, benchmark_dir, voxel_size: (str(tmp_path / "vol"), {"raw_prefix": str(tmp_path / "vol"), "pc_project": 0, "projection": {"n_pcs": 0}}))
    monkeypatch.setattr(benchmark_cmd, "_write_dataset", fake_write_dataset)
    monkeypatch.setattr(benchmark_cmd.synthetic_dataset, "load_heterogeneous_reconstruction", lambda *_args, **_kwargs: _FakeHVD())
    monkeypatch.setattr(benchmark_cmd.halfsets, "load_halfset_dataset_from_args", lambda *_args, **_kwargs: _FakeDataset())
    monkeypatch.setattr(
        benchmark_cmd.latent_density,
        "compute_latent_quadratic_forms_in_batch",
        lambda *_args, **_kwargs: np.array([[0.1], [0.2]], dtype=np.float32),
    )
    monkeypatch.setattr(benchmark_cmd.kb, "choose_bandwidth_bins", lambda *_args, **_kwargs: np.array([0.1, 0.2, 0.3], dtype=np.float32))
    monkeypatch.setattr(
        benchmark_cmd.kb,
        "compute_candidate_estimates",
        lambda *_args, **_kwargs: (
            [np.zeros((3, 2, 2, 2), dtype=np.float32), np.zeros((3, 2, 2, 2), dtype=np.float32)],
            [np.zeros((2, 2, 2), dtype=np.float32), np.zeros((2, 2, 2), dtype=np.float32)],
            [np.ones((2, 2, 2), dtype=np.float32), np.ones((2, 2, 2), dtype=np.float32)],
        ),
    )
    monkeypatch.setattr(benchmark_cmd.kb, "save_summary_csv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(benchmark_cmd.kb, "save_plots", lambda *_args, **_kwargs: None)

    args = SimpleNamespace(
        output_dir=str(tmp_path / "out"),
        project=None,
        grid_size=2,
        n_states=3,
        n_images=4,
        noise_level=0.0,
        contrast_std=0.0,
        noise_model="radial1",
        seed=0,
        trajectory_source="pdb5nrl",
        pc_project=0,
        volume_distribution="uniform",
        embedding_source="gt-pc",
        zdim=1,
        gt_z_sigma=0.05,
        target_state=None,
        n_bandwidths=3,
        n_min_particles=1,
        q_max=0.95,
        batch_size=None,
        heterogeneity_kernel="parabola",
        save_candidate_volumes=False,
        lazy=False,
        premultiplied_ctf=False,
        bfactor=80.0,
        max_rotation_degrees=10.0,
        pdb_path=None,
        pipeline_output=None,
    )

    summary = benchmark_cmd.run_benchmark(args, tmp_path / "out")

    assert summary["choice_match_rate"] == 1.0
    assert (tmp_path / "out" / "bandwidth_benchmark" / "summary.json").exists()

"""Unit tests for the 1D kernel-bandwidth benchmark helpers."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from recovar.commands import benchmark_kernel_bandwidth_1d as benchmark_cmd
from recovar.commands import extract_image_subset as extract_subset_cmd
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


def test_choose_bandwidth_bins_matches_production_helper():
    distances_by_half = [np.array([100.0, 200.0], dtype=np.float32), np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)]
    bins = kb.choose_bandwidth_bins(distances_by_half, n_bandwidths=5, n_min_particles=1)
    expected = kb.hv.pick_heterogeneity_bins2(-1, distances_by_half[1], q=0.5, min_images=1, n_bins=5)

    assert np.allclose(bins, expected)


def test_affine_align_points_matches_target_path():
    source = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    target = np.array([[1.0, -1.0], [3.0, 1.0], [5.0, 3.0]], dtype=np.float32)

    aligned = kb._affine_align_points(source, target)

    assert aligned.shape == target.shape
    assert np.allclose(aligned, target)


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


def test_shellwise_oracle_and_cv_accepts_cv_focus_mask():
    target = np.zeros((2, 2, 2), dtype=np.float32)
    estimates0 = np.zeros((2, 2, 2, 2), dtype=np.float32)
    estimates1 = np.zeros((2, 2, 2, 2), dtype=np.float32)
    estimates0[0, 0, 0, 0] = 10.0
    estimates1[1, 1, 1, 1] = 1.0
    estimates = [estimates0, estimates1]
    cv = [np.zeros((2, 2, 2), dtype=np.float32), np.zeros((2, 2, 2), dtype=np.float32)]
    lhs = [np.ones((2, 2, 2), dtype=np.float32), np.ones((2, 2, 2), dtype=np.float32)]
    mask = np.zeros((2, 2, 2), dtype=np.float32)
    mask[1, 1, 1] = 1.0

    unmasked = kb.compute_shellwise_oracle_and_cv(estimates, cv, lhs, target)
    masked = kb.compute_shellwise_oracle_and_cv(estimates, cv, lhs, target, cv_focus_mask=mask)

    assert unmasked["oracle_choice"][0] == 1
    assert masked["oracle_choice"][0] == 0
    assert masked["cv_choice"][0] == 0
    assert masked["cv_focus_mask_used"] is True
    assert masked["cv_focus_mask_fraction"] == 1.0 / 8.0


def test_save_bandwidth_ladder_figure(tmp_path):
    candidate_bins = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    est0 = np.zeros((3, 4, 4, 4), dtype=np.float32)
    est1 = np.zeros((3, 4, 4, 4), dtype=np.float32)
    est0[1, 2, 2, 2] = 1.0
    est1[2, 1, 1, 1] = -1.0

    path = tmp_path / "ladder.pdf"
    kb._save_bandwidth_ladder_figure(path, candidate_bins, [est0, est1])

    assert path.exists()
    assert path.stat().st_size > 0


def test_save_compute_state_walkthrough_plots_smoke(tmp_path, monkeypatch):
    compute_state_root = tmp_path / "compute_state"
    diag_dir = compute_state_root / "diagnostics" / "state000"
    diag_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "heterogeneity_bins": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "n_images_per_bin": np.array([1, 3, 4], dtype=np.int32),
        "ml_choice": np.array([[0, 1], [1, 2]], dtype=np.int32),
        "ml_errors": np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[0.5, 1.5], [2.5, 3.5]],
                [[0.25, 1.25], [2.25, 3.25]],
            ],
            dtype=np.float32,
        ),
        "locres_sampling": 1.0,
        "locres_maskrad": 1.0,
        "voxel_size": 1.0,
    }
    benchmark_cmd.utils.pickle_dump(params, diag_dir / "params.pkl")
    benchmark_cmd.utils.write_mrc(diag_dir / "local_resolution.mrc", np.ones((4, 4, 4), dtype=np.float32), voxel_size=1.0)
    np.savetxt(diag_dir / "heterogeneity_distances.txt", np.linspace(0.0, 1.0, 4, dtype=np.float32))

    def fake_extract_image_subset(input_dir, output_path, subvolume_idx, mask, coordinate):
        benchmark_cmd.utils.pickle_dump(np.array([0, 2], dtype=np.int32), output_path)

    monkeypatch.setattr(extract_subset_cmd, "extract_image_subset", fake_extract_image_subset)

    image_stack = np.arange(4 * 4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4, 4)
    particle_zs = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0], [3.0, 1.5]], dtype=np.float32)
    particle_labels = np.array([0, 1, 1, 2], dtype=np.int32)
    gt_state_scores = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    moving_mask = np.zeros((4, 4, 4), dtype=np.float32)
    moving_mask[1:3, 1:3, 1:3] = 1.0

    saved, summary = kb.save_compute_state_walkthrough_plots(
        tmp_path / "plots",
        compute_state_root,
        image_stack=image_stack,
        particle_zs=particle_zs,
        particle_labels=particle_labels,
        gt_state_scores=gt_state_scores,
        moving_mask=moving_mask,
        diagnostic_zs=particle_zs,
    )

    assert "compute_state_bandwidth_selection.pdf" in saved
    assert "compute_state_shell_choice.pdf" in saved
    assert "compute_state_subset_embedding.pdf" in saved
    assert "compute_state_subset_montage.pdf" in saved
    assert "compute_state_shell_choices.csv" in saved
    assert summary["compute_state_subset_count"] == 2
    assert len(summary["compute_state_shell_choice_head"]) == 2
    for path in saved.values():
        assert Path(path).exists()


def test_write_dataset_forces_zero_contrast(tmp_path, monkeypatch):
    calls = {}

    def fake_generate_synthetic_dataset(*_args, **kwargs):
        calls["contrast_std"] = kwargs["contrast_std"]
        calls["noise_scale_std"] = kwargs["noise_scale_std"]
        image_stack = np.zeros((2, 2, 2, 2), dtype=np.float32)
        sim_info = {"noise_variance": np.array([1.0], dtype=np.float32)}
        return image_stack, sim_info

    monkeypatch.setattr(benchmark_cmd.simulator, "generate_synthetic_dataset", fake_generate_synthetic_dataset)

    image_stack, sim_info = benchmark_cmd._write_dataset(
        "vol",
        tmp_path,
        voxel_size=1.0,
        n_images=2,
        grid_size=2,
        volume_distribution=np.array([1.0], dtype=np.float32),
        noise_level=0.1,
        contrast_std=0.25,
        noise_model="radial1",
        seed=0,
        premultiplied_ctf=False,
    )

    assert image_stack.shape == (2, 2, 2, 2)
    assert "noise_variance" in sim_info
    assert calls["contrast_std"] == 0.0
    assert calls["noise_scale_std"] == 0.0


def test_run_pipeline_if_needed_uses_moving_mask_and_disables_contrast(tmp_path, monkeypatch):
    class _FakeHVD:
        volume_shape = (2, 2, 2)

    calls = {}

    def fake_union_mask(_gt, shape):
        return np.ones(shape, dtype=np.float32), np.ones(shape, dtype=bool)

    def fake_moving_mask(_gt, shape):
        mask = np.zeros(shape, dtype=np.float32)
        mask[0, 0, 0] = 1.0
        return mask, mask.astype(bool)

    def fake_standard_recovar_pipeline(parsed_args):
        calls["parsed_args"] = parsed_args

    monkeypatch.setattr(benchmark_cmd.synthetic_dataset, "load_heterogeneous_reconstruction", lambda *_args, **_kwargs: _FakeHVD())
    monkeypatch.setattr(benchmark_cmd.metrics, "make_union_gt_mask_from_hvd", fake_union_mask)
    monkeypatch.setattr(benchmark_cmd.metrics, "make_moving_gt_mask_from_hvd", fake_moving_mask)
    monkeypatch.setattr(benchmark_cmd.pipeline, "standard_recovar_pipeline", fake_standard_recovar_pipeline)

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "particles.star").write_text("star")
    (dataset_dir / "poses.pkl").write_bytes(b"poses")
    (dataset_dir / "ctf.pkl").write_bytes(b"ctf")
    sim_info = {"grid_size": 2}

    args = SimpleNamespace(
        pipeline_output=None,
        grid_size=2,
        noise_model="radial1",
        lazy=False,
        low_memory_option=False,
        very_low_memory_option=False,
        premultiplied_ctf=False,
    )

    output_dir = benchmark_cmd._run_pipeline_if_needed(args, dataset_dir, sim_info, 1.0, enabled=True)

    assert output_dir == str(dataset_dir / "pipeline_output")
    parsed_args = calls["parsed_args"]
    assert parsed_args.correct_contrast is False
    assert parsed_args.mask == str(dataset_dir / "gt_masks" / "gt_union_mask.mrc")
    assert parsed_args.focus_mask == str(dataset_dir / "gt_masks" / "gt_moving_mask.mrc")


def test_benchmark_command_add_args_parses_expected_flags():
    parser = benchmark_cmd.add_args(argparse.ArgumentParser())
    default_args = parser.parse_args(["--output-dir", "/tmp/out"])
    args = parser.parse_args(
        [
            "--output-dir",
            "/tmp/out",
            "--trajectory-source",
            "pdb5nrl",
            "--embedding-source",
            "gt-pc",
            "--pc-project",
            "1",
        ]
    )

    assert args.output_dir == "/tmp/out"
    assert default_args.trajectory_source == "pdb5nrl"
    assert default_args.contrast_std == 0.0
    assert args.embedding_source == "gt-pc"
    assert default_args.diagnostic_embedding_source == "recovar"
    assert default_args.cv_focus_moving_mask is False
    assert default_args.skip_compute_state is False
    assert default_args.compute_state_maskrad_fraction == 0.5
    assert default_args.compute_state_n_min_particles == 100
    assert default_args.compute_state_save_all_estimates is False
    assert default_args.low_memory_option is False
    assert args.pc_project == 1
    assert args.zdim == 1


def test_load_embedding_components_uses_pipeline_embedding(monkeypatch):
    calls = {}

    class _FakeDataset:
        volume_size = 8

        pass

    class _FakePipelineOutput:
        def __init__(self, _path):
            self.dataset = _FakeDataset()
            self.values = {
                "dataset": self.dataset,
                "lazy_dataset": self.dataset,
                "input_args": SimpleNamespace(ignore_zero_frequency=True),
                "mean": np.ones((8,), dtype=np.complex64),
                "u": np.ones((2, 8), dtype=np.complex64),
                "s": np.array([2.0, 1.0], dtype=np.float32),
                "volume_mask": np.ones((2, 2, 2), dtype=np.float32),
            }

        def get(self, key):
            return self.values[key]

    def fake_embedding(mean, u, s, n_pcs_to_use, dataset, volume_mask, gpu_memory, disc_type, **kwargs):
        calls["mean_shape"] = np.asarray(mean).shape
        calls["u_shape"] = np.asarray(u).shape
        calls["s_shape"] = np.asarray(s).shape
        calls["n_pcs_to_use"] = n_pcs_to_use
        calls["dataset_is_fake"] = isinstance(dataset, _FakeDataset)
        calls["volume_mask_shape"] = np.asarray(volume_mask).shape
        calls["gpu_memory"] = gpu_memory
        calls["disc_type"] = disc_type
        calls["kwargs"] = kwargs
        zs = np.zeros((4, n_pcs_to_use), dtype=np.float32)
        covs = np.repeat(np.eye(n_pcs_to_use, dtype=np.float32)[None, :, :], 4, axis=0)
        contrasts = np.ones((4,), dtype=np.float32)
        return zs, covs, contrasts, None

    monkeypatch.setattr(benchmark_cmd.o, "PipelineOutput", _FakePipelineOutput)
    monkeypatch.setattr(benchmark_cmd.embedding, "get_per_image_embedding", fake_embedding)
    monkeypatch.setattr(benchmark_cmd.utils, "get_gpu_memory_total", lambda: 48.0)

    dataset, zs, cov_zs, contrasts = benchmark_cmd._load_embedding_components("/tmp/pipeline_output", 2, lazy=False)

    assert isinstance(dataset, _FakeDataset)
    assert zs.shape == (4, 2)
    assert cov_zs.shape == (4, 2, 2)
    assert contrasts.shape == (4,)
    assert calls["mean_shape"] == (8,)
    assert calls["u_shape"] == (8, 2)
    assert calls["s_shape"] == (2,)
    assert calls["n_pcs_to_use"] == 2
    assert calls["dataset_is_fake"] is True
    assert calls["volume_mask_shape"] == (2, 2, 2)
    assert calls["gpu_memory"] == 48.0
    assert calls["disc_type"] == "linear_interp"
    assert calls["kwargs"]["contrast_option"] == "none"
    assert calls["kwargs"]["ignore_zero_frequency"] is True


def test_run_benchmark_smoke(monkeypatch, tmp_path):
    class _FakeHVD:
        volume_shape = (2, 2, 2)
        volumes = np.zeros((3, 2, 2, 2), dtype=np.float32)

    class _FakeDataset:
        volume_shape = (2, 2, 2)

        def set_noise(self, noise_variance):
            self.noise_variance = np.asarray(noise_variance)

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
            "noise_variance": np.array([1.0], dtype=np.float32),
        }
        return np.zeros((4, 2, 2, 2), dtype=np.float32), sim_info

    monkeypatch.setattr(
        benchmark_cmd,
        "_prepare_trajectory_volumes",
        lambda args, benchmark_dir, voxel_size: (
            str(tmp_path / "vol"),
            {"raw_prefix": str(tmp_path / "vol"), "pc_project": 0, "trajectory_source": "pdb5nrl", "projection": {"n_pcs": 0}},
            np.zeros((3, 2, 2, 2), dtype=np.float32),
            np.zeros((3, 2, 2, 2), dtype=np.float32),
        ),
    )
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
    monkeypatch.setattr(benchmark_cmd.kb, "save_presentation_plots", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(benchmark_cmd.compute_state_cmd, "compute_state", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(benchmark_cmd.kb, "save_compute_state_walkthrough_plots", lambda *_args, **_kwargs: ({}, {}))
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
        diagnostic_embedding_source="none",
        diagnostic_zdim=3,
        zdim=1,
        gt_z_sigma=0.05,
        target_state=None,
        n_bandwidths=3,
        n_min_particles=1,
        batch_size=None,
        heterogeneity_kernel="parabola",
        cv_focus_moving_mask=False,
        skip_compute_state=True,
        compute_state_maskrad_fraction=0.5,
        compute_state_n_min_particles=100,
        compute_state_save_all_estimates=False,
        save_candidate_volumes=False,
        lazy=False,
        low_memory_option=False,
        very_low_memory_option=False,
        premultiplied_ctf=False,
        bfactor=80.0,
        max_rotation_degrees=5.0,
        pdb_path=None,
        pipeline_output=None,
    )

    summary = benchmark_cmd.run_benchmark(args, tmp_path / "out")

    assert summary["choice_match_rate"] == 1.0
    assert (tmp_path / "out" / "bandwidth_benchmark" / "summary.json").exists()
    assert (tmp_path / "out" / "bandwidth_benchmark" / "report.md").exists()
    assert (tmp_path / "out" / "bandwidth_benchmark" / "trace.json").exists()

"""Unit tests for the student PDB walkthrough command."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from recovar.commands import benchmark_kernel_bandwidth_1d as walkthrough
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


def test_add_args_defaults_are_student_friendly():
    parser = walkthrough.add_args(argparse.ArgumentParser())
    args = parser.parse_args(["--output-dir", "/tmp/out"])

    assert args.output_dir == "/tmp/out"
    assert args.noise_level == 1.0
    assert args.pc_project == 0
    assert args.n_states == 50
    assert args.n_bins == 50
    assert args.compute_state_maskrad_fraction == 4.0
    assert args.compute_state_n_min_particles == 100
    assert args.low_memory_option is False
    assert args.very_low_memory_option is False
    assert args.path == "symmetric"


def test_add_args_path_choices_include_arm_only():
    parser = walkthrough.add_args(argparse.ArgumentParser())
    args = parser.parse_args(["--output-dir", "/tmp/out", "--path", "arm_only"])
    assert args.path == "arm_only"


def test_write_active_volumes_saves_projection_metadata(tmp_path):
    volumes = np.zeros((3, 2, 2, 2), dtype=np.float32)
    volumes[1, 0, 0, 0] = 1.0
    volumes[2, 0, 0, 0] = 2.0
    args = SimpleNamespace(pc_project=1)

    prefix, active, meta = walkthrough._write_active_volumes(volumes, args, tmp_path, voxel_size=1.0)

    assert prefix == tmp_path / "02_active_volumes" / "vol"
    assert active.shape == volumes.shape
    assert meta["n_pcs"] == 1
    assert (tmp_path / "02_active_volumes" / "vol0000.mrc").exists()
    assert (tmp_path / "02_active_volumes" / "pca_scores.npy").exists()
    assert (tmp_path / "02_active_volumes" / "README.json").exists()


def test_simulate_dataset_disables_contrast(tmp_path, monkeypatch):
    calls = {}

    def fake_generate_synthetic_dataset(*_args, **kwargs):
        calls.update(kwargs)
        image_stack = np.zeros((4, 2, 2, 2), dtype=np.float32)
        sim_info = {"image_assignment": np.array([0, 1, 0, 1], dtype=np.int32)}
        return image_stack, sim_info

    monkeypatch.setattr(walkthrough.simulator, "generate_synthetic_dataset", fake_generate_synthetic_dataset)
    args = SimpleNamespace(
        n_images=4,
        grid_size=2,
        n_states=2,
        noise_level=1.0,
        noise_model="radial1",
        seed=0,
        premultiplied_ctf=False,
        overwrite=True,
    )

    image_stack, sim_info = walkthrough._simulate_dataset(args, tmp_path, tmp_path / "vol", voxel_size=1.0)

    assert image_stack.shape == (4, 2, 2, 2)
    assert np.array_equal(sim_info["image_assignment"], np.array([0, 1, 0, 1], dtype=np.int32))
    assert calls["contrast_std"] == 0.0
    assert calls["noise_scale_std"] == 0.0
    assert calls["per_particle_contrast"] is False


def test_write_gt_masks_and_volumes(tmp_path, monkeypatch):
    from recovar.core import mask as core_mask

    class FakeHVD:
        volumes = np.ones((2, 2, 2, 2), dtype=np.float32)

    seen = {}

    def fake_union(_real_vols, shape, dilation_iters=None, **_kwargs):
        seen["union_dilation"] = dilation_iters
        return np.ones(shape, dtype=np.float32), np.ones(shape, dtype=bool)

    def fake_moving(_real_vols, shape, dilation_iters=None, **_kwargs):
        seen["moving_dilation"] = dilation_iters
        mask = np.zeros(shape, dtype=np.float32)
        mask[0, 0, 0] = 1.0
        return mask, mask.astype(bool)

    monkeypatch.setattr(walkthrough.synthetic_dataset, "load_heterogeneous_reconstruction", lambda _sim_info: FakeHVD())
    monkeypatch.setattr(core_mask, "make_union_gt_mask", fake_union)
    monkeypatch.setattr(core_mask, "make_moving_gt_mask", fake_moving)

    paths = walkthrough._write_gt_masks_and_volumes(tmp_path, {}, grid_size=2, voxel_size=1.0, mask_dilation_iters=1)

    assert Path(paths["volume_mask"]).name == "volume_mask_union.mrc"
    assert Path(paths["focus_mask"]).name == "focus_mask_moving.mrc"
    assert Path(paths["volume_mask"]).exists()
    assert Path(paths["focus_mask"]).exists()
    assert (tmp_path / "04_ground_truth" / "gt_volumes_used_by_simulator.npy").exists()
    assert seen["union_dilation"] == 1
    assert seen["moving_dilation"] == 1


def test_run_pipeline_disables_contrast_and_uses_masks(tmp_path, monkeypatch):
    calls = {}

    def fake_pipeline(parsed_args):
        calls["args"] = parsed_args

    monkeypatch.setattr(walkthrough.pipeline, "standard_recovar_pipeline", fake_pipeline)
    dataset_dir = tmp_path / "03_dataset"
    dataset_dir.mkdir()
    args = SimpleNamespace(
        noise_model="radial1",
        lazy=False,
        low_memory_option=False,
        very_low_memory_option=False,
        pipeline_gpu_memory=None,
        premultiplied_ctf=False,
        overwrite=True,
    )
    mask_paths = {
        "volume_mask": str(tmp_path / "05_masks" / "volume_mask_union.mrc"),
        "focus_mask": str(tmp_path / "05_masks" / "focus_mask_moving.mrc"),
    }

    pipeline_dir = walkthrough._run_pipeline(args, tmp_path, mask_paths)

    assert pipeline_dir == tmp_path / "06_pipeline"
    parsed = calls["args"]
    assert parsed.correct_contrast is False
    assert parsed.mask == mask_paths["volume_mask"]
    assert parsed.focus_mask == mask_paths["focus_mask"]


def test_middle_state_latent_point_uses_pipeline_embedding(monkeypatch):
    class FakePipelineOutput:
        def __init__(self, _path):
            pass

        def get(self, key):
            assert key == "latent_coords"
            return {1: np.array([[0.0], [2.0], [4.0], [6.0]], dtype=np.float32)}

    monkeypatch.setattr(walkthrough.o, "PipelineOutput", FakePipelineOutput)
    sim_info = {"image_assignment": np.array([0, 1, 1, 2], dtype=np.int32)}

    latent = walkthrough._middle_state_latent_point(Path("/tmp/pipeline"), sim_info, target_state=1, zdim=1)

    assert latent.shape == (1, 1)
    assert np.allclose(latent, [[3.0]])


def test_run_walkthrough_smoke(monkeypatch, tmp_path):
    calls = {}

    def fake_raw(_args, out, _voxel_size):
        vols = np.zeros((3, 2, 2, 2), dtype=np.float32)
        return out / "01_raw_volumes" / "vol", vols

    def fake_active(raw_volumes, _args, out, _voxel_size):
        meta = {"explained_energy": np.array([1.0, 0.0, 0.0], dtype=np.float32)}
        return out / "02_active_volumes" / "vol", raw_volumes, meta

    def fake_dataset(_args, _out, _prefix, _voxel_size):
        return np.zeros((4, 2, 2, 2), dtype=np.float32), {"image_assignment": np.array([0, 1, 1, 2], dtype=np.int32)}

    def fake_masks(out, _sim_info, _grid_size, _voxel_size, **_kwargs):
        return {"gt_dir": str(out / "04_ground_truth"), "volume_mask": "mask.mrc", "focus_mask": "focus.mrc"}

    def fake_pipeline(_args, out, _mask_paths):
        return out / "06_pipeline"

    def fake_latent(_pipeline_dir, _sim_info, target_state, zdim):
        calls["target_state"] = target_state
        calls["zdim"] = zdim
        return np.array([[1.5]], dtype=np.float32)

    def fake_compute_state(_args, out, _pipeline_dir, latent_point):
        calls["latent_point"] = latent_point
        return out / "07_compute_state"

    monkeypatch.setattr(walkthrough, "_write_raw_pdb_volumes", fake_raw)
    monkeypatch.setattr(walkthrough, "_write_active_volumes", fake_active)
    monkeypatch.setattr(walkthrough, "_simulate_dataset", fake_dataset)
    monkeypatch.setattr(walkthrough, "_write_gt_masks_and_volumes", fake_masks)
    monkeypatch.setattr(walkthrough, "_run_pipeline", fake_pipeline)
    monkeypatch.setattr(walkthrough, "_middle_state_latent_point", fake_latent)
    monkeypatch.setattr(walkthrough, "_run_compute_state", fake_compute_state)

    args = SimpleNamespace(
        output_dir=str(tmp_path),
        grid_size=2,
        n_states=3,
        n_images=4,
        noise_level=1.0,
        noise_model="radial1",
        seed=0,
        pc_project=1,
        target_state=None,
        zdim=1,
        n_bins=3,
        compute_state_maskrad_fraction=4.0,
        compute_state_n_min_particles=100,
        compute_state_save_all_estimates=False,
        lazy=False,
        low_memory_option=False,
        very_low_memory_option=False,
        pipeline_gpu_memory=None,
        premultiplied_ctf=False,
        bfactor=80.0,
        max_rotation_degrees=5.0,
        pdb_path=None,
        overwrite=True,
        use_oracle_pipeline=False,
        path="symmetric",
        mask_dilation_iters=None,
    )

    summary = walkthrough.run_walkthrough(args, tmp_path)

    assert summary["target_state"] == 1
    assert summary["contrast_std"] == 0.0
    assert calls["target_state"] == 1
    assert np.allclose(calls["latent_point"], [[1.5]])
    assert summary["use_oracle_pipeline"] is False
    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "config.json").exists()


def test_use_oracle_pipeline_flag_dispatches_to_oracle(monkeypatch, tmp_path):
    calls = {}

    def fake_raw(_args, out, _voxel_size):
        return out / "01_raw_volumes" / "vol", np.zeros((3, 2, 2, 2), dtype=np.float32)

    def fake_active(raw_volumes, _args, out, _voxel_size):
        meta = {"explained_energy": np.array([1.0, 0.0, 0.0], dtype=np.float32)}
        return out / "02_active_volumes" / "vol", raw_volumes, meta

    def fake_dataset(_args, _out, _prefix, _voxel_size):
        return np.zeros((4, 2, 2, 2), dtype=np.float32), {"image_assignment": np.array([0, 1, 1, 2], dtype=np.int32)}

    def fake_masks(out, _sim_info, _grid_size, _voxel_size, **_kwargs):
        return {"gt_dir": str(out / "04_ground_truth"), "volume_mask": "mask.mrc", "focus_mask": "focus.mrc"}

    def fake_run_pipeline(*_args, **_kwargs):
        calls["dispatched"] = "real_pipeline"
        return tmp_path / "06_pipeline"

    def fake_oracle(args, out, mask_paths, sim_info, voxel_size):
        calls["dispatched"] = "oracle_pipeline"
        calls["mask_paths"] = mask_paths
        calls["voxel_size"] = voxel_size
        calls["sim_info_keys"] = sorted(sim_info.keys())
        return out / "06_pipeline"

    def fake_latent(_pipeline_dir, _sim_info, _target_state, _zdim):
        return np.array([[2.5]], dtype=np.float32)

    def fake_compute_state(_args, out, _pipeline_dir, _latent_point):
        return out / "07_compute_state"

    monkeypatch.setattr(walkthrough, "_write_raw_pdb_volumes", fake_raw)
    monkeypatch.setattr(walkthrough, "_write_active_volumes", fake_active)
    monkeypatch.setattr(walkthrough, "_simulate_dataset", fake_dataset)
    monkeypatch.setattr(walkthrough, "_write_gt_masks_and_volumes", fake_masks)
    monkeypatch.setattr(walkthrough, "_run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(walkthrough, "_run_oracle_pipeline", fake_oracle)
    monkeypatch.setattr(walkthrough, "_middle_state_latent_point", fake_latent)
    monkeypatch.setattr(walkthrough, "_run_compute_state", fake_compute_state)

    args = SimpleNamespace(
        output_dir=str(tmp_path),
        grid_size=2,
        n_states=3,
        n_images=4,
        noise_level=1.0,
        noise_model="radial1",
        seed=0,
        pc_project=1,
        target_state=None,
        zdim=1,
        n_bins=3,
        compute_state_maskrad_fraction=4.0,
        compute_state_n_min_particles=100,
        compute_state_save_all_estimates=False,
        lazy=False,
        low_memory_option=False,
        very_low_memory_option=False,
        pipeline_gpu_memory=None,
        premultiplied_ctf=False,
        bfactor=80.0,
        max_rotation_degrees=5.0,
        pdb_path=None,
        overwrite=True,
        use_oracle_pipeline=True,
        path="arm_only",
        mask_dilation_iters=1,
    )

    summary = walkthrough.run_walkthrough(args, tmp_path)

    assert calls["dispatched"] == "oracle_pipeline"
    assert summary["path"] == "arm_only"
    assert summary["use_oracle_pipeline"] is True
    assert "image_assignment" in calls["sim_info_keys"]
    assert calls["mask_paths"]["volume_mask"] == "mask.mrc"


def test_oracle_pipeline_run_oracle_invokes_writer(monkeypatch, tmp_path):
    """`_run_oracle_pipeline` should call into the oracle writer with the right args."""
    captured = {}

    def fake_write_oracle(**kwargs):
        captured.update(kwargs)
        Path(kwargs["pipeline_dir"]).mkdir(parents=True, exist_ok=True)
        return {
            "pipeline_dir": str(kwargs["pipeline_dir"]),
            "zdims": [1],
            "n_pcs_available": 1,
            "n_particles": 4,
            "halfset_sizes": [2, 2],
            "noise_variance_length": 1,
            "voxel_size": 1.0,
            "volume_shape": [2, 2, 2],
        }

    monkeypatch.setattr(walkthrough.oracle_pipeline, "write_oracle_pipeline_output", fake_write_oracle)

    mask_path = tmp_path / "mask.mrc"
    focus_path = tmp_path / "focus.mrc"
    from recovar import utils as recovar_utils

    recovar_utils.write_mrc(mask_path, np.ones((2, 2, 2), dtype=np.float32), voxel_size=1.0)
    recovar_utils.write_mrc(focus_path, np.zeros((2, 2, 2), dtype=np.float32), voxel_size=1.0)

    args = SimpleNamespace(
        overwrite=True,
        zdim=1,
        pipeline_gpu_memory=None,
        premultiplied_ctf=False,
        noise_model="radial1",
    )
    sim_info = {
        "image_assignment": np.array([0, 1, 1, 2], dtype=np.int32),
        "noise_variance": np.ones(1, dtype=np.float32),
    }

    pipeline_dir = walkthrough._run_oracle_pipeline(
        args,
        tmp_path,
        {"volume_mask": str(mask_path), "focus_mask": str(focus_path)},
        sim_info,
        voxel_size=1.0,
    )

    assert pipeline_dir == tmp_path / "06_pipeline"
    assert captured["pipeline_dir"] == tmp_path / "06_pipeline"
    assert captured["voxel_size"] == 1.0
    assert captured["noise_model"] == "radial"  # radial1 is mapped to radial
    assert captured["premultiplied_ctf"] is False
    assert captured["focus_mask"].shape == (2, 2, 2)
    assert (tmp_path / "06_pipeline" / "oracle_pipeline_info.json").exists()

"""Unit tests for the oracle pipeline output writer.

These tests cover the pieces that don't require a real CryoEMDataset or
GPU: input_args construction, halfset/embedding layout on disk, and the
oracle PCA helper. The full end-to-end embedding path is exercised by
the GPU smoke command on Slurm.
"""

import argparse
from types import SimpleNamespace

import numpy as np
import pytest

from recovar import utils
from recovar.simulation import oracle_pipeline

pytestmark = pytest.mark.unit


def test_compute_oracle_basis_matches_simulator_pca(monkeypatch):
    class FakeHVD:
        def __init__(self):
            self.volume_shape = (2, 2, 2)

        def get_mean(self):
            return np.zeros(8, dtype=np.complex64)

        def get_u(self):
            u = np.zeros((8, 1), dtype=np.complex64)
            u[0, 0] = 1.0
            return u

        def get_s(self):
            return np.array([3.0], dtype=np.float32)

        def get_probs_of_state(self):
            return np.array([0.5, 0.5], dtype=np.float32)

    monkeypatch.setattr(
        oracle_pipeline.synthetic_dataset,
        "load_heterogeneous_reconstruction",
        lambda _sim_info: FakeHVD(),
    )

    out = oracle_pipeline.compute_oracle_basis({})

    assert out["mean"].shape == (8,)
    assert out["u"].shape == (8, 1)
    assert out["s"].shape == (1,)
    assert out["volume_shape"] == (2, 2, 2)
    assert out["mean_real"].shape == (2, 2, 2)
    assert out["u_real"].shape == (1, 2, 2, 2)
    assert np.allclose(out["probs"], [0.5, 0.5])


def test_build_oracle_input_args_has_required_fields(tmp_path):
    args = oracle_pipeline._build_oracle_input_args(
        pipeline_dir=tmp_path,
        particles_path=tmp_path / "particles.star",
        poses_path=tmp_path / "poses.pkl",
        ctf_path=tmp_path / "ctf.pkl",
        halfsets_pickle_path=tmp_path / "model" / "particles_halfsets.pkl",
        zdims=[1, 2],
        premultiplied_ctf=False,
        correct_contrast=False,
        noise_model="radial",
        volume_mask_path=tmp_path / "model" / "mask.mrc",
        focus_mask_path=None,
    )

    # Fields HalfsetDatasetSpec.from_args reads.
    assert args.particles == str(tmp_path / "particles.star")
    assert args.poses == str(tmp_path / "poses.pkl")
    assert args.ctf == str(tmp_path / "ctf.pkl")
    assert args.tilt_series is False
    assert args.tilt_series_ctf == "cryoem"
    assert args.uninvert_data == "false"

    # Fields PipelineOutput / compute_state inspect.
    assert args.halfsets == str(tmp_path / "model" / "particles_halfsets.pkl")
    assert args.zdim == [1, 2]
    assert args.correct_contrast is False
    assert args.outdir == str(tmp_path)


def test_nan_pad_to_original_scatters_into_full_array():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    sorted_indices = np.array([0, 3])

    out = oracle_pipeline._nan_pad_to_original(arr, sorted_indices)

    assert out.shape == (4, 2)
    assert np.allclose(out[0], [1.0, 2.0])
    assert np.allclose(out[3], [3.0, 4.0])
    assert np.isnan(out[1]).all()
    assert np.isnan(out[2]).all()


def test_save_embeddings_per_zdim_writes_files(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    embedding_dict = {
        "latent_coords": {1: np.array([[1.0], [2.0]], dtype=np.float32)},
        "latent_coords_noreg": {1: np.array([[1.1], [2.1]], dtype=np.float32)},
        "latent_precision": {1: np.array([[[1.0]], [[1.0]]], dtype=np.float32)},
        "latent_precision_noreg": {1: np.array([[[1.0]], [[1.0]]], dtype=np.float32)},
        "contrasts": {1: np.array([1.0, 1.0], dtype=np.float32)},
        "contrasts_noreg": {1: np.array([1.0, 1.0], dtype=np.float32)},
    }

    oracle_pipeline._save_embeddings_per_zdim(model_dir, embedding_dict, np.array([0, 1]))

    zdim_dir = model_dir / "zdim_1"
    assert (zdim_dir / "latent_coords.npy").exists()
    assert (zdim_dir / "latent_precision.npy").exists()
    assert (zdim_dir / "contrasts.npy").exists()
    saved_zs = np.load(zdim_dir / "latent_coords.npy")
    assert saved_zs.shape == (2, 1)
    assert np.allclose(saved_zs, [[1.0], [2.0]])


def test_write_oracle_pipeline_output_skeleton(monkeypatch, tmp_path):
    """End-to-end test of the writer with embedding mocked.

    This exercises:
      - oracle PCA helper called
      - dataset directory inputs validated
      - halfsets pickled
      - per-zdim embedding files written
      - params.pkl written with the expected keys
      - output/volumes/mean.mrc written
    """
    dataset_dir = tmp_path / "03_dataset"
    dataset_dir.mkdir()
    # Required dataset files for the writer's existence checks.
    (dataset_dir / "particles.64.mrcs").write_bytes(b"")
    (dataset_dir / "particles.star").write_text("# fake star\n")
    (dataset_dir / "poses.pkl").write_bytes(b"")
    (dataset_dir / "ctf.pkl").write_bytes(b"")

    class FakeHVD:
        def __init__(self):
            self.volume_shape = (2, 2, 2)

        def get_mean(self):
            return np.zeros(8, dtype=np.complex64)

        def get_u(self):
            u = np.zeros((8, 2), dtype=np.complex64)
            u[0, 0] = 1.0
            u[1, 1] = 1.0
            return u

        def get_s(self):
            return np.array([2.0, 1.0], dtype=np.float32)

        def get_probs_of_state(self):
            return np.array([0.5, 0.5], dtype=np.float32)

    monkeypatch.setattr(
        oracle_pipeline.synthetic_dataset,
        "load_heterogeneous_reconstruction",
        lambda _sim_info: FakeHVD(),
    )

    # Stub dataset loader so we do not parse a real STAR file.
    fake_dataset = SimpleNamespace()
    fake_dataset.set_radial_noise_model = lambda *a, **kw: None
    monkeypatch.setattr(
        oracle_pipeline.halfset_io,
        "load_halfset_dataset_from_args",
        lambda *_a, **_kw: fake_dataset,
    )

    # Stub the embedding step to return shape-checked outputs.
    def fake_embed(mean, u, s, basis_size, dataset, mask, gpu_memory, **kwargs):
        n = 4
        zs = np.tile(np.arange(basis_size, dtype=np.float32)[None, :], (n, 1))
        cov = np.tile(np.eye(basis_size, dtype=np.float32)[None, :, :], (n, 1, 1))
        contrasts = np.ones(n, dtype=np.float32)
        return zs, cov, contrasts, None

    monkeypatch.setattr(oracle_pipeline.embedding, "get_per_image_embedding", fake_embed)

    sim_info = {
        "image_assignment": np.array([0, 1, 0, 1], dtype=np.int32),
        "noise_variance": np.ones(1, dtype=np.float32),
    }
    volume_mask = np.ones((2, 2, 2), dtype=np.float32)

    summary = oracle_pipeline.write_oracle_pipeline_output(
        pipeline_dir=tmp_path / "06_pipeline",
        dataset_dir=dataset_dir,
        voxel_size=1.0,
        volume_mask=volume_mask,
        sim_info=sim_info,
        zdims=[1],
        gpu_memory=2.0,
    )

    assert summary["zdims"] == [1]
    assert summary["n_pcs_available"] == 2
    assert summary["n_particles"] == 4
    assert summary["halfset_sizes"] == [2, 2]

    pipeline_dir = tmp_path / "06_pipeline"
    assert (pipeline_dir / "model" / "params.pkl").exists()
    assert (pipeline_dir / "model" / "halfsets.pkl").exists()
    assert (pipeline_dir / "model" / "particles_halfsets.pkl").exists()
    assert (pipeline_dir / "model" / "zdim_1" / "latent_coords.npy").exists()
    assert (pipeline_dir / "model" / "zdim_1" / "latent_precision.npy").exists()
    assert (pipeline_dir / "model" / "zdim_1" / "contrasts.npy").exists()
    assert (pipeline_dir / "output" / "volumes" / "mean.mrc").exists()
    assert (pipeline_dir / "output" / "volumes" / "mask.mrc").exists()
    assert (pipeline_dir / "output" / "volumes" / "eigen_pos0000.mrc").exists()

    params = utils.pickle_load(pipeline_dir / "model" / "params.pkl")
    assert params["version"] == "0.7"
    assert params["volume_shape"] == (2, 2, 2)
    assert params["voxel_size"] == 1.0
    assert isinstance(params["input_args"], argparse.Namespace)
    assert params["input_args"].correct_contrast is False

    # zs were saved in NaN-padded original-image-index order; with all
    # particles assigned, the array length equals n_particles and rows are
    # in sorted-original order.
    saved_zs = np.load(pipeline_dir / "model" / "zdim_1" / "latent_coords.npy")
    assert saved_zs.shape == (4, 1)
    assert not np.isnan(saved_zs).any()


def test_write_oracle_pipeline_output_rejects_all_outliers(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "03_dataset"
    dataset_dir.mkdir()
    (dataset_dir / "particles.64.mrcs").write_bytes(b"")
    (dataset_dir / "particles.star").write_text("")
    (dataset_dir / "poses.pkl").write_bytes(b"")
    (dataset_dir / "ctf.pkl").write_bytes(b"")

    class FakeHVD:
        volume_shape = (2, 2, 2)

        def get_mean(self):
            return np.zeros(8, dtype=np.complex64)

        def get_u(self):
            return np.eye(8, 1, dtype=np.complex64)

        def get_s(self):
            return np.array([1.0], dtype=np.float32)

        def get_probs_of_state(self):
            return np.array([1.0], dtype=np.float32)

    monkeypatch.setattr(
        oracle_pipeline.synthetic_dataset,
        "load_heterogeneous_reconstruction",
        lambda _sim_info: FakeHVD(),
    )

    sim_info = {
        "image_assignment": np.array([-1, -1], dtype=np.int32),
        "noise_variance": np.ones(1, dtype=np.float32),
    }
    with pytest.raises(ValueError, match="outliers"):
        oracle_pipeline.write_oracle_pipeline_output(
            pipeline_dir=tmp_path / "06_pipeline",
            dataset_dir=dataset_dir,
            voxel_size=1.0,
            volume_mask=np.ones((2, 2, 2), dtype=np.float32),
            sim_info=sim_info,
        )

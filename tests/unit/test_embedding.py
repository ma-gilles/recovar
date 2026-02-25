import numpy as np
import pytest

pytest.importorskip("jax")

from recovar import embedding
from recovar.dataset import CryoEMHalfsets

pytestmark = pytest.mark.unit


class _Cryo:
    def __init__(self, n_images):
        self.n_images = n_images


def test_split_weights_partitions_by_cryo_sizes():
    w = np.arange(10, dtype=np.float32)
    cryos = [_Cryo(3), _Cryo(4), _Cryo(3)]
    out = embedding.split_weights(w, cryos)
    assert len(out) == 3
    assert np.allclose(out[0], [0, 1, 2])
    assert np.allclose(out[1], [3, 4, 5, 6])
    assert np.allclose(out[2], [7, 8, 9])


def test_generate_conformation_from_reprojection_linear_combination():
    # mean: (1, vol_size), u: (vol_size, latent_dim), xs: (n_states, latent_dim)
    mean = np.array([[10.0, 20.0]], dtype=np.float32)
    u = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    xs = np.array([[1.0, 2.0], [-1.0, 0.5]], dtype=np.float32)

    out = embedding.generate_conformation_from_reprojection(xs, mean, u)
    expected = np.array(
        [
            [11.0, 24.0],  # mean + [1,4]
            [9.0, 21.0],   # mean + [-1,1]
        ],
        dtype=np.float32,
    )
    assert out.shape == expected.shape
    assert np.allclose(out, expected)


class _DummyCryo:
    def __init__(self, volume_size=4, image_size=16, n_images=3, dtype=np.complex64):
        self.volume_size = volume_size
        self.image_size = image_size
        self.n_images = n_images
        self.dtype = dtype


def test_get_per_image_embedding_clamps_batch_size_to_at_least_one(monkeypatch):
    cryo0 = _DummyCryo(volume_size=4, image_size=16, n_images=3)
    cryo1 = _DummyCryo(volume_size=4, image_size=16, n_images=2)
    mean = np.zeros((4,), dtype=np.complex64)
    u = np.zeros((4, 2), dtype=np.complex64)
    s = np.ones((2,), dtype=np.float32)
    volume_mask = np.ones((4,), dtype=np.float32)

    monkeypatch.setattr(embedding, "USE_CUBIC", False)
    monkeypatch.setattr(embedding.utils, "get_embedding_batch_size", lambda *_args, **_kwargs: 0)

    captured = {"batch_sizes": []}

    def fake_get_coords(
        experiment_dataset,
        mean_estimate,
        basis,
        eigenvalues,
        volume_mask_in,
        contrast_grid,
        batch_size,
        disc_type,
        **kwargs,
    ):
        _ = (mean_estimate, eigenvalues, volume_mask_in, contrast_grid, disc_type, kwargs)
        captured["batch_sizes"].append(batch_size)
        bsz = basis.shape[0]
        n = experiment_dataset.n_images
        xs = np.zeros((n, bsz), dtype=np.complex64)
        cov = np.zeros((n, bsz, bsz), dtype=np.complex64)
        contrasts = np.ones((n,), dtype=np.float32)
        bias = np.zeros((n, bsz, bsz), dtype=np.complex64)
        return xs, cov, contrasts, bias

    monkeypatch.setattr(embedding, "get_coords_in_basis_and_contrast_3", fake_get_coords)

    zs, cov_zs, est_contrasts, bias = embedding.get_per_image_embedding(
        mean=mean,
        u=u,
        s=s,
        basis_size=2,
        cryos=CryoEMHalfsets(cryo0, cryo1),
        volume_mask=volume_mask,
        gpu_memory=1,
        disc_type="linear_interp",
        contrast_option="none",
        to_real=True,
        compute_covariances=True,
        compute_bias=True,
    )

    assert captured["batch_sizes"] == [1, 1]
    assert zs.shape == (5, 2)
    assert cov_zs.shape == (5, 2, 2)
    assert est_contrasts.shape == (5,)
    assert bias.shape == (5, 2, 2)
    assert np.isrealobj(zs)
    assert np.isrealobj(cov_zs)
    assert np.isrealobj(bias)


def test_get_per_image_embedding_ignore_zero_frequency_overrides_volume_mask(monkeypatch):
    cryo0 = _DummyCryo(volume_size=4, image_size=16, n_images=2)
    cryo1 = _DummyCryo(volume_size=4, image_size=16, n_images=1)
    mean = np.zeros((4,), dtype=np.complex64)
    u = np.zeros((4, 1), dtype=np.complex64)
    s = np.ones((1,), dtype=np.float32)
    volume_mask = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)

    monkeypatch.setattr(embedding, "USE_CUBIC", False)
    monkeypatch.setattr(embedding.utils, "get_embedding_batch_size", lambda *_args, **_kwargs: 10)

    captured = {"masks": [], "contrast_grid_size": []}

    def fake_get_coords(
        experiment_dataset,
        mean_estimate,
        basis,
        eigenvalues,
        volume_mask_in,
        contrast_grid,
        batch_size,
        disc_type,
        **kwargs,
    ):
        _ = (mean_estimate, eigenvalues, batch_size, disc_type, kwargs)
        captured["masks"].append(np.asarray(volume_mask_in))
        captured["contrast_grid_size"].append(int(np.asarray(contrast_grid).size))
        n = experiment_dataset.n_images
        bsz = basis.shape[0]
        return (
            np.zeros((n, bsz), dtype=np.complex64),
            np.zeros((n, bsz, bsz), dtype=np.complex64),
            np.ones((n,), dtype=np.float32),
            np.zeros((n, bsz, bsz), dtype=np.complex64),
        )

    monkeypatch.setattr(embedding, "get_coords_in_basis_and_contrast_3", fake_get_coords)

    embedding.get_per_image_embedding(
        mean=mean,
        u=u,
        s=s,
        basis_size=1,
        cryos=CryoEMHalfsets(cryo0, cryo1),
        volume_mask=volume_mask,
        gpu_memory=1,
        disc_type="linear_interp",
        contrast_option="none",
        ignore_zero_frequency=True,
        to_real=True,
        compute_covariances=False,
        compute_bias=False,
    )

    assert len(captured["masks"]) == 2
    np.testing.assert_allclose(captured["masks"][0], np.ones((4,), dtype=np.float32))
    np.testing.assert_allclose(captured["masks"][1], np.ones((4,), dtype=np.float32))
    assert captured["contrast_grid_size"] == [1, 1]


def test_get_per_image_embedding_supports_single_cryo_list(monkeypatch):
    cryo = _DummyCryo(volume_size=4, image_size=16, n_images=2)
    mean = np.zeros((4,), dtype=np.complex64)
    u = np.zeros((4, 1), dtype=np.complex64)
    s = np.ones((1,), dtype=np.float32)
    volume_mask = np.ones((4,), dtype=np.float32)

    monkeypatch.setattr(embedding, "USE_CUBIC", False)
    monkeypatch.setattr(embedding.utils, "get_embedding_batch_size", lambda *_args, **_kwargs: 10)
    monkeypatch.setattr(
        embedding,
        "get_coords_in_basis_and_contrast_3",
        lambda experiment_dataset, _mean_estimate, basis, _eigenvalues, _volume_mask_in, _contrast_grid, _batch_size, _disc_type, **_kwargs: (
            np.zeros((experiment_dataset.n_images, basis.shape[0]), dtype=np.complex64),
            np.zeros((experiment_dataset.n_images, basis.shape[0], basis.shape[0]), dtype=np.complex64),
            np.ones((experiment_dataset.n_images,), dtype=np.float32),
            np.zeros((experiment_dataset.n_images, basis.shape[0], basis.shape[0]), dtype=np.complex64),
        ),
    )

    zs, cov_zs, est_contrasts, bias = embedding.get_per_image_embedding(
        mean=mean,
        u=u,
        s=s,
        basis_size=1,
        cryos=CryoEMHalfsets(cryo, cryo),
        volume_mask=volume_mask,
        gpu_memory=1,
        disc_type="linear_interp",
        contrast_option="none",
        to_real=True,
        compute_covariances=True,
        compute_bias=True,
    )

    assert zs.shape == (4, 1)
    assert cov_zs.shape == (4, 1, 1)
    assert est_contrasts.shape == (4,)
    assert bias.shape == (4, 1, 1)

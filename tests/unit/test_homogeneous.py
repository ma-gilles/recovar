import numpy as np
import pytest

pytest.importorskip("jax")

from recovar.reconstruction import homogeneous
from recovar.reconstruction import regularization
from recovar.reconstruction import relion_functions

pytestmark = pytest.mark.unit


class _FakeHalfsetDataset:
    def __init__(self, tag, volume_shape):
        self.tag = tag
        self.volume_shape = volume_shape


class _FakeDataset:
    def __init__(self):
        self.volume_shape = (4,)
        self.halfset_indices = [
            np.array([0, 1], dtype=np.int32),
            np.array([2, 3], dtype=np.int32),
        ]

    def materialize_halfset_datasets(self):
        return (
            _FakeHalfsetDataset(0, self.volume_shape),
            _FakeHalfsetDataset(1, self.volume_shape),
        )


def test_get_mean_conformation_relion_flow_and_restore(monkeypatch):
    def fake_triangular_kernel(dataset, noise_variance, batch_size, index_subset=None, **kwargs):
        tag = dataset.tag
        return (
            np.ones(4, dtype=np.float32) * (1 + tag),
            np.ones(4, dtype=np.float32) * (10 + tag),
        )

    def fake_post_process_from_filter_v2(ft_ctf, ft_y, volume_shape, upsampling_factor, tau=None, **kwargs):
        base = float(np.mean(ft_ctf) + np.mean(ft_y) + (0 if tau is None else np.mean(tau)))
        return np.full(4, base, dtype=np.float32)

    monkeypatch.setattr(relion_functions, "relion_style_triangular_kernel", fake_triangular_kernel)
    monkeypatch.setattr(relion_functions, "post_process_from_filter_v2", fake_post_process_from_filter_v2)
    monkeypatch.setattr(
        regularization,
        "compute_relion_prior",
        lambda dataset, noise, m0, m1, batch: (np.ones(4, dtype=np.float32) * 2.0, np.array([0.9]), np.array([0.0])),
    )

    ds = _FakeDataset()
    means, mean_prior, fsc = homogeneous.get_mean_conformation_relion(
        dataset=ds,
        batch_size=2,
        noise_variance=np.ones(4, dtype=np.float32),
        use_regularization=False,
        upsampling_factor=3,
    )

    assert means.combined is not None
    assert means.corrected0 is not None
    assert means.corrected1 is not None
    assert means.corrected0reg is not None
    assert means.corrected1reg is not None
    assert means.lhs is not None
    assert means.prior is not None
    assert means.combined.shape == (4,)
    assert isinstance(mean_prior, np.ndarray)
    assert np.asarray(fsc).size > 0


def test_get_mean_conformation_relion_use_regularization_switch(monkeypatch):
    monkeypatch.setattr(
        relion_functions,
        "relion_style_triangular_kernel",
        lambda dataset, *_args, **_kwargs: (np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32)),
    )

    def fake_pp_v2(ft_ctf, ft_y, volume_shape, upsampling_factor, tau=None, **kwargs):
        if tau is None:
            return np.array([1.0, 1.0], dtype=np.float32)
        return np.array([3.0, 3.0], dtype=np.float32)

    monkeypatch.setattr(relion_functions, "post_process_from_filter_v2", fake_pp_v2)
    monkeypatch.setattr(
        regularization,
        "compute_relion_prior",
        lambda *_args, **_kwargs: (np.ones(2, dtype=np.float32), np.array([0.8]), np.array([0.0])),
    )

    ds = _FakeDataset()
    ds.volume_shape = (2,)
    ds.halfset_indices = [np.array([0], dtype=np.int32), np.array([1], dtype=np.int32)]

    means_unreg, *_ = homogeneous.get_mean_conformation_relion(
        dataset=ds, batch_size=1,
        noise_variance=np.ones(2, dtype=np.float32),
        use_regularization=False,
    )
    means_reg, *_ = homogeneous.get_mean_conformation_relion(
        dataset=ds, batch_size=1,
        noise_variance=np.ones(2, dtype=np.float32),
        use_regularization=True,
    )

    # unregularized combined = average of tau=None outputs = 1.0
    assert np.allclose(means_unreg.combined, 1.0)
    # regularized combined = average of tau!=None outputs = 3.0
    assert np.allclose(means_reg.combined, 3.0)

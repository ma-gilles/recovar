import numpy as np
import pytest

pytest.importorskip("jax")

from recovar.reconstruction import homogeneous
from recovar.reconstruction import regularization
from recovar.reconstruction import relion_functions

pytestmark = pytest.mark.unit


class _FakeCryo:
    def __init__(self, tag):
        self.tag = tag
        self.volume_upsampling_factor = 1

    def update_volume_upsampling_factor(self, v):
        self.volume_upsampling_factor = v


def test_get_mean_conformation_relion_flow_and_restore(monkeypatch):
    def fake_triangular_kernel(cryo, noise_variance, batch_size, disc_type):
        _ = (noise_variance, batch_size, disc_type)
        return np.ones(4, dtype=np.float32) * (1 + cryo.tag), np.ones(4, dtype=np.float32) * (10 + cryo.tag)

    def fake_post_process_from_filter(cryo, ft_ctf, ft_y, tau, **kwargs):
        _ = kwargs
        base = float(np.mean(ft_ctf) + np.mean(ft_y) + (0 if tau is None else np.mean(tau)))
        return np.full(4, base + cryo.tag, dtype=np.float32)

    monkeypatch.setattr(relion_functions, "relion_style_triangular_kernel", fake_triangular_kernel)
    monkeypatch.setattr(relion_functions, "post_process_from_filter", fake_post_process_from_filter)
    monkeypatch.setattr(
        regularization,
        "compute_relion_prior",
        lambda cryos, noise, m0, m1, batch: (np.ones(4, dtype=np.float32) * 2.0, np.array([0.9]), np.array([0.0])),
    )

    cryos = [_FakeCryo(0), _FakeCryo(1)]
    means, mean_prior, fsc = homogeneous.get_mean_conformation_relion(
        cryos=cryos,
        batch_size=2,
        noise_variance=np.ones(4, dtype=np.float32),
        use_regularization=False,
        upsampling_factor=3,
        disc_type="linear_interp",
    )

    assert "combined" in means
    assert "combined_regularized" in means
    assert means["combined"].shape == (4,)
    assert means["prior"].shape == (4,)
    assert isinstance(mean_prior, np.ndarray)
    assert np.asarray(fsc).size > 0
    # Ensure upsampling factor restored.
    assert cryos[0].volume_upsampling_factor == 1
    assert cryos[1].volume_upsampling_factor == 1


def test_get_mean_conformation_relion_use_regularization_switch(monkeypatch):
    monkeypatch.setattr(
        relion_functions,
        "relion_style_triangular_kernel",
        lambda cryo, *_args, **_kwargs: (np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32)),
    )

    def fake_pp(cryo, ft_ctf, ft_y, tau, **kwargs):
        _ = (cryo, ft_ctf, ft_y, kwargs)
        if tau is None:
            return np.array([1.0, 1.0], dtype=np.float32)
        return np.array([3.0, 3.0], dtype=np.float32)

    monkeypatch.setattr(relion_functions, "post_process_from_filter", fake_pp)
    monkeypatch.setattr(
        regularization,
        "compute_relion_prior",
        lambda *_args, **_kwargs: (np.ones(2, dtype=np.float32), np.array([0.8]), np.array([0.0])),
    )

    cryos = [_FakeCryo(0), _FakeCryo(1)]
    means, *_ = homogeneous.get_mean_conformation_relion(
        cryos=cryos,
        batch_size=1,
        noise_variance=np.ones(2, dtype=np.float32),
        use_regularization=True,
    )
    assert np.allclose(means["combined"], means["combined_regularized"])

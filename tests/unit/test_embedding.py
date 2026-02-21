import numpy as np
import pytest

pytest.importorskip("jax")

from recovar import embedding

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

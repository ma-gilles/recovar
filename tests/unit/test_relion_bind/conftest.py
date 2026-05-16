"""Shared fixtures for relion_bind tests."""

import numpy as np
import pytest


@pytest.fixture(params=[16, 32, 64])
def box_size(request):
    """Test across multiple box sizes."""
    return request.param


@pytest.fixture()
def rng():
    """Seeded RNG for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture()
def random_real_volume(box_size, rng):
    """Random real-space volume in RELION convention."""
    return rng.standard_normal((box_size, box_size, box_size))


@pytest.fixture()
def random_fftw_half(random_real_volume):
    """FFTW rfftn of the random volume."""
    return np.fft.rfftn(random_real_volume)


@pytest.fixture()
def random_recovar_centered(random_real_volume):
    """recovar centered full-complex of the random volume."""
    return np.fft.fftshift(np.fft.fftn(random_real_volume))

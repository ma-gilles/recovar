from types import SimpleNamespace

import numpy as np
import pytest
import matplotlib.pyplot as plt

pytest.importorskip("jax")

from recovar import plot_utils

pytestmark = pytest.mark.unit


class _FakePipelineOutput:
    def __init__(self, values):
        self._values = values

    def get(self, key):
        return self._values.get(key)


def test_plot_power_spectrum_returns_shell_average():
    vol = np.ones((8, 8, 8), dtype=np.complex64)
    avg = plot_utils.plot_power_spectrum(vol)
    assert hasattr(avg, "shape")
    assert avg.ndim == 1
    assert avg.size > 0


def test_plot_noise_profile_1d():
    po = _FakePipelineOutput(
        {
            "noise_var_used": np.linspace(1.0, 2.0, 16).astype(np.float32),
            "input_args": SimpleNamespace(ignore_zero_frequency=False),
            "image_PS": np.linspace(1.2, 2.2, 16).astype(np.float32),
            "std_image_PS": np.full(16, 0.1, dtype=np.float32),
            "masked_image_PS": None,
            "std_masked_image_PS": None,
            "noise_var_from_hf": None,
            "radial_noise_var_outside_mask": None,
        }
    )
    fig, ax = plot_utils.plot_noise_profile(po, yscale="linear")
    assert fig is not None and ax is not None


def test_plot_noise_profile_2d_per_tilt():
    po = _FakePipelineOutput(
        {
            "noise_var_used": np.stack(
                [
                    np.linspace(1.0, 1.5, 12),
                    np.linspace(1.1, 1.6, 12),
                    np.linspace(1.2, 1.7, 12),
                ],
                axis=0,
            ).astype(np.float32),
            "input_args": SimpleNamespace(ignore_zero_frequency=True),
            "image_PS": None,
            "std_image_PS": None,
            "masked_image_PS": np.linspace(1.0, 2.0, 12).astype(np.float32),
            "std_masked_image_PS": np.full(12, 0.05, dtype=np.float32),
            "noise_var_from_hf": np.linspace(0.8, 0.9, 12).astype(np.float32),
            "radial_noise_var_outside_mask": np.linspace(0.7, 0.85, 12).astype(np.float32),
        }
    )
    fig, ax = plot_utils.plot_noise_profile(po, yscale="log")
    assert fig is not None and ax is not None


def test_plot_summary_t_prefers_selective_get_u_real_api():
    calls = {"get_u_real": 0}

    class _PO:
        def get(self, key):
            if key == "u_real":
                raise AssertionError("legacy get('u_real') path should not be used")
            if key in {"mean", "volume_mask", "variance"}:
                return np.zeros(8, dtype=np.float32)
            raise KeyError(key)

        def get_u_real(self, n_pcs):
            calls["get_u_real"] += 1
            assert n_pcs == 2
            return np.zeros((2, 2, 2, 2), dtype=np.float32)

    plot_utils.plot_summary_t(_PO(), n_eigs=2, filename=None)
    assert calls["get_u_real"] == 1
    plt.close("all")


def test_plot_summary_t_clamps_n_eigs_to_available_components():
    class _PO:
        def get(self, key):
            if key in {"mean", "volume_mask", "variance"}:
                return np.zeros(8, dtype=np.float32)
            if key == "u_real":
                return np.zeros((1, 2, 2, 2), dtype=np.float32)
            raise KeyError(key)

    # Request more eigenvectors than available: should not raise.
    plot_utils.plot_summary_t(_PO(), n_eigs=5, filename=None)
    plt.close("all")

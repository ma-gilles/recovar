"""Test the compute-capability preflight in cuda_backproject."""
import pathlib
from unittest import mock

import pytest


def test_preflight_raises_on_unsupported_gpu():
    """When the GPU's compute cap isn't in the .so, the error message
    must contain the three critical diagnostic sections."""
    from recovar import cuda_backproject

    # Reset cached state
    cuda_backproject._preflight_ok = None

    fake_so = pathlib.Path("/tmp/fake_libcuda_backproject.so")

    with (
        mock.patch.object(
            cuda_backproject, "_detect_gpu_compute_cap",
            return_value=("NVIDIA Tesla P100", "60"),
        ),
        mock.patch.object(
            cuda_backproject, "_detect_so_arches",
            return_value=({"80", "86", "89", "90"}, set()),
        ),
        mock.patch.object(
            cuda_backproject, "_detect_nvcc_version",
            return_value="12.8",
        ),
    ):
        with pytest.raises(RuntimeError, match="compute capability sm_60"):
            cuda_backproject._preflight_check(fake_so)

    # Verify all three diagnostic sections are in the message
    cuda_backproject._preflight_ok = None
    with (
        mock.patch.object(
            cuda_backproject, "_detect_gpu_compute_cap",
            return_value=("NVIDIA Tesla P100", "60"),
        ),
        mock.patch.object(
            cuda_backproject, "_detect_so_arches",
            return_value=({"80", "86", "89", "90"}, set()),
        ),
        mock.patch.object(
            cuda_backproject, "_detect_nvcc_version",
            return_value="12.8",
        ),
    ):
        try:
            cuda_backproject._preflight_check(fake_so)
            assert False, "Should have raised"
        except RuntimeError as e:
            msg = str(e)
            assert "compute capability sm_60" in msg, "Missing GPU info"
            assert "RECOVAR_DISABLE_CUDA=1" in msg, "Missing bypass instruction"
            assert "2x slower" in msg, "Missing slowdown caveat"
            assert "make clean" in msg, "Missing rebuild instruction"

    # Reset for other tests
    cuda_backproject._preflight_ok = None


def test_preflight_passes_when_gpu_covered():
    """When the GPU's compute cap is in the .so, no error is raised."""
    from recovar import cuda_backproject

    cuda_backproject._preflight_ok = None
    fake_so = pathlib.Path("/tmp/fake_libcuda_backproject.so")

    with (
        mock.patch.object(
            cuda_backproject, "_detect_gpu_compute_cap",
            return_value=("NVIDIA A100", "80"),
        ),
        mock.patch.object(
            cuda_backproject, "_detect_so_arches",
            return_value=({"70", "75", "80", "86", "89", "90"}, {"75"}),
        ),
    ):
        cuda_backproject._preflight_check(fake_so)  # should not raise

    cuda_backproject._preflight_ok = None


def test_preflight_passes_with_ptx_fallback():
    """A PTX target <= GPU cap should be sufficient."""
    from recovar import cuda_backproject

    cuda_backproject._preflight_ok = None
    fake_so = pathlib.Path("/tmp/fake_libcuda_backproject.so")

    with (
        mock.patch.object(
            cuda_backproject, "_detect_gpu_compute_cap",
            return_value=("NVIDIA H200", "100"),  # future arch
        ),
        mock.patch.object(
            cuda_backproject, "_detect_so_arches",
            return_value=({"70", "75", "80", "86", "89", "90"}, {"75"}),
        ),
    ):
        cuda_backproject._preflight_check(fake_so)  # PTX 75 <= 100, should pass

    cuda_backproject._preflight_ok = None


def test_preflight_cuda13_nvcc_warning():
    """CUDA 13 + old GPU should include the toolkit note."""
    from recovar import cuda_backproject

    cuda_backproject._preflight_ok = None
    fake_so = pathlib.Path("/tmp/fake_libcuda_backproject.so")

    with (
        mock.patch.object(
            cuda_backproject, "_detect_gpu_compute_cap",
            return_value=("NVIDIA V100", "70"),
        ),
        mock.patch.object(
            cuda_backproject, "_detect_so_arches",
            return_value=({"80", "86", "89", "90"}, set()),
        ),
        mock.patch.object(
            cuda_backproject, "_detect_nvcc_version",
            return_value="13.0",
        ),
    ):
        try:
            cuda_backproject._preflight_check(fake_so)
            assert False, "Should have raised"
        except RuntimeError as e:
            msg = str(e)
            assert "nvcc 13.0" in msg, "Missing CUDA 13 note"
            assert "cuda-toolkit=12.4" in msg, "Missing toolkit install suggestion"

    cuda_backproject._preflight_ok = None

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


def _elf_with_fatbin(entries):
    """A minimal 64-bit ELF whose .nv_fatbin section holds one fat binary of ``entries`` [(kind, arch)]."""
    import struct

    body = b""
    for kind, arch in entries:
        payload = b"\0" * 16
        header = struct.pack("<HHII", kind, 0x0101, 64, len(payload)) + b"\0" * 16 + struct.pack("<I", arch)  # arch at byte 28
        body += header.ljust(64, b"\0") + payload
    fatbin = struct.pack("<IHHQ", 0xBA55ED50, 1, 16, len(body)) + body
    names = b"\0.shstrtab\0.nv_fatbin\0"
    shstr_off, fatbin_off = 64, 64 + len(names)
    shoff = (fatbin_off + len(fatbin) + 7) // 8 * 8
    ident = b"\x7fELF" + bytes([2, 1, 1]) + b"\0" * 9
    elf_header = ident + struct.pack("<HHIQQQIHHHHHH", 3, 62, 1, 0, 0, shoff, 0, 64, 0, 0, 64, 3, 1)

    def section(name, offset, size):
        return struct.pack("<IIQQQQIIQQ", name, 1, 0, 0, offset, size, 0, 0, 1, 0)

    sections = b"\0" * 64 + section(1, shstr_off, len(names)) + section(11, fatbin_off, len(fatbin))
    return (elf_header + names + fatbin).ljust(shoff, b"\0") + sections


def test_fatbin_arches_without_cuobjdump(tmp_path):
    """The fat binary gives the SASS and PTX targets when cuobjdump is not installed (a GPU node without the
    toolkit: Polar 413492 skipped the preflight and failed later with 'no kernel image', CUDA error 209)."""
    from recovar import cuda_backproject

    so = tmp_path / "libfake.so"
    so.write_bytes(_elf_with_fatbin([(2, 60), (2, 70), (2, 80), (1, 90)]))
    assert cuda_backproject._fatbin_arches(so) == ({"60", "70", "80"}, {"90"})
    with mock.patch.object(cuda_backproject, "_cuobjdump_arches", return_value=(set(), set())):
        assert cuda_backproject._detect_so_arches(so) == ({"60", "70", "80"}, {"90"})
    not_elf = tmp_path / "not_elf.so"
    not_elf.write_bytes(b"plain text")
    assert cuda_backproject._fatbin_arches(not_elf) == (set(), set())


def test_preflight_checks_every_library(tmp_path):
    """A covered first library does not exempt the next one (relax's library after recovar's)."""
    from recovar import cuda_backproject

    covered, uncovered = tmp_path / "libcovered.so", tmp_path / "libuncovered.so"
    covered.write_bytes(_elf_with_fatbin([(2, 60), (2, 80)]))
    uncovered.write_bytes(_elf_with_fatbin([(2, 80), (2, 90)]))
    cuda_backproject._preflight_ok = None
    with (
        mock.patch.object(cuda_backproject, "_detect_gpu_compute_cap", return_value=("Tesla P100", "60")),
        mock.patch.object(cuda_backproject, "_cuobjdump_arches", return_value=(set(), set())),
        mock.patch.object(cuda_backproject, "_detect_nvcc_version", return_value=None),
    ):
        cuda_backproject._preflight_check(covered)
        with pytest.raises(RuntimeError, match="libuncovered.so cannot run on your GPU") as err:
            cuda_backproject._preflight_check(uncovered, make_dir=tmp_path)
        assert f"cd {tmp_path}" in str(err.value)
    cuda_backproject._preflight_ok = None

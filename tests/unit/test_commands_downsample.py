"""
Unit tests for recovar.commands.downsample.

Covers:
  main()              – argument registration and validation
  _get_pixel_size()   – pixel-size extraction from MRC headers
  downsample_to_disk  – input validation (odd D, target > orig)
"""

import argparse
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from recovar.commands import downsample as ds_cmd
from recovar.data_io import downsample as ds_io

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _get_pixel_size
# ---------------------------------------------------------------------------


def test_get_pixel_size_from_mrc(tmp_path):
    """_get_pixel_size should read voxel_size.x from an MRC file."""
    import mrcfile

    mrc_path = tmp_path / "particles.mrcs"
    with mrcfile.new(str(mrc_path), overwrite=True) as m:
        m.set_data(np.zeros((4, 8, 8), dtype=np.float32))
        m.voxel_size = 1.75

    apix = ds_cmd._get_pixel_size(str(mrc_path))
    assert apix == pytest.approx(1.75, rel=1e-3)


def test_get_pixel_size_returns_none_for_unknown_extension(tmp_path):
    """Unsupported extensions should return None."""
    txt_path = tmp_path / "particles.txt"
    txt_path.write_text("dummy")
    assert ds_cmd._get_pixel_size(str(txt_path)) is None


def test_get_pixel_size_returns_none_for_mrc_with_zero_voxel_size(tmp_path):
    """An MRC with voxel_size=0 should return None."""
    import mrcfile

    mrc_path = tmp_path / "zero_vox.mrcs"
    with mrcfile.new(str(mrc_path), overwrite=True) as m:
        m.set_data(np.zeros((2, 4, 4), dtype=np.float32))
        m.voxel_size = 0.0

    assert ds_cmd._get_pixel_size(str(mrc_path)) is None


# ---------------------------------------------------------------------------
# downsample_to_disk – input validation
# ---------------------------------------------------------------------------


def test_downsample_to_disk_rejects_odd_target_D(tmp_path):
    """Target box size must be even."""
    with pytest.raises(ValueError, match="even"):
        ds_cmd.downsample_to_disk(
            particles_file="dummy.mrcs",
            target_D=65,
            outdir=str(tmp_path),
        )


def test_downsample_to_disk_uses_explicit_gpu_memory_cap(monkeypatch, tmp_path):
    class _FakeLoader:
        num_images = 2
        image_size = 64

        def get(self, indices):
            return np.zeros((len(indices), 64, 64), dtype=np.float32)

    calls = {}

    monkeypatch.setattr(ds_cmd, "_get_pixel_size", lambda _p: 1.5)
    monkeypatch.setattr(ds_cmd, "_write_output_star", lambda *args, **kwargs: None)
    monkeypatch.setattr(ds_io, "_gpu_available", lambda: True)

    def _record_batch_size(orig_D, gpu_memory_gb):
        calls["batch_args"] = (orig_D, gpu_memory_gb)
        return 4

    monkeypatch.setattr(ds_io, "get_downsample_batch_size", _record_batch_size)
    monkeypatch.setattr(ds_io, "downsample_images", lambda images, target_D, use_gpu=None: images[:, :target_D, :target_D])
    monkeypatch.setattr("recovar.data_io.image_loader.load_images", lambda *args, **kwargs: _FakeLoader())

    import recovar.utils.helpers as helpers

    monkeypatch.setattr(helpers, "get_gpu_memory_total", lambda: (_ for _ in ()).throw(AssertionError("unexpected")))

    ds_cmd.downsample_to_disk(
        particles_file="particles.star",
        target_D=32,
        outdir=str(tmp_path),
        gpu_memory_gb=12.0,
    )

    assert calls["batch_args"] == (64, 12.0)


def test_downsample_to_disk_retries_with_smaller_batch_on_gpu_oom(monkeypatch, tmp_path):
    class _FakeLoader:
        num_images = 5
        image_size = 64

        def get(self, indices):
            return np.zeros((len(indices), 64, 64), dtype=np.float32)

    batch_sizes = []
    state = {"failed": False}

    monkeypatch.setattr(ds_cmd, "_get_pixel_size", lambda _p: 1.5)
    monkeypatch.setattr(ds_cmd, "_write_output_star", lambda *args, **kwargs: None)
    monkeypatch.setattr(ds_io, "_gpu_available", lambda: True)
    monkeypatch.setattr(ds_io, "get_downsample_batch_size", lambda _orig_D, _gpu_memory_gb: 4)
    monkeypatch.setattr("recovar.data_io.image_loader.load_images", lambda *args, **kwargs: _FakeLoader())

    def _fake_downsample_images(images, target_D, use_gpu=None):
        batch_sizes.append(len(images))
        if len(images) == 4 and not state["failed"]:
            state["failed"] = True
            raise RuntimeError("RESOURCE_EXHAUSTED: out of memory")
        return images[:, :target_D, :target_D]

    monkeypatch.setattr(ds_io, "downsample_images", _fake_downsample_images)

    ds_cmd.downsample_to_disk(
        particles_file="particles.star",
        target_D=32,
        outdir=str(tmp_path),
        gpu_memory_gb=12.0,
    )

    assert batch_sizes == [4, 2, 2, 1]


# ---------------------------------------------------------------------------
# main – argument parsing
# ---------------------------------------------------------------------------


def test_main_registers_particles_positional():
    """'particles' is a required positional argument."""
    parser = argparse.ArgumentParser()
    parser.add_argument("particles")
    parser.add_argument("-D", "--target-D", type=int, required=True)
    parser.add_argument("-o", "--outdir", required=True)
    positionals = [a.dest for a in parser._actions if not a.option_strings]
    assert "particles" in positionals


def test_main_registers_target_D():
    parser = argparse.ArgumentParser()
    parser.add_argument("particles")
    parser.add_argument("-D", "--target-D", type=int, required=True)
    parser.add_argument("-o", "--outdir", required=True)
    actions = parser._option_string_actions
    assert "-D" in actions or "--target-D" in actions


def test_main_registers_outdir():
    parser = argparse.ArgumentParser()
    parser.add_argument("particles")
    parser.add_argument("-D", "--target-D", type=int, required=True)
    parser.add_argument("-o", "--outdir", required=True)
    actions = parser._option_string_actions
    assert "-o" in actions or "--outdir" in actions


def test_main_registers_batch_size_with_default():
    parser = argparse.ArgumentParser()
    parser.add_argument("particles")
    parser.add_argument("-D", "--target-D", type=int, required=True)
    parser.add_argument("-o", "--outdir", required=True)
    parser.add_argument("--batch-size", type=int, default=1000)
    action = parser._option_string_actions["--batch-size"]
    assert action.default == 1000


def test_main_registers_gpu_memory_cap():
    parser = argparse.ArgumentParser()
    parser.add_argument("particles")
    parser.add_argument("-D", "--target-D", type=int, required=True)
    parser.add_argument("-o", "--outdir", required=True)
    parser.add_argument("--gpu-gb", "--gpu-memory", dest="gpu_memory", type=float, default=None)
    action = parser._option_string_actions["--gpu-gb"]
    assert action.dest == "gpu_memory"


def test_main_exits_on_odd_target_D(monkeypatch, tmp_path):
    """main() should exit with code 1 when given an odd -D."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["downsample", "particles.mrcs", "-D", "65", "-o", str(tmp_path)],
    )
    with pytest.raises(SystemExit) as exc_info:
        ds_cmd.main()
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# _write_minimal_star
# ---------------------------------------------------------------------------


def test_write_minimal_star_creates_file(tmp_path):
    """_write_minimal_star should create a valid STAR file."""
    star_path = str(tmp_path / "particles.128.star")
    ds_cmd._write_minimal_star(
        star_path=star_path,
        mrcs_rel="particles.128.mrcs",
        target_D=128,
        new_apix=2.0,
        n_images=10,
    )
    assert (tmp_path / "particles.128.star").exists()

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
import pandas as pd
import pytest

from recovar.commands import downsample as ds_cmd
from recovar.data_io import downsample as ds_io
from recovar.data_io import metadata_readers as metadata_parsing
from recovar.data_io.starfile import StarFile, write_star

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


def test_build_project_downsample_cache_dir_is_stable(tmp_path):
    cache_dir = ds_cmd.build_project_downsample_cache_dir(
        project_root=str(tmp_path / "project"),
        particles_file=str(tmp_path / "Particles" / "ribosome.star"),
        target_D=128,
        datadir=str(tmp_path / "data"),
        strip_prefix="Extract/job001",
    )
    assert cache_dir.endswith("Cache/downsample/ribosome_d128_" + cache_dir.split("_")[-1])
    cache_dir_2 = ds_cmd.build_project_downsample_cache_dir(
        project_root=str(tmp_path / "project"),
        particles_file=str(tmp_path / "Particles" / "ribosome.star"),
        target_D=128,
        datadir=str(tmp_path / "data"),
        strip_prefix="Extract/job001",
    )
    assert cache_dir == cache_dir_2


def test_write_downsample_cache_metadata_creates_manifest(tmp_path):
    ds_dir = tmp_path / "Cache" / "downsample" / "ribosome_d128"
    ds_dir.mkdir(parents=True)
    ds_cmd.write_downsample_cache_metadata(
        ds_dir=str(ds_dir),
        particles_file=str(tmp_path / "particles.star"),
        target_D=128,
        datadir=str(tmp_path / "data"),
        strip_prefix="Extract/job001",
    )
    manifest = ds_dir / "cache.json"
    assert manifest.exists()
    payload = __import__("json").loads(manifest.read_text())
    assert payload["target_D"] == 128
    assert payload["strip_prefix"] == "Extract/job001"


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



def _make_relion31_star_input(tmp_path, orig_D=288, orig_apix=1.5, n_images=4):
    star_path = tmp_path / "input.star"
    optics = pd.DataFrame(
        {
            "_rlnOpticsGroup": [1],
            "_rlnOpticsGroupName": ["opticsGroup1"],
            "_rlnImagePixelSize": [orig_apix],
            "_rlnImageSize": [orig_D],
            "_rlnImageDimensionality": [2],
            "_rlnVoltage": [300.0],
            "_rlnSphericalAberration": [2.7],
            "_rlnAmplitudeContrast": [0.1],
        }
    )
    trans_px = np.array(
        [
            [3.25, -1.5],
            [-4.0, 2.75],
            [0.5, 0.25],
            [-1.125, -2.5],
        ],
        dtype=np.float64,
    )[:n_images]
    particles = pd.DataFrame(
        {
            "_rlnImageName": [f"{i + 1}@input.mrcs" for i in range(n_images)],
            "_rlnOpticsGroup": np.ones(n_images, dtype=int),
            "_rlnAngleRot": [10.0, 20.0, 30.0, 40.0][:n_images],
            "_rlnAngleTilt": [15.0, 25.0, 35.0, 45.0][:n_images],
            "_rlnAnglePsi": [5.0, 12.0, 18.0, 24.0][:n_images],
            "_rlnOriginXAngst": trans_px[:, 0] * orig_apix,
            "_rlnOriginYAngst": trans_px[:, 1] * orig_apix,
            "_rlnDefocusU": np.linspace(10000.0, 13000.0, n_images),
            "_rlnDefocusV": np.linspace(11000.0, 14000.0, n_images),
            "_rlnDefocusAngle": np.linspace(0.0, 90.0, n_images),
            "_rlnPhaseShift": np.linspace(0.0, 12.0, n_images),
        }
    )
    write_star(str(star_path), particles, optics)
    return str(star_path), trans_px, float(orig_apix), int(orig_D), n_images


def _make_cs_input(tmp_path, orig_D=288, orig_apix=1.5, n_images=4):
    cs_path = tmp_path / "input.cs"
    rotvecs = np.array(
        [
            [0.05, -0.02, 0.01],
            [-0.03, 0.04, 0.02],
            [0.01, 0.02, -0.05],
            [0.06, 0.01, 0.03],
        ],
        dtype=np.float32,
    )[:n_images]
    trans_px = np.array(
        [
            [3.25, -1.5],
            [-4.0, 2.75],
            [0.5, 0.25],
            [-1.125, -2.5],
        ],
        dtype=np.float32,
    )[:n_images]
    dtype = np.dtype(
        [
            ("blob/idx", np.int32),
            ("blob/path", "U200"),
            ("blob/shape", np.int32, (2,)),
            ("blob/psize_A", np.float32),
            ("alignments3D/pose", np.float32, (3,)),
            ("alignments3D/shift", np.float32, (2,)),
            ("ctf/df1_A", np.float32),
            ("ctf/df2_A", np.float32),
            ("ctf/df_angle_rad", np.float32),
            ("ctf/accel_kv", np.float32),
            ("ctf/cs_mm", np.float32),
            ("ctf/amp_contrast", np.float32),
            ("ctf/phase_shift_rad", np.float32),
        ]
    )
    data = np.zeros(n_images, dtype=dtype)
    data["blob/idx"] = np.arange(n_images, dtype=np.int32)
    data["blob/path"] = "input.mrcs"
    data["blob/shape"] = [orig_D, orig_D]
    data["blob/psize_A"] = np.float32(orig_apix)
    data["alignments3D/pose"] = rotvecs
    data["alignments3D/shift"] = trans_px
    data["ctf/df1_A"] = np.linspace(10000.0, 13000.0, n_images, dtype=np.float32)
    data["ctf/df2_A"] = np.linspace(11000.0, 14000.0, n_images, dtype=np.float32)
    data["ctf/df_angle_rad"] = np.deg2rad(np.linspace(0.0, 90.0, n_images)).astype(np.float32)
    data["ctf/accel_kv"] = 300.0
    data["ctf/cs_mm"] = 2.7
    data["ctf/amp_contrast"] = 0.1
    data["ctf/phase_shift_rad"] = np.deg2rad(np.linspace(0.0, 12.0, n_images)).astype(np.float32)
    with open(cs_path, "wb") as f:
        np.save(f, data)
    return str(cs_path), trans_px.astype(np.float64), float(orig_apix), int(orig_D), n_images


@pytest.mark.parametrize("target_D", [288, 192, 128, 96])
def test_write_output_star_preserves_star_metadata_across_downsampling(tmp_path, target_D):
    input_star, trans_px, orig_apix, orig_D, n_images = _make_relion31_star_input(tmp_path)
    expected_rot, expected_trans = metadata_parsing.parse_poses_from_star(input_star, target_D)
    expected_ctf = metadata_parsing.parse_ctf_from_star(input_star, target_D)

    out_star = tmp_path / f"particles.{target_D}.star"
    out_mrcs = tmp_path / f"particles.{target_D}.mrcs"
    new_apix = orig_apix * orig_D / float(target_D)

    ds_cmd._write_output_star(
        input_path=input_star,
        mrcs_path=str(out_mrcs),
        star_path=str(out_star),
        target_D=target_D,
        new_apix=new_apix,
        n_images=n_images,
    )

    sf = StarFile.load(str(out_star))
    assert np.all(sf.resolution == target_D)
    assert np.allclose(sf.apix, new_apix)
    assert np.allclose(sf.apix * sf.resolution, orig_apix * orig_D)

    got_rot, got_trans = metadata_parsing.parse_poses_from_star(str(out_star), target_D)
    got_ctf = metadata_parsing.parse_ctf_from_star(str(out_star), target_D)

    np.testing.assert_allclose(got_rot, expected_rot, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(got_trans, expected_trans, atol=1e-10, rtol=1e-12)
    np.testing.assert_allclose(got_trans, trans_px / float(orig_D), atol=1e-10, rtol=1e-12)
    np.testing.assert_allclose(got_ctf, expected_ctf, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(got_ctf[:, 0] * target_D, orig_apix * orig_D, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("target_D", [288, 192, 128, 96])
def test_write_output_star_preserves_cs_metadata_across_downsampling(tmp_path, target_D):
    input_cs, trans_px, orig_apix, orig_D, n_images = _make_cs_input(tmp_path)
    expected_rot, expected_trans = metadata_parsing.parse_poses_from_cs(input_cs, target_D)
    expected_ctf = metadata_parsing.parse_ctf_from_cs(input_cs, target_D)

    out_star = tmp_path / f"particles.{target_D}.star"
    out_mrcs = tmp_path / f"particles.{target_D}.mrcs"
    new_apix = orig_apix * orig_D / float(target_D)

    ds_cmd._write_output_star(
        input_path=input_cs,
        mrcs_path=str(out_mrcs),
        star_path=str(out_star),
        target_D=target_D,
        new_apix=new_apix,
        n_images=n_images,
    )

    sf = StarFile.load(str(out_star))
    assert np.all(sf.resolution == target_D)
    assert np.allclose(sf.apix, new_apix)
    assert np.allclose(sf.apix * sf.resolution, orig_apix * orig_D)

    got_rot, got_trans = metadata_parsing.parse_poses_from_star(str(out_star), target_D)
    got_ctf = metadata_parsing.parse_ctf_from_star(str(out_star), target_D)

    np.testing.assert_allclose(got_rot, expected_rot, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(got_trans, expected_trans, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(got_trans, trans_px / float(orig_D), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(got_ctf, expected_ctf, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(got_ctf[:, 0] * target_D, orig_apix * orig_D, atol=1e-6, rtol=1e-6)

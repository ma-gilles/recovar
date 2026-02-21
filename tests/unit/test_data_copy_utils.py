from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from recovar import starfile, utils
from recovar.utils_core import data_copy

pytestmark = pytest.mark.unit


def _make_basic_files(tmp_path):
    particles = tmp_path / "particles.mrcs"
    poses = tmp_path / "poses.pkl"
    ctf = tmp_path / "ctf.pkl"

    utils.write_mrc(str(particles), np.zeros((2, 8, 8), dtype=np.float32))
    rots = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
    trans = np.zeros((2, 2), dtype=np.float32)
    utils.pickle_dump((rots, trans), str(poses))
    utils.pickle_dump(np.zeros((2, 9), dtype=np.float32), str(ctf))
    return particles, poses, ctf


def test_copy_data_to_temp_folder_copies_and_rewrites_paths(tmp_path):
    particles, poses, ctf = _make_basic_files(tmp_path)
    temp_folder = tmp_path / "tmpcopy"
    args = SimpleNamespace(
        copy_to_folder=str(temp_folder),
        particles=str(particles),
        poses=str(poses),
        ctf=str(ctf),
        datadir=None,
        strip_prefix=None,
        mask=None,
        focus_mask=None,
        ind=None,
        tilt_ind=None,
        halfsets=None,
    )

    mapping = data_copy.copy_data_to_temp_folder(args)
    assert mapping is not None
    assert args.particles.startswith(str(temp_folder))
    assert args.poses.startswith(str(temp_folder))
    assert args.ctf.startswith(str(temp_folder))
    assert (temp_folder / particles.name).exists()
    assert (temp_folder / poses.name).exists()
    assert (temp_folder / ctf.name).exists()


def test_copy_data_to_temp_folder_cache_skips_redundant_copy(tmp_path):
    data_copy.clear_file_copy_cache()
    particles, poses, ctf = _make_basic_files(tmp_path)
    temp_folder = tmp_path / "tmpcopy"

    args1 = SimpleNamespace(
        copy_to_folder=str(temp_folder),
        particles=str(particles),
        poses=str(poses),
        ctf=str(ctf),
        datadir=None,
        strip_prefix=None,
        mask=None,
        focus_mask=None,
        ind=None,
        tilt_ind=None,
        halfsets=None,
    )
    data_copy.copy_data_to_temp_folder(args1)
    stats1 = data_copy.get_cache_stats()
    assert stats1["cached_files"] >= 3

    # Second call with same source files and temp folder should hit cache path.
    args2 = SimpleNamespace(
        copy_to_folder=str(temp_folder),
        particles=str(particles),
        poses=str(poses),
        ctf=str(ctf),
        datadir=None,
        strip_prefix=None,
        mask=None,
        focus_mask=None,
        ind=None,
        tilt_ind=None,
        halfsets=None,
    )
    data_copy.copy_data_to_temp_folder(args2)
    stats2 = data_copy.get_cache_stats()
    assert stats2["cached_files"] == stats1["cached_files"]


def test_copy_data_to_temp_folder_star_requires_datadir(tmp_path):
    stack = tmp_path / "stack.mrcs"
    utils.write_mrc(str(stack), np.zeros((2, 8, 8), dtype=np.float32))
    df = pd.DataFrame({"_rlnImageName": [f"1@{stack.name}", f"2@{stack.name}"]})
    star_path = tmp_path / "particles.star"
    starfile.write_star(str(star_path), data=df)
    _, poses, ctf = _make_basic_files(tmp_path)

    args = SimpleNamespace(
        copy_to_folder=str(tmp_path / "tmpcopy"),
        particles=str(star_path),
        poses=str(poses),
        ctf=str(ctf),
        datadir=None,
        strip_prefix=None,
        mask=None,
        focus_mask=None,
        ind=None,
        tilt_ind=None,
        halfsets=None,
    )
    with pytest.raises(RuntimeError, match="must also provide --datadir"):
        data_copy.copy_data_to_temp_folder(args)


def test_copy_data_to_temp_folder_star_copies_referenced_files(tmp_path):
    datadir = tmp_path / "data"
    datadir.mkdir()
    stack = datadir / "stack.mrcs"
    utils.write_mrc(str(stack), np.zeros((2, 8, 8), dtype=np.float32))

    # STAR references path with prefix that should be stripped.
    df = pd.DataFrame({"_rlnImageName": ["1@prefix/stack.mrcs", "2@prefix/stack.mrcs"]})
    star_path = tmp_path / "particles.star"
    starfile.write_star(str(star_path), data=df)
    _, poses, ctf = _make_basic_files(tmp_path)

    temp_folder = tmp_path / "tmpcopy"
    args = SimpleNamespace(
        copy_to_folder=str(temp_folder),
        particles=str(star_path),
        poses=str(poses),
        ctf=str(ctf),
        datadir=str(datadir),
        strip_prefix="prefix/",
        mask=None,
        focus_mask=None,
        ind=None,
        tilt_ind=None,
        halfsets=None,
    )
    data_copy.copy_data_to_temp_folder(args)
    assert (temp_folder / "stack.mrcs").exists()


def test_save_original_paths_info_and_cleanup(tmp_path):
    temp_folder = tmp_path / "tmpcopy"
    temp_folder.mkdir()
    mapping = {
        "original_particles": "/orig/particles.mrcs",
        "original_poses": "/orig/poses.pkl",
        "original_ctf": "/orig/ctf.pkl",
        "original_datadir": None,
        "temp_folder": str(temp_folder),
        "temp_particles": str(temp_folder / "particles.mrcs"),
    }
    outdir = tmp_path / "out"
    outdir.mkdir()

    data_copy.save_original_paths_info(mapping, str(outdir))
    info_file = outdir / "original_paths.txt"
    assert info_file.exists()
    text = info_file.read_text()
    assert "Original particles file" in text
    assert "temp_particles" in text

    data_copy.cleanup_temp_files(mapping)
    assert not temp_folder.exists()


def test_copy_specific_files_to_temp_and_clear_cache(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    d = tmp_path / "d"
    d.mkdir()
    (d / "k.txt").write_text("y")
    temp_folder = tmp_path / "tmpcopy"

    mapping = data_copy.copy_specific_files_to_temp(
        {"file_a": str(f), "dir_d": str(d)},
        str(temp_folder),
    )
    assert (temp_folder / "a.txt").exists()
    assert (temp_folder / "d" / "k.txt").exists()
    assert "temp_file_a" in mapping
    assert "temp_dir_d" in mapping

    data_copy.clear_file_copy_cache()
    assert data_copy.get_cache_stats()["cached_files"] == 0


def test_copy_data_to_temp_folder_returns_none_when_disabled(tmp_path):
    particles, poses, ctf = _make_basic_files(tmp_path)
    args = SimpleNamespace(
        copy_to_folder=None,
        particles=str(particles),
        poses=str(poses),
        ctf=str(ctf),
        datadir=None,
        strip_prefix=None,
        mask=None,
        focus_mask=None,
        ind=None,
        tilt_ind=None,
        halfsets=None,
    )
    assert data_copy.copy_data_to_temp_folder(args) is None


def test_copy_data_to_temp_folder_copies_datadir_tree(tmp_path):
    particles, poses, ctf = _make_basic_files(tmp_path)
    datadir = tmp_path / "datadir"
    datadir.mkdir()
    nested = datadir / "nested"
    nested.mkdir()
    (nested / "file.txt").write_text("ok")

    temp_folder = tmp_path / "tmpcopy"
    args = SimpleNamespace(
        copy_to_folder=str(temp_folder),
        particles=str(particles),
        poses=str(poses),
        ctf=str(ctf),
        datadir=str(datadir),
        strip_prefix=None,
        mask=None,
        focus_mask=None,
        ind=None,
        tilt_ind=None,
        halfsets=None,
    )
    mapping = data_copy.copy_data_to_temp_folder(args)
    assert "temp_datadir" in mapping
    assert (temp_folder / "datadir" / "nested" / "file.txt").exists()


def test_copy_specific_files_to_temp_respects_file_types_filter(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    d = tmp_path / "d"
    d.mkdir()
    (d / "k.txt").write_text("y")
    temp_folder = tmp_path / "tmpcopy"

    mapping = data_copy.copy_specific_files_to_temp(
        {"file_a": str(f), "dir_d": str(d)},
        str(temp_folder),
        file_types=["file_a"],
    )
    assert (temp_folder / "a.txt").exists()
    assert not (temp_folder / "d").exists()
    assert "temp_file_a" in mapping
    assert "temp_dir_d" not in mapping


def test_cleanup_temp_files_noop_on_none():
    # Should not raise.
    data_copy.cleanup_temp_files(None)


def test_copy_data_to_temp_folder_auto_uses_mkdtemp(tmp_path, monkeypatch):
    particles, poses, ctf = _make_basic_files(tmp_path)
    auto_dir = tmp_path / "auto_tmp"
    auto_dir.mkdir()
    monkeypatch.setattr(data_copy.tempfile, "mkdtemp", lambda prefix="": str(auto_dir))

    args = SimpleNamespace(
        copy_to_folder="auto",
        particles=str(particles),
        poses=str(poses),
        ctf=str(ctf),
        datadir=None,
        strip_prefix=None,
        mask=None,
        focus_mask=None,
        ind=None,
        tilt_ind=None,
        halfsets=None,
    )
    mapping = data_copy.copy_data_to_temp_folder(args)
    assert mapping["temp_folder"] == str(auto_dir)
    assert args.particles.startswith(str(auto_dir))


def test_copy_data_to_temp_folder_star_missing_reference_raises(tmp_path):
    datadir = tmp_path / "data"
    datadir.mkdir()
    # STAR references a file that does not exist in datadir.
    df = pd.DataFrame({"_rlnImageName": ["1@prefix/missing_stack.mrcs"]})
    star_path = tmp_path / "particles.star"
    starfile.write_star(str(star_path), data=df)
    _, poses, ctf = _make_basic_files(tmp_path)

    args = SimpleNamespace(
        copy_to_folder=str(tmp_path / "tmpcopy"),
        particles=str(star_path),
        poses=str(poses),
        ctf=str(ctf),
        datadir=str(datadir),
        strip_prefix="prefix/",
        mask=None,
        focus_mask=None,
        ind=None,
        tilt_ind=None,
        halfsets=None,
    )
    with pytest.raises(FileNotFoundError, match="File referenced in star file not found"):
        data_copy.copy_data_to_temp_folder(args)


def test_copy_data_to_temp_folder_copies_optional_mask_and_index_files(tmp_path):
    particles, poses, ctf = _make_basic_files(tmp_path)
    mask_path = tmp_path / "mask.mrc"
    focus_mask_path = tmp_path / "focus_mask.mrc"
    utils.write_mrc(str(mask_path), np.zeros((8, 8, 8), dtype=np.float32))
    utils.write_mrc(str(focus_mask_path), np.zeros((8, 8, 8), dtype=np.float32))

    ind_path = tmp_path / "ind.pkl"
    tilt_ind_path = tmp_path / "tilt_ind.pkl"
    halfsets_path = tmp_path / "halfsets.pkl"
    utils.pickle_dump(np.array([0, 1], dtype=np.int32), str(ind_path))
    utils.pickle_dump(np.array([0], dtype=np.int32), str(tilt_ind_path))
    utils.pickle_dump([np.array([0], dtype=np.int32), np.array([1], dtype=np.int32)], str(halfsets_path))

    temp_folder = tmp_path / "tmpcopy"
    args = SimpleNamespace(
        copy_to_folder=str(temp_folder),
        particles=str(particles),
        poses=str(poses),
        ctf=str(ctf),
        datadir=None,
        strip_prefix=None,
        mask=str(mask_path),
        focus_mask=str(focus_mask_path),
        ind=str(ind_path),
        tilt_ind=str(tilt_ind_path),
        halfsets=str(halfsets_path),
    )
    mapping = data_copy.copy_data_to_temp_folder(args)
    for name in [
        mask_path.name,
        focus_mask_path.name,
        ind_path.name,
        tilt_ind_path.name,
        halfsets_path.name,
    ]:
        assert (temp_folder / name).exists()
    assert "original_mask" in mapping and "temp_mask" in mapping
    assert "original_focus_mask" in mapping and "temp_focus_mask" in mapping
    assert "original_ind" in mapping and "temp_ind" in mapping
    assert "original_tilt_ind" in mapping and "temp_particle_ind" in mapping
    assert "original_halfsets" in mapping and "temp_halfsets" in mapping


def test_copy_data_from_pipeline_output_handles_none_input_args(monkeypatch):
    class _PO:
        def get(self, key):
            assert key == "input_args"
            return None

    assert data_copy.copy_data_from_pipeline_output(_PO(), "/tmp/whatever") is None


def test_copy_data_from_pipeline_output_from_path_delegates(monkeypatch, tmp_path):
    particles, poses, ctf = _make_basic_files(tmp_path)
    input_args = SimpleNamespace(
        copy_to_folder=None,
        particles=str(particles),
        poses=str(poses),
        ctf=str(ctf),
        datadir=None,
        strip_prefix=None,
        mask=None,
        focus_mask=None,
        ind=None,
        tilt_ind=None,
        halfsets=None,
    )

    class _PO:
        def get(self, key):
            assert key == "input_args"
            return input_args

    import recovar.output as output_mod
    monkeypatch.setattr(output_mod, "PipelineOutput", lambda _path: _PO())
    mapping = data_copy.copy_data_from_pipeline_output(str(tmp_path / "pipeline_out"), str(tmp_path / "copy_tmp"))
    assert mapping is not None
    assert input_args.copy_to_folder == str(tmp_path / "copy_tmp")


def test_cleanup_temp_files_ignores_rmtree_errors(tmp_path, monkeypatch):
    temp_folder = tmp_path / "tmpcopy"
    temp_folder.mkdir()
    mapping = {"temp_folder": str(temp_folder)}

    def _boom(_path):
        raise OSError("nope")

    monkeypatch.setattr(data_copy.shutil, "rmtree", _boom)
    # Should not raise.
    data_copy.cleanup_temp_files(mapping)

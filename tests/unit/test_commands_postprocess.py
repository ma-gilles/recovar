from pathlib import Path
import sys

import numpy as np
import pytest

from recovar.commands import postprocess

pytestmark = pytest.mark.unit


def _write_mrc(path: Path, *, voxel_size: float = 1.5):
    import mrcfile

    with mrcfile.new(str(path), overwrite=True) as m:
        m.set_data(np.zeros((8, 8, 8), dtype=np.float32))
        m.voxel_size = voxel_size


def test_find_halfmap2_uses_known_filename_pattern(tmp_path):
    half1 = tmp_path / "half1_unfil.mrc"
    half2 = tmp_path / "half2_unfil.mrc"
    _write_mrc(half1)
    _write_mrc(half2)

    found = postprocess.find_halfmap2(str(half1))
    assert found == str(half2)


def test_find_halfmap2_falls_back_to_common_name(tmp_path):
    half1 = tmp_path / "custom_input_name.mrc"
    half2 = tmp_path / "halfmap2.mrc"
    _write_mrc(half1)
    _write_mrc(half2)

    found = postprocess.find_halfmap2(str(half1))
    assert found == str(half2)


def test_find_volume_directories_prefers_vol_prefix(tmp_path):
    vol2 = tmp_path / "vol0002"
    vol1 = tmp_path / "vol0001"
    other = tmp_path / "something_else"
    vol2.mkdir()
    vol1.mkdir()
    other.mkdir()
    _write_mrc(other / "half1_unfil.mrc")

    found = postprocess.find_volume_directories(str(tmp_path))
    assert found == [str(vol1), str(vol2)]


def test_find_volume_directories_falls_back_to_halfmap_subdirs(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    c = tmp_path / "c"
    a.mkdir()
    b.mkdir()
    c.mkdir()
    _write_mrc(a / "half1_unfil.mrc")
    _write_mrc(c / "halfmap1.mrc")

    found = postprocess.find_volume_directories(str(tmp_path))
    assert found == [str(a), str(c)]


def test_get_voxel_size_from_mrc_roundtrip(tmp_path):
    mrc = tmp_path / "map.mrc"
    _write_mrc(mrc, voxel_size=2.25)
    assert postprocess.get_voxel_size_from_mrc(str(mrc)) == pytest.approx(2.25)


def test_estimate_bfactor_from_halfmaps_rejects_small_volumes():
    halfmap = np.zeros((16, 16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="Volume too small"):
        postprocess.estimate_bfactor_from_halfmaps(halfmap, voxel_size=1.0)


def test_main_batch_local_dispatches_to_local_handler(monkeypatch, tmp_path):
    in_dir = tmp_path / "batch_in"
    out_dir = tmp_path / "batch_out"
    in_dir.mkdir()
    captured = {}

    def _fake_batch_local(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"vol0000": {"success": True}}

    monkeypatch.setattr(postprocess, "batch_process_volumes_local", _fake_batch_local)
    monkeypatch.setattr(
        postprocess,
        "batch_process_volumes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("global batch path should not be used")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["postprocess", str(in_dir), "--batch", "--local", "--output", str(out_dir)],
    )

    rc = postprocess.main()
    assert rc == 0
    assert captured["args"][0] == str(in_dir)
    assert captured["args"][1] == str(out_dir)


def test_main_batch_requires_directory_input(monkeypatch, tmp_path):
    not_a_dir = tmp_path / "single.mrc"
    _write_mrc(not_a_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        ["postprocess", str(not_a_dir), "--batch", "--output", str(tmp_path / "out")],
    )

    rc = postprocess.main()
    assert rc == 1


def test_main_batch_global_dispatches_to_global_handler(monkeypatch, tmp_path):
    in_dir = tmp_path / "batch_in"
    out_dir = tmp_path / "batch_out"
    in_dir.mkdir()
    captured = {}

    def _fake_batch_global(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"vol0000": {"success": True}}

    monkeypatch.setattr(postprocess, "batch_process_volumes", _fake_batch_global)
    monkeypatch.setattr(
        postprocess,
        "batch_process_volumes_local",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("local batch path should not be used")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["postprocess", str(in_dir), "--batch", "--output", str(out_dir)],
    )

    rc = postprocess.main()
    assert rc == 0
    assert captured["args"][0] == str(in_dir)
    assert captured["args"][1] == str(out_dir)


def test_main_single_global_auto_halfmap2_success(monkeypatch, tmp_path):
    half1 = tmp_path / "half1_unfil.mrc"
    half2 = tmp_path / "half2_unfil.mrc"
    out = tmp_path / "out" / "filtered.mrc"
    _write_mrc(half1)
    _write_mrc(half2)
    calls = {}

    def _fake_postprocess_halfmaps(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return 3.25

    monkeypatch.setattr(postprocess, "postprocess_halfmaps", _fake_postprocess_halfmaps)
    monkeypatch.setattr(
        postprocess,
        "local_filter_halfmaps",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("local path should not be used")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["postprocess", str(half1), "--output", str(out)],
    )

    rc = postprocess.main()
    assert rc == 0
    assert calls["args"][0] == str(half1)
    assert calls["args"][1] == str(half2)
    assert calls["args"][3] == str(out)


def test_main_single_local_dispatches_to_local_filter(monkeypatch, tmp_path):
    half1 = tmp_path / "half1_unfil.mrc"
    half2 = tmp_path / "half2_unfil.mrc"
    out = tmp_path / "out" / "filtered.mrc"
    _write_mrc(half1)
    _write_mrc(half2)
    calls = {}

    def _fake_local_filter(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return 6.0

    monkeypatch.setattr(postprocess, "local_filter_halfmaps", _fake_local_filter)
    monkeypatch.setattr(
        postprocess,
        "postprocess_halfmaps",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("global path should not be used")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["postprocess", str(half1), "--output", str(out), "--local"],
    )

    rc = postprocess.main()
    assert rc == 0
    assert calls["args"][0] == str(half1)
    assert calls["args"][1] == str(half2)
    assert calls["args"][3] == str(out)


def test_main_single_returns_error_when_halfmap2_missing(monkeypatch, tmp_path):
    half1 = tmp_path / "lonely_halfmap.mrc"
    out = tmp_path / "filtered.mrc"
    _write_mrc(half1)
    monkeypatch.setattr(
        sys,
        "argv",
        ["postprocess", str(half1), "--output", str(out)],
    )

    rc = postprocess.main()
    assert rc == 1


def test_main_single_returns_error_when_apply_mask_missing(monkeypatch, tmp_path):
    half1 = tmp_path / "half1_unfil.mrc"
    half2 = tmp_path / "half2_unfil.mrc"
    out = tmp_path / "filtered.mrc"
    _write_mrc(half1)
    _write_mrc(half2)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "postprocess",
            str(half1),
            "--halfmap2",
            str(half2),
            "--output",
            str(out),
            "--apply-mask",
            str(tmp_path / "missing_mask.mrc"),
        ],
    )

    rc = postprocess.main()
    assert rc == 1


def test_batch_process_volumes_handles_success_skip_and_failure(monkeypatch, tmp_path):
    volumes_dir = tmp_path / "volumes"
    out_dir = tmp_path / "out"
    vol_ok = volumes_dir / "vol0000"
    vol_fail = volumes_dir / "vol0001"
    vol_skip = volumes_dir / "vol0002"
    vol_ok.mkdir(parents=True)
    vol_fail.mkdir(parents=True)
    vol_skip.mkdir(parents=True)

    ok_half1 = vol_ok / "half1_unfil.mrc"
    ok_half2 = vol_ok / "half2_unfil.mrc"
    fail_half1 = vol_fail / "half1_unfil.mrc"
    fail_half2 = vol_fail / "half2_unfil.mrc"
    _write_mrc(ok_half1)
    _write_mrc(ok_half2)
    _write_mrc(fail_half1)
    _write_mrc(fail_half2)

    def _fake_postprocess(half1, *_args, **_kwargs):
        if str(half1).endswith("vol0001/half1_unfil.mrc"):
            raise RuntimeError("boom")
        return 4.5

    monkeypatch.setattr(postprocess, "postprocess_halfmaps", _fake_postprocess)

    results = postprocess.batch_process_volumes(
        str(volumes_dir),
        str(out_dir),
        voxel_size=1.0,
    )

    assert results["vol0000"]["success"] is True
    assert results["vol0001"]["success"] is False
    assert "boom" in results["vol0001"]["error"]
    assert "vol0002" not in results  # skipped due to missing halfmap1


def test_batch_process_volumes_local_handles_success_skip_and_failure(monkeypatch, tmp_path):
    volumes_dir = tmp_path / "volumes_local"
    out_dir = tmp_path / "out_local"
    vol_ok = volumes_dir / "vol0000"
    vol_fail = volumes_dir / "vol0001"
    vol_skip = volumes_dir / "vol0002"
    vol_ok.mkdir(parents=True)
    vol_fail.mkdir(parents=True)
    vol_skip.mkdir(parents=True)

    ok_half1 = vol_ok / "half1_unfil.mrc"
    ok_half2 = vol_ok / "half2_unfil.mrc"
    fail_half1 = vol_fail / "half1_unfil.mrc"
    fail_half2 = vol_fail / "half2_unfil.mrc"
    _write_mrc(ok_half1)
    _write_mrc(ok_half2)
    _write_mrc(fail_half1)
    _write_mrc(fail_half2)

    def _fake_local_filter(half1, *_args, **_kwargs):
        if str(half1).endswith("vol0001/half1_unfil.mrc"):
            raise RuntimeError("local-boom")
        return 7.5

    monkeypatch.setattr(postprocess, "local_filter_halfmaps", _fake_local_filter)

    results = postprocess.batch_process_volumes_local(
        str(volumes_dir),
        str(out_dir),
        voxel_size=1.0,
    )

    assert results["vol0000"]["success"] is True
    assert results["vol0001"]["success"] is False
    assert "local-boom" in results["vol0001"]["error"]
    assert "vol0002" not in results  # skipped due to missing halfmap1


def test_batch_process_volumes_reads_voxel_size_when_missing(monkeypatch, tmp_path):
    volumes_dir = tmp_path / "volumes"
    out_dir = tmp_path / "out"
    vol = volumes_dir / "vol0000"
    vol.mkdir(parents=True)
    half1 = vol / "half1_unfil.mrc"
    half2 = vol / "half2_unfil.mrc"
    _write_mrc(half1)
    _write_mrc(half2)
    captured = {}

    monkeypatch.setattr(postprocess, "get_voxel_size_from_mrc", lambda _p: 2.25)

    def _fake_postprocess(_half1, _half2, voxel_size, output_path, **_kwargs):
        captured["voxel_size"] = voxel_size
        captured["output_path"] = output_path
        return 5.0

    monkeypatch.setattr(postprocess, "postprocess_halfmaps", _fake_postprocess)

    results = postprocess.batch_process_volumes(
        str(volumes_dir),
        str(out_dir),
        voxel_size=None,
    )

    assert results["vol0000"]["success"] is True
    assert captured["voxel_size"] == pytest.approx(2.25)
    assert captured["output_path"].endswith("vol0000_filtered.mrc")


def test_batch_process_volumes_local_reads_voxel_size_when_missing(monkeypatch, tmp_path):
    volumes_dir = tmp_path / "volumes_local2"
    out_dir = tmp_path / "out_local2"
    vol = volumes_dir / "vol0000"
    vol.mkdir(parents=True)
    half1 = vol / "half1_unfil.mrc"
    half2 = vol / "half2_unfil.mrc"
    _write_mrc(half1)
    _write_mrc(half2)
    captured = {}

    monkeypatch.setattr(postprocess, "get_voxel_size_from_mrc", lambda _p: 3.0)

    def _fake_local_filter(_half1, _half2, voxel_size, output_path, **_kwargs):
        captured["voxel_size"] = voxel_size
        captured["output_path"] = output_path
        return 8.0

    monkeypatch.setattr(postprocess, "local_filter_halfmaps", _fake_local_filter)

    results = postprocess.batch_process_volumes_local(
        str(volumes_dir),
        str(out_dir),
        voxel_size=None,
    )

    assert results["vol0000"]["success"] is True
    assert captured["voxel_size"] == pytest.approx(3.0)
    assert captured["output_path"].endswith("vol0000_local_filtered.mrc")


def test_local_filter_halfmaps_applies_mask_and_writes_expected_outputs(monkeypatch, tmp_path):
    half1_path = tmp_path / "half1.mrc"
    half2_path = tmp_path / "half2.mrc"
    fsc_mask_path = tmp_path / "fsc_mask.mrc"
    apply_mask_path = tmp_path / "apply_mask.mrc"
    output_path = tmp_path / "filtered.mrc"

    half1 = np.ones((4, 4, 4), dtype=np.float32)
    half2 = np.ones((4, 4, 4), dtype=np.float32) * 2
    fsc_mask = np.ones((4, 4, 4), dtype=np.float32)
    apply_mask = np.zeros((4, 4, 4), dtype=np.float32)
    apply_mask[:2] = 1.0
    writes = {}
    captured = {}

    def _fake_load_mrc(path):
        path = str(path)
        if path == str(half1_path):
            return half1.copy()
        if path == str(half2_path):
            return half2.copy()
        if path == str(fsc_mask_path):
            return fsc_mask.copy()
        if path == str(apply_mask_path):
            return apply_mask.copy()
        raise AssertionError(f"unexpected mrc path: {path}")

    def _fake_local_resolution(h1, h2, bfactor, voxel_size, **kwargs):
        captured["h1"] = np.array(h1)
        captured["h2"] = np.array(h2)
        captured["bfactor"] = bfactor
        captured["voxel_size"] = voxel_size
        captured["kwargs"] = kwargs
        filtered = np.ones_like(h1, dtype=np.float32) * 9.0
        local_res = np.array(h1, dtype=np.float32) + 5.0
        local_auc = np.ones_like(h1, dtype=np.float32) * 0.5
        return filtered, local_res, local_auc, None, None

    monkeypatch.setattr(postprocess.utils, "load_mrc", _fake_load_mrc)
    monkeypatch.setattr(postprocess.locres, "local_resolution", _fake_local_resolution)
    monkeypatch.setattr(
        postprocess.utils,
        "write_mrc",
        lambda path, arr, voxel_size=1.0: writes.setdefault(str(path), (np.array(arr), float(voxel_size))),
    )

    median = postprocess.local_filter_halfmaps(
        str(half1_path),
        str(half2_path),
        voxel_size=1.5,
        output_path=str(output_path),
        B_factor=12.0,
        fsc_mask_path=str(fsc_mask_path),
        apply_mask_path=str(apply_mask_path),
        estimate_B_factor=False,
        locres_sampling=20.0,
        locres_maskrad=10.0,
        locres_edgwidth=4.0,
        locres_minres=30.0,
        fsc_threshold=1 / 7,
        filter_edgewidth=2,
    )

    np.testing.assert_allclose(captured["h1"], half1 * apply_mask)
    np.testing.assert_allclose(captured["h2"], half2 * apply_mask)
    assert captured["bfactor"] == pytest.approx(12.0)
    assert captured["voxel_size"] == pytest.approx(1.5)
    assert writes[str(output_path)][0].shape == (4, 4, 4)
    assert str(output_path).replace(".mrc", "_local_resol.mrc") in writes
    assert str(output_path).replace(".mrc", "_local_auc.mrc") in writes
    expected_median = float(np.median((half1 * apply_mask) + 5.0))
    assert median == pytest.approx(expected_median)

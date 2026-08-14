from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_k1_half1_raw_accumulator import (
    _load_native_bpref,
    _load_recovar,
    _metric,
    _require_matching_bpref_stage,
)


def _write_bpref(path: Path, *, schema: str = "recovar-bpref-prejoin-v2") -> None:
    np.savez(
        path,
        schema=np.asarray(schema),
        iteration=np.int32(1),
        current_size=np.int32(56),
        padding_factor=np.int32(2),
        grid_size=np.int32(128),
        volume_shape=np.asarray((128, 128, 128), dtype=np.int32),
        mstep_accumulator_shape=np.asarray((123, 123, 123), dtype=np.int32),
        Ft_y_0=np.asarray([1 + 2j], dtype=np.complex64),
        Ft_y_1=np.asarray([3 + 4j], dtype=np.complex64),
        Ft_ctf_0=np.asarray([5], dtype=np.float32),
        Ft_ctf_1=np.asarray([6], dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("half", "expected_numerator", "expected_weight"),
    ((1, 1 + 2j, 5.0), (2, 3 + 4j, 6.0)),
)
def test_load_recovar_selects_requested_half(
    tmp_path: Path,
    half: int,
    expected_numerator: complex,
    expected_weight: float,
) -> None:
    path = tmp_path / "prejoin.npz"
    _write_bpref(path)

    loaded = _load_recovar(path, half=half)

    assert loaded["half"] == half
    assert loaded["numerator"].item() == expected_numerator
    assert loaded["weight"].item() == expected_weight
    assert loaded["stage"] == "pre_lowres_join"


def test_load_recovar_accepts_post_lowres_join_dump(tmp_path: Path) -> None:
    path = tmp_path / "postjoin.npz"
    _write_bpref(path, schema="recovar-bpref-accum-v2")

    loaded = _load_recovar(path, half=1)

    assert loaded["stage"] == "post_lowres_join"
    assert loaded["numerator"].item() == 1 + 2j


def test_load_recovar_rejects_invalid_half(tmp_path: Path) -> None:
    path = tmp_path / "prejoin.npz"
    _write_bpref(path)

    with pytest.raises(ValueError, match="half must be 1 or 2"):
        _load_recovar(path, half=0)


def test_load_native_bpref_state_v1(tmp_path: Path) -> None:
    path = tmp_path / "bpref_data.bin"
    shape = np.asarray((3, 3, 2), dtype=np.int64)
    values = np.arange(18, dtype=np.float64).astype(np.complex128).reshape(3, 3, 2)
    with path.open("wb") as stream:
        shape.tofile(stream)
        values.tofile(stream)

    loaded_shape, loaded_values = _load_native_bpref(path, value_dtype=np.complex128)

    assert np.array_equal(loaded_shape, shape)
    assert np.array_equal(loaded_values, values)


def test_metric_reports_exact_first_mismatch_telemetry() -> None:
    source = np.asarray([1.0, 2.0, 4.0], dtype=np.float32)
    target = np.asarray([1.0, 2.5, 4.0], dtype=np.float32)

    result = _metric(source, target, allow_sign=False)

    assert result["exact_equal"] is False
    assert result["mismatch_count"] == 1
    assert result["first_mismatch_flat_index"] == 1
    assert result["max_absolute"] == pytest.approx(0.5)


def test_require_matching_bpref_stage_rejects_cross_stage_comparison() -> None:
    _require_matching_bpref_stage("post_lowres_join", "post_lowres_join")

    with pytest.raises(ValueError, match="BPref stage mismatch"):
        _require_matching_bpref_stage("pre_lowres_join", "post_lowres_join")

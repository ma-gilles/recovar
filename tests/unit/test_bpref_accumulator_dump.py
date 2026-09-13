"""Accumulator captures remain readable by the existing boundary analyzer."""

import numpy as np
import pytest

from recovar.em.diagnostics.iteration import _save_bpref_accumulators
from scripts.analyze_k1_half1_raw_accumulator import _load_recovar


@pytest.mark.unit
@pytest.mark.parametrize(
    ("stage", "loaded_stage"),
    [("prejoin", "pre_lowres_join"), ("accum", "post_lowres_join")],
)
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("run_id", [None, "repeat-2"])
def test_dump_round_trip(tmp_path, monkeypatch, stage, loaded_stage, dtype, run_id):
    if run_id is None:
        monkeypatch.delenv("RECOVAR_BPREF_BOUNDARY_DUMP_RUN_ID", raising=False)
    else:
        monkeypatch.setenv("RECOVAR_BPREF_BOUNDARY_DUMP_RUN_ID", run_id)
    numerators = [np.array([1 + 2j, 3 - 4j], dtype=dtype), np.array([5j, 6], dtype=dtype)]
    weights = [np.array([7 + 8j, 9], dtype=dtype), np.array([10, 11j], dtype=dtype)]
    original = [value.copy() for value in numerators + weights]
    dump_dir = tmp_path / "new" / "capture"

    _save_bpref_accumulators(
        str(dump_dir),
        stage=stage,
        iteration=2,
        current_size=56,
        padding_factor=2,
        grid_size=128,
        voxel_size=4.25,
        volume_shape=(128, 128, 128),
        accumulator_shape=(123, 123, 123),
        Ft_y_0=numerators[0],
        Ft_y_1=numerators[1],
        Ft_ctf_0=weights[0],
        Ft_ctf_1=weights[1],
    )

    path = dump_dir / f"recovar_bpref_{stage}_it003.npz"
    assert list(dump_dir.iterdir()) == [path]
    with np.load(path, allow_pickle=False) as saved:
        assert saved["run_id"].item() == ("unset" if run_id is None else run_id)
        assert saved["iteration"].item() == 3
        for key in (
            "iteration",
            "current_size",
            "padding_factor",
            "grid_size",
            "volume_shape",
            "mstep_accumulator_shape",
        ):
            assert saved[key].dtype == np.int32
        assert saved["voxel_size"].dtype == np.float32
    for half in (1, 2):
        loaded = _load_recovar(path, half=half)
        assert loaded["stage"] == loaded_stage
        assert loaded["numerator"].dtype == numerators[half - 1].dtype
        assert loaded["weight"].dtype == weights[half - 1].real.dtype
        np.testing.assert_array_equal(loaded["numerator"], numerators[half - 1])
        np.testing.assert_array_equal(loaded["weight"], weights[half - 1].real)
    for value, before in zip(numerators + weights, original, strict=True):
        np.testing.assert_array_equal(value, before)

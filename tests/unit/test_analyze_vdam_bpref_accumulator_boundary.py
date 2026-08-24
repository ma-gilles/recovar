from pathlib import Path

import numpy as np
import pytest

from recovar.em.bpref_contribution_replay import BPrefAccumulatorReplay
from scripts.analyze_vdam_bpref_accumulator_boundary import (
    _geometry,
    _rank_particle_sources,
    _to_relion_bpref_frame,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_rank_particle_sources_aggregates_rows_and_orders_by_data_norm():
    rows = {
        "active_original_indices": np.asarray([9, 2, 9], dtype=np.int64),
        "active_summed": np.asarray([[3 + 4j, 0], [2 + 0j, 0], [0, 12 + 0j]]),
        "active_ctf_probs": np.asarray([[1.0, 2.0], [4.0, 0.0], [3.0, 0.0]]),
    }

    ranking = _rank_particle_sources(rows)

    assert [row["original_index"] for row in ranking] == [9, 2]
    assert ranking[0]["row_count"] == 2
    assert ranking[0]["data_l2"] == 13.0
    assert ranking[0]["weight_l1"] == 6.0


@pytest.mark.unit
def test_geometry_separates_parallel_and_orthogonal_candidate_error():
    control = np.zeros(2, dtype=np.float64)
    native = np.asarray([2.0, 0.0])
    candidate = np.asarray([1.0, 3.0])

    metric = _geometry(candidate, native, control)

    assert metric == {
        "candidate_projection_on_native_delta": 0.5,
        "candidate_orthogonal_over_native_delta": 1.5,
    }


@pytest.mark.unit
def test_relion_bpref_frame_conversion_applies_fft_sign_and_scales():
    replay = BPrefAccumulatorReplay(
        data=np.asarray([1 + 2j], dtype=np.complex128),
        weight=np.asarray([3.0], dtype=np.float64),
        backend="raw",
        order="execution",
        precision="complex128/float64",
        launch_topology="fixture",
    )

    converted = _to_relion_bpref_frame(replay, ori_size=4)

    np.testing.assert_array_equal(converted.data, np.asarray([-16 - 32j]))
    np.testing.assert_array_equal(converted.weight, np.asarray([768.0]))
    assert converted.backend == "raw_relion_bpref_frame"
    assert converted.order == replay.order
    assert converted.precision == replay.precision
    assert converted.launch_topology == replay.launch_topology


@pytest.mark.unit
def test_big_jit_bpref_capture_observes_production_tensors_without_disabling_path():
    source = (
        REPO_ROOT / "recovar/em/dense_single_volume/local_em_engine.py"
    ).read_text()

    use_big_jit_block = source.split("use_big_jit_buckets = (", 1)[1].split(")\n", 1)[0]
    assert "bpref_contribution_capture_active" not in use_big_jit_block
    assert "or bpref_contribution_capture_active" in source
    assert "big-JIT BPref contribution capture requires returned M-step tensors and scores" in source
    assert source.count("_maybe_dump_exact_local_bpref_contribution_rows(") >= 3

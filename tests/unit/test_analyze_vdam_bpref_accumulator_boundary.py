from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_vdam_bpref_accumulator_boundary import _geometry, _rank_particle_sources

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
def test_big_jit_bpref_capture_observes_production_tensors_without_disabling_path():
    source = (
        REPO_ROOT / "recovar/em/dense_single_volume/local_em_engine.py"
    ).read_text()

    use_big_jit_block = source.split("use_big_jit_buckets = (", 1)[1].split(")\n", 1)[0]
    assert "bpref_contribution_capture_active" not in use_big_jit_block
    assert "or bpref_contribution_capture_active" in source
    assert "big-JIT BPref contribution capture requires returned M-step tensors and scores" in source
    assert source.count("_maybe_dump_exact_local_bpref_contribution_rows(") >= 3

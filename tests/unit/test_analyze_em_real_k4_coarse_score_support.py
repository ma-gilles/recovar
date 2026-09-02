from __future__ import annotations

import numpy as np
import pytest

from scripts import analyze_em_real_k4_coarse_score_support as analyzer


def test_stable_top_count_reports_boundary_ties_without_changing_count():
    scores = np.asarray([5.0, 4.0, 4.0, 1.0])

    selected, report = analyzer._stable_top_count_mask(scores, 2)

    assert selected.tolist() == [True, True, False, False]
    assert report["requested_count"] == 2
    assert report["cutoff_equal_count"] == 2
    assert report["boundary_tie"] is True


def test_native_count_counterfactual_separates_cutoff_from_score_order():
    scores = np.asarray([9.0, 8.0, 7.0, 6.0])
    native = np.asarray([True, True, False, False])
    recovar = np.asarray([True, False, True, False])

    report = analyzer._analyze_score_support(scores, native, recovar)

    assert report["observed"]["jaccard"] == pytest.approx(1 / 3)
    counterfactual = report["native_count_stable_top_score_counterfactual"]
    assert counterfactual["exact"] is True
    assert counterfactual["boundary_tie"] is False
    native_only = report["score_categories"]["native_only"]
    assert native_only["margin_to_recovar_observed_cutoff"]["median"] == 1.0


def test_native_count_counterfactual_exposes_score_order_difference():
    scores = np.asarray([9.0, 6.0, 8.0, 7.0])
    native = np.asarray([True, True, False, False])
    recovar = np.asarray([True, False, True, False])

    report = analyzer._analyze_score_support(scores, native, recovar)

    assert report["native_count_stable_top_score_counterfactual"]["exact"] is False
    assert report["score_categories"]["native_only"]["descending_rank"]["median"] == 3.0


def test_aggregate_metrics_uses_microaveraged_jaccard():
    first = analyzer._support_metric(
        np.asarray([True, True, False]),
        np.asarray([True, False, True]),
    )
    second = analyzer._support_metric(
        np.asarray([True, False]),
        np.asarray([True, False]),
    )

    report = analyzer._aggregate_metrics([first, second])

    assert report["records"] == 2
    assert report["exact_records"] == 1
    assert report["intersection"] == 2
    assert report["union"] == 4
    assert report["jaccard"] == 0.5


def test_parse_indices_rejects_duplicates_and_empty_values():
    assert analyzer._parse_indices("7,9,82") == (7, 9, 82)
    with pytest.raises(analyzer.AnalysisError, match="invalid"):
        analyzer._parse_indices("7,7")
    with pytest.raises(analyzer.AnalysisError, match="empty"):
        analyzer._parse_indices("")


def test_dump_identity_uses_reduced_dataset_indices_not_stack_offsets():
    dumps = [
        {"original_index": 7, "stack_index_one_based": 126},
        {"original_index": 9, "stack_index_one_based": 133},
    ]

    indexed = analyzer._index_dumps_by_dataset_index(dumps, (7, 9))

    assert sorted(indexed) == [7, 9]
    with pytest.raises(analyzer.AnalysisError, match="reduced-dataset"):
        analyzer._index_dumps_by_dataset_index(dumps, (125, 132))


def test_dump_identity_rejects_duplicate_dataset_indices():
    dumps = [{"original_index": 7}, {"original_index": 7}]

    with pytest.raises(analyzer.AnalysisError, match="duplicate"):
        analyzer._index_dumps_by_dataset_index(dumps, (7,))


def test_causal_particle_ids_join_to_reordered_dataset_rows_via_stack_identity():
    dataset_stacks = np.asarray([1001, 126, 308, 133])
    causal_rows = {
        7: {"stack_index_one_based": 126},
        9: {"stack_index_one_based": 133},
        29: {"stack_index_one_based": 308},
        82: {"stack_index_one_based": 1001},
    }

    mapping = analyzer._causal_to_dataset_indices(
        dataset_stack_indices_one_based=dataset_stacks,
        causal_rows=causal_rows,
        expected_indices=(7, 9, 29, 82),
    )

    assert mapping == {7: 1, 9: 3, 29: 2, 82: 0}


def test_causal_particle_join_rejects_missing_and_duplicate_stack_identities():
    causal_rows = {7: {"stack_index_one_based": 126}}
    with pytest.raises(analyzer.AnalysisError, match="invalid or duplicated"):
        analyzer._causal_to_dataset_indices(
            dataset_stack_indices_one_based=np.asarray([126, 126]),
            causal_rows=causal_rows,
            expected_indices=(7,),
        )
    with pytest.raises(analyzer.AnalysisError, match="lacks causal stack"):
        analyzer._causal_to_dataset_indices(
            dataset_stack_indices_one_based=np.asarray([125]),
            causal_rows=causal_rows,
            expected_indices=(7,),
        )

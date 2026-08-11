import numpy as np

from scripts.analyze_em_k1_pass2_ab import GROUP_FIELDS, analyze, array_metric


def _write_panel(path, score_delta=0.0):
    values = {}
    for fields in GROUP_FIELDS.values():
        for field in fields:
            values[field] = np.asarray([1.0], dtype=np.float64)
    values.update(
        {
            "original_index": np.int64(7),
            "local_index": np.int64(3),
            "current_size": np.int64(56),
            "n_fine_trans": np.int64(1),
            "candidate_mask": np.asarray([[True]]),
            "reconstruction_mask": np.asarray([[True]]),
            "probs": np.asarray([[1.0]]),
            "reconstruction_probs": np.asarray([[1.0]]),
            "reconstruction_n_significant": np.int64(1),
            "scores_pre_prior": np.asarray([[2.0 + score_delta]]),
        }
    )
    np.savez(path, **values)


def test_pass2_ab_classifies_first_raw_score_difference(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    name = "pass2_orig000007_cs056.npz"
    _write_panel(left / name)
    _write_panel(right / name, score_delta=0.25)

    report = analyze(left, right, expected_count=1, current_size=56)

    assert report["first_unequal_group"] == "raw_score"
    assert report["fixed_metric"]["passing"] == len(GROUP_FIELDS) - 1
    metric = report["particles"][0]["groups"]["raw_score"]["fields"]["scores_pre_prior"]
    assert metric["max_abs_finite_delta"] == 0.25
    assert metric["correlation_used"] is False


def test_pass2_ab_treats_local_index_as_execution_coordinate(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    name = "pass2_orig000007_cs056.npz"
    _write_panel(left / name)
    _write_panel(right / name)
    with np.load(right / name) as archive:
        values = {key: archive[key] for key in archive.files}
    values["local_index"] = np.int64(99)
    np.savez(right / name, **values)

    report = analyze(left, right, expected_count=1, current_size=56)

    assert report["first_unequal_group"] == "all_fields_exact"
    assert report["particles"][0]["left_local_index"] == 3
    assert report["particles"][0]["right_local_index"] == 99


def test_array_metric_preserves_imaginary_differences():
    metric = array_metric(
        np.asarray([1.0 + 2.0j], dtype=np.complex64),
        np.asarray([1.0 + 3.0j], dtype=np.complex64),
    )

    assert metric["finite_delta_l2"] == 1.0
    assert metric["max_abs_finite_delta"] == 1.0

from __future__ import annotations

import struct
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import audit_em_real_k4_shared200_causal_replay as auditor
from scripts import launch_em_real_k4_shared200_causal_replay_slurm as launcher


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _passing_metrics() -> dict[str, float]:
    return {
        "candidate_tuple_exact_fraction": 1.0,
        "centered_raw_score_relative_l2": 0.0,
        "centered_combined_score_relative_l2": 0.0,
        "posterior_relative_l2": 0.0,
        "posterior_row_sum_abs_error": 0.0,
        "support_jaccard": 1.0,
        "winner_agreement": 1.0,
        "pmax_rmse": 0.0,
        "pmax_abs_error": 0.0,
        "assignment_accuracy": 1.0,
        "cross_engine_map_fsc_auc": 1.0,
        "native_control_map_fsc_auc": 1.0,
        "native_capture_inertness_map_fsc_auc": 1.0,
        "frozen_target_map_fsc_auc": 1.0,
        "native_assignment_inertness": 1.0,
        "frozen_target_assignment_accuracy": 1.0,
        "minimum_class_fraction": 0.25,
    }


def test_gate_evaluation_accepts_only_complete_fixed_scorecard():
    result = auditor.evaluate_gates(_passing_metrics(), launcher.THRESHOLDS)
    assert result["accepted"] is True
    assert not result["failures"]
    assert all(item["passed"] for item in result["gates"].values())


def test_gate_evaluation_reports_scale_sensitive_failure():
    metrics = _passing_metrics()
    metrics["posterior_relative_l2"] = 2.0 * launcher.THRESHOLDS["maximum_posterior_relative_l2"]
    result = auditor.evaluate_gates(metrics, launcher.THRESHOLDS)
    assert result["accepted"] is False
    assert result["gates"]["posterior_relative_l2"]["passed"] is False
    assert "posterior_relative_l2" in result["failures"][0]


def test_centered_error_accumulator_removes_only_scalar_offset():
    accumulator = auditor.ErrorAccumulator()
    accumulator.add(
        np.asarray([10.0, 12.0, 17.0]),
        np.asarray([1.0, 3.0, 8.0]),
        center=True,
    )
    assert accumulator.report()["relative_l2"] == pytest.approx(0.0)

    changed = auditor.ErrorAccumulator()
    changed.add(
        np.asarray([10.0, 12.0, 18.0]),
        np.asarray([1.0, 3.0, 8.0]),
        center=True,
    )
    assert changed.report()["relative_l2"] > 0.0


def test_native_scalar_diagnostics_reports_bounded_cross_replay_drift():
    joined = [
        {
            "weight_norm_bits": _bits(100.0 + offset),
            "significant_weight_bits": _bits(0.25 + offset / 400.0),
        }
        for offset in (0.0, 0.005, 0.01, 0.002)
    ]

    report = auditor._native_scalar_diagnostics(joined)

    assert report["weight_norm_distinct_bit_patterns"] == 4
    assert report["significant_weight_distinct_bit_patterns"] == 4
    assert 0.0 < report["maximum_relative_range"] < auditor.MAX_NATIVE_SCALAR_RELATIVE_RANGE
    assert report["maximum_allowed_relative_range"] == auditor.MAX_NATIVE_SCALAR_RELATIVE_RANGE

    joined[-1]["weight_norm_bits"] = _bits(101.0)
    with pytest.raises(auditor.AuditError, match="exceeds the sealed diagnostic ceiling"):
        auditor._native_scalar_diagnostics(joined)


def test_load_recovar_allows_absent_empty_enrichment_but_rejects_partial(tmp_path):
    path = tmp_path / "pass2.npz"
    base = {
        "original_index": np.asarray(3),
        "class_index": np.asarray(1),
        "current_size": np.asarray(auditor.CASE.current_size),
        "rotations": np.eye(3, dtype=np.float32)[None, ...],
        "candidate_mask": np.zeros((1, 2), dtype=bool),
        "scores_with_prior": np.zeros((1, 2), dtype=np.float32),
        "probs": np.zeros((1, 2), dtype=np.float32),
        "rotation_log_prior": np.zeros(1, dtype=np.float32),
        "translation_log_prior": np.zeros(2, dtype=np.float32),
    }
    np.savez(path, **base)

    values = auditor._load_recovar(path, subset_local_index=3, class_id=2)
    assert not bool(values["_reconstruction_capture_present"])

    np.savez(path, **base, reconstruction_probs=np.zeros((1, 2), dtype=np.float32))
    with pytest.raises(auditor.AuditError, match="reconstruction capture is partial"):
        auditor._load_recovar(path, subset_local_index=3, class_id=2)


def test_particle_mass_diagnostics_normalizes_once_across_every_class():
    joined = []
    for class_id, native, recovar, native_support, recovar_support in (
        (
            1,
            [[0.4, 0.1]],
            [[0.3, 0.2]],
            [[True, False]],
            [[True, False]],
        ),
        (
            2,
            [[0.2, 0.3]],
            [[0.1, 0.4]],
            [[True, True]],
            [[True, True]],
        ),
    ):
        native_array = np.asarray(native, dtype=np.float64)
        recovar_array = np.asarray(recovar, dtype=np.float64)
        native_mask = np.asarray(native_support, dtype=bool)
        recovar_mask = np.asarray(recovar_support, dtype=bool)
        joined.append(
            {
                "class_id": class_id,
                "native_posterior": native_array,
                "recovar_posterior": recovar_array,
                "native_reconstruction_posterior": np.where(native_mask, native_array, 0.0),
                "recovar_reconstruction_posterior": np.where(recovar_mask, recovar_array, 0.0),
                "native_support": native_mask,
                "recovar_support": recovar_mask,
            }
        )

    report = auditor._particle_mass_diagnostics(joined)

    assert report["native_full_mass"] == pytest.approx(1.0)
    assert report["recovar_full_mass"] == pytest.approx(1.0)
    assert report["native_retained_mass"] == pytest.approx(0.9)
    assert report["recovar_retained_mass"] == pytest.approx(0.8)
    assert report["recovar_joint_retained_normalization_factor"] == pytest.approx(1.25)
    assert [row["recovar_jointly_normalized_retained_mass"] for row in report["class_rows"]] == pytest.approx(
        [0.375, 0.625]
    )
    assert sum(row["recovar_jointly_normalized_retained_mass"] for row in report["class_rows"]) == pytest.approx(
        1.0
    )


def test_particle_mass_diagnostics_rejects_per_class_or_incomplete_topology():
    base = {
        "native_posterior": np.asarray([[1.0]]),
        "recovar_posterior": np.asarray([[1.0]]),
        "native_reconstruction_posterior": np.asarray([[1.0]]),
        "recovar_reconstruction_posterior": np.asarray([[1.0]]),
        "native_support": np.asarray([[True]]),
        "recovar_support": np.asarray([[True]]),
    }
    with pytest.raises(auditor.AuditError, match="complete and ordered"):
        auditor._particle_mass_diagnostics([{**base, "class_id": 2}])


def test_empty_sparse_support_join_requires_matching_explicit_sentinels(monkeypatch, tmp_path):
    factor_header = [0] * 64
    factor_header[9:13] = [1, 2, 117, 17]
    factor_header[20] = 1
    factor_header[21] = 2
    factor_header[25] = _bits(0.25)
    factor_header[26] = _bits(1.0)
    factor_header[46] = 1
    rotations = np.zeros(1, dtype=[("matrix", "<f4", (9,))])
    rotations["matrix"] = np.eye(3, dtype=np.float32).reshape(1, 9)
    factor = SimpleNamespace(
        geometry_only=True,
        empty_sparse_support=True,
        header=tuple(factor_header),
        rotations=rotations,
    )
    score_header = [0] * 48
    score_header[4:8] = [1, 2, 117, 17]
    score = SimpleNamespace(
        empty_sparse_support=True,
        header=tuple(score_header),
        candidates=np.empty(0, dtype=np.float32),
    )
    recovar = {
        "rotations": np.eye(3, dtype=np.float32)[None, ...],
        "candidate_mask": np.zeros((1, 2), dtype=bool),
        "probs": np.zeros((1, 2), dtype=np.float64),
        "reconstruction_probs": np.zeros((1, 2), dtype=np.float64),
        "reconstruction_mask": np.zeros((1, 2), dtype=bool),
        "relion_raw_diff2": np.zeros((1, 2), dtype=np.float32),
    }
    monkeypatch.setattr(auditor, "load_factor_capture", lambda _path: factor)
    monkeypatch.setattr(auditor, "load_fine_score_capture", lambda _path: score)
    monkeypatch.setattr(auditor, "_load_recovar", lambda *_args, **_kwargs: recovar)

    joined = auditor._join_class(
        stack=17,
        subset_local_index=0,
        class_id=2,
        factor_path=tmp_path / "factor.bin",
        score_path=tmp_path / "score.bin",
        pass2_path=tmp_path / "pass2.npz",
    )

    assert joined["empty_sparse_support"] is True
    assert joined["candidate_exact"] is True
    assert joined["native_raw"].size == 0
    assert joined["native_posterior"].shape == (1, 2)
    assert not np.any(joined["native_posterior"])

    score.empty_sparse_support = False
    with pytest.raises(auditor.AuditError, match="sentinels disagree"):
        auditor._join_class(
            stack=17,
            subset_local_index=0,
            class_id=2,
            factor_path=tmp_path / "factor.bin",
            score_path=tmp_path / "score.bin",
            pass2_path=tmp_path / "pass2.npz",
        )


def test_empty_sparse_support_join_rejects_nonzero_recovar_mass(monkeypatch, tmp_path):
    factor_header = [0] * 64
    factor_header[9:13] = [1, 2, 117, 17]
    factor_header[21] = 2
    factor_header[25] = _bits(0.25)
    factor_header[26] = _bits(1.0)
    score_header = [0] * 48
    score_header[4:8] = [1, 2, 117, 17]
    monkeypatch.setattr(
        auditor,
        "load_factor_capture",
        lambda _path: SimpleNamespace(
            geometry_only=True,
            empty_sparse_support=True,
            header=tuple(factor_header),
            rotations=np.empty(0, dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        auditor,
        "load_fine_score_capture",
        lambda _path: SimpleNamespace(
            empty_sparse_support=True,
            header=tuple(score_header),
            candidates=np.empty(0, dtype=np.float32),
        ),
    )
    recovar = {
        "rotations": np.eye(3, dtype=np.float32)[None, ...],
        "candidate_mask": np.zeros((1, 2), dtype=bool),
        "probs": np.asarray([[0.25, 0.0]], dtype=np.float64),
        "reconstruction_probs": np.zeros((1, 2), dtype=np.float64),
        "reconstruction_mask": np.zeros((1, 2), dtype=bool),
        "relion_raw_diff2": np.zeros((1, 2), dtype=np.float32),
    }
    monkeypatch.setattr(auditor, "_load_recovar", lambda *_args, **_kwargs: recovar)

    with pytest.raises(auditor.AuditError, match="posterior is nonzero"):
        auditor._join_class(
            stack=17,
            subset_local_index=0,
            class_id=2,
            factor_path=tmp_path / "factor.bin",
            score_path=tmp_path / "score.bin",
            pass2_path=tmp_path / "pass2.npz",
        )


def test_nonempty_native_join_reports_disjoint_empty_recovar_candidate_set(monkeypatch, tmp_path):
    factor_header = [0] * 64
    factor_header[9:13] = [1, 1, 116, 17]
    factor_header[21] = 2
    factor_header[25] = _bits(0.25)
    factor_header[26] = _bits(1.0)
    factor_header[45] = 1
    rotations = np.zeros(1, dtype=[("matrix", "<f4", (9,))])
    rotations["matrix"] = np.eye(3, dtype=np.float32).reshape(1, 9)
    factor = SimpleNamespace(
        geometry_only=True,
        empty_sparse_support=False,
        header=tuple(factor_header),
        rotations=rotations,
    )
    score_header = [0] * 48
    score_header[4:8] = [1, 1, 116, 17]
    candidate_dtype = np.dtype(
        [
            ("flags", "<u4"),
            ("rotation_local", "<i4"),
            ("translation_id", "<i4"),
            ("post_exponent_weight", "<f4"),
            ("raw_diff2", "<f4"),
            ("combined_preexponent", "<f4"),
        ]
    )
    candidates = np.zeros(1, dtype=candidate_dtype)
    candidates["flags"] = auditor.ACTIVE
    candidates["rotation_local"] = 0
    candidates["translation_id"] = 1
    candidates["post_exponent_weight"] = 1.0
    score = SimpleNamespace(
        empty_sparse_support=False,
        header=tuple(score_header),
        candidates=candidates,
    )
    recovar = {
        "rotations": np.eye(3, dtype=np.float32)[None, ...],
        "candidate_mask": np.zeros((1, 2), dtype=bool),
        "probs": np.zeros((1, 2), dtype=np.float64),
        "scores_with_prior": np.zeros((1, 2), dtype=np.float32),
    }
    monkeypatch.setattr(auditor, "load_factor_capture", lambda _path: factor)
    monkeypatch.setattr(auditor, "load_fine_score_capture", lambda _path: score)
    monkeypatch.setattr(auditor, "_load_recovar", lambda *_args, **_kwargs: recovar)

    joined = auditor._join_class(
        stack=17,
        subset_local_index=0,
        class_id=1,
        factor_path=tmp_path / "factor.bin",
        score_path=tmp_path / "score.bin",
        pass2_path=tmp_path / "pass2.npz",
    )

    assert joined["empty_sparse_support"] is False
    assert joined["candidate_exact"] is False
    assert joined["candidate_intersection"] == 0
    assert joined["candidate_union"] == 1
    assert joined["native_raw"].size == joined["recovar_raw"].size == 0
    assert joined["native_posterior"][0, 1] == pytest.approx(1.0)
    assert not np.any(joined["recovar_posterior"])
    assert joined["recovar_reconstruction_capture_present"] is False


def test_exact_keyed_paths_rejects_duplicate_and_missing(tmp_path):
    paths = [tmp_path / "1-1", tmp_path / "1-2"]
    expected = {(1, 1), (1, 2)}
    result = auditor._exact_keyed_paths(
        paths,
        key=lambda path: tuple(int(value) for value in path.name.split("-")),
        expected=expected,
        label="unit",
    )
    assert set(result) == expected

    with pytest.raises(auditor.AuditError, match="duplicate"):
        auditor._exact_keyed_paths(
            [paths[0], paths[0]],
            key=lambda path: tuple(int(value) for value in path.name.split("-")),
            expected=expected,
            label="unit",
        )
    with pytest.raises(auditor.AuditError, match="topology differs"):
        auditor._exact_keyed_paths(
            paths[:1],
            key=lambda path: tuple(int(value) for value in path.name.split("-")),
            expected=expected,
            label="unit",
        )


def test_inventory_is_exact_across_every_stack_and_class(monkeypatch, tmp_path):
    monkeypatch.setattr(
        auditor,
        "CASE",
        SimpleNamespace(K=2, particle_count=2, current_size=56),
    )
    for class_id in (1, 2):
        directory = tmp_path / f"native/class{class_id}/factors"
        directory.mkdir(parents=True)
        for stack in (3, 7):
            (directory / f"part{stack + 20}_stack{stack}_img0_class{class_id}.bpre-v2.bin").touch()
            (directory / f"part{stack + 20}_stack{stack}_class{class_id}.fine-score-v1.bin").touch()
    for arm in ("control_a", "control_b", "class1", "class2", "class3", "class4"):
        output = tmp_path / f"native/{arm}/output"
        output.mkdir(parents=True)
        (output / "run_it001_data.star").touch()
        for class_id in (1, 2):
            (output / f"run_it001_class{class_id:03d}.mrc").touch()
    pass2 = tmp_path / "recovar/pass2"
    pass2.mkdir(parents=True)
    for subset_local_index, stack in enumerate((3, 7)):
        for class_id in (1, 2):
            (pass2 / f"pass2_orig{subset_local_index:06d}_class{class_id:03d}_cs056.npz").touch()

    assignments = {3: 1, 7: 2}
    frozen_assignments = {3: 2, 7: 2}
    monkeypatch.setattr(auditor, "_native_assignments", lambda _path: assignments)
    inventory = auditor.discover_inventory(
        tmp_path,
        [3, 7],
        reference_assignments=frozen_assignments,
    )
    assert inventory["counts"] == {
        "native_factors": 4,
        "native_fine_scores": 4,
        "recovar_pass2": 4,
        "native_data_stars": 6,
    }
    assert inventory["native_assignment_inertness"] == 1.0
    assert inventory["frozen_target_assignment_accuracy"] == 0.5
    assert inventory["pass2"][(3, 1)].name == "pass2_orig000000_class001_cs056.npz"
    assert inventory["pass2"][(7, 1)].name == "pass2_orig000001_class001_cs056.npz"

    (pass2 / "pass2_orig000000_class002_cs056.npz").unlink()
    with pytest.raises(auditor.AuditError):
        auditor.discover_inventory(
            tmp_path,
            [3, 7],
            reference_assignments=frozen_assignments,
        )

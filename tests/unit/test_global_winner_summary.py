from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from recovar.em.global_winner_analysis import (
    analyze_summaries,
    load_recovar_summary,
    load_relion_summary,
)
from recovar.em.global_winner_summary import (
    MAX_SUPPORTED_BYTES,
    maybe_dump_global_winner_summary,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recovar_env(monkeypatch, path: Path, *, n_images: int = 4, iteration: int = 1):
    values = {
        "RECOVAR_GLOBAL_WINNER_SUMMARY_PATH": str(path),
        "RECOVAR_GLOBAL_WINNER_SUMMARY_ITERATION": str(iteration),
        "RECOVAR_GLOBAL_WINNER_SUMMARY_EXPECTED_PARTICLES": str(n_images),
        "RECOVAR_GLOBAL_WINNER_SUMMARY_EXPECTED_CLASSES": "4",
        "RECOVAR_GLOBAL_WINNER_SUMMARY_MAX_BYTES": str(MAX_SUPPORTED_BYTES),
        "RECOVAR_GLOBAL_WINNER_SUMMARY_RUN_ID": "unit-test",
        "RECOVAR_GLOBAL_WINNER_SUMMARY_SOURCE_ID": "a" * 64,
        "RECOVAR_GLOBAL_WINNER_SUMMARY_EXECUTABLE_SHA256": "b" * 64,
        "RECOVAR_GLOBAL_WINNER_SUMMARY_GPU_UUID": "GPU-unit-test",
        "RECOVAR_GLOBAL_WINNER_SUMMARY_INPUT_MANIFEST_SHA256": "c" * 64,
        "RECOVAR_GLOBAL_WINNER_SUMMARY_DISPATCH_ORACLE_SHA256": "d" * 64,
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def _full_stats(n_images: int = 4):
    scores = np.asarray(
        [
            [4.0, 1.0, 1.0, 1.0],
            [3.0, 4.0, 2.0, 2.0],
            [2.0, 3.0, 4.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
        ],
        dtype=np.float32,
    )[:, :n_images]
    best_poses = np.tile(np.arange(n_images, dtype=np.int32), (4, 1))
    return {
        "class_best_log_score_per_image": scores,
        "class_second_best_log_score_per_image": scores - np.float32(0.5),
        "class_hard_assignments": best_poses,
        "class_second_hard_assignments": (best_poses + 1) % 6,
        "class_log_evidence_per_image": scores.astype(np.float64) + 0.25,
        "class_assignments": np.argmax(scores, axis=0).astype(np.int32),
        "normalization_log_z": np.arange(n_images, dtype=np.float64) + 10.0,
    }


def _dataset(original_indices, *, dataset_indices=None):
    original_indices = np.asarray(original_indices, dtype=np.int64)

    def original_image_indices_from_local(local_indices):
        return original_indices[np.asarray(local_indices, dtype=np.int64)]

    return SimpleNamespace(
        dataset_indices=(
            np.arange(original_indices.size, dtype=np.int64)
            if dataset_indices is None
            else np.asarray(dataset_indices, dtype=np.int64)
        ),
        original_image_indices_from_local=original_image_indices_from_local,
    )


def test_recovar_summary_round_trip_and_semantics(monkeypatch, tmp_path):
    path = tmp_path / "recovar.npz"
    _recovar_env(monkeypatch, path)
    output = maybe_dump_global_winner_summary(
        experiment_dataset=_dataset(np.arange(4, dtype=np.int64)),
        full_stats=_full_stats(),
        n_classes=4,
        n_rotations=3,
        n_translations=2,
        iteration=1,
    )
    assert output == path.resolve()
    assert path.stat().st_size < MAX_SUPPORTED_BYTES
    summary = load_recovar_summary(path, label="recovar_a")
    assert summary.metadata["raw_score_semantics"].startswith("per-class best normalized-CC")
    assert summary.class_log_evidence is not None
    assert summary.global_log_z is not None
    np.testing.assert_array_equal(summary.winner, [0, 1, 2, 3])
    np.testing.assert_array_equal(summary.class_posterior_mass.sum(axis=1), 1.0)


def test_recovar_summary_uses_original_image_mapping_not_dataset_indices(monkeypatch, tmp_path):
    path = tmp_path / "recovar.npz"
    _recovar_env(monkeypatch, path)
    original_indices = np.asarray([17, 3, 29, 11], dtype=np.int64)
    maybe_dump_global_winner_summary(
        experiment_dataset=_dataset(original_indices, dataset_indices=[0, 1, 2, 3]),
        full_stats=_full_stats(),
        n_classes=4,
        n_rotations=3,
        n_translations=2,
        iteration=1,
    )
    summary = load_recovar_summary(path, label="recovar_a")
    np.testing.assert_array_equal(summary.identity, original_indices)


def test_recovar_summary_preserves_actual_winner_when_stored_float32_scores_tie(monkeypatch, tmp_path):
    path = tmp_path / "recovar_tie.npz"
    _recovar_env(monkeypatch, path)
    full_stats = _full_stats()
    full_stats["class_best_log_score_per_image"][1, 0] = full_stats["class_best_log_score_per_image"][0, 0]
    full_stats["class_assignments"][0] = 1
    maybe_dump_global_winner_summary(
        experiment_dataset=_dataset(np.arange(4)),
        full_stats=full_stats,
        n_classes=4,
        n_rotations=3,
        n_translations=2,
        iteration=1,
    )
    summary = load_recovar_summary(path, label="recovar_tie")
    assert summary.winner[0] == 1
    assert summary.runner_up[0] == 0
    assert summary.margin[0] == 0.0


@pytest.mark.parametrize("indices", [np.asarray([0, 1, 1, 3]), np.asarray([0, 1, 2])])
def test_recovar_summary_rejects_duplicate_or_missing_original_identity(monkeypatch, tmp_path, indices):
    path = tmp_path / "recovar.npz"
    _recovar_env(monkeypatch, path)
    with pytest.raises(RuntimeError, match="original identit|one original identity"):
        maybe_dump_global_winner_summary(
            experiment_dataset=_dataset(indices),
            full_stats=_full_stats(),
            n_classes=4,
            n_rotations=3,
            n_translations=2,
            iteration=1,
        )


def test_recovar_summary_rejects_unexpected_k_or_n(monkeypatch, tmp_path):
    path = tmp_path / "recovar.npz"
    _recovar_env(monkeypatch, path, n_images=5)
    with pytest.raises(RuntimeError, match="shape"):
        maybe_dump_global_winner_summary(
            experiment_dataset=_dataset(np.arange(4)),
            full_stats=_full_stats(),
            n_classes=4,
            n_rotations=3,
            n_translations=2,
            iteration=1,
        )
    monkeypatch.setenv("RECOVAR_GLOBAL_WINNER_SUMMARY_EXPECTED_CLASSES", "3")
    with pytest.raises(RuntimeError, match="K=4"):
        maybe_dump_global_winner_summary(
            experiment_dataset=_dataset(np.arange(4)),
            full_stats=_full_stats(),
            n_classes=4,
            n_rotations=3,
            n_translations=2,
            iteration=1,
        )


def test_recovar_loader_rejects_artifact_over_hard_cap(tmp_path):
    path = tmp_path / "oversized.npz"
    with path.open("wb") as handle:
        handle.truncate(MAX_SUPPORTED_BYTES + 1)
    with pytest.raises(ValueError, match="exceeds"):
        load_recovar_summary(path, label="oversized")


def test_recovar_writer_enforces_configured_artifact_cap(monkeypatch, tmp_path):
    path = tmp_path / "recovar.npz"
    _recovar_env(monkeypatch, path)
    monkeypatch.setenv("RECOVAR_GLOBAL_WINNER_SUMMARY_MAX_BYTES", "1")
    with pytest.raises(RuntimeError, match="exceeding cap"):
        maybe_dump_global_winner_summary(
            experiment_dataset=_dataset(np.arange(4)),
            full_stats=_full_stats(),
            n_classes=4,
            n_rotations=3,
            n_translations=2,
            iteration=1,
        )
    assert not path.exists()


def _write_relion_fixture(tmp_path: Path):
    star = tmp_path / "run_it001_data.star"
    star.write_text(
        "data_particles\n\nloop_\n_rlnImageName #1\n_rlnClassNumber #2\n"
        "1@particles.mrcs 1\n2@particles.mrcs 2\n3@particles.mrcs 3\n4@particles.mrcs 4\n"
    )
    input_manifest = tmp_path / "fixture_input.star"
    input_manifest.write_text("# immutable unit-test fixture manifest\n")
    executable = tmp_path / "relion_refine_mpi"
    executable.write_bytes(b"unit-test-relion-binary")
    dispatch = tmp_path / "dispatch.tsv"
    dispatch.write_text("# RELION_DISPATCH_LOG_SCHEMA_V2\n2\t1\t1\t0\t2\n2\t1\t1\t1\t0\n2\t1\t1\t2\t3\n2\t1\t1\t3\t1\n")
    schedule = tmp_path / "dispatch_schedule.npz"
    np.savez(
        schedule,
        schema_version=np.asarray(3),
        relion_iterations=np.asarray([1]),
        owner_by_sorted_position=np.asarray([[0, 0, 0, 0]]),
        original_particle_id_by_sorted_position=np.asarray([[2, 0, 3, 1]]),
    )
    summary_dir = tmp_path / "relion_summary"
    summary_dir.mkdir()
    shard = summary_dir / "rank1.tsv"
    metadata = {
        "schema": "k4_global_winner_summary_v1",
        "run_id": "unit-test",
        "source_id": "a" * 64,
        "executable_sha256": _sha256(executable),
        "gpu_uuid": "GPU-unit-test",
        "input_manifest_sha256": _sha256(input_manifest),
        "dispatch_oracle_sha256": _sha256(schedule),
        "iteration": "1",
        "expected_particles": "4",
        "expected_classes": "4",
        "max_bytes": str(MAX_SUPPORTED_BYTES),
        "score_mode": "firstiter_cc_raw_diff2_lower_is_better",
        "raw_score_semantics": "per-class minimum native diff2 before priors and WTA",
        "total_score_semantics": "identical to raw diff2 because firstiter_cc bypasses priors",
        "posterior_semantics": "post-firstiter_cc one-hot class mass",
        "winner_semantics": "actual device getArgMin of native float32 joint class-pose scores before WTA",
        "support_semantics": "post-firstiter_cc exactly one global class-pose sample per particle",
        "pre_wta_support_semantics": "all coarse candidates scored; no posterior threshold before WTA",
        "significant_count_semantics": "post-WTA global class-pose support cardinality; exactly one",
        "within_class_runner_up_semantics": (
            "second-lowest distinct class-local coarse pose diff2 before priors and WTA"
        ),
        "evidence_availability": "unavailable",
        "evidence_unavailable_reason": "firstiter_cc bypasses exponentiation and logsumexp evidence",
    }
    columns = [
        "schema_version",
        "iteration",
        "mpi_rank",
        "part_id_zero_based",
        "class_min_zero_based",
        "class_max_zero_based",
        "nr_dir",
        "nr_psi",
        "nr_trans",
        "score_element_bytes",
        "winner_class_zero_based",
        "runner_up_class_zero_based",
        "winner_flat_index",
        "runner_up_flat_index",
        "winner_score",
        "runner_up_score",
        "winner_margin",
        "significant_count",
    ]
    for class_index in range(4):
        columns.extend(
            [
                f"class{class_index}_best_flat",
                f"class{class_index}_best_raw_diff2_pre_prior",
                f"class{class_index}_best_total_diff2",
                f"class{class_index}_second_flat",
                f"class{class_index}_second_raw_diff2_pre_prior",
                f"class{class_index}_second_total_diff2",
                f"class{class_index}_within_pose_margin",
            ]
        )
    columns.extend(f"class{class_index}_posterior_mass" for class_index in range(4))
    lines = [*(f"# {key}={value}" for key, value in metadata.items()), "\t".join(columns)]
    for part_id in range(4):
        scores = np.asarray([5.0, 6.0, 7.0, 8.0], dtype=np.float32) + part_id
        scores = np.roll(scores, part_id)
        winner = int(np.argmin(scores))
        masked = scores.copy()
        masked[winner] = np.inf
        runner = int(np.argmin(masked))
        stride = 6
        row = [
            1,
            1,
            1,
            part_id,
            0,
            3,
            2,
            1,
            3,
            4,
            winner,
            runner,
            winner * stride,
            runner * stride,
            scores[winner],
            scores[runner],
            np.float32(scores[runner] - scores[winner]),
            1,
        ]
        for class_index in range(4):
            second_score = np.float32(scores[class_index] + 0.5)
            row.extend(
                [
                    class_index * stride,
                    scores[class_index],
                    scores[class_index],
                    class_index * stride + 1,
                    second_score,
                    second_score,
                    np.float32(0.5),
                ]
            )
        row.extend(int(class_index == winner) for class_index in range(4))
        lines.append("\t".join(map(str, row)))
    shard.write_text("\n".join(lines) + "\n")
    return summary_dir, star, input_manifest, executable, dispatch, schedule, shard


def test_relion_summary_round_trip_and_evidence_limitation(tmp_path):
    summary_dir, star, input_manifest, executable, dispatch, schedule, _shard = _write_relion_fixture(tmp_path)
    summary = load_relion_summary(
        summary_dir,
        data_star=star,
        input_manifest=input_manifest,
        executable=executable,
        dispatch_log=dispatch,
        dispatch_schedule=schedule,
        label="relion_a",
    )
    assert summary.class_log_evidence is None
    assert summary.global_log_z is None
    np.testing.assert_array_equal(summary.identity, np.arange(4))
    np.testing.assert_array_equal(summary.class_posterior_mass.sum(axis=1), 1.0)


def test_relion_summary_rejects_dispatch_manifest_mismatch(tmp_path):
    summary_dir, star, input_manifest, executable, dispatch, schedule, _shard = _write_relion_fixture(tmp_path)
    with np.load(schedule) as payload:
        arrays = {key: payload[key] for key in payload.files}
    arrays["owner_by_sorted_position"] = np.asarray([[0, 1, 0, 0]])
    np.savez(schedule, **arrays)
    with pytest.raises(ValueError, match="manifest SHA-256"):
        load_relion_summary(
            summary_dir,
            data_star=star,
            input_manifest=input_manifest,
            executable=executable,
            dispatch_log=dispatch,
            dispatch_schedule=schedule,
            label="relion_a",
        )


def test_relion_summary_rejects_dispatch_v2_owner_mismatch(tmp_path):
    summary_dir, star, input_manifest, executable, dispatch, schedule, _shard = _write_relion_fixture(tmp_path)
    dispatch.write_text(dispatch.read_text().replace("2\t1\t1\t0\t2", "2\t1\t2\t0\t2"))
    with pytest.raises(ValueError, match="dispatch-v2 log disagrees"):
        load_relion_summary(
            summary_dir,
            data_star=star,
            input_manifest=input_manifest,
            executable=executable,
            dispatch_log=dispatch,
            dispatch_schedule=schedule,
            label="relion_a",
        )


@pytest.mark.parametrize("mode", ["duplicate", "missing"])
def test_relion_summary_rejects_duplicate_or_missing_internal_identity(tmp_path, mode):
    summary_dir, star, input_manifest, executable, dispatch, schedule, shard = _write_relion_fixture(tmp_path)
    lines = shard.read_text().splitlines()
    header_index = next(index for index, line in enumerate(lines) if not line.startswith("#"))
    columns = lines[header_index].split("\t")
    if mode == "duplicate":
        first = lines[header_index + 1].split("\t")
        first[columns.index("part_id_zero_based")] = "1"
        lines[header_index + 1] = "\t".join(first)
        match = "part IDs are not unique"
    else:
        lines.pop()
        match = "records have 3 rows"
    shard.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match=match):
        load_relion_summary(
            summary_dir,
            data_star=star,
            input_manifest=input_manifest,
            executable=executable,
            dispatch_log=dispatch,
            dispatch_schedule=schedule,
            label="relion_a",
        )


def test_relion_summary_rejects_executable_sha_mismatch(tmp_path):
    summary_dir, star, input_manifest, executable, dispatch, schedule, _shard = _write_relion_fixture(tmp_path)
    executable.write_bytes(b"different binary")
    with pytest.raises(ValueError, match="executable SHA-256"):
        load_relion_summary(
            summary_dir,
            data_star=star,
            input_manifest=input_manifest,
            executable=executable,
            dispatch_log=dispatch,
            dispatch_schedule=schedule,
            label="relion_a",
        )


@pytest.mark.parametrize(
    ("column", "replacement", "match"),
    [
        ("score_element_bytes", "8", "element bytes"),
        ("iteration", "2", "iteration"),
        ("winner_flat_index", "6", "device argmin index"),
        ("significant_count", "2", "significant count"),
        ("class0_second_flat", "0", "pose indices must differ"),
    ],
)
def test_relion_summary_rejects_invalid_native_schema(tmp_path, column, replacement, match):
    summary_dir, star, input_manifest, executable, dispatch, schedule, shard = _write_relion_fixture(tmp_path)
    lines = shard.read_text().splitlines()
    header_index = next(index for index, line in enumerate(lines) if not line.startswith("#"))
    columns = lines[header_index].split("\t")
    row = lines[header_index + 1].split("\t")
    row[columns.index(column)] = replacement
    lines[header_index + 1] = "\t".join(row)
    shard.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match=match):
        load_relion_summary(
            summary_dir,
            data_star=star,
            input_manifest=input_manifest,
            executable=executable,
            dispatch_log=dispatch,
            dispatch_schedule=schedule,
            label="relion_a",
        )


def test_analysis_reports_repeat_normalization_sign_and_ulp(monkeypatch, tmp_path):
    paths = []
    for label in ("a", "b"):
        path = tmp_path / f"recovar_{label}.npz"
        _recovar_env(monkeypatch, path)
        monkeypatch.setenv("RECOVAR_GLOBAL_WINNER_SUMMARY_RUN_ID", label)
        maybe_dump_global_winner_summary(
            experiment_dataset=_dataset(np.arange(4)),
            full_stats=_full_stats(),
            n_classes=4,
            n_rotations=3,
            n_translations=2,
            iteration=1,
        )
        paths.append(path)
    report = analyze_summaries(
        [
            load_recovar_summary(paths[0], label="a"),
            load_recovar_summary(paths[1], label="b"),
        ]
    )
    pair = report["pairwise"][0]
    assert pair["mismatch_count"] == 0
    assert pair["native_float32_ulp"]["max_ulp"] == 0
    assert pair["best_pose_exact_mismatches"]["total_element_count"] == 0
    assert pair["second_pose_exact_mismatches"]["total_element_count"] == 0
    assert pair["native_repeat_normalized"]["centered_class_loss_max_ratio"] == 0.0
    assert set(pair["class_pair_signed_margin_confusion"]) == {
        "0_vs_1",
        "0_vs_2",
        "0_vs_3",
        "1_vs_2",
        "1_vs_3",
        "2_vs_3",
    }


def test_analysis_rejects_pose_topology_mismatch(monkeypatch, tmp_path):
    path = tmp_path / "recovar.npz"
    _recovar_env(monkeypatch, path)
    maybe_dump_global_winner_summary(
        experiment_dataset=_dataset(np.arange(4)),
        full_stats=_full_stats(),
        n_classes=4,
        n_rotations=3,
        n_translations=2,
        iteration=1,
    )
    summary = load_recovar_summary(path, label="a")
    with pytest.raises(ValueError, match="incompatible class-local pose topologies"):
        analyze_summaries([summary, replace(summary, label="b", pose_topology=(2, 3))])

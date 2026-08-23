import numpy as np
import pytest

from scripts.replay_bpref_contribution_bundle import (
    _classify_replay_difference,
    _map_fsc_metrics,
)
from recovar.em.bpref_contribution_replay import (
    BPrefAccumulatorReplay,
    accumulator_replay_metrics,
    dense_fftw_half_rows,
    exact_array_metrics,
    load_bpref_contribution_bundle,
    load_bpref_contribution_shard,
    replay_relion_double,
    summarize_bpref_contribution_bundle,
)


def _write_shard(
    path,
    *,
    original_indices=(4, 7),
    rotation_offset=0,
    call=0,
    dump=0,
    operand_precision="float32",
):
    original_indices = np.asarray(original_indices, dtype=np.int64)
    active_particle = np.asarray([0, 0, 1], dtype=np.int32)
    active_row = np.asarray([0, 1, 0], dtype=np.int32)
    rotations = np.asarray(
        [[10 + rotation_offset, 11 + rotation_offset], [20 + rotation_offset, 21 + rotation_offset]]
    )
    active_global = rotations[active_particle, active_row]
    real_dtype = np.float32 if operand_precision == "float32" else np.float64
    complex_dtype = np.complex64 if operand_precision == "float32" else np.complex128
    np.savez(
        path,
        magic=np.asarray("RECOVAR_BPREF_CONTRIBUTION_ROWS"),
        schema=np.asarray("recovar-bpref-contribution-rows-v3"),
        schema_version=np.int32(3),
        iteration=np.int32(7),
        half=np.int32(2),
        rank=np.int32(0),
        pass_index=np.int32(2),
        class_index=np.int32(0),
        run_id=np.asarray("fixture"),
        call_index=np.int64(call),
        dump_index=np.int64(dump),
        current_size=np.int64(4),
        source_stack_sha256=np.asarray("a" * 64),
        disc_type=np.asarray("linear_interp"),
        reconstruction_padding_factor=np.int32(2),
        image_shape=np.asarray([4, 4], dtype=np.int32),
        volume_shape=np.asarray([11, 11, 11], dtype=np.int32),
        window_indices=np.arange(6, dtype=np.int32),
        actual_counts=np.asarray([2, 1]),
        active_particle_rows=active_particle,
        active_rotation_rows=active_row,
        original_indices=original_indices,
        active_original_indices=original_indices[active_particle],
        oversampled_rotation_indices=rotations,
        active_global_rotation_indices=active_global,
        active_summed=np.ones((3, 6), dtype=complex_dtype) * (call + 1),
        active_ctf_probs=np.ones((3, 6), dtype=real_dtype),
        active_rotations=np.broadcast_to(np.eye(3, dtype=np.float32), (3, 3, 3)),
    )


def test_load_shard_validates_and_closes_active_identity(tmp_path):
    path = tmp_path / "rows.npz"
    _write_shard(path)

    shard = load_bpref_contribution_shard(path)

    assert shard.row_count == 3
    assert shard.row_identity.tolist() == [[4, 10, 0], [4, 11, 1], [7, 20, 0]]


def test_load_shard_preserves_upstream_high_precision_operands(tmp_path):
    path = tmp_path / "rows.npz"
    _write_shard(path, operand_precision="float64")

    shard = load_bpref_contribution_shard(path)

    assert shard.values["active_summed"].dtype == np.complex128
    assert shard.values["active_ctf_probs"].dtype == np.float64


def test_bundle_preserves_execution_order_and_builds_canonical_order(tmp_path):
    later = tmp_path / "later.npz"
    earlier = tmp_path / "earlier.npz"
    _write_shard(later, original_indices=(8, 9), rotation_offset=20, call=2, dump=2)
    _write_shard(earlier, original_indices=(1, 3), call=1, dump=1)

    bundle = load_bpref_contribution_bundle([later, earlier])
    execution = bundle.concatenate("execution")
    canonical = bundle.concatenate("canonical")

    assert execution["active_original_indices"].tolist() == [1, 1, 3, 8, 8, 9]
    assert canonical["active_original_indices"].tolist() == [1, 1, 3, 8, 8, 9]
    assert execution["source_shard"].tolist() == [0, 0, 0, 1, 1, 1]

    summary = summarize_bpref_contribution_bundle(bundle)
    assert summary["status"] == "PASS"
    assert summary["shard_count"] == 2
    assert summary["row_count"] == 6
    assert summary["unique_particle_count"] == 4
    assert summary["quality_gate"].endswith("no correlation metric")


def test_bundle_rejects_overlapping_semantic_rows(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_shard(first, call=0, dump=0)
    _write_shard(second, call=1, dump=1)

    with pytest.raises(ValueError, match="overlap"):
        load_bpref_contribution_bundle([first, second])


def test_shard_rejects_identity_not_matching_candidate_table(tmp_path):
    path = tmp_path / "rows.npz"
    _write_shard(path)
    with np.load(path, allow_pickle=False) as archive:
        values = {key: np.asarray(archive[key]) for key in archive.files}
    values["active_global_rotation_indices"] = values["active_global_rotation_indices"].copy()
    values["active_global_rotation_indices"][0] += 1
    np.savez(path, **values)

    with pytest.raises(ValueError, match="rotation identities"):
        load_bpref_contribution_shard(path)


def test_exact_array_metrics_never_reports_correlation():
    result = exact_array_metrics(np.asarray([1.0, 2.0]), np.asarray([1.0, 2.5]))

    assert result["mismatch_count"] == 1
    assert result["max_abs"] == pytest.approx(0.5)
    assert "correlation" not in result


def test_exact_array_metrics_uses_wide_norm_for_float32():
    left = np.full(1000, 1e30, dtype=np.float32)
    right = left.copy()
    right[0] = np.nextafter(right[0], np.float32(0.0))

    result = exact_array_metrics(left, right)

    assert np.isfinite(result["relative_l2"])
    assert result["mismatch_count"] == 1


def test_dense_fftw_half_rows_places_compact_pixels():
    rows = np.asarray([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])
    dense = dense_fftw_half_rows(
        rows,
        np.asarray([0, 5]),
        (4, 4),
        dtype=np.complex128,
    )

    assert dense.shape == (2, 4, 3)
    assert dense.reshape(2, -1)[:, [0, 5]].tolist() == rows.tolist()
    assert np.count_nonzero(dense) == 4


def test_relion_double_replay_preserves_requested_row_order(tmp_path):
    later = tmp_path / "later.npz"
    earlier = tmp_path / "earlier.npz"
    _write_shard(later, original_indices=(8, 9), rotation_offset=20, call=2, dump=2)
    _write_shard(earlier, original_indices=(1, 3), call=1, dump=1)
    bundle = load_bpref_contribution_bundle([later, earlier])
    calls = []

    def fake_backproject(images, rotations, weights, **kwargs):
        calls.append((images.copy(), rotations.copy(), weights.copy(), kwargs))
        return (
            np.zeros((11, 11, 6), dtype=np.complex128),
            np.zeros((11, 11, 6), dtype=np.float64),
        )

    replay = replay_relion_double(
        bundle,
        order="canonical",
        get_backprojector_data=fake_backproject,
        interpolator=1,
    )

    assert replay.backend == "relion_cpu_backprojector"
    assert replay.order == "canonical"
    assert replay.precision == "complex128/float64"
    assert calls[0][0].dtype == np.complex128
    assert calls[0][2].dtype == np.float64
    assert calls[0][3]["ori_size"] == 4
    assert calls[0][3]["current_size"] == 4


def test_accumulator_replay_metrics_reports_data_and_weight_without_correlation():
    left = BPrefAccumulatorReplay(
        data=np.asarray([1 + 2j], dtype=np.complex64),
        weight=np.asarray([3], dtype=np.float32),
        backend="left",
        order="execution",
        precision="complex64/float32",
        launch_topology="fixture",
    )
    right = BPrefAccumulatorReplay(
        data=np.asarray([1 + 3j], dtype=np.complex128),
        weight=np.asarray([4], dtype=np.float64),
        backend="right",
        order="canonical",
        precision="complex128/float64",
        launch_topology="fixture",
    )

    metrics = accumulator_replay_metrics(left, right)

    assert metrics["data"]["max_abs"] == pytest.approx(1.0)
    assert metrics["weight"]["max_abs"] == pytest.approx(1.0)
    assert "correlation" not in metrics["data"]


def test_map_replay_metrics_use_fsc_auc_without_correlation():
    rng = np.random.default_rng(0)
    volume = rng.normal(size=(8, 8, 8))

    metrics = _map_fsc_metrics(volume, volume.copy())

    assert metrics["fsc_auc"] == pytest.approx(1.0)
    assert metrics["min_fsc_non_dc"] == pytest.approx(1.0)
    assert "correlation" not in metrics


def test_replay_classification_identifies_precision_dominated_difference():
    def comparison(scale):
        return {
            "data": {"relative_l2": scale},
            "weight": {"relative_l2": scale / 2},
        }

    result = _classify_replay_difference(
        {
            "control_repeat_f32_canonical_1": comparison(1e-7),
            "gpu_order_only_f32": comparison(1e-6),
            "gpu_precision_canonical": comparison(4e-2),
            "gpu_vs_relion_f64_canonical": comparison(1e-14),
        },
        {"gpu_precision_canonical": {"fsc_auc": 0.998}},
    )

    assert result["classification"] == "scatter_precision"
    assert result["precision_control_unregularized_map_fsc_auc"] == pytest.approx(0.998)

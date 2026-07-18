import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers import compact_candidate_capture as capture
from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
    _pass2_dump_requested_for_bucket,
)


class _ArrayConversionForbidden:
    def __array__(self, *args, **kwargs):
        raise AssertionError("disabled capture converted a production array")


def _capture_inputs(batch=2):
    image_indices = np.arange(batch, dtype=np.int64)
    rotations = np.stack(
        [np.eye(3, dtype=np.float32), np.diag([-1.0, -1.0, 1.0]).astype(np.float32)]
    )
    per_image_inputs = {
        "oversampled_rots": [rotations.copy() for _ in range(batch)],
        "oversampled_rot_indices": [np.asarray([20, 21], dtype=np.int64) for _ in range(batch)],
        "parent_map": [np.asarray([0, 1], dtype=np.int32) for _ in range(batch)],
        "unique_rot": [np.asarray([100, 101], dtype=np.int32) for _ in range(batch)],
    }
    score = np.broadcast_to(
        np.asarray([[[-1.0, -2.0], [-3.0, -4.0]]], dtype=np.float32),
        (batch, 2, 2),
    ).copy()
    candidate_mask = np.ones_like(score, dtype=bool)
    posterior = np.exp(score, dtype=np.float32)
    posterior /= posterior.sum(axis=(1, 2), keepdims=True, dtype=np.float32)
    significant = posterior >= np.sort(posterior.reshape(batch, -1), axis=1)[:, -2][:, None, None]
    return {
        "iteration": 3,
        "half": 1,
        "image_indices": image_indices,
        "original_indices": image_indices + 1000,
        "per_image_inputs": per_image_inputs,
        "current_size": 64,
        "fine_translations": np.asarray([[0.0, 0.0], [1.0, -1.0]], dtype=np.float32),
        "fine_translation_parent": np.asarray([0, 0], dtype=np.int32),
        "scores": score,
        "probs": posterior,
        "rotation_log_prior": np.zeros((batch, 2), dtype=np.float32),
        "translation_log_prior": np.zeros((batch, 2), dtype=np.float32),
        "candidate_mask": candidate_mask,
        "reconstruction_mask": significant,
        "log_z": np.log(np.exp(score, dtype=np.float32).sum(axis=(1, 2), dtype=np.float32)),
        "best_log_score": score[:, 0, 0],
        "best_argmax": np.zeros(batch, dtype=np.int32),
        "max_posterior": posterior[:, 0, 0],
    }


@pytest.mark.unit
def test_disabled_capture_returns_before_array_conversion(monkeypatch):
    monkeypatch.delenv(capture.CAPTURE_DIR_ENV, raising=False)
    blocked = _ArrayConversionForbidden()
    kwargs = {
        name: blocked
        for name in (
            "image_indices",
            "original_indices",
            "per_image_inputs",
            "fine_translations",
            "fine_translation_parent",
            "scores",
            "probs",
            "rotation_log_prior",
            "translation_log_prior",
            "candidate_mask",
            "reconstruction_mask",
            "log_z",
            "best_log_score",
            "best_argmax",
            "max_posterior",
        )
    }
    assert capture.maybe_capture_k1_production_bucket(
        iteration=3, half=1, current_size=64, **kwargs
    ) == 0


@pytest.mark.unit
def test_disabled_chunked_capture_returns_before_array_conversion(monkeypatch):
    monkeypatch.delenv(capture.CAPTURE_DIR_ENV, raising=False)
    blocked = _ArrayConversionForbidden()

    assert capture.maybe_capture_k1_production_bucket_chunked(
        iteration=3,
        half=1,
        image_indices=blocked,
        original_indices=blocked,
        per_image_inputs=blocked,
        current_size=64,
        fine_translations=blocked,
        fine_translation_parent=blocked,
        score_chunks=(blocked,),
        prob_chunks=(blocked,),
        rotation_log_prior=blocked,
        translation_log_prior=blocked,
        candidate_mask=blocked,
        reconstruction_mask_chunks=(blocked,),
        log_z=blocked,
        best_log_score=blocked,
        best_argmax=blocked,
        max_posterior=blocked,
    ) == 0


@pytest.mark.unit
def test_chunked_capture_capacity_is_bounded(monkeypatch):
    monkeypatch.setattr(capture, "MAX_CHUNKED_CAPTURE_INPUT_BYTES", 100)
    assert capture.require_chunked_capture_capacity(1, 2, 2) == 68
    with pytest.raises(capture.CompactCaptureError, match="bounded host assembly cap"):
        capture.require_chunked_capture_capacity(1, 3, 2)


@pytest.mark.unit
def test_enabled_capture_preserves_native_arrays_and_reopens_atomically(tmp_path, monkeypatch):
    monkeypatch.setenv(capture.CAPTURE_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(capture.CAPTURE_ITERATION_ENV, "3")
    monkeypatch.setattr(capture, "_capture_counter", 0)
    kwargs = _capture_inputs()
    before = {name: np.asarray(kwargs[name]).copy() for name in ("scores", "probs", "candidate_mask")}

    assert capture.maybe_capture_k1_production_bucket(**kwargs) == 2

    paths = list(tmp_path.glob("*.npz"))
    assert len(paths) == 1
    assert not list(tmp_path.glob("*.partial"))
    with np.load(paths[0], allow_pickle=False) as shard:
        np.testing.assert_array_equal(shard["candidate_offset"], [0, 4, 8])
        np.testing.assert_array_equal(shard["original_indices"], [1000, 1001])
        np.testing.assert_array_equal(shard["significant_count"], [2, 2])
        assert shard["raw_combined_score"].dtype == np.float32
        assert shard["posterior"].dtype == np.float32
        assert np.all(
            np.abs(shard["posterior_sum_float32_order"] - shard["posterior_sum_float64_exact"])
            <= shard["posterior_sum_float32_bound"]
        )
    for name, expected in before.items():
        np.testing.assert_array_equal(kwargs[name], expected)


@pytest.mark.unit
def test_capture_splits_raw_shards_at_particle_bound(tmp_path, monkeypatch):
    monkeypatch.setenv(capture.CAPTURE_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(capture, "_capture_counter", 0)
    kwargs = _capture_inputs(batch=257)

    assert capture.maybe_capture_k1_production_bucket(**kwargs) == 257

    paths = sorted(tmp_path.glob("*.npz"))
    assert len(paths) == 2
    with np.load(paths[0], allow_pickle=False) as first, np.load(paths[1], allow_pickle=False) as second:
        assert first["original_indices"].size == 256
        assert second["original_indices"].size == 1
        assert first["candidate_offset"][-1] <= capture.MAX_CANDIDATES_PER_RAW_SHARD


@pytest.mark.unit
def test_capture_splits_raw_shards_at_candidate_bound(tmp_path, monkeypatch):
    monkeypatch.setenv(capture.CAPTURE_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(capture, "_capture_counter", 0)
    monkeypatch.setattr(capture, "MAX_CANDIDATES_PER_RAW_SHARD", 5)

    assert capture.maybe_capture_k1_production_bucket(**_capture_inputs(batch=2)) == 2

    paths = sorted(tmp_path.glob("*.npz"))
    assert len(paths) == 2
    with np.load(paths[0], allow_pickle=False) as first, np.load(paths[1], allow_pickle=False) as second:
        assert first["candidate_offset"][-1] == 4
        assert second["candidate_offset"][-1] == 4


@pytest.mark.unit
def test_raw_capture_validation_and_complete_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv(capture.CAPTURE_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(capture, "_capture_counter", 0)
    kwargs = _capture_inputs(batch=3)
    capture.maybe_capture_k1_production_bucket(**kwargs)
    half2_kwargs = _capture_inputs(batch=3)
    half2_kwargs["half"] = 2
    half2_kwargs["original_indices"] = np.arange(2000, 2003, dtype=np.int64)
    capture.maybe_capture_k1_production_bucket(**half2_kwargs)
    shard = next(tmp_path.glob("*.npz"))

    inventory = capture.validate_raw_capture_shard(shard)
    assert inventory["particle_count"] == 3
    assert inventory["candidate_count"] == 12
    marker = capture.finalize_raw_capture_directory(
        tmp_path,
        expected_original_indices_by_half={
            1: kwargs["original_indices"],
            2: half2_kwargs["original_indices"],
        },
        expected_iteration=3,
    )
    assert marker["particle_count"] == 6
    assert marker["candidate_count"] == 24
    assert marker["halves"] == [1, 2]
    assert (tmp_path / "RAW_CAPTURE.sha256").is_file()
    assert (tmp_path / "RAW_CAPTURE_COMPLETE.json").is_file()


@pytest.mark.unit
def test_raw_capture_finalize_rejects_half_identity_swap(tmp_path, monkeypatch):
    monkeypatch.setenv(capture.CAPTURE_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(capture, "_capture_counter", 0)
    half1 = _capture_inputs(batch=2)
    half2 = _capture_inputs(batch=2)
    half2["half"] = 2
    half2["original_indices"] = np.arange(2000, 2002, dtype=np.int64)
    capture.maybe_capture_k1_production_bucket(**half1)
    capture.maybe_capture_k1_production_bucket(**half2)

    with pytest.raises(capture.CompactCaptureError, match="half-1 identity set"):
        capture.finalize_raw_capture_directory(
            tmp_path,
            expected_original_indices_by_half={
                1: half2["original_indices"],
                2: half1["original_indices"],
            },
            expected_iteration=3,
        )


@pytest.mark.unit
def test_raw_capture_validation_rejects_corrupted_pmax(tmp_path, monkeypatch):
    monkeypatch.setenv(capture.CAPTURE_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(capture, "_capture_counter", 0)
    capture.maybe_capture_k1_production_bucket(**_capture_inputs())
    shard = next(tmp_path.glob("*.npz"))
    with np.load(shard, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    arrays["pmax"] = arrays["pmax"].copy()
    arrays["pmax"][0] = np.nextafter(arrays["pmax"][0], np.float32(0.0))
    np.savez(shard, **arrays)

    with pytest.raises(capture.CompactCaptureError, match="reproduce Pmax"):
        capture.validate_raw_capture_shard(shard)


@pytest.mark.unit
def test_capture_rejects_nonorthogonal_rotation(tmp_path, monkeypatch):
    monkeypatch.setenv(capture.CAPTURE_DIR_ENV, str(tmp_path))
    kwargs = _capture_inputs()
    kwargs["per_image_inputs"]["oversampled_rots"][0][0, 0, 0] = 2.0
    with pytest.raises(capture.CompactCaptureError, match="proper orthogonal"):
        capture.maybe_capture_k1_production_bucket(**kwargs)


@pytest.mark.unit
def test_compact_capture_env_does_not_request_materialized_dump(tmp_path, monkeypatch):
    monkeypatch.setenv(capture.CAPTURE_DIR_ENV, str(tmp_path))
    monkeypatch.delenv("RECOVAR_PASS2_DUMP_DIR", raising=False)
    assert not _pass2_dump_requested_for_bucket(
        experiment_dataset=object(),
        image_indices=np.asarray([0], dtype=np.int64),
        current_size=64,
    )

"""The BPref contribution dump must scope, align and label class-segmented rows.

Before this fix the local routes reached the writer with no class identity: it took a
contiguous prefix of ``actual_rotation_counts``, which runs off the end of class 0's real
rows into its padding and on into class 1, and stored the default ``class_index=0``. A
live capture (job 14030697) showed exactly that.

Every assertion below reads a serialized NPZ produced by the real writer, reloaded with
``allow_pickle=False``. The argument set follows
``tests/unit/test_bpref_device_signature_scope.py``.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from recovar.em.diagnostics import bpref_diagnostics

# Unequal segments on purpose: class 0 has 3 real rows, class 1 has 1, class 2 has 2,
# inside segments of length 4, so padding sits between every class.
COUNTS = np.array([[3, 1, 2]], dtype=np.int64)
SEGMENT = 4
N_ROWS = SEGMENT * COUNTS.shape[1]
REAL_ROWS = [0, 1, 2, 4, 8, 9]
PADDING_ROWS = [3, 5, 6, 7, 10, 11]


class _FakeDataset:
    def original_image_indices_from_local(self, local):
        return np.asarray(local, dtype=np.int64) + 7  # distinct from local ids


def _dump(tmp_path, *, requested, counts=COUNTS, segment=SEGMENT):
    """Run the real writer to completion and return the reloaded NPZ."""
    out = Path(tmp_path)
    mapping = out / "image_names.npy"
    np.save(mapping, np.array([f"{i + 1}@{out / 'stack.mrcs'}" for i in range(32)]),
            allow_pickle=False)
    env = {
        "RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR": str(out),
        "RECOVAR_BPREF_CONTRIBUTION_DUMP_ITERATION": "8",
        "RECOVAR_BPREF_CONTRIBUTION_DUMP_HALF": "1",
        "RECOVAR_BPREF_CONTRIBUTION_DUMP_CURRENT_SIZE": "4",
        "RECOVAR_BPREF_CONTRIBUTION_STACK_SHA256": "0" * 64,
        "RECOVAR_BPREF_CONTRIBUTION_IMAGE_NAMES_NPY": str(mapping),
    }
    if requested is not None:
        env[bpref_diagnostics._BPREF_CONTRIBUTION_DUMP_CLASS_ENV] = str(requested)

    n = N_ROWS if counts is not None else 5
    # Distinct per-row values so a wrong row selection cannot pass by coincidence.
    summed = (np.arange(n, dtype=np.float32) + 100.0).reshape(1, n, 1).astype(np.complex64)
    ctf_probs = (np.arange(n, dtype=np.float32) + 0.5).reshape(1, n, 1)
    scores = -(np.arange(n, dtype=np.float32) + 1.0).reshape(1, n, 1)
    probs = np.exp(scores).astype(np.float32)
    probs /= probs.sum(axis=(1, 2), keepdims=True)

    with mock.patch.dict(os.environ, env, clear=False):
        bpref_diagnostics.set_bpref_contribution_dump_context(iteration=8, half=1)
        try:
            bpref_diagnostics._maybe_dump_bpref_contribution_rows(
                experiment_dataset=_FakeDataset(),
                image_indices=np.asarray([0]),
                current_size=4,
                summed=summed,
                ctf_probs=ctf_probs,
                rotations=np.broadcast_to(np.eye(3), (1, n, 3, 3)),
                actual_counts=np.asarray([int(np.asarray(counts).sum()) if counts is not None else 3]),
                class_actual_rotation_counts=counts,
                class_segment_rotation_count=segment,
                rotation_indices=np.arange(n, dtype=np.int64)[None, :] + 1000,
                fine_translations=np.asarray([[0.0, 0.0]]),
                scores=scores, preprior_scores=scores, probs=probs,
                rotation_log_prior=np.zeros((1, n)), translation_log_prior=np.zeros((1, 1)),
                log_z=np.zeros((1,)), best_log_score=np.zeros((1,)),
                reconstruction_probs=np.zeros_like(probs),
                reconstruction_mask=np.zeros_like(probs, dtype=bool),
                reconstruction_sum_weight=np.zeros((1,)), reconstruction_threshold=np.zeros((1,)),
                candidate_mask=np.ones_like(probs, dtype=bool),
                high_precision_operand_bundle=False, raw_batch_data=None, ctf_params=None,
                noise_variance_half=None, integer_pre_shifts=None, batch_image_corrections=None,
                batch_scale_corrections=None, relion_preprocess_normalization_factors=None,
                relion_cuda_preprocess=False, score_with_masked_images=False, image_mask=None,
                image_mask_mode="not-captured", voxel_size=1.0, ctf_mode="not-captured",
                ctf_dose_per_tilt=0.0, ctf_angle_per_tilt=0.0, disc_type="linear_interp",
                projection_padding_factor=2, reconstruction_padding_factor=2,
                use_relion_x_half_mstep=True, winner_take_all=False, max_r=2,
                window_indices=np.arange(6), image_shape=(4, 4), volume_shape=(4, 4, 4),
                shadow_only_mode=True, shadow_score_bitwise_equal=True,
                shadow_reduction_agreement={
                    "data_rel_l1": 0.0, "data_normalized_max": 0.0, "weight_rel_l1": 0.0,
                    "weight_normalized_max": 0.0, "rel_l1_bound": 1e-3,
                    "normalized_max_bound": 1e-3},
            )
        finally:
            bpref_diagnostics.clear_bpref_contribution_dump_context()
    path = next(p for p in out.glob("*.npz") if "image_names" not in p.name)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}, probs


def _rows_from(payload):
    """The bucket rows the writer actually serialized."""
    return np.asarray(payload["active_rotation_rows"]).ravel().tolist()


def test_serialized_rows_are_per_segment_and_exclude_padding(tmp_path):
    payload, _ = _dump(tmp_path, requested=None)
    rows = _rows_from(payload)
    assert rows == REAL_ROWS
    assert rows != list(range(int(COUNTS.sum())))       # not the old contiguous prefix
    assert not set(rows) & set(PADDING_ROWS)
    assert payload["active_class_indices"].tolist() == [0, 0, 0, 1, 2, 2]


def test_serialized_operands_align_row_for_row(tmp_path):
    payload, _ = _dump(tmp_path, requested=None)
    rows = np.asarray(_rows_from(payload))
    assert np.allclose(np.asarray(payload["active_summed"]).ravel().real, rows + 100.0)
    assert np.allclose(np.asarray(payload["active_ctf_probs"]).ravel(), rows + 0.5)
    assert np.asarray(payload["active_rotations"]).shape[0] == rows.size
    # the global rotation identity travels with the same rows
    assert np.asarray(payload["active_global_rotation_indices"]).tolist() == (rows + 1000).tolist()
    # and the particle identity is the original id, not the local index
    assert np.unique(np.asarray(payload["active_original_indices"])).tolist() == [7]


def test_requested_class_two_emits_only_zero_based_one_with_consistent_identity(tmp_path):
    payload, _ = _dump(tmp_path, requested=2)
    assert _rows_from(payload) == [4]
    assert payload["active_class_indices"].tolist() == [1]
    # The scalar a legacy reader consults must not say class 0.
    assert payload["class_index"].item() == 1
    assert str(payload["class_scope"]) == "segmented-class001"
    assert str(payload["requested_class_one_based"]) == "2"
    assert np.allclose(np.asarray(payload["active_summed"]).ravel().real, [104.0])


def test_mixed_class_scope_is_explicit_and_not_class_zero(tmp_path):
    payload, _ = _dump(tmp_path, requested=None)
    assert payload["class_index"].item() == -1
    assert str(payload["class_scope"]) == "segmented-mixed-classes"
    assert payload["class_segment_rotation_count"].item() == SEGMENT
    assert payload["class_actual_rotation_counts"].tolist() == COUNTS.tolist()


def test_full_joint_probabilities_are_unchanged_by_scoping(tmp_path):
    """Scoping selects rows to emit; it must not renormalize the joint posterior."""
    scoped, probs = _dump(tmp_path, requested=2)
    assert np.isclose(probs.sum(), 1.0)
    assert np.allclose(np.asarray(scoped["posterior_probs"]).ravel(), probs.ravel())


def test_single_class_route_keeps_its_prefix_and_its_scalar(tmp_path):
    payload, _ = _dump(tmp_path, requested=None, counts=None, segment=None)
    assert _rows_from(payload) == [0, 1, 2]
    assert str(payload["class_scope"]) == "single-class-prefix"
    assert payload["class_index"].item() == 0
    assert payload["active_class_indices"].size == 0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(counts=np.array([[5, 1, 2]], dtype=np.int64)), "exceeds the segment length"),
        (dict(segment=None), "requires class_segment_rotation_count"),
        (dict(requested=9), "selects no class"),
    ],
)
def test_unsupported_geometry_or_request_fails_closed(tmp_path, kwargs, message):
    call = dict(requested=None)
    call.update(kwargs)
    with pytest.raises(ValueError, match=message):
        _dump(tmp_path, **call)


def test_both_local_call_sites_forward_the_segment_geometry():
    """Both local routes can reach a segmented bucket; neither may omit the geometry."""
    src = Path(bpref_diagnostics.__file__).resolve().parents[1] / "local" / "local_em_engine.py"
    lines = src.read_text().splitlines()
    starts = [i for i, l in enumerate(lines)
              if l.strip().startswith("_maybe_dump_exact_local_bpref_contribution_rows(")]
    assert len(starts) == 2, f"expected two local call sites, found {len(starts)}"
    for start in starts:
        body = "\n".join(lines[start:start + 60])
        assert "class_actual_rotation_counts=" in body
        assert "class_segment_rotation_count=" in body

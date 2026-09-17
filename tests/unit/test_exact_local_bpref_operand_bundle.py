"""The exact-local BPref capture must honour the existing operand-bundle request.

``RECOVAR_BPREF_HIGH_PRECISION_OPERAND_BUNDLE`` already selects the image-side
operands on the bucketed sparse-pass-2 route, but the exact-local route -- the one
class-segmented K>1 buckets take -- hardwired ``high_precision_operand_bundle=False``
and every operand to ``None``.  A live class-2 capture (job 14031860) therefore stored
``raw_real_images``, ``ctf_params``, ``noise_variance_half``, ``integer_pre_shifts``,
``image_corrections``, ``scale_corrections`` and ``image_mask`` as zero-length arrays,
which is why no same-state replay could be built from it.

Every assertion reads a serialized NPZ produced by the real writer, reloaded with
``allow_pickle=False``, or exercises the real bundle helper.  Nothing here asserts
score parity: a populated capture is an input, not evidence of agreement.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from recovar.em.diagnostics import bpref_diagnostics, local_bpref_capture

BUNDLE_ENV = local_bpref_capture._HIGH_PRECISION_OPERAND_BUNDLE_ENV
IMAGE_SHAPE = (4, 4)
UNPADDED = 2
PADDED = 5  # the big-JIT route pads the image axis; only UNPADDED rows are real.
BUNDLE_KEYS = (
    "raw_real_images", "ctf_params", "noise_variance_half", "integer_pre_shifts",
    "image_corrections", "scale_corrections", "image_mask",
)


class _FakeDataset:
    """Only what the capture path touches; the mask comes from the helper it calls."""

    image_mask = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(IMAGE_SHAPE)

    def original_image_indices_from_local(self, local):
        return np.asarray(local, dtype=np.int64) + 7


class _MasklessDataset(_FakeDataset):
    image_mask = None


def _static_kwargs(**over):
    base = dict(
        high_precision_operand_bundle=False, raw_batch_data=None, ctf_params=None,
        noise_variance_half=None, integer_pre_shifts=None, batch_image_corrections=None,
        batch_scale_corrections=None, relion_preprocess_normalization_factors=None,
        relion_cuda_preprocess=False, score_with_masked_images=False, image_mask=None,
        image_mask_mode="not-captured", voxel_size=1.0, ctf_mode="not-captured",
        ctf_dose_per_tilt=0.0, ctf_angle_per_tilt=0.0, disc_type="linear_interp",
        projection_padding_factor=2, reconstruction_padding_factor=2,
        use_relion_x_half_mstep=True, winner_take_all=False, max_r=2,
        window_indices=np.arange(6), image_shape=IMAGE_SHAPE, volume_shape=(4, 4, 4),
        shadow_only_mode=True, shadow_score_bitwise_equal=True,
        shadow_reduction_agreement={
            "data_rel_l1": 0.0, "data_normalized_max": 0.0, "weight_rel_l1": 0.0,
            "weight_normalized_max": 0.0, "rel_l1_bound": 1e-3,
            "normalized_max_bound": 1e-3},
    )
    base.update(over)
    return base


# Distinct per-row values so a wrong slice or a wrong index cannot pass by coincidence.
RAW = (np.arange(PADDED * 16, dtype=np.float32) + 1.0).reshape(PADDED, *IMAGE_SHAPE)
CTF = (np.arange(PADDED * 3, dtype=np.float32) + 50.0).reshape(PADDED, 3)
NOISE_HALF = np.linspace(0.25, 2.0, 12, dtype=np.float32)
PRE_SHIFTS = np.array([[1, -2], [-3, 4], [9, 9], [9, 9], [9, 9]], dtype=np.int32)
IMAGE_INDICES = np.array([11, 4], dtype=np.int64)
INTEGRAL_SHIFTS = np.zeros((16, 2), dtype=np.float64)
INTEGRAL_SHIFTS[IMAGE_INDICES] = [[1.0, -2.0], [-3.0, 4.0]]
N_IMAGES = 16
ALL_CORR = (np.arange(N_IMAGES, dtype=np.float32) + 0.125)
ALL_SCALE = (np.arange(N_IMAGES, dtype=np.float32) + 0.875)
MASK = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(IMAGE_SHAPE)


class _CtfEvaluator:
    """Shaped like config.ctf: a named mode plus tilt-series fields."""

    class mode:
        name = "SPA"

    dose_per_tilt = 0.0
    angle_per_tilt = 0.0


CTF_EVAL = _CtfEvaluator()


def _bundle(**over):
    kw = dict(
        experiment_dataset=_FakeDataset(), image_shape=IMAGE_SHAPE,
        preprocess_path="split_exact", exact_source_star_ctf=True,
        applied_image_mask=MASK, applied_image_mask_mode="relion_background_fill",
        raw_batch_data=RAW,
        ctf_params=CTF, noise_variance_half=NOISE_HALF,
        image_pre_shifts=INTEGRAL_SHIFTS, integer_pre_shifts=PRE_SHIFTS,
        real_space_pre_shift_applied=True,
        image_corrections=None, scale_corrections=None, image_indices=IMAGE_INDICES,
        unpadded_batch_size=UNPADDED,
    )
    kw.update(over)
    static = kw.pop("static_kwargs", _static_kwargs())
    return local_bpref_capture._exact_local_bpref_operand_bundle(static, **kw)


# --- default behaviour is unchanged -----------------------------------------------

def test_request_absent_returns_static_kwargs_unchanged():
    static = _static_kwargs()
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(BUNDLE_ENV, None)
        out = _bundle(static_kwargs=static)
    assert out is static
    assert out["high_precision_operand_bundle"] is False
    assert out["raw_batch_data"] is None
    assert out["image_mask_mode"] == "not-captured"


@pytest.mark.parametrize("value", ["0", "false", "off", "no", ""])
def test_falsey_request_values_keep_operands_absent(value):
    static = _static_kwargs()
    with mock.patch.dict(os.environ, {BUNDLE_ENV: value}, clear=False):
        assert _bundle(static_kwargs=static) is static


# --- missing inputs fail explicitly rather than being substituted -------------------

def test_request_without_raw_rows_fails_closed():
    with mock.patch.dict(os.environ, {BUNDLE_ENV: "1"}, clear=False):
        with pytest.raises(RuntimeError, match="raw real-space image batches"):
            _bundle(raw_batch_data=None)


# --- requested bundle carries the engine's own rows ---------------------------------

def test_requested_bundle_slices_padding_away_and_keeps_row_order():
    with mock.patch.dict(os.environ, {BUNDLE_ENV: "1"}, clear=False):
        out = _bundle()
    assert out["high_precision_operand_bundle"] is True
    assert np.array_equal(out["raw_batch_data"], RAW[:UNPADDED])
    assert np.array_equal(out["ctf_params"], CTF[:UNPADDED])
    assert np.array_equal(out["integer_pre_shifts"], PRE_SHIFTS[:UNPADDED])
    # The padded tail must not leak in under any of the three.
    assert not np.array_equal(out["raw_batch_data"], RAW)
    assert 9 not in np.asarray(out["integer_pre_shifts"]).ravel().tolist()
    assert np.array_equal(out["noise_variance_half"], NOISE_HALF)


def test_absent_corrections_record_unit_operands_not_zeros():
    with mock.patch.dict(os.environ, {BUNDLE_ENV: "1"}, clear=False):
        out = _bundle()
    assert np.array_equal(out["batch_image_corrections"], np.ones(UNPADDED, dtype=np.float32))
    assert np.array_equal(out["batch_scale_corrections"], np.ones(UNPADDED, dtype=np.float32))


def test_present_corrections_are_indexed_by_image_id_not_by_row():
    with mock.patch.dict(os.environ, {BUNDLE_ENV: "1"}, clear=False):
        out = _bundle(image_corrections=ALL_CORR, scale_corrections=ALL_SCALE)
    assert np.array_equal(out["batch_image_corrections"], ALL_CORR[IMAGE_INDICES])
    assert np.array_equal(out["batch_scale_corrections"], ALL_SCALE[IMAGE_INDICES])
    # A row-order read would take entries 0 and 1; the image ids are 11 and 4.
    assert not np.array_equal(out["batch_image_corrections"], ALL_CORR[:UNPADDED])


def test_absent_pre_shifts_record_explicit_zero_shifts():
    """No shift at all is genuinely a zero shift; only that case may record zeros."""
    with mock.patch.dict(os.environ, {BUNDLE_ENV: "1"}, clear=False):
        out = _bundle(image_pre_shifts=None, integer_pre_shifts=None,
                      real_space_pre_shift_applied=False)
    assert np.array_equal(out["integer_pre_shifts"], np.zeros((UNPADDED, 2), dtype=np.int32))


def test_fourier_phase_shift_path_is_refused_not_recorded_as_zero():
    """A non-integral shift means the engine used Fourier phases, not RELION's
    zero-filled integer real-space shift; zeros would claim the wrong path."""
    fractional = INTEGRAL_SHIFTS.copy()
    fractional[IMAGE_INDICES[0]] = [0.5, -1.25]
    with mock.patch.dict(os.environ, {BUNDLE_ENV: "1"}, clear=False):
        with pytest.raises(RuntimeError, match="non-integral image_pre_shifts"):
            _bundle(image_pre_shifts=fractional, integer_pre_shifts=None,
                    real_space_pre_shift_applied=False)


def test_recorded_integer_shifts_match_the_helper_the_kernel_path_uses():
    """The bundle must agree with integer_pre_shifts_or_none on the same inputs."""
    from recovar.em.helpers.image_shifts import integer_pre_shifts_or_none

    expected = integer_pre_shifts_or_none(
        INTEGRAL_SHIFTS, IMAGE_INDICES, batch=RAW[:UNPADDED]
    )
    assert expected is not None
    with mock.patch.dict(os.environ, {BUNDLE_ENV: "1"}, clear=False):
        out = _bundle(integer_pre_shifts=expected)
    assert np.array_equal(out["integer_pre_shifts"], expected)
    assert np.array_equal(out["integer_pre_shifts"], np.array([[1, -2], [-3, 4]], dtype=np.int32))


def test_recorded_preprocessing_policy_fields_are_not_inherited_claims():
    with mock.patch.dict(os.environ, {BUNDLE_ENV: "1"}, clear=False):
        out = _bundle()
    # Derived from the dataset, not inherited from the static defaults.
    assert out["relion_cuda_preprocess"] is False
    assert np.array_equal(
        out["relion_preprocess_normalization_factors"], np.ones(UNPADDED, dtype=np.float32)
    )
    # On the exact branch the CTF comes from the source STAR, which the metadata says.
    assert out["ctf_mode"] == "relion_exact_source_star"


def test_corrections_that_do_not_round_trip_to_float32_are_rejected():
    """The writer stores float32; a kernel operand at another precision must not be
    silently narrowed and then described as the operand the kernel received."""
    lossy = np.full(16, np.float64(1.0) + np.float64(2.0) ** -40, dtype=np.float64)
    assert not np.array_equal(lossy.astype(np.float32).astype(np.float64), lossy)
    with mock.patch.dict(os.environ, {BUNDLE_ENV: "1"}, clear=False):
        with pytest.raises(RuntimeError, match="cannot record image_corrections exactly"):
            _bundle(image_corrections=lossy)


def test_float32_corrections_round_trip_and_are_accepted():
    exact = ALL_CORR.astype(np.float32)
    with mock.patch.dict(os.environ, {BUNDLE_ENV: "1"}, clear=False):
        out = _bundle(image_corrections=exact)
    assert np.array_equal(out["batch_image_corrections"], exact[IMAGE_INDICES])


def test_float64_raw_images_that_do_not_round_trip_are_rejected():
    lossy = RAW.astype(np.float64)
    lossy[0, 0, 0] = np.float64(1.0) + np.float64(2.0) ** -40
    with mock.patch.dict(os.environ, {BUNDLE_ENV: "1"}, clear=False):
        with pytest.raises(RuntimeError, match="cannot record raw_batch_data exactly"):
            _bundle(raw_batch_data=lossy)


def test_masked_scoring_without_a_supplied_mask_fails_closed():
    """The mask must be the one the bucket scored with; none may be invented here."""
    static = _static_kwargs(score_with_masked_images=True)
    with mock.patch.dict(os.environ, {BUNDLE_ENV: "1"}, clear=False):
        with pytest.raises(RuntimeError, match="requires the image mask this bucket scored with"):
            _bundle(static_kwargs=static, applied_image_mask=None)


def test_supplied_mask_is_recorded_verbatim():
    with mock.patch.dict(os.environ, {BUNDLE_ENV: "1"}, clear=False):
        out = _bundle(static_kwargs=_static_kwargs(score_with_masked_images=True))
    assert np.array_equal(out["image_mask"], MASK)
    assert out["image_mask_mode"] == "relion_background_fill"


# --- the selected preprocessing branch, never the dataset backend -------------------

def test_exact_branch_on_relion_cuda_dataset_is_accepted():
    """local_preprocessing._process_half takes _big_jit_preprocess_half when
    relion_exact_bpref_operands is set, bypassing backend preprocessing entirely, so a
    relion_cuda-CONFIGURED dataset still has no normalization applied to this bucket."""
    with mock.patch.dict(os.environ, {BUNDLE_ENV: "1"}, clear=False):
        with mock.patch.object(
            local_bpref_capture, "uses_relion_cuda_image_preprocessing", return_value=True
        ):
            out = _bundle(preprocess_path="split_exact", exact_source_star_ctf=True)
    assert out["high_precision_operand_bundle"] is True
    # The metadata describes what was APPLIED, not how the dataset is configured.
    assert out["relion_cuda_preprocess"] is False
    assert np.array_equal(
        out["relion_preprocess_normalization_factors"], np.ones(UNPADDED, dtype=np.float32)
    )
    assert out["ctf_mode"] == "relion_exact_source_star"


def test_big_jit_cuda_path_requires_the_kernels_normalization_operand():
    with mock.patch.dict(os.environ, {BUNDLE_ENV: "1"}, clear=False):
        with pytest.raises(RuntimeError, match="image_only_corrections"):
            _bundle(preprocess_path="big_jit_relion_cuda", exact_source_star_ctf=True,
                    relion_cuda_preprocess_radius=37.5,
                    relion_cuda_preprocess_cosine_width=3.25,
                    static_kwargs=_static_kwargs(score_with_masked_images=False))


def test_big_jit_cuda_path_records_the_real_normalization_when_unmasked():
    norm = np.linspace(0.5, 1.5, UNPADDED, dtype=np.float32)
    with mock.patch.dict(os.environ, {BUNDLE_ENV: "1"}, clear=False):
        out = _bundle(preprocess_path="big_jit_relion_cuda", exact_source_star_ctf=True,
                      relion_preprocess_normalization=norm,
                      relion_cuda_preprocess_radius=37.5,
                      relion_cuda_preprocess_cosine_width=3.25,
                      static_kwargs=_static_kwargs(score_with_masked_images=False))
    assert out["relion_cuda_preprocess"] is True
    assert np.array_equal(out["relion_preprocess_normalization_factors"], norm)
    assert out["relion_cuda_preprocess_radius"] == 37.5
    assert out["relion_cuda_preprocess_cosine_width"] == 3.25


def test_k_gt_1_capture_path_records_a_non_exact_ctf_and_keeps_its_mask():
    """At K>1 use_exact_local_relion_operands is False (sparse_pass2_estep.py:826-830
    requires state.K == 1), so every K=4 capture runs big_jit_jax with
    config.compute_ctf_half and the array mask resolved at local_em_engine.py:1529."""
    with mock.patch.dict(os.environ, {BUNDLE_ENV: "1"}, clear=False):
        out = _bundle(preprocess_path="big_jit_jax", exact_source_star_ctf=False,
                      production_ctf=CTF_EVAL,
                      static_kwargs=_static_kwargs(score_with_masked_images=True))
    assert out["high_precision_operand_bundle"] is True
    assert out["relion_cuda_preprocess"] is False
    # The evaluator's own mode, not the sentinel and not the exact-path label.
    assert out["ctf_mode"] == "SPA"
    assert out["ctf_dose_per_tilt"] == 0.0 and out["ctf_angle_per_tilt"] == 0.0
    assert np.array_equal(out["image_mask"], MASK)
    assert out["image_mask_mode"] == "relion_background_fill"


def test_big_jit_jax_with_exact_operands_does_report_the_source_star_ctf():
    with mock.patch.dict(os.environ, {BUNDLE_ENV: "1"}, clear=False):
        out = _bundle(preprocess_path="big_jit_jax", exact_source_star_ctf=True,
                      static_kwargs=_static_kwargs(score_with_masked_images=True))
    assert out["ctf_mode"] == "relion_exact_source_star"


def test_non_exact_path_without_the_production_evaluator_is_refused():
    """The static sentinel describes no CTF construction, so it may not stand in."""
    with mock.patch.dict(os.environ, {BUNDLE_ENV: "1"}, clear=False):
        with pytest.raises(RuntimeError, match="production config.ctf evaluator"):
            _bundle(preprocess_path="big_jit_jax", exact_source_star_ctf=False,
                    production_ctf=None)


def test_unknown_preprocess_path_is_refused():
    with mock.patch.dict(os.environ, {BUNDLE_ENV: "1"}, clear=False):
        with pytest.raises(RuntimeError, match="explicit preprocessing path"):
            _bundle(preprocess_path="whatever", exact_source_star_ctf=False)


def test_ordinary_branch_on_relion_cuda_dataset_is_refused():
    """Here the backend really does normalize, and those factors never reach this
    boundary, so unit factors would claim a policy the bucket did not use."""
    with mock.patch.dict(os.environ, {BUNDLE_ENV: "1"}, clear=False):
        with mock.patch.object(
            local_bpref_capture, "uses_relion_cuda_image_preprocessing", return_value=True
        ):
            with pytest.raises(RuntimeError, match="ordinary local_preprocessing branch"):
                _bundle(preprocess_path="split_backend", exact_source_star_ctf=False,
                      production_ctf=CTF_EVAL)


def test_ordinary_branch_without_relion_cuda_is_accepted():
    with mock.patch.dict(os.environ, {BUNDLE_ENV: "1"}, clear=False):
        with mock.patch.object(
            local_bpref_capture, "uses_relion_cuda_image_preprocessing", return_value=False
        ):
            out = _bundle(preprocess_path="split_backend", exact_source_star_ctf=False,
                          production_ctf=CTF_EVAL)
    assert out["relion_cuda_preprocess"] is False
    # No exact-branch CTF construction was used, so the evaluator's own mode is recorded
    # rather than the exact-path label or the sentinel that describes nothing.
    assert out["ctf_mode"] == "SPA"


def test_branch_is_never_inferred_from_the_dataset_backend():
    """Same dataset, both branches: the outcomes must differ, which is only possible if
    the branch is threaded in rather than read off the backend."""
    with mock.patch.dict(os.environ, {BUNDLE_ENV: "1"}, clear=False):
        with mock.patch.object(
            local_bpref_capture, "uses_relion_cuda_image_preprocessing", return_value=True
        ):
            accepted = _bundle(preprocess_path="split_exact", exact_source_star_ctf=True)
            with pytest.raises(RuntimeError):
                _bundle(preprocess_path="split_backend", exact_source_star_ctf=False,
                      production_ctf=CTF_EVAL)
    assert accepted["high_precision_operand_bundle"] is True


# --- shifts: a cached application must not be recorded as no shift ------------------

def test_applied_shift_without_an_integer_array_is_refused():
    """processed_half_cache applies the shift and then reports integer_pre_shifts=None;
    zeros there would claim the bucket was never shifted."""
    with mock.patch.dict(os.environ, {BUNDLE_ENV: "1"}, clear=False):
        with pytest.raises(RuntimeError, match="pre-shift"):
            _bundle(image_pre_shifts=None, integer_pre_shifts=None,
                    real_space_pre_shift_applied=True)


# --- the writer serializes the bundle instead of zero-length arrays ------------------

def _dump_through_real_writer(tmp_path, static_kwargs):
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
    n = 3
    summed = (np.arange(UNPADDED * n, dtype=np.float32) + 100.0).reshape(UNPADDED, n, 1).astype(np.complex64)
    ctf_probs = (np.arange(UNPADDED * n, dtype=np.float32) + 0.5).reshape(UNPADDED, n, 1)
    scores = -(np.arange(UNPADDED * n, dtype=np.float32) + 1.0).reshape(UNPADDED, n, 1)
    probs = np.exp(scores).astype(np.float32)
    probs /= probs.sum(axis=(1, 2), keepdims=True)
    with mock.patch.dict(os.environ, env, clear=False):
        bpref_diagnostics.set_bpref_contribution_dump_context(iteration=8, half=1)
        try:
            bpref_diagnostics._maybe_dump_bpref_contribution_rows(
                experiment_dataset=_FakeDataset(),
                image_indices=np.asarray([0, 1]),
                current_size=4, summed=summed, ctf_probs=ctf_probs,
                rotations=np.broadcast_to(np.eye(3), (UNPADDED, n, 3, 3)),
                actual_counts=np.asarray([n, n]),
                rotation_indices=np.arange(n, dtype=np.int64) + 1000,
                fine_translations=np.asarray([[0.0, 0.0]]),
                scores=scores, preprior_scores=scores, probs=probs,
                rotation_log_prior=np.zeros((UNPADDED, n)),
                translation_log_prior=np.zeros((UNPADDED, 1)),
                log_z=np.zeros((UNPADDED,)), best_log_score=np.zeros((UNPADDED,)),
                reconstruction_probs=np.zeros_like(probs),
                reconstruction_mask=np.zeros_like(probs, dtype=bool),
                reconstruction_sum_weight=np.zeros((UNPADDED,)),
                reconstruction_threshold=np.zeros((UNPADDED,)),
                candidate_mask=np.ones_like(probs, dtype=bool),
                **static_kwargs,
            )
        finally:
            bpref_diagnostics.clear_bpref_contribution_dump_context()
    path = next(p for p in out.glob("*.npz") if "image_names" not in p.name)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def test_unrequested_capture_reproduces_the_empty_fields_seen_in_job_14031860(tmp_path):
    payload = _dump_through_real_writer(tmp_path, _static_kwargs())
    assert bool(payload["high_precision_operand_bundle"]) is False
    for key in BUNDLE_KEYS:
        assert np.asarray(payload[key]).size == 0, key


def test_requested_capture_serializes_every_operand_round_trip(tmp_path):
    with mock.patch.dict(os.environ, {BUNDLE_ENV: "1"}, clear=False):
        merged = _bundle(static_kwargs=_static_kwargs(score_with_masked_images=True))
    payload = _dump_through_real_writer(tmp_path, merged)
    assert bool(payload["high_precision_operand_bundle"]) is True
    for key in BUNDLE_KEYS:
        assert np.asarray(payload[key]).size > 0, key
    assert np.array_equal(payload["raw_real_images"], RAW[:UNPADDED])
    assert np.array_equal(payload["ctf_params"], CTF[:UNPADDED])
    assert np.array_equal(payload["integer_pre_shifts"], PRE_SHIFTS[:UNPADDED])
    assert np.array_equal(payload["noise_variance_half"], NOISE_HALF)
    assert str(payload["image_mask_mode"]) != "not-captured"
    assert np.asarray(payload["raw_real_images"]).dtype == np.float32

import numpy as np
import pytest

from scripts import analyze_em_k4_preprocess_replay as analyzer


def _runs():
    normalized = np.ones((analyzer.REPLAY_COUNT, 1, 4, 4), dtype=np.float32)
    masked = normalized.copy()
    fourier = np.fft.rfft2(masked).astype(np.complex64)
    return normalized, masked, fourier


def _analyze(normalized, masked, fourier):
    return analyzer.analyze_replay_arrays(
        normalized_runs=normalized,
        masked_runs=masked,
        masked_fourier_runs=fourier,
    )


def test_all_exact_replays_have_fixed_denominator():
    normalized, masked, fourier = _runs()

    report = _analyze(normalized, masked, fourier)

    assert report["classification"] == "preprocessing_replays_bitwise_exact"
    assert report["fixed_metric"] == {
        "evaluated_comparisons": 9,
        "expected_comparisons": 9,
        "bitwise_equal_comparisons": 9,
        "within_fixed_material_floor_comparisons": 9,
    }
    assert report["scorecard_change_admissible"] is False
    assert report["correlation_used"] is False


def test_softmask_roundoff_below_fixed_floor_is_localized():
    normalized, masked, fourier = _runs()
    masked[2, 0, 0, 0] += np.float32(1.0e-7)
    fourier[2] = np.fft.rfft2(masked[2]).astype(np.complex64)

    report = _analyze(normalized, masked, fourier)

    assert report["classification"] == "softmask_background_reduction_drift_within_fixed_material_floor"
    assert report["stages"]["normalized_shifted_real"]["bitwise_equal_comparison_count"] == 3
    assert report["stages"]["masked_real"]["bitwise_equal_comparison_count"] == 2
    assert report["fixed_metric"]["within_fixed_material_floor_comparisons"] == 9


def test_normalization_roundoff_precedes_softmask():
    normalized, masked, fourier = _runs()
    normalized[1, 0, 0, 0] += np.float32(1.0e-7)

    report = _analyze(normalized, masked, fourier)

    assert report["classification"] == "normalization_or_translation_roundoff_within_fixed_material_floor"


def test_material_softmask_drift_is_not_hidden_by_roundoff_label():
    normalized, masked, fourier = _runs()
    masked[3, 0, 0, 0] += np.float32(1.0e-3)
    fourier[3] = np.fft.rfft2(masked[3]).astype(np.complex64)

    report = _analyze(normalized, masked, fourier)

    assert report["classification"] == "material_drift_begins_at_softmask_background"
    assert report["fixed_metric"]["within_fixed_material_floor_comparisons"] < 9


def test_wrong_replay_count_is_rejected():
    normalized, masked, fourier = _runs()

    with pytest.raises(ValueError, match="exactly 4 executions"):
        _analyze(normalized[:3], masked, fourier)


# --- parametric CUDA mask payloads -------------------------------------------------
#
# relion_preprocess_real_f32 masks with two scalars, so a big-JIT CUDA capture stores an
# empty image_mask. _load_bundle previously required a full array unconditionally and
# would have rejected the very payload the capture checker accepts, and run_gpu_replays
# derived the mask radius from the CLI while ignoring the captured geometry.

import numpy as _np  # noqa: E402

SHAPE = (4, 4)
RADIUS, WIDTH, VOXEL, DIAMETER = 25.0, 3.0, 2.0, 100.0  # DIAMETER/(2*VOXEL) == RADIUS


def _bundle_values(parametric: bool, **over):
    values = {
        "class_index": _np.int64(1),
        "current_size": _np.int64(4),
        "half": _np.int64(1),
        "high_precision_operand_bundle": _np.bool_(True),
        "image_corrections": _np.ones(1, _np.float32),
        "image_shape": _np.asarray(SHAPE, dtype=_np.int64),
        "integer_pre_shifts": _np.zeros((1, 2), _np.int32),
        "iteration": _np.int64(8),
        "original_indices": _np.asarray([2131], dtype=_np.int64),
        "preprocess_backend": _np.str_("relion_cuda"),
        "raw_real_images": _np.ones((1,) + SHAPE, _np.float32),
        "relion_cuda_preprocess": _np.bool_(True),
        "relion_preprocess_normalization_factors": _np.ones(1, _np.float32),
        "scale_corrections": _np.ones(1, _np.float32),
        "schema": _np.str_("recovar-bpref-contribution-rows-v3"),
        "score_with_masked_images": _np.bool_(True),
        "voxel_size": _np.float64(VOXEL),
    }
    if parametric:
        values.update(
            image_mask=_np.empty((0,), _np.float32),
            image_mask_mode=_np.str_("relion_cuda_parametric"),
            relion_cuda_preprocess_radius=_np.float64(RADIUS),
            relion_cuda_preprocess_cosine_width=_np.float64(WIDTH),
        )
    else:
        mask = _np.zeros(SHAPE, _np.float32)
        mask[1:, 1:] = 1.0
        values.update(
            image_mask=mask,
            image_mask_mode=_np.str_("relion_background_fill"),
            relion_cuda_preprocess_radius=_np.float64(_np.nan),
            relion_cuda_preprocess_cosine_width=_np.float64(_np.nan),
        )
    values.update(over)
    return values


def _write_bundle(tmp_path, values):
    path = tmp_path / "bpref_contribution_rows_it008_h1_c001.npz"
    _np.savez_compressed(path, **values)
    return path


def _load(tmp_path, values):
    return analyzer._load_bundle(
        _write_bundle(tmp_path, values),
        expected_original_index=2131,
        expected_iteration=8,
        expected_class_one_based=2,
    )


def test_load_bundle_accepts_the_parametric_mask_payload(tmp_path):
    loaded = _load(tmp_path, _bundle_values(parametric=True))
    assert _np.asarray(loaded["image_mask"]).size == 0
    assert float(_np.asarray(loaded["relion_cuda_preprocess_radius"])) == RADIUS


def test_load_bundle_still_accepts_the_array_mask_payload(tmp_path):
    loaded = _load(tmp_path, _bundle_values(parametric=False))
    assert _np.asarray(loaded["image_mask"]).shape == SHAPE


@pytest.mark.parametrize(
    "over,message",
    [
        ({"relion_cuda_preprocess_cosine_width": _np.float64(0.0)}, "cosine width"),
        ({"relion_cuda_preprocess_radius": _np.float64(0.0)}, "radius"),
        ({"relion_cuda_preprocess_radius": _np.float64(_np.nan)}, "radius"),
        ({"image_mask": _np.ones(SHAPE, _np.float32)}, "also carries an array mask"),
    ],
)
def test_load_bundle_rejects_invalid_parametric_geometry(tmp_path, over, message):
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, _bundle_values(parametric=True, **over))


def _jax_gpu_available() -> bool:
    try:
        import jax

        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(
    not _jax_gpu_available(), reason="run_gpu_replays keeps its production JAX GPU guard"
)


def _run_replay(tmp_path, values, *, diameter, width=WIDTH):
    """Execute run_gpu_replays far enough to settle the mask geometry.

    Only two things are mocked: device identity (``_allocated_gpu_uuid``, which shells out
    to nvidia-smi with a UUID this test does not own) and the native boundary
    (``relion_preprocess_real_f32``). The production ``jax.default_backend() == "gpu"``
    guard is left intact, so this runs on a real device and never on CPU, and it needs no
    scheduled GPU job of its own.
    """
    from unittest import mock

    from recovar import cuda_backproject

    seen = {}

    class _Stop(Exception):
        pass

    def _capture(images, normalization, shifts, radius, cosine_width, apply_mask,
                 **kwargs):
        seen["radius"] = float(radius)
        seen["cosine_width"] = float(cosine_width)
        seen["apply_mask"] = bool(apply_mask)
        raise _Stop

    path = _write_bundle(tmp_path, values)
    with mock.patch.object(analyzer, "_allocated_gpu_uuid", lambda expected: "GPU-test"), \
            mock.patch.object(cuda_backproject, "relion_preprocess_real_f32", _capture):
        try:
            analyzer.run_gpu_replays(
                bundle_path=path, expected_gpu_uuid="GPU-test", expected_original_index=2131,
                expected_iteration=8, expected_class_one_based=2,
                particle_diameter_angstrom=diameter, mask_edge_width_pixels=width,
            )
        except _Stop:
            pass
    return seen


@requires_gpu
def test_replay_passes_the_captured_mask_geometry_to_the_kernel(tmp_path):
    """DIAMETER/(2*VOXEL) equals the captured radius, so the call must proceed and the
    kernel must receive the CAPTURED scalars."""
    seen = _run_replay(tmp_path, _bundle_values(parametric=True), diameter=DIAMETER)
    assert seen == {"radius": RADIUS, "cosine_width": WIDTH, "apply_mask": True}


@requires_gpu
def test_replay_rejects_a_cli_geometry_that_disagrees_with_the_capture(tmp_path):
    """120.0/(2*2.0) is 30.0, not the captured 25.0; the replay must not silently
    prefer either value."""
    with pytest.raises(ValueError, match="differs from the CLI"):
        _run_replay(tmp_path, _bundle_values(parametric=True), diameter=120.0)


@requires_gpu
def test_replay_rejects_a_cli_mask_width_that_disagrees_with_the_capture(tmp_path):
    with pytest.raises(ValueError, match="cosine width .* differs from the CLI"):
        _run_replay(tmp_path, _bundle_values(parametric=True), diameter=DIAMETER,
                    width=WIDTH + 1.0)


@requires_gpu
def test_array_mask_payload_still_uses_the_cli_derivation(tmp_path):
    seen = _run_replay(tmp_path, _bundle_values(parametric=False), diameter=DIAMETER)
    assert seen["radius"] == DIAMETER / (2.0 * VOXEL)

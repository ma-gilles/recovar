"""Real native-oracle tests of the shared fixed-capacity projector setup."""

import json

import jax
import numpy as np
import pytest

from recovar.em.relion.relion_projector_setup import setup_relion_projector

pytestmark = pytest.mark.unit


def _crop(projector, r_max, padding):
    size = 2 * (padding * r_max + 1) + 1
    start = projector.shape[0] // 2 - size // 2
    return projector[start : start + size, start : start + size, : size // 2 + 1]


def _relative_metrics(left, right):
    left, right = np.asarray(left, dtype=np.complex128), np.asarray(right, dtype=np.complex128)
    delta = left - right
    tiny = np.finfo(np.float64).tiny
    return {
        "relative_l2": float(
            np.linalg.norm(delta.ravel()) / max(np.linalg.norm(left.ravel()), np.linalg.norm(right.ravel()), tiny)
        ),
        "relative_max": float(np.max(np.abs(delta)) / max(np.max(np.abs(left)), np.max(np.abs(right)), tiny)),
        "relative_mean_abs": float(np.mean(np.abs(delta)) / max(np.mean(np.abs(left)), np.mean(np.abs(right)), tiny)),
    }


def _existing_float32_policy(control1, candidate1, candidate2, control2):
    # Existing static-key ABBA policy: control repeat <=4eps-f32; paired and
    # candidate-repeat metrics <=min(control repeat+4eps-f32,8eps-f32).
    # Reproduced here so this unit test does not depend on a scratch analyzer.
    floor = 4 * np.finfo(np.float32).eps
    control = _relative_metrics(control1, control2)
    assert all(value <= floor for value in control.values())
    panels = [
        _relative_metrics(control1, candidate1),
        _relative_metrics(control2, candidate2),
        _relative_metrics(candidate1, candidate2),
    ]
    for panel in panels:
        assert all(
            value <= np.nextafter(min(control[key] + floor, 2 * floor), np.inf) for key, value in panel.items()
        ), panel
    return panels


@pytest.mark.parametrize("size", [8, 16])
@pytest.mark.parametrize("padding", [1, 2])
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("gridding", [False, True])
def test_native_projector_and_power_fp64_and_consumer_cast(size, padding, full, gridding):
    from recovar.relion_bind import _relion_bind_core as bind

    reference = np.random.default_rng(61).normal(size=(size,) * 3).astype(np.float64)
    before = reference.copy()
    radius = size // (2 if full else 4)
    controls = [
        bind.compute_fourier_transform_map(reference, size, padding, 1, 2 * radius, gridding, 2) for _ in range(2)
    ]
    candidates = [
        setup_relion_projector(reference, np.int32(radius), ori_size=size, padding_factor=padding, do_gridding=gridding)
        for _ in range(2)
    ]
    records = {}
    for field in (0, 1):
        controls_field = [np.asarray(value[field]) for value in controls]
        candidates_field = [np.asarray(value[field]) for value in candidates]
        if field == 0:
            candidates_field = [_crop(value, radius, padding) for value in candidates_field]
            assert candidates_field[0].dtype == np.complex128
            np.testing.assert_array_equal(candidates_field[0] == 0, controls_field[0] == 0)
        else:
            assert candidates_field[0].dtype == np.float64
        metrics = _relative_metrics(controls_field[0], candidates_field[0])
        # Existing native projector FP64 contract (test_e1_padding_parity.py).
        # More than five orders tighter than the four-eps-f32 consumer panel.
        assert all(value < 1e-12 for value in metrics.values()), metrics
        cast_dtype = np.complex64 if field == 0 else np.float32
        records[str(field)] = {
            "fp64": metrics,
            "cast_panel": _existing_float32_policy(
                controls_field[0].astype(cast_dtype),
                candidates_field[0].astype(cast_dtype),
                candidates_field[1].astype(cast_dtype),
                controls_field[1].astype(cast_dtype),
            ),
        }
    np.testing.assert_array_equal(reference, before)
    print(json.dumps({"size": size, "padding": padding, "full": full, "gridding": gridding, "metrics": records}))


def test_radius_and_gridding_reuse_one_compiled_shape():
    setup_relion_projector.clear_cache()
    reference = np.ones((8,) * 3, dtype=np.float64)
    outputs = []
    for radius, corrected in [(0, True), (2, True), (4, False), (-1, False)]:
        result = setup_relion_projector(
            reference, np.int32(radius), ori_size=8, padding_factor=2, do_gridding=corrected
        )
        jax.block_until_ready(result)
        outputs.append(result)
    assert setup_relion_projector._cache_size() == 1
    assert all(value[0].shape == (19, 19, 10) and value[1].shape == (5,) for value in outputs)
    np.testing.assert_array_equal(outputs[2][0], outputs[3][0])


def test_positive_nyquist_only_and_inclusive_sphere():
    from recovar.relion_bind import _relion_bind_core as bind

    reference = np.zeros((8,) * 3, dtype=np.float64)
    reference[4, 4, 4] = 1.0
    candidate, power = setup_relion_projector(reference, np.int32(4), ori_size=8, do_gridding=False)
    actual = np.asarray(candidate)
    native, native_power, *_ = bind.compute_fourier_transform_map(reference, 8, 1, 1, 8, False, 2)
    np.testing.assert_array_equal(actual, native)
    np.testing.assert_allclose(power, native_power, rtol=1e-12, atol=0)
    center = actual.shape[0] // 2
    assert actual[center + 4, center, 0] != 0
    assert actual[center - 4, center, 0] == 0
    assert actual[center, center + 4, 0] != 0
    assert actual[center, center - 4, 0] == 0
    assert actual[center, center, 4] != 0
    assert actual[center + 4, center + 1, 0] == 0

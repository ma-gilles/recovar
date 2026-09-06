"""Native-oracle coverage of the opt-in device projector adapter boundary."""

from dataclasses import replace

import numpy as np
import pytest

from recovar.em.initial_model import dense_adapter as adapter
from recovar.em.initial_model import initialise_denovo_state
from recovar.utils.helpers import recovar_volume_to_relion

pytestmark = pytest.mark.unit


def _relative_metrics(left, right):
    left, right = np.asarray(left, np.complex128), np.asarray(right, np.complex128)
    delta = np.abs(left - right)
    tiny = np.finfo(np.float64).tiny
    return np.array(
        [
            np.linalg.norm(delta.ravel()) / max(np.linalg.norm(left.ravel()), np.linalg.norm(right.ravel()), tiny),
            np.max(delta) / max(np.max(np.abs(left)), np.max(np.abs(right)), tiny),
            np.mean(delta) / max(np.mean(np.abs(left)), np.mean(np.abs(right)), tiny),
        ]
    )


def _assert_existing_consumer_policy(control1, candidate1, candidate2, control2):
    # Immutable static-key ABBA policy: control repeat <= 4eps-f32, paired
    # and candidate-repeat <= min(control repeat + 4eps-f32, 8eps-f32).
    floor = 4 * np.finfo(np.float32).eps
    repeat = _relative_metrics(control1, control2)
    assert np.all(repeat <= floor)
    limit = np.nextafter(np.minimum(repeat + floor, 2 * floor), np.inf)
    for left, right in [(control1, candidate1), (control2, candidate2), (candidate1, candidate2)]:
        metrics = _relative_metrics(left, right)
        assert np.all(metrics <= limit), metrics


@pytest.mark.parametrize("size", [8, 16])
@pytest.mark.parametrize("padding", [1, 2])
@pytest.mark.parametrize("current", ["negative", "zero", "one", "partial_odd", "full", "oversize"])
def test_adapter_native_radius_layout_frame_and_consumer_policy(size, padding, current, monkeypatch):
    from recovar.em.dense_single_volume.helpers import relion_projector_setup as setup
    from recovar.relion_bind import _relion_bind_core as bind

    current_size = {
        "negative": -1,
        "zero": 0,
        "one": 1,
        "partial_odd": size // 2 + 1,
        "full": size,
        "oversize": size + 4,
    }[current]
    references = np.random.default_rng(29).normal(size=(2, size, size, size))
    before = references.copy()
    original_setup = setup.setup_relion_projector
    raw = []

    def capture(*args, **kwargs):
        result = original_setup(*args, **kwargs)
        raw.append(result)
        return result

    monkeypatch.setattr(setup, "setup_relion_projector", capture)
    kwargs = dict(current_size=current_size, padding_factor=padding)
    controls = [adapter.reference_to_relion_projector_half_maps_and_power(references, **kwargs) for _ in range(2)]
    candidates = [
        adapter.reference_to_relion_projector_half_maps_and_power(references, **kwargs, projector_setup_backend="jax")
        for _ in range(2)
    ]
    assert len(raw) == 4  # Native-default calls never enter the device helper.
    assert candidates[0][2] == controls[0][2]
    assert candidates[0][0].shape == controls[0][0].shape
    assert candidates[0][0].dtype == np.complex64
    assert candidates[0][1].dtype == np.float64
    np.testing.assert_array_equal(candidates[0][0] == 0, controls[0][0] == 0)
    for field in (0, 1):
        _assert_existing_consumer_policy(
            controls[0][field], candidates[0][field], candidates[1][field], controls[1][field]
        )
    # FP64 companion checks the same setup before the consumer cast, against
    # the real native oracle under its existing 1e-12 projector contract.
    for reference, (full, power) in zip(references, raw):
        native = bind.compute_fourier_transform_map(
            np.asarray(recovar_volume_to_relion(reference), np.float64),
            size,
            padding,
            1,
            current_size,
            True,
            2,
        )
        logical_size = native[0].shape[0]
        start = full.shape[0] // 2 - logical_size // 2
        cropped = full[start : start + logical_size, start : start + logical_size, : native[0].shape[2]]
        assert np.all(_relative_metrics(native[0], cropped) < 1e-12)
        assert np.all(_relative_metrics(native[1], power) < 1e-12)
    np.testing.assert_array_equal(references, before)


@pytest.mark.parametrize("size,padding,interpolator", [(8, 1, 0), (8, 3, 1), (9, 1, 1)])
def test_unsupported_projector_geometry_uses_native(size, padding, interpolator, monkeypatch):
    from recovar.em.dense_single_volume.helpers import relion_projector_setup as setup

    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported geometry must stay native")

    monkeypatch.setattr(setup, "setup_relion_projector", forbidden)
    refs = np.random.default_rng(31).normal(size=(1, size, size, size))
    kwargs = dict(current_size=size, padding_factor=padding, interpolator=interpolator)
    native = adapter.reference_to_relion_projector_half_maps_and_power(refs, **kwargs)
    requested = adapter.reference_to_relion_projector_half_maps_and_power(refs, **kwargs, projector_setup_backend="jax")
    for left, right in zip(native, requested):
        np.testing.assert_array_equal(left, right)


def test_config_opt_in_state_default_size_and_dump(monkeypatch, tmp_path):
    state = initialise_denovo_state(ori_size=8, pixel_size=1.0, K=1, nr_iter=2, n_directions=3, pseudo_halfsets=True)
    state.Iref = np.random.default_rng(33).normal(size=state.Iref.shape)
    state.current_size = 0
    config = adapter.DenseInitialModelEstepConfig(
        noise_variance=np.ones(5),
        rotations=np.eye(3)[None],
        translations=np.zeros((1, 2)),
        relion_projector_frame=True,
    )
    assert config.projector_setup_backend == "native"
    monkeypatch.setenv(adapter._EXACT_RELION_PROJECTOR_ENV, "1")
    native = adapter._resolve_class_inputs(state, config)
    monkeypatch.setenv(adapter._RELION_PROJECTOR_DUMP_DIR_ENV, str(tmp_path))
    candidate = adapter._resolve_class_inputs(state, replace(config, projector_setup_backend="jax"))
    assert candidate[3] == native[3] == 4
    for field in (0, 1, 2):
        _assert_existing_consumer_policy(native[field], candidate[field], candidate[field], native[field])
    with np.load(tmp_path / "iter000_relion_projector_half.npz") as dumped:
        np.testing.assert_array_equal(dumped["projector_half"], candidate[2])
        assert int(dumped["current_size"]) == 8
    inputs, power = adapter.prepare_relion_projector_class_inputs_and_power(
        state, padding_factor=1, projector_setup_backend="jax"
    )
    np.testing.assert_array_equal(inputs[2], candidate[2])
    assert power.shape == (1, 5)


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="Unknown projector_setup_backend"):
        adapter.reference_to_relion_projector_half_maps_and_power(
            np.zeros((1, 8, 8, 8)), current_size=8, projector_setup_backend="typo"
        )

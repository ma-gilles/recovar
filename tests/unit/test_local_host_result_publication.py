"""Exact publication contracts; the same tests also run on a pinned GPU."""

from dataclasses import fields, is_dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume import k_class, k_class_results
from recovar.em.dense_single_volume.helpers.types import (
    LocalEMResult,
    _stats_array,
    make_noise_stats,
    make_relion_stats,
)
from recovar.em.dense_single_volume.local_layout import LocalHypothesisLayout

pytestmark = pytest.mark.unit


def _same_bytes(actual, expected):
    if actual is None or expected is None:
        assert actual is expected
    elif is_dataclass(actual):
        assert type(actual) is type(expected)
        for field in fields(actual):
            _same_bytes(getattr(actual, field.name), getattr(expected, field.name))
    elif isinstance(actual, tuple):
        assert type(actual) is type(expected)
        for a, b in zip(actual, expected, strict=True):
            _same_bytes(a, b)
    else:
        a, b = np.asarray(actual), np.asarray(expected)
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("dtype", [None, np.float32, np.float64])
@pytest.mark.parametrize("device_input", [False, True])
def test_host_factory_cast_matches_jax_bytes(x64, dtype, device_input):
    with jax.enable_x64(x64):
        values = np.asarray(
            [0.0, -0.0, 1.0, -2.5, 1 + 2**-24, 1 + 3 * 2**-24, 2**-149, -(2**-149), 2**-126, np.inf, -np.inf, np.nan],
            dtype=np.float64,
        )
        if device_input:
            values = jnp.asarray(values)
        actual = _stats_array(values, dtype, True)
        assert isinstance(actual, np.ndarray)
        _same_bytes(actual, _stats_array(values, dtype, False))


def _noise(values, *, host=False, optional=True):
    return make_noise_stats(
        wsum_sigma2_noise=values,
        wsum_img_power=values,
        wsum_sigma2_offset=-0.0,
        sumw=-0.0,
        **(
            {
                name: values
                for name in (
                    "wsum_noise_a2",
                    "wsum_noise_xa",
                    "wsum_norm_correction",
                    "wsum_scale_correction_xa",
                    "wsum_scale_correction_aa",
                )
            }
            if optional
            else {}
        ),
        host_arrays=host,
    )


@pytest.mark.parametrize("optional", [False, True])
def test_single_class_reduction_preserves_payloads_and_scalar_sum(optional):
    values = np.asarray(
        [0, 0x80000000, 0x7FC00021, 0xFFC00022, 0x7F800000, 0xFF800000, 1, 0x80000001, 0x3F800000],
        dtype=np.uint32,
    ).view(np.float32)
    stats = _noise(values, optional=optional)
    actual = k_class_results._sum_noise_stats((stats,), host_arrays=True)
    expected = k_class_results._sum_noise_stats((stats,))
    _same_bytes(actual, expected)
    assert isinstance(actual.wsum_sigma2_noise, np.ndarray)
    _same_bytes(np.asarray(jnp.sum(jnp.stack([jnp.asarray(values)]), axis=0)), values)


@pytest.mark.parametrize("host", [False, True])
def test_complete_result_publication_matches_device_result(host):
    values = np.asarray([0.0, -0.0, 0.25], dtype=np.float32)
    stats = make_relion_stats(
        log_evidence_per_image=[4.0, 5.0, 6.0],
        best_log_score_per_image=[3.0, 4.0, 5.0],
        max_posterior_per_image=[0.5, 0.8, 0.9],
        rotation_posterior_sums=values,
        host_arrays=host,
    )
    noise = _noise(values, host=host)._replace(sumw=2.0)
    kwargs = dict(
        class_log_evidence=np.asarray([[4.0, 5.0, 6.0]]),
        new_means=None,
        Ft_y=[np.asarray([1 + 2j, 3 - 4j], dtype=np.complex64)],
        Ft_ctf=[np.asarray([5.0, 6.0], dtype=np.float32)],
        per_class_hard_assignments=np.asarray([[3, 1, 7]], dtype=np.int32),
        per_class_stats=(stats,),
        noise_stats=(noise,),
        per_class_best_pose_rotation_ids=[np.asarray([3, 1, 7], dtype=np.int32)],
        per_class_best_pose_translations=[np.zeros((3, 2), dtype=np.float32)],
        per_class_best_pose_rotations=[np.tile(np.eye(3, dtype=np.float32), (3, 1, 1))],
    )
    expected = k_class_results._assemble_result(**kwargs)
    actual = k_class_results._assemble_result(
        **kwargs,
        host_accumulators=True,
        host_stats_publication=True,
    )
    _same_bytes(actual, expected)
    for name in (
        "Ft_y",
        "Ft_ctf",
        "class_responsibilities",
        "class_posterior_sums",
        "class_mstep_posterior_sums",
        "class_assignments",
        "pose_assignments",
    ):
        assert isinstance(getattr(actual, name), np.ndarray)
    assert all(isinstance(field, np.ndarray) for field in actual.stats)


@pytest.mark.parametrize("n_classes", [1, 4])
@pytest.mark.parametrize("host_finalize", [False, True])
@pytest.mark.parametrize("flag", ["0", "1"])
def test_local_result_route_is_opt_in_and_single_class(monkeypatch, n_classes, host_finalize, flag):
    monkeypatch.setenv(k_class._LOCAL_HOST_RESULT_PUBLICATION_ENV, flag)
    calls = []
    layout = LocalHypothesisLayout(
        n_global_rotations=2,
        n_pixels=1,
        n_psi=2,
        rotation_offsets=np.asarray([0, 1, 2], dtype=np.int64),
        rotation_ids_flat=np.asarray([0, 1], dtype=np.int32),
        rotations_flat=np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)),
        rotation_log_priors_flat=np.zeros(2, dtype=np.float32),
        rotation_counts=np.ones(2, dtype=np.int32),
        translation_grid=np.zeros((1, 2), dtype=np.float32),
        translation_log_priors=np.zeros((2, 1), dtype=np.float32),
    )

    def engine(_dataset, mean, *_args, **kwargs):
        calls.append(kwargs)
        stats = make_relion_stats(
            log_evidence_per_image=np.asarray([4, 5], dtype=np.float32),
            best_log_score_per_image=np.asarray([3, 4], dtype=np.float32),
            max_posterior_per_image=np.asarray([0.5, 0.8], dtype=np.float32),
            rotation_posterior_sums=np.asarray([1, 2], dtype=np.float32),
            host_arrays=kwargs.get("host_stats_publication", False),
        )
        return LocalEMResult(
            Ft_y=np.zeros(mean.shape, dtype=np.complex64),
            Ft_ctf=np.zeros(mean.shape, dtype=np.float32),
            hard_assignments=np.asarray([1, 0], dtype=np.int32),
            stats=stats,
        )

    monkeypatch.setattr(k_class, "run_local_em_exact", engine)
    result = k_class.run_local_k_class_em(
        SimpleNamespace(n_units=2),
        jnp.zeros((n_classes, 4), dtype=jnp.complex64),
        jnp.ones(4),
        jnp.ones(4),
        layout,
        "linear_interp",
        class_log_evidence=np.zeros((n_classes, 2)),
        host_accumulator_finalize=host_finalize,
    )
    selected = flag == "1" and host_finalize and n_classes == 1
    assert len(calls) == n_classes
    assert all(call.get("host_stats_publication", False) == selected for call in calls)
    assert isinstance(result.Ft_y, np.ndarray if selected else jax.Array)
    assert isinstance(result.stats.rotation_posterior_sums, np.ndarray if selected else jax.Array)


@pytest.mark.parametrize("token", ["", "true", "2", " 1"])
def test_invalid_publication_selector_rejected(monkeypatch, token):
    monkeypatch.setenv(k_class._LOCAL_HOST_RESULT_PUBLICATION_ENV, token)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        k_class._local_host_result_publication_requested()


def test_host_publication_default_and_multiclass_rejection(monkeypatch):
    monkeypatch.delenv(k_class._LOCAL_HOST_RESULT_PUBLICATION_ENV, raising=False)
    assert not k_class._local_host_result_publication_requested()
    stats = _noise(np.ones(2, dtype=np.float32))
    with pytest.raises(ValueError, match="exactly one class"):
        k_class_results._sum_noise_stats((stats, stats), host_arrays=True)
    with pytest.raises(TypeError, match="must be a bool"):
        _stats_array([1.0], None, 1)


def test_actual_local_engine_publishes_exact_host_statistics():
    # Reuse the deterministic tiny EM fixture used by the fast guard.
    import test_refine_relion_mode as fixture

    dataset = fixture.MockDataset(1, np.random.default_rng(17))
    layout = LocalHypothesisLayout(
        n_global_rotations=2,
        n_pixels=2,
        n_psi=1,
        rotation_offsets=np.asarray([0, 2], dtype=np.int64),
        rotation_ids_flat=np.asarray([0, 1], dtype=np.int32),
        rotations_flat=np.asarray(fixture._make_rotations(2, seed=99)),
        rotation_log_priors_flat=np.zeros(2, dtype=np.float32),
        rotation_counts=np.asarray([2], dtype=np.int32),
        translation_grid=np.zeros((1, 2), dtype=np.float32),
        translation_log_priors=np.zeros((1, 1), dtype=np.float32),
    )
    args = (
        dataset,
        fixture._hermitian_volume(fixture.VOLUME_SHAPE, seed=101),
        jnp.full(fixture.VOLUME_SIZE, 10.0, dtype=jnp.float32),
        jnp.ones(fixture.IMAGE_SIZE, dtype=jnp.float32),
        layout,
        "linear_interp",
    )
    kwargs = dict(
        image_batch_size=1,
        rotation_block_size=4,
        current_size=6,
        accumulate_noise=True,
        reconstruct_significant_only=False,
        host_accumulator_finalize=True,
    )
    control = fixture.run_local_em_exact(*args, **kwargs)
    candidate = fixture.run_local_em_exact(*args, **kwargs, host_stats_publication=True)
    _same_bytes(candidate, control)
    assert all(isinstance(value, np.ndarray) for value in candidate.stats)
    assert isinstance(candidate.noise_stats.wsum_norm_correction, np.ndarray)

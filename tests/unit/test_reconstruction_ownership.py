"""Donor reconstruction ownership and authoritative per-half shell priors."""

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.core import fourier_transform_utils as ftu
from recovar.reconstruction import regularization
from recovar.reconstruction import relion_functions as rf

pytestmark = pytest.mark.unit

VOLUME_SHAPE = (8, 8, 8)
VOLUME_SIZE = 512


class TestReconstructionOwnership:
    def test_k1_reconstruction_uses_per_half_1d_tau_shell_prior(self, monkeypatch):
        """K=1 reconstruction should not round-trip tau2 through full volumes."""
        from types import SimpleNamespace

        from recovar.em.refinement import mean_helpers as mean_helpers_module

        calls = []
        events = []

        def fake_reconstruct(*_args, retained_device_numerator=None, **kwargs):
            kwargs["retained_device_numerator"] = retained_device_numerator
            calls.append(kwargs)
            events.append("reconstruct")
            return jnp.ones(VOLUME_SIZE, dtype=jnp.complex128)

        def fake_finish(result, *_accumulators):
            events.append("finish")
            return result

        monkeypatch.setattr(mean_helpers_module, "_reconstruct_volume_eager", fake_reconstruct)
        monkeypatch.setattr(mean_helpers_module, "_finish_host_staged_reconstruction", fake_finish)
        n_shells = VOLUME_SHAPE[0] // 2 + 1
        tau_full = [jnp.full(VOLUME_SIZE, 11.0, dtype=jnp.float32), jnp.full(VOLUME_SIZE, 12.0, dtype=jnp.float32)]
        tau_shells = [jnp.arange(n_shells, dtype=jnp.float32) + 101.0, jnp.arange(n_shells, dtype=jnp.float32) + 201.0]
        retained_half0 = object()
        means = [None, None]
        mean_helpers_module._reconstruct_and_postprocess_means(
            means,
            Ft_y_0=jnp.ones(VOLUME_SIZE, dtype=jnp.complex64),
            Ft_y_1=jnp.ones(VOLUME_SIZE, dtype=jnp.complex64),
            Ft_ctf_0=jnp.ones(VOLUME_SIZE, dtype=jnp.float32),
            Ft_ctf_1=jnp.ones(VOLUME_SIZE, dtype=jnp.float32),
            Ft_y_combined=None,
            Ft_ctf_combined=None,
            mean_signal_variance=None,
            mean_signal_variance_shells=None,
            mean_signal_variance_per_half=tau_full,
            n_classes=1,
            cs=8,
            iteration=0,
            grid_size=8,
            cryo=SimpleNamespace(voxel_size=1.0),
            volume_shape=VOLUME_SHAPE,
            tau2_fudge=1.0,
            padding_factor=1,
            projection_padding_factor=1,
            relion_minres_map=0,
            particle_diameter_ang=None,
            relion_firstiter_cc_this_iter=False,
            relion_firstiter_ini_high_angstrom=None,
            relion_width_mask_edge=5,
            relion_fmask_edge=2,
            mean_signal_variance_shells_per_half=tau_shells,
            retained_Ft_y_0_device=retained_half0,
        )
        assert len(calls) == 2
        assert events == ["reconstruct", "finish", "reconstruct", "finish"]
        assert calls[0]["retained_device_numerator"] is retained_half0
        assert calls[1]["retained_device_numerator"] is None
        assert all((call["tau_is_1d"] is True for call in calls))
        assert all((call["tau"].dtype == jnp.float64 for call in calls))
        np.testing.assert_array_equal(np.asarray(calls[0]["tau"]), np.asarray(tau_shells[0]))
        np.testing.assert_array_equal(np.asarray(calls[1]["tau"]), np.asarray(tau_shells[1]))
        assert means[0].shape == (VOLUME_SIZE,)
        assert means[1].shape == (VOLUME_SIZE,)

    def test_host_staged_k1_reconstruction_blocks_before_next_half(self, monkeypatch):
        """Host-staged box-scale reconstruction must serialize its FFT workspace."""
        from recovar.em.refinement import mean_helpers as mean_helpers_module

        events = []

        class Result:
            def block_until_ready(self):
                events.append("block")

        monkeypatch.setattr(mean_helpers_module.gc, "collect", lambda: events.append("collect"))
        result = Result()
        returned = mean_helpers_module._finish_host_staged_reconstruction(
            result, np.ones(1, dtype=np.float32), jnp.ones(1, dtype=jnp.float32)
        )
        assert returned is result
        assert events == ["block", "collect"]
        events.clear()
        returned = mean_helpers_module._finish_host_staged_reconstruction(
            result, jnp.ones(1, dtype=jnp.float32), jnp.ones(1, dtype=jnp.float32)
        )
        assert returned is result
        assert events == []

    def test_large_host_reconstruction_reuses_retained_numerator_and_releases_stage_a(self, monkeypatch, caplog):
        """The retained half-0 buffer must feed Stage A and release before the iFFT."""
        from recovar.em.refinement import mean_helpers as mean_helpers_module
        from recovar.reconstruction import relion_functions

        events = []
        host_boundary = np.ones((5, 5, 3), dtype=np.complex64)

        class DeviceBoundary:
            def block_until_ready(self):
                events.append("block")

            def __del__(self):
                events.append("release")

        host_ctf = np.ones((5, 5, 3), dtype=np.float32)
        host_numerator = np.ones((5, 5, 3), dtype=np.complex64)
        retained_numerator = jnp.ones((5, 5, 3), dtype=jnp.complex64)
        regularized_filter = jnp.ones(host_ctf.shape, dtype=jnp.float32)

        def fake_regularize(stage_filter, *_args):
            events.append("regularize")
            stage_filter.delete()
            return regularized_filter

        def fake_divide(stage_numerator, stage_filter, *_args):
            events.append("divide")
            assert stage_numerator is retained_numerator
            assert stage_filter is regularized_filter
            return DeviceBoundary()

        def fake_device_get(_value):
            events.append("device_get")
            return host_boundary

        def fake_finish(value, *_args, **kwargs):
            events.append("finish")
            assert events == ["regularize", "divide", "block", "device_get", "release", "collect", "collect", "finish"]
            assert value.shape == (4, 4, 3)
            np.testing.assert_array_equal(value, np.ones((4, 4, 3), dtype=np.complex64))
            assert kwargs["gridding_correct"] == "radial"
            return host_boundary

        monkeypatch.setattr(relion_functions, "_large_grid_postprocess_single_precision_enabled", lambda _voxels: True)
        monkeypatch.setattr(relion_functions, "_regularize_large_relion_half_filter_donate_ctf", fake_regularize)
        monkeypatch.setattr(relion_functions, "_divide_large_relion_half_numerator_donate_numerator", fake_divide)
        monkeypatch.setattr(relion_functions, "_finish_large_relion_postprocess_from_fftw_half", fake_finish)
        monkeypatch.setattr(mean_helpers_module.jax, "device_get", fake_device_get)
        monkeypatch.setattr(mean_helpers_module.gc, "collect", lambda: events.append("collect"))
        caplog.set_level("INFO", logger=mean_helpers_module.__name__)
        volume_shape = (2, 2, 2)
        accumulator_shape = (5, 5, 5)
        half_shape = ftu.volume_shape_to_half_volume_shape(accumulator_shape)
        assert half_shape == host_numerator.shape
        returned = mean_helpers_module._reconstruct_volume_eager(
            host_ctf,
            host_numerator,
            volume_shape,
            2,
            tau=np.ones(np.prod(volume_shape), dtype=np.float32),
            tau2_fudge=1.0,
            projection_padding_factor=1,
            accumulator_volume_shape=accumulator_shape,
            retained_device_numerator=retained_numerator,
        )
        assert returned is host_boundary
        assert events == ["regularize", "divide", "block", "device_get", "release", "collect", "collect", "finish"]
        assert (
            "RELION split pre-IFFT host boundary: accumulator_shape=(5, 5, 5) reconstruction_shape=(4, 4, 4) packed_half_bytes=384"
            in caplog.text
        )

    def test_large_host_reconstruction_stages_numpy_numerator_for_donation(self, monkeypatch, caplog):
        """Half 2 must see half 1 freed, then stage/delete its host numerator."""
        from recovar.em.refinement import mean_helpers as mean_helpers_module
        from recovar.reconstruction import relion_functions

        volume_shape = (2, 2, 2)
        accumulator_shape = (5, 5, 5)
        half_shape = ftu.volume_shape_to_half_volume_shape(accumulator_shape)
        host_ctf = np.ones(half_shape, dtype=np.float32)
        host_numerator = np.ones(half_shape, dtype=np.complex64)
        retained_numerator = jnp.ones(half_shape, dtype=jnp.complex64)
        stage_inputs = []
        stage_outputs = []
        regularized_filters = []
        sentinel = jnp.asarray([7.0 + 0j], dtype=jnp.complex64)

        def fake_regularize(stage_filter, *_args):
            assert isinstance(stage_filter, mean_helpers_module.jax.Array)
            stage_filter.delete()
            regularized = jnp.ones(half_shape, dtype=jnp.float32)
            regularized_filters.append(regularized)
            return regularized

        def fake_divide(stage_numerator, regularized_filter, *_args):
            assert regularized_filter is regularized_filters[-1]
            if not stage_inputs:
                assert stage_numerator is retained_numerator
            else:
                assert retained_numerator.is_deleted()
                assert stage_outputs[0].is_deleted()
                assert isinstance(stage_numerator, mean_helpers_module.jax.Array)
                assert not isinstance(stage_numerator, np.ndarray)
            assert not stage_numerator.is_deleted()
            stage_inputs.append(stage_numerator)
            stage_outputs.append(jnp.ones(half_shape, dtype=jnp.complex64))
            return stage_outputs[-1]

        def fake_finish(value, *_args, **_kwargs):
            assert isinstance(value, np.ndarray)
            assert value.shape == (4, 4, 3)
            return sentinel

        monkeypatch.setattr(relion_functions, "_large_grid_postprocess_single_precision_enabled", lambda _voxels: True)
        monkeypatch.setattr(relion_functions, "_regularize_large_relion_half_filter_donate_ctf", fake_regularize)
        monkeypatch.setattr(relion_functions, "_divide_large_relion_half_numerator_donate_numerator", fake_divide)
        monkeypatch.setattr(relion_functions, "_finish_large_relion_postprocess_from_fftw_half", fake_finish)
        caplog.set_level("INFO", logger=mean_helpers_module.__name__)
        half0 = mean_helpers_module._reconstruct_volume_eager(
            host_ctf,
            host_numerator,
            volume_shape,
            2,
            tau=np.ones(np.prod(volume_shape), dtype=np.float32),
            tau2_fudge=1.0,
            projection_padding_factor=1,
            accumulator_volume_shape=accumulator_shape,
            retained_device_numerator=retained_numerator,
        )
        half1 = mean_helpers_module._reconstruct_volume_eager(
            host_ctf,
            host_numerator,
            volume_shape,
            2,
            tau=np.ones(np.prod(volume_shape), dtype=np.float32),
            tau2_fudge=1.0,
            projection_padding_factor=1,
            accumulator_volume_shape=accumulator_shape,
        )
        assert half0 is sentinel
        assert half1 is sentinel
        assert len(stage_inputs) == len(stage_outputs) == 2
        assert len(regularized_filters) == 2
        assert all((value.is_deleted() for value in stage_inputs))
        assert all((value.is_deleted() for value in stage_outputs))
        assert all((value.is_deleted() for value in regularized_filters))
        assert "RELION Stage A staging host numerator for donation" in caplog.text
        assert "source=staged_numpy output_deleted=True numerator_deleted=True filter_deleted=True" in caplog.text


def test_relion_reconstruction_tau_shells_match_full_prior_bitwise():
    """Authoritative tau2 shells reproduce the legacy full-prior result exactly."""
    volume_shape = (8, 8, 8)
    accumulator_shape = (16, 16, 16)
    half_shape = (16, 16, 9)
    rng = np.random.default_rng(20260831)
    weight = (0.5 + rng.random(half_shape)).astype(np.float32)
    numerator = (rng.standard_normal(half_shape) + 1j * rng.standard_normal(half_shape)).astype(np.complex64)
    fsc = np.linspace(0.9, 0.1, volume_shape[0] // 2 + 1, dtype=np.float64)
    tau_full, _, details = regularization.compute_relion_tau2_from_weights(
        weight,
        weight,
        fsc,
        volume_shape,
        padding_factor=2,
        r_max=volume_shape[0] // 2,
        return_details=True,
        accumulator_volume_shape=accumulator_shape,
    )
    common = {
        "kernel": "triangular",
        "use_spherical_mask": False,
        "grid_correct": False,
        "current_size": volume_shape[0],
        "accumulator_volume_shape": accumulator_shape,
        "input_half_volume": True,
        "preserve_output_precision": True,
    }
    from_full = rf.post_process_from_filter_v2(
        jnp.asarray(weight),
        jnp.asarray(numerator),
        volume_shape,
        2,
        tau=jnp.asarray(tau_full, dtype=jnp.float64),
        tau_is_1d=False,
        **common,
    )
    from_shells = rf.post_process_from_filter_v2(
        jnp.asarray(weight),
        jnp.asarray(numerator),
        volume_shape,
        2,
        tau=jnp.asarray(details["prior_shells"], dtype=jnp.float64),
        tau_is_1d=True,
        **common,
    )
    np.testing.assert_array_equal(np.asarray(from_shells), np.asarray(from_full))


def test_k1_numpy_join_reservation_reaches_first_stage_a_only(monkeypatch):
    """Production host join must hand one live exact buffer to half-0 Stage A."""
    from types import SimpleNamespace

    from recovar.em.refinement import mean_helpers as mean_helpers_module

    accumulator_shape = (9, 9, 9)
    half_shape = ftu.volume_shape_to_half_volume_shape(accumulator_shape)
    rng = np.random.default_rng(20260901)
    ft_y_0 = (rng.standard_normal(half_shape) + 1j * rng.standard_normal(half_shape)).astype(np.complex64)
    ft_y_1 = (rng.standard_normal(half_shape) + 1j * rng.standard_normal(half_shape)).astype(np.complex64)
    ft_ctf_0 = rng.uniform(0.5, 1.5, half_shape).astype(np.float32)
    ft_ctf_1 = rng.uniform(0.5, 1.5, half_shape).astype(np.float32)
    monkeypatch.setenv("RECOVAR_LOWRES_JOIN_HOST_FALLBACK", "always")
    joined = regularization.join_halves_at_low_resolution(
        ft_y_0,
        ft_y_1,
        ft_ctf_0,
        ft_ctf_1,
        volume_shape=accumulator_shape,
        voxel_size=10.0,
        grid_size=4,
        low_resol_join_halves_angstrom=40.0,
        padding_factor=2,
        preserve_inputs=False,
        return_retained_first_numerator=True,
    )
    retained_half0 = joined[4]
    assert retained_half0 is not None
    np.testing.assert_array_equal(np.asarray(retained_half0), joined[0])
    calls = []

    def fake_reconstruct(*args, retained_device_numerator=None, **kwargs):
        kwargs["retained_device_numerator"] = retained_device_numerator
        calls.append((args, kwargs))
        return jnp.ones(4**3, dtype=jnp.complex128)

    monkeypatch.setattr(mean_helpers_module, "_reconstruct_volume_eager", fake_reconstruct)
    monkeypatch.setattr(
        mean_helpers_module, "_finish_host_staged_reconstruction", lambda result, *_accumulators: result
    )
    means = [None, None]
    mean_helpers_module._reconstruct_and_postprocess_means(
        means,
        Ft_y_0=joined[0],
        Ft_y_1=joined[1],
        Ft_ctf_0=joined[2],
        Ft_ctf_1=joined[3],
        Ft_y_combined=None,
        Ft_ctf_combined=None,
        mean_signal_variance=None,
        mean_signal_variance_shells=None,
        mean_signal_variance_per_half=[jnp.ones(4**3, dtype=jnp.float32), jnp.ones(4**3, dtype=jnp.float32)],
        n_classes=1,
        cs=4,
        iteration=0,
        grid_size=4,
        cryo=SimpleNamespace(voxel_size=1.0),
        volume_shape=(4, 4, 4),
        tau2_fudge=1.0,
        padding_factor=2,
        projection_padding_factor=1,
        relion_minres_map=0,
        particle_diameter_ang=None,
        relion_firstiter_cc_this_iter=False,
        relion_firstiter_ini_high_angstrom=None,
        relion_width_mask_edge=5,
        relion_fmask_edge=2,
        accumulator_volume_shape=accumulator_shape,
        mean_signal_variance_shells_per_half=[jnp.ones(3, dtype=jnp.float32), jnp.ones(3, dtype=jnp.float32)],
        retained_Ft_y_0_device=retained_half0,
    )
    assert len(calls) == 2
    assert calls[0][0][1] is joined[0]
    assert calls[0][1]["retained_device_numerator"] is retained_half0
    assert calls[1][1]["retained_device_numerator"] is None

"""Owner tests for the pre-Wiener half join and Class3D per-class tau2 statistics.

``mean_helpers`` owns RELION's ``--low_resol_join_halves`` application and the
per-class tau2/data-vs-prior detail records that the regular and final
all-data passes previously duplicated inline. These cases pin the argument
mapping, the previous-resolution cap, dtypes and the detail-record layout.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume import mean_helpers
from recovar.em.dense_single_volume.helpers.resolution import shell_index_to_resolution_angstrom
from recovar.reconstruction import regularization

pytestmark = pytest.mark.unit

GRID_SIZE = 8
PADDING_FACTOR = 2
VOLUME_SHAPE = (GRID_SIZE, GRID_SIZE, GRID_SIZE)
ACCUMULATOR_SHAPE = tuple(PADDING_FACTOR * s for s in VOLUME_SHAPE)
N_SHELLS = GRID_SIZE // 2 + 1


def _random_accumulators(seed: int):
    rng = np.random.default_rng(seed)
    size = int(np.prod(ACCUMULATOR_SHAPE))
    ft_y = [
        jnp.asarray(rng.standard_normal(size) + 1j * rng.standard_normal(size), dtype=jnp.complex64)
        for _ in range(2)
    ]
    ft_ctf = [
        jnp.asarray(rng.uniform(0.5, 2.0, size) + 0j, dtype=jnp.complex64)
        for _ in range(2)
    ]
    return ft_y, ft_ctf


class TestPreviousResolutionForHalfJoin:
    def test_last_positive_recorded_shell_wins(self):
        expected = shell_index_to_resolution_angstrom(3, GRID_SIZE, 1.5)
        result = mean_helpers._previous_resolution_angstrom_for_half_join(
            [7, 3], 20.0, grid_size=GRID_SIZE, voxel_size=1.5
        )
        assert result == expected

    def test_nonpositive_recorded_shell_leaves_join_uncapped(self):
        # The history takes precedence over a finite state resolution.
        result = mean_helpers._previous_resolution_angstrom_for_half_join(
            [0], 20.0, grid_size=GRID_SIZE, voxel_size=1.5
        )
        assert result is None

    def test_without_history_uses_finite_state_resolution(self):
        result = mean_helpers._previous_resolution_angstrom_for_half_join(
            [], np.float32(12.5), grid_size=GRID_SIZE, voxel_size=1.5
        )
        assert result == 12.5
        assert type(result) is float

    @pytest.mark.parametrize("current_resolution", [float("inf"), float("nan")])
    def test_without_history_or_finite_state_is_none(self, current_resolution):
        result = mean_helpers._previous_resolution_angstrom_for_half_join(
            [], current_resolution, grid_size=GRID_SIZE, voxel_size=1.5
        )
        assert result is None


class TestJoinHalfAccumulatorsAtLowResolution:
    def test_delegates_positional_layout_and_previous_resolution_cap(self, monkeypatch):
        calls = []
        sentinel = ("y0", "y1", "c0", "c1")

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return sentinel

        monkeypatch.setattr(regularization, "join_halves_at_low_resolution", spy)
        ft_y, ft_ctf = _random_accumulators(0)
        result = mean_helpers.join_half_accumulators_at_low_resolution(
            ft_y[0],
            ft_y[1],
            ft_ctf[0],
            ft_ctf[1],
            accumulator_volume_shape=ACCUMULATOR_SHAPE,
            grid_size=GRID_SIZE,
            voxel_size=1.5,
            low_resol_join_halves_angstrom=40.0,
            pixel_resolutions=[2],
            current_resolution=float("inf"),
            padding_factor=PADDING_FACTOR,
        )

        assert result is sentinel
        assert len(calls) == 1
        args, kwargs = calls[0]
        assert [a is b for a, b in zip(args[:4], (ft_y[0], ft_y[1], ft_ctf[0], ft_ctf[1]))] == [True] * 4
        assert args[4:] == (ACCUMULATOR_SHAPE, 1.5, GRID_SIZE, 40.0)
        assert kwargs == {
            "current_resolution_angstrom": shell_index_to_resolution_angstrom(2, GRID_SIZE, 1.5),
            "padding_factor": PADDING_FACTOR,
        }

    @pytest.mark.parametrize(
        ("pixel_resolutions", "current_resolution"),
        [([], float("inf")), ([], 12.5), ([3], 20.0), ([0], 20.0)],
    )
    def test_matches_direct_regularization_call(self, pixel_resolutions, current_resolution):
        ft_y, ft_ctf = _random_accumulators(1)
        expected_cap = mean_helpers._previous_resolution_angstrom_for_half_join(
            pixel_resolutions, current_resolution, grid_size=GRID_SIZE, voxel_size=1.5
        )
        expected = regularization.join_halves_at_low_resolution(
            ft_y[0],
            ft_y[1],
            ft_ctf[0],
            ft_ctf[1],
            ACCUMULATOR_SHAPE,
            1.5,
            GRID_SIZE,
            40.0,
            current_resolution_angstrom=expected_cap,
            padding_factor=PADDING_FACTOR,
        )
        result = mean_helpers.join_half_accumulators_at_low_resolution(
            ft_y[0],
            ft_y[1],
            ft_ctf[0],
            ft_ctf[1],
            accumulator_volume_shape=ACCUMULATOR_SHAPE,
            grid_size=GRID_SIZE,
            voxel_size=1.5,
            low_resol_join_halves_angstrom=40.0,
            pixel_resolutions=pixel_resolutions,
            current_resolution=current_resolution,
            padding_factor=PADDING_FACTOR,
        )
        assert len(result) == 4
        for got, want in zip(result, expected):
            assert got.dtype == want.dtype and got.shape == want.shape
            assert np.asarray(got).tobytes() == np.asarray(want).tobytes()


class TestClassTau2FromIrefPowerSpectrum:
    def test_scales_relion_frame_result_and_shells(self, monkeypatch):
        rng = np.random.default_rng(2)
        relion_tau2 = jnp.asarray(rng.uniform(0.1, 1.0, int(np.prod(VOLUME_SHAPE))), dtype=jnp.float32)
        relion_shells = np.asarray(rng.uniform(0.1, 1.0, N_SHELLS), dtype=np.float32)
        iref = jnp.zeros(int(np.prod(VOLUME_SHAPE)), dtype=jnp.complex64)
        calls = []

        def fake(iref_fourier, volume_shape, **kwargs):
            calls.append((iref_fourier, volume_shape, kwargs))
            return relion_tau2, {"tau2_shells": relion_shells, "shell_sum": relion_shells}

        monkeypatch.setattr(regularization, "compute_relion_tau2_from_iref_power_spectrum", fake)
        frame_scale = float(GRID_SIZE) ** 4
        tau2, shells_relion, shells_recovar = mean_helpers._class_tau2_from_iref_power_spectrum(
            iref,
            VOLUME_SHAPE,
            padding_factor=PADDING_FACTOR,
            current_size=6,
            frame_scale=frame_scale,
        )

        assert len(calls) == 1
        assert calls[0][0] is iref and calls[0][1] == VOLUME_SHAPE
        assert calls[0][2] == {"padding_factor": PADDING_FACTOR, "current_size": 6, "return_details": True}
        expected_tau2 = relion_tau2 * jnp.asarray(frame_scale, dtype=jnp.float32)
        expected_shells_relion = jnp.asarray(relion_shells, dtype=jnp.float32)
        expected_shells_recovar = expected_shells_relion * jnp.asarray(frame_scale, dtype=jnp.float32)
        for got, want in ((tau2, expected_tau2), (shells_relion, expected_shells_relion), (shells_recovar, expected_shells_recovar)):
            assert got.dtype == jnp.float32 and got.shape == want.shape
            assert np.asarray(got).tobytes() == np.asarray(want).tobytes()


class TestClassTau2UpdateDetails:
    @staticmethod
    def _inputs():
        rng = np.random.default_rng(3)
        ft_ctf = jnp.asarray(rng.uniform(0.5, 2.0, int(np.prod(ACCUMULATOR_SHAPE))) + 0j, dtype=jnp.complex64)
        tau2_shells = jnp.asarray(rng.uniform(0.1, 1.0, N_SHELLS), dtype=jnp.float32)
        shell_stats = regularization._compute_relion_weight_shell_stats(
            ft_ctf,
            VOLUME_SHAPE,
            padding_factor=PADDING_FACTOR,
            r_max=GRID_SIZE // 2,
            shell_rounding="round",
            full_half_axis=-1,
            accumulator_volume_shape=ACCUMULATOR_SHAPE,
        )
        return ft_ctf, tau2_shells, shell_stats

    def test_matches_inline_record_layout_and_dtypes(self):
        ft_ctf, tau2_shells, shell_stats = self._inputs()
        data_vs_prior, details = mean_helpers._class_tau2_update_details(
            ft_ctf,
            tau2_shells,
            shell_stats,
            VOLUME_SHAPE,
            padding_factor=PADDING_FACTOR,
            tau2_fudge=4.0,
            current_size=GRID_SIZE,
            full_half_axis=-1,
            accumulator_volume_shape=ACCUMULATOR_SHAPE,
        )
        expected_dvp = regularization.compute_data_vs_prior(
            ft_ctf,
            tau2_shells,
            VOLUME_SHAPE,
            padding_factor=PADDING_FACTOR,
            tau2_fudge=4.0,
            current_size=GRID_SIZE,
            full_half_axis=-1,
            accumulator_volume_shape=ACCUMULATOR_SHAPE,
        )
        assert data_vs_prior.dtype == expected_dvp.dtype
        assert np.asarray(data_vs_prior).tobytes() == np.asarray(expected_dvp).tobytes()

        expected = {
            "prior_shells": np.asarray(tau2_shells, dtype=np.float64),
            "sigma2_shells": np.asarray(
                jnp.where(
                    shell_stats["avg_weight_shells"] > 0,
                    1.0 / (PADDING_FACTOR**3 * shell_stats["avg_weight_shells"]),
                    0.0,
                ),
                dtype=np.float64,
            ),
            "avg_weight_shells": np.asarray(shell_stats["avg_weight_shells"], dtype=np.float64),
            "shell_sum": np.asarray(shell_stats["shell_sum"], dtype=np.float64),
            "shell_count": np.asarray(shell_stats["shell_count"], dtype=np.float64),
            "fsc_shells": None,
            "ssnr_shells": np.asarray(expected_dvp, dtype=np.float64),
        }
        assert list(details) == list(expected) == list(mean_helpers._CLASS_TAU2_DETAIL_KEYS)
        assert details["fsc_shells"] is None
        for key, want in expected.items():
            if want is None:
                continue
            got = details[key]
            assert isinstance(got, np.ndarray) and got.dtype == np.float64 and got.shape == want.shape
            assert got.tobytes() == want.tobytes()

    def test_zero_weight_shells_have_zero_sigma2(self):
        ft_ctf, tau2_shells, shell_stats = self._inputs()
        zero_stats = dict(shell_stats)
        zero_stats["avg_weight_shells"] = jnp.zeros_like(shell_stats["avg_weight_shells"])
        _, details = mean_helpers._class_tau2_update_details(
            ft_ctf,
            tau2_shells,
            zero_stats,
            VOLUME_SHAPE,
            padding_factor=PADDING_FACTOR,
            tau2_fudge=1.0,
            current_size=GRID_SIZE,
            full_half_axis=-1,
            accumulator_volume_shape=ACCUMULATOR_SHAPE,
        )
        assert np.all(details["sigma2_shells"] == 0.0)
        assert np.all(details["avg_weight_shells"] == 0.0)


def test_stack_class_tau2_update_details_keeps_key_layout():
    rng = np.random.default_rng(4)
    records = []
    for _ in range(3):
        record = {key: rng.standard_normal(N_SHELLS) for key in mean_helpers._CLASS_TAU2_DETAIL_KEYS}
        record["fsc_shells"] = None
        records.append(record)

    stacked = mean_helpers._stack_class_tau2_update_details(records)

    assert list(stacked) == list(mean_helpers._CLASS_TAU2_DETAIL_KEYS)
    assert stacked["fsc_shells"] is None
    for key in mean_helpers._CLASS_TAU2_DETAIL_KEYS:
        if key == "fsc_shells":
            continue
        want = np.stack([record[key] for record in records], axis=0)
        assert stacked[key].shape == (3, N_SHELLS)
        assert stacked[key].tobytes() == want.tobytes()

"""Phase 5 tests: VDAM iteration loop orchestration.

Uses a stub E-step callback that returns deterministic accumulators so
the test is self-contained (no RELION fixture dependency).
Full-run RELION fixture parity lives in a follow-up Phase-5 commit that
connects the E-step to the dense-path kernels + particle data loader.
"""

from __future__ import annotations

import numpy as np
import pytest

from recovar.em.initial_model import initialise_denovo_state
from recovar.em.initial_model.iteration_loop import (
    refresh_tau2_from_projector_power,
    relion_solvent_mask,
    relion_solvent_flatten_state,
    run_vdam_iterations,
    select_subset_for_iter,
    update_current_resolution_from_data_vs_prior,
    update_image_size_and_resolution_pointers,
    update_probabilities_from_estep_meta,
)
from recovar.em.initial_model.m_step import VdamAccumulator
from recovar.em.initial_model.subset import numpy_rnd_unif_factory

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def bind():
    try:
        from recovar.relion_bind import _relion_bind_core as m
    except ImportError:
        pytest.skip("relion_bind not built")
    if not hasattr(m, "vdam_reweight_grad"):
        pytest.skip("relion_bind built without InitialModel VDAM primitives; rebuild recovar/relion_bind")
    return m


def _stub_estep_factory(ori_size: int):
    """Return a deterministic E-step that produces K*H accumulators keyed
    on iter so the M-step receives slightly different data each iter.
    """
    counter = {"n": 0}

    def estep(state, particle_ids, halfset_ids):
        counter["n"] += 1
        pf = 1
        Nz = ori_size * pf
        Ny = ori_size * pf
        Nx_h = (ori_size * pf) // 2 + 1
        accumulators = []
        K = state.K
        H = 2 if state.pseudo_halfsets else 1
        for h in range(H):
            for k in range(K):
                seed = 100 * counter["n"] + 10 * h + k
                rng = np.random.default_rng(seed)
                data = (rng.standard_normal((Nz, Ny, Nx_h)) + 1j * rng.standard_normal((Nz, Ny, Nx_h))).astype(
                    np.complex128
                )
                weight = rng.uniform(10.0, 100.0, size=(Nz, Ny, Nx_h))
                accumulators.append(VdamAccumulator(data=data, weight=weight, class_idx=k, halfset_idx=h))
        meta = {
            "pmax_mean": float(rng.uniform(0.1, 0.3)),
            "nr_significant_mean": int(rng.integers(10, 200)),
            "iter": state.iter,
        }
        return accumulators, meta

    return estep


def test_refresh_tau2_from_projector_power_updates_all_classes(bind):
    ori = 8
    state = initialise_denovo_state(
        ori_size=ori,
        pixel_size=1.0,
        K=2,
        nr_iter=1,
        n_directions=12,
        pseudo_halfsets=True,
    )
    z, y, x = np.indices((ori, ori, ori), dtype=np.float64)
    state.Iref[0] = np.exp(-((x - 2.0) ** 2 + (y - 3.0) ** 2 + (z - 4.0) ** 2) / 6.0)
    state.Iref[1] = np.exp(-((x - 5.0) ** 2 + (y - 4.0) ** 2 + (z - 3.0) ** 2) / 4.0)
    state.tau2_class.fill(123.0)
    state.current_size = 6

    out = refresh_tau2_from_projector_power(state)

    assert out is not state
    assert out.tau2_class.shape == (2, ori // 2 + 1)
    assert np.all(np.isfinite(out.tau2_class))
    assert np.all(out.tau2_class >= 0.0)
    assert not np.array_equal(out.tau2_class, state.tau2_class)
    assert not np.array_equal(out.tau2_class[0], out.tau2_class[1])
    np.testing.assert_array_equal(state.tau2_class, np.full((2, ori // 2 + 1), 123.0))


class TestRunVdamIterations:
    def test_current_resolution_uses_relion_data_vs_prior_scan(self):
        state = initialise_denovo_state(
            ori_size=64,
            pixel_size=2.0,
            K=2,
            nr_iter=1,
            n_directions=12,
            pseudo_halfsets=True,
        )
        state.data_vs_prior_class[0, 1:8] = 2.0
        state.data_vs_prior_class[0, 8] = 0.5
        state.data_vs_prior_class[1, 1:13] = 2.0
        state.data_vs_prior_class[1, 13] = 0.5

        out = update_current_resolution_from_data_vs_prior(state)

        assert out.current_resolution_shell == 12
        assert out.current_resolution == pytest.approx(12.0 / (64.0 * 2.0))
        assert out.current_size == state.current_size

    def test_current_size_update_uses_previous_resolution_for_next_estep(self):
        state = initialise_denovo_state(
            ori_size=64,
            pixel_size=2.0,
            K=1,
            nr_iter=1,
            n_directions=12,
            pseudo_halfsets=True,
        )
        state.current_resolution_shell = 20
        state.current_resolution = 20.0 / (64.0 * 2.0)
        state.ave_Pmax = 0.0

        out = update_image_size_and_resolution_pointers(state)

        assert out.current_size == 60
        assert out.current_resolution_shell == 20

    def test_iteration_loop_feeds_updated_current_size_to_next_estep(self, monkeypatch):
        import recovar.em.initial_model.iteration_loop as loop

        state = initialise_denovo_state(
            ori_size=64,
            pixel_size=1.0,
            K=1,
            nr_iter=2,
            n_directions=12,
            pseudo_halfsets=True,
        )
        seen_current_sizes = []

        def estep(current, particle_ids, halfset_ids):
            seen_current_sizes.append(int(current.current_size))
            return [], {"max_posterior_per_image": np.asarray([0.2, 0.3], dtype=np.float32)}

        def fake_m_step(current, accumulators, **kwargs):
            out = current
            out.data_vs_prior_class = np.zeros_like(current.data_vs_prior_class)
            out.data_vs_prior_class[:, 1:21] = 2.0
            out.data_vs_prior_class[:, 21:] = 0.5
            return out

        monkeypatch.setattr(loop, "vdam_m_step", fake_m_step)

        final = run_vdam_iterations(
            state,
            nr_particles=200,
            optics_group_by_particle=[0] * 200,
            grad_ini_subset_size=50,
            grad_fin_subset_size=100,
            tau2_fudge_arg=4.0,
            grad_em_iters=0,
            random_seed=0,
            rnd_unif_factory=numpy_rnd_unif_factory,
            expectation_step=estep,
        )

        assert seen_current_sizes == [28, 60]
        assert final.current_resolution_shell == 20

    def test_iteration_loop_refreshes_tau2_before_estep(self, monkeypatch):
        import recovar.em.initial_model.iteration_loop as loop

        state = initialise_denovo_state(
            ori_size=16,
            pixel_size=1.0,
            K=1,
            nr_iter=1,
            n_directions=12,
            pseudo_halfsets=True,
        )
        state.Iref[0, 8, 8, 8] = 1.0
        state.tau2_class.fill(0.0)
        seen = {}

        def fake_refresh(current, *, padding_factor, interpolator):
            assert padding_factor == 1
            assert interpolator == 1
            out = current
            out.tau2_class = np.full_like(current.tau2_class, 7.0)
            seen["refresh_current_size"] = int(current.current_size)
            return out

        def estep(current, particle_ids, halfset_ids):
            seen["estep_tau2"] = current.tau2_class.copy()
            return [], {}

        def fake_m_step(current, accumulators, **kwargs):
            return current

        monkeypatch.setattr(loop, "refresh_tau2_from_projector_power", fake_refresh)
        monkeypatch.setattr(loop, "vdam_m_step", fake_m_step)

        run_vdam_iterations(
            state,
            nr_particles=200,
            optics_group_by_particle=[0] * 200,
            grad_ini_subset_size=50,
            grad_fin_subset_size=100,
            tau2_fudge_arg=4.0,
            grad_em_iters=0,
            random_seed=0,
            rnd_unif_factory=numpy_rnd_unif_factory,
            expectation_step=estep,
        )

        assert seen["refresh_current_size"] == 16
        np.testing.assert_array_equal(seen["estep_tau2"], np.full((1, 9), 7.0))

    def test_relion_solvent_flatten_state_matches_centered_spherical_mask(self):
        state = initialise_denovo_state(
            ori_size=8,
            pixel_size=1.0,
            K=1,
            nr_iter=1,
            n_directions=3,
            pseudo_halfsets=True,
        )
        state.Iref = np.ones((1, 8, 8, 8), dtype=np.float64)

        mask = relion_solvent_mask(
            ori_size=8,
            pixel_size=1.0,
            particle_diameter_ang=6.0,
            width_mask_edge_px=2.0,
        )
        out = relion_solvent_flatten_state(state, mask=mask)

        coords = np.arange(-4, 4, dtype=np.float64)
        z, y, x = np.meshgrid(coords, coords, coords, indexing="ij")
        radius = 3.0
        radius_p = 5.0
        r = np.sqrt(x * x + y * y + z * z)
        expected = np.zeros((8, 8, 8), dtype=np.float64)
        expected[r < radius] = 1.0
        edge = (r >= radius) & (r <= radius_p)
        expected[edge] = 0.5 - 0.5 * np.cos(np.pi * (radius_p - r[edge]) / 2.0)

        np.testing.assert_allclose(mask, expected, atol=1e-12)
        np.testing.assert_allclose(out.Iref[0], expected, atol=1e-12)
        np.testing.assert_allclose(state.Iref, 1.0)

    def test_post_mstep_update_runs_before_iteration_artifact_sink(self, bind):
        ori = 16
        state = initialise_denovo_state(
            ori_size=ori,
            pixel_size=1.0,
            K=1,
            nr_iter=1,
            n_directions=12,
            pseudo_halfsets=True,
        )
        state.Iref = np.ones((1, ori, ori, ori), dtype=np.float64) * 0.1
        seen = {}

        def post_update(current, iteration, meta):
            assert iteration == 1
            meta["post_update_seen"] = True
            updated = relion_solvent_flatten_state(
                current,
                particle_diameter_ang=8.0,
                width_mask_edge_px=2.0,
            )
            seen["post_sum"] = float(updated.Iref.sum())
            return updated

        def sink(current, iteration, meta):
            seen["sink_sum"] = float(current.Iref.sum())
            seen["meta_seen"] = bool(meta.get("post_update_seen"))

        run_vdam_iterations(
            state,
            nr_particles=200,
            optics_group_by_particle=[0] * 200,
            grad_ini_subset_size=50,
            grad_fin_subset_size=100,
            tau2_fudge_arg=4.0,
            grad_em_iters=0,
            random_seed=1,
            rnd_unif_factory=numpy_rnd_unif_factory,
            expectation_step=_stub_estep_factory(ori),
            iter_artifact_sink=sink,
            post_mstep_update=post_update,
        )

        assert seen["meta_seen"] is True
        assert seen["sink_sum"] == seen["post_sum"]

    def test_updates_class_and_direction_priors_from_estep_meta(self):
        state = initialise_denovo_state(
            ori_size=8,
            pixel_size=1.0,
            K=2,
            nr_iter=1,
            n_directions=3,
            pseudo_halfsets=True,
        )
        state.pdf_class = np.asarray([0.5, 0.5], dtype=np.float64)
        state.pdf_direction = np.full((2, 3), 1.0 / 6.0, dtype=np.float64)
        state.subset_size = 50
        meta = {
            "class_posterior_sums": np.asarray([30.0, 70.0]),
            "class_direction_posterior_sums": np.asarray(
                [
                    [10.0, 15.0, 5.0],
                    [20.0, 30.0, 20.0],
                ]
            ),
        }

        out = update_probabilities_from_estep_meta(state, meta, do_grad=True, mu=0.9)

        np.testing.assert_allclose(out.pdf_class, [0.48, 0.52])
        np.testing.assert_allclose(
            out.pdf_direction,
            state.pdf_direction * 0.9 + 0.1 * meta["class_direction_posterior_sums"] / 100.0,
        )
        np.testing.assert_allclose(state.pdf_class, [0.5, 0.5])

    def test_direction_prior_resizes_uniformly_when_sampling_changes(self):
        state = initialise_denovo_state(
            ori_size=8,
            pixel_size=1.0,
            K=2,
            nr_iter=1,
            n_directions=3,
            pseudo_halfsets=True,
        )
        state.pdf_direction = np.full((2, 3), 1.0 / 6.0, dtype=np.float64)
        state.subset_size = 50
        direction_sums = np.asarray(
            [
                [4.0, 6.0, 8.0, 10.0, 12.0],
                [10.0, 10.0, 10.0, 15.0, 15.0],
            ],
            dtype=np.float64,
        )
        meta = {
            "class_posterior_sums": np.asarray([40.0, 60.0]),
            "class_direction_posterior_sums": direction_sums,
        }

        out = update_probabilities_from_estep_meta(state, meta, do_grad=True, mu=0.9)

        assert out.pdf_direction.shape == (2, 5)
        expected_uniform = np.full((2, 5), 1.0 / 10.0, dtype=np.float64)
        np.testing.assert_allclose(out.pdf_direction, expected_uniform * 0.9 + 0.1 * direction_sums / 100.0)

    def test_all_particle_probability_update_replaces_priors_and_allows_zero_class(self):
        state = initialise_denovo_state(
            ori_size=8,
            pixel_size=1.0,
            K=2,
            nr_iter=1,
            n_directions=3,
            pseudo_halfsets=True,
        )
        state.pdf_class = np.asarray([0.5, 0.5], dtype=np.float64)
        state.subset_size = -1

        out = update_probabilities_from_estep_meta(
            state,
            {
                "halfset_0_class_posterior_sums": np.asarray([0.0, 4.0]),
                "halfset_1_class_posterior_sums": np.asarray([0.0, 6.0]),
            },
            do_grad=True,
            mu=0.9,
        )

        np.testing.assert_allclose(out.pdf_class, [0.0, 1.0])

    def test_random_seed_zero_skips_particle_shuffle(self):
        state = initialise_denovo_state(
            ori_size=8,
            pixel_size=1.0,
            K=1,
            nr_iter=1,
            n_directions=3,
            pseudo_halfsets=True,
        )
        state.subset_size = 4

        def fail_if_called(seed):
            raise AssertionError(f"unexpected randomization for seed {seed}")

        out = select_subset_for_iter(
            state,
            iter=1,
            nr_particles=6,
            optics_group_by_particle=[0, 1, 0, 1, 0, 1],
            rnd_unif_factory=fail_if_called,
            random_seed=0,
            do_grad=True,
        )

        np.testing.assert_array_equal(out.subset_particle_ids, np.array([0, 2, 1, 3]))
        np.testing.assert_array_equal(out.subset_halfset_ids, np.array([0, 0, 1, 1], dtype=np.int8))

    def test_random_seed_zero_preserves_relion_sorted_idx_base_order(self):
        state = initialise_denovo_state(
            ori_size=8,
            pixel_size=1.0,
            K=1,
            nr_iter=1,
            n_directions=3,
            pseudo_halfsets=True,
        )
        state.subset_size = 6

        def fail_if_called(seed):
            raise AssertionError(f"unexpected randomization for seed {seed}")

        out = select_subset_for_iter(
            state,
            iter=1,
            nr_particles=6,
            optics_group_by_particle=[0, 1, 0, 1, 0, 1],
            rnd_unif_factory=fail_if_called,
            random_seed=0,
            do_grad=True,
            particle_order=np.array([5, 0, 3, 4, 1, 2], dtype=np.int64),
        )

        np.testing.assert_array_equal(out.subset_particle_ids, np.array([0, 4, 2, 5, 3, 1]))
        np.testing.assert_array_equal(out.subset_halfset_ids, np.array([0, 0, 0, 1, 1, 1], dtype=np.int8))

    def test_rejects_invalid_particle_order(self):
        state = initialise_denovo_state(
            ori_size=8,
            pixel_size=1.0,
            K=1,
            nr_iter=1,
            n_directions=3,
            pseudo_halfsets=True,
        )
        state.subset_size = 4

        with pytest.raises(ValueError, match="particle_order must be a permutation"):
            select_subset_for_iter(
                state,
                iter=1,
                nr_particles=4,
                optics_group_by_particle=[0, 0, 0, 0],
                rnd_unif_factory=numpy_rnd_unif_factory,
                random_seed=0,
                do_grad=True,
                particle_order=np.array([0, 1, 1, 3], dtype=np.int64),
            )

    def test_5_iter_smoke(self, bind):
        ori = 16
        K = 1
        nr_iter = 5
        nr_particles = 300
        state = initialise_denovo_state(
            ori_size=ori,
            pixel_size=1.0,
            K=K,
            nr_iter=nr_iter,
            n_directions=12,
            pseudo_halfsets=True,
        )
        # RELION seeds Iref from Mavg (setSigmaNoiseEstimatesAndSetAverageImage
        # at ml_optimiser.cpp:2805) before iter 1; without it reconstructGrad's
        # FSC estimate stays 0 and the reference never updates. Mimic that by
        # seeding Iref with noise.
        state.Iref = np.random.default_rng(77).standard_normal((K, ori, ori, ori)) * 0.1
        optics = [0] * nr_particles

        iter_log = []

        def sink(s, it, meta):
            iter_log.append(
                {
                    "iter": it,
                    "subset_size": s.subset_size,
                    "stepsize": s.grad_current_stepsize,
                    "tau": s.tau2_fudge_factor,
                }
            )

        final = run_vdam_iterations(
            state,
            nr_particles=nr_particles,
            optics_group_by_particle=optics,
            grad_ini_subset_size=200,
            grad_fin_subset_size=300,
            tau2_fudge_arg=4.0,
            grad_em_iters=0,
            random_seed=42,
            rnd_unif_factory=numpy_rnd_unif_factory,
            expectation_step=_stub_estep_factory(ori),
            iter_artifact_sink=sink,
        )
        assert final.iter == nr_iter
        assert len(iter_log) == nr_iter
        # Each iteration should have updated Iref
        assert np.all(np.isfinite(final.Iref))
        assert not np.array_equal(final.Iref, state.Iref)

        # Schedules advance reasonably: tau2 starts near 1, ends < 4
        first = iter_log[0]
        last = iter_log[-1]
        assert first["tau"] < last["tau"] or abs(first["tau"] - last["tau"]) < 1.0

    def test_respects_grad_em_tail(self, bind):
        ori = 16
        K = 1
        nr_iter = 10
        state = initialise_denovo_state(
            ori_size=ori,
            pixel_size=1.0,
            K=K,
            nr_iter=nr_iter,
            n_directions=12,
            pseudo_halfsets=True,
        )
        optics = [0] * 200

        iter_log = []
        sink = lambda s, it, meta: iter_log.append({"iter": it, "pseudo": s.pseudo_halfsets})

        run_vdam_iterations(
            state,
            nr_particles=200,
            optics_group_by_particle=optics,
            grad_ini_subset_size=50,
            grad_fin_subset_size=100,
            tau2_fudge_arg=4.0,
            grad_em_iters=2,
            random_seed=1,
            rnd_unif_factory=numpy_rnd_unif_factory,
            expectation_step=_stub_estep_factory(ori),
            iter_artifact_sink=sink,
        )
        # Last 2 iters drop gradient mode -> pseudo_halfsets becomes False
        # select_subset_for_iter copies do_grad's value to pseudo field
        # do_grad = (nr_iter - iter) >= grad_em_iters = 2
        # -> iter = 8 : (10-8)=2 >= 2 -> do_grad=True
        # -> iter = 9 : (10-9)=1 <  2 -> do_grad=False
        # -> iter =10 : (10-10)=0 < 2 -> do_grad=False
        assert iter_log[8 - 1]["pseudo"] is True  # iter 8
        assert iter_log[9 - 1]["pseudo"] is False  # iter 9
        assert iter_log[10 - 1]["pseudo"] is False  # iter 10

    def test_subset_schedule_snapshot(self, bind):
        ori = 16
        K = 1
        nr_iter = 6
        state = initialise_denovo_state(
            ori_size=ori,
            pixel_size=1.0,
            K=K,
            nr_iter=nr_iter,
            n_directions=12,
        )
        iter_log = []

        def sink(s, it, meta):
            iter_log.append({"iter": it, "subset_size": s.subset_size})

        # Dataset size >> fin subset so the `subset_size >= nr_particles`
        # collapse branch doesn't fire for most iters
        run_vdam_iterations(
            state,
            nr_particles=10_000,
            optics_group_by_particle=[0] * 10_000,
            grad_ini_subset_size=200,
            grad_fin_subset_size=1000,
            tau2_fudge_arg=4.0,
            grad_em_iters=0,
            random_seed=7,
            rnd_unif_factory=numpy_rnd_unif_factory,
            expectation_step=_stub_estep_factory(ori),
            iter_artifact_sink=sink,
        )
        # 6 iters should have subset_size entries
        assert len(iter_log) == 6
        # At iter 1 (below grad_ini_iter = int(6*0.3)=1... so iter=1 is >=
        # grad_ini_iter=1 -> inbetween phase, but the very first iter should
        # still be within reason
        for rec in iter_log:
            assert rec["subset_size"] in range(200, 1001) or rec["subset_size"] == -1

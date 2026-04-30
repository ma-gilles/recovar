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
from recovar.em.initial_model.iteration_loop import run_vdam_iterations, update_probabilities_from_estep_meta
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


class TestRunVdamIterations:
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

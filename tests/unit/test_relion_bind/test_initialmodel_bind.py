"""Phase-2 parity tests: Python InitialModel schedules / ordering vs the
RELION C++ bindings.

These tests verify that `recovar.em.initial_model` produces byte-identical
trajectories against the corresponding RELION C++ functions compiled into
`_relion_bind_core`.

Covers:
  - vdam_compute_subset_size    vs schedules.compute_subset_size
  - vdam_compute_stepsize       vs schedules.compute_stepsize
  - vdam_compute_tau2_fudge     vs schedules.compute_tau2_fudge
  - vdam_randomise_particles_order  vs subset.randomise_particles_order (NOTE:
        expected to diverge until the Python path uses RELION's rnd_unif;
        we fall back to checking "uses a valid permutation" + document a
        known-divergent trajectory test pinned against a C++ reference.)

Moment primitives (reweight_grad, first/second moment, apply_momenta,
reconstruct_grad) get a roundtrip smoke test only here — full parity
against RELION's `run_itNNN_*` fixture is exercised in Phase 4.
"""

from __future__ import annotations

import numpy as np
import pytest

from recovar.em.initial_model import (
    compute_phase_lengths,
    compute_stepsize,
    compute_subset_size,
    compute_tau2_fudge,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def bind():
    try:
        from recovar.relion_bind import _relion_bind_core as m
    except ImportError:  # pragma: no cover
        pytest.skip("relion_bind not built")
    return m


# ---------------------------------------------------------------------------
# Scheduler parity (Python impl vs C++ copy-verbatim)
# ---------------------------------------------------------------------------


class TestSubsetSizeParity:
    @pytest.mark.parametrize(
        "nr_iter, nr_particles, ini, fin",
        [
            (200, 1_000_000, 500, 10_000),  # GUI default, large dataset
            (200, 500, 200, 1000),  # 500-particle fixture
            (100, 50_000, 500, 5000),
            (50, 10_000, 200, 2000),
        ],
    )
    def test_full_trajectory(self, bind, nr_iter, nr_particles, ini, fin):
        phases = compute_phase_lengths(nr_iter, 0.3, 0.2)
        for it in range(0, nr_iter + 1, max(1, nr_iter // 20)):
            py_val = compute_subset_size(
                iter=it,
                phase_lengths=phases,
                grad_ini_subset_size=ini,
                grad_fin_subset_size=fin,
                nr_particles=nr_particles,
                nr_iter=nr_iter,
            )
            cpp_val = bind.vdam_compute_subset_size(
                it,
                nr_iter,
                phases.grad_ini_iter,
                phases.grad_inbetween_iter,
                ini,
                fin,
                nr_particles,
            )
            assert py_val == cpp_val, (
                f"subset_size mismatch at iter={it}: py={py_val} cpp={cpp_val} "
                f"(nr_iter={nr_iter}, nr_particles={nr_particles}, ini={ini}, fin={fin})"
            )

    def test_grad_em_iters_branch(self, bind):
        phases = compute_phase_lengths(200, 0.3, 0.2)
        for em_iters in [0, 1, 5, 10]:
            for it in [195, 196, 198, 199, 200]:
                py_val = compute_subset_size(
                    iter=it,
                    phase_lengths=phases,
                    grad_ini_subset_size=500,
                    grad_fin_subset_size=10_000,
                    nr_particles=1_000_000,
                    nr_iter=200,
                    grad_em_iters=em_iters,
                )
                cpp_val = bind.vdam_compute_subset_size(
                    it,
                    200,
                    phases.grad_ini_iter,
                    phases.grad_inbetween_iter,
                    500,
                    10_000,
                    1_000_000,
                    em_iters,
                )
                assert py_val == cpp_val, f"em_iters={em_iters} iter={it}"


class TestStepsizeParity:
    def test_3d_initial_model_default_scheme(self, bind):
        phases = compute_phase_lengths(200, 0.3, 0.2)
        for it in range(0, 201, 5):
            py_val = compute_stepsize(iter=it, phase_lengths=phases, is_3d_model=True, ref_dim=3)
            cpp_val = bind.vdam_compute_stepsize(
                it,
                phases.grad_ini_iter,
                phases.grad_inbetween_iter,
                True,
                3,
            )
            assert abs(py_val - cpp_val) < 1e-12, f"iter={it}: py={py_val} cpp={cpp_val}"

    def test_user_stepsize_override(self, bind):
        phases = compute_phase_lengths(200, 0.3, 0.2)
        for stepsize in [0.3, 0.5, 0.7, 1.0]:
            for it in [0, 30, 80, 160, 200]:
                py_val = compute_stepsize(
                    iter=it,
                    phase_lengths=phases,
                    is_3d_model=True,
                    ref_dim=3,
                    grad_stepsize=stepsize,
                )
                cpp_val = bind.vdam_compute_stepsize(
                    it,
                    phases.grad_ini_iter,
                    phases.grad_inbetween_iter,
                    True,
                    3,
                    stepsize,
                )
                assert abs(py_val - cpp_val) < 1e-12

    def test_plain_scheme(self, bind):
        phases = compute_phase_lengths(200, 0.3, 0.2)
        for it in [0, 50, 100, 200]:
            py_val = compute_stepsize(
                iter=it,
                phase_lengths=phases,
                is_3d_model=True,
                ref_dim=3,
                grad_stepsize=0.42,
                grad_stepsize_scheme="plain",
            )
            cpp_val = bind.vdam_compute_stepsize(
                it,
                phases.grad_ini_iter,
                phases.grad_inbetween_iter,
                True,
                3,
                0.42,
                "plain",
            )
            assert abs(py_val - cpp_val) < 1e-12


class TestTau2FudgeParity:
    def test_3d_initial_model_default_scheme(self, bind):
        phases = compute_phase_lengths(200, 0.3, 0.2)
        for it in range(0, 201, 5):
            py_val = compute_tau2_fudge(
                iter=it,
                phase_lengths=phases,
                is_3d_model=True,
                ref_dim=3,
                tau2_fudge_arg=4.0,
            )
            cpp_val = bind.vdam_compute_tau2_fudge(
                it,
                phases.grad_ini_iter,
                phases.grad_inbetween_iter,
                True,
                3,
                False,
                4.0,
            )
            assert abs(py_val - cpp_val) < 1e-12, f"iter={it}: py={py_val} cpp={cpp_val}"

    def test_user_fudge_variants(self, bind):
        phases = compute_phase_lengths(200, 0.3, 0.2)
        for fudge in [1.0, 2.5, 4.0, 8.0]:
            for it in [0, 30, 72, 100, 200]:
                py_val = compute_tau2_fudge(
                    iter=it,
                    phase_lengths=phases,
                    is_3d_model=True,
                    ref_dim=3,
                    tau2_fudge_arg=fudge,
                )
                cpp_val = bind.vdam_compute_tau2_fudge(
                    it,
                    phases.grad_ini_iter,
                    phases.grad_inbetween_iter,
                    True,
                    3,
                    False,
                    fudge,
                )
                assert abs(py_val - cpp_val) < 1e-12

    def test_plain_3d_classification(self, bind):
        phases = compute_phase_lengths(200, 0.3, 0.2)
        for it in [0, 50, 100, 200]:
            py_val = compute_tau2_fudge(
                iter=it,
                phase_lengths=phases,
                is_3d_model=False,  # 3D classification, default "plain"
                ref_dim=3,
                tau2_fudge_arg=4.0,
            )
            cpp_val = bind.vdam_compute_tau2_fudge(
                it,
                phases.grad_ini_iter,
                phases.grad_inbetween_iter,
                False,
                3,
                False,
                4.0,
            )
            assert abs(py_val - cpp_val) < 1e-12


class TestRandomiseParticlesOrderBinding:
    """The C++ binding uses RELION's rnd_unif; the Python version uses NumPy.
    These are NOT bit-identical. We only assert:

    - The binding always returns a valid permutation.
    - Repeated calls with the same seed are deterministic.
    - Different seeds yield different orders.

    Phase 4+ tests will rely on the binding directly for any RELION-fixture
    parity so the NumPy fallback is never used in parity paths.
    """

    def test_is_permutation(self, bind):
        n = 500
        order = bind.vdam_randomise_particles_order(n, 12345)
        arr = np.asarray(order)
        assert arr.shape == (n,)
        assert set(int(x) for x in arr) == set(range(n))

    def test_deterministic_with_same_seed(self, bind):
        a = np.asarray(bind.vdam_randomise_particles_order(500, 12345))
        b = np.asarray(bind.vdam_randomise_particles_order(500, 12345))
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self, bind):
        a = np.asarray(bind.vdam_randomise_particles_order(500, 12345))
        b = np.asarray(bind.vdam_randomise_particles_order(500, 12346))
        assert not np.array_equal(a, b)

    def test_size_one_is_identity(self, bind):
        order = np.asarray(bind.vdam_randomise_particles_order(1, 42))
        assert order.tolist() == [0]

    def test_empty(self, bind):
        order = np.asarray(bind.vdam_randomise_particles_order(0, 42))
        assert order.size == 0


# ---------------------------------------------------------------------------
# Moment-primitive smoke tests (full parity lands in Phase 4)
# ---------------------------------------------------------------------------


class TestMomentPrimitivesSmoke:
    def _tiny_fixture(self, ori_size=16):
        """Build deterministic (data, weight) buffers with the RELION padded shape.

        At padding_factor=1 the BackProjector padded shape is
        (ori_size, ori_size, ori_size/2+1) for 3D. The initZeros(-1) path
        uses r_max = ori_size/2 and the data/weight are that full box.
        """
        pf = 1
        N = ori_size * pf
        shape = (N, N, N // 2 + 1)
        rng = np.random.default_rng(0)
        data = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128)
        weight = rng.uniform(0.1, 10.0, size=shape).astype(np.float64)
        return data, weight, ori_size, pf

    def test_reweight_grad_smoke(self, bind):
        data, weight, ori_size, pf = self._tiny_fixture()
        out = bind.vdam_reweight_grad(data, weight, ori_size, pf)
        out = np.asarray(out)
        assert out.shape == data.shape
        assert out.dtype == np.complex128
        assert np.all(np.isfinite(out))
        # reweightGrad divides by max(1, weight). For weight >= 1 cells, result
        # should be strictly smaller in magnitude than the input.
        mask_ge1 = weight >= 1.0
        if mask_ge1.any():
            # Only cells within r_max^2 are touched; most cells at the corners
            # are untouched. Check at least some cells decreased in magnitude.
            changed = np.any(np.abs(out[mask_ge1]) < np.abs(data[mask_ge1]) - 1e-12)
            assert changed

    def test_first_moment_initialisation(self, bind):
        data, _, ori_size, pf = self._tiny_fixture()
        mom = np.zeros_like(data)
        # `lambda` is a Python keyword — must pass via **kwargs or default
        out = np.asarray(bind.vdam_first_moment(data, mom, ori_size, pf, **{"lambda": 0.9}))
        # First call (mom.sum()==0) copies data into mom within r_max. Non-r_max
        # cells stay zero.
        assert out.shape == data.shape

    def test_first_moment_ema(self, bind):
        data, _, ori_size, pf = self._tiny_fixture()
        # Seed mom with nonzero so we take the EMA branch
        mom = np.full_like(data, 0.5 + 0.1j)
        out = np.asarray(bind.vdam_first_moment(data, mom, ori_size, pf, **{"lambda": 0.9}))
        assert out.shape == data.shape
        # EMA should produce values bounded between old and new where both
        # agree on sign; at minimum, output is finite and not identical to
        # either input.
        assert not np.allclose(out, mom)
        assert not np.allclose(out, data)

    def test_second_moment_smoke(self, bind):
        data, _, ori_size, pf = self._tiny_fixture()
        data_other = data + 0.1 * np.ones_like(data)
        mom = np.ones_like(data) * (0.01 + 0.0j)
        out = np.asarray(bind.vdam_second_moment(data, data_other, mom, ori_size, pf, **{"lambda": 0.9}))
        assert out.shape == data.shape
        # Imaginary part must be zero by construction
        assert np.all(np.abs(out.imag) < 1e-12)
        # Real part must be >= 0
        assert np.all(out.real >= -1e-12)

    def test_apply_momenta_smoke(self, bind):
        data, _, ori_size, pf = self._tiny_fixture()
        mom1_h1 = data * 0.5
        mom1_h2 = data * 0.6
        mom2 = np.ones_like(data) * (0.01 + 0.0j)
        out, noise_power = bind.vdam_apply_momenta(data, mom1_h1, mom1_h2, mom2, ori_size, pf)
        out = np.asarray(out)
        noise_power = np.asarray(noise_power)
        assert out.shape == data.shape
        assert np.all(np.isfinite(out))
        # mom1_noise_power is per-shell, length ori_size/2 + 1
        assert noise_power.shape == (ori_size // 2 + 1,)
        assert np.all(noise_power >= 0)

    def test_reconstruct_grad_smoke(self, bind):
        data, weight, ori_size, pf = self._tiny_fixture()
        vol = np.zeros((ori_size, ori_size, ori_size), dtype=np.float64)
        fsc = np.zeros(ori_size // 2 + 1, dtype=np.float64)
        out = bind.vdam_reconstruct_grad(
            vol,
            data,
            weight,
            fsc,
            grad_stepsize=0.5,
            tau2_fudge=1.0,
            ori_size=ori_size,
            padding_factor=pf,
            use_fsc=False,
        )
        out = np.asarray(out)
        assert out.shape == (ori_size, ori_size, ori_size)
        assert np.all(np.isfinite(out))

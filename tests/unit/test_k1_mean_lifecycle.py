"""Focused lifetime guards for K=1 references between EM iterations."""

import inspect
from pathlib import Path

import numpy as np
import pytest

from recovar.em.dense_single_volume import iteration_loop

pytestmark = pytest.mark.unit


def test_snapshot_and_release_previous_k1_means_owns_host_copies(monkeypatch):
    first = np.arange(24, dtype=np.float64).view(np.complex128).reshape(3, 4)
    second = (first + np.complex128(2.0 + 3.0j)).copy()
    means = [first, second]
    collect_states = []

    def _collect_after_release():
        collect_states.append(tuple(means))
        return 0

    monkeypatch.setattr(iteration_loop.gc, "collect", _collect_after_release)
    snapshots = iteration_loop._snapshot_and_release_previous_k1_means(means)

    assert means == [None, None]
    assert collect_states == [(None, None)]
    for snapshot, original in zip(snapshots, (first, second), strict=True):
        assert type(snapshot) is np.ndarray
        assert snapshot.flags.owndata
        assert not np.shares_memory(snapshot, original)
        np.testing.assert_array_equal(snapshot, original)

    first[...] = np.complex128(-9.0 + 4.0j)
    assert not np.array_equal(snapshots[0], first)


def test_k1_mean_release_precedes_tau_and_reconstruction():
    source = inspect.getsource(iteration_loop._run_relion_iteration_loop)
    initial_alias_release = source.index("del init_volume")
    release = source.index(
        "previous_means = _snapshot_and_release_previous_k1_means(means)"
    )
    tau_update = source.index(
        "regularization.compute_relion_tau2_from_weights(",
        release,
    )
    reconstruction = source.index(
        "_reconstruct_and_postprocess_means(",
        tau_update,
    )

    assert initial_alias_release < release < tau_update < reconstruction


def test_production_runner_leaves_cold_start_host_owned_until_normalization():
    repo_root = Path(iteration_loop.__file__).resolve().parents[3]
    runner_source = (repo_root / "scripts" / "run_full_refinement.py").read_text()
    call_start = runner_source.index("result = refine_single_volume(")
    call_stop = runner_source.index("options=RefinementOptions(", call_start)
    production_call = runner_source[call_start:call_stop]

    assert "init_volume=init_vol_ft," in production_call
    assert "init_volume=jnp.asarray(init_vol_ft)" not in production_call

"""The exact local engine must not clear JAX's caches on the production path.

`run_local_em_exact` does contain a `jax.clear_caches()`, but it sits inside the
score-dump branch that needs a dump directory, `RECOVAR_LOCAL_SCORE_DUMP_FORCE_SPLIT`
and a bucket holding a dump target. A report once read that call as running at the
end of every bucket and concluded that every program the local iteration touches is
re-traced per bucket. It is not: measured on the 10k/256 fixture at two local states
(Slurm job 14181654), the production path made zero calls across 80 and 162 buckets,
and the eager-dispatch cache grew monotonically while its hits climbed.

Forcing the call after every bucket in the same job took the local iteration from
1586 traces and 220 compiles to 34296 and 2194, so the difference is not subtle.
These tests pin the production path at zero calls, so a future change cannot make
the premise true without failing here.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from helpers.em_arrays import _hermitian_volume, _make_rotations
from test_refine_relion_mode import IMAGE_SIZE, VOLUME_SHAPE, MockDataset

from recovar.em.local.local_em_engine import run_local_em_exact
from recovar.em.local.local_layout import LocalHypothesisLayout

pytestmark = pytest.mark.unit


def _layout(n_images=4, per_image=2):
    total = n_images * per_image
    return LocalHypothesisLayout(
        n_global_rotations=total,
        n_pixels=n_images,
        n_psi=per_image,
        rotation_offsets=np.arange(0, total + 1, per_image, dtype=np.int64),
        rotation_ids_flat=np.arange(total, dtype=np.int32),
        rotations_flat=np.asarray(_make_rotations(total, seed=5), dtype=np.float32),
        rotation_log_priors_flat=np.zeros(total, dtype=np.float32),
        rotation_counts=np.full(n_images, per_image, dtype=np.int32),
        translation_grid=np.zeros((1, 2), dtype=np.float32),
        translation_log_priors=np.zeros((n_images, 1), dtype=np.float32),
    )


def _run(monkeypatch, image_batch_size, **kwargs):
    """Drive the engine with `jax.clear_caches` counted, and return the count."""

    calls = []
    original = jax.clear_caches

    def counting(*args, **kw):
        calls.append(True)
        return original(*args, **kw)

    monkeypatch.setattr(jax, "clear_caches", counting)
    # The engine reaches it through the module object, so patch that binding too.
    import recovar.em.local.local_em_engine as engine

    monkeypatch.setattr(engine.jax, "clear_caches", counting, raising=False)

    dataset = MockDataset(4, np.random.default_rng(20260920))
    result = run_local_em_exact(
        dataset,
        _hermitian_volume(VOLUME_SHAPE, seed=17),
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        _layout(),
        "linear_interp",
        image_batch_size=image_batch_size,
        rotation_block_size=4,
        current_size=6,
        reconstruct_significant_only=False,
        return_profile=False,
        **{"accumulate_noise": True, **kwargs},
    )
    assert result is not None
    return len(calls)


@pytest.mark.parametrize("image_batch_size", [1, 2, 4])
def test_production_local_iteration_never_clears_jax_caches(monkeypatch, image_batch_size):
    """One bucket or four, the production path clears nothing."""

    monkeypatch.delenv("RECOVAR_LOCAL_SCORE_DUMP_DIR", raising=False)
    monkeypatch.delenv("RECOVAR_LOCAL_SCORE_DUMP_FORCE_SPLIT", raising=False)
    assert _run(monkeypatch, image_batch_size) == 0


def test_score_only_parent_probe_never_clears_jax_caches(monkeypatch):
    """The pass-1 parent probe is what the local search actually runs here."""

    monkeypatch.delenv("RECOVAR_LOCAL_SCORE_DUMP_DIR", raising=False)
    monkeypatch.delenv("RECOVAR_LOCAL_SCORE_DUMP_FORCE_SPLIT", raising=False)
    assert (
        _run(
            monkeypatch,
            2,
            score_only=True,
            disable_adjoint_y=True,
            disable_adjoint_ctf=True,
            accumulate_noise=False,
        )
        == 0
    )


def test_the_clear_is_reachable_only_behind_the_score_dump_guard():
    """Pin the guard, so the call cannot drift out of the debug branch.

    Read as source rather than executed: reaching the branch needs a dump
    directory, the force-split flag and a bucket holding a dump target, which no
    production or harness environment sets.
    """

    import inspect

    from recovar.em.local import local_em_engine

    source = inspect.getsource(local_em_engine.run_local_em_exact)
    assert source.count("jax.clear_caches()") == 1
    guard = "if debug_score_dump_force_split and debug_score_dump_bucket_matches:"
    assert guard in source
    body = source.split(guard, 1)[1]
    head, _, _ = body.partition("timing.host_stats_s +=")
    assert "jax.clear_caches()" in head, (
        "the local engine's clear_caches left the score-dump branch"
    )

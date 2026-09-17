"""Opt-in prototype: block BPref accumulation for soft-posterior iterations."""

import numpy as np
import pytest

import jax.numpy as jnp

from recovar.em.sparse_pass2 import sparse_pass2_adjoint as adjoint_mod
from recovar.em.sparse_pass2 import sparse_pass2_bucketed as bucketed_mod
from recovar.em.helpers.env_flags import parse_env_flag


def test_prototype_is_off_by_default(monkeypatch):
    monkeypatch.delenv(bucketed_mod._SOFT_POSTERIOR_BLOCK_BPREF_PROTOTYPE_ENV, raising=False)
    assert parse_env_flag(bucketed_mod._SOFT_POSTERIOR_BLOCK_BPREF_PROTOTYPE_ENV, default=False) is False


@pytest.mark.parametrize(
    "enabled, live, wta, expected",
    [
        (False, True, False, False),   # default: never changes the live mode
        (True, True, False, True),     # soft-posterior iteration under the guard -> block path
        (True, True, True, False),     # winner-take-all firstiter keeps particle launches
        (True, False, False, False),   # nothing to switch when launches are already block
    ],
)
def test_prototype_decision(enabled, live, wta, expected):
    assert (
        bucketed_mod._soft_posterior_block_bpref_active(
            prototype_enabled=enabled, live_per_particle_launches=live, winner_take_all=wta
        )
        is expected
    )


def test_block_and_per_particle_paths_accumulate_the_same_active_rows(monkeypatch):
    """Fixed-operand contract on a mocked adjoint: identical active (row, rotation) multiset.

    The per-particle path issues one (data, ctf) launch pair per particle with that
    particle's first ``count`` rows; the block path issues one launch pair with every
    flattened row, padded rows carrying exact zeros.  Both must feed the adjoint the
    same non-zero rows with the same rotations.
    """
    rng = np.random.default_rng(0)
    B, R, P = 3, 4, 5
    counts = np.asarray([4, 1, 2], dtype=np.int32)
    values = jnp.asarray(rng.standard_normal((B, R, P)) + 1j * rng.standard_normal((B, R, P)), dtype=jnp.complex64)
    ctf = jnp.asarray(np.abs(rng.standard_normal((B, R, P))), dtype=jnp.float32)
    rot = jnp.asarray(rng.standard_normal((B, R, 3, 3)), dtype=jnp.float32)
    pad = jnp.arange(R)[None, :] < jnp.asarray(counts)[:, None]
    values = jnp.where(pad[:, :, None], values, 0)
    ctf = jnp.where(pad[:, :, None], ctf, 0)

    seen = {"per_particle": [], "block": []}
    mode = {"k": None}

    def fake_adjoint(half_block, window_indices, rotations_block, volume_in, *a, **k):
        seen[mode["k"]].append((np.asarray(half_block), np.asarray(rotations_block)))
        return volume_in

    monkeypatch.setattr(adjoint_mod, "_adjoint_slice_volume_windowed", fake_adjoint)
    common = dict(window_indices=jnp.arange(P, dtype=jnp.int32), image_shape=(8, 8), volume_shape=(8, 8, 8),
                  disc_type="linear_interp", half_volume=True, max_r=2.0)
    mode["k"] = "per_particle"
    adjoint_mod._accumulate_relion_x_half_per_particle_launches(
        values, ctf, rot, counts, jnp.zeros(1, jnp.complex64), jnp.zeros(1, jnp.float32),
        log_label_prefix="t", winner_take_all=False, strict_particle_order=True, **common,
    )
    mode["k"] = "block"
    flat_v = values.reshape(B * R, P); flat_c = ctf.reshape(B * R, P); flat_r = rot.reshape(B * R, 3, 3)
    adjoint_mod._accumulate_adjoint_block_chunked(
        flat_v, flat_r, jnp.zeros(1, jnp.complex64), window_indices=common["window_indices"], use_windowed_adjoint=True,
        image_shape=(8, 8), volume_shape=(8, 8, 8), disc_type="linear_interp", half_image=True, half_volume=True,
        max_r=2.0, relion_x_half=True, max_block_bytes=1 << 30, log_label="t",
    )
    adjoint_mod._accumulate_adjoint_block_chunked(
        flat_c, flat_r, jnp.zeros(1, jnp.float32), window_indices=common["window_indices"], use_windowed_adjoint=True,
        image_shape=(8, 8), volume_shape=(8, 8, 8), disc_type="linear_interp", half_image=True, half_volume=True,
        max_r=2.0, relion_x_half=True, max_block_bytes=1 << 30, log_label="t",
    )
    assert len(seen["per_particle"]) == 2 * B and len(seen["block"]) == 2

    def active_rows(calls):
        out = []
        for block, rots in calls:
            for row, r in zip(block, rots):
                if np.any(row != 0):
                    out.append((row.tobytes(), r.tobytes()))
        return sorted(out)

    # data launches: per-particle calls alternate (data, ctf); block calls are (data), (ctf)
    pp_data = active_rows(seen["per_particle"][0::2]); pp_ctf = active_rows(seen["per_particle"][1::2])
    bl_data = active_rows(seen["block"][0:1]); bl_ctf = active_rows(seen["block"][1:2])
    assert pp_data == bl_data and len(pp_data) == int(counts.sum())
    assert pp_ctf == bl_ctf and len(pp_ctf) == int(counts.sum())


def test_run_chunking_default_and_rungs():
    from recovar.em.scoring.sparse_bucket_arrays import _split_run_into_chunks

    run = list(range(11))
    assert [len(c) for c in _split_run_into_chunks(run, 8, image_rungs=False)] == [8, 3]
    assert [len(c) for c in _split_run_into_chunks(run, 8, image_rungs=True)] == [8, 2, 1]
    assert [len(c) for c in _split_run_into_chunks(list(range(6)), 8, image_rungs=True)] == [4, 2]
    assert [len(c) for c in _split_run_into_chunks(list(range(16)), 8, image_rungs=True)] == [8, 8]
    # every image appears exactly once, in order
    for rungs in (False, True):
        chunks = _split_run_into_chunks(run, 8, image_rungs=rungs)
        assert sum(chunks, []) == run


def test_bucket_builder_rungs_yield_power_of_two_image_counts():
    import numpy as np
    from recovar.em.scoring.sparse_bucket_arrays import _bucket_pass2_inputs

    rng = np.random.default_rng(0)
    n = 37
    counts = rng.integers(600, 9000, size=n)
    per_image_inputs = {"oversampled_rots": [np.zeros((int(c), 3, 3), dtype=np.float32) for c in counts]}
    common = dict(n_fine_trans=84, rotation_block_size_for_quantization=4096, max_hypotheses_per_microbatch=7_653_710,
                  max_images_per_microbatch=212, processing_order_override=np.arange(n), processing_order_group_by_bucket_size=True)
    plain = _bucket_pass2_inputs(per_image_inputs, **common)
    rungs = _bucket_pass2_inputs(per_image_inputs, **common, group_chunk_image_rungs=True)
    covered = sorted(int(i) for b in rungs for i in b["image_indices"])
    assert covered == list(range(n))
    for b in rungs:
        k = len(b["image_indices"])
        assert k & (k - 1) == 0, k  # power of two
    # same padded rotation rows per image (grouping unchanged; only chunk boundaries move)
    rows = lambda bs: sum(int(b["bucket_size"]) * len(b["image_indices"]) for b in bs)
    assert rows(plain) == rows(rungs)

"""Fine-projection cache admission for the HEALPix-3 / current_size-92 regime.

The 10k EMPIAR-10097 convergence run (job 14031616) skipped the sparse pass-2
projection cache in every hp3 iteration ("estimated transient 18.36 GiB
exceeds cap 7.96 GiB") and recomputed fine projections per image chunk.
These checks pin the admission rule that lets an 80 GB device cache the
294912-rotation hp3 projections while a 40 GB device still declines.
"""

import numpy as np

from recovar.em.sparse_pass2.sparse_pass2_budget import (
    _projection_cache_fits_budget,
    _projection_cache_max_bytes_for_pass,
    _projection_cache_transient_bytes,
    _projection_call_max_bytes_for_pass,
)

_HP3_FINE_ROTATIONS = 294912
_CS92_HALF_PIXELS = 3386  # windowed half-spectrum score pixels at current_size 92, 256^2
_H100_BYTES = int(79.65 * 1024**3)
_A100_40_BYTES = 40 * 1024**3


def _clear_env(monkeypatch):
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES", raising=False)
    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS", raising=False)


def test_hp3_cs92_cache_estimate_matches_logged_value():
    transient = _projection_cache_transient_bytes(
        _HP3_FINE_ROTATIONS, _CS92_HALF_PIXELS, projection_complex_dtype=np.complex64, include_abs2=True,
    )
    # score (complex64) + recon (complex64 shares the score cache when not
    # windowed) is not double counted here; the logged 18.36 GiB includes the
    # windowed score+recon+abs2 triple, so only check the single-cache term.
    assert transient == _HP3_FINE_ROTATIONS * _CS92_HALF_PIXELS * (8 + 4)


def test_hp3_cache_admitted_on_80gb_rejected_on_40gb(monkeypatch):
    _clear_env(monkeypatch)
    logged_hp3_estimate = int(18.36 * 1024**3)
    assert _projection_cache_fits_budget(logged_hp3_estimate, _projection_cache_max_bytes_for_pass(_H100_BYTES))
    assert not _projection_cache_fits_budget(logged_hp3_estimate, _projection_cache_max_bytes_for_pass(_A100_40_BYTES))


def test_cache_cap_is_quarter_of_device_memory_without_override(monkeypatch):
    _clear_env(monkeypatch)
    assert _projection_cache_max_bytes_for_pass(_H100_BYTES) == int(_H100_BYTES * 0.25)


def test_cache_cap_change_leaves_per_call_rotation_budget_alone(monkeypatch):
    _clear_env(monkeypatch)
    # The per-call projected-rotation budget (809 rotations per call at cs 92 on
    # an H100 in the measured runs) must not grow with the cache admission cap.
    assert _projection_call_max_bytes_for_pass(_H100_BYTES) == int(_H100_BYTES * 0.040)
    assert _projection_call_max_bytes_for_pass(_H100_BYTES) < _projection_cache_max_bytes_for_pass(_H100_BYTES)


def test_env_override_still_wins(monkeypatch):
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_PROJECTION_CACHE_MAX_BYTES", "654321")
    assert _projection_cache_max_bytes_for_pass(_H100_BYTES) == 654321


def test_cache_build_rotations_per_call_scales_scoring_budget(monkeypatch):
    from recovar.em.sparse_pass2.sparse_pass2_budget import _projection_cache_build_max_rotations_per_call

    _clear_env(monkeypatch)
    assert _projection_cache_build_max_rotations_per_call(809, 294912) == 4 * 809
    # never more rotations than the fine grid holds, never below one
    assert _projection_cache_build_max_rotations_per_call(809, 1000) == 1000
    assert _projection_cache_build_max_rotations_per_call(None, 294912) is None
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_MAX_PROJECTED_ROTATIONS", "4096")
    assert _projection_cache_build_max_rotations_per_call(809, 294912) == 4096


def test_flatten_bucket_rotations_is_host_side_for_numpy_and_not_jitted():
    from recovar.em.local import local_backprojection as lb

    rots = np.arange(2 * 3 * 9, dtype=np.float32).reshape(2, 3, 3, 3)
    out = lb.flatten_bucket_rotations(rots)
    assert isinstance(out, np.ndarray) and out.shape == (6, 3, 3)
    np.testing.assert_array_equal(out, rots.reshape(6, 3, 3))
    assert not hasattr(lb.flatten_bucket_rotations, "lower")  # plain function, not a jit wrapper
    import jax.numpy as jnp
    dev = lb.flatten_bucket_rotations(jnp.asarray(rots))
    assert dev.shape == (6, 3, 3)
    np.testing.assert_array_equal(np.asarray(dev), rots.reshape(6, 3, 3))


def test_per_particle_launch_rung_is_power_of_two_at_or_above_count():
    from recovar.em.sparse_pass2.sparse_pass2_adjoint import _per_particle_launch_rung

    assert [_per_particle_launch_rung(c, 4096) for c in (0, 1, 2, 3, 4, 5, 100, 1000)] == [0, 1, 2, 4, 4, 8, 128, 1024]
    assert _per_particle_launch_rung(3000, 2048) == 2048  # never beyond the bucket rows


def test_per_particle_launches_pad_to_rungs_with_zeroed_spare_rows(monkeypatch):
    import jax.numpy as jnp
    from recovar.em.sparse_pass2 import sparse_pass2_adjoint as adjoint_mod

    values = (1.0 + jnp.arange(2 * 6 * 2, dtype=jnp.float32)).reshape(2, 6, 2).astype(jnp.complex64)
    ctf_values = (100.0 + jnp.arange(2 * 6 * 2, dtype=jnp.float32)).reshape(2, 6, 2)
    rotations = jnp.arange(2 * 6 * 9, dtype=jnp.float32).reshape(2, 6, 3, 3)
    actual_counts = np.asarray([3, 5], dtype=np.int32)  # rungs 4 and 8 -> 8 capped to 6 bucket rows
    calls = []

    def fake_adjoint(block, window_indices, rotations_block, volume, image_shape, volume_shape, disc_type, half_image, half_volume, max_r, relion_x_half):
        calls.append((np.asarray(block).copy(), np.asarray(rotations_block).copy()))
        return volume

    monkeypatch.setattr(adjoint_mod, "_adjoint_slice_volume_windowed", fake_adjoint)
    monkeypatch.delenv("RECOVAR_RELION_X_HALF_BP_PARTICLE_POOL_SIZE", raising=False)
    adjoint_mod._accumulate_relion_x_half_per_particle_launches(
        values, ctf_values, rotations, actual_counts, jnp.zeros((4,), jnp.complex64), jnp.zeros((4,), jnp.float32),
        window_indices=jnp.arange(2), image_shape=(8, 8), volume_shape=(8, 8, 8), disc_type="linear_interp",
        half_volume=True, max_r=4.0, log_label_prefix="test",
    )
    assert [c[0].shape[0] for c in calls] == [4, 4, 6, 6]
    # particle 0: rows 0-2 live, row 3 zeroed; particle 1: rows 0-4 live, row 5 zeroed
    np.testing.assert_array_equal(calls[0][0][:3], np.asarray(values[0, :3])); assert np.all(calls[0][0][3:] == 0)
    np.testing.assert_array_equal(calls[1][0][:3], np.asarray(ctf_values[0, :3])); assert np.all(calls[1][0][3:] == 0)
    np.testing.assert_array_equal(calls[2][0][:5], np.asarray(values[1, :5])); assert np.all(calls[2][0][5:] == 0)
    np.testing.assert_array_equal(calls[0][1], np.asarray(rotations[0, :4]))


def test_large_bucket_pow2_rung_is_opt_in(monkeypatch):
    from recovar.em.scoring import sparse_bucket_arrays as sba

    monkeypatch.delenv("RECOVAR_SPARSE_PASS2_LARGE_BUCKET_POW2", raising=False)
    monkeypatch.delenv("RECOVAR_LOCAL_BUCKET_QUANTUM", raising=False)
    default = [sba._pass2_bucket_rotation_size(c, 5000) for c in (7, 100, 900, 5000, 9000, 20000, 100000, 217000)]
    assert default[:3] == [16, 128, 1024]  # small supports: shared power-of-two rule, unchanged
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_LARGE_BUCKET_POW2", "1")
    pow2 = [sba._pass2_bucket_rotation_size(c, 5000) for c in (7, 100, 900, 5000, 9000, 20000, 100000, 217000)]
    assert pow2[:3] == default[:3]
    assert pow2[3:] == [8192, 16384, 32768, 131072, 262144]
    assert all(p >= d for p, d in zip(pow2, default))  # never smaller than the shared quantiser

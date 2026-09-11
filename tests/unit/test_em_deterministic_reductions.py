"""Fixed-order segment sums behind RECOVAR_EM_DETERMINISTIC_REDUCTIONS.

The opt-in replaces duplicate-index scatter-adds (shell binning, per-group
scale terms, per-image residual totals) with masked reductions.  These checks
pin the helper's semantics against the host reference and prove the flagged
``bin_shell_values_jax`` path matches the scatter path on the same operands.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from recovar.em.dense_single_volume.helpers import deterministic_reduce as dr
from recovar.em.dense_single_volume.helpers.half_spectrum import (
    bin_shell_values_jax,
    bin_shell_values_np,
)

jax.config.update("jax_platform_name", "cpu")


def _shell_case(seed: int = 0):
    rng = np.random.default_rng(seed)
    n_pix, n_shells = 517, 9
    # Include sentinel (== n_shells) and out-of-range ids, as the engines do.
    ids = rng.integers(-1, n_shells + 2, size=n_pix).astype(np.int32)
    values = rng.standard_normal(n_pix)
    return values, ids, n_shells


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv(dr.DETERMINISTIC_REDUCTIONS_ENV, raising=False)
    assert dr.deterministic_reductions_enabled() is False
    monkeypatch.setenv(dr.DETERMINISTIC_REDUCTIONS_ENV, "1")
    assert dr.deterministic_reductions_enabled() is True


def test_fixed_order_segment_sum_matches_host_bincount_and_drops_out_of_range():
    values, ids, n = _shell_case()
    got = np.asarray(dr.fixed_order_segment_sum(jnp.asarray(values), ids, n))
    expected = bin_shell_values_np(values, ids, n)
    assert got.shape == (n,)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
    # Integer-valued float32 inputs reduce exactly in any order.
    ints = np.round(values * 4).astype(np.float32)
    got32 = np.asarray(dr.fixed_order_segment_sum(jnp.asarray(ints), ids, n))
    assert got32.dtype == np.float32
    np.testing.assert_array_equal(got32, bin_shell_values_np(ints, ids, n).astype(np.float32))


def test_fixed_order_segment_sum_leading_axes_and_shape_check():
    values, ids, n = _shell_case(1)
    stacked = np.stack([values, 2.0 * values, -values], axis=0)
    got = np.asarray(dr.fixed_order_segment_sum(jnp.asarray(stacked), ids, n))
    assert got.shape == (3, n)
    for row, scale in zip(got, (1.0, 2.0, -1.0)):
        np.testing.assert_allclose(row, scale * bin_shell_values_np(values, ids, n), atol=1e-12)
    with pytest.raises(ValueError, match="segment ids"):
        dr.fixed_order_segment_sum(jnp.asarray(values[:-1]), ids, n)


def test_add_segment_sum_matches_scatter_under_both_flag_states(monkeypatch):
    rng = np.random.default_rng(3)
    n_groups, batch = 4, 37
    group_ids = rng.integers(0, n_groups, size=batch).astype(np.int32)
    per_image = rng.standard_normal(batch).astype(np.float32)
    acc0 = jnp.asarray(rng.standard_normal(n_groups).astype(np.float32))
    monkeypatch.delenv(dr.DETERMINISTIC_REDUCTIONS_ENV, raising=False)
    scatter = np.asarray(dr.add_segment_sum(acc0, group_ids, per_image))
    monkeypatch.setenv(dr.DETERMINISTIC_REDUCTIONS_ENV, "1")
    fixed = np.asarray(dr.add_segment_sum(acc0, group_ids, per_image))
    assert fixed.dtype == scatter.dtype == np.float32
    reference = np.asarray(acc0, dtype=np.float64) + np.bincount(
        group_ids, weights=per_image.astype(np.float64), minlength=n_groups
    )
    np.testing.assert_allclose(scatter, reference, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(fixed, reference, rtol=1e-5, atol=1e-6)


def test_bin_shell_values_jax_flagged_path_matches_scatter_path(monkeypatch):
    values, ids, n = _shell_case(5)
    monkeypatch.delenv(dr.DETERMINISTIC_REDUCTIONS_ENV, raising=False)
    scatter = np.asarray(bin_shell_values_jax(jnp.asarray(values), ids, n))
    monkeypatch.setenv(dr.DETERMINISTIC_REDUCTIONS_ENV, "1")
    fixed = np.asarray(bin_shell_values_jax(jnp.asarray(values), ids, n))
    assert fixed.shape == scatter.shape == (n,)
    np.testing.assert_allclose(fixed, scatter, rtol=0, atol=1e-12)
    np.testing.assert_allclose(fixed, bin_shell_values_np(values, ids, n), atol=1e-12)


def test_fixed_order_segment_sum_is_bitwise_repeatable_under_jit():
    values, ids, n = _shell_case(7)
    fn = jax.jit(lambda v: dr.fixed_order_segment_sum(v, ids, n))
    first = np.asarray(fn(jnp.asarray(values, dtype=jnp.float32)))
    for _ in range(3):
        np.testing.assert_array_equal(np.asarray(fn(jnp.asarray(values, dtype=jnp.float32))), first)


def test_scatter_flat_local_rows_flagged_path_matches_set_and_fixes_duplicates(monkeypatch):
    from recovar.em.dense_single_volume.helpers.flat_local_rows import scatter_flat_local_rows

    rng = np.random.default_rng(11)
    batch, n_rot, n_trans = 3, 5, 4
    values = (rng.standard_normal((9, n_trans)) + 1j * rng.standard_normal((9, n_trans))).astype(np.complex64)
    image = np.array([0, 0, 1, 1, 2, 2, 2, 0, 0], dtype=np.int32)
    rot = np.array([0, 3, 1, 4, 2, 0, 3, 1, 1], dtype=np.int32)  # rows 7 and 8 duplicate (0, 1); last two are padding
    present = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=bool)
    kw = dict(batch_size=batch, dense_rotation_count=n_rot, fill_value=np.inf)
    monkeypatch.delenv(dr.DETERMINISTIC_REDUCTIONS_ENV, raising=False)
    scatter = np.asarray(scatter_flat_local_rows(values, image, rot, present, **kw))
    monkeypatch.setenv(dr.DETERMINISTIC_REDUCTIONS_ENV, "1")
    fixed = np.asarray(scatter_flat_local_rows(values, image, rot, present, **kw))
    assert fixed.shape == scatter.shape == (batch, n_rot, n_trans)
    np.testing.assert_array_equal(fixed, scatter)  # unique present rows: identical result, padding dropped
    assert np.isinf(fixed[0, 1]).all() and np.isinf(fixed[2, 4]).all()
    # Duplicate present rows: the flagged path deterministically keeps the highest packed row id.
    present_dup = present.copy(); present_dup[7] = present_dup[8] = True
    fixed_dup = np.asarray(scatter_flat_local_rows(values, image, rot, present_dup, **kw))
    np.testing.assert_array_equal(fixed_dup[0, 1], values[8])
    for _ in range(3):
        np.testing.assert_array_equal(np.asarray(scatter_flat_local_rows(values, image, rot, present_dup, **kw)), fixed_dup)


def test_powerclass_spectrum_flagged_path_matches_host_binning(monkeypatch):
    """Under the opt-in the RELION powerClass spectrum is binned in XLA with the kernel's shell rule."""
    from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
        _relion_cuda_powerclass_spectrum_norm_units,
    )

    n, half = 16, 9
    rng = np.random.default_rng(2)
    centred = (rng.standard_normal((3, n, half)) + 1j * rng.standard_normal((3, n, half))).astype(np.complex64)
    monkeypatch.setenv(dr.DETERMINISTIC_REDUCTIONS_ENV, "1")
    got = np.asarray(_relion_cuda_powerclass_spectrum_norm_units(centred.reshape(3, -1), image_shape=(n, n), current_size=8))
    assert got.shape == (3, half)
    # Host reference: RELION layout (rows rolled by n//2), 1/N^2 scaling, shell = rint(sqrt(x^2+y^2)),
    # shells 1..half-1, half-row x==0,y<0 excluded, then N^4 back-scaling.
    rel = np.roll(centred, -(n // 2), axis=1) / np.float32(n * n)
    rows = np.arange(n)[:, None]; cols = np.arange(half)[None, :]
    y = np.where(rows < half, rows, rows - n); shell = np.rint(np.sqrt((cols * cols + y * y).astype(np.float32))).astype(int)
    valid = (shell > 0) & (shell < half) & ~((cols == 0) & (y < 0))
    power = (rel.real.astype(np.float64) ** 2 + rel.imag.astype(np.float64) ** 2)
    ref = np.stack([np.bincount(shell[valid], weights=power[i][valid], minlength=half)[:half] for i in range(3)]) * float(n * n) ** 2
    np.testing.assert_allclose(got, ref, rtol=2e-6, atol=0)


def test_mstep_fixed_order_shell_sums_match_scatter_rule():
    from recovar.em.dense_single_volume.helpers.deterministic_reduce import (
        fixed_order_shell_sums,
        static_shell_voxel_lists,
    )

    capacity, pf, n_shells = 19, 1, 9  # ori_size 16, padding 1 -> capacity 19
    lists = static_shell_voxel_lists(capacity, pf, n_shells)
    clamped = static_shell_voxel_lists(capacity, pf, n_shells, clamp_to_last=True)
    assert sum(len(l) for l in clamped) == capacity * capacity * (capacity // 2 + 1)
    coord = np.arange(capacity) - capacity // 2; x = np.arange(capacity // 2 + 1)
    r2 = coord[:, None, None] ** 2 + coord[None, :, None] ** 2 + x[None, None, :] ** 2
    shells = np.floor(np.sqrt(r2.astype(np.float64)) / pf + 0.5).astype(int).reshape(-1)
    assert sum(len(l) for l in lists) == int((shells < n_shells).sum())
    rng = np.random.default_rng(4); flat = rng.random(shells.size).astype(np.float32)
    got = np.asarray(fixed_order_shell_sums(jnp.asarray(flat), lists, jnp.float32))
    ref = np.bincount(np.where(shells < n_shells, shells, n_shells), weights=flat.astype(np.float64), minlength=n_shells + 1)[:n_shells]
    np.testing.assert_allclose(got, ref, rtol=2e-6)


def test_score_only_microbatch_cap_ignores_live_memory_under_opt_in(monkeypatch):
    # the cap lives in the batch planner on this branch, not in the engine module
    from recovar.em.dense_single_volume import local_batch_planning as le

    calls = []
    monkeypatch.setattr(le, "_exact_local_runtime_free_memory_bytes", lambda: calls.append(1) or (1 << 20))
    monkeypatch.setattr(le, "_exact_local_planned_hypotheses_floor", lambda *a, **k: 4096)
    monkeypatch.setattr(le, "_exact_local_max_hypotheses_per_microbatch", lambda *a, **k: 2048)
    monkeypatch.setattr(le, "_exact_local_microbatch_env_overridden", lambda: False)
    monkeypatch.setattr(le, "_exact_local_auto_microbatch_boost", lambda: 2.0)
    kw = dict(n_trans=8, n_recon_windowed=None, local_layout=None, image_batch_size=8, rotation_block_size=8, score_only=True)
    monkeypatch.delenv(dr.DETERMINISTIC_REDUCTIONS_ENV, raising=False)
    probed = le._exact_local_effective_max_hypotheses_per_microbatch(None, 512, **kw)
    assert calls and probed <= 4096
    monkeypatch.setenv(dr.DETERMINISTIC_REDUCTIONS_ENV, "1")
    calls.clear()
    fixed = le._exact_local_effective_max_hypotheses_per_microbatch(None, 512, **kw)
    assert fixed == 2048 and not calls

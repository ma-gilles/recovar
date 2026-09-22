"""Recovered stable-window and fused-chunk donor regressions."""
import logging
import re
import numpy as np
import pytest
from test_compact_real_rows_integration import _fused_kclass_multibucket_fixture, _fused_kclass_result_arrays, _assert_fused_arrays_identical

@pytest.mark.parametrize("logical_size", [22, 26, 28])
def test_fused_translate_scorer_runtime_logical_size_matches_static_kernel(
    monkeypatch, custom_cuda_lib, gpu_device, logical_size
):
    """The compact fused-translate scorer, given a physical pixel capacity and a runtime
    logical size, returns bit-identical costs to the static kernel on the logical operands,
    and the runtime program is shared across logical sizes in one physical class."""
    import jax
    import jax.numpy as jnp
    import recovar.cuda_backproject as cuda_backproject
    from recovar.em.sparse_pass2 import sparse_pass2_bucketed as bucketed_mod
    from recovar.em.helpers.fourier_window import make_stable_fourier_window_shape_plan

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    rng = np.random.default_rng(2026_09_15)
    image_size = 32
    image_shape = (image_size, image_size)
    n_half = image_size * (image_size // 2 + 1)
    plan = make_stable_fourier_window_shape_plan(
        image_shape, logical_size, n_half, enabled=True, quantum=8,
    )
    # quantum-8 ladder in a 32 box: 22 -> 24, 26 and 28 -> 30; every case carries a real tail
    assert plan.physical_current_size > logical_size
    logical_spec = plan.logical_spec
    packed = plan.packed_physical_spec()
    n_logical = int(logical_spec.n_score)
    n_physical = int(packed.n_score)
    assert n_physical > n_logical, "the physical class must add tail pixels or the test is vacuous"
    batch, rows, n_trans, pairs = 2, 3, 4, 5
    proj = (rng.normal(size=(batch, rows, n_logical)) + 1j * rng.normal(size=(batch, rows, n_logical))).astype(np.complex64)
    image = (rng.normal(size=(batch, n_logical)) + 1j * rng.normal(size=(batch, n_logical))).astype(np.complex64)
    corr = rng.uniform(0.5, 2.0, size=(batch, n_logical)).astype(np.float32)
    half_weights = rng.uniform(0.5, 2.0, size=(n_logical,)).astype(np.float32)
    angles = rng.normal(0, 0.2, size=(n_trans, 2)).astype(np.float32)
    local_rotation_row = rng.integers(0, rows, size=(batch, pairs)).astype(np.int32)
    translation_idx = rng.integers(0, n_trans, size=(batch, pairs)).astype(np.int32)
    pair_mask = np.ones((batch, pairs), dtype=bool); pair_mask[1, -1] = False
    highres = rng.uniform(0, 1, size=(batch,)).astype(np.float32)
    logical_lookup = bucketed_mod._relion_cuda_fine_full_to_compact_lookup(
        image_shape, logical_size, logical_spec.score_indices_np,
    )
    pad = n_physical - n_logical
    def padc(a): return np.pad(a, [(0, 0)] * (a.ndim - 1) + [(0, pad)], constant_values=np.complex64(9 + 4j))
    def padf(a): return np.pad(a, [(0, 0)] * (a.ndim - 1) + [(0, pad)], constant_values=np.float32(3.5))
    physical_rect = plan.physical_rectangle_pixels
    physical_lookup = np.pad(logical_lookup, (0, physical_rect - logical_lookup.size), constant_values=-1)
    with jax.default_device(gpu_device):
        expected = bucketed_mod._score_pass2_pairs_relion_gpu_diff2_raw_fused_translate(
            jnp.asarray(image), jnp.asarray(corr), jnp.asarray(proj), jnp.asarray(half_weights),
            jnp.asarray(angles), jnp.asarray(local_rotation_row), jnp.asarray(translation_idx),
            jnp.asarray(pair_mask), jnp.asarray(logical_lookup), jnp.asarray(highres),
            current_size=logical_size,
        )
        actual = bucketed_mod._score_pass2_pairs_relion_gpu_diff2_raw_fused_translate(
            jnp.asarray(padc(image)), jnp.asarray(padf(corr)), jnp.asarray(padc(proj)), jnp.asarray(padf(half_weights)),
            jnp.asarray(angles), jnp.asarray(local_rotation_row), jnp.asarray(translation_idx),
            jnp.asarray(pair_mask), jnp.asarray(physical_lookup), jnp.asarray(highres),
            current_size=plan.physical_current_size, logical_current_size=logical_size,
        )
        expected, actual = jax.block_until_ready((expected, actual))
    expected = np.asarray(expected); actual = np.asarray(actual)
    assert expected.shape == actual.shape == (batch, pairs)
    assert np.all(np.isinf(expected[~pair_mask])) and np.all(np.isinf(actual[~pair_mask]))
    np.testing.assert_array_equal(actual[pair_mask].view(np.uint32), expected[pair_mask].view(np.uint32))


def _stable_window_fixture_env(monkeypatch, flag):
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MAX_IMAGES_PER_MICROBATCH", "4")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_FUSED_TRANSLATE", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_ADJOINT_REAL_ROWS", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_STABLE_WINDOWS", flag)


def test_stable_windows_fall_back_to_the_logical_window_without_the_fused_scorer(monkeypatch, caplog):
    """Without the fused-translate Gaussian scorer (here: CUDA disabled) the flag must leave
    every output untouched and say so."""
    from recovar.em.sparse_pass2 import sparse_pass2_bucketed as bucketed_mod

    def run(flag):
        monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")
        _stable_window_fixture_env(monkeypatch, flag)
        kwargs = _fused_kclass_multibucket_fixture(n_images=13)
        kwargs["accumulate_noise"] = True
        return _fused_kclass_result_arrays(bucketed_mod.compute_k_class_pass2_stats_sparse_fused(**kwargs))

    base = run("0")
    with caplog.at_level(logging.INFO):
        on = run("1")
    assert any("stable windows requested but not applicable" in r.getMessage() for r in caplog.records)
    _assert_fused_arrays_identical(base, on, "stable windows fallback")


def test_fused_chunk_scoring_matches_the_chunk_loop_on_gpu(monkeypatch, caplog, custom_cuda_lib, gpu_device):
    """Step 1 of the fused chunk program: per-class raw diff2, the joint minimum, the score
    conversion and the log normalizers in one program. Same kernels in the same order, so
    every output except the two adjoint volumes (GPU atomics) is bit-identical to the loop."""
    import jax
    import recovar.cuda_backproject as cuda_backproject
    from recovar.em.sparse_pass2 import sparse_pass2_bucketed as bucketed_mod

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    calls = []
    original = bucketed_mod._fused_chunk_scores_and_log_z

    def spy(*a, **kw):
        calls.append(len(a[6]))
        return original(*a, **kw)

    monkeypatch.setattr(bucketed_mod, "_fused_chunk_scores_and_log_z", spy)

    def run(flag):
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS", "1")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE", "1")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MAX_IMAGES_PER_MICROBATCH", "4")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_FUSED_TRANSLATE", "1")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_ADJOINT_REAL_ROWS", "1")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_FUSED_CHUNK", flag)
        calls.clear()
        kwargs = _fused_kclass_multibucket_fixture(n_images=13)
        kwargs["accumulate_noise"] = True
        kwargs["relion_exact_fine_gaussian"] = True
        kwargs["relion_f32_fine_posterior"] = True
        kwargs["relion_x_half_mstep"] = True
        with jax.default_device(gpu_device):
            return _fused_kclass_result_arrays(bucketed_mod.compute_k_class_pass2_stats_sparse_fused(**kwargs))

    base_a = run("0")
    assert not calls
    base_b = run("0")
    on = run("1")
    assert calls and all(k == 2 for k in calls), calls[:5]
    # Which outputs does this configuration reproduce at all? The adjoint volumes
    # (CUDA atomics) and wsum_img_power (atomic shell scatter-add) differ between two
    # control runs; every key the control reproduces bit for bit must not move under
    # the flag, and a key it does not reproduce is held to the same float32 atomics
    # bound, with the control's own spread reported next to the flag's delta.
    reproducible, spread = [], {}
    for k in base_a:
        a, b = np.asarray(base_a[k]), np.asarray(base_b[k])
        d = float(np.max(np.abs(np.nan_to_num(a) - np.nan_to_num(b)))) if a.size else 0.0
        (reproducible.append(k) if d == 0.0 else spread.__setitem__(k, d))
    assert reproducible, "the two control runs agreed on nothing"
    _assert_fused_arrays_identical(
        {k: base_a[k] for k in reproducible}, {k: on[k] for k in reproducible}, "fused chunk scoring",
    )
    for k in sorted(spread):
        x = np.asarray(base_a[k]); y = np.asarray(on[k])
        scale = float(np.max(np.abs(np.nan_to_num(x)))) or 1.0
        err = float(np.max(np.abs(np.nan_to_num(y) - np.nan_to_num(x))))
        np.testing.assert_allclose(np.nan_to_num(y), np.nan_to_num(x), rtol=1e-5, atol=1e-5 * scale,
                                   err_msg=f"{k}: flag delta {err:.3e} (control spread {spread[k]:.3e}), max|ref|={scale:.3e}")


@pytest.mark.parametrize("current_size", [2, 4])  # the fixture box is 8 pixels; both sizes get a physical tail
def test_stable_windows_match_the_logical_window_on_gpu(monkeypatch, caplog, custom_cuda_lib, gpu_device, current_size):
    """Inside a physical Fourier-window class the compact engine reproduces the logical
    window: scores, posteriors, assignments and evidence bit for bit (the runtime kernel
    stops at the logical size); noise statistics and the two adjoint volumes within the
    bounds this file uses for reductions over a differently shaped axis and for GPU
    atomics. The planner log must show a physical class larger than the logical size,
    or the test is vacuous."""
    import jax
    import recovar.cuda_backproject as cuda_backproject
    from recovar.em.sparse_pass2 import sparse_pass2_bucketed as bucketed_mod

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    def run(flag):
        _stable_window_fixture_env(monkeypatch, flag)
        kwargs = _fused_kclass_multibucket_fixture(n_images=13)
        kwargs["accumulate_noise"] = True
        kwargs["current_size"] = current_size
        kwargs["relion_exact_fine_gaussian"] = True
        kwargs["relion_f32_fine_posterior"] = True  # production posterior dtype; the native dual sums require it
        kwargs["relion_x_half_mstep"] = True
        with jax.default_device(gpu_device):
            return _fused_kclass_result_arrays(bucketed_mod.compute_k_class_pass2_stats_sparse_fused(**kwargs))

    base = run("0")
    with caplog.at_level(logging.INFO):
        on = run("1")
    plan_lines = [r.getMessage() for r in caplog.records if "stable windows: logical current_size" in r.getMessage()]
    why = [r.getMessage() for r in caplog.records if "stable windows requested but not applicable" in r.getMessage()]
    assert plan_lines, f"the stable-window plan was not applied: {why}"
    m = re.search(r"logical current_size (\d+) -> physical class (\d+) \(score pixels (\d+) -> (\d+)", plan_lines[-1])
    assert m and int(m.group(2)) > int(m.group(1)) and int(m.group(4)) > int(m.group(3)), plan_lines[-1]
    bounded = {k for k in base if k.startswith(("Ft_y", "Ft_ctf", "noise_stats"))}
    _assert_fused_arrays_identical(
        {k: v for k, v in base.items() if k not in bounded},
        {k: v for k, v in on.items() if k not in bounded},
        "stable windows",
    )
    for k in sorted(bounded):
        x = np.asarray(base[k]); y = np.asarray(on[k])
        assert x.shape == y.shape and x.dtype == y.dtype, k
        tol = 1e-5 if k.startswith(("Ft_y", "Ft_ctf")) else 1e-6
        scale = float(np.max(np.abs(np.nan_to_num(x)))) or 1.0
        err = float(np.max(np.abs(np.nan_to_num(y) - np.nan_to_num(x))))
        np.testing.assert_allclose(np.nan_to_num(y), np.nan_to_num(x), rtol=tol, atol=tol * scale,
                                   err_msg=f"{k}: max|diff|={err:.3e}, max|ref|={scale:.3e}, rel={err/scale:.3e}")

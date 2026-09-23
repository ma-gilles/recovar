"""Recovered real-row adjoint, native sums and group-scale regressions."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_sparse_pass2_bucketed_perf import MockDataset, VOLUME_SHAPE, IMAGE_SIZE, _hermitian_volume
from test_compact_capacity_integration import _fused_kclass_result_arrays

def _fused_kclass_multibucket_fixture(n_images=12, seed=5):
    """Heterogeneous candidate counts so images land in several pair-width buckets.

    The single-bucket fixture cannot see a record/leaf misalignment in the deferred
    host-statistics replay; at 100k/256 the replay raised the padding guard for class 4
    in iteration 2 while the immediate path passed (job 13799858).
    """
    from recovar.em.sampling import rotation_grid_size

    rng = np.random.default_rng(seed)
    n_classes = 2
    rotation_grid_size(0)
    fine_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 4, axis=0)
    fine_parent = np.asarray([0, 1, 2, 3], dtype=np.int64)
    fine_translations = np.asarray([[0.0, 0.0], [0.25, 0.0], [0.0, 0.25]], dtype=np.float32)
    fine_translation_parent = np.zeros(3, dtype=np.int32)
    significant_by_class = [
        [np.sort(rng.choice(4, size=int(rng.integers(1, 5)), replace=False)).astype(np.int32) for _ in range(n_images)]
        for _ in range(n_classes)
    ]
    volumes = jnp.stack(
        [_hermitian_volume(VOLUME_SHAPE, seed=2027), _hermitian_volume(VOLUME_SHAPE, seed=2029)]
    )
    return dict(
        experiment_dataset=MockDataset(n_images=n_images, seed=2039),
        volumes=volumes,
        noise_variance=jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        translations=np.asarray([[0.0, 0.0]], dtype=np.float32),
        significant_sample_indices_by_class=significant_by_class,
        rotation_log_priors_by_class=[None] * n_classes,
        translation_log_prior=np.array([-0.25], dtype=np.float32),
        nside_level=0,
        disc_type="linear_interp",
        oversampling_order=0,
        current_size=4,
        half_spectrum_scoring=True,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
        fine_translations_override=fine_translations,
        fine_translation_parent_override=fine_translation_parent,
        relion_x_half_mstep=False,
        relion_fine_mstep_prune_mode="joint",
        adaptive_fraction=0.9,
    )


def _assert_fused_arrays_identical(baseline, candidate, label):
    assert baseline and set(baseline) == set(candidate)
    mismatched = [
        name for name, expected in baseline.items()
        if expected.shape != candidate[name].shape
        or not np.array_equal(
            np.nan_to_num(expected, nan=0.0, posinf=1e30, neginf=-1e30),
            np.nan_to_num(candidate[name], nan=0.0, posinf=1e30, neginf=-1e30),
        )
    ]
    assert not mismatched, f"{label} changed: {mismatched}"


@pytest.mark.parametrize("noise_mode", ["no_noise", "noise"])
def test_compact_adjoint_real_rows_matches_dense_rows(monkeypatch, noise_mode):
    """Skipping the padded rotation rows in the M-step adjoint must not change any output.

    Rows at or beyond ``actual_counts`` carry exactly zero posterior, so the adjoint
    of the dense rows and of the real rows agree bit for bit on CPU. The row indices
    come from host metadata (no device pull). At 100k/256 only 27 % of the padded
    rows are real and the adjoint was 203 s of iteration 2 (job 13804810)."""
    from recovar.em.sparse_pass2 import sparse_pass2_bucketed as bucketed_mod

    calls = []
    original = bucketed_mod._real_flat_row_indices_from_actual_counts

    def counting(*a, **kw):
        out = original(*a, **kw)
        calls.append(int(out[2]))
        return out

    monkeypatch.setattr(bucketed_mod, "_real_flat_row_indices_from_actual_counts", counting)

    def run(flag):
        monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS", "1")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE", "1")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_ACTIVE_ROWS", "0")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MAX_IMAGES_PER_MICROBATCH", "4")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_ADJOINT_REAL_ROWS", flag)
        kwargs = _fused_kclass_multibucket_fixture(n_images=13)
        if noise_mode == "noise":
            kwargs["accumulate_noise"] = True
        return _fused_kclass_result_arrays(
            bucketed_mod.compute_k_class_pass2_stats_sparse_fused(**kwargs)
        )

    dense = run("0")
    assert not calls
    real = run("1")
    assert calls and all(c > 0 for c in calls), "real-rows adjoint never engaged"
    # Everything except the two adjoint volumes is bit-identical. Gathering the
    # real rows changes the XLA scatter's reduction order, which on CPU moved 7
    # of 512 Ft_y voxels by one float64 ULP (rel 2e-16); bound that explicitly.
    volumes = {k for k in dense if k.startswith(("Ft_y", "Ft_ctf"))}
    _assert_fused_arrays_identical(
        {k: v for k, v in dense.items() if k not in volumes},
        {k: v for k, v in real.items() if k not in volumes},
        f"real-rows adjoint ({noise_mode})",
    )
    for k in sorted(volumes):
        x = np.asarray(dense[k]); y = np.asarray(real[k])
        assert x.shape == y.shape and x.dtype == y.dtype
        np.testing.assert_allclose(y, x, rtol=4e-16 * 8, atol=0.0, err_msg=k)


def test_real_flat_row_indices_from_actual_counts_layout():
    from recovar.em.sparse_pass2 import sparse_pass2_bucketed as bucketed_mod

    idx, mask, count = bucketed_mod._real_flat_row_indices_from_actual_counts([2, 0, 3], 4, pad_multiple=1)
    np.testing.assert_array_equal(idx, [0, 1, 8, 9, 10])
    np.testing.assert_array_equal(mask, [1, 1, 1, 1, 1])
    assert count == 5
    idx, mask, count = bucketed_mod._real_flat_row_indices_from_actual_counts([2, 0, 3], 4, pad_multiple=4)
    np.testing.assert_array_equal(idx, [0, 1, 8, 9, 10, 0, 0, 0])
    np.testing.assert_array_equal(mask, [1, 1, 1, 1, 1, 0, 0, 0])
    assert count == 5
    idx, mask, count = bucketed_mod._real_flat_row_indices_from_actual_counts([0, 0], 4)
    assert idx.size == 0 and mask.size == 0 and count == 0
    # pad_multiple > 1 pads to the multiple, and snaps to a power of two only when that
    # wastes at most an eighth of the rows: 5 rows at multiple 3 -> 6 (a snap to 8 would
    # waste a third), 11 rows at multiple 3 -> 12 (already the row ceiling).
    idx, mask, count = bucketed_mod._real_flat_row_indices_from_actual_counts([2, 0, 3], 4, pad_multiple=3)
    assert idx.shape == (6,) and count == 5 and mask.sum() == 5
    idx, mask, count = bucketed_mod._real_flat_row_indices_from_actual_counts([4, 4, 3], 4, pad_multiple=3)
    assert idx.shape == (12,) and count == 11
    # a count just below a power of two does snap (15 rows at multiple 1 stays 15; at
    # multiple 5 -> 15 -> 16 wastes a fifteenth)
    idx, mask, count = bucketed_mod._real_flat_row_indices_from_actual_counts([5, 5, 5], 8, pad_multiple=5)
    assert idx.shape == (16,) and count == 15 and mask.sum() == 15


@pytest.mark.gpu
@pytest.mark.parametrize("device_scalars", ["0", "1"])
@pytest.mark.parametrize("defer_flag", ["0", "1"])
@pytest.mark.parametrize("noise_mode", ["noise", "noise_with_scale_groups"])
def test_flat_real_rows_sums_and_noise_are_bit_identical(monkeypatch, custom_cuda_lib, gpu_device, device_scalars, defer_flag, noise_mode):
    """RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_FLAT_ROWS computes the pair-sparse weighted sums,
    CTF sums and noise terms only for the real rotation rows (flat [rows, pixel] layout)
    instead of the padded [images, rows, pixel] layout. Each real row is the padded row
    bit for bit and padded rows carry exactly zero mass, so every output must be
    bit-identical to the padded path with the real-rows adjoint."""
    import recovar.cuda_backproject as cuda_backproject
    from recovar.em.cuda import kernels as em_cuda_kernels
    from recovar.em.sparse_pass2 import sparse_pass2_bucketed as bucketed_mod

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    calls = []
    original = em_cuda_kernels.dual_weighted_sums_pairs_rows_f32

    def spy(*a, **kw):
        calls.append(1)
        return original(*a, **kw)

    def run(flat):
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS", "1")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE", "1")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MAX_IMAGES_PER_MICROBATCH", "4")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_DEVICE_CHUNK_SCALARS", device_scalars)
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_DEFERRED_HOST_STATS", defer_flag)
        monkeypatch.setenv("RECOVAR_SPARSE_PASS2_IMAGE_CAPACITY", "1")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_LAZY_TABLES", "1")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_ADJOINT_REAL_ROWS", "1")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_NATIVE_PAIR_SPARSE_SUMS", "1")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_FLAT_ROWS", flat)
        monkeypatch.setattr(em_cuda_kernels, "dual_weighted_sums_pairs_rows_f32", spy)
        # the fused stage is jitted: retrace so the (Python-level) spy sees the kernel call
        bucketed_mod._compact_pair_weighted_sums_and_noise_native.clear_cache()
        kwargs = _fused_kclass_multibucket_fixture(n_images=13)
        kwargs["accumulate_noise"] = True
        if noise_mode == "noise_with_scale_groups":
            # the group-scale XA/AA terms take their own flat-row form (job 13829375 crashed here)
            kwargs["group_ids"] = np.arange(13) % 3
            kwargs["scale_corrections"] = np.linspace(0.9, 1.1, 13).astype(np.float32)
        # the fused native sums/noise path (a prerequisite) is the exact Gaussian x-half M-step contract
        kwargs["relion_x_half_mstep"] = True
        kwargs["relion_f32_fine_posterior"] = True  # current native F32 posterior contract
        with jax.default_device(gpu_device):
            return _fused_kclass_result_arrays(
                bucketed_mod.compute_k_class_pass2_stats_sparse_fused(**kwargs)
            )

    padded = run("0")
    assert not calls, "the flat-row kernel must not run with the flag off"
    flat = run("1")
    assert calls, "the flat-row kernel must run with the flag on"
    # The x-half BPref adjoint accumulates with device atomics, so the reconstruction
    # volumes are order-dependent at the float32 ULP level whenever the rows arrive in a
    # different launch layout (the real-rows adjoint pin, f089f37e6, bounds them the same
    # way); every other output — sums, CTF sums, noise statistics, posteriors, best poses —
    # must be bit-identical.
    volume_keys = {k for k in padded if k.startswith(("Ft_y", "Ft_ctf"))}
    # The posterior-mass reductions now sit in a different XLA program (the flat-row
    # stage), so XLA fuses them differently and the noise statistics they feed move by a
    # float32 ULP or two (as ctf_probs did when the pair-sparse sums landed, 048c30d20).
    # Bound them; posteriors, best poses and class evidences stay bitwise.
    noise_keys = {k for k in padded if k.startswith("noise_stats")}
    for name in sorted(noise_keys):
        a = np.asarray(padded[name]); b = np.asarray(flat[name])
        assert a.shape == b.shape and a.dtype == b.dtype, name
        np.testing.assert_allclose(b, a, rtol=8 * np.finfo(np.float32).eps, atol=0.0, err_msg=name)
    for name in sorted(padded):
        a = np.asarray(padded[name]); b = np.asarray(flat[name])
        if a.shape != b.shape or not np.array_equal(np.nan_to_num(a), np.nan_to_num(b)):
            diff = np.abs(np.nan_to_num(a).astype(np.complex128) - np.nan_to_num(b).astype(np.complex128))
            scale = np.abs(np.nan_to_num(a).astype(np.complex128))
            print(f"DIFF {name}: shape {a.shape} dtype {a.dtype} n_mismatch {int((diff > 0).sum())} max_abs {diff.max():.3e} max_rel {np.max(diff / np.maximum(scale, 1e-30)):.3e}")
    for name in sorted(volume_keys):
        a = np.asarray(padded[name]); b = np.asarray(flat[name])
        assert a.shape == b.shape and a.dtype == b.dtype, name
        eps = np.finfo(np.asarray(a).real.dtype).eps
        np.testing.assert_allclose(b, a, rtol=4 * eps, atol=4 * eps * max(1.0, float(np.max(np.abs(a)))), err_msg=name)
    bounded = volume_keys | noise_keys
    _assert_fused_arrays_identical(
        {k: v for k, v in padded.items() if k not in bounded},
        {k: v for k, v in flat.items() if k not in bounded},
        f"flat real rows ({noise_mode}, defer={defer_flag}, device_scalars={device_scalars})",
    )


def test_flat_real_rows_flag_requires_its_prerequisites(monkeypatch):
    from recovar.em.sparse_pass2 import sparse_pass2_bucketed as bucketed_mod

    monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_FLAT_ROWS", "1")
    monkeypatch.delenv("RECOVAR_SPARSE_KCLASS_COMPACT_ADJOINT_REAL_ROWS", raising=False)
    kwargs = _fused_kclass_multibucket_fixture(n_images=6)
    with pytest.raises(ValueError, match="compact pair flat rows require"):
        bucketed_mod.compute_k_class_pass2_stats_sparse_fused(**kwargs)




def test_fused_translate_capacity_skips_only_absent_pair_gather(monkeypatch):
    from recovar.em.sparse_pass2.sparse_pass2_adjoint import _split_compact_pair_buckets_by_projection_gather_budget
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_GROUP_PAIR_BUCKETS_BY_ROTATION_SIGNATURE", "0")
    bucket = {"pair_bucket_size": 1024, "image_indices": np.array([0, 1]), "class_bucket_sizes": (4,)}
    kwargs = dict(n_score_pixels=16, n_recon_pixels=16, projection_complex_dtype=np.complex64,
                  max_gather_bytes=65536, max_prepare_images_per_microbatch=16,
                  rotation_block_size_for_quantization=4)
    gathered = _split_compact_pair_buckets_by_projection_gather_budget([bucket], (), **kwargs)
    fused = _split_compact_pair_buckets_by_projection_gather_budget([bucket], (), skip_diff2_gather_budget=True, **kwargs)
    assert gathered[0]["image_capacity_budget"] == 1  # 2 * 1024 * 16 * sizeof(complex64)
    assert fused[0]["image_capacity_budget"] == 16  # still limited by preparation
    # The byte budget also bounds the chunk size (as in the S donor splitter), so
    # the one-image diff2 budget splits the gathered bucket per image, while the
    # fused route, which skips only that absent gather, keeps both images.
    assert [bucket["image_indices"].tolist() for bucket in gathered] == [[0], [1]]
    assert all(bucket["image_capacity_budget"] == 1 for bucket in gathered)
    assert [bucket["image_indices"].tolist() for bucket in fused] == [[0, 1]]

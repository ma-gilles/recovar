"""Donor fused-noise guards adapted to bounded statistics ownership."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_sparse_pass2_bucketed_perf import MockDataset, VOLUME_SHAPE, VOLUME_SIZE, IMAGE_SIZE, _hermitian_volume
from test_compact_capacity_integration import _fused_kclass_result_arrays
from test_compact_real_rows_integration import _fused_kclass_multibucket_fixture, _assert_fused_arrays_identical
from recovar.em.classification.k_class import _run_sparse_k_class_adaptive_pass2
from recovar.em.classification import k_class_results as results

def build_device_chunk_scalars_gpu_fixture():
    """Inputs of the fused-noise device-chunk-scalars guard.

    Module level so the GPU guard and the atomic-spread probe that measures the
    unchanged-source run-to-run band use byte-identical inputs.
    """

    from recovar.em.sampling import rotation_grid_size

    n_images = 7
    n_coarse_rot = rotation_grid_size(1)
    fine_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 6, axis=0)
    fine_parent = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    fine_translations = np.asarray([[0.0, 0.0], [0.5, 0.0], [0.0, 1.0], [0.5, 1.0]], dtype=np.float32)
    fine_translation_parent = np.asarray([0, 0, 1, 1], dtype=np.int32)
    coarse_translations = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    rng = np.random.default_rng(5)
    significant_by_class = [
        [np.sort(rng.choice(12, size=int(rng.integers(2, 5)), replace=False)).astype(np.int32) for _ in range(n_images)]
        for _ in range(2)
    ]
    ds = MockDataset(n_images=n_images, seed=93)
    volumes = jnp.stack([_hermitian_volume(VOLUME_SHAPE, seed=111), _hermitian_volume(VOLUME_SHAPE, seed=113)])
    return dict(
        experiment_dataset=ds,
        means_array=volumes,
        mean_variance=jnp.ones(VOLUME_SIZE, dtype=jnp.float32) * 10.0,
        noise_variance=jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        coarse_rotations_np=np.repeat(np.eye(3, dtype=np.float32)[None], n_coarse_rot, axis=0),
        coarse_translations_np=coarse_translations,
        fine_rotations_np=fine_rotations,
        fine_mstep_rotations_np=None,
        rot_parent_map_np=fine_parent,
        fine_translations_np=fine_translations,
        trans_parent_map_np=fine_translation_parent,
        sig_sample_indices_by_class=significant_by_class,
        disc_type="linear_interp",
        class_log_priors=np.log(np.asarray([0.45, 0.55], dtype=np.float64)),
        accumulate_noise=True,
        return_best_pose_details=True,
        oversampling_order=1,
        random_perturbation=0.0,
        engine_kwargs={
            "current_size": None,
            "relion_half_volume_mstep": False,
            "mstep_relion_x_half": True,
            # Native fused sums require the joint float32 posterior.
            "relion_fine_mstep_prune": True,
            "adaptive_fraction": 0.75,
        },
    )

@pytest.mark.gpu
def test_device_chunk_scalars_gpu_fused_noise_accumulator_calls(monkeypatch, custom_cuda_lib, gpu_device):
    """GPU-only guard: with the fused weighted-sums/noise path (native CUDA dual sums, RELION
    x-half M-step, image capacity padding) RECOVAR_SPARSE_KCLASS_DEVICE_CHUNK_SCALARS must
    actually run the device noise accumulator (call count asserted, lead review 2026-09-13);
    the assignments and the two noise totals must equal the host accumulation bit for bit,
    while the CUDA-atomic Ft_y/Ft_ctf are bounded at 1e-6 relative."""

    import jax

    import recovar.cuda_backproject as cb
    from recovar.em.sparse_pass2 import sparse_pass2_bucketed as bucketed_mod
    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cb, "_cuda_ok", None)

    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("device chunk scalars fused-noise guard requires a JAX GPU device")
    if not cb.cuda_available():
        pytest.skip(cb.cuda_unavailable_error())

    kwargs = build_device_chunk_scalars_gpu_fixture()
    calls = []
    native_calls = []
    original_native = bucketed_mod._compact_pair_weighted_sums_and_noise_native

    def native_spy(*args, **kwargs):
        native_calls.append(1)
        return original_native(*args, **kwargs)

    monkeypatch.setattr(bucketed_mod, "_compact_pair_weighted_sums_and_noise_native", native_spy)
    original = results._accumulate_noise_totals_device

    # Named parameters, not *args: the accumulator's signature gained the leak flag and a
    # positional spy silently mis-indexed it (job 13838987).
    def spy(wsum_total, norm_total, padded_leak, image_indices, n_real_images, block_shells, block_norm_residual):
        calls.append(int(np.asarray(image_indices).shape[0]) - int(n_real_images))
        return original(
            wsum_total, norm_total, padded_leak, image_indices, n_real_images, block_shells, block_norm_residual
        )

    def run(flag):
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_FUSED", "1")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS", "1")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE", "1")
        # 7 images in one microbatch of up to 16: the capacity ladder floor is 16, the
        # growth limit max(16, 2*7) = 16 and the budget 16, so the chunk pads to 16 images
        # (nine duplicate indices).
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MAX_IMAGES_PER_MICROBATCH", "16")
        monkeypatch.setenv("RECOVAR_SPARSE_PASS2_IMAGE_CAPACITY", "1")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_NATIVE_PAIR_SPARSE_SUMS", "1")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_NATIVE_DUAL_WEIGHTED_SUMS", "1")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_FUSED_MSTEP_NOISE", "1")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_DEFERRED_HOST_STATS", "1")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_DEVICE_CHUNK_SCALARS", flag)
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_RELION_FINE_MSTEP_PRUNE", "joint")
        monkeypatch.setattr(results, "_accumulate_noise_totals_device", spy)
        calls.clear()
        native_calls.clear()
        with jax.default_device(gpu_device):
            result = _run_sparse_k_class_adaptive_pass2(**kwargs)
        assert native_calls, "the native fused weighted-sums/noise path must execute"
        return result, list(calls)

    host, host_calls = run("0")
    assert host_calls == [], "the device accumulator must not run with the flag off"
    device, device_calls = run("1")
    assert device_calls, "the fused-noise path must reach the device accumulator with the flag on"
    assert any(padded > 0 for padded in device_calls), "image-capacity padding must exercise duplicate indices"
    # The RELION x-half BPref accumulators use CUDA atomics whose order varies run to
    # run (1 ULP, documented for the flat-row pin); the flag does not touch them, so
    # they are bounded rather than bitwise. Every statistic the flag produces is bitwise.
    for name in ("Ft_y", "Ft_ctf"):
        np.testing.assert_allclose(
            np.asarray(getattr(device, name)), np.asarray(getattr(host, name)), rtol=1e-6, atol=0.0, err_msg=name
        )
    for name in ("per_class_hard_assignments", "class_assignments", "pose_assignments"):
        np.testing.assert_array_equal(np.asarray(getattr(device, name)), np.asarray(getattr(host, name)), err_msg=name)
    # Scope (lead review 2026-09-13): the three assignment fields exact; the two noise
    # totals the flag produces exact for every class; Ft_y/Ft_ctf bounded above.
    host_noise = host.noise_stats
    device_noise = device.noise_stats
    assert host_noise is not None and device_noise is not None
    assert len(host_noise) == len(device_noise) == 2
    for class_index, (h, d) in enumerate(zip(host_noise, device_noise)):
        for field in ("wsum_sigma2_noise", "wsum_norm_correction"):
            np.testing.assert_array_equal(np.asarray(getattr(d, field)), np.asarray(getattr(h, field)), err_msg=f"{field} class {class_index}")

@pytest.mark.gpu
@pytest.mark.parametrize("noise_mode", ["noise", "noise_with_scale_groups"])
def test_defer_fused_noise_totals_is_bit_identical_and_defers_two_leaves(
    monkeypatch, custom_cuda_lib, gpu_device, noise_mode
):
    """Compare repeated immediate controls with bounded deferred residual callbacks.

    Every reproducible output remains exact; other outputs use the unchanged
    donor bound and report the measured control spread. Each deferred residual
    callback must carry exactly two leaves. The current owner has no separate
    fused-noise deferral switch: host mode0/1 exercises its immediate/deferred path.
    """
    import recovar.cuda_backproject as cuda_backproject
    from recovar.em.sparse_pass2 import sparse_pass2_bucketed as bucketed_mod

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    original_append = results.DeferredHostUpdates.append
    native_calls = []
    original_native = bucketed_mod._compact_pair_weighted_sums_and_noise_native
    def native_spy(*args, **kwargs):
        native_calls.append(1)
        return original_native(*args, **kwargs)
    monkeypatch.setattr(bucketed_mod, "_compact_pair_weighted_sums_and_noise_native", native_spy)
    def run(flag):
        native_calls.clear()
        deferred_residual_leaves = []
        def spy(queue, update, *, host, device):
            if getattr(update, "__name__", "") == "residual" and queue.max_records > 1:
                assert set(device) == {"block_noise_shells", "block_norm_residual"}
                deferred_residual_leaves.append(len(device))
            return original_append(queue, update, host=host, device=device)
        monkeypatch.setattr(results.DeferredHostUpdates, "append", spy)
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS", "1")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE", "1")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MAX_IMAGES_PER_MICROBATCH", "4")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_DEFERRED_HOST_STATS", flag)
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_DEVICE_CHUNK_SCALARS", "0")
        monkeypatch.setenv("RECOVAR_SPARSE_PASS2_IMAGE_CAPACITY", "1")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_ADJOINT_REAL_ROWS", "1")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_NATIVE_DUAL_WEIGHTED_SUMS", "1")
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_FUSED_MSTEP_NOISE", "1")
        kwargs = _fused_kclass_multibucket_fixture(n_images=13)
        kwargs["accumulate_noise"] = True
        kwargs["relion_f32_fine_posterior"] = True
        if noise_mode == "noise_with_scale_groups":
            kwargs["group_ids"] = np.arange(13) % 3
            kwargs["scale_corrections"] = np.linspace(0.9, 1.1, 13).astype(np.float32)
        kwargs["relion_x_half_mstep"] = True
        with jax.default_device(gpu_device):
            result = _fused_kclass_result_arrays(bucketed_mod.compute_k_class_pass2_stats_sparse_fused(**kwargs))
        assert native_calls, "the native fused weighted-sums/noise path must execute"
        return result, sum(deferred_residual_leaves)

    off_a, leaves_off = run("0")
    off_b, _ = run("0")
    on, leaves_on = run("1")
    label = f"RECOVAR_SPARSE_KCLASS_DEFERRED_HOST_STATS ({noise_mode})"

    def _num(value):
        arr = np.asarray(value)
        if np.iscomplexobj(arr):
            return np.nan_to_num(arr.astype(np.complex128))
        return np.nan_to_num(arr.astype(np.float64))

    # Which outputs does this configuration actually reproduce? The fused M-step and the
    # x-half BPref adjoint both accumulate with CUDA atomics, so some outputs differ
    # between two processes at unchanged configuration. Measure that instead of assuming
    # it: run the control twice. A key the two controls reproduce bit for bit must not
    # move at all under the flag; a key they do not reproduce is bounded by the same
    # float32 atomics bound the reviewed padded-versus-flat-rows test uses, and the
    # failure message reports the control spread next to the flag's delta.
    assert set(off_a) == set(off_b) == set(on)
    reproducible, nondeterministic = [], {}
    for name in sorted(off_a):
        a, b = _num(off_a[name]), _num(off_b[name])
        if a.shape != b.shape:
            raise AssertionError(f"{label}: {name} changed shape between two control runs")
        spread = float(np.max(np.abs(a - b))) if a.size else 0.0
        if spread == 0.0:
            reproducible.append(name)
        else:
            nondeterministic[name] = spread
    assert reproducible, f"{label}: the two control runs agreed on nothing"
    _assert_fused_arrays_identical(
        {k: off_a[k] for k in reproducible},
        {k: on[k] for k in reproducible},
        label,
    )
    for name, spread in sorted(nondeterministic.items()):
        a, c = _num(off_a[name]), _num(on[name])
        flag_delta = float(np.max(np.abs(a - c)))
        scale = max(1.0, float(np.max(np.abs(a))))
        bound = 4 * np.finfo(np.float32).eps * scale
        assert flag_delta <= max(bound, spread), (
            f"{label}: {name} moved by {flag_delta:.3e} under the flag, beyond both the "
            f"{bound:.3e} float32 atomics bound and the {spread:.3e} spread measured "
            "between two runs of the control"
        )
    assert leaves_on > leaves_off, (
        "the mode must actually defer residual records; the bounded queue "
        f"carried {leaves_on} leaves with the flag on and {leaves_off} with it off, so "
        "no additional residual leaves were deferred"
    )
    assert (leaves_on - leaves_off) % 2 == 0, (
        "each deferred fused-noise record contributes exactly two leaves (shells and "
        f"norm residual); saw {leaves_on - leaves_off} extra"
    )

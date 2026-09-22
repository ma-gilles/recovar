"""Integrated compact-index/capacity regression using the donor's small fixture."""
import numpy as np
import pytest
import jax.numpy as jnp
from test_sparse_pass2_bucketed_perf import MockDataset, VOLUME_SHAPE, IMAGE_SIZE, _hermitian_volume
from recovar.em.sparse_pass2 import sparse_pass2_bucketed as bucketed_mod

def _fused_kclass_capacity_fixture():
    """Small exactly-K2 fused pass-2 call used to compare image-axis capacities."""

    from recovar.em.sampling import rotation_grid_size

    n_images = 3
    n_classes = 2
    rotation_grid_size(0)
    fine_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0)
    fine_parent = np.asarray([0, 1, 2], dtype=np.int64)
    fine_translations = np.asarray([[0.0, 0.0], [0.25, 0.0]], dtype=np.float32)
    fine_translation_parent = np.zeros(2, dtype=np.int32)
    significant_by_class = [
        [np.asarray([0, 1], dtype=np.int32), np.asarray([1, 2], dtype=np.int32), np.asarray([0, 2], dtype=np.int32)],
        [np.asarray([0, 2], dtype=np.int32), np.asarray([0, 1], dtype=np.int32), np.asarray([1, 2], dtype=np.int32)],
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

def _fused_kclass_result_arrays(result):
    """Every array-valued field of a fused pass-2 result, for exact comparison."""

    arrays = {}
    for name in dir(result):
        if name.startswith("_"):
            continue
        value = getattr(result, name)
        if callable(value):
            continue
        if isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                if hasattr(item, "shape") or isinstance(item, (int, float)):
                    arrays[f"{name}[{i}]"] = np.asarray(item)
                elif item is not None and not isinstance(item, (str, bytes, dict)):
                    # one level into per-class records (noise statistics carry the
                    # translation-prior sigma2 offset as a scalar attribute)
                    for sub in dir(item):
                        if sub.startswith("_"):
                            continue
                        sub_value = getattr(item, sub)
                        if callable(sub_value):
                            continue
                        if hasattr(sub_value, "shape") or isinstance(sub_value, (int, float)):
                            arrays[f"{name}[{i}].{sub}"] = np.asarray(sub_value)
        elif hasattr(value, "shape") or isinstance(value, (int, float)):
            arrays[name] = np.asarray(value)
    return arrays

@pytest.mark.parametrize("noise", [False, True])
@pytest.mark.parametrize("device_index", [False, True])
def test_capacity_and_device_indices_preserve_all_results(monkeypatch, noise, device_index):
    monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_IMAGE_CAPACITY", "0")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_DEVICE_INDEX", "0")
    kwargs = _fused_kclass_capacity_fixture()
    kwargs["accumulate_noise"] = noise
    expected = _fused_kclass_result_arrays(bucketed_mod.compute_k_class_pass2_stats_sparse_fused(**kwargs))
    pads = []
    original = bucketed_mod._build_compact_pair_bucket_arrays_from_per_image_inputs
    def build(bucket, inputs, **kw):
        if kw.get("capacity_rows") is not None:
            pads.append((len(bucket["image_indices"]), kw["capacity_rows"]))
        return original(bucket, inputs, **kw)
    monkeypatch.setattr(bucketed_mod, "_build_compact_pair_bucket_arrays_from_per_image_inputs", build)
    # Tiny fixture byte budgets are intentionally overridden to exercise padding.
    # The capacity arithmetic and budget ceilings have separate donor tests.
    monkeypatch.setattr(bucketed_mod, "quantized_image_capacity", lambda n, **kw: max(16, 1 << (n - 1).bit_length()))
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_IMAGE_CAPACITY", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_DEVICE_INDEX", str(int(device_index)))
    actual = _fused_kclass_result_arrays(bucketed_mod.compute_k_class_pass2_stats_sparse_fused(**kwargs))
    assert pads and all(cap > real for real, cap in pads)
    assert expected.keys() == actual.keys()
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)

@pytest.mark.parametrize("noise", [False, True])
def test_device_chunk_scalars_preserve_multibucket_results(monkeypatch, noise):
    monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MAX_IMAGES_PER_MICROBATCH", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_IMAGE_CAPACITY", "1")
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_DEVICE_INDEX", "1")
    monkeypatch.setattr(bucketed_mod, "quantized_image_capacity", lambda n, **kw: 4)
    kwargs = _fused_kclass_capacity_fixture()
    kwargs["accumulate_noise"] = noise
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_DEVICE_CHUNK_SCALARS", "0")
    expected = _fused_kclass_result_arrays(bucketed_mod.compute_k_class_pass2_stats_sparse_fused(**kwargs))
    calls = []
    original = bucketed_mod._log_score_offset_from_min_diff2_device
    def offset(min_diff2):
        calls.append(min_diff2.shape)
        return original(min_diff2)
    monkeypatch.setattr(bucketed_mod, "_log_score_offset_from_min_diff2_device", offset)
    monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_DEVICE_CHUNK_SCALARS", "1")
    actual = _fused_kclass_result_arrays(bucketed_mod.compute_k_class_pass2_stats_sparse_fused(**kwargs))
    assert len(calls) > 1, "must exercise multiple chunks and bounded host flushes"
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)


@pytest.mark.parametrize("failure", [None, "padding", "host_mutation"])
def test_device_noise_totals_order_and_fail_closed(failure):
    from types import SimpleNamespace
    from recovar.em.classification.k_class_results import SparseKClassDeviceNoiseTotals
    host = SimpleNamespace(noise_wsum_total=[np.zeros(2)], noise_norm_correction_total=[np.zeros(2)])
    totals = SparseKClassDeviceNoiseTotals(host)
    indices = np.array([0, 1, 1], dtype=np.int64)
    expected = np.zeros(2)
    expected_shells = np.zeros(2)
    for values in ([1e16, 1.0, 0.0], [-1e16, 2.0, 0.0], [1.0, 3.0, 0.0]):
        values = np.array(values)
        shells = values[:2]
        totals.add(0, indices, 2, jnp.asarray(values), jnp.asarray(shells))
        np.add.at(expected, indices, values)
        expected_shells += shells
    if failure == "padding":
        totals.add(0, indices, 2, jnp.array([0.0, 0.0, 1.0]))
    elif failure == "host_mutation":
        host.noise_norm_correction_total[0][0] = 9.0
    if failure is not None:
        with pytest.raises(RuntimeError, match="padded image|host noise totals"):
            totals.finish()
    else:
        totals.finish()
        np.testing.assert_array_equal(host.noise_norm_correction_total[0], expected)
        np.testing.assert_array_equal(host.noise_wsum_total[0], expected_shells)

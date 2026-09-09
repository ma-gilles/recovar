"""Prefix ABI/dispatch checks on CPU; marked tests require the new real CUDA ABI."""

from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar import cuda_backproject as cb
from recovar.em.dense_single_volume import local_big_jit as local
from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as sparse

pytestmark = pytest.mark.unit


def _inputs(random=False):
    rng = np.random.default_rng(719)

    def values(shape, complex=False):
        if random:
            x = rng.normal(size=shape)
            if complex:
                x = x + 1j * rng.normal(size=shape)
        else:
            x = (np.arange(np.prod(shape)).reshape(shape) % 5 - 2) / 8
            if complex:
                x = x + 0.25j
        return jnp.asarray(x, dtype=jnp.complex64 if complex else jnp.float32)

    raw = values((2, 3, 6), complex=True)
    posterior = jnp.full((2, 3, 3), 0.125, dtype=jnp.float32)
    power = sparse._relion_wavg_rectangle_image_power(raw, posterior)
    return [
        raw,
        power,
        values((2, 3, 4), complex=True),
        values((2, 8)).astype(jnp.float64),
        jnp.ones(2, dtype=jnp.float32),
        posterior,
        jnp.asarray([0, 2, 4, 5], dtype=jnp.int32),
        jnp.asarray([0, 3, 1, 7], dtype=jnp.int32),
        jnp.asarray(3, dtype=jnp.int32),
        jnp.asarray(5, dtype=jnp.int32),
    ]


def _legacy(args):
    raw, _, proj, ctf, scale, posterior, positions, indices, ne, nr = args
    exact = cb.relion_wavg_sequential_runtime_triplet_f32(
        proj, ctf[:, indices].astype(jnp.float32), scale, raw[:, :, positions], posterior, ne
    )
    rectangle = sparse._relion_wavg_rectangle_triplet_terms(exact, raw, posterior, positions)
    atomic = cb.relion_wavg_rotation_atomic_runtime_triplet_add_f32(
        rectangle, jnp.zeros((raw.shape[0], raw.shape[2], 3), dtype=jnp.float32), nr
    )
    return atomic, rectangle


def test_factored_power_keeps_original_contraction_bitwise():
    raw, _, _, _, _, posterior, *_ = _inputs(random=True)

    @jax.jit
    def original(raw, posterior):
        power = (raw.real * raw.real).astype(jnp.float32)
        power = jax.lax.optimization_barrier(power)
        power = (power + raw.imag * raw.imag).astype(jnp.float32)
        return jnp.einsum("brt,btp->brp", posterior, power, preferred_element_type=jnp.float32).astype(jnp.float32)

    np.testing.assert_array_equal(
        original(raw, posterior), jax.jit(sparse._relion_wavg_rectangle_image_power)(raw, posterior)
    )


@pytest.mark.parametrize(
    "index,dtype", [(0, jnp.complex128), (1, jnp.float64), (3, jnp.float32), (8, jnp.int64), (9, jnp.float32)]
)
def test_operand_dtype_rejected_before_registration(monkeypatch, index, dtype):
    args = _inputs()
    args[index] = args[index].astype(dtype)
    monkeypatch.setattr(cb, "_ensure_wavg_native_prefix_ffi", lambda: pytest.fail("registered invalid ABI"))
    with pytest.raises(ValueError, match="dtype"):
        cb.relion_wavg_native_prefix_f32(*args)


@pytest.mark.parametrize("index", [1, 4, 5, 6, 7, 8, 9])
def test_shape_rejected_before_registration(monkeypatch, index):
    args = _inputs()
    args[index] = args[index].reshape(-1) if index in (8, 9) else args[index][:-1]
    monkeypatch.setattr(cb, "_ensure_wavg_native_prefix_ffi", lambda: pytest.fail("registered invalid ABI"))
    with pytest.raises(ValueError, match="geometry"):
        cb.relion_wavg_native_prefix_f32(*args)


@pytest.mark.parametrize("debug", [False, True])
def test_exact_ffi_operand_forwarding_without_materialization(monkeypatch, debug):
    args = _inputs()
    captured = {}
    monkeypatch.setattr(cb, "_ensure_wavg_native_prefix_ffi", lambda: None)

    def ffi(target, outputs, **options):
        captured.update(target=target, outputs=outputs, options=options)

        def call(*values):
            captured["values"] = values
            return "result"

        return call

    monkeypatch.setattr(jax.ffi, "ffi_call", ffi)
    assert cb.relion_wavg_native_prefix_f32(*args, debug=debug) == "result"
    assert all(a is b for a, b in zip(args, captured["values"], strict=True))
    assert captured["target"] == (
        cb._TARGET_RELION_WAVG_NATIVE_PREFIX_DEBUG_F32 if debug else cb._TARGET_RELION_WAVG_NATIVE_PREFIX_F32
    )
    if debug:
        assert captured["outputs"][1].shape == (2, 3, 6, 3)
    else:
        assert captured["outputs"].shape == (2, 6, 3)
    assert captured["options"] == {"vmap_method": "sequential"}


def test_old_library_remains_compatible_but_optional_target_fails(monkeypatch):
    monkeypatch.setattr(cb, "_wavg_native_prefix_ffi_registered", False)
    monkeypatch.setattr(cb, "_ensure_ffi", lambda: None)
    monkeypatch.setattr(cb, "_get_lib", lambda: SimpleNamespace())
    assert all("NativePrefix" not in symbol for _, symbol in cb._FFI_REGISTRATIONS)
    with pytest.raises(RuntimeError, match="explicitly rebuilt"):
        cb._ensure_wavg_native_prefix_ffi()
    assert not cb._wavg_native_prefix_ffi_registered


def test_shell_boundary_native_receives_one_original_translation(monkeypatch):
    args = _inputs()
    raw, power, proj, ctf, scale, posterior, positions, indices, ne, nr = args
    calls = []

    def translate(*unused):
        calls.append("translate")
        return raw

    monkeypatch.setattr(local, "_relion_cuda_translate_wavg_norm_images", translate)

    def native(*received, **kwargs):
        calls.append("native")
        assert received[0] is raw and received[2] is proj
        np.testing.assert_array_equal(received[1], power)
        return jnp.ones((2, 6, 3), dtype=jnp.float32)

    monkeypatch.setattr(cb, "relion_wavg_native_prefix_f32", native)
    shells, cutoff = local._relion_wavg_direct_triplet_shells(
        jnp.zeros((2, 8), dtype=jnp.complex64),
        jnp.zeros((3, 2), dtype=jnp.float32),
        jnp.arange(6, dtype=jnp.int32),
        positions,
        jnp.asarray([0, 1, 1, 2, 2, -1], dtype=jnp.int32),
        indices,
        proj,
        ctf,
        scale,
        posterior,
        jnp.asarray([True, False]),
        image_shape=(4, 4),
        shell_count=3,
        cutoff_shell=2,
        relion_wavg_sequential_cuda=True,
        logical_recon_pixel_count=ne,
        logical_rectangle_pixel_count=nr,
        return_per_image_cutoff=True,
        native_prefix=True,
    )
    assert calls == ["translate", "native"]
    np.testing.assert_array_equal(shells, [[1, 2, 2]] * 3)
    np.testing.assert_array_equal(cutoff, [[2, 2, 2], [0, 0, 0]])


@pytest.mark.gpu
@pytest.mark.parametrize("random", [False, True])
def test_real_cuda_preatomic_bitwise_and_dyadic_atomic(random):
    assert jax.default_backend() == "gpu" and jax.config.x64_enabled
    args = _inputs(random=random)
    actual, terms = cb.relion_wavg_native_prefix_f32(*args, debug=True)
    expected, expected_terms = _legacy(args)
    np.testing.assert_array_equal(np.asarray(terms).view(np.uint32), np.asarray(expected_terms).view(np.uint32))
    if not random:
        np.testing.assert_array_equal(np.asarray(actual).view(np.uint32), np.asarray(expected).view(np.uint32))
    np.testing.assert_array_equal(np.asarray(terms)[:, :, 5], 0.0)
    # A second executable explicitly exercises the production scratch path.
    production = cb.relion_wavg_native_prefix_f32(*args)
    if not random:
        np.testing.assert_array_equal(production, expected)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "case",
    [
        "duplicate",
        "negative",
        "beyond",
        "tail_into_logical",
        "active_into_tail",
        "bad_ctf",
        "negative_count",
        "large_count",
        "large_rectangle",
    ],
)
def test_real_cuda_invalid_map_or_count_poisoned(case):
    args = _inputs()
    if case == "duplicate":
        args[6] = args[6].at[1].set(0)
    elif case == "negative":
        args[6] = args[6].at[0].set(-1)
    elif case == "beyond":
        args[6] = args[6].at[0].set(6)
    elif case == "tail_into_logical":
        args[6] = args[6].at[3].set(3)
    elif case == "active_into_tail":
        args[6] = args[6].at[2].set(5)
    elif case == "bad_ctf":
        args[7] = args[7].at[0].set(8)
    elif case == "negative_count":
        args[8] = jnp.asarray(-1, dtype=jnp.int32)
    elif case == "large_count":
        args[8] = jnp.asarray(5, dtype=jnp.int32)
    elif case == "large_rectangle":
        args[9] = jnp.asarray(7, dtype=jnp.int32)
    assert np.isnan(np.asarray(cb.relion_wavg_native_prefix_f32(*args))).all()


@pytest.mark.gpu
def test_real_cuda_runtime_counts_reuse_one_executable():
    args = _inputs()
    compiled = jax.jit(cb.relion_wavg_native_prefix_f32)
    first = compiled(*args)
    changed = list(args)
    changed[8] = jnp.asarray(4, dtype=jnp.int32)
    changed[9] = jnp.asarray(6, dtype=jnp.int32)
    second = compiled(*changed)
    zero = list(args)
    zero[8] = jnp.asarray(0, dtype=jnp.int32)
    zero[9] = jnp.asarray(0, dtype=jnp.int32)
    empty = compiled(*zero)
    np.testing.assert_array_equal(np.asarray(empty), 0.0)
    assert compiled._cache_size() == 1
    assert np.isfinite(np.asarray(first)).all() and np.isfinite(np.asarray(second)).all()


@pytest.mark.gpu
def test_nine_shared_noise_outputs_with_single_rotation(monkeypatch):
    from test_shared_local_exact_noise import _make_noise_inputs

    inputs = _make_noise_inputs(42)
    for name in (
        "scalar_reconstruction_probs",
        "pixel_reconstruction_probs",
        "pixel_proj_for_noise",
        "pixel_ctf_probs",
    ):
        inputs[name] = inputs[name][:, :1]
    # Dyadic operands and one image per scale group make this a deterministic
    # nine-output integration oracle, independent of cross-rotation/group atomics.
    inputs["scalar_reconstruction_probs"] = jnp.full((42, 1, 3), 0.125, dtype=jnp.float32)
    inputs["pixel_reconstruction_probs"] = inputs["scalar_reconstruction_probs"]
    inputs["pixel_proj_for_noise"] = jnp.full((42, 1, 4), 0.25 + 0.5j, dtype=jnp.complex64)
    inputs["pixel_ctf_probs"] = jnp.full((42, 1, 4), 0.5, dtype=jnp.float32)
    inputs["shifted_noise_split"] = jnp.full((42, 3, 4), 0.5 + 0.25j, dtype=jnp.complex64)
    inputs["translation_sqdist_ang"] = jnp.full((42, 3), 0.25, dtype=jnp.float32)
    inputs["noise_variance_for_noise"] = jnp.ones(4, dtype=jnp.float32)
    inputs["image_only_corr"] = jnp.ones(42, dtype=jnp.float32)
    inputs["batch_scale"] = jnp.ones(42, dtype=jnp.float32)
    inputs["group_ids"] = jnp.arange(42, dtype=jnp.int32)
    inputs["noise_scale_xa"] = jnp.full(42, 0.25, dtype=jnp.float32)
    inputs["noise_scale_aa"] = jnp.full(42, 0.5, dtype=jnp.float32)
    inputs["processed_score_half"] = jnp.ones((42, 12), dtype=jnp.complex64) * (0.25 + 0.5j)
    inputs["shell_indices_half"] = jnp.asarray([0, 1, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2], dtype=jnp.int32)
    inputs["ctf_rfloat_half"] = jnp.ones((42, 12), dtype=jnp.float64)
    inputs["image_shape"] = (4, 4)
    options = dict(
        norm_current_size=None,
        stable_fourier_window_shapes=False,
        include_unweighted_norm_high_shell=True,
        use_relion_cuda_powerclass_spectrum=False,
        source_faithful_spectrum_norm=False,
        accumulate_scale_correction=True,
        return_noise_split=True,
        use_relion_wavg_cutoff=True,
        relion_wavg_sequential_cuda=True,
    )
    original = local._relion_wavg_direct_triplet_shells
    local.run_deferred_local_exact_noise_jit.clear_cache()
    try:
        expected = local.run_deferred_local_exact_noise_jit(**inputs, **options)
        jax.block_until_ready(expected)
        local.run_deferred_local_exact_noise_jit.clear_cache()
        monkeypatch.setattr(local, "_relion_wavg_direct_triplet_shells", partial(original, native_prefix=True))
        actual = local.run_deferred_local_exact_noise_jit(**inputs, **options)
        assert len(actual) == len(expected) == 9
        for a, b in zip(actual, expected, strict=True):
            assert a.dtype == b.dtype and a.shape == b.shape
            np.testing.assert_array_equal(a, b)
    finally:
        local.run_deferred_local_exact_noise_jit.clear_cache()

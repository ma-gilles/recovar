"""The physical noise core preserves the existing scalar/image arithmetic."""

import ast
import importlib.util
import inspect
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume import local_big_jit as noise

pytestmark = pytest.mark.unit

# Reuse the already reviewed tiny fixture and pre-extraction oracle without
# copying a second implementation of the complete noise science into this test.
_spec = importlib.util.spec_from_file_location(
    "_shared_noise_reference", Path(__file__).with_name("test_shared_local_exact_noise.py")
)
_reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reference)

CORE_DYNAMIC = (
    "scalar_reconstruction_probs",
    "processed_score_half",
    "image_only_corr",
    "translation_sqdist_ang",
    "valid_image_mask",
    "shell_indices_half",
    "runtime_logical_current_size",
)
CORE_STATIC = (
    "image_shape",
    "shell_count",
    "norm_current_size",
    "stable_fourier_window_shapes",
    "include_unweighted_norm_high_shell",
    "use_relion_cuda_powerclass_spectrum",
    "source_faithful_spectrum_norm",
    "unweighted_high_shell_image_power",
)


def _core_options(inputs, source_faithful):
    return dict(
        image_shape=inputs["image_shape"],
        shell_count=inputs["shell_count"],
        norm_current_size=None,
        stable_fourier_window_shapes=False,
        include_unweighted_norm_high_shell=True,
        use_relion_cuda_powerclass_spectrum=False,
        source_faithful_spectrum_norm=source_faithful,
    )


def _core(inputs, source_faithful):
    return noise.run_deferred_local_exact_noise_core_jit(
        *(inputs[key] for key in CORE_DYNAMIC), **_core_options(inputs, source_faithful)
    )


def _literal_core(inputs, source_faithful):
    """Only the four extracted outputs, literally using the original operations."""
    support_mass, _, offset = noise.compute_local_noise_scalar_terms(
        inputs["scalar_reconstruction_probs"],
        inputs["translation_sqdist_ang"],
        inputs["valid_image_mask"],
    )
    processed = inputs["processed_score_half"] * inputs["image_only_corr"][:, None]
    shells, per_image = noise._noise_image_power_shells_and_per_image(
        processed,
        support_mass,
        inputs["shell_indices_half"],
        inputs["valid_image_mask"],
        inputs["runtime_logical_current_size"] // 2,
        shell_count=inputs["shell_count"],
        image_shape=inputs["image_shape"],
        current_size=None,
        runtime_current_size=None,
        include_unweighted_high_shell=True,
        use_relion_cuda_powerclass_spectrum=False,
        source_faithful_spectrum_norm=source_faithful,
    )
    return support_mass, offset, shells, per_image


def _deferred(inputs, *, source_faithful, noise_split, scale, prepared_core=None):
    function = noise.run_deferred_local_exact_noise_jit
    arguments = {name: inputs[name] for name in inspect.signature(function).parameters if name in inputs}
    arguments.update(_core_options(inputs, source_faithful))
    arguments.update(
        accumulate_scale_correction=scale,
        return_noise_split=noise_split,
        use_relion_wavg_cutoff=False,
        relion_wavg_sequential_cuda=True,
        prepared_core=prepared_core,
    )
    return function(**arguments)


def _assert_bitwise(actual, expected):
    assert len(actual) == len(expected)
    for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
        left, right = np.asarray(left), np.asarray(right)
        assert left.shape == right.shape, index
        assert left.dtype == right.dtype, index
        assert left.tobytes() == right.tobytes(), f"output {index} differs bitwise"


@pytest.mark.parametrize("pixel_batch", (42, 32))
@pytest.mark.parametrize("source_faithful", (False, True), ids=("spectrum_f32", "spectrum_f64"))
def test_physical_core_matches_literal_pre_extraction_operations(pixel_batch, source_faithful):
    inputs = _reference._make_noise_inputs(pixel_batch)
    actual = _core(inputs, source_faithful)
    _assert_bitwise(actual, _literal_core(inputs, source_faithful))
    assert tuple(value.shape for value in actual) == ((42,), (), (4,), (42,))
    assert actual.batch_img_power_per_image.dtype == (jnp.float64 if source_faithful else jnp.float32)
    np.testing.assert_array_equal(np.asarray(actual.support_mass)[pixel_batch:], 0)
    np.testing.assert_array_equal(np.asarray(actual.batch_img_power_per_image)[pixel_batch:], 0)


@pytest.mark.parametrize("pixel_batch", (42, 32))
@pytest.mark.parametrize("source_faithful", (False, True), ids=("spectrum_f32", "spectrum_f64"))
@pytest.mark.parametrize("noise_split", (False, True))
@pytest.mark.parametrize("scale", (False, True))
def test_split_core_and_deferred_consumer_preserve_all_nine_outputs(pixel_batch, source_faithful, noise_split, scale):
    inputs = _reference._make_noise_inputs(pixel_batch)
    inline = _deferred(inputs, source_faithful=source_faithful, noise_split=noise_split, scale=scale)
    split = _deferred(
        inputs,
        source_faithful=source_faithful,
        noise_split=noise_split,
        scale=scale,
        prepared_core=_core(inputs, source_faithful),
    )
    assert len(split) == len(inline) == 9
    _assert_bitwise(split, inline)
    if source_faithful:
        # Existing independently spelled-out complete oracle uses F64 spectrum
        # norms. F32 core math is checked independently in the test above.
        legacy = _reference._legacy_inline_noise(
            inputs, return_noise_split=noise_split, accumulate_scale_correction=scale
        )
        global_norm = inputs["noise_norm_correction"].at[inputs["bucket_image_indices"]].add(legacy[6])
        _assert_bitwise(inline, (*legacy[:6], global_norm, *legacy[7:9]))


def test_core_abi_and_cache_exclude_ragged_pixel_operands():
    function = noise.run_deferred_local_exact_noise_core_jit
    signature = inspect.signature(function)
    assert tuple(signature.parameters) == (*CORE_DYNAMIC, *CORE_STATIC)
    tree = ast.parse(inspect.getsource(function))
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    static = [
        ast.literal_eval(keyword.value)
        for decorator in definition.decorator_list
        if isinstance(decorator, ast.Call)
        for keyword in decorator.keywords
        if keyword.arg == "static_argnames"
    ]
    assert static == [CORE_STATIC]
    assert not hasattr(noise.compute_local_exact_noise_core, "lower")
    function.clear_cache()
    try:
        for pixel_batch in (42, 32):
            inputs = _reference._make_noise_inputs(pixel_batch)
            # Pixel B/rotation buffers differ, but all seven core operand shapes
            # retain B42. A logical radius value also cannot add a static key.
            for current_size in (2, 4):
                changed = dict(inputs, runtime_logical_current_size=jnp.asarray(current_size, dtype=jnp.int32))
                _core(changed, True)
                assert function._cache_size() == 1
    finally:
        function.clear_cache()


@pytest.mark.parametrize(
    "field,bad_shape",
    (
        ("support_mass", (41,)),
        ("noise_sumw_offset", (1,)),
        ("batch_img_power_shells", (5,)),
        ("batch_img_power_per_image", (41,)),
    ),
)
def test_prepared_core_rejects_nonphysical_shapes(field, bad_shape):
    inputs = _reference._make_noise_inputs(32)
    core = _core(inputs, True)
    malformed = core._replace(**{field: jnp.zeros(bad_shape, dtype=getattr(core, field).dtype)})
    with pytest.raises(ValueError, match="Prepared noise core must retain physical batch and shell shapes"):
        _deferred(inputs, source_faithful=True, noise_split=True, scale=True, prepared_core=malformed)

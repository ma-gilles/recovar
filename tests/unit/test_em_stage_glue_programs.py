"""Program-count reductions in the coarse pass that must not move any number.

Three changes are covered, all of them program-count changes only:

* ``RECOVAR_COARSE_PAD_FINAL_IMAGE_BATCH`` pads a half set's last coarse image
  batch up to ``image_batch_size`` by repeating image row zero, so the coarse
  pass, significance and image preprocessing see a single image extent per run
  instead of one per remainder. The repeated rows are dropped from every
  science output.
* ``RECOVAR_EM_JIT_STAGE_GLUE`` runs the post-transform preprocessing chain and
  the windowed score-operand gather as one jitted program each, instead of one
  XLA program per primitive per extent.
* ``_collate_batch_to_jax`` builds the host array before the device transfer,
  which removes the ``convert_element_type`` program JAX compiles for every
  distinct Python-list length.

Each test compares the changed path against the unchanged one bit for bit.
"""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp
from helpers.em_arrays import _hermitian_volume, _make_rotations, _raw_real_image_2d

import recovar.core.fourier_transform_utils as ftu

pytestmark = pytest.mark.unit

IMAGE_SHAPE = (16, 16)
IMAGE_SIZE = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
VOLUME_SHAPE = (16, 16, 16)
VOLUME_SIZE = VOLUME_SHAPE[0] * VOLUME_SHAPE[1] * VOLUME_SHAPE[2]


def _identity_ctf(params, image_shape=None, voxel_size=None, *, half_image=False):
    if half_image:
        height, width = image_shape if image_shape is not None else IMAGE_SHAPE
        size = height * (width // 2 + 1)
    else:
        size = IMAGE_SIZE
    return jnp.ones((params.shape[0], size), dtype=jnp.float32)


def _raw_real_process(batch, apply_image_mask=False):
    _ = apply_image_mask
    images = jnp.asarray(batch)
    return ftu.get_dft2(images).reshape((images.shape[0], -1)).astype(jnp.complex64)


def _raw_real_process_half(batch, apply_image_mask=False):
    _ = apply_image_mask
    images = jnp.asarray(batch)
    return ftu.get_dft2_real(images).reshape((images.shape[0], -1)).astype(jnp.complex64)


class _StageGlueDataset:
    """Minimal dataset with the attributes the coarse significance path reads."""

    def __init__(self, n_images, seed=907):
        self.image_shape = IMAGE_SHAPE
        self.image_size = IMAGE_SIZE
        self.grid_size = IMAGE_SHAPE[0]
        self.volume_shape = VOLUME_SHAPE
        self.volume_size = VOLUME_SIZE
        self.n_images = n_images
        self.n_units = n_images
        self.voxel_size = 1.0
        self.dtype = jnp.complex64
        self.CTF_params = np.zeros((n_images, 9), dtype=np.float32)
        self.ctf_evaluator = staticmethod(_identity_ctf)
        self.process_images = staticmethod(_raw_real_process)
        self.process_images_half = staticmethod(_raw_real_process_half)
        self.rotation_matrices = np.tile(np.eye(3, dtype=np.float32), (n_images, 1, 1))
        self.translations = np.zeros((n_images, 2), dtype=np.float32)
        self.premultiplied_ctf = False

        rng = np.random.default_rng(seed)
        self._images = np.zeros((n_images, *IMAGE_SHAPE), dtype=np.float32)
        for index in range(n_images):
            self._images[index] = _raw_real_image_2d(IMAGE_SHAPE, seed=int(rng.integers(10000)))

        class _ImageSource:
            process_images = staticmethod(_raw_real_process)
            process_images_half = staticmethod(_raw_real_process_half)

        self.image_source = _ImageSource()

    def iter_batches(self, batch_size, *, indices=None, by_image=False, **kwargs):
        _ = kwargs
        if indices is None:
            indices = np.arange(self.n_images)
        indices = np.asarray(indices)
        step = max(1, int(batch_size))
        for start in range(0, len(indices), step):
            idx = np.asarray(indices[start : start + step])
            yield (
                jnp.asarray(self._images[idx]),
                jnp.asarray(self.rotation_matrices[idx]),
                jnp.asarray(self.translations[idx]),
                jnp.asarray(self.CTF_params[idx]),
                None,
                idx,
                idx,
            )

    def get_valid_frequency_indices(self, pixel_res):
        return np.ones(self.volume_size, dtype=bool)


def _significance_call(n_classes=2):
    """Arguments for a coarse significance call whose last batch is a remainder."""

    volume = _hermitian_volume(VOLUME_SHAPE, seed=915)
    args = (
        _StageGlueDataset(n_images=7, seed=913),
        jnp.stack([volume * (1.0 + 0.01 * k) for k in range(n_classes)]),
        jnp.ones(IMAGE_SIZE, dtype=jnp.float32),
        _make_rotations(5, seed=921),
        jnp.array([[0.0, 0.0], [1.0, -1.0], [-1.0, 0.0]], dtype=jnp.float32),
        "linear_interp",
    )
    kwargs = dict(
        class_log_priors=np.log(np.arange(1, n_classes + 1) / sum(range(1, n_classes + 1))),
        rotation_log_prior=np.linspace(0.0, -0.4, 5, dtype=np.float32),
        translation_log_prior=np.linspace(
            0.0, -0.3, 7 * 3, dtype=np.float32
        ).reshape(7, 3),
        adaptive_fraction=0.9,
        max_significants=6,
        # 7 images in batches of 3 leaves a 1-image tail batch.
        image_batch_size=3,
        rotation_block_size=2,
        current_size=8,
        half_spectrum_scoring=True,
        return_class_best=True,
    )
    return args, kwargs


def _assert_significance_results_identical(candidate, control):
    for actual, expected in zip(candidate[:4], control[:4]):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    for actual_class, expected_class in zip(candidate[4], control[4]):
        for actual, expected in zip(actual_class, expected_class):
            if actual is None or expected is None:
                assert actual is expected
            else:
                np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert set(candidate[5]) == set(control[5])
    for key, expected in control[5].items():
        actual = candidate[5][key]
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
            assert actual.dtype == expected.dtype


# P3-B measured this path's own floor on a GPU: repeating the identical
# configuration moves 0 to 3 of 10080 entries, each by one float32 ulp, always
# in a reporting field and never in a mask, a count, a sample-index array or an
# assignment (its report, section 5a). The padding cannot be held to bitwise on
# a GPU, where the batch extent changes the reduction shape, so the contract
# below is that measured band.
#
# Two of its three parts transfer to this fixture and one does not. The
# magnitude (one float32 ulp) and the scope (discrete outputs and the non-
# reporting results bitwise) are asserted. The *count* is fixture-dependent and
# is not asserted here: measured on a claimed A100 at this commit, this fixture
# compares 77 float entries, its null control (the identical configuration
# twice in one process) moves 0, and the padded path moves 6, all of them in
# four ``full_stats`` reporting fields and none by more than one float32 ulp.
# Six of 77 is not three of 10080; asserting either number on the other fixture
# would be a tolerance, not a measurement.
_NULL_BAND_MAX_FLOAT32_ULP = 1.0


def _float32_ulp_distance(actual, expected):
    """Distance in **float32** units in the last place.

    The band is a float32 ulp whatever the container is: the coarse posterior
    is float32 data, and the reporting fields that move are float64
    accumulators over it, so float64 spacing would read a single float32 step
    as hundreds of millions of ulp.
    """

    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
    lo = np.minimum(np.abs(actual), np.abs(expected))
    spacing = np.spacing(lo.astype(np.float32)).astype(np.float64)
    spacing = np.where(spacing == 0, np.float64(np.finfo(np.float32).tiny), spacing)
    steps = np.zeros(actual.shape, dtype=np.float64)
    finite = np.isfinite(actual) & np.isfinite(expected)
    steps[finite] = np.abs(actual[finite] - expected[finite]) / spacing[finite]
    return steps


def _assert_significance_results_within_null_band(candidate, control):
    """Everything bitwise except reporting fields, which may move one ulp."""

    moved_fields = []

    def compare(actual, expected, label, *, reporting):
        actual = np.asarray(actual)
        expected = np.asarray(expected)
        assert actual.shape == expected.shape, label
        assert actual.dtype == expected.dtype, label
        if actual.dtype.kind not in "fc" or not reporting:
            # masks, counts, sample indices, assignments and every non-
            # reporting result: bitwise, always.
            np.testing.assert_array_equal(actual, expected, err_msg=label)
            return
        if actual.dtype.kind == "c":
            steps = np.maximum(
                _float32_ulp_distance(actual.real, expected.real),
                _float32_ulp_distance(actual.imag, expected.imag),
            )
        else:
            steps = _float32_ulp_distance(actual, expected)
        worst = float(steps.max(initial=0.0))
        assert worst <= _NULL_BAND_MAX_FLOAT32_ULP, (
            f"{label}: {int((steps > 0).sum())} entries moved, max {worst} "
            f"float32 ulp, band is {_NULL_BAND_MAX_FLOAT32_ULP}"
        )
        if worst > 0.0:
            moved_fields.append((label, int((steps > 0).sum()), worst))

    for i, (actual, expected) in enumerate(zip(candidate[:4], control[:4])):
        compare(actual, expected, f"result[{i}]", reporting=False)
    for c, (actual_class, expected_class) in enumerate(zip(candidate[4], control[4])):
        for i, (actual, expected) in enumerate(zip(actual_class, expected_class)):
            if actual is None or expected is None:
                assert actual is expected, f"class[{c}][{i}]"
                continue
            compare(actual, expected, f"class[{c}][{i}]", reporting=False)
    assert set(candidate[5]) == set(control[5])
    for key, expected in control[5].items():
        actual = candidate[5][key]
        if isinstance(expected, np.ndarray):
            compare(actual, expected, f"full_stats[{key!r}]", reporting=True)
    return moved_fields


def _run_padding_pair(monkeypatch, *, jit_glue=False):
    """The unpadded control and the padded candidate, in this process."""

    from recovar.em.scoring import significance

    args, kwargs = _significance_call()
    monkeypatch.delenv("RECOVAR_COARSE_PAD_FINAL_IMAGE_BATCH", raising=False)
    monkeypatch.delenv("RECOVAR_EM_JIT_STAGE_GLUE", raising=False)
    control = significance._compute_k_class_significance_batched(*args, **kwargs)
    monkeypatch.setenv("RECOVAR_COARSE_PAD_FINAL_IMAGE_BATCH", "1")
    if jit_glue:
        monkeypatch.setenv("RECOVAR_EM_JIT_STAGE_GLUE", "1")
    candidate = significance._compute_k_class_significance_batched(*args, **kwargs)
    return candidate, control


def test_coarse_pad_env_flag_preserves_every_significance_output(monkeypatch):
    """The padded tail batch must reproduce the unpadded outputs exactly.

    CPU-only: on a GPU the padding changes the coarse reduction's shape, and
    the outputs move inside the null band the GPU sibling below asserts.
    """

    if jax.default_backend() == "gpu":
        pytest.skip("CPU-only contract; the GPU band is the sibling test")
    candidate, control = _run_padding_pair(monkeypatch)
    _assert_significance_results_identical(candidate, control)


@pytest.mark.skipif(
    jax.default_backend() != "gpu", reason="the null band is a GPU measurement"
)
def test_coarse_pad_env_flag_stays_inside_the_null_band_on_gpu(monkeypatch):
    """On a GPU the padded path must stay inside P3-B's measured floor."""

    candidate, control = _run_padding_pair(monkeypatch)
    _assert_significance_results_within_null_band(candidate, control)


def test_coarse_pad_env_flag_gives_every_batch_one_image_extent(monkeypatch):
    """With the flag on, preprocessing sees ``image_batch_size`` rows every time."""

    from recovar.em.helpers import preprocessing
    from recovar.em.scoring import significance

    args, kwargs = _significance_call(n_classes=1)
    original = preprocessing.preprocess_batch
    seen = []

    def record(experiment_dataset, batch, *rest, **batch_kwargs):
        seen.append(int(np.asarray(batch).shape[0]))
        return original(experiment_dataset, batch, *rest, **batch_kwargs)

    monkeypatch.setattr(preprocessing, "preprocess_batch", record)

    monkeypatch.delenv("RECOVAR_COARSE_PAD_FINAL_IMAGE_BATCH", raising=False)
    significance._compute_k_class_significance_batched(*args, **kwargs)
    unpadded = list(seen)

    seen.clear()
    monkeypatch.setenv("RECOVAR_COARSE_PAD_FINAL_IMAGE_BATCH", "1")
    significance._compute_k_class_significance_batched(*args, **kwargs)
    padded = list(seen)

    assert unpadded == [3, 3, 1]
    assert padded == [3, 3, 3]
    assert len(set(padded)) == 1


def test_jit_stage_glue_preserves_every_significance_output(monkeypatch):
    """The jitted preprocessing/window glue must be bit-for-bit the eager path."""

    from recovar.em.scoring import significance

    args, kwargs = _significance_call()
    monkeypatch.delenv("RECOVAR_COARSE_PAD_FINAL_IMAGE_BATCH", raising=False)
    monkeypatch.delenv("RECOVAR_EM_JIT_STAGE_GLUE", raising=False)
    control = significance._compute_k_class_significance_batched(*args, **kwargs)
    monkeypatch.setenv("RECOVAR_EM_JIT_STAGE_GLUE", "1")
    candidate = significance._compute_k_class_significance_batched(*args, **kwargs)
    _assert_significance_results_identical(candidate, control)


def test_jit_stage_glue_and_padding_together_preserve_outputs(monkeypatch):
    """Both opt-ins at once, which is how the candidate arm runs.

    CPU-only for the same reason as the padding test above.
    """

    if jax.default_backend() == "gpu":
        pytest.skip("CPU-only contract; the GPU band is the sibling test")
    candidate, control = _run_padding_pair(monkeypatch, jit_glue=True)
    _assert_significance_results_identical(candidate, control)


@pytest.mark.skipif(
    jax.default_backend() != "gpu", reason="the null band is a GPU measurement"
)
def test_both_opt_ins_stay_inside_the_null_band_on_gpu(monkeypatch):
    """Both opt-ins on a GPU, against the same measured floor."""

    candidate, control = _run_padding_pair(monkeypatch, jit_glue=True)
    _assert_significance_results_within_null_band(candidate, control)


def test_preprocess_batch_jitted_elementwise_matches_eager():
    """The jitted elementwise chain agrees bitwise with the eager one."""

    from recovar.em.helpers import preprocessing

    rng = np.random.default_rng(3)
    n_images, n_half_pixels, n_trans = 4, 9, 3
    processed = jnp.asarray(
        rng.standard_normal((n_images, n_half_pixels))
        + 1j * rng.standard_normal((n_images, n_half_pixels)),
        dtype=jnp.complex64,
    )
    ctf = jnp.asarray(rng.uniform(0.3, 1.7, (n_images, n_half_pixels)), dtype=jnp.float32)
    noise = jnp.asarray(rng.uniform(0.5, 2.0, n_half_pixels), dtype=jnp.float32)
    phases = jnp.asarray(
        rng.standard_normal((n_trans, n_half_pixels))
        + 1j * rng.standard_normal((n_trans, n_half_pixels)),
        dtype=jnp.complex64,
    )
    weights = jnp.asarray(rng.uniform(1.0, 2.0, n_half_pixels), dtype=jnp.float32)

    common = dict(
        score_complex_dtype=jnp.complex64,
        score_real_dtype=jnp.float32,
        norm_real_dtype=None,
    )
    eager = preprocessing._preprocess_batch_elementwise(
        processed, ctf, noise, phases, weights, **common
    )
    jitted = preprocessing._preprocess_batch_elementwise_jit(
        processed, ctf, noise, phases, weights, **common
    )
    assert len(eager) == len(jitted) == 4
    for actual, expected in zip(jitted, eager):
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_collate_list_batch_makes_no_program_and_same_array():
    """``_collate_batch_to_jax`` must stop compiling one program per list length."""

    import jax._src.dispatch as dispatch

    from recovar.data_io.image_backends import _collate_batch_to_jax

    batches = [
        [int(value) for value in range(n)] for n in (5, 6, 7)
    ]
    expected = [jnp.asarray(batch) for batch in batches]

    original = dispatch.xla_primitive_callable
    calls = []

    def counting(*args, **kwargs):
        calls.append(args[0] if args else None)
        return original(*args, **kwargs)

    dispatch.xla_primitive_callable = counting
    try:
        actual = [_collate_batch_to_jax(batch) for batch in batches]
    finally:
        dispatch.xla_primitive_callable = original

    assert calls == [], f"collation compiled {len(calls)} eager programs: {calls}"
    for got, want in zip(actual, expected):
        assert got.dtype == want.dtype and got.shape == want.shape
        assert got.weak_type == want.weak_type
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_collate_keeps_device_arrays_on_device():
    """A list of device arrays must not be pulled back through the host."""

    from recovar.data_io.image_backends import _collate_batch_to_jax

    batch = [jnp.asarray(1.0, dtype=jnp.float32), jnp.asarray(2.0, dtype=jnp.float32)]
    collated = _collate_batch_to_jax(batch)
    assert isinstance(collated, jax.Array)
    np.testing.assert_array_equal(np.asarray(collated), np.asarray([1.0, 2.0], dtype=np.float32))


@pytest.mark.parametrize(
    "name, reader",
    [
        (
            "RECOVAR_COARSE_PAD_FINAL_IMAGE_BATCH",
            "recovar.em.scoring.significance:_coarse_pad_final_image_batch_enabled",
        ),
        (
            "RECOVAR_EM_JIT_STAGE_GLUE",
            "recovar.em.helpers.preprocessing:jit_stage_glue_enabled",
        ),
    ],
)
def test_stage_glue_flags_fail_closed_on_bad_tokens(monkeypatch, name, reader):
    import importlib

    module_name, attribute = reader.split(":")
    read = getattr(importlib.import_module(module_name), attribute)

    monkeypatch.delenv(name, raising=False)
    assert read() is False
    monkeypatch.setenv(name, "1")
    assert read() is True
    monkeypatch.setenv(name, "0")
    assert read() is False
    monkeypatch.setenv(name, "maybe")
    with pytest.raises(ValueError, match=name):
        read()


def test_source_star_ctf_pads_with_the_rest_of_the_coarse_batch():
    """The firstiter-CC operand rebuilt from the source STAR must pad too.

    ``RECOVAR_COARSE_PAD_FINAL_IMAGE_BATCH`` pads the coarse batch's images,
    CTF parameters, pre-shifts, corrections and scales, but the normalized-CC
    tree-rescore branch of ``_compute_k_class_significance_batched`` rebuilds one
    more per-image operand from the source STAR at the *unpadded* ``indices``.
    With the flag on, the first iteration of a K=1 end-to-end died there:

        TypeError: div got incompatible shapes for broadcasting:
                   (250, 1), (216, 33024)

    (250 = the padded batch scale, 216 = the half set's last batch). The branch
    itself needs RELION CUDA preprocessing and a real source STAR, so the
    end-to-end is its regression check; this test pins the padding contract the
    fix relies on: the repeat-padded operand keeps every live row, repeats row
    zero, and broadcasts against the padded per-image scale.
    """

    from recovar.em.relion.relion_coarse_operands import _repeat_pad_batch_axis
    from recovar.em.sparse_pass2.sparse_pass2_scoring import (
        _relion_cuda_pixel_correction_from_rfloat_ctf,
    )

    actual, padded_size, pixels = 216, 250, 12
    rng = np.random.default_rng(20260920)
    ctf = jnp.asarray(rng.uniform(0.5, 1.5, (actual, pixels)), dtype=jnp.float32)
    padded = jnp.asarray(_repeat_pad_batch_axis(ctf, padded_size))

    assert padded.shape == (padded_size, pixels)
    np.testing.assert_array_equal(np.asarray(padded[:actual]), np.asarray(ctf))
    np.testing.assert_array_equal(
        np.asarray(padded[actual:]),
        np.repeat(np.asarray(ctf[:1]), padded_size - actual, axis=0),
    )

    scale = jnp.asarray(rng.uniform(0.9, 1.1, (padded_size, 1)), dtype=jnp.float32)
    correction = _relion_cuda_pixel_correction_from_rfloat_ctf(scale, padded)
    assert correction.shape == (padded_size, pixels)
    # The live rows are the unpadded answer: padding may not move a science row.
    unpadded_correction = _relion_cuda_pixel_correction_from_rfloat_ctf(
        scale[:actual], ctf
    )
    np.testing.assert_array_equal(
        np.asarray(correction[:actual]), np.asarray(unpadded_correction)
    )

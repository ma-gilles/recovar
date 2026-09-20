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


def test_coarse_pad_env_flag_preserves_every_significance_output(monkeypatch):
    """The padded tail batch must reproduce the unpadded outputs exactly."""

    from recovar.em.scoring import significance

    args, kwargs = _significance_call()
    monkeypatch.delenv("RECOVAR_COARSE_PAD_FINAL_IMAGE_BATCH", raising=False)
    control = significance._compute_k_class_significance_batched(*args, **kwargs)
    monkeypatch.setenv("RECOVAR_COARSE_PAD_FINAL_IMAGE_BATCH", "1")
    candidate = significance._compute_k_class_significance_batched(*args, **kwargs)
    _assert_significance_results_identical(candidate, control)


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
    """Both opt-ins at once, which is how the candidate arm runs."""

    from recovar.em.scoring import significance

    args, kwargs = _significance_call()
    monkeypatch.delenv("RECOVAR_COARSE_PAD_FINAL_IMAGE_BATCH", raising=False)
    monkeypatch.delenv("RECOVAR_EM_JIT_STAGE_GLUE", raising=False)
    control = significance._compute_k_class_significance_batched(*args, **kwargs)
    monkeypatch.setenv("RECOVAR_COARSE_PAD_FINAL_IMAGE_BATCH", "1")
    monkeypatch.setenv("RECOVAR_EM_JIT_STAGE_GLUE", "1")
    candidate = significance._compute_k_class_significance_batched(*args, **kwargs)
    _assert_significance_results_identical(candidate, control)


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

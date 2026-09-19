"""Once-per-half per-image operands of the device-resident pass 2 (T16).

Ticket: em_parity_tickets_20260918/T16_resident_operands_once_per_half.md.

The change moves work in time, not in arithmetic: operands the driver used to
rebuild for every capacity chunk are built once for a whole half and gathered
per chunk, and the translation that used to be applied while building them is
applied inside T15's M-step kernel instead. Both claims are checked against the
path they replace:

* the resident per-image arrays, translated with the primitives
  ``_prepare_bucket_io`` uses, must equal its pre-shifted tiles **bitwise**;
* the kernel's ``summed``/``summed_masked``/``ctf_probs`` must equal
  ``_resident_block_weighted_sums`` **bitwise** at the production translation
  count, where XLA's reduction over translations is sequential too.

The exact-BPref reconstruction operand needs a RELION source STAR and RELION
CUDA preprocessing, which this fixture's dataset does not have; that
configuration is covered on a real chunk by the driver's
``RECOVAR_SPARSE_PASS2_RESIDENT_OPERANDS_VERIFY=1`` arm, which runs the same
comparison against the per-chunk preparation inside a production pass.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp
from test_sparse_pass2_bucketed_parity import IMAGE_SHAPE, MockDataset

from recovar.core.configs import ForwardModelConfig
from recovar.em.helpers.batch_fetch import fetch_indexed_batch
from recovar.em.helpers.preprocessing import (
    apply_half_translation_phases,
    half_translation_phase_table,
)
from recovar.em.sparse_pass2.resident_operands import (
    ResidentOperandsUnsupported,
    gather_resident_chunk_operands,
    prepare_resident_half_operands,
    resident_half_operand_bytes,
)
from recovar.em.sparse_pass2.sparse_pass2_bucket_io import (
    _prepare_bucket_io,
    _relion_cuda_score_translation_angles_if_available,
    prepare_unshifted_bucket_operands,
)
from recovar.reconstruction import noise as noise_utils

pytestmark = pytest.mark.unit

N_IMAGES = 10
N_FINE_TRANS = 21
HALF_PIXELS = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)


def _recon_window(image_shape, current_size):
    """Centred half-layout indices of the current-size window."""

    n_rows, n_cols = image_shape
    half_width = n_cols // 2 + 1
    flat = np.arange(n_rows * half_width)
    ky = flat // half_width - n_rows // 2
    kx = flat % half_width
    keep = (ky >= -(current_size // 2) + 1) & (ky <= current_size // 2) & (kx <= current_size // 2)
    return flat[keep].astype(np.int32)


def _fine_translations(seed=20260919):
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [np.zeros((1, 2)), rng.uniform(-2.0, 2.0, (N_FINE_TRANS - 1, 2))]
    ).astype(np.float32)


def _case(*, relion_angles, current_size=6, seed=20260919):
    """Dataset, window and the ``_prepare_bucket_io`` keywords the driver passes.

    ``score_with_masked_images`` is on, as production has it, so the score and
    reconstruction images really are two different arrays and the two translate
    conventions of the M-step are both exercised.
    """

    dataset = MockDataset(n_images=N_IMAGES, seed=seed % 2**31)
    image_shape = dataset.image_shape
    window_indices = _recon_window(image_shape, current_size)
    fine_translations = _fine_translations(seed)
    noise_variance = jnp.linspace(0.4, 1.3, HALF_PIXELS, dtype=jnp.float32)
    noise_variance_half = noise_utils.to_batched_half_pixel_noise(
        noise_variance, image_shape
    ).squeeze()
    config = ForwardModelConfig.from_dataset(
        dataset, disc_type="linear_interp", process_fn=dataset.process_images
    )
    translation_angles = (
        _relion_cuda_score_translation_angles_if_available(
            fine_translations, image_shape, enabled=True, dtype=np.float32
        )
        if relion_angles
        else None
    )
    kwargs = dict(
        noise_variance_half=noise_variance_half,
        fine_translations=fine_translations,
        config=config,
        n_trans=N_FINE_TRANS,
        score_with_masked_images=True,
        half_spectrum_scoring=True,
        image_corrections=None,
        scale_corrections=None,
        image_pre_shifts=None,
        use_float64_scoring=False,
        score_only=False,
        score_mode="gaussian",
        window_indices=window_indices,
        recon_window_indices=window_indices,
        translation_phases_half=half_translation_phase_table(fine_translations, image_shape),
        relion_score_translation_angles=translation_angles,
        return_windowed_shifted=False,
        relion_exact_normalized_cc_operands=False,
        relion_exact_bpref_operands=False,
    )
    return dict(
        dataset=dataset,
        image_shape=image_shape,
        current_size=current_size,
        window_indices=window_indices,
        fine_translations=fine_translations,
        translation_angles=translation_angles,
        bucket_io_kwargs=kwargs,
    )


def _prepared(case, image_indices):
    batch_data, ctf_params, fetched = fetch_indexed_batch(
        case["dataset"], np.asarray(image_indices)
    )
    return (
        np.asarray(fetched),
        _prepare_bucket_io(
            case["dataset"],
            jnp.asarray(batch_data),
            ctf_params,
            np.asarray(fetched),
            return_direct_scoring_io=True,
            **case["bucket_io_kwargs"],
        ),
        jnp.asarray(batch_data),
        ctf_params,
    )


def _assert_bitwise(actual, expected, label):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape, f"{label}: {actual.shape} vs {expected.shape}"
    assert actual.dtype == expected.dtype, f"{label}: {actual.dtype} vs {expected.dtype}"
    if np.iscomplexobj(expected):
        np.testing.assert_array_equal(actual.real, expected.real, err_msg=label)
        np.testing.assert_array_equal(actual.imag, expected.imag, err_msg=label)
    else:
        np.testing.assert_array_equal(actual, expected, err_msg=label)


def test_unshifted_record_is_the_prepare_bucket_io_prologue():
    """The extracted record holds exactly the locals the tail consumed."""

    case = _case(relion_angles=False)
    fetched, prepared, batch, ctf_params = _prepared(case, np.arange(N_IMAGES))
    kwargs = case["bucket_io_kwargs"]
    unshifted = prepare_unshifted_bucket_operands(
        case["dataset"],
        batch,
        ctf_params,
        fetched,
        noise_variance_half=kwargs["noise_variance_half"],
        config=kwargs["config"],
        score_with_masked_images=kwargs["score_with_masked_images"],
        image_corrections=kwargs["image_corrections"],
        scale_corrections=kwargs["scale_corrections"],
        image_pre_shifts=kwargs["image_pre_shifts"],
        use_float64_scoring=kwargs["use_float64_scoring"],
        return_direct_scoring_io=True,
        score_only=False,
        score_mode=kwargs["score_mode"],
        window_indices=kwargs["window_indices"],
        relion_exact_normalized_cc_operands=False,
        relion_exact_bpref_operands=False,
    )
    (
        _shifted_score_half,
        shifted_recon_half,
        batch_norm,
        _ctf2_over_nv_half,
        ctf2_over_nv_half_with_dc,
        shifted_score_half_with_dc,
        processed_score_half_for_noise,
        *_rest,
    ) = prepared
    direct_score_input = prepared[8]

    _assert_bitwise(unshifted.batch_norm, batch_norm, "batch_norm")
    _assert_bitwise(
        unshifted.processed_score_half_for_noise,
        processed_score_half_for_noise,
        "processed_score_half_for_noise",
    )
    _assert_bitwise(unshifted.sparse_score_input_half, direct_score_input, "direct_score_input")
    _assert_bitwise(
        unshifted.ctf2_over_nv_recon_half, ctf2_over_nv_half_with_dc, "ctf2_over_nv_half_with_dc"
    )

    # Masked scoring must really separate the two reconstruction operands, or
    # the convention pairing below would be untested.
    assert not np.array_equal(
        np.asarray(unshifted.score_weighted_half), np.asarray(unshifted.recon_weighted_half)
    ), "the fixture's image mask is a no-op; the two operands must differ"

    phases = case["bucket_io_kwargs"]["translation_phases_half"]
    _assert_bitwise(
        apply_half_translation_phases(unshifted.recon_weighted_half, phases),
        shifted_recon_half,
        "translate(recon_weighted_half)",
    )
    _assert_bitwise(
        apply_half_translation_phases(unshifted.score_weighted_half, phases),
        shifted_score_half_with_dc,
        "translate(score_weighted_half)",
    )


def test_column_gather_commutes_with_the_translation():
    """Windowing before the shift is the same arithmetic as windowing after it.

    The resident path stores window columns and translates them; the per-chunk
    path translated the whole half and took the columns afterwards. The phase of
    a pixel depends only on that pixel, so the two agree bitwise -- this pins
    that property rather than assuming it.
    """

    case = _case(relion_angles=False)
    _fetched, prepared, _batch, _ctf = _prepared(case, np.arange(N_IMAGES))
    shifted_recon_half = np.asarray(prepared[1]).reshape(N_IMAGES, N_FINE_TRANS, -1)
    window = np.asarray(case["window_indices"], dtype=np.int32)

    phases_full = case["bucket_io_kwargs"]["translation_phases_half"]
    windowed_phases = np.asarray(phases_full)[:, window]
    unshifted = prepare_unshifted_bucket_operands(
        case["dataset"],
        *_prepared(case, np.arange(N_IMAGES))[2:4],
        np.arange(N_IMAGES, dtype=np.int64),
        noise_variance_half=case["bucket_io_kwargs"]["noise_variance_half"],
        config=case["bucket_io_kwargs"]["config"],
        score_with_masked_images=True,
        image_corrections=None,
        scale_corrections=None,
        image_pre_shifts=None,
        use_float64_scoring=False,
        return_direct_scoring_io=True,
        window_indices=case["window_indices"],
    )
    windowed_then_shifted = apply_half_translation_phases(
        jnp.asarray(unshifted.recon_weighted_half)[:, jnp.asarray(window)],
        jnp.asarray(windowed_phases),
    ).reshape(N_IMAGES, N_FINE_TRANS, window.size)
    _assert_bitwise(
        windowed_then_shifted,
        shifted_recon_half[:, :, window],
        "windowed translate",
    )


def test_operand_bytes_estimate_is_the_sum_of_the_stored_arrays():
    estimate = resident_half_operand_bytes(
        n_images=100,
        n_score_pixels=50,
        n_recon_pixels=40,
        n_half_pixels=200,
        n_fine_trans=21,
    )
    expected = 100 * (50 * 12 + 40 * (16 + 8 + 8) + 200 * 8 + 21 * 4 + 16)
    assert estimate == expected


def test_unmasked_scoring_is_refused():
    """The kernel cannot pair a BPref reconstruction sum with a BPref noise sum."""

    case = _case(relion_angles=False)
    kwargs = dict(case["bucket_io_kwargs"])
    kwargs["score_with_masked_images"] = False
    with pytest.raises(ResidentOperandsUnsupported, match="unmasked scoring"):
        prepare_resident_half_operands(
            case["dataset"],
            np.arange(N_IMAGES),
            bucket_io_kwargs=kwargs,
            window_indices=case["window_indices"],
            recon_window_indices=case["window_indices"],
            image_shape=case["image_shape"],
            n_fine_trans=N_FINE_TRANS,
        )


def _gpu_case(monkeypatch, custom_cuda_lib):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    if not cuda_backproject.relion_translate_sum_flat_rows_f32_supported():
        pytest.skip("loaded CUDA library predates the translate-and-sum target")
    return cuda_backproject


def _resident_operands(case):
    return prepare_resident_half_operands(
        case["dataset"],
        np.arange(N_IMAGES),
        bucket_io_kwargs=case["bucket_io_kwargs"],
        window_indices=case["window_indices"],
        recon_window_indices=case["window_indices"],
        image_shape=case["image_shape"],
        n_fine_trans=N_FINE_TRANS,
        image_batch_size=4,
    )


@pytest.mark.gpu
def test_resident_operands_translate_to_the_per_chunk_tiles(
    monkeypatch, custom_cuda_lib, gpu_device
):
    """One pass over the half reproduces every per-chunk operand bitwise."""

    cuda_backproject = _gpu_case(monkeypatch, custom_cuda_lib)
    with jax.default_device(gpu_device):
        case = _case(relion_angles=True)
        assert case["translation_angles"] is not None
        operands = _resident_operands(case)
        _fetched, prepared, _batch, _ctf = _prepared(case, np.arange(N_IMAGES))
        window = jnp.asarray(case["window_indices"], dtype=jnp.int32)
        angles = jnp.asarray(case["translation_angles"], dtype=jnp.float32)

        translated_recon = cuda_backproject.relion_translate_score_f32(
            jnp.asarray(operands.recon_image, dtype=jnp.complex64), angles, window, IMAGE_SHAPE
        )
        translated_noise = cuda_backproject.relion_translate_score_f32(
            jnp.asarray(operands.noise_image, dtype=jnp.complex64), angles, window, IMAGE_SHAPE
        )
        n_pixels = int(window.shape[0])
        reference_recon = np.asarray(prepared[1]).reshape(N_IMAGES, N_FINE_TRANS, -1)[:, :, np.asarray(window)]
        reference_noise = np.asarray(prepared[5]).reshape(N_IMAGES, N_FINE_TRANS, -1)[:, :, np.asarray(window)]

    assert operands.recon_weight is None, "the fixture has no exact-BPref operands"
    _assert_bitwise(
        np.asarray(translated_recon).reshape(N_IMAGES, N_FINE_TRANS, n_pixels),
        reference_recon,
        "resident recon operand",
    )
    _assert_bitwise(
        np.asarray(translated_noise).reshape(N_IMAGES, N_FINE_TRANS, n_pixels),
        reference_noise,
        "resident noise operand",
    )
    _assert_bitwise(
        operands.score_input,
        np.asarray(prepared[8])[:, np.asarray(window)],
        "resident score_input",
    )
    _assert_bitwise(
        operands.ctf2_over_nv_recon,
        np.asarray(prepared[4])[:, np.asarray(window)],
        "resident ctf2_over_nv_recon",
    )
    _assert_bitwise(
        operands.processed_image_half, np.asarray(prepared[6]), "resident processed_image_half"
    )


@pytest.mark.gpu
@pytest.mark.parametrize("kernel_ctf_probs", [False, True])
def test_translate_sum_kernel_matches_the_block_reduction_bitwise(
    monkeypatch, custom_cuda_lib, gpu_device, kernel_ctf_probs
):
    """T15's kernel equals the gathered-tile reduction at 21 translations.

    ``summed`` and ``summed_masked`` are bitwise in both forms. ``ctf_probs`` is
    bitwise only in the default form, which keeps the XLA statement: the
    kernel's own ``probs_sum_t`` reduces the translations sequentially, and XLA
    does not at this block shape, which moves the fourth output by up to one
    relative ulp. The failure is recorded here rather than tolerated silently.
    """

    from recovar.em.sparse_pass2.resident_pass2 import (
        _resident_block_weighted_sums,
        _resident_block_weighted_sums_kernel,
    )

    cuda_backproject = _gpu_case(monkeypatch, custom_cuda_lib)
    rng = np.random.default_rng(31415)
    block_rows = 64
    with jax.default_device(gpu_device):
        case = _case(relion_angles=True)
        operands = _resident_operands(case)
        window = jnp.asarray(case["window_indices"], dtype=jnp.int32)
        angles = jnp.asarray(case["translation_angles"], dtype=jnp.float32)
        n_pixels = int(window.shape[0])

        shifted_recon = cuda_backproject.relion_translate_score_f32(
            jnp.asarray(operands.recon_image, dtype=jnp.complex64), angles, window, IMAGE_SHAPE
        ).reshape(N_IMAGES, N_FINE_TRANS, n_pixels)
        shifted_noise = cuda_backproject.relion_translate_score_f32(
            jnp.asarray(operands.noise_image, dtype=jnp.complex64), angles, window, IMAGE_SHAPE
        ).reshape(N_IMAGES, N_FINE_TRANS, n_pixels)

        row_image = rng.integers(0, N_IMAGES, size=block_rows).astype(np.int32)
        posterior = rng.random((block_rows, N_FINE_TRANS)).astype(np.float32)
        # A padded row and a zero-mass row, the two cases the kernel short-circuits.
        posterior[3] = 0.0
        row_image_ids = row_image.copy()
        row_image_ids[7] = -1
        posterior[7] = 0.0

        summed_ref, masked_ref, ctf_ref, _mass_ref = jax.block_until_ready(
            _resident_block_weighted_sums(
                jnp.asarray(posterior),
                jnp.asarray(row_image),
                shifted_recon,
                shifted_noise,
                jnp.asarray(operands.ctf2_over_nv_recon),
            )
        )
        summed, masked, ctf_probs, _mass = jax.block_until_ready(
            _resident_block_weighted_sums_kernel(
                jnp.asarray(posterior),
                jnp.asarray(row_image_ids),
                jnp.asarray(row_image),
                operands.recon_image,
                operands.recon_weight,
                operands.noise_image,
                operands.ctf2_over_nv_recon,
                window,
                angles,
                image_shape=IMAGE_SHAPE,
                n_recon_pixels=n_pixels,
                kernel_ctf_probs=kernel_ctf_probs,
                cuda_backproject=cuda_backproject,
            )
        )

    live = np.ones(block_rows, dtype=bool)
    live[7] = False
    _assert_bitwise(np.asarray(summed)[live], np.asarray(summed_ref)[live], "summed")
    _assert_bitwise(np.asarray(masked)[live], np.asarray(masked_ref)[live], "summed_masked")
    assert np.all(np.asarray(summed)[7] == 0)
    assert np.all(np.asarray(summed)[3] == 0)
    if not kernel_ctf_probs:
        _assert_bitwise(np.asarray(ctf_probs)[live], np.asarray(ctf_ref)[live], "ctf_probs")
        assert np.all(np.asarray(ctf_probs)[7] == 0) and np.all(np.asarray(ctf_probs)[3] == 0)
    else:
        # The fused fourth output is the measured exception: a bounded ulp gap,
        # not an equality. Record the bound the arm is allowed to show.
        actual = np.asarray(ctf_probs)[live].astype(np.float64)
        expected = np.asarray(ctf_ref)[live].astype(np.float64)
        relative = np.abs(actual - expected) / np.maximum(np.abs(expected), 1e-30)
        assert float(relative.max()) <= 4.0 * np.finfo(np.float32).eps, float(relative.max())


@pytest.mark.gpu
def test_chunk_gather_reproduces_the_capacity_padding(
    monkeypatch, custom_cuda_lib, gpu_device
):
    """Padded slots carry the per-chunk path's padding values, not stale rows."""

    _gpu_case(monkeypatch, custom_cuda_lib)
    with jax.default_device(gpu_device):
        case = _case(relion_angles=True)
        operands = _resident_operands(case)
        image_capacity = 8
        slots = np.full(image_capacity, -1, dtype=np.int32)
        slots[:5] = np.arange(5, dtype=np.int32)
        rect_indices = np.asarray(case["window_indices"], dtype=np.int32)
        gathered = gather_resident_chunk_operands(
            operands,
            slots,
            translation_angles=case["translation_angles"],
            rect_indices=rect_indices,
            exact_positions=np.arange(rect_indices.size, dtype=np.int32),
            image_shape=IMAGE_SHAPE,
            current_size=case["current_size"],
            use_exact_relion_gaussian=False,
            accumulate_noise=False,
            source_faithful_spectrum_norm=False,
        )

    for name, source in (
        ("score_input", operands.score_input),
        ("recon_image", operands.recon_image),
        ("noise_image", operands.noise_image),
        ("ctf2_over_nv_recon", operands.ctf2_over_nv_recon),
        ("processed_image_half", operands.processed_image_half),
    ):
        value = np.asarray(gathered[name])
        _assert_bitwise(value[:5], np.asarray(source)[:5], f"{name} live rows")
        assert np.all(value[5:] == 0), f"{name} padded rows are not zero"
    assert np.all(np.asarray(gathered["scale"])[5:] == 1.0)
    assert np.all(np.asarray(gathered["group_ids"])[5:] == -1)
    assert np.all(np.asarray(gathered["raw_translated_wavg_rectangle"])[5:] == 0)

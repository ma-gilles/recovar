"""Flat-row translate-and-sum against the resident M-step's XLA reduction.

``resident_pass2._resident_block_weighted_sums`` gathers a pre-shifted
``[images, translations, pixels]`` tile per row and contracts it with the
posterior at ``Precision.HIGHEST``.  The CUDA kernel applies RELION's
translation inside the reduction instead, so the tile never exists.  Every
reference below is therefore built by the primitive production uses --
``relion_translate_score_f32``, or ``relion_translate_bpref_f32`` for the
reconstruction operand when ``relion_exact_bpref_operands`` is selected --
followed by the XLA reduction, so a comparison isolates the summation and not
the translation.  The two paths differ only in how the products over
translations are summed: the kernel sums sequentially in increasing ``t`` with
a rounded multiply and a rounded add, XLA contracts the same products in its
reduce fusion.

Measured on an A100 with nvcc 13.3: at the production translation count (21)
XLA emits a sequential reduction over ``t``, so the two paths agree **bitwise**
at both the early (415) and hp3 (3386) reconstruction pixel counts, and at
T = 1, 5 and 13.  From T = 32 XLA changes the reduction order and the paths
differ by at most 4 ulp of the magnitude the summation works at,
``sum_t |posterior[r, t]| * |image[id, p]|``.  That is the meaningful bound for
a sum of randomly phased complex terms: in the few percent of cells where such
a sum cancels far below its terms, the same absolute gap reads as hundreds of
ulp *of the result* while the arithmetic is still correct to half an ulp of the
summation.  The tests assert the scale-relative bound everywhere and bitwise
equality at the shapes the resident M-step actually runs.
"""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

pytestmark = pytest.mark.unit

IMAGE_SHAPE = (128, 128)
HALF_WIDTH = IMAGE_SHAPE[1] // 2 + 1
HALF_PIXELS = IMAGE_SHAPE[0] * HALF_WIDTH

# Both paths form the same products and differ only in the order of the
# additions over translations, so the gap is bounded by the rounding of that
# summation: ``sum_t |posterior| * |image|`` times a few float32 eps.  An
# ulp-of-result bound is not usable here because a sum of randomly phased
# complex terms can cancel to far below the magnitude of its terms.
_MAX_ULP_OF_SUM_SCALE = 4.0
# ``probs_sum_t`` sums non-negative weights, where the sequential order and any
# tree order differ by at most (T - 1) eps relative.
_MAX_MASS_ULP_PER_TRANSLATION = 1.0


def _cuda_backproject(monkeypatch, custom_cuda_lib):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    if not cuda_backproject.relion_translate_sum_flat_rows_f32_supported():
        pytest.skip("loaded CUDA library predates the translate-and-sum target")
    return cuda_backproject


def _operands(
    rng,
    *,
    rows,
    image_capacity,
    n_trans,
    n_pixels,
    zero_rows=(),
    padded_rows=(),
):
    pixel_indices = np.sort(
        rng.choice(HALF_PIXELS, size=n_pixels, replace=False)
    ).astype(np.int32)
    posterior = rng.random((rows, n_trans)).astype(np.float32)
    for row in zero_rows:
        posterior[row, :] = 0.0
    row_image_ids = rng.integers(0, image_capacity, size=rows).astype(np.int32)
    for row in padded_rows:
        row_image_ids[row] = -1
    recon_image = (
        rng.standard_normal((image_capacity, n_pixels))
        + 1j * rng.standard_normal((image_capacity, n_pixels))
    ).astype(np.complex64)
    noise_image = (
        rng.standard_normal((image_capacity, n_pixels))
        + 1j * rng.standard_normal((image_capacity, n_pixels))
    ).astype(np.complex64)
    shifts = rng.uniform(-4.0, 4.0, size=(n_trans, 2))
    translation_angles = (
        -2.0 * np.pi * shifts / float(IMAGE_SHAPE[0])
    ).astype(np.float32)
    ctf2 = rng.random((image_capacity, n_pixels)).astype(np.float32)
    recon_weight = rng.uniform(-2.0, 2.0, (image_capacity, n_pixels)).astype(
        np.float32
    )
    return dict(
        pixel_indices=pixel_indices,
        recon_weight=recon_weight,
        posterior=posterior,
        row_image_ids=row_image_ids,
        recon_image=recon_image,
        noise_image=noise_image,
        translation_angles=translation_angles,
        ctf2=ctf2,
    )


def _reference(cuda_backproject, operands, *, reference_row_ids=None, bpref=False):
    """The production tile builder followed by the XLA weighted sums."""

    from recovar.em.sparse_pass2.resident_pass2 import _resident_block_weighted_sums

    n_images, n_pixels = operands["recon_image"].shape
    n_trans = operands["translation_angles"].shape[0]
    shifted = []
    for key in ("recon_image", "noise_image"):
        if key == "recon_image" and bpref:
            tile = cuda_backproject.relion_translate_bpref_f32(
                jnp.asarray(operands[key]),
                jnp.asarray(operands["recon_weight"]),
                jnp.asarray(operands["translation_angles"]),
                jnp.asarray(operands["pixel_indices"]),
                IMAGE_SHAPE,
            )
        else:
            tile = cuda_backproject.relion_translate_score_f32(
                jnp.asarray(operands[key]),
                jnp.asarray(operands["translation_angles"]),
                jnp.asarray(operands["pixel_indices"]),
                IMAGE_SHAPE,
            )
        shifted.append(tile.reshape(n_images, n_trans, n_pixels))
    ids = (
        operands["row_image_ids"]
        if reference_row_ids is None
        else reference_row_ids
    )
    summed, masked, _ctf, mass = _resident_block_weighted_sums(
        jnp.asarray(operands["posterior"]),
        jnp.asarray(ids),
        shifted[0],
        shifted[1],
        jnp.asarray(operands["ctf2"]),
    )
    return jax.block_until_ready((summed, masked, mass))


def _kernel(
    cuda_backproject,
    operands,
    *,
    n_valid_rows,
    logical_pixels,
    rows_per_block=0,
    bpref=False,
):
    return jax.block_until_ready(
        cuda_backproject.relion_translate_sum_flat_rows_f32(
            jnp.asarray(operands["recon_image"]),
            jnp.asarray(operands["noise_image"]),
            jnp.asarray(operands["row_image_ids"]),
            jnp.asarray(operands["posterior"]),
            jnp.asarray(operands["translation_angles"]),
            jnp.asarray(operands["pixel_indices"]),
            jnp.asarray(n_valid_rows, dtype=jnp.int32),
            jnp.asarray(logical_pixels, dtype=jnp.int32),
            jnp.asarray(operands["recon_weight"]) if bpref else None,
            image_shape=IMAGE_SHAPE,
            rows_per_block=rows_per_block,
        )
    )


def _ulp_of_result(actual, expected):
    a = np.asarray(actual, dtype=np.float32).astype(np.float64)
    b = np.asarray(expected, dtype=np.float32).astype(np.float64)
    scale = np.maximum(np.abs(a), np.abs(b)).astype(np.float32)
    spacing = np.spacing(scale)
    out = np.abs(a - b) / np.where(scale == 0, np.float32(1.0), spacing)
    return np.where(scale == 0, 0.0, out)


def _assert_close(actual, expected, sum_scale, label, max_ulp_of_scale=None):
    """Bound every cell against the summation scale; report both measures."""

    actual = np.asarray(actual)
    expected = np.asarray(expected)
    eps = float(np.finfo(np.float32).eps)
    limit = _MAX_ULP_OF_SUM_SCALE if max_ulp_of_scale is None else max_ulp_of_scale
    floor = np.finfo(np.float64).tiny
    worst_scale = 0.0
    worst_result = 0.0
    components = ("real", "imag") if expected.dtype.kind == "c" else (None,)
    for component in components:
        a = getattr(actual, component) if component else actual
        b = getattr(expected, component) if component else expected
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        scaled = diff / np.maximum(eps * sum_scale, floor)
        worst_scale = max(worst_scale, float(scaled.max()))
        worst_result = max(worst_result, float(_ulp_of_result(a, b).max()))
        assert np.all(scaled <= limit), (
            f"{label}.{component}: {scaled.max()} ulp of the summation scale "
            f"exceeds {limit}"
        )
    return worst_scale, worst_result


def _sum_scale(operands, key, row_ids=None):
    ids = operands["row_image_ids"] if row_ids is None else row_ids
    weights = np.abs(operands["posterior"]).sum(axis=1).astype(np.float64)
    magnitude = np.abs(operands[key]).astype(np.float64)[np.asarray(ids)]
    return weights[:, None] * magnitude


def _assert_bitwise(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.dtype.kind == "c":
        np.testing.assert_array_equal(actual.real, expected.real)
        np.testing.assert_array_equal(actual.imag, expected.imag)
    else:
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("rows", "image_capacity", "n_trans", "n_pixels", "bitwise"),
    [
        (4096, 128, 21, 415, True),
        (1024, 128, 21, 3386, True),
        (512, 32, 64, 311, False),
    ],
    ids=["early", "hp3", "many_translations"],
)
def test_translate_sum_matches_resident_block_weighted_sums(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
    rows,
    image_capacity,
    n_trans,
    n_pixels,
    bitwise,
):
    cuda_backproject = _cuda_backproject(monkeypatch, custom_cuda_lib)
    rng = np.random.default_rng(1500 + n_pixels + n_trans)
    operands = _operands(
        rng,
        rows=rows,
        image_capacity=image_capacity,
        n_trans=n_trans,
        n_pixels=n_pixels,
    )
    with jax.default_device(gpu_device):
        summed_ref, masked_ref, mass_ref = _reference(cuda_backproject, operands)
        summed, masked, mass = _kernel(
            cuda_backproject,
            operands,
            n_valid_rows=rows,
            logical_pixels=n_pixels,
        )
    worst = [
        _assert_close(
            summed, summed_ref, _sum_scale(operands, "recon_image"), "summed"
        ),
        _assert_close(
            masked, masked_ref, _sum_scale(operands, "noise_image"), "summed_masked"
        ),
    ]
    mass_scale = np.abs(operands["posterior"]).sum(axis=1).astype(np.float64)
    worst.append(
        _assert_close(
            mass,
            mass_ref,
            mass_scale,
            "probs_sum_t",
            max_ulp_of_scale=_MAX_MASS_ULP_PER_TRANSLATION * n_trans,
        )
    )
    print(
        f"{rows}x{n_trans}x{n_pixels}: max ulp of the summation scale "
        f"{max(entry[0] for entry in worst)}, max ulp of the result "
        f"{max(entry[1] for entry in worst)}"
    )
    if bitwise:
        # At the translation counts the resident M-step runs, XLA reduces
        # sequentially over t, which is exactly the kernel's order. This is the
        # property T14's integration inherits; a jaxlib that reduces
        # differently must be re-qualified rather than silently tolerated.
        _assert_bitwise(summed, summed_ref)
        _assert_bitwise(masked, masked_ref)


@pytest.mark.gpu
def test_translate_sum_is_bitwise_for_a_single_translation(
    monkeypatch, custom_cuda_lib, gpu_device
):
    cuda_backproject = _cuda_backproject(monkeypatch, custom_cuda_lib)
    rng = np.random.default_rng(97)
    operands = _operands(
        rng, rows=257, image_capacity=9, n_trans=1, n_pixels=311
    )
    with jax.default_device(gpu_device):
        summed_ref, masked_ref, mass_ref = _reference(cuda_backproject, operands)
        summed, masked, mass = _kernel(
            cuda_backproject, operands, n_valid_rows=257, logical_pixels=311
        )
    # One translation leaves nothing to sum: the products must be identical.
    _assert_bitwise(summed, summed_ref)
    _assert_bitwise(masked, masked_ref)
    _assert_bitwise(mass, mass_ref)


@pytest.mark.gpu
@pytest.mark.parametrize("rows_per_block", [1, 2, 4, 8])
def test_rows_per_block_does_not_change_the_result(
    monkeypatch, custom_cuda_lib, gpu_device, rows_per_block
):
    cuda_backproject = _cuda_backproject(monkeypatch, custom_cuda_lib)
    rng = np.random.default_rng(3113)
    operands = _operands(rng, rows=333, image_capacity=17, n_trans=21, n_pixels=511)
    with jax.default_device(gpu_device):
        baseline = _kernel(
            cuda_backproject,
            operands,
            n_valid_rows=333,
            logical_pixels=511,
            rows_per_block=1,
        )
        tiled = _kernel(
            cuda_backproject,
            operands,
            n_valid_rows=333,
            logical_pixels=511,
            rows_per_block=rows_per_block,
        )
    for actual, expected in zip(tiled, baseline):
        _assert_bitwise(actual, expected)


@pytest.mark.gpu
@pytest.mark.parametrize("rows_per_block", [1, 4])
def test_padded_zero_mass_rows_and_pixel_tail_are_zero(
    monkeypatch, custom_cuda_lib, gpu_device, rows_per_block
):
    cuda_backproject = _cuda_backproject(monkeypatch, custom_cuda_lib)
    rng = np.random.default_rng(654)
    rows, n_pixels, logical_pixels = 96, 200, 137
    n_valid_rows = 70
    padded_rows = (0, 3, 4, 5, 37, 69)
    zero_rows = (1, 2, 36, 68)
    operands = _operands(
        rng,
        rows=rows,
        image_capacity=11,
        n_trans=21,
        n_pixels=n_pixels,
        zero_rows=zero_rows,
        padded_rows=padded_rows,
    )
    with jax.default_device(gpu_device):
        summed, masked, mass = _kernel(
            cuda_backproject,
            operands,
            n_valid_rows=n_valid_rows,
            logical_pixels=logical_pixels,
            rows_per_block=rows_per_block,
        )
    summed = np.asarray(summed)
    masked = np.asarray(masked)
    mass = np.asarray(mass)

    for row in padded_rows:
        assert np.all(summed[row] == 0), f"padded row {row} wrote data"
        assert np.all(masked[row] == 0), f"padded row {row} wrote data"
        assert mass[row] == 0.0
    for row in zero_rows:
        assert np.all(summed[row] == 0), f"zero-mass row {row} is not zero"
        assert np.all(masked[row] == 0), f"zero-mass row {row} is not zero"
        assert mass[row] == 0.0
    assert np.all(summed[n_valid_rows:] == 0)
    assert np.all(masked[n_valid_rows:] == 0)
    assert np.all(mass[n_valid_rows:] == 0)
    assert np.all(summed[:, logical_pixels:] == 0)
    assert np.all(masked[:, logical_pixels:] == 0)

    # Every live row must still equal the XLA reduction over its own prefix.
    reference_ids = operands["row_image_ids"].copy()
    live = np.ones(rows, dtype=bool)
    live[list(padded_rows)] = False
    live[n_valid_rows:] = False
    reference_ids[~live] = 0
    with jax.default_device(gpu_device):
        summed_ref, masked_ref, mass_ref = _reference(
            cuda_backproject, operands, reference_row_ids=reference_ids
        )
    summed_ref = np.asarray(summed_ref)[:, :logical_pixels]
    masked_ref = np.asarray(masked_ref)[:, :logical_pixels]
    scale_recon = _sum_scale(operands, "recon_image", reference_ids)[:, :logical_pixels]
    scale_noise = _sum_scale(operands, "noise_image", reference_ids)[:, :logical_pixels]
    _assert_close(
        summed[live][:, :logical_pixels],
        summed_ref[live],
        scale_recon[live],
        "summed(live)",
    )
    _assert_close(
        masked[live][:, :logical_pixels],
        masked_ref[live],
        scale_noise[live],
        "summed_masked(live)",
    )
    mass_scale = np.abs(operands["posterior"]).sum(axis=1).astype(np.float64)
    _assert_close(
        mass[live],
        np.asarray(mass_ref)[live],
        mass_scale[live],
        "probs_sum_t(live)",
        max_ulp_of_scale=_MAX_MASS_ULP_PER_TRANSLATION * 21,
    )
    _assert_bitwise(summed[live][:, :logical_pixels], summed_ref[live])
    _assert_bitwise(masked[live][:, :logical_pixels], masked_ref[live])


@pytest.mark.gpu
def test_wrapper_is_traceable_with_static_shapes(
    monkeypatch, custom_cuda_lib, gpu_device
):
    """T14 calls the wrapper from a ``fori_loop`` body: only shapes are static."""

    cuda_backproject = _cuda_backproject(monkeypatch, custom_cuda_lib)
    rng = np.random.default_rng(8)
    rows, n_pixels = 64, 96
    operands = _operands(rng, rows=rows, image_capacity=5, n_trans=7, n_pixels=n_pixels)

    def body(carry, n_valid_rows, logical_pixels):
        summed, masked, mass = cuda_backproject.relion_translate_sum_flat_rows_f32(
            jnp.asarray(operands["recon_image"]),
            jnp.asarray(operands["noise_image"]),
            jnp.asarray(operands["row_image_ids"]),
            jnp.asarray(operands["posterior"]),
            jnp.asarray(operands["translation_angles"]),
            jnp.asarray(operands["pixel_indices"]),
            n_valid_rows,
            logical_pixels,
            image_shape=IMAGE_SHAPE,
        )
        return carry + jnp.sum(jnp.abs(summed)) + jnp.sum(jnp.abs(masked)) + jnp.sum(mass)

    traced = jax.jit(body)
    with jax.default_device(gpu_device):
        first = traced(
            jnp.float32(0.0),
            jnp.asarray(rows, dtype=jnp.int32),
            jnp.asarray(n_pixels, dtype=jnp.int32),
        )
        second = traced(
            jnp.float32(0.0),
            jnp.asarray(rows // 2, dtype=jnp.int32),
            jnp.asarray(n_pixels // 2, dtype=jnp.int32),
        )
        first, second = jax.block_until_ready((first, second))
    # Both calls reuse one traced program; the device scalars change the work.
    assert float(first) > float(second) > 0.0


@pytest.mark.gpu
@pytest.mark.parametrize("rows_per_block", [1, 4])
def test_exact_bpref_recon_operand_matches_its_own_translate(
    monkeypatch, custom_cuda_lib, gpu_device, rows_per_block
):
    """``relion_exact_bpref_operands`` builds the recon tile with a different
    primitive: ``relion_translate_bpref_f32`` rounds the imaginary component as
    ``cosine * v.y + sine * v.x`` and multiplies by the weighted CTF after the
    rotation. The kernel must reproduce that tile, not the score one, when the
    weight is supplied -- and the noise tile must stay on the score path."""

    cuda_backproject = _cuda_backproject(monkeypatch, custom_cuda_lib)
    rng = np.random.default_rng(2025)
    rows, n_pixels, n_trans = 1024, 415, 21
    operands = _operands(
        rng, rows=rows, image_capacity=32, n_trans=n_trans, n_pixels=n_pixels
    )
    with jax.default_device(gpu_device):
        summed_ref, masked_ref, _mass_ref = _reference(
            cuda_backproject, operands, bpref=True
        )
        summed, masked, _mass = _kernel(
            cuda_backproject,
            operands,
            n_valid_rows=rows,
            logical_pixels=n_pixels,
            rows_per_block=rows_per_block,
            bpref=True,
        )
    _assert_bitwise(summed, summed_ref)
    _assert_bitwise(masked, masked_ref)

    # The two conventions are genuinely different arithmetic, so the score-mode
    # result must NOT equal the BPref reference; otherwise this test would pass
    # for the wrong reason.
    with jax.default_device(gpu_device):
        score_mode, _m, _p = _kernel(
            cuda_backproject,
            operands,
            n_valid_rows=rows,
            logical_pixels=n_pixels,
            rows_per_block=rows_per_block,
            bpref=False,
        )
    assert not np.array_equal(np.asarray(score_mode), np.asarray(summed_ref))


@pytest.mark.gpu
@pytest.mark.parametrize("bpref", [False, True], ids=["score", "bpref"])
def test_in_kernel_translation_phases_are_bitwise(
    monkeypatch, custom_cuda_lib, gpu_device, bpref
):
    """The in-kernel translation must equal the production translate primitive.

    One translation with unit posterior turns the reduction into the identity
    (``1.0 * x`` is exact), so the outputs are the translated operands
    themselves and any phase or rotation difference shows up directly.
    """

    cuda_backproject = _cuda_backproject(monkeypatch, custom_cuda_lib)
    rng = np.random.default_rng(555)
    n_images, n_pixels = 24, 617
    operands = _operands(
        rng, rows=n_images, image_capacity=n_images, n_trans=1, n_pixels=n_pixels
    )
    operands["row_image_ids"] = np.arange(n_images, dtype=np.int32)
    operands["posterior"] = np.ones((n_images, 1), dtype=np.float32)

    with jax.default_device(gpu_device):
        if bpref:
            recon_ref = cuda_backproject.relion_translate_bpref_f32(
                jnp.asarray(operands["recon_image"]),
                jnp.asarray(operands["recon_weight"]),
                jnp.asarray(operands["translation_angles"]),
                jnp.asarray(operands["pixel_indices"]),
                IMAGE_SHAPE,
            )
        else:
            recon_ref = cuda_backproject.relion_translate_score_f32(
                jnp.asarray(operands["recon_image"]),
                jnp.asarray(operands["translation_angles"]),
                jnp.asarray(operands["pixel_indices"]),
                IMAGE_SHAPE,
            )
        noise_ref = cuda_backproject.relion_translate_score_f32(
            jnp.asarray(operands["noise_image"]),
            jnp.asarray(operands["translation_angles"]),
            jnp.asarray(operands["pixel_indices"]),
            IMAGE_SHAPE,
        )
        summed, masked, mass = _kernel(
            cuda_backproject,
            operands,
            n_valid_rows=n_images,
            logical_pixels=n_pixels,
            bpref=bpref,
        )
    _assert_bitwise(summed, np.asarray(recon_ref).reshape(n_images, n_pixels))
    _assert_bitwise(masked, np.asarray(noise_ref).reshape(n_images, n_pixels))
    np.testing.assert_array_equal(np.asarray(mass), np.ones(n_images, np.float32))

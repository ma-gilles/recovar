"""GPU tests for the device-resident pass-2 scoring stage (T6).

Ticket: em_parity_tickets_20260918/T6_resident_scoring_stage.md.
Design: em_device_resident_pass2_design_20260918.md (stages 1-3).

The stage under test is a layout change, not an arithmetic change: its raw
diff2 must equal the compact engine's raw diff2 for the same rows, the same
cached projections and the same ``_prepare_bucket_io`` operands. Two
comparisons are made, because the compact engine has two routes:

* against the compact *pairs* fused-translate scorer, which reaches the same
  CUDA kernel body with a different row addressing: bitwise;
* against the pre-shifted rectangular masked path, which shifts the image with
  a separate kernel before scoring: reported as a ULP distribution, since the
  two translate implementations are allowed to differ in the last bits.
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
from recovar.em.helpers.half_spectrum import make_scoring_half_image_weights
from recovar.em.helpers.preprocessing import half_translation_phase_table
from recovar.em.scoring.compact_candidates import _candidate_mask_to_dense
from recovar.em.scoring.significant_samples import ComplementSignificantSampleIndices
from recovar.em.scoring.sparse_bucket_arrays import _prepare_per_image_pass2_inputs
from recovar.em.sparse_pass2 import sparse_pass2_scoring as spb
from recovar.em.sparse_pass2.resident_candidates import (
    build_resident_candidate_tables,
    materialize_chunk,
    plan_capacity_chunks,
)
from recovar.em.sparse_pass2.resident_scoring import (
    materialize_chunk_device,
    prepare_resident_image_operands,
    score_all_chunks,
    score_resident_chunk,
)
from recovar.em.sparse_pass2.sparse_pass2_bucket_io import (
    _prepare_bucket_io,
    _relion_cuda_score_translation_angles_if_available,
)
from recovar.reconstruction import noise as noise_utils

N_COARSE_ROT = 6
CHILDREN = 2
N_COARSE_TRANS = 4
N_FINE_TRANS = 8
FINE_TRANS_PARENT = np.repeat(np.arange(N_COARSE_TRANS, dtype=np.int32), 2)
N_IMAGES = 10


def _z_rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a.view(np.int32).astype(np.int64) - b.view(np.int32).astype(np.int64))


def _fine_rotation_grid():
    rotations = np.stack(
        [_z_rotation(0.07 * k) for k in range(N_COARSE_ROT * CHILDREN)]
    ).astype(np.float32)
    parent = np.repeat(np.arange(N_COARSE_ROT, dtype=np.int32), CHILDREN)
    return rotations, parent


def _significant_sample_indices(rng):
    """One significant (coarse rot, coarse trans) set per image.

    Images 0 and 1 are the degenerate modes the candidate table supports
    (``full`` support and an ``empty`` set); image 2 uses the sparse
    complement; the rest are ordinary explicit sample lists of varying size,
    which is what makes the row counts per image uneven.
    """

    total = N_COARSE_ROT * N_COARSE_TRANS
    samples = [None, np.asarray([], dtype=np.int32)]
    samples.append(
        ComplementSignificantSampleIndices(
            excluded_indices=np.asarray([0, 5, 9, 17], dtype=np.int32),
            total_size=total,
        )
    )
    for image in range(3, N_IMAGES):
        count = int(rng.integers(1, total))
        samples.append(
            np.sort(rng.choice(total, size=count, replace=False).astype(np.int32))
        )
    return samples


def _build_candidate_tables(seed=20260918):
    rng = np.random.default_rng(seed)
    fine_rotations, fine_parent = _fine_rotation_grid()
    per_image_inputs = _prepare_per_image_pass2_inputs(
        _significant_sample_indices(rng),
        n_coarse_rot=N_COARSE_ROT,
        n_coarse_trans=N_COARSE_TRANS,
        nside_level=0,
        oversampling_order=0,
        n_fine_trans=N_FINE_TRANS,
        fine_translation_parent=FINE_TRANS_PARENT,
        rotation_log_prior=np.linspace(-0.4, 0.4, N_COARSE_ROT, dtype=np.float32),
        random_perturbation=0.0,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
        dtype=np.float32,
    )
    modes = [mask.mode for mask in per_image_inputs["candidate_mask"]]
    assert modes[0] == "full" and modes[1] == "empty" and modes[2] == "coarse_exclude", (
        f"fixture built modes {modes[:3]}; update the fixture, not the assertion"
    )
    assert "coarse" in modes, "fixture must exercise the explicit coarse mask mode"
    tables = build_resident_candidate_tables(
        per_image_inputs,
        n_coarse_trans=N_COARSE_TRANS,
        n_fine_trans=N_FINE_TRANS,
        fine_translation_parent=FINE_TRANS_PARENT,
    )
    return per_image_inputs, tables, fine_rotations.shape[0]


def _bucket_io_kwargs(dataset, *, current_size, window_indices, translation_angles, fine_translations):
    """The keyword arguments the K=1 loop passes to ``_prepare_bucket_io``."""

    image_shape = dataset.image_shape
    n_half = image_shape[0] * (image_shape[1] // 2 + 1)
    noise_variance = jnp.ones(n_half, dtype=jnp.float32) * 0.7
    noise_variance_half = noise_utils.to_batched_half_pixel_noise(
        noise_variance, image_shape
    ).squeeze()
    config = ForwardModelConfig.from_dataset(
        dataset, disc_type="linear_interp", process_fn=dataset.process_images
    )
    return dict(
        noise_variance_half=noise_variance_half,
        fine_translations=fine_translations,
        config=config,
        n_trans=N_FINE_TRANS,
        score_with_masked_images=False,
        half_spectrum_scoring=True,
        image_corrections=None,
        scale_corrections=None,
        image_pre_shifts=None,
        use_float64_scoring=False,
        score_only=True,
        score_mode="gaussian",
        window_indices=window_indices,
        recon_window_indices=window_indices,
        translation_phases_half=half_translation_phase_table(fine_translations, image_shape),
        relion_score_translation_angles=translation_angles,
        return_windowed_shifted=False,
        relion_exact_normalized_cc_operands=False,
        relion_exact_bpref_operands=False,
    )


def _score_window(image_shape, current_size):
    """Centred half-layout indices of the current-size score window.

    Mirrors the production window: the FFTW row ``+cs/2`` (positive Nyquist)
    is kept and ``-cs/2`` is not, which is the convention both the RELION fine
    kernel and the fused kernel apply to the translation phase.
    """

    n_rows, n_cols = image_shape
    half_width = n_cols // 2 + 1
    flat = np.arange(n_rows * half_width)
    ky = flat // half_width - n_rows // 2
    kx = flat % half_width
    keep = (ky >= -(current_size // 2) + 1) & (ky <= current_size // 2) & (kx <= current_size // 2)
    return flat[keep].astype(np.int32)


def _build_case(current_size, *, row_capacity_ladder, image_capacity_ladder, seed=20260918):
    """Fixture bundle: dataset operands, candidate tables, chunks, projections."""

    per_image_inputs, tables, n_fine_rot = _build_candidate_tables(seed=seed)
    dataset = MockDataset(n_images=N_IMAGES, seed=seed % 2**31)
    image_shape = dataset.image_shape
    n_half = image_shape[0] * (image_shape[1] // 2 + 1)
    use_window = current_size < image_shape[0]
    window_indices = _score_window(image_shape, current_size) if use_window else None

    rng = np.random.default_rng(seed + 1)
    fine_translations = np.concatenate(
        [np.zeros((1, 2)), rng.uniform(-2.0, 2.0, (N_FINE_TRANS - 1, 2))]
    )
    translation_angles = _relion_cuda_score_translation_angles_if_available(
        fine_translations, image_shape, enabled=True, dtype=np.float32
    )
    assert translation_angles is not None and translation_angles.dtype == np.float32

    compact_indices = (
        window_indices if use_window else np.arange(n_half, dtype=np.int32)
    )
    full_to_compact = spb._relion_cuda_fine_full_to_compact_lookup(
        image_shape, current_size, compact_indices
    )
    half_weights = make_scoring_half_image_weights(
        image_shape, relion_half_sum=True, exclude_relion_redundant_x0=True
    )
    if use_window:
        half_weights = half_weights[jnp.asarray(window_indices, dtype=jnp.int32)]

    kwargs = _bucket_io_kwargs(
        dataset,
        current_size=current_size,
        window_indices=window_indices,
        translation_angles=translation_angles,
        fine_translations=fine_translations,
    )
    translation_prior = rng.normal(scale=0.05, size=(N_IMAGES, N_FINE_TRANS)).astype(np.float32)
    operands = prepare_resident_image_operands(
        dataset,
        np.arange(N_IMAGES),
        bucket_io_kwargs=kwargs,
        half_weights=half_weights,
        full_to_compact=full_to_compact,
        translation_angles=translation_angles,
        window_indices=window_indices,
        windowed_prepare=False,
        image_shape=image_shape,
        current_size=current_size,
        n_fine_trans=N_FINE_TRANS,
        use_exact_relion_gaussian=True,
        accumulate_noise=False,
        source_faithful_spectrum_norm=False,
        fine_translation_prior_2d=translation_prior,
        image_batch_size=4,
    )
    n_pixels = operands.n_score_pixels

    projection_cache = (
        rng.normal(0.0, 0.03, (n_fine_rot, n_pixels))
        + 1j * rng.normal(0.0, 0.03, (n_fine_rot, n_pixels))
    ).astype(np.complex64)

    chunks = plan_capacity_chunks(
        tables,
        row_capacity_ladder=row_capacity_ladder,
        image_capacity_ladder=image_capacity_ladder,
    )
    return dict(
        dataset=dataset,
        image_shape=image_shape,
        current_size=current_size,
        window_indices=window_indices,
        bucket_io_kwargs=kwargs,
        per_image_inputs=per_image_inputs,
        tables=tables,
        chunks=chunks,
        operands=operands,
        projection_cache=jnp.asarray(projection_cache),
        half_weights=half_weights,
        full_to_compact=full_to_compact,
        translation_angles=translation_angles,
    )


def _compact_reference_for_image(case, image):
    """Raw diff2 of one image from both compact-engine routes.

    Returns ``(fused_pairs, rectangular, candidate_mask)``, each
    ``[n_rows_i, n_fine_trans]`` (the mask as bool).
    """

    case_tables = case["tables"]
    start = int(case_tables.row_offsets[image])
    stop = int(case_tables.row_offsets[image + 1])
    n_rows_i = stop - start
    rot_ids = np.asarray(case_tables.row_fine_rot[start:stop], dtype=np.int32)
    mask = _candidate_mask_to_dense(case["per_image_inputs"]["candidate_mask"][image])
    assert mask.shape == (n_rows_i, N_FINE_TRANS)

    dataset = case["dataset"]
    batch_data, ctf_params, fetched = fetch_indexed_batch(dataset, np.asarray([image]))
    prepared = _prepare_bucket_io(
        dataset,
        jnp.asarray(batch_data),
        ctf_params,
        np.asarray(fetched),
        return_direct_scoring_io=True,
        **case["bucket_io_kwargs"],
    )
    ctf2_over_nv_half = prepared[3]
    processed_score_half_for_noise = prepared[6]
    shifted_corrected_score_half = prepared[7]
    direct_score_input = prepared[8]

    window_indices = case["window_indices"]
    if window_indices is not None:
        gather = jnp.asarray(window_indices, dtype=jnp.int32)
        corr = ctf2_over_nv_half[:, gather]
        unshifted = direct_score_input[:, gather]
        shifted = shifted_corrected_score_half[:, gather]
    else:
        corr = ctf2_over_nv_half
        unshifted = direct_score_input
        shifted = shifted_corrected_score_half
    highres_xi2_half, _ = spb._relion_powerclass_noise_terms(
        processed_score_half_for_noise,
        image_shape=case["image_shape"],
        current_size=case["current_size"],
        use_exact_relion_gaussian=True,
        accumulate_noise=False,
        source_faithful_spectrum_norm=False,
    )

    proj_half = case["projection_cache"][jnp.asarray(rot_ids)][None, :, :]
    shifted_split = shifted.reshape(1, N_FINE_TRANS, -1)

    local_rotation_row = np.repeat(np.arange(n_rows_i, dtype=np.int32), N_FINE_TRANS)[None, :]
    translation_idx = np.tile(np.arange(N_FINE_TRANS, dtype=np.int32), n_rows_i)[None, :]
    pair_mask = mask.reshape(1, -1)
    fused_pairs = spb._score_pass2_pairs_relion_gpu_diff2_raw_fused_translate(
        unshifted,
        corr,
        proj_half,
        case["half_weights"],
        case["translation_angles"],
        jnp.asarray(local_rotation_row),
        jnp.asarray(translation_idx),
        jnp.asarray(pair_mask),
        case["full_to_compact"],
        highres_xi2_half,
        current_size=int(case["current_size"]),
    )
    rectangular = spb._score_pass2_bucket_relion_gpu_diff2_raw(
        shifted_split,
        corr,
        proj_half,
        case["half_weights"],
        case["full_to_compact"],
        highres_xi2_half,
        use_fused_ffi=True,
        candidate_mask=jnp.asarray(mask[None]),
    )
    return (
        np.asarray(fused_pairs).reshape(n_rows_i, N_FINE_TRANS),
        np.asarray(rectangular)[0],
        mask,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("current_size", [IMAGE_SHAPE[0], 6])
def test_resident_raw_diff2_matches_the_compact_engine_rows(
    monkeypatch, custom_cuda_lib, gpu_device, current_size
):
    """Every valid cell of every image agrees with the compact engine.

    The fused-translate comparison is asserted bitwise (same kernel body, same
    operands, only the row addressing differs); the pre-shifted rectangular
    comparison is measured and reported in ULP, and bounded loosely so a real
    regression still fails here.
    """

    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    with jax.default_device(gpu_device):
        case = _build_case(
            current_size,
            row_capacity_ladder=(16, 64, 256),
            image_capacity_ladder=(2, 4, 8),
        )
        chunks = case["chunks"]
        assert len(chunks) > 1, "fixture must produce more than one chunk"
        results = score_all_chunks(
            case["tables"],
            chunks,
            case["operands"],
            case["projection_cache"],
            FINE_TRANS_PARENT,
        )
        results = jax.block_until_ready(results)

        tables = case["tables"]
        fused_mismatches = 0
        fused_cells = 0
        rect_ulps = []
        rect_exact = 0
        rect_cells = 0
        for chunk, result in zip(chunks, results, strict=True):
            raw = np.asarray(result.raw_diff2)
            scores = np.asarray(result.scores)
            min_diff2 = np.asarray(result.min_diff2)
            assert raw.shape == (chunk.row_capacity, N_FINE_TRANS)
            assert scores.shape == raw.shape
            assert min_diff2.shape == (chunk.image_capacity,)
            # Padded rows carry no candidate at all.
            assert np.all(np.isposinf(raw[chunk.n_valid_rows :]))
            assert np.all(np.isneginf(scores[chunk.n_valid_rows :]))

            for image in range(chunk.image_start, chunk.image_stop):
                start = int(tables.row_offsets[image]) - chunk.row_start
                stop = int(tables.row_offsets[image + 1]) - chunk.row_start
                if stop == start:
                    continue
                got = raw[start:stop]
                fused, rect, mask = _compact_reference_for_image(case, image)
                # The fused-pairs reference masks exactly like this stage, so
                # the whole block (finite cells and +inf cells) must agree.
                fused_cells += int(mask.size)
                fused_mismatches += int(
                    np.count_nonzero(got.view(np.uint32) != fused.view(np.uint32))
                )
                if np.any(mask):
                    rect_cells += int(mask.sum())
                    ulp = _ulp_distance(got[mask], rect[mask])
                    rect_exact += int(np.count_nonzero(ulp == 0))
                    rect_ulps.append(ulp)

                # Per-image common minimum and the score conversion.
                expected_min = (
                    float(np.min(got[mask])) if np.any(mask) else 0.0
                )
                local_image = image - chunk.image_start
                np.testing.assert_array_equal(
                    np.float32(min_diff2[local_image]), np.float32(expected_min)
                )

        max_ulp = int(np.concatenate(rect_ulps).max()) if rect_ulps else 0
        exact_fraction = rect_exact / max(rect_cells, 1)
        print(
            f"[T6 current_size={current_size}] fused-vs-fused cells={fused_cells} "
            f"mismatches={fused_mismatches}; rectangular cells={rect_cells} "
            f"max_ulp={max_ulp} exact_fraction={exact_fraction:.6f}"
        )
        assert fused_mismatches == 0, (
            f"{fused_mismatches}/{fused_cells} cells differ from the compact fused-translate scorer"
        )
        if current_size < IMAGE_SHAPE[0]:
            # A production score window never contains the ky = -cs/2 row, and
            # without it the in-kernel and pre-shift translations agree bitwise.
            assert max_ulp == 0, f"rectangular pre-shifted path is {max_ulp} ULP away"
        else:
            # The full half does contain that row, where the two translate
            # conventions differ by construction (see the focused test below).
            assert max_ulp > 0, (
                "the full half is expected to disagree on the ky = -N/2 row; "
                "if it no longer does, the conventions were reconciled and this "
                "test and the T6 report should be updated"
            )


@pytest.mark.gpu
def test_full_half_gap_against_the_preshifted_path_is_only_the_minus_nyquist_row(
    monkeypatch, custom_cuda_lib, gpu_device
):
    """Attribute the full-half disagreement to one Fourier row, and no other.

    At ``current_size == image_size`` there is no score window, so the summed
    support contains the FFTW row ``cs/2``, whose true frequency is
    ``ky = -N/2``. The fused in-kernel translation wraps ``y > cs/2`` and so
    treats that row as ``+N/2`` (which is how RELION represents it, see
    ``make_scoring_half_image_weights``); the pre-shift kernel computes
    ``y = index // half_width - N/2`` on the centred lattice and treats it as
    ``-N/2``. Dropping that row from the shared full-to-compact lookup must
    collapse the difference to nothing, which shows the resident stage
    introduces no other arithmetic change.
    """

    import dataclasses

    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    with jax.default_device(gpu_device):
        case = _build_case(
            IMAGE_SHAPE[0],
            row_capacity_ladder=(16, 64, 256),
            image_capacity_ladder=(2, 4, 8),
        )
        half_width = IMAGE_SHAPE[1] // 2 + 1
        lookup = np.asarray(case["full_to_compact"]).copy()
        fftw_row = np.arange(lookup.size) // half_width
        dropped = lookup.copy()
        dropped[fftw_row == IMAGE_SHAPE[0] // 2] = -1
        assert np.count_nonzero(dropped != lookup) == half_width

        lookup_device = jnp.asarray(dropped, dtype=jnp.int32)
        case["full_to_compact"] = lookup_device
        case["operands"] = dataclasses.replace(
            case["operands"], full_to_compact=lookup_device
        )
        results = jax.block_until_ready(
            score_all_chunks(
                case["tables"],
                case["chunks"],
                case["operands"],
                case["projection_cache"],
                FINE_TRANS_PARENT,
            )
        )
        tables = case["tables"]
        cells = 0
        worst = 0
        for chunk, result in zip(case["chunks"], results, strict=True):
            raw = np.asarray(result.raw_diff2)
            for image in range(chunk.image_start, chunk.image_stop):
                start = int(tables.row_offsets[image]) - chunk.row_start
                stop = int(tables.row_offsets[image + 1]) - chunk.row_start
                if stop == start:
                    continue
                _fused, rect, mask = _compact_reference_for_image(case, image)
                if not mask.any():
                    continue
                ulp = _ulp_distance(raw[start:stop][mask], rect[mask])
                cells += int(ulp.size)
                worst = max(worst, int(ulp.max()))
        print(f"[T6 full half without the ky=-N/2 row] cells={cells} max_ulp={worst}")
        assert cells > 0
        assert worst == 0, (
            f"dropping the ky=-N/2 row leaves {worst} ULP; the resident stage differs "
            "from the pre-shifted path for some other reason too"
        )


@pytest.mark.gpu
def test_resident_scores_match_the_relion_conversion_on_the_same_raw_costs(
    monkeypatch, custom_cuda_lib, gpu_device
):
    """``scores`` equals ``_relion_cuda_fine_diff2_to_scores`` on this stage's raw costs."""

    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    with jax.default_device(gpu_device):
        case = _build_case(
            IMAGE_SHAPE[0],
            row_capacity_ladder=(16, 64, 256),
            image_capacity_ladder=(2, 4, 8),
        )
        tables = case["tables"]
        results = jax.block_until_ready(
            score_all_chunks(
                tables,
                case["chunks"],
                case["operands"],
                case["projection_cache"],
                FINE_TRANS_PARENT,
            )
        )
        translation_prior = np.asarray(case["operands"].translation_prior)
        for chunk, result in zip(case["chunks"], results, strict=True):
            raw = np.asarray(result.raw_diff2)
            got_scores = np.asarray(result.scores)
            for image in range(chunk.image_start, chunk.image_stop):
                start = int(tables.row_offsets[image]) - chunk.row_start
                stop = int(tables.row_offsets[image + 1]) - chunk.row_start
                if stop == start:
                    continue
                mask = _candidate_mask_to_dense(
                    case["per_image_inputs"]["candidate_mask"][image]
                )
                rows_raw = raw[start:stop]
                row_prior = np.asarray(
                    case["per_image_inputs"]["log_prior"][image], dtype=np.float32
                )
                expected = spb._relion_cuda_fine_diff2_to_scores(
                    jnp.asarray(rows_raw)[None],
                    jnp.asarray(row_prior)[None, :, None],
                    jnp.asarray(translation_prior[image])[None, None, :],
                    jnp.asarray(mask)[None],
                )
                np.testing.assert_array_equal(
                    np.asarray(expected)[0], got_scores[start:stop]
                )


@pytest.mark.gpu
def test_resident_program_compiles_once_per_capacity_class(
    monkeypatch, custom_cuda_lib, gpu_device
):
    """Chunks of one capacity class reuse one compiled program.

    Occupancy (``n_valid_rows``/``n_valid_images``) and the RELION current size
    are runtime operands, so the only compile keys are the capacity class and
    the pixel/translation counts.
    """

    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    with jax.default_device(gpu_device):
        case = _build_case(
            IMAGE_SHAPE[0],
            row_capacity_ladder=(256,),
            image_capacity_ladder=(2,),
        )
        chunks = case["chunks"]
        classes = {(chunk.row_capacity, chunk.image_capacity) for chunk in chunks}
        assert len(chunks) >= 3 and len(classes) == 1, (
            f"fixture must give several chunks of one class, got {len(chunks)} chunks in {classes}"
        )
        chunk_arrays = [materialize_chunk_device(case["tables"], chunk) for chunk in chunks]

        score_resident_chunk._clear_cache()
        for chunk, arrays in zip(chunks, chunk_arrays, strict=True):
            jax.block_until_ready(
                score_resident_chunk(
                    arrays["row_image_local"],
                    arrays["row_fine_rot"],
                    arrays["row_log_prior"],
                    arrays["row_mask_bits"],
                    arrays["row_mask_mode"],
                    arrays["n_valid_rows"],
                    arrays["image_ids"],
                    case["projection_cache"],
                    case["operands"].score_input,
                    case["operands"].corr_img_score,
                    case["operands"].highres_xi2_half,
                    case["operands"].translation_prior,
                    half_weights=case["operands"].half_weights,
                    translation_angles=case["operands"].translation_angles,
                    full_to_compact=case["operands"].full_to_compact,
                    fine_translation_parent=jnp.asarray(FINE_TRANS_PARENT, dtype=jnp.int32),
                    logical_current_size=jnp.asarray(case["current_size"], dtype=jnp.int32),
                    row_capacity=int(chunk.row_capacity),
                    image_capacity=int(chunk.image_capacity),
                    n_fine_trans=N_FINE_TRANS,
                    n_score_pixels=int(case["operands"].n_score_pixels),
                )
            )
        assert score_resident_chunk._cache_size() == 1, (
            f"expected one compiled program, got {score_resident_chunk._cache_size()}"
        )


@pytest.mark.gpu
def test_padded_image_slots_and_empty_images_do_not_contribute(
    monkeypatch, custom_cuda_lib, gpu_device
):
    """An all-empty candidate image yields +inf rows, -inf scores and min 0."""

    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    with jax.default_device(gpu_device):
        case = _build_case(
            IMAGE_SHAPE[0],
            row_capacity_ladder=(16, 64, 256),
            image_capacity_ladder=(2, 4, 8),
        )
        tables = case["tables"]
        results = jax.block_until_ready(
            score_all_chunks(
                tables,
                case["chunks"],
                case["operands"],
                case["projection_cache"],
                FINE_TRANS_PARENT,
            )
        )
        empty_image = 1  # the "empty" candidate-mask mode of the fixture
        for chunk, result in zip(case["chunks"], results, strict=True):
            if not (chunk.image_start <= empty_image < chunk.image_stop):
                continue
            start = int(tables.row_offsets[empty_image]) - chunk.row_start
            stop = int(tables.row_offsets[empty_image + 1]) - chunk.row_start
            raw = np.asarray(result.raw_diff2)[start:stop]
            scores = np.asarray(result.scores)[start:stop]
            assert np.all(np.isposinf(raw))
            assert np.all(np.isneginf(scores))
            assert float(np.asarray(result.min_diff2)[empty_image - chunk.image_start]) == 0.0
            break
        else:  # pragma: no cover - the fixture always places image 1 in a chunk
            pytest.fail("the empty-mask image was not found in any chunk")

        # Padded image slots never carry a valid row.
        for chunk, result in zip(case["chunks"], results, strict=True):
            host = materialize_chunk(tables, chunk)
            padded = np.asarray(host["image_ids"]) < 0
            if np.any(padded):
                assert np.all(np.asarray(result.min_diff2)[padded] == 0.0)


@pytest.mark.gpu
@pytest.mark.parametrize("current_size", [IMAGE_SHAPE[0], 6])
def test_resident_operands_match_a_single_image_prepare_bitwise(
    monkeypatch, custom_cuda_lib, gpu_device, current_size
):
    """Batching ``_prepare_bucket_io`` over images does not change its result.

    ``prepare_resident_image_operands`` runs the production owner in fixed
    image batches; every operand it keeps must equal what the same owner
    returns for that image alone, bit for bit, or the resident stage would not
    be scoring the compact engine's operands.
    """

    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)

    with jax.default_device(gpu_device):
        case = _build_case(
            current_size,
            row_capacity_ladder=(16, 64, 256),
            image_capacity_ladder=(2, 4, 8),
        )
        operands = case["operands"]
        window_indices = case["window_indices"]
        for image in range(N_IMAGES):
            batch_data, ctf_params, fetched = fetch_indexed_batch(
                case["dataset"], np.asarray([image])
            )
            prepared = _prepare_bucket_io(
                case["dataset"],
                jnp.asarray(batch_data),
                ctf_params,
                np.asarray(fetched),
                return_direct_scoring_io=True,
                **case["bucket_io_kwargs"],
            )
            ctf2_over_nv_half = prepared[3]
            processed_score_half_for_noise = prepared[6]
            direct_score_input = prepared[8]
            if window_indices is not None:
                gather = jnp.asarray(window_indices, dtype=jnp.int32)
                expected_input = direct_score_input[:, gather]
                expected_corr = ctf2_over_nv_half[:, gather]
            else:
                expected_input = direct_score_input
                expected_corr = ctf2_over_nv_half
            expected_xi2, _ = spb._relion_powerclass_noise_terms(
                processed_score_half_for_noise,
                image_shape=case["image_shape"],
                current_size=current_size,
                use_exact_relion_gaussian=True,
                accumulate_noise=False,
                source_faithful_spectrum_norm=False,
            )
            np.testing.assert_array_equal(
                np.asarray(operands.score_input[image]), np.asarray(expected_input)[0]
            )
            np.testing.assert_array_equal(
                np.asarray(operands.corr_img_score[image]), np.asarray(expected_corr)[0]
            )
            np.testing.assert_array_equal(
                np.asarray(operands.highres_xi2_half[image]),
                np.asarray(expected_xi2).reshape(-1)[0],
            )

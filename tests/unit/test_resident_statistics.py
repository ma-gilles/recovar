"""Tests for the device-resident pass-2 statistics stage (T9a, CPU only).

Ticket: em_parity_tickets_20260918/T9a_device_statistics_stage.md.
Design: em_device_resident_pass2_design_20260918.md (stage 7).

The reference in every end-to-end case is the production host tail of
``recovar.em.sparse_pass2.sparse_pass2_bucketed.compute_pass2_stats_sparse_bucketed``
(``_bucket_tail``) transcribed statement by statement into
:func:`_host_tail_reference` below, driven on the same operands. Run with
``JAX_PLATFORMS=cpu RECOVAR_DISABLE_CUDA=1``.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.helpers.projection import (
    compute_norm_residual_per_image,
    compute_scale_correction_terms_per_image,
)
from recovar.em.local.local_backprojection import flatten_bucket_rows
from recovar.em.sparse_pass2.resident_statistics import (
    ChunkStatisticsOperands,
    ResidentStatisticsTables,
    accumulate_chunk_statistics,
    finalize_statistics,
    make_resident_statistics,
    resolve_statistics_config,
    segment_sum_by_image,
)
from recovar.em.sparse_pass2.sparse_pass2_noise_blocks import _compute_noise_block_chunked
from recovar.em.sparse_pass2.sparse_pass2_wavg import (
    _replace_low_shell_noise_with_relion_wavg_direct_residual,
    _replace_low_shell_noise_with_relion_wavg_direct_residual_jnp,
    _weighted_image_power_shells_and_per_image,
)

pytestmark = pytest.mark.unit


# --- Fixture shapes ---------------------------------------------------------
# Small but structurally realistic: several images per chunk, a ragged number
# of candidate rows per image, more than one scale group, and a Wavg rectangle
# strictly larger than the reconstruction window (exact_positions is a subset).
N_IMAGES = 7
N_SHELLS = 6
N_FINE_TRANS = 4
N_PIX = 11
N_PIX_HALF = 13
N_RECT = 15
N_COARSE_ROT = 9
N_GROUPS = 3
CURRENT_SIZE = 8  # -> norm cutoff 4, direct-noise exclusive shell stop 5
ROWS_PER_IMAGE = (3, 1, 4, 2, 2, 5, 1)


def _rng():
    return np.random.default_rng(20260918)


def _make_tables(rng, *, with_sqdist=True, float_dtype=np.float32):
    shell_indices_noise = rng.integers(-1, N_SHELLS + 1, size=N_PIX).astype(np.int32)
    shell_indices_half = rng.integers(-1, N_SHELLS + 1, size=N_PIX_HALF).astype(np.int32)
    wavg_shells = rng.integers(-1, N_SHELLS, size=N_RECT).astype(np.int32)
    exact_positions = np.sort(rng.choice(N_RECT, size=N_PIX, replace=False)).astype(np.int32)
    scale_pixel_mask = rng.random(N_PIX) > 0.3
    wavg_scale_mask = np.zeros(N_RECT, dtype=bool)
    wavg_scale_mask[exact_positions] = scale_pixel_mask
    return {
        "noise_variance": rng.random(N_PIX).astype(float_dtype) + 0.5,
        "shell_indices_noise": shell_indices_noise,
        "shell_indices_half": shell_indices_half,
        "wavg_shell_indices": wavg_shells,
        "wavg_scale_pixel_mask": wavg_scale_mask,
        "scale_pixel_mask": scale_pixel_mask,
        "translation_sqdist_ang": (
            rng.random(N_FINE_TRANS).astype(float_dtype) * 4.0 if with_sqdist else None
        ),
        "exact_positions": exact_positions,
    }


def _make_bucket(rng, image_indices, n_rot, tables, *, zero_posterior=False, float_dtype=np.float32):
    """One rectangular (B, R, T) bucket exactly as the host loop builds it.

    ``n_rot`` is the padded rotation count; rows beyond an image's own
    ``actual_count`` carry zero posterior, which is the host's padding rule and
    the resident row-padding rule alike. ``float_dtype`` selects the production
    float32 scoring stream or the float64 diagnostic companion; the Wavg
    triplet stays float32 either way because RELION's atomic stream is float32.
    """

    complex_dtype = np.complex64 if float_dtype == np.float32 else np.complex128
    batch = len(image_indices)
    actual_counts = np.asarray([ROWS_PER_IMAGE[i] for i in image_indices], dtype=np.int32)
    assert int(actual_counts.max()) <= n_rot

    probs = rng.random((batch, n_rot, N_FINE_TRANS)).astype(float_dtype)
    # RELION's fine M-step prune zeroes most cells; keep the sparsity realistic.
    probs = np.where(rng.random(probs.shape) < 0.45, probs, float_dtype(0.0))
    for row, count in enumerate(actual_counts):
        probs[row, count:, :] = 0.0
    if zero_posterior:
        probs[:] = 0.0

    proj = (rng.random((batch, n_rot, N_PIX)) + 1j * rng.random((batch, n_rot, N_PIX))).astype(
        complex_dtype
    )
    proj_abs2 = (np.abs(proj) ** 2).astype(float_dtype)
    ctf_probs = rng.random((batch, n_rot, N_PIX)).astype(float_dtype)
    ctf_probs = np.where(rng.random(ctf_probs.shape) < 0.8, ctf_probs, float_dtype(0.0))
    summed = (
        rng.random((batch, n_rot, N_PIX)) + 1j * rng.random((batch, n_rot, N_PIX))
    ).astype(complex_dtype)
    # Rows with no posterior mass carry no weighted sums and no CTF mass, which
    # is what neutralises both host rotation padding and resident row padding.
    dead_rows = probs.sum(axis=-1) == 0.0
    ctf_probs[dead_rows] = 0.0
    summed[dead_rows] = 0.0

    triplet = rng.standard_normal((batch, N_RECT, 3)).astype(np.float32)
    triplet[np.asarray(probs.sum(axis=(1, 2)) == 0.0)] = 0.0

    return {
        "image_indices": np.asarray(image_indices, dtype=np.int64),
        "actual_counts": actual_counts,
        "probs": probs,
        "proj": proj,
        "proj_abs2": proj_abs2,
        "ctf_probs": ctf_probs,
        "summed": summed,
        "triplet": triplet,
        "processed_half": (
            rng.standard_normal((batch, N_PIX_HALF)) + 1j * rng.standard_normal((batch, N_PIX_HALF))
        ).astype(complex_dtype),
        "relion_norm_high_shell": rng.random(batch).astype(np.float64),
        "scale": (rng.random(batch).astype(float_dtype) + 0.5),
        "group_ids": np.asarray([i % N_GROUPS for i in image_indices], dtype=np.int32),
        "min_diff2": rng.random(batch).astype(float_dtype) * 100.0,
        "class_log_z": rng.random(batch).astype(np.float64) * 10.0,
        "best_log_score": rng.random(batch).astype(np.float64) * 10.0,
        "max_posterior": rng.random(batch).astype(float_dtype),
        "best_rot": np.asarray(
            [rng.integers(0, max(1, c)) for c in actual_counts], dtype=np.int32
        ),
        "best_trans": rng.integers(0, N_FINE_TRANS, size=batch).astype(np.int32),
        "coarse_rot": rng.integers(0, N_COARSE_ROT, size=(batch, n_rot)).astype(np.int32),
        "fine_rot": rng.integers(0, 1000, size=(batch, n_rot)).astype(np.int32),
    }


# --- Host reference ---------------------------------------------------------


def _host_tail_reference(buckets, tables, *, atomic_scale=True, direct_noise=True):
    """The production ``_bucket_tail`` statements, transcribed and run in numpy.

    Configuration: ``accumulate_noise=True``, ``use_relion_fine_mstep_prune=True``,
    ``relion_wavg_atomic_scale_aa=True``, ``relion_wavg_atomic_direct_noise=True``,
    ``relion_wavg_atomic_direct_norm=False``, ``translated_wavg_norm=False``,
    ``use_exact_relion_gaussian=True``, ``include_unweighted_norm_high_shell=True``,
    ``source_faithful_spectrum_norm=True``.
    """

    noise_wsum_total = np.zeros(N_SHELLS, dtype=np.float64)
    noise_img_power_total = np.zeros(N_SHELLS, dtype=np.float64)
    noise_norm_correction_total = np.zeros(N_IMAGES, dtype=np.float64)
    noise_scale_xa_total = np.zeros(N_GROUPS, dtype=np.float64)
    noise_scale_aa_total = np.zeros(N_GROUPS, dtype=np.float64)
    rotation_posterior_sums = np.zeros(N_COARSE_ROT, dtype=np.float64)
    noise_sumw_total = 0.0
    noise_sigma2_offset_total = 0.0
    log_evidence = np.full(N_IMAGES, np.nan, dtype=np.float64)
    score_log_z = np.full(N_IMAGES, np.nan, dtype=np.float64)
    best_log_score = np.full(N_IMAGES, np.nan, dtype=np.float64)
    score_real_dtype = buckets[0]["max_posterior"].dtype
    max_posterior = np.full(N_IMAGES, np.nan, dtype=score_real_dtype)
    hard_assignment = np.full(N_IMAGES, -1, dtype=np.int32)
    best_rotation_indices = np.full(N_IMAGES, -1, dtype=np.int64)

    noise_variance = jnp.asarray(tables["noise_variance"])
    shell_indices_noise = jnp.asarray(tables["shell_indices_noise"])
    shell_indices_half = jnp.asarray(tables["shell_indices_half"])

    for bucket in buckets:
        noise_probs = jnp.asarray(bucket["probs"])
        image_indices = bucket["image_indices"]

        translation_posterior = np.asarray(jnp.sum(noise_probs, axis=1), dtype=np.float64)
        noise_sigma2_offset_total += float(
            np.sum(translation_posterior * tables["translation_sqdist_ang"], dtype=np.float64)
        )

        support_mass = jnp.sum(noise_probs, axis=(1, 2))
        weighted_img_shells, weighted_img_per_image = _weighted_image_power_shells_and_per_image(
            jnp.asarray(bucket["processed_half"]),
            shell_indices_half,
            support_mass,
            shell_count=N_SHELLS,
            norm_unweighted_shell_cutoff=CURRENT_SIZE // 2,
            norm_unweighted_high_shell=jnp.asarray(bucket["relion_norm_high_shell"]),
            include_unweighted_high_shell=True,
            source_faithful_spectrum_norm=True,
        )
        support_mass_np = np.asarray(support_mass, dtype=np.float64)
        weighted_img_shells_np = np.asarray(weighted_img_shells, dtype=np.float64)
        noise_norm_correction_total[image_indices] += np.asarray(
            weighted_img_per_image, dtype=np.float64
        )
        noise_sumw_total += float(np.sum(support_mass_np, dtype=np.float64))

        block_noise_shells, _, _ = _compute_noise_block_chunked(
            flatten_bucket_rows(jnp.asarray(bucket["proj"])),
            flatten_bucket_rows(jnp.asarray(bucket["proj_abs2"])),
            flatten_bucket_rows(jnp.asarray(bucket["summed"])),
            flatten_bucket_rows(jnp.asarray(bucket["ctf_probs"])),
            noise_variance,
            shell_indices_noise,
            N_SHELLS,
            max_block_bytes=None,
        )
        block_noise_shells_np = np.asarray(block_noise_shells, dtype=np.float64)

        if direct_noise:
            direct_residual_shells, direct_image_power_shells = (
                _replace_low_shell_noise_with_relion_wavg_direct_residual(
                    block_noise_shells_np,
                    weighted_img_shells_np,
                    bucket["triplet"][:, :, 2],
                    tables["wavg_shell_indices"],
                    exclusive_shell_stop=CURRENT_SIZE // 2 + 1,
                )
            )
            noise_wsum_total += direct_residual_shells
            noise_img_power_total += direct_image_power_shells
        else:
            noise_wsum_total += block_noise_shells_np
            noise_img_power_total += weighted_img_shells_np

        block_norm_residual = compute_norm_residual_per_image(
            jnp.asarray(bucket["proj"]),
            jnp.asarray(bucket["proj_abs2"]),
            jnp.asarray(bucket["summed"]),
            jnp.asarray(bucket["ctf_probs"]),
            noise_variance,
        )
        noise_norm_correction_total[image_indices] += np.asarray(
            block_norm_residual, dtype=np.float64
        )

        scale_xa_per_image, scale_aa_per_image = compute_scale_correction_terms_per_image(
            jnp.asarray(bucket["proj"]),
            jnp.asarray(bucket["proj_abs2"]),
            jnp.asarray(bucket["summed"]),
            jnp.asarray(bucket["ctf_probs"]),
            noise_variance,
            jnp.asarray(bucket["scale"]),
            jnp.asarray(tables["scale_pixel_mask"]),
        )
        if atomic_scale:
            scale_pixel_mask_np = np.asarray(tables["wavg_scale_pixel_mask"]).reshape(1, -1)
            scale_xa_per_image = np.sum(
                np.where(scale_pixel_mask_np, bucket["triplet"][:, :, 0], np.float32(0.0)),
                axis=1,
                dtype=np.float64,
            )
            scale_aa_per_image = np.sum(
                np.where(scale_pixel_mask_np, bucket["triplet"][:, :, 1], np.float32(0.0)),
                axis=1,
                dtype=np.float64,
            )
        np.add.at(
            noise_scale_xa_total,
            np.asarray(bucket["group_ids"], dtype=np.int64),
            np.asarray(scale_xa_per_image, dtype=np.float64),
        )
        np.add.at(
            noise_scale_aa_total,
            np.asarray(bucket["group_ids"], dtype=np.int64),
            np.asarray(scale_aa_per_image, dtype=np.float64),
        )

        log_score_offset = -np.asarray(bucket["min_diff2"], dtype=np.float64)
        class_log_Z_np = np.asarray(bucket["class_log_z"], dtype=np.float64)
        best_log_score_np = np.asarray(bucket["best_log_score"], dtype=np.float64)
        max_posterior_np = np.asarray(bucket["max_posterior"], dtype=score_real_dtype)
        for row, image_idx in enumerate(image_indices.tolist()):
            r = int(bucket["best_rot"][row])
            t = int(bucket["best_trans"][row])
            hard_assignment[image_idx] = r * N_FINE_TRANS + t
            best_rotation_indices[image_idx] = bucket["fine_rot"][row, r]
            if np.isfinite(best_log_score_np[row]):
                log_evidence[image_idx] = float(class_log_Z_np[row] + log_score_offset[row])
                score_log_z[image_idx] = float(class_log_Z_np[row] + log_score_offset[row])
            else:
                log_evidence[image_idx] = -np.inf
                score_log_z[image_idx] = -np.inf
            best_log_score[image_idx] = float(best_log_score_np[row] + log_score_offset[row])
            max_posterior[image_idx] = float(max_posterior_np[row])

        probs_sum_t = np.asarray(jnp.sum(noise_probs, axis=-1), dtype=np.float64)
        for row, image_idx in enumerate(image_indices.tolist()):
            cnt = int(bucket["actual_counts"][row])
            if cnt == 0:
                continue
            np.add.at(
                rotation_posterior_sums,
                bucket["coarse_rot"][row, :cnt].astype(np.int64),
                probs_sum_t[row, :cnt],
            )

    return {
        "wsum_sigma2_noise": noise_wsum_total,
        "wsum_img_power": noise_img_power_total,
        "wsum_sigma2_offset": noise_sigma2_offset_total,
        "sumw": noise_sumw_total,
        "wsum_norm_correction": noise_norm_correction_total,
        "wsum_scale_correction_xa": noise_scale_xa_total,
        "wsum_scale_correction_aa": noise_scale_aa_total,
        "rotation_posterior_sums": rotation_posterior_sums,
        "log_evidence_per_image": log_evidence,
        "score_log_z_per_image": score_log_z,
        "best_log_score_per_image": best_log_score,
        "max_posterior_per_image": max_posterior,
        "hard_assignment": hard_assignment,
        "best_fine_rotation_indices": best_rotation_indices,
    }


# --- Rectangular bucket -> flat resident chunk ------------------------------


def _chunk_from_buckets(buckets, *, row_capacity, image_capacity, with_triplet=True):
    """Flatten one or more rectangular buckets into one padded resident chunk.

    Only an image's own ``actual_count`` rows become candidate rows; the
    bucket's rotation padding is dropped here exactly as the resident data
    model never materialises it.
    """

    row_image, row_fine, row_coarse, row_probs = [], [], [], []
    rows_proj, rows_abs2, rows_summed, rows_ctf = [], [], [], []
    image_ids, row_starts, row_counts, group_ids = [], [], [], []
    processed, high_shell, scale, triplet = [], [], [], []
    best_row, best_trans, class_log_z, min_diff2 = [], [], [], []
    best_log_score, max_posterior = [], []

    slot = 0
    for bucket in buckets:
        for row, image_idx in enumerate(bucket["image_indices"].tolist()):
            count = int(bucket["actual_counts"][row])
            row_starts.append(len(row_image))
            row_counts.append(count)
            best_row.append(len(row_image) + int(bucket["best_rot"][row]))
            for r in range(count):
                row_image.append(slot)
                row_fine.append(int(bucket["fine_rot"][row, r]))
                row_coarse.append(int(bucket["coarse_rot"][row, r]))
                row_probs.append(bucket["probs"][row, r])
                rows_proj.append(bucket["proj"][row, r])
                rows_abs2.append(bucket["proj_abs2"][row, r])
                rows_summed.append(bucket["summed"][row, r])
                rows_ctf.append(bucket["ctf_probs"][row, r])
            image_ids.append(int(image_idx))
            group_ids.append(int(bucket["group_ids"][row]))
            processed.append(bucket["processed_half"][row])
            high_shell.append(bucket["relion_norm_high_shell"][row])
            scale.append(bucket["scale"][row])
            triplet.append(bucket["triplet"][row])
            best_trans.append(int(bucket["best_trans"][row]))
            class_log_z.append(bucket["class_log_z"][row])
            min_diff2.append(bucket["min_diff2"][row])
            best_log_score.append(bucket["best_log_score"][row])
            max_posterior.append(bucket["max_posterior"][row])
            slot += 1

    n_valid_rows = len(row_image)
    n_valid_images = slot
    assert n_valid_rows <= row_capacity and n_valid_images <= image_capacity

    def pad_rows(values, dtype, pad_value=0):
        out = np.full((row_capacity,) + np.shape(values[0] if values else 0), pad_value, dtype=dtype)
        if values:
            out[:n_valid_rows] = np.asarray(values, dtype=dtype)
        return out

    def pad_images(values, dtype, pad_value=0):
        out = np.full(
            (image_capacity,) + np.shape(values[0] if values else 0), pad_value, dtype=dtype
        )
        if values:
            out[:n_valid_images] = np.asarray(values, dtype=dtype)
        return out

    float_dtype = buckets[0]["probs"].dtype
    complex_dtype = buckets[0]["proj"].dtype
    operands = ChunkStatisticsOperands(
        # Padded rows point at the last image slot (the T5 materialize_chunk
        # contract) and carry zero posterior / CTF mass / weighted sums.
        row_image_local=jnp.asarray(
            pad_rows(row_image, np.int32, pad_value=image_capacity - 1)
        ),
        row_fine_rot=jnp.asarray(pad_rows(row_fine, np.int32)),
        row_coarse_rot=jnp.asarray(pad_rows(row_coarse, np.int32, pad_value=N_COARSE_ROT)),
        row_posterior=jnp.asarray(pad_rows(row_probs, float_dtype)),
        proj=jnp.asarray(pad_rows(rows_proj, complex_dtype)),
        proj_abs2=jnp.asarray(pad_rows(rows_abs2, float_dtype)),
        summed_masked=jnp.asarray(pad_rows(rows_summed, complex_dtype)),
        ctf_probs=jnp.asarray(pad_rows(rows_ctf, float_dtype)),
        image_ids=jnp.asarray(pad_images(image_ids, np.int32, pad_value=-1)),
        image_row_start=jnp.asarray(pad_images(row_starts, np.int32)),
        image_row_count=jnp.asarray(pad_images(row_counts, np.int32)),
        group_ids=jnp.asarray(pad_images(group_ids, np.int32, pad_value=-1)),
        processed_image_half=jnp.asarray(pad_images(processed, complex_dtype)),
        relion_norm_high_shell=jnp.asarray(pad_images(high_shell, np.float64)),
        scale=jnp.asarray(pad_images(scale, float_dtype)),
        wavg_triplet_pixels=(
            jnp.asarray(pad_images(triplet, np.float32)) if with_triplet else None
        ),
        best_row_local=jnp.asarray(pad_images(best_row, np.int32)),
        best_translation=jnp.asarray(pad_images(best_trans, np.int32)),
        class_log_z=jnp.asarray(pad_images(class_log_z, np.float64)),
        min_diff2=jnp.asarray(pad_images(min_diff2, float_dtype)),
        best_log_score=jnp.asarray(pad_images(best_log_score, np.float64)),
        max_posterior=jnp.asarray(pad_images(max_posterior, float_dtype)),
        batch_norm=None,
    )
    return operands


def _resident_tables(tables, *, with_sqdist=True):
    return ResidentStatisticsTables(
        translation_sqdist_ang=(
            jnp.asarray(tables["translation_sqdist_ang"]) if with_sqdist else None
        ),
        noise_variance=jnp.asarray(tables["noise_variance"]),
        shell_indices_noise=jnp.asarray(tables["shell_indices_noise"]),
        shell_indices_half=jnp.asarray(tables["shell_indices_half"]),
        wavg_shell_indices=jnp.asarray(tables["wavg_shell_indices"]),
        wavg_scale_pixel_mask=jnp.asarray(tables["wavg_scale_pixel_mask"]),
        scale_pixel_mask=jnp.asarray(tables["scale_pixel_mask"]),
    )


def _config(*, atomic_scale=True, direct_noise=True):
    return resolve_statistics_config(
        n_shells=N_SHELLS,
        n_fine_trans=N_FINE_TRANS,
        n_images=N_IMAGES,
        n_coarse_rot=N_COARSE_ROT,
        n_scale_groups=N_GROUPS,
        current_size=CURRENT_SIZE,
        include_unweighted_high_shell=True,
        use_exact_relion_gaussian=True,
        relion_wavg_atomic_direct_noise=direct_noise,
        relion_wavg_atomic_scale_aa=atomic_scale,
        accumulate_scale=True,
        source_faithful_spectrum_norm=True,
    )


# --- Helper twins -----------------------------------------------------------


def test_direct_residual_twin_is_bitwise_when_order_cannot_matter():
    """One summand per shell: the twin must match the numpy original bitwise."""

    rng = _rng()
    # One image and distinct shells per pixel -> every shell receives at most
    # one float64 summand, so association order is not a degree of freedom.
    shells = np.asarray([0, 1, 2, 3, 4, 5, -1, 7], dtype=np.int32)
    atomic = rng.standard_normal((1, shells.size)).astype(np.float32)
    residual = rng.standard_normal(N_SHELLS).astype(np.float64)
    image_power = rng.standard_normal(N_SHELLS).astype(np.float64)

    expected = _replace_low_shell_noise_with_relion_wavg_direct_residual(
        residual, image_power, atomic, shells, exclusive_shell_stop=5
    )
    got = _replace_low_shell_noise_with_relion_wavg_direct_residual_jnp(
        jnp.asarray(residual),
        jnp.asarray(image_power),
        jnp.asarray(atomic),
        jnp.asarray(shells),
        exclusive_shell_stop=5,
        shell_count=N_SHELLS,
    )
    for expected_part, got_part in zip(expected, got):
        assert np.array_equal(expected_part, np.asarray(got_part))


@pytest.mark.parametrize("shell_stop", [0, 3, 5, N_SHELLS, N_SHELLS + 4])
def test_direct_residual_twin_matches_numpy_original(shell_stop):
    """Many images and repeated shells: float64 association order only."""

    rng = np.random.default_rng(7 + shell_stop)
    shells = rng.integers(-1, N_SHELLS + 2, size=N_RECT).astype(np.int32)
    atomic = rng.standard_normal((5, N_RECT)).astype(np.float32)
    residual = rng.standard_normal(N_SHELLS).astype(np.float64)
    image_power = rng.standard_normal(N_SHELLS).astype(np.float64)

    expected_residual, expected_power = (
        _replace_low_shell_noise_with_relion_wavg_direct_residual(
            residual, image_power, atomic, shells, exclusive_shell_stop=shell_stop
        )
    )
    got_residual, got_power = _replace_low_shell_noise_with_relion_wavg_direct_residual_jnp(
        jnp.asarray(residual),
        jnp.asarray(image_power),
        jnp.asarray(atomic),
        jnp.asarray(shells),
        exclusive_shell_stop=shell_stop,
        shell_count=N_SHELLS,
    )
    np.testing.assert_allclose(np.asarray(got_residual), expected_residual, rtol=1e-14, atol=0.0)
    # The replaced image-power entries are exact zeros, so this half is bitwise.
    assert np.array_equal(np.asarray(got_power), expected_power)


def test_segment_sum_by_image_matches_numpy_add_at():
    """The row -> image reduction is ``np.add.at`` with the row axis first."""

    rng = _rng()
    row_image = rng.integers(0, 4, size=17).astype(np.int32)
    values = rng.standard_normal((17, 3)).astype(np.float64)
    expected = np.zeros((4, 3), dtype=np.float64)
    np.add.at(expected, row_image, values)
    got = np.asarray(segment_sum_by_image(jnp.asarray(values), jnp.asarray(row_image), 4))
    np.testing.assert_allclose(got, expected, rtol=1e-14, atol=0.0)


def test_flat_row_norm_residual_matches_host_per_image_helper():
    """The flat-row A2/XA partials reproduce ``compute_norm_residual_per_image``."""

    from recovar.em.sparse_pass2.resident_statistics import _flat_row_norm_and_scale_terms

    rng = _rng()
    batch, n_rot = 3, 4
    proj = (rng.standard_normal((batch, n_rot, N_PIX)) + 1j * rng.standard_normal((batch, n_rot, N_PIX))).astype(np.complex64)
    proj_abs2 = (np.abs(proj) ** 2).astype(np.float32)
    summed = (rng.standard_normal((batch, n_rot, N_PIX)) + 1j * rng.standard_normal((batch, n_rot, N_PIX))).astype(np.complex64)
    ctf = rng.random((batch, n_rot, N_PIX)).astype(np.float32)
    nv = rng.random(N_PIX).astype(np.float32) + 0.5

    expected = np.asarray(
        compute_norm_residual_per_image(
            jnp.asarray(proj), jnp.asarray(proj_abs2), jnp.asarray(summed), jnp.asarray(ctf), jnp.asarray(nv)
        ),
        dtype=np.float64,
    )
    row_image = np.repeat(np.arange(batch, dtype=np.int32), n_rot)
    a2_row, xa_row = _flat_row_norm_and_scale_terms(
        jnp.asarray(proj.reshape(batch * n_rot, N_PIX)),
        jnp.asarray(proj_abs2.reshape(batch * n_rot, N_PIX)),
        jnp.asarray(summed.reshape(batch * n_rot, N_PIX)),
        jnp.asarray(ctf.reshape(batch * n_rot, N_PIX)),
        jnp.asarray(nv),
    )
    a2 = np.asarray(segment_sum_by_image(a2_row, jnp.asarray(row_image), batch), dtype=np.float64)
    xa = np.asarray(segment_sum_by_image(xa_row, jnp.asarray(row_image), batch), dtype=np.float64)
    np.testing.assert_allclose(a2 - 2.0 * xa, expected, rtol=1e-6, atol=0.0)


def test_flat_row_scale_terms_match_host_scale_helper():
    """The masked flat-row partials reproduce the algebraic scale statistics."""

    from recovar.em.sparse_pass2.resident_statistics import _flat_row_norm_and_scale_terms

    rng = _rng()
    batch, n_rot = 3, 4
    proj = (rng.standard_normal((batch, n_rot, N_PIX)) + 1j * rng.standard_normal((batch, n_rot, N_PIX))).astype(np.complex64)
    proj_abs2 = (np.abs(proj) ** 2).astype(np.float32)
    summed = (rng.standard_normal((batch, n_rot, N_PIX)) + 1j * rng.standard_normal((batch, n_rot, N_PIX))).astype(np.complex64)
    ctf = rng.random((batch, n_rot, N_PIX)).astype(np.float32)
    nv = rng.random(N_PIX).astype(np.float32) + 0.5
    scale = rng.random(batch).astype(np.float32) + 0.5
    mask = rng.random(N_PIX) > 0.3

    expected_xa, expected_aa = compute_scale_correction_terms_per_image(
        jnp.asarray(proj), jnp.asarray(proj_abs2), jnp.asarray(summed), jnp.asarray(ctf),
        jnp.asarray(nv), jnp.asarray(scale), jnp.asarray(mask),
    )
    row_image = np.repeat(np.arange(batch, dtype=np.int32), n_rot)
    a2_row, xa_row = _flat_row_norm_and_scale_terms(
        jnp.asarray(proj.reshape(batch * n_rot, N_PIX)),
        jnp.asarray(proj_abs2.reshape(batch * n_rot, N_PIX)),
        jnp.asarray(summed.reshape(batch * n_rot, N_PIX)),
        jnp.asarray(ctf.reshape(batch * n_rot, N_PIX)),
        jnp.asarray(nv),
        pixel_mask=jnp.asarray(mask),
    )
    a2 = segment_sum_by_image(a2_row, jnp.asarray(row_image), batch)
    xa = segment_sum_by_image(xa_row, jnp.asarray(row_image), batch)
    safe = np.maximum(scale.astype(np.float64), 1e-30)
    np.testing.assert_allclose(
        np.asarray(xa, dtype=np.float64) / safe, np.asarray(expected_xa, dtype=np.float64),
        rtol=1e-6, atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(a2, dtype=np.float64) / safe**2, np.asarray(expected_aa, dtype=np.float64),
        rtol=1e-6, atol=0.0,
    )


# --- End-to-end -------------------------------------------------------------


def _run_device(buckets_per_chunk, tables, config, *, row_capacity, image_capacity):
    float_dtype = buckets_per_chunk[0][0]["probs"].dtype
    stats = make_resident_statistics(config, max_posterior_dtype=float_dtype)
    resident_tables = _resident_tables(tables)
    for chunk_buckets in buckets_per_chunk:
        operands = _chunk_from_buckets(
            chunk_buckets, row_capacity=row_capacity, image_capacity=image_capacity
        )
        stats = accumulate_chunk_statistics(stats, operands, resident_tables, config=config)
    return finalize_statistics(stats, config=config)


# Per-accumulator tolerance table. A value is float32-reduction limited when
# its producer stream is float32 (the production scoring/projection dtype) and
# the device stage reassociates that reduction: the same float32 summands, a
# different association order. Everything the device stage keeps in float64,
# and everything it computes element-wise, is held to float64 rounding or to
# bitwise equality. ``test_..._float64_companion`` reruns the identical
# comparison with float64 producers and tightens every entry to 1e-13.
FLOAT32_STREAM_RTOL = 1e-6
FLOAT64_STREAM_RTOL = 1e-13

_REDUCTION_ACCUMULATORS = (
    "wsum_sigma2_noise",
    "wsum_img_power",
    "wsum_norm_correction",
    "wsum_scale_correction_xa",
    "wsum_scale_correction_aa",
    "rotation_posterior_sums",
)
_SCALAR_ACCUMULATORS = ("wsum_sigma2_offset", "sumw")
# Element-wise float64 arithmetic on float64 operands: no reduction, so the
# device stage must reproduce the host bit for bit.
_BITWISE_FLOAT_FIELDS = (
    "log_evidence_per_image",
    "best_log_score_per_image",
    "score_log_z_per_image",
    "max_posterior_per_image",
)
_BITWISE_INT_FIELDS = ("hard_assignment", "best_fine_rotation_indices")


def _max_rel(got, expected):
    got = np.asarray(got, dtype=np.float64).reshape(-1)
    expected = np.asarray(expected, dtype=np.float64).reshape(-1)
    scale = np.where(expected == 0.0, 1.0, np.abs(expected))
    return float(np.max(np.abs(got - expected) / scale)) if got.size else 0.0


def _assert_finalized_matches_host(got, expected, *, rtol, label=""):
    """Compare every accumulator and log the observed relative deviation."""

    observed = {}
    for name in _REDUCTION_ACCUMULATORS + _SCALAR_ACCUMULATORS:
        observed[name] = _max_rel(getattr(got, name), expected[name])
        np.testing.assert_allclose(
            np.asarray(getattr(got, name), dtype=np.float64),
            np.asarray(expected[name], dtype=np.float64),
            rtol=rtol,
            atol=0.0,
            err_msg=f"{label}accumulator {name} outside {rtol} relative",
        )
    for name in _BITWISE_FLOAT_FIELDS:
        assert np.array_equal(
            np.asarray(getattr(got, name)), np.asarray(expected[name])
        ), f"{label}{name} is not bitwise equal to the host tail"
        observed[name] = 0.0
    for name in _BITWISE_INT_FIELDS:
        assert np.array_equal(
            np.asarray(getattr(got, name)), np.asarray(expected[name])
        ), f"{label}{name} is not bitwise equal to the host tail"
    logging.info("T9a %stolerances: %s", label, observed)
    return observed


def test_device_stage_matches_host_tail_single_chunk():
    """One chunk holding every image reproduces the whole production host tail."""

    rng = _rng()
    tables = _make_tables(rng)
    bucket = _make_bucket(rng, list(range(N_IMAGES)), n_rot=max(ROWS_PER_IMAGE), tables=tables)
    expected = _host_tail_reference([bucket], tables)
    got = _run_device([[bucket]], tables, _config(), row_capacity=64, image_capacity=8)
    _assert_finalized_matches_host(got, expected, rtol=FLOAT32_STREAM_RTOL, label="f32 single ")


def test_device_stage_matches_host_tail_single_chunk_float64_companion():
    """Float64 companion: with float64 producers every entry tightens to 1e-13."""

    rng = _rng()
    tables = _make_tables(rng, float_dtype=np.float64)
    bucket = _make_bucket(
        rng, list(range(N_IMAGES)), n_rot=max(ROWS_PER_IMAGE), tables=tables, float_dtype=np.float64
    )
    expected = _host_tail_reference([bucket], tables)
    got = _run_device([[bucket]], tables, _config(), row_capacity=64, image_capacity=8)
    _assert_finalized_matches_host(got, expected, rtol=FLOAT64_STREAM_RTOL, label="f64 single ")


def test_device_stage_matches_host_tail_multiple_chunks():
    """Several chunks accumulate exactly like several host buckets."""

    rng = _rng()
    tables = _make_tables(rng)
    groups = [[0, 1, 2], [3, 4], [5, 6]]
    buckets = [
        _make_bucket(rng, ids, n_rot=max(ROWS_PER_IMAGE[i] for i in ids), tables=tables)
        for ids in groups
    ]
    expected = _host_tail_reference(buckets, tables)
    got = _run_device([[b] for b in buckets], tables, _config(), row_capacity=32, image_capacity=4)
    _assert_finalized_matches_host(got, expected, rtol=FLOAT32_STREAM_RTOL, label="f32 multi ")


def test_device_stage_matches_host_tail_multiple_chunks_float64_companion():
    """Float64 companion for the multi-chunk accumulation order."""

    rng = _rng()
    tables = _make_tables(rng, float_dtype=np.float64)
    groups = [[0, 1, 2], [3, 4], [5, 6]]
    buckets = [
        _make_bucket(
            rng, ids, n_rot=max(ROWS_PER_IMAGE[i] for i in ids), tables=tables,
            float_dtype=np.float64,
        )
        for ids in groups
    ]
    expected = _host_tail_reference(buckets, tables)
    got = _run_device([[b] for b in buckets], tables, _config(), row_capacity=32, image_capacity=4)
    _assert_finalized_matches_host(got, expected, rtol=FLOAT64_STREAM_RTOL, label="f64 multi ")


def test_device_stage_matches_host_tail_algebraic_scale_branch():
    """The non-atomic scale branch reproduces the algebraic host statistics."""

    rng = _rng()
    tables = _make_tables(rng)
    bucket = _make_bucket(rng, list(range(N_IMAGES)), n_rot=max(ROWS_PER_IMAGE), tables=tables)
    expected = _host_tail_reference([bucket], tables, atomic_scale=False, direct_noise=False)
    config = _config(atomic_scale=False, direct_noise=False)
    got = _run_device([[bucket]], tables, config, row_capacity=64, image_capacity=8)
    _assert_finalized_matches_host(got, expected, rtol=FLOAT32_STREAM_RTOL, label="f32 algebraic ")


def test_device_stage_matches_host_tail_algebraic_scale_branch_float64_companion():
    """Float64 companion for the algebraic scale / plain-noise branch."""

    rng = _rng()
    tables = _make_tables(rng, float_dtype=np.float64)
    bucket = _make_bucket(
        rng, list(range(N_IMAGES)), n_rot=max(ROWS_PER_IMAGE), tables=tables, float_dtype=np.float64
    )
    expected = _host_tail_reference([bucket], tables, atomic_scale=False, direct_noise=False)
    config = _config(atomic_scale=False, direct_noise=False)
    got = _run_device([[bucket]], tables, config, row_capacity=64, image_capacity=8)
    _assert_finalized_matches_host(got, expected, rtol=FLOAT64_STREAM_RTOL, label="f64 algebraic ")


def test_padded_rows_and_images_contribute_nothing():
    """Padding past the valid extent adds nothing beyond its own reassociation.

    A larger capacity class is a different traced program, so its float
    reductions associate differently; what padding must never do is contribute
    a summand. The per-image score and pose fields are element-wise, so those
    are held to bitwise equality.
    """

    rng = _rng()
    tables = _make_tables(rng)
    bucket = _make_bucket(rng, list(range(N_IMAGES)), n_rot=max(ROWS_PER_IMAGE), tables=tables)
    tight = _run_device([[bucket]], tables, _config(), row_capacity=18, image_capacity=7)
    padded = _run_device([[bucket]], tables, _config(), row_capacity=256, image_capacity=32)

    for name in _REDUCTION_ACCUMULATORS + _SCALAR_ACCUMULATORS:
        np.testing.assert_allclose(
            np.asarray(getattr(padded, name), dtype=np.float64),
            np.asarray(getattr(tight, name), dtype=np.float64),
            rtol=FLOAT32_STREAM_RTOL,
            atol=0.0,
            err_msg=f"padding moved {name} beyond its reassociation band",
        )
    for name in _BITWISE_FLOAT_FIELDS + _BITWISE_INT_FIELDS + ("best_translation_indices",):
        assert np.array_equal(
            np.asarray(getattr(padded, name)), np.asarray(getattr(tight, name))
        ), f"padding changed {name}"


def test_all_padding_chunk_is_a_no_op():
    """A chunk whose images all carry zero posterior adds exactly zero."""

    rng = _rng()
    tables = _make_tables(rng)
    bucket = _make_bucket(rng, list(range(N_IMAGES)), n_rot=max(ROWS_PER_IMAGE), tables=tables)
    empty = _make_bucket(
        rng, list(range(N_IMAGES)), n_rot=max(ROWS_PER_IMAGE), tables=tables, zero_posterior=True
    )
    config = _config()
    resident_tables = _resident_tables(tables)

    stats = make_resident_statistics(config, max_posterior_dtype=jnp.float32)
    operands = _chunk_from_buckets([bucket], row_capacity=64, image_capacity=8)
    stats = accumulate_chunk_statistics(stats, operands, resident_tables, config=config)
    reference = finalize_statistics(stats, config=config)

    # An all-zero-posterior chunk still writes its per-image score fields, so
    # compare only the accumulated statistics.
    empty_operands = _chunk_from_buckets([empty], row_capacity=64, image_capacity=8)
    stats_after = accumulate_chunk_statistics(stats, empty_operands, resident_tables, config=config)
    after = finalize_statistics(stats_after, config=config)
    for name in (
        "wsum_sigma2_noise",
        "wsum_scale_correction_xa",
        "wsum_scale_correction_aa",
        "rotation_posterior_sums",
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(after, name), dtype=np.float64),
            np.asarray(getattr(reference, name), dtype=np.float64),
            rtol=0.0,
            atol=0.0,
            err_msg=f"an empty chunk changed {name}",
        )
    assert float(after.sumw) == float(reference.sumw)
    assert float(after.wsum_sigma2_offset) == float(reference.wsum_sigma2_offset)


def test_padded_image_slots_never_reach_the_last_real_image():
    """The sentinel scatter guard: ``-1`` ids must not land on image n-1."""

    rng = _rng()
    tables = _make_tables(rng)
    subset = [0, 1]
    bucket = _make_bucket(rng, subset, n_rot=max(ROWS_PER_IMAGE[i] for i in subset), tables=tables)
    config = _config()
    stats = make_resident_statistics(config, max_posterior_dtype=jnp.float32)
    operands = _chunk_from_buckets([bucket], row_capacity=32, image_capacity=16)
    stats = accumulate_chunk_statistics(stats, operands, _resident_tables(tables), config=config)
    finalized = finalize_statistics(stats, config=config, require_all_images=False)
    untouched = np.asarray(finalized.wsum_norm_correction)[2:]
    assert np.all(untouched == 0.0), untouched


def test_one_program_per_capacity_class():
    """Chunks of one capacity class trace a single program."""

    rng = _rng()
    tables = _make_tables(rng)
    config = _config()
    resident_tables = _resident_tables(tables)
    stats = make_resident_statistics(config, max_posterior_dtype=jnp.float32)
    from recovar.em.sparse_pass2 import resident_statistics as module

    module._accumulate_chunk_statistics_jit.clear_cache()
    for ids in ([0, 1, 2], [3, 4], [5, 6]):
        bucket = _make_bucket(
            rng, ids, n_rot=max(ROWS_PER_IMAGE[i] for i in ids), tables=tables
        )
        operands = _chunk_from_buckets([bucket], row_capacity=32, image_capacity=4)
        stats = accumulate_chunk_statistics(stats, operands, resident_tables, config=config)
    jax.block_until_ready(stats.wsum_sigma2_noise)
    assert module._accumulate_chunk_statistics_jit._cache_size() == 1


def test_finalize_rejects_a_winner_in_row_padding():
    """The host's ``best rotation points into padding`` guard is preserved."""

    rng = _rng()
    tables = _make_tables(rng)
    bucket = _make_bucket(rng, list(range(N_IMAGES)), n_rot=max(ROWS_PER_IMAGE), tables=tables)
    config = _config()
    operands = _chunk_from_buckets([bucket], row_capacity=64, image_capacity=8)
    broken = operands._replace(
        best_row_local=operands.best_row_local.at[0].set(
            operands.image_row_start[0] + operands.image_row_count[0]
        )
    )
    stats = make_resident_statistics(config, max_posterior_dtype=jnp.float32)
    stats = accumulate_chunk_statistics(stats, broken, _resident_tables(tables), config=config)
    with pytest.raises(RuntimeError, match="outside their own row range"):
        finalize_statistics(stats, config=config)


def test_finalize_reports_images_no_chunk_covered():
    """A chunk plan that misses an image must not silently return zeros."""

    rng = _rng()
    tables = _make_tables(rng)
    subset = [0, 1, 2]
    bucket = _make_bucket(rng, subset, n_rot=max(ROWS_PER_IMAGE[i] for i in subset), tables=tables)
    config = _config()
    stats = make_resident_statistics(config, max_posterior_dtype=jnp.float32)
    operands = _chunk_from_buckets([bucket], row_capacity=32, image_capacity=4)
    stats = accumulate_chunk_statistics(stats, operands, _resident_tables(tables), config=config)
    with pytest.raises(RuntimeError, match="never written by a chunk"):
        finalize_statistics(stats, config=config)

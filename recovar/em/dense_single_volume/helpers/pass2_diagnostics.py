"""Environment-gated sparse pass-2 score captures.

The sparse scorer calls these writers at the numerical boundary being captured.
This module owns target-row selection and NPZ serialization; bucket scheduling
and production reductions remain in ``sparse_pass2_bucketed``. Iteration/half
identity comes from the shared ``bpref_diagnostics`` context.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers import bpref_diagnostics
from recovar.em.dense_single_volume.helpers.batch_fetch import original_image_indices
from recovar.em.dense_single_volume.helpers.env_flags import parse_env_flag, parse_env_int_set

_PASS2_DUMP_DIR_ENV = "RECOVAR_PASS2_DUMP_DIR"
_PASS2_DUMP_RAW_OPERANDS_ENV = "RECOVAR_PASS2_DUMP_RAW_OPERANDS"
_PASS2_DUMP_ROTATION_ROWS_ENV = "RECOVAR_PASS2_DUMP_ROTATION_ROWS"


def _pass2_dump_target_rows(
    *,
    experiment_dataset,
    image_indices,
    current_size,
) -> np.ndarray:
    """Return batch rows selected by the explicit pass-2 dump contract."""

    dump_dir = os.environ.get("RECOVAR_PASS2_DUMP_DIR")
    if not dump_dir:
        return np.empty((0,), dtype=np.int64)
    target_original_indices = parse_env_int_set("RECOVAR_PASS2_DUMP_ORIGINAL_INDICES")
    if not target_original_indices:
        target_original_indices = parse_env_int_set("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES")
    if not target_original_indices:
        return np.empty((0,), dtype=np.int64)
    target_iteration = os.environ.get("RECOVAR_PASS2_DUMP_ITERATION")
    context_iteration = int(bpref_diagnostics._bpref_contribution_context["iteration"])
    if target_iteration and context_iteration != int(target_iteration):
        return np.empty((0,), dtype=np.int64)
    target_current_size = os.environ.get("RECOVAR_PASS2_DUMP_CURRENT_SIZE")
    if target_current_size:
        if current_size is None or int(current_size) != int(target_current_size):
            return np.empty((0,), dtype=np.int64)

    local_indices = np.asarray(image_indices, dtype=np.int64)
    original_indices = original_image_indices(experiment_dataset, local_indices)
    return np.flatnonzero(
        np.isin(
            original_indices,
            np.fromiter(target_original_indices, dtype=np.int64),
        )
    ).astype(np.int64, copy=False)


def _k1_raw_operand_fields(
    *,
    schema,
    rotation_count,
    row,
    rotation_rows,
    raw_diff2,
    shifted_corrected,
    ctf2_over_nv,
    projections,
    half_weights,
    full_to_compact,
    highres_xi2_half,
):
    """Serialize selected or full K1 rows with the same effective operand dtypes.

    Select batch/rotation rows here so host conversions retain their order in
    both capture paths. ``raw_diff2`` is already the selected float32 array;
    both saved score names refer to it.
    """
    return {
        "raw_operand_schema": np.asarray(schema),
        "raw_operand_actual_rotation_count": np.int64(rotation_count),
        "raw_operand_raw_diff2": raw_diff2,
        "raw_operand_shifted_corrected": np.asarray(shifted_corrected[row], dtype=np.complex64),
        "raw_operand_corr_img_score": np.asarray(ctf2_over_nv[row], dtype=np.float32),
        "raw_operand_proj_half": np.asarray(projections[row, rotation_rows, :], dtype=np.complex64),
        "raw_operand_half_weights": np.asarray(half_weights, dtype=np.float32),
        "raw_operand_relion_full_to_compact": full_to_compact,
        "raw_operand_highres_xi2_half": np.float32(highres_xi2_half[row]),
        "relion_raw_diff2": raw_diff2,
    }


def _pass2_dump_context(current_size):
    """Resolve the shared K1/K-class capture gate in validation order."""
    dump_dir = os.environ.get("RECOVAR_PASS2_DUMP_DIR")
    if not dump_dir:
        return None
    target_original_indices = parse_env_int_set("RECOVAR_PASS2_DUMP_ORIGINAL_INDICES")
    if not target_original_indices:
        target_original_indices = parse_env_int_set("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES")
    if not target_original_indices:
        return None
    target_current_size = os.environ.get("RECOVAR_PASS2_DUMP_CURRENT_SIZE")
    if target_current_size:
        if current_size is None or int(current_size) != int(target_current_size):
            return None
    context_iteration = int(bpref_diagnostics._bpref_contribution_context["iteration"])
    context_half = int(bpref_diagnostics._bpref_contribution_context["half"])
    target_iteration = os.environ.get("RECOVAR_PASS2_DUMP_ITERATION")
    if target_iteration and context_iteration != int(target_iteration):
        return None
    return dump_dir, target_original_indices, context_iteration, context_half


def _maybe_dump_pass2_bucket(
    *,
    experiment_dataset,
    image_indices,
    per_image_inputs,
    current_size,
    n_fine_trans,
    fine_translations,
    scores,
    probs,
    rotation_log_prior,
    translation_log_prior,
    candidate_mask,
    ctf2_over_nv_score,
    proj_half,
    half_weights_used,
    window_indices,
    shifted_corrected_score_split=None,
    direct_score_input=None,
    direct_preprocessed_score_input=None,
    direct_pixel_correction=None,
    direct_inverse_noise_score=None,
    direct_ctf_rfloat_score=None,
    direct_preprocess_normalization_factors=None,
    direct_integer_pre_shifts=None,
    direct_batch_image_corrections=None,
    direct_batch_scale_corrections=None,
    shifted_recon_split=None,
    ctf2_over_nv_recon=None,
    recon_window_indices=None,
    reconstruction_mask=None,
    reconstruction_probs=None,
    reconstruction_n_significant=None,
    relion_highres_xi2_half=None,
    relion_min_diff2=None,
    relion_raw_diff2=None,
    relion_full_to_compact=None,
    raw_score_mode="gaussian",
):
    """Env-gated sparse pass-2 dump for RELION operand parity debugging."""
    dump_context = _pass2_dump_context(current_size)
    if dump_context is None:
        return 0
    dump_dir, target_original_indices, context_iteration, context_half = dump_context
    local_indices = np.asarray(image_indices, dtype=np.int64)
    original_indices = original_image_indices(experiment_dataset, local_indices)

    wanted_rows = [i for i, original_idx in enumerate(original_indices) if int(original_idx) in target_original_indices]
    if not wanted_rows:
        return 0

    raw_operands_requested = parse_env_flag(
        _PASS2_DUMP_RAW_OPERANDS_ENV,
        default=False,
    )
    raw_diff2_np = None
    raw_full_to_compact_np = None
    raw_highres_np = None
    if raw_operands_requested:
        if relion_raw_diff2 is None:
            if raw_score_mode != "normalized_cc":
                raise ValueError(
                    f"{_PASS2_DUMP_RAW_OPERANDS_ENV}=1 requires the production "
                    "K=1 RELION raw-diff2 tensor"
                )
            # Fresh firstiter-CC produces a score tensor directly rather than
            # a Gaussian ``raw_diff2`` tensor. Preserve the effective
            # float32 kernel result by removing the additive priors here;
            # this is capture-only and does not alter scoring or selection.
            relion_raw_diff2 = (
                jnp.asarray(scores)
                - jnp.asarray(rotation_log_prior)[:, :, None]
                - jnp.asarray(translation_log_prior)[:, None, :]
            )
        if shifted_corrected_score_split is None:
            raise ValueError(
                f"{_PASS2_DUMP_RAW_OPERANDS_ENV}=1 requires the effective "
                "shifted score image"
            )
        raw_diff2_np = np.asarray(
            jax.block_until_ready(relion_raw_diff2),
            dtype=np.float32,
        )
        expected_raw_shape = tuple(np.shape(scores))
        if raw_diff2_np.shape != expected_raw_shape:
            raise ValueError(
                "K=1 raw diff2 shape differs from scores: "
                f"{raw_diff2_np.shape} != {expected_raw_shape}"
            )
        raw_pixel_count = int(np.shape(proj_half)[-1])
        raw_full_to_compact_np = (
            np.arange(raw_pixel_count, dtype=np.int32)
            if relion_full_to_compact is None
            else np.asarray(relion_full_to_compact, dtype=np.int32)
        )
        raw_highres_np = (
            np.zeros(local_indices.size, dtype=np.float32)
            if relion_highres_xi2_half is None
            else np.asarray(relion_highres_xi2_half, dtype=np.float32)
        )
        if raw_highres_np.shape != local_indices.shape:
            raise ValueError(
                "K=1 high-resolution raw operand shape differs from the dump batch: "
                f"{raw_highres_np.shape} != {local_indices.shape}"
            )

    requested_rotation_rows = parse_env_int_set(_PASS2_DUMP_ROTATION_ROWS_ENV)
    if requested_rotation_rows:
        rotation_rows = np.asarray(sorted(requested_rotation_rows), dtype=np.int64)
        os.makedirs(dump_dir, exist_ok=True)
        dump_count = 0
        for row in wanted_rows:
            image_idx = int(local_indices[row])
            original_idx = int(original_indices[row])
            cnt = int(per_image_inputs["oversampled_rots"][image_idx].shape[0])
            if np.any(rotation_rows < 0) or np.any(rotation_rows >= cnt):
                raise ValueError(
                    f"{_PASS2_DUMP_ROTATION_ROWS_ENV} contains a row outside "
                    f"[0, {cnt}) for original particle {original_idx}",
                )
            selected_scores = np.asarray(
                jnp.take(scores[row], jnp.asarray(rotation_rows), axis=0),
                dtype=np.float64,
            )
            selected_rotation_prior = np.asarray(
                jnp.take(rotation_log_prior[row], jnp.asarray(rotation_rows), axis=0),
                dtype=np.float64,
            )
            translation_prior = np.asarray(translation_log_prior[row], dtype=np.float64)
            pre_prior = (
                selected_scores
                - selected_rotation_prior[:, None]
                - translation_prior[None, :]
            )
            full_mask = jnp.asarray(candidate_mask[row], dtype=bool)
            full_scores = jnp.asarray(scores[row])
            full_probs = jnp.asarray(probs[row])
            masked_scores = jnp.where(full_mask, full_scores, -jnp.inf)
            masked_probs = jnp.where(full_mask, full_probs, 0.0)
            score_argmax_flat = int(np.asarray(jnp.argmax(masked_scores)))
            prob_argmax_flat = int(np.asarray(jnp.argmax(masked_probs)))
            score_argmax_rotation, score_argmax_translation = divmod(
                score_argmax_flat, int(n_fine_trans)
            )
            prob_argmax_rotation, prob_argmax_translation = divmod(
                prob_argmax_flat, int(n_fine_trans)
            )
            selected_reconstruction_fields = {}
            if reconstruction_mask is not None:
                selected_reconstruction_fields["reconstruction_mask"] = np.asarray(
                    jnp.take(reconstruction_mask[row], jnp.asarray(rotation_rows), axis=0),
                    dtype=bool,
                )
            if reconstruction_probs is not None:
                selected_reconstruction_fields["reconstruction_probs"] = np.asarray(
                    jnp.take(reconstruction_probs[row], jnp.asarray(rotation_rows), axis=0),
                    dtype=np.float64,
                )
            if reconstruction_n_significant is not None:
                recon_n_sig = np.asarray(reconstruction_n_significant, dtype=np.int64)
                selected_reconstruction_fields["reconstruction_n_significant"] = (
                    recon_n_sig[row] if recon_n_sig.ndim else recon_n_sig
                )
            if relion_highres_xi2_half is not None:
                selected_reconstruction_fields["relion_highres_xi2_half"] = np.asarray(
                    relion_highres_xi2_half,
                )[row]
            if relion_min_diff2 is not None:
                selected_reconstruction_fields["relion_min_diff2"] = np.asarray(
                    relion_min_diff2,
                )[row]
            raw_operand_fields = {}
            if raw_operands_requested:
                selected_raw_diff2 = raw_diff2_np[row, rotation_rows, :]
                raw_operand_fields = _k1_raw_operand_fields(
                    schema='recovar-k1-pass2-selected-raw-operands-v1',
                    rotation_count=rotation_rows.size,
                    row=row,
                    rotation_rows=rotation_rows,
                    raw_diff2=selected_raw_diff2,
                    shifted_corrected=shifted_corrected_score_split,
                    ctf2_over_nv=ctf2_over_nv_score,
                    projections=proj_half,
                    half_weights=half_weights_used,
                    full_to_compact=raw_full_to_compact_np,
                    highres_xi2_half=raw_highres_np,
                )
            out_path = os.path.join(
                dump_dir,
                f"pass2_orig{original_idx:06d}_cs{(-1 if current_size is None else int(current_size)):03d}.npz",
            )
            np.savez_compressed(
                out_path,
                schema=np.asarray("recovar.em.k1_pass2_selected_rotations.v1"),
                iteration=np.int64(context_iteration),
                half=np.int64(context_half),
                original_index=np.int64(original_idx),
                local_index=np.int64(image_idx),
                current_size=np.int64(-1 if current_size is None else int(current_size)),
                n_fine_trans=np.int64(n_fine_trans),
                rotation_rows_global=rotation_rows,
                fine_translations=np.asarray(fine_translations),
                rotations=np.asarray(
                    per_image_inputs["oversampled_rots"][image_idx],
                )[rotation_rows],
                oversampled_rot_indices=np.asarray(
                    per_image_inputs["oversampled_rot_indices"][image_idx],
                    dtype=np.int64,
                )[rotation_rows],
                parent_map=np.asarray(
                    per_image_inputs["parent_map"][image_idx],
                    dtype=np.int32,
                )[rotation_rows],
                candidate_mask=np.asarray(
                    jnp.take(candidate_mask[row], jnp.asarray(rotation_rows), axis=0),
                    dtype=bool,
                ),
                candidate_rotation_count=np.int64(cnt),
                candidate_mask_total_count=np.int64(
                    np.asarray(jnp.count_nonzero(full_mask))
                ),
                score_max=np.float64(np.asarray(jnp.max(masked_scores))),
                score_argmax_rotation=np.int64(score_argmax_rotation),
                score_argmax_translation=np.int64(score_argmax_translation),
                posterior_sum=np.float64(np.asarray(jnp.sum(masked_probs))),
                posterior_max=np.float64(np.asarray(jnp.max(masked_probs))),
                posterior_argmax_rotation=np.int64(prob_argmax_rotation),
                posterior_argmax_translation=np.int64(prob_argmax_translation),
                scores_with_prior=selected_scores,
                scores_pre_prior=pre_prior,
                probs=np.asarray(
                    jnp.take(probs[row], jnp.asarray(rotation_rows), axis=0),
                    dtype=np.float64,
                ),
                rotation_log_prior=selected_rotation_prior,
                translation_log_prior=translation_prior,
                shifted_corrected=(
                    np.asarray(shifted_corrected_score_split[row])
                    if shifted_corrected_score_split is not None
                    else np.empty((0,), dtype=np.complex64)
                ),
                direct_score_input=(
                    np.asarray(direct_score_input[row])
                    if direct_score_input is not None
                    else np.empty((0,), dtype=np.complex64)
                ),
                direct_preprocessed_score_input=(
                    np.asarray(direct_preprocessed_score_input[row])
                    if direct_preprocessed_score_input is not None
                    else np.empty((0,), dtype=np.complex64)
                ),
                direct_pixel_correction=(
                    np.asarray(direct_pixel_correction[row])
                    if direct_pixel_correction is not None
                    else np.empty((0,), dtype=np.float32)
                ),
                direct_inverse_noise_score=(
                    np.asarray(direct_inverse_noise_score)
                    if direct_inverse_noise_score is not None
                    else np.empty((0,), dtype=np.float32)
                ),
                direct_ctf_rfloat_score=(
                    np.asarray(direct_ctf_rfloat_score[row], dtype=np.float64)
                    if direct_ctf_rfloat_score is not None
                    else np.empty((0,), dtype=np.float64)
                ),
                relion_preprocess_normalization_factor=(
                    np.asarray(direct_preprocess_normalization_factors)[row]
                    if direct_preprocess_normalization_factors is not None
                    else np.float32(np.nan)
                ),
                relion_integer_pre_shift=(
                    np.asarray(direct_integer_pre_shifts, dtype=np.int32)[row]
                    if direct_integer_pre_shifts is not None
                    else np.empty((0,), dtype=np.int32)
                ),
                batch_image_correction=(
                    np.asarray(direct_batch_image_corrections)[row]
                    if direct_batch_image_corrections is not None
                    else np.float32(np.nan)
                ),
                batch_scale_correction=(
                    np.asarray(direct_batch_scale_corrections)[row]
                    if direct_batch_scale_corrections is not None
                    else np.float32(np.nan)
                ),
                ctf2_over_nv_score=np.asarray(ctf2_over_nv_score[row], dtype=np.float64),
                shifted_recon=(
                    np.asarray(shifted_recon_split[row])
                    if shifted_recon_split is not None
                    else np.empty((0,), dtype=np.complex64)
                ),
                ctf2_over_nv_recon=(
                    np.asarray(ctf2_over_nv_recon[row], dtype=np.float64)
                    if ctf2_over_nv_recon is not None
                    else np.empty((0,), dtype=np.float64)
                ),
                proj_half=np.asarray(
                    jnp.take(proj_half[row], jnp.asarray(rotation_rows), axis=0),
                ),
                half_weights=np.asarray(half_weights_used),
                window_indices=(
                    np.asarray(window_indices, dtype=np.int32)
                    if window_indices is not None
                    else np.empty((0,), dtype=np.int32)
                ),
                recon_window_indices=(
                    np.asarray(recon_window_indices, dtype=np.int32)
                    if recon_window_indices is not None
                    else np.empty((0,), dtype=np.int32)
                ),
                **selected_reconstruction_fields,
                **raw_operand_fields,
            )
            dump_count += 1
        return dump_count

    os.makedirs(dump_dir, exist_ok=True)
    scores_np = np.asarray(scores, dtype=np.float64)
    probs_np = np.asarray(probs, dtype=np.float64)
    rot_prior_np = np.asarray(rotation_log_prior, dtype=np.float64)
    trans_prior_np = np.asarray(translation_log_prior, dtype=np.float64)
    mask_np = np.asarray(candidate_mask, dtype=bool)
    recon_mask_np = None if reconstruction_mask is None else np.asarray(reconstruction_mask, dtype=bool)
    recon_probs_np = None if reconstruction_probs is None else np.asarray(reconstruction_probs, dtype=np.float64)
    recon_n_sig_np = (
        None if reconstruction_n_significant is None else np.asarray(reconstruction_n_significant, dtype=np.int64)
    )
    ctf2_np = np.asarray(ctf2_over_nv_score, dtype=np.float64)
    proj_np = np.asarray(proj_half)
    shifted_corrected_np = (
        None if shifted_corrected_score_split is None else np.asarray(shifted_corrected_score_split)
    )
    direct_score_input_np = (
        None if direct_score_input is None else np.asarray(direct_score_input)
    )
    direct_preprocessed_score_input_np = (
        None
        if direct_preprocessed_score_input is None
        else np.asarray(direct_preprocessed_score_input)
    )
    direct_pixel_correction_np = (
        None if direct_pixel_correction is None else np.asarray(direct_pixel_correction)
    )
    direct_inverse_noise_score_np = (
        None
        if direct_inverse_noise_score is None
        else np.asarray(direct_inverse_noise_score)
    )
    direct_ctf_rfloat_score_np = (
        None
        if direct_ctf_rfloat_score is None
        else np.asarray(direct_ctf_rfloat_score, dtype=np.float64)
    )
    shifted_recon_np = None if shifted_recon_split is None else np.asarray(shifted_recon_split)
    ctf2_recon_np = None if ctf2_over_nv_recon is None else np.asarray(ctf2_over_nv_recon, dtype=np.float64)
    recon_window_indices_np = None if recon_window_indices is None else np.asarray(recon_window_indices, dtype=np.int32)
    highres_np = (
        None if relion_highres_xi2_half is None else np.asarray(relion_highres_xi2_half)
    )
    min_diff2_np = None if relion_min_diff2 is None else np.asarray(relion_min_diff2)

    dump_count = 0
    for row in wanted_rows:
        image_idx = int(local_indices[row])
        original_idx = int(original_indices[row])
        cnt = int(per_image_inputs["oversampled_rots"][image_idx].shape[0])
        scores_row = scores_np[row, :cnt, :]
        pre_prior = scores_row - rot_prior_np[row, :cnt, None] - trans_prior_np[row, None, :]
        out_path = os.path.join(
            dump_dir,
            f"pass2_orig{original_idx:06d}_cs{(-1 if current_size is None else int(current_size)):03d}.npz",
        )
        reconstruction_fields = {}
        if recon_mask_np is not None:
            reconstruction_fields["reconstruction_mask"] = recon_mask_np[row, :cnt, :]
        if recon_probs_np is not None:
            reconstruction_fields["reconstruction_probs"] = recon_probs_np[row, :cnt, :]
        if recon_n_sig_np is not None:
            reconstruction_fields["reconstruction_n_significant"] = (
                recon_n_sig_np[row] if recon_n_sig_np.ndim else recon_n_sig_np
            )
        if highres_np is not None:
            reconstruction_fields["relion_highres_xi2_half"] = highres_np[row]
        if min_diff2_np is not None:
            reconstruction_fields["relion_min_diff2"] = min_diff2_np[row]
        raw_operand_fields = {}
        if raw_operands_requested:
            selected_raw_diff2 = raw_diff2_np[row, :cnt, :]
            raw_operand_fields = _k1_raw_operand_fields(
                schema='recovar-k1-pass2-effective-raw-operands-v1',
                rotation_count=cnt,
                row=row,
                rotation_rows=slice(None, cnt),
                raw_diff2=selected_raw_diff2,
                shifted_corrected=shifted_corrected_np,
                ctf2_over_nv=ctf2_np,
                projections=proj_np,
                half_weights=half_weights_used,
                full_to_compact=raw_full_to_compact_np,
                highres_xi2_half=raw_highres_np,
            )
        np.savez_compressed(
            out_path,
            iteration=np.int64(context_iteration),
            half=np.int64(context_half),
            original_index=np.int64(original_idx),
            local_index=np.int64(image_idx),
            current_size=np.int64(-1 if current_size is None else int(current_size)),
            n_fine_trans=np.int64(n_fine_trans),
            fine_translations=np.asarray(fine_translations),
            rotations=np.asarray(per_image_inputs["oversampled_rots"][image_idx]),
            oversampled_rot_indices=np.asarray(per_image_inputs["oversampled_rot_indices"][image_idx], dtype=np.int64),
            parent_map=np.asarray(per_image_inputs["parent_map"][image_idx], dtype=np.int32),
            candidate_mask=mask_np[row, :cnt, :],
            scores_with_prior=scores_row,
            scores_pre_prior=pre_prior,
            probs=probs_np[row, :cnt, :],
            rotation_log_prior=rot_prior_np[row, :cnt],
            translation_log_prior=trans_prior_np[row],
            shifted_corrected=(
                shifted_corrected_np[row] if shifted_corrected_np is not None else np.empty((0,), dtype=np.complex64)
            ),
            direct_score_input=(
                direct_score_input_np[row]
                if direct_score_input_np is not None
                else np.empty((0,), dtype=np.complex64)
            ),
            direct_preprocessed_score_input=(
                direct_preprocessed_score_input_np[row]
                if direct_preprocessed_score_input_np is not None
                else np.empty((0,), dtype=np.complex64)
            ),
            direct_pixel_correction=(
                direct_pixel_correction_np[row]
                if direct_pixel_correction_np is not None
                else np.empty((0,), dtype=np.float32)
            ),
            direct_inverse_noise_score=(
                direct_inverse_noise_score_np
                if direct_inverse_noise_score_np is not None
                else np.empty((0,), dtype=np.float32)
            ),
            direct_ctf_rfloat_score=(
                direct_ctf_rfloat_score_np[row]
                if direct_ctf_rfloat_score_np is not None
                else np.empty((0,), dtype=np.float64)
            ),
            relion_preprocess_normalization_factor=(
                np.asarray(direct_preprocess_normalization_factors)[row]
                if direct_preprocess_normalization_factors is not None
                else np.float32(np.nan)
            ),
            relion_integer_pre_shift=(
                np.asarray(direct_integer_pre_shifts, dtype=np.int32)[row]
                if direct_integer_pre_shifts is not None
                else np.empty((0,), dtype=np.int32)
            ),
            batch_image_correction=(
                np.asarray(direct_batch_image_corrections)[row]
                if direct_batch_image_corrections is not None
                else np.float32(np.nan)
            ),
            batch_scale_correction=(
                np.asarray(direct_batch_scale_corrections)[row]
                if direct_batch_scale_corrections is not None
                else np.float32(np.nan)
            ),
            ctf2_over_nv_score=ctf2_np[row],
            shifted_recon=(
                shifted_recon_np[row] if shifted_recon_np is not None else np.empty((0,), dtype=np.complex64)
            ),
            ctf2_over_nv_recon=(
                ctf2_recon_np[row] if ctf2_recon_np is not None else np.empty((0,), dtype=np.float64)
            ),
            proj_half=proj_np[row, :cnt, :],
            half_weights=np.asarray(half_weights_used),
            window_indices=(
                np.asarray(window_indices, dtype=np.int32) if window_indices is not None else np.empty((0,), dtype=np.int32)
            ),
            recon_window_indices=(
                recon_window_indices_np if recon_window_indices_np is not None else np.empty((0,), dtype=np.int32)
            ),
            **reconstruction_fields,
            **raw_operand_fields,
        )
        dump_count += 1
    return dump_count


def _capture_k_class_pass2_raw_operands(
    *,
    raw_diff2,
    target_rows,
    actual_counts,
    shifted_corrected,
    corr_img_score,
    proj_half,
    half_weights,
    relion_full_to_compact,
    highres_xi2_half,
    pair_mask=None,
    pair_rotation_row=None,
    pair_translation_idx=None,
):
    """Stage the effective raw-diff2 operands after scoring completes."""

    raw_diff2 = np.asarray(jax.block_until_ready(raw_diff2))
    real_dtype = raw_diff2.dtype
    complex_dtype = np.complex128 if real_dtype == np.dtype(np.float64) else np.complex64
    target_rows = np.asarray(target_rows, dtype=np.int64)
    actual_counts = np.asarray(actual_counts, dtype=np.int64)
    shifted_corrected = np.asarray(shifted_corrected, dtype=complex_dtype)
    corr_img_score = np.asarray(corr_img_score, dtype=real_dtype)
    proj_half = np.asarray(proj_half, dtype=complex_dtype)
    half_weights = np.asarray(half_weights, dtype=real_dtype)
    if relion_full_to_compact is None:
        relion_full_to_compact = np.arange(
            proj_half.shape[-1],
            dtype=np.int32,
        )
    else:
        relion_full_to_compact = np.asarray(
            relion_full_to_compact,
            dtype=np.int32,
        )
    if highres_xi2_half is None:
        highres_xi2_half = np.zeros(shifted_corrected.shape[0], dtype=real_dtype)
    else:
        highres_xi2_half = np.asarray(highres_xi2_half, dtype=real_dtype)
    if pair_mask is None:
        pair_mask = np.empty((shifted_corrected.shape[0], 0), dtype=bool)
        pair_rotation_row = np.empty(
            (shifted_corrected.shape[0], 0),
            dtype=np.int32,
        )
        pair_translation_idx = np.empty(
            (shifted_corrected.shape[0], 0),
            dtype=np.int32,
        )
    else:
        pair_mask = np.asarray(pair_mask, dtype=bool)
        pair_rotation_row = np.asarray(pair_rotation_row, dtype=np.int32)
        pair_translation_idx = np.asarray(pair_translation_idx, dtype=np.int32)

    captured = {}
    for row in target_rows:
        row = int(row)
        n_rot = int(actual_counts[row])
        captured[row] = {
            "actual_rotation_count": np.int64(n_rot),
            "raw_diff2": np.array(raw_diff2[row], copy=True),
            "shifted_corrected": np.array(
                shifted_corrected[row],
                copy=True,
            ),
            "corr_img_score": np.array(corr_img_score[row], copy=True),
            "proj_half": np.array(proj_half[row], copy=True),
            "half_weights": np.array(half_weights, copy=True),
            "relion_full_to_compact": np.array(
                relion_full_to_compact,
                copy=True,
            ),
            "highres_xi2_half": np.asarray(highres_xi2_half[row], dtype=real_dtype)[()],
            "pair_mask": np.array(pair_mask[row], copy=True),
            "pair_rotation_row": np.array(pair_rotation_row[row], copy=True),
            "pair_translation_idx": np.array(
                pair_translation_idx[row],
                copy=True,
            ),
        }
    return captured


def _maybe_dump_k_class_pass2_bucket(
    *,
    experiment_dataset,
    image_indices,
    class_index,
    per_image_inputs,
    class_bucket_arrays,
    compact_pair_arrays,
    current_size,
    n_fine_trans,
    fine_translations,
    scores,
    probs,
    bucket_translation_prior,
    compact_pairs,
    fine_translation_parent=None,
    reconstruction_mask=None,
    reconstruction_probs=None,
    raw_diff2_by_batch_row=None,
    raw_operands_by_batch_row=None,
    relion_min_diff2=None,
):
    """Env-gated K-class sparse pass-2 dump for RELION parity debugging."""

    dump_context = _pass2_dump_context(current_size)
    if dump_context is None:
        return 0
    dump_dir, target_original_indices, context_iteration, context_half = dump_context
    target_class = os.environ.get("RECOVAR_PASS2_DUMP_CLASS")
    if target_class and int(target_class) != int(class_index) + 1:
        return 0

    local_indices = np.asarray(image_indices, dtype=np.int64)
    original_indices = original_image_indices(experiment_dataset, local_indices)
    wanted_rows = [i for i, original_idx in enumerate(original_indices) if int(original_idx) in target_original_indices]
    if not wanted_rows:
        return 0

    os.makedirs(dump_dir, exist_ok=True)
    scores_np = np.asarray(scores, dtype=np.float64)
    probs_np = np.asarray(probs, dtype=np.float64)
    trans_prior_np = np.asarray(bucket_translation_prior, dtype=np.float64)
    fine_translation_parent_np = (
        np.arange(int(n_fine_trans), dtype=np.int32)
        if fine_translation_parent is None
        else np.asarray(fine_translation_parent, dtype=np.int32)
    )
    recon_mask_np = None if reconstruction_mask is None else np.asarray(reconstruction_mask, dtype=bool)
    recon_probs_np = None if reconstruction_probs is None else np.asarray(reconstruction_probs, dtype=np.float64)
    min_diff2_np = (
        None
        if relion_min_diff2 is None
        else np.asarray(relion_min_diff2, dtype=np.float32)
    )
    if raw_diff2_by_batch_row is not None:
        if min_diff2_np is None:
            raise ValueError("raw diff2 pass-2 dump requires the common RELION minimum")
        if min_diff2_np.shape != local_indices.shape:
            raise ValueError(
                "RELION minimum shape differs from the pass-2 dump batch: "
                f"{min_diff2_np.shape} != {local_indices.shape}"
            )

    dump_count = 0
    for row in wanted_rows:
        image_idx = int(local_indices[row])
        original_idx = int(original_indices[row])
        rot_indices = np.asarray(per_image_inputs["oversampled_rot_indices"][image_idx], dtype=np.int64)
        rotations = np.asarray(per_image_inputs["oversampled_rots"][image_idx], dtype=np.float32)
        parent_map = np.asarray(per_image_inputs["parent_map"][image_idx], dtype=np.int32)
        rotation_prior = np.asarray(per_image_inputs["log_prior"][image_idx], dtype=np.float64)
        n_rot = int(rot_indices.shape[0])

        if compact_pairs:
            pair_mask = np.asarray(compact_pair_arrays["pair_mask"][row], dtype=bool)
            pair_rot_rows = np.asarray(compact_pair_arrays["local_rotation_row"][row], dtype=np.int64)
            pair_trans = np.asarray(compact_pair_arrays["translation_idx"][row], dtype=np.int64)
            pair_scores = scores_np[row]
            pair_probs = probs_np[row]
            valid = (
                pair_mask
                & (pair_rot_rows >= 0)
                & (pair_rot_rows < n_rot)
                & (pair_trans >= 0)
                & (pair_trans < int(n_fine_trans))
            )
            scores_with = np.full((n_rot, int(n_fine_trans)), -np.inf, dtype=np.float64)
            prob_dense = np.zeros((n_rot, int(n_fine_trans)), dtype=np.float64)
            candidate_mask = np.zeros((n_rot, int(n_fine_trans)), dtype=bool)
            reconstruction_mask_dense = None
            reconstruction_probs_dense = None
            raw_diff2_dense = None
            if np.any(valid):
                rr = pair_rot_rows[valid]
                tt = pair_trans[valid]
                scores_with[rr, tt] = pair_scores[valid]
                prob_dense[rr, tt] = pair_probs[valid]
                candidate_mask[rr, tt] = True
                if recon_mask_np is not None:
                    reconstruction_mask_dense = np.zeros((n_rot, int(n_fine_trans)), dtype=bool)
                    reconstruction_mask_dense[rr, tt] = recon_mask_np[row][valid]
                if recon_probs_np is not None:
                    reconstruction_probs_dense = np.zeros((n_rot, int(n_fine_trans)), dtype=np.float64)
                    reconstruction_probs_dense[rr, tt] = recon_probs_np[row][valid]
                if raw_diff2_by_batch_row is not None:
                    if row not in raw_diff2_by_batch_row:
                        raise ValueError(
                            f"raw diff2 pass-2 dump is missing batch row {row}"
                        )
                    raw_diff2_pair = np.asarray(
                        raw_diff2_by_batch_row[row],
                        dtype=np.float32,
                    )
                    if raw_diff2_pair.shape != pair_scores.shape:
                        raise ValueError(
                            "compact raw diff2 shape differs from scores: "
                            f"{raw_diff2_pair.shape} != {pair_scores.shape}"
                        )
                    raw_diff2_dense = np.full(
                        (n_rot, int(n_fine_trans)),
                        np.nan,
                        dtype=np.float32,
                    )
                    raw_diff2_dense[rr, tt] = raw_diff2_pair[valid]
        else:
            scores_with = scores_np[row, :n_rot, :]
            prob_dense = probs_np[row, :n_rot, :]
            candidate_mask = np.asarray(class_bucket_arrays["candidate_mask"][row, :n_rot, :], dtype=bool)
            reconstruction_mask_dense = (
                None if recon_mask_np is None else np.asarray(recon_mask_np[row, :n_rot, :], dtype=bool)
            )
            reconstruction_probs_dense = (
                None if recon_probs_np is None else np.asarray(recon_probs_np[row, :n_rot, :], dtype=np.float64)
            )
            raw_diff2_dense = None
            if raw_diff2_by_batch_row is not None:
                if row not in raw_diff2_by_batch_row:
                    raise ValueError(
                        f"raw diff2 pass-2 dump is missing batch row {row}"
                    )
                raw_diff2_dense = np.asarray(
                    raw_diff2_by_batch_row[row],
                    dtype=np.float32,
                )[:n_rot, :]
                if raw_diff2_dense.shape != scores_with.shape:
                    raise ValueError(
                        "dense raw diff2 shape differs from scores: "
                        f"{raw_diff2_dense.shape} != {scores_with.shape}"
                    )

        scores_pre = (
            scores_with
            - rotation_prior[:, None]
            - trans_prior_np[row, None, :]
        )
        reconstruction_fields = {}
        if reconstruction_mask_dense is not None:
            reconstruction_fields["reconstruction_mask"] = reconstruction_mask_dense
            reconstruction_fields["reconstruction_n_significant"] = np.int64(np.count_nonzero(reconstruction_mask_dense))
        if reconstruction_probs_dense is not None:
            reconstruction_fields["reconstruction_probs"] = reconstruction_probs_dense
        raw_diff2_fields = {}
        if raw_diff2_dense is not None:
            raw_diff2_fields = {
                "relion_raw_diff2": raw_diff2_dense,
                "relion_min_diff2": np.float32(min_diff2_np[row]),
            }
        raw_operand_fields = {}
        if raw_operands_by_batch_row is not None:
            if row not in raw_operands_by_batch_row:
                raise ValueError(
                    f"raw operand pass-2 dump is missing batch row {row}"
                )
            raw_operands = raw_operands_by_batch_row[row]
            raw_operand_fields = {
                "raw_operand_schema": np.asarray(
                    "recovar-kclass-pass2-effective-raw-operands-v2"
                ),
                "raw_operand_actual_rotation_count": np.int64(
                    raw_operands["actual_rotation_count"]
                ),
                "raw_operand_raw_diff2": np.asarray(
                    raw_operands["raw_diff2"],
                    dtype=np.float32,
                ),
                "raw_operand_shifted_corrected": np.asarray(
                    raw_operands["shifted_corrected"],
                    dtype=np.complex64,
                ),
                "raw_operand_corr_img_score": np.asarray(
                    raw_operands["corr_img_score"],
                    dtype=np.float32,
                ),
                "raw_operand_proj_half": np.asarray(
                    raw_operands["proj_half"],
                    dtype=np.complex64,
                ),
                "raw_operand_half_weights": np.asarray(
                    raw_operands["half_weights"],
                    dtype=np.float32,
                ),
                "raw_operand_relion_full_to_compact": np.asarray(
                    raw_operands["relion_full_to_compact"],
                    dtype=np.int32,
                ),
                "raw_operand_highres_xi2_half": np.float32(
                    raw_operands["highres_xi2_half"]
                ),
                "raw_operand_pair_mask": np.asarray(
                    raw_operands["pair_mask"],
                    dtype=bool,
                ),
                "raw_operand_pair_rotation_row": np.asarray(
                    raw_operands["pair_rotation_row"],
                    dtype=np.int32,
                ),
                "raw_operand_pair_translation_idx": np.asarray(
                    raw_operands["pair_translation_idx"],
                    dtype=np.int32,
                ),
            }
        out_path = os.path.join(
            dump_dir,
            f"pass2_orig{original_idx:06d}_class{int(class_index) + 1:03d}_cs"
            f"{(-1 if current_size is None else int(current_size)):03d}.npz",
        )
        np.savez_compressed(
            out_path,
            iteration=np.int64(context_iteration),
            half=np.int64(context_half),
            original_index=np.int64(original_idx),
            local_index=np.int64(image_idx),
            class_index=np.int64(class_index),
            current_size=np.int64(-1 if current_size is None else int(current_size)),
            n_fine_trans=np.int64(n_fine_trans),
            fine_translations=np.asarray(fine_translations, dtype=np.float32),
            fine_translation_parent=fine_translation_parent_np,
            rotations=rotations,
            oversampled_rot_indices=rot_indices,
            parent_map=parent_map,
            candidate_mask=candidate_mask,
            scores_with_prior=scores_with,
            scores_pre_prior=scores_pre,
            probs=prob_dense,
            rotation_log_prior=rotation_prior,
            translation_log_prior=trans_prior_np[row],
            compact_pair_dump=np.bool_(compact_pairs),
            **reconstruction_fields,
            **raw_diff2_fields,
            **raw_operand_fields,
        )
        dump_count += 1
    return dump_count

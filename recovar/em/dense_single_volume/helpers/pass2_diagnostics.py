"""Environment-gated sparse pass-2 score and noise captures.

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
    dump_dir = os.environ.get("RECOVAR_PASS2_DUMP_DIR")
    if not dump_dir:
        return 0
    target_original_indices = parse_env_int_set("RECOVAR_PASS2_DUMP_ORIGINAL_INDICES")
    if not target_original_indices:
        target_original_indices = parse_env_int_set("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES")
    if not target_original_indices:
        return 0
    target_current_size = os.environ.get("RECOVAR_PASS2_DUMP_CURRENT_SIZE")
    if target_current_size:
        if current_size is None or int(current_size) != int(target_current_size):
            return 0
    context_iteration = int(bpref_diagnostics._bpref_contribution_context["iteration"])
    context_half = int(bpref_diagnostics._bpref_contribution_context["half"])
    target_iteration = os.environ.get("RECOVAR_PASS2_DUMP_ITERATION")
    if target_iteration and context_iteration != int(target_iteration):
        return 0
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
                raw_operand_fields = {
                    "raw_operand_schema": np.asarray(
                        "recovar-k1-pass2-selected-raw-operands-v1"
                    ),
                    "raw_operand_actual_rotation_count": np.int64(
                        rotation_rows.size
                    ),
                    "raw_operand_raw_diff2": selected_raw_diff2,
                    "raw_operand_shifted_corrected": np.asarray(
                        shifted_corrected_score_split[row],
                        dtype=np.complex64,
                    ),
                    "raw_operand_corr_img_score": np.asarray(
                        ctf2_over_nv_score[row],
                        dtype=np.float32,
                    ),
                    "raw_operand_proj_half": np.asarray(
                        proj_half[row, rotation_rows, :],
                        dtype=np.complex64,
                    ),
                    "raw_operand_half_weights": np.asarray(
                        half_weights_used,
                        dtype=np.float32,
                    ),
                    "raw_operand_relion_full_to_compact": raw_full_to_compact_np,
                    "raw_operand_highres_xi2_half": np.float32(raw_highres_np[row]),
                    "relion_raw_diff2": selected_raw_diff2,
                }
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
            raw_operand_fields = {
                "raw_operand_schema": np.asarray(
                    "recovar-k1-pass2-effective-raw-operands-v1"
                ),
                "raw_operand_actual_rotation_count": np.int64(cnt),
                "raw_operand_raw_diff2": selected_raw_diff2,
                "raw_operand_shifted_corrected": np.asarray(
                    shifted_corrected_np[row],
                    dtype=np.complex64,
                ),
                "raw_operand_corr_img_score": np.asarray(
                    ctf2_np[row],
                    dtype=np.float32,
                ),
                "raw_operand_proj_half": np.asarray(
                    proj_np[row, :cnt, :],
                    dtype=np.complex64,
                ),
                "raw_operand_half_weights": np.asarray(
                    half_weights_used,
                    dtype=np.float32,
                ),
                "raw_operand_relion_full_to_compact": raw_full_to_compact_np,
                "raw_operand_highres_xi2_half": np.float32(raw_highres_np[row]),
                "relion_raw_diff2": selected_raw_diff2,
            }
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


def _maybe_dump_norm_residual_inputs(
    *,
    experiment_dataset,
    image_indices,
    current_size,
    proj_for_noise,
    proj_abs2_for_noise,
    summed_masked_noise,
    ctf_probs,
    ctf2_over_nv_recon,
    posterior_probs,
    rotations_for_noise,
    noise_variance_for_noise,
    block_norm_residual,
    processed_score_half_for_noise,
    shell_indices_half,
    support_mass,
    relion_norm_high_shell,
    weighted_img_per_image,
    relion_score_translation_angles,
    recon_window_indices,
    score_window_indices,
    image_shape,
    bucket_scale_for_stats,
    scale_correction_pixel_mask,
    scale_shell_indices,
    bucket_group_ids,
    relion_wavg_atomic_diff2_rectangle=None,
    relion_wavg_atomic_rectangle_shell_indices=None,
):
    """Capture norm and group-scale AA inputs before any global reduction.

    This diagnostic is deliberately separate from the ordinary pass-2 dump:
    that dump records score-window projections, while the normalization update
    consumes the reconstruction/noise window and its squared projections.
    """

    if not parse_env_flag("RECOVAR_PASS2_DUMP_NORM_RESIDUAL_INPUTS", default=False):
        return 0
    dump_dir = os.environ.get(_PASS2_DUMP_DIR_ENV)
    if not dump_dir:
        raise ValueError(
            "RECOVAR_PASS2_DUMP_NORM_RESIDUAL_INPUTS requires RECOVAR_PASS2_DUMP_DIR"
        )
    target_iteration = os.environ.get("RECOVAR_PASS2_DUMP_ITERATION")
    context_iteration = int(bpref_diagnostics._bpref_contribution_context["iteration"])
    if target_iteration and context_iteration != int(target_iteration):
        return 0
    target_rows = _pass2_dump_target_rows(
        experiment_dataset=experiment_dataset,
        image_indices=image_indices,
        current_size=current_size,
    )
    if target_rows.size == 0:
        return 0

    selected = jnp.asarray(target_rows, dtype=jnp.int32)
    raw_translated_recon = None
    raw_translated_wavg = None
    if relion_score_translation_angles is not None:
        from recovar import cuda_backproject

        recon_indices_jax = jnp.asarray(recon_window_indices, dtype=jnp.int32)
        translation_angles_jax = jnp.asarray(
            relion_score_translation_angles,
            dtype=jnp.float32,
        )
        raw_translated_recon = cuda_backproject.relion_translate_score_f32(
            jnp.asarray(processed_score_half_for_noise)[selected][:, recon_indices_jax],
            translation_angles_jax,
            recon_indices_jax,
            image_shape,
        ).reshape(target_rows.size, translation_angles_jax.shape[0], -1)
        score_indices_jax = jnp.asarray(score_window_indices, dtype=jnp.int32)
        raw_translated_wavg = cuda_backproject.relion_translate_score_f32(
            jnp.asarray(processed_score_half_for_noise)[selected][:, score_indices_jax],
            translation_angles_jax,
            score_indices_jax,
            image_shape,
        ).reshape(target_rows.size, translation_angles_jax.shape[0], -1)
    selected_proj_abs2 = jnp.asarray(proj_abs2_for_noise)[selected]
    selected_ctf_probs = jnp.asarray(ctf_probs)[selected]
    selected_posterior_probs = jnp.asarray(posterior_probs)[selected]
    selected_ctf2_over_nv = jnp.asarray(ctf2_over_nv_recon)[selected]
    selected_scale = jnp.asarray(bucket_scale_for_stats)[selected]
    selected_rotations = jnp.asarray(rotations_for_noise)[selected]
    scale_mask = jnp.asarray(scale_correction_pixel_mask, dtype=bool).reshape(-1)
    selected_noise = jnp.asarray(noise_variance_for_noise)
    norm_ctf_has_mass = selected_ctf_probs != 0.0
    norm_ctf_probs_raw = jnp.where(
        norm_ctf_has_mass,
        selected_ctf_probs * selected_noise[None, None, :],
        0.0,
    )
    norm_a2_terms = jnp.where(
        norm_ctf_has_mass,
        selected_proj_abs2 * norm_ctf_probs_raw,
        0.0,
    )
    selected_summed = jnp.asarray(summed_masked_noise)[selected]
    norm_cross_terms = jnp.where(
        selected_summed != 0.0,
        jnp.asarray(proj_for_noise)[selected] * jnp.conj(selected_summed),
        0.0,
    )
    norm_xa_terms = selected_noise[None, None, :] * norm_cross_terms.real
    norm_a2_per_image = jnp.sum(norm_a2_terms, axis=(1, 2)).astype(jnp.float32)
    norm_xa_per_image = jnp.sum(norm_xa_terms, axis=(1, 2)).astype(jnp.float32)
    ctf_has_mass = (selected_ctf_probs != 0.0) & scale_mask[None, None, :]
    ctf_probs_raw = jnp.where(
        ctf_has_mass,
        selected_ctf_probs * selected_noise[None, None, :],
        0.0,
    )
    aa_terms_before_scale = jnp.where(
        ctf_has_mass,
        selected_proj_abs2 * ctf_probs_raw,
        0.0,
    )
    safe_scale = jnp.maximum(
        selected_scale.astype(selected_proj_abs2.real.dtype),
        1e-30,
    )
    aa_terms = aa_terms_before_scale / (safe_scale[:, None, None] ** 2)
    aa_per_pixel = jnp.sum(aa_terms, axis=1)
    aa_per_image = jnp.sum(aa_terms, axis=(1, 2)).astype(jnp.float32)

    staged = jax.block_until_ready(
        (
            jnp.asarray(proj_for_noise)[selected],
            selected_proj_abs2,
            jnp.asarray(summed_masked_noise)[selected],
            selected_ctf_probs,
            selected_ctf2_over_nv,
            selected_posterior_probs,
            selected_rotations,
            jnp.asarray(block_norm_residual)[selected],
            jnp.asarray(processed_score_half_for_noise)[selected],
            jnp.asarray(support_mass)[selected],
            jnp.asarray(weighted_img_per_image)[selected],
            selected_scale,
            ctf_has_mass,
            ctf_probs_raw,
            aa_terms_before_scale,
            aa_terms,
            aa_per_pixel,
            aa_per_image,
            norm_ctf_has_mass,
            norm_ctf_probs_raw,
            norm_a2_terms,
            norm_cross_terms,
            norm_xa_terms,
            norm_a2_per_image,
            norm_xa_per_image,
            (
                jnp.empty((target_rows.size, 0, 0), dtype=jnp.complex64)
                if raw_translated_recon is None
                else raw_translated_recon
            ),
            (
                jnp.empty((target_rows.size, 0, 0), dtype=jnp.complex64)
                if raw_translated_wavg is None
                else raw_translated_wavg
            ),
        )
    )
    (
        proj_np,
        proj_abs2_np,
        summed_np,
        ctf_probs_np,
        ctf2_over_nv_np,
        posterior_probs_np,
        rotations_np,
        residual_np,
        processed_image_np,
        support_mass_np,
        weighted_image_power_np,
        scale_np,
        ctf_has_mass_np,
        ctf_probs_raw_np,
        aa_terms_before_scale_np,
        aa_terms_np,
        aa_per_pixel_np,
        aa_per_image_np,
        norm_ctf_has_mass_np,
        norm_ctf_probs_raw_np,
        norm_a2_terms_np,
        norm_cross_terms_np,
        norm_xa_terms_np,
        norm_a2_per_image_np,
        norm_xa_per_image_np,
        raw_translated_recon_np,
        raw_translated_wavg_np,
    ) = (np.asarray(value) for value in staged)
    noise_np = np.asarray(jax.block_until_ready(noise_variance_for_noise))
    shell_indices_np = np.asarray(jax.block_until_ready(shell_indices_half), dtype=np.int32)
    scale_shell_indices_np = np.asarray(
        jax.block_until_ready(scale_shell_indices),
        dtype=np.int32,
    )
    scale_mask_np = np.asarray(jax.block_until_ready(scale_mask), dtype=bool)
    relion_high_shell_np = (
        np.empty((0,), dtype=np.float32)
        if relion_norm_high_shell is None
        else np.asarray(jax.block_until_ready(relion_norm_high_shell), dtype=np.float32)
    )
    translation_angles_np = (
        np.empty((0, 2), dtype=np.float32)
        if relion_score_translation_angles is None
        else np.asarray(
            jax.block_until_ready(relion_score_translation_angles),
            dtype=np.float32,
        )
    )
    local_indices = np.asarray(image_indices, dtype=np.int64)
    group_ids_np = (
        np.zeros(local_indices.shape, dtype=np.int64)
        if bucket_group_ids is None
        else np.asarray(bucket_group_ids, dtype=np.int64)
    )
    atomic_rectangle_np = (
        None
        if relion_wavg_atomic_diff2_rectangle is None
        else np.asarray(relion_wavg_atomic_diff2_rectangle, dtype=np.float32)
    )
    atomic_rectangle_shells_np = (
        None
        if relion_wavg_atomic_rectangle_shell_indices is None
        else np.asarray(
            relion_wavg_atomic_rectangle_shell_indices,
            dtype=np.int32,
        ).reshape(-1)
    )
    if (atomic_rectangle_np is None) != (atomic_rectangle_shells_np is None):
        raise ValueError(
            "ordinary Wavg rectangle values and shell labels must be supplied together"
        )
    if atomic_rectangle_np is not None and atomic_rectangle_np.shape != (
        local_indices.size,
        atomic_rectangle_shells_np.size,
    ):
        raise ValueError("ordinary Wavg rectangle diff2 topology changed")
    original_indices = original_image_indices(experiment_dataset, local_indices)
    os.makedirs(dump_dir, exist_ok=True)
    context_half = int(bpref_diagnostics._bpref_contribution_context["half"])
    size_label = -1 if current_size is None else int(current_size)
    for selected_row, bucket_row in enumerate(target_rows.tolist()):
        original_index = int(original_indices[bucket_row])
        out_path = os.path.join(
            dump_dir,
            f"norm_residual_orig{original_index:06d}_half{context_half}_cs{size_label:03d}.npz",
        )
        aa_shells = np.zeros(int(np.max(scale_shell_indices_np, initial=-1)) + 1, dtype=np.float64)
        valid_scale_shell = (
            (scale_shell_indices_np >= 0)
            & scale_mask_np
            & (scale_shell_indices_np < aa_shells.size)
        )
        np.add.at(
            aa_shells,
            scale_shell_indices_np[valid_scale_shell],
            aa_per_pixel_np[selected_row, valid_scale_shell].astype(np.float64),
        )
        payload = dict(
            schema=np.asarray("recovar-k1-norm-residual-inputs-v3"),
            iteration=np.int64(context_iteration),
            half=np.int64(context_half),
            original_index=np.int64(original_index),
            local_index=np.int64(local_indices[bucket_row]),
            bucket_row=np.int64(bucket_row),
            current_size=np.int64(size_label),
            proj_for_noise=proj_np[selected_row],
            proj_abs2_for_noise=proj_abs2_np[selected_row],
            summed_masked_noise=summed_np[selected_row],
            ctf_probs=ctf_probs_np[selected_row],
            ctf2_over_nv_recon=ctf2_over_nv_np[selected_row],
            posterior_probs=posterior_probs_np[selected_row],
            rotations_for_noise=rotations_np[selected_row],
            relion_score_translation_angles=translation_angles_np,
            noise_variance_for_noise=noise_np,
            block_norm_residual=np.asarray(residual_np[selected_row]),
            processed_score_half_for_noise=processed_image_np[selected_row],
            shell_indices_half=shell_indices_np,
            support_mass=np.asarray(support_mass_np[selected_row]),
            relion_norm_high_shell=(
                relion_high_shell_np
                if relion_high_shell_np.size == 0
                else np.asarray(relion_high_shell_np[bucket_row])
            ),
            weighted_img_per_image=np.asarray(weighted_image_power_np[selected_row]),
            group_id=np.int64(group_ids_np[bucket_row]),
            scale_for_stats=np.asarray(scale_np[selected_row]),
            scale_correction_pixel_mask=scale_mask_np,
            scale_shell_indices=scale_shell_indices_np,
            scale_ctf_has_mass=ctf_has_mass_np[selected_row],
            scale_ctf_probs_raw=ctf_probs_raw_np[selected_row],
            scale_aa_terms_before_scale=aa_terms_before_scale_np[selected_row],
            scale_aa_terms=aa_terms_np[selected_row],
            scale_aa_per_pixel=aa_per_pixel_np[selected_row],
            scale_aa_per_shell=aa_shells,
            scale_aa_per_image=np.asarray(aa_per_image_np[selected_row]),
            norm_ctf_has_mass=norm_ctf_has_mass_np[selected_row],
            norm_ctf_probs_raw=norm_ctf_probs_raw_np[selected_row],
            norm_a2_terms=norm_a2_terms_np[selected_row],
            norm_cross_terms=norm_cross_terms_np[selected_row],
            norm_xa_terms=norm_xa_terms_np[selected_row],
            norm_a2_per_image=np.asarray(norm_a2_per_image_np[selected_row]),
            norm_xa_per_image=np.asarray(norm_xa_per_image_np[selected_row]),
            raw_translated_recon=raw_translated_recon_np[selected_row],
            raw_translated_wavg=raw_translated_wavg_np[selected_row],
            recon_window_indices=np.asarray(recon_window_indices, dtype=np.int32),
            wavg_window_indices=np.asarray(score_window_indices, dtype=np.int32),
        )
        if atomic_rectangle_np is not None:
            rectangle_pixels = atomic_rectangle_np[bucket_row]
            valid_rectangle = atomic_rectangle_shells_np >= 0
            payload.update(
                wavg_diff2_atomic_rectangle_per_pixel=rectangle_pixels,
                wavg_diff2_atomic_rectangle_shell_indices=atomic_rectangle_shells_np,
                wavg_diff2_atomic_rectangle_per_image=np.float64(
                    np.sum(rectangle_pixels[valid_rectangle], dtype=np.float64)
                ),
            )
        np.savez_compressed(out_path, **payload)
    return int(target_rows.size)


def _write_chunked_scale_aa_dump(
    *,
    dump_dir,
    experiment_dataset,
    image_indices,
    target_rows,
    current_size,
    bucket_group_ids,
    bucket_scale_for_stats,
    scale_correction_pixel_mask,
    scale_shell_indices,
    chunk_ranges,
    posterior_mass_chunks,
    proj_abs2_sum_chunks,
    ctf_probs_raw_sum_chunks,
    xa_per_pixel_chunks,
    xa_per_image_chunks,
    norm_a2_per_pixel_chunks,
    norm_a2_per_image_chunks,
    norm_xa_per_pixel_chunks,
    norm_xa_per_image_chunks,
    aa_before_scale_per_pixel_chunks,
    aa_per_pixel_chunks,
    aa_per_image_chunks,
    posterior_probs_chunks=None,
    rotation_matrix_chunks=None,
    fine_translations=None,
    aa_feature_per_shell_chunks=None,
    aa_feature_shell_ids=None,
    atomic_xa_per_pixel=None,
    atomic_aa_per_pixel=None,
    atomic_diff2_per_pixel=None,
    atomic_diff2_rectangle=None,
    atomic_diff2_rectangle_shell_indices=None,
    noise_variance_for_noise=None,
    weighted_img_per_image=None,
    relion_norm_high_shell=None,
    norm_shifted_images=None,
):
    """Write compact Wavg XA/AA/diff2 intermediates for a target bucket."""

    target_rows = np.asarray(target_rows, dtype=np.int64)
    if target_rows.size == 0:
        return 0
    local_indices = np.asarray(image_indices, dtype=np.int64)
    original_indices = original_image_indices(experiment_dataset, local_indices)
    group_ids = np.asarray(bucket_group_ids, dtype=np.int64)[target_rows]
    scales = np.asarray(bucket_scale_for_stats, dtype=np.float32)[target_rows]
    mask = np.asarray(scale_correction_pixel_mask, dtype=bool).reshape(-1)
    shells = np.asarray(scale_shell_indices, dtype=np.int32).reshape(-1)
    chunk_ranges_np = np.asarray(chunk_ranges, dtype=np.int64)
    posterior_mass = np.stack(posterior_mass_chunks, axis=1)
    proj_abs2_sum = np.stack(proj_abs2_sum_chunks, axis=1)
    ctf_probs_raw_sum = np.stack(ctf_probs_raw_sum_chunks, axis=1)
    xa_per_pixel_by_chunk = np.stack(xa_per_pixel_chunks, axis=1)
    xa_per_image_by_chunk = np.stack(xa_per_image_chunks, axis=1)
    norm_a2_per_pixel_by_chunk = np.stack(norm_a2_per_pixel_chunks, axis=1)
    norm_a2_per_image_by_chunk = np.stack(norm_a2_per_image_chunks, axis=1)
    norm_xa_per_pixel_by_chunk = np.stack(norm_xa_per_pixel_chunks, axis=1)
    norm_xa_per_image_by_chunk = np.stack(norm_xa_per_image_chunks, axis=1)
    aa_before_scale = np.stack(aa_before_scale_per_pixel_chunks, axis=1)
    aa_per_pixel_by_chunk = np.stack(aa_per_pixel_chunks, axis=1)
    aa_per_image_by_chunk = np.stack(aa_per_image_chunks, axis=1)
    atomic_aa_per_pixel_np = (
        None
        if atomic_aa_per_pixel is None
        else np.asarray(atomic_aa_per_pixel, dtype=np.float32)
    )
    atomic_diff2_per_pixel_np = (
        None
        if atomic_diff2_per_pixel is None
        else np.asarray(atomic_diff2_per_pixel, dtype=np.float32)
    )
    atomic_xa_per_pixel_np = (
        None
        if atomic_xa_per_pixel is None
        else np.asarray(atomic_xa_per_pixel, dtype=np.float32)
    )
    atomic_diff2_rectangle_np = (
        None
        if atomic_diff2_rectangle is None
        else np.asarray(atomic_diff2_rectangle, dtype=np.float32)
    )
    atomic_diff2_rectangle_shells_np = (
        None
        if atomic_diff2_rectangle_shell_indices is None
        else np.asarray(atomic_diff2_rectangle_shell_indices, dtype=np.int32).reshape(-1)
    )
    if atomic_xa_per_pixel_np is not None and atomic_xa_per_pixel_np.shape != (
        local_indices.size,
        mask.size,
    ):
        raise ValueError("chunked scale-XA atomic pixel topology changed")
    if atomic_aa_per_pixel_np is not None and atomic_aa_per_pixel_np.shape != (
        local_indices.size,
        mask.size,
    ):
        raise ValueError("chunked scale-AA atomic pixel topology changed")
    if atomic_diff2_per_pixel_np is not None and atomic_diff2_per_pixel_np.shape != (
        local_indices.size,
        mask.size,
    ):
        raise ValueError("chunked Wavg diff2 atomic pixel topology changed")
    if (atomic_diff2_rectangle_np is None) != (atomic_diff2_rectangle_shells_np is None):
        raise ValueError("chunked Wavg rectangle values and shell labels must be supplied together")
    if atomic_diff2_rectangle_np is not None and atomic_diff2_rectangle_np.shape != (
        local_indices.size,
        atomic_diff2_rectangle_shells_np.size,
    ):
        raise ValueError("chunked Wavg rectangle diff2 topology changed")
    candidate_arrays_present = posterior_probs_chunks is not None or rotation_matrix_chunks is not None
    if candidate_arrays_present:
        if posterior_probs_chunks is None or rotation_matrix_chunks is None or fine_translations is None:
            raise ValueError("chunked scale-AA candidate capture is incomplete")
        posterior_probs = np.concatenate(posterior_probs_chunks, axis=1)
        rotation_matrices = np.concatenate(rotation_matrix_chunks, axis=1)
        fine_translations_np = np.asarray(fine_translations, dtype=np.float32)
        if posterior_probs.shape[:2] != rotation_matrices.shape[:2]:
            raise ValueError("chunked scale-AA candidate rotation topology changed")
        if posterior_probs.shape[2] != fine_translations_np.shape[0]:
            raise ValueError("chunked scale-AA candidate translation topology changed")
        if aa_feature_per_shell_chunks is None or aa_feature_shell_ids is None:
            raise ValueError("chunked scale-AA candidate shell features are missing")
        aa_feature_per_shell = np.concatenate(aa_feature_per_shell_chunks, axis=1)
        aa_feature_shell_ids_np = np.asarray(aa_feature_shell_ids, dtype=np.int32)
        if aa_feature_per_shell.shape[:2] != posterior_probs.shape[:2]:
            raise ValueError("chunked scale-AA candidate shell-feature topology changed")
        if aa_feature_per_shell.shape[2] != aa_feature_shell_ids_np.size:
            raise ValueError("chunked scale-AA candidate shell labels changed")
        norm_shifted_images_np = np.asarray(norm_shifted_images, dtype=np.complex64)
        if norm_shifted_images_np.shape != (
            target_rows.size,
            fine_translations_np.shape[0],
            mask.size,
        ):
            raise ValueError("chunked norm shifted-image topology changed")
    if not (
        posterior_mass.shape[:2]
        == proj_abs2_sum.shape[:2]
        == ctf_probs_raw_sum.shape[:2]
        == xa_per_pixel_by_chunk.shape[:2]
        == xa_per_image_by_chunk.shape[:2]
        == norm_a2_per_pixel_by_chunk.shape[:2]
        == norm_a2_per_image_by_chunk.shape[:2]
        == norm_xa_per_pixel_by_chunk.shape[:2]
        == norm_xa_per_image_by_chunk.shape[:2]
        == aa_before_scale.shape[:2]
        == aa_per_pixel_by_chunk.shape[:2]
        == aa_per_image_by_chunk.shape[:2]
        == (target_rows.size, chunk_ranges_np.shape[0])
    ):
        raise ValueError("chunked scale-AA capture topology changed")

    os.makedirs(dump_dir, exist_ok=True)
    context_iteration = int(bpref_diagnostics._bpref_contribution_context["iteration"])
    context_half = int(bpref_diagnostics._bpref_contribution_context["half"])
    size_label = -1 if current_size is None else int(current_size)
    shell_count = int(np.max(shells, initial=-1)) + 1
    for selected_row, bucket_row in enumerate(target_rows.tolist()):
        xa_per_pixel = np.zeros(mask.shape, dtype=np.float32)
        aa_per_pixel = np.zeros(mask.shape, dtype=np.float32)
        aa_before_scale_per_pixel = np.zeros(mask.shape, dtype=np.float32)
        ctf_probs_raw_per_pixel = np.zeros(mask.shape, dtype=np.float32)
        proj_abs2_per_pixel = np.zeros(mask.shape, dtype=np.float32)
        for chunk_index in range(chunk_ranges_np.shape[0]):
            xa_per_pixel = (
                xa_per_pixel + xa_per_pixel_by_chunk[selected_row, chunk_index]
            ).astype(np.float32)
            aa_per_pixel = (aa_per_pixel + aa_per_pixel_by_chunk[selected_row, chunk_index]).astype(np.float32)
            aa_before_scale_per_pixel = (
                aa_before_scale_per_pixel + aa_before_scale[selected_row, chunk_index]
            ).astype(np.float32)
            ctf_probs_raw_per_pixel = (
                ctf_probs_raw_per_pixel + ctf_probs_raw_sum[selected_row, chunk_index]
            ).astype(np.float32)
            proj_abs2_per_pixel = (
                proj_abs2_per_pixel + proj_abs2_sum[selected_row, chunk_index]
            ).astype(np.float32)
        aa_per_shell = np.zeros(shell_count, dtype=np.float64)
        valid = mask & (shells >= 0) & (shells < shell_count)
        np.add.at(
            aa_per_shell,
            shells[valid],
            aa_per_pixel[valid].astype(np.float64),
        )
        production_aa_total = float(
            np.sum(aa_per_image_by_chunk[selected_row], dtype=np.float64)
        )
        production_xa_total = float(
            np.sum(xa_per_image_by_chunk[selected_row], dtype=np.float64)
        )
        production_norm_a2_total = float(
            np.sum(norm_a2_per_image_by_chunk[selected_row], dtype=np.float64)
        )
        production_norm_xa_total = float(
            np.sum(norm_xa_per_image_by_chunk[selected_row], dtype=np.float64)
        )
        original_index = int(original_indices[bucket_row])
        out_path = os.path.join(
            dump_dir,
            f"scale_aa_chunked_orig{original_index:06d}_half{context_half}_cs{size_label:03d}.npz",
        )
        payload = dict(
            schema=np.asarray("recovar-k1-scale-xa-aa-chunked-v4"),
            iteration=np.int64(context_iteration),
            half=np.int64(context_half),
            original_index=np.int64(original_index),
            local_index=np.int64(local_indices[bucket_row]),
            bucket_row=np.int64(bucket_row),
            current_size=np.int64(size_label),
            group_id=np.int64(group_ids[selected_row]),
            scale_for_stats=np.float32(scales[selected_row]),
            chunk_ranges=chunk_ranges_np,
            posterior_mass_per_chunk=posterior_mass[selected_row],
            proj_abs2_sum_per_pixel_by_chunk=proj_abs2_sum[selected_row],
            ctf_probs_raw_sum_per_pixel_by_chunk=ctf_probs_raw_sum[selected_row],
            aa_before_scale_per_pixel_by_chunk=aa_before_scale[selected_row],
            aa_per_pixel_by_chunk=aa_per_pixel_by_chunk[selected_row],
            aa_per_image_by_chunk=aa_per_image_by_chunk[selected_row],
            scale_correction_pixel_mask=mask,
            scale_shell_indices=shells,
            proj_abs2_sum_per_pixel=proj_abs2_per_pixel,
            ctf_probs_raw_sum_per_pixel=ctf_probs_raw_per_pixel,
            scale_xa_per_pixel_by_chunk=xa_per_pixel_by_chunk[selected_row],
            scale_xa_per_image_by_chunk=xa_per_image_by_chunk[selected_row],
            scale_xa_per_pixel=xa_per_pixel,
            scale_xa_per_image=np.float64(production_xa_total),
            xa_pixel_sum_minus_production_total=np.float64(
                np.sum(xa_per_pixel, dtype=np.float64) - production_xa_total
            ),
            scale_aa_terms_before_scale_per_pixel=aa_before_scale_per_pixel,
            scale_aa_per_pixel=aa_per_pixel,
            scale_aa_per_shell=aa_per_shell,
            scale_aa_per_image=np.float64(production_aa_total),
            pixel_sum_minus_production_total=np.float64(
                np.sum(aa_per_pixel, dtype=np.float64) - production_aa_total
            ),
            norm_a2_per_pixel_by_chunk=norm_a2_per_pixel_by_chunk[selected_row],
            norm_a2_per_image_by_chunk=norm_a2_per_image_by_chunk[selected_row],
            norm_a2_per_image=np.float64(production_norm_a2_total),
            norm_xa_per_pixel_by_chunk=norm_xa_per_pixel_by_chunk[selected_row],
            norm_xa_per_image_by_chunk=norm_xa_per_image_by_chunk[selected_row],
            norm_xa_per_image=np.float64(production_norm_xa_total),
            norm_residual_per_image=np.float64(
                production_norm_a2_total - 2.0 * production_norm_xa_total
            ),
            noise_variance_for_noise=np.asarray(noise_variance_for_noise, dtype=np.float32),
            weighted_img_per_image=np.float64(
                np.asarray(weighted_img_per_image, dtype=np.float64)[bucket_row]
            ),
            relion_norm_high_shell=np.float64(
                np.asarray(relion_norm_high_shell, dtype=np.float64)[bucket_row]
            ),
        )
        if atomic_xa_per_pixel_np is not None:
            atomic_xa_pixels = atomic_xa_per_pixel_np[bucket_row]
            atomic_xa_shells = np.zeros(shell_count, dtype=np.float64)
            np.add.at(
                atomic_xa_shells,
                shells[valid],
                atomic_xa_pixels[valid].astype(np.float64),
            )
            payload.update(
                scale_xa_atomic_per_pixel=atomic_xa_pixels,
                scale_xa_atomic_per_shell=atomic_xa_shells,
                scale_xa_atomic_per_image=np.float64(
                    np.sum(atomic_xa_pixels, dtype=np.float64)
                ),
            )
        if atomic_aa_per_pixel_np is not None:
            atomic_pixels = atomic_aa_per_pixel_np[bucket_row]
            atomic_shells = np.zeros(shell_count, dtype=np.float64)
            np.add.at(
                atomic_shells,
                shells[valid],
                atomic_pixels[valid].astype(np.float64),
            )
            payload.update(
                scale_aa_atomic_per_pixel=atomic_pixels,
                scale_aa_atomic_per_shell=atomic_shells,
                scale_aa_atomic_per_image=np.float64(
                    np.sum(atomic_pixels, dtype=np.float64)
                ),
            )
        if atomic_diff2_per_pixel_np is not None:
            atomic_diff2_pixels = atomic_diff2_per_pixel_np[bucket_row]
            atomic_diff2_shells = np.zeros(shell_count, dtype=np.float64)
            valid_current_size = (shells >= 0) & (shells < shell_count)
            np.add.at(
                atomic_diff2_shells,
                shells[valid_current_size],
                atomic_diff2_pixels[valid_current_size].astype(np.float64),
            )
            payload.update(
                wavg_diff2_atomic_per_pixel=atomic_diff2_pixels,
                wavg_diff2_atomic_per_shell=atomic_diff2_shells,
                wavg_diff2_atomic_per_image=np.float64(
                    np.sum(atomic_diff2_pixels, dtype=np.float64)
                ),
            )
        if atomic_diff2_rectangle_np is not None:
            rectangle_pixels = atomic_diff2_rectangle_np[bucket_row]
            rectangle_shells = np.zeros(shell_count, dtype=np.float64)
            valid_rectangle = (
                (atomic_diff2_rectangle_shells_np >= 0)
                & (atomic_diff2_rectangle_shells_np < shell_count)
            )
            np.add.at(
                rectangle_shells,
                atomic_diff2_rectangle_shells_np[valid_rectangle],
                rectangle_pixels[valid_rectangle].astype(np.float64),
            )
            payload.update(
                wavg_diff2_atomic_rectangle_per_pixel=rectangle_pixels,
                wavg_diff2_atomic_rectangle_shell_indices=atomic_diff2_rectangle_shells_np,
                wavg_diff2_atomic_rectangle_per_shell=rectangle_shells,
                wavg_diff2_atomic_rectangle_per_image=np.float64(
                    np.sum(rectangle_pixels[valid_rectangle], dtype=np.float64)
                ),
            )
        if candidate_arrays_present:
            payload.update(
                candidate_posterior_probs=np.asarray(posterior_probs[selected_row], dtype=np.float32),
                candidate_rotation_matrices=np.asarray(rotation_matrices[selected_row], dtype=np.float32),
                fine_translations=fine_translations_np,
                candidate_aa_feature_per_shell=np.asarray(
                    aa_feature_per_shell[selected_row],
                    dtype=np.float32,
                ),
                candidate_aa_feature_shell_ids=aa_feature_shell_ids_np,
                norm_shifted_images=np.asarray(
                    norm_shifted_images_np[selected_row],
                    dtype=np.complex64,
                ),
            )
        np.savez_compressed(out_path, **payload)
    return int(target_rows.size)


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

    dump_dir = os.environ.get("RECOVAR_PASS2_DUMP_DIR")
    if not dump_dir:
        return 0
    target_original_indices = parse_env_int_set("RECOVAR_PASS2_DUMP_ORIGINAL_INDICES")
    if not target_original_indices:
        target_original_indices = parse_env_int_set("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES")
    if not target_original_indices:
        return 0
    target_current_size = os.environ.get("RECOVAR_PASS2_DUMP_CURRENT_SIZE")
    if target_current_size:
        if current_size is None or int(current_size) != int(target_current_size):
            return 0
    context_iteration = int(bpref_diagnostics._bpref_contribution_context["iteration"])
    context_half = int(bpref_diagnostics._bpref_contribution_context["half"])
    target_iteration = os.environ.get("RECOVAR_PASS2_DUMP_ITERATION")
    if target_iteration and context_iteration != int(target_iteration):
        return 0
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

"""BPref diagnostic capture, scoped execution checks and artifact writing.

This module owns the shared numbered-half context and capture counters used by
sparse and exact-local EM. Diagnostic output preserves the existing schemas,
array precision and capture order. Execution-policy flags here select explicit
diagnostic overrides; callers still own the production scoring and M-step.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers.batch_fetch import original_image_indices
from recovar.em.dense_single_volume.helpers.env_flags import parse_env_flag, parse_env_int_set
from recovar.em.dense_single_volume.helpers.preprocessing import image_preprocess_backend
from recovar.em.dense_single_volume.local_backprojection import (
    relion_x_half_sequential_translation_reduction_enabled,
)
from recovar.utils.file_hash import sha256_file

_BPREF_MEMBERSHIP_DUMP_DIR_ENV = "RECOVAR_BPREF_MEMBERSHIP_DUMP_DIR"
_BPREF_MEMBERSHIP_DUMP_ITERATION_ENV = "RECOVAR_BPREF_MEMBERSHIP_DUMP_ITERATION"
_BPREF_MEMBERSHIP_DUMP_HALF_ENV = "RECOVAR_BPREF_MEMBERSHIP_DUMP_HALF"

_bpref_membership_dump_counter = 0


_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH_ENV = "RECOVAR_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH"


_RELION_X_HALF_BP_FUSED_ATOMICS_ENV = "RECOVAR_RELION_X_HALF_BP_FUSED_ATOMICS"


_BPREF_CONTRIBUTION_DUMP_CLASS_ENV = "RECOVAR_BPREF_CONTRIBUTION_DUMP_CLASS"


_BPREF_CONTRIBUTION_STOP_AFTER_TARGET_ENV = "RECOVAR_BPREF_CONTRIBUTION_STOP_AFTER_TARGET"


_native_mstep_dump_counter = 0


_bpref_contribution_dump_counter = 0


_bpref_contribution_call_counter = 0


_bpref_contribution_context = {"iteration": -1, "half": -1}


_bpref_image_identity_cache: dict[str, np.ndarray] = {}


_BPrefPanelKey = tuple[int, int, str, int]


_bpref_device_panel_accumulators: dict[_BPrefPanelKey, tuple[jax.Array, jax.Array]] = {}


_bpref_device_panel_launch_counters: dict[_BPrefPanelKey, int] = {}


_bpref_device_panel_metadata: dict[_BPrefPanelKey, dict[str, object]] = {}


def set_bpref_contribution_dump_context(*, iteration: int, half: int) -> None:
    """Set explicit one-based iteration/half labels for diagnostic row dumps."""

    _bpref_contribution_context["iteration"] = int(iteration)
    _bpref_contribution_context["half"] = int(half)


def clear_bpref_contribution_dump_context() -> None:
    """Mark contribution and native M-step dumps as outside a numbered half."""

    _bpref_contribution_context["iteration"] = -1
    _bpref_contribution_context["half"] = -1


class BPrefContributionDumpComplete(RuntimeError):
    """Raised after an explicitly targeted BPref diagnostic bundle is written."""

    def __init__(
        self,
        *,
        contribution_path: str | Path,
        device_signature_path: str | Path | None,
    ):
        self.contribution_path = Path(contribution_path)
        self.device_signature_path = None if device_signature_path is None else Path(device_signature_path)
        message = f"requested RECOVAR BPref contribution target was written (contribution_path={self.contribution_path}"
        if self.device_signature_path is not None:
            message += f", device_signature_path={self.device_signature_path}"
        super().__init__(message + ")")


def _maybe_stop_after_bpref_contribution_dump(
    *,
    contribution_path: str | Path,
    device_signature_path: str | Path | None,
) -> None:
    """Stop an explicit diagnostic only after all requested files exist."""

    if os.environ.get(_BPREF_CONTRIBUTION_STOP_AFTER_TARGET_ENV) != "1":
        return
    contribution_path = Path(contribution_path)
    if not contribution_path.is_file():
        raise RuntimeError(
            f"RECOVAR BPref contribution stop target is missing its contribution file: {contribution_path}"
        )
    device_dump_requested = bool(os.environ.get("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", "").strip())
    resolved_device_path = None if device_signature_path is None else Path(device_signature_path)
    if device_dump_requested and (resolved_device_path is None or not resolved_device_path.is_file()):
        raise RuntimeError(
            "RECOVAR BPref contribution stop target is missing its requested "
            f"device-signature file: {resolved_device_path}"
        )
    raise BPrefContributionDumpComplete(
        contribution_path=contribution_path,
        device_signature_path=resolved_device_path,
    )


def _bpref_contribution_target_rows(experiment_dataset, image_indices) -> np.ndarray:
    """Return bucket rows selected by the optional frozen original-index target."""

    local_indices = np.asarray(image_indices, dtype=np.int64)
    target_values = parse_env_int_set(
        "RECOVAR_BPREF_CONTRIBUTION_DUMP_ORIGINAL_INDICES"
    )
    if not target_values:
        return np.arange(local_indices.size, dtype=np.int64)
    targets = np.asarray(
        sorted(target_values),
        dtype=np.int64,
    )
    original_indices = original_image_indices(experiment_dataset, local_indices)
    return np.flatnonzero(np.isin(original_indices, targets)).astype(np.int64, copy=False)


def _bpref_diagnostic_ownership_indices(
    image_indices,
    target_particle_rows,
    *,
    device_signature_requested: bool,
) -> np.ndarray:
    """Return particle owners relevant to the requested diagnostic.

    A scoped device signature captures only the configured target rows.  The
    surrounding sparse bucket may be ordered by support size rather than by
    particle id, so requiring every unrelated bucket row to be monotone can
    abort an otherwise target-only observational capture.  Unscoped
    per-particle diagnostics retain the original full-bucket ordering gate.
    """

    owners = np.asarray(image_indices, dtype=np.int64)
    if not device_signature_requested:
        return owners
    rows = np.asarray(target_particle_rows, dtype=np.int64)
    if rows.size == 0:
        return np.empty((0,), dtype=np.int64)
    if np.any(rows < 0) or np.any(rows >= owners.size):
        raise RuntimeError("BPref device signature target row is outside the sparse bucket")
    return owners[rows]


def _validate_bpref_diagnostic_ownership(
    owners,
    *,
    device_signature_requested: bool,
) -> None:
    """Validate ownership without imposing particle-id order on scoped captures."""

    owners = np.asarray(owners, dtype=np.int64)
    if owners.size < 2:
        return
    if device_signature_requested:
        if np.unique(owners).size != owners.size:
            raise RuntimeError("Scoped BPref device signature requires unique particle ownership")
    elif not np.all(np.diff(owners) > 0):
        raise RuntimeError(
            "RELION per-particle launch diagnostic requires strictly increasing particle ownership order"
        )


def _resolve_bpref_bucket_diagnostic_modes(
    *,
    device_signature_requested: bool,
    contribution_diagnostics_active: bool,
    target_particle_rows,
    high_precision_operand_bundle_requested: bool,
) -> dict[str, bool]:
    """Limit scoped device diagnostics to buckets containing a target row."""

    target_bucket_active = bool(device_signature_requested and np.asarray(target_particle_rows).size)
    bucket_contribution_diagnostics_active = bool(
        contribution_diagnostics_active and (not device_signature_requested or target_bucket_active)
    )
    return {
        "device_signature_requested": target_bucket_active,
        "contribution_diagnostics_active": bucket_contribution_diagnostics_active,
        "shadow_only": target_bucket_active,
        "high_precision_operand_bundle": bool(
            bucket_contribution_diagnostics_active and high_precision_operand_bundle_requested
        ),
    }


def _bpref_contribution_class_enabled(class_index: int) -> bool:
    """Return whether a zero-based class belongs to the scoped capture.

    The environment value is one-based to match RELION's class numbering and
    the class labels used by the pre-scatter comparison scripts.
    """

    value = os.environ.get(_BPREF_CONTRIBUTION_DUMP_CLASS_ENV, "").strip()
    if not value:
        return True
    try:
        requested = int(value)
    except ValueError as exc:
        raise ValueError(f"{_BPREF_CONTRIBUTION_DUMP_CLASS_ENV} must be a positive integer") from exc
    if requested <= 0:
        raise ValueError(f"{_BPREF_CONTRIBUTION_DUMP_CLASS_ENV} must be a positive integer")
    return int(class_index) + 1 == requested


def _validate_bpref_positive_rotation_rows(
    positive_rotation_rows,
    target_particle_rows,
    *,
    device_signature_requested: bool,
    winner_take_all: bool,
    posterior_partitioned_across_classes: bool = False,
) -> None:
    """Validate positive-row support for owners represented by a diagnostic.

    A soft posterior is allowed to leave one positive rotation row for every
    particle after reconstruction pruning.  This check runs independently for
    each sparse bucket, so requiring a multi-row witness here would incorrectly
    reject a valid bucket even when other buckets contain soft multi-row
    particles.  A fused K-class capture is a slice of a jointly normalized
    posterior: a particle may therefore have zero rows in the requested class
    while retaining support in another class.
    """

    counts = np.asarray(positive_rotation_rows, dtype=np.int64)
    if device_signature_requested:
        rows = np.asarray(target_particle_rows, dtype=np.int64)
        if rows.size == 0:
            return
        if np.any(rows < 0) or np.any(rows >= counts.size):
            raise RuntimeError("BPref device signature target row is outside the sparse bucket")
        counts = counts[rows]
    if np.any(counts < 0):
        raise RuntimeError("BPref positive rotation-row count cannot be negative")
    if posterior_partitioned_across_classes:
        if winner_take_all and np.any(counts > 1):
            raise RuntimeError(
                "RELION K-class WTA diagnostic permits at most one positive rotation row per particle and class"
            )
        return
    if winner_take_all:
        if not np.all(counts == 1):
            raise RuntimeError(
                "RELION WTA per-particle diagnostic requires exactly one positive rotation row per particle"
            )
    elif np.any(counts < 1):
        raise RuntimeError("RECOVAR soft-particle causal arm requires at least one positive row per particle")


def _empty_bpref_device_signature_arrays(
    dense_pixel_count: int,
    *,
    image_identity_dtype,
) -> dict[str, np.ndarray]:
    """Return a schema-valid signature payload for an all-zero class slice."""

    pixels = int(dense_pixel_count)
    if pixels <= 0:
        raise ValueError("BPref device signature dense pixel count must be positive")
    return {
        "rotation_keys": np.empty((0, pixels), dtype=np.int32),
        "pixel_indices": np.empty((0, pixels), dtype=np.int32),
        "row_flags": np.empty((0, pixels), dtype=np.int32),
        "source_values": np.empty((0, pixels, 6), dtype=np.float32),
        "neighbor_indices": np.empty((0, pixels, 8), dtype=np.int32),
        "neighbor_coefficients": np.empty((0, pixels, 8), dtype=np.float32),
        "neighbor_flags": np.empty((0, pixels, 8), dtype=np.int32),
        "launch_ordinals": np.empty((0,), dtype=np.int64),
        "particle_local_rows": np.empty((0,), dtype=np.int32),
        "image_identities": np.empty((0,), dtype=np.dtype(image_identity_dtype)),
        "original_indices": np.empty((0,), dtype=np.int64),
        "contributor_rotation_keys": np.empty((0,), dtype=np.int32),
    }


def _guard_bpref_target_rotation_chunking(
    rotation_chunk_size,
    *,
    bucket_size: int,
    target_particle_rows,
):
    """Preserve live chunk planning and reject only a genuinely chunked target."""

    target_count = int(np.asarray(target_particle_rows).size)
    if target_count and rotation_chunk_size is not None and int(rotation_chunk_size) < int(bucket_size):
        raise RuntimeError(
            "BPref device signature target bucket is rotation-chunked in the "
            "authoritative production plan; capture refuses to change that plan "
            f"(bucket_size={int(bucket_size)}, rotation_chunk_size={int(rotation_chunk_size)}, "
            f"target_particles={target_count})"
        )
    return rotation_chunk_size


def _bpref_image_identities_for_original_indices(original_indices: np.ndarray) -> np.ndarray:
    """Return exact ``rlnImageName`` identities for diagnostic particles.

    The explicit mapping is required for cross-engine diagnostics because a
    local dataset row or original integer index is not, by itself, a stable
    identity across STAR readers.  Object arrays are deliberately rejected so
    the diagnostic never needs pickle.
    """

    mapping_path = os.environ.get("RECOVAR_BPREF_CONTRIBUTION_IMAGE_NAMES_NPY", "").strip()
    if not mapping_path:
        raise RuntimeError(
            "RECOVAR_BPREF_CONTRIBUTION_IMAGE_NAMES_NPY is required when RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR is enabled"
        )
    resolved = str(Path(mapping_path).expanduser().resolve())
    identities = _bpref_image_identity_cache.get(resolved)
    if identities is None:
        identities = np.load(resolved, allow_pickle=False)
        if identities.ndim != 1 or identities.dtype.kind not in {"U", "S"}:
            raise ValueError(
                "BPref image identity mapping must be a rank-1 fixed-width string NPY, "
                f"got shape={identities.shape} dtype={identities.dtype}"
            )
        identities = identities.astype(str, copy=False)
        _bpref_image_identity_cache[resolved] = identities
    original_indices = np.asarray(original_indices, dtype=np.int64)
    if original_indices.size and (int(original_indices.min()) < 0 or int(original_indices.max()) >= identities.size):
        raise IndexError(
            "BPref original particle index is outside the explicit image identity mapping: "
            f"range=[{int(original_indices.min())}, {int(original_indices.max())}] "
            f"mapping_size={identities.size}"
        )
    selected = identities[original_indices]
    if np.any(np.char.find(selected, "@") <= 0):
        raise ValueError("Every BPref image identity must be an exact 1-based-index@stack-path string")
    for identity in selected.tolist():
        _, stack_path = identity.split("@", 1)
        if not Path(stack_path).is_absolute():
            raise ValueError(f"BPref image identity stack path must be absolute, got {identity!r}")
    return selected


def _bpref_required_stack_checksum() -> str:
    checksum = os.environ.get("RECOVAR_BPREF_CONTRIBUTION_STACK_SHA256", "").strip().lower()
    if len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum):
        raise RuntimeError("RECOVAR_BPREF_CONTRIBUTION_STACK_SHA256 must contain the frozen source stack SHA256")
    return checksum


def flush_bpref_device_panel_accumulator(*, iteration: int, half: int) -> None:
    """Write and release every exact native class panel for one half."""

    dump_dir = os.environ.get("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", "").strip()
    if not dump_dir:
        return
    run_id = os.environ.get("RECOVAR_BPREF_CONTRIBUTION_DUMP_RUN_ID", "unset")
    prefix = (int(iteration), int(half), run_id)
    keys = sorted(key for key in _bpref_device_panel_metadata if key[:3] == prefix)
    if not keys:
        raise RuntimeError(f"No RECOVAR device panel metadata exists for {prefix}")
    output = Path(dump_dir)
    output.mkdir(parents=True, exist_ok=True)
    for key in keys:
        accumulators = _bpref_device_panel_accumulators.pop(key, None)
        launch_count = _bpref_device_panel_launch_counters.pop(key, 0)
        metadata = _bpref_device_panel_metadata.pop(key)
        if accumulators is None:
            raise RuntimeError(f"No RECOVAR device panel accumulator exists for {key}")
        data_accumulator, weight_accumulator = accumulators
        class_index = int(metadata["class_index"])
        np.savez(
            output
            / (
                f"recovar_device_panel_native_it{int(iteration):03d}_h{int(half)}"
                f"_class{class_index + 1:03d}_rank{int(metadata['rank']):03d}.npz"
            ),
            magic=np.asarray("RECOVAR_DEVICE_PANEL_NATIVE"),
            schema=np.asarray("recovar-device-panel-native-v1"),
            schema_version=np.int32(1),
            run_id=np.asarray(run_id),
            iteration=np.int32(iteration),
            half=np.int32(half),
            class_index=np.int32(class_index),
            rank=np.int32(metadata["rank"]),
            launch_count=np.int64(launch_count),
            current_size=np.int32(metadata["current_size"]),
            max_r=np.float32(metadata["max_r"]),
            image_shape=np.asarray(metadata["image_shape"], dtype=np.int32),
            volume_shape=np.asarray(metadata["volume_shape"], dtype=np.int32),
            reconstruction_padding_factor=np.int32(metadata["reconstruction_padding_factor"]),
            source_stack_sha256=np.asarray(metadata["source_stack_sha256"]),
            causal_arm=np.asarray(metadata["causal_arm"]),
            winner_take_all=np.bool_(metadata["winner_take_all"]),
            topology_claim=np.asarray("causal-arm-not-relion-hypothesis-arithmetic-closure"),
            accumulator_field_legend=np.asarray("data=complex64 x-half;weight=float32 x-half;flat C order"),
            data_accumulator=np.asarray(data_accumulator),
            weight_accumulator=np.asarray(weight_accumulator),
        )


def _maybe_dump_native_half_mstep(
    Ft_y_total,
    Ft_ctf_total,
    *,
    current_size,
    n_images,
    recon_volume_shape,
    stage,
):
    dump_dir = os.environ.get("RECOVAR_SPARSE_PASS2_NATIVE_DUMP_DIR")
    if not dump_dir:
        return
    context_iteration = int(_bpref_contribution_context["iteration"])
    context_half = int(_bpref_contribution_context["half"])
    target_iteration = os.environ.get("RECOVAR_SPARSE_PASS2_NATIVE_DUMP_ITERATION")
    if target_iteration and context_iteration != int(target_iteration):
        return

    global _native_mstep_dump_counter
    dump_idx = _native_mstep_dump_counter
    _native_mstep_dump_counter += 1

    path = Path(dump_dir)
    path.mkdir(parents=True, exist_ok=True)
    run_id = os.environ.get("RECOVAR_SPARSE_PASS2_NATIVE_DUMP_RUN_ID", "unset")
    np.savez_compressed(
        path
        / (
            f"native_half_mstep_it{context_iteration:03d}_h{context_half}"
            f"_dump{dump_idx:03d}_{stage}_n{int(n_images):04d}_cs{int(current_size):03d}.npz"
        ),
        schema=np.asarray("recovar-native-half-mstep-v2"),
        dump_index=np.int64(dump_idx),
        iteration=np.int32(context_iteration),
        half=np.int32(context_half),
        run_id=np.asarray(run_id),
        Ft_y=np.asarray(Ft_y_total),
        Ft_ctf=np.asarray(Ft_ctf_total),
        current_size=np.int32(current_size),
        n_images=np.int32(n_images),
        recon_volume_shape=np.asarray(recon_volume_shape, dtype=np.int32),
        stage=np.asarray(stage),
    )


def _materialize_k_class_capture_rows(
    *,
    image_indices,
    target_particle_rows,
    per_image_inputs,
    class_bucket_arrays,
    compact_pair_arrays,
    scores,
    probs,
    reconstruction_mask,
    reconstruction_probs,
    bucket_translation_prior,
    n_fine_trans: int,
):
    """Materialize only selected fused-K rows in rectangular diagnostic form."""

    rows = np.asarray(target_particle_rows, dtype=np.int64)
    if rows.ndim != 1 or rows.size == 0:
        raise ValueError("fused K-class capture requires at least one target particle row")
    image_indices_np = np.asarray(image_indices, dtype=np.int64)
    if np.any(rows < 0) or np.any(rows >= image_indices_np.size):
        raise ValueError("fused K-class capture target row is outside the bucket")

    selected_image_indices = image_indices_np[rows]
    n_selected = int(rows.size)
    n_rot = int(class_bucket_arrays["bucket_size"])
    n_trans = int(n_fine_trans)

    def _selected(values):
        return np.asarray(jnp.asarray(values)[jnp.asarray(rows, dtype=jnp.int32)])

    selected_scores = _selected(scores)
    selected_probs = _selected(probs)
    selected_reconstruction_mask = (
        None if reconstruction_mask is None else _selected(reconstruction_mask).astype(bool, copy=False)
    )
    selected_reconstruction_probs = (
        None if reconstruction_probs is None else _selected(reconstruction_probs)
    )

    rotation_log_prior = np.zeros((n_selected, n_rot), dtype=np.float32)
    for selected_row, image_index in enumerate(selected_image_indices.tolist()):
        prior = np.asarray(per_image_inputs["log_prior"][int(image_index)], dtype=np.float32)
        if prior.size > n_rot:
            raise ValueError("fused K-class capture rotation prior exceeds its bucket")
        rotation_log_prior[selected_row, : prior.size] = prior

    if compact_pair_arrays is None:
        candidate_mask = _selected(class_bucket_arrays["candidate_mask"]).astype(bool, copy=False)
        dense_scores = selected_scores
        dense_probs = selected_probs
        dense_reconstruction_mask = selected_reconstruction_mask
        dense_reconstruction_probs = selected_reconstruction_probs
    else:
        pair_rows = _selected(compact_pair_arrays["local_rotation_row"]).astype(np.int64, copy=False)
        pair_translations = _selected(compact_pair_arrays["translation_idx"]).astype(np.int64, copy=False)
        pair_mask = _selected(compact_pair_arrays["pair_mask"]).astype(bool, copy=False)
        dense_scores = np.full((n_selected, n_rot, n_trans), -np.inf, dtype=selected_scores.dtype)
        dense_probs = np.zeros((n_selected, n_rot, n_trans), dtype=selected_probs.dtype)
        candidate_mask = np.zeros((n_selected, n_rot, n_trans), dtype=bool)
        dense_reconstruction_mask = (
            None
            if selected_reconstruction_mask is None
            else np.zeros((n_selected, n_rot, n_trans), dtype=bool)
        )
        dense_reconstruction_probs = (
            None
            if selected_reconstruction_probs is None
            else np.zeros((n_selected, n_rot, n_trans), dtype=selected_reconstruction_probs.dtype)
        )
        for selected_row in range(n_selected):
            valid = (
                pair_mask[selected_row]
                & (pair_rows[selected_row] >= 0)
                & (pair_rows[selected_row] < n_rot)
                & (pair_translations[selected_row] >= 0)
                & (pair_translations[selected_row] < n_trans)
            )
            rr = pair_rows[selected_row, valid]
            tt = pair_translations[selected_row, valid]
            if np.unique(rr * n_trans + tt).size != rr.size:
                raise RuntimeError("fused K-class capture encountered duplicate compact candidate pairs")
            dense_scores[selected_row, rr, tt] = selected_scores[selected_row, valid]
            dense_probs[selected_row, rr, tt] = selected_probs[selected_row, valid]
            candidate_mask[selected_row, rr, tt] = True
            if dense_reconstruction_mask is not None:
                dense_reconstruction_mask[selected_row, rr, tt] = selected_reconstruction_mask[
                    selected_row, valid
                ]
            if dense_reconstruction_probs is not None:
                dense_reconstruction_probs[selected_row, rr, tt] = selected_reconstruction_probs[
                    selected_row, valid
                ]

    if dense_reconstruction_probs is None:
        mstep_probs = dense_probs
    else:
        mstep_probs = dense_reconstruction_probs
    if dense_reconstruction_mask is None:
        dense_reconstruction_mask = mstep_probs > 0

    return {
        "image_indices": selected_image_indices,
        "batch_rows": rows,
        "scores": dense_scores,
        "probs": dense_probs,
        "candidate_mask": candidate_mask,
        "reconstruction_mask": dense_reconstruction_mask,
        "reconstruction_probs": mstep_probs,
        "rotation_log_prior": rotation_log_prior,
        "translation_log_prior": _selected(bucket_translation_prior),
        "rotations": _selected(class_bucket_arrays["mstep_rotations"]),
        "rotation_indices": _selected(class_bucket_arrays["rotation_indices"]),
        "actual_counts": _selected(class_bucket_arrays["actual_counts"]).astype(np.int64, copy=False),
    }


def _maybe_dump_bpref_contribution_rows(
    *,
    experiment_dataset,
    image_indices,
    current_size,
    summed,
    ctf_probs,
    rotations,
    actual_counts,
    rotation_indices,
    fine_translations,
    scores,
    preprior_scores,
    probs,
    rotation_log_prior,
    translation_log_prior,
    log_z,
    best_log_score,
    reconstruction_probs,
    reconstruction_mask,
    reconstruction_sum_weight,
    reconstruction_threshold,
    candidate_mask,
    high_precision_operand_bundle,
    raw_batch_data,
    ctf_params,
    noise_variance_half,
    integer_pre_shifts,
    batch_image_corrections,
    batch_scale_corrections,
    relion_preprocess_normalization_factors,
    relion_cuda_preprocess,
    score_with_masked_images,
    image_mask,
    image_mask_mode,
    voxel_size,
    ctf_mode,
    ctf_dose_per_tilt,
    ctf_angle_per_tilt,
    disc_type,
    projection_padding_factor,
    reconstruction_padding_factor,
    use_relion_x_half_mstep,
    winner_take_all,
    max_r,
    window_indices,
    image_shape,
    volume_shape,
    shadow_only_mode,
    shadow_score_bitwise_equal,
    shadow_reduction_agreement,
    device_signature_active: bool | None = None,
    class_index: int = 0,
    mstep_shifted_recon=None,
    mstep_ctf2_over_nv=None,
    inline_projector_data_volumes=None,
    inline_projector_weight_volumes=None,
    reconstruction_group_ids=None,
):
    """Dump posterior-reduced active rows for whole-accumulator scatter replay.

    This diagnostic boundary is immediately before the x-half backprojection.
    Files retain bucket execution order, particle ownership, and every valid
    rotation row, including exact-zero rows.  The companion device signature
    limits only its signature-only output arrays to exact positive-weight
    contributors; its native accumulator launch still receives every row.
    Replaying every contribution file in counter order therefore permits a
    streaming closure check without materializing one 3-D accumulator per
    particle.
    """

    if os.environ.get("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", "").strip() and device_signature_active is not True:
        return
    dump_dir = os.environ.get("RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR")
    if not dump_dir:
        return
    class_index = int(class_index)
    if class_index < 0:
        raise ValueError("BPref contribution class_index must be non-negative")
    global _bpref_contribution_call_counter
    call_idx = _bpref_contribution_call_counter
    _bpref_contribution_call_counter += 1
    context_iteration = int(_bpref_contribution_context["iteration"])
    context_half = int(_bpref_contribution_context["half"])
    target_iteration = os.environ.get("RECOVAR_BPREF_CONTRIBUTION_DUMP_ITERATION")
    if target_iteration and context_iteration != int(target_iteration):
        return
    target_half = os.environ.get("RECOVAR_BPREF_CONTRIBUTION_DUMP_HALF")
    if target_half:
        if int(target_half) not in {1, 2}:
            raise ValueError("RECOVAR_BPREF_CONTRIBUTION_DUMP_HALF must be 1 or 2")
        if context_half != int(target_half):
            return
    target_current_size = os.environ.get("RECOVAR_BPREF_CONTRIBUTION_DUMP_CURRENT_SIZE")
    if target_current_size:
        if current_size is None or int(current_size) != int(target_current_size):
            return

    preprocess_backend_object = image_preprocess_backend(experiment_dataset)
    relion_native_lane_reduction = bool(getattr(preprocess_backend_object, "relion_native_lane_reduction", False))
    if relion_native_lane_reduction and not relion_cuda_preprocess:
        raise ValueError("native-lane preprocessing telemetry requires the RELION CUDA backend")

    local_indices = np.asarray(image_indices, dtype=np.int64)
    original_indices = original_image_indices(experiment_dataset, local_indices)
    image_identities = _bpref_image_identities_for_original_indices(original_indices)
    stack_sha256 = _bpref_required_stack_checksum()
    selected_particle_rows = _bpref_contribution_target_rows(
        experiment_dataset,
        local_indices,
    )
    if os.environ.get("RECOVAR_BPREF_CONTRIBUTION_DUMP_ORIGINAL_INDICES", "").strip():
        if selected_particle_rows.size == 0:
            return
        local_indices = local_indices[selected_particle_rows]
        original_indices = original_indices[selected_particle_rows]
        image_identities = image_identities[selected_particle_rows]

    def _select_particle_axis(values):
        values_np = np.asarray(values)
        if values_np.ndim > 0 and values_np.shape[0] == np.asarray(image_indices).size:
            return values_np[selected_particle_rows]
        return values_np

    actual_counts_np = _select_particle_axis(actual_counts).astype(np.int64, copy=False)
    if reconstruction_group_ids is None:
        captured_reconstruction_group_ids = np.empty((0,), dtype=np.int32)
    else:
        captured_reconstruction_group_ids = _select_particle_axis(
            reconstruction_group_ids
        ).astype(np.int32, copy=False)
        if captured_reconstruction_group_ids.shape != (original_indices.size,):
            raise ValueError(
                "BPref contribution dump reconstruction_group_ids shape mismatch"
            )
        if np.any(captured_reconstruction_group_ids < 0):
            raise ValueError(
                "BPref contribution dump reconstruction_group_ids must be non-negative"
            )
    summed_np = _select_particle_axis(summed)
    ctf_probs_np = _select_particle_axis(ctf_probs)
    rotations_np = _select_particle_axis(rotations)
    if inline_projector_data_volumes is None:
        inline_projector_data_volumes_np = np.empty((0,), dtype=np.complex64)
        inline_projector_weight_volumes_np = np.empty((0,), dtype=np.float32)
    else:
        if inline_projector_weight_volumes is None:
            raise ValueError(
                "inline-projector contribution data requires matching weight volumes"
            )
        inline_projector_data_volumes_np = np.asarray(
            inline_projector_data_volumes,
            dtype=np.complex64,
        )
        inline_projector_weight_volumes_np = np.asarray(
            inline_projector_weight_volumes,
            dtype=np.float32,
        )
        volume_shape_tuple = tuple(int(v) for v in volume_shape)
        expected_volume_shape = (
            original_indices.size,
            volume_shape_tuple[0]
            * volume_shape_tuple[1]
            * (volume_shape_tuple[2] // 2 + 1),
        )
        if inline_projector_data_volumes_np.shape != expected_volume_shape:
            raise ValueError(
                "inline-projector contribution data shape mismatch: "
                f"{inline_projector_data_volumes_np.shape} vs {expected_volume_shape}"
            )
        if inline_projector_weight_volumes_np.shape != expected_volume_shape:
            raise ValueError(
                "inline-projector contribution weight shape mismatch: "
                f"{inline_projector_weight_volumes_np.shape} vs {expected_volume_shape}"
            )
    if summed_np.shape[:2] != ctf_probs_np.shape[:2] or summed_np.shape[:2] != rotations_np.shape[:2]:
        raise ValueError("BPref contribution dump requires matching particle/rotation axes")
    if actual_counts_np.shape != (summed_np.shape[0],):
        raise ValueError("BPref contribution dump actual_counts shape mismatch")

    rotation_rows = np.arange(summed_np.shape[1], dtype=np.int64)[None, :]
    valid = rotation_rows < actual_counts_np[:, None]
    # Preserve every valid rotation row, including exact-zero rows.  A strict
    # RELION/RECOVAR four-arm replay must distinguish a genuine support/value
    # difference from a row silently omitted by the diagnostic writer.
    active = valid
    active_particle_rows, active_rotation_rows = np.nonzero(active)
    rotation_indices_np = _select_particle_axis(rotation_indices).astype(np.int64, copy=False)
    if rotation_indices_np.ndim == 1:
        rotation_indices_np = np.broadcast_to(rotation_indices_np[None, :], summed_np.shape[:2])
    if rotation_indices_np.shape[:2] != summed_np.shape[:2]:
        raise ValueError("BPref contribution dump rotation_indices shape mismatch")

    global _bpref_contribution_dump_counter
    dump_idx = _bpref_contribution_dump_counter
    _bpref_contribution_dump_counter += 1
    path = Path(dump_dir)
    path.mkdir(parents=True, exist_ok=True)
    run_id = os.environ.get("RECOVAR_BPREF_CONTRIBUTION_DUMP_RUN_ID", "unset")
    stack_indices = np.asarray([int(value.split("@", 1)[0]) for value in image_identities], dtype=np.int64)
    stack_paths = np.asarray([value.split("@", 1)[1] for value in image_identities])
    if high_precision_operand_bundle:
        raw_real_images = _select_particle_axis(raw_batch_data)
        if np.iscomplexobj(raw_real_images):
            raise ValueError("BPref raw source images must be real, not Fourier/complex samples")
        expected_raw_shape = (raw_real_images.shape[0], int(image_shape[0]), int(image_shape[1]))
        if raw_real_images.ndim == 2 and raw_real_images.shape[1] == int(np.prod(image_shape)):
            raw_real_images = raw_real_images.reshape(expected_raw_shape)
        if raw_real_images.shape != expected_raw_shape:
            raise ValueError(
                "BPref raw source images must have shape (B,H,W) before FFT/preprocessing, "
                f"got {raw_real_images.shape}, expected {expected_raw_shape}"
            )
        raw_source_dtype = str(raw_real_images.dtype)
        raw_real_images = raw_real_images.astype(np.float32, copy=False)
        captured_ctf_params = _select_particle_axis(ctf_params)
        captured_noise_variance_half = np.asarray(noise_variance_half)
        captured_integer_pre_shifts = _select_particle_axis(integer_pre_shifts).astype(np.int32, copy=False)
        captured_image_corrections = _select_particle_axis(batch_image_corrections).astype(np.float32, copy=False)
        captured_scale_corrections = _select_particle_axis(batch_scale_corrections).astype(np.float32, copy=False)
        captured_normalization_factors = _select_particle_axis(relion_preprocess_normalization_factors).astype(
            np.float32, copy=False
        )
        captured_image_mask = np.asarray(image_mask, dtype=np.float32)
        captured_mstep_shifted_recon = (
            np.empty((0,), dtype=np.complex64)
            if mstep_shifted_recon is None
            else _select_particle_axis(mstep_shifted_recon)
        )
        captured_mstep_ctf2_over_nv = (
            np.empty((0,), dtype=np.float32)
            if mstep_ctf2_over_nv is None
            else _select_particle_axis(mstep_ctf2_over_nv)
        )
    else:
        raw_real_images = np.empty((0,), dtype=np.float32)
        raw_source_dtype = ""
        captured_ctf_params = np.empty((0,), dtype=np.float32)
        captured_noise_variance_half = np.empty((0,), dtype=np.float32)
        captured_integer_pre_shifts = np.empty((0, 2), dtype=np.int32)
        captured_image_corrections = np.empty((0,), dtype=np.float32)
        captured_scale_corrections = np.empty((0,), dtype=np.float32)
        captured_normalization_factors = np.empty((0,), dtype=np.float32)
        captured_image_mask = np.empty((0,), dtype=np.float32)
        captured_mstep_shifted_recon = np.empty((0,), dtype=np.complex64)
        captured_mstep_ctf2_over_nv = np.empty((0,), dtype=np.float32)
    rotation_log_prior_np = _select_particle_axis(rotation_log_prior).astype(np.float64, copy=False)
    translation_log_prior_np = _select_particle_axis(translation_log_prior).astype(np.float64, copy=False)
    combined_scores_np = _select_particle_axis(scores).astype(np.float64, copy=False)
    preprior_scores_np = _select_particle_axis(preprior_scores).astype(np.float64, copy=False)
    best_log_score_np = _select_particle_axis(best_log_score).astype(np.float64, copy=False)
    log_z_np = _select_particle_axis(log_z).astype(np.float64, copy=False)
    normalized_sum_exp = np.exp(log_z_np - best_log_score_np)
    captured_reconstruction_probs = _select_particle_axis(reconstruction_probs)
    if captured_reconstruction_probs.dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError(
            "BPref reconstruction probabilities must retain native float32/float64 dtype, "
            f"got {captured_reconstruction_probs.dtype}"
        )
    scores_f32 = combined_scores_np.astype(np.float32)
    best_f32 = np.max(np.where(np.isfinite(scores_f32), scores_f32, -np.inf), axis=(1, 2))
    exponent_shift_f32 = np.float32(50.0) - best_f32
    shifted_f32 = scores_f32 + exponent_shift_f32[:, None, None]
    raw_exp_weights_f32 = np.where(
        np.isfinite(shifted_f32) & (shifted_f32 >= np.float32(-88.0)),
        np.exp(shifted_f32, dtype=np.float32),
        np.float32(0.0),
    ).astype(np.float32, copy=False)
    contribution_path = path / (
        f"bpref_contribution_rows_it{context_iteration:03d}_h{context_half}"
        f"_call{call_idx:06d}_dump{dump_idx:06d}_cs{int(current_size):03d}.npz"
    )
    np.savez(
        contribution_path,
        magic=np.asarray("RECOVAR_BPREF_CONTRIBUTION_ROWS"),
        schema=np.asarray("recovar-bpref-contribution-rows-v3"),
        schema_version=np.int32(3),
        dump_index=np.int64(dump_idx),
        call_index=np.int64(call_idx),
        iteration=np.int32(context_iteration),
        half=np.int32(context_half),
        rank=np.int32(int(os.environ.get("RECOVAR_BPREF_CONTRIBUTION_RANK", "0"))),
        pass_index=np.int32(2),
        class_index=np.int32(class_index),
        run_id=np.asarray(run_id),
        current_size=np.int64(current_size),
        # ``current_size`` is the scoring window. During fresh firstiter-CC,
        # RELION may keep the BPref/model support one shell smaller. Persist
        # the actual scatter radius so focused replay never infers it from the
        # score window or the odd accumulator shape.
        mstep_max_r=np.float64(np.nan if max_r is None else float(max_r)),
        mstep_current_size=np.int64(-1 if max_r is None else 2 * int(round(float(max_r)))),
        image_shape=np.asarray(image_shape, dtype=np.int32),
        volume_shape=np.asarray(volume_shape, dtype=np.int32),
        window_indices=np.asarray(window_indices, dtype=np.int32),
        local_indices=local_indices,
        original_indices=original_indices,
        star_rows=original_indices,
        image_identities=image_identities,
        stack_indices_1based=stack_indices,
        resolved_stack_paths=stack_paths,
        source_stack_sha256=np.asarray(stack_sha256),
        shadow_only_mode=np.bool_(shadow_only_mode),
        shadow_score_bitwise_equal=np.bool_(shadow_score_bitwise_equal),
        shadow_reduction_data_rel_l1=np.float64(
            np.nan if shadow_reduction_agreement is None else shadow_reduction_agreement["data_rel_l1"]
        ),
        shadow_reduction_data_normalized_max=np.float64(
            np.nan if shadow_reduction_agreement is None else shadow_reduction_agreement["data_normalized_max"]
        ),
        shadow_reduction_weight_rel_l1=np.float64(
            np.nan if shadow_reduction_agreement is None else shadow_reduction_agreement["weight_rel_l1"]
        ),
        shadow_reduction_weight_normalized_max=np.float64(
            np.nan if shadow_reduction_agreement is None else shadow_reduction_agreement["weight_normalized_max"]
        ),
        shadow_reduction_rel_l1_bound=np.float64(
            np.nan if shadow_reduction_agreement is None else shadow_reduction_agreement["rel_l1_bound"]
        ),
        shadow_reduction_normalized_max_bound=np.float64(
            np.nan if shadow_reduction_agreement is None else shadow_reduction_agreement["normalized_max_bound"]
        ),
        high_precision_operand_bundle=np.bool_(high_precision_operand_bundle),
        raw_real_images=raw_real_images,
        raw_source_dtype=np.asarray(raw_source_dtype),
        raw_source_shape=np.asarray(raw_real_images.shape, dtype=np.int64),
        ctf_params=captured_ctf_params,
        ctf_parameter_convention=np.asarray(
            "recovar.CTFParamIndex-v1:DFU[A],DFV[A],DFANG[deg],VOLT[kV],CS[mm],"
            "W[amplitude_fraction],PHASE_SHIFT[deg],BFACTOR[A^2],CONTRAST,DOSE[e-/A^2],TILT_ANGLE[deg]"
        ),
        noise_variance_half=captured_noise_variance_half,
        integer_pre_shifts=captured_integer_pre_shifts,
        image_corrections=captured_image_corrections,
        scale_corrections=captured_scale_corrections,
        relion_preprocess_normalization_factors=captured_normalization_factors,
        relion_cuda_preprocess=np.bool_(relion_cuda_preprocess),
        relion_native_lane_reduction=np.bool_(relion_native_lane_reduction),
        preprocess_backend=np.asarray("relion_cuda" if relion_cuda_preprocess else "dataset_native"),
        preprocess_convention=np.asarray("recovar-half-preprocess-v1"),
        score_with_masked_images=np.bool_(score_with_masked_images),
        image_mask=captured_image_mask,
        image_mask_mode=np.asarray(image_mask_mode),
        voxel_size=np.float64(voxel_size),
        ctf_mode=np.asarray(ctf_mode),
        ctf_dose_per_tilt=np.float64(ctf_dose_per_tilt),
        ctf_angle_per_tilt=np.float64(ctf_angle_per_tilt),
        disc_type=np.asarray(disc_type),
        projection_padding_factor=np.int32(projection_padding_factor),
        reconstruction_padding_factor=np.int32(reconstruction_padding_factor),
        actual_counts=actual_counts_np,
        oversampled_rotation_indices=rotation_indices_np,
        fine_translations=np.asarray(fine_translations),
        candidate_preprior_scores=preprior_scores_np,
        candidate_rotation_log_prior=rotation_log_prior_np,
        candidate_translation_log_prior=translation_log_prior_np,
        candidate_combined_scores=combined_scores_np,
        candidate_best_log_score=best_log_score_np,
        candidate_log_z=log_z_np,
        candidate_normalized_sum_exp=normalized_sum_exp,
        candidate_exponent_shift_f32=exponent_shift_f32,
        candidate_raw_exp_weights_f32=raw_exp_weights_f32,
        posterior_probs=_select_particle_axis(probs).astype(np.float64, copy=False),
        reconstruction_probs=captured_reconstruction_probs,
        reconstruction_probs_native_dtype=np.asarray(str(captured_reconstruction_probs.dtype)),
        reconstruction_probs_native_itemsize=np.int32(captured_reconstruction_probs.dtype.itemsize),
        reconstruction_probs_native_nbytes=np.int64(captured_reconstruction_probs.nbytes),
        # Additive v3 fields: old readers ignore unknown NPZ members, while
        # new high-precision replay fails closed if any member is missing.
        reconstruction_probs_storage_policy=np.asarray("native-dtype-preserved;dtype-itemsize-nbytes-bound"),
        mstep_shifted_recon=captured_mstep_shifted_recon,
        mstep_ctf2_over_nv=captured_mstep_ctf2_over_nv,
        reconstruction_mask=_select_particle_axis(reconstruction_mask).astype(bool, copy=False),
        reconstruction_sum_weight=_select_particle_axis(reconstruction_sum_weight).astype(np.float64, copy=False),
        reconstruction_threshold=_select_particle_axis(reconstruction_threshold).astype(np.float64, copy=False),
        candidate_mask=_select_particle_axis(candidate_mask).astype(bool, copy=False),
        active_particle_rows=active_particle_rows.astype(np.int32, copy=False),
        active_rotation_rows=active_rotation_rows.astype(np.int32, copy=False),
        active_original_indices=original_indices[active_particle_rows],
        reconstruction_group_ids=captured_reconstruction_group_ids,
        active_reconstruction_group_ids=(
            captured_reconstruction_group_ids[active_particle_rows]
            if captured_reconstruction_group_ids.size
            else np.empty((0,), dtype=np.int32)
        ),
        active_global_rotation_indices=rotation_indices_np[active_particle_rows, active_rotation_rows],
        active_summed=summed_np[active_particle_rows, active_rotation_rows],
        active_ctf_probs=ctf_probs_np[active_particle_rows, active_rotation_rows],
        active_rotations=rotations_np[active_particle_rows, active_rotation_rows],
        inline_projector_original_indices=original_indices[
            : inline_projector_data_volumes_np.shape[0]
        ],
        inline_projector_data_volumes=inline_projector_data_volumes_np,
        inline_projector_weight_volumes=inline_projector_weight_volumes_np,
    )

    device_signature_path = None
    device_dump_dir = os.environ.get("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", "").strip()
    if device_dump_dir:
        from recovar import cuda_backproject

        _require_bpref_device_soft_particle_arm(
            use_relion_x_half_mstep=bool(use_relion_x_half_mstep),
        )
        if max_r is None:
            raise RuntimeError("RECOVAR device signature requires the explicit production support radius")
        if context_iteration <= 0 or context_half not in {1, 2}:
            raise RuntimeError("RECOVAR device signature requires explicit positive iteration/half context")
        accumulator_key = (context_iteration, context_half, run_id, int(class_index))
        accumulators = _bpref_device_panel_accumulators.get(accumulator_key)
        accumulator_size = int(volume_shape[0] * volume_shape[1] * (volume_shape[2] // 2 + 1))
        if accumulators is None:
            accumulators = (
                jnp.zeros((accumulator_size,), dtype=jnp.complex64),
                jnp.zeros((accumulator_size,), dtype=jnp.float32),
            )
        launch_ordinal = _bpref_device_panel_launch_counters.get(accumulator_key, 0)
        signature_chunks = [[] for _ in range(7)]
        signature_launch_ordinals = []
        signature_particle_local_rows = []
        signature_image_identities = []
        signature_original_indices = []
        signature_contributor_rotation_keys = []
        particle_launch_ordinals = []
        particle_total_row_counts = []
        particle_contributor_row_counts = []
        particle_noncontributor_row_counts = []
        particle_noncontributor_zero_sha256 = []
        particle_image_identities = []
        particle_original_indices = []
        # Production uses one stream-ordered CUDA launch per particle.  Keep
        # those launch boundaries exact; flattening several particles into one
        # grid changes inter-particle atomic scheduling.
        for particle_row in range(summed_np.shape[0]):
            row_count = int(actual_counts_np[particle_row])
            if row_count <= 0:
                continue
            particle_rotation_keys = np.asarray(rotation_indices_np[particle_row, :row_count], dtype=np.int64)
            if particle_rotation_keys.size and (
                int(particle_rotation_keys.min()) < np.iinfo(np.int32).min
                or int(particle_rotation_keys.max()) > np.iinfo(np.int32).max
            ):
                raise OverflowError("RECOVAR canonical rotation key exceeds device int32 range")
            particle_summed = np.asarray(summed_np[particle_row, :row_count], dtype=np.complex64)
            particle_weights = np.asarray(ctf_probs_np[particle_row, :row_count], dtype=np.float32)
            if not np.all(np.isfinite(particle_summed)) or not np.all(np.isfinite(particle_weights)):
                raise RuntimeError("RECOVAR soft-particle causal arm encountered a nonfinite scatter operand")
            if np.any(particle_weights < 0):
                raise RuntimeError("RECOVAR soft-particle causal arm encountered a negative scatter weight")
            contributor_rows = np.flatnonzero(np.any(particle_weights > 0, axis=1)).astype(np.int32, copy=False)
            noncontributor_mask = np.ones(row_count, dtype=bool)
            noncontributor_mask[contributor_rows] = False
            noncontributor_summed = np.ascontiguousarray(particle_summed[noncontributor_mask])
            noncontributor_weights = np.ascontiguousarray(particle_weights[noncontributor_mask])
            noncontributor_rows = np.flatnonzero(noncontributor_mask).astype(np.int32, copy=False)
            noncontributor_rotation_keys = particle_rotation_keys[noncontributor_mask].astype(np.int32, copy=False)
            if np.any(noncontributor_summed != 0) or np.any(noncontributor_weights != 0):
                raise RuntimeError("RECOVAR device signature requires every omitted signature row to be exactly zero")
            zero_digest = hashlib.sha256()
            zero_digest.update(noncontributor_rows.tobytes(order="C"))
            zero_digest.update(noncontributor_rotation_keys.tobytes(order="C"))
            zero_digest.update(str(noncontributor_summed.dtype).encode("ascii"))
            zero_digest.update(np.asarray(noncontributor_summed.shape, dtype=np.int64).tobytes())
            zero_digest.update(noncontributor_summed.tobytes(order="C"))
            zero_digest.update(str(noncontributor_weights.dtype).encode("ascii"))
            zero_digest.update(np.asarray(noncontributor_weights.shape, dtype=np.int64).tobytes())
            zero_digest.update(noncontributor_weights.tobytes(order="C"))

            ffi_args = (
                accumulators[0],
                accumulators[1],
                jnp.asarray(particle_summed, dtype=jnp.complex64),
                jnp.asarray(particle_weights, dtype=jnp.float32),
                jnp.asarray(window_indices, dtype=jnp.int32),
                jnp.asarray(rotations_np[particle_row, :row_count], dtype=jnp.float32),
            )
            if contributor_rows.size:
                signature_outputs = cuda_backproject.relion_fused_x_half_backproject_signature_indexed(
                    *ffi_args,
                    jnp.asarray(particle_rotation_keys, dtype=jnp.int32),
                    jnp.asarray(contributor_rows, dtype=jnp.int32),
                    tuple(int(value) for value in image_shape),
                    tuple(int(value) for value in volume_shape),
                    float(max_r),
                )
                accumulators = signature_outputs[:2]
                for output_index, output in enumerate(signature_outputs[2:]):
                    signature_chunks[output_index].append(np.asarray(output))
                signature_launch_ordinals.append(np.full(contributor_rows.size, launch_ordinal, dtype=np.int64))
                signature_particle_local_rows.append(contributor_rows)
                signature_image_identities.append(np.full(contributor_rows.size, image_identities[particle_row]))
                signature_original_indices.append(
                    np.full(contributor_rows.size, original_indices[particle_row], dtype=np.int64)
                )
                signature_contributor_rotation_keys.append(
                    particle_rotation_keys[contributor_rows].astype(np.int32, copy=False)
                )
            else:
                # Preserve the native all-row launch even when no row passes
                # its Fweight>0 gate; only the signature-only launch is absent.
                accumulators = cuda_backproject.relion_fused_x_half_backproject_indexed(
                    *ffi_args,
                    tuple(int(value) for value in image_shape),
                    tuple(int(value) for value in volume_shape),
                    float(max_r),
                )
            particle_launch_ordinals.append(launch_ordinal)
            particle_total_row_counts.append(row_count)
            particle_contributor_row_counts.append(int(contributor_rows.size))
            particle_noncontributor_row_counts.append(int(row_count - contributor_rows.size))
            particle_noncontributor_zero_sha256.append(zero_digest.hexdigest())
            particle_image_identities.append(image_identities[particle_row])
            particle_original_indices.append(original_indices[particle_row])
            launch_ordinal += 1
        if not particle_launch_ordinals:
            raise RuntimeError("RECOVAR device signature selected no particle launches")
        _bpref_device_panel_accumulators[accumulator_key] = accumulators
        _bpref_device_panel_launch_counters[accumulator_key] = launch_ordinal
        metadata = {
            "current_size": int(current_size),
            "max_r": float(max_r),
            "image_shape": tuple(int(value) for value in image_shape),
            "volume_shape": tuple(int(value) for value in volume_shape),
            "reconstruction_padding_factor": int(reconstruction_padding_factor),
            "source_stack_sha256": stack_sha256,
            "rank": int(os.environ.get("RECOVAR_BPREF_CONTRIBUTION_RANK", "0")),
            "causal_arm": (
                "winner-take-all-per-particle-fused-xhalf"
                if winner_take_all
                else "soft-posterior-per-particle-fused-xhalf"
            ),
            "winner_take_all": bool(winner_take_all),
            "class_index": int(class_index),
        }
        previous_metadata = _bpref_device_panel_metadata.setdefault(accumulator_key, metadata)
        if previous_metadata != metadata:
            raise RuntimeError("RECOVAR device panel metadata changed within one half")
        dense_height = 2 * int(round(float(max_r)))
        dense_pixel_count = dense_height * (dense_height // 2 + 1)
        if signature_chunks[0]:
            (
                signature_rotation_keys,
                signature_pixel_indices,
                signature_row_flags,
                signature_source_values,
                signature_neighbor_indices,
                signature_neighbor_coefficients,
                signature_neighbor_flags,
            ) = (np.concatenate(chunks, axis=0) for chunks in signature_chunks)
            signature_launch_ordinals = np.concatenate(signature_launch_ordinals)
            signature_particle_local_rows = np.concatenate(signature_particle_local_rows)
            signature_image_identities = np.concatenate(signature_image_identities)
            signature_original_indices = np.concatenate(signature_original_indices)
            signature_contributor_rotation_keys = np.concatenate(signature_contributor_rotation_keys)
        else:
            empty_signature = _empty_bpref_device_signature_arrays(
                dense_pixel_count,
                image_identity_dtype=np.asarray(image_identities).dtype,
            )
            signature_rotation_keys = empty_signature["rotation_keys"]
            signature_pixel_indices = empty_signature["pixel_indices"]
            signature_row_flags = empty_signature["row_flags"]
            signature_source_values = empty_signature["source_values"]
            signature_neighbor_indices = empty_signature["neighbor_indices"]
            signature_neighbor_coefficients = empty_signature["neighbor_coefficients"]
            signature_neighbor_flags = empty_signature["neighbor_flags"]
            signature_launch_ordinals = empty_signature["launch_ordinals"]
            signature_particle_local_rows = empty_signature["particle_local_rows"]
            signature_image_identities = empty_signature["image_identities"]
            signature_original_indices = empty_signature["original_indices"]
            signature_contributor_rotation_keys = empty_signature["contributor_rotation_keys"]
        device_path = Path(device_dump_dir)
        device_path.mkdir(parents=True, exist_ok=True)
        contribution_sha256 = sha256_file(contribution_path)
        device_signature_path = device_path / f"{contribution_path.stem}.device.npz"
        np.savez(
            device_signature_path,
            magic=np.asarray("RECOVAR_DEVICE_SCATTER_SIGNATURE"),
            schema=np.asarray("recovar-device-scatter-signature-v1"),
            schema_version=np.int32(1),
            run_id=np.asarray(run_id),
            iteration=np.int32(context_iteration),
            half=np.int32(context_half),
            rank=np.int32(int(os.environ.get("RECOVAR_BPREF_CONTRIBUTION_RANK", "0"))),
            pass_index=np.int32(2),
            class_index=np.int32(class_index),
            call_index=np.int64(call_idx),
            dump_index=np.int64(dump_idx),
            source_stack_sha256=np.asarray(stack_sha256),
            companion_contribution_path=np.asarray(str(contribution_path.resolve())),
            companion_contribution_sha256=np.asarray(contribution_sha256),
            image_shape=np.asarray(image_shape, dtype=np.int32),
            volume_shape=np.asarray(volume_shape, dtype=np.int32),
            current_size=np.int32(current_size),
            max_r=np.float32(max_r),
            causal_arm=np.asarray(
                "winner-take-all-per-particle-fused-xhalf"
                if winner_take_all
                else "soft-posterior-per-particle-fused-xhalf"
            ),
            winner_take_all=np.bool_(winner_take_all),
            topology_claim=np.asarray("causal-arm-not-relion-hypothesis-arithmetic-closure"),
            signature_inertness_gate=np.asarray("bitwise-post-accum-shadow-and-operand-exact"),
            signature_inertness_gate_passed=np.bool_(True),
            signature_accumulator_shadow_bitwise_equal=np.bool_(True),
            signature_prepared_operands_bitwise_equal=np.bool_(True),
            signature_kernel_accumulate=np.bool_(False),
            reconstruction_padding_factor=np.int32(reconstruction_padding_factor),
            particle_launch_ordinals=np.asarray(particle_launch_ordinals, dtype=np.int64),
            particle_total_row_counts=np.asarray(particle_total_row_counts, dtype=np.int32),
            particle_contributor_row_counts=np.asarray(particle_contributor_row_counts, dtype=np.int32),
            particle_noncontributor_row_counts=np.asarray(particle_noncontributor_row_counts, dtype=np.int32),
            particle_noncontributor_exact_zero=np.ones(len(particle_launch_ordinals), dtype=bool),
            particle_noncontributor_zero_sha256=np.asarray(particle_noncontributor_zero_sha256),
            particle_image_identities=np.asarray(particle_image_identities),
            particle_original_indices=np.asarray(particle_original_indices, dtype=np.int64),
            signature_bytes_per_dense_row_pixel=np.int32(132),
            signature_estimated_uncompressed_bytes=np.int64(
                int(signature_contributor_rotation_keys.size) * dense_pixel_count * 132
            ),
            launch_ordinal=signature_launch_ordinals,
            particle_local_row=signature_particle_local_rows,
            image_identity=signature_image_identities,
            original_indices=signature_original_indices,
            contributor_canonical_rotation_keys=signature_contributor_rotation_keys,
            canonical_rotation_keys=signature_rotation_keys,
            canonical_pixel_indices=signature_pixel_indices,
            row_flags=signature_row_flags,
            source_values=signature_source_values,
            neighbor_indices=signature_neighbor_indices,
            neighbor_coefficients=signature_neighbor_coefficients,
            neighbor_flags=signature_neighbor_flags,
            program_row=signature_particle_local_rows,
            program_lane=np.arange(dense_pixel_count, dtype=np.int32) % np.int32(128),
            program_serial_pass=np.arange(dense_pixel_count, dtype=np.int32) // np.int32(128),
            program_neighbor=np.arange(8, dtype=np.int32),
            program_axis_sizes=np.asarray(
                [
                    int(signature_contributor_rotation_keys.size),
                    dense_pixel_count,
                    8,
                ],
                dtype=np.int64,
            ),
            signature_tensor_axis_legend=np.asarray(
                "row-major [contributor_row,dense_pixel,neighbor]; program_row is the "
                "particle-local source rotation row; lane=dense_pixel%128; "
                "serial_pass=dense_pixel//128; neighbor=d0*4+d1*2+d2"
            ),
            atomic_component_program_order_legend=np.asarray(
                "for each valid neighbor: atomicAdd(data_real), then atomicAdd(data_imag), then atomicAdd(weight)"
            ),
            row_flag_legend=np.asarray(
                "1=redundant-x0;2=2d-radius;4=nonpositive-weight;8=3d-radius;"
                "16=orientation-fold;32=compact-oob;64=reached-scatter"
            ),
            neighbor_flag_legend=np.asarray("1=valid;2=hermitian-fold;4=nyquist;8=oob"),
            source_value_legend=np.asarray("data_re,data_im,Fweight,rk0,rk1,rk2 (pre-orientation-fold)"),
        )
    _maybe_stop_after_bpref_contribution_dump(
        contribution_path=contribution_path,
        device_signature_path=device_signature_path,
    )


def relion_x_half_bp_per_particle_launch_enabled() -> bool:
    """Return whether the diagnostic x-half path launches once per particle."""

    return parse_env_flag(_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH_ENV, default=False)


def relion_x_half_bp_fused_atomics_enabled() -> bool:
    """Return whether the diagnostic fused data/weight scatter is enabled."""

    return parse_env_flag(_RELION_X_HALF_BP_FUSED_ATOMICS_ENV, default=False)


def _scoped_bpref_diagnostic_flags(*, active: bool) -> dict[str, bool]:
    """Resolve process flags against an explicit device-capture boundary."""

    device_signature_configured = bool(os.environ.get("RECOVAR_BPREF_DEVICE_SIGNATURE_DUMP_DIR", "").strip())
    scope_active = bool(active or not device_signature_configured)
    return {
        "device_signature_configured": device_signature_configured,
        "sequential_translation_reduction": bool(
            scope_active and relion_x_half_sequential_translation_reduction_enabled()
        ),
        "per_particle_launches": bool(scope_active and relion_x_half_bp_per_particle_launch_enabled()),
        "fused_atomics": bool(scope_active and relion_x_half_bp_fused_atomics_enabled()),
        "high_precision_operand_bundle": bool(
            scope_active
            and parse_env_flag(
                "RECOVAR_BPREF_HIGH_PRECISION_OPERAND_BUNDLE",
                default=False,
            )
        ),
    }


def _resolve_bpref_execution_modes(
    scoped_diagnostic_flags: dict[str, bool],
    *,
    device_signature_requested: bool,
    production_firstiter_xhalf_topology: bool = False,
) -> dict[str, bool]:
    """Separate requested diagnostic shadows from authoritative live modes."""

    shadow_only = bool(device_signature_requested)
    diagnostic_sequential = bool(scoped_diagnostic_flags["sequential_translation_reduction"])
    diagnostic_per_particle = bool(scoped_diagnostic_flags["per_particle_launches"])
    return {
        "shadow_only": shadow_only,
        "diagnostic_sequential_translation_reduction": diagnostic_sequential,
        "diagnostic_per_particle_launches": diagnostic_per_particle,
        "live_sequential_translation_reduction": bool(
            production_firstiter_xhalf_topology or (diagnostic_sequential and not shadow_only)
        ),
        "live_per_particle_launches": bool(
            production_firstiter_xhalf_topology or (diagnostic_per_particle and not shadow_only)
        ),
    }


def _require_bpref_shadow_exact(label: str, authoritative, shadow) -> None:
    """Fail closed unless a target-only shadow exactly matches live output."""

    authoritative_np = np.asarray(authoritative)
    shadow_np = np.asarray(shadow)
    if authoritative_np.shape != shadow_np.shape or authoritative_np.dtype != shadow_np.dtype:
        raise RuntimeError(
            f"BPref {label} shadow shape/dtype mismatch: "
            f"{authoritative_np.shape}/{authoritative_np.dtype} vs "
            f"{shadow_np.shape}/{shadow_np.dtype}"
        )
    if not np.array_equal(authoritative_np, shadow_np):
        mismatch_count = int(np.count_nonzero(authoritative_np != shadow_np))
        raise RuntimeError(
            f"BPref {label} shadow is not bitwise equal to the authoritative path "
            f"({mismatch_count}/{authoritative_np.size} elements differ)"
        )


def _require_bpref_reduction_shadow_agreement(
    authoritative_summed,
    authoritative_weights,
    shadow_summed,
    shadow_weights,
    *,
    rel_l1_bound: float = 1e-3,
    normalized_max_bound: float = 1e-3,
) -> dict[str, float]:
    """Gate the sequential-f32 diagnostic rows against ordinary live rows."""

    metrics: dict[str, float] = {}
    for label, authoritative, shadow in (
        ("data", authoritative_summed, shadow_summed),
        ("weight", authoritative_weights, shadow_weights),
    ):
        authoritative_np = np.asarray(authoritative)
        shadow_np = np.asarray(shadow)
        if authoritative_np.shape != shadow_np.shape:
            raise RuntimeError(
                f"BPref {label} reduction shadow shape mismatch: {authoritative_np.shape} vs {shadow_np.shape}"
            )
        metric_dtype = (
            np.complex128 if (np.iscomplexobj(authoritative_np) or np.iscomplexobj(shadow_np)) else np.float64
        )
        authoritative_metric = authoritative_np.astype(metric_dtype, copy=False)
        shadow_metric = shadow_np.astype(metric_dtype, copy=False)
        if not np.all(np.isfinite(authoritative_metric)) or not np.all(np.isfinite(shadow_metric)):
            raise RuntimeError(f"BPref {label} reduction shadow contains nonfinite values")
        difference = np.abs(authoritative_metric - shadow_metric)
        scale_l1 = max(float(np.sum(np.abs(authoritative_metric))), np.finfo(np.float64).tiny)
        scale_max = max(float(np.max(np.abs(authoritative_metric), initial=0.0)), np.finfo(np.float64).tiny)
        rel_l1 = float(np.sum(difference) / scale_l1)
        normalized_max = float(np.max(difference, initial=0.0) / scale_max)
        metrics[f"{label}_rel_l1"] = rel_l1
        metrics[f"{label}_normalized_max"] = normalized_max
        if rel_l1 > rel_l1_bound or normalized_max > normalized_max_bound:
            raise RuntimeError(
                f"BPref {label} reduction shadow exceeds the ordinary-path envelope: "
                f"rel_l1={rel_l1:.6g} (bound={rel_l1_bound:.6g}), "
                f"normalized_max={normalized_max:.6g} "
                f"(bound={normalized_max_bound:.6g})"
            )
    metrics["rel_l1_bound"] = float(rel_l1_bound)
    metrics["normalized_max_bound"] = float(normalized_max_bound)
    return metrics


def _require_bpref_device_soft_particle_arm(*, use_relion_x_half_mstep: bool) -> None:
    """Fail closed unless capture shares the explicit soft-particle causal arm.

    This arm is deliberately not called baseline production parity: RECOVAR
    first reduces translations into one row per orientation, whereas RELION may
    scatter orientation-by-translation hypotheses.  It is useful only as a
    controlled causal arm, with ordinary-vs-arm and plain-vs-instrumented
    controls recorded separately.
    """

    if not use_relion_x_half_mstep:
        raise RuntimeError("RECOVAR device signature requires the RELION x-half M-step")
    if not relion_x_half_bp_per_particle_launch_enabled():
        raise RuntimeError("RECOVAR device signature requires RECOVAR_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH=1")
    if not relion_x_half_sequential_translation_reduction_enabled():
        raise RuntimeError("RECOVAR device signature requires RECOVAR_RELION_X_HALF_SEQUENTIAL_TRANSLATION_REDUCTION=1")
    if not relion_x_half_bp_fused_atomics_enabled():
        raise RuntimeError("RECOVAR device signature requires RECOVAR_RELION_X_HALF_BP_FUSED_ATOMICS=1")
    from recovar import cuda_backproject

    if not cuda_backproject.relion_x_half_bp_block_topology_requested():
        raise RuntimeError("RECOVAR device signature requires RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY=1")


def _bpref_membership_dump_requested():
    dump_dir = os.environ.get(_BPREF_MEMBERSHIP_DUMP_DIR_ENV, "").strip()
    if not dump_dir:
        return False
    context_iteration = int(_bpref_contribution_context["iteration"])
    context_half = int(_bpref_contribution_context["half"])
    target_iteration = os.environ.get(_BPREF_MEMBERSHIP_DUMP_ITERATION_ENV)
    if target_iteration and context_iteration != int(target_iteration):
        return False
    target_half = os.environ.get(_BPREF_MEMBERSHIP_DUMP_HALF_ENV)
    if target_half:
        if int(target_half) not in {1, 2}:
            raise ValueError(f"{_BPREF_MEMBERSHIP_DUMP_HALF_ENV} must be 1 or 2")
        if context_half != int(target_half):
            return False
    return True


def _maybe_dump_k1_bpref_rotation_mass(
    *,
    experiment_dataset,
    image_indices,
    current_size,
    actual_counts,
    rotations,
    rotation_indices,
    candidate_translation_count,
    posterior_rotation_mass,
    reconstruction_rotation_mass,
    significant_translation_count,
    reconstruction_sum_weight,
    reconstruction_threshold,
):
    """Dump the sufficient per-rotation inputs to the BPref denominator."""

    if not _bpref_membership_dump_requested():
        return
    dump_dir = os.environ[_BPREF_MEMBERSHIP_DUMP_DIR_ENV].strip()
    context_iteration = int(_bpref_contribution_context["iteration"])
    context_half = int(_bpref_contribution_context["half"])

    local_indices = np.asarray(image_indices, dtype=np.int64)
    original_indices = original_image_indices(experiment_dataset, local_indices)
    counts = np.asarray(actual_counts, dtype=np.int64)
    rotations_np = np.asarray(rotations, dtype=np.float32)
    rotation_indices_np = np.asarray(rotation_indices, dtype=np.int64)
    candidate_count_np = np.asarray(candidate_translation_count, dtype=np.int32)
    posterior_mass_np = np.asarray(posterior_rotation_mass)
    reconstruction_mass_np = np.asarray(reconstruction_rotation_mass)
    significant_count_np = np.asarray(significant_translation_count, dtype=np.int32)
    sum_weight_np = np.asarray(reconstruction_sum_weight)
    threshold_np = np.asarray(reconstruction_threshold)

    batch = local_indices.size
    topology = posterior_mass_np.shape
    if counts.shape != (batch,) or len(topology) != 2 or topology[0] != batch:
        raise ValueError("BPref rotation-mass topology mismatch")
    if (
        reconstruction_mass_np.shape != topology
        or candidate_count_np.shape != topology
        or significant_count_np.shape != topology
    ):
        raise ValueError("BPref rotation-mass arrays have inconsistent topology")
    if rotations_np.shape != (*topology, 3, 3):
        raise ValueError("BPref rotation-mass rotation topology mismatch")
    if rotation_indices_np.ndim == 1:
        rotation_indices_np = np.broadcast_to(rotation_indices_np[None, :], topology)
    if rotation_indices_np.shape != topology:
        raise ValueError("BPref rotation-mass index topology mismatch")
    if np.any(counts < 0) or np.any(counts > topology[1]):
        raise ValueError("BPref rotation counts are outside the padded rotation axis")
    if np.any(candidate_count_np < 0) or np.any(significant_count_np < 0):
        raise ValueError("BPref translation counts are negative")
    if np.any(significant_count_np > candidate_count_np):
        raise ValueError("BPref significant translations exceed candidate translations")
    if np.any(posterior_mass_np < 0) or np.any(reconstruction_mass_np < 0):
        raise ValueError("BPref rotation masses are negative")
    if np.any(reconstruction_mass_np > posterior_mass_np + np.finfo(np.float32).eps):
        raise ValueError("BPref reconstruction mass exceeds posterior mass")
    padded = np.arange(topology[1])[None, :] >= counts[:, None]
    if (
        np.any(candidate_count_np[padded])
        or np.any(significant_count_np[padded])
        or np.any(posterior_mass_np[padded])
        or np.any(reconstruction_mass_np[padded])
    ):
        raise ValueError("BPref padded rotations carry membership or mass")
    if np.max(candidate_count_np, initial=0) > np.iinfo(np.uint16).max:
        raise ValueError("BPref candidate translation count exceeds uint16")

    global _bpref_membership_dump_counter
    dump_index = _bpref_membership_dump_counter
    _bpref_membership_dump_counter += 1
    path = Path(dump_dir)
    path.mkdir(parents=True, exist_ok=True)
    output = path / (
        f"bpref_membership_it{context_iteration:03d}_h{context_half}"
        f"_dump{dump_index:06d}_cs{int(current_size):03d}.npz"
    )
    np.savez(
        output,
        schema=np.asarray("recovar-bpref-rotation-mass-v2"),
        iteration=np.int32(context_iteration),
        half=np.int32(context_half),
        current_size=np.int32(current_size),
        local_indices=local_indices,
        original_indices=original_indices,
        stack_indices_1based=original_indices + 1,
        actual_counts=counts,
        rotations=rotations,
        rotation_indices=rotation_indices_np,
        candidate_translation_count=candidate_count_np.astype(np.uint16),
        posterior_rotation_mass=posterior_mass_np,
        reconstruction_rotation_mass=reconstruction_mass_np,
        significant_translation_count=significant_count_np.astype(np.uint16),
        reconstruction_sum_weight=sum_weight_np,
        reconstruction_threshold=threshold_np,
    )


def _maybe_dump_k1_bpref_membership(
    *,
    experiment_dataset,
    image_indices,
    current_size,
    actual_counts,
    rotations,
    rotation_indices,
    fine_translations,
    candidate_mask,
    posterior_probs,
    reconstruction_probs,
    reconstruction_mask,
    reconstruction_sum_weight,
    reconstruction_threshold,
):
    """Collapse a rectangular fine posterior to sufficient rotation masses."""

    if not _bpref_membership_dump_requested():
        return
    candidate_mask_np = np.asarray(candidate_mask, dtype=bool)
    posterior_np = np.asarray(posterior_probs)
    reconstruction_np = np.asarray(reconstruction_probs)
    reconstruction_mask_np = np.asarray(reconstruction_mask, dtype=bool)
    if posterior_np.ndim != 3:
        raise ValueError("BPref membership posterior topology mismatch")
    if reconstruction_np.shape != posterior_np.shape:
        raise ValueError("BPref membership reconstruction-posterior shape mismatch")
    if candidate_mask_np.shape != posterior_np.shape:
        raise ValueError("BPref membership candidate-mask shape mismatch")
    if reconstruction_mask_np.shape != posterior_np.shape:
        raise ValueError("BPref membership reconstruction-mask shape mismatch")
    if not np.array_equal(reconstruction_mask_np, reconstruction_np > 0):
        raise ValueError("BPref membership mask does not equal positive reconstruction posterior")
    _maybe_dump_k1_bpref_rotation_mass(
        experiment_dataset=experiment_dataset,
        image_indices=image_indices,
        current_size=current_size,
        actual_counts=actual_counts,
        rotations=rotations,
        rotation_indices=rotation_indices,
        candidate_translation_count=np.sum(candidate_mask_np, axis=-1, dtype=np.int32),
        posterior_rotation_mass=np.sum(posterior_np, axis=-1),
        reconstruction_rotation_mass=np.sum(reconstruction_np, axis=-1),
        significant_translation_count=np.sum(reconstruction_mask_np, axis=-1, dtype=np.int32),
        reconstruction_sum_weight=reconstruction_sum_weight,
        reconstruction_threshold=reconstruction_threshold,
    )

#!/usr/bin/env python3
"""Compare authoritative native and RECOVAR K=4 raw fine-pass costs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_em_k4_authoritative_native_scores import (
    EXPECTED_CLASS,
    EXPECTED_CURRENT_SIZE,
    EXPECTED_ITERATION,
    EXPECTED_PARTICLE_ID,
    EXPECTED_ROTATIONS,
    EXPECTED_STACK,
    EXPECTED_SUPPORT,
    TARGET_GPU_UUID,
    TARGET_RECOVAR_ROTATION,
    TARGET_TRANSLATIONS,
    _read_allocation_table,
    _rotation_permutation,
    _validate_completion,
    float32_metric,
)
from scripts.validate_relion_bpref_factor_capture import load_factor_capture
from scripts.validate_relion_fine_score_capture import (
    ACTIVE,
    load_fine_score_capture,
)

SCHEMA = "relion-k4-it2-raw-diff2-parity-v3"
PASS_CLASSIFICATION = "exact_device_k4_raw_diff2_and_common_min_bitwise_match"
PASS_SCORE_CLASSIFICATION = (
    "exact_device_k4_raw_priors_and_combined_scores_bitwise_match"
)
RECOVAR_CAPTURE_HEAD = "ec68f651a4408ed14ed7ebce0ddf3d54a74e0d41"
RECOVAR_CAPTURE_SCHEMA = "recovar-k4-it2-selected-raw-diff2-job-v1"
NATIVE_SCIENCE_JOB_ID = 11_787_017


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _float32_from_bits(value: int) -> np.float32:
    return np.asarray(np.uint32(value)).view(np.float32)[()]


def classify_raw_diff2_parity(
    *,
    support_exact: bool,
    common_min_bitwise_exact: bool,
    raw_diff2_bitwise_exact: bool,
    centered_pre_prior_bitwise_exact: bool,
    native_target_tied: bool,
    recovar_target_tied: bool,
) -> str:
    gates = {
        "support": support_exact,
        "common_min": common_min_bitwise_exact,
        "raw_diff2": raw_diff2_bitwise_exact,
        "centered_pre_prior": centered_pre_prior_bitwise_exact,
        "native_target_tie": native_target_tied,
        "recovar_target_tie": recovar_target_tied,
    }
    failures = [name for name, passed in gates.items() if not passed]
    if not failures:
        return PASS_CLASSIFICATION
    return "exact_device_k4_raw_diff2_mismatch__" + "__".join(failures)


def classify_score_path_parity(
    *,
    support_exact: bool,
    rotation_prior_bitwise_exact: bool,
    translation_prior_bitwise_exact: bool,
    saved_score_replay_bitwise_exact: bool,
    combined_score_bitwise_exact: bool,
    maximum_tie_sets_exact: bool,
) -> str:
    gates = {
        "support": support_exact,
        "rotation_prior": rotation_prior_bitwise_exact,
        "translation_prior": translation_prior_bitwise_exact,
        "saved_score_replay": saved_score_replay_bitwise_exact,
        "combined_score": combined_score_bitwise_exact,
        "maximum_tie_sets": maximum_tie_sets_exact,
    }
    failures = [name for name, passed in gates.items() if not passed]
    if not failures:
        return PASS_SCORE_CLASSIFICATION
    return "exact_device_k4_score_path_mismatch__" + "__".join(failures)


def _relion_score_replay(
    raw_diff2: np.ndarray,
    rotation_prior: np.ndarray,
    translation_prior: np.ndarray,
    min_diff2: np.float32,
) -> np.ndarray:
    """Replay RELION's float32 prior/min/raw operation order."""

    return np.subtract(
        np.add(
            np.add(
                np.asarray(rotation_prior, dtype=np.float32),
                np.asarray(translation_prior, dtype=np.float32),
                dtype=np.float32,
            ),
            np.float32(min_diff2),
            dtype=np.float32,
        ),
        np.asarray(raw_diff2, dtype=np.float32),
        dtype=np.float32,
    )


def _raw_mismatch_strata(
    *,
    native_raw: np.ndarray,
    recovar_raw: np.ndarray,
    native_candidate_index: np.ndarray,
    native_rotation_local: np.ndarray,
    mapped_recovar_rotation: np.ndarray,
    translation_id: np.ndarray,
) -> dict[str, Any]:
    """Partition bitwise raw-cost mismatches by fixed rotation/translation."""

    native_raw = np.asarray(native_raw, dtype=np.float32)
    recovar_raw = np.asarray(recovar_raw, dtype=np.float32)
    native_candidate_index = np.asarray(
        native_candidate_index,
        dtype=np.int64,
    )
    native_rotation_local = np.asarray(
        native_rotation_local,
        dtype=np.int64,
    )
    mapped_recovar_rotation = np.asarray(
        mapped_recovar_rotation,
        dtype=np.int64,
    )
    translation_id = np.asarray(translation_id, dtype=np.int64)
    arrays = (
        native_raw,
        recovar_raw,
        native_candidate_index,
        native_rotation_local,
        mapped_recovar_rotation,
        translation_id,
    )
    _require(
        len({array.shape for array in arrays}) == 1
        and native_raw.ndim == 1
        and native_raw.size > 0,
        "raw-mismatch arrays must be nonempty aligned vectors",
    )
    _require(
        np.unique(native_candidate_index).size == native_candidate_index.size,
        "native candidate indices must be unique",
    )
    _require(
        np.all(np.isfinite(native_raw))
        and np.all(np.isfinite(recovar_raw))
        and np.all(native_raw >= 0)
        and np.all(recovar_raw >= 0),
        "raw costs must be finite nonnegative float32 values",
    )
    _require(
        np.all(native_rotation_local >= 0)
        and np.all(mapped_recovar_rotation >= 0)
        and np.all(translation_id >= 0),
        "raw-mismatch identities must be nonnegative",
    )

    native_bits = native_raw.view(np.uint32)
    recovar_bits = recovar_raw.view(np.uint32)
    mismatch = native_bits != recovar_bits
    delta = np.asarray(
        recovar_raw.astype(np.float64) - native_raw.astype(np.float64),
        dtype=np.float64,
    )
    mismatch_rows = np.flatnonzero(mismatch)
    mismatch_rows = mismatch_rows[
        np.argsort(
            native_candidate_index[mismatch_rows],
            kind="stable",
        )
    ]
    mismatch_count = int(mismatch_rows.size)
    signed_delta = math.fsum(float(delta[row]) for row in mismatch_rows)
    delta_l1 = math.fsum(abs(float(delta[row])) for row in mismatch_rows)
    maximum_abs_delta = float(
        0.0
        if mismatch_count == 0
        else np.max(np.abs(delta[mismatch_rows]))
    )

    def candidate_record(row: int) -> dict[str, Any]:
        native_value = native_raw[row]
        recovar_value = recovar_raw[row]
        return {
            "native_candidate_index": int(native_candidate_index[row]),
            "native_rotation_local": int(native_rotation_local[row]),
            "mapped_recovar_rotation": int(mapped_recovar_rotation[row]),
            "translation_id": int(translation_id[row]),
            "native_raw_diff2": float(native_value),
            "native_raw_diff2_bits": int(native_bits[row]),
            "recovar_raw_diff2": float(recovar_value),
            "recovar_raw_diff2_bits": int(recovar_bits[row]),
            "delta_recovar_minus_native": float(delta[row]),
            "absolute_delta": float(abs(delta[row])),
            "ulp_distance": abs(
                int(recovar_bits[row]) - int(native_bits[row])
            ),
        }

    def partition(
        identity: np.ndarray,
        identity_name: str,
        *,
        paired_identity: np.ndarray | None = None,
        paired_identity_name: str | None = None,
    ) -> dict[str, Any]:
        records = []
        flattened_rows = []
        for identity_value in np.unique(identity):
            identity_value = int(identity_value)
            active_rows = np.flatnonzero(identity == identity_value)
            rows = active_rows[mismatch[active_rows]]
            rows = rows[
                np.argsort(
                    native_candidate_index[rows],
                    kind="stable",
                )
            ]
            flattened_rows.extend(int(row) for row in rows)
            stratum_signed = math.fsum(float(delta[row]) for row in rows)
            stratum_l1 = math.fsum(
                abs(float(delta[row])) for row in rows
            )
            if rows.size:
                absolute = np.abs(delta[rows])
                stratum_max = float(np.max(absolute))
                representatives = rows[absolute == stratum_max]
                representative_row = int(
                    representatives[
                        np.argmin(
                            native_candidate_index[representatives]
                        )
                    ]
                )
                representative = candidate_record(representative_row)
            else:
                stratum_max = 0.0
                representative = None
            record = {
                identity_name: identity_value,
                "active_candidate_count": int(active_rows.size),
                "mismatch_count": int(rows.size),
                "mismatch_fraction": float(
                    rows.size / active_rows.size
                ),
                "signed_raw_delta": float(stratum_signed),
                "raw_delta_l1": float(stratum_l1),
                "maximum_absolute_raw_delta": stratum_max,
                "positive_mismatch_count": int(
                    np.count_nonzero(delta[rows] > 0)
                ),
                "negative_mismatch_count": int(
                    np.count_nonzero(delta[rows] < 0)
                ),
                "zero_delta_bitwise_mismatch_count": int(
                    np.count_nonzero(delta[rows] == 0)
                ),
                "maximum_absolute_representative": representative,
            }
            if paired_identity is not None:
                paired_values = np.unique(paired_identity[active_rows])
                _require(
                    paired_identity_name is not None
                    and paired_values.size == 1,
                    f"{identity_name} does not map to one "
                    f"{paired_identity_name}",
                )
                record[paired_identity_name] = int(paired_values[0])
            else:
                _require(
                    paired_identity_name is None,
                    f"{identity_name} paired identity lacks values",
                )
            records.append(record)

        flattened_signed = math.fsum(
            float(delta[row]) for row in flattened_rows
        )
        flattened_l1 = math.fsum(
            abs(float(delta[row])) for row in flattened_rows
        )
        _require(
            len(flattened_rows) == mismatch_count
            and flattened_signed == signed_delta
            and flattened_l1 == delta_l1,
            f"{identity_name} mismatch partition does not replay globally",
        )
        ranked = sorted(
            records,
            key=lambda record: (
                -record["raw_delta_l1"],
                record[identity_name],
            ),
        )
        for rank, record in enumerate(ranked, start=1):
            record["raw_delta_l1_rank_1based"] = rank

        def concentration(top_n: int) -> dict[str, Any]:
            contribution = math.fsum(
                float(record["raw_delta_l1"])
                for record in ranked[:top_n]
            )
            return {
                "requested_top_n": top_n,
                "available_strata_used": min(top_n, len(ranked)),
                "raw_delta_l1": float(contribution),
                "share_of_raw_delta_l1": float(
                    0.0 if delta_l1 == 0.0 else contribution / delta_l1
                ),
            }

        rounded_signed = math.fsum(
            float(record["signed_raw_delta"]) for record in records
        )
        rounded_l1 = math.fsum(
            float(record["raw_delta_l1"]) for record in records
        )
        return {
            "stratum_identity": identity_name,
            "group_count": len(records),
            "active_candidate_count": int(identity.size),
            "mismatch_count": mismatch_count,
            "ranking_rule": (
                f"descending_raw_delta_l1_then_ascending_{identity_name}"
            ),
            "flattened_partition_signed_replay": float(
                flattened_signed
            ),
            "flattened_partition_l1_replay": float(flattened_l1),
            "rounded_group_signed_sum": float(rounded_signed),
            "rounded_group_signed_replay_residual": float(
                rounded_signed - signed_delta
            ),
            "rounded_group_l1_sum": float(rounded_l1),
            "rounded_group_l1_replay_residual": float(
                rounded_l1 - delta_l1
            ),
            "l1_concentration": {
                "top_1": concentration(1),
                "top_3": concentration(3),
                "top_10": concentration(10),
            },
            "top_10": ranked[:10],
            "all_strata": records,
        }

    rotation = partition(
        mapped_recovar_rotation,
        "mapped_recovar_rotation",
        paired_identity=native_rotation_local,
        paired_identity_name="native_rotation_local",
    )
    translation = partition(translation_id, "translation_id")

    if mismatch_count:
        selected_rotation = int(
            rotation["top_10"][0]["mapped_recovar_rotation"]
        )
        eligible = np.flatnonzero(
            mismatch & (mapped_recovar_rotation == selected_rotation)
        )
        absolute = np.abs(delta[eligible])
        largest = np.max(absolute)
        representatives = eligible[absolute == largest]
        selected_row = int(
            representatives[
                np.argmin(native_candidate_index[representatives])
            ]
        )
        representative = candidate_record(selected_row)
        representative["selection_rule"] = (
            "top_rotation_by_descending_mismatch_raw_delta_l1_then_"
            "ascending_rotation; within_rotation_largest_absolute_raw_"
            "delta_then_lowest_native_candidate_index"
        )
    else:
        representative = None

    return {
        "active_candidate_count": int(native_raw.size),
        "mismatch_count": mismatch_count,
        "mismatch_fraction": float(mismatch_count / native_raw.size),
        "signed_raw_delta": float(signed_delta),
        "raw_delta_l1": float(delta_l1),
        "maximum_absolute_raw_delta": maximum_abs_delta,
        "positive_mismatch_count": int(
            np.count_nonzero(delta[mismatch_rows] > 0)
        ),
        "negative_mismatch_count": int(
            np.count_nonzero(delta[mismatch_rows] < 0)
        ),
        "zero_delta_bitwise_mismatch_count": int(
            np.count_nonzero(delta[mismatch_rows] == 0)
        ),
        "rotation_strata": rotation,
        "translation_strata": translation,
        "selected_representative": representative,
        "partition_replay_exact": True,
    }


def _validate_recovar_completion(
    path: Path,
    *,
    expected_job_id: int,
) -> dict[str, Any]:
    report = json.loads(path.read_text())
    _require(
        report.get("schema") == RECOVAR_CAPTURE_SCHEMA,
        "unexpected RECOVAR raw-diff2 completion schema",
    )
    _require(report.get("status") == "complete", "RECOVAR raw-diff2 capture is incomplete")
    _require(
        int(report.get("slurm_job_id")) == expected_job_id,
        "RECOVAR raw-diff2 Slurm identity changed",
    )
    _require(
        report.get("integration_head") == RECOVAR_CAPTURE_HEAD,
        "RECOVAR raw-diff2 source commit changed",
    )
    _require(
        report.get("gpu_uuid") == TARGET_GPU_UUID,
        "RECOVAR raw-diff2 GPU UUID changed",
    )
    _require(
        report.get("grid_correction") == "unset_default_off"
        and report.get("final_all_data_after_max_iter") == "unset",
        "RECOVAR raw-diff2 grid/finalization contract changed",
    )
    _require(
        report.get("scorecard_change_admissible") is False,
        "RECOVAR raw-diff2 completion incorrectly permits a scorecard change",
    )
    return report


def _comparison(
    *,
    factor_path: Path,
    fine_score_path: Path,
    recovar_pass2_path: Path,
) -> dict[str, Any]:
    factor = load_factor_capture(factor_path)
    score = load_fine_score_capture(fine_score_path)
    _require(factor.geometry_only, "expected geometry-only BPref capture")
    _require(factor.stack_index == EXPECTED_STACK, "factor stack changed")
    _require(
        score.header[4:8]
        == (
            EXPECTED_ITERATION,
            EXPECTED_CLASS,
            EXPECTED_PARTICLE_ID,
            EXPECTED_STACK,
        ),
        "native fine-score identity changed",
    )
    with np.load(recovar_pass2_path, allow_pickle=False) as archive:
        recovar = {key: np.asarray(archive[key]) for key in archive.files}
    required = {
        "original_index",
        "class_index",
        "current_size",
        "rotations",
        "candidate_mask",
        "relion_raw_diff2",
        "relion_min_diff2",
        "rotation_log_prior",
        "translation_log_prior",
        "scores_with_prior",
    }
    _require(required.issubset(recovar), "RECOVAR raw-diff2 artifact schema is incomplete")
    _require(int(recovar["original_index"]) == 53_722, "RECOVAR particle changed")
    _require(
        int(recovar["class_index"]) == 0
        and int(recovar["current_size"]) == EXPECTED_CURRENT_SIZE,
        "RECOVAR class/current-size identity changed",
    )

    native_rotations = (
        np.asarray(factor.rotations["matrix"], dtype=np.float32)
        .reshape(-1, 3, 3)
        .transpose(0, 2, 1)
    )
    native_to_recovar = _rotation_permutation(
        native_rotations,
        np.asarray(recovar["rotations"], dtype=np.float32),
    )

    candidates = score.candidates
    active = (candidates["flags"] & ACTIVE) != 0
    native_rotation = np.asarray(candidates["rotation_local"], dtype=np.int64)
    native_translation = np.asarray(candidates["translation_id"], dtype=np.int64)
    mapped_rotation = native_to_recovar[native_rotation]
    candidate_mask = np.asarray(recovar["candidate_mask"], dtype=bool)
    recovar_raw_table = np.asarray(recovar["relion_raw_diff2"], dtype=np.float32)
    recovar_score_table = np.asarray(recovar["scores_with_prior"], dtype=np.float32)
    recovar_rotation_prior = np.asarray(
        recovar["rotation_log_prior"],
        dtype=np.float32,
    )
    recovar_translation_prior = np.asarray(
        recovar["translation_log_prior"],
        dtype=np.float32,
    )
    _require(
        recovar_raw_table.shape
        == recovar_score_table.shape
        == candidate_mask.shape,
        "RECOVAR raw/score/candidate-mask shapes differ",
    )
    _require(
        recovar_rotation_prior.shape == (candidate_mask.shape[0],)
        and recovar_translation_prior.shape == (candidate_mask.shape[1],),
        "RECOVAR prior-table shapes differ from the candidate table",
    )
    _require(
        np.all(np.isfinite(recovar_raw_table[candidate_mask]))
        and np.all(np.isfinite(recovar_score_table[candidate_mask])),
        "RECOVAR active raw diff2 or score contains non-finite values",
    )

    native_support = np.zeros(candidate_mask.shape, dtype=bool)
    native_support[
        mapped_rotation[active],
        native_translation[active],
    ] = True
    _require(
        np.count_nonzero(native_support) == EXPECTED_SUPPORT
        and np.count_nonzero(candidate_mask) == EXPECTED_SUPPORT,
        "fixed active-support denominator changed",
    )
    support_exact = bool(np.array_equal(native_support, candidate_mask))

    active_candidate_indices = np.flatnonzero(active)
    native_raw = np.asarray(candidates["raw_diff2"][active], dtype=np.float32)
    recovar_raw = np.asarray(
        recovar_raw_table[
            mapped_rotation[active],
            native_translation[active],
        ],
        dtype=np.float32,
    )
    native_min = _float32_from_bits(score.header[18])
    recovar_min = np.float32(recovar["relion_min_diff2"])
    native_centered = np.subtract(native_min, native_raw, dtype=np.float32)
    recovar_centered = np.subtract(recovar_min, recovar_raw, dtype=np.float32)
    common_min_exact = bool(
        native_min.view(np.uint32) == recovar_min.view(np.uint32)
    )
    raw_metric = float32_metric(native_raw, recovar_raw)
    centered_metric = float32_metric(native_centered, recovar_centered)
    raw_mismatch_strata = _raw_mismatch_strata(
        native_raw=native_raw,
        recovar_raw=recovar_raw,
        native_candidate_index=active_candidate_indices,
        native_rotation_local=native_rotation[active],
        mapped_recovar_rotation=mapped_rotation[active],
        translation_id=native_translation[active],
    )

    native_rotation_prior = np.asarray(
        candidates["orientation_log_prior"][active],
        dtype=np.float32,
    )
    native_translation_prior = np.asarray(
        candidates["translation_log_prior"][active],
        dtype=np.float32,
    )
    native_combined = np.asarray(
        candidates["combined_preexponent"][active],
        dtype=np.float32,
    )
    recovar_rotation_prior_active = recovar_rotation_prior[
        mapped_rotation[active]
    ]
    recovar_translation_prior_active = recovar_translation_prior[
        native_translation[active]
    ]
    recovar_saved_score = recovar_score_table[
        mapped_rotation[active],
        native_translation[active],
    ]
    recovar_replay_score = _relion_score_replay(
        recovar_raw,
        recovar_rotation_prior_active,
        recovar_translation_prior_active,
        recovar_min,
    )
    rotation_prior_metric = float32_metric(
        native_rotation_prior,
        recovar_rotation_prior_active,
    )
    translation_prior_metric = float32_metric(
        native_translation_prior,
        recovar_translation_prior_active,
    )
    saved_score_replay_metric = float32_metric(
        recovar_saved_score,
        recovar_replay_score,
    )
    combined_score_metric = float32_metric(
        native_combined,
        recovar_replay_score,
    )
    native_maximum = np.max(native_combined)
    recovar_maximum = np.max(recovar_replay_score)
    native_maximum_ties = native_combined.view(np.uint32) == native_maximum.view(
        np.uint32
    )
    recovar_maximum_ties = (
        recovar_replay_score.view(np.uint32)
        == recovar_maximum.view(np.uint32)
    )
    maximum_tie_sets_exact = bool(
        np.array_equal(native_maximum_ties, recovar_maximum_ties)
    )
    score_path_classification = classify_score_path_parity(
        support_exact=support_exact,
        rotation_prior_bitwise_exact=rotation_prior_metric["bitwise_exact"],
        translation_prior_bitwise_exact=translation_prior_metric[
            "bitwise_exact"
        ],
        saved_score_replay_bitwise_exact=saved_score_replay_metric[
            "bitwise_exact"
        ],
        combined_score_bitwise_exact=combined_score_metric["bitwise_exact"],
        maximum_tie_sets_exact=maximum_tie_sets_exact,
    )

    inverse_target = np.flatnonzero(
        native_to_recovar == TARGET_RECOVAR_ROTATION
    )
    _require(
        inverse_target.size == 1,
        "target RECOVAR rotation does not have one native match",
    )
    target_native_rotation = int(inverse_target[0])
    target_records = []
    for translation in TARGET_TRANSLATIONS:
        matches = np.flatnonzero(
            (native_rotation[active] == target_native_rotation)
            & (native_translation[active] == translation)
        )
        _require(
            matches.size == 1,
            f"target translation {translation} does not have one native row",
        )
        active_position = int(matches[0])
        candidate_index = int(active_candidate_indices[active_position])
        native_value = np.float32(candidates[candidate_index]["raw_diff2"])
        native_combined_value = np.float32(
            candidates[candidate_index]["combined_preexponent"]
        )
        recovar_value = np.float32(
            recovar_raw_table[TARGET_RECOVAR_ROTATION, translation]
        )
        recovar_combined_value = np.float32(
            recovar_replay_score[active_position]
        )
        target_records.append(
            {
                "translation_id": translation,
                "native_raw_diff2": float(native_value),
                "native_raw_diff2_bits": int(native_value.view(np.uint32)),
                "recovar_raw_diff2": float(recovar_value),
                "recovar_raw_diff2_bits": int(recovar_value.view(np.uint32)),
                "delta_recovar_minus_native": float(
                    np.float64(recovar_value) - np.float64(native_value)
                ),
                "native_combined_score": float(native_combined_value),
                "native_combined_score_bits": int(
                    native_combined_value.view(np.uint32)
                ),
                "recovar_replayed_combined_score": float(
                    recovar_combined_value
                ),
                "recovar_replayed_combined_score_bits": int(
                    recovar_combined_value.view(np.uint32)
                ),
            }
        )
    native_target_tied = bool(
        target_records[0]["native_raw_diff2_bits"]
        == target_records[1]["native_raw_diff2_bits"]
    )
    recovar_target_tied = bool(
        target_records[0]["recovar_raw_diff2_bits"]
        == target_records[1]["recovar_raw_diff2_bits"]
    )
    classification = classify_raw_diff2_parity(
        support_exact=support_exact,
        common_min_bitwise_exact=common_min_exact,
        raw_diff2_bitwise_exact=raw_metric["bitwise_exact"],
        centered_pre_prior_bitwise_exact=centered_metric["bitwise_exact"],
        native_target_tied=native_target_tied,
        recovar_target_tied=recovar_target_tied,
    )
    return {
        "classification": classification,
        "accepted": classification == PASS_CLASSIFICATION,
        "rotation_mapping": {
            "count": EXPECTED_ROTATIONS,
            "bitwise_exact_bijection": True,
            "native_target_rotation_local": target_native_rotation,
            "recovar_target_rotation_row": TARGET_RECOVAR_ROTATION,
        },
        "support": {
            "native_active_count": int(np.count_nonzero(native_support)),
            "recovar_active_count": int(np.count_nonzero(candidate_mask)),
            "exact": support_exact,
        },
        "common_min_diff2": {
            "native": float(native_min),
            "native_bits": int(native_min.view(np.uint32)),
            "recovar": float(recovar_min),
            "recovar_bits": int(recovar_min.view(np.uint32)),
            "bitwise_exact": common_min_exact,
        },
        "raw_diff2": raw_metric,
        "raw_mismatch_strata": raw_mismatch_strata,
        "centered_pre_prior": centered_metric,
        "score_path": {
            "classification": score_path_classification,
            "accepted": (
                score_path_classification == PASS_SCORE_CLASSIFICATION
            ),
            "rotation_prior": rotation_prior_metric,
            "translation_prior": translation_prior_metric,
            "saved_score_replay": saved_score_replay_metric,
            "native_vs_recovar_replayed_combined_score": (
                combined_score_metric
            ),
            "maximum_tie_sets_exact": maximum_tie_sets_exact,
            "native_maximum": float(native_maximum),
            "native_maximum_bits": int(native_maximum.view(np.uint32)),
            "recovar_replayed_maximum": float(recovar_maximum),
            "recovar_replayed_maximum_bits": int(
                recovar_maximum.view(np.uint32)
            ),
        },
        "target": {
            "records": target_records,
            "native_raw_diff2_tied": native_target_tied,
            "recovar_raw_diff2_tied": recovar_target_tied,
        },
    }


def build_report(
    *,
    factor_path: Path,
    fine_score_path: Path,
    recovar_pass2_path: Path,
    native_completion_path: Path,
    recovar_completion_path: Path,
    recovar_allocation_path: Path,
    expected_recovar_job_id: int,
) -> dict[str, Any]:
    native_completion = _validate_completion(
        native_completion_path,
        expected_job_id=NATIVE_SCIENCE_JOB_ID,
    )
    recovar_completion = _validate_recovar_completion(
        recovar_completion_path,
        expected_job_id=expected_recovar_job_id,
    )
    allocation = _read_allocation_table(recovar_allocation_path)
    comparison = _comparison(
        factor_path=factor_path,
        fine_score_path=fine_score_path,
        recovar_pass2_path=recovar_pass2_path,
    )
    inputs = {
        "factor": factor_path,
        "fine_score": fine_score_path,
        "recovar_pass2": recovar_pass2_path,
        "native_completion": native_completion_path,
        "recovar_completion": recovar_completion_path,
        "recovar_allocation": recovar_allocation_path,
    }
    return {
        "schema": SCHEMA,
        "status": "complete",
        "classification_ready": True,
        "classification": comparison.pop("classification"),
        "accepted": comparison.pop("accepted"),
        "scorecard_change_admissible": False,
        "metric_policy": (
            "fixed exact-device K4 iteration-2 raw-cost diagnostic; "
            "bitwise rotation/support/common-min/raw/centered-score and "
            "direct raw-prior-min combined-score replay gates; deterministic "
            "raw-mismatch rotation/translation partitions and representative; "
            "no fitted scale, sign, or correlation; no map acceptance claim"
        ),
        "fixed_contract": {
            "native_slurm_job_id": NATIVE_SCIENCE_JOB_ID,
            "recovar_slurm_job_id": expected_recovar_job_id,
            "target_gpu_uuid": TARGET_GPU_UUID,
            "expected_support": EXPECTED_SUPPORT,
            "expected_rotations": EXPECTED_ROTATIONS,
            "target_recovar_rotation": TARGET_RECOVAR_ROTATION,
            "target_translations": list(TARGET_TRANSLATIONS),
            "recovar_capture_head": RECOVAR_CAPTURE_HEAD,
        },
        "hardware": {
            "allocation": allocation,
            "target_gpu_present": True,
        },
        "native_completion": {
            "status": native_completion["status"],
            "slurm_job_id": native_completion["slurm_job_id"],
        },
        "recovar_completion": {
            "status": recovar_completion["status"],
            "slurm_job_id": recovar_completion["slurm_job_id"],
        },
        **comparison,
        "inputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for name, path in inputs.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor", type=Path, required=True)
    parser.add_argument("--fine-score", type=Path, required=True)
    parser.add_argument("--recovar-pass2", type=Path, required=True)
    parser.add_argument("--native-completion", type=Path, required=True)
    parser.add_argument("--recovar-completion", type=Path, required=True)
    parser.add_argument("--recovar-allocation", type=Path, required=True)
    parser.add_argument("--expected-recovar-job-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"refusing to overwrite {args.output}")
    report = build_report(
        factor_path=args.factor,
        fine_score_path=args.fine_score,
        recovar_pass2_path=args.recovar_pass2,
        native_completion_path=args.native_completion,
        recovar_completion_path=args.recovar_completion,
        recovar_allocation_path=args.recovar_allocation,
        expected_recovar_job_id=args.expected_recovar_job_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "accepted": report["accepted"],
                "classification": report["classification"],
            }
        )
    )


if __name__ == "__main__":
    main()

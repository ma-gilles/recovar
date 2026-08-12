#!/usr/bin/env python3
"""Compare a bounded RELION/RECOVAR K=1 coarse pass-1 score surface."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


HEADER = struct.Struct("<16s32Q")
FOOTER = struct.Struct("<16sQQ")
FLOAT32 = np.dtype("<f4")
UINT8 = np.dtype("u1")
INVALID_DIFF2 = np.finfo(np.float32).min


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _float32_from_bits(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0]


@dataclass(frozen=True)
class NativeCoarseScore:
    path: Path
    sha256: str
    header: tuple[int, ...]
    raw_diff2: np.ndarray
    combined: np.ndarray
    post_exponent: np.ndarray
    orientation_prior: np.ndarray
    translation_prior: np.ndarray
    orientation_zero: np.ndarray
    translation_zero: np.ndarray
    significant: np.ndarray

    @property
    def stack_index(self) -> int:
        return self.header[6]

    @property
    def part_id(self) -> int:
        return self.header[5]


def load_native(path: Path) -> NativeCoarseScore:
    payload = Path(path).read_bytes()
    _require(len(payload) >= HEADER.size + FOOTER.size, f"truncated capture: {path}")
    magic, *values = HEADER.unpack_from(payload, 0)
    header = tuple(int(value) for value in values)
    _require(magic.rstrip(b"\0") == b"RLNCRSC1HEADER", f"header magic mismatch: {path}")
    _require(header[0] == 1 and header[1] == HEADER.size, f"header schema mismatch: {path}")
    _require(header[2] == FOOTER.size, f"footer size mismatch: {path}")
    n_dir, n_psi, n_trans = header[9:12]
    candidate_count = header[12]
    significant_count = header[13]
    orientation_count = n_dir * n_psi
    _require(candidate_count == orientation_count * n_trans, f"topology mismatch: {path}")
    expected = (
        HEADER.size
        + 3 * candidate_count * FLOAT32.itemsize
        + orientation_count * FLOAT32.itemsize
        + n_trans * FLOAT32.itemsize
        + orientation_count * UINT8.itemsize
        + n_trans * UINT8.itemsize
        + candidate_count * UINT8.itemsize
        + FOOTER.size
    )
    _require(len(payload) == expected, f"byte count mismatch: {path}")
    offset = HEADER.size

    def take(dtype: np.dtype, count: int) -> np.ndarray:
        nonlocal offset
        result = np.frombuffer(payload, dtype=dtype, count=count, offset=offset).copy()
        offset += count * dtype.itemsize
        return result

    raw = take(FLOAT32, candidate_count)
    combined = take(FLOAT32, candidate_count)
    post = take(FLOAT32, candidate_count)
    orientation_prior = take(FLOAT32, orientation_count)
    translation_prior = take(FLOAT32, n_trans)
    orientation_zero = take(UINT8, orientation_count).astype(bool)
    translation_zero = take(UINT8, n_trans).astype(bool)
    significant = take(UINT8, candidate_count).astype(bool)
    footer_magic, footer_candidates, footer_significant = FOOTER.unpack_from(payload, offset)
    _require(footer_magic.rstrip(b"\0") == b"RLNCRSC1FOOTER", f"footer magic mismatch: {path}")
    _require(footer_candidates == candidate_count, f"footer candidate mismatch: {path}")
    _require(footer_significant == significant_count, f"footer support mismatch: {path}")
    _require(np.count_nonzero(significant) == significant_count, f"support count mismatch: {path}")
    _require(np.all(np.isfinite(post)) and np.all(post >= 0), f"invalid weights: {path}")
    return NativeCoarseScore(
        path=Path(path),
        sha256=_sha256(Path(path)),
        header=header,
        raw_diff2=raw.reshape(orientation_count, n_trans),
        combined=combined.reshape(orientation_count, n_trans),
        post_exponent=post.reshape(orientation_count, n_trans),
        orientation_prior=orientation_prior,
        translation_prior=translation_prior,
        orientation_zero=orientation_zero,
        translation_zero=translation_zero,
        significant=significant.reshape(orientation_count, n_trans),
    )


def _direction_major_to_psi_major(values: np.ndarray, n_dir: int, n_psi: int) -> np.ndarray:
    array = np.asarray(values)
    _require(array.shape[0] == n_dir * n_psi, "rotation topology mismatch")
    shaped = array.reshape(n_dir, n_psi, *array.shape[1:])
    return shaped.transpose(1, 0, *range(2, shaped.ndim)).reshape(array.shape)


def _metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    left = np.asarray(reference, dtype=np.float64).reshape(-1)
    right = np.asarray(candidate, dtype=np.float64).reshape(-1)
    _require(left.shape == right.shape and left.size > 0, "metric topology mismatch")
    delta = right - left
    denominator = max(float(np.linalg.norm(left)), np.finfo(np.float64).tiny)
    absolute = np.abs(delta)
    return {
        "count": int(left.size),
        "exact_equal": bool(np.array_equal(left, right)),
        "mismatch_count": int(np.count_nonzero(left != right)),
        "relative_l2_over_relion": float(np.linalg.norm(delta) / denominator),
        "median_abs": float(np.median(absolute)),
        "p95_abs": float(np.percentile(absolute, 95)),
        "max_abs": float(np.max(absolute)),
    }


def _center(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[mask] - np.max(array[mask])


def _raw_residual_structure(
    native_raw_diff2: np.ndarray,
    recovar_log_score: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    """Split the centered raw residual into rotation-constant and interaction terms."""

    residual = np.asarray(recovar_log_score, dtype=np.float64) + np.asarray(
        native_raw_diff2,
        dtype=np.float64,
    )
    selected = residual[mask]
    centered = selected - np.mean(selected)
    total_energy = float(np.sum(centered**2))
    _require(total_energy > 0.0, "raw residual has zero centered energy")

    row_mean = np.zeros(residual.shape[0], dtype=np.float64)
    for row in range(residual.shape[0]):
        row_mask = mask[row]
        if np.any(row_mask):
            row_mean[row] = np.mean(residual[row, row_mask])
    rotation_constant_fit = np.broadcast_to(row_mean[:, None], residual.shape)
    interaction = residual[mask] - rotation_constant_fit[mask]
    interaction_abs = np.abs(interaction)
    return {
        "interpretation": (
            "fraction removed by one additive constant per rotation; the remainder "
            "must vary with translation within a rotation"
        ),
        "centered_total_energy": total_energy,
        "rotation_constant_energy_removal_fraction": float(
            1.0 - np.sum(interaction**2) / total_energy
        ),
        "translation_varying_median_abs": float(np.median(interaction_abs)),
        "translation_varying_p95_abs": float(np.percentile(interaction_abs, 95)),
        "translation_varying_max_abs": float(np.max(interaction_abs)),
    }


def _load_recovar(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        scores_pre = np.asarray(payload["scores_pre_prior_per_class"], dtype=np.float64)
        scores_with = np.asarray(payload["scores_with_prior_per_class"], dtype=np.float64)
        _require(scores_pre.ndim == 3 and scores_pre.shape[0] == 1, "expected K=1 scores")
        _require(scores_with.shape == scores_pre.shape, "RECOVAR score topology mismatch")
        n_rot, n_trans = scores_pre.shape[1:]
        probabilities = np.asarray(payload["weights_per_class"], dtype=np.float64).reshape(n_rot, n_trans)
        significant = np.asarray(payload["significant_mask"], dtype=bool).reshape(n_rot, n_trans)
        return {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "original_index": int(np.asarray(payload["original_index"]).item()),
            "n_rot": n_rot,
            "n_trans": n_trans,
            "scores_pre": scores_pre[0],
            "scores_with": scores_with[0],
            "probabilities": probabilities,
            "significant": significant,
            "rotation_prior": np.asarray(payload["rotation_log_prior"], dtype=np.float64).reshape(-1),
            "translation_prior": np.asarray(payload["translation_log_prior"], dtype=np.float64),
            "n_significant": int(np.asarray(payload["n_significant"]).item()),
            "hard_assignment": int(np.asarray(payload["hard_assignment"]).item()),
        }


def _compare(native: NativeCoarseScore, recovar: dict[str, Any]) -> dict[str, Any]:
    n_dir, n_psi, n_trans = native.header[9:12]
    _require(recovar["n_rot"] == n_dir * n_psi and recovar["n_trans"] == n_trans, "cross-engine topology mismatch")
    raw = _direction_major_to_psi_major(native.raw_diff2, n_dir, n_psi)
    combined = _direction_major_to_psi_major(native.combined, n_dir, n_psi)
    post = _direction_major_to_psi_major(native.post_exponent, n_dir, n_psi)
    native_significant = _direction_major_to_psi_major(native.significant, n_dir, n_psi)
    native_orientation_prior = _direction_major_to_psi_major(native.orientation_prior, n_dir, n_psi)
    native_orientation_zero = _direction_major_to_psi_major(native.orientation_zero, n_dir, n_psi)

    native_prior_support = (~native_orientation_zero[:, None]) & (~native.translation_zero[None, :])
    recovar_prior_support = np.isfinite(recovar["scores_with"])
    native_raw_valid = np.isfinite(raw) & (raw != INVALID_DIFF2)
    recovar_raw_valid = np.isfinite(recovar["scores_pre"])
    common = native_prior_support & recovar_prior_support & native_raw_valid & recovar_raw_valid
    _require(np.any(common), "no common prior support")
    native_raw_score = -np.asarray(raw, dtype=np.float64)
    raw_metric = _metric(_center(native_raw_score, common), _center(recovar["scores_pre"], common))
    raw_residual_structure = _raw_residual_structure(raw, recovar["scores_pre"], common)
    combined_metric = _metric(_center(combined, common), _center(recovar["scores_with"], common))
    orientation_common = ~native_orientation_zero
    orientation_prior_metric = _metric(
        native_orientation_prior[orientation_common],
        recovar["rotation_prior"][orientation_common],
    )
    translation_common = ~native.translation_zero
    recovar_translation_prior = np.asarray(recovar["translation_prior"], dtype=np.float64).reshape(-1)
    translation_prior_metric = _metric(
        native.translation_prior[translation_common],
        recovar_translation_prior[translation_common],
    )

    native_probability = np.asarray(post, dtype=np.float64)
    native_probability /= np.sum(native_probability, dtype=np.float64)
    recovar_probability = np.asarray(recovar["probabilities"], dtype=np.float64)
    recovar_probability /= np.sum(recovar_probability, dtype=np.float64)
    posterior_metric = _metric(native_probability, recovar_probability)
    posterior_tv = float(0.5 * np.sum(np.abs(native_probability - recovar_probability)))
    support_mismatch = native_significant != recovar["significant"]
    native_parent = np.any(native_significant, axis=1)
    recovar_parent = np.any(recovar["significant"], axis=1)

    stage_exact = {
        "prior_support": bool(np.array_equal(native_prior_support, recovar_prior_support)),
        "raw_diff2_centered": bool(raw_metric["exact_equal"]),
        "orientation_prior": bool(orientation_prior_metric["exact_equal"]),
        "translation_prior": bool(translation_prior_metric["exact_equal"]),
        "combined_log_weight_centered": bool(combined_metric["exact_equal"]),
        "normalized_posterior": bool(posterior_metric["exact_equal"]),
        "significant_support": bool(not np.any(support_mismatch)),
    }
    first_unequal = next((name for name, equal in stage_exact.items() if not equal), "coarse_boundary_exact")
    return {
        "stack_index_one_based": native.stack_index,
        "original_index_zero_based": recovar["original_index"],
        "relion_part_id": native.part_id,
        "first_unequal_exact_boundary": first_unequal,
        "stage_exact": stage_exact,
        "prior_support": {
            "exact": stage_exact["prior_support"],
            "mismatch_count": int(np.count_nonzero(native_prior_support != recovar_prior_support)),
            "common_count": int(np.count_nonzero(common)),
        },
        "raw_centered_score": raw_metric,
        "raw_residual_structure": raw_residual_structure,
        "orientation_log_prior": orientation_prior_metric,
        "translation_log_prior": translation_prior_metric,
        "combined_centered_log_weight": combined_metric,
        "normalized_posterior": posterior_metric,
        "posterior_total_variation": posterior_tv,
        "support": {
            "exact": stage_exact["significant_support"],
            "candidate_mismatch_count": int(np.count_nonzero(support_mismatch)),
            "parent_mismatch_count": int(np.count_nonzero(native_parent != recovar_parent)),
            "relion_significant_count": int(np.count_nonzero(native_significant)),
            "recovar_significant_count": int(np.count_nonzero(recovar["significant"])),
            "relion_parent_count": int(np.count_nonzero(native_parent)),
            "recovar_parent_count": int(np.count_nonzero(recovar_parent)),
            "relion_only_parent_ids_recovar_psi_major": np.flatnonzero(native_parent & ~recovar_parent).tolist(),
            "recovar_only_parent_ids_recovar_psi_major": np.flatnonzero(recovar_parent & ~native_parent).tolist(),
        },
        "recorded_relion": {
            "min_diff2": _float32_from_bits(native.header[14]),
            "significant_weight": _float32_from_bits(native.header[15]),
            "sum_weight": _float32_from_bits(native.header[16]),
            "max_weight": _float32_from_bits(native.header[17]),
        },
        "artifacts": {
            "relion": str(native.path.resolve()),
            "relion_sha256": native.sha256,
            "recovar": recovar["path"],
            "recovar_sha256": recovar["sha256"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--recovar-directory", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection_json.read_text())
    targets = {int(row["stack_index_one_based"]): row for row in selection["targets"]}
    native_paths = sorted(args.native_directory.glob("*.coarse-score-v1.bin"))
    _require(len(native_paths) == len(targets), "native capture completeness mismatch")
    rows = []
    for native_path in native_paths:
        native = load_native(native_path)
        _require(native.stack_index in targets, f"unexpected native stack {native.stack_index}")
        target = targets[native.stack_index]
        matches = sorted(
            args.recovar_directory.glob(
                f"significance_orig{int(target['original_index_zero_based']):06d}*_cs*.npz"
            )
        )
        _require(len(matches) == 1, f"RECOVAR capture lookup failed for stack {native.stack_index}")
        rows.append(_compare(native, _load_recovar(matches[0])))
    rows.sort(key=lambda row: row["stack_index_one_based"])
    report = {
        "schema": "recovar.em.k1_coarse_score_boundary.v2",
        "case_id": int(selection["case_id"]),
        "physical_iteration": int(selection["physical_iteration"]),
        "metric_policy": "exact and relative-L2 intermediate metrics; no correlation",
        "particle_count": len(rows),
        "first_unequal_exact_boundaries": {
            str(row["stack_index_one_based"]): row["first_unequal_exact_boundary"] for row in rows
        },
        "particles": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

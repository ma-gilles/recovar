#!/usr/bin/env python3
"""Compare one RECOVAR production Wavg norm total with RELION's native update."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


SCHEMA = "recovar.em.k1_target_norm_update_boundary.v1"
CAPTURE_SCHEMAS = {
    "recovar-k1-scale-xa-aa-chunked-v4",
    "recovar-k1-norm-residual-inputs-v3",
}
_NATIVE_UPDATE = re.compile(
    r"RELION_P1_NORM_UPDATE_OPERANDS_V1 iter=(?P<iteration>\d+) "
    r"part_id=(?P<part_id>\d+) previous_norm=(?P<previous_norm>\S+) "
    r"previous_avg=(?P<previous_avg>\S+) old_norm_over_avg=(?P<old_norm_over_avg>\S+) "
    r"wsum_norm=(?P<wsum_norm>\S+) sqrt_2_wsum=(?P<sqrt_2_wsum>\S+) "
    r"new_norm=(?P<new_norm>\S+)"
)
_NATIVE_SPLIT = re.compile(
    r"RELION_P1_NORM_SPLIT_OPERANDS_V1 iter=(?P<iteration>\d+) "
    r"part_id=(?P<part_id>\d+) current_size=(?P<current_size>\S+) "
    r"high_shell=(?P<high_shell>\S+) total=(?P<total>\S+)"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _native_update(path: Path, *, iteration: int, part_id: int) -> dict[str, float | int]:
    records = []
    for line in path.read_text(errors="replace").splitlines():
        match = _NATIVE_UPDATE.search(line)
        if match is None:
            continue
        if int(match["iteration"]) != iteration or int(match["part_id"]) != part_id:
            continue
        records.append(
            {
                "iteration": iteration,
                "part_id": part_id,
                **{
                    name: float.fromhex(match[name])
                    for name in (
                        "previous_norm",
                        "previous_avg",
                        "old_norm_over_avg",
                        "wsum_norm",
                        "sqrt_2_wsum",
                        "new_norm",
                    )
                },
            }
        )
    _require(
        len(records) == 1,
        f"expected one native update for iteration {iteration}, part {part_id}; found {len(records)}",
    )
    return records[0]


def _native_split(
    path: Path,
    *,
    iteration: int,
    part_id: int,
) -> dict[str, float | int] | None:
    records = []
    for line in path.read_text(errors="replace").splitlines():
        match = _NATIVE_SPLIT.search(line)
        if match is None:
            continue
        if int(match["iteration"]) != iteration or int(match["part_id"]) != part_id:
            continue
        records.append(
            {
                "iteration": iteration,
                "part_id": part_id,
                **{
                    name: float.fromhex(match[name])
                    for name in ("current_size", "high_shell", "total")
                },
            }
        )
    _require(len(records) <= 1, "native norm split is duplicated")
    return None if not records else records[0]


def analyze(
    capture_path: Path,
    native_log: Path,
    *,
    image_size: int,
    iteration: int,
    part_id: int,
    source_index: int,
    iteration_state_path: Path | None = None,
) -> dict[str, object]:
    _require(image_size > 0, "image size must be positive")
    native = _native_update(native_log, iteration=iteration, part_id=part_id)
    native_split = _native_split(native_log, iteration=iteration, part_id=part_id)
    with np.load(capture_path, allow_pickle=False) as capture:
        _require(capture["schema"].item() in CAPTURE_SCHEMAS, "RECOVAR capture schema changed")
        _require(int(capture["iteration"]) == iteration, "RECOVAR iteration changed")
        _require(int(capture["original_index"]) == source_index, "RECOVAR particle identity changed")
        high_internal = float(capture["relion_norm_high_shell"])
        atomic_total_internal = None
        atomic_current_internal = None
        if "wavg_diff2_atomic_rectangle_per_image" in capture.files:
            atomic_current_internal = float(capture["wavg_diff2_atomic_rectangle_per_image"])
            atomic_total_internal = atomic_current_internal + high_internal
            atomic_pixels = np.asarray(
                capture["wavg_diff2_atomic_rectangle_per_pixel"], dtype=np.float32
            )
            atomic_shells = np.asarray(
                capture["wavg_diff2_atomic_rectangle_shell_indices"], dtype=np.int32
            )
            valid = atomic_shells >= 0
            sequential_current = float(np.sum(atomic_pixels[valid], dtype=np.float64))
            _require(
                sequential_current == atomic_current_internal,
                "captured atomic Wavg current-size total does not close by sequential host sum",
            )
            atomic_pixel_count = int(atomic_pixels.size)
            atomic_valid_pixel_count = int(np.count_nonzero(valid))
        else:
            atomic_pixel_count = None
            atomic_valid_pixel_count = None
        algebraic_total_internal = None
        if (
            "weighted_img_per_image" in capture.files
            and "block_norm_residual" in capture.files
        ):
            weighted_image_internal = float(capture["weighted_img_per_image"])
            residual_internal = float(capture["block_norm_residual"])
            algebraic_total_internal = weighted_image_internal + residual_internal
        _require(
            atomic_total_internal is not None or algebraic_total_internal is not None,
            "neither atomic nor production-algebraic Wavg norm total is present",
        )
        capture_schema = capture["schema"].item()
        if capture_schema == "recovar-k1-norm-residual-inputs-v3":
            _require(
                algebraic_total_internal is not None,
                "ordinary production-algebraic capture is incomplete",
            )
            norm_path = "production_algebraic_weighted_image_plus_residual"
            total_internal = algebraic_total_internal
            current_internal = total_internal - high_internal
        else:
            _require(
                atomic_total_internal is not None,
                "chunked direct-Wavg capture lacks its atomic total",
            )
            norm_path = "direct_wavg_atomic_rectangle_plus_high_shell"
            total_internal = atomic_total_internal
            current_internal = atomic_current_internal
        posterior_key = (
            "candidate_posterior_probs"
            if "candidate_posterior_probs" in capture.files
            else "posterior_probs"
        )
        posterior = np.asarray(capture[posterior_key], dtype=np.float32)
        posterior_chunk_mass = (
            None
            if "posterior_mass_per_chunk" not in capture.files
            else float(
                np.sum(
                    np.asarray(capture["posterior_mass_per_chunk"], dtype=np.float32),
                    dtype=np.float64,
                )
            )
        )

    divisor = float(image_size**4)
    stopped_capture_wsum = total_internal / divisor
    authoritative_state = None
    if iteration_state_path is not None:
        with np.load(iteration_state_path, allow_pickle=False) as state:
            matches = []
            for half in (1, 2):
                indices = np.asarray(
                    state[f"half{half}_original_image_indices"],
                    dtype=np.int64,
                )
                rows = np.flatnonzero(indices == source_index)
                if rows.size:
                    _require(rows.size == 1, "source identity is duplicated in full iteration state")
                    matches.append((half, int(rows[0])))
            _require(len(matches) == 1, "source identity must occur in exactly one full-state half")
            half, row = matches[0]
            full_internal = float(state[f"half{half}_wsum_norm_correction"][row])
            authoritative_state = {
                "half": half,
                "physical_row": row,
                "wsum_norm_internal": full_internal,
                "wsum_norm_relion_units": full_internal / divisor,
                "norm_correction_internal": float(state[f"half{half}_norm_corrections"][row]),
                "average_norm_correction_internal": float(
                    state[f"half{half}_avg_norm_correction"]
                ),
                "image_correction": float(state[f"half{half}_image_corrections"][row]),
                "scale_correction": float(state[f"half{half}_scale_corrections"][row]),
            }
    recovar_wsum = (
        stopped_capture_wsum
        if authoritative_state is None
        else float(authoritative_state["wsum_norm_relion_units"])
    )
    recovar_sqrt = float(np.sqrt(2.0 * recovar_wsum))
    recovar_new_norm = float(native["old_norm_over_avg"] * recovar_sqrt)
    native_wsum = float(native["wsum_norm"])
    atomic_wsum = (
        None if atomic_total_internal is None else atomic_total_internal / divisor
    )
    algebraic_wsum = (
        None if algebraic_total_internal is None else algebraic_total_internal / divisor
    )
    stopped_high_wsum = high_internal / divisor
    stopped_atomic_current_wsum = (
        None if atomic_current_internal is None else atomic_current_internal / divisor
    )
    stopped_algebraic_current_wsum = (
        None
        if algebraic_total_internal is None
        else (algebraic_total_internal - high_internal) / divisor
    )
    split_comparison = None
    if native_split is not None:
        _require(
            float(native_split["current_size"]) + float(native_split["high_shell"])
            == float(native_split["total"]),
            "native current-size plus high-shell split does not close its total",
        )
        _require(
            float(native_split["total"]) == native_wsum,
            "native split total does not match native update total",
        )
        split_comparison = {
            "stopped_high_shell_delta": stopped_high_wsum - float(native_split["high_shell"]),
            "stopped_atomic_current_size_delta": (
                None
                if stopped_atomic_current_wsum is None
                else stopped_atomic_current_wsum - float(native_split["current_size"])
            ),
            "stopped_algebraic_current_size_delta": (
                None
                if stopped_algebraic_current_wsum is None
                else stopped_algebraic_current_wsum - float(native_split["current_size"])
            ),
        }
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "identity": {
            "iteration": iteration,
            "part_id": part_id,
            "source_index_zero_based": source_index,
            "stack_index_one_based": source_index + 1,
            "image_size": image_size,
        },
        "recovar": {
            "current_size_norm_internal": current_internal,
            "high_shell_power_internal": high_internal,
            "total_internal": total_internal,
            "norm_path": norm_path,
            "production_algebraic_total_internal": algebraic_total_internal,
            "production_algebraic_wsum_norm_relion_units": algebraic_wsum,
            "direct_atomic_total_internal": atomic_total_internal,
            "direct_atomic_wsum_norm_relion_units": atomic_wsum,
            "stopped_high_shell_wsum_relion_units": stopped_high_wsum,
            "stopped_atomic_current_size_wsum_relion_units": stopped_atomic_current_wsum,
            "stopped_algebraic_current_size_wsum_relion_units": stopped_algebraic_current_wsum,
            "relion_unit_divisor": int(image_size**4),
            "wsum_norm_relion_units": recovar_wsum,
            "stopped_capture_wsum_norm_relion_units": stopped_capture_wsum,
            "authoritative_full_iteration": authoritative_state,
            "sqrt_2_wsum": recovar_sqrt,
            "new_norm_using_native_old_ratio": recovar_new_norm,
            "atomic_rectangle_pixel_count": atomic_pixel_count,
            "atomic_rectangle_valid_pixel_count": atomic_valid_pixel_count,
            "posterior_mass_float64_sum": float(np.sum(posterior, dtype=np.float64)),
            "posterior_chunk_mass_float64_sum": posterior_chunk_mass,
        },
        "relion_native": {**native, "split": native_split},
        "comparison": {
            "wsum_norm_delta": recovar_wsum - native_wsum,
            "wsum_norm_relative_error": (recovar_wsum - native_wsum) / native_wsum,
            "sqrt_2_wsum_delta": recovar_sqrt - float(native["sqrt_2_wsum"]),
            "new_norm_delta": recovar_new_norm - float(native["new_norm"]),
            "production_algebraic_wsum_delta": (
                None if algebraic_wsum is None else algebraic_wsum - native_wsum
            ),
            "direct_atomic_wsum_delta": (
                None if atomic_wsum is None else atomic_wsum - native_wsum
            ),
            "native_split": split_comparison,
            "first_compared_boundary": "weighted_norm_total_before_sqrt",
            "authoritative_recovar_source": (
                "stopped_target_capture"
                if authoritative_state is None
                else "full_iteration_state"
            ),
            "bit_exact_float64": bool(
                np.float64(recovar_wsum).view(np.uint64)
                == np.float64(native_wsum).view(np.uint64)
            ),
        },
        "inputs": {
            "recovar_capture": str(capture_path.resolve()),
            "recovar_capture_sha256": _sha256(capture_path),
            "native_log": str(native_log.resolve()),
            "native_log_sha256": _sha256(native_log),
            "recovar_iteration_state": (
                None if iteration_state_path is None else str(iteration_state_path.resolve())
            ),
            "recovar_iteration_state_sha256": (
                None if iteration_state_path is None else _sha256(iteration_state_path)
            ),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovar-capture", type=Path, required=True)
    parser.add_argument("--native-log", type=Path, required=True)
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--part-id", type=int, required=True)
    parser.add_argument("--source-index", type=int, required=True)
    parser.add_argument("--recovar-iteration-state", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        args.recovar_capture,
        args.native_log,
        image_size=args.image_size,
        iteration=args.iteration,
        part_id=args.part_id,
        source_index=args.source_index,
        iteration_state_path=args.recovar_iteration_state,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["comparison"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

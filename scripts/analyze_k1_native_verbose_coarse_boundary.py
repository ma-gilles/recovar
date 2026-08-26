#!/usr/bin/env python3
"""Join native RELION verbose pass-0 arrays to a RECOVAR coarse capture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

if __package__:
    from scripts.parse_relion_dump_dir import parse_dump_dir
else:
    from parse_relion_dump_dir import parse_dump_dir

REQUIRED_NATIVE = {
    "pass0_coarse_candidate_rot_idx",
    "pass0_coarse_candidate_trans_idx",
    "pass0_coarse_candidate_weight_normalized",
    "pass0_coarse_raw_diff2",
    "pass0_coarse_log_weight_preexp",
    "pass0_coarse_candidate_in_threshold_set",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stats(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    _require(absolute.size > 0 and np.all(np.isfinite(absolute)), "invalid residual")
    return {
        "median_abs": float(np.median(absolute)),
        "p95_abs": float(np.percentile(absolute, 95)),
        "max_abs": float(np.max(absolute)),
        "rms": float(np.sqrt(np.mean(np.square(absolute)))),
    }


def _native_to_recovar_rotation(
    native_rotation: np.ndarray,
    *,
    n_directions: int,
    n_psi: int,
) -> np.ndarray:
    """Convert RELION direction-major ids to RECOVAR psi-major ids."""

    native = np.asarray(native_rotation, dtype=np.int64)
    count = n_directions * n_psi
    _require(n_directions > 0 and n_psi > 0, "rotation dimensions must be positive")
    _require(
        bool(np.all((native >= 0) & (native < count))),
        "native rotation id is outside the direction/psi grid",
    )
    direction = native // n_psi
    psi = native % n_psi
    return psi * n_directions + direction


def _load_recovar(path: Path) -> dict[str, np.ndarray]:
    required = {
        "current_size",
        "original_index",
        "n_rot",
        "n_trans",
        "scores_pre_prior_per_class",
        "scores_with_prior_per_class",
        "weights_per_class",
        "significant_mask",
        "n_significant",
        "hard_assignment",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        _require(not missing, f"RECOVAR capture misses fields: {sorted(missing)}")
        return {name: np.asarray(archive[name]) for name in required}


def _load_native(
    directory: Path,
    *,
    n_directions: int,
    n_psi: int,
) -> dict[str, np.ndarray]:
    payload = parse_dump_dir(directory, include_names=REQUIRED_NATIVE)
    missing = REQUIRED_NATIVE - set(payload)
    _require(not missing, f"native dump misses fields: {sorted(missing)}")
    native_rotation = np.asarray(
        payload["pass0_coarse_candidate_rot_idx"], dtype=np.int64
    ).reshape(-1)
    translation = np.asarray(
        payload["pass0_coarse_candidate_trans_idx"], dtype=np.int64
    ).reshape(-1)
    recovar_rotation = _native_to_recovar_rotation(
        native_rotation,
        n_directions=n_directions,
        n_psi=n_psi,
    )
    count = native_rotation.size
    arrays = {
        "native_rotation": native_rotation,
        "recovar_rotation": recovar_rotation,
        "translation": translation,
        "raw_cost": np.asarray(payload["pass0_coarse_raw_diff2"], dtype=np.float32).reshape(-1),
        "total_log_weight": np.asarray(
            payload["pass0_coarse_log_weight_preexp"], dtype=np.float32
        ).reshape(-1),
        "probability": np.asarray(
            payload["pass0_coarse_candidate_weight_normalized"], dtype=np.float32
        ).reshape(-1),
        "selected": np.asarray(
            payload["pass0_coarse_candidate_in_threshold_set"], dtype=np.int32
        ).reshape(-1).astype(bool),
    }
    _require(
        all(value.size == count for value in arrays.values()),
        "native coarse candidate arrays are misaligned",
    )
    n_rot = n_directions * n_psi
    n_trans = int(np.max(translation)) + 1
    _require(
        count == n_rot * n_trans,
        f"native coarse table is not dense: {count} != {n_rot} * {n_trans}",
    )
    flat = recovar_rotation * n_trans + translation
    _require(
        np.array_equal(np.sort(flat), np.arange(count, dtype=np.int64)),
        "mapped native coarse keys are not a dense bijection",
    )
    arrays["flat"] = flat
    arrays["n_rot"] = np.asarray(n_rot)
    arrays["n_trans"] = np.asarray(n_trans)
    return arrays


def analyze(
    *,
    native_dump_dir: Path,
    recovar_path: Path,
    n_directions: int,
    n_psi: int,
    target_rotation: int,
    target_translation: int,
) -> dict[str, object]:
    native = _load_native(
        native_dump_dir,
        n_directions=n_directions,
        n_psi=n_psi,
    )
    recovar = _load_recovar(recovar_path)
    n_rot = int(native["n_rot"])
    n_trans = int(native["n_trans"])
    _require(
        int(np.asarray(recovar["n_rot"]).item()) == n_rot
        and int(np.asarray(recovar["n_trans"]).item()) == n_trans,
        "native and RECOVAR coarse topologies differ",
    )
    _require(
        0 <= target_rotation < n_rot and 0 <= target_translation < n_trans,
        "target coordinate is outside the coarse grid",
    )

    flat = np.asarray(native["flat"], dtype=np.int64)

    def table(name: str, dtype) -> np.ndarray:
        output = np.empty(n_rot * n_trans, dtype=dtype)
        output[flat] = np.asarray(native[name], dtype=dtype)
        return output.reshape(n_rot, n_trans)

    native_raw = table("raw_cost", np.float32)
    native_total = table("total_log_weight", np.float32)
    native_probability = table("probability", np.float32)
    native_selected = table("selected", bool)
    recovar_raw = np.asarray(recovar["scores_pre_prior_per_class"], dtype=np.float32)[0]
    recovar_total = np.asarray(recovar["scores_with_prior_per_class"], dtype=np.float32)[0]
    recovar_probability = np.asarray(recovar["weights_per_class"], dtype=np.float32)[0].reshape(
        n_rot, n_trans
    )
    recovar_selected = np.asarray(recovar["significant_mask"], dtype=bool).reshape(
        n_rot, n_trans
    )

    valid = (
        np.isfinite(native_raw)
        & np.isfinite(recovar_raw)
        & (native_raw != np.finfo(np.float32).min)
    )
    _require(bool(np.any(valid)), "native and RECOVAR raw-score support is empty")
    raw_offset = float(
        np.median(recovar_raw[valid].astype(np.float64) + native_raw[valid].astype(np.float64))
    )
    raw_residual_full = (
        recovar_raw.astype(np.float64) + native_raw.astype(np.float64) - raw_offset
    )
    total_valid = np.isfinite(native_total) & np.isfinite(recovar_total)
    _require(bool(np.any(total_valid)), "native and RECOVAR total-score support is empty")
    native_total_centered = native_total.astype(np.float64) - float(
        np.max(native_total[total_valid])
    )
    recovar_total_centered = recovar_total.astype(np.float64) - float(
        np.max(recovar_total[total_valid])
    )
    total_residual = recovar_total_centered[total_valid] - native_total_centered[total_valid]
    probability_residual = (
        recovar_probability.astype(np.float64) - native_probability.astype(np.float64)
    )
    mismatch = np.argwhere(recovar_selected != native_selected)
    native_best = np.unravel_index(np.argmax(native_probability), native_probability.shape)
    recovar_best = np.unravel_index(np.argmax(recovar_probability), recovar_probability.shape)
    target = (target_rotation, target_translation)

    mismatch_records = []
    for rotation, translation in mismatch[:64]:
        mismatch_records.append(
            {
                "rotation": int(rotation),
                "translation": int(translation),
                "native_probability": float(native_probability[rotation, translation]),
                "recovar_probability": float(recovar_probability[rotation, translation]),
                "native_selected": bool(native_selected[rotation, translation]),
                "recovar_selected": bool(recovar_selected[rotation, translation]),
                "raw_centered_residual": float(raw_residual_full[rotation, translation]),
                "total_centered_residual": float(
                    recovar_total_centered[rotation, translation]
                    - native_total_centered[rotation, translation]
                ),
            }
        )

    return {
        "schema": "recovar.em.k1_native_verbose_coarse_boundary.v1",
        "status": "complete",
        "identity": {
            "source_row_zero_based": int(np.asarray(recovar["original_index"]).item()),
            "stack_index_one_based": int(np.asarray(recovar["original_index"]).item()) + 1,
            "current_size": int(np.asarray(recovar["current_size"]).item()),
            "n_directions": n_directions,
            "n_psi": n_psi,
            "n_rotations": n_rot,
            "n_translations": n_trans,
            "candidate_count": n_rot * n_trans,
        },
        "summary": {
            "native_selected_count": int(np.count_nonzero(native_selected)),
            "recovar_selected_count": int(np.count_nonzero(recovar_selected)),
            "support_mismatch_count": int(mismatch.shape[0]),
            "native_probability_sum": float(np.sum(native_probability, dtype=np.float64)),
            "recovar_probability_sum": float(np.sum(recovar_probability, dtype=np.float64)),
            "posterior_total_variation": float(
                0.5 * np.sum(np.abs(probability_residual), dtype=np.float64)
            ),
            "native_best": [int(native_best[0]), int(native_best[1])],
            "recovar_best": [int(recovar_best[0]), int(recovar_best[1])],
            "best_exact": bool(native_best == recovar_best),
        },
        "residuals": {
            "raw_score_centered": _stats(raw_residual_full[valid]),
            "total_log_weight_centered": _stats(total_residual),
            "posterior": _stats(probability_residual),
        },
        "target": {
            "coordinate": [target_rotation, target_translation],
            "native_rotation_direction_major": int(
                (target_rotation % n_directions) * n_psi
                + target_rotation // n_directions
            ),
            "native_raw_cost": float(native_raw[target]),
            "recovar_raw_score": float(recovar_raw[target]),
            "raw_centered_residual": float(raw_residual_full[target]),
            "native_total_log_weight_centered": float(native_total_centered[target]),
            "recovar_total_log_weight_centered": float(recovar_total_centered[target]),
            "total_centered_residual": float(
                recovar_total_centered[target] - native_total_centered[target]
            ),
            "native_probability": float(native_probability[target]),
            "recovar_probability": float(recovar_probability[target]),
            "native_selected": bool(native_selected[target]),
            "recovar_selected": bool(recovar_selected[target]),
        },
        "support_mismatches_first": mismatch_records,
        "artifacts": {
            "native_dump_dir": str(native_dump_dir.resolve()),
            "native_inputs": {
                name: {
                    "path": str((native_dump_dir / f"{name}.bin").resolve()),
                    "sha256": _sha256(native_dump_dir / f"{name}.bin"),
                }
                for name in sorted(REQUIRED_NATIVE)
            },
            "recovar": {
                "path": str(recovar_path.resolve()),
                "sha256": _sha256(recovar_path),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dump-dir", type=Path, required=True)
    parser.add_argument("--recovar", type=Path, required=True)
    parser.add_argument("--n-directions", type=int, required=True)
    parser.add_argument("--n-psi", type=int, required=True)
    parser.add_argument("--target-rotation", type=int, required=True)
    parser.add_argument("--target-translation", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    report = analyze(
        native_dump_dir=args.native_dump_dir,
        recovar_path=args.recovar,
        n_directions=args.n_directions,
        n_psi=args.n_psi,
        target_rotation=args.target_rotation,
        target_translation=args.target_translation,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()

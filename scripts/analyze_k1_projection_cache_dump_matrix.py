#!/usr/bin/env python3
"""Compare the four K=1 projection-cache/pass-2-dump discriminator arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from scripts.summarize_em_completion_bench import normalized_fsc_auc, shell_fsc
else:
    from summarize_em_completion_bench import normalized_fsc_auc, shell_fsc

ARMS = (
    "no_dump_cache_on",
    "no_dump_cache_off",
    "dump_cache_on",
    "dump_cache_off",
)


def _array_delta(left: np.ndarray, right: np.ndarray) -> dict[str, float | int | bool]:
    lhs = np.asarray(left)
    rhs = np.asarray(right)
    if lhs.shape != rhs.shape:
        raise ValueError(f"array shapes differ: {lhs.shape} versus {rhs.shape}")
    equal = lhs == rhs
    if np.issubdtype(lhs.dtype, np.floating):
        equal = equal | (np.isnan(lhs) & np.isnan(rhs))
    delta = lhs.astype(np.float64) - rhs.astype(np.float64)
    finite = np.isfinite(delta)
    denominator = np.linalg.norm(rhs.astype(np.float64).reshape(-1))
    return {
        "bitwise_equal": bool(np.array_equal(lhs, rhs, equal_nan=True)),
        "different_count": int(np.count_nonzero(~equal)),
        "max_abs": float(np.max(np.abs(delta[finite]))) if np.any(finite) else 0.0,
        "relative_l2": float(np.linalg.norm(delta[finite]) / max(float(denominator), np.finfo(float).tiny)),
    }


def _map_delta(left: np.ndarray, right: np.ndarray) -> dict[str, float | int | bool]:
    out = _array_delta(left, right)
    grid_size = int(round(np.asarray(left).size ** (1.0 / 3.0)))
    if grid_size**3 != np.asarray(left).size:
        raise ValueError(f"map array is not a cube: {np.asarray(left).shape}")
    lhs = np.asarray(left, dtype=np.float64).reshape((grid_size,) * 3)
    rhs = np.asarray(right, dtype=np.float64).reshape((grid_size,) * 3)
    curve = np.asarray(shell_fsc(lhs, rhs), dtype=np.float64)
    out["signed_fsc_auc_non_dc"] = float(normalized_fsc_auc(curve))
    out["finite_non_dc_shells"] = int(np.count_nonzero(np.isfinite(curve[1:])))
    return out


def _target_metrics(archive: np.lib.npyio.NpzFile, source_index: int) -> dict[str, float | int | str]:
    for half in (1, 2):
        original = np.asarray(archive[f"half{half}_original_image_indices"], dtype=np.int64)
        rows = np.flatnonzero(original == int(source_index))
        if rows.size == 1:
            row = int(rows[0])
            factor = np.float32(archive[f"half{half}_norm_corrections"][row])
            image_factor = np.float32(archive[f"half{half}_image_corrections"][row])
            return {
                "half": half,
                "physical_row": row,
                "wsum_norm_correction": float(archive[f"half{half}_wsum_norm_correction"][row]),
                "norm_correction": float(factor),
                "norm_correction_float32_hex": f"0x{int(factor.view(np.uint32)):08x}",
                "image_correction_factor": float(image_factor),
                "image_correction_factor_float32_hex": f"0x{int(image_factor.view(np.uint32)):08x}",
                "best_log_score": float(archive[f"half{half}_best_log_score"][row]),
                "max_posterior": float(archive[f"half{half}_max_posterior"][row]),
                "hard_assignment": int(archive[f"half{half}_hard_assignment"][row]),
            }
    raise ValueError(f"source index {source_index} was not found exactly once")


def _compare(left: np.lib.npyio.NpzFile, right: np.lib.npyio.NpzFile) -> dict[str, object]:
    result: dict[str, object] = {}
    for half in (1, 2):
        prefix = f"half{half}_"
        result[f"half{half}"] = {
            "best_log_score": _array_delta(left[prefix + "best_log_score"], right[prefix + "best_log_score"]),
            "max_posterior": _array_delta(left[prefix + "max_posterior"], right[prefix + "max_posterior"]),
            "hard_assignment": _array_delta(left[prefix + "hard_assignment"], right[prefix + "hard_assignment"]),
            "best_eulers": _array_delta(left[prefix + "best_eulers"], right[prefix + "best_eulers"]),
            "best_translations": _array_delta(
                left[prefix + "best_translations"], right[prefix + "best_translations"]
            ),
            "wsum_sigma2_noise": _array_delta(
                left[prefix + "wsum_sigma2_noise"], right[prefix + "wsum_sigma2_noise"]
            ),
            "wsum_img_power": _array_delta(
                left[prefix + "wsum_img_power"], right[prefix + "wsum_img_power"]
            ),
            "wsum_norm_correction": _array_delta(
                left[prefix + "wsum_norm_correction"], right[prefix + "wsum_norm_correction"]
            ),
            "norm_corrections": _array_delta(
                np.asarray(left[prefix + "norm_corrections"], dtype=np.float32),
                np.asarray(right[prefix + "norm_corrections"], dtype=np.float32),
            ),
            "Ft_y_total": _array_delta(left[prefix + "Ft_y_total"], right[prefix + "Ft_y_total"]),
            "Ft_ctf_total": _array_delta(left[prefix + "Ft_ctf_total"], right[prefix + "Ft_ctf_total"]),
            "regularized_map": _map_delta(left[prefix + "mean_real_ds"], right[prefix + "mean_real_ds"]),
            "unregularized_map": _map_delta(
                left[prefix + "unreg_mean_real_ds"], right[prefix + "unreg_mean_real_ds"]
            ),
        }
    result["sigma2_noise"] = _array_delta(left["sigma2_noise"], right["sigma2_noise"])
    result["fsc"] = _array_delta(left["fsc"], right["fsc"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-index", type=int, default=66)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    paths = {arm: args.root / arm / "parity" / "iter_001.npz" for arm in ARMS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing parity archives: {missing}")
    archives = {arm: np.load(path, allow_pickle=False) for arm, path in paths.items()}
    pairs = {
        "cache_effect_no_dump": ("no_dump_cache_on", "no_dump_cache_off"),
        "cache_effect_dump": ("dump_cache_on", "dump_cache_off"),
        "dump_effect_cache_on": ("no_dump_cache_on", "dump_cache_on"),
        "dump_effect_cache_off": ("no_dump_cache_off", "dump_cache_off"),
    }
    report = {
        "schema": "k1_projection_cache_dump_matrix_v1",
        "metric_policy": "exact/array deltas for state; signed shellwise FSC/FSC-AUC for maps",
        "source_index": int(args.source_index),
        "inputs": {arm: str(path.resolve()) for arm, path in paths.items()},
        "target": {
            arm: _target_metrics(archive, args.source_index) for arm, archive in archives.items()
        },
        "comparisons": {
            label: {
                "left": left,
                "right": right,
                "metrics": _compare(archives[left], archives[right]),
            }
            for label, (left, right) in pairs.items()
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["target"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

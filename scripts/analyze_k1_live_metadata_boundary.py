#!/usr/bin/env python3
"""Compare exact in-memory RELION final metadata with RECOVAR by source ID.

The RELION live-metadata capture is stored in physical dispatch order, whereas
RECOVAR's ``*_by_image`` arrays and the input STAR are stored in source-row
order.  This analyzer joins them by the immutable one-based stack index and
refuses to continue unless the half assignment agrees for every particle.

This is a diagnostic boundary comparison.  It reports pose, translation, and
Pmax discrepancies but does not weaken or replace the signed FSC-AUC gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import starfile


SCHEMA = "recovar.em.k1_live_metadata_boundary.v1"
METADATA_ROT = 0
METADATA_TILT = 1
METADATA_PSI = 2
METADATA_XOFF = 3
METADATA_YOFF = 4
METADATA_PMAX = 8
METADATA_NR_SIGN = 9
TOP_EXAMPLES = 16


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _particle_table(path: Path) -> pd.DataFrame:
    document = starfile.read(path)
    if isinstance(document, pd.DataFrame):
        candidates = [document] if "rlnImageName" in document.columns else []
    else:
        candidates = [
            table
            for table in document.values()
            if isinstance(table, pd.DataFrame) and "rlnImageName" in table.columns
        ]
    _require(len(candidates) == 1, f"{path}: expected one particle table, found {len(candidates)}")
    return candidates[0]


def _stack_indices(image_names: np.ndarray) -> np.ndarray:
    result = np.empty(len(image_names), dtype=np.int64)
    for row, value in enumerate(image_names):
        match = re.fullmatch(r"([1-9][0-9]*)@(.+)", str(value))
        _require(match is not None, f"source row {row}: invalid rlnImageName {value!r}")
        result[row] = int(match.group(1))
    _require(np.unique(result).size == result.size, "input STAR has duplicate stack identities")
    return result


def align_random_subsets_by_stack(
    source_stack_indices: np.ndarray,
    half_stack_indices: np.ndarray,
    half_random_subsets: np.ndarray,
) -> np.ndarray:
    """Align RELION half labels to RECOVAR source rows by immutable stack ID."""

    source_stack_indices = np.asarray(source_stack_indices, dtype=np.int64).reshape(-1)
    half_stack_indices = np.asarray(half_stack_indices, dtype=np.int64).reshape(-1)
    half_random_subsets = np.asarray(half_random_subsets, dtype=np.int64).reshape(-1)
    _require(half_stack_indices.shape == half_random_subsets.shape, "half STAR arrays differ in shape")
    _require(np.unique(source_stack_indices).size == source_stack_indices.size, "source identities repeat")
    _require(np.unique(half_stack_indices).size == half_stack_indices.size, "half identities repeat")
    _require(
        np.array_equal(np.sort(source_stack_indices), np.sort(half_stack_indices)),
        "source and half STAR identity sets differ",
    )
    _require(np.all(np.isin(half_random_subsets, (1, 2))), "RELION half labels must be one or two")
    half_by_stack = {
        int(stack): int(half)
        for stack, half in zip(half_stack_indices, half_random_subsets, strict=True)
    }
    return np.asarray([half_by_stack[int(stack)] for stack in source_stack_indices], dtype=np.int64)


def source_rows_for_live_order(
    source_stack_indices: np.ndarray,
    source_random_subsets: np.ndarray,
    live_stack_indices: np.ndarray,
    live_follower_ranks: np.ndarray,
) -> np.ndarray:
    """Return source rows for physical live order after strict identity checks."""

    source_stack_indices = np.asarray(source_stack_indices, dtype=np.int64).reshape(-1)
    source_random_subsets = np.asarray(source_random_subsets, dtype=np.int64).reshape(-1)
    live_stack_indices = np.asarray(live_stack_indices, dtype=np.int64).reshape(-1)
    live_follower_ranks = np.asarray(live_follower_ranks, dtype=np.int64).reshape(-1)
    n_images = source_stack_indices.size
    _require(source_random_subsets.shape == (n_images,), "source half array has the wrong shape")
    _require(live_stack_indices.shape == (n_images,), "live identity array has the wrong shape")
    _require(live_follower_ranks.shape == (n_images,), "live rank array has the wrong shape")
    _require(np.all(np.isin(source_random_subsets, (1, 2))), "source halves must be one or two")
    _require(np.all(np.isin(live_follower_ranks, (1, 2))), "live follower ranks must be one or two")
    _require(np.unique(source_stack_indices).size == n_images, "source stack identities are not unique")
    _require(np.unique(live_stack_indices).size == n_images, "live stack identities are not unique")
    _require(
        np.array_equal(np.sort(source_stack_indices), np.sort(live_stack_indices)),
        "source and live stack identity sets differ",
    )

    source_position = {int(stack): row for row, stack in enumerate(source_stack_indices)}
    source_rows = np.asarray([source_position[int(stack)] for stack in live_stack_indices], dtype=np.int64)
    expected_ranks = source_random_subsets[source_rows]
    mismatch = np.flatnonzero(expected_ranks != live_follower_ranks)
    _require(
        mismatch.size == 0,
        "live follower rank disagrees with rlnRandomSubset at "
        f"{mismatch.size} particles; first physical rows={mismatch[:8].tolist()}",
    )
    return source_rows


def _relion_euler_matrices(eulers_deg: np.ndarray) -> np.ndarray:
    eulers = np.asarray(eulers_deg, dtype=np.float64).reshape(-1, 3)
    alpha, beta, gamma = np.deg2rad(eulers).T
    ca, cb, cg = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sa, sb, sg = np.sin(alpha), np.sin(beta), np.sin(gamma)
    cc, cs, sc, ss = cb * ca, cb * sa, sb * ca, sb * sa
    matrices = np.empty((eulers.shape[0], 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = cg * cc - sg * sa
    matrices[:, 0, 1] = cg * cs + sg * ca
    matrices[:, 0, 2] = -cg * sb
    matrices[:, 1, 0] = -sg * cc - cg * sa
    matrices[:, 1, 1] = -sg * cs + cg * ca
    matrices[:, 1, 2] = sg * sb
    matrices[:, 2, 0] = sc
    matrices[:, 2, 1] = ss
    matrices[:, 2, 2] = cb
    return matrices


def angular_error_deg(lhs_eulers: np.ndarray, rhs_eulers: np.ndarray) -> np.ndarray:
    lhs_eulers = np.asarray(lhs_eulers, dtype=np.float64).reshape(-1, 3)
    rhs_eulers = np.asarray(rhs_eulers, dtype=np.float64).reshape(-1, 3)
    _require(lhs_eulers.shape == rhs_eulers.shape, "Euler arrays have different shapes")
    lhs = _relion_euler_matrices(lhs_eulers)
    rhs = _relion_euler_matrices(rhs_eulers)
    relative = np.einsum("nij,nkj->nik", lhs, rhs)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    skew = np.stack(
        (
            relative[:, 2, 1] - relative[:, 1, 2],
            relative[:, 0, 2] - relative[:, 2, 0],
            relative[:, 1, 0] - relative[:, 0, 1],
        ),
        axis=1,
    )
    sine = 0.5 * np.linalg.norm(skew, axis=1)
    result = np.degrees(np.arctan2(sine, cosine))
    result[np.all(lhs_eulers == rhs_eulers, axis=1)] = 0.0
    return result


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "count": int(values.size),
            "finite_count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "rms": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }
    return {
        "count": int(values.size),
        "finite_count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "rms": float(np.sqrt(np.mean(np.square(finite)))),
        "p50": float(np.quantile(finite, 0.50)),
        "p95": float(np.quantile(finite, 0.95)),
        "p99": float(np.quantile(finite, 0.99)),
    }


def _metric_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "signed": _summary(values),
        "absolute": _summary(np.abs(values)),
        "exact_zero_count": int(np.count_nonzero(values == 0.0)),
        "nonzero_count": int(np.count_nonzero(values != 0.0)),
    }


def _top_rows(
    values: np.ndarray,
    live_stack_indices: np.ndarray,
    live_follower_ranks: np.ndarray,
    *,
    count: int = TOP_EXAMPLES,
) -> list[dict[str, float | int]]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(np.abs(values), kind="stable")[::-1][: min(count, values.size)]
    return [
        {
            "physical_row": int(row),
            "stack_index_one_based": int(live_stack_indices[row]),
            "half": int(live_follower_ranks[row]),
            "value": float(values[row]),
            "absolute_value": float(abs(values[row])),
        }
        for row in order
    ]


def analyze_arrays(
    *,
    source_stack_indices: np.ndarray,
    source_random_subsets: np.ndarray,
    live_stack_indices: np.ndarray,
    live_follower_ranks: np.ndarray,
    metadata_input: np.ndarray,
    metadata_output: np.ndarray,
    recovar_pmax_by_image: np.ndarray,
    recovar_eulers_by_image: np.ndarray,
    recovar_translations_by_image: np.ndarray,
    panel_stack_indices: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata_input = np.asarray(metadata_input)
    metadata_output = np.asarray(metadata_output)
    _require(metadata_input.shape == metadata_output.shape, "live metadata shapes differ")
    _require(metadata_input.ndim == 2 and metadata_input.shape[1] >= 25, "invalid metadata shape")
    _require(np.all(np.isfinite(metadata_input)), "live input metadata contains nonfinite values")
    _require(np.all(np.isfinite(metadata_output)), "live output metadata contains nonfinite values")
    source_rows = source_rows_for_live_order(
        source_stack_indices,
        source_random_subsets,
        live_stack_indices,
        live_follower_ranks,
    )
    n_images = source_rows.size
    recovar_pmax_by_image = np.asarray(recovar_pmax_by_image, dtype=np.float64)
    recovar_eulers_by_image = np.asarray(recovar_eulers_by_image, dtype=np.float64)
    recovar_translations_by_image = np.asarray(recovar_translations_by_image, dtype=np.float64)
    _require(recovar_pmax_by_image.shape == (n_images,), "RECOVAR Pmax has the wrong shape")
    _require(recovar_eulers_by_image.shape == (n_images, 3), "RECOVAR Euler array has the wrong shape")
    _require(recovar_translations_by_image.shape == (n_images, 2), "RECOVAR translations have the wrong shape")
    for label, values in (
        ("RECOVAR Pmax", recovar_pmax_by_image),
        ("RECOVAR Eulers", recovar_eulers_by_image),
        ("RECOVAR translations", recovar_translations_by_image),
    ):
        _require(np.all(np.isfinite(values)), f"{label} contains nonfinite values")

    relion_pmax = np.asarray(metadata_output[:, METADATA_PMAX], dtype=np.float64)
    relion_eulers = np.asarray(
        metadata_output[:, (METADATA_ROT, METADATA_TILT, METADATA_PSI)], dtype=np.float64
    )
    relion_translations = np.asarray(
        metadata_output[:, (METADATA_XOFF, METADATA_YOFF)], dtype=np.float64
    )
    recovar_pmax = recovar_pmax_by_image[source_rows]
    recovar_eulers = recovar_eulers_by_image[source_rows]
    recovar_translations = recovar_translations_by_image[source_rows]
    pmax_delta = recovar_pmax - relion_pmax
    translation_delta = recovar_translations - relion_translations
    translation_norm = np.linalg.norm(translation_delta, axis=1)
    rotation_error = angular_error_deg(recovar_eulers, relion_eulers)
    support = np.rint(metadata_output[:, METADATA_NR_SIGN]).astype(np.int64)
    _require(np.all(support >= 0), "RELION significant-support counts are negative")

    panel_mask = np.zeros(n_images, dtype=bool)
    if panel_stack_indices is not None:
        panel_stack_indices = np.asarray(panel_stack_indices, dtype=np.int64).reshape(-1)
        _require(np.unique(panel_stack_indices).size == panel_stack_indices.size, "panel identities repeat")
        live_set = set(np.asarray(live_stack_indices, dtype=np.int64).tolist())
        missing = sorted(set(panel_stack_indices.tolist()) - live_set)
        _require(not missing, f"panel identities are absent from live capture: {missing[:8]}")
        panel_mask = np.isin(live_stack_indices, panel_stack_indices)

    def cohort(mask: np.ndarray) -> dict[str, Any]:
        return {
            "particle_count": int(np.count_nonzero(mask)),
            "pmax_delta_recovar_minus_relion": _metric_summary(pmax_delta[mask]),
            "translation_error_pixels": _summary(translation_norm[mask]),
            "rotation_geodesic_error_deg": _summary(rotation_error[mask]),
            "relion_significant_support": _summary(support[mask]),
        }

    all_mask = np.ones(n_images, dtype=bool)
    report = {
        "schema": SCHEMA,
        "classification": "diagnostic_exact_live_metadata_no_fsc_gate_change",
        "identity_alignment": (
            "RELION physical rows joined to RECOVAR source rows by exact one-based stack index"
        ),
        "half_alignment": "exact follower rank equals input rlnRandomSubset for every particle",
        "particle_count": int(n_images),
        "cohorts": {
            "all": cohort(all_mask),
            "half1": cohort(np.asarray(live_follower_ranks) == 1),
            "half2": cohort(np.asarray(live_follower_ranks) == 2),
            "declared_panel": cohort(panel_mask) if panel_stack_indices is not None else None,
        },
        "relion_expectation_update": {
            "pmax": _metric_summary(
                metadata_output[:, METADATA_PMAX] - metadata_input[:, METADATA_PMAX]
            ),
            "translation_norm_pixels": _summary(
                np.linalg.norm(
                    metadata_output[:, (METADATA_XOFF, METADATA_YOFF)]
                    - metadata_input[:, (METADATA_XOFF, METADATA_YOFF)],
                    axis=1,
                )
            ),
            "rotation_geodesic_deg": _summary(
                angular_error_deg(
                    metadata_output[:, (METADATA_ROT, METADATA_TILT, METADATA_PSI)],
                    metadata_input[:, (METADATA_ROT, METADATA_TILT, METADATA_PSI)],
                )
            ),
            "significant_support_changed_count": int(
                np.count_nonzero(
                    metadata_output[:, METADATA_NR_SIGN] != metadata_input[:, METADATA_NR_SIGN]
                )
            ),
        },
        "largest_discrepancies": {
            "absolute_pmax": _top_rows(
                pmax_delta, live_stack_indices, live_follower_ranks
            ),
            "translation_norm_pixels": _top_rows(
                translation_norm, live_stack_indices, live_follower_ranks
            ),
            "rotation_geodesic_deg": _top_rows(
                rotation_error, live_stack_indices, live_follower_ranks
            ),
        },
    }
    arrays = {
        "source_row": source_rows,
        "stack_index_one_based": np.asarray(live_stack_indices, dtype=np.int64),
        "half": np.asarray(live_follower_ranks, dtype=np.int64),
        "panel_mask": panel_mask,
        "relion_pmax": relion_pmax,
        "recovar_pmax": recovar_pmax,
        "pmax_delta_recovar_minus_relion": pmax_delta,
        "relion_eulers_deg": relion_eulers,
        "recovar_eulers_deg": recovar_eulers,
        "rotation_geodesic_error_deg": rotation_error,
        "relion_translations_pixels": relion_translations,
        "recovar_translations_pixels": recovar_translations,
        "translation_delta_pixels": translation_delta,
        "translation_error_pixels": translation_norm,
        "relion_significant_support": support,
    }
    return report, arrays


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-metadata-npz", required=True, type=Path)
    parser.add_argument("--recovar-results", required=True, type=Path)
    parser.add_argument(
        "--source-particles-star",
        required=True,
        type=Path,
        help="Input STAR whose particle row order defines RECOVAR *_by_image arrays",
    )
    parser.add_argument(
        "--relion-half-star",
        required=True,
        type=Path,
        help="RELION data STAR providing rlnRandomSubset; its row order is ignored",
    )
    parser.add_argument("--selection-json", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    for path in (
        args.live_metadata_npz,
        args.recovar_results,
        args.source_particles_star,
        args.relion_half_star,
    ):
        _require(path.is_file(), f"missing input: {path}")
    source_table = _particle_table(args.source_particles_star)
    half_table = _particle_table(args.relion_half_star)
    _require("rlnRandomSubset" in half_table.columns, "RELION half STAR lacks rlnRandomSubset")
    source_stack_indices = _stack_indices(
        source_table["rlnImageName"].astype(str).to_numpy()
    )
    source_random_subsets = align_random_subsets_by_stack(
        source_stack_indices,
        _stack_indices(half_table["rlnImageName"].astype(str).to_numpy()),
        np.asarray(half_table["rlnRandomSubset"], dtype=np.int64),
    )
    panel = None
    if args.selection_json is not None:
        _require(args.selection_json.is_file(), f"missing selection: {args.selection_json}")
        selection = json.loads(args.selection_json.read_text())
        panel = np.asarray(selection["capture_stack_indices_one_based"], dtype=np.int64)

    with np.load(args.live_metadata_npz, allow_pickle=False) as live:
        required_live = {
            "stack_index_one_based",
            "follower_rank",
            "metadata_input",
            "metadata_output",
        }
        _require(required_live <= set(live.files), f"live NPZ lacks {sorted(required_live - set(live.files))}")
        live_values = {name: np.asarray(live[name]) for name in required_live}
    with np.load(args.recovar_results, allow_pickle=False) as recovar:
        required_recovar = {
            "pmax_final_all_data_by_image",
            "best_rotation_eulers_final_all_data_by_image",
            "best_translations_final_all_data_by_image",
        }
        _require(
            required_recovar <= set(recovar.files),
            f"RECOVAR results lack {sorted(required_recovar - set(recovar.files))}",
        )
        recovar_values = {name: np.asarray(recovar[name]) for name in required_recovar}

    report, arrays = analyze_arrays(
        source_stack_indices=source_stack_indices,
        source_random_subsets=source_random_subsets,
        live_stack_indices=live_values["stack_index_one_based"],
        live_follower_ranks=live_values["follower_rank"],
        metadata_input=live_values["metadata_input"],
        metadata_output=live_values["metadata_output"],
        recovar_pmax_by_image=recovar_values["pmax_final_all_data_by_image"],
        recovar_eulers_by_image=recovar_values[
            "best_rotation_eulers_final_all_data_by_image"
        ],
        recovar_translations_by_image=recovar_values[
            "best_translations_final_all_data_by_image"
        ],
        panel_stack_indices=panel,
    )
    report["inputs"] = {
        label: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for label, path in (
            ("live_metadata_npz", args.live_metadata_npz),
            ("recovar_results", args.recovar_results),
            ("source_particles_star", args.source_particles_star),
            ("relion_half_star", args.relion_half_star),
        )
    }
    if args.selection_json is not None:
        report["inputs"]["selection_json"] = {
            "path": str(args.selection_json.resolve()),
            "sha256": _sha256(args.selection_json),
        }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez(args.output_npz, **arrays)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

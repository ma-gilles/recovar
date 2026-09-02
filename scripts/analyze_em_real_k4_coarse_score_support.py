#!/usr/bin/env python3
"""Test whether K=4 coarse-support drift is only a cutoff-count effect.

The RELION fine-pass capture proves which coarse parents were expanded.  A
passive RECOVAR significance dump supplies the untouched coarse score surface
and production support.  This analyzer asks a deliberately narrow causal
question: if RECOVAR's scores are truncated to RELION's observed number of
parents, do they select RELION's parent set?
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from scripts.audit_em_real_k4_shared200_causal_replay import (
    ROTATION_CHILDREN_PER_PARENT,
    TRANSLATION_CHILDREN_PER_PARENT,
    _canonical_native_coarse_rotation_ids,
)
from scripts.validate_relion_fine_score_capture import load_fine_score_capture

SCHEMA = "recovar.em_real_k4_coarse_score_support.v1"


class AnalysisError(RuntimeError):
    """Raised when the bounded score/support evidence is incomplete."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_indices(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item.strip())
    _require(bool(values), "index list is empty")
    _require(len(values) == len(set(values)) and min(values) >= 0, "indices are invalid")
    return values


def _parse_optional_indices(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item.strip())
    _require(
        len(values) == len(set(values)) and all(item >= 0 for item in values),
        "indices are invalid",
    )
    return values


def _support_metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    left = np.asarray(reference, dtype=bool).reshape(-1)
    right = np.asarray(candidate, dtype=bool).reshape(-1)
    _require(left.shape == right.shape, "support shapes differ")
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    return {
        "exact": bool(np.array_equal(left, right)),
        "intersection": intersection,
        "union": union,
        "native_only": int(np.count_nonzero(left & ~right)),
        "recovar_only": int(np.count_nonzero(right & ~left)),
        "native_count": int(np.count_nonzero(left)),
        "recovar_count": int(np.count_nonzero(right)),
        "jaccard": 1.0 if union == 0 else intersection / float(union),
    }


def _stable_top_count_mask(scores: np.ndarray, count: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Select exactly ``count`` finite scores, reporting any boundary tie."""

    values = np.asarray(scores, dtype=np.float64)
    flat = values.reshape(-1)
    finite_indices = np.flatnonzero(np.isfinite(flat))
    _require(0 <= count <= finite_indices.size, "top-count request exceeds finite score support")
    result = np.zeros(flat.size, dtype=bool)
    if count == 0:
        return result.reshape(values.shape), {
            "requested_count": 0,
            "finite_score_count": int(finite_indices.size),
            "cutoff_score": None,
            "cutoff_equal_count": 0,
            "boundary_tie": False,
            "tie_break": "stable flattened class-rotation-translation order",
        }

    finite_scores = flat[finite_indices]
    order = np.argsort(-finite_scores, kind="stable")
    selected = finite_indices[order[:count]]
    result[selected] = True
    cutoff = float(flat[selected[-1]])
    equal_count = int(np.count_nonzero(finite_scores == cutoff))
    next_score = None if count == finite_indices.size else float(finite_scores[order[count]])
    boundary_tie = next_score is not None and next_score == cutoff
    return result.reshape(values.shape), {
        "requested_count": int(count),
        "finite_score_count": int(finite_indices.size),
        "cutoff_score": cutoff,
        "cutoff_equal_count": equal_count,
        "next_score": next_score,
        "boundary_tie": boundary_tie,
        "tie_break": "stable flattened class-rotation-translation order",
    }


def _quantiles(values: Iterable[float]) -> dict[str, float] | None:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return None
    _require(np.isfinite(array).all(), "quantile input is non-finite")
    return {
        "minimum": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def _score_category(
    scores: np.ndarray,
    category: np.ndarray,
    *,
    observed_cutoff: float | None,
) -> dict[str, Any]:
    flat_scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    selected = np.asarray(category, dtype=bool).reshape(-1)
    _require(flat_scores.shape == selected.shape, "score/category shapes differ")
    values = flat_scores[selected]
    finite = np.isfinite(values)

    all_finite_indices = np.flatnonzero(np.isfinite(flat_scores))
    ranks = np.full(flat_scores.size, -1, dtype=np.int64)
    if all_finite_indices.size:
        order = np.argsort(-flat_scores[all_finite_indices], kind="stable")
        ranks[all_finite_indices[order]] = np.arange(all_finite_indices.size, dtype=np.int64)
    selected_ranks = ranks[selected]
    selected_ranks = selected_ranks[selected_ranks >= 0]
    rank_denominator = max(int(all_finite_indices.size) - 1, 1)
    margins = (
        values[finite] - observed_cutoff
        if observed_cutoff is not None
        else np.empty(0, dtype=np.float64)
    )
    return {
        "count": int(values.size),
        "finite_count": int(np.count_nonzero(finite)),
        "nonfinite_count": int(np.count_nonzero(~finite)),
        "score": _quantiles(values[finite]),
        "margin_to_recovar_observed_cutoff": _quantiles(margins),
        "descending_rank": _quantiles(selected_ranks.astype(np.float64)),
        "normalized_descending_rank": _quantiles(
            selected_ranks.astype(np.float64) / float(rank_denominator)
        ),
    }


def _analyze_score_support(
    scores: np.ndarray,
    native_support: np.ndarray,
    recovar_support: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    native = np.asarray(native_support, dtype=bool)
    recovar = np.asarray(recovar_support, dtype=bool)
    _require(values.shape == native.shape == recovar.shape, "score/support topology differs")
    _require(not np.any(recovar & ~np.isfinite(values)), "RECOVAR selected a non-finite score")

    observed = _support_metric(native, recovar)
    recovar_cutoff = float(np.min(values[recovar])) if np.any(recovar) else None
    counterfactual, cutoff = _stable_top_count_mask(values, observed["native_count"])
    counterfactual_metric = _support_metric(native, counterfactual)
    return {
        "observed": observed,
        "native_count_stable_top_score_counterfactual": {
            **counterfactual_metric,
            **cutoff,
        },
        "recovar_observed_minimum_selected_score": recovar_cutoff,
        "score_categories": {
            "shared": _score_category(values, native & recovar, observed_cutoff=recovar_cutoff),
            "native_only": _score_category(
                values,
                native & ~recovar,
                observed_cutoff=recovar_cutoff,
            ),
            "recovar_only": _score_category(
                values,
                recovar & ~native,
                observed_cutoff=recovar_cutoff,
            ),
        },
    }


def _load_recovar_dump(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "original_index",
            "debug_iteration",
            "current_size",
            "n_classes",
            "n_rot",
            "n_trans",
            "significant_mask",
            "n_significant",
            "scores_pre_prior_per_class",
            "scores_with_prior_per_class",
            "score_capture_mode",
        }
        _require(required <= set(payload.files), f"RECOVAR dump lacks {sorted(required - set(payload.files))}")
        n_classes = int(np.asarray(payload["n_classes"]).item())
        n_rot = int(np.asarray(payload["n_rot"]).item())
        n_trans = int(np.asarray(payload["n_trans"]).item())
        scores_pre = np.asarray(payload["scores_pre_prior_per_class"], dtype=np.float64)
        scores_with = np.asarray(payload["scores_with_prior_per_class"], dtype=np.float64)
        significant = np.asarray(payload["significant_mask"], dtype=bool).reshape(
            n_classes,
            n_rot,
            n_trans,
        )
        _require(
            scores_pre.shape == scores_with.shape == significant.shape,
            "RECOVAR score/support shapes differ",
        )
        _require(
            int(np.asarray(payload["n_significant"]).item()) == int(np.count_nonzero(significant)),
            "RECOVAR significant count differs from its mask",
        )
        return {
            "path": path.resolve(),
            "sha256": _sha256(path),
            "original_index": int(np.asarray(payload["original_index"]).item()),
            "debug_iteration": int(np.asarray(payload["debug_iteration"]).item()),
            "current_size": int(np.asarray(payload["current_size"]).item()),
            "n_classes": n_classes,
            "n_rot": n_rot,
            "n_trans": n_trans,
            "scores_pre": scores_pre,
            "scores_with": scores_with,
            "significant": significant,
            "score_capture_mode": str(np.asarray(payload["score_capture_mode"]).item()),
        }


def _index_dumps_by_dataset_index(
    dumps: Iterable[dict[str, Any]],
    expected_indices: Iterable[int],
) -> dict[int, dict[str, Any]]:
    """Index captures in the reduced dataset's explicit index domain.

    ``run_k_class_parity`` loads the frozen 200-row STAR as a new dataset, so
    ``original_index`` in a significance dump is the row-local dataset index
    (0--199).  The underlying ``rlnImageName`` stack offset is useful
    provenance, but it is not RECOVAR's index-layout identity for this view.
    """

    indexed: dict[int, dict[str, Any]] = {}
    for dump in dumps:
        dataset_index = int(dump["original_index"])
        _require(dataset_index not in indexed, "duplicate RECOVAR dataset-index dump")
        indexed[dataset_index] = dump
    _require(
        set(indexed) == {int(index) for index in expected_indices},
        "RECOVAR reduced-dataset dump identities differ",
    )
    return indexed


def _causal_to_dataset_indices(
    *,
    dataset_stack_indices_one_based: np.ndarray,
    causal_rows: dict[int, dict[str, Any]],
    expected_indices: Iterable[int],
) -> dict[int, int]:
    """Join native particle IDs to reduced-STAR rows through stack identity."""

    stack_indices = np.asarray(dataset_stack_indices_one_based, dtype=np.int64).reshape(-1)
    _require(
        stack_indices.size == np.unique(stack_indices).size and np.all(stack_indices > 0),
        "reduced STAR stack identities are invalid or duplicated",
    )
    dataset_index_by_stack = {
        int(stack_index): int(dataset_index)
        for dataset_index, stack_index in enumerate(stack_indices)
    }
    result: dict[int, int] = {}
    for causal_index in expected_indices:
        index = int(causal_index)
        _require(index in causal_rows, "causal report lacks requested target")
        stack_index = int(causal_rows[index]["stack_index_one_based"])
        _require(stack_index in dataset_index_by_stack, "reduced STAR lacks causal stack target")
        result[index] = dataset_index_by_stack[stack_index]
    _require(
        len(set(result.values())) == len(result),
        "causal targets do not map one-to-one onto reduced-STAR rows",
    )
    return result


def _load_dataset_stack_indices_one_based(data_star_path: Path) -> np.ndarray:
    import starfile

    document = starfile.read(data_star_path)
    particles = document["particles"] if isinstance(document, dict) else document
    _require("rlnImageName" in particles, "reduced STAR lacks rlnImageName")
    values = []
    for image_name in particles["rlnImageName"]:
        stack_token = str(image_name).split("@", maxsplit=1)[0]
        try:
            values.append(int(stack_token))
        except ValueError as exc:
            raise AnalysisError(f"invalid rlnImageName stack identity: {image_name}") from exc
    return np.asarray(values, dtype=np.int64)


def _load_native_parent_support(
    causal_root: Path,
    *,
    original_index: int,
    n_classes: int,
    n_rot: int,
    n_trans: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    support = np.zeros((n_classes, n_rot, n_trans), dtype=bool)
    artifacts = []
    children_per_parent = ROTATION_CHILDREN_PER_PARENT * TRANSLATION_CHILDREN_PER_PARENT
    for class_index in range(n_classes):
        class_id = class_index + 1
        matches = sorted(
            (causal_root / "native" / f"class{class_id}" / "factors").glob(
                f"part{original_index}_stack*_class{class_id}.fine-score-v1.bin"
            )
        )
        _require(len(matches) == 1, f"native class-{class_id} capture lookup is ambiguous")
        capture = load_fine_score_capture(matches[0])
        _require(int(capture.header[6]) == original_index, "native particle identity differs")
        _require(int(capture.header[5]) == class_id, "native class identity differs")
        direction_count = int(capture.header[12])
        psi_count = int(capture.header[13])
        _require(direction_count * psi_count == n_rot, "native/RECOVAR rotation count differs")
        _require(int(capture.header[14]) == n_trans, "native/RECOVAR translation count differs")
        candidates = capture.candidates
        if candidates.size:
            rotation = _canonical_native_coarse_rotation_ids(
                candidates["rotation_id"],
                direction_count=direction_count,
                psi_count=psi_count,
            )
            translation = np.asarray(candidates["coarse_translation"], dtype=np.int64)
            pairs = np.stack((rotation, translation), axis=1)
            unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
            _require(
                np.all(counts == children_per_parent),
                "native support is not a complete 8x4 fine expansion",
            )
            support[class_index, unique_pairs[:, 0], unique_pairs[:, 1]] = True
        artifacts.append(
            {
                "class_id_one_based": class_id,
                "path": str(capture.path.resolve()),
                "sha256": capture.sha256,
                "fine_candidate_count": int(candidates.size),
                "coarse_parent_count": int(np.count_nonzero(support[class_index])),
            }
        )
    return support, artifacts


def _validate_causal_metric(observed: dict[str, Any], expected: dict[str, Any]) -> None:
    for key in (
        "exact",
        "intersection",
        "union",
        "native_only",
        "recovar_only",
        "native_count",
        "recovar_count",
    ):
        _require(observed[key] == expected[key], f"fresh passive support changed causal metric {key}")
    _require(
        math.isclose(observed["jaccard"], expected["jaccard"], rel_tol=0.0, abs_tol=1e-15),
        "fresh passive support changed causal Jaccard",
    )


def _sum_metrics(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    return _aggregate_metrics([record[field] for record in records])


def _aggregate_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    _require(bool(metrics), "cannot aggregate an empty metric list")
    intersection = sum(int(metric["intersection"]) for metric in metrics)
    union = sum(int(metric["union"]) for metric in metrics)
    return {
        "records": len(metrics),
        "exact_records": sum(bool(metric["exact"]) for metric in metrics),
        "intersection": intersection,
        "union": union,
        "jaccard": 1.0 if union == 0 else intersection / float(union),
        "native_count": sum(int(metric["native_count"]) for metric in metrics),
        "recovar_count": sum(int(metric["recovar_count"]) for metric in metrics),
    }


def build_report(
    *,
    significance_dir: Path,
    causal_root: Path,
    causal_report_path: Path,
    data_star_path: Path,
    expected_indices: tuple[int, ...],
    exact_control_indices: tuple[int, ...],
) -> dict[str, Any]:
    _require(set(exact_control_indices) < set(expected_indices), "controls must be a proper target subset")
    causal_report = json.loads(causal_report_path.read_text())
    _require(
        causal_report.get("schema") == "recovar.em_real_k4_shared200_causal_replay_audit.v6",
        "causal report schema differs",
    )
    causal_rows = {
        int(row["native_particle_id_zero_based"]): row
        for row in causal_report["causal_boundary"]["particle_rows"]
    }
    _require(set(expected_indices) <= set(causal_rows), "causal report lacks requested targets")
    dataset_stack_indices = _load_dataset_stack_indices_one_based(data_star_path)
    dataset_index_by_causal = _causal_to_dataset_indices(
        dataset_stack_indices_one_based=dataset_stack_indices,
        causal_rows=causal_rows,
        expected_indices=expected_indices,
    )

    dump_paths = sorted(significance_dir.glob("significance_*.npz"))
    _require(len(dump_paths) == len(expected_indices), "RECOVAR dump count differs")
    dumps = [_load_recovar_dump(path) for path in dump_paths]
    dump_by_dataset_index = _index_dumps_by_dataset_index(
        dumps,
        dataset_index_by_causal.values(),
    )

    particle_records = []
    class_records = []
    global_analyses = []
    for original_index in expected_indices:
        causal_row = causal_rows[original_index]
        dataset_index = dataset_index_by_causal[original_index]
        stack_index_zero_based = int(causal_row["stack_index_one_based"]) - 1
        dump = dump_by_dataset_index[dataset_index]
        _require(dump["n_classes"] == 4, "probe is not K=4")
        _require(
            dump["debug_iteration"] == -1 and dump["current_size"] == 20,
            "probe boundary differs",
        )
        _require(
            dump["score_capture_mode"] == "passive_cached_after_support",
            "score capture was not passive-after-support",
        )
        native, native_artifacts = _load_native_parent_support(
            causal_root,
            original_index=original_index,
            n_classes=dump["n_classes"],
            n_rot=dump["n_rot"],
            n_trans=dump["n_trans"],
        )
        analysis = _analyze_score_support(dump["scores_with"], native, dump["significant"])
        global_analyses.append(analysis)
        per_class = []
        for class_index in range(dump["n_classes"]):
            class_analysis = _analyze_score_support(
                dump["scores_with"][class_index],
                native[class_index],
                dump["significant"][class_index],
            )
            expected = causal_row["class_topology"][class_index]["coarse_parent_support"]["joint"]
            _validate_causal_metric(class_analysis["observed"], expected)
            record = {
                "native_particle_id_zero_based": original_index,
                "dataset_index_zero_based": dataset_index,
                "stack_index_zero_based": stack_index_zero_based,
                "class_id_one_based": class_index + 1,
                "stratum": "exact_control" if original_index in exact_control_indices else "mismatch_probe",
                **class_analysis,
            }
            per_class.append(record)
            class_records.append(record)
        if original_index in exact_control_indices:
            _require(analysis["observed"]["exact"], "declared exact control no longer reproduces")
        else:
            _require(not analysis["observed"]["exact"], "declared mismatch probe is now exact")
        particle_records.append(
            {
                "native_particle_id_zero_based": original_index,
                "dataset_index_zero_based": dataset_index,
                "stack_index_zero_based": stack_index_zero_based,
                "stack_index_one_based": int(causal_row["stack_index_one_based"]),
                "stratum": "exact_control" if original_index in exact_control_indices else "mismatch_probe",
                "score_capture_mode": dump["score_capture_mode"],
                "recovar_dump": {
                    "path": str(dump["path"]),
                    "sha256": dump["sha256"],
                },
                "native_artifacts": native_artifacts,
                "global_across_classes": analysis,
                "per_class": per_class,
            }
        )

    class_observed = [record["observed"] for record in class_records]
    class_counterfactual = [
        record["native_count_stable_top_score_counterfactual"] for record in class_records
    ]
    mismatch_class_records = [
        record for record in class_records if record["stratum"] == "mismatch_probe"
    ]
    global_observed = [analysis["observed"] for analysis in global_analyses]
    global_counterfactual = [
        analysis["native_count_stable_top_score_counterfactual"]
        for analysis in global_analyses
    ]
    mismatch_positions = [
        position
        for position, original_index in enumerate(expected_indices)
        if original_index not in exact_control_indices
    ]
    mismatch_global_observed = [global_observed[position] for position in mismatch_positions]
    mismatch_global_counterfactual = [
        global_counterfactual[position] for position in mismatch_positions
    ]
    observed_summary = _aggregate_metrics(global_observed)
    counterfactual_summary = _aggregate_metrics(global_counterfactual)
    mismatch_exact_after = sum(bool(metric["exact"]) for metric in mismatch_global_counterfactual)
    mismatch_total = len(mismatch_global_counterfactual)
    if mismatch_exact_after == mismatch_total and not any(
        bool(metric["boundary_tie"]) for metric in mismatch_global_counterfactual
    ):
        conclusion = "native_parent_count_alone_reproduces_every_probed_mismatch"
    elif counterfactual_summary["jaccard"] > observed_summary["jaccard"]:
        conclusion = "native_parent_count_improves_support_but_score_ordering_also_differs"
    else:
        conclusion = "native_parent_count_does_not_improve_support"

    return {
        "schema": SCHEMA,
        "status": "complete",
        "scientific_scope": (
            f"{len(expected_indices)} frozen EMPIAR-10076 shared-200 particles at K=4 "
            "iteration 1/coarse current-size 20; "
            f"{len(expected_indices) - len(exact_control_indices)} mismatch probes plus "
            f"{len(exact_control_indices)} exact controls"
        ),
        "metric_policy": (
            "direct parent-set identities and stable score ranks; no correlation; native-count "
            "counterfactual changes only the truncation count and never the score values"
        ),
        "inputs": {
            "significance_directory": str(significance_dir.resolve()),
            "causal_root": str(causal_root.resolve()),
            "causal_report": str(causal_report_path.resolve()),
            "causal_report_sha256": _sha256(causal_report_path),
            "data_star": str(data_star_path.resolve()),
            "data_star_sha256": _sha256(data_star_path),
            "expected_native_particle_ids_zero_based": list(expected_indices),
            "exact_control_native_particle_ids_zero_based": list(exact_control_indices),
            "dataset_index_by_native_particle_id": {
                str(key): value for key, value in dataset_index_by_causal.items()
            },
        },
        "summary": {
            "classification": conclusion,
            "class_records": len(class_records),
            "particle_records": len(global_analyses),
            "observed": observed_summary,
            "native_count_counterfactual": counterfactual_summary,
            "mismatch_particle_records": mismatch_total,
            "mismatch_particles_exact_after_native_count": mismatch_exact_after,
            "observed_particle_exact_vector": [bool(metric["exact"]) for metric in global_observed],
            "counterfactual_particle_exact_vector": [
                bool(metric["exact"]) for metric in global_counterfactual
            ],
            "mismatch_observed_jaccard": _aggregate_metrics(mismatch_global_observed)[
                "jaccard"
            ],
            "mismatch_counterfactual_jaccard": _aggregate_metrics(
                mismatch_global_counterfactual
            )["jaccard"],
            "per_class_diagnostic": {
                "observed": _aggregate_metrics(class_observed),
                "native_class_count_counterfactual": _aggregate_metrics(class_counterfactual),
                "mismatch_class_records": len(mismatch_class_records),
            },
        },
        "particles": particle_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--significance-dir", type=Path, required=True)
    parser.add_argument("--causal-root", type=Path, required=True)
    parser.add_argument("--causal-report", type=Path, required=True)
    parser.add_argument("--data-star", type=Path, required=True)
    parser.add_argument("--expected-indices", required=True)
    parser.add_argument("--exact-control-indices", default="")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.output_json}")
    report = build_report(
        significance_dir=args.significance_dir,
        causal_root=args.causal_root,
        causal_report_path=args.causal_report,
        data_star_path=args.data_star,
        expected_indices=_parse_indices(args.expected_indices),
        exact_control_indices=_parse_optional_indices(args.exact_control_indices),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_json.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit RECOVAR/RELION per-particle EM state distributions by image identity.

The auditor compares every matched numbered iteration.  It reports Pmax and
significant-support distributions, with optional pose, translation, class,
half-set, and defocus-cohort diagnostics.  A second RELION trajectory can be
supplied as a numerical control envelope.  Correlation is intentionally not
computed or used as a gate.

RECOVAR ``*_by_image_iter_NNN`` arrays are in the original input-image order,
so ``--recovar-particles-star`` is required to give those rows durable image
identities.  RELION ``run_itNNN_data.star`` rows may be in any order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from recovar.data_io.starfile import read_star

SCHEMA = "em_particle_state_distribution_audit_v1"
STAR_ITERATION_RE = re.compile(r"(?:^|_)it(\d+)(?:_|$)")
RECOVAR_ITERATION_RE = re.compile(
    r"^(?:pmax_per_(?:image_by_image|half_order|image)|sig_counts(?:_by_image|_half_order)?|"
    r"best_rotation_eulers_by_image|best_translations_by_image|class_assignments_by_image)_iter_(\d{3})$"
)


class AuditError(RuntimeError):
    """Raised when required particle state or identity provenance is invalid."""


class MissingRelionIterationsError(AuditError):
    """Raised when a RECOVAR iteration has no supplied RELION state."""

    def __init__(self, missing_relion_iterations: list[int]):
        self.missing_relion_iterations = [int(value) for value in missing_relion_iterations]
        super().__init__(f"missing_relion_iterations={self.missing_relion_iterations}")


def _not_measured(reason: str) -> dict[str, str]:
    return {"status": "not_measured", "reason": reason}


def _column(table, name: str):
    """Return a RELION column while accepting repo-loader underscore spelling."""
    wanted = name.lstrip("_")
    matches = [column for column in table.columns if str(column).lstrip("_") == wanted]
    if len(matches) > 1:
        raise AuditError(f"ambiguous STAR column {name}: {matches}")
    return None if not matches else table[matches[0]]


def _particle_table(path: Path):
    try:
        table, _optics = read_star(str(path))
    except Exception as exc:
        raise AuditError(f"failed to read STAR file {path}: {exc}") from exc
    return table


def _identity_array(table, *, source: Path) -> np.ndarray:
    values = _column(table, "rlnImageName")
    if values is None:
        raise AuditError(f"{source} is missing required rlnImageName identities")
    identities = np.asarray(values.astype(str).to_numpy(), dtype=str)
    if identities.size == 0 or np.any(np.char.str_len(np.char.strip(identities)) == 0):
        raise AuditError(f"{source} contains empty rlnImageName identities")
    unique, counts = np.unique(identities, return_counts=True)
    duplicates = unique[counts > 1]
    if duplicates.size:
        preview = duplicates[:3].tolist()
        raise AuditError(f"{source} contains duplicate rlnImageName identities: {preview}")
    return identities


def _aligned_order(reference: np.ndarray, candidate: np.ndarray, *, source: Path) -> np.ndarray:
    if reference.size != candidate.size:
        raise AuditError(f"identity count mismatch for {source}: RECOVAR={reference.size}, RELION={candidate.size}")
    candidate_index = {name: index for index, name in enumerate(candidate.tolist())}
    missing = [name for name in reference.tolist() if name not in candidate_index]
    extras = sorted(set(candidate.tolist()) - set(reference.tolist()))
    if missing or extras:
        raise AuditError(f"identity set mismatch for {source}: missing={missing[:3]}, extra={extras[:3]}")
    return np.asarray([candidate_index[name] for name in reference.tolist()], dtype=np.int64)


def _as_finite_1d(values: Any, *, label: str, n_images: int, integer: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != n_images:
        raise AuditError(f"{label} has {array.size} rows, expected {n_images}")
    if not np.isfinite(array).all():
        raise AuditError(f"{label} contains non-finite values")
    if integer:
        if np.any(array < 0) or np.any(array != np.floor(array)):
            raise AuditError(f"{label} contains negative or non-integral counts")
        return array.astype(np.int64)
    return array


def _numeric_column(table, name: str, order: np.ndarray, *, source: Path, required: bool = False):
    column = _column(table, name)
    if column is None:
        if required:
            raise AuditError(f"{source} is missing required {name}")
        return None
    try:
        values = np.asarray(column.astype(float).to_numpy(), dtype=np.float64)[order]
    except Exception as exc:
        raise AuditError(f"{source} has non-numeric {name}: {exc}") from exc
    return values


def _summary(values: Any) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size != array.size:
        raise AuditError("internal metric array contains non-finite values")
    if finite.size == 0:
        return {"n": 0}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _difference_summary(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(lhs, dtype=np.float64) - np.asarray(rhs, dtype=np.float64)
    return {
        "signed": _summary(delta),
        "absolute": _summary(np.abs(delta)),
        "exact_equal_count": int(np.count_nonzero(delta == 0)),
        "different_count": int(np.count_nonzero(delta != 0)),
    }


def _relion_euler_matrices(eulers_deg: np.ndarray) -> np.ndarray:
    """Vectorized RELION ``Euler_angles2matrix`` for a dependency-light CLI."""
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


def _angular_error_deg(lhs_eulers: np.ndarray, rhs_eulers: np.ndarray) -> np.ndarray:
    # Angular distance is unchanged by the transpose that converts RELION's
    # projector matrix into RECOVAR's rotation-frame representation.
    lhs_eulers = np.asarray(lhs_eulers, dtype=np.float64).reshape(-1, 3)
    rhs_eulers = np.asarray(rhs_eulers, dtype=np.float64).reshape(-1, 3)
    lhs = _relion_euler_matrices(lhs_eulers)
    rhs = _relion_euler_matrices(rhs_eulers)
    relative = np.einsum("nij,nkj->nik", lhs, rhs)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    skew = np.stack(
        [
            relative[:, 2, 1] - relative[:, 1, 2],
            relative[:, 0, 2] - relative[:, 2, 0],
            relative[:, 1, 0] - relative[:, 0, 1],
        ],
        axis=1,
    )
    sine = 0.5 * np.linalg.norm(skew, axis=1)
    angles = np.degrees(np.arctan2(sine, cosine))
    angles[np.all(lhs_eulers == rhs_eulers, axis=1)] = 0.0
    return angles


def _class_summary(
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    lhs_zero_based: bool = True,
    class_mapping: dict[int, int] | None = None,
) -> dict[str, Any]:
    lhs = np.asarray(lhs, dtype=np.int64).reshape(-1)
    rhs = np.asarray(rhs, dtype=np.int64).reshape(-1)
    if lhs_zero_based:
        # RECOVAR serializes internal K-class ids zero-based; STAR ids are
        # one-based.  Convert unconditionally: class zero may be absent from a
        # collapsed iteration or a subgroup.
        lhs = lhs + 1
    if lhs.size == 0:
        return {
            "status": "measured",
            "n": 0,
            "agreement": None,
            "agreement_count": 0,
            "raw_agreement": None,
            "raw_agreement_count": 0,
            "matching_scope": "whole_iteration" if class_mapping is not None else "current_cohort",
            "labels": [],
            "hungarian_recovar_to_relion": [],
            "confusion_rows_recovar_cols_relion": [],
        }
    labels = sorted(set(lhs.tolist()) | set(rhs.tolist()))
    positions = {label: index for index, label in enumerate(labels)}
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for rec_class, rel_class in zip(lhs, rhs, strict=True):
        confusion[positions[int(rec_class)], positions[int(rel_class)]] += 1
    if class_mapping is None:
        rows, columns = linear_sum_assignment(-confusion)
        class_mapping = {
            labels[int(row)]: labels[int(column)] for row, column in zip(rows, columns, strict=True)
        }
        matching_scope = "current_cohort"
    else:
        class_mapping = {int(key): int(value) for key, value in class_mapping.items()}
        matching_scope = "whole_iteration"
    matched_count = int(
        np.count_nonzero(
            np.asarray([class_mapping.get(int(rec_class), -1) for rec_class in lhs], dtype=np.int64) == rhs
        )
    )
    raw_count = int(np.count_nonzero(lhs == rhs))
    return {
        "status": "measured",
        "n": int(lhs.size),
        # Class identifiers are arbitrary across independent K-class runs. The
        # primary agreement uses the maximum-overlap permutation; raw agreement
        # remains available to expose label stability.
        "agreement": float(matched_count / lhs.size),
        "agreement_count": matched_count,
        "raw_agreement": float(raw_count / lhs.size),
        "raw_agreement_count": raw_count,
        "matching_scope": matching_scope,
        "labels": labels,
        "hungarian_recovar_to_relion": [
            {"recovar_class": rec_class, "relion_class": rel_class}
            for rec_class, rel_class in sorted(class_mapping.items())
        ],
        "confusion_rows_recovar_cols_relion": confusion.tolist(),
    }


def _control_error_comparison(
    rec: dict[str, Any],
    rel: dict[str, Any],
    control: dict[str, Any],
    control_reference: dict[str, Any],
) -> dict[str, Any]:
    """Compare RECOVAR errors with the aligned RELION-repeat error envelope."""
    result: dict[str, Any] = {"status": "measured", "metrics": {}}
    for label, key in (("pmax", "pmax"), ("significant_support", "support")):
        rec_abs = np.abs(np.asarray(rec[key], dtype=np.float64) - np.asarray(rel[key], dtype=np.float64))
        control_abs = np.abs(
            np.asarray(control[key], dtype=np.float64) - np.asarray(control_reference[key], dtype=np.float64)
        )
        rec_summary = _summary(rec_abs)
        control_summary = _summary(control_abs)
        ratios: dict[str, float | None] = {}
        undefined = []
        for statistic in ("mean", "p95", "p99", "max"):
            denominator = float(control_summary[statistic])
            if denominator == 0.0:
                ratios[statistic] = None
                undefined.append(statistic)
            else:
                ratios[statistic] = float(rec_summary[statistic]) / denominator
        control_max = float(control_summary["max"])
        exceed = rec_abs > control_max
        result["metrics"][label] = {
            "recovar_vs_relion_absolute_error": rec_summary,
            "relion_control_vs_relion_absolute_error": control_summary,
            "recovar_to_control_absolute_error_ratio": ratios,
            "ratio_undefined_zero_control_statistics": undefined,
            "count_recovar_abs_error_gt_control_max": int(np.count_nonzero(exceed)),
            "fraction_recovar_abs_error_gt_control_max": float(np.mean(exceed)),
        }
    return result


def _load_recovar_array(npz, stem: str, iteration: int, n_images: int, *, integer: bool = False):
    by_image_key = f"{stem}_by_image_iter_{iteration:03d}"
    if by_image_key in npz.files:
        return _as_finite_1d(npz[by_image_key], label=by_image_key, n_images=n_images, integer=integer)

    half_order_stem = "pmax_per" if stem == "pmax_per_image" else stem
    half_order_key = f"{half_order_stem}_half_order_iter_{iteration:03d}"
    legacy_key = f"{stem}_iter_{iteration:03d}"
    key = half_order_key if half_order_key in npz.files else legacy_key if legacy_key in npz.files else None
    if key is None:
        raise AuditError(f"RECOVAR iteration {iteration:03d} is missing required {stem} state")
    for half_key in ("half1_indices", "half2_indices"):
        if half_key not in npz.files:
            raise AuditError(f"{key} is half-order but {half_key} is missing")
    half_order = np.concatenate(
        [np.asarray(npz["half1_indices"], dtype=np.int64), np.asarray(npz["half2_indices"], dtype=np.int64)]
    )
    raw = _as_finite_1d(npz[key], label=key, n_images=n_images, integer=integer)
    if np.unique(half_order).size != n_images or np.any(np.sort(half_order) != np.arange(n_images)):
        raise AuditError("RECOVAR half-set indices do not form a unique full image permutation")
    out = np.empty_like(raw)
    out[half_order] = raw
    return out


def _optional_recovar_matrix(npz, stem: str, iteration: int, n_images: int, width: int):
    key = f"{stem}_by_image_iter_{iteration:03d}"
    if key not in npz.files:
        return None
    array = np.asarray(npz[key], dtype=np.float64)
    if array.shape != (n_images, width) or not np.isfinite(array).all():
        raise AuditError(f"{key} has invalid shape/values: {array.shape}, expected {(n_images, width)}")
    return array


def _optional_recovar_class(npz, iteration: int, n_images: int):
    key = f"class_assignments_by_image_iter_{iteration:03d}"
    if key not in npz.files:
        return None
    return _as_finite_1d(npz[key], label=key, n_images=n_images, integer=True)


def _half_labels(npz, source_table, n_images: int) -> np.ndarray:
    source_half = _column(source_table, "rlnRandomSubset")
    if source_half is not None:
        halves = np.asarray(source_half.astype(int).to_numpy(), dtype=np.int64)
        if halves.shape != (n_images,) or not set(np.unique(halves)).issubset({1, 2}):
            raise AuditError("RECOVAR particle STAR has invalid rlnRandomSubset values")
        return halves
    halves = np.zeros(n_images, dtype=np.int64)
    for half, key in ((1, "half1_indices"), (2, "half2_indices")):
        if key not in npz.files:
            raise AuditError("half-set subgroup requested but neither STAR subsets nor NPZ half indices are complete")
        indices = np.asarray(npz[key], dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= n_images) or np.unique(indices).size != indices.size:
            raise AuditError(f"invalid {key}")
        halves[indices] = half
    if np.any(halves == 0):
        raise AuditError("NPZ half indices do not cover every image")
    return halves


def _cohort_metrics(
    mask: np.ndarray,
    rec: dict[str, Any],
    rel: dict[str, Any],
    *,
    rec_classes_zero_based: bool = True,
    class_mapping: dict[int, int] | None = None,
) -> dict[str, Any]:
    indices = np.asarray(mask, dtype=bool)
    result: dict[str, Any] = {
        "status": "measured",
        "n": int(np.count_nonzero(indices)),
        "pmax": _difference_summary(rec["pmax"][indices], rel["pmax"][indices]),
        "significant_support": _difference_summary(rec["support"][indices], rel["support"][indices]),
    }
    if rec.get("eulers") is not None and rel.get("eulers") is not None:
        result["angular_error_deg"] = _summary(_angular_error_deg(rec["eulers"][indices], rel["eulers"][indices]))
    else:
        result["angular_error_deg"] = _not_measured("Euler angles unavailable in one or both engines")
    if rec.get("translations") is not None and rel.get("translations") is not None:
        result["translation_error"] = {
            "status": "measured",
            "units": rel["translation_units"],
            **_summary(np.linalg.norm(rec["translations"][indices] - rel["translations"][indices], axis=1)),
        }
    else:
        result["translation_error"] = _not_measured("translations unavailable in one or both engines")
    if rec.get("classes") is not None and rel.get("classes") is not None:
        result["class_assignment"] = _class_summary(
            rec["classes"][indices],
            rel["classes"][indices],
            lhs_zero_based=rec_classes_zero_based,
            class_mapping=class_mapping,
        )
    else:
        result["class_assignment"] = _not_measured("class assignments unavailable in one or both engines")
    return result


def _fixed_pmax_groups(values: np.ndarray) -> list[tuple[str, np.ndarray]]:
    edges = (0.0, 0.5, 0.9, 0.99, math.inf)
    labels = ("lt_0.5", "0.5_to_0.9", "0.9_to_0.99", "ge_0.99")
    return [
        (label, (values >= lo) & (values < hi)) for label, lo, hi in zip(labels, edges[:-1], edges[1:], strict=True)
    ]


def _quantile_groups(values: np.ndarray, *, prefix: str, quantiles: Iterable[float]) -> list[tuple[str, np.ndarray]]:
    edges = np.unique(np.quantile(np.asarray(values, dtype=np.float64), list(quantiles)))
    if edges.size < 2:
        return [(f"{prefix}_all", np.ones(values.size, dtype=bool))]
    groups = []
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        upper = values <= hi if index == edges.size - 2 else values < hi
        groups.append((f"{prefix}_{index + 1}_{lo:.6g}_to_{hi:.6g}", (values >= lo) & upper))
    return groups


def _systematic_labels(
    group_sets: dict[str, dict[str, Any]], overall: dict[str, Any], n_images: int
) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    minimum_n = max(10, int(math.ceil(0.01 * n_images)))
    baselines = {
        "pmax": float(overall["pmax"]["absolute"]["mean"]),
        "significant_support": float(overall["significant_support"]["absolute"]["mean"]),
    }
    for family, groups in group_sets.items():
        for label, metrics in groups.items():
            if metrics["n"] < minimum_n:
                continue
            for metric in ("pmax", "significant_support"):
                value = float(metrics[metric]["absolute"]["mean"])
                baseline = baselines[metric]
                ratio = math.inf if baseline == 0 and value > 0 else (value / baseline if baseline else 1.0)
                if ratio >= 2.0:
                    labels.append(
                        {
                            "cohort_family": family,
                            "cohort": label,
                            "metric": metric,
                            "n": metrics["n"],
                            "abs_mean": value,
                            "overall_abs_mean": baseline,
                            "ratio": ratio,
                        }
                    )
    return labels


def _iteration_from_spec(spec: str) -> tuple[int, Path]:
    explicit = re.match(r"^(\d+):(.*)$", spec)
    if explicit:
        iteration, raw_path = int(explicit.group(1)), explicit.group(2)
    else:
        raw_path = spec
        match = STAR_ITERATION_RE.search(Path(raw_path).name)
        if match is None:
            raise AuditError(f"cannot infer iteration from {spec!r}; use ITERATION:/absolute/path.star")
        iteration = int(match.group(1))
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise AuditError(f"missing STAR file: {path}")
    return iteration, path


def _star_specs(specs: list[str], *, label: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for spec in specs:
        iteration, path = _iteration_from_spec(spec)
        if iteration in result:
            raise AuditError(f"duplicate {label} iteration {iteration}: {result[iteration]} and {path}")
        result[iteration] = path
    return result


def _load_relion_state(path: Path, identities: np.ndarray) -> tuple[dict[str, Any], Any]:
    table = _particle_table(path)
    order = _aligned_order(identities, _identity_array(table, source=path), source=path)
    n_images = identities.size
    pmax = _as_finite_1d(
        _numeric_column(table, "rlnMaxValueProbDistribution", order, source=path, required=True),
        label=f"{path}:rlnMaxValueProbDistribution",
        n_images=n_images,
    )
    support = _as_finite_1d(
        _numeric_column(table, "rlnNrOfSignificantSamples", order, source=path, required=True),
        label=f"{path}:rlnNrOfSignificantSamples",
        n_images=n_images,
        integer=True,
    )
    euler_columns = [
        _numeric_column(table, name, order, source=path) for name in ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")
    ]
    eulers = None if any(value is None for value in euler_columns) else np.column_stack(euler_columns)
    translations = None
    units = None
    angst = [_numeric_column(table, name, order, source=path) for name in ("rlnOriginXAngst", "rlnOriginYAngst")]
    pixel = [_numeric_column(table, name, order, source=path) for name in ("rlnOriginX", "rlnOriginY")]
    if all(value is not None for value in angst):
        translations, units = np.column_stack(angst), "angstrom"
    elif all(value is not None for value in pixel):
        translations, units = np.column_stack(pixel), "pixel"
    classes_raw = _numeric_column(table, "rlnClassNumber", order, source=path)
    classes = (
        None
        if classes_raw is None
        else _as_finite_1d(classes_raw, label=f"{path}:rlnClassNumber", n_images=n_images, integer=True)
    )
    return {
        "pmax": pmax,
        "support": support,
        "eulers": eulers,
        "translations": translations,
        "translation_units": units,
        "classes": classes,
    }, table


def _load_recovar_state(npz, iteration: int, n_images: int, relion_units: str | None) -> dict[str, Any]:
    pmax = _load_recovar_array(npz, "pmax_per_image", iteration, n_images)
    support = _load_recovar_array(npz, "sig_counts", iteration, n_images, integer=True)
    eulers = _optional_recovar_matrix(npz, "best_rotation_eulers", iteration, n_images, 3)
    translations = _optional_recovar_matrix(npz, "best_translations", iteration, n_images, 2)
    if translations is not None and relion_units == "angstrom":
        if "voxel_size" not in npz.files:
            raise AuditError("RELION translations are in Angstrom but RECOVAR voxel_size is missing")
        translations = translations * float(np.asarray(npz["voxel_size"]))
    return {
        "pmax": pmax,
        "support": support,
        "eulers": eulers,
        "translations": translations,
        "translation_units": relion_units,
        "classes": _optional_recovar_class(npz, iteration, n_images),
    }


def _identity_sha256(identities: np.ndarray) -> str:
    payload = "\n".join(identities.tolist()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def audit(
    *,
    recovar_results: Path,
    recovar_particles_star: Path,
    relion_stars: dict[int, Path],
    control_stars: dict[int, Path] | None = None,
    control_reference_stars: dict[int, Path] | None = None,
    recovar_iterations: set[int] | None = None,
) -> dict[str, Any]:
    recovar_results = recovar_results.expanduser().resolve()
    recovar_particles_star = recovar_particles_star.expanduser().resolve()
    if not recovar_results.is_file():
        raise AuditError(f"missing RECOVAR result archive: {recovar_results}")
    source_table = _particle_table(recovar_particles_star)
    identities = _identity_array(source_table, source=recovar_particles_star)
    n_images = identities.size

    with np.load(recovar_results, allow_pickle=False) as npz:
        if "n_images" in npz.files and int(np.asarray(npz["n_images"])) != n_images:
            raise AuditError(
                f"RECOVAR NPZ n_images={int(np.asarray(npz['n_images']))} disagrees with identity STAR rows={n_images}"
            )
        available_rec_iterations = sorted(
            {int(match.group(1)) for key in npz.files if (match := RECOVAR_ITERATION_RE.match(key)) is not None}
        )
        if recovar_iterations is None:
            rec_iterations = available_rec_iterations
        else:
            requested = sorted(int(iteration) for iteration in recovar_iterations)
            missing_requested = sorted(set(requested) - set(available_rec_iterations))
            if missing_requested:
                raise AuditError(
                    "requested RECOVAR iterations are absent from the result archive: "
                    f"{missing_requested}; available={available_rec_iterations}"
                )
            rec_iterations = requested
        expected_relion_iterations = {iteration + 1 for iteration in rec_iterations}
        missing_relion = sorted(expected_relion_iterations - set(relion_stars))
        if missing_relion:
            raise MissingRelionIterationsError(missing_relion)
        if not rec_iterations:
            raise AuditError(
                f"no matched iterations: RECOVAR zero-based={rec_iterations}, RELION one-based={sorted(relion_stars)}"
            )
        matched = [(rec_iteration, rec_iteration + 1) for rec_iteration in rec_iterations]
        halves = _half_labels(npz, source_table, n_images)
        defocus_u = _column(source_table, "rlnDefocusU")
        defocus_v = _column(source_table, "rlnDefocusV")
        defocus = None
        if defocus_u is not None and defocus_v is not None:
            defocus = 0.5 * (
                np.asarray(defocus_u.astype(float).to_numpy(), dtype=np.float64)
                + np.asarray(defocus_v.astype(float).to_numpy(), dtype=np.float64)
            )

        rows = []
        for rec_iteration, rel_iteration in matched:
            rel_state, _ = _load_relion_state(relion_stars[rel_iteration], identities)
            rec_state = _load_recovar_state(npz, rec_iteration, n_images, rel_state["translation_units"])
            overall = _cohort_metrics(np.ones(n_images, dtype=bool), rec_state, rel_state)
            class_mapping = None
            if overall["class_assignment"]["status"] == "measured":
                class_mapping = {
                    int(item["recovar_class"]): int(item["relion_class"])
                    for item in overall["class_assignment"]["hungarian_recovar_to_relion"]
                }
            group_masks: dict[str, list[tuple[str, np.ndarray]]] = {
                "half": [(f"half{half}", halves == half) for half in sorted(np.unique(halves))],
                "relion_pmax_bin": _fixed_pmax_groups(rel_state["pmax"]),
                "relion_support_quantile": _quantile_groups(
                    rel_state["support"], prefix="q", quantiles=(0.0, 0.25, 0.5, 0.75, 1.0)
                ),
            }
            if defocus is not None:
                group_masks["defocus_quantile_angstrom"] = _quantile_groups(
                    defocus, prefix="q", quantiles=(0.0, 1 / 3, 2 / 3, 1.0)
                )
            groups = {
                family: {
                    label: _cohort_metrics(mask, rec_state, rel_state, class_mapping=class_mapping)
                    for label, mask in masks
                }
                for family, masks in group_masks.items()
            }

            control = _not_measured("no RELION/RELION control STAR supplied for this iteration")
            relative_to_control = _not_measured("no RELION/RELION control STAR supplied for this iteration")
            if control_stars and rel_iteration in control_stars:
                control_state, _ = _load_relion_state(control_stars[rel_iteration], identities)
                control_reference_state = rel_state
                if control_reference_stars and rel_iteration in control_reference_stars:
                    control_reference_state, _ = _load_relion_state(
                        control_reference_stars[rel_iteration], identities
                    )
                control = _cohort_metrics(
                    np.ones(n_images, dtype=bool),
                    control_state,
                    control_reference_state,
                    rec_classes_zero_based=False,
                )
                relative_to_control = _control_error_comparison(
                    rec_state, rel_state, control_state, control_reference_state
                )

            rows.append(
                {
                    "recovar_iteration": rec_iteration,
                    "relion_iteration": rel_iteration,
                    "recovar_vs_relion": overall,
                    "relion_control_vs_relion": control,
                    "recovar_vs_relion_relative_to_control": relative_to_control,
                    "subgroups": groups,
                    "systematic_cohorts": _systematic_labels(groups, overall, n_images),
                }
            )

    unused_relion = sorted(set(relion_stars) - set(iteration + 1 for iteration in rec_iterations))
    return {
        "schema": SCHEMA,
        "status": "complete",
        "quality_metric_policy": "No correlation computed; this state auditor reports exact array/distribution errors. Map gates belong to FSC/FSC-AUC trajectory audits.",
        "n_images": int(n_images),
        "identity_sha256": _identity_sha256(identities),
        "sources": {
            "recovar_results": str(recovar_results),
            "recovar_particles_star": str(recovar_particles_star),
            "relion_stars": {str(key): str(value.resolve()) for key, value in sorted(relion_stars.items())},
            "relion_control_stars": {
                str(key): str(value.resolve()) for key, value in sorted((control_stars or {}).items())
            },
            "relion_control_reference_stars": {
                str(key): str(value.resolve())
                for key, value in sorted((control_reference_stars or {}).items())
            },
        },
        "iteration_alignment": {
            "recovar_zero_based_to_relion_one_based": True,
            "selected_recovar_iterations": rec_iterations,
            "missing_relion_iterations": [],
            "unused_relion_iterations": unused_relion,
        },
        "systematic_cohort_rule": {
            "minimum_n": "max(10, ceil(1% of particles))",
            "label_when": "cohort absolute-mean error >= 2x whole-iteration absolute-mean error",
            "interpretation": "aggregate triage label only; not a particle-level quality gate",
        },
        "iterations": rows,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovar-results", required=True, type=Path, help="RECOVAR refinement_results.npz")
    parser.add_argument(
        "--recovar-particles-star",
        required=True,
        type=Path,
        help="Input STAR defining RECOVAR by-image row identities and metadata",
    )
    parser.add_argument(
        "--relion-star",
        required=True,
        action="append",
        help="RELION run_itNNN_data.star; repeat per iteration (or use N:/path.star)",
    )
    parser.add_argument(
        "--relion-control-star",
        action="append",
        default=[],
        help="Optional RELION repeat/control run_itNNN_data.star; repeat per iteration",
    )
    parser.add_argument(
        "--relion-control-reference-star",
        action="append",
        default=[],
        help=(
            "Optional first arm of an independent RELION/RELION repeat pair. "
            "When omitted, --relion-star is the control reference."
        ),
    )
    parser.add_argument(
        "--recovar-iteration",
        action="append",
        type=int,
        help=(
            "Optional zero-based RECOVAR iteration to audit; repeat to select an explicit complete boundary subset. "
            "Unselected iterations are not claimed."
        ),
    )
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output_json.expanduser().resolve()
    try:
        report = audit(
            recovar_results=args.recovar_results,
            recovar_particles_star=args.recovar_particles_star,
            relion_stars=_star_specs(args.relion_star, label="RELION"),
            control_stars=_star_specs(args.relion_control_star, label="RELION control"),
            control_reference_stars=_star_specs(
                args.relion_control_reference_star, label="RELION control reference"
            ),
            recovar_iterations=None if args.recovar_iteration is None else set(args.recovar_iteration),
        )
        status = 0
    except MissingRelionIterationsError as exc:
        report = {
            "schema": SCHEMA,
            "status": "error",
            "earliest_failure": str(exc),
            "missing_relion_iterations": exc.missing_relion_iterations,
        }
        status = 2
    except (AuditError, OSError, ValueError) as exc:
        report = {"schema": SCHEMA, "status": "error", "earliest_failure": str(exc)}
        status = 2
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"status={report['status']} output={output}")
    if status:
        print(f"error={report['earliest_failure']}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())

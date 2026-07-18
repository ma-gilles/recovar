#!/usr/bin/env python3
"""Audit RECOVAR/RELION per-particle EM state distributions by image identity.

The auditor compares every matched numbered iteration and the final all-data
state when both engines expose it.  It reports scalar schedule/convergence
state plus Pmax, significant-support, pose, translation, class, half-set, and
defocus-cohort distributions.  A second RELION trajectory can be supplied as
a numerical control envelope.  Correlation is intentionally not computed or
used as a gate.

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
import shlex
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

SCHEMA = "em_particle_state_distribution_audit_v1"
ARRAY_SCHEMA = "em_particle_state_distribution_arrays_v1"
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


def read_star(path: str):
    """Return the particle loop without importing RECOVAR/JAX.

    This intentionally small parser accepts both RELION's underscored labels
    and the repository writer's legacy bare ``rln...`` labels.  It only reads
    the loop containing ``rlnImageName``; scalar/model STAR parsing is handled
    separately by :func:`_star_scalar_values`.
    """
    source = Path(path)
    lines = source.read_text(errors="replace").splitlines()
    for start, line in enumerate(lines):
        if line.strip().lower() != "loop_":
            continue
        columns: list[str] = []
        cursor = start + 1
        while cursor < len(lines):
            stripped = lines[cursor].strip()
            if not stripped or stripped.startswith("#"):
                cursor += 1
                continue
            token = stripped.split()[0]
            if token.lstrip("_").startswith("rln"):
                columns.append(token)
                cursor += 1
                continue
            break
        if not any(column.lstrip("_") == "rlnImageName" for column in columns):
            continue
        rows: list[list[str]] = []
        while cursor < len(lines):
            stripped = lines[cursor].strip()
            if not stripped:
                if rows:
                    break
                cursor += 1
                continue
            if stripped.startswith("#"):
                cursor += 1
                continue
            if stripped.lower() == "loop_" or stripped.lower().startswith("data_"):
                break
            fields = shlex.split(stripped, comments=True)
            if len(fields) != len(columns):
                raise AuditError(
                    f"{source} particle row has {len(fields)} fields, expected {len(columns)} at line {cursor + 1}"
                )
            rows.append(fields)
            cursor += 1
        if not rows:
            raise AuditError(f"{source} particle loop contains no rows")
        return pd.DataFrame(rows, columns=columns), None
    raise AuditError(f"{source} has no loop containing rlnImageName")


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
        "p50": float(np.percentile(finite, 50)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "rmse": float(np.sqrt(np.mean(np.square(finite)))),
    }


def _threshold_fractions(values: Any, thresholds: Iterable[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        f"le_{threshold:g}": float(np.mean(array <= threshold)) if array.size else None
        for threshold in thresholds
    }


def _error_summary(values: Any, thresholds: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return {**_summary(array), "threshold_fractions": _threshold_fractions(array, thresholds)}


def _difference_summary(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(lhs, dtype=np.float64) - np.asarray(rhs, dtype=np.float64)
    return {
        "signed": _summary(delta),
        "absolute": _error_summary(np.abs(delta), (1e-6, 1e-5, 1e-4, 1e-3)),
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


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms > 1e-15, norms, 1.0)


def _view_inplane_error_deg(lhs_eulers: np.ndarray, rhs_eulers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return view-direction and in-plane errors using RELION matrix rows."""
    lhs = _relion_euler_matrices(lhs_eulers)
    rhs = _relion_euler_matrices(rhs_eulers)
    lhs_view = _normalize_rows(lhs[:, 2, :])
    rhs_view = _normalize_rows(rhs[:, 2, :])
    view = np.degrees(np.arccos(np.clip(np.sum(lhs_view * rhs_view, axis=1), -1.0, 1.0)))

    lhs_x = lhs[:, 0, :] - np.sum(lhs[:, 0, :] * rhs_view, axis=1, keepdims=True) * rhs_view
    rhs_x = rhs[:, 0, :] - np.sum(rhs[:, 0, :] * rhs_view, axis=1, keepdims=True) * rhs_view
    lhs_x = _normalize_rows(lhs_x)
    rhs_x = _normalize_rows(rhs_x)
    inplane = np.abs(
        np.degrees(
            np.arctan2(
                np.sum(rhs_view * np.cross(rhs_x, lhs_x), axis=1),
                np.sum(rhs_x * lhs_x, axis=1),
            )
        )
    )
    exact = np.all(np.asarray(lhs_eulers) == np.asarray(rhs_eulers), axis=1)
    view[exact] = 0.0
    inplane[exact] = 0.0
    return view, inplane


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
    result: dict[str, Any] = {"status": "measured", "n": int(np.count_nonzero(indices))}
    if rec.get("pmax") is not None and rel.get("pmax") is not None:
        result["pmax"] = _difference_summary(rec["pmax"][indices], rel["pmax"][indices])
    else:
        result["pmax"] = _not_measured("Pmax unavailable in one or both engines")
    if rec.get("support") is not None and rel.get("support") is not None:
        support_delta = np.abs(rec["support"][indices] - rel["support"][indices])
        result["significant_support"] = _difference_summary(
            rec["support"][indices], rel["support"][indices]
        )
        result["significant_support"]["absolute"]["threshold_fractions"] = _threshold_fractions(
            support_delta, (0, 1, 2, 5, 10)
        )
    else:
        result["significant_support"] = _not_measured(
            "significant-support count unavailable in one or both engines"
        )
    if rec.get("eulers") is not None and rel.get("eulers") is not None:
        rec_eulers = rec["eulers"][indices]
        rel_eulers = rel["eulers"][indices]
        view_error, inplane_error = _view_inplane_error_deg(rec_eulers, rel_eulers)
        result["angular_error_deg"] = _error_summary(
            _angular_error_deg(rec_eulers, rel_eulers), (0.01, 0.1, 0.5, 1, 5)
        )
        result["view_direction_error_deg"] = _error_summary(view_error, (0.01, 0.1, 0.5, 1, 5))
        result["inplane_error_deg"] = _error_summary(inplane_error, (0.01, 0.1, 0.5, 1, 5))
    else:
        result["angular_error_deg"] = _not_measured("Euler angles unavailable in one or both engines")
        result["view_direction_error_deg"] = _not_measured("Euler angles unavailable in one or both engines")
        result["inplane_error_deg"] = _not_measured("Euler angles unavailable in one or both engines")
    if rec.get("translations") is not None and rel.get("translations") is not None:
        result["translation_error"] = {
            "status": "measured",
            "units": rel["translation_units"],
            **_error_summary(
                np.linalg.norm(rec["translations"][indices] - rel["translations"][indices], axis=1),
                (0.01, 0.1, 0.5, 1, 2),
            ),
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


def _compact_error_arrays(rec: dict[str, Any], rel: dict[str, Any]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for label, key in (("pmax", "pmax"), ("support", "support")):
        if rec.get(key) is not None and rel.get(key) is not None:
            arrays[f"{label}_recovar"] = np.asarray(rec[key])
            arrays[f"{label}_relion"] = np.asarray(rel[key])
            arrays[f"{label}_delta"] = np.asarray(rec[key], dtype=np.float64) - np.asarray(
                rel[key], dtype=np.float64
            )
    if rec.get("eulers") is not None and rel.get("eulers") is not None:
        arrays["rotation_geodesic_deg"] = _angular_error_deg(rec["eulers"], rel["eulers"])
        view, inplane = _view_inplane_error_deg(rec["eulers"], rel["eulers"])
        arrays["rotation_view_deg"] = view
        arrays["rotation_inplane_deg"] = inplane
    if rec.get("translations") is not None and rel.get("translations") is not None:
        arrays["translation_l2"] = np.linalg.norm(rec["translations"] - rel["translations"], axis=1)
    if rec.get("classes") is not None and rel.get("classes") is not None:
        arrays["class_recovar_zero_based"] = np.asarray(rec["classes"], dtype=np.int32)
        arrays["class_relion_one_based"] = np.asarray(rel["classes"], dtype=np.int32)
    return arrays


def _binary_tail_enrichment(exposure: np.ndarray, pose_tail: np.ndarray) -> dict[str, Any]:
    """Summarize whether an identity-aligned exposure precedes a pose tail."""
    exposure = np.asarray(exposure, dtype=bool).reshape(-1)
    pose_tail = np.asarray(pose_tail, dtype=bool).reshape(-1)
    if exposure.shape != pose_tail.shape:
        raise AuditError(
            f"cross-iteration enrichment shape mismatch: exposure={exposure.shape}, pose_tail={pose_tail.shape}"
        )
    n = int(exposure.size)
    exposed = int(np.count_nonzero(exposure))
    unexposed = n - exposed
    tail = int(np.count_nonzero(pose_tail))
    exposed_and_tail = int(np.count_nonzero(exposure & pose_tail))
    exposed_only = exposed - exposed_and_tail
    tail_only = tail - exposed_and_tail
    neither = n - exposed_and_tail - exposed_only - tail_only

    rate_exposed = None if exposed == 0 else float(exposed_and_tail / exposed)
    rate_unexposed = None if unexposed == 0 else float(tail_only / unexposed)
    capture_fraction = None if tail == 0 else float(exposed_and_tail / tail)
    enrichment = (
        None
        if rate_exposed is None or rate_unexposed is None or rate_unexposed == 0.0
        else float(rate_exposed / rate_unexposed)
    )
    undefined = []
    if exposed == 0:
        undefined.append("pose_tail_rate_given_exposure: exposure_count=0")
    if unexposed == 0:
        undefined.append("pose_tail_rate_without_exposure: unexposed_count=0")
    if tail == 0:
        undefined.append("tail_capture_fraction: pose_tail_count=0")
    if rate_unexposed == 0.0:
        undefined.append("enrichment: pose_tail_rate_without_exposure=0")
    elif rate_unexposed is None or rate_exposed is None:
        undefined.append("enrichment: conditional_rate_unavailable")
    return {
        "n": n,
        "contingency": {
            "exposure_and_next_pose_tail": exposed_and_tail,
            "exposure_only": exposed_only,
            "next_pose_tail_only": tail_only,
            "neither": neither,
        },
        "exposure_count": exposed,
        "unexposed_count": unexposed,
        "next_pose_tail_count": tail,
        "exposure_fraction": None if n == 0 else float(exposed / n),
        "next_pose_tail_fraction": None if n == 0 else float(tail / n),
        "conditional_rates": {
            "next_pose_tail_given_exposure": rate_exposed,
            "next_pose_tail_without_exposure": rate_unexposed,
        },
        "enrichment_vs_unexposed": enrichment,
        "next_pose_tail_capture_fraction": capture_fraction,
        "undefined_zero_denominators": undefined,
    }


def _top_fraction_mask(values: np.ndarray, fraction: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Select an exact top fraction, breaking cutoff ties by identity row."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise AuditError("top-fraction diagnostic received non-finite values")
    if not 0.0 < fraction <= 1.0:
        raise AuditError(f"top fraction must be in (0, 1], got {fraction}")
    count = min(values.size, int(math.ceil(fraction * values.size)))
    order = np.argsort(-values, kind="stable")
    selected = order[:count]
    mask = np.zeros(values.size, dtype=bool)
    mask[selected] = True
    return mask, {
        "requested_fraction": float(fraction),
        "selected_count": int(count),
        "selected_fraction": None if values.size == 0 else float(count / values.size),
        "cutoff_value": None if count == 0 else float(values[selected[-1]]),
        "tie_break": "descending value; stable original exact-identity row order",
    }


def _cross_iteration_tail_enrichment(
    aligned_errors: dict[int, dict[str, np.ndarray]],
    *,
    n_images: int,
    artifact_arrays: dict[str, np.ndarray] | None = None,
    pose_tail_threshold_deg: float = 0.1,
    pmax_top_fraction: float = 0.05,
) -> dict[str, Any]:
    """Relate state mismatch at iteration t to pose tails at iteration t+1."""
    boundaries = []
    iterations = sorted(aligned_errors)
    for previous, current in zip(iterations[:-1], iterations[1:], strict=True):
        if current != previous + 1:
            continue
        previous_errors = aligned_errors[previous]
        current_errors = aligned_errors[current]
        pose_error = current_errors.get("rotation_geodesic_deg")
        if pose_error is None:
            boundaries.append(
                {
                    "previous_relion_iteration": previous,
                    "current_relion_iteration": current,
                    "status": "not_measured",
                    "reason": "next-iteration Euler angles unavailable in one or both engines",
                }
            )
            continue
        for name, values in (
            ("support_delta", previous_errors["support_delta"]),
            ("pmax_delta", previous_errors["pmax_delta"]),
            ("rotation_geodesic_deg", pose_error),
        ):
            if np.asarray(values).reshape(-1).size != n_images:
                raise AuditError(
                    f"cross-iteration {previous}->{current} {name} has "
                    f"{np.asarray(values).size} identities, expected {n_images}"
                )
        support_exposure = np.asarray(previous_errors["support_delta"]) != 0
        absolute_pmax = np.abs(np.asarray(previous_errors["pmax_delta"], dtype=np.float64))
        pmax_exposure, pmax_selection = _top_fraction_mask(absolute_pmax, pmax_top_fraction)
        pose_tail = np.asarray(pose_error, dtype=np.float64) > float(pose_tail_threshold_deg)
        label = f"it{previous:03d}_to_it{current:03d}"
        if artifact_arrays is not None:
            artifact_arrays[f"{label}_support_mismatch_at_t"] = support_exposure
            artifact_arrays[f"{label}_top5_abs_pmax_delta_at_t"] = pmax_exposure
            artifact_arrays[f"{label}_pose_tail_at_t_plus_1"] = pose_tail
        boundaries.append(
            {
                "previous_relion_iteration": previous,
                "current_relion_iteration": current,
                "status": "measured",
                "significant_support_count_mismatch_at_t": _binary_tail_enrichment(
                    support_exposure, pose_tail
                ),
                "top_5pct_absolute_pmax_delta_at_t": {
                    "selection": pmax_selection,
                    **_binary_tail_enrichment(pmax_exposure, pose_tail),
                },
            }
        )
    return {
        "status": "measured" if boundaries else "not_measured",
        "diagnostic_only": True,
        "identity_alignment": "exact rlnImageName row identity across every numbered state",
        "pose_tail_definition": f"cross-engine rotation geodesic error > {pose_tail_threshold_deg:g} degrees at t+1",
        "support_exposure_definition": "RECOVAR significant-support count != RELION count at t",
        "pmax_exposure_definition": (
            f"exact top ceil({pmax_top_fraction:g} * n_images) absolute cross-engine Pmax deltas at t"
        ),
        "zero_denominator_policy": "undefined conditional rates, enrichment, or capture fractions are null and named explicitly",
        "quality_gate": "none; descriptive aggregate triage only",
        "correlation": "not computed",
        "boundaries": boundaries,
    }


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


def _optional_final_recovar_array(npz, key: str, n_images: int, *, width: int | None = None):
    if key not in npz.files:
        return None
    array = np.asarray(npz[key], dtype=np.float64)
    expected = (n_images,) if width is None else (n_images, width)
    if array.shape != expected or not np.isfinite(array).all():
        raise AuditError(f"{key} has invalid shape/values: {array.shape}, expected {expected}")
    return array


def _load_final_recovar_state(npz, n_images: int, relion_units: str | None) -> dict[str, Any]:
    translations = _optional_final_recovar_array(
        npz, "best_translations_final_all_data_by_image", n_images, width=2
    )
    if translations is not None and relion_units == "angstrom":
        if "voxel_size" not in npz.files:
            raise AuditError("RELION final translations are in Angstrom but RECOVAR voxel_size is missing")
        translations = translations * float(np.asarray(npz["voxel_size"]))
    classes = _optional_final_recovar_array(npz, "class_assignments_final_all_data_by_image", n_images)
    return {
        "pmax": _optional_final_recovar_array(npz, "pmax_final_all_data_by_image", n_images),
        "support": _optional_final_recovar_array(npz, "sig_counts_final_all_data_by_image", n_images),
        "eulers": _optional_final_recovar_array(
            npz, "best_rotation_eulers_final_all_data_by_image", n_images, width=3
        ),
        "translations": translations,
        "translation_units": relion_units,
        "classes": None if classes is None else classes.astype(np.int64),
    }


def _star_scalar_values(path: Path) -> dict[str, Any]:
    """Read scalar ``_rlnName value`` rows without materializing STAR loops."""
    if not path.is_file():
        return {}
    values: dict[str, Any] = {}
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        first_token = stripped.split(maxsplit=1)[0] if stripped else ""
        if not first_token.lstrip("_").startswith("rln"):
            continue
        try:
            fields = shlex.split(stripped, comments=True)
        except ValueError as exc:
            raise AuditError(f"failed to parse scalar STAR line in {path}: {stripped!r}: {exc}") from exc
        if len(fields) != 2:
            continue
        name, token = fields
        try:
            value: Any = int(token)
        except ValueError:
            try:
                value = float(token)
            except ValueError:
                value = token
        values[name.lstrip("_")] = value
    return values


def _star_loop_numeric_values(path: Path, column_name: str) -> np.ndarray | None:
    """Read one numeric STAR-loop column, accepting bare or underscored labels."""
    if not path.is_file():
        return None
    wanted = column_name.lstrip("_")
    lines = path.read_text(errors="replace").splitlines()
    for start, line in enumerate(lines):
        if line.strip().lower() != "loop_":
            continue
        columns: list[str] = []
        cursor = start + 1
        while cursor < len(lines):
            stripped = lines[cursor].strip()
            if not stripped or stripped.startswith("#"):
                cursor += 1
                continue
            token = stripped.split()[0]
            if token.lstrip("_").startswith("rln"):
                columns.append(token.lstrip("_"))
                cursor += 1
                continue
            break
        if wanted not in columns:
            continue
        column_index = columns.index(wanted)
        values: list[float] = []
        while cursor < len(lines):
            stripped = lines[cursor].strip()
            if not stripped:
                if values:
                    break
                cursor += 1
                continue
            if stripped.startswith("#"):
                cursor += 1
                continue
            if stripped.lower() == "loop_" or stripped.lower().startswith("data_"):
                break
            try:
                fields = shlex.split(stripped, comments=True)
            except ValueError as exc:
                raise AuditError(f"failed to parse STAR loop row in {path}: {stripped!r}: {exc}") from exc
            if len(fields) != len(columns):
                raise AuditError(
                    f"{path} loop row has {len(fields)} fields, expected {len(columns)} at line {cursor + 1}"
                )
            try:
                values.append(float(fields[column_index]))
            except ValueError as exc:
                raise AuditError(
                    f"{path} {wanted} is non-numeric at line {cursor + 1}: {fields[column_index]!r}"
                ) from exc
            cursor += 1
        if not values:
            raise AuditError(f"{path} loop column {wanted} contains no values")
        result = np.asarray(values, dtype=np.float64)
        if not np.isfinite(result).all():
            raise AuditError(f"{path} loop column {wanted} contains non-finite values")
        return result
    return None


def _relion_state_paths(data_path: Path) -> dict[str, Path]:
    suffix = "_data.star"
    if not data_path.name.endswith(suffix):
        return {}
    prefix = data_path.name[: -len(suffix)]
    return {
        "data": data_path,
        "model_half1": data_path.with_name(f"{prefix}_half1_model.star"),
        "sampling": data_path.with_name(f"{prefix}_sampling.star"),
        "optimiser": data_path.with_name(f"{prefix}_optimiser.star"),
    }


def _relion_scalar_state(data_path: Path, pmax: np.ndarray) -> dict[str, Any]:
    paths = _relion_state_paths(data_path)
    model = _star_scalar_values(paths["model_half1"]) if "model_half1" in paths else {}
    estimated_resolutions = (
        _star_loop_numeric_values(paths["model_half1"], "rlnEstimatedResolution")
        if "model_half1" in paths
        else None
    )
    if estimated_resolutions is not None and estimated_resolutions.size != 1:
        raise AuditError(
            "K=1 scalar audit expects exactly one model-class rlnEstimatedResolution; "
            f"found {estimated_resolutions.size} in {paths['model_half1']}"
        )
    estimated_resolution = None if estimated_resolutions is None else float(estimated_resolutions[0])
    sampling = _star_scalar_values(paths["sampling"]) if "sampling" in paths else {}
    optimiser = _star_scalar_values(paths["optimiser"]) if "optimiser" in paths else {}
    fields = {
        "current_image_size": model.get("rlnCurrentImageSize"),
        "current_resolution_angstrom": estimated_resolution,
        "estimated_resolution_angstrom": estimated_resolution,
        "scheduling_current_resolution_angstrom": model.get("rlnCurrentResolution"),
        "average_pmax_particles": float(np.mean(pmax)),
        "average_pmax_mstep": model.get("rlnAveragePmax"),
        "healpix_order": sampling.get("rlnHealpixOrder"),
        "offset_range_angstrom": sampling.get("rlnOffsetRange"),
        "offset_step_angstrom": sampling.get("rlnOffsetStep"),
        "sampling_perturbation": sampling.get("rlnSamplingPerturbInstance"),
        "accuracy_rotations_deg": optimiser.get("rlnOverallAccuracyRotations"),
        "accuracy_translations_angstrom": optimiser.get("rlnOverallAccuracyTranslationsAngst"),
        "changes_optimal_orientations_deg": optimiser.get("rlnChangesOptimalOrientations"),
        "changes_optimal_offsets_angstrom": optimiser.get("rlnChangesOptimalOffsets"),
        "changes_optimal_classes": optimiser.get("rlnChangesOptimalClasses"),
        "smallest_changes_orientations_deg": optimiser.get("rlnSmallestChangesOrientations"),
        "smallest_changes_offsets_angstrom": optimiser.get("rlnSmallestChangesOffsets"),
        "smallest_changes_classes": optimiser.get("rlnSmallestChangesClasses"),
        "iterations_without_resolution_gain": optimiser.get("rlnNumberOfIterWithoutResolutionGain"),
        "iterations_without_assignment_change": optimiser.get("rlnNumberOfIterWithoutChangingAssignments"),
        "has_high_fsc_at_resolution_limit": optimiser.get("rlnHasHighFscAtResolLimit"),
        "has_converged": optimiser.get("rlnHasConverged"),
        "current_iteration": optimiser.get("rlnCurrentIteration"),
    }
    return {
        "fields": fields,
        "artifacts": {name: {"path": str(path), "present": path.is_file()} for name, path in paths.items()},
    }


def _npz_trajectory_value(npz, key: str, iteration: int):
    if key not in npz.files:
        return None
    array = np.asarray(npz[key]).reshape(-1)
    if iteration >= array.size:
        return None
    value = array[iteration]
    value = value.item() if isinstance(value, np.generic) else value
    # Some trajectories intentionally use NaN for an unavailable first
    # boundary (for example exact expected accuracy under firstiter_cc).
    # JSON reports represent an unavailable scalar as null/not-measured.
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _recovar_scalar_state(npz, iteration: int) -> dict[str, Any]:
    resolution_shell = _npz_trajectory_value(npz, "pixel_resolutions", iteration)
    resolution_angstrom = None
    if resolution_shell is not None and float(resolution_shell) > 0 and "voxel_size" in npz.files:
        shape_key = "volume_shape" if "volume_shape" in npz.files else "image_shape" if "image_shape" in npz.files else None
        if shape_key is not None:
            box_size = int(np.asarray(npz[shape_key]).reshape(-1)[-1])
            resolution_angstrom = box_size * float(np.asarray(npz["voxel_size"])) / float(resolution_shell)
    return {
        "current_image_size": _npz_trajectory_value(npz, "current_sizes", iteration),
        "current_resolution_shell_index": resolution_shell,
        "current_resolution_angstrom": resolution_angstrom,
        "average_pmax_particles": _npz_trajectory_value(npz, "ave_Pmax_trajectory", iteration),
        "healpix_order": _npz_trajectory_value(npz, "healpix_order_trajectory", iteration),
        "accuracy_rotations_deg": _npz_trajectory_value(npz, "acc_rot_trajectory", iteration),
        "accuracy_translations_angstrom": _npz_trajectory_value(npz, "acc_trans_trajectory", iteration),
        "changes_optimal_orientations_deg": _npz_trajectory_value(
            npz, "smallest_change_angles_trajectory", iteration
        ),
        "changes_optimal_offsets_angstrom": _npz_trajectory_value(
            npz, "smallest_change_offsets_trajectory", iteration
        ),
        "fraction_assignments_changed": _npz_trajectory_value(npz, "frac_changed_trajectory", iteration),
    }


def _scalar_comparison(recovar: dict[str, Any], relion: dict[str, Any]) -> dict[str, Any]:
    comparisons = {}
    for key in sorted(set(recovar) & set(relion)):
        lhs, rhs = recovar[key], relion[key]
        if lhs is None or rhs is None or not isinstance(lhs, (int, float, np.number)) or not isinstance(
            rhs, (int, float, np.number)
        ):
            comparisons[key] = _not_measured("scalar missing or non-numeric in one engine")
            continue
        if not np.isfinite(float(lhs)) or not np.isfinite(float(rhs)):
            comparisons[key] = _not_measured("scalar is non-finite in one engine")
            continue
        comparisons[key] = {
            "status": "measured",
            "recovar_minus_relion": float(lhs) - float(rhs),
            "exact_equal": bool(float(lhs) == float(rhs)),
        }
    return comparisons


def _identity_sha256(identities: np.ndarray) -> str:
    payload = "\n".join(identities.tolist()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _metric_value(metrics: dict[str, Any], path: tuple[str, ...]):
    value: Any = metrics
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _apply_thresholds(
    report: dict[str, Any],
    *,
    thresholds: dict[str, float],
    require_exact_schedule: bool,
    require_exact_convergence: bool,
) -> list[str]:
    failures: list[str] = []
    checks = {
        "max_pmax_abs_p95": (("pmax", "absolute", "p95"), "max"),
        "max_pmax_abs_max": (("pmax", "absolute", "max"), "max"),
        "max_support_abs_p95": (("significant_support", "absolute", "p95"), "max"),
        "max_rotation_geodesic_p95_deg": (("angular_error_deg", "p95"), "max"),
        "max_rotation_view_p95_deg": (("view_direction_error_deg", "p95"), "max"),
        "max_rotation_inplane_p95_deg": (("inplane_error_deg", "p95"), "max"),
        "max_translation_p95": (("translation_error", "p95"), "max"),
        "min_class_agreement": (("class_assignment", "agreement"), "min"),
    }
    states = [(f"it{row['relion_iteration']:03d}", row["recovar_vs_relion"]) for row in report["iterations"]]
    final = report.get("final_all_data", {})
    if final.get("status") == "measured":
        states.append(("final", final["recovar_vs_relion"]))
    for threshold_name, threshold in thresholds.items():
        if threshold_name not in checks:
            raise AuditError(f"unknown threshold: {threshold_name}")
        path, direction = checks[threshold_name]
        for label, metrics in states:
            value = _metric_value(metrics, path)
            if value is None:
                failures.append(f"{label} {'.'.join(path)} is not measured")
            elif direction == "max" and float(value) > float(threshold):
                failures.append(f"{label} {'.'.join(path)}={float(value):.9g} > {float(threshold):.9g}")
            elif direction == "min" and float(value) < float(threshold):
                failures.append(f"{label} {'.'.join(path)}={float(value):.9g} < {float(threshold):.9g}")

    if require_exact_schedule:
        for row in report["iterations"]:
            comparison = row["scalar_state"]["comparison"]
            for field in ("current_image_size", "healpix_order"):
                metric = comparison.get(field, {})
                if metric.get("status") != "measured":
                    failures.append(f"it{row['relion_iteration']:03d} schedule {field} is not measured")
                elif not metric.get("exact_equal"):
                    failures.append(f"it{row['relion_iteration']:03d} schedule {field} differs")
    if require_exact_convergence:
        rec = report["convergence_topology"]["recovar"]
        rel = report["convergence_topology"]["relion"]
        for field in ("iteration", "has_converged"):
            if rec.get(field) is None or rel.get(field) is None:
                failures.append(f"convergence {field} is not measured in both engines")
            elif int(rec[field]) != int(rel[field]):
                failures.append(f"convergence {field} differs: RECOVAR={rec[field]} RELION={rel[field]}")
        if rec.get("final_all_data_ran") is not True or rel.get("final_data_star_present") is not True:
            failures.append(
                "finalization topology differs or is incomplete: "
                f"RECOVAR final_all_data_ran={rec.get('final_all_data_ran')} "
                f"RELION final_data_star_present={rel.get('final_data_star_present')}"
            )
    return failures


def audit(
    *,
    recovar_results: Path,
    recovar_particles_star: Path,
    relion_stars: dict[int, Path],
    control_stars: dict[int, Path] | None = None,
    control_reference_stars: dict[int, Path] | None = None,
    recovar_iterations: set[int] | None = None,
    relion_iteration_offset: int = 1,
    relion_final_star: Path | None = None,
    artifact_arrays: dict[str, np.ndarray] | None = None,
    thresholds: dict[str, float] | None = None,
    require_exact_schedule: bool = False,
    require_exact_convergence: bool = False,
) -> dict[str, Any]:
    recovar_results = recovar_results.expanduser().resolve()
    recovar_particles_star = recovar_particles_star.expanduser().resolve()
    if not recovar_results.is_file():
        raise AuditError(f"missing RECOVAR result archive: {recovar_results}")
    source_table = _particle_table(recovar_particles_star)
    identities = _identity_array(source_table, source=recovar_particles_star)
    n_images = identities.size
    relion_iteration_offset = int(relion_iteration_offset)
    if relion_iteration_offset < 1:
        raise AuditError(
            "relion_iteration_offset must be at least 1 because RELION numbered "
            f"iterations are one-based, got {relion_iteration_offset}"
        )
    if artifact_arrays is not None:
        artifact_arrays["schema"] = np.asarray(ARRAY_SCHEMA)
        artifact_arrays["identity_row_index"] = np.arange(n_images, dtype=np.int64)
        artifact_arrays["identity_sha256"] = np.asarray(_identity_sha256(identities))

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
            if rec_iterations != list(range(len(rec_iterations))):
                raise AuditError(
                    "RECOVAR numbered iteration topology is not contiguous zero-based: "
                    f"{rec_iterations}"
                )
        else:
            requested = sorted(int(iteration) for iteration in recovar_iterations)
            missing_requested = sorted(set(requested) - set(available_rec_iterations))
            if missing_requested:
                raise AuditError(
                    "requested RECOVAR iterations are absent from the result archive: "
                    f"{missing_requested}; available={available_rec_iterations}"
                )
            rec_iterations = requested
        expected_relion_iterations = {
            iteration + relion_iteration_offset for iteration in rec_iterations
        }
        missing_relion = sorted(expected_relion_iterations - set(relion_stars))
        if missing_relion:
            raise MissingRelionIterationsError(missing_relion)
        if not rec_iterations:
            raise AuditError(
                f"no matched iterations: RECOVAR zero-based={rec_iterations}, RELION one-based={sorted(relion_stars)}"
            )
        matched = [
            (rec_iteration, rec_iteration + relion_iteration_offset)
            for rec_iteration in rec_iterations
        ]
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
        aligned_errors: dict[int, dict[str, np.ndarray]] = {}
        for rec_iteration, rel_iteration in matched:
            rel_state, _ = _load_relion_state(relion_stars[rel_iteration], identities)
            rec_state = _load_recovar_state(npz, rec_iteration, n_images, rel_state["translation_units"])
            overall = _cohort_metrics(np.ones(n_images, dtype=bool), rec_state, rel_state)
            relion_scalar = _relion_scalar_state(relion_stars[rel_iteration], rel_state["pmax"])
            recovar_scalar = _recovar_scalar_state(npz, rec_iteration)
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

            compact_errors = _compact_error_arrays(rec_state, rel_state)
            aligned_errors[rel_iteration] = compact_errors
            rows.append(
                {
                    "recovar_iteration": rec_iteration,
                    "relion_iteration": rel_iteration,
                    "recovar_vs_relion": overall,
                    "relion_control_vs_relion": control,
                    "recovar_vs_relion_relative_to_control": relative_to_control,
                    "subgroups": groups,
                    "systematic_cohorts": _systematic_labels(groups, overall, n_images),
                    "scalar_state": {
                        "recovar": recovar_scalar,
                        "relion": relion_scalar,
                        "comparison": _scalar_comparison(recovar_scalar, relion_scalar["fields"]),
                    },
                }
            )
            if artifact_arrays is not None:
                artifact_arrays.update(
                    {
                        f"it{rel_iteration:03d}_{name}": value
                        for name, value in compact_errors.items()
                    }
                )

        cross_iteration_tail_enrichment = _cross_iteration_tail_enrichment(
            aligned_errors,
            n_images=n_images,
            artifact_arrays=artifact_arrays,
        )

        final = _not_measured("no complete RECOVAR/RELION final all-data state was available")
        final_all_data_ran = (
            bool(np.asarray(npz["final_all_data_ran"]).reshape(()))
            if "final_all_data_ran" in npz.files
            else None
        )
        if relion_final_star is not None:
            relion_final_star = relion_final_star.expanduser().resolve()
            if not relion_final_star.is_file():
                raise AuditError(f"missing RELION final STAR file: {relion_final_star}")
            final_rel_state, _ = _load_relion_state(relion_final_star, identities)
            final_rec_state = _load_final_recovar_state(npz, n_images, final_rel_state["translation_units"])
            if any(value is not None for key, value in final_rec_state.items() if key != "translation_units"):
                final = {
                    "status": "measured",
                    "recovar_vs_relion": _cohort_metrics(
                        np.ones(n_images, dtype=bool), final_rec_state, final_rel_state
                    ),
                    "relion_star": str(relion_final_star),
                    "scalar_state": {
                        "recovar": {
                            "convergence_iteration": (
                                int(np.asarray(npz["convergence_iteration"]).reshape(()))
                                if "convergence_iteration" in npz.files
                                else None
                            ),
                            "has_converged": (
                                bool(np.asarray(npz["convergence_has_converged"]).reshape(()))
                                if "convergence_has_converged" in npz.files
                                else None
                            ),
                            "final_all_data_ran": final_all_data_ran,
                        },
                        "relion": _relion_scalar_state(relion_final_star, final_rel_state["pmax"]),
                    },
                }
                if artifact_arrays is not None:
                    artifact_arrays.update(
                        {
                            f"final_{name}": value
                            for name, value in _compact_error_arrays(final_rec_state, final_rel_state).items()
                        }
                    )

        convergence = {
            "recovar": {
                "iteration": (
                    int(np.asarray(npz["convergence_iteration"]).reshape(()))
                    if "convergence_iteration" in npz.files
                    else None
                ),
                "has_converged": (
                    bool(np.asarray(npz["convergence_has_converged"]).reshape(()))
                    if "convergence_has_converged" in npz.files
                    else None
                ),
                "final_all_data_ran": final_all_data_ran,
            },
            "relion": {},
        }
        final_scalar_path = relion_final_star
        if final_scalar_path is None and relion_stars:
            last_path = relion_stars[max(relion_stars)]
            final_scalar_path = last_path.with_name(re.sub(r"_it\d+_data\.star$", "_data.star", last_path.name))
        if final_scalar_path is not None and final_scalar_path.is_file():
            final_paths = _relion_state_paths(final_scalar_path)
            final_opt = _star_scalar_values(final_paths.get("optimiser", Path("/__missing__")))
            relion_convergence_iteration = final_opt.get("rlnCurrentIteration")
            if (
                final_opt.get("rlnHasConverged") in (1, True)
                and (relion_convergence_iteration is None or int(relion_convergence_iteration) < 0)
                and relion_stars
            ):
                # RELION's unnumbered final optimiser writes CurrentIteration
                # as -1.  The convergence boundary is the highest numbered
                # state that immediately preceded this converged final pass.
                relion_convergence_iteration = max(relion_stars)
            convergence["relion"] = {
                "iteration": relion_convergence_iteration,
                "has_converged": final_opt.get("rlnHasConverged"),
                "final_data_star_present": final_scalar_path.is_file(),
            }

    used_relion_iterations = {
        iteration + relion_iteration_offset for iteration in rec_iterations
    }
    unused_relion = sorted(set(relion_stars) - used_relion_iterations)
    report = {
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
            "relion_final_star": str(relion_final_star) if relion_final_star is not None else None,
        },
        "iteration_alignment": {
            "recovar_zero_based_to_relion_one_based": relion_iteration_offset == 1,
            "relion_iteration_offset": relion_iteration_offset,
            "mapping": "relion_iteration = recovar_iteration + relion_iteration_offset",
            "selected_recovar_iterations": rec_iterations,
            "missing_relion_iterations": [],
            "unused_relion_iterations": unused_relion,
            "numbered_topology_valid": True,
            "identity_alignment": "exact rlnImageName set; RELION row order ignored",
        },
        "systematic_cohort_rule": {
            "minimum_n": "max(10, ceil(1% of particles))",
            "label_when": "cohort absolute-mean error >= 2x whole-iteration absolute-mean error",
            "interpretation": "aggregate triage label only; not a particle-level quality gate",
        },
        "cross_iteration_tail_enrichment": cross_iteration_tail_enrichment,
        "iterations": rows,
        "final_all_data": final,
        "convergence_topology": convergence,
        "thresholds": thresholds or {},
        "threshold_failures": [],
    }
    failures = _apply_thresholds(
        report,
        thresholds=thresholds or {},
        require_exact_schedule=require_exact_schedule,
        require_exact_convergence=require_exact_convergence,
    )
    report["threshold_failures"] = failures
    report["gating"] = {
        "enabled": bool(thresholds or require_exact_schedule or require_exact_convergence),
        "require_exact_schedule": bool(require_exact_schedule),
        "require_exact_convergence": bool(require_exact_convergence),
        "defaults": "diagnostic/non-gating",
    }
    if report["gating"]["enabled"]:
        report["status"] = "pass" if not failures else "fail"
    return report


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
    parser.add_argument(
        "--relion-iteration-offset",
        type=int,
        default=1,
        help=(
            "Map RECOVAR local iteration i to RELION physical iteration i+OFFSET. "
            "Use 2 for a one-step replay of physical iteration 2."
        ),
    )
    parser.add_argument(
        "--relion-final-star",
        type=Path,
        help="Optional RELION run_data.star; otherwise inferred beside the numbered trajectory when present",
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument(
        "--output-npz",
        type=Path,
        help="Compact aligned error arrays (default: <output-json stem>_arrays.npz)",
    )
    parser.add_argument(
        "--output-hash-manifest",
        type=Path,
        help="SHA-256 manifest for JSON and compact NPZ (default: <output-json>.sha256)",
    )
    parser.add_argument("--max-pmax-abs-p95", type=float)
    parser.add_argument("--max-pmax-abs-max", type=float)
    parser.add_argument("--max-support-abs-p95", type=float)
    parser.add_argument("--max-rotation-geodesic-p95-deg", type=float)
    parser.add_argument("--max-rotation-view-p95-deg", type=float)
    parser.add_argument("--max-rotation-inplane-p95-deg", type=float)
    parser.add_argument("--max-translation-p95", type=float)
    parser.add_argument("--min-class-agreement", type=float)
    parser.add_argument("--require-exact-schedule", action="store_true")
    parser.add_argument("--require-exact-convergence", action="store_true")
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _infer_final_star(relion_stars: dict[int, Path]) -> Path | None:
    candidates = {
        path.with_name(re.sub(r"_it\d+_data\.star$", "_data.star", path.name))
        for path in relion_stars.values()
        if re.search(r"_it\d+_data\.star$", path.name)
    }
    present = sorted(path for path in candidates if path.is_file())
    if len(present) > 1:
        raise AuditError(f"ambiguous inferred RELION final STAR files: {present}")
    return present[0] if present else None


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output_json.expanduser().resolve()
    output_npz = (
        args.output_npz.expanduser().resolve()
        if args.output_npz is not None
        else output.with_name(f"{output.stem}_arrays.npz")
    )
    output_manifest = (
        args.output_hash_manifest.expanduser().resolve()
        if args.output_hash_manifest is not None
        else output.with_suffix(output.suffix + ".sha256")
    )
    artifact_arrays: dict[str, np.ndarray] = {}
    try:
        relion_stars = _star_specs(args.relion_star, label="RELION")
        relion_final_star = (
            args.relion_final_star.expanduser().resolve()
            if args.relion_final_star is not None
            else _infer_final_star(relion_stars)
        )
        threshold_names = (
            "max_pmax_abs_p95",
            "max_pmax_abs_max",
            "max_support_abs_p95",
            "max_rotation_geodesic_p95_deg",
            "max_rotation_view_p95_deg",
            "max_rotation_inplane_p95_deg",
            "max_translation_p95",
            "min_class_agreement",
        )
        thresholds = {
            name: float(getattr(args, name))
            for name in threshold_names
            if getattr(args, name) is not None
        }
        report = audit(
            recovar_results=args.recovar_results,
            recovar_particles_star=args.recovar_particles_star,
            relion_stars=relion_stars,
            control_stars=_star_specs(args.relion_control_star, label="RELION control"),
            control_reference_stars=_star_specs(
                args.relion_control_reference_star, label="RELION control reference"
            ),
            recovar_iterations=None if args.recovar_iteration is None else set(args.recovar_iteration),
            relion_iteration_offset=args.relion_iteration_offset,
            relion_final_star=relion_final_star,
            artifact_arrays=artifact_arrays,
            thresholds=thresholds,
            require_exact_schedule=args.require_exact_schedule,
            require_exact_convergence=args.require_exact_convergence,
        )
        status = 1 if report["status"] == "fail" else 0
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
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **artifact_arrays)
    report["artifacts"] = {
        "compact_npz": {"path": str(output_npz), "sha256": _sha256_file(output_npz)},
        "hash_manifest": str(output_manifest),
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        f"{_sha256_file(output)}  {output}\n{_sha256_file(output_npz)}  {output_npz}\n"
    )
    print(f"status={report['status']} output={output}")
    print(f"arrays={output_npz} hashes={output_manifest}")
    if status == 2:
        print(f"error={report['earliest_failure']}")
    elif status == 1:
        print(f"threshold_failures={len(report['threshold_failures'])}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())

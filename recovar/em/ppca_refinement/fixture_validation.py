"""Real-fixture validation for K-class-to-PPCA initialization."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from recovar.em.ppca_refinement.initialization import (
    PPCAInitialization,
    initialize_ppca_from_gt_volumes,
    initialize_ppca_from_kclass_volumes,
    load_volume_stack,
)
from recovar.em.ppca_refinement.schedule import loading_subspace_agreement


@dataclass(frozen=True)
class KClassPPCAFixtureValidation:
    """Summary of a real K-class-to-PPCA initialization validation."""

    passed: bool
    summary: dict[str, Any]
    failures: tuple[str, ...] = field(default_factory=tuple)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _resolve_existing_path(path: str | Path) -> Path:
    p = Path(path)
    if p.exists():
        return p
    text = str(path)
    replacements = (
        ("/home/mg6942/myscratch/", "/scratch/gpfs/GILLES/mg6942/"),
        ("/scratch/gpfs/GILLES/mg6942/", "/home/mg6942/myscratch/"),
    )
    for old, new in replacements:
        if text.startswith(old):
            candidate = Path(new + text[len(old) :])
            if candidate.exists():
                return candidate
    raise FileNotFoundError(path)


def _volume_paths_from_summary(summary: dict[str, Any], *, map_key: str) -> list[Path]:
    values = summary.get(map_key)
    if not values:
        raise ValueError(f"summary does not contain any paths under {map_key!r}")
    return [_resolve_existing_path(path) for path in values]


def _weights_from_summary(summary: dict[str, Any], *, weight_source: str, n_classes: int) -> np.ndarray | None:
    if weight_source == "uniform":
        return None
    if weight_source == "recovar":
        key = "recovar_class_weights"
    elif weight_source == "relion":
        key = "relion_class_weights_in_recovar_order"
    else:
        raise ValueError("weight_source must be one of 'recovar', 'relion', or 'uniform'")
    weights = summary.get(key)
    if weights is None:
        raise ValueError(f"summary is missing {key!r}")
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (n_classes,):
        raise ValueError(f"{key} has shape {weights.shape}, expected ({n_classes},)")
    return weights


def _relative_error(a: float, b: float) -> float:
    denom = max(abs(float(a)), abs(float(b)), np.finfo(np.float64).eps)
    return float(abs(float(a) - float(b)) / denom)


def _safe_correlation(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _volume_stats(volumes: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(volumes)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "finite": bool(np.all(np.isfinite(arr))),
        "rms": float(np.sqrt(np.mean(np.abs(arr) ** 2))),
        "abs_max": float(np.max(np.abs(arr))) if arr.size else 0.0,
        "mean_abs": float(np.mean(np.abs(arr))) if arr.size else 0.0,
    }


def _covariance_trace(volumes: np.ndarray, weights: np.ndarray) -> float:
    flat = np.asarray(volumes).reshape(volumes.shape[0], -1)
    weights = np.asarray(weights, dtype=np.float64)
    mu = np.sum(weights[:, None] * flat, axis=0)
    centered = flat - mu[None, :]
    return float(np.real(np.sum(weights * np.sum(np.abs(centered) ** 2, axis=1))))


def _loading_trace(W: np.ndarray) -> float:
    return float(np.real(np.sum(np.abs(np.asarray(W)) ** 2)))


def _manifest_volume_paths_in_recovar_order(
    manifest_path: str | Path,
    *,
    recovar_to_relion: list[int] | tuple[int, ...] | np.ndarray | None,
) -> list[Path]:
    manifest = json.loads(Path(manifest_path).read_text())
    by_class = {int(row["class_index"]): _resolve_existing_path(row["volume_path"]) for row in manifest}
    if recovar_to_relion is None:
        order = sorted(by_class)
    else:
        order = [int(idx) for idx in recovar_to_relion]
    return [by_class[idx] for idx in order]


def validate_kclass_to_ppca_initialization(
    summary_json: str | Path,
    *,
    q: int | None = None,
    map_key: str = "output_maps",
    kclass_frame: str = "relion",
    weight_source: str = "recovar",
    class_manifest_json: str | Path | None = None,
    gt_frame: str = "recovar",
    covariance_trace_rtol: float = 1e-5,
    double_load_rtol: float = 1e-6,
    double_load_subspace_atol: float = 1e-4,
    min_explained_covariance: float = 0.999999,
    min_gt_mean_correlation: float | None = None,
    min_gt_subspace_agreement: float | None = None,
) -> tuple[KClassPPCAFixtureValidation, PPCAInitialization, PPCAInitialization | None]:
    """Validate PPCA initialization from a real K-class parity summary.

    The K-class parity harness writes RELION-frame MRC maps. By default this
    function therefore loads paths with ``frame="relion"`` and separately
    verifies that preloading those maps once into RECOVAR arrays gives the same
    PPCA mean/subspace. That check catches accidental double conversion.
    """
    summary_path = Path(summary_json)
    summary_in = json.loads(summary_path.read_text())
    volume_paths = _volume_paths_from_summary(summary_in, map_key=map_key)
    n_classes = len(volume_paths)
    q = max(n_classes - 1, 0) if q is None else int(q)
    weights = _weights_from_summary(summary_in, weight_source=weight_source, n_classes=n_classes)

    init = initialize_ppca_from_kclass_volumes(volume_paths, q=q, class_weights=weights, frame=kclass_frame)
    expected_trace = _covariance_trace(init.aligned_volumes, init.weights)
    actual_trace = _loading_trace(init.W)
    trace_rel_error = _relative_error(actual_trace, expected_trace)
    explained_covariance = 1.0 if expected_trace == 0.0 else float(actual_trace / expected_trace)

    preloaded = load_volume_stack(volume_paths, frame=kclass_frame)
    init_preloaded = initialize_ppca_from_kclass_volumes(preloaded, q=q, class_weights=weights, frame="recovar")
    mu_double_rel_error = _relative_error(float(np.linalg.norm(init.mu)), float(np.linalg.norm(init_preloaded.mu)))
    if np.linalg.norm(init.mu) or np.linalg.norm(init_preloaded.mu):
        mu_double_rel_error = float(
            np.linalg.norm(init.mu - init_preloaded.mu)
            / max(float(np.linalg.norm(init.mu)), float(np.linalg.norm(init_preloaded.mu)), np.finfo(np.float64).eps)
        )
    W_subspace_double = loading_subspace_agreement(init.W, init_preloaded.W)
    W_trace_double_rel_error = _relative_error(_loading_trace(init.W), _loading_trace(init_preloaded.W))

    gt_init = None
    gt_summary: dict[str, Any] = {}
    if class_manifest_json is not None:
        recovar_to_relion = summary_in.get("best_permutation", {}).get("recovar_to_relion")
        gt_paths = _manifest_volume_paths_in_recovar_order(
            class_manifest_json,
            recovar_to_relion=recovar_to_relion,
        )
        gt_init = initialize_ppca_from_gt_volumes(gt_paths, q=q, weights=init.weights, frame=gt_frame)
        gt_summary = {
            "manifest_json": str(class_manifest_json),
            "volume_paths": [str(path) for path in gt_paths],
            "frame": gt_frame,
            "mean_correlation": _safe_correlation(init.mu, gt_init.mu),
            "W_subspace_agreement": loading_subspace_agreement(init.W, gt_init.W),
            "covariance_trace": _loading_trace(gt_init.W),
            "kclass_to_gt_covariance_trace_ratio": (
                None if _loading_trace(gt_init.W) == 0.0 else float(_loading_trace(init.W) / _loading_trace(gt_init.W))
            ),
        }

    failures: list[str] = []
    if not np.all(np.isfinite(init.aligned_volumes)):
        failures.append("nonfinite K-class input volume values")
    if not np.all(np.isfinite(init.mu)) or not np.all(np.isfinite(init.W)):
        failures.append("nonfinite PPCA mu/W")
    if trace_rel_error > covariance_trace_rtol:
        failures.append(
            f"loading covariance trace mismatch: rel_error={trace_rel_error:.3g} > {covariance_trace_rtol:.3g}"
        )
    if explained_covariance + covariance_trace_rtol < min_explained_covariance:
        failures.append(
            f"explained covariance too low: {explained_covariance:.9g} < {min_explained_covariance:.9g}"
        )
    if mu_double_rel_error > double_load_rtol:
        failures.append(f"path/preloaded mean mismatch: rel_error={mu_double_rel_error:.3g} > {double_load_rtol:.3g}")
    if W_trace_double_rel_error > double_load_rtol:
        failures.append(
            f"path/preloaded W trace mismatch: rel_error={W_trace_double_rel_error:.3g} > {double_load_rtol:.3g}"
        )
    # The path-vs-preloaded check compares identical arrays through a large
    # float32 QR/SVD diagnostic. On 128^3+ maps the subspace score is more
    # sensitive to linear algebra roundoff than the scalar trace and mean
    # checks, so keep a separate tolerance for this diagnostic.
    if W_subspace_double < 1.0 - double_load_subspace_atol:
        failures.append(f"path/preloaded W subspace mismatch: agreement={W_subspace_double:.9g}")
    if min_gt_mean_correlation is not None and gt_summary:
        if float(gt_summary["mean_correlation"]) < float(min_gt_mean_correlation):
            failures.append(
                f"GT mean correlation too low: {gt_summary['mean_correlation']:.9g} < {min_gt_mean_correlation:.9g}"
            )
    if min_gt_subspace_agreement is not None and gt_summary:
        if float(gt_summary["W_subspace_agreement"]) < float(min_gt_subspace_agreement):
            failures.append(
                "GT W subspace agreement too low: "
                f"{gt_summary['W_subspace_agreement']:.9g} < {min_gt_subspace_agreement:.9g}"
            )

    validation_summary = {
        "summary_json": str(summary_path),
        "map_key": map_key,
        "volume_paths": [str(path) for path in volume_paths],
        "kclass_frame": kclass_frame,
        "weight_source": weight_source,
        "weights": init.weights,
        "n_classes": n_classes,
        "q": q,
        "initializer_diagnostics": init.diagnostics,
        "kclass_volume_stats": _volume_stats(init.aligned_volumes),
        "mu_stats": _volume_stats(init.mu),
        "W_stats": _volume_stats(init.W),
        "covariance_trace": {
            "expected_weighted_centered_trace": expected_trace,
            "loading_trace": actual_trace,
            "relative_error": trace_rel_error,
            "explained_covariance": explained_covariance,
        },
        "loader_frame_check": {
            "preloaded_frame": "recovar",
            "mu_relative_error": mu_double_rel_error,
            "W_trace_relative_error": W_trace_double_rel_error,
            "W_subspace_agreement": W_subspace_double,
            "W_subspace_atol": double_load_subspace_atol,
        },
        "gt_comparison": gt_summary or None,
    }
    validation_summary = _jsonable(validation_summary)
    return (
        KClassPPCAFixtureValidation(
            passed=not failures,
            summary=validation_summary,
            failures=tuple(failures),
        ),
        init,
        gt_init,
    )

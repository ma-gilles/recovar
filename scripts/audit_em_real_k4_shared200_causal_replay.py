#!/usr/bin/env python3
"""Fail-closed audit of the EMPIAR-10076 K=4 shared-200 causal replay.

The audit joins every native RELION and RECOVAR fine-search table by the
immutable stack/class/rotation/translation identity.  It gates direct array
errors (never correlation), global posterior/support/winner agreement,
particle-class assignments, and signed shellwise FSC/FSC-AUC for maps.
Malformed or incomplete topology raises before a report is written.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from recovar.data_io.starfile import read_star
from scripts.analyze_em_k4_allclass_native_boundary import exact_rotation_permutation
from scripts.launch_em_real_k4_shared200_causal_replay_slurm import (
    CASE,
    TARGET_SCHEMA,
    validate_manifest,
)
from scripts.summarize_em_completion_bench import (
    _load_recovar_volume,
    _load_relion_volume,
    normalized_fsc_auc,
    shell_fsc,
)
from scripts.validate_relion_bpref_factor_capture import load_factor_capture
from scripts.validate_relion_fine_score_capture import ACTIVE, load_fine_score_capture

SCHEMA = "recovar.em_real_k4_shared200_causal_replay_audit.v2"
PASS2_NAME = re.compile(r"pass2_orig(?P<original>[0-9]{6})_class(?P<class_>[0-9]{3})_cs(?P<size>[0-9]{3})[.]npz")
FACTOR_NAME = re.compile(
    r"part(?P<part>[0-9]+)_stack(?P<stack>[0-9]+)_img(?P<img>[0-9]+)_class(?P<class_>[0-9]+)[.]bpre-v2[.]bin"
)
SCORE_NAME = re.compile(r"part(?P<part>[0-9]+)_stack(?P<stack>[0-9]+)_class(?P<class_>[0-9]+)[.]fine-score-v1[.]bin")


class AuditError(RuntimeError):
    """Raised when causal-replay evidence is incomplete or ambiguous."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _finite(value: float, *, label: str) -> float:
    value = float(value)
    _require(math.isfinite(value), f"{label} is non-finite")
    return value


def _float32_from_bits(value: int) -> np.float32:
    return np.asarray(value & 0xFFFFFFFF, dtype=np.uint32).view(np.float32)[()]


@dataclass
class ErrorAccumulator:
    """Scale-sensitive L2 error accumulator with optional offset removal."""

    numerator_sq: float = 0.0
    denominator_sq: float = 0.0
    maximum_abs: float = 0.0
    count: int = 0

    def add(self, left: np.ndarray, right: np.ndarray, *, center: bool) -> None:
        left64 = np.asarray(left, dtype=np.float64).reshape(-1)
        right64 = np.asarray(right, dtype=np.float64).reshape(-1)
        _require(left64.shape == right64.shape and left64.size > 0, "metric arrays differ or are empty")
        _require(np.isfinite(left64).all() and np.isfinite(right64).all(), "metric array is non-finite")
        if center:
            left64 = left64 - math.fsum(float(value) for value in left64) / left64.size
            right64 = right64 - math.fsum(float(value) for value in right64) / right64.size
        delta = left64 - right64
        self.numerator_sq += math.fsum(float(value) * float(value) for value in delta)
        self.denominator_sq += math.fsum(float(value) * float(value) for value in right64)
        self.maximum_abs = max(self.maximum_abs, float(np.max(np.abs(delta), initial=0.0)))
        self.count += int(delta.size)

    def report(self) -> dict[str, Any]:
        _require(self.count > 0, "metric accumulator is empty")
        denominator = max(math.sqrt(self.denominator_sq), np.finfo(np.float64).tiny)
        return {
            "count": self.count,
            "relative_l2": math.sqrt(self.numerator_sq) / denominator,
            "maximum_abs": self.maximum_abs,
            "correlation_used": False,
        }


def _exact_keyed_paths(
    paths: Iterable[Path],
    *,
    key: Callable[[Path], tuple[int, int]],
    expected: set[tuple[int, int]],
    label: str,
) -> dict[tuple[int, int], Path]:
    result: dict[tuple[int, int], Path] = {}
    for path in paths:
        item = key(path)
        _require(item not in result, f"duplicate {label} key {item}: {path}")
        result[item] = path
    _require(
        set(result) == expected,
        f"{label} topology differs: missing={sorted(expected - set(result))[:10]}, "
        f"extra={sorted(set(result) - expected)[:10]}",
    )
    return result


def _factor_key(path: Path) -> tuple[int, int]:
    match = FACTOR_NAME.fullmatch(path.name)
    _require(match is not None, f"unexpected factor filename: {path.name}")
    return int(match["stack"]), int(match["class_"])


def _score_key(path: Path) -> tuple[int, int]:
    match = SCORE_NAME.fullmatch(path.name)
    _require(match is not None, f"unexpected fine-score filename: {path.name}")
    return int(match["stack"]), int(match["class_"])


def _pass2_key(path: Path) -> tuple[int, int]:
    match = PASS2_NAME.fullmatch(path.name)
    _require(match is not None, f"unexpected RECOVAR pass-2 filename: {path.name}")
    _require(int(match["size"]) == CASE.current_size, f"RECOVAR current-size filename drift: {path}")
    return int(match["original"]) + 1, int(match["class_"])


def _column(table, name: str) -> str:
    for candidate in (name, f"_{name}" if not name.startswith("_") else name[1:]):
        if candidate in table.columns:
            return candidate
    raise AuditError(f"STAR table lacks {name}")


def _stack_index(value: object) -> int:
    prefix, separator, _ = str(value).partition("@")
    _require(bool(separator), f"invalid image identity: {value!r}")
    try:
        result = int(prefix)
    except ValueError as exc:
        raise AuditError(f"invalid image identity: {value!r}") from exc
    _require(result > 0, f"invalid stack index: {value!r}")
    return result


def _native_assignments(path: Path) -> dict[int, int]:
    particles, _ = read_star(str(path))
    image_column = _column(particles, "rlnImageName")
    class_column = _column(particles, "rlnClassNumber")
    stacks = [_stack_index(value) for value in particles[image_column]]
    _require(len(stacks) == len(set(stacks)), f"duplicate image identities: {path}")
    classes = np.asarray(particles[class_column], dtype=np.int64)
    _require(np.all((classes >= 0) & (classes <= CASE.K)), f"class labels out of range: {path}")
    return {stack: int(class_id) for stack, class_id in zip(stacks, classes, strict=True) if class_id > 0}


def discover_inventory(
    run_root: Path,
    stacks: list[int],
    *,
    reference_assignments: dict[int, int],
) -> dict[str, Any]:
    expected = {(stack, class_id) for stack in stacks for class_id in range(1, CASE.K + 1)}
    factors: list[Path] = []
    scores: list[Path] = []
    for class_id in range(1, CASE.K + 1):
        directory = run_root / f"native/class{class_id}/factors"
        _require(directory.is_dir(), f"missing native capture directory: {directory}")
        class_factors = sorted(directory.glob("*.bpre-v2.bin"))
        class_scores = sorted(directory.glob("*.fine-score-v1.bin"))
        _require(len(class_factors) == CASE.particle_count, f"class {class_id} factor count drift")
        _require(len(class_scores) == CASE.particle_count, f"class {class_id} score count drift")
        _require(
            len(list(directory.glob("*.bin"))) == 2 * CASE.particle_count,
            f"class {class_id} has unexpected binary captures",
        )
        factors.extend(class_factors)
        scores.extend(class_scores)

    pass2_directory = run_root / "recovar/pass2"
    _require(pass2_directory.is_dir(), f"missing RECOVAR pass-2 directory: {pass2_directory}")
    pass2_paths = sorted(pass2_directory.glob("*.npz"))
    _require(len(pass2_paths) == CASE.particle_count * CASE.K, "RECOVAR pass-2 count drift")
    native_assignments = {}
    for arm in ("control_a", "control_b", "class1", "class2", "class3", "class4"):
        output = run_root / f"native/{arm}/output"
        data_star = output / "run_it001_data.star"
        _require(data_star.is_file(), f"missing native data STAR: {data_star}")
        for class_id in range(1, CASE.K + 1):
            _require(
                (output / f"run_it001_class{class_id:03d}.mrc").is_file(),
                f"missing native class map: arm={arm} class={class_id}",
            )
        observed = _native_assignments(data_star)
        _require(set(observed) == set(stacks), f"native assigned set drift: {arm}")
        native_assignments[arm] = observed
    _require(set(reference_assignments) == set(stacks), "frozen target assigned set drift")
    # Instrumentation inertness is a same-replay question. Compare every arm
    # with the uncaptured replay, and measure restart drift against the frozen
    # fresh-run checkpoint separately.
    control_assignments = native_assignments["control_a"]
    inertness_matches = [
        int(native_assignments[arm][stack] == control_assignments[stack])
        for arm in native_assignments
        for stack in stacks
    ]
    frozen_target_matches = [
        int(control_assignments[stack] == reference_assignments[stack]) for stack in stacks
    ]
    return {
        "factors": _exact_keyed_paths(factors, key=_factor_key, expected=expected, label="factor"),
        "scores": _exact_keyed_paths(scores, key=_score_key, expected=expected, label="fine-score"),
        "pass2": _exact_keyed_paths(pass2_paths, key=_pass2_key, expected=expected, label="pass-2"),
        "counts": {
            "native_factors": len(factors),
            "native_fine_scores": len(scores),
            "recovar_pass2": len(pass2_paths),
            "native_data_stars": len(native_assignments),
        },
        "native_assignments": native_assignments,
        "native_assignment_inertness": float(np.mean(inertness_matches)),
        "frozen_target_assignment_accuracy": float(np.mean(frozen_target_matches)),
    }


def _load_recovar(path: Path, *, stack: int, class_id: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "original_index",
        "class_index",
        "current_size",
        "rotations",
        "candidate_mask",
        "scores_with_prior",
        "probs",
        "reconstruction_probs",
        "rotation_log_prior",
        "translation_log_prior",
        "reconstruction_mask",
        "relion_raw_diff2",
    }
    _require(required <= set(values), f"RECOVAR pass-2 capture lacks {sorted(required - set(values))}: {path}")
    _require(int(values["original_index"]) == stack - 1, f"RECOVAR stack identity drift: {path}")
    _require(int(values["class_index"]) == class_id - 1, f"RECOVAR class identity drift: {path}")
    _require(int(values["current_size"]) == CASE.current_size, f"RECOVAR current size drift: {path}")
    return values


def _join_class(
    *,
    stack: int,
    class_id: int,
    factor_path: Path,
    score_path: Path,
    pass2_path: Path,
) -> dict[str, Any]:
    factor = load_factor_capture(factor_path)
    score = load_fine_score_capture(score_path)
    recovar = _load_recovar(pass2_path, stack=stack, class_id=class_id)
    _require(factor.geometry_only, f"factor capture is not geometry-only: {factor_path}")
    _require(
        (score.header[4], score.header[5], score.header[7]) == (CASE.iteration, class_id, stack),
        f"fine-score identity drift: {score_path}",
    )
    _require(
        (factor.header[9], factor.header[10], factor.header[12]) == (CASE.iteration, class_id, stack),
        f"factor identity drift: {factor_path}",
    )
    _require(score.header[6] == factor.header[11], f"native particle identity differs: stack {stack} class {class_id}")

    recovar_rotations = np.asarray(recovar["rotations"], dtype=np.float32)
    candidate_mask = np.asarray(recovar["candidate_mask"], dtype=bool)
    recovar_posterior = np.asarray(recovar["probs"], dtype=np.float64)
    recovar_reconstruction_posterior = np.asarray(
        recovar["reconstruction_probs"],
        dtype=np.float64,
    )
    recovar_support = np.asarray(recovar["reconstruction_mask"], dtype=bool)
    _require(
        recovar_rotations.shape == (candidate_mask.shape[0], 3, 3)
        and candidate_mask.shape[1] == int(factor.header[21]),
        f"candidate geometry drift: stack {stack} class {class_id}",
    )
    _require(
        recovar_posterior.shape
        == recovar_reconstruction_posterior.shape
        == candidate_mask.shape
        == recovar_support.shape,
        "RECOVAR posterior/support geometry drift",
    )
    _require(np.all(recovar_support <= candidate_mask), "RECOVAR support is outside candidates")
    _require(
        np.isfinite(recovar_posterior).all()
        and np.isfinite(recovar_reconstruction_posterior).all(),
        "RECOVAR posterior is non-finite",
    )
    _require(
        np.all(recovar_posterior >= 0.0)
        and np.all(recovar_reconstruction_posterior >= 0.0),
        "RECOVAR posterior is negative",
    )
    _require(
        np.array_equal(recovar_support, recovar_reconstruction_posterior > 0.0),
        "RECOVAR reconstruction mask does not equal its positive posterior support",
    )
    _require(
        np.array_equal(
            recovar_reconstruction_posterior,
            np.where(recovar_support, recovar_posterior, 0.0),
        ),
        "RECOVAR retained posterior does not equal the full posterior on support",
    )

    _require(
        score.empty_sparse_support == factor.empty_sparse_support,
        f"native empty-support sentinels disagree: stack {stack} class {class_id}",
    )
    significant_weight = _float32_from_bits(int(factor.header[25]))
    weight_norm = _float32_from_bits(int(factor.header[26]))
    _require(
        np.isfinite(significant_weight) and np.isfinite(weight_norm) and weight_norm > 0,
        "invalid native posterior scalars",
    )
    if score.empty_sparse_support:
        _require(score.candidates.size == 0, "empty fine-score sentinel contains candidates")
        if factor.rotations.size:
            native_rotations = (
                np.asarray(factor.rotations["matrix"], dtype=np.float32)
                .reshape(-1, 3, 3)
                .transpose(0, 2, 1)
            )
            exact_rotation_permutation(native_rotations, recovar_rotations)
        _require(not np.any(candidate_mask), "RECOVAR candidate mask is nonempty for native empty support")
        _require(not np.any(recovar_posterior), "RECOVAR posterior is nonzero for native empty support")
        _require(
            not np.any(recovar_reconstruction_posterior) and not np.any(recovar_support),
            "RECOVAR retained support is nonzero for native empty support",
        )
        empty_scores = np.empty(0, dtype=np.float32)
        native_posterior = np.zeros(candidate_mask.shape, dtype=np.float32)
        native_support = np.zeros(candidate_mask.shape, dtype=bool)
        return {
            "stack": stack,
            "class_id": class_id,
            "particle_id": int(score.header[6]),
            "empty_sparse_support": True,
            "candidate_exact": True,
            "candidate_intersection": 0,
            "candidate_union": 0,
            "native_raw": empty_scores,
            "recovar_raw": empty_scores.copy(),
            "native_combined": empty_scores.copy(),
            "recovar_combined": empty_scores.copy(),
            "native_posterior": native_posterior,
            "recovar_posterior": recovar_posterior,
            "native_reconstruction_posterior": native_posterior.copy(),
            "recovar_reconstruction_posterior": recovar_reconstruction_posterior,
            "native_support": native_support,
            "recovar_support": recovar_support,
            "significant_weight_bits": int(factor.header[25]),
            "weight_norm_bits": int(factor.header[26]),
        }

    native_rotations = np.asarray(factor.rotations["matrix"], dtype=np.float32).reshape(-1, 3, 3).transpose(0, 2, 1)
    native_to_recovar = exact_rotation_permutation(native_rotations, recovar_rotations)
    _require(
        candidate_mask.shape == (native_to_recovar.size, int(factor.header[21])),
        f"candidate geometry drift: stack {stack} class {class_id}",
    )
    candidates = score.candidates
    active = (candidates["flags"] & ACTIVE) != 0
    native_rotation = np.asarray(candidates["rotation_local"], dtype=np.int64)
    translations = np.asarray(candidates["translation_id"], dtype=np.int64)
    _require(
        np.all((native_rotation >= 0) & (native_rotation < native_to_recovar.size)),
        "native rotation index is out of range",
    )
    _require(
        np.all((translations >= 0) & (translations < candidate_mask.shape[1])),
        "native translation index is out of range",
    )
    mapped_rotation = native_to_recovar[native_rotation]
    native_candidates = np.zeros(candidate_mask.shape, dtype=bool)
    native_candidates[mapped_rotation[active], translations[active]] = True
    _require(
        int(np.count_nonzero(native_candidates)) == int(np.count_nonzero(active)),
        "native active tuple keys are not unique",
    )
    tuple_exact = bool(np.array_equal(native_candidates, candidate_mask))
    common = active & candidate_mask[mapped_rotation, translations]
    _require(np.any(common), "native/RECOVAR candidate intersection is empty")
    mapped_common_rotation = mapped_rotation[common]
    common_translation = translations[common]

    native_posterior = np.zeros(candidate_mask.shape, dtype=np.float32)
    native_posterior[mapped_rotation[active], translations[active]] = np.divide(
        candidates["post_exponent_weight"][active], weight_norm, dtype=np.float32
    )
    native_support = np.zeros(candidate_mask.shape, dtype=bool)
    native_significant_rows = active & (candidates["post_exponent_weight"] >= significant_weight)
    native_support[mapped_rotation[native_significant_rows], translations[native_significant_rows]] = True
    _require(
        int(np.count_nonzero(native_support)) == int(factor.header[45]),
        "native support does not replay the BPref header",
    )
    return {
        "stack": stack,
        "class_id": class_id,
        "particle_id": int(score.header[6]),
        "empty_sparse_support": False,
        "candidate_exact": tuple_exact,
        "candidate_intersection": int(np.count_nonzero(native_candidates & candidate_mask)),
        "candidate_union": int(np.count_nonzero(native_candidates | candidate_mask)),
        "native_raw": np.asarray(candidates["raw_diff2"][common], dtype=np.float32),
        "recovar_raw": np.asarray(
            recovar["relion_raw_diff2"][mapped_common_rotation, common_translation], dtype=np.float32
        ),
        "native_combined": np.asarray(candidates["combined_preexponent"][common], dtype=np.float32),
        "recovar_combined": np.asarray(
            recovar["scores_with_prior"][mapped_common_rotation, common_translation], dtype=np.float32
        ),
        "native_posterior": native_posterior,
        "recovar_posterior": recovar_posterior,
        "native_reconstruction_posterior": np.where(
            native_support,
            native_posterior,
            np.float32(0.0),
        ),
        "recovar_reconstruction_posterior": recovar_reconstruction_posterior,
        "native_support": native_support,
        "recovar_support": recovar_support,
        "significant_weight_bits": int(factor.header[25]),
        "weight_norm_bits": int(factor.header[26]),
    }


def _fsc_metric(left: np.ndarray, right: np.ndarray) -> tuple[float, list[float | None]]:
    curve = np.asarray(shell_fsc(left, right), dtype=np.float64).reshape(-1)
    _require(curve.size > 1 and np.isfinite(curve[1:]).any(), "map comparison has no finite non-DC FSC shells")
    auc = _finite(normalized_fsc_auc(curve), label="FSC-AUC")
    return auc, [float(value) if np.isfinite(value) else None for value in curve]


def _particle_mass_diagnostics(joined: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare full and BPref-retained mass with one joint K-class normalization.

    The counterfactual denominator is deliberately computed once after
    concatenating every class and every retained pose/translation hypothesis
    for the particle.  Normalizing one class at a time would erase the class
    posterior and is not a valid K-class causal arm.
    """

    _require(bool(joined), "particle mass diagnostics require at least one class")
    _require(
        [int(item["class_id"]) for item in joined] == list(range(1, len(joined) + 1)),
        "particle classes must be complete and ordered",
    )
    native_full = np.concatenate(
        [np.asarray(item["native_posterior"], dtype=np.float64).reshape(-1) for item in joined]
    )
    recovar_full = np.concatenate(
        [np.asarray(item["recovar_posterior"], dtype=np.float64).reshape(-1) for item in joined]
    )
    native_retained = np.concatenate(
        [
            np.asarray(item["native_reconstruction_posterior"], dtype=np.float64).reshape(-1)
            for item in joined
        ]
    )
    recovar_retained = np.concatenate(
        [
            np.asarray(item["recovar_reconstruction_posterior"], dtype=np.float64).reshape(-1)
            for item in joined
        ]
    )
    for label, values in (
        ("native full posterior", native_full),
        ("RECOVAR full posterior", recovar_full),
        ("native retained posterior", native_retained),
        ("RECOVAR retained posterior", recovar_retained),
    ):
        _require(values.size > 0 and np.isfinite(values).all(), f"{label} is empty or non-finite")
        _require(np.all(values >= 0.0), f"{label} is negative")

    native_full_mass = float(np.sum(native_full, dtype=np.float64))
    recovar_full_mass = float(np.sum(recovar_full, dtype=np.float64))
    native_retained_mass = float(np.sum(native_retained, dtype=np.float64))
    recovar_retained_mass = float(np.sum(recovar_retained, dtype=np.float64))
    _require(native_retained_mass > 0.0, "native retained posterior mass is zero")
    _require(recovar_retained_mass > 0.0, "RECOVAR retained posterior mass is zero")

    normalized_native_retained = native_retained / native_retained_mass
    normalized_recovar_retained = recovar_retained / recovar_retained_mass
    native_norm = max(float(np.linalg.norm(native_retained)), np.finfo(np.float64).tiny)
    normalized_native_norm = max(
        float(np.linalg.norm(normalized_native_retained)),
        np.finfo(np.float64).tiny,
    )

    class_rows = []
    normalized_offset = 0
    for item in joined:
        native_class_full = np.asarray(item["native_posterior"], dtype=np.float64).reshape(-1)
        recovar_class_full = np.asarray(item["recovar_posterior"], dtype=np.float64).reshape(-1)
        native_class_retained = np.asarray(
            item["native_reconstruction_posterior"], dtype=np.float64
        ).reshape(-1)
        recovar_class_retained = np.asarray(
            item["recovar_reconstruction_posterior"], dtype=np.float64
        ).reshape(-1)
        normalized_stop = normalized_offset + recovar_class_retained.size
        native_support = np.asarray(item["native_support"], dtype=bool)
        recovar_support = np.asarray(item["recovar_support"], dtype=bool)
        intersection = int(np.count_nonzero(native_support & recovar_support))
        union = int(np.count_nonzero(native_support | recovar_support))
        native_class_full_mass = float(np.sum(native_class_full, dtype=np.float64))
        recovar_class_full_mass = float(np.sum(recovar_class_full, dtype=np.float64))
        native_class_retained_mass = float(np.sum(native_class_retained, dtype=np.float64))
        recovar_class_retained_mass = float(np.sum(recovar_class_retained, dtype=np.float64))
        class_rows.append(
            {
                "class_id_one_based": int(item["class_id"]),
                "native_full_mass": native_class_full_mass,
                "recovar_full_mass": recovar_class_full_mass,
                "native_retained_mass": native_class_retained_mass,
                "recovar_retained_mass": recovar_class_retained_mass,
                "native_discarded_mass": native_class_full_mass - native_class_retained_mass,
                "recovar_discarded_mass": recovar_class_full_mass - recovar_class_retained_mass,
                "recovar_jointly_normalized_retained_mass": float(
                    np.sum(
                        normalized_recovar_retained[normalized_offset:normalized_stop],
                        dtype=np.float64,
                    )
                ),
                "native_support_count": int(np.count_nonzero(native_support)),
                "recovar_support_count": int(np.count_nonzero(recovar_support)),
                "support_intersection": intersection,
                "support_union": union,
                "support_jaccard": 1.0 if union == 0 else intersection / float(union),
            }
        )
        normalized_offset = normalized_stop
    _require(normalized_offset == normalized_recovar_retained.size, "class mass slices are incomplete")

    live_delta = recovar_retained - native_retained
    shape_delta = normalized_recovar_retained - normalized_native_retained
    causal_delta = normalized_recovar_retained - native_retained
    return {
        "native_full_mass": native_full_mass,
        "recovar_full_mass": recovar_full_mass,
        "native_retained_mass": native_retained_mass,
        "recovar_retained_mass": recovar_retained_mass,
        "native_discarded_mass": native_full_mass - native_retained_mass,
        "recovar_discarded_mass": recovar_full_mass - recovar_retained_mass,
        "recovar_joint_retained_normalization_factor": 1.0 / recovar_retained_mass,
        "live_retained_relative_l2": float(np.linalg.norm(live_delta)) / native_norm,
        "jointly_normalized_retained_shape_relative_l2": (
            float(np.linalg.norm(shape_delta)) / normalized_native_norm
        ),
        "recovar_normalized_only_vs_native_live_relative_l2": (
            float(np.linalg.norm(causal_delta)) / native_norm
        ),
        "class_rows": class_rows,
        "normalization_scope": "one denominator over all retained classes, rotations, and translations",
    }


def _map_metrics(
    run_root: Path,
    permutation: np.ndarray,
    *,
    frozen_target_maps: list[Path],
) -> dict[str, Any]:
    _require(permutation.shape == (CASE.K,), "class permutation shape drift")
    _require(np.array_equal(np.sort(permutation), np.arange(CASE.K)), "class permutation is not bijective")
    controls = {
        label: [
            _load_relion_volume(run_root / f"native/control_{label}/output/run_it001_class{class_id:03d}.mrc")
            for class_id in range(1, CASE.K + 1)
        ]
        for label in ("a", "b")
    }
    cross_engine = []
    control_repeat = []
    capture_inertness = []
    frozen_target = []
    for recovar_index, relion_index in enumerate(permutation):
        recovar = _load_recovar_volume(run_root / f"recovar/output/recovar_class{recovar_index + 1:03d}.mrc")
        auc, curve = _fsc_metric(recovar, controls["a"][int(relion_index)])
        cross_engine.append(
            {
                "recovar_class": recovar_index + 1,
                "relion_class": int(relion_index) + 1,
                "fsc_auc": auc,
                "shellwise_fsc": curve,
            }
        )
    for class_index in range(CASE.K):
        auc, curve = _fsc_metric(controls["a"][class_index], controls["b"][class_index])
        control_repeat.append({"class": class_index + 1, "fsc_auc": auc, "shellwise_fsc": curve})
        frozen = _load_relion_volume(frozen_target_maps[class_index])
        target_auc, target_curve = _fsc_metric(controls["a"][class_index], frozen)
        frozen_target.append({"class": class_index + 1, "fsc_auc": target_auc, "shellwise_fsc": target_curve})
    for capture_class in range(1, CASE.K + 1):
        for map_class in range(1, CASE.K + 1):
            captured = _load_relion_volume(
                run_root / f"native/class{capture_class}/output/run_it001_class{map_class:03d}.mrc"
            )
            auc, curve = _fsc_metric(captured, controls["a"][map_class - 1])
            capture_inertness.append(
                {
                    "capture_class": capture_class,
                    "map_class": map_class,
                    "fsc_auc": auc,
                    "shellwise_fsc": curve,
                }
            )
    return {
        "cross_engine": cross_engine,
        "native_control_repeatability": control_repeat,
        "native_capture_inertness": capture_inertness,
        "frozen_target_replay": frozen_target,
        "minimum_cross_engine_fsc_auc": min(row["fsc_auc"] for row in cross_engine),
        "minimum_native_control_fsc_auc": min(row["fsc_auc"] for row in control_repeat),
        "minimum_native_capture_inertness_fsc_auc": min(row["fsc_auc"] for row in capture_inertness),
        "minimum_frozen_target_fsc_auc": min(row["fsc_auc"] for row in frozen_target),
        "correlation_used": False,
    }


def evaluate_gates(metrics: dict[str, float], thresholds: dict[str, float]) -> dict[str, Any]:
    specifications = (
        ("candidate_tuple_exact_fraction", "minimum_candidate_tuple_exact_fraction", "minimum"),
        ("centered_raw_score_relative_l2", "maximum_centered_raw_score_relative_l2", "maximum"),
        ("centered_combined_score_relative_l2", "maximum_centered_combined_score_relative_l2", "maximum"),
        ("posterior_relative_l2", "maximum_posterior_relative_l2", "maximum"),
        ("posterior_row_sum_abs_error", "maximum_posterior_row_sum_abs_error", "maximum"),
        ("support_jaccard", "minimum_support_jaccard", "minimum"),
        ("winner_agreement", "minimum_winner_agreement", "minimum"),
        ("pmax_rmse", "maximum_pmax_rmse", "maximum"),
        ("pmax_abs_error", "maximum_pmax_abs_error", "maximum"),
        ("assignment_accuracy", "minimum_assignment_accuracy", "minimum"),
        ("cross_engine_map_fsc_auc", "minimum_per_class_map_fsc_auc", "minimum"),
        ("native_control_map_fsc_auc", "minimum_native_control_map_fsc_auc", "minimum"),
        ("native_capture_inertness_map_fsc_auc", "minimum_capture_inertness_map_fsc_auc", "minimum"),
        ("frozen_target_map_fsc_auc", "minimum_frozen_target_map_fsc_auc", "minimum"),
        ("native_assignment_inertness", "minimum_native_assignment_inertness", "minimum"),
        (
            "frozen_target_assignment_accuracy",
            "minimum_frozen_target_assignment_accuracy",
            "minimum",
        ),
        ("minimum_class_fraction", "minimum_class_fraction", "minimum"),
    )
    gates: dict[str, Any] = {}
    failures: list[str] = []
    for metric_name, threshold_name, direction in specifications:
        _require(metric_name in metrics, f"missing gate metric {metric_name}")
        _require(threshold_name in thresholds, f"missing gate threshold {threshold_name}")
        value = _finite(metrics[metric_name], label=metric_name)
        threshold = _finite(thresholds[threshold_name], label=threshold_name)
        passed = value >= threshold if direction == "minimum" else value <= threshold
        gates[metric_name] = {
            "value": value,
            "comparison": ">=" if direction == "minimum" else "<=",
            "threshold": threshold,
            "passed": passed,
        }
        if not passed:
            failures.append(f"{metric_name}={value:.12g} {gates[metric_name]['comparison']} {threshold:.12g} failed")
    return {"accepted": not failures, "gates": gates, "failures": failures}


def _parity_metrics(path: Path, stacks: list[int]) -> tuple[dict[str, Any], np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "mapped_recovar_class",
        "relion_class",
        "recovar_pmax",
        "relion_pmax",
        "recovar_to_relion",
    }
    _require(required <= set(values), f"parity arrays lack {sorted(required - set(values))}")
    for key in ("mapped_recovar_class", "relion_class", "recovar_pmax", "relion_pmax"):
        _require(values[key].reshape(-1).size == len(stacks), f"parity array {key} count drift")
    mapped = values["mapped_recovar_class"].reshape(-1).astype(np.int64)
    relion = values["relion_class"].reshape(-1).astype(np.int64)
    _require(np.all((mapped >= 1) & (mapped <= CASE.K)), "mapped RECOVAR class is out of range")
    _require(np.all((relion >= 1) & (relion <= CASE.K)), "RELION class is out of range")
    recovar_pmax = values["recovar_pmax"].reshape(-1).astype(np.float64)
    relion_pmax = values["relion_pmax"].reshape(-1).astype(np.float64)
    _require(np.isfinite(recovar_pmax).all() and np.isfinite(relion_pmax).all(), "parity Pmax is non-finite")
    delta = recovar_pmax - relion_pmax
    recovar_fractions = [float(np.mean(mapped == class_id)) for class_id in range(1, CASE.K + 1)]
    relion_fractions = [float(np.mean(relion == class_id)) for class_id in range(1, CASE.K + 1)]
    return (
        {
            "assignment_accuracy": float(np.mean(mapped == relion)),
            "pmax_rmse": float(np.sqrt(np.mean(delta * delta))),
            "pmax_abs_error": float(np.max(np.abs(delta), initial=0.0)),
            "recovar_class_fractions": recovar_fractions,
            "relion_class_fractions": relion_fractions,
            "minimum_class_fraction": min(recovar_fractions + relion_fractions),
        },
        values["recovar_to_relion"].reshape(-1).astype(np.int64),
    )


def build_report(manifest_path: Path) -> dict[str, Any]:
    manifest = validate_manifest(manifest_path)
    run_root = Path(manifest["run_root"])
    targets = json.loads(Path(manifest["targets"]["path"]).read_text())
    _require(targets.get("schema") == TARGET_SCHEMA, "target schema drift")
    stacks = [int(value) for value in targets.get("stack_indices_one_based", [])]
    _require(len(stacks) == CASE.particle_count and len(set(stacks)) == len(stacks), "target stacks drift")
    input_by_role = {record["role"]: Path(record["path"]) for record in manifest["input_records"]}
    frozen_assignments = _native_assignments(input_by_role["RELION iteration-1 data"])
    inventory = discover_inventory(
        run_root,
        stacks,
        reference_assignments=frozen_assignments,
    )

    raw = ErrorAccumulator()
    combined = ErrorAccumulator()
    posterior = ErrorAccumulator()
    exact_count = 0
    empty_sparse_support_count = 0
    support_intersection = 0
    support_union = 0
    winner_matches = 0
    native_pmax: list[float] = []
    recovar_pmax: list[float] = []
    row_sum_errors: list[float] = []
    particle_rows = []
    particle_mass_rows = []
    for stack in stacks:
        joined = []
        for class_id in range(1, CASE.K + 1):
            key = (stack, class_id)
            item = _join_class(
                stack=stack,
                class_id=class_id,
                factor_path=inventory["factors"][key],
                score_path=inventory["scores"][key],
                pass2_path=inventory["pass2"][key],
            )
            joined.append(item)
            exact_count += int(item["candidate_exact"])
            empty_sparse_support_count += int(item["empty_sparse_support"])
            if item["native_raw"].size:
                raw.add(item["native_raw"], item["recovar_raw"], center=True)
                combined.add(item["native_combined"], item["recovar_combined"], center=True)
            if item["native_posterior"].size:
                posterior.add(item["native_posterior"], item["recovar_posterior"], center=False)
            support_intersection += int(np.count_nonzero(item["native_support"] & item["recovar_support"]))
            support_union += int(np.count_nonzero(item["native_support"] | item["recovar_support"]))

        _require(
            len({item["particle_id"] for item in joined}) == 1,
            f"native particle id differs across classes for stack {stack}",
        )
        _require(
            len({item["weight_norm_bits"] for item in joined}) == 1
            and len({item["significant_weight_bits"] for item in joined}) == 1,
            f"native global posterior scalars differ across class arms for stack {stack}",
        )
        native_flat = np.concatenate([item["native_posterior"].reshape(-1) for item in joined])
        recovar_flat = np.concatenate([item["recovar_posterior"].reshape(-1) for item in joined])
        native_winner = int(np.argmax(native_flat))
        recovar_winner = int(np.argmax(recovar_flat))
        winner_matches += int(native_winner == recovar_winner)
        native_max = float(native_flat[native_winner])
        recovar_max = float(recovar_flat[recovar_winner])
        native_pmax.append(native_max)
        recovar_pmax.append(recovar_max)
        mass_diagnostics = _particle_mass_diagnostics(joined)
        native_mass = mass_diagnostics["native_full_mass"]
        recovar_mass = mass_diagnostics["recovar_full_mass"]
        row_sum_errors.extend((abs(native_mass - 1.0), abs(recovar_mass - 1.0)))
        particle_mass_rows.append(
            {
                "stack_index_one_based": stack,
                "native_particle_id_zero_based": joined[0]["particle_id"],
                **mass_diagnostics,
            }
        )
        particle_rows.append(
            {
                "stack_index_one_based": stack,
                "native_particle_id_zero_based": joined[0]["particle_id"],
                "candidate_classes_exact": sum(item["candidate_exact"] for item in joined),
                "empty_sparse_support_classes": sum(
                    item["empty_sparse_support"] for item in joined
                ),
                "native_posterior_mass": native_mass,
                "recovar_posterior_mass": recovar_mass,
                "winner_exact": native_winner == recovar_winner,
                "native_pmax": native_max,
                "recovar_pmax": recovar_max,
            }
        )

    posterior_report = posterior.report()
    raw_report = raw.report()
    combined_report = combined.report()
    native_pmax_values = np.asarray(native_pmax, dtype=np.float64)
    recovar_pmax_values = np.asarray(recovar_pmax, dtype=np.float64)
    pmax_delta = recovar_pmax_values - native_pmax_values
    retained_live_errors = np.asarray(
        [row["live_retained_relative_l2"] for row in particle_mass_rows],
        dtype=np.float64,
    )
    retained_shape_errors = np.asarray(
        [row["jointly_normalized_retained_shape_relative_l2"] for row in particle_mass_rows],
        dtype=np.float64,
    )
    retained_causal_errors = np.asarray(
        [row["recovar_normalized_only_vs_native_live_relative_l2"] for row in particle_mass_rows],
        dtype=np.float64,
    )
    full_mass_delta = np.asarray(
        [row["recovar_full_mass"] - row["native_full_mass"] for row in particle_mass_rows],
        dtype=np.float64,
    )
    retained_mass_delta = np.asarray(
        [row["recovar_retained_mass"] - row["native_retained_mass"] for row in particle_mass_rows],
        dtype=np.float64,
    )
    mass_diagnostic_summary = {
        "particle_count": len(particle_mass_rows),
        "maximum_full_mass_abs_delta": float(np.max(np.abs(full_mass_delta), initial=0.0)),
        "maximum_retained_mass_abs_delta": float(
            np.max(np.abs(retained_mass_delta), initial=0.0)
        ),
        "mean_live_retained_relative_l2": float(np.mean(retained_live_errors)),
        "maximum_live_retained_relative_l2": float(
            np.max(retained_live_errors, initial=0.0)
        ),
        "mean_jointly_normalized_retained_shape_relative_l2": float(
            np.mean(retained_shape_errors)
        ),
        "maximum_jointly_normalized_retained_shape_relative_l2": float(
            np.max(retained_shape_errors, initial=0.0)
        ),
        "mean_recovar_normalized_only_vs_native_live_relative_l2": float(
            np.mean(retained_causal_errors)
        ),
        "maximum_recovar_normalized_only_vs_native_live_relative_l2": float(
            np.max(retained_causal_errors, initial=0.0)
        ),
        "minimum_recovar_joint_retained_normalization_factor": min(
            row["recovar_joint_retained_normalization_factor"] for row in particle_mass_rows
        ),
        "maximum_recovar_joint_retained_normalization_factor": max(
            row["recovar_joint_retained_normalization_factor"] for row in particle_mass_rows
        ),
        "normalization_scope": (
            "one denominator per particle over all retained classes, rotations, and translations"
        ),
        "correlation_used": False,
    }
    causal_metrics = {
        "candidate_tuple_exact_fraction": exact_count / float(CASE.particle_count * CASE.K),
        "centered_raw_score_relative_l2": raw_report["relative_l2"],
        "centered_combined_score_relative_l2": combined_report["relative_l2"],
        "posterior_relative_l2": posterior_report["relative_l2"],
        "posterior_row_sum_abs_error": max(row_sum_errors),
        "support_jaccard": (
            1.0 if support_union == 0 else support_intersection / float(support_union)
        ),
        "winner_agreement": winner_matches / float(CASE.particle_count),
        "pmax_rmse": float(np.sqrt(np.mean(pmax_delta * pmax_delta))),
        "pmax_abs_error": float(np.max(np.abs(pmax_delta), initial=0.0)),
    }
    parity, permutation = _parity_metrics(run_root / "recovar/output/k_class_parity_arrays.npz", stacks)
    maps = _map_metrics(
        run_root,
        permutation,
        frozen_target_maps=[input_by_role[f"RELION target class {class_id}"] for class_id in range(1, CASE.K + 1)],
    )
    gate_metrics = {
        **causal_metrics,
        "assignment_accuracy": parity["assignment_accuracy"],
        "minimum_class_fraction": parity["minimum_class_fraction"],
        "cross_engine_map_fsc_auc": maps["minimum_cross_engine_fsc_auc"],
        "native_control_map_fsc_auc": maps["minimum_native_control_fsc_auc"],
        "native_capture_inertness_map_fsc_auc": maps["minimum_native_capture_inertness_fsc_auc"],
        "frozen_target_map_fsc_auc": maps["minimum_frozen_target_fsc_auc"],
        "native_assignment_inertness": inventory["native_assignment_inertness"],
        "frozen_target_assignment_accuracy": inventory["frozen_target_assignment_accuracy"],
    }
    evaluation = evaluate_gates(gate_metrics, manifest["thresholds"])
    return {
        "schema": SCHEMA,
        "status": "complete",
        "accepted": evaluation["accepted"],
        "scientific_scope": manifest["scientific_scope"],
        "scientific_limitations": manifest["scientific_limitations"],
        "metric_policy": (
            "exact immutable candidate keys; centered scale-sensitive L2 for score tables; "
            "direct posterior/support/winner errors; signed shellwise FSC/FSC-AUC for maps; no correlation"
        ),
        "correlation_used": False,
        "launch_manifest": str(manifest_path.resolve()),
        "inventory": inventory["counts"],
        "causal_boundary": {
            "metrics": causal_metrics,
            "empty_sparse_support_records": empty_sparse_support_count,
            "centered_raw_score": raw_report,
            "centered_combined_score": combined_report,
            "posterior": posterior_report,
            "support_intersection": support_intersection,
            "support_union": support_union,
            "particle_rows": particle_rows,
            "retained_mass_diagnostics": mass_diagnostic_summary,
            "retained_mass_particle_rows": particle_mass_rows,
        },
        "parity_output": parity,
        "maps": maps,
        "gate_metrics": gate_metrics,
        "thresholds": manifest["thresholds"],
        "gates": evaluation["gates"],
        "failures": evaluation["failures"],
        "scorecard_change_admissible": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.manifest = args.manifest.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _require(not args.output.exists(), f"refusing to overwrite {args.output}")
    report = build_report(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "accepted": report["accepted"],
                "failures": report["failures"],
                "gate_metrics": report["gate_metrics"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

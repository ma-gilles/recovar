#!/usr/bin/env python3
"""Locate the first unequal K=1 coarse-posterior boundary for fixed particles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_em_k1_coarse_pass1_boundary import (
    _map_relion_table,
    _translation_permutation,
)
from scripts.validate_relion_coarse_pass1_components import (
    RELION_INVALID_DIFF2,
    load_artifact,
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


def _center(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values - np.max(values)


def _stats(delta: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(delta, dtype=np.float64).reshape(-1)
    absolute = np.abs(delta)
    return {
        "count": int(delta.size),
        "exact_equal_count": int(np.count_nonzero(delta == 0.0)),
        "median_abs": float(np.median(absolute)),
        "p95_abs": float(np.percentile(absolute, 95)),
        "max_abs": float(np.max(absolute)),
    }


def _load_recovar(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        scores_pre = np.asarray(payload["scores_pre_prior_per_class"], dtype=np.float64)
        scores_with = np.asarray(payload["scores_with_prior_per_class"], dtype=np.float64)
        _require(scores_pre.ndim == 3 and scores_pre.shape[0] == 1, "expected K=1 scores")
        _require(scores_with.shape == scores_pre.shape, "RECOVAR score topology mismatch")
        n_rot, n_trans = scores_pre.shape[1:]
        weights = np.asarray(payload["weights_per_class"], dtype=np.float64).reshape(n_rot, n_trans)
        significant = np.asarray(payload["significant_mask"], dtype=bool).reshape(n_rot, n_trans)
        return {
            "path": path.resolve(),
            "sha256": _sha256(path),
            "original_index": int(payload["original_index"]),
            "iteration": int(payload["debug_iteration"]),
            "current_size": int(payload["current_size"]),
            "scores_pre": scores_pre[0],
            "scores_with": scores_with[0],
            "weights": weights,
            "significant": significant,
            "translations": np.asarray(payload["translations"], dtype=np.float64),
            "translation_log_prior": np.asarray(
                payload["translation_log_prior"], dtype=np.float64
            ),
            "hard_assignment": int(payload["hard_assignment"]),
        }


def _candidate_keys(mask: np.ndarray, n_trans: int) -> list[dict[str, int]]:
    rows, translations = np.nonzero(mask)
    return [
        {
            "rotation_recovar_psi_major": int(rotation),
            "translation_recovar": int(translation),
            "flat_recovar": int(rotation * n_trans + translation),
        }
        for rotation, translation in zip(rows, translations, strict=True)
    ]


def _fixed_count_support(values: np.ndarray, count: int) -> np.ndarray:
    """Return a deterministic top-``count`` mask for a support counterfactual."""

    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    _require(0 < int(count) <= flat.size, "invalid fixed support count")
    order = np.argsort(-flat, kind="stable")
    selected = np.zeros(flat.size, dtype=bool)
    selected[order[: int(count)]] = True
    return selected.reshape(values.shape)


def _native_direction_prior_counterfactual(
    *,
    native_raw: np.ndarray,
    native_weights: np.ndarray,
    native_significant: np.ndarray,
    recovar_scores_pre: np.ndarray,
    recovar_translation_log_prior: np.ndarray,
) -> dict[str, Any]:
    """Swap only the direction term while retaining RECOVAR translation priors.

    Native unnormalised posterior weights satisfy, up to one particle-wide
    additive constant,

        log(weight) + diff2 = direction_log_prior + translation_log_prior.

    The native direction term is therefore inferred independently for every
    active rotation after subtracting RECOVAR's translation prior.  Taking the
    median across translations suppresses the already-small translation-prior
    and score-rounding residual.  The counterfactual keeps RECOVAR's raw score
    and translation prior and replaces only this inferred direction vector.
    """

    translation_prior = np.asarray(recovar_translation_log_prior, dtype=np.float64)
    _require(
        translation_prior.shape == (recovar_scores_pre.shape[1],),
        "translation-prior topology mismatch",
    )
    positive = (
        (native_weights > 0.0)
        & (native_raw != RELION_INVALID_DIFF2)
        & np.isfinite(recovar_scores_pre)
    )
    log_native_weight = np.full(native_weights.shape, np.nan, dtype=np.float64)
    log_native_weight[positive] = np.log(native_weights[positive])
    inferred_combined_prior = log_native_weight + native_raw
    direction_prior = np.full(recovar_scores_pre.shape[0], np.nan, dtype=np.float64)
    active_rotation = np.any(positive, axis=1)
    for rotation in np.flatnonzero(active_rotation):
        active_translation = positive[rotation]
        direction_prior[rotation] = np.median(
            inferred_combined_prior[rotation, active_translation]
            - translation_prior[active_translation]
        )

    additive_closure = (
        inferred_combined_prior[positive]
        - np.broadcast_to(direction_prior[:, None], positive.shape)[positive]
        - np.broadcast_to(translation_prior[None, :], positive.shape)[positive]
    )
    additive_closure -= np.median(additive_closure)

    counterfactual_log_weight = (
        recovar_scores_pre
        + direction_prior[:, None]
        + translation_prior[None, :]
    )
    counterfactual_log_weight[~np.isfinite(counterfactual_log_weight)] = -np.inf
    counterfactual_significant = _fixed_count_support(
        counterfactual_log_weight,
        int(np.count_nonzero(native_significant)),
    )
    mismatch = counterfactual_significant != native_significant
    native_only = native_significant & ~counterfactual_significant
    counterfactual_only = counterfactual_significant & ~native_significant
    native_parent = np.any(native_significant, axis=1)
    counterfactual_parent = np.any(counterfactual_significant, axis=1)
    n_trans = int(native_significant.shape[1])
    return {
        "method": (
            "native direction log prior inferred per rotation from positive native "
            "weights; RECOVAR raw scores and translation prior retained; support "
            "count fixed to native significant count"
        ),
        "positive_native_weight_count": int(np.count_nonzero(positive)),
        "active_rotation_count": int(np.count_nonzero(active_rotation)),
        "native_prior_additive_closure_after_constant": _stats(additive_closure),
        "candidate_mismatch_count": int(np.count_nonzero(mismatch)),
        "parent_mismatch_count": int(
            np.count_nonzero(native_parent != counterfactual_parent)
        ),
        "native_only": _candidate_keys(native_only, n_trans),
        "counterfactual_only": _candidate_keys(counterfactual_only, n_trans),
    }


def _compare(native_path: Path, recovar_path: Path) -> dict[str, Any]:
    native = load_artifact(native_path)
    recovar = _load_recovar(recovar_path)
    _require(native.stack_index - 1 == recovar["original_index"], "particle identity mismatch")
    _require(native.header[5] == recovar["iteration"], "iteration mismatch")
    _require(native.header[27] == recovar["current_size"], "current-size mismatch")
    n_directions, n_psi, n_trans = native.header[10:13]
    permutation, translation_mapping = _translation_permutation(
        native.translations,
        recovar["translations"],
    )

    def mapped(values: np.ndarray) -> np.ndarray:
        return _map_relion_table(
            values,
            n_directions=n_directions,
            n_psi=n_psi,
            relion_to_recovar_translation=permutation,
        )

    native_raw = mapped(native.raw_diff2)
    native_weights = mapped(native.weights).astype(np.float64)
    native_significant = mapped(native.significant_mask).astype(bool)
    native_norm = mapped(native.reference_norms)
    native_cross = mapped(native.cross_terms)
    _require(native_raw.shape == recovar["scores_pre"].shape, "candidate topology mismatch")
    active = native_raw != RELION_INVALID_DIFF2
    recovar_active = np.isfinite(recovar["scores_with"])
    common = active & recovar_active
    _require(np.any(common), "no common candidate support")

    raw_delta = _center(recovar["scores_pre"][common]) - _center(-native_raw[common])
    positive = (native_weights > 0.0) & common
    _require(np.any(positive), "no positive native posterior values")
    native_log_weight = np.log(native_weights[positive])
    with_prior_delta = (
        _center(recovar["scores_with"][positive]) - _center(native_log_weight)
    )
    native_prior = native_log_weight - (-native_raw[positive])
    recovar_prior = recovar["scores_with"][positive] - recovar["scores_pre"][positive]
    prior_delta = _center(recovar_prior) - _center(native_prior)

    native_probability = native_weights / np.sum(native_weights, dtype=np.float64)
    recovar_probability = recovar["weights"] / np.sum(recovar["weights"], dtype=np.float64)
    support_delta = native_significant != recovar["significant"]
    relion_only = native_significant & ~recovar["significant"]
    recovar_only = recovar["significant"] & ~native_significant
    native_parent = np.any(native_significant, axis=1)
    recovar_parent = np.any(recovar["significant"], axis=1)
    component_closure = native_raw[active] - native_norm[active] - native_cross[active]
    component_closure -= np.median(component_closure)
    direction_prior_counterfactual = _native_direction_prior_counterfactual(
        native_raw=native_raw,
        native_weights=native_weights,
        native_significant=native_significant,
        recovar_scores_pre=recovar["scores_pre"],
        recovar_translation_log_prior=recovar["translation_log_prior"],
    )

    return {
        "stack_index_one_based": native.stack_index,
        "original_index_zero_based": recovar["original_index"],
        "relion_part_id": native.part_id,
        "physical_iteration": recovar["iteration"],
        "current_size": recovar["current_size"],
        "topology": {
            "n_directions": int(n_directions),
            "n_psi": int(n_psi),
            "n_translations": int(n_trans),
            "candidate_count": int(native_raw.size),
            "common_active_count": int(np.count_nonzero(common)),
            "active_mask_mismatch_count": int(np.count_nonzero(active != recovar_active)),
        },
        "raw_centered_score_delta_recovar_minus_relion": _stats(raw_delta),
        "prior_centered_delta_recovar_minus_relion": _stats(prior_delta),
        "with_prior_centered_delta_recovar_minus_relion": _stats(with_prior_delta),
        "posterior": {
            "total_variation": float(0.5 * np.sum(np.abs(recovar_probability - native_probability))),
            "max_abs": float(np.max(np.abs(recovar_probability - native_probability))),
            "relion_winner_flat_recovar": int(np.argmax(native_probability)),
            "recovar_winner_flat": int(np.argmax(recovar_probability)),
            "winner_exact": bool(np.argmax(native_probability) == np.argmax(recovar_probability)),
        },
        "significant_support": {
            "relion_count": int(np.count_nonzero(native_significant)),
            "recovar_count": int(np.count_nonzero(recovar["significant"])),
            "candidate_mismatch_count": int(np.count_nonzero(support_delta)),
            "parent_mismatch_count": int(np.count_nonzero(native_parent != recovar_parent)),
            "relion_only": _candidate_keys(relion_only, n_trans),
            "recovar_only": _candidate_keys(recovar_only, n_trans),
        },
        "native_direction_prior_counterfactual": direction_prior_counterfactual,
        "native_component_closure_after_constant": _stats(component_closure),
        "translation_mapping": translation_mapping,
        "artifacts": {
            "relion": str(native.path.resolve()),
            "relion_sha256": native.sha256,
            "recovar": str(recovar["path"]),
            "recovar_sha256": recovar["sha256"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--recovar-directory", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    native_by_stack = {
        load_artifact(path).stack_index: path
        for path in sorted(args.native_directory.glob("*.p1-v2.bin"))
    }
    recovar_by_stack = {}
    for path in sorted(args.recovar_directory.glob("significance_*.npz")):
        artifact = _load_recovar(path)
        recovar_by_stack[artifact["original_index"] + 1] = path
    _require(native_by_stack.keys() == recovar_by_stack.keys(), "capture identity sets differ")
    particles = [
        _compare(native_by_stack[stack], recovar_by_stack[stack])
        for stack in sorted(native_by_stack)
    ]
    report = {
        "schema": "recovar.em.k1_coarse_components_boundary.v1",
        "metric_policy": "exact identities plus centered score residuals and posterior total variation",
        "particles": particles,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

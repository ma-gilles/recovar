#!/usr/bin/env python
"""Compare one RELION ACC E-step dump with one RECOVAR pass-2 dump.

The comparison is intentionally candidate-based: it matches hypotheses by
global fine-rotation index and fine-translation index, then reports posterior
and score agreement on the common set. This makes the diagnostic useful even
when the two sides prune different tails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.parse_relion_dump_dir import parse_dump_dir


_ACC_TABLE_WEIGHT_SUFFIXES = ("diff2_weights", "sorted_weights")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _get_by_suffix(
    payload: dict[str, np.ndarray],
    suffix: str,
    *,
    prefer_prefix: str | None = None,
) -> np.ndarray | None:
    if suffix in payload:
        return payload[suffix]
    exact_layout_matches = [
        (key, value) for key, value in sorted(payload.items()) if _relion_layout_name(key) == suffix
    ]
    if prefer_prefix is not None:
        preferred = [
            (key, value)
            for key, value in exact_layout_matches
            if key == prefer_prefix or key.startswith(prefer_prefix + "_")
        ]
        if preferred:
            exact_layout_matches = preferred
    for _key, value in exact_layout_matches:
        if np.asarray(value).size:
            return value
    if exact_layout_matches:
        return exact_layout_matches[0][1]

    matches = [value for key, value in sorted(payload.items()) if key.endswith("_" + suffix)]
    for value in matches:
        if np.asarray(value).size:
            return value
    return matches[0] if matches else None


def _relion_layout_name(name: str) -> str:
    while True:
        stripped = name
        for prefix in ("pass", "over", "img", "part", "class"):
            import re

            stripped = re.sub(rf"^{prefix}\d+_", "", stripped, count=1)
        if stripped == name:
            return name
        name = stripped


def _finite_centered(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return arr
    out = arr.copy()
    out[finite] -= np.max(out[finite])
    return out


def _safe_stats(values: np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"count": int(arr.size), "finite_count": 0, "mean": None, "max_abs": None, "p95_abs": None}
    return {
        "count": int(arr.size),
        "finite_count": int(finite.size),
        "mean": float(np.mean(finite)),
        "max_abs": float(np.max(np.abs(finite))),
        "p95_abs": float(np.percentile(np.abs(finite), 95)),
    }


def _read_relion_flat_real(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    raw = path.read_bytes()
    if len(raw) < 4:
        return None
    count = int(np.frombuffer(raw[:4], dtype=np.int32)[0])
    data = np.frombuffer(raw[4:], dtype=np.float64, count=count).copy()
    return data


def _read_relion_fine_rotation_matrices(dump_dir: Path, *, min_rows: int | None = None) -> np.ndarray | None:
    candidates: list[tuple[int, np.ndarray]] = []
    for path in sorted(dump_dir.glob("*class*_fine_eulers.bin")):
        data = _read_relion_flat_real(path)
        if data is None or data.size == 0 or data.size % 9:
            continue
        matrices = data.reshape(-1, 3, 3)
        if min_rows is not None and matrices.shape[0] < int(min_rows):
            continue
        candidates.append((int(matrices.shape[0]), matrices))
    if not candidates:
        return None
    # K-class dumps include one fine_eulers table per class.  The candidate
    # compact rotation rows are local to the dumped class table, so choose the
    # smallest table that can contain the observed compact row ids.
    return min(candidates, key=lambda item: item[0])[1]


def _candidate_table_from_recovar(
    path: Path,
    *,
    reconstruction_only: bool,
    class_index_override: int | None = None,
    parent_rotation_divisor: int | None = None,
    parent_translation_divisor: int | None = None,
) -> dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    def zget(key: str, default: Any) -> Any:
        return z[key] if key in z.files else default

    if "pass2_scores_total" in z or ("posterior" in z and "local_rotation_indices" in z):
        probs = np.asarray(z["posterior"], dtype=np.float64)
        if probs.ndim == 3:
            probs = probs[0]
        if probs.ndim != 2:
            raise ValueError(f"RECOVAR local posterior dump must have shape (1, R, T) or (R, T), got {probs.shape}")
        n_rot, n_trans = probs.shape

        rotation_log_prior = np.asarray(zget("rotation_log_prior", np.zeros((1, n_rot))), dtype=np.float64)
        if rotation_log_prior.ndim == 2:
            rotation_log_prior = rotation_log_prior[0]
        if rotation_log_prior.size != n_rot:
            rotation_log_prior = np.zeros(n_rot, dtype=np.float64)
        translation_log_prior = np.asarray(zget("translation_log_prior", np.zeros((1, n_trans))), dtype=np.float64)
        if translation_log_prior.ndim == 2:
            translation_log_prior = translation_log_prior[0]
        if translation_log_prior.size != n_trans:
            translation_log_prior = np.zeros(n_trans, dtype=np.float64)

        if "pass2_scores_total" in z:
            scores_with = np.asarray(z["pass2_scores_total"], dtype=np.float64)
            if scores_with.ndim == 3:
                scores_with = scores_with[0]
        else:
            log_z = float(np.asarray(zget("log_Z", np.array([0.0])), dtype=np.float64).reshape(-1)[0])
            scores_with = np.where(probs > 0.0, np.log(probs) + log_z, -np.inf)
        if "pass2_scores_raw" in z:
            scores_pre = np.asarray(z["pass2_scores_raw"], dtype=np.float64)
            if scores_pre.ndim == 3:
                scores_pre = scores_pre[0]
        else:
            scores_pre = scores_with - rotation_log_prior[:, None] - translation_log_prior[None, :]
        if scores_with.shape != (n_rot, n_trans) or scores_pre.shape != (n_rot, n_trans):
            raise ValueError(
                "RECOVAR local score dump score arrays must match posterior shape "
                f"{(n_rot, n_trans)}; got scores_with={scores_with.shape} scores_pre={scores_pre.shape}"
            )

        if reconstruction_only and "reconstruction_sample_mask" in z:
            selected = np.asarray(z["reconstruction_sample_mask"], dtype=bool)
            if selected.ndim == 3:
                selected = selected[0]
            selected_field = "reconstruction_sample_mask"
        else:
            selected = np.isfinite(scores_with)
            selected_field = "finite_pass2_scores_total"
        flat_selected = selected.reshape(-1)

        local_rot_indices = np.arange(n_rot, dtype=np.int64)
        rot_indices = np.asarray(z["local_rotation_indices"], dtype=np.int64).reshape(-1)
        if rot_indices.size != n_rot:
            rot_indices = local_rot_indices
        trans_indices = np.arange(n_trans, dtype=np.int64)
        rot_grid, trans_grid = np.meshgrid(rot_indices, trans_indices, indexing="ij")
        local_rot_grid, _ = np.meshgrid(local_rot_indices, trans_indices, indexing="ij")

        keys_parent = None
        parent_mapping_details: dict[str, Any] = {}
        if "local_rotation_parent_indices" in z:
            parent_rot_indices = np.asarray(z["local_rotation_parent_indices"], dtype=np.int64).reshape(-1)
            if parent_rot_indices.size == n_rot:
                parent_trans_indices = np.asarray(
                    zget("translation_parent_indices", np.arange(n_trans, dtype=np.int64)),
                    dtype=np.int64,
                ).reshape(-1)
                if parent_trans_indices.size != n_trans:
                    parent_trans_indices = np.arange(n_trans, dtype=np.int64)
                parent_rot_grid, parent_trans_grid = np.meshgrid(
                    parent_rot_indices,
                    parent_trans_indices,
                    indexing="ij",
                )
                keys_parent = np.stack(
                    [parent_rot_grid.reshape(-1)[flat_selected], parent_trans_grid.reshape(-1)[flat_selected]],
                    axis=1,
                )
                parent_mapping_details["recovar_parent_rotation_mapping"] = "local_rotation_parent_indices"
                parent_mapping_details["recovar_parent_translation_mapping"] = "translation_parent_indices"

        rotations = np.asarray(
            zget(
                "local_rotation_matrices",
                zget("rotations", np.zeros((n_rot, 3, 3), dtype=np.float32)),
            ),
            dtype=np.float64,
        )
        if rotations.shape[0] != n_rot:
            rotations = np.zeros((n_rot, 3, 3), dtype=np.float64)

        original_index = np.asarray(
            zget(
                "selected_global_image_indices",
                zget("original_index", np.array([-1], dtype=np.int64)),
            ),
        ).reshape(-1)
        local_index = np.asarray(
            zget(
                "selected_local_image_indices",
                zget("local_index", np.array([-1], dtype=np.int64)),
            ),
        ).reshape(-1)
        current_size = np.asarray(zget("current_size", np.array([-1], dtype=np.int64))).reshape(-1)
        class_index = np.asarray(zget("class_index", np.array([-1], dtype=np.int64))).reshape(-1)
        return {
            "source": str(path),
            "original_index": int(original_index[0]),
            "local_index": int(local_index[0]),
            "class_index": int(class_index[0]) if int(class_index[0]) >= 0 else None,
            "current_size": int(current_size[0]),
            "selected_field": selected_field,
            "reconstruction_n_significant": int(np.asarray(z["n_significant_samples"]).reshape(-1)[0])
            if "n_significant_samples" in z
            else None,
            "keys_global": np.stack(
                [rot_grid.reshape(-1)[flat_selected], trans_grid.reshape(-1)[flat_selected]],
                axis=1,
            ),
            "keys_local": np.stack(
                [local_rot_grid.reshape(-1)[flat_selected], trans_grid.reshape(-1)[flat_selected]],
                axis=1,
            ),
            "keys_parent": keys_parent,
            "parent_mapping_details": parent_mapping_details,
            "rotations": rotations,
            "prob": probs.reshape(-1)[flat_selected],
            "score_pre_prior": scores_pre.reshape(-1)[flat_selected],
            "score_with_prior": scores_with.reshape(-1)[flat_selected],
            "rotation_log_prior": np.repeat(rotation_log_prior, n_trans)[flat_selected],
            "translation_log_prior": np.tile(translation_log_prior, n_rot)[flat_selected],
            "combined_log_prior": (
                np.repeat(rotation_log_prior, n_trans) + np.tile(translation_log_prior, n_rot)
            )[flat_selected],
        }

    if "scores_pre_prior_per_class" in z:
        scores_by_class = np.asarray(z["scores_pre_prior_per_class"], dtype=np.float64)
        if scores_by_class.ndim != 3:
            raise ValueError(
                "RECOVAR significance dump scores_pre_prior_per_class must have shape "
                f"(K, R, T), got {scores_by_class.shape}"
            )
        if class_index_override is not None:
            class_index = int(class_index_override)
        elif "class_index" in z:
            class_index = int(np.asarray(z["class_index"]).reshape(-1)[0])
        elif "class_assignment" in z:
            class_index = int(np.asarray(z["class_assignment"]).reshape(-1)[0])
        else:
            class_index = 0
        if class_index < 0 or class_index >= scores_by_class.shape[0]:
            raise ValueError(
                f"RECOVAR significance class_index={class_index} is outside "
                f"available class range [0, {scores_by_class.shape[0]})",
            )
        scores_pre = scores_by_class[class_index]
        n_rot, n_trans = scores_pre.shape
        scores_with_by_class = (
            np.asarray(z["scores_with_prior_per_class"], dtype=np.float64)
            if "scores_with_prior_per_class" in z
            else scores_by_class
        )
        scores_with = scores_with_by_class[class_index]
        if "weights_per_class" in z:
            probs = np.asarray(z["weights_per_class"], dtype=np.float64)[class_index].reshape(n_rot, n_trans)
        else:
            weights_full = np.asarray(z["weights_full"], dtype=np.float64)
            if weights_full.size == scores_by_class.size:
                probs = weights_full.reshape(scores_by_class.shape)[class_index]
            else:
                probs = weights_full.reshape(-1)[: n_rot * n_trans].reshape(n_rot, n_trans)
        rot_indices = np.arange(n_rot, dtype=np.int64)
        trans_indices = np.arange(n_trans, dtype=np.int64)
        selected = np.isfinite(scores_with)
        rot_grid, trans_grid = np.meshgrid(rot_indices, trans_indices, indexing="ij")
        flat_selected = selected.reshape(-1)
        rotations = np.asarray(z["rotations"], dtype=np.float64)
        rotation_log_prior = np.asarray(z["rotation_log_prior"], dtype=np.float64)
        if rotation_log_prior.ndim == 2:
            rotation_log_prior = rotation_log_prior[class_index]
        if rotation_log_prior.size != n_rot:
            rotation_log_prior = np.zeros(n_rot, dtype=np.float64)
        translation_log_prior = np.asarray(z["translation_log_prior"], dtype=np.float64)
        if translation_log_prior.size != n_trans:
            translation_log_prior = np.zeros(n_trans, dtype=np.float64)
        return {
            "source": str(path),
            "original_index": int(np.asarray(z["original_index"]).item()),
            "local_index": int(np.asarray(z["local_index"]).item()),
            "class_index": int(class_index),
            "current_size": int(np.asarray(z["current_size"]).item()),
            "selected_field": "finite_scores_pre_prior_per_class",
            "reconstruction_n_significant": None,
            "keys_global": np.stack(
                [rot_grid.reshape(-1)[flat_selected], trans_grid.reshape(-1)[flat_selected]],
                axis=1,
            ),
            "keys_local": np.stack(
                [rot_grid.reshape(-1)[flat_selected], trans_grid.reshape(-1)[flat_selected]],
                axis=1,
            ),
            "keys_parent": None,
            "rotations": rotations,
            "prob": probs.reshape(-1)[flat_selected],
            "score_pre_prior": scores_pre.reshape(-1)[flat_selected],
            "score_with_prior": scores_with.reshape(-1)[flat_selected],
            "rotation_log_prior": np.repeat(rotation_log_prior, n_trans)[flat_selected],
            "translation_log_prior": np.tile(translation_log_prior, n_rot)[flat_selected],
            "combined_log_prior": (
                np.repeat(rotation_log_prior, n_trans) + np.tile(translation_log_prior, n_rot)
            )[flat_selected],
        }

    probs = np.asarray(z["probs"], dtype=np.float64)
    selected_probs = probs
    scores_pre = np.asarray(z["scores_pre_prior"], dtype=np.float64)
    scores_with = np.asarray(z["scores_with_prior"], dtype=np.float64)
    rot_indices = np.asarray(z["oversampled_rot_indices"], dtype=np.int64)
    mask = np.asarray(z["candidate_mask"], dtype=bool)

    selected_field = "candidate_mask"
    if reconstruction_only and "reconstruction_mask" in z:
        selected = np.asarray(z["reconstruction_mask"], dtype=bool)
        if "reconstruction_probs" in z:
            selected_probs = np.asarray(z["reconstruction_probs"], dtype=np.float64)
        selected_field = "reconstruction_mask"
    elif reconstruction_only:
        selected = mask
    else:
        selected = np.isfinite(scores_with)
        selected_field = "finite_scores"

    local_rot_indices = np.arange(probs.shape[0], dtype=np.int64)
    rot_grid, trans_grid = np.meshgrid(rot_indices, np.arange(probs.shape[1], dtype=np.int64), indexing="ij")
    local_rot_grid, _ = np.meshgrid(local_rot_indices, np.arange(probs.shape[1], dtype=np.int64), indexing="ij")
    flat_selected = selected.reshape(-1)
    keys_global = np.stack([rot_grid.reshape(-1)[flat_selected], trans_grid.reshape(-1)[flat_selected]], axis=1)
    keys_local = np.stack([local_rot_grid.reshape(-1)[flat_selected], trans_grid.reshape(-1)[flat_selected]], axis=1)
    keys_parent = None
    parent_mapping_details: dict[str, Any] = {}
    if "parent_map" in z:
        parent_map = np.asarray(z["parent_map"], dtype=np.int64).reshape(-1)
        if parent_rotation_divisor is not None:
            parent_rotation_divisor = int(parent_rotation_divisor)
            if parent_rotation_divisor <= 0:
                raise ValueError(
                    "parent_rotation_divisor must be positive when provided, "
                    f"got {parent_rotation_divisor}",
                )
            parent_rot_indices = rot_indices // parent_rotation_divisor
            parent_mapping_details["recovar_parent_rotation_mapping"] = "oversampled_rot_indices_divisor"
            parent_mapping_details["recovar_parent_rotation_divisor"] = parent_rotation_divisor
        else:
            parent_rot_indices = parent_map
            parent_mapping_details["recovar_parent_rotation_mapping"] = "parent_map"
        if parent_rot_indices.size == probs.shape[0]:
            parent_trans_indices = np.arange(probs.shape[1], dtype=np.int64)
            if parent_translation_divisor is not None:
                parent_translation_divisor = int(parent_translation_divisor)
                if parent_translation_divisor <= 0:
                    raise ValueError(
                        "parent_translation_divisor must be positive when provided, "
                        f"got {parent_translation_divisor}",
                    )
                parent_trans_indices = parent_trans_indices // parent_translation_divisor
                parent_mapping_details["recovar_parent_translation_mapping"] = "fine_trans_indices_divisor"
                parent_mapping_details["recovar_parent_translation_divisor"] = parent_translation_divisor
            else:
                parent_mapping_details["recovar_parent_translation_mapping"] = "fine_trans_indices"
            parent_rot_grid, parent_trans_grid = np.meshgrid(parent_rot_indices, parent_trans_indices, indexing="ij")
            keys_parent = np.stack(
                [parent_rot_grid.reshape(-1)[flat_selected], parent_trans_grid.reshape(-1)[flat_selected]],
                axis=1,
            )
    return {
        "source": str(path),
        "original_index": int(np.asarray(z["original_index"]).item()),
        "local_index": int(np.asarray(z["local_index"]).item()),
        "class_index": int(np.asarray(z["class_index"]).item()) if "class_index" in z else None,
        "current_size": int(np.asarray(z["current_size"]).item()),
        "selected_field": selected_field,
        "reconstruction_n_significant": int(np.asarray(z["reconstruction_n_significant"]).item())
        if "reconstruction_n_significant" in z
        else None,
        "keys_global": keys_global,
        "keys_local": keys_local,
        "keys_parent": keys_parent,
        "parent_mapping_details": parent_mapping_details,
        "rotations": np.asarray(z["rotations"], dtype=np.float64),
        "prob": selected_probs.reshape(-1)[flat_selected],
        "score_pre_prior": scores_pre.reshape(-1)[flat_selected],
        "score_with_prior": scores_with.reshape(-1)[flat_selected],
        "rotation_log_prior": np.repeat(np.asarray(z["rotation_log_prior"], dtype=np.float64), probs.shape[1])[
            flat_selected
        ],
        "translation_log_prior": np.tile(np.asarray(z["translation_log_prior"], dtype=np.float64), probs.shape[0])[
            flat_selected
        ],
        "combined_log_prior": (
            np.repeat(np.asarray(z["rotation_log_prior"], dtype=np.float64), probs.shape[1])
            + np.tile(np.asarray(z["translation_log_prior"], dtype=np.float64), probs.shape[0])
        )[flat_selected],
    }


def _read_prefixed_relion_acc_table(
    payload: dict[str, np.ndarray],
    prefix: str,
    *,
    reconstruction_only: bool = False,
    class_index: int | None = None,
) -> dict[str, Any] | None:
    weights_key = f"{prefix}_diff2_weights"
    weights = payload.get(weights_key)
    weight_mode = "diff2"
    if weights is None:
        weights_key = f"{prefix}_sorted_weights"
        weights = payload.get(weights_key)
        weight_mode = "sorted_weights"
    if weights is None:
        return None
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    full_storewavg_selected_mask: np.ndarray | None = None
    storewavg_selected_field = f"acc_storewavg_positive_weights:{prefix}"
    if weight_mode == "sorted_weights":
        positive = np.isfinite(weights) & (weights > 0.0)
        full_storewavg_selected_mask = positive
        if reconstruction_only:
            significant_weight_value = payload.get(f"{prefix}_significant_weight")
            if significant_weight_value is not None:
                significant_weight = float(np.asarray(significant_weight_value, dtype=np.float64).reshape(-1)[0])
                significant = np.zeros_like(positive, dtype=bool)
                if np.isfinite(significant_weight) and significant_weight > 0.0 and np.any(positive):
                    positive_rows = np.flatnonzero(positive)
                    order = positive_rows[np.argsort(weights[positive_rows], kind="stable")[::-1]]
                    cumulative = np.cumsum(weights[order], dtype=np.float64)
                    tolerance = max(abs(significant_weight), 1.0) * 1.0e-6
                    keep = cumulative <= significant_weight + tolerance
                    if not np.any(keep) and abs(float(weights[order[0]]) - significant_weight) <= tolerance:
                        keep[0] = True
                    significant[order[keep]] = True
                full_storewavg_selected_mask = significant
                storewavg_selected_field = f"acc_storewavg_significant_weights:{prefix}"

    orientation_num = payload.get(f"{prefix}_orientation_num")
    translation_num = payload.get(f"{prefix}_translation_num")
    if orientation_num is None or translation_num is None:
        raise ValueError(
            f"RELION ACC table prefix {prefix!r} has {weights_key} but is missing "
            f"{prefix}_orientation_num/{prefix}_translation_num",
        )
    n_rot = int(np.asarray(orientation_num).item())
    n_trans = int(np.asarray(translation_num).item())
    sliced_class_index = None
    nr_classes_value = payload.get(f"{prefix}_nr_classes")
    nr_classes = int(np.asarray(nr_classes_value).item()) if nr_classes_value is not None else None
    iclass_min_value = payload.get(f"{prefix}_iclass_min")
    iclass_min = int(np.asarray(iclass_min_value).item()) if iclass_min_value is not None else 0
    class_entries = _get_by_suffix(payload, "fine_class_entries")
    class_idx = _get_by_suffix(payload, "fine_class_idx")
    fine_iorientclasses = _get_by_suffix(payload, "fine_iorientclasses")
    fine_iover_rots = _get_by_suffix(payload, "fine_iover_rots")
    pdf_orientation = _get_by_suffix(payload, "pdf_orientation", prefer_prefix="pass1")
    selected_mask_override: np.ndarray | None = None
    if (
        n_rot <= 0
        and weight_mode == "sorted_weights"
        and class_entries is not None
        and class_index is not None
        and n_trans > 0
    ):
        entries = np.asarray(class_entries, dtype=np.int64).reshape(-1)
        class_stop = entries.size if nr_classes is None else min(entries.size, iclass_min + max(int(nr_classes), 0))
        if iclass_min <= int(class_index) < class_stop:
            start_row = int(np.sum(entries[iclass_min : int(class_index)]))
            class_n_rot = int(entries[int(class_index)])
            total_rows = int(np.sum(entries[iclass_min:class_stop]))
            if weights.size == total_rows * n_trans:
                start = start_row * n_trans
                stop = start + class_n_rot * n_trans
                weights = weights[start:stop]
                if full_storewavg_selected_mask is not None:
                    selected_mask_override = full_storewavg_selected_mask[start:stop]
                n_rot = class_n_rot
                sliced_class_index = int(class_index)
    if n_rot <= 0 and n_trans > 0 and weights.size % n_trans == 0:
        n_rot = int(weights.size // n_trans)
    expected = n_rot * n_trans
    if weights.size != expected:
        if nr_classes is None and expected > 0 and weights.size % expected == 0:
            nr_classes = int(weights.size // expected)
        sliced_variable_class_entries = False
        if (
            weight_mode == "sorted_weights"
            and class_entries is not None
            and class_index is not None
            and n_trans > 0
        ):
            entries = np.asarray(class_entries, dtype=np.int64).reshape(-1)
            class_stop = entries.size if nr_classes is None else min(entries.size, iclass_min + max(int(nr_classes), 0))
            if iclass_min <= int(class_index) < class_stop:
                start_row = int(np.sum(entries[iclass_min : int(class_index)]))
                class_n_rot = int(entries[int(class_index)])
                total_rows = int(np.sum(entries[iclass_min:class_stop]))
                if weights.size == total_rows * n_trans:
                    start = start_row * n_trans
                    stop = start + class_n_rot * n_trans
                    weights = weights[start:stop]
                    if full_storewavg_selected_mask is not None:
                        selected_mask_override = full_storewavg_selected_mask[start:stop]
                    n_rot = class_n_rot
                    expected = n_rot * n_trans
                    sliced_class_index = int(class_index)
                    sliced_variable_class_entries = True
        if sliced_variable_class_entries:
            pass
        elif (
            weight_mode == "sorted_weights"
            and nr_classes is not None
            and nr_classes > 1
            and weights.size == expected * nr_classes
            and class_index is not None
        ):
            class_offset = int(class_index) - iclass_min
            if class_offset < 0 or class_offset >= nr_classes:
                raise ValueError(
                    f"RECOVAR class_index={class_index} is outside RELION "
                    f"{prefix!r} class range [{iclass_min}, {iclass_min + nr_classes})",
                )
            start = class_offset * expected
            weights = weights[start : start + expected]
            if full_storewavg_selected_mask is not None:
                selected_mask_override = full_storewavg_selected_mask[start : start + expected]
            sliced_class_index = int(class_index)
        else:
            raise ValueError(
                f"RELION ACC table {weights_key} has {weights.size} values, expected "
                f"{n_rot} * {n_trans} = {expected}",
            )
    local_rotation_rows = np.arange(n_rot, dtype=np.int64)
    global_rotation_rows = local_rotation_rows.copy()
    rotation_key_mode = "local_row"
    if (
        weight_mode == "sorted_weights"
        and sliced_class_index is not None
        and class_idx is not None
        and fine_iorientclasses is not None
        and fine_iover_rots is not None
        and pdf_orientation is not None
    ):
        starts = np.asarray(class_idx, dtype=np.int64).reshape(-1)
        iorient = np.asarray(fine_iorientclasses, dtype=np.int64).reshape(-1)
        iover = np.asarray(fine_iover_rots, dtype=np.int64).reshape(-1)
        n_base_rot = int(np.asarray(pdf_orientation).reshape(-1).size)
        if (
            0 <= int(sliced_class_index) < starts.size
            and n_base_rot > 0
            and iorient.size == iover.size
            and iorient.size >= int(starts[int(sliced_class_index)]) + n_rot
        ):
            start_row = int(starts[int(sliced_class_index)])
            stop_row = start_row + n_rot
            iover_slice = iover[start_row:stop_row]
            nr_oversampled_rot = int(np.max(iover) + 1) if iover.size else 1
            if nr_oversampled_rot > 0 and iover_slice.size == n_rot:
                global_rotation_rows = (
                    (iorient[start_row:stop_row] % n_base_rot) * nr_oversampled_rot
                    + iover_slice
                ).astype(np.int64, copy=False)
                rotation_key_mode = "fine_iorientclasses_mod_pdf_orientation_times_iover_rots"

    rot_idx = np.repeat(global_rotation_rows, n_trans)
    local_rot_idx = np.repeat(local_rotation_rows, n_trans)
    trans_idx = np.tile(np.arange(n_trans, dtype=np.int64), n_rot)
    selected_mask = np.ones(expected, dtype=bool)
    if weight_mode == "diff2":
        prob = np.zeros(expected, dtype=np.float64)
        finite = np.isfinite(weights)
        if np.any(finite):
            finite_indices = np.flatnonzero(finite)
            prob[finite_indices[int(np.argmin(weights[finite]))]] = 1.0
        score_pre = weights
        selected_field = f"acc_full_grid:{prefix}"
    else:
        # Older RELION debug builds dumped StoreWeightedSums' local table as
        # positive unnormalised weights, with impossible entries set to
        # -FLT_MAX.  Treat strictly positive finite values as the active
        # posterior support.  For reconstruction-only diagnostics, RELION's
        # significant_weight is a global StoreWeightedSums threshold; apply it
        # before class slicing so low-probability tails in non-winning classes
        # are not mistaken for M-step contributors.
        if selected_mask_override is not None:
            selected_mask = np.asarray(selected_mask_override, dtype=bool).reshape(-1)
        elif full_storewavg_selected_mask is not None and full_storewavg_selected_mask.size == expected:
            selected_mask = np.asarray(full_storewavg_selected_mask, dtype=bool).reshape(-1)
        else:
            selected_mask = np.isfinite(weights) & (weights > 0.0)
        prob = np.zeros(expected, dtype=np.float64)
        norm = float(np.sum(weights[selected_mask])) if np.any(selected_mask) else 0.0
        if norm > 0.0:
            prob[selected_mask] = weights[selected_mask] / norm
        score_pre = np.full(expected, np.inf, dtype=np.float64)
        score_pre[selected_mask] = -np.log(weights[selected_mask])
        selected_field = storewavg_selected_field
        if sliced_class_index is not None:
            selected_field += f":class{sliced_class_index}"
    rot_matrices = payload.get(f"{prefix}_eulers_matrices")
    if rot_matrices is not None:
        rot_matrices = np.asarray(rot_matrices, dtype=np.float64).reshape(-1, 3, 3)
    return {
        "rot_idx": rot_idx,
        "local_rot_idx": local_rot_idx,
        "trans_idx": trans_idx,
        "prob": prob,
        "score_pre": score_pre,
        "rot_matrices": rot_matrices,
        "rotation_count": n_rot,
        "rotation_key_mode": rotation_key_mode,
        "selected_field": selected_field,
        "selected_mask": selected_mask,
        "weight_mode": weight_mode,
    }


def _discover_relion_acc_table_prefixes(payload: dict[str, np.ndarray]) -> list[str]:
    """Return full RELION ACC table prefixes present in a parsed dump.

    A single dump directory may contain several part-specific broad score
    tables. Generic ``pass*_img*`` operands are overwritten by whichever
    particle dumped last, so diagnostics must not assume they identify the
    same particle as a manually chosen ``img*_part*`` score table.
    """

    prefixes: list[str] = []
    for key in sorted(payload):
        for suffix in _ACC_TABLE_WEIGHT_SUFFIXES:
            marker = "_" + suffix
            if not key.endswith(marker):
                continue
            prefix = key[: -len(marker)]
            if f"{prefix}_orientation_num" in payload and f"{prefix}_translation_num" in payload:
                prefixes.append(prefix)
            break
    return sorted(set(prefixes))


def _candidate_table_from_relion(
    path: Path,
    *,
    reconstruction_only: bool,
    acc_table_prefix: str | None = None,
    class_index: int | None = None,
    parent_rotation_divisor: int | None = None,
    parent_translation_divisor: int | None = None,
) -> dict[str, Any]:
    payload = parse_dump_dir(path)

    prefixed_table = None
    prefixed_selected_mask = None
    if acc_table_prefix is not None:
        prefixed_table = _read_prefixed_relion_acc_table(
            payload,
            acc_table_prefix,
            reconstruction_only=reconstruction_only,
            class_index=class_index,
        )
        if prefixed_table is None:
            raise ValueError(f"RELION dump {path} has no ACC table for prefix {acc_table_prefix!r}")

    if prefixed_table is None:
        # RELION's ACC names are easy to misread: rot_id is the global
        # orientation id, while rot_idx is only the index inside the compact
        # significant-orientation list. Use rot_id for RECOVAR key matching.
        rot_id = _get_by_suffix(payload, "acc_rot_id")
        compact_rot_idx = _get_by_suffix(payload, "acc_rot_idx")
        rot_idx = rot_id if rot_id is not None else compact_rot_idx
        trans_idx = _get_by_suffix(payload, "acc_trans_idx")
        coarse_trans_idx = _get_by_suffix(payload, "candidate_coarse_trans_idx")
        prob = _get_by_suffix(payload, "candidate_weight_normalized")
        if prob is None:
            prob = _get_by_suffix(payload, "exp_Mweight_posterior")
        score_pre = _get_by_suffix(payload, "exp_Mweight_raw_preprior")
        rot_prior = _get_by_suffix(payload, "candidate_orientation_log_prior")
        trans_prior = _get_by_suffix(payload, "candidate_offset_log_prior")
        combined_prior = _get_by_suffix(payload, "candidate_combined_log_prior")
        candidate_class_idx = _get_by_suffix(payload, "candidate_class_idx")
        reconstruction_mask = _get_by_suffix(payload, "candidate_in_reconstruction_set")
        firstiter_raw_preonehot = None
        if rot_idx is None:
            rot_idx = _get_by_suffix(payload, "firstiter_cc_raw_rot_idx", prefer_prefix="pass1")
            rot_id = rot_idx
        if compact_rot_idx is None:
            compact_rot_idx = rot_idx
        if trans_idx is None:
            trans_idx = _get_by_suffix(payload, "firstiter_cc_raw_trans_idx", prefer_prefix="pass1")
        if coarse_trans_idx is None:
            coarse_trans_idx = trans_idx
        if score_pre is None:
            firstiter_raw_preonehot = _get_by_suffix(
                payload,
                "firstiter_cc_exp_Mweight_raw_preonehot",
                prefer_prefix="pass1",
            )
            score_pre = firstiter_raw_preonehot
        elif _get_by_suffix(payload, "firstiter_cc_exp_Mweight_raw_preonehot", prefer_prefix="pass1") is not None:
            firstiter_raw_preonehot = _get_by_suffix(
                payload,
                "firstiter_cc_exp_Mweight_raw_preonehot",
                prefer_prefix="pass1",
            )
        rot_matrices = None
        rotation_count = None
        selected_field = "all_candidates"
    else:
        rot_idx = prefixed_table["rot_idx"]
        trans_idx = prefixed_table["trans_idx"]
        prob = prefixed_table["prob"]
        score_pre = prefixed_table["score_pre"]
        rot_prior = None
        trans_prior = None
        combined_prior = None
        candidate_class_idx = None
        reconstruction_mask = None
        firstiter_raw_preonehot = None
        rot_matrices = prefixed_table["rot_matrices"]
        rotation_count = prefixed_table["rotation_count"]
        selected_field = prefixed_table["selected_field"]
        prefixed_selected_mask = prefixed_table.get("selected_mask")
        compact_rot_idx = prefixed_table.get("local_rot_idx", rot_idx)
        coarse_trans_idx = trans_idx

    missing = [
        name
        for name, value in {
            "acc_rot_id/acc_rot_idx": rot_idx,
            "acc_trans_idx": trans_idx,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(f"RELION dump is missing required arrays: {', '.join(missing)}")

    rot_idx = np.asarray(rot_idx, dtype=np.int64).reshape(-1)
    trans_idx = np.asarray(trans_idx, dtype=np.int64).reshape(-1)
    compact_rot_idx = np.asarray(compact_rot_idx, dtype=np.int64).reshape(-1) if compact_rot_idx is not None else rot_idx
    coarse_trans_idx = np.asarray(coarse_trans_idx, dtype=np.int64).reshape(-1) if coarse_trans_idx is not None else trans_idx
    if prob is not None:
        prob = np.asarray(prob, dtype=np.float64).reshape(-1)
    elif firstiter_raw_preonehot is not None:
        score_for_wta = -np.asarray(firstiter_raw_preonehot, dtype=np.float64).reshape(-1)
        prob = np.zeros_like(score_for_wta, dtype=np.float64)
        finite = np.isfinite(score_for_wta)
        if np.any(finite):
            finite_indices = np.flatnonzero(finite)
            prob[finite_indices[int(np.argmax(score_for_wta[finite]))]] = 1.0
    else:
        raise ValueError(
            "RELION dump is missing required arrays: "
            "candidate_weight_normalized/exp_Mweight_posterior or firstiter_cc_exp_Mweight_raw_preonehot"
        )
    n = min(rot_idx.size, trans_idx.size, prob.size)
    if n == 0:
        if prefixed_table is not None:
            empty_i = np.empty((0, 2), dtype=np.int64)
            empty_f = np.empty((0,), dtype=np.float64)
            return {
                "source": str(path),
                "keys": empty_i,
                "keys_global": empty_i,
                "keys_local": empty_i,
                "keys_parent": None,
                "parent_mapping_details": {},
                "rot_matrices": rot_matrices,
                "rotation_count": rotation_count,
                "selected_field": selected_field,
                "prob": empty_f,
                "score_pre_prior": empty_f,
                "score_with_prior": empty_f,
                "rotation_log_prior": empty_f,
                "translation_log_prior": empty_f,
                "combined_log_prior": empty_f,
                "available_keys": sorted(payload.keys()),
            }
        raise ValueError(f"RELION dump {path} has no candidates")
    if rot_matrices is None:
        compact_for_matrix = compact_rot_idx[:n]
        min_rows = int(np.max(compact_for_matrix)) + 1 if compact_for_matrix.size else None
        rot_matrices = _read_relion_fine_rotation_matrices(path, min_rows=min_rows)

    def trim_or_nan(arr: np.ndarray | None) -> np.ndarray:
        if arr is None:
            return np.full(n, np.nan, dtype=np.float64)
        out = np.asarray(arr, dtype=np.float64).reshape(-1)
        if out.size < n:
            padded = np.full(n, np.nan, dtype=np.float64)
            padded[: out.size] = out
            return padded
        return out[:n]

    # RELION's ACC dump name is historical: exp_Mweight_raw_preprior is the
    # raw diff2/cost term, so lower is better. Convert it into RECOVAR's
    # log-score convention before adding log-priors or centering differences.
    score_pre_trimmed = -trim_or_nan(score_pre)
    rot_prior_trimmed = trim_or_nan(rot_prior)
    trans_prior_trimmed = trim_or_nan(trans_prior)
    combined_prior_trimmed = trim_or_nan(combined_prior)
    if not np.any(np.isfinite(combined_prior_trimmed)):
        combined_from_parts = rot_prior_trimmed + trans_prior_trimmed
        if np.any(np.isfinite(combined_from_parts)):
            combined_prior_trimmed = combined_from_parts
    if np.any(np.isfinite(combined_prior_trimmed)):
        score_with = score_pre_trimmed + combined_prior_trimmed
    else:
        score_with = score_pre_trimmed.copy()
    if not np.any(np.isfinite(score_with)):
        score_with = trim_or_nan(_get_by_suffix(payload, "coarse_log_weight_preexp"))
    if reconstruction_only and reconstruction_mask is not None:
        selected = np.asarray(reconstruction_mask).reshape(-1)[:n].astype(bool)
        selected_field = "candidate_in_reconstruction_set"
    elif prefixed_table is not None and prefixed_selected_mask is not None:
        selected = np.asarray(prefixed_selected_mask, dtype=bool).reshape(-1)[:n]
    else:
        selected = np.ones(n, dtype=bool)
    if prefixed_table is None and class_index is not None and candidate_class_idx is not None:
        class_selected = np.asarray(candidate_class_idx, dtype=np.int64).reshape(-1)[:n] == int(class_index)
        selected = selected & class_selected
        selected_field += f":class{int(class_index)}"
    keys_global = np.stack([rot_idx[:n], trans_idx[:n]], axis=1)
    keys_local = np.stack([compact_rot_idx[:n], trans_idx[:n]], axis=1)
    keys_parent = None
    parent_mapping_details: dict[str, Any] = {}
    if parent_rotation_divisor is not None:
        parent_rotation_divisor = int(parent_rotation_divisor)
        if parent_rotation_divisor <= 0:
            raise ValueError(
                "relion parent_rotation_divisor must be positive when provided, "
                f"got {parent_rotation_divisor}",
            )
        parent_rot = compact_rot_idx[:n] // parent_rotation_divisor
        parent_trans = coarse_trans_idx[:n]
        parent_mapping_details["relion_parent_rotation_mapping"] = "acc_rot_idx_divisor"
        parent_mapping_details["relion_parent_rotation_divisor"] = parent_rotation_divisor
        if parent_translation_divisor is not None:
            parent_translation_divisor = int(parent_translation_divisor)
            if parent_translation_divisor <= 0:
                raise ValueError(
                    "relion parent_translation_divisor must be positive when provided, "
                    f"got {parent_translation_divisor}",
                )
            parent_trans = trans_idx[:n] // parent_translation_divisor
            parent_mapping_details["relion_parent_translation_mapping"] = "acc_trans_idx_divisor"
            parent_mapping_details["relion_parent_translation_divisor"] = parent_translation_divisor
        elif coarse_trans_idx is not None:
            parent_mapping_details["relion_parent_translation_mapping"] = "candidate_coarse_trans_idx"
        keys_parent = np.stack([parent_rot, parent_trans], axis=1)

    return {
        "source": str(path),
        "keys": keys_global[selected],
        "keys_global": keys_global[selected],
        "keys_local": keys_local[selected],
        "keys_parent": keys_parent[selected] if keys_parent is not None else None,
        "parent_mapping_details": parent_mapping_details,
        "rot_matrices": rot_matrices,
        "rotation_count": rotation_count,
        "relion_rotation_key_mode": prefixed_table.get("rotation_key_mode") if prefixed_table is not None else None,
        "selected_field": selected_field,
        "prob": prob[:n][selected],
        "score_pre_prior": score_pre_trimmed[selected],
        "score_with_prior": score_with[selected],
        "rotation_log_prior": rot_prior_trimmed[selected],
        "translation_log_prior": trans_prior_trimmed[selected],
        "combined_log_prior": combined_prior_trimmed[selected],
        "available_keys": sorted(payload.keys()),
    }


def _key_index(keys: np.ndarray) -> dict[tuple[int, int], int]:
    return {(int(rot), int(trans)): idx for idx, (rot, trans) in enumerate(np.asarray(keys, dtype=np.int64))}


def _collapse_duplicate_keys(keys: np.ndarray, table: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any], int]:
    keys = np.asarray(keys, dtype=np.int64)
    if keys.size == 0:
        return keys.reshape(0, 2), table, 0

    key_to_row: dict[tuple[int, int], int] = {}
    unique_keys: list[tuple[int, int]] = []
    for key in keys:
        tup = (int(key[0]), int(key[1]))
        if tup not in key_to_row:
            key_to_row[tup] = len(unique_keys)
            unique_keys.append(tup)

    duplicate_count = int(keys.shape[0] - len(unique_keys))
    if duplicate_count == 0:
        return keys, table, 0

    out = dict(table)
    out_keys = np.asarray(unique_keys, dtype=np.int64)
    prob = np.zeros(len(unique_keys), dtype=np.float64)
    score_fields = {
        name: np.full(len(unique_keys), -np.inf, dtype=np.float64)
        for name in (
            "score_pre_prior",
            "score_with_prior",
            "rotation_log_prior",
            "translation_log_prior",
            "combined_log_prior",
        )
    }
    for source_row, key in enumerate(keys):
        target_row = key_to_row[(int(key[0]), int(key[1]))]
        p = float(np.asarray(table["prob"], dtype=np.float64)[source_row])
        if np.isfinite(p):
            prob[target_row] += p
        for name, dest in score_fields.items():
            value = float(np.asarray(table[name], dtype=np.float64)[source_row])
            if np.isfinite(value) and (not np.isfinite(dest[target_row]) or value > dest[target_row]):
                dest[target_row] = value

    for name, values in score_fields.items():
        values[~np.isfinite(values)] = np.nan
        out[name] = values
    out["prob"] = prob
    return out_keys, out, duplicate_count


def _nearest_rotation_rows_by_matrix(
    rel_rots: np.ndarray,
    rec_rots: np.ndarray,
    *,
    chunk_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Map RELION rotation matrices to nearest RECOVAR local rotation rows."""

    rel_rots = np.asarray(rel_rots, dtype=np.float64)
    rec_rots = np.asarray(rec_rots, dtype=np.float64)
    rec_direct = rec_rots.reshape(rec_rots.shape[0], -1)
    rec_transposed = np.swapaxes(rec_rots, 1, 2).reshape(rec_rots.shape[0], -1)
    rec_direct_norm = np.einsum("ij,ij->i", rec_direct, rec_direct)
    rec_transposed_norm = np.einsum("ij,ij->i", rec_transposed, rec_transposed)

    def match_against(rec_flat: np.ndarray, rec_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nearest = np.empty(rel_rots.shape[0], dtype=np.int64)
        min_d2 = np.empty(rel_rots.shape[0], dtype=np.float64)
        for start in range(0, rel_rots.shape[0], int(chunk_size)):
            stop = min(start + int(chunk_size), rel_rots.shape[0])
            rel_flat = rel_rots[start:stop].reshape(stop - start, -1)
            rel_norm = np.einsum("ij,ij->i", rel_flat, rel_flat)
            d2 = rel_norm[:, None] + rec_norm[None, :] - 2.0 * (rel_flat @ rec_flat.T)
            d2 = np.maximum(d2, 0.0)
            argmin = np.argmin(d2, axis=1)
            nearest[start:stop] = argmin
            min_d2[start:stop] = d2[np.arange(stop - start), argmin]
        return nearest, np.sqrt(min_d2)

    direct_nearest, direct_min = match_against(rec_direct, rec_direct_norm)
    transposed_nearest, transposed_min = match_against(rec_transposed, rec_transposed_norm)
    if float(np.median(transposed_min)) < float(np.median(direct_min)):
        return transposed_nearest, transposed_min, "transpose"
    return direct_nearest, direct_min, "direct"


def _matrix_mapped_relion_keys(relion: dict[str, Any], recovar: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]] | None:
    rel_rots = relion.get("rot_matrices")
    if rel_rots is None:
        return None
    rel_rots = np.asarray(rel_rots, dtype=np.float64)
    rec_rots = np.asarray(recovar["rotations"], dtype=np.float64)
    rel_keys = np.asarray(relion.get("keys_local", relion["keys"]), dtype=np.int64)
    if rel_rots.ndim != 3 or rel_rots.shape[1:] != (3, 3):
        return None
    if rec_rots.ndim != 3 or rec_rots.shape[1:] != (3, 3):
        return None
    if rel_keys.size == 0 or np.max(rel_keys[:, 0]) >= rel_rots.shape[0]:
        return None

    unique_relion_rows, inverse = np.unique(rel_keys[:, 0], return_inverse=True)
    nearest_unique, min_dist_unique, orientation = _nearest_rotation_rows_by_matrix(
        rel_rots[unique_relion_rows],
        rec_rots,
    )

    mapped = rel_keys.copy()
    mapped[:, 0] = nearest_unique[inverse]
    return mapped, {
        "rotation_matrix_orientation": orientation,
        "rotation_matrix_match_median_frobenius": float(np.median(min_dist_unique)),
        "rotation_matrix_match_max_frobenius": float(np.max(min_dist_unique)),
        "rotation_matrix_unique_relion_rows": int(unique_relion_rows.size),
        "rotation_matrix_recovar_rows": int(rec_rots.shape[0]),
        "rotation_matrix_matcher": "chunked_unique_relion_rows",
    }


def _relion_grid_mapped_keys(relion: dict[str, Any], recovar: dict[str, Any], n_psi: int | None) -> tuple[np.ndarray, dict[str, Any]] | None:
    if n_psi is None:
        return None
    n_psi = int(n_psi)
    if n_psi <= 0:
        raise ValueError(f"relion_n_psi must be positive, got {n_psi}")
    relion_keys = np.asarray(relion["keys"], dtype=np.int64)
    relion_rotation_count = relion.get("rotation_count")
    if relion_rotation_count is not None:
        n_rot = int(relion_rotation_count)
    else:
        rec_rots = np.asarray(recovar["rotations"])
        n_rot = int(rec_rots.shape[0])
    if n_rot <= 0 or n_rot % n_psi:
        raise ValueError(f"RELION rotation count {n_rot} is not divisible by relion_n_psi={n_psi}")
    n_pixels = n_rot // n_psi
    mapped = relion_keys.copy()
    rel_rot = mapped[:, 0]
    if rel_rot.size and (int(np.max(rel_rot)) >= n_rot or int(np.min(rel_rot)) < 0):
        raise ValueError("RELION rotation indices are outside the RELION first-iteration rotation grid")
    pixel = rel_rot // n_psi
    psi = rel_rot % n_psi
    mapped[:, 0] = psi * n_pixels + pixel
    return mapped, {
        "rotation_index_mapping": "relion_pixel_major_to_recovar_psi_major",
        "relion_n_psi": n_psi,
        "relion_n_pixels": int(n_pixels),
        "relion_rotation_count": int(n_rot),
    }


def _choose_match_keys(
    *,
    relion: dict[str, Any],
    recovar: dict[str, Any],
    match_mode: str,
    relion_n_psi: int | None = None,
) -> tuple[str, np.ndarray, np.ndarray, dict[str, Any]]:
    if match_mode not in {"auto", "global", "local", "matrix", "relion_grid", "relion_grid_parent"}:
        raise ValueError(
            "match_mode must be auto, global, local, matrix, relion_grid, or "
            f"relion_grid_parent; got {match_mode!r}"
        )

    relion_keys = np.asarray(relion["keys"], dtype=np.int64)
    relion_global_keys = np.asarray(relion.get("keys_global", relion_keys), dtype=np.int64)
    relion_local_keys = np.asarray(relion.get("keys_local", relion_keys), dtype=np.int64)
    choices: list[tuple[str, np.ndarray, np.ndarray, dict[str, Any]]] = [
        ("global", relion_global_keys, recovar["keys_global"], {}),
        ("local", relion_local_keys, recovar["keys_local"], {}),
    ]
    recovar_parent_keys = recovar.get("keys_parent")
    relion_parent_keys = relion.get("keys_parent")
    if recovar_parent_keys is not None and relion_parent_keys is not None:
        choices.append(
            (
                "relion_grid_parent",
                relion_parent_keys,
                recovar_parent_keys,
                {
                    "relion_key_mapping": "fine_pass2_candidate_to_coarse_parent",
                    "recovar_key_mapping": "fine_pass2_candidate_to_coarse_parent",
                    **relion.get("parent_mapping_details", {}),
                    **recovar.get("parent_mapping_details", {}),
                },
            )
        )
    try:
        relion_grid = _relion_grid_mapped_keys(relion, recovar, relion_n_psi)
    except ValueError:
        if match_mode == "relion_grid_parent" and recovar_parent_keys is not None and relion_parent_keys is not None:
            relion_grid = None
        else:
            raise
    if relion_grid is not None:
        mapped_keys, details = relion_grid
        choices.append(("relion_grid", mapped_keys, recovar["keys_global"], details))
        if recovar_parent_keys is not None:
            if relion_parent_keys is None:
                parent_details = {
                    **details,
                    "recovar_key_mapping": "fine_pass2_candidate_to_coarse_parent",
                    **recovar.get("parent_mapping_details", {}),
                }
                choices.append(("relion_grid_parent", mapped_keys, recovar_parent_keys, parent_details))

    if match_mode != "auto":
        for choice in choices:
            if choice[0] == match_mode:
                return choice
        if match_mode == "matrix":
            matrix = _matrix_mapped_relion_keys(relion, recovar)
            if matrix is not None:
                mapped_keys, details = matrix
                return ("matrix", mapped_keys, recovar["keys_local"], details)
        raise ValueError(f"match_mode={match_mode!r} is unavailable or too large for these dumps")

    matrix = _matrix_mapped_relion_keys(relion, recovar)
    if matrix is not None:
        mapped_keys, details = matrix
        choices.append(("matrix", mapped_keys, recovar["keys_local"], details))

    def common_count(choice: tuple[str, np.ndarray, np.ndarray, dict[str, Any]]) -> int:
        _, rel_keys, rec_keys, _ = choice
        return len(set(_key_index(rel_keys)) & set(_key_index(rec_keys)))

    return max(choices, key=common_count)


def _rank_relion_acc_table_prefix(
    *,
    relion_dump_dir: Path,
    recovar: dict[str, Any],
    prefix: str,
    reconstruction_only: bool,
    match_mode: str,
    relion_n_psi: int | None,
    relion_parent_rot_divisor: int | None,
    relion_parent_trans_divisor: int | None,
) -> dict[str, Any]:
    try:
        relion = _candidate_table_from_relion(
            relion_dump_dir,
            reconstruction_only=reconstruction_only,
            acc_table_prefix=prefix,
            class_index=recovar.get("class_index"),
            parent_rotation_divisor=relion_parent_rot_divisor,
            parent_translation_divisor=relion_parent_trans_divisor,
        )
        chosen_match_mode, relion_keys, recovar_keys, match_details = _choose_match_keys(
            relion=relion,
            recovar=recovar,
            match_mode=match_mode,
            relion_n_psi=relion_n_psi,
        )
        relion_keys, relion_matched, relion_duplicate_keys_collapsed = _collapse_duplicate_keys(relion_keys, relion)
        recovar_keys, recovar_matched, recovar_duplicate_keys_collapsed = _collapse_duplicate_keys(
            recovar_keys,
            recovar,
        )
        rel_idx = _key_index(relion_keys)
        rec_idx = _key_index(recovar_keys)
        common_keys = sorted(set(rel_idx) & set(rec_idx))
        out: dict[str, Any] = {
            "prefix": prefix,
            "match_mode": chosen_match_mode,
            "match_details": match_details,
            "relion_rotation_key_mode": relion.get("relion_rotation_key_mode"),
            "common_candidate_count": int(len(common_keys)),
            "relion_candidate_count": int(len(rel_idx)),
            "recovar_candidate_count": int(len(rec_idx)),
            "relion_duplicate_keys_collapsed": int(relion_duplicate_keys_collapsed),
            "recovar_duplicate_keys_collapsed": int(recovar_duplicate_keys_collapsed),
            "relion_top_key": list(max(rel_idx, key=lambda key: relion_matched["prob"][rel_idx[key]]))
            if rel_idx
            else None,
            "recovar_top_key": list(max(rec_idx, key=lambda key: recovar_matched["prob"][rec_idx[key]]))
            if rec_idx
            else None,
        }
        if common_keys:
            common_rel = np.array([rel_idx[key] for key in common_keys], dtype=np.int64)
            common_rec = np.array([rec_idx[key] for key in common_keys], dtype=np.int64)
            rel_values = _finite_centered(np.asarray(relion_matched["score_pre_prior"], dtype=np.float64)[common_rel])
            rec_values = _finite_centered(np.asarray(recovar_matched["score_pre_prior"], dtype=np.float64)[common_rec])
            diff = rec_values - rel_values
            finite = np.isfinite(rel_values) & np.isfinite(rec_values)
            out["score_pre_prior_centered_diff"] = _safe_stats(diff)
            out["score_pre_prior_centered_corr"] = (
                float(np.corrcoef(rel_values[finite], rec_values[finite])[0, 1]) if int(np.sum(finite)) > 1 else None
            )
        return out
    except Exception as exc:  # pragma: no cover - retained in JSON diagnostics.
        return {"prefix": prefix, "error": f"{type(exc).__name__}: {exc}"}


def _rank_relion_acc_table_prefixes(
    *,
    relion_dump_dir: Path,
    recovar: dict[str, Any],
    requested_prefix: str | None,
    reconstruction_only: bool,
    match_mode: str,
    relion_n_psi: int | None,
    relion_parent_rot_divisor: int | None,
    relion_parent_trans_divisor: int | None,
) -> dict[str, Any] | None:
    payload = parse_dump_dir(relion_dump_dir)
    prefixes = _discover_relion_acc_table_prefixes(payload)
    if not prefixes:
        return None

    rankings = [
        _rank_relion_acc_table_prefix(
            relion_dump_dir=relion_dump_dir,
            recovar=recovar,
            prefix=prefix,
            reconstruction_only=reconstruction_only,
            match_mode=match_mode,
            relion_n_psi=relion_n_psi,
            relion_parent_rot_divisor=relion_parent_rot_divisor,
            relion_parent_trans_divisor=relion_parent_trans_divisor,
        )
        for prefix in prefixes
    ]

    def sort_key(item: dict[str, Any]) -> tuple[int, float, float, int]:
        if "error" in item:
            return (-1, -np.inf, np.inf, 0)
        diff = item.get("score_pre_prior_centered_diff") or {}
        corr = item.get("score_pre_prior_centered_corr")
        corr_value = float(corr) if corr is not None and np.isfinite(corr) else -np.inf
        max_abs = diff.get("max_abs")
        max_abs_value = float(max_abs) if max_abs is not None and np.isfinite(max_abs) else np.inf
        return (0, corr_value, -max_abs_value, int(item.get("common_candidate_count", 0)))

    rankings = sorted(rankings, key=sort_key, reverse=True)
    requested_rank = None
    if requested_prefix is not None:
        for idx, item in enumerate(rankings, start=1):
            if item.get("prefix") == requested_prefix:
                requested_rank = idx
                break
    return {
        "available_relion_acc_table_prefixes": prefixes,
        "best_relion_acc_table_prefix": rankings[0].get("prefix") if rankings else None,
        "requested_relion_acc_table_prefix": requested_prefix,
        "requested_relion_acc_table_prefix_rank": requested_rank,
        "relion_acc_table_prefix_rankings": rankings,
    }


def compare_dumps(
    relion_dump_dir: Path,
    recovar_pass2_npz: Path,
    *,
    reconstruction_only: bool = False,
    match_mode: str = "auto",
    relion_acc_table_prefix: str | None = None,
    recovar_class_index: int | None = None,
    relion_n_psi: int | None = None,
    relion_parent_rot_divisor: int | None = None,
    relion_parent_trans_divisor: int | None = None,
    recovar_parent_rot_divisor: int | None = None,
    recovar_parent_trans_divisor: int | None = None,
) -> dict[str, Any]:
    recovar = _candidate_table_from_recovar(
        recovar_pass2_npz,
        reconstruction_only=reconstruction_only,
        class_index_override=recovar_class_index,
        parent_rotation_divisor=recovar_parent_rot_divisor,
        parent_translation_divisor=recovar_parent_trans_divisor,
    )
    auto_prefix_rankings: dict[str, Any] | None = None
    requested_relion_acc_table_prefix = relion_acc_table_prefix
    if relion_acc_table_prefix == "auto":
        auto_prefix_rankings = _rank_relion_acc_table_prefixes(
            relion_dump_dir=relion_dump_dir,
            recovar=recovar,
            requested_prefix=None,
            reconstruction_only=reconstruction_only,
            match_mode=match_mode,
            relion_n_psi=relion_n_psi,
            relion_parent_rot_divisor=relion_parent_rot_divisor,
            relion_parent_trans_divisor=relion_parent_trans_divisor,
        )
        if auto_prefix_rankings is None or auto_prefix_rankings.get("best_relion_acc_table_prefix") is None:
            raise ValueError(f"RELION dump {relion_dump_dir} has no part-specific ACC tables to auto-select")
        relion_acc_table_prefix = str(auto_prefix_rankings["best_relion_acc_table_prefix"])
    relion = _candidate_table_from_relion(
        relion_dump_dir,
        reconstruction_only=reconstruction_only,
        acc_table_prefix=relion_acc_table_prefix,
        class_index=recovar.get("class_index"),
        parent_rotation_divisor=relion_parent_rot_divisor,
        parent_translation_divisor=relion_parent_trans_divisor,
    )
    chosen_match_mode, relion_keys, recovar_keys, match_details = _choose_match_keys(
        relion=relion,
        recovar=recovar,
        match_mode=match_mode,
        relion_n_psi=relion_n_psi,
    )
    relion_keys, relion_matched, relion_duplicate_keys_collapsed = _collapse_duplicate_keys(relion_keys, relion)
    recovar_keys, recovar_matched, recovar_duplicate_keys_collapsed = _collapse_duplicate_keys(recovar_keys, recovar)

    rel_idx = _key_index(relion_keys)
    rec_idx = _key_index(recovar_keys)
    raw_rel_idx = _key_index(relion["keys"])
    common_keys = sorted(set(rel_idx) & set(rec_idx))
    rel_only = sorted(set(rel_idx) - set(rec_idx))
    rec_only = sorted(set(rec_idx) - set(rel_idx))
    common_rel = np.array([rel_idx[key] for key in common_keys], dtype=np.int64)
    common_rec = np.array([rec_idx[key] for key in common_keys], dtype=np.int64)

    result: dict[str, Any] = {
        "relion_source": relion["source"],
        "recovar_source": recovar["source"],
        "recovar_original_index": recovar["original_index"],
        "recovar_local_index": recovar["local_index"],
        "recovar_class_index": recovar["class_index"],
        "recovar_current_size": recovar["current_size"],
        "reconstruction_only": bool(reconstruction_only),
        "relion_selected_field": relion["selected_field"],
        "relion_rotation_key_mode": relion.get("relion_rotation_key_mode"),
        "recovar_selected_field": recovar["selected_field"],
        "recovar_reconstruction_n_significant": recovar["reconstruction_n_significant"],
        "match_mode": chosen_match_mode,
        "match_details": match_details,
        "relion_candidate_count": int(len(rel_idx)),
        "recovar_candidate_count": int(len(rec_idx)),
        "relion_duplicate_keys_collapsed": int(relion_duplicate_keys_collapsed),
        "recovar_duplicate_keys_collapsed": int(recovar_duplicate_keys_collapsed),
        "common_candidate_count": int(len(common_keys)),
        "relion_only_count": int(len(rel_only)),
        "recovar_only_count": int(len(rec_only)),
        "candidate_jaccard": float(len(common_keys) / max(1, len(set(rel_idx) | set(rec_idx)))),
        "relion_raw_top_key": list(max(raw_rel_idx, key=lambda key: relion["prob"][raw_rel_idx[key]]))
        if raw_rel_idx
        else None,
        "relion_matched_top_key": list(max(rel_idx, key=lambda key: relion_matched["prob"][rel_idx[key]]))
        if rel_idx
        else None,
        "relion_top_key": list(max(rel_idx, key=lambda key: relion_matched["prob"][rel_idx[key]])) if rel_idx else None,
        "recovar_top_key": list(max(rec_idx, key=lambda key: recovar_matched["prob"][rec_idx[key]])) if rec_idx else None,
        "sample_relion_only_keys": [list(key) for key in rel_only[:10]],
        "sample_recovar_only_keys": [list(key) for key in rec_only[:10]],
    }

    if common_keys:
        rel_prob = np.asarray(relion_matched["prob"], dtype=np.float64)[common_rel]
        rec_prob = np.asarray(recovar_matched["prob"], dtype=np.float64)[common_rec]
        rel_prob_norm = rel_prob / max(float(np.sum(rel_prob)), np.finfo(np.float64).tiny)
        rec_prob_norm = rec_prob / max(float(np.sum(rec_prob)), np.finfo(np.float64).tiny)
        result.update(
            {
                "common_relion_prob_mass": float(np.sum(rel_prob)),
                "common_recovar_prob_mass": float(np.sum(rec_prob)),
                "common_prob_l1_after_common_renorm": float(np.sum(np.abs(rel_prob_norm - rec_prob_norm))),
                "common_prob_corr": float(np.corrcoef(rel_prob_norm, rec_prob_norm)[0, 1])
                if len(common_keys) > 1
                else None,
            }
        )
        for field in (
            "score_pre_prior",
            "score_with_prior",
            "rotation_log_prior",
            "translation_log_prior",
            "combined_log_prior",
        ):
            rel_raw = np.asarray(relion_matched[field], dtype=np.float64)[common_rel]
            rec_raw = np.asarray(recovar_matched[field], dtype=np.float64)[common_rec]
            if field.endswith("_log_prior"):
                result[f"common_{field}_diff"] = _safe_stats(rec_raw - rel_raw)
            rel_values = _finite_centered(rel_raw)
            rec_values = _finite_centered(rec_raw)
            result[f"common_{field}_centered_diff"] = _safe_stats(rec_values - rel_values)

    if auto_prefix_rankings is not None:
        result.update(auto_prefix_rankings)
        result["requested_relion_acc_table_prefix"] = requested_relion_acc_table_prefix
        result["selected_relion_acc_table_prefix"] = relion_acc_table_prefix
    elif relion_acc_table_prefix is not None:
        prefix_rankings = _rank_relion_acc_table_prefixes(
            relion_dump_dir=relion_dump_dir,
            recovar=recovar,
            requested_prefix=relion_acc_table_prefix,
            reconstruction_only=reconstruction_only,
            match_mode=match_mode,
            relion_n_psi=relion_n_psi,
            relion_parent_rot_divisor=relion_parent_rot_divisor,
            relion_parent_trans_divisor=relion_parent_trans_divisor,
        )
        if prefix_rankings is not None:
            result.update(prefix_rankings)
            result["selected_relion_acc_table_prefix"] = relion_acc_table_prefix

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relion-dump-dir", required=True, type=Path)
    parser.add_argument("--recovar-pass2-npz", required=True, type=Path)
    parser.add_argument("--reconstruction-only", action="store_true")
    parser.add_argument(
        "--relion-acc-table-prefix",
        default=None,
        help=(
            "Use a part-specific full RELION ACC table, e.g. "
            "img0_part7778_pass1_class0_pass1 for "
            "img0_part7778_pass1_class0_pass1_diff2_weights.bin. "
            "Use 'auto' to rank and select the best part-specific table."
        ),
    )
    parser.add_argument(
        "--recovar-class-index",
        type=int,
        default=None,
        help="Select this zero-based class from a RECOVAR significance dump.",
    )
    parser.add_argument(
        "--match-mode",
        choices=("auto", "global", "local", "matrix", "relion_grid", "relion_grid_parent"),
        default="auto",
        help=(
            "Match rotations by RECOVAR global oversampled index, local pass-2 row index, "
            "RELION-to-RECOVAR nearest rotation matrix, RELION pixel-major firstiter grid, "
            "RELION pixel-major grid collapsed to RECOVAR pass-2 coarse parents, "
            "or whichever overlaps more."
        ),
    )
    parser.add_argument(
        "--relion-n-psi",
        type=int,
        default=None,
        help="Number of psi samples in a RELION pixel-major firstiter grid; use 48 for healpix_order 3.",
    )
    parser.add_argument(
        "--relion-parent-rot-divisor",
        type=int,
        default=None,
        help=(
            "For relion_grid_parent matching on ACC candidate dumps, map RELION compact fine "
            "rotation rows to parent rows by integer division with this factor."
        ),
    )
    parser.add_argument(
        "--relion-parent-trans-divisor",
        type=int,
        default=None,
        help=(
            "For relion_grid_parent matching, map RELION fine translation ids to parent "
            "translation ids by integer division. By default candidate_coarse_trans_idx is "
            "used when present."
        ),
    )
    parser.add_argument(
        "--recovar-parent-rot-divisor",
        type=int,
        default=None,
        help=(
            "For relion_grid_parent matching, map RECOVAR fine global rotation ids "
            "to coarse parent ids by integer division with this factor. When omitted, "
            "the dump's parent_map field is used."
        ),
    )
    parser.add_argument(
        "--recovar-parent-trans-divisor",
        type=int,
        default=None,
        help=(
            "For relion_grid_parent matching, map fine RECOVAR translation ids "
            "to coarse parent ids by integer division with this factor."
        ),
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    result = compare_dumps(
        args.relion_dump_dir,
        args.recovar_pass2_npz,
        reconstruction_only=args.reconstruction_only,
        match_mode=args.match_mode,
        relion_acc_table_prefix=args.relion_acc_table_prefix,
        recovar_class_index=args.recovar_class_index,
        relion_n_psi=args.relion_n_psi,
        relion_parent_rot_divisor=args.relion_parent_rot_divisor,
        relion_parent_trans_divisor=args.relion_parent_trans_divisor,
        recovar_parent_rot_divisor=args.recovar_parent_rot_divisor,
        recovar_parent_trans_divisor=args.recovar_parent_trans_divisor,
    )
    text = json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text)
        print(f"Wrote {args.output_json}")
    print(text, end="")


if __name__ == "__main__":
    main()

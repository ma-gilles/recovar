"""Default-off captures for InitialModel K-class localization.

Each records a quantity no output file carries, writes only when its environment
variable names a directory, and changes no production value.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np


_KCLASS_STATS_DUMP_ENV = "RECOVAR_VDAM_KCLASS_STATS_DUMP_DIR"


def k_class_statistics_capture_enabled() -> bool:
    """Whether the K-class statistics capture is on, for producers of its inputs.

    The engine buffer and the per-class driver copy of the pre-cast normalizer
    feed only this module, so both ask here rather than reading the environment
    separately. With the capture off neither is allocated, pulled or copied.
    """
    return bool(os.environ.get(_KCLASS_STATS_DUMP_ENV))


def _maybe_dump_k_class_statistics(result, *, iteration: int, halfset: int, image_indices) -> None:
    """Record the per-class score and normalization arrays when asked.

    Pmax is published as ``exp(global_best - global_log_evidence)``, so localizing a
    posterior difference between two engines needs the per-class best scores and log
    evidences that feed it, which no output file carries. Diagnostic only, default off.
    """

    directory = os.environ.get(_KCLASS_STATS_DUMP_ENV)
    if not directory:
        return
    target = pathlib.Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    per_class = getattr(result, "per_class_stats", None) or ()
    payload = {
        "image_indices": np.asarray(image_indices, dtype=np.int64),
        "halfset": np.asarray(int(halfset), dtype=np.int64),
        "joint_best": np.asarray(result.stats.best_log_score_per_image, dtype=np.float64),
        "joint_log_evidence": np.asarray(result.stats.log_evidence_per_image, dtype=np.float64),
        "joint_max_posterior": np.asarray(result.stats.max_posterior_per_image, dtype=np.float64),
        "class_responsibilities": np.asarray(result.class_responsibilities, dtype=np.float64),
        "class_posterior_sums": np.asarray(result.class_posterior_sums, dtype=np.float64),
    }
    uncast = getattr(result, "uncast_log_evidence_per_image", None)
    if uncast is not None:
        payload["uncast_log_evidence"] = np.asarray(uncast, dtype=np.float64)
    published = payload["joint_max_posterior"].shape[0]
    if payload["image_indices"].shape[0] != published:
        raise ValueError(
            "k-class statistics dump: %d image ids for %d published rows"
            % (payload["image_indices"].shape[0], published)
        )
    for class_index, stats in enumerate(per_class):
        payload[f"class{class_index}_best"] = np.asarray(stats.best_log_score_per_image, dtype=np.float64)
        payload[f"class{class_index}_log_evidence"] = np.asarray(stats.log_evidence_per_image, dtype=np.float64)
        payload[f"class{class_index}_max_posterior"] = np.asarray(stats.max_posterior_per_image, dtype=np.float64)
    np.savez(target / f"kclass_stats_it{int(iteration):03d}_half{int(halfset)}.npz", **payload)


_CLASS_CANDIDATE_COUNT_DUMP_ENV = "RECOVAR_VDAM_CLASS_CANDIDATE_COUNT_DUMP_DIR"


def _maybe_dump_class_candidate_counts(class_layouts, *, iteration: int, halfset: int) -> None:
    """Record the per-image, per-class pass-2 candidate counts when asked.

    Diagnostic only, default off. The counts decide how a shared class-segment
    width pads relative to per-class widths, which is a layout design question
    that cannot be answered from a synthetic distribution.
    """

    directory = os.environ.get(_CLASS_CANDIDATE_COUNT_DUMP_ENV)
    if not directory or not class_layouts:
        return
    target = pathlib.Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    counts = np.stack(
        [np.asarray(layout.rotation_counts, dtype=np.int64) for layout in class_layouts],
        axis=1,
    )
    np.save(target / f"class_candidate_counts_it{int(iteration):03d}_half{int(halfset)}.npy", counts)

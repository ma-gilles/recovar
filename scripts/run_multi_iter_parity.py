#!/usr/bin/env python
"""Run N iterations of recovar in RELION mode, save results for diff comparison.

Usage:
  pixi run python scripts/run_multi_iter_parity.py \
    --relion_dir .../relion_ref_os0 \
    --data_star .../particles.star \
    --iter 3 --max_iter 15 \
    --output_dir .../recovar_15iter
"""

import argparse
import json
import logging
import os
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np

from recovar.em.dense_single_volume.relion_replay import (
    read_relion_single_optics_sigma2_noise as _read_relion_single_optics_sigma2_noise,
)
from recovar.em.dense_single_volume.relion_replay import (
    relion_mpi_process_start_scoring_noise_pair as _relion_mpi_process_start_scoring_noise_pair,
)
from recovar.em.initial_model.gt_metrics import (
    DEFAULT_GT_ALIGN_HEALPIX_ORDER,
    DEFAULT_GT_ALIGN_MAX_SHELL,
)
from recovar.utils.parity_provenance import (
    _safe_git_commit,
)
from recovar.utils.parity_provenance import (
    assert_parity_ancestors_or_exit as _print_provenance_banner_and_assert_parity_ancestors,
)

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)


def build_gt_postprocess_command(
    *,
    recovar_dir: str | Path,
    relion_dir: str | Path,
    relion_start_iter: int,
    relion_run_prefix: str,
    gt_volume: str | Path,
    max_iter: int,
    intermediates_dir: str | Path | None = None,
    gt_align: bool = False,
    gt_align_healpix_order: int = DEFAULT_GT_ALIGN_HEALPIX_ORDER,
    gt_align_max_shell: int = DEFAULT_GT_ALIGN_MAX_SHELL,
    gt_align_no_mirror: bool = False,
    gt_align_allow_sign: bool = False,
    gt_align_all_series: bool = False,
) -> list[str]:
    """Build the GT postprocessor command using import-safe module execution."""
    command = [
        sys.executable,
        "-m",
        "scripts.postprocess_multi_iter_gt",
        "--recovar_dir",
        str(recovar_dir),
        "--relion_dir",
        str(relion_dir),
        "--relion_start_iter",
        str(relion_start_iter),
        "--relion_run_prefix",
        str(relion_run_prefix),
        "--gt_volume",
        str(gt_volume),
        "--max_iter",
        str(max_iter),
    ]
    if intermediates_dir is not None:
        command.extend(["--intermediates_dir", str(intermediates_dir)])
    if gt_align:
        command.extend(
            [
                "--gt_align",
                "--gt_align_healpix_order",
                str(gt_align_healpix_order),
                "--gt_align_max_shell",
                str(gt_align_max_shell),
            ]
        )
        if gt_align_no_mirror:
            command.append("--gt_align_no_mirror")
        if gt_align_allow_sign:
            command.append("--gt_align_allow_sign")
        if gt_align_all_series:
            command.append("--gt_align_all_series")
    return command


def stack_index_from_image_name(name: str) -> int:
    """Return the zero-based stack row encoded in a RELION image name."""
    m = re.match(r"(\d+)@", str(name))
    return int(m.group(1)) - 1 if m else -1


def read_relion_model_pixel_size(path: str | Path) -> float:
    """Read the model pixel size from a RELION reference MRC header."""

    import mrcfile

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"RELION model reference does not exist: {source}")
    with mrcfile.open(source, mode="r", header_only=True) as handle:
        pixel_size = float(handle.voxel_size.x)
    if not np.isfinite(pixel_size) or pixel_size <= 0.0:
        raise ValueError(
            f"RELION model reference has invalid pixel size {pixel_size}: {source}"
        )
    return pixel_size


def read_relion_optics_image_geometry(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Read the image size and pixel size for every RELION optics group."""

    import starfile

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"RELION particle STAR does not exist: {source}")
    star = starfile.read(source)
    optics = star.get("optics") if isinstance(star, dict) else None
    if optics is None:
        raise ValueError(f"RELION particle STAR has no optics table: {source}")
    required = {"rlnImageSize", "rlnImagePixelSize"}
    missing = required.difference(optics.columns)
    if missing:
        raise ValueError(
            "RELION optics table is missing "
            + ", ".join(sorted(missing))
            + f": {source}"
        )
    image_sizes = np.asarray(optics["rlnImageSize"], dtype=np.int64).reshape(-1)
    pixel_sizes = np.asarray(optics["rlnImagePixelSize"], dtype=np.float64).reshape(-1)
    if image_sizes.size == 0 or image_sizes.shape != pixel_sizes.shape:
        raise ValueError(f"RELION optics geometry is empty or misaligned: {source}")
    if np.any(image_sizes <= 0) or np.any(~np.isfinite(pixel_sizes)) or np.any(pixel_sizes <= 0.0):
        raise ValueError(f"RELION optics geometry contains invalid values: {source}")
    return image_sizes, pixel_sizes


def replay_previous_relion_iteration(init_relion_iteration: int, recovar_iteration: int) -> int:
    """Return the RELION iteration whose particle metadata seeds this replay step."""
    return int(init_relion_iteration) + int(recovar_iteration)


def replay_control_relion_iteration(init_relion_iteration: int, recovar_iteration: int) -> int:
    """Return the RELION iteration whose control variables govern this replay step."""
    return replay_previous_relion_iteration(init_relion_iteration, recovar_iteration) + 1


def replay_override_iteration_pairs(init_relion_iteration: int, max_iter: int) -> list[tuple[int, int, int]]:
    """Return numbered replay states, including the state that seeds final all-data."""
    return [
        (
            recovar_iteration,
            replay_previous_relion_iteration(init_relion_iteration, recovar_iteration),
            replay_control_relion_iteration(init_relion_iteration, recovar_iteration),
        )
        for recovar_iteration in range(1, int(max_iter) + 1)
    ]


def map_pose_arrays_to_particle_order(our_names, gt_rot_all, gt_trans_all=None):
    """Map pose arrays indexed by stack row onto the current particle ordering."""
    n_total = len(our_names)
    gt_rotations_orig = np.full((n_total, 3, 3), np.nan, dtype=np.float64)
    gt_translations_orig = np.full((n_total, 2), np.nan, dtype=np.float64) if gt_trans_all is not None else None
    for j, name in enumerate(our_names):
        pose_idx = stack_index_from_image_name(name)
        if 0 <= pose_idx < len(gt_rot_all):
            gt_rotations_orig[j] = gt_rot_all[pose_idx]
        if gt_translations_orig is not None and 0 <= pose_idx < len(gt_trans_all):
            gt_translations_orig[j] = gt_trans_all[pose_idx]
    return gt_rotations_orig, gt_translations_orig


def map_relion_scale_groups_to_half_order(
    group_numbers,
    relion_identity_to_position,
    half_identities,
):
    """Map one-based RELION scale groups to zero-based half-local rows."""

    group_numbers = np.asarray(group_numbers, dtype=np.int64).reshape(-1)
    if group_numbers.size and int(np.min(group_numbers)) < 1:
        raise ValueError("RELION rlnGroupNumber values must be 1-based positive integers")
    try:
        group_ids = np.asarray(
            [group_numbers[relion_identity_to_position[identity]] - 1 for identity in half_identities],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise ValueError(f"RELION scale-group table is missing particle identity {exc.args[0]!r}") from exc
    group_count = int(np.max(group_numbers)) if group_numbers.size else 0
    return group_ids, group_count


def retain_group_scale_update_state(
    *,
    max_iter: int,
    skip_final_iteration: bool,
    diagnostic_retain_terminal_state: bool = False,
) -> bool:
    """Whether a later score can consume group scales updated by this replay.

    The legacy dense ``adaptive_oversampling=0`` engine does not accumulate
    RELION XA/AA group-scale sufficient statistics. A single explicitly
    terminal numbered pass may still use the loaded per-particle scale factors
    for scoring because its newly estimated scale state has no downstream
    consumer. All nonterminal replays retain group IDs and therefore keep the
    engine's fail-closed protection against silently skipping an update.
    """

    return bool(diagnostic_retain_terminal_state) or not (
        int(max_iter) == 1 and bool(skip_final_iteration)
    )


def add_significant_count_artifacts(save_dict, significant_counts, half_indices, n_images):
    """Save parity support counts in explicit half and source-image order."""
    half_order_indices = np.concatenate(
        [np.asarray(indices, dtype=np.int64) for indices in half_indices],
    )
    for iteration, counts in enumerate(significant_counts):
        if counts is None:
            continue
        if isinstance(counts, (list, tuple)):
            present = [np.asarray(value) for value in counts if value is not None]
            if not present:
                continue
            counts_half_order = np.concatenate(present, axis=0)
        else:
            counts_half_order = np.asarray(counts)
        flat_counts = counts_half_order.reshape(-1)
        legacy_key = f"sig_counts_iter_{iteration:03d}"
        save_dict[legacy_key] = counts_half_order
        save_dict[f"sig_counts_half_order_iter_{iteration:03d}"] = counts_half_order
        if flat_counts.shape[0] != half_order_indices.shape[0]:
            continue
        counts_by_image = np.full(int(n_images), -1, dtype=flat_counts.dtype)
        counts_by_image[half_order_indices] = flat_counts
        save_dict[f"sig_counts_by_image_iter_{iteration:03d}"] = counts_by_image


def particle_half_indices(
    random_subsets,
    *,
    fresh_order_seed: int | None = None,
    optics_group_ids=None,
    first_iteration: int = 1,
):
    """Return source-order or reconstructed fresh RELION half orders."""

    subsets = np.asarray(random_subsets)
    if fresh_order_seed is not None:
        from recovar.em.dense_single_volume.helpers.expected_accuracy import (
            relion_auto_refine_half_orders,
        )

        return relion_auto_refine_half_orders(
            subsets,
            int(fresh_order_seed),
            int(first_iteration),
            optics_group_ids=optics_group_ids,
        )
    return (
        np.flatnonzero(subsets == 1).astype(np.int64),
        np.flatnonzero(subsets == 2).astype(np.int64),
    )


def map_relion_half_orders_to_dataset_rows(
    dataset_image_names,
    relion_image_names,
    relion_half_orders,
):
    """Map RELION source-row orders to dataset rows by immutable image identity."""

    dataset_names = tuple(str(name) for name in dataset_image_names)
    relion_names = tuple(str(name) for name in relion_image_names)
    dataset_rows = {name: row for row, name in enumerate(dataset_names)}
    if len(dataset_rows) != len(dataset_names):
        raise ValueError("dataset rlnImageName identities are not unique")
    if len(set(relion_names)) != len(relion_names):
        raise ValueError("RELION rlnImageName identities are not unique")
    if set(dataset_names) != set(relion_names):
        raise ValueError("dataset and RELION rlnImageName identities differ")
    return tuple(
        np.asarray(
            [dataset_rows[relion_names[int(row)]] for row in order],
            dtype=np.int64,
        )
        for order in relion_half_orders
    )


def _count_compile_lines(log_path):
    if log_path is None:
        return None
    path = Path(log_path)
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    return sum("Compiling" in line for line in text.splitlines())


def _collect_local_profile_rows(save_intermediates_dir):
    rows = []
    scalar_keys = [
        "n_chunks",
        "em_time_s",
        "accounted_em_time_s",
        "unattributed_em_time_s",
        "sum_union_rows",
        "sum_padded_rows",
        "sum_nonzero_posterior_rows",
        "sum_reconstruction_rows",
        "sum_significant_samples",
        "unique_global_rotations",
        "unique_nonzero_global_rotations",
        "unique_reconstruction_global_rotations",
        "duplicate_rotation_factor",
        "reconstruction_duplicate_rotation_factor",
        "sum_union_row_pixels",
        "adjoint_seconds_per_row_pixel",
        "union_waste_fraction",
        "padded_waste_fraction",
        "padding_only_waste_fraction",
        "preprocess_time_s",
        "preprocess_integer_shift_s",
        "preprocess_translation_phase_s",
        "preprocess_score_process_s",
        "preprocess_recon_process_s",
        "preprocess_ctf_s",
        "preprocess_tile_shift_score_s",
        "preprocess_tile_shift_recon_s",
        "preprocess_norm_s",
        "projection_time_s",
        "fused_score_mstep_s",
        "local_score_s",
        "local_normalize_s",
        "local_significance_s",
        "local_mstep_s",
        "local_pack_s",
        "local_backproject_y_s",
        "local_backproject_ctf_s",
        "local_noise_s",
        "local_postprocess_s",
        "local_host_stats_s",
        "local_final_accumulator_s",
        "local_stats_finalize_s",
        "selector_time_s",
        "metadata_build_time_s",
        "translation_prior_time_s",
        "raw_cache_build_time_s",
        "bucket_build_time_s",
        "batch_fetch_time_s",
        "transfer_total_to_host_s",
        "transfer_reconstruction_mask_to_host_s",
        "transfer_mstep_posterior_sum_to_host_s",
        "transfer_postprocess_argmax_to_host_s",
        "transfer_postprocess_scores_to_host_s",
        "transfer_postprocess_posterior_to_host_s",
        "transfer_final_noise_to_host_s",
        "local_total_hypotheses",
        "local_mean_rotations_per_image",
        "local_mean_significant_samples_per_image",
        "local_mean_reconstruction_rows_per_image",
        "local_num_buckets",
        "local_pad_fraction",
        "max_hypotheses_per_microbatch",
        "n_windowed",
        "native_half_preprocess",
        "native_half_preprocess_mode",
        "fused_score_mstep_enabled",
        "raw_cache_enabled",
    ]
    for npz_path in sorted(Path(save_intermediates_dir).glob("*_local_profile.npz")):
        with np.load(npz_path) as profile_npz:
            row = {"path": str(npz_path)}
            for key in scalar_keys:
                if key in profile_npz:
                    value = profile_npz[key]
                    row[key] = value.item() if np.ndim(value) == 0 else np.asarray(value).tolist()
            rows.append(row)
    return rows


def _profile_value_to_jsonable(value):
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    return arr.tolist()


def _collect_local_profile_history(result):
    return [
        {key: _profile_value_to_jsonable(value) for key, value in row.items()}
        for row in result.get("local_profile_history", [])
    ]


def _summarize_local_profile_rows(rows, wall_times):
    """Aggregate exact-local profile rows for timing ledgers."""
    if not rows:
        return {}
    sum_keys = [
        "em_time_s",
        "accounted_em_time_s",
        "unattributed_em_time_s",
        "preprocess_time_s",
        "projection_time_s",
        "fused_score_mstep_s",
        "local_score_s",
        "local_normalize_s",
        "local_significance_s",
        "local_mstep_s",
        "local_pack_s",
        "local_backproject_y_s",
        "local_backproject_ctf_s",
        "local_noise_s",
        "local_postprocess_s",
        "local_host_stats_s",
        "local_final_accumulator_s",
        "local_stats_finalize_s",
        "selector_time_s",
        "raw_cache_build_time_s",
        "bucket_build_time_s",
        "batch_fetch_time_s",
        "transfer_total_to_host_s",
    ]
    summary = {
        "n_profile_rows": len(rows),
        "sum_wall_times_s": float(np.sum(np.asarray(wall_times, dtype=np.float64))) if wall_times else None,
    }
    for key in sum_keys:
        values = [float(row[key]) for row in rows if key in row]
        if values:
            summary[f"sum_{key}"] = float(np.sum(values))
    if summary["sum_wall_times_s"] is not None and "sum_em_time_s" in summary:
        summary["wall_minus_exact_local_s"] = float(summary["sum_wall_times_s"] - summary["sum_em_time_s"])
    if "sum_em_time_s" in summary and "sum_accounted_em_time_s" in summary:
        summary["exact_local_unaccounted_check_s"] = float(
            summary["sum_em_time_s"] - summary["sum_accounted_em_time_s"]
        )
    for key in (
        "native_half_preprocess",
        "native_half_preprocess_mode",
        "fused_score_mstep_enabled",
    ):
        values = [row[key] for row in rows if key in row]
        if values:
            summary[key] = values[0]
    return summary


def _read_relion_pmax_column(relion_df):
    """Return RELION per-particle Pmax values when available.

    Older benchmark directories do not always carry
    ``rlnMaxValueProbDistribution``. Keep those runs usable for timing and
    structural parity checks by treating the field as optional.
    """

    if "rlnMaxValueProbDistribution" not in relion_df:
        return None
    return np.array(relion_df["rlnMaxValueProbDistribution"], dtype=np.float64)


def _read_relion_scheduling_average_pmax(model, *, relion_iteration: int) -> float:
    """Read the model scalar RELION uses for scheduling the next iteration.

    The arithmetic mean of ``data.star::rlnMaxValueProbDistribution`` is a
    useful per-particle comparison metric, but it is not RELION's optimizer
    state.  Current-size growth consumes ``model.star::rlnAveragePmax``.
    """

    model_general = model.get("model_general", {}) if isinstance(model, dict) else {}
    value = model_general.get("rlnAveragePmax")
    if value is not None and np.isfinite(float(value)):
        return float(value)
    if int(relion_iteration) == 0:
        return 0.0
    raise ValueError(
        "RELION model is missing finite rlnAveragePmax required for exact "
        f"iteration-{int(relion_iteration)} scheduling replay"
    )


def parse_relion_optimiser_cli_flags(opt_text: str) -> dict[str, object]:
    """Extract selected CLI flags from RELION's optimiser STAR header."""
    cli_line = next(
        (line.lstrip("#").strip() for line in opt_text.splitlines() if line.lstrip().startswith("# --")),
        "",
    )
    ini_high_match = re.search(r"(?:^|\s)--ini_high\s+(\S+)", cli_line)
    return {
        "cli_line": cli_line,
        "do_firstiter_cc": bool(re.search(r"(?:^|\s)--firstiter_cc(?:\s|$)", cli_line)),
        "ini_high_angstrom": float(ini_high_match.group(1)) if ini_high_match else None,
    }


def resolve_firstiter_cc_mode(mode: str, *, oracle_enabled: bool, start_iteration: int) -> bool:
    """Resolve the typed firstiter-CC replay policy against the RELION oracle."""
    if mode not in {"auto", "on", "off"}:
        raise ValueError(f"unknown firstiter-CC mode {mode!r}")
    if mode == "on" and int(start_iteration) != 0:
        raise ValueError("--firstiter-cc-mode on requires --iter 0")
    if int(start_iteration) != 0:
        return False
    if mode == "auto":
        return bool(oracle_enabled)
    return mode == "on"


def resolve_relion_final_oracle_paths(
    relion_dir: str | Path,
    *,
    run_prefix: str = "run",
    start_iteration: int,
    completed_iterations: int,
    final_all_data_ran: bool,
) -> tuple[str, dict[str, Path]]:
    """Resolve the RELION maps with the same finalization semantics as RECOVAR."""
    relion_dir = Path(relion_dir)
    if final_all_data_ran:
        return "all_data", {"merged": relion_dir / f"{run_prefix}_class001.mrc"}
    final_numbered_iteration = int(start_iteration) + int(completed_iterations)
    return "split_half", {
        label: relion_dir / f"{run_prefix}_it{final_numbered_iteration:03d}_{label}_class001.mrc"
        for label in ("half1", "half2")
    }


def _normalized_fsc_auc(fsc: np.ndarray) -> float:
    """Integrate finite non-DC FSC shells on a normalized frequency axis."""
    values = np.asarray(fsc, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    if finite.size:
        finite[0] = False
    if np.count_nonzero(finite) < 2:
        return float("nan")
    x = np.flatnonzero(finite).astype(np.float64)
    x = (x - x[0]) / (x[-1] - x[0])
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(values[finite], x))


def relion_final_gt_series(relion_final_ft: dict[str, np.ndarray], relion_merged_ft) -> dict[str, np.ndarray]:
    """Return only the RELION final maps available for GT reporting."""
    series = {}
    if "half1" in relion_final_ft and "half2" in relion_final_ft:
        series["relion_half1"] = relion_final_ft["half1"]
        series["relion_half2"] = relion_final_ft["half2"]
    if relion_merged_ft is not None:
        series["relion_merged"] = np.asarray(relion_merged_ft, dtype=np.complex64)
    return series


def final_output_fourier_volumes(result):
    """Return final half maps and RELION-semantic joined reconstruction.

    The final all-data path reconstructs the combined BPref accumulator as
    ``result["mean"]``.  Averaging the two separately regularized half maps is
    not equivalent because Wiener reconstruction is nonlinear in the weights.
    """
    half1 = np.asarray(result["means"][0], dtype=np.complex64).reshape(-1)
    half2 = np.asarray(result["means"][1], dtype=np.complex64).reshape(-1)
    merged = np.asarray(result["mean"], dtype=np.complex64).reshape(-1)
    return half1, half2, merged


def load_initial_fourier_volume(path: str | Path, volume_shape: tuple[int, int, int]) -> np.ndarray:
    """Load one sealed internal Fourier reference without an MRC round-trip."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"initial Fourier reference does not exist: {source}")
    with np.load(source, allow_pickle=False) as payload:
        if "mean_vol_ft" not in payload.files:
            raise ValueError(f"initial Fourier reference is missing mean_vol_ft: {source}")
        volume = np.asarray(payload["mean_vol_ft"])
    expected_size = int(np.prod(volume_shape))
    if volume.size != expected_size:
        raise ValueError(
            f"initial Fourier reference {source} has {volume.size} elements, expected {expected_size}"
        )
    if not np.issubdtype(volume.dtype, np.complexfloating) or not np.isfinite(volume).all():
        raise ValueError(f"initial Fourier reference {source} must be finite complex data")
    return volume.reshape(-1)


def filter_fresh_initial_reference(
    volume_real: np.ndarray,
    *,
    pixel_size: float,
    ini_high_angstrom: float,
) -> np.ndarray:
    """Return RELION's in-memory startup reference before its first E-step.

    ``run_it000_*_class001.mrc`` is a rounded serialization of this state.
    Reloading that MRC changes borderline first-iteration CC winners.  A fresh
    trajectory must instead repeat ``initialLowPassFilterReferences`` and pass
    its binary64 real-space result directly to the initial projector.
    """

    from recovar.em.initial_model.bootstrap_iref import (
        initial_low_pass_filter_references,
    )

    volume_real = np.asarray(volume_real, dtype=np.float64)
    if volume_real.ndim != 3 or len(set(volume_real.shape)) != 1:
        raise ValueError(
            "fresh initial reference must be one cubic real-space volume, "
            f"got {volume_real.shape}",
        )
    if not np.isfinite(volume_real).all():
        raise ValueError("fresh initial reference must contain only finite values")
    if not np.isfinite(pixel_size) or float(pixel_size) <= 0.0:
        raise ValueError("fresh initial reference pixel size must be positive")
    if not np.isfinite(ini_high_angstrom) or float(ini_high_angstrom) <= 0.0:
        raise ValueError("fresh initial reference ini_high must be positive")
    return np.asarray(
        initial_low_pass_filter_references(
            volume_real[None, ...],
            ori_size=int(volume_real.shape[0]),
            pixel_size=float(pixel_size),
            ini_high_ang=float(ini_high_angstrom),
            filter_edgewidth=2.0,
        )[0],
        dtype=np.float64,
    )


def validate_fresh_initial_reference_args(
    *,
    fresh_initial_reference_mrc: str | None,
    start_iteration: int,
    initial_half1_mrc: str | None,
    initial_half1_ft_npz: str | None,
) -> None:
    """Keep the process-resident startup reference confined to fresh runs."""

    if fresh_initial_reference_mrc is None:
        return
    if int(start_iteration) != 0:
        raise ValueError("--fresh-initial-reference-mrc requires --iter 0")
    if initial_half1_mrc is not None or initial_half1_ft_npz is not None:
        raise ValueError(
            "--fresh-initial-reference-mrc is mutually exclusive with initial half references",
        )


def validate_fresh_particle_order_args(
    *,
    fresh_particle_order_seed: int | None,
    preserve_bpref_particle_order: bool,
    start_iteration: int,
    initial_half1_mrc: str | None,
    initial_half2_mrc: str | None,
    initial_half1_ft_npz: str | None,
    initial_half2_ft_npz: str | None,
    final_replay_fields: str | None,
) -> None:
    """Confine RELION's one-time physical order to an unsealed fresh K=1 run."""

    if preserve_bpref_particle_order and fresh_particle_order_seed is None:
        raise ValueError(
            "--diagnostic-preserve-bpref-particle-order requires "
            "--diagnostic-fresh-particle-order-seed",
        )
    if fresh_particle_order_seed is None:
        return
    if int(start_iteration) != 0:
        raise ValueError("--diagnostic-fresh-particle-order-seed requires --iter 0")
    if any(
        value is not None
        for value in (
            initial_half1_mrc,
            initial_half2_mrc,
            initial_half1_ft_npz,
            initial_half2_ft_npz,
        )
    ):
        raise ValueError(
            "--diagnostic-fresh-particle-order-seed cannot be applied to an initial half boundary",
        )
    if final_replay_fields:
        raise ValueError(
            "--diagnostic-fresh-particle-order-seed cannot be combined with --final-replay-fields",
        )


def validate_native_relion_particle_order_args(
    *,
    native_order_seed: int | None,
    fresh_order_seed: int | None,
    start_iteration: int,
) -> None:
    """Confine imported RELION physical order to a non-fresh replay boundary."""

    if native_order_seed is None:
        return
    if fresh_order_seed is not None:
        raise ValueError(
            "native RELION and fresh particle-order diagnostics are mutually exclusive"
        )
    if int(start_iteration) <= 0:
        raise ValueError(
            "--diagnostic-native-relion-particle-order-seed requires --iter greater than 0"
        )


def load_initial_noise_variance(path: str | Path, image_shape: tuple[int, int]) -> np.ndarray:
    """Load one sealed full-pixel noise-variance state without STAR rounding."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"initial noise variance does not exist: {source}")
    variance = np.asarray(np.load(source, allow_pickle=False))
    expected_size = int(np.prod(image_shape))
    if variance.size != expected_size:
        raise ValueError(
            f"initial noise variance {source} has {variance.size} elements, expected {expected_size}"
        )
    if np.iscomplexobj(variance) or not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError(f"initial noise variance {source} must be finite, real, and positive")
    return variance.reshape(-1)


def load_initial_direction_prior(path: str | Path, expected_size: int) -> np.ndarray:
    """Load one sealed direction-probability vector without STAR rounding."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"initial direction prior does not exist: {source}")
    prior = np.asarray(np.load(source, allow_pickle=False))
    if prior.size != expected_size:
        raise ValueError(
            f"initial direction prior {source} has {prior.size} elements, expected {expected_size}"
        )
    if np.iscomplexobj(prior) or not np.isfinite(prior).all() or np.any(prior < 0):
        raise ValueError(f"initial direction prior {source} must be finite, real, and nonnegative")
    if not float(np.sum(prior, dtype=np.float64)) > 0.0:
        raise ValueError(f"initial direction prior {source} must have positive total mass")
    return prior.reshape(-1)


def validate_final_only_replay_args(
    *,
    max_iter: int,
    force_final_after_zero_iterations: bool,
    initial_half1_mrc: str | None,
    initial_half2_mrc: str | None,
    initial_half1_ft_npz: str | None = None,
    initial_half2_ft_npz: str | None = None,
    initial_noise_half1_npy: str | None = None,
    initial_noise_half2_npy: str | None = None,
    initial_direction_prior_half1_npy: str | None = None,
    initial_direction_prior_half2_npy: str | None = None,
) -> None:
    """Validate the diagnostic that enters finalization from saved half maps."""

    if force_final_after_zero_iterations and int(max_iter) != 0:
        raise ValueError("--force-final-after-zero-iterations requires --max_iter 0")
    if (initial_half1_mrc is None) != (initial_half2_mrc is None):
        raise ValueError("--initial-half1-mrc and --initial-half2-mrc must be provided together")
    if (initial_half1_ft_npz is None) != (initial_half2_ft_npz is None):
        raise ValueError("--initial-half1-ft-npz and --initial-half2-ft-npz must be provided together")
    if initial_half1_mrc is not None and initial_half1_ft_npz is not None:
        raise ValueError("initial MRC and Fourier-NPZ reference inputs are mutually exclusive")
    if (initial_noise_half1_npy is None) != (initial_noise_half2_npy is None):
        raise ValueError("--initial-noise-half1-npy and --initial-noise-half2-npy must be provided together")
    if (initial_direction_prior_half1_npy is None) != (initial_direction_prior_half2_npy is None):
        raise ValueError(
            "--initial-direction-prior-half1-npy and --initial-direction-prior-half2-npy "
            "must be provided together"
        )


def initial_scoring_noise_pair(noise_half1, noise_half2, *, continuous_relion_noise_state: bool):
    """Resolve restart-faithful versus uninterrupted RELION scoring noise.

    A new RELION MPI process broadcasts half 1's spectrum to both followers.
    A controlled N-to-N+1 substitution against an uninterrupted trajectory
    instead needs the independently updated spectrum from each numbered half.
    """

    return _relion_mpi_process_start_scoring_noise_pair(
        noise_half1,
        noise_half2,
        split_random_halves=not bool(continuous_relion_noise_state),
    )


def final_only_replay_override(replay_iteration_overrides, *, enabled: bool):
    """Return the explicit final-boundary state for a zero-iteration replay.

    A lone replay slot zero is intentionally ignored by the EM loop's
    automatic final replay because ordinary refinements use that slot for
    cold-start state.  The zero-iteration diagnostic is different: slot zero
    is the requested last-numbered RELION state and must be passed through the
    dedicated final override argument.
    """

    if not enabled:
        return None
    if not replay_iteration_overrides or replay_iteration_overrides[0] is None:
        raise ValueError("final-only replay is missing its RELION boundary state")
    return replay_iteration_overrides[0]


_FINAL_REPLAY_FIELD_GROUPS = {
    "poses": frozenset(
        {
            "previous_best_translations",
            "previous_best_rotations",
            "previous_best_rotation_eulers",
        }
    ),
    "sampling": frozenset(
        {
            "translation_sigma_angstrom",
            "translation_sigma_angstrom_per_half",
        }
    ),
    "normalization": frozenset({"image_corrections", "scale_corrections"}),
    "noise": frozenset({"noise_variance"}),
    "direction_prior": frozenset({"direction_prior"}),
}
_FINAL_REPLAY_FIELD_GROUPS["corrections"] = frozenset().union(
    _FINAL_REPLAY_FIELD_GROUPS["normalization"],
    _FINAL_REPLAY_FIELD_GROUPS["noise"],
    _FINAL_REPLAY_FIELD_GROUPS["direction_prior"],
)
_FINAL_REPLAY_FIELD_GROUPS["all"] = frozenset().union(
    *(_FINAL_REPLAY_FIELD_GROUPS[name] for name in (
        "poses",
        "sampling",
        "normalization",
        "noise",
        "direction_prior",
    ))
)


def select_final_replay_override(replay_iteration_overrides, field_groups: str | None):
    """Select RELION fields substituted only at the final K=1 boundary.

    The source is the last replay slot, which represents the last-numbered
    RELION particle/model state for an ordinary continuation.  Keeping this
    selector here makes the diagnostic independent of the production CLI's
    differently named serialized correction fields.
    """

    requested_groups = [
        item.strip().lower()
        for item in str(field_groups or "").split(",")
        if item.strip()
    ]
    if not requested_groups:
        return None
    unknown = sorted(set(requested_groups) - set(_FINAL_REPLAY_FIELD_GROUPS))
    if unknown:
        raise ValueError(
            "unknown final replay field group(s): "
            f"{','.join(unknown)}; expected one or more of "
            f"{','.join(sorted(_FINAL_REPLAY_FIELD_GROUPS))}"
        )
    if not replay_iteration_overrides or replay_iteration_overrides[-1] is None:
        raise ValueError("final replay field selection is missing the last-numbered RELION state")

    selected_keys = frozenset().union(
        *(_FINAL_REPLAY_FIELD_GROUPS[group] for group in requested_groups)
    )
    source = replay_iteration_overrides[-1]
    missing = sorted(selected_keys - set(source))
    if missing:
        raise ValueError(f"last-numbered RELION state is missing selected fields: {missing}")
    return {key: source[key] for key in sorted(selected_keys)}


def parse_iteration_normalization_factor_overrides(specs):
    """Parse diagnostic per-scoring-iteration normalization overrides."""
    parsed = {}
    for spec in specs:
        try:
            iteration_text, stack_text, factor_text = spec.split(":")
            scoring_iteration = int(iteration_text)
            stack_index = int(stack_text)
            factor = np.float32(factor_text)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(
                "iteration normalization override must be "
                "SCORING_ITER:ZERO_BASED_STACK:FACTOR"
            ) from error
        if scoring_iteration < 1:
            raise ValueError("normalization override scoring iteration must be positive")
        if stack_index < 0:
            raise ValueError("normalization override stack row must be nonnegative")
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("normalization override factor must be finite and positive")
        key = (scoring_iteration, stack_index)
        if key in parsed:
            raise ValueError(
                "duplicate iteration normalization override for "
                f"scoring iteration {scoring_iteration}, stack row {stack_index}"
            )
        parsed[key] = factor
    return parsed


def apply_iteration_normalization_factor_overrides(
    corrections,
    scale_corrections,
    *,
    half_stack_indices,
    scoring_iteration,
    overrides,
):
    """Apply explicit diagnostic factors to copies of half-ordered corrections."""
    corrected = [np.array(values, copy=True) for values in corrections]
    stack_locations = {}
    for half_index, stack_indices in enumerate(half_stack_indices):
        for half_position, stack_index in enumerate(stack_indices):
            stack_index = int(stack_index)
            if stack_index in stack_locations:
                raise ValueError(f"stack row occurs in both particle halves: {stack_index}")
            stack_locations[stack_index] = (half_index, half_position)

    applied = []
    for (override_iteration, stack_index), factor in overrides.items():
        if override_iteration != int(scoring_iteration):
            continue
        if stack_index not in stack_locations:
            raise ValueError(f"normalization-factor override stack row is absent: {stack_index}")
        half_index, half_position = stack_locations[stack_index]
        corrected[half_index][half_position] = (
            factor * np.asarray(scale_corrections[half_index])[half_position]
        )
        applied.append(
            {
                "scoring_iteration": int(scoring_iteration),
                "stack_index": stack_index,
                "half": half_index + 1,
                "half_position": half_position,
                "factor": float(factor),
            }
        )
    return corrected, applied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--relion_dir", required=True)
    parser.add_argument(
        "--relion_run_prefix",
        default="run",
        help="RELION output prefix inside --relion_dir (default: run)",
    )
    parser.add_argument("--data_star", required=True)
    parser.add_argument("--iter", type=int, default=3, help="RELION iteration to start from")
    parser.add_argument("--max_iter", type=int, default=15)
    parser.add_argument(
        "--continuous-relion-noise-state",
        action="store_true",
        help=(
            "Diagnostic N-to-N+1 substitution only: preserve each numbered half's "
            "sigma2_noise as used by an uninterrupted RELION trajectory. The default "
            "emulates a true RELION MPI restart, which broadcasts half-1 noise to both halves."
        ),
    )
    parser.add_argument(
        "--diagnostic-fresh-particle-order-seed",
        type=int,
        default=None,
        help=(
            "Diagnostic exact-state A/B only: reconstruct RELION's one-time fresh "
            "paired half order using this optimizer seed. Ordinary continuation "
            "preserves source order."
        ),
    )
    parser.add_argument(
        "--diagnostic-native-relion-particle-order-seed",
        type=int,
        default=None,
        help=(
            "Diagnostic imported-boundary A/B only: reconstruct the native RELION "
            "run's one-time paired half order using this optimizer seed, and preserve "
            "that order through K=1 BPref. Requires --iter greater than 0."
        ),
    )
    parser.add_argument(
        "--diagnostic-preserve-bpref-particle-order",
        action="store_true",
        help=(
            "Diagnostic exact-state A/B only: preserve the selected physical "
            "particle order through the K=1 sparse BPref accumulation."
        ),
    )
    parser.add_argument(
        "--initial-half1-mrc",
        type=str,
        default=None,
        help="Diagnostic RECOVAR-frame half-1 map replacing the starting RELION map.",
    )
    parser.add_argument(
        "--initial-half2-mrc",
        type=str,
        default=None,
        help="Diagnostic RECOVAR-frame half-2 map replacing the starting RELION map.",
    )
    parser.add_argument(
        "--initial-mrc-projector-direct",
        action="store_true",
        help=(
            "Diagnostic map intervention only: construct first-iteration RELION "
            "Projector::data directly from the loaded --initial-half*-mrc arrays, "
            "without the internal Fourier forward/inverse round trip."
        ),
    )
    parser.add_argument(
        "--fresh-initial-reference-mrc",
        type=str,
        default=None,
        help=(
            "Diagnostic fresh-run RECOVAR-frame reference. Repeats RELION's "
            "initial low-pass in memory and bypasses the lossy run_it000 MRC boundary."
        ),
    )
    parser.add_argument(
        "--initial-half1-ft-npz",
        type=str,
        default=None,
        help=(
            "Diagnostic internal Fourier half-1 reference. The NPZ must contain "
            "mean_vol_ft; bypasses the lossy MRC import round-trip."
        ),
    )
    parser.add_argument(
        "--initial-half2-ft-npz",
        type=str,
        default=None,
        help=(
            "Diagnostic internal Fourier half-2 reference. The NPZ must contain "
            "mean_vol_ft; bypasses the lossy MRC import round-trip."
        ),
    )
    parser.add_argument(
        "--initial-noise-half1-npy",
        type=str,
        default=None,
        help="Diagnostic full-pixel internal half-1 noise variance replacing STAR-derived noise.",
    )
    parser.add_argument(
        "--initial-noise-half2-npy",
        type=str,
        default=None,
        help="Diagnostic full-pixel internal half-2 noise variance replacing STAR-derived noise.",
    )
    parser.add_argument(
        "--initial-direction-prior-half1-npy",
        type=str,
        default=None,
        help="Diagnostic internal half-1 direction prior replacing the serialized model-STAR vector.",
    )
    parser.add_argument(
        "--initial-direction-prior-half2-npy",
        type=str,
        default=None,
        help="Diagnostic internal half-2 direction prior replacing the serialized model-STAR vector.",
    )
    parser.add_argument(
        "--force-final-after-zero-iterations",
        action="store_true",
        help=(
            "Diagnostic only: with --max_iter 0, enter the K=1 final all-data pass "
            "from the supplied/current half maps and the pinned RELION state."
        ),
    )
    parser.add_argument(
        "--final-replay-fields",
        default=None,
        metavar="GROUP[,GROUP...]",
        help=(
            "Diagnostic K=1 final-boundary substitution from the last-numbered "
            "RELION state. Groups: poses, sampling, normalization, noise, "
            "direction_prior, corrections, all. Numbered iterations are unchanged."
        ),
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--save_intermediates_dir", type=str, default=None, help="Directory for manifest NPZ dumps (for replay)"
    )
    parser.add_argument(
        "--image_batch_size",
        type=int,
        default=500,
        help="Images per GPU batch in the score+reconstruct passes. Reduce on 256³ to avoid OOM/IMA.",
    )
    parser.add_argument(
        "--rotation_block_size",
        type=int,
        default=5000,
        help="Rotations per dispatch in the score pass. Reduce on 256³ to bound transient peak working set.",
    )
    parser.add_argument(
        "--image-fourier-backend",
        choices=("host_numpy", "jax_gpu", "relion_cuda"),
        default="host_numpy",
        help=(
            "Fourier preprocessing backend for RELION-masked images. Use "
            "relion_cuda for source-level CUDA operand comparisons."
        ),
    )
    parser.add_argument(
        "--relion-native-lane-softmask-reduction",
        action="store_true",
        help=(
            "Diagnostic only: reproduce RELION's per-lane CUDA soft-mask "
            "background reduction when using --image-fourier-backend relion_cuda."
        ),
    )
    parser.add_argument(
        "--normalization-factor-override",
        action="append",
        default=[],
        metavar="ZERO_BASED_STACK:FACTOR",
        help=(
            "Diagnostic only: replace avg_norm/normcorr for one stack row. "
            "May be repeated; the factor is multiplied by the row's scale correction."
        ),
    )
    parser.add_argument(
        "--normalization-factor-override-at-iteration",
        action="append",
        default=[],
        metavar="SCORING_ITER:ZERO_BASED_STACK:FACTOR",
        help=(
            "Diagnostic only: replace avg_norm/normcorr for one stack row at one "
            "physical scoring iteration. May be repeated; the factor is multiplied "
            "by the row's scale correction."
        ),
    )
    parser.add_argument("--max_healpix_order", type=int, default=8)
    parser.add_argument("--skip_final_iteration", action="store_true", help="Skip the final combined-data Nyquist iter")
    parser.add_argument(
        "--diagnostic-retain-terminal-group-scale-state",
        action="store_true",
        help=(
            "Diagnostic only: retain scale-group IDs and XA/AA sufficient "
            "statistics for a one-iteration --skip_final_iteration replay."
        ),
    )
    parser.add_argument(
        "--force_max_iter_after_convergence",
        action="store_true",
        help=(
            "Continue RECOVAR parity replay until --max_iter even if the RELION-mode "
            "convergence state has already converged. Use only for fixed-length diagnostics."
        ),
    )
    parser.add_argument(
        "--max_particles", type=int, default=None, help="Subsample to at most N particles (N/2 per half)"
    )
    parser.add_argument(
        "--keep_stack_indices",
        type=str,
        default=None,
        help=(
            "Comma/space-separated zero-based particle stack indices to keep. "
            "Use for focused RELION-vs-RECOVAR E-step score dumps."
        ),
    )
    parser.add_argument(
        "--gt_volume",
        type=str,
        default=None,
        help="Optional recovar-frame GT MRC for FSC/correlation checks. Defaults to sibling reference_gt.mrc if present.",
    )
    parser.add_argument(
        "--gt_align",
        action="store_true",
        help="Also compute alignment-aware GT metrics for ab-initio/global-pose ambiguous maps.",
    )
    parser.add_argument(
        "--gt_align_healpix_order",
        type=int,
        default=DEFAULT_GT_ALIGN_HEALPIX_ORDER,
        help="RELION/RECOVAR rotation-grid order used for GT alignment.",
    )
    parser.add_argument(
        "--gt_align_max_shell",
        type=int,
        default=DEFAULT_GT_ALIGN_MAX_SHELL,
        help="Maximum Fourier shell used to score coarse GT alignment.",
    )
    parser.add_argument(
        "--gt_align_no_mirror",
        action="store_true",
        help="Do not test the x-axis mirror handedness ambiguity during GT alignment.",
    )
    parser.add_argument(
        "--gt_align_allow_sign",
        action="store_true",
        help="Allow a global sign flip during GT alignment. Off by default.",
    )
    parser.add_argument(
        "--gt_align_all_series",
        action="store_true",
        help="Compute aligned GT metrics for half maps too; default is merged maps only.",
    )
    parser.add_argument(
        "--force_oversampling",
        type=int,
        default=None,
        help="Override RELION's adaptive oversampling order for debugging ablations.",
    )
    parser.add_argument(
        "--max_significants",
        type=int,
        default=None,
        help="Override RELION's maximum significant poses. Default: read _rlnMaximumSignificantPoses from optimiser.star.",
    )
    parser.add_argument(
        "--local_search_profile",
        choices=["auto", "on", "off"],
        default="auto",
        help="Control exact-local profile collection. 'auto' profiles only when intermediates are enabled.",
    )
    parser.add_argument(
        "--local_search_translation_prior_mode",
        choices=["perturbed", "coarse"],
        default="coarse",
        help="Evaluate local-search translation priors on the perturbed candidate grid or the unperturbed coarse RELION grid.",
    )
    parser.add_argument(
        "--firstiter-cc-mode",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Control RELION firstiter-CC emulation. 'auto' (default) follows the "
            "optimiser STAR command, 'on' forces strict normalized-CC/hard-winner "
            "semantics for --iter 0, and 'off' is the explicit ablation path."
        ),
    )
    parser.add_argument(
        "--first_iteration_score_mode",
        choices=["gaussian", "normalized_cc"],
        default="gaussian",
        help="Diagnostic override for the iter-0 score metric.",
    )
    parser.add_argument(
        "--first_iteration_reconstruction_mode",
        choices=["soft", "hard"],
        default="soft",
        help="Diagnostic override for the iter-0 reconstruction weights.",
    )
    parser.add_argument(
        "--relion_ini_high",
        type=float,
        default=None,
        help="Optional override for RELION --ini_high. Defaults to the optimiser flag value, or 30 A.",
    )
    parser.add_argument(
        "--disable_adjoint_y",
        action="store_true",
        help="Experimental ablation: disable weighted-image adjoint accumulation.",
    )
    parser.add_argument(
        "--disable_adjoint_ctf",
        action="store_true",
        help="Experimental ablation: disable CTF adjoint accumulation.",
    )
    parser.add_argument(
        "--benchmark_ledger_json",
        type=str,
        default=None,
        help="Optional JSON path for a machine-readable benchmark/perf ledger summary.",
    )
    parser.add_argument(
        "--timing_only",
        action="store_true",
        help=(
            "Run refinement and write only the benchmark ledger. Skips "
            "diagnostic volumes, per-particle comparisons, and diff scripts so "
            "wall time reflects refinement rather than audit I/O."
        ),
    )
    parser.add_argument(
        "--compile_log",
        type=str,
        default=None,
        help="Optional log path to scan for JAX compile lines when building the benchmark ledger.",
    )
    parser.add_argument(
        "--jax_cache_dir",
        type=str,
        default=None,
        help="Optional persistent JAX compilation cache directory for cross-process warm starts.",
    )
    args = parser.parse_args()
    iteration_normalization_overrides = parse_iteration_normalization_factor_overrides(
        args.normalization_factor_override_at_iteration
    )
    validate_fresh_particle_order_args(
        fresh_particle_order_seed=args.diagnostic_fresh_particle_order_seed,
        preserve_bpref_particle_order=args.diagnostic_preserve_bpref_particle_order,
        start_iteration=args.iter,
        initial_half1_mrc=args.initial_half1_mrc,
        initial_half2_mrc=args.initial_half2_mrc,
        initial_half1_ft_npz=args.initial_half1_ft_npz,
        initial_half2_ft_npz=args.initial_half2_ft_npz,
        final_replay_fields=args.final_replay_fields,
    )
    validate_native_relion_particle_order_args(
        native_order_seed=args.diagnostic_native_relion_particle_order_seed,
        fresh_order_seed=args.diagnostic_fresh_particle_order_seed,
        start_iteration=args.iter,
    )
    validate_final_only_replay_args(
        max_iter=args.max_iter,
        force_final_after_zero_iterations=args.force_final_after_zero_iterations,
        initial_half1_mrc=args.initial_half1_mrc,
        initial_half2_mrc=args.initial_half2_mrc,
        initial_half1_ft_npz=args.initial_half1_ft_npz,
        initial_half2_ft_npz=args.initial_half2_ft_npz,
        initial_noise_half1_npy=args.initial_noise_half1_npy,
        initial_noise_half2_npy=args.initial_noise_half2_npy,
        initial_direction_prior_half1_npy=args.initial_direction_prior_half1_npy,
        initial_direction_prior_half2_npy=args.initial_direction_prior_half2_npy,
    )
    validate_fresh_initial_reference_args(
        fresh_initial_reference_mrc=args.fresh_initial_reference_mrc,
        start_iteration=args.iter,
        initial_half1_mrc=args.initial_half1_mrc,
        initial_half1_ft_npz=args.initial_half1_ft_npz,
    )

    _print_provenance_banner_and_assert_parity_ancestors()

    if args.jax_cache_dir:
        os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", args.jax_cache_dir)
        os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
        os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
        os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")

    import jax
    import jax.numpy as jnp
    import jaxlib
    import starfile

    from recovar import utils
    from recovar.core import fourier_transform_utils as ftu
    from recovar.data_io.cryoem_dataset import load_dataset
    from recovar.em.dense_single_volume.iteration_loop import refine_single_volume
    from recovar.em.sampling import read_relion_sampling_metadata
    from recovar.output.output import save_volume
    from recovar.reconstruction import noise as recon_noise
    from recovar.reconstruction import regularization
    from recovar.utils import helpers

    def _rotation_matrices_from_eulers_deg(eulers_deg):
        return utils.R_from_relion(np.asarray(eulers_deg, dtype=np.float64))

    def _angular_distance_from_dots(dot_vals):
        return np.rad2deg(np.arccos(np.clip(np.asarray(dot_vals, dtype=np.float64), -1.0, 1.0)))

    def _angular_error_deg_from_rotations(lhs_rot, rhs_rot):
        lhs_rot = np.asarray(lhs_rot, dtype=np.float64)
        rhs_rot = np.asarray(rhs_rot, dtype=np.float64)
        rdiff = np.einsum("nij,njk->nik", np.transpose(lhs_rot, (0, 2, 1)), rhs_rot)
        traces = np.trace(rdiff, axis1=1, axis2=2)
        return _angular_distance_from_dots((traces - 1.0) / 2.0)

    def _normalize_rows(vectors):
        vectors = np.asarray(vectors, dtype=np.float64)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms > 1e-12, norms, 1.0)
        return vectors / norms

    def _view_direction_error_deg_from_rotations(lhs_rot, rhs_rot):
        lhs_view = _normalize_rows(np.asarray(lhs_rot, dtype=np.float64)[:, 2, :])
        rhs_view = _normalize_rows(np.asarray(rhs_rot, dtype=np.float64)[:, 2, :])
        return _angular_distance_from_dots(np.sum(lhs_view * rhs_view, axis=1))

    def _inplane_error_deg_from_rotations(lhs_rot, rhs_rot):
        lhs_rot = np.asarray(lhs_rot, dtype=np.float64)
        rhs_rot = np.asarray(rhs_rot, dtype=np.float64)
        rhs_view = _normalize_rows(rhs_rot[:, 2, :])

        lhs_x = lhs_rot[:, 0, :]
        rhs_x = rhs_rot[:, 0, :]
        lhs_x = lhs_x - np.sum(lhs_x * rhs_view, axis=1, keepdims=True) * rhs_view
        rhs_x = rhs_x - np.sum(rhs_x * rhs_view, axis=1, keepdims=True) * rhs_view
        lhs_x = _normalize_rows(lhs_x)
        rhs_x = _normalize_rows(rhs_x)

        cross = np.cross(rhs_x, lhs_x)
        signed = np.rad2deg(
            np.arctan2(
                np.sum(rhs_view * cross, axis=1),
                np.sum(rhs_x * lhs_x, axis=1),
            )
        )
        return np.abs(signed)

    def _angular_error_deg_from_eulers(lhs_eulers_deg, rhs_eulers_deg):
        return _angular_error_deg_from_rotations(
            _rotation_matrices_from_eulers_deg(lhs_eulers_deg),
            _rotation_matrices_from_eulers_deg(rhs_eulers_deg),
        )

    def _view_direction_error_deg_from_eulers(lhs_eulers_deg, rhs_eulers_deg):
        return _view_direction_error_deg_from_rotations(
            _rotation_matrices_from_eulers_deg(lhs_eulers_deg),
            _rotation_matrices_from_eulers_deg(rhs_eulers_deg),
        )

    def _inplane_error_deg_from_eulers(lhs_eulers_deg, rhs_eulers_deg):
        return _inplane_error_deg_from_rotations(
            _rotation_matrices_from_eulers_deg(lhs_eulers_deg),
            _rotation_matrices_from_eulers_deg(rhs_eulers_deg),
        )

    def _rotations_in_gt_frame_from_relion_eulers(eulers_deg, transpose_relion_convention):
        rot = _rotation_matrices_from_eulers_deg(eulers_deg)
        if transpose_relion_convention:
            rot = np.transpose(rot, (0, 2, 1))
        return rot

    def _format_error_summary(values, unit, thresholds):
        values = np.asarray(values, dtype=np.float64)
        percentiles = np.percentile(values, [90, 95, 99])
        frac_terms = [f"<= {thr:g}{unit}: {(100.0 * np.mean(values <= thr)):.1f}%" for thr in thresholds]
        return (
            f"mean={values.mean():.4f}{unit}, "
            f"median={np.median(values):.4f}{unit}, "
            f"p90={percentiles[0]:.4f}{unit}, "
            f"p95={percentiles[1]:.4f}{unit}, "
            f"p99={percentiles[2]:.4f}{unit}, "
            f"max={values.max():.4f}{unit}; " + ", ".join(frac_terms)
        )

    def _first_shell_below_threshold(fsc_values, threshold):
        fsc_values = np.asarray(fsc_values, dtype=np.float64)
        below = np.where(fsc_values < float(threshold))[0]
        return int(below[0]) if below.size else None

    def _shell_to_resolution_angstrom(shell_idx):
        if shell_idx is None or shell_idx <= 0:
            return np.nan
        return float(N * pixel_size) / float(shell_idx)

    def _compute_fsc_vs_gt(volume_ft_flat, gt_ft_flat):
        return np.asarray(
            regularization.get_fsc_gpu(
                jnp.asarray(volume_ft_flat),
                jnp.asarray(gt_ft_flat),
                (N, N, N),
            ),
            dtype=np.float64,
        )

    relion_dir = Path(args.relion_dir)
    run_prefix = args.relion_run_prefix
    iteration = args.iter
    prefix = str(relion_dir / f"{run_prefix}_it{iteration:03d}")

    # ---- Load RELION state ----
    model_h1 = starfile.read(f"{prefix}_half1_model.star")
    model_h2 = starfile.read(f"{prefix}_half2_model.star")
    control_model_h1 = model_h1
    control_model_path = (
        relion_dir / f"{run_prefix}_it{replay_control_relion_iteration(iteration, 0):03d}_half1_model.star"
    )
    if control_model_path.exists():
        control_model_h1 = starfile.read(control_model_path)
    N = int(model_h1["model_general"]["rlnOriginalImageSize"])
    current_size = int(control_model_h1["model_general"]["rlnCurrentImageSize"])
    pixel_size = float(model_h1["model_general"]["rlnPixelSize"])

    sigma2_h1 = _read_relion_single_optics_sigma2_noise(
        model_h1,
        context=f"RELION iteration {iteration} half 1",
    )
    sigma2_h2 = _read_relion_single_optics_sigma2_noise(
        model_h2,
        context=f"RELION iteration {iteration} half 2",
    )
    if sigma2_h1 is None or sigma2_h2 is None:
        raise ValueError(f"RELION iteration {iteration} model is missing rlnSigma2Noise")
    class1 = model_h1["model_class_1"]
    # Prefer rlnReferenceTau2 (signal power, EMDL_MLMODEL_TAU2_REF) which is
    # what RELION's BackProjector::reconstruct uses for the Wiener prior.
    # rlnReferenceSigma2 (noise power, EMDL_MLMODEL_SIGMA2_REF) would over-
    # regularise when used as the prior; safe at iter≥3 K=1 because the
    # production loop recomputes mean_signal_variance from M-step weights
    # via compute_relion_tau2_from_weights, but matches RELION's intent.
    tau2_col = "rlnReferenceTau2" if "rlnReferenceTau2" in class1 else "rlnReferenceSigma2"
    tau2 = np.array(class1[tau2_col])
    fsc_col = "rlnGoldStandardFsc" if "rlnGoldStandardFsc" in class1 else "rlnFourierShellCorrelationCorrected"
    fsc = np.array(class1[fsc_col])

    opt_text = (relion_dir / f"{run_prefix}_it{iteration:03d}_optimiser.star").read_text()
    m_pd = re.search(r"_rlnParticleDiameter\s+(\S+)", opt_text)
    particle_diameter = float(m_pd.group(1)) if m_pd else 544.0
    m_os = re.search(r"_rlnAdaptiveOversampleOrder\s+(\d+)", opt_text)
    oversampling = int(m_os.group(1)) if m_os else 0
    m_ms = re.search(r"_rlnMaximumSignificantPoses\s+(-?\d+)", opt_text)
    max_significants = int(m_ms.group(1)) if m_ms else 500
    optimiser_cli_flags = parse_relion_optimiser_cli_flags(opt_text)
    oracle_firstiter_cc = bool(optimiser_cli_flags["do_firstiter_cc"])
    do_firstiter_cc = resolve_firstiter_cc_mode(
        args.firstiter_cc_mode,
        oracle_enabled=oracle_firstiter_cc,
        start_iteration=args.iter,
    )
    relion_ini_high = (
        float(args.relion_ini_high)
        if args.relion_ini_high is not None
        else float(optimiser_cli_flags["ini_high_angstrom"])
        if optimiser_cli_flags["ini_high_angstrom"] is not None
        else 30.0
    )

    sampling_meta = read_relion_sampling_metadata(relion_dir / f"{run_prefix}_it{iteration:03d}_sampling.star")
    hp_order = int(sampling_meta["healpix_order"])
    offset_range = float(sampling_meta["offset_range"])
    offset_step = float(sampling_meta["offset_step"])

    # Scheduling ave_Pmax comes from RELION's model state.  Keep the particle
    # column separate: it is useful for per-particle posterior diagnostics but
    # is not the scalar consumed by current-size growth.
    relion_data = starfile.read(f"{prefix}_data.star")
    relion_df = relion_data["particles"] if isinstance(relion_data, dict) else relion_data
    relion_pmax = _read_relion_pmax_column(relion_df)
    ave_Pmax = _read_relion_scheduling_average_pmax(
        model_h1,
        relion_iteration=iteration,
    )
    if relion_pmax is None:
        print(
            "  Initial RELION data STAR has no rlnMaxValueProbDistribution; "
            "per-particle Pmax comparison is unavailable",
        )

    # has_high_fsc_at_limit (sticky flag)
    has_high_fsc_at_limit = False
    for it in range(1, iteration + 1):
        try:
            m = starfile.read(str(relion_dir / f"{run_prefix}_it{it:03d}_half1_model.star"))
            fc = np.array(m["model_class_1"][fsc_col])
            oc = (relion_dir / f"{run_prefix}_it{it:03d}_optimiser.star").read_text()
            cs_it = (
                int(re.search(r"_rlnCurrentImageSize\s+(\d+)", oc).group(1))
                if re.search(r"_rlnCurrentImageSize", oc)
                else None
            )
            if cs_it is None:
                mc = starfile.read(str(relion_dir / f"{run_prefix}_it{it:03d}_half1_model.star"))
                cs_it = int(mc["model_general"]["rlnCurrentImageSize"])
            shell_at_limit = cs_it // 2 - 1
            if shell_at_limit < len(fc) and fc[shell_at_limit] > 0.2:
                has_high_fsc_at_limit = True
        except Exception:
            pass

    print(f"RELION state: N={N}, hp={hp_order}, os={oversampling}, cs={current_size}")
    print(f"  pixel_size={pixel_size}, particle_diameter={particle_diameter}")
    print(
        f"  ave_Pmax={ave_Pmax:.4f}, has_high_fsc_at_limit={has_high_fsc_at_limit}, max_significants={max_significants}"
    )
    print(
        "  RELION firstiter_cc: "
        f"oracle={oracle_firstiter_cc}, mode={args.firstiter_cc_mode}, effective={do_firstiter_cc}; "
        f"ini_high={relion_ini_high}"
    )
    if args.force_oversampling is not None:
        print(f"  Oversampling override: {oversampling} -> {args.force_oversampling}")
        oversampling = int(args.force_oversampling)
    if args.max_significants is not None:
        print(f"  Max significants override: {max_significants} -> {args.max_significants}")
        max_significants = int(args.max_significants)

    # ---- Init volumes ----
    # RELION FFT normalization: F_relion = FFT(img)/N^d, so sigma2/tau2 from
    # model.star are in RELION's convention.  recovar uses unnormalized FFT,
    # so power spectra scale by N^4.
    n4 = N**4
    noise_variance_h1 = jnp.asarray(recon_noise.make_radial_noise(sigma2_h1 * n4, (N, N)))
    noise_variance_h2 = jnp.asarray(recon_noise.make_radial_noise(sigma2_h2 * n4, (N, N)))
    if args.initial_noise_half1_npy is not None:
        noise_variance_h1 = jnp.asarray(
            load_initial_noise_variance(args.initial_noise_half1_npy, (N, N))
        )
        noise_variance_h2 = jnp.asarray(
            load_initial_noise_variance(args.initial_noise_half2_npy, (N, N))
        )
        print(
            "  Diagnostic initial internal noise variance: "
            f"half1={args.initial_noise_half1_npy}, half2={args.initial_noise_half2_npy}"
        )
    process_start_noise = initial_scoring_noise_pair(
        noise_variance_h1.reshape(-1),
        noise_variance_h2.reshape(-1),
        continuous_relion_noise_state=args.continuous_relion_noise_state,
    )
    noise_variance = jnp.stack(process_start_noise, axis=0)
    print(
        "  initial scoring noise: "
        + (
            "uninterrupted numbered half-specific state"
            if args.continuous_relion_noise_state
            else "RELION MPI process-start half-1 broadcast"
        )
    )
    mean_variance = jnp.asarray(utils.make_radial_image(tau2 * n4, (N, N, N), extend_last_frequency=True))

    # Volume: get_dft3(vol_real) produces the unnormalized centered DFT.
    # This matches the internal convention expected by the refinement code.
    model_reference_path = Path(f"{prefix}_half1_class001.mrc")
    relion_model_pixel_size = read_relion_model_pixel_size(model_reference_path)
    relion_optics_image_sizes, relion_optics_pixel_sizes = read_relion_optics_image_geometry(
        args.data_star
    )
    print(
        "  RELION model pixel size: "
        f"{relion_model_pixel_size:.9g} A/px from {model_reference_path}"
    )
    print(
        "  RELION optics image geometry: "
        f"sizes={relion_optics_image_sizes.tolist()} "
        f"pixel_sizes={relion_optics_pixel_sizes.tolist()} from {args.data_star}"
    )
    initial_reference_real_for_projector = None
    if args.fresh_initial_reference_mrc is not None:
        unfiltered_real = helpers.load_mrc(args.fresh_initial_reference_mrc)
        filtered_real = filter_fresh_initial_reference(
            unfiltered_real,
            pixel_size=pixel_size,
            ini_high_angstrom=relion_ini_high,
        )
        initial_reference_real_for_projector = [filtered_real, filtered_real]
        filtered_for_fourier = filtered_real.astype(np.float32, copy=False)
        vol_ft = np.asarray(ftu.get_dft3(jnp.asarray(filtered_for_fourier))).reshape(-1)
        vol_ft_h1 = vol_ft
        vol_ft_h2 = vol_ft
        print(
            "  Fresh process-resident initial reference: "
            f"source={args.fresh_initial_reference_mrc}, ini_high={relion_ini_high:.2f} A, "
            "fmask_edge=2 shells"
        )
    elif args.initial_half1_ft_npz is not None:
        vol_ft_h1 = load_initial_fourier_volume(args.initial_half1_ft_npz, (N, N, N))
        vol_ft_h2 = load_initial_fourier_volume(args.initial_half2_ft_npz, (N, N, N))
        print(
            "  Diagnostic initial internal Fourier references: "
            f"half1={args.initial_half1_ft_npz}, half2={args.initial_half2_ft_npz}"
        )
    elif args.initial_half1_mrc is not None:
        vol_h1 = helpers.load_mrc(args.initial_half1_mrc)
        vol_h2 = helpers.load_mrc(args.initial_half2_mrc)
        print(
            "  Diagnostic initial half maps (RECOVAR frame): "
            f"half1={args.initial_half1_mrc}, half2={args.initial_half2_mrc}"
        )
        vol_ft_h1 = np.array(ftu.get_dft3(jnp.array(vol_h1))).reshape(-1)
        vol_ft_h2 = np.array(ftu.get_dft3(jnp.array(vol_h2))).reshape(-1)
        if args.initial_mrc_projector_direct:
            initial_reference_real_for_projector = [
                np.asarray(vol_h1, dtype=np.float64),
                np.asarray(vol_h2, dtype=np.float64),
            ]
            print(
                "  Diagnostic initial MRC projector: direct process-resident real maps "
                "(no Fourier forward/inverse round trip)"
            )
    else:
        vol_h1 = helpers.load_relion_volume(f"{prefix}_half1_class001.mrc")
        vol_h2 = helpers.load_relion_volume(f"{prefix}_half2_class001.mrc")
        vol_ft_h1 = np.array(ftu.get_dft3(jnp.array(vol_h1))).reshape(-1)
        vol_ft_h2 = np.array(ftu.get_dft3(jnp.array(vol_h2))).reshape(-1)

    # ---- Dataset + half-set split ----
    ds = load_dataset(args.data_star)
    if args.relion_native_lane_softmask_reduction:
        if args.image_fourier_backend != "relion_cuda":
            raise ValueError(
                "--relion-native-lane-softmask-reduction requires "
                "--image-fourier-backend relion_cuda"
            )
        backend = getattr(getattr(ds, "image_source", None), "backend", None)
        if backend is None or not hasattr(backend, "set_relion_native_lane_reduction"):
            raise ValueError("Dataset backend does not support native-lane soft-mask reduction")
        backend.set_relion_native_lane_reduction(True)
        print("  RELION native-lane soft-mask reduction: enabled")
    relion_subsets = np.array(relion_df["rlnRandomSubset"])
    relion_names = list(relion_df["rlnImageName"])
    our_particles = starfile.read(args.data_star)
    our_particles = our_particles["particles"] if isinstance(our_particles, dict) else our_particles
    our_names = list(our_particles["rlnImageName"])

    def _idx(name):
        return stack_index_from_image_name(name)

    relion_idx_map = {_idx(relion_names[i]): relion_subsets[i] for i in range(len(relion_names))}
    our_subsets = np.array([relion_idx_map.get(_idx(n), 0) for n in our_names])
    selected_order_seed = (
        args.diagnostic_fresh_particle_order_seed
        if args.diagnostic_fresh_particle_order_seed is not None
        else args.diagnostic_native_relion_particle_order_seed
    )
    if selected_order_seed is not None:
        if args.keep_stack_indices or args.max_particles is not None:
            raise ValueError(
                "particle-order diagnostics require the complete particle table"
            )
        relion_optics = (
            np.asarray(relion_df["rlnOpticsGroup"], dtype=np.int64)
            if "rlnOpticsGroup" in relion_df.columns
            else None
        )
        relion_half_orders = particle_half_indices(
            relion_subsets,
            fresh_order_seed=selected_order_seed,
            optics_group_ids=relion_optics,
        )
        half1_indices, half2_indices = map_relion_half_orders_to_dataset_rows(
            our_names,
            relion_names,
            relion_half_orders,
        )
        order_scope = (
            "fresh"
            if args.diagnostic_fresh_particle_order_seed is not None
            else "imported native RELION boundary"
        )
        print(
            f"  Diagnostic reconstructed {order_scope} particle order: "
            f"optimizer_seed={selected_order_seed}, effective_seed={selected_order_seed + 1}"
        )
    else:
        half1_indices = None
        half2_indices = None

    # Keep exact known-bad particles when debugging per-image score parity.
    if args.keep_stack_indices:
        keep_indices = set()
        for token in args.keep_stack_indices.replace(",", " ").split():
            token = token.strip()
            if token:
                keep_indices.add(int(token))
        keep_mask = np.zeros_like(our_subsets, dtype=bool)
        observed = set()
        for i, name in enumerate(our_names):
            stack_idx = stack_index_from_image_name(name)
            if stack_idx in keep_indices:
                keep_mask[i] = True
                observed.add(stack_idx)
        our_subsets[~keep_mask] = 0
        print(
            "  Focused particle selection: "
            f"requested={sorted(keep_indices)}, kept={int(np.sum(keep_mask))}, "
            f"half1={int(np.sum(our_subsets == 1))}, half2={int(np.sum(our_subsets == 2))}, "
            f"missing={sorted(keep_indices - observed)}"
        )

    # Subsample if requested (for fast debugging)
    if args.max_particles is not None:
        rng = np.random.RandomState(42)
        h1_idx = np.where(our_subsets == 1)[0]
        h2_idx = np.where(our_subsets == 2)[0]
        n_per_half = args.max_particles // 2
        if n_per_half < len(h1_idx):
            drop_h1 = rng.choice(h1_idx, size=len(h1_idx) - n_per_half, replace=False)
            our_subsets[drop_h1] = 0
        if n_per_half < len(h2_idx):
            drop_h2 = rng.choice(h2_idx, size=len(h2_idx) - n_per_half, replace=False)
            our_subsets[drop_h2] = 0
        print(f"  Subsampled to max_particles={args.max_particles}")

    if half1_indices is None or half2_indices is None:
        half1_indices, half2_indices = particle_half_indices(our_subsets)

    ds_half1 = ds.subset(half1_indices)
    ds_half2 = ds.subset(half2_indices)
    print(f"  Half-sets: {len(half1_indices)} + {len(half2_indices)}")

    # ---- Image corrections (RELION parity: normcorr + scale) ----
    # RELION: img *= avg_norm_correction / normcorr  (ml_optimiser.cpp:6240)
    # then   Frefctf *= scale                        (ml_optimiser.cpp:7298)
    normcorr = np.array(relion_df["rlnNormCorrection"], dtype=np.float64)
    general_h1 = model_h1["model_general"]
    general_h2 = model_h2["model_general"]
    avg_norm_h1 = float(
        general_h1["rlnNormCorrectionAverage"]
        if isinstance(general_h1, dict)
        else general_h1["rlnNormCorrectionAverage"].iloc[0]
    )
    avg_norm_h2 = float(
        general_h2["rlnNormCorrectionAverage"]
        if isinstance(general_h2, dict)
        else general_h2["rlnNormCorrectionAverage"].iloc[0]
    )
    groups_h1 = model_h1.get("model_groups", None)
    groups_h2 = model_h2.get("model_groups", None)
    scale_h1 = (
        np.array(groups_h1["rlnGroupScaleCorrection"], dtype=np.float64)
        if groups_h1 is not None and "rlnGroupScaleCorrection" in groups_h1.columns
        else np.array([1.0])
    )
    scale_h2 = (
        np.array(groups_h2["rlnGroupScaleCorrection"], dtype=np.float64)
        if groups_h2 is not None and "rlnGroupScaleCorrection" in groups_h2.columns
        else np.array([1.0])
    )
    group_numbers = (
        np.array(relion_df["rlnGroupNumber"], dtype=int)
        if "rlnGroupNumber" in relion_df.columns
        else np.ones(len(relion_df), dtype=int)
    )
    pp_scale_h1 = scale_h1[np.clip(group_numbers - 1, 0, len(scale_h1) - 1)]
    pp_scale_h2 = scale_h2[np.clip(group_numbers - 1, 0, len(scale_h2) - 1)]

    combined_h1 = (avg_norm_h1 / normcorr) * pp_scale_h1
    combined_h2 = (avg_norm_h2 / normcorr) * pp_scale_h2

    # Map to dataset ordering per half-set
    relion_idx_to_pos = {_idx(relion_names[i]): i for i in range(len(relion_names))}
    half1_our_idx = [_idx(our_names[i]) for i in half1_indices]
    half2_our_idx = [_idx(our_names[i]) for i in half2_indices]
    group_ids_h1, group_count = map_relion_scale_groups_to_half_order(
        group_numbers,
        relion_idx_to_pos,
        half1_our_idx,
    )
    group_ids_h2, group_count_h2 = map_relion_scale_groups_to_half_order(
        group_numbers,
        relion_idx_to_pos,
        half2_our_idx,
    )
    if group_count_h2 != group_count:
        raise RuntimeError("RELION scale-group axis differs between half mappings")
    print(
        "  RELION scale groups: "
        f"full_axis={group_count}, half1_unique={np.unique(group_ids_h1).size}, "
        f"half2_unique={np.unique(group_ids_h2).size}"
    )
    corr_h1 = np.array([combined_h1[relion_idx_to_pos[idx]] for idx in half1_our_idx], dtype=np.float32)
    corr_h2 = np.array([combined_h2[relion_idx_to_pos[idx]] for idx in half2_our_idx], dtype=np.float32)
    scale_corr_h1 = np.array([pp_scale_h1[relion_idx_to_pos[idx]] for idx in half1_our_idx], dtype=np.float32)
    scale_corr_h2 = np.array([pp_scale_h2[relion_idx_to_pos[idx]] for idx in half2_our_idx], dtype=np.float32)
    for override in args.normalization_factor_override:
        try:
            stack_text, factor_text = override.split(":", maxsplit=1)
            stack_index = int(stack_text)
            normalization_factor = np.float32(factor_text)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "--normalization-factor-override must be ZERO_BASED_STACK:FACTOR"
            ) from error
        if not np.isfinite(normalization_factor) or normalization_factor <= 0:
            raise ValueError("normalization-factor override must be finite and positive")
        if stack_index in half1_our_idx:
            half_position = half1_our_idx.index(stack_index)
            corr_h1[half_position] = normalization_factor * scale_corr_h1[half_position]
            half_number = 1
        elif stack_index in half2_our_idx:
            half_position = half2_our_idx.index(stack_index)
            corr_h2[half_position] = normalization_factor * scale_corr_h2[half_position]
            half_number = 2
        else:
            raise ValueError(f"normalization-factor override stack row is absent: {stack_index}")
        print(
            "  Diagnostic normalization-factor override: "
            f"stack={stack_index} half={half_number} factor={float(normalization_factor):.9g}"
        )
    print(
        "  Image corrections: "
        f"avg_norm_h1={avg_norm_h1:.6f}, avg_norm_h2={avg_norm_h2:.6f}, "
        f"corr_h1 mean={corr_h1.mean():.4f}, corr_h2 mean={corr_h2.mean():.4f}"
    )

    # ---- Previous best translations (RELION parity: pre-centering) ----
    # RELION pre-centers images by old_offset before scoring
    if "rlnOriginXAngst" in relion_df.columns:
        offsets_x = np.array(relion_df["rlnOriginXAngst"], dtype=np.float64) / pixel_size
        offsets_y = np.array(relion_df["rlnOriginYAngst"], dtype=np.float64) / pixel_size
        offsets = np.stack([offsets_x, offsets_y], axis=1)
        trans_h1 = np.array([offsets[relion_idx_to_pos[idx]] for idx in half1_our_idx], dtype=np.float32)
        trans_h2 = np.array([offsets[relion_idx_to_pos[idx]] for idx in half2_our_idx], dtype=np.float32)
        print(
            f"  Pre-centering offsets: h1 mean_abs={np.abs(trans_h1).mean():.3f} px, h2 mean_abs={np.abs(trans_h2).mean():.3f} px"
        )
    else:
        trans_h1 = None
        trans_h2 = None

    angle_cols = ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]
    euler_h1 = None
    euler_h2 = None
    if all(col in relion_df.columns for col in angle_cols):
        eulers = np.stack([np.array(relion_df[col], dtype=np.float64) for col in angle_cols], axis=1)
        euler_h1 = np.array([eulers[relion_idx_to_pos[idx]] for idx in half1_our_idx], dtype=np.float32)
        euler_h2 = np.array([eulers[relion_idx_to_pos[idx]] for idx in half2_our_idx], dtype=np.float32)
        print(f"  Previous best eulers: h1={euler_h1.shape[0]} particles, h2={euler_h2.shape[0]} particles")
    else:
        print("  Previous best eulers: None (angle columns not found)")

    # ---- Sigma offset from model star ----
    # RELION scores iteration N+1 with the model state written at iteration N.
    # The next model file is useful for replaying control outputs such as the
    # bootstrapped current image size, but its sigma offset is the result of
    # the E/M-step we are trying to reproduce, not an input to it.
    def _model_general_scalar(general, key):
        return float(general[key] if isinstance(general, dict) else general[key].iloc[0])

    sigma_offset_angst_per_half = [
        _model_general_scalar(model_h1["model_general"], "rlnSigmaOffsetsAngst"),
        _model_general_scalar(model_h2["model_general"], "rlnSigmaOffsetsAngst"),
    ]
    sigma_offset_angst = float(np.mean(sigma_offset_angst_per_half))
    print(
        "  sigma_offset = "
        f"half1 {sigma_offset_angst_per_half[0]:.4f} A, "
        f"half2 {sigma_offset_angst_per_half[1]:.4f} A, mean {sigma_offset_angst:.4f} A"
    )

    # ---- Direction prior from model star (RELION's pdf_orientation) ----
    pdf_orient_key = "model_pdf_orient_class_1"
    if pdf_orient_key in model_h1 and pdf_orient_key in model_h2:
        direction_prior = [
            np.array(model_h1[pdf_orient_key]["rlnOrientationDistribution"], dtype=np.float32),
            np.array(model_h2[pdf_orient_key]["rlnOrientationDistribution"], dtype=np.float32),
        ]
        print(
            "  direction_prior: "
            f"h1 {direction_prior[0].shape[0]} directions range=[{direction_prior[0].min():.6f}, {direction_prior[0].max():.6f}] zeros={int(np.sum(direction_prior[0] == 0))}; "
            f"h2 {direction_prior[1].shape[0]} directions range=[{direction_prior[1].min():.6f}, {direction_prior[1].max():.6f}] zeros={int(np.sum(direction_prior[1] == 0))}"
        )
    else:
        direction_prior = None
        print("  direction_prior: None (not found in model star)")

    diagnostic_direction_prior_override = None
    if args.initial_direction_prior_half1_npy is not None:
        if direction_prior is None:
            raise ValueError("diagnostic direction-prior override requires model-STAR direction priors")
        diagnostic_direction_prior_override = [
            load_initial_direction_prior(
                args.initial_direction_prior_half1_npy,
                direction_prior[0].size,
            ),
            load_initial_direction_prior(
                args.initial_direction_prior_half2_npy,
                direction_prior[1].size,
            ),
        ]
        direction_prior = diagnostic_direction_prior_override
        print(
            "  Diagnostic initial internal direction prior: "
            f"half1={args.initial_direction_prior_half1_npy}, "
            f"half2={args.initial_direction_prior_half2_npy}"
        )

    def _load_relion_iteration_override(
        previous_relion_iteration,
        control_relion_iteration,
        *,
        process_start=False,
    ):
        iter_prefix = relion_dir / f"{run_prefix}_it{previous_relion_iteration:03d}"
        model_h1_iter = starfile.read(f"{iter_prefix}_half1_model.star")
        model_h2_iter = starfile.read(f"{iter_prefix}_half2_model.star")
        relion_iter_data = starfile.read(f"{iter_prefix}_data.star")
        relion_iter_df = relion_iter_data["particles"] if isinstance(relion_iter_data, dict) else relion_iter_data
        relion_iter_names = list(relion_iter_df["rlnImageName"])
        relion_iter_idx_to_pos = {_idx(relion_iter_names[i]): i for i in range(len(relion_iter_names))}
        general_h1_iter = model_h1_iter["model_general"]
        general_h2_iter = model_h2_iter["model_general"]
        avg_norm_h1_iter = float(
            general_h1_iter["rlnNormCorrectionAverage"]
            if isinstance(general_h1_iter, dict)
            else general_h1_iter["rlnNormCorrectionAverage"].iloc[0]
        )
        avg_norm_h2_iter = float(
            general_h2_iter["rlnNormCorrectionAverage"]
            if isinstance(general_h2_iter, dict)
            else general_h2_iter["rlnNormCorrectionAverage"].iloc[0]
        )
        sigma_offset_iter_per_half = [
            _model_general_scalar(general_h1_iter, "rlnSigmaOffsetsAngst"),
            _model_general_scalar(general_h2_iter, "rlnSigmaOffsetsAngst"),
        ]
        sigma_offset_iter = float(np.mean(sigma_offset_iter_per_half))
        sigma2_h1_iter = _read_relion_single_optics_sigma2_noise(
            model_h1_iter,
            context=f"RELION iteration {previous_relion_iteration} half 1",
        )
        sigma2_h2_iter = _read_relion_single_optics_sigma2_noise(
            model_h2_iter,
            context=f"RELION iteration {previous_relion_iteration} half 2",
        )
        if sigma2_h1_iter is None or sigma2_h2_iter is None:
            raise ValueError(
                f"RELION iteration {previous_relion_iteration} model is missing rlnSigma2Noise"
            )
        noise_pair_iter = _relion_mpi_process_start_scoring_noise_pair(
            jnp.asarray(recon_noise.make_radial_noise(sigma2_h1_iter * n4, (N, N))).reshape(-1),
            jnp.asarray(recon_noise.make_radial_noise(sigma2_h2_iter * n4, (N, N))).reshape(-1),
            split_random_halves=process_start,
        )
        noise_variance_iter = jnp.stack(noise_pair_iter, axis=0)

        normcorr_iter = np.array(relion_iter_df["rlnNormCorrection"], dtype=np.float64)
        groups_h1_iter = model_h1_iter.get("model_groups", None)
        groups_h2_iter = model_h2_iter.get("model_groups", None)
        scale_h1_iter = (
            np.array(groups_h1_iter["rlnGroupScaleCorrection"], dtype=np.float64)
            if groups_h1_iter is not None and "rlnGroupScaleCorrection" in groups_h1_iter.columns
            else np.array([1.0])
        )
        scale_h2_iter = (
            np.array(groups_h2_iter["rlnGroupScaleCorrection"], dtype=np.float64)
            if groups_h2_iter is not None and "rlnGroupScaleCorrection" in groups_h2_iter.columns
            else np.array([1.0])
        )
        iter_group_numbers = (
            np.array(relion_iter_df["rlnGroupNumber"], dtype=int)
            if "rlnGroupNumber" in relion_iter_df.columns
            else np.ones(len(relion_iter_df), dtype=int)
        )
        pp_scale_h1_iter = scale_h1_iter[np.clip(iter_group_numbers - 1, 0, len(scale_h1_iter) - 1)]
        pp_scale_h2_iter = scale_h2_iter[np.clip(iter_group_numbers - 1, 0, len(scale_h2_iter) - 1)]
        combined_h1_iter = (avg_norm_h1_iter / normcorr_iter) * pp_scale_h1_iter
        combined_h2_iter = (avg_norm_h2_iter / normcorr_iter) * pp_scale_h2_iter

        corr_h1_iter = np.array(
            [combined_h1_iter[relion_iter_idx_to_pos[idx]] for idx in half1_our_idx], dtype=np.float32
        )
        corr_h2_iter = np.array(
            [combined_h2_iter[relion_iter_idx_to_pos[idx]] for idx in half2_our_idx], dtype=np.float32
        )
        scale_corr_h1_iter = np.array(
            [pp_scale_h1_iter[relion_iter_idx_to_pos[idx]] for idx in half1_our_idx],
            dtype=np.float32,
        )
        scale_corr_h2_iter = np.array(
            [pp_scale_h2_iter[relion_iter_idx_to_pos[idx]] for idx in half2_our_idx],
            dtype=np.float32,
        )
        (corr_h1_iter, corr_h2_iter), applied_normalization_overrides = (
            apply_iteration_normalization_factor_overrides(
                [corr_h1_iter, corr_h2_iter],
                [scale_corr_h1_iter, scale_corr_h2_iter],
                half_stack_indices=[half1_our_idx, half2_our_idx],
                scoring_iteration=control_relion_iteration,
                overrides=iteration_normalization_overrides,
            )
        )
        for applied in applied_normalization_overrides:
            print(
                "  Diagnostic iteration normalization-factor override: "
                f"scoring_iteration={applied['scoring_iteration']} "
                f"stack={applied['stack_index']} half={applied['half']} "
                f"factor={applied['factor']:.9g}"
            )

        if "rlnOriginXAngst" in relion_iter_df.columns:
            offsets_x_iter = np.array(relion_iter_df["rlnOriginXAngst"], dtype=np.float64) / pixel_size
            offsets_y_iter = np.array(relion_iter_df["rlnOriginYAngst"], dtype=np.float64) / pixel_size
            offsets_iter = np.stack([offsets_x_iter, offsets_y_iter], axis=1)
            trans_h1_iter = np.array(
                [offsets_iter[relion_iter_idx_to_pos[idx]] for idx in half1_our_idx],
                dtype=np.float32,
            )
            trans_h2_iter = np.array(
                [offsets_iter[relion_iter_idx_to_pos[idx]] for idx in half2_our_idx],
                dtype=np.float32,
            )
        else:
            trans_h1_iter = None
            trans_h2_iter = None

        rot_h1_iter = None
        rot_h2_iter = None
        euler_h1_iter = None
        euler_h2_iter = None
        if all(col in relion_iter_df.columns for col in angle_cols):
            eulers_iter = np.stack([np.array(relion_iter_df[col], dtype=np.float64) for col in angle_cols], axis=1)
            rotations_iter = utils.R_from_relion(eulers_iter).astype(np.float32)
            rot_h1_iter = np.array(
                [rotations_iter[relion_iter_idx_to_pos[idx]] for idx in half1_our_idx],
                dtype=np.float32,
            )
            rot_h2_iter = np.array(
                [rotations_iter[relion_iter_idx_to_pos[idx]] for idx in half2_our_idx],
                dtype=np.float32,
            )
            euler_h1_iter = np.array(
                [eulers_iter[relion_iter_idx_to_pos[idx]] for idx in half1_our_idx],
                dtype=np.float32,
            )
            euler_h2_iter = np.array(
                [eulers_iter[relion_iter_idx_to_pos[idx]] for idx in half2_our_idx],
                dtype=np.float32,
            )

        pdf_iter = None
        if pdf_orient_key in model_h1_iter and pdf_orient_key in model_h2_iter:
            pdf_iter = [
                np.array(model_h1_iter[pdf_orient_key]["rlnOrientationDistribution"], dtype=np.float32),
                np.array(model_h2_iter[pdf_orient_key]["rlnOrientationDistribution"], dtype=np.float32),
            ]

        return {
            "translation_sigma_angstrom": np.float32(sigma_offset_iter),
            "translation_sigma_angstrom_per_half": np.asarray(
                sigma_offset_iter_per_half,
                dtype=np.float32,
            ),
            "image_corrections": [corr_h1_iter, corr_h2_iter],
            "scale_corrections": [scale_corr_h1_iter, scale_corr_h2_iter],
            "previous_best_translations": [trans_h1_iter, trans_h2_iter],
            "previous_best_rotations": [rot_h1_iter, rot_h2_iter],
            "previous_best_rotation_eulers": [euler_h1_iter, euler_h2_iter],
            "direction_prior": pdf_iter,
            "noise_variance": noise_variance_iter,
        }

    # The extra slot at index max_iter seeds the post-convergence all-data
    # pass from the last numbered RELION particle/model state.
    replay_iteration_overrides = [None] * (args.max_iter + 1)
    for recovar_iter, relion_prev_iter, relion_control_iter in replay_override_iteration_pairs(
        iteration,
        args.max_iter,
    ):
        if not (relion_dir / f"{run_prefix}_it{relion_prev_iter:03d}_data.star").exists():
            print(
                f"  Replay state for recovar iter {recovar_iter + 1}: RELION iter {relion_prev_iter:03d} not found, leaving override unset"
            )
            continue
        replay_iteration_overrides[recovar_iter] = _load_relion_iteration_override(
            relion_prev_iter,
            relion_control_iter,
        )
        override = replay_iteration_overrides[recovar_iter]
        if recovar_iter == 0 and diagnostic_direction_prior_override is not None:
            override["direction_prior"] = diagnostic_direction_prior_override
        trans_msg = "none"
        if override["previous_best_translations"][0] is not None:
            trans_msg = (
                f"h1 mean_abs={np.abs(override['previous_best_translations'][0]).mean():.3f} px, "
                f"h2 mean_abs={np.abs(override['previous_best_translations'][1]).mean():.3f} px"
            )
        print(
            f"  Replay state for recovar iter {recovar_iter + 1}: RELION prev={relion_prev_iter:03d}, control={relion_control_iter:03d}, "
            "sigma_offset="
            f"({float(override['translation_sigma_angstrom_per_half'][0]):.4f}, "
            f"{float(override['translation_sigma_angstrom_per_half'][1]):.4f}) A, "
            f"corr means=({override['image_corrections'][0].mean():.4f}, {override['image_corrections'][1].mean():.4f}), "
            f"pre-shifts={trans_msg}"
        )
    if args.force_final_after_zero_iterations:
        replay_iteration_overrides[0] = _load_relion_iteration_override(
            iteration,
            iteration + 1,
            process_start=False,
        )
        os.environ["RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER"] = "1"
        print(
            "  Diagnostic final-only replay: forcing K=1 final all-data after zero numbered iterations "
            f"from RELION state {iteration:03d}"
        )
    explicit_final_replay_override = final_only_replay_override(
        replay_iteration_overrides,
        enabled=args.force_final_after_zero_iterations,
    )
    selected_final_replay_override = select_final_replay_override(
        replay_iteration_overrides,
        args.final_replay_fields,
    )
    if selected_final_replay_override is not None:
        if explicit_final_replay_override is not None:
            raise ValueError(
                "--final-replay-fields and --force-final-after-zero-iterations are mutually exclusive"
            )
        explicit_final_replay_override = selected_final_replay_override
        print(
            "  Diagnostic final-only RELION field substitution: "
            + ",".join(sorted(explicit_final_replay_override))
        )

    # ---- Output directory ----
    out_dir = args.output_dir or str(relion_dir.parent / "_agent_scratch" / f"{args.max_iter}iter_parity")
    os.makedirs(out_dir, exist_ok=True)
    Path(out_dir).joinpath("SAFE_TO_DELETE").touch()
    if args.timing_only and args.save_intermediates_dir is None:
        save_intermediates_dir = None
    else:
        save_intermediates_dir = args.save_intermediates_dir or os.path.join(out_dir, "intermediates")
        os.makedirs(save_intermediates_dir, exist_ok=True)
    print(f"  Intermediate dumps: {save_intermediates_dir if save_intermediates_dir is not None else '<disabled>'}")

    gt_path = None
    if args.gt_volume is not None:
        gt_path = Path(args.gt_volume)
    else:
        candidate_gt = Path(args.data_star).with_name("reference_gt.mrc")
        if candidate_gt.exists():
            gt_path = candidate_gt
    gt_real = None
    gt_ft = None
    gt_align_rotations = None
    if gt_path is not None and gt_path.exists():
        gt_real = helpers.load_mrc(str(gt_path))
        gt_ft = np.asarray(ftu.get_dft3(jnp.asarray(gt_real))).reshape(-1)
        print(f"  GT volume: {gt_path}")
        if args.gt_align:
            from recovar.em.initial_model.gt_metrics import relion_alignment_rotations

            gt_align_rotations = relion_alignment_rotations(args.gt_align_healpix_order)
            print(
                "  GT alignment: "
                f"healpix_order={args.gt_align_healpix_order}, rotations={gt_align_rotations.shape[0]}, "
                f"score_shell<={args.gt_align_max_shell}, mirror={not args.gt_align_no_mirror}, "
                f"allow_sign={args.gt_align_allow_sign}, "
                f"series={'all' if args.gt_align_all_series else 'merged-only'}"
            )
    elif args.gt_volume is not None:
        print(f"  GT volume requested but not found: {args.gt_volume}")

    print(f"  Local-search profile: {args.local_search_profile}")
    print(f"  Local translation prior mode: {args.local_search_translation_prior_mode}")
    print(f"  First-iteration score mode: {args.first_iteration_score_mode}")
    print(f"  First-iteration reconstruction mode: {args.first_iteration_reconstruction_mode}")
    print(f"  Emulate RELION iter-1 CC: {do_firstiter_cc}")
    print(f"  RELION ini_high: {relion_ini_high}")
    print(f"  Adjoint ablations: disable_y={args.disable_adjoint_y}, disable_ctf={args.disable_adjoint_ctf}")

    # ---- Run ----
    print(f"\nRunning {args.max_iter} iterations...")
    t0 = time.time()
    keep_group_scale_update_state = retain_group_scale_update_state(
        max_iter=args.max_iter,
        skip_final_iteration=args.skip_final_iteration,
        diagnostic_retain_terminal_state=args.diagnostic_retain_terminal_group_scale_state,
    )
    if not keep_group_scale_update_state:
        print(
            "  Explicitly terminal one-iteration replay: loaded group scales are "
            "used for scoring; no downstream XA/AA group-scale update is requested"
        )
    result = refine_single_volume(
        experiment_datasets=[ds_half1, ds_half2],
        init_volume=[jnp.asarray(vol_ft_h1), jnp.asarray(vol_ft_h2)],
        init_reference_real=initial_reference_real_for_projector,
        init_noise_variance=noise_variance,
        init_mean_variance=mean_variance.reshape(-1),
        rotations=None,
        translations=None,
        disc_type="linear_interp",
        max_iter=args.max_iter,
        image_batch_size=args.image_batch_size,
        rotation_block_size=args.rotation_block_size,
        init_current_size=current_size,
        fsc_threshold=1.0 / 7.0,
        adaptive_oversampling=oversampling,
        max_significants=max_significants,
        init_healpix_order=hp_order,
        max_healpix_order=args.max_healpix_order,
        init_translation_range=offset_range / pixel_size,
        init_translation_step=offset_step / pixel_size,
        init_translation_sigma_angstrom=sigma_offset_angst_per_half,
        particle_diameter_ang=particle_diameter,
        tau2_fudge=1.0,
        perturb_factor=0.5,
        perturb_replay_relion_dir=str(relion_dir),
        perturb_replay_relion_prefix=run_prefix,
        init_relion_iteration=iteration,
        init_fsc=fsc,
        init_ave_Pmax=ave_Pmax,
        init_has_high_fsc_at_limit=has_high_fsc_at_limit,
        init_image_corrections=[corr_h1, corr_h2],
        init_scale_corrections=[scale_corr_h1, scale_corr_h2],
        init_group_ids=(
            [group_ids_h1, group_ids_h2]
            if keep_group_scale_update_state
            else None
        ),
        init_group_count=(group_count if keep_group_scale_update_state else None),
        init_previous_best_translations=[trans_h1, trans_h2],
        init_previous_best_rotation_eulers=[euler_h1, euler_h2],
        init_direction_prior=direction_prior,
        replay_iteration_overrides=replay_iteration_overrides,
        final_replay_override=explicit_final_replay_override,
        save_intermediates_dir=save_intermediates_dir,
        skip_final_iteration=args.skip_final_iteration,
        local_search_profile_mode=args.local_search_profile,
        local_search_translation_prior_mode=args.local_search_translation_prior_mode,
        disable_adjoint_y=args.disable_adjoint_y,
        disable_adjoint_ctf=args.disable_adjoint_ctf,
        emulate_relion_firstiter_cc=do_firstiter_cc,
        relion_firstiter_ini_high_angstrom=relion_ini_high if args.iter == 0 else None,
        first_iteration_score_mode=args.first_iteration_score_mode,
        first_iteration_reconstruction_mode=args.first_iteration_reconstruction_mode,
        force_max_iter_after_convergence=args.force_max_iter_after_convergence,
        image_fourier_backend=args.image_fourier_backend,
        preserve_bpref_particle_order=(
            args.diagnostic_preserve_bpref_particle_order
            or args.diagnostic_native_relion_particle_order_seed is not None
        ),
        allow_replayed_bpref_particle_order=(
            args.diagnostic_native_relion_particle_order_seed is not None
        ),
        relion_optics_image_sizes=relion_optics_image_sizes,
        relion_optics_pixel_sizes=relion_optics_pixel_sizes,
        relion_model_pixel_size=relion_model_pixel_size,
    )
    elapsed = time.time() - t0
    completed_iters = len(result.get("current_sizes", []))
    if completed_iters != args.max_iter:
        print(
            f"\nCompleted {completed_iters} emitted iterations in {elapsed:.1f}s "
            f"(requested --max_iter {args.max_iter}; stopped by convergence)"
        )
    else:
        print(f"\nCompleted {completed_iters} emitted iterations in {elapsed:.1f}s")

    if args.timing_only:
        ledger_path = args.benchmark_ledger_json or os.path.join(out_dir, "benchmark_ledger.json")
        local_profile_rows = _collect_local_profile_history(result)
        if not local_profile_rows and save_intermediates_dir is not None:
            local_profile_rows = _collect_local_profile_rows(save_intermediates_dir)
        wall_times = [float(x) for x in result.get("wall_times", [])]
        ledger = {
            "git_commit": _safe_git_commit(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "jax_version": getattr(jax, "__version__", None),
            "jaxlib_version": getattr(jaxlib, "__version__", None),
            "jax_devices": [str(device) for device in jax.devices()],
            "relion_dir": str(relion_dir),
            "data_star": str(args.data_star),
            "iter_start": int(args.iter),
            "max_iter": int(args.max_iter),
            "completed_iterations": int(completed_iters),
            "force_max_iter_after_convergence": bool(args.force_max_iter_after_convergence),
            "elapsed_s": float(elapsed),
            "timing_only": True,
            "local_search_profile_mode": args.local_search_profile,
            "disable_adjoint_y": bool(args.disable_adjoint_y),
            "disable_adjoint_ctf": bool(args.disable_adjoint_ctf),
            "compile_count_from_log": _count_compile_lines(args.compile_log),
            "wall_times_trajectory": wall_times,
            "current_sizes": [int(x) for x in result.get("current_sizes", [])],
            "pixel_resolutions": [float(x) for x in result.get("pixel_resolutions", [])],
            "ave_Pmax_trajectory": [float(x) for x in result.get("ave_Pmax_trajectory", [])],
            "local_profile_rows": local_profile_rows,
            "local_profile_summary": _summarize_local_profile_rows(local_profile_rows, wall_times),
        }
        with open(ledger_path, "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=2, sort_keys=True)
        print(f"Saved timing-only benchmark ledger: {ledger_path}")
        return

    # ---- Save results ----
    save_dict = {
        "volume_shape": np.array([N, N, N]),
        "voxel_size": np.float64(pixel_size),
        "current_sizes": np.array(result["current_sizes"]),
        "pixel_resolutions": np.array(result["pixel_resolutions"]),
        "n_half1_particles": np.int32(half1_indices.size),
        "n_half2_particles": np.int32(half2_indices.size),
        "half1_indices": half1_indices,
        "half2_indices": half2_indices,
        "max_significants": np.int32(max_significants),
        "local_search_profile_mode": np.array(args.local_search_profile),
        "local_search_translation_prior_mode": np.array(args.local_search_translation_prior_mode),
        "first_iteration_score_mode": np.array(args.first_iteration_score_mode),
        "first_iteration_reconstruction_mode": np.array(args.first_iteration_reconstruction_mode),
        "firstiter_cc_mode": np.array(args.firstiter_cc_mode),
        "firstiter_cc_oracle_enabled": np.bool_(oracle_firstiter_cc),
        "firstiter_cc_effective": np.bool_(do_firstiter_cc),
        "relion_ini_high_angstrom": np.float64(relion_ini_high),
        "continuous_relion_noise_state": np.bool_(args.continuous_relion_noise_state),
        "disable_adjoint_y": np.bool_(args.disable_adjoint_y),
        "disable_adjoint_ctf": np.bool_(args.disable_adjoint_ctf),
        "final_all_data_ran": np.bool_(result.get("final_all_data_ran", False)),
    }
    if result.get("ave_Pmax_trajectory"):
        save_dict["ave_Pmax_trajectory"] = np.array(result["ave_Pmax_trajectory"])
    if result.get("pmax_per_image_history"):
        for i, pmax_arr in enumerate(result["pmax_per_image_history"]):
            save_dict[f"pmax_per_image_iter_{i:03d}"] = np.array(pmax_arr, dtype=np.float32)
    if result.get("healpix_order_trajectory"):
        save_dict["healpix_order_trajectory"] = np.array(result["healpix_order_trajectory"])
    if result.get("wall_times"):
        save_dict["wall_times_trajectory"] = np.array(result["wall_times"], dtype=np.float64)
    if result.get("sigma_offset_trajectory"):
        save_dict["sigma_offset_trajectory"] = np.array(result["sigma_offset_trajectory"], dtype=np.float64)
    if result.get("sigma_offset_used_trajectory"):
        save_dict["sigma_offset_used_trajectory"] = np.array(result["sigma_offset_used_trajectory"], dtype=np.float64)
    if result.get("sigma_offset_per_half_trajectory"):
        save_dict["sigma_offset_per_half_trajectory"] = np.asarray(
            result["sigma_offset_per_half_trajectory"], dtype=np.float64
        )
    if result.get("sigma_offset_used_per_half_trajectory"):
        save_dict["sigma_offset_used_per_half_trajectory"] = np.asarray(
            result["sigma_offset_used_per_half_trajectory"], dtype=np.float64
        )
    if result.get("direction_prior_trajectory_per_half"):
        save_dict["direction_prior_trajectory_per_half"] = np.asarray(
            result["direction_prior_trajectory_per_half"], dtype=object
        )
    for scalar_name in [
        "frac_changed_trajectory",
        "acc_rot_trajectory",
        "smallest_change_angles_trajectory",
        "smallest_change_offsets_trajectory",
    ]:
        if result.get(scalar_name):
            save_dict[scalar_name] = np.array(result[scalar_name], dtype=np.float64)

    def _save_array_or_half_sequence(key, value, dtype=None):
        try:
            arr = np.asarray(value, dtype=dtype) if dtype is not None else np.asarray(value)
            if arr.dtype != object:
                save_dict[key] = arr
                return
        except (TypeError, ValueError):
            pass
        if isinstance(value, (list, tuple)):
            saved_any = False
            for half_idx, half_value in enumerate(value):
                if half_value is None:
                    continue
                save_dict[f"{key}_half{half_idx + 1}"] = (
                    np.asarray(half_value, dtype=dtype) if dtype is not None else np.asarray(half_value)
                )
                saved_any = True
            if saved_any:
                return
        save_dict[key] = np.asarray(value, dtype=object)

    def _concat_half_sequence(value, dtype):
        if isinstance(value, (list, tuple)):
            return np.concatenate([np.asarray(v, dtype=dtype) for v in value if v is not None], axis=0)
        return np.asarray(value, dtype=dtype)

    for traj_name, prefix_name in [
        ("fsc_history", "fsc_iter"),
        ("data_vs_prior_trajectory", "data_vs_prior_iter"),
        ("noise_radial_trajectory", "noise_radial_iter"),
        ("noise_radial_per_half_trajectory", "noise_radial_per_half_iter"),
        ("tau2_radial_trajectory", "tau2_radial_iter"),
        ("tau2_sigma2_trajectory", "tau2_sigma2_iter"),
        ("tau2_avg_weight_trajectory", "tau2_avg_weight_iter"),
        ("tau2_shell_sum_trajectory", "tau2_shell_sum_iter"),
        ("tau2_shell_count_trajectory", "tau2_shell_count_iter"),
        ("tau2_fsc_used_trajectory", "tau2_fsc_used_iter"),
        ("tau2_ssnr_trajectory", "tau2_ssnr_iter"),
        ("rotation_posterior_trajectory_per_half", "rotation_posterior_per_half_iter"),
    ]:
        if result.get(traj_name):
            for i, arr_i in enumerate(result[traj_name]):
                if arr_i is not None:
                    _save_array_or_half_sequence(f"{prefix_name}_{i:03d}", arr_i)
    if result.get("significant_counts"):
        add_significant_count_artifacts(
            save_dict,
            result["significant_counts"],
            [half1_indices, half2_indices],
            len(our_subsets),
        )
    for traj_name, prefix_name in [
        ("best_rotation_eulers_history", "best_rotation_eulers_iter"),
        ("best_translations_history", "best_translations_iter"),
    ]:
        if result.get(traj_name):
            for i, arr_i in enumerate(result[traj_name]):
                if arr_i is not None:
                    _save_array_or_half_sequence(f"{prefix_name}_{i:03d}", arr_i, dtype=np.float32)

    for key in (
        "final_all_data_best_rotation_eulers",
        "final_all_data_best_translations",
        "final_all_data_max_posterior",
    ):
        value = result.get(key)
        if value is not None:
            save_dict[key] = _concat_half_sequence(value, np.float32)
    for key in (
        "final_all_data_fsc",
        "tau2_radial_final_all_data",
        "tau2_fsc_used_final_all_data",
        "tau2_ssnr_final_all_data",
    ):
        value = result.get(key)
        if value is not None:
            save_dict[key] = np.asarray(value)
    for key in (
        "final_all_data_sampling_perturbation",
        "final_all_data_sampling_perturbation_applied",
        "final_all_data_sampling_relion_iteration",
        "final_all_data_sampling_offset_range",
        "final_all_data_sampling_offset_step",
        "final_all_data_grid_correct",
    ):
        if key in result:
            save_dict[key] = np.asarray(result[key])
    for key in (
        "final_all_data_sampling_star",
        "final_all_data_sampling_star_source",
        "final_all_data_gridding_correct",
        "tau2_weight_combination_final_all_data",
    ):
        if result.get(key) is not None:
            save_dict[key] = np.asarray(str(result[key]))

    final_half1_ft, final_half2_ft, final_merged_ft = final_output_fourier_volumes(result)

    save_dict["final_half1_ft"] = final_half1_ft
    save_dict["final_half2_ft"] = final_half2_ft
    save_dict["final_merged_ft"] = final_merged_ft

    save_volume(
        np.asarray(final_half1_ft),
        os.path.join(out_dir, "recovar_final_half1"),
        volume_shape=(N, N, N),
        from_ft=True,
        voxel_size=pixel_size,
    )
    save_volume(
        np.asarray(final_half2_ft),
        os.path.join(out_dir, "recovar_final_half2"),
        volume_shape=(N, N, N),
        from_ft=True,
        voxel_size=pixel_size,
    )
    save_volume(
        np.asarray(final_merged_ft),
        os.path.join(out_dir, "recovar_final_merged"),
        volume_shape=(N, N, N),
        from_ft=True,
        voxel_size=pixel_size,
    )
    print(
        f"Saved final volumes: {os.path.join(out_dir, 'recovar_final_half1.mrc')}, recovar_final_half2.mrc, recovar_final_merged.mrc"
    )

    # ---- Summary table ----
    n_iters = len(result["current_sizes"])
    print(f"\n{'iter':>4} {'cs':>4} {'pixres':>6} {'pmax':>8} {'hp':>3} {'FSC@0.5':>8} {'res(A)':>8}")
    print("-" * 50)
    for i in range(n_iters):
        cs_i = result["current_sizes"][i]
        pr_i = result["pixel_resolutions"][i]
        pmax_i = (
            result["ave_Pmax_trajectory"][i]
            if result.get("ave_Pmax_trajectory") and i < len(result["ave_Pmax_trajectory"])
            else 0
        )
        hp_i = (
            result["healpix_order_trajectory"][i]
            if result.get("healpix_order_trajectory") and i < len(result["healpix_order_trajectory"])
            else hp_order
        )
        fsc_i = (
            np.array(result["fsc_history"][i]) if result.get("fsc_history") and i < len(result["fsc_history"]) else None
        )
        fsc05 = 0
        if fsc_i is not None:
            for s in range(1, len(fsc_i)):
                if fsc_i[s] >= 0.5:
                    fsc05 = s
        res = (N * pixel_size) / max(fsc05, 1)
        print(f"{i + 1:4d} {cs_i:4d} {pr_i:6.1f} {pmax_i:8.4f} {hp_i:3d} {fsc05:8d} {res:8.1f}")

    # ---- Compare final volume with the semantically matched RELION oracle ----
    final_oracle_mode, final_oracle_paths = resolve_relion_final_oracle_paths(
        relion_dir,
        run_prefix=run_prefix,
        start_iteration=iteration,
        completed_iterations=completed_iters,
        final_all_data_ran=bool(result.get("final_all_data_ran", False)),
    )
    save_dict["relion_final_oracle_mode"] = np.array(final_oracle_mode)
    relion_final_real = {}
    relion_final_ft = {}
    recovar_final_real = {
        "half1": np.real(np.array(ftu.get_idft3(jnp.asarray(final_half1_ft.reshape(N, N, N))))),
        "half2": np.real(np.array(ftu.get_idft3(jnp.asarray(final_half2_ft.reshape(N, N, N))))),
        "merged": np.real(np.array(ftu.get_idft3(jnp.asarray(final_merged_ft.reshape(N, N, N))))),
    }
    for label, target_path in final_oracle_paths.items():
        if not target_path.exists():
            print(f"  Final {label}: matched RELION oracle not found: {target_path}")
            continue
        relion_vol = helpers.load_relion_volume(str(target_path))
        relion_final_real[label] = relion_vol
        relion_final_ft[label] = np.asarray(ftu.get_dft3(jnp.asarray(relion_vol))).reshape(-1)
        corr = float(np.corrcoef(recovar_final_real[label].ravel(), relion_vol.ravel())[0, 1])
        print(f"  Final {label} vs RELION {final_oracle_mode} oracle {target_path.name}: corr={corr:.6f}")
        save_dict[f"final_{label}_corr_vs_relion"] = np.float64(corr)
        save_dict[f"relion_final_oracle_path_{label}"] = np.array(str(target_path))

    relion_merged_ft = None
    if "merged" in relion_final_ft:
        relion_merged_ft = relion_final_ft["merged"]
    elif "half1" in relion_final_ft and "half2" in relion_final_ft:
        relion_merged_ft = (
            relion_final_ft["half1"].astype(np.complex128) + relion_final_ft["half2"].astype(np.complex128)
        ) / 2.0
    if relion_merged_ft is not None:
        if final_oracle_mode == "all_data":
            relion_merged_real = relion_final_real["merged"]
        else:
            relion_merged_real = np.real(
                np.array(ftu.get_idft3(jnp.asarray(relion_merged_ft.reshape(N, N, N))))
            )
        merged_corr = float(np.corrcoef(recovar_final_real["merged"].ravel(), relion_merged_real.ravel())[0, 1])
        merged_fsc = _compute_fsc_vs_gt(final_merged_ft, relion_merged_ft)
        merged_fsc_auc = _normalized_fsc_auc(merged_fsc)
        save_dict["final_merged_corr_vs_relion"] = np.float64(merged_corr)
        save_dict["final_merged_fsc_vs_relion"] = merged_fsc
        save_dict["final_merged_fsc_auc_vs_relion"] = np.float64(merged_fsc_auc)
        print(f"  Final merged vs RELION: corr={merged_corr:.6f}, FSC-AUC={merged_fsc_auc:.6f}")
    if "half1" in relion_final_ft and "half2" in relion_final_ft:
        save_volume(
            np.asarray(relion_final_ft["half1"]),
            os.path.join(out_dir, "relion_final_half1"),
            volume_shape=(N, N, N),
            from_ft=True,
            voxel_size=pixel_size,
        )
        save_volume(
            np.asarray(relion_final_ft["half2"]),
            os.path.join(out_dir, "relion_final_half2"),
            volume_shape=(N, N, N),
            from_ft=True,
            voxel_size=pixel_size,
        )
    if relion_merged_ft is not None:
        save_volume(
            np.asarray(relion_merged_ft, dtype=np.complex64),
            os.path.join(out_dir, "relion_final_merged"),
            volume_shape=(N, N, N),
            from_ft=True,
            voxel_size=pixel_size,
        )
        print(f"  Saved matched RELION final map: {os.path.join(out_dir, 'relion_final_merged.mrc')}")

    gt_ledger_summary = {}
    if gt_ft is not None:
        print("\n=== Final FSC vs GT ===")
        from recovar.em.initial_model.gt_metrics import align_volume_to_reference

        gt_summary = {}
        recovar_final_series = {
            "recovar_half1": final_half1_ft,
            "recovar_half2": final_half2_ft,
            "recovar_merged": final_merged_ft,
        }
        recovar_final_series.update(relion_final_gt_series(relion_final_ft, relion_merged_ft))

        for label, vol_ft in recovar_final_series.items():
            fsc_vs_gt = _compute_fsc_vs_gt(vol_ft, gt_ft)
            fsc_auc_vs_gt = _normalized_fsc_auc(fsc_vs_gt)
            shell_05 = _first_shell_below_threshold(fsc_vs_gt, 0.5)
            shell_0143 = _first_shell_below_threshold(fsc_vs_gt, 0.143)
            real_vol = np.real(np.array(ftu.get_idft3(jnp.asarray(np.asarray(vol_ft).reshape(N, N, N)))))
            corr_vs_gt = float(np.corrcoef(real_vol.ravel(), gt_real.ravel())[0, 1])
            print(
                f"  {label:<14s} FSC-AUC={fsc_auc_vs_gt:.6f}, corr={corr_vs_gt:.6f}, "
                f"FSC<0.5 shell={shell_05}, res={_shell_to_resolution_angstrom(shell_05):.2f} A, "
                f"FSC<0.143 shell={shell_0143}, res={_shell_to_resolution_angstrom(shell_0143):.2f} A"
            )
            gt_summary[f"{label}_fsc_vs_gt"] = fsc_vs_gt
            gt_summary[f"{label}_fsc_auc_vs_gt"] = np.float64(fsc_auc_vs_gt)
            gt_summary[f"{label}_corr_vs_gt"] = np.float64(corr_vs_gt)
            gt_summary[f"{label}_shell_05"] = np.int32(-1 if shell_05 is None else shell_05)
            gt_summary[f"{label}_shell_0143"] = np.int32(-1 if shell_0143 is None else shell_0143)
            gt_ledger_summary[f"{label}_corr_vs_gt"] = float(corr_vs_gt)
            gt_ledger_summary[f"{label}_fsc_auc_vs_gt"] = float(fsc_auc_vs_gt)
            gt_ledger_summary[f"{label}_shell_05"] = int(-1 if shell_05 is None else shell_05)
            gt_ledger_summary[f"{label}_shell_0143"] = int(-1 if shell_0143 is None else shell_0143)
            if args.gt_align and (args.gt_align_all_series or label.endswith("_merged")):
                if gt_align_rotations is None:
                    raise RuntimeError("--gt_align requested but no GT alignment rotation grid was initialized")
                alignment = align_volume_to_reference(
                    real_vol,
                    gt_real,
                    gt_align_rotations,
                    score_max_shell=int(args.gt_align_max_shell),
                    allow_mirror=not bool(args.gt_align_no_mirror),
                    allow_sign=bool(args.gt_align_allow_sign),
                )
                aligned_ft = np.asarray(ftu.get_dft3(jnp.asarray(alignment.aligned_volume))).reshape(-1)
                aligned_fsc_vs_gt = _compute_fsc_vs_gt(aligned_ft, gt_ft)
                aligned_fsc_auc_vs_gt = _normalized_fsc_auc(aligned_fsc_vs_gt)
                aligned_shell_05 = _first_shell_below_threshold(aligned_fsc_vs_gt, 0.5)
                aligned_shell_0143 = _first_shell_below_threshold(aligned_fsc_vs_gt, 0.143)
                print(
                    f"  {label:<14s} aligned_FSC-AUC={aligned_fsc_auc_vs_gt:.6f}, "
                    f"aligned_corr={alignment.corr:.6f}, "
                    f"aligned_FSC<0.5 shell={aligned_shell_05}, "
                    f"res={_shell_to_resolution_angstrom(aligned_shell_05):.2f} A, "
                    f"aligned_FSC<0.143 shell={aligned_shell_0143}, "
                    f"res={_shell_to_resolution_angstrom(aligned_shell_0143):.2f} A, "
                    f"rot_idx={alignment.rotation_index}, mirror_x={alignment.mirror_x}, sign={alignment.sign}"
                )
                gt_summary[f"{label}_aligned_fsc_vs_gt"] = aligned_fsc_vs_gt
                gt_summary[f"{label}_aligned_fsc_auc_vs_gt"] = np.float64(aligned_fsc_auc_vs_gt)
                gt_summary[f"{label}_aligned_corr_vs_gt"] = np.float64(alignment.corr)
                gt_summary[f"{label}_aligned_score_vs_gt"] = np.float64(alignment.score)
                gt_summary[f"{label}_aligned_shell_05"] = np.int32(-1 if aligned_shell_05 is None else aligned_shell_05)
                gt_summary[f"{label}_aligned_shell_0143"] = np.int32(
                    -1 if aligned_shell_0143 is None else aligned_shell_0143
                )
                gt_summary[f"{label}_gt_align_rotation_index"] = np.int32(alignment.rotation_index)
                gt_summary[f"{label}_gt_align_rotation_matrix"] = alignment.rotation_matrix
                gt_summary[f"{label}_gt_align_mirror_x"] = np.bool_(alignment.mirror_x)
                gt_summary[f"{label}_gt_align_sign"] = np.int32(alignment.sign)
                gt_summary[f"{label}_gt_align_score_max_shell"] = np.int32(args.gt_align_max_shell)
                gt_summary[f"{label}_gt_align_healpix_order"] = np.int32(args.gt_align_healpix_order)
                gt_ledger_summary[f"{label}_aligned_corr_vs_gt"] = float(alignment.corr)
                gt_ledger_summary[f"{label}_aligned_fsc_auc_vs_gt"] = float(aligned_fsc_auc_vs_gt)
                gt_ledger_summary[f"{label}_aligned_score_vs_gt"] = float(alignment.score)
                gt_ledger_summary[f"{label}_aligned_shell_05"] = int(
                    -1 if aligned_shell_05 is None else aligned_shell_05
                )
                gt_ledger_summary[f"{label}_aligned_shell_0143"] = int(
                    -1 if aligned_shell_0143 is None else aligned_shell_0143
                )
                gt_ledger_summary[f"{label}_gt_align_rotation_index"] = int(alignment.rotation_index)
                gt_ledger_summary[f"{label}_gt_align_mirror_x"] = bool(alignment.mirror_x)
                gt_ledger_summary[f"{label}_gt_align_sign"] = int(alignment.sign)

        save_dict.update(gt_summary)
        gt_npz_path = os.path.join(out_dir, "gt_comparison_final.npz")
        np.savez(gt_npz_path, **gt_summary)
        print(f"  Saved GT comparison: {gt_npz_path}")

    npz_path = os.path.join(out_dir, "refinement_results.npz")
    np.savez(npz_path, **save_dict)
    print(f"Saved: {npz_path}")

    ledger_path = args.benchmark_ledger_json or os.path.join(out_dir, "benchmark_ledger.json")
    local_profile_rows = _collect_local_profile_rows(save_intermediates_dir)
    wall_times = [float(x) for x in result.get("wall_times", [])]
    ledger = {
        "git_commit": _safe_git_commit(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "jax_version": getattr(jax, "__version__", None),
        "jaxlib_version": getattr(jaxlib, "__version__", None),
        "jax_devices": [str(device) for device in jax.devices()],
        "relion_dir": str(relion_dir),
        "data_star": str(args.data_star),
        "iter_start": int(args.iter),
        "max_iter": int(args.max_iter),
        "completed_iterations": int(completed_iters),
        "gt_volume": str(gt_path) if gt_path is not None else None,
        "gt_align_enabled": bool(args.gt_align and gt_ft is not None),
        "gt_align_healpix_order": int(args.gt_align_healpix_order),
        "gt_align_max_shell": int(args.gt_align_max_shell),
        "gt_align_allow_mirror": not bool(args.gt_align_no_mirror),
        "gt_align_allow_sign": bool(args.gt_align_allow_sign),
        "gt_align_all_series": bool(args.gt_align_all_series),
        "gt_metrics": gt_ledger_summary,
        "force_max_iter_after_convergence": bool(args.force_max_iter_after_convergence),
        "final_all_data_ran": bool(result.get("final_all_data_ran", False)),
        "relion_final_oracle_mode": final_oracle_mode,
        "relion_final_oracle_paths": {label: str(path) for label, path in final_oracle_paths.items()},
        "final_merged_corr_vs_relion": (
            float(save_dict["final_merged_corr_vs_relion"])
            if "final_merged_corr_vs_relion" in save_dict
            else None
        ),
        "final_merged_fsc_auc_vs_relion": (
            float(save_dict["final_merged_fsc_auc_vs_relion"])
            if "final_merged_fsc_auc_vs_relion" in save_dict
            else None
        ),
        "elapsed_s": float(elapsed),
        "local_search_profile_mode": args.local_search_profile,
        "disable_adjoint_y": bool(args.disable_adjoint_y),
        "disable_adjoint_ctf": bool(args.disable_adjoint_ctf),
        "compile_count_from_log": _count_compile_lines(args.compile_log),
        "wall_times_trajectory": wall_times,
        "current_sizes": [int(x) for x in result.get("current_sizes", [])],
        "pixel_resolutions": [float(x) for x in result.get("pixel_resolutions", [])],
        "local_profile_rows": local_profile_rows,
        "local_profile_summary": _summarize_local_profile_rows(local_profile_rows, wall_times),
    }
    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2, sort_keys=True)
    print(f"Saved benchmark ledger: {ledger_path}")

    # ---- Per-particle Pmax comparison with RELION ----
    # pmax_per_image_history entries are in (half1, half2) concatenated order.
    # Map them back to original particle ordering for matched comparison.
    n_total = len(our_names)
    gt_pose_path = Path(args.data_star).with_name("poses.pkl")
    gt_rotations_orig = None
    gt_translations_orig = None
    gt_transpose_relion_convention = None
    if gt_pose_path.exists():
        gt_pose_data = utils.pickle_load(str(gt_pose_path))
        if isinstance(gt_pose_data, tuple) and len(gt_pose_data) >= 1:
            gt_rot_all = np.asarray(gt_pose_data[0], dtype=np.float64)
            gt_trans_all = np.asarray(gt_pose_data[1], dtype=np.float64) if len(gt_pose_data) >= 2 else None
            gt_rotations_orig, gt_translations_orig = map_pose_arrays_to_particle_order(
                our_names,
                gt_rot_all,
                gt_trans_all,
            )
            print(f"  GT poses: {gt_pose_path}")
        else:
            print(f"  GT poses present but not in expected tuple format: {gt_pose_path}")

    if result.get("pmax_per_image_history"):
        for i_iter, pmax_arr in enumerate(result["pmax_per_image_history"]):
            target_it = iteration + 1 + i_iter
            target_data_star = relion_dir / f"{run_prefix}_it{target_it:03d}_data.star"
            if not target_data_star.exists():
                print(
                    f"\n  Iter {i_iter + 1}: RELION data star it{target_it:03d} not found, skipping per-particle comparison"
                )
                continue
            relion_data_it = starfile.read(str(target_data_star))
            relion_df_it = relion_data_it["particles"] if isinstance(relion_data_it, dict) else relion_data_it
            relion_pmax_raw = _read_relion_pmax_column(relion_df_it)
            if relion_pmax_raw is None:
                print(
                    f"\n  Iter {i_iter + 1}: RELION data star it{target_it:03d} lacks rlnMaxValueProbDistribution, skipping per-particle comparison"
                )
                continue

            # Map RELION particles to original ordering by stack index
            relion_names_it = list(relion_df_it["rlnImageName"])
            relion_idx_to_pos = {_idx(relion_names_it[j]): j for j in range(len(relion_names_it))}
            relion_pmax_map = {_idx(relion_names_it[j]): relion_pmax_raw[j] for j in range(len(relion_names_it))}

            # Reconstruct recovar Pmax in original particle ordering
            # pmax_arr = [half1_pmax (n_half1,), half2_pmax (n_half2,)] concatenated
            pmax_arr_np = np.asarray(pmax_arr, dtype=np.float64)
            n_h1 = len(half1_indices)
            recovar_pmax_orig = np.full(n_total, np.nan, dtype=np.float64)
            recovar_pmax_orig[half1_indices] = pmax_arr_np[:n_h1]
            recovar_pmax_orig[half2_indices] = pmax_arr_np[n_h1:]

            # Build matched RELION array in original ordering
            relion_pmax_orig = np.full(n_total, np.nan, dtype=np.float64)
            relion_eulers_orig = np.full((n_total, 3), np.nan, dtype=np.float64)
            relion_trans_orig = np.full((n_total, 2), np.nan, dtype=np.float64)
            has_relion_eulers = all(
                col in relion_df_it.columns for col in ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]
            )
            has_relion_trans = all(col in relion_df_it.columns for col in ["rlnOriginXAngst", "rlnOriginYAngst"])
            relion_eulers_raw = (
                np.stack(
                    [
                        np.array(relion_df_it[col], dtype=np.float64)
                        for col in ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]
                    ],
                    axis=1,
                )
                if has_relion_eulers
                else None
            )
            relion_trans_raw = (
                np.stack(
                    [
                        np.array(relion_df_it["rlnOriginXAngst"], dtype=np.float64) / pixel_size,
                        np.array(relion_df_it["rlnOriginYAngst"], dtype=np.float64) / pixel_size,
                    ],
                    axis=1,
                )
                if has_relion_trans
                else None
            )
            recovar_trans_orig = None
            for j, name in enumerate(our_names):
                idx = _idx(name)
                if idx in relion_pmax_map:
                    relion_pmax_orig[j] = relion_pmax_map[idx]
                    rel_pos = relion_idx_to_pos[idx]
                    if relion_eulers_raw is not None:
                        relion_eulers_orig[j] = relion_eulers_raw[rel_pos]
                    if relion_trans_raw is not None:
                        relion_trans_orig[j] = relion_trans_raw[rel_pos]

            # Compare only particles present in both
            valid = ~(np.isnan(recovar_pmax_orig) | np.isnan(relion_pmax_orig))
            recovar_pmax = recovar_pmax_orig[valid]
            relion_pmax_matched = relion_pmax_orig[valid]

            diff = recovar_pmax - relion_pmax_matched
            abs_diff = np.abs(diff)
            corr = float(np.corrcoef(recovar_pmax, relion_pmax_matched)[0, 1])

            print(f"\n=== Per-particle Pmax comparison: iter {i_iter + 1} (RELION it{target_it:03d}) ===")
            print(f"  N particles matched: {valid.sum()} / {n_total}")
            print(f"  recovar  ave_Pmax = {recovar_pmax.mean():.6f}")
            print(f"  RELION   ave_Pmax = {relion_pmax_matched.mean():.6f}")
            print(f"  Gap (recovar - RELION) = {diff.mean():.6f}")
            print(
                f"  Abs diff:  mean={abs_diff.mean():.6f}, median={np.median(abs_diff):.6f}, max={abs_diff.max():.6f}"
            )
            print(f"  Std diff:  {diff.std():.6f}")
            print(f"  Correlation: {corr:.6f}")
            print("  Percentiles of (recovar - RELION):")
            for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                print(f"    p{pct:2d}: {np.percentile(diff, pct):+.6f}")

            # Save full per-particle comparison
            comp_path = os.path.join(out_dir, f"pmax_comparison_iter{i_iter:03d}.npz")
            np.savez(
                comp_path,
                recovar_pmax=recovar_pmax_orig,
                relion_pmax=relion_pmax_orig,
                diff_valid=diff,
                half1_indices=half1_indices,
                half2_indices=half2_indices,
            )
            print(f"  Saved per-particle comparison: {comp_path}")

            best_eulers_hist = result.get("best_rotation_eulers_history")
            best_trans_hist = result.get("best_translations_history")
            if best_eulers_hist and i_iter < len(best_eulers_hist) and best_eulers_hist[i_iter] is not None:
                best_eulers_arr = _concat_half_sequence(best_eulers_hist[i_iter], np.float64)
                recovar_eulers_orig = np.full((n_total, 3), np.nan, dtype=np.float64)
                recovar_eulers_orig[half1_indices] = best_eulers_arr[:n_h1]
                recovar_eulers_orig[half2_indices] = best_eulers_arr[n_h1:]
                valid_angle = ~(np.isnan(recovar_eulers_orig).any(axis=1) | np.isnan(relion_eulers_orig).any(axis=1))
                if np.any(valid_angle):
                    ang_err_deg = _angular_error_deg_from_eulers(
                        recovar_eulers_orig[valid_angle],
                        relion_eulers_orig[valid_angle],
                    )
                    view_err_deg = _view_direction_error_deg_from_eulers(
                        recovar_eulers_orig[valid_angle],
                        relion_eulers_orig[valid_angle],
                    )
                    inplane_err_deg = _inplane_error_deg_from_eulers(
                        recovar_eulers_orig[valid_angle],
                        relion_eulers_orig[valid_angle],
                    )
                    print(f"  Angular error (deg): {_format_error_summary(ang_err_deg, '°', [5, 10, 20])}")
                    print(f"  View-dir error (deg): {_format_error_summary(view_err_deg, '°', [2, 5, 10])}")
                    print(f"  In-plane error (deg): {_format_error_summary(inplane_err_deg, '°', [2, 5, 10])}")
                else:
                    ang_err_deg = None
                    view_err_deg = None
                    inplane_err_deg = None

                recovar_gt_ang_err_deg = None
                recovar_gt_view_err_deg = None
                recovar_gt_inplane_err_deg = None
                relion_gt_ang_err_deg = None
                relion_gt_view_err_deg = None
                relion_gt_inplane_err_deg = None
                if gt_rotations_orig is not None:
                    valid_relion_gt = ~(
                        np.isnan(relion_eulers_orig).any(axis=1) | np.isnan(gt_rotations_orig).any(axis=(1, 2))
                    )
                    if np.any(valid_relion_gt) and gt_transpose_relion_convention is None:
                        relion_gt_direct = _rotation_matrices_from_eulers_deg(relion_eulers_orig[valid_relion_gt])
                        direct_err = _angular_error_deg_from_rotations(
                            relion_gt_direct,
                            gt_rotations_orig[valid_relion_gt],
                        )
                        transpose_err = _angular_error_deg_from_rotations(
                            np.transpose(relion_gt_direct, (0, 2, 1)),
                            gt_rotations_orig[valid_relion_gt],
                        )
                        gt_transpose_relion_convention = bool(np.nanmedian(transpose_err) < np.nanmedian(direct_err))
                        mode = "transpose" if gt_transpose_relion_convention else "direct"
                        print(
                            "  GT rotation convention: using "
                            f"{mode} RELION-like rotations "
                            f"(RELION-vs-GT median direct={np.nanmedian(direct_err):.4f}°, "
                            f"transpose={np.nanmedian(transpose_err):.4f}°)"
                        )

                    valid_recovar_gt = ~(
                        np.isnan(recovar_eulers_orig).any(axis=1) | np.isnan(gt_rotations_orig).any(axis=(1, 2))
                    )
                    if np.any(valid_recovar_gt):
                        recovar_rot_gt = _rotations_in_gt_frame_from_relion_eulers(
                            recovar_eulers_orig[valid_recovar_gt],
                            gt_transpose_relion_convention if gt_transpose_relion_convention is not None else True,
                        )
                        gt_rot_valid = gt_rotations_orig[valid_recovar_gt]
                        recovar_gt_ang_err_deg = _angular_error_deg_from_rotations(recovar_rot_gt, gt_rot_valid)
                        recovar_gt_view_err_deg = _view_direction_error_deg_from_rotations(recovar_rot_gt, gt_rot_valid)
                        recovar_gt_inplane_err_deg = _inplane_error_deg_from_rotations(recovar_rot_gt, gt_rot_valid)
                        print(
                            "  RECOVAR vs GT angle error: "
                            f"{_format_error_summary(recovar_gt_ang_err_deg, '°', [2, 5, 10])}"
                        )
                        print(
                            "  RECOVAR vs GT view-dir: "
                            f"{_format_error_summary(recovar_gt_view_err_deg, '°', [2, 5, 10])}"
                        )
                        print(
                            "  RECOVAR vs GT in-plane: "
                            f"{_format_error_summary(recovar_gt_inplane_err_deg, '°', [2, 5, 10])}"
                        )
                    if np.any(valid_relion_gt):
                        relion_rot_gt = _rotations_in_gt_frame_from_relion_eulers(
                            relion_eulers_orig[valid_relion_gt],
                            gt_transpose_relion_convention if gt_transpose_relion_convention is not None else True,
                        )
                        gt_rot_valid = gt_rotations_orig[valid_relion_gt]
                        relion_gt_ang_err_deg = _angular_error_deg_from_rotations(relion_rot_gt, gt_rot_valid)
                        relion_gt_view_err_deg = _view_direction_error_deg_from_rotations(relion_rot_gt, gt_rot_valid)
                        relion_gt_inplane_err_deg = _inplane_error_deg_from_rotations(relion_rot_gt, gt_rot_valid)
                        print(
                            "  RELION  vs GT angle error: "
                            f"{_format_error_summary(relion_gt_ang_err_deg, '°', [2, 5, 10])}"
                        )
                        print(
                            "  RELION  vs GT view-dir: "
                            f"{_format_error_summary(relion_gt_view_err_deg, '°', [2, 5, 10])}"
                        )
                        print(
                            "  RELION  vs GT in-plane: "
                            f"{_format_error_summary(relion_gt_inplane_err_deg, '°', [2, 5, 10])}"
                        )
                if best_trans_hist and i_iter < len(best_trans_hist) and best_trans_hist[i_iter] is not None:
                    best_trans_arr = _concat_half_sequence(best_trans_hist[i_iter], np.float64)
                    recovar_trans_orig = np.full((n_total, 2), np.nan, dtype=np.float64)
                    recovar_trans_orig[half1_indices] = best_trans_arr[:n_h1]
                    recovar_trans_orig[half2_indices] = best_trans_arr[n_h1:]
                    recovar_gt_trans_err_px = None
                    relion_gt_trans_err_px = None
                    valid_trans = ~(np.isnan(recovar_trans_orig).any(axis=1) | np.isnan(relion_trans_orig).any(axis=1))
                    if np.any(valid_trans):
                        trans_err_px = np.linalg.norm(
                            recovar_trans_orig[valid_trans] - relion_trans_orig[valid_trans],
                            axis=1,
                        )
                        trans_err_ang = trans_err_px * pixel_size
                        print(
                            "  Translation error: "
                            f"{_format_error_summary(trans_err_px, ' px', [0.25, 0.5, 1.0])} "
                            f"(mean={trans_err_ang.mean():.4f} A)"
                        )
                    else:
                        trans_err_px = None
                    if gt_translations_orig is not None:
                        valid_recovar_gt_trans = ~(
                            np.isnan(recovar_trans_orig).any(axis=1) | np.isnan(gt_translations_orig).any(axis=1)
                        )
                        if np.any(valid_recovar_gt_trans):
                            recovar_gt_trans_err_px = np.linalg.norm(
                                recovar_trans_orig[valid_recovar_gt_trans]
                                - gt_translations_orig[valid_recovar_gt_trans],
                                axis=1,
                            )
                            print(
                                "  RECOVAR vs GT translation: "
                                f"{_format_error_summary(recovar_gt_trans_err_px, ' px', [0.25, 0.5, 1.0])}"
                            )
                        valid_relion_gt_trans = ~(
                            np.isnan(relion_trans_orig).any(axis=1) | np.isnan(gt_translations_orig).any(axis=1)
                        )
                        if np.any(valid_relion_gt_trans):
                            relion_gt_trans_err_px = np.linalg.norm(
                                relion_trans_orig[valid_relion_gt_trans] - gt_translations_orig[valid_relion_gt_trans],
                                axis=1,
                            )
                            print(
                                "  RELION  vs GT translation: "
                                f"{_format_error_summary(relion_gt_trans_err_px, ' px', [0.25, 0.5, 1.0])}"
                            )
                else:
                    trans_err_px = None
                    recovar_gt_trans_err_px = None
                    relion_gt_trans_err_px = None

                pose_path = os.path.join(out_dir, f"pose_comparison_iter{i_iter:03d}.npz")
                np.savez(
                    pose_path,
                    recovar_eulers=recovar_eulers_orig,
                    relion_eulers=relion_eulers_orig,
                    angular_error_deg=ang_err_deg if ang_err_deg is not None else np.array([]),
                    view_direction_error_deg=view_err_deg if view_err_deg is not None else np.array([]),
                    inplane_error_deg=inplane_err_deg if inplane_err_deg is not None else np.array([]),
                    gt_rotations=gt_rotations_orig if gt_rotations_orig is not None else np.array([]),
                    gt_translations=gt_translations_orig if gt_translations_orig is not None else np.array([]),
                    gt_transpose_relion_convention=np.array(
                        gt_transpose_relion_convention if gt_transpose_relion_convention is not None else False,
                        dtype=np.bool_,
                    ),
                    recovar_vs_gt_angular_error_deg=(
                        recovar_gt_ang_err_deg if recovar_gt_ang_err_deg is not None else np.array([])
                    ),
                    recovar_vs_gt_view_direction_error_deg=(
                        recovar_gt_view_err_deg if recovar_gt_view_err_deg is not None else np.array([])
                    ),
                    recovar_vs_gt_inplane_error_deg=(
                        recovar_gt_inplane_err_deg if recovar_gt_inplane_err_deg is not None else np.array([])
                    ),
                    relion_vs_gt_angular_error_deg=(
                        relion_gt_ang_err_deg if relion_gt_ang_err_deg is not None else np.array([])
                    ),
                    relion_vs_gt_view_direction_error_deg=(
                        relion_gt_view_err_deg if relion_gt_view_err_deg is not None else np.array([])
                    ),
                    relion_vs_gt_inplane_error_deg=(
                        relion_gt_inplane_err_deg if relion_gt_inplane_err_deg is not None else np.array([])
                    ),
                    recovar_translations=recovar_trans_orig if recovar_trans_orig is not None else np.array([]),
                    relion_translations=relion_trans_orig,
                    translation_error_px=trans_err_px if trans_err_px is not None else np.array([]),
                    recovar_vs_gt_translation_error_px=(
                        recovar_gt_trans_err_px if recovar_gt_trans_err_px is not None else np.array([])
                    ),
                    relion_vs_gt_translation_error_px=(
                        relion_gt_trans_err_px if relion_gt_trans_err_px is not None else np.array([])
                    ),
                    half1_indices=half1_indices,
                    half2_indices=half2_indices,
                )
                print(f"  Saved pose comparison: {pose_path}")

    if gt_ft is not None:
        print("\n=== Postprocessing per-iteration map quality vs GT/RELION ===")
        import subprocess

        gt_postprocess_cmd = build_gt_postprocess_command(
            recovar_dir=out_dir,
            relion_dir=relion_dir,
            relion_start_iter=iteration,
            relion_run_prefix=run_prefix,
            gt_volume=gt_path,
            max_iter=args.max_iter,
            intermediates_dir=save_intermediates_dir,
            gt_align=args.gt_align,
            gt_align_healpix_order=args.gt_align_healpix_order,
            gt_align_max_shell=args.gt_align_max_shell,
            gt_align_no_mirror=args.gt_align_no_mirror,
            gt_align_allow_sign=args.gt_align_allow_sign,
            gt_align_all_series=args.gt_align_all_series,
        )
        subprocess.run(gt_postprocess_cmd, check=True)

    # ---- Run diff script ----
    print("\n=== Running diff_relion_recovar_per_iter.py ===")
    import subprocess

    subprocess.run(
        [
            sys.executable,
            "scripts/diff_relion_recovar_per_iter.py",
            "--relion_dir",
            str(relion_dir),
            "--recovar_dir",
            out_dir,
            "--relion_start_iter",
            str(iteration),
            "--relion_run_prefix",
            run_prefix,
            "--max_iter",
            str(completed_iters + 1),
            "--tol",
            "0.05",
            "--shells",
            "10",
        ]
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        if (
            exc.__class__.__name__ == "SignificanceDumpComplete"
            and os.environ.get("RECOVAR_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET") == "1"
        ):
            print(
                "RECOVAR coarse-significance dump completed; stopping before "
                f"pass-2/M-step work: {exc}"
            )
            sys.exit(0)
        raise

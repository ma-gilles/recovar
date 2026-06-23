#!/usr/bin/env python3
"""Compare halfmap FSC to map-to-model FSC for uniform noise=3 state 50.

RECOVAR has state-50 halfmaps from ``compute_state``.  For RECOVAR the
map-to-model curve is computed against GT state 50 using both the filtered map
and the unfiltered halfmap average.

3DFlex has highres flex-map halfmaps rather than state-specific generated
halfmaps.  For 3DFlex the halfmap curve is converted to the expected full-map
correlation envelope, then compared to the best map-to-model curve across all
100 GT states, selected by masked FSC area under curve.  The generated state-50
mean-latent map is also scored against GT state 50 for context.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import mrcfile
import numpy as np


SWEEP_ROOT = Path("/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise3_b80_20260531")
SOURCE_ROOT = Path("/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise3_b80_20260531")
BROAD_MASK = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_direct_volume_shell_metrics_20260523/"
    "full_gt_vols_plus_masks_20260524/masks/broad_mask.mrc"
)
NOT_MOVING_MASK = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_direct_volume_shell_metrics_20260523/"
    "full_gt_vols_plus_masks_20260524/masks/not_moving_spike_mask_soft_20260601.mrc"
)
METHOD_ORDER = ("recovar", "3dflex")
N_ORDER = (10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000)
N_LABELS = {
    10_000: "10k",
    30_000: "30k",
    100_000: "100k",
    300_000: "300k",
    1_000_000: "1M",
    3_000_000: "3M",
}
N_COLORS = {
    10_000: "#4c1d7a",
    30_000: "#287796",
    100_000: "#2bb07f",
    300_000: "#c9df1a",
    1_000_000: "#ff9f1c",
    3_000_000: "#c1121f",
}
HALFMAP_EXPECTED_COLOR = "#006d2c"
MAPMODEL_GT_COLOR = "#08519c"
HALFMAP_RAW_COLOR = "#74c476"
PHASE_RANDOMIZED_COLOR = "#b35806"
MAPMODEL_RAW_COLOR = "#6baed6"
MAPMODEL_PHASE_RANDOMIZED_COLOR = "#756bb1"
DELTA_COLOR = "#4a4a4a"
CURVE_STROKE = [pe.Stroke(linewidth=7.0, foreground="white"), pe.Normal()]
DELTA_STROKE = [pe.Stroke(linewidth=6.0, foreground="white"), pe.Normal()]
THREE_DFLEX_HIGHRES_JOBS = {
    10_000: "J479",
    30_000: "J485",
    100_000: "J490",
    300_000: "J497",
    1_000_000: "J502",
    3_000_000: "J524",
}
THREE_DFLEX_STATE50_JOBS = {
    10_000: "J504",
    30_000: "J506",
    100_000: "J518",
    300_000: "J526",
    1_000_000: "J528",
    3_000_000: "J544",
}


@dataclass(frozen=True)
class MaskConfig:
    key: str
    label: str
    mask: Path


MASK_CONFIGS = {
    "broad": MaskConfig("broad", "broad_mask", BROAD_MASK),
    "notmoving": MaskConfig("notmoving", "not_moving_mask", NOT_MOVING_MASK),
}


@dataclass(frozen=True)
class RecovarMaps:
    n_images: int
    filtered: Path
    unfiltered: Path
    half1: Path
    half2: Path
    gt50: Path


@dataclass(frozen=True)
class ThreeDFlexMaps:
    n_images: int
    job_uid: str
    full_map: Path
    half_a: Path
    half_b: Path
    generated_state50: Path
    gt50: Path


def _load_mrc(path: Path) -> np.ndarray:
    with mrcfile.open(path, permissive=True) as handle:
        return np.asarray(handle.data, dtype=np.float32)


def _gt_path(n_images: int, state: int) -> Path:
    n_tag = f"n{n_images:08d}"
    return SOURCE_ROOT / n_tag / "runs" / f"{n_tag}_seed0000" / "04_ground_truth" / f"gt_vol{state:04d}.mrc"


def _recovar_maps(n_images: int) -> RecovarMaps | None:
    root = SWEEP_ROOT / "recovar" / f"n{n_images:08d}" / "compute_state_zdim4_noreg_focus" / "state0050"
    maps = RecovarMaps(
        n_images=n_images,
        filtered=root / "state000.mrc",
        unfiltered=root / "state000_unfil.mrc",
        half1=root / "state000_half1_unfil.mrc",
        half2=root / "state000_half2_unfil.mrc",
        gt50=_gt_path(n_images, 50),
    )
    return maps if all(path.exists() for path in (maps.filtered, maps.unfiltered, maps.half1, maps.half2, maps.gt50)) else None


def _three_dflex_maps(n_images: int) -> ThreeDFlexMaps | None:
    highres_job = THREE_DFLEX_HIGHRES_JOBS[n_images]
    gen_job = THREE_DFLEX_STATE50_JOBS[n_images]
    highres_root = Path(f"/projects/CRYOEM/singerlab/mg6942/CS-testres/{highres_job}")
    gen_root = Path(f"/projects/CRYOEM/singerlab/mg6942/CS-testres/{gen_job}")
    maps = ThreeDFlexMaps(
        n_images=n_images,
        job_uid=highres_job,
        full_map=highres_root / f"{highres_job}_flex_map.mrc",
        half_a=highres_root / f"{highres_job}_flex_map_half_A.mrc",
        half_b=highres_root / f"{highres_job}_flex_map_half_B.mrc",
        generated_state50=gen_root / f"{gen_job}_series_000" / f"{gen_job}_series_000_frame_002.mrc",
        gt50=_gt_path(n_images, 50),
    )
    required = (maps.full_map, maps.half_a, maps.half_b, maps.generated_state50, maps.gt50)
    return maps if all(path.exists() for path in required) else None


def _shell_labels(shape: tuple[int, int, int]) -> tuple[np.ndarray, int]:
    coords = np.indices(shape, dtype=np.float32)
    center = np.asarray(shape, dtype=np.float32)[:, None, None, None] // 2
    labels = np.rint(np.sqrt(np.sum((coords - center) ** 2, axis=0))).astype(np.int32)
    n_shells = min(shape) // 2 - 1
    return np.clip(labels, 0, n_shells - 1), n_shells


def _fft_masked(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftn(np.fft.fftshift(volume * mask)))


def _fsc_from_fts(a_ft: np.ndarray, b_ft: np.ndarray, labels: np.ndarray, n_shells: int) -> np.ndarray:
    flat = labels.ravel()
    cross = np.bincount(flat, weights=np.real(np.conj(a_ft).ravel() * b_ft.ravel()), minlength=n_shells)
    a_power = np.bincount(flat, weights=np.abs(a_ft).ravel() ** 2, minlength=n_shells)
    b_power = np.bincount(flat, weights=np.abs(b_ft).ravel() ** 2, minlength=n_shells)
    with np.errstate(divide="ignore", invalid="ignore"):
        fsc = cross / np.sqrt(a_power * b_power)
    fsc[~np.isfinite(fsc)] = 0.0
    if fsc.size > 1:
        fsc[0] = fsc[1]
    return fsc.astype(np.float32)


def _masked_fsc(a: np.ndarray, b: np.ndarray, mask: np.ndarray, labels: np.ndarray, n_shells: int) -> np.ndarray:
    return _fsc_from_fts(_fft_masked(a, mask), _fft_masked(b, mask), labels, n_shells)


def _fftfreq_radius(shape: tuple[int, ...], voxel_size: float) -> np.ndarray:
    grids = np.meshgrid(
        *[np.fft.fftfreq(size, d=voxel_size) for size in shape],
        indexing="ij",
    )
    radius = np.zeros(shape, dtype=np.float64)
    for grid in grids:
        radius += grid * grid
    return np.sqrt(radius)


def _phase_randomize_volume(
    volume: np.ndarray,
    cutoff_A: float,
    voxel_size: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomize Fourier phases above ``cutoff_A`` while preserving amplitudes.

    The random phase shifts are antisymmetrized so Hermitian symmetry is kept
    and the inverse FFT is real-valued.
    """
    if cutoff_A <= 0:
        raise ValueError(f"phase-randomization cutoff must be positive, got {cutoff_A}")
    spectrum = np.fft.fftn(np.asarray(volume, dtype=np.float32))
    radius = _fftfreq_radius(volume.shape, voxel_size)
    high = radius >= (1.0 / cutoff_A)
    flat_index = np.arange(volume.size, dtype=np.int64).reshape(volume.shape)
    neg_indexer = np.ix_(*[((-np.arange(size)) % size) for size in volume.shape])
    neg_flat_index = flat_index[neg_indexer]
    canonical = (flat_index < neg_flat_index) & high
    phases = np.zeros(volume.shape, dtype=np.float32)
    coords = np.nonzero(canonical)
    theta = rng.uniform(-np.pi, np.pi, size=coords[0].size).astype(np.float32)
    phases[coords] = theta
    neg_coords = tuple(((-coord) % size) for coord, size in zip(coords, volume.shape, strict=True))
    phases[neg_coords] = -theta
    randomized = spectrum.copy()
    randomized[high] *= np.exp(1j * phases[high])
    return np.fft.ifftn(randomized).real.astype(np.float32)


def _phase_randomized_masked_fsc(
    half1: np.ndarray,
    half2: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    n_shells: int,
    cutoff_A: float,
    voxel_size: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    randomized1 = _phase_randomize_volume(half1, cutoff_A, voxel_size, rng)
    randomized2 = _phase_randomize_volume(half2, cutoff_A, voxel_size, rng)
    return _masked_fsc(randomized1, randomized2, mask, labels, n_shells)


def _mask_corrected_fsc(
    masked_fsc: np.ndarray,
    randomized_masked_fsc: np.ndarray,
    frequency: np.ndarray,
    cutoff_A: float,
) -> np.ndarray:
    corrected = np.asarray(masked_fsc, dtype=np.float32).copy()
    high = frequency >= (1.0 / cutoff_A)
    denominator = 1.0 - randomized_masked_fsc
    with np.errstate(divide="ignore", invalid="ignore"):
        corrected_high = (masked_fsc - randomized_masked_fsc) / denominator
    corrected[high] = corrected_high[high]
    corrected[~np.isfinite(corrected)] = 0.0
    return np.clip(corrected, -1.0, 1.0).astype(np.float32)


def _high_shell_only(curve: np.ndarray, frequency: np.ndarray, cutoff_A: float) -> np.ndarray:
    out = np.asarray(curve, dtype=np.float32).copy()
    out[frequency < (1.0 / cutoff_A)] = np.nan
    return out


def _resolution(frequency: np.ndarray, curve: np.ndarray, threshold: float) -> float:
    valid = np.isfinite(frequency) & np.isfinite(curve) & (frequency > 0)
    x = np.asarray(frequency[valid], dtype=np.float64)
    y = np.asarray(curve[valid], dtype=np.float64)
    if x.size == 0:
        return math.nan
    below = np.flatnonzero(y < threshold)
    if below.size == 0:
        return float(1.0 / x[-1])
    idx = int(below[0])
    if idx == 0:
        return math.nan
    x0, x1 = x[idx - 1], x[idx]
    y0, y1 = y[idx - 1], y[idx]
    cross = x1 if y1 == y0 else x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
    return float(1.0 / cross) if cross > 0 else math.nan


def _auc(curve: np.ndarray, frequency: np.ndarray) -> float:
    valid = np.isfinite(curve) & np.isfinite(frequency) & (frequency > 0)
    if not np.any(valid):
        return math.nan
    return float(np.nanmean(curve[valid]))


def _rh_power_from_halfmap(halfmap_fsc: np.ndarray) -> np.ndarray:
    """Squared full-map/true-map correlation: 2h/(1+h).

    The plotted map-model-scale envelope is the square root of this quantity.
    """
    clipped = np.clip(halfmap_fsc, -0.999999, 0.999999)
    out = 2.0 * clipped / (1.0 + clipped)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _rh_corr_from_halfmap(halfmap_fsc: np.ndarray) -> np.ndarray:
    power = np.clip(_rh_power_from_halfmap(halfmap_fsc), 0.0, 1.0)
    return np.sqrt(power).astype(np.float32)


def _write_shell_csv(
    path: Path,
    frequency: np.ndarray,
    curves: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["shell", "frequency_1_per_A", "resolution_A", *curves.keys()]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for shell, freq in enumerate(frequency):
            writer.writerow(
                [
                    shell,
                    freq,
                    math.inf if freq <= 0 else 1.0 / freq,
                    *[curves[name][shell] for name in curves],
                ]
            )


def _score_recovar(
    maps: RecovarMaps,
    mask: np.ndarray,
    labels: np.ndarray,
    n_shells: int,
    frequency: np.ndarray,
    out_dir: Path,
    mask_config: MaskConfig,
    voxel_size: float,
    phase_randomization_cutoff_A: float,
    phase_randomization_seed: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    half1 = _load_mrc(maps.half1)
    half2 = _load_mrc(maps.half2)
    filtered = _load_mrc(maps.filtered)
    unfiltered = _load_mrc(maps.unfiltered)
    gt = _load_mrc(maps.gt50)
    halfmap = _masked_fsc(half1, half2, mask, labels, n_shells)
    predicted_power = _rh_power_from_halfmap(halfmap)
    predicted_corr = _rh_corr_from_halfmap(halfmap)
    phase_randomized = _phase_randomized_masked_fsc(
        half1,
        half2,
        mask,
        labels,
        n_shells,
        phase_randomization_cutoff_A,
        voxel_size,
        phase_randomization_seed + maps.n_images + 17,
    )
    corrected_halfmap = _mask_corrected_fsc(halfmap, phase_randomized, frequency, phase_randomization_cutoff_A)
    corrected_predicted_power = _rh_power_from_halfmap(corrected_halfmap)
    corrected_predicted_corr = _rh_corr_from_halfmap(corrected_halfmap)
    mapmodel_filtered = _masked_fsc(filtered, gt, mask, labels, n_shells)
    mapmodel_unfiltered = _masked_fsc(unfiltered, gt, mask, labels, n_shells)
    mapmodel_filtered_phase_randomized = _phase_randomized_masked_fsc(
        filtered,
        gt,
        mask,
        labels,
        n_shells,
        phase_randomization_cutoff_A,
        voxel_size,
        phase_randomization_seed + maps.n_images + 200017,
    )
    mapmodel_unfiltered_phase_randomized = _phase_randomized_masked_fsc(
        unfiltered,
        gt,
        mask,
        labels,
        n_shells,
        phase_randomization_cutoff_A,
        voxel_size,
        phase_randomization_seed + maps.n_images + 300017,
    )
    corrected_mapmodel_filtered = _mask_corrected_fsc(
        mapmodel_filtered,
        mapmodel_filtered_phase_randomized,
        frequency,
        phase_randomization_cutoff_A,
    )
    corrected_mapmodel_unfiltered = _mask_corrected_fsc(
        mapmodel_unfiltered,
        mapmodel_unfiltered_phase_randomized,
        frequency,
        phase_randomization_cutoff_A,
    )
    curves = {
        "halfmap_fsc": halfmap,
        "phase_randomized_masked_fsc": phase_randomized,
        "phase_randomized_masked_fsc_high_shells": _high_shell_only(
            phase_randomized, frequency, phase_randomization_cutoff_A
        ),
        "corrected_halfmap_fsc": corrected_halfmap,
        "expected_fullmap_corr_sq_2h_over_1_plus_h": predicted_power,
        "fsc_full_sqrt_expected_corr": predicted_corr,
        "corrected_expected_fullmap_corr_sq_2h_over_1_plus_h": corrected_predicted_power,
        "corrected_fsc_full_sqrt_expected_corr": corrected_predicted_corr,
        "mapmodel_fsc_filtered_vs_gt50": mapmodel_filtered,
        "mapmodel_fsc_unfiltered_vs_gt50": mapmodel_unfiltered,
        "mapmodel_filtered_phase_randomized_masked_fsc": mapmodel_filtered_phase_randomized,
        "mapmodel_filtered_phase_randomized_masked_fsc_high_shells": _high_shell_only(
            mapmodel_filtered_phase_randomized, frequency, phase_randomization_cutoff_A
        ),
        "corrected_mapmodel_fsc_filtered_vs_gt50": corrected_mapmodel_filtered,
        "mapmodel_unfiltered_phase_randomized_masked_fsc": mapmodel_unfiltered_phase_randomized,
        "mapmodel_unfiltered_phase_randomized_masked_fsc_high_shells": _high_shell_only(
            mapmodel_unfiltered_phase_randomized, frequency, phase_randomization_cutoff_A
        ),
        "corrected_mapmodel_fsc_unfiltered_vs_gt50": corrected_mapmodel_unfiltered,
    }
    shell_csv = out_dir / "shell_metrics" / "recovar" / f"n{maps.n_images:08d}" / "state0050_halfmap_vs_mapmodel.csv"
    _write_shell_csv(shell_csv, frequency, curves)
    row = {
        "method": "recovar",
        "n_images": maps.n_images,
        "n_label": N_LABELS[maps.n_images],
        "halfmap_threshold": 1.0 / 7.0,
        "phase_randomization_cutoff_A": phase_randomization_cutoff_A,
        "phase_randomization_cutoff_frequency_1_per_A": 1.0 / phase_randomization_cutoff_A,
        "halfmap_resolution_0p143_A": _resolution(frequency, halfmap, 1.0 / 7.0),
        "corrected_halfmap_resolution_0p143_A": _resolution(frequency, corrected_halfmap, 1.0 / 7.0),
        "rh_corr_resolution_0p5_A": _resolution(frequency, predicted_corr, 0.5),
        "rh_power_resolution_0p5_A": _resolution(frequency, predicted_power, 0.5),
        "corrected_rh_corr_resolution_0p5_A": _resolution(frequency, corrected_predicted_corr, 0.5),
        "corrected_rh_power_resolution_0p5_A": _resolution(frequency, corrected_predicted_power, 0.5),
        "phase_randomized_masked_fsc_high_shell_auc": _auc(
            _high_shell_only(phase_randomized, frequency, phase_randomization_cutoff_A), frequency
        ),
        "mapmodel_filtered_resolution_0p5_A": _resolution(frequency, mapmodel_filtered, 0.5),
        "mapmodel_unfiltered_resolution_0p5_A": _resolution(frequency, mapmodel_unfiltered, 0.5),
        "corrected_mapmodel_filtered_resolution_0p5_A": _resolution(frequency, corrected_mapmodel_filtered, 0.5),
        "corrected_mapmodel_unfiltered_resolution_0p5_A": _resolution(frequency, corrected_mapmodel_unfiltered, 0.5),
        "mapmodel_filtered_phase_randomized_masked_fsc_high_shell_auc": _auc(
            _high_shell_only(mapmodel_filtered_phase_randomized, frequency, phase_randomization_cutoff_A), frequency
        ),
        "best_gt_state": 50,
        "best_gt_auc": _auc(corrected_mapmodel_filtered, frequency),
        "half1": str(maps.half1),
        "half2": str(maps.half2),
        "full_map": str(maps.filtered),
        "unfiltered_map": str(maps.unfiltered),
        "gt": str(maps.gt50),
        "mask": str(mask_config.mask),
        "mask_mode": mask_config.key,
        "shell_metrics_csv": str(shell_csv),
    }
    return row, curves


def _score_3dflex(
    maps: ThreeDFlexMaps,
    mask: np.ndarray,
    labels: np.ndarray,
    n_shells: int,
    frequency: np.ndarray,
    out_dir: Path,
    gt_state_fts: dict[int, tuple[Path, np.ndarray]],
    mask_config: MaskConfig,
    voxel_size: float,
    phase_randomization_cutoff_A: float,
    phase_randomization_seed: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    half_a = _load_mrc(maps.half_a)
    half_b = _load_mrc(maps.half_b)
    full_map = _load_mrc(maps.full_map)
    generated = _load_mrc(maps.generated_state50)
    gt50 = _load_mrc(maps.gt50)
    full_ft = _fft_masked(full_map, mask)
    halfmap = _masked_fsc(half_a, half_b, mask, labels, n_shells)
    predicted_power = _rh_power_from_halfmap(halfmap)
    predicted_corr = _rh_corr_from_halfmap(halfmap)
    phase_randomized = _phase_randomized_masked_fsc(
        half_a,
        half_b,
        mask,
        labels,
        n_shells,
        phase_randomization_cutoff_A,
        voxel_size,
        phase_randomization_seed + maps.n_images + 1000003,
    )
    corrected_halfmap = _mask_corrected_fsc(halfmap, phase_randomized, frequency, phase_randomization_cutoff_A)
    corrected_predicted_power = _rh_power_from_halfmap(corrected_halfmap)
    corrected_predicted_corr = _rh_corr_from_halfmap(corrected_halfmap)
    state50_generated = _masked_fsc(generated, gt50, mask, labels, n_shells)
    best_state = -1
    best_auc = -math.inf
    best_curve: np.ndarray | None = None
    best_gt_path: Path | None = None
    for state, (gt_path, gt_ft) in gt_state_fts.items():
        curve = _fsc_from_fts(full_ft, gt_ft, labels, n_shells)
        auc = _auc(curve, frequency)
        if auc > best_auc:
            best_auc = auc
            best_state = state
            best_curve = curve
            best_gt_path = gt_path
    if best_curve is None or best_gt_path is None:
        raise RuntimeError(f"no GT state comparison succeeded for {maps.job_uid}")
    best_gt = _load_mrc(best_gt_path)
    best_mapmodel_phase_randomized = _phase_randomized_masked_fsc(
        full_map,
        best_gt,
        mask,
        labels,
        n_shells,
        phase_randomization_cutoff_A,
        voxel_size,
        phase_randomization_seed + maps.n_images + 1200003,
    )
    corrected_best_curve = _mask_corrected_fsc(
        best_curve,
        best_mapmodel_phase_randomized,
        frequency,
        phase_randomization_cutoff_A,
    )
    generated_state50_phase_randomized = _phase_randomized_masked_fsc(
        generated,
        gt50,
        mask,
        labels,
        n_shells,
        phase_randomization_cutoff_A,
        voxel_size,
        phase_randomization_seed + maps.n_images + 1300003,
    )
    corrected_state50_generated = _mask_corrected_fsc(
        state50_generated,
        generated_state50_phase_randomized,
        frequency,
        phase_randomization_cutoff_A,
    )
    curves = {
        "halfmap_fsc": halfmap,
        "phase_randomized_masked_fsc": phase_randomized,
        "phase_randomized_masked_fsc_high_shells": _high_shell_only(
            phase_randomized, frequency, phase_randomization_cutoff_A
        ),
        "corrected_halfmap_fsc": corrected_halfmap,
        "expected_fullmap_corr_sq_2h_over_1_plus_h": predicted_power,
        "fsc_full_sqrt_expected_corr": predicted_corr,
        "corrected_expected_fullmap_corr_sq_2h_over_1_plus_h": corrected_predicted_power,
        "corrected_fsc_full_sqrt_expected_corr": corrected_predicted_corr,
        "best_gt_state_mapmodel_fsc": best_curve,
        "best_gt_state_mapmodel_phase_randomized_masked_fsc": best_mapmodel_phase_randomized,
        "best_gt_state_mapmodel_phase_randomized_masked_fsc_high_shells": _high_shell_only(
            best_mapmodel_phase_randomized, frequency, phase_randomization_cutoff_A
        ),
        "corrected_best_gt_state_mapmodel_fsc": corrected_best_curve,
        "generated_state50_mapmodel_fsc_vs_gt50": state50_generated,
        "generated_state50_mapmodel_phase_randomized_masked_fsc": generated_state50_phase_randomized,
        "generated_state50_mapmodel_phase_randomized_masked_fsc_high_shells": _high_shell_only(
            generated_state50_phase_randomized, frequency, phase_randomization_cutoff_A
        ),
        "corrected_generated_state50_mapmodel_fsc_vs_gt50": corrected_state50_generated,
    }
    shell_csv = out_dir / "shell_metrics" / "3dflex" / f"n{maps.n_images:08d}" / "highres_halfmap_vs_best_gt.csv"
    _write_shell_csv(shell_csv, frequency, curves)
    row = {
        "method": "3dflex",
        "n_images": maps.n_images,
        "n_label": N_LABELS[maps.n_images],
        "job_uid": maps.job_uid,
        "halfmap_threshold": 1.0 / 7.0,
        "phase_randomization_cutoff_A": phase_randomization_cutoff_A,
        "phase_randomization_cutoff_frequency_1_per_A": 1.0 / phase_randomization_cutoff_A,
        "halfmap_resolution_0p143_A": _resolution(frequency, halfmap, 1.0 / 7.0),
        "corrected_halfmap_resolution_0p143_A": _resolution(frequency, corrected_halfmap, 1.0 / 7.0),
        "rh_corr_resolution_0p5_A": _resolution(frequency, predicted_corr, 0.5),
        "rh_power_resolution_0p5_A": _resolution(frequency, predicted_power, 0.5),
        "corrected_rh_corr_resolution_0p5_A": _resolution(frequency, corrected_predicted_corr, 0.5),
        "corrected_rh_power_resolution_0p5_A": _resolution(frequency, corrected_predicted_power, 0.5),
        "phase_randomized_masked_fsc_high_shell_auc": _auc(
            _high_shell_only(phase_randomized, frequency, phase_randomization_cutoff_A), frequency
        ),
        "mapmodel_best_gt_resolution_0p5_A": _resolution(frequency, best_curve, 0.5),
        "generated_state50_resolution_0p5_A": _resolution(frequency, state50_generated, 0.5),
        "corrected_mapmodel_best_gt_resolution_0p5_A": _resolution(frequency, corrected_best_curve, 0.5),
        "corrected_generated_state50_resolution_0p5_A": _resolution(frequency, corrected_state50_generated, 0.5),
        "mapmodel_best_gt_phase_randomized_masked_fsc_high_shell_auc": _auc(
            _high_shell_only(best_mapmodel_phase_randomized, frequency, phase_randomization_cutoff_A), frequency
        ),
        "best_gt_state": best_state,
        "best_gt_auc": _auc(corrected_best_curve, frequency),
        "half1": str(maps.half_a),
        "half2": str(maps.half_b),
        "full_map": str(maps.full_map),
        "generated_state50_map": str(maps.generated_state50),
        "best_gt": str(best_gt_path),
        "gt_state50": str(maps.gt50),
        "mask": str(mask_config.mask),
        "mask_mode": mask_config.key,
        "shell_metrics_csv": str(shell_csv),
    }
    return row, curves


def _precompute_gt_state_fts(
    reference_n_images: int,
    max_state: int,
    mask: np.ndarray,
) -> dict[int, tuple[Path, np.ndarray]]:
    out: dict[int, tuple[Path, np.ndarray]] = {}
    for state in range(max_state + 1):
        gt_path = _gt_path(reference_n_images, state)
        if not gt_path.exists():
            continue
        out[state] = (gt_path, _fft_masked(_load_mrc(gt_path), mask))
    if not out:
        raise RuntimeError(f"no GT volumes found for n={reference_n_images}")
    return out


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _raw_mapmodel_curve_name(method: str) -> str:
    if method == "recovar":
        return "mapmodel_fsc_filtered_vs_gt50"
    if method == "3dflex":
        return "best_gt_state_mapmodel_fsc"
    raise ValueError(f"unknown method: {method}")


def _plot_curves(
    out_dir: Path,
    frequency: np.ndarray,
    rows: list[dict[str, object]],
    curves_by_key: dict[tuple[str, int], dict[str, np.ndarray]],
    mask_config: MaskConfig,
    plot_n_images: tuple[int, ...] | None,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    n_suffix = ""
    n_title = ""
    if plot_n_images is not None:
        labels = [N_LABELS[n] for n in plot_n_images]
        n_suffix = "_" + "_".join(f"n{n:08d}" for n in plot_n_images) + "_only"
        n_title = " | " + ", ".join(labels) + " only"
    for method in METHOD_ORDER:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        fig, ax = plt.subplots(figsize=(8.8, 5.4), dpi=220)
        cutoff_freq = None
        for row in sorted(method_rows, key=lambda r: int(r["n_images"])):
            n_images = int(row["n_images"])
            curves = curves_by_key[(method, n_images)]
            label_prefix = N_LABELS[n_images]
            cutoff_freq = float(row.get("phase_randomization_cutoff_frequency_1_per_A", math.nan))
            if method == "recovar":
                raw_actual_name = "mapmodel_fsc_filtered_vs_gt50"
                corrected_actual_name = "corrected_mapmodel_fsc_filtered_vs_gt50"
                actual_randomized_name = "mapmodel_filtered_phase_randomized_masked_fsc_high_shells"
            else:
                raw_actual_name = "best_gt_state_mapmodel_fsc"
                corrected_actual_name = "corrected_best_gt_state_mapmodel_fsc"
                actual_randomized_name = "best_gt_state_mapmodel_phase_randomized_masked_fsc_high_shells"
            ax.plot(
                frequency,
                curves[corrected_actual_name],
                color=MAPMODEL_GT_COLOR,
                lw=3.8,
                label=f"{label_prefix} corrected map-model",
            )
            ax.plot(
                frequency,
                curves[raw_actual_name],
                color=MAPMODEL_RAW_COLOR,
                lw=2.2,
                ls=":",
                alpha=0.90,
                label=f"{label_prefix} raw map-model",
            )
            ax.plot(
                frequency,
                curves[actual_randomized_name],
                color=MAPMODEL_PHASE_RANDOMIZED_COLOR,
                lw=2.0,
                ls="-.",
                alpha=0.82,
                label=f"{label_prefix} map-model phase-rand FSC",
            )
            ax.plot(
                frequency,
                curves["corrected_fsc_full_sqrt_expected_corr"],
                color=HALFMAP_EXPECTED_COLOR,
                lw=3.4,
                ls="--",
                alpha=0.95,
                label=f"{label_prefix} corrected FSC_full_sqrt",
            )
            ax.plot(
                frequency,
                curves["fsc_full_sqrt_expected_corr"],
                color=HALFMAP_RAW_COLOR,
                lw=2.2,
                ls=":",
                alpha=0.90,
                label=f"{label_prefix} raw FSC_full_sqrt",
            )
            ax.plot(
                frequency,
                curves["phase_randomized_masked_fsc_high_shells"],
                color=PHASE_RANDOMIZED_COLOR,
                lw=2.2,
                ls="-.",
                alpha=0.90,
                label=f"{label_prefix} phase-rand masked FSC",
            )
        ax.axhline(0.5, color="0.35", ls=":", lw=1.2)
        if cutoff_freq is not None and math.isfinite(cutoff_freq):
            ax.axvline(cutoff_freq, color="0.45", ls="--", lw=1.0, alpha=0.65)
        ax.set_xlim(0.0, 0.40)
        ax.set_ylim(-0.04, 1.03)
        ax.set_xlabel("spatial frequency (1/A)")
        ax.set_ylabel("FSC / predicted correlation")
        ax.grid(True, color="0.86", lw=0.7)
        ax.legend(fontsize=7.5, ncol=2)
        title = (
            f"RECOVAR state-50 halfmaps vs GT50 | {mask_config.label}"
            if method == "recovar"
            else f"3DFlex highres halfmaps vs best GT state | {mask_config.label}"
        )
        ax.set_title(f"{title}{n_title}", weight="bold")
        fig.tight_layout()
        png = out_dir / f"uniform_noise3_state50_{method}_halfmap_vs_mapmodel_curves{n_suffix}.png"
        fig.savefig(png)
        fig.savefig(png.with_suffix(".pdf"))
        plt.close(fig)
        outputs[f"{method}_curves_png"] = str(png)

    fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=220)
    x = np.arange(len(N_ORDER), dtype=np.float64)
    offsets = {"recovar": -0.17, "3dflex": 0.17}
    for method in METHOD_ORDER:
        method_rows = {int(row["n_images"]): row for row in rows if row["method"] == method}
        actual: list[float] = []
        predicted: list[float] = []
        for n_images in N_ORDER:
            row = method_rows.get(n_images)
            if row is None:
                actual.append(math.nan)
                predicted.append(math.nan)
            elif method == "recovar":
                actual.append(float(row["corrected_mapmodel_filtered_resolution_0p5_A"]))
                predicted.append(float(row["corrected_rh_corr_resolution_0p5_A"]))
            else:
                actual.append(float(row["corrected_mapmodel_best_gt_resolution_0p5_A"]))
                predicted.append(float(row["corrected_rh_corr_resolution_0p5_A"]))
        color = "#1b9e77" if method == "recovar" else "#7570b3"
        ax.plot(x + offsets[method], actual, marker="o", lw=2.4, color=color, label=f"{method} corrected map-model")
        ax.plot(
            x + offsets[method],
            predicted,
            marker="s",
            lw=1.8,
            ls="--",
            color=color,
            alpha=0.75,
            label=f"{method} corrected halfmap->corr",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([N_LABELS[n] for n in N_ORDER])
    ax.set_ylabel("FSC0.5 resolution (A)")
    ax.invert_yaxis()
    ax.grid(True, color="0.86", lw=0.7)
    ax.legend(fontsize=8)
    ax.set_title(
        f"Uniform noise=3 state 50: halfmap prediction vs map-to-model resolution | {mask_config.label}{n_title}",
        weight="bold",
    )
    fig.tight_layout()
    png = out_dir / f"uniform_noise3_state50_halfmap_vs_mapmodel_resolution_summary{n_suffix}.png"
    fig.savefig(png)
    fig.savefig(png.with_suffix(".pdf"))
    plt.close(fig)
    outputs["resolution_summary_png"] = str(png)
    outputs.update(_plot_full_sqrt_same_scale(out_dir, frequency, rows, curves_by_key, mask_config, plot_n_images))
    outputs.update(_plot_full_sqrt_delta(out_dir, frequency, rows, curves_by_key, mask_config, plot_n_images))
    outputs.update(_plot_raw_only_same_scale(out_dir, frequency, rows, curves_by_key, mask_config, plot_n_images))
    outputs.update(_plot_raw_only_delta(out_dir, frequency, rows, curves_by_key, mask_config, plot_n_images))
    return outputs


def _plot_full_sqrt_same_scale(
    out_dir: Path,
    frequency: np.ndarray,
    rows: list[dict[str, object]],
    curves_by_key: dict[tuple[str, int], dict[str, np.ndarray]],
    mask_config: MaskConfig,
    plot_n_images: tuple[int, ...] | None,
) -> dict[str, str]:
    n_suffix = ""
    n_title = ""
    if plot_n_images is not None:
        labels = [N_LABELS[n] for n in plot_n_images]
        n_suffix = "_" + "_".join(f"n{n:08d}" for n in plot_n_images) + "_only"
        n_title = " | " + ", ".join(labels) + " only"
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.3), dpi=240, sharex=True, sharey=True)
    for ax, method in zip(axes, METHOD_ORDER, strict=True):
        method_rows = [row for row in rows if row["method"] == method]
        cutoff_freq = None
        for row in sorted(method_rows, key=lambda r: int(r["n_images"])):
            n_images = int(row["n_images"])
            curves = curves_by_key[(method, n_images)]
            cutoff_freq = float(row.get("phase_randomization_cutoff_frequency_1_per_A", math.nan))
            if method == "recovar":
                raw_actual_name = "mapmodel_fsc_filtered_vs_gt50"
                corrected_actual_name = "corrected_mapmodel_fsc_filtered_vs_gt50"
                actual_randomized_name = "mapmodel_filtered_phase_randomized_masked_fsc_high_shells"
            else:
                raw_actual_name = "best_gt_state_mapmodel_fsc"
                corrected_actual_name = "corrected_best_gt_state_mapmodel_fsc"
                actual_randomized_name = "best_gt_state_mapmodel_phase_randomized_masked_fsc_high_shells"
            label = N_LABELS[n_images]
            ax.plot(
                frequency,
                curves["corrected_fsc_full_sqrt_expected_corr"],
                color=HALFMAP_EXPECTED_COLOR,
                lw=3.4,
                ls="--",
                alpha=0.95,
            )
            ax.plot(
                frequency,
                curves["fsc_full_sqrt_expected_corr"],
                color=HALFMAP_RAW_COLOR,
                lw=2.2,
                ls=":",
                alpha=0.90,
            )
            ax.plot(
                frequency,
                curves["phase_randomized_masked_fsc_high_shells"],
                color=PHASE_RANDOMIZED_COLOR,
                lw=2.0,
                ls="-.",
                alpha=0.82,
            )
            ax.plot(
                frequency,
                curves[corrected_actual_name],
                color=MAPMODEL_GT_COLOR,
                lw=3.8,
                label=label,
            )
            ax.plot(
                frequency,
                curves[raw_actual_name],
                color=MAPMODEL_RAW_COLOR,
                lw=2.2,
                ls=":",
                alpha=0.90,
            )
            ax.plot(
                frequency,
                curves[actual_randomized_name],
                color=MAPMODEL_PHASE_RANDOMIZED_COLOR,
                lw=2.0,
                ls="-.",
                alpha=0.82,
            )
        ax.axhline(0.5, color="0.25", ls=":", lw=1.2)
        if cutoff_freq is not None and math.isfinite(cutoff_freq):
            ax.axvline(cutoff_freq, color="0.45", ls="--", lw=1.0, alpha=0.65)
        ax.set_xlim(0.0, 0.40)
        ax.set_ylim(-0.04, 1.03)
        ax.grid(True, color="0.86", lw=0.7)
        ax.set_title("RECOVAR" if method == "recovar" else "3DFlex", weight="bold")
        ax.set_xlabel("spatial frequency (1/A)")
        ax.legend(title="n images", fontsize=8, title_fontsize=9, ncol=2)
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
            spine.set_color("0.18")
    axes[0].set_ylabel("map-model FSC and expected full-map correlation")
    fig.suptitle(
        f"Uniform noise=3 state 50 | {mask_config.label}{n_title} | corrected map-model and corrected FSC_full_sqrt",
        weight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = out_dir / f"uniform_noise3_state50_fsc_full_sqrt_same_scale{n_suffix}.png"
    fig.savefig(png)
    fig.savefig(png.with_suffix(".pdf"))
    plt.close(fig)
    return {"fsc_full_sqrt_same_scale_png": str(png), "fsc_full_sqrt_same_scale_pdf": str(png.with_suffix(".pdf"))}


def _plot_full_sqrt_delta(
    out_dir: Path,
    frequency: np.ndarray,
    rows: list[dict[str, object]],
    curves_by_key: dict[tuple[str, int], dict[str, np.ndarray]],
    mask_config: MaskConfig,
    plot_n_images: tuple[int, ...] | None,
) -> dict[str, str]:
    n_suffix = ""
    n_title = ""
    if plot_n_images is not None:
        labels = [N_LABELS[n] for n in plot_n_images]
        n_suffix = "_" + "_".join(f"n{n:08d}" for n in plot_n_images) + "_only"
        n_title = " | " + ", ".join(labels) + " only"
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14.4, 8.0),
        dpi=240,
        sharex="col",
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.10, "wspace": 0.13},
    )
    for col, method in enumerate(METHOD_ORDER):
        ax_curve = axes[0, col]
        ax_delta = axes[1, col]
        method_rows = [row for row in rows if row["method"] == method]
        deltas: list[np.ndarray] = []
        cutoff_freq = None
        for row in sorted(method_rows, key=lambda r: int(r["n_images"])):
            n_images = int(row["n_images"])
            curves = curves_by_key[(method, n_images)]
            cutoff_freq = float(row.get("phase_randomization_cutoff_frequency_1_per_A", math.nan))
            if method == "recovar":
                raw_actual_name = "mapmodel_fsc_filtered_vs_gt50"
                corrected_actual_name = "corrected_mapmodel_fsc_filtered_vs_gt50"
                actual_randomized_name = "mapmodel_filtered_phase_randomized_masked_fsc_high_shells"
            else:
                raw_actual_name = "best_gt_state_mapmodel_fsc"
                corrected_actual_name = "corrected_best_gt_state_mapmodel_fsc"
                actual_randomized_name = "best_gt_state_mapmodel_phase_randomized_masked_fsc_high_shells"
            expected = curves["corrected_fsc_full_sqrt_expected_corr"]
            actual = curves[corrected_actual_name]
            label = N_LABELS[n_images]
            ax_curve.plot(
                frequency,
                expected,
                color=HALFMAP_EXPECTED_COLOR,
                lw=3.4,
                ls="--",
                alpha=0.95,
                label=f"{label} corrected FSC_full_sqrt",
            )
            ax_curve.plot(
                frequency,
                curves["fsc_full_sqrt_expected_corr"],
                color=HALFMAP_RAW_COLOR,
                lw=2.2,
                ls=":",
                alpha=0.90,
                label=f"{label} raw FSC_full_sqrt",
            )
            ax_curve.plot(
                frequency,
                curves["phase_randomized_masked_fsc_high_shells"],
                color=PHASE_RANDOMIZED_COLOR,
                lw=2.0,
                ls="-.",
                alpha=0.82,
                label=f"{label} phase-rand masked FSC",
            )
            ax_curve.plot(
                frequency,
                actual,
                color=MAPMODEL_GT_COLOR,
                lw=3.8,
                label=f"{label} corrected map-model",
            )
            ax_curve.plot(
                frequency,
                curves[raw_actual_name],
                color=MAPMODEL_RAW_COLOR,
                lw=2.2,
                ls=":",
                alpha=0.90,
                label=f"{label} raw map-model",
            )
            ax_curve.plot(
                frequency,
                curves[actual_randomized_name],
                color=MAPMODEL_PHASE_RANDOMIZED_COLOR,
                lw=2.0,
                ls="-.",
                alpha=0.82,
                label=f"{label} map-model phase-rand FSC",
            )
            delta = actual - expected
            deltas.append(delta)
            ax_delta.plot(
                frequency,
                delta,
                color=DELTA_COLOR,
                lw=3.0,
                label=label,
            )
        ax_curve.axhline(0.5, color="0.25", ls=":", lw=1.2)
        if cutoff_freq is not None and math.isfinite(cutoff_freq):
            ax_curve.axvline(cutoff_freq, color="0.45", ls="--", lw=1.0, alpha=0.65)
            ax_delta.axvline(cutoff_freq, color="0.45", ls="--", lw=1.0, alpha=0.65)
        ax_curve.set_xlim(0.0, 0.40)
        ax_curve.set_ylim(-0.04, 1.03)
        ax_curve.grid(True, color="0.86", lw=0.7)
        ax_curve.set_title("RECOVAR" if method == "recovar" else "3DFlex", weight="bold")
        ax_curve.legend(fontsize=7.2, ncol=1)
        ax_delta.axhline(0.0, color="0.25", ls=":", lw=1.2)
        ax_delta.set_xlim(0.0, 0.40)
        if deltas:
            valid = (frequency >= 0.025) & (frequency <= 0.36)
            max_abs = max(float(np.nanmax(np.abs(delta[valid]))) for delta in deltas)
            ylim = max(0.08, min(0.60, 1.15 * max_abs))
            ax_delta.set_ylim(-ylim, ylim)
        ax_delta.grid(True, color="0.86", lw=0.7)
        ax_delta.set_xlabel("spatial frequency (1/A)")
        ax_delta.set_ylabel("corrected map-model\nminus corrected halfmap")
        for ax in (ax_curve, ax_delta):
            for spine in ax.spines.values():
                spine.set_linewidth(0.9)
                spine.set_color("0.18")
    axes[0, 0].set_ylabel("FSC / expected correlation")
    fig.suptitle(
        f"Uniform noise=3 state 50 | {mask_config.label}{n_title} | phase-randomization-corrected map-model and halfmap",
        weight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = out_dir / f"uniform_noise3_state50_fsc_full_sqrt_same_scale_with_delta{n_suffix}.png"
    fig.savefig(png)
    fig.savefig(png.with_suffix(".pdf"))
    plt.close(fig)
    return {
        "fsc_full_sqrt_same_scale_with_delta_png": str(png),
        "fsc_full_sqrt_same_scale_with_delta_pdf": str(png.with_suffix(".pdf")),
    }


def _plot_raw_only_same_scale(
    out_dir: Path,
    frequency: np.ndarray,
    rows: list[dict[str, object]],
    curves_by_key: dict[tuple[str, int], dict[str, np.ndarray]],
    mask_config: MaskConfig,
    plot_n_images: tuple[int, ...] | None,
) -> dict[str, str]:
    n_suffix = ""
    n_title = ""
    if plot_n_images is not None:
        labels = [N_LABELS[n] for n in plot_n_images]
        n_suffix = "_" + "_".join(f"n{n:08d}" for n in plot_n_images) + "_only"
        n_title = " | " + ", ".join(labels) + " only"
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.3), dpi=260, sharex=True, sharey=True)
    for ax, method in zip(axes, METHOD_ORDER, strict=True):
        method_rows = [row for row in rows if row["method"] == method]
        for row in sorted(method_rows, key=lambda r: int(r["n_images"])):
            n_images = int(row["n_images"])
            curves = curves_by_key[(method, n_images)]
            label = N_LABELS[n_images]
            actual = curves[_raw_mapmodel_curve_name(method)]
            expected = curves["fsc_full_sqrt_expected_corr"]
            ax.plot(
                frequency,
                actual,
                color=MAPMODEL_GT_COLOR,
                lw=4.4,
                ls="-",
                alpha=0.98,
                zorder=2,
                path_effects=CURVE_STROKE,
                label=f"{label} map to GT",
            )
            ax.plot(
                frequency,
                expected,
                color=HALFMAP_EXPECTED_COLOR,
                lw=5.0,
                ls=(0, (7, 3)),
                alpha=0.98,
                zorder=3,
                path_effects=CURVE_STROKE,
                label=f"{label} halfmap -> expected",
            )
        ax.axhline(0.5, color="0.25", ls=":", lw=1.4)
        ax.set_xlim(0.0, 0.40)
        ax.set_ylim(-0.04, 1.03)
        ax.grid(True, color="0.84", lw=0.8)
        ax.set_title("RECOVAR" if method == "recovar" else "3DFlex", weight="bold", fontsize=16)
        ax.set_xlabel("spatial frequency (1/A)", fontsize=12)
        ax.legend(fontsize=9.5, ncol=1, framealpha=0.90)
        ax.tick_params(axis="both", labelsize=11)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color("0.16")
    axes[0].set_ylabel("raw FSC / expected correlation", fontsize=12)
    fig.suptitle(
        f"Uniform noise=3 state 50 | {mask_config.label}{n_title} | raw curves only",
        weight="bold",
        y=0.995,
        fontsize=17,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = out_dir / f"uniform_noise3_state50_raw_only_same_scale{n_suffix}.png"
    fig.savefig(png)
    fig.savefig(png.with_suffix(".pdf"))
    plt.close(fig)
    return {"raw_only_same_scale_png": str(png), "raw_only_same_scale_pdf": str(png.with_suffix(".pdf"))}


def _plot_raw_only_delta(
    out_dir: Path,
    frequency: np.ndarray,
    rows: list[dict[str, object]],
    curves_by_key: dict[tuple[str, int], dict[str, np.ndarray]],
    mask_config: MaskConfig,
    plot_n_images: tuple[int, ...] | None,
) -> dict[str, str]:
    n_suffix = ""
    n_title = ""
    if plot_n_images is not None:
        labels = [N_LABELS[n] for n in plot_n_images]
        n_suffix = "_" + "_".join(f"n{n:08d}" for n in plot_n_images) + "_only"
        n_title = " | " + ", ".join(labels) + " only"
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14.6, 8.0),
        dpi=260,
        sharex="col",
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.10, "wspace": 0.13},
    )
    for col, method in enumerate(METHOD_ORDER):
        ax_curve = axes[0, col]
        ax_delta = axes[1, col]
        method_rows = [row for row in rows if row["method"] == method]
        deltas: list[np.ndarray] = []
        for row in sorted(method_rows, key=lambda r: int(r["n_images"])):
            n_images = int(row["n_images"])
            curves = curves_by_key[(method, n_images)]
            label = N_LABELS[n_images]
            actual = curves[_raw_mapmodel_curve_name(method)]
            expected = curves["fsc_full_sqrt_expected_corr"]
            delta = actual - expected
            deltas.append(delta)
            ax_curve.plot(
                frequency,
                actual,
                color=MAPMODEL_GT_COLOR,
                lw=4.4,
                alpha=0.98,
                zorder=2,
                path_effects=CURVE_STROKE,
                label=f"{label} map to GT",
            )
            ax_curve.plot(
                frequency,
                expected,
                color=HALFMAP_EXPECTED_COLOR,
                lw=5.0,
                ls=(0, (7, 3)),
                alpha=0.98,
                zorder=3,
                path_effects=CURVE_STROKE,
                label=f"{label} halfmap -> expected",
            )
            ax_delta.plot(
                frequency,
                delta,
                color=DELTA_COLOR,
                lw=3.8,
                alpha=0.98,
                path_effects=DELTA_STROKE,
                label=label,
            )
        ax_curve.axhline(0.5, color="0.25", ls=":", lw=1.4)
        ax_curve.set_xlim(0.0, 0.40)
        ax_curve.set_ylim(-0.04, 1.03)
        ax_curve.grid(True, color="0.84", lw=0.8)
        ax_curve.set_title("RECOVAR" if method == "recovar" else "3DFlex", weight="bold", fontsize=16)
        ax_curve.legend(fontsize=9.0, ncol=1, framealpha=0.90)
        ax_delta.axhline(0.0, color="0.25", ls=":", lw=1.4)
        ax_delta.set_xlim(0.0, 0.40)
        if deltas:
            valid = (frequency >= 0.025) & (frequency <= 0.36)
            max_abs = max(float(np.nanmax(np.abs(delta[valid]))) for delta in deltas)
            ylim = max(0.06, min(0.55, 1.18 * max_abs))
            ax_delta.set_ylim(-ylim, ylim)
        ax_delta.grid(True, color="0.84", lw=0.8)
        ax_delta.set_xlabel("spatial frequency (1/A)", fontsize=12)
        ax_delta.set_ylabel("map to GT\nminus halfmap", fontsize=11)
        ax_curve.tick_params(axis="both", labelsize=11)
        ax_delta.tick_params(axis="both", labelsize=11)
        for ax in (ax_curve, ax_delta):
            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
                spine.set_color("0.16")
    axes[0, 0].set_ylabel("raw FSC / expected correlation", fontsize=12)
    fig.suptitle(
        f"Uniform noise=3 state 50 | {mask_config.label}{n_title} | raw curves only",
        weight="bold",
        y=0.995,
        fontsize=17,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = out_dir / f"uniform_noise3_state50_raw_only_same_scale_with_delta{n_suffix}.png"
    fig.savefig(png)
    fig.savefig(png.with_suffix(".pdf"))
    plt.close(fig)
    return {
        "raw_only_same_scale_with_delta_png": str(png),
        "raw_only_same_scale_with_delta_pdf": str(png.with_suffix(".pdf")),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SWEEP_ROOT / "state50_halfmap_vs_mapmodel_fsc_20260607",
    )
    parser.add_argument("--voxel-size", type=float, default=1.25)
    parser.add_argument("--max-state", type=int, default=99)
    parser.add_argument("--gt-reference-n", type=int, default=10_000)
    parser.add_argument("--mask-mode", choices=sorted(MASK_CONFIGS), default="broad")
    parser.add_argument(
        "--phase-randomization-cutoff-A",
        type=float,
        default=10.0,
        help="Randomize halfmap Fourier phases at and beyond this resolution cutoff in Angstrom.",
    )
    parser.add_argument("--phase-randomization-seed", type=int, default=20260608)
    parser.add_argument(
        "--plot-n-images",
        help="Comma-separated image counts to score and include in plots, e.g. 1000000.",
    )
    return parser.parse_args(argv)


def _parse_plot_n_images(value: str | None) -> tuple[int, ...] | None:
    if value is None or value.strip() == "":
        return None
    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    invalid = [n_images for n_images in parsed if n_images not in N_ORDER]
    if invalid:
        raise ValueError(f"unsupported --plot-n-images values: {invalid}")
    return parsed


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_config = MASK_CONFIGS[args.mask_mode]
    mask = _load_mrc(mask_config.mask)
    labels, n_shells = _shell_labels(mask.shape)
    frequency = np.arange(n_shells, dtype=np.float64) / (mask.shape[0] * args.voxel_size)
    gt_state_fts = _precompute_gt_state_fts(args.gt_reference_n, args.max_state, mask)
    plot_n_images = _parse_plot_n_images(args.plot_n_images)
    n_order = plot_n_images if plot_n_images is not None else N_ORDER
    rows: list[dict[str, object]] = []
    curves_by_key: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for n_images in n_order:
        recovar = _recovar_maps(n_images)
        if recovar is not None:
            row, curves = _score_recovar(
                recovar,
                mask,
                labels,
                n_shells,
                frequency,
                args.output_dir,
                mask_config,
                args.voxel_size,
                args.phase_randomization_cutoff_A,
                args.phase_randomization_seed,
            )
            rows.append(row)
            curves_by_key[("recovar", n_images)] = curves
        three_dflex = _three_dflex_maps(n_images)
        if three_dflex is not None:
            row, curves = _score_3dflex(
                three_dflex,
                mask,
                labels,
                n_shells,
                frequency,
                args.output_dir,
                gt_state_fts,
                mask_config,
                args.voxel_size,
                args.phase_randomization_cutoff_A,
                args.phase_randomization_seed,
            )
            rows.append(row)
            curves_by_key[("3dflex", n_images)] = curves
    if not rows:
        raise RuntimeError(f"no rows scored for requested n_images={n_order}")
    summary_csv = args.output_dir / "uniform_noise3_state50_halfmap_vs_mapmodel_summary.csv"
    _write_summary(summary_csv, rows)
    plots = _plot_curves(args.output_dir, frequency, rows, curves_by_key, mask_config, plot_n_images)
    audit = {
        "script": str(Path(__file__).resolve()),
        "sweep_root": str(SWEEP_ROOT),
        "source_root": str(SOURCE_ROOT),
        "mask": str(mask_config.mask),
        "mask_mode": mask_config.key,
        "state": 50,
        "noise_level": 3,
        "bfactor": 80,
        "voxel_size_A": args.voxel_size,
        "gt_reference_n_images_for_3dflex_best_state": args.gt_reference_n,
        "plot_n_images": None if plot_n_images is None else list(plot_n_images),
        "phase_randomization_cutoff_A": args.phase_randomization_cutoff_A,
        "phase_randomization_cutoff_frequency_1_per_A": 1.0 / args.phase_randomization_cutoff_A,
        "phase_randomization_seed": args.phase_randomization_seed,
        "formulae": {
            "halfmap_raw": "h = FSC(half1, half2)",
            "phase_randomized_masked": "r = FSC(mask * phase_randomize(half1), mask * phase_randomize(half2))",
            "mask_corrected_halfmap": "h_corr = (h - r)/(1 - r) for shells at/above phase-randomization cutoff; h below cutoff",
            "squared_expected_correlation": "2*h_corr/(1+h_corr)",
            "FSC_full_sqrt_expected_correlation": "sqrt(max(0, 2*h_corr/(1+h_corr)))",
            "mapmodel_raw": "m = FSC(mask * estimate, mask * GT_or_model)",
            "mapmodel_phase_randomized_masked": "q = FSC(mask * phase_randomize(estimate), mask * phase_randomize(GT_or_model))",
            "mask_corrected_mapmodel": "m_corr = (m - q)/(1 - q) for shells at/above phase-randomization cutoff; m below cutoff",
            "full_map_snr_not_plotted_as_fsc": "2*h_corr/(1-h_corr)",
        },
        "3dflex_note": (
            "3DFlex halfmaps are highres flex-map halfmaps.  The map-to-model curve "
            f"is the highres full flex map against the best GT state by {mask_config.label} FSC AUC; "
            "the generated state-50 mean-latent map is scored separately for context."
        ),
        "summary_csv": str(summary_csv),
        "plots": plots,
        "n_rows": len(rows),
    }
    audit_path = args.output_dir / "uniform_noise3_state50_halfmap_vs_mapmodel_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(args.output_dir), "summary_csv": str(summary_csv), "plots": plots}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])

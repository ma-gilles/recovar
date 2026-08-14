"""Optional per-tilt homogeneous reconstructions and compact diagnostics."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from recovar import core
from recovar.core import fourier_transform_utils
from recovar.output import output
from recovar.reconstruction import regularization, relion_functions

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TiltDiagnosticGroup:
    index: int
    pre_exposure: float
    image_indices: np.ndarray


def group_preexposures(values, atol=1e-5):
    """Return sorted acquisition groups, tolerating STAR float roundoff."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return []
    if not np.all(np.isfinite(values)):
        raise ValueError("pre-exposure values contain NaN or Inf")
    order = np.argsort(values, kind="stable")
    groups = []
    start = 0
    while start < order.size:
        anchor = values[order[start]]
        stop = start + 1
        while stop < order.size and abs(values[order[stop]] - anchor) <= atol:
            stop += 1
        indices = np.sort(order[start:stop].astype(np.int32, copy=False))
        groups.append(
            TiltDiagnosticGroup(
                index=len(groups),
                pre_exposure=float(np.median(values[indices])),
                image_indices=indices,
            )
        )
        start = stop
    return groups


def _radial_power(volume):
    volume_ft = fourier_transform_utils.get_dft3(np.asarray(volume))
    return np.asarray(regularization.average_over_shells(np.abs(volume_ft) ** 2, volume.shape))


def _radial_ctf_power(ctf, image_shape):
    return np.asarray(regularization.batch_average_over_shells(np.abs(ctf) ** 2, image_shape, 0))


def _safe_relative_amplitude(power, reference_power):
    power = np.asarray(power, dtype=np.float64)
    reference_power = np.asarray(reference_power, dtype=np.float64)
    floor = np.finfo(np.float32).tiny
    return np.sqrt(np.maximum(power, floor) / np.maximum(reference_power, floor))


def _neutralized_group_dataset(dataset, group, source_ctf_params):
    group_dataset = dataset.subset(group.image_indices)
    ctf_params = np.asarray(source_ctf_params[group.image_indices]).copy()
    ctf_params[:, core.CTFParamIndex.CONTRAST] = 1.0
    ctf_params[:, core.CTFParamIndex.BFACTOR] = 0.0
    if ctf_params.shape[1] > core.CTFParamIndex.DOSE:
        ctf_params[:, core.CTFParamIndex.DOSE] = 0.0
    if ctf_params.shape[1] > core.CTFParamIndex.TILT_ANGLE:
        ctf_params[:, core.CTFParamIndex.TILT_ANGLE] = 0.0
    group_dataset.update_ctf(ctf_params)
    group_dataset.update_ctf_evaluator(core.CTFEvaluator(mode=core.CTFMode.SPA))
    unit_noise = np.ones(group_dataset.grid_size // 2 - 1, dtype=group_dataset.dtype_real)
    group_dataset.set_radial_noise_model(unit_noise)
    return group_dataset, unit_noise


def _group_transfer_profiles(dataset, group, source_ctf_params, source_ctf_evaluator):
    params = np.asarray(source_ctf_params[group.image_indices]).copy()
    input_scales = params[:, core.CTFParamIndex.CONTRAST].copy()
    full_ctf = source_ctf_evaluator(params, dataset.image_shape, dataset.voxel_size)
    params[:, core.CTFParamIndex.CONTRAST] = 1.0
    dose_ctf = source_ctf_evaluator(params, dataset.image_shape, dataset.voxel_size)
    params[:, core.CTFParamIndex.BFACTOR] = 0.0
    base_ctf = core.CTFEvaluator(mode=core.CTFMode.SPA)(
        params,
        dataset.image_shape,
        dataset.voxel_size,
    )
    return {
        "full": np.mean(_radial_ctf_power(full_ctf, dataset.image_shape), axis=0),
        "dose": np.mean(_radial_ctf_power(dose_ctf, dataset.image_shape), axis=0),
        "base": np.mean(_radial_ctf_power(base_ctf, dataset.image_shape), axis=0),
        "median_input_scale": float(np.median(input_scales)),
    }


def _shared_limits(images):
    finite = np.concatenate([np.asarray(image, dtype=np.float32).ravel() for image in images])
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(finite, [1.0, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(finite)), float(np.max(finite) + 1e-6)
    return float(lo), float(hi)


def _six_volume_views(volume):
    center = tuple(size // 2 for size in volume.shape)
    projections = [np.mean(volume, axis=axis) for axis in range(3)]
    slices = [
        volume[center[0], :, :],
        volume[:, center[1], :],
        volume[:, :, center[2]],
    ]
    return projections + slices


def _plot_six_view_rows(volumes, records, output_path):
    """Match the pipeline six-view layout with one row per acquisition rank."""

    rows = [_six_volume_views(volume) for volume in volumes]
    # Mean projections preserve density units closely enough that a single
    # global display range remains useful for both projections and slices.
    lo, hi = _shared_limits([panel for row in rows for panel in row])
    fig, axes = plt.subplots(len(rows), 6, figsize=(17, max(3.0, 2.65 * len(rows))), squeeze=False)
    column_titles = ["mean projection 0", "mean projection 1", "mean projection 2", "slice 0", "slice 1", "slice 2"]
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=10)
    last_image = None
    for row_idx, (panels, record) in enumerate(zip(rows, records)):
        for col_idx, panel in enumerate(panels):
            last_image = axes[row_idx, col_idx].imshow(
                panel.T,
                cmap="gray",
                origin="lower",
                vmin=lo,
                vmax=hi,
            )
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
        angle_prefix = "|tilt|≈" if record["tilt_angle_inferred"] else "tilt "
        axes[row_idx, 0].set_ylabel(
            f"rank {record['group_index']}\ndose {record['pre_exposure']:.1f}\n"
            f"{angle_prefix}{record['tilt_angle_deg']:.1f}°\nn={record['n_images']}",
            fontsize=8,
        )
    colorbar_axis = fig.add_axes([0.925, 0.36, 0.012, 0.28])
    fig.colorbar(last_image, cax=colorbar_axis, label="density (shared scale)")
    fig.suptitle("Per-tilt homogeneous means: six views, identical display scale", fontsize=14, y=0.995)
    fig.subplots_adjust(left=0.08, right=0.90, top=0.94, bottom=0.02, hspace=0.08, wspace=0.04)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_reconstruction_power_comparisons(
    volume_powers,
    predicted_powers,
    observed_amplitudes,
    predicted_amplitudes,
    records,
    frequencies,
    output_path,
    detail_dir,
):
    """Plot all reconstruction spectra and one rank-zero comparison per tilt."""

    eps = np.finfo(np.float32).tiny
    group_indices = np.arange(len(records))
    colors = plt.cm.plasma(np.linspace(0.05, 0.95, len(records)))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for group_idx, color in zip(group_indices, colors):
        axes[0].plot(
            frequencies,
            np.maximum(volume_powers[group_idx], eps),
            color=color,
            linewidth=1.8,
            label=f"rank {group_idx}",
        )
        axes[1].plot(frequencies, observed_amplitudes[group_idx], color=color, linewidth=1.8)
        axes[1].plot(
            frequencies,
            predicted_amplitudes[group_idx],
            color=color,
            linestyle="--",
            linewidth=1.4,
        )
    axes[0].set_yscale("log")
    axes[0].set_title("Spherically averaged reconstruction power — all tilts")
    axes[0].set_ylabel("power")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].axhline(1.0, color="black", linewidth=1.0)
    axes[1].set_yscale("log")
    axes[1].set_title("Observed (solid) vs predicted CTF scale×dose (dashed)")
    axes[1].set_ylabel("amplitude relative to rank 0")
    axes[1].plot([], [], color="black", linewidth=1.8, label="observed")
    axes[1].plot([], [], color="black", linestyle="--", linewidth=1.4, label="predicted")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("spatial frequency (1/Å)")
        ax.grid(alpha=0.25)
    fig.suptitle("Per-tilt reconstruction power and predicted transfer", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    os.makedirs(detail_dir, exist_ok=True)
    reference_power = np.maximum(volume_powers[0], eps)
    for group_idx, color in zip(group_indices, colors):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
        axes[0].plot(frequencies, reference_power, color="black", linewidth=2.0, label="rank 0 observed")
        if group_idx != 0:
            axes[0].plot(
                frequencies,
                np.maximum(volume_powers[group_idx], eps),
                color=color,
                linewidth=2.0,
                label=f"rank {group_idx} observed",
            )
        axes[0].plot(
            frequencies,
            np.maximum(predicted_powers[group_idx], eps),
            color=color,
            linestyle="--",
            linewidth=1.7,
            label="rank 0 × predicted transfer²",
        )
        axes[0].set_yscale("log")
        axes[0].set_title("Observed and CTF-predicted reconstruction power")
        axes[0].set_ylabel("power")
        axes[0].legend(fontsize=8)

        axes[1].plot(
            frequencies,
            observed_amplitudes[group_idx],
            color=color,
            linewidth=2.0,
            label="observed amplitude ratio",
        )
        axes[1].plot(
            frequencies,
            predicted_amplitudes[group_idx],
            color=color,
            linestyle="--",
            linewidth=1.7,
            label="predicted scale×dose ratio",
        )
        axes[1].axhline(1.0, color="black", linewidth=1.0)
        axes[1].set_yscale("log")
        axes[1].set_title("Fit relative to acquisition rank 0")
        axes[1].set_ylabel("amplitude ratio")
        axes[1].legend(fontsize=8)
        for ax in axes:
            ax.set_xlabel("spatial frequency (1/Å)")
            ax.grid(alpha=0.25)
        fig.suptitle(
            f"Reconstruction spectrum: rank {group_idx} vs 0 | dose {records[group_idx]['pre_exposure']:.1f}",
            fontsize=13,
        )
        fig.tight_layout()
        fig.savefig(os.path.join(detail_dir, f"tilt_{group_idx:03d}_vs_000.png"), dpi=180, bbox_inches="tight")
        plt.close(fig)


def _plot_spectral_summary(volume_powers, transfer_profiles, records, voxel_size, grid_size, output_path):
    volume_powers = np.asarray(volume_powers)
    full_transfer = np.asarray([profile["full"] for profile in transfer_profiles])
    dose_transfer = np.asarray([profile["dose"] for profile in transfer_profiles])
    base_transfer = np.asarray([profile["base"] for profile in transfer_profiles])
    n_shells = min(volume_powers.shape[1], full_transfer.shape[1])
    frequencies = np.arange(n_shells, dtype=np.float64) / (grid_size * voxel_size)
    observed = _safe_relative_amplitude(volume_powers[:, :n_shells], volume_powers[0, :n_shells])
    total_envelope = np.sqrt(
        np.maximum(full_transfer[:, :n_shells], np.finfo(np.float32).tiny)
        / np.maximum(base_transfer[:, :n_shells], np.finfo(np.float32).tiny)
    )
    total_envelope /= np.maximum(total_envelope[0], np.finfo(np.float32).tiny)
    dose_envelope = np.sqrt(
        np.maximum(dose_transfer[:, :n_shells], np.finfo(np.float32).tiny)
        / np.maximum(base_transfer[:, :n_shells], np.finfo(np.float32).tiny)
    )
    dose_envelope /= np.maximum(dose_envelope[0], np.finfo(np.float32).tiny)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    extent = (frequencies[0], frequencies[-1], -0.5, len(records) - 0.5)
    image = axes[0, 0].imshow(
        np.log10(np.maximum(volume_powers[:, :n_shells], np.finfo(np.float32).tiny)),
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
    )
    axes[0, 0].set_title("Reconstruction radial power spectra")
    fig.colorbar(image, ax=axes[0, 0], label="log10 power")
    image = axes[0, 1].imshow(
        np.log2(np.maximum(observed, np.finfo(np.float32).tiny)),
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    axes[0, 1].set_title("Observed amplitude ratio to first exposure")
    fig.colorbar(image, ax=axes[0, 1], label="log2 amplitude ratio")
    for ax in axes[0]:
        ax.set_xlabel("spatial frequency (1/Å)")
        ax.set_ylabel("tilt/dose group")

    group_indices = np.arange(len(records))
    colors = plt.cm.plasma(np.linspace(0.05, 0.95, group_indices.size))
    for group_idx, color in zip(group_indices, colors):
        axes[1, 0].plot(frequencies, observed[group_idx], color=color, linewidth=1.8, label=f"group {group_idx}")
        axes[1, 0].plot(frequencies, total_envelope[group_idx], color=color, linestyle="--", linewidth=1.4)
    axes[1, 0].axhline(1.0, color="black", linewidth=1)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Observed (solid) vs predicted scale×dose envelope (dashed)")
    axes[1, 0].set_xlabel("spatial frequency (1/Å)")
    axes[1, 0].set_ylabel("amplitude relative to first exposure")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(fontsize=8, ncol=2)

    mid = slice(max(1, n_shells // 4), max(2, 3 * n_shells // 4))
    group_axis = np.arange(len(records))
    axes[1, 1].plot(group_axis, np.nanmedian(observed[:, mid], axis=1), "o-", label="observed")
    axes[1, 1].plot(group_axis, np.nanmedian(total_envelope[:, mid], axis=1), "o-", label="scale × dose")
    axes[1, 1].plot(group_axis, np.nanmedian(dose_envelope[:, mid], axis=1), "o-", label="dose only")
    axes[1, 1].plot(
        group_axis,
        np.asarray([abs(record["input_ctf_scale_relative"]) for record in records]),
        "o-",
        label="input scale only",
    )
    axes[1, 1].set_title("Mid-frequency attenuation across acquisition")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("tilt/dose group")
    axes[1, 1].set_ylabel("amplitude relative to first exposure")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()
    fig.suptitle("Per-tilt reconstruction transfer diagnostics", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    predicted_powers = volume_powers[0, :n_shells] * total_envelope**2
    return observed, total_envelope, dose_envelope, predicted_powers, frequencies


def run_tilt_diagnostics(
    dataset,
    *,
    source_ctf_params,
    source_ctf_evaluator,
    output_dir,
    plots_dir,
    batch_size,
):
    """Reconstruct one neutral-weight mean per acquisition exposure group."""

    if not dataset.tilt_series_flag:
        raise ValueError("--tilt-diagnostics requires --tilt-series")
    source_ctf_params = np.asarray(source_ctf_params)
    if source_ctf_params.shape != dataset.CTF_params.shape:
        raise ValueError("source CTF parameter snapshot does not match the loaded dataset")

    groups = group_preexposures(source_ctf_params[:, core.CTFParamIndex.DOSE])
    if not groups:
        raise ValueError("no pre-exposure groups found for tilt diagnostics")
    diagnostics_dir = os.path.join(output_dir, "tilt_diagnostics")
    volumes_dir = os.path.join(diagnostics_dir, "volumes")
    if os.path.exists(diagnostics_dir):
        raise FileExistsError(f"tilt diagnostics output already exists: {diagnostics_dir}")
    os.makedirs(volumes_dir)

    volumes = []
    volume_powers = []
    transfer_profiles = []
    records = []
    for group in groups:
        logger.info(
            "Tilt diagnostics %d/%d: pre-exposure %.6f (%d images)",
            group.index + 1,
            len(groups),
            group.pre_exposure,
            group.image_indices.size,
        )
        group_dataset, unit_noise = _neutralized_group_dataset(dataset, group, source_ctf_params)
        ft_ctf, ft_y = relion_functions.relion_style_triangular_kernel(
            group_dataset,
            unit_noise,
            2 * batch_size,
            upsampling_factor=2,
        )
        mean_ft = relion_functions.post_process_from_filter_v2(
            ft_ctf,
            ft_y,
            group_dataset.volume_shape,
            2,
        )
        volume = np.asarray(
            fourier_transform_utils.get_idft3(mean_ft.reshape(group_dataset.volume_shape)).real,
            dtype=np.float32,
        )
        volume_path = os.path.join(volumes_dir, f"tilt_mean_{group.index:03d}.mrc")
        output.save_volume(
            volume,
            os.path.splitext(volume_path)[0],
            group_dataset.volume_shape,
            from_ft=False,
            voxel_size=group_dataset.voxel_size,
        )
        transfer = _group_transfer_profiles(dataset, group, source_ctf_params, source_ctf_evaluator)
        source_angles = source_ctf_params[group.image_indices, core.CTFParamIndex.TILT_ANGLE]
        angle = float(np.median(source_angles)) if source_angles.size else 0.0
        inferred = False
        if np.isclose(angle, 0.0) and 0 < abs(transfer["median_input_scale"]) <= 1:
            angle = float(np.degrees(np.arccos(np.clip(abs(transfer["median_input_scale"]), 0.0, 1.0))))
            inferred = True
        records.append(
            {
                "group_index": group.index,
                "pre_exposure": group.pre_exposure,
                "n_images": int(group.image_indices.size),
                "tilt_angle_deg": angle,
                "tilt_angle_inferred": inferred,
                "median_input_ctf_scale": transfer["median_input_scale"],
                "input_ctf_scale_relative": 0.0,
                "volume": volume_path,
            }
        )
        volumes.append(volume)
        volume_powers.append(_radial_power(volume))
        transfer_profiles.append(transfer)

    reference_scale = records[0]["median_input_ctf_scale"]
    for record in records:
        record["input_ctf_scale_relative"] = (
            record["median_input_ctf_scale"] / reference_scale if reference_scale != 0 else np.nan
        )

    _plot_six_view_rows(
        volumes,
        records,
        os.path.join(plots_dir, "tilt_diagnostics_six_view_shared_scale.png"),
    )
    observed, total_envelope, dose_envelope, predicted_powers, frequencies = _plot_spectral_summary(
        volume_powers,
        transfer_profiles,
        records,
        float(dataset.voxel_size),
        int(dataset.grid_size),
        os.path.join(plots_dir, "tilt_diagnostics_summary.png"),
    )
    n_shells = observed.shape[1]
    _plot_reconstruction_power_comparisons(
        np.asarray(volume_powers)[:, :n_shells],
        predicted_powers,
        observed,
        total_envelope,
        records,
        frequencies,
        os.path.join(plots_dir, "tilt_reconstruction_power.png"),
        os.path.join(plots_dir, "tilt_reconstruction_power_individual"),
    )
    np.savez_compressed(
        os.path.join(diagnostics_dir, "spectra.npz"),
        reconstruction_power=np.asarray(volume_powers),
        observed_relative_amplitude=observed,
        predicted_scale_dose_relative_amplitude=total_envelope,
        predicted_dose_relative_amplitude=dose_envelope,
        predicted_reconstruction_power=predicted_powers,
        ctf_full_power=np.asarray([profile["full"] for profile in transfer_profiles]),
        ctf_dose_power=np.asarray([profile["dose"] for profile in transfer_profiles]),
        ctf_base_power=np.asarray([profile["base"] for profile in transfer_profiles]),
    )
    with open(os.path.join(diagnostics_dir, "summary.json"), "w", encoding="utf-8") as stream:
        json.dump(
            {
                "conventions": {
                    "grouping": "sorted unique pre-exposure",
                    "reconstruction_contrast": 1.0,
                    "reconstruction_ctf_scale": 1.0,
                    "reconstruction_dose_envelope": "disabled",
                    "reconstruction_ctf_bfactor_envelope": "disabled",
                    "reconstruction_noise_weighting": "unit radial variance",
                    "physical_ctf": "retained",
                    "prediction": "source RECOVAR CTF evaluator; amplitude relative to first exposure",
                },
                "groups": records,
            },
            stream,
            indent=2,
        )
    logger.info("Saved tilt diagnostics under %s", diagnostics_dir)
    return records

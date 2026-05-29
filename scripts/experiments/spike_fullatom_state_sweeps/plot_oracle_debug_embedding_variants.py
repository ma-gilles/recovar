#!/usr/bin/env python3
"""Clean oracle-embedding diagnostic plot for the 3M noise=30 state-50 run."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_noise1_sweep_with_gt_nearest import (
    DEFAULT_MASK,
    FREQ_MAX,
    TARGET_STATE,
    fsc05_resolution,
    load_mrc,
    masked_metrics,
    metric_context,
    relerr_resolution,
    state_weighted_gt_mrc,
)


DEFAULT_RUN = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_consistency_grid256_noise30_b80_parallel_20260518/"
    "n03000000/runs/n03000000_seed0000"
)
DEFAULT_OUT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/tmp/"
    "spike_fullatom_direct_volume_shell_metrics_20260523/"
    "oracle_debug_embedding_variants_repro_20260528"
)


@dataclass(frozen=True)
class Variant:
    label: str
    relpath: str
    color: str
    linestyle: str
    meaning: str


@dataclass(frozen=True)
class NearestControl:
    label: str
    embedding_relpath: str
    target_point_name: str
    color: str
    linestyle: str
    meaning: str


ZDIM4_VARIANTS = (
    Variant(
        label="source oracle",
        relpath="07_compute_state_oracle_regfix_zdim4_reg_lazy",
        color="#111111",
        linestyle="-",
        meaning="original source-oracle embedding, zdim4, regularized cov-distance",
    ),
    Variant(
        label="source oracle noreg",
        relpath="07_compute_state_oracle_regfix_zdim4_noreg_lazy",
        color="#666666",
        linestyle="--",
        meaning="same source-oracle latent coordinates with the noreg/no-regularization precision path",
    ),
    Variant(
        label="pipeline",
        relpath="07_compute_state_pipeline_zdim4_reg_lazy",
        color="#1f77b4",
        linestyle="-",
        meaning="normal pipeline output embedding, zdim4, regularized cov-distance",
    ),
    Variant(
        label="GT clean",
        relpath="07_compute_state_oracle_gap_gt_clean_identity_zdim4_reg_lazy",
        color="#2ca02c",
        linestyle="-",
        meaning="true GT-PC coordinates as embedding with identity precision",
    ),
    Variant(
        label="GT clean + source cov",
        relpath="07_compute_state_oracle_gap_gt_clean_sourceprec_zdim4_reg_lazy",
        color="#17becf",
        linestyle="--",
        meaning="true GT-PC coordinates with source-oracle covariance/precision attached",
    ),
    Variant(
        label="GT + exact residual",
        relpath="07_compute_state_oracle_gap_gt_residcopy_trueunits_zdim4_reg_lazy",
        color="#ff7f0e",
        linestyle="-",
        meaning="true GT-PC coordinates plus the exact affine source-oracle residual",
    ),
    Variant(
        label="GT + sampled cov",
        relpath="07_compute_state_oracle_gap_gt_covnoise_trueunits_zdim4_reg_lazy",
        color="#d62728",
        linestyle="-",
        meaning="true GT-PC coordinates plus newly sampled Gaussian noise from source-oracle covariance",
    ),
    Variant(
        label="GT + shuffled cov",
        relpath="07_compute_state_oracle_gap_gt_covshuffle_trueunits_zdim4_reg_lazy",
        color="#9467bd",
        linestyle="-",
        meaning="same sampled-covariance experiment after shuffling covariances across images",
    ),
)

ZDIM2_VARIANTS = tuple(
    Variant(
        label=variant.label,
        relpath=variant.relpath.replace("zdim4", "zdim2"),
        color=variant.color,
        linestyle=variant.linestyle,
        meaning=variant.meaning.replace("zdim4", "zdim2"),
    )
    for variant in ZDIM4_VARIANTS
    if "pipeline_zdim4" not in variant.relpath and "zdim4_noreg" not in variant.relpath
)

GROUPS = (
    ("Source, pipeline, exact residual", ("source oracle", "source oracle noreg", "pipeline", "GT clean", "GT clean + source cov", "GT + exact residual")),
    ("Sampled covariance controls", ("source oracle", "GT clean + source cov", "GT + sampled cov", "GT + shuffled cov")),
)

NEAREST_CONTROLS = {
    "zdim4": (
        NearestControl(
            label="source nearest-100 GT",
            embedding_relpath="06_pipeline_oracle_regfix_20260526/model/zdim_4/latent_coords.npy",
            target_point_name="target_latent_point_oracle_regfix_zdim4_reg_state0050.txt",
            color="#005a32",
            linestyle="-.",
            meaning="GT-volume mixture from the 100 particles nearest to state 50 in the source-oracle zdim4 embedding",
        ),
    ),
    "zdim2": (
        NearestControl(
            label="source nearest-100 GT",
            embedding_relpath="06_pipeline_oracle_regfix_20260526/model/zdim_2/latent_coords.npy",
            target_point_name="target_latent_point_oracle_regfix_zdim2_reg_state0050.txt",
            color="#005a32",
            linestyle="-.",
            meaning="GT-volume mixture from the 100 particles nearest to state 50 in the source-oracle zdim2 embedding",
        ),
    ),
}


def distribution_mean_mrc(run: Path) -> np.ndarray:
    paths = [run / "04_ground_truth" / f"gt_vol{i:04d}.mrc" for i in range(100)]
    total = np.zeros_like(load_mrc(paths[0]), dtype=np.float64)
    for path in paths:
        total += load_mrc(path)
    return (total / len(paths)).astype(np.float32)


def load_variant_curves(
    run: Path,
    variants: tuple[Variant, ...],
    mask: np.ndarray,
    labels: np.ndarray,
    n_shells: int,
    target_ft: np.ndarray,
    target_power: np.ndarray,
) -> dict[str, dict[str, object]]:
    loaded: dict[str, dict[str, object]] = {}
    for variant in variants:
        map_path = run / variant.relpath / "state000_unfil.mrc"
        if not map_path.exists():
            print(f"SKIP missing {variant.label}: {map_path}", flush=True)
            continue
        fsc, err = masked_metrics(load_mrc(map_path), mask, labels, n_shells, target_ft, target_power)
        loaded[variant.label] = {
            "variant": variant,
            "path": map_path,
            "fsc": fsc,
            "err": err,
        }
    return loaded


def load_nearest_control_curves(
    run: Path,
    controls: tuple[NearestControl, ...],
    n_nearest: int,
    mask: np.ndarray,
    labels: np.ndarray,
    n_shells: int,
    target_ft: np.ndarray,
    target_power: np.ndarray,
) -> dict[str, dict[str, object]]:
    loaded: dict[str, dict[str, object]] = {}
    states = np.asarray(np.load(run / "03_dataset" / "state_assignment.npy"), dtype=np.int64).reshape(-1)
    for control in controls:
        z_path = run / control.embedding_relpath
        target_path = run / control.target_point_name
        if not z_path.exists() or not target_path.exists():
            print(f"SKIP missing nearest control {control.label}: {z_path} / {target_path}", flush=True)
            continue
        z = np.asarray(np.load(z_path), dtype=np.float64)
        target = np.asarray(np.loadtxt(target_path), dtype=np.float64).reshape(-1)
        if z.shape[0] != states.size:
            raise ValueError(f"Embedding/state length mismatch for {z_path}: {z.shape[0]} vs {states.size}")
        dist2 = np.sum((z[:, : target.size] - target[None, :]) ** 2, axis=1)
        n = min(n_nearest, dist2.size)
        nearest = np.argpartition(dist2, n - 1)[:n] if n < dist2.size else np.arange(dist2.size)
        counts = np.bincount(states[nearest], minlength=100).astype(np.float64)
        weights = counts / counts.sum()
        top_states = " ".join(
            f"{int(state)}:{int(counts[state])}"
            for state in np.argsort(counts)[::-1][:12]
            if counts[state] > 0
        )
        volume = state_weighted_gt_mrc(run, weights)
        fsc, err = masked_metrics(volume, mask, labels, n_shells, target_ft, target_power)
        loaded[control.label] = {
            "control": control,
            "fsc": fsc,
            "err": err,
            "embedding": z_path,
            "target_point": target_path,
            "n_nearest": int(n),
            "nearest_radius": float(np.sqrt(np.max(dist2[nearest]))),
            "state50_fraction": float(weights[TARGET_STATE]),
            "top_states": top_states,
        }
    return loaded


def plot_grouped(
    out_dir: Path,
    tag: str,
    loaded: dict[str, dict[str, object]],
    nearest_controls: dict[str, dict[str, object]],
    mean_fsc: np.ndarray,
    mean_err: np.ndarray,
    freq: np.ndarray,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 9.0), sharex=True)
    fig.suptitle(
        f"Oracle-debug embedding variants | {tag} | 3M noise=30 | unfiltered final maps",
        fontweight="bold",
    )
    legend: dict[str, object] = {}
    for row, (group_title, labels) in enumerate(GROUPS):
        ax_fsc, ax_err = axes[row]
        for label in labels:
            if label not in loaded:
                continue
            item = loaded[label]
            variant: Variant = item["variant"]  # type: ignore[assignment]
            line = ax_fsc.plot(
                freq,
                item["fsc"],
                color=variant.color,
                ls=variant.linestyle,
                lw=2.15,
                label=variant.label,
            )[0]
            ax_err.semilogy(
                freq,
                np.maximum(item["err"], 1e-30),
                color=variant.color,
                ls=variant.linestyle,
                lw=2.15,
            )
            legend[variant.label] = line
        mean_line = ax_fsc.plot(freq, mean_fsc, color="0.35", ls=":", lw=2.0, label="GT mean")[0]
        ax_err.semilogy(freq, np.maximum(mean_err, 1e-30), color="0.35", ls=":", lw=2.0)
        legend["GT mean"] = mean_line
        for item in nearest_controls.values():
            control: NearestControl = item["control"]  # type: ignore[assignment]
            line = ax_fsc.plot(
                freq,
                item["fsc"],
                color=control.color,
                ls=control.linestyle,
                lw=2.1,
                label=control.label,
            )[0]
            ax_err.semilogy(
                freq,
                np.maximum(item["err"], 1e-30),
                color=control.color,
                ls=control.linestyle,
                lw=2.1,
            )
            legend[control.label] = line
        ax_fsc.axhline(0.5, color="0.5", ls=":", lw=1.0)
        ax_err.axhline(0.1, color="0.5", ls=":", lw=1.0)
        ax_fsc.set_title(f"{group_title}: FSC")
        ax_err.set_title(f"{group_title}: relative error")
        ax_fsc.set_ylabel("FSC vs GT state 50")
        ax_err.set_ylabel("relative Fourier error")
        ax_fsc.set_ylim(-0.08, 1.03)
        ax_err.set_ylim(1e-3, 1e3)
        for ax in (ax_fsc, ax_err):
            ax.set_xlim(0.0, FREQ_MAX)
            ax.grid(alpha=0.25, which="both")
    for ax in axes[-1]:
        ax.set_xlabel("spatial frequency (1/A)")
    fig.legend(
        legend.values(),
        legend.keys(),
        loc="outside lower center",
        ncols=4,
        fontsize=8.0,
        frameon=True,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    png = out_dir / f"oracle_debug_{tag}_grouped_fsc_error.png"
    fig.savefig(png, dpi=180)
    fig.savefig(out_dir / f"oracle_debug_{tag}_grouped_fsc_error.pdf")
    plt.close(fig)
    return png


def write_key(
    out_dir: Path,
    tag: str,
    loaded: dict[str, dict[str, object]],
    nearest_controls: dict[str, dict[str, object]],
    freq: np.ndarray,
    mean_fsc: np.ndarray,
    mean_err: np.ndarray,
) -> None:
    with (out_dir / f"oracle_debug_{tag}_key.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "meaning", "map_path"])
        for item in loaded.values():
            variant: Variant = item["variant"]  # type: ignore[assignment]
            writer.writerow([variant.label, variant.meaning, item["path"]])
        for item in nearest_controls.values():
            control: NearestControl = item["control"]  # type: ignore[assignment]
            writer.writerow(
                [
                    control.label,
                    f"{control.meaning}; top states {item['top_states']}; state50 fraction {item['state50_fraction']:.3f}",
                    f"{item['embedding']} target={item['target_point']}",
                ]
            )
        writer.writerow(["GT mean", "mean of gt_volNNNN.mrc maps", "04_ground_truth/gt_vol*.mrc"])
    with (out_dir / f"oracle_debug_{tag}_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "fsc05_resolution_A", "relerr10_resolution_A", "map_path"])
        for item in loaded.values():
            variant: Variant = item["variant"]  # type: ignore[assignment]
            writer.writerow(
                [
                    variant.label,
                    fsc05_resolution(freq, item["fsc"]),
                    relerr_resolution(freq, item["err"], 0.1),
                    item["path"],
                ]
            )
        for item in nearest_controls.values():
            control: NearestControl = item["control"]  # type: ignore[assignment]
            writer.writerow(
                [
                    control.label,
                    fsc05_resolution(freq, item["fsc"]),
                    relerr_resolution(freq, item["err"], 0.1),
                    f"{item['embedding']} target={item['target_point']}",
                ]
            )
        writer.writerow(["GT mean", fsc05_resolution(freq, mean_fsc), relerr_resolution(freq, mean_err, 0.1), "04_ground_truth/gt_vol*.mrc"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--zdim", choices=("2", "4", "both"), default="4")
    parser.add_argument("--nearest-count", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target = load_mrc(args.run_dir / "04_ground_truth" / f"gt_vol{TARGET_STATE:04d}.mrc")
    mask = np.clip(load_mrc(args.mask), 0.0, 1.0)
    labels, n_shells, target_ft, target_power, freq = metric_context(target, mask)
    mean_fsc, mean_err = masked_metrics(
        distribution_mean_mrc(args.run_dir),
        mask,
        labels,
        n_shells,
        target_ft,
        target_power,
    )

    tags_and_variants: list[tuple[str, tuple[Variant, ...]]] = []
    if args.zdim in ("4", "both"):
        tags_and_variants.append(("zdim4", ZDIM4_VARIANTS))
    if args.zdim in ("2", "both"):
        tags_and_variants.append(("zdim2", ZDIM2_VARIANTS))

    for tag, variants in tags_and_variants:
        loaded = load_variant_curves(args.run_dir, variants, mask, labels, n_shells, target_ft, target_power)
        if not loaded:
            print(f"SKIP no completed variants for {tag}", flush=True)
            continue
        nearest_controls = load_nearest_control_curves(
            args.run_dir,
            NEAREST_CONTROLS.get(tag, ()),
            args.nearest_count,
            mask,
            labels,
            n_shells,
            target_ft,
            target_power,
        )
        png = plot_grouped(args.out_dir, tag, loaded, nearest_controls, mean_fsc, mean_err, freq)
        write_key(args.out_dir, tag, loaded, nearest_controls, freq, mean_fsc, mean_err)
        print(f"PLOT {png}")
        print(f"KEY {args.out_dir / f'oracle_debug_{tag}_key.csv'}")
        print(f"SUMMARY {args.out_dir / f'oracle_debug_{tag}_summary.csv'}")


if __name__ == "__main__":
    main()

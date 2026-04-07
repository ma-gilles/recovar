#!/usr/bin/env python
"""Score PPCA refit sweep results and aggregate into comparison tables + plots.

Reads existing baseline scores (covariance, ppca, ppca_projected_covariance) from the
reference run, scores new refit methods, and produces:
- per-dataset comparison_summary.json (compatible with existing format)
- aggregate CSV table
- relvar summary plot
- embedding_squared_error summary plot
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from recovar import utils
from recovar.output import output as output_module
from recovar.simulation import synthetic_dataset

logger = logging.getLogger(__name__)

# All methods we compare: baseline + new refit methods
BASELINE_METHODS = ("covariance", "ppca", "ppca_projected_covariance")
REFIT_METHODS = (
    "refit_b",
    "temperature_scalar",
    "temperature_diag",
    "stiefel_ub",
    "whitening_manifold_ub",
    "coord_reg_grid",
    "coord_reg_physical",
)
ALL_METHODS = BASELINE_METHODS + REFIT_METHODS

METHOD_LABELS = {
    "covariance": "Covariance",
    "ppca": "PPCA",
    "ppca_projected_covariance": "PPCA+ProjCov",
    "refit_b": "Alg1: Refit B",
    "temperature_scalar": "Alg5s: Scalar T",
    "temperature_diag": "Alg5d: Diag T",
    "stiefel_ub": "Alg2: Stiefel U/B",
    "whitening_manifold_ub": "Alg3: Whiten+B",
    "coord_reg_grid": "Alg4A: Grid Reg",
    "coord_reg_physical": "Alg4B: Phys Reg",
}

METHOD_COLORS = {
    "covariance": "#d55e00",
    "ppca": "#0072b2",
    "ppca_projected_covariance": "#009e73",
    "refit_b": "#cc79a7",
    "temperature_scalar": "#e69f00",
    "temperature_diag": "#f0e442",
    "stiefel_ub": "#56b4e9",
    "whitening_manifold_ub": "#882255",
    "coord_reg_grid": "#999999",
    "coord_reg_physical": "#332288",
}


def _score_one_method(result_dir: str, gt_results, metric_context: dict, zdim: int) -> dict | None:
    """Score a single method's PipelineOutput against ground truth."""
    params_path = os.path.join(result_dir, "model", "params.pkl")
    if not os.path.isfile(params_path):
        logger.warning("Missing params.pkl at %s", params_path)
        return None

    try:
        from recovar.ppca.compare_covariance_vs_ppca_pipeline import score_pipeline_output
        method_root = os.path.dirname(result_dir)
        return score_pipeline_output(result_dir, method_root, gt_results, metric_context, zdim)
    except Exception as e:
        logger.error("Failed to score %s: %s", result_dir, e)
        return None


def _build_metric_context(gt_results, sim_info, volume_shape):
    """Build metric context dict (same as compare driver)."""
    from recovar.output import metrics
    gt_mean = np.asarray(gt_results.get_mean())
    gt_union_soft_mask, gt_union_binary_mask = metrics.make_union_gt_mask_from_hvd(gt_results, volume_shape)
    u_gt_real, s_gt_real, _ = gt_results.get_vol_svd(contrasted=False, real_space=True, random_svd_pcs=200)
    particle_assignment = sim_info["image_assignment"]
    preferred_labels = [0, (int(np.max(particle_assignment)) + 1) // 2]
    return {
        "gt_mean": gt_mean,
        "gt_union_soft_mask": gt_union_soft_mask,
        "gt_union_binary_mask": gt_union_binary_mask,
        "u_gt_real": u_gt_real,
        "s_gt_real": s_gt_real,
        "particle_assignment": particle_assignment,
        "preferred_labels": preferred_labels,
        "gt_contrasts": np.asarray(gt_results.contrasts),
        "sim_info": sim_info,
    }


def score_dataset(dataset_dir: str, zdim: int) -> dict:
    """Score all available methods for one dataset directory."""
    sim_dir = os.path.join(dataset_dir, "simulated_data")
    sim_info_path = os.path.join(sim_dir, "simulation_info.pkl")

    # Follow symlinks
    if os.path.islink(sim_dir):
        sim_dir = os.readlink(sim_dir)
    if os.path.islink(sim_info_path):
        sim_info_path = os.path.realpath(sim_info_path)

    sim_info = utils.pickle_load(sim_info_path)
    gt_results = synthetic_dataset.load_heterogeneous_reconstruction(sim_info_path)

    grid_size = 128  # from dataset config
    config_path = os.path.join(dataset_dir, "dataset_config.json")
    if os.path.isfile(config_path) or os.path.islink(config_path):
        with open(os.path.realpath(config_path), "r") as fh:
            config = json.load(fh)
        grid_size = config.get("grid_size", 128)

    volume_shape = (grid_size, grid_size, grid_size)
    metric_context = _build_metric_context(gt_results, sim_info, volume_shape)

    scores = {}
    for method in ALL_METHODS:
        result_dir = os.path.join(dataset_dir, method, "result")
        if os.path.islink(result_dir):
            result_dir = os.path.realpath(result_dir)
        if not os.path.isdir(result_dir):
            continue
        logger.info("Scoring %s at %s", method, result_dir)
        score = _score_one_method(result_dir, gt_results, metric_context, zdim)
        if score is not None:
            scores[method] = score

    return scores


def _flatten_scalar_metrics(prefix: str, value) -> dict:
    """Flatten nested metrics dict to flat key-value pairs."""
    flat = {}
    if isinstance(value, dict):
        for key, subval in value.items():
            child_prefix = f"{prefix}_{key}" if prefix else str(key)
            flat.update(_flatten_scalar_metrics(child_prefix, subval))
        return flat
    if isinstance(value, (list, tuple)):
        return flat
    if value is None:
        flat[prefix] = None
        return flat
    if isinstance(value, (bool, np.bool_)):
        flat[prefix] = bool(value)
        return flat
    if isinstance(value, (int, float, np.integer, np.floating)):
        flat[prefix] = float(value)
        return flat
    return flat


def build_aggregate_table(results_root: str, zdim: int) -> list[dict]:
    """Score all datasets and build aggregate rows."""
    rows = []
    results_path = Path(results_root)

    for dataset_dir in sorted(results_path.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name == "aggregate":
            continue
        # Parse dataset name from directory
        parts = dataset_dir.name.split("_g")
        if len(parts) < 2:
            continue
        dataset_name = parts[0]

        logger.info("Processing dataset: %s", dataset_name)
        scores = score_dataset(str(dataset_dir), zdim)

        row = {
            "dataset": dataset_name,
            "run_dir": str(dataset_dir),
        }

        for method in ALL_METHODS:
            if method not in scores:
                continue
            s = scores[method]
            row[f"{method}_relvar_mean"] = float(s["rel_var_mean"])
            row[f"{method}_mean_error"] = float(s["mean_error"])
            row[f"{method}_result_dir"] = s["result_dir"]

            # Extract key pipeline metrics
            pm = s.get("pipeline_metrics", {})
            embed_key = f"embedding_squared_error_{zdim}"
            if embed_key in pm:
                row[f"{method}_embedding_squared_error_{zdim}"] = float(pm[embed_key])
            for state_key in ("state_0_locres_90pct", "state_0_locres_median",
                              "state_1_locres_90pct", "state_1_locres_median"):
                if state_key in pm:
                    row[f"{method}_{state_key}"] = float(pm[state_key])

        # Save per-dataset summary
        summary = {"dataset": dataset_name, "zdim": zdim, "scores": scores}
        summary_path = dataset_dir / "refit_comparison_summary.json"
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)

        rows.append(row)

    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    """Write aggregate CSV."""
    if not rows:
        return
    # Collect all keys preserving order
    all_keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                all_keys.append(key)
                seen.add(key)

    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_relvar(rows: list[dict], out_path: Path, zdim: int) -> None:
    """Bar chart of relvar_mean for all methods."""
    labels = [row["dataset"] for row in rows]
    x = np.arange(len(rows), dtype=np.float32)
    methods = [m for m in ALL_METHODS if f"{m}_relvar_mean" in rows[0]]

    fig, ax = plt.subplots(figsize=(max(12, len(rows) * 2.5), 6))
    width = 0.8 / max(len(methods), 1)
    center_offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width

    for offset, method in zip(center_offsets, methods):
        vals = [row.get(f"{method}_relvar_mean", 0) for row in rows]
        ax.bar(
            x + offset, vals, width=width,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, "#aaaaaa"),
            alpha=0.9,
        )
    ax.set_ylabel("Mean RelVar", fontsize=12)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x, labels, fontsize=11)
    ax.set_title(f"Relative Variance (zdim={zdim}, no contrast)", fontsize=14)
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved relvar plot: %s", out_path)


def plot_embedding_error(rows: list[dict], out_path: Path, zdim: int) -> None:
    """Bar chart of embedding_squared_error for all methods."""
    labels = [row["dataset"] for row in rows]
    x = np.arange(len(rows), dtype=np.float32)
    embed_key = f"embedding_squared_error_{zdim}"
    methods = [m for m in ALL_METHODS if f"{m}_{embed_key}" in rows[0]]

    if not methods:
        logger.warning("No embedding error data found, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(max(12, len(rows) * 2.5), 6))
    width = 0.8 / max(len(methods), 1)
    center_offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width

    for offset, method in zip(center_offsets, methods):
        vals = [row.get(f"{method}_{embed_key}", float("nan")) for row in rows]
        ax.bar(
            x + offset, vals, width=width,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, "#aaaaaa"),
            alpha=0.9,
        )
    ax.set_ylabel(f"Embedding Squared Error (top {zdim})", fontsize=12)
    ax.set_xticks(x, labels, fontsize=11)
    ax.set_title(f"Embedding Error (zdim={zdim}, no contrast)", fontsize=14)
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved embedding error plot: %s", out_path)


def print_comparison_table(rows: list[dict], zdim: int) -> None:
    """Print compact comparison table to stderr for visibility."""
    embed_key = f"embedding_squared_error_{zdim}"
    header = f"{'Dataset':<15}"
    methods_present = [m for m in ALL_METHODS if any(f"{m}_relvar_mean" in row for row in rows)]

    for m in methods_present:
        short = METHOD_LABELS.get(m, m)[:12]
        header += f"  {short:>12} rv  {short:>12} em"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for row in rows:
        line = f"{row['dataset']:<15}"
        for m in methods_present:
            rv = row.get(f"{m}_relvar_mean")
            em = row.get(f"{m}_{embed_key}")
            rv_str = f"{rv:.3f}" if rv is not None else "  ---"
            em_str = f"{em:.3f}" if em is not None else "  ---"
            line += f"  {rv_str:>14}  {em_str:>14}"
        print(line, flush=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Score PPCA refit sweep results.")
    parser.add_argument("results_root", type=str)
    parser.add_argument("--reference-root", type=str, default=None,
                        help="Reference run root (for loading baseline scores)")
    parser.add_argument("--zdim", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.results_root) / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Scoring results in %s", args.results_root)
    rows = build_aggregate_table(args.results_root, args.zdim)

    if not rows:
        logger.error("No results found to score")
        return

    # Write outputs
    write_csv(rows, output_dir / "comparison_table.csv")
    plot_relvar(rows, output_dir / "relvar_summary.png", args.zdim)
    plot_embedding_error(rows, output_dir / "embedding_error_summary.png", args.zdim)

    aggregate = {
        "results_root": args.results_root,
        "reference_root": args.reference_root,
        "zdim": args.zdim,
        "n_datasets": len(rows),
        "rows": rows,
    }
    with (output_dir / "aggregate_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(aggregate, fh, indent=2, default=str)

    print("\n=== AGGREGATE RESULTS ===\n", flush=True)
    print_comparison_table(rows, args.zdim)
    print(f"\nOutputs saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

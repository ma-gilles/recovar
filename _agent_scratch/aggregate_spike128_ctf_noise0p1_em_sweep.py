from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recovar import utils
from recovar.commands import spike_kernel_report as skr


DEFAULT_RUN_DIR = Path(
    "/scratch/gpfs/GILLES/mg6942/runs/spike_grid128_box320_ctf_noise0p1_n200k_raw_index_embsigma100_20260514"
)
RUN_DIR = DEFAULT_RUN_DIR
MASK_PATH = RUN_DIR / "05_masks/mask_crafted_level0p0126_cosine3_128.mrc"
BASE_TAG = "em_deg3_i1_h1_32"
SWEEP_TAGS = [
    "em_d3_i2_q5_plain_h1_32",
    "em_d3_i2_q7_plain_h1_32",
    "em_d3_i2_q5_temp2_damp0p5_h1_32",
    "em_d3_i2_q7_temp2_mix0p1_damp0p5_h1_32",
    "em_d3_i2_q5_temp2_damp0p5_tauMean1_h1_32",
    "em_d3_i2_q5_temp2_damp0p5_tauMean10_h1_32",
    "em_d5_i1_q5_plain_h1_32",
    "em_d5_i1_q7_plain_h1_32",
    "em_d5_i2_q7_temp2_damp0p5_tauMean10_h1_32",
    "em_d3_i3_q5_plain_h1_32",
    "em_d3_i4_q5_plain_h1_32",
    "em_d3_i3_q5_tauMean0p1_h1_32",
    "em_d3_i3_q5_tauMean1_h1_32",
    "em_d3_i3_q5_tauMean10_h1_32",
    "em_d5_i2_q5_plain_h1_32",
    "em_d5_i2_q5_tauMean1_h1_32",
    "em_d5_i3_q5_tauMean1_h1_32",
]


def _finite_float(value, default=np.nan):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def report_dir(tag: str) -> Path:
    return RUN_DIR / f"08_kernel_report_mask_crafted_level0p0126_cosine3_local_poly_{tag}_lazy"


def discover_tags() -> list[str]:
    prefix = "08_kernel_report_mask_crafted_level0p0126_cosine3_local_poly_"
    suffix = "_lazy"
    tags = [BASE_TAG] + SWEEP_TAGS
    seen = set(tags)
    for path in sorted(RUN_DIR.glob(f"{prefix}*{suffix}")):
        name = path.name
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        tag = name[len(prefix) : -len(suffix)]
        if tag and tag not in seen:
            tags.append(tag)
            seen.add(tag)
    return tags


def read_summary(tag: str) -> dict | None:
    path = report_dir(tag) / "summary.json"
    if not path.exists():
        return None
    with path.open() as f:
        summary = json.load(f)
    return summary


def _local_poly_em_params(report: Path) -> dict:
    metrics_path = report / "candidate_metrics.csv"
    if not metrics_path.exists():
        return {}
    with metrics_path.open() as f:
        for row in csv.DictReader(f):
            if row["mode"] != "local_poly_em":
                continue
            candidate_path = Path(row["path"])
            params_path = candidate_path.parent.parent / "params.pkl"
            if not params_path.exists():
                return {}
            params = utils.pickle_load(params_path)
            return params.get("local_poly_em", params.get("local_poly", {}))
    return {}


def _summarize_diagnostics(params: dict) -> dict:
    out = {
        "basis": params.get("basis"),
        "reg_type": params.get("pol_reg_type"),
        "eta": params.get("pol_reg_eta"),
        "reg_power": params.get("pol_reg_power"),
        "degree": params.get("degree"),
        "q": params.get("n_quadrature"),
        "em_iters": params.get("n_iterations"),
        "basis_gram_condition_median": np.nan,
        "basis_gram_condition_p95": np.nan,
        "basis_gram_condition_max": np.nan,
        "normal_eq_condition_median": np.nan,
        "normal_eq_condition_p95": np.nan,
        "normal_eq_condition_max": np.nan,
        "posterior_entropy_mean": np.nan,
        "effective_quadrature_nodes_mean": np.nan,
        "fraction_max_gamma_gt_0p9": np.nan,
        "fraction_max_gamma_gt_0p99": np.nan,
    }
    half_diagnostics = params.get("half_diagnostics") or []
    basis_conds = []
    normal_medians = []
    normal_p95 = []
    normal_max = []
    entropies = []
    eff_nodes = []
    frac09 = []
    frac099 = []
    for half_diag in half_diagnostics:
        for stage in half_diag:
            for info in stage.get("basis_info", []) or []:
                val = _finite_float(info.get("basis_gram_condition"))
                if np.isfinite(val):
                    basis_conds.append(val)
            for info in stage.get("solve_diagnostics", []) or []:
                val = _finite_float(info.get("normal_eq_condition_median"))
                if np.isfinite(val):
                    normal_medians.append(val)
                val = _finite_float(info.get("normal_eq_condition_p95"))
                if np.isfinite(val):
                    normal_p95.append(val)
                val = _finite_float(info.get("normal_eq_condition_max"))
                if np.isfinite(val):
                    normal_max.append(val)
            for info in stage.get("bandwidths", []) or []:
                val = _finite_float(info.get("mean_gamma_entropy"))
                if np.isfinite(val):
                    entropies.append(val)
                val = _finite_float(info.get("effective_quadrature_nodes"))
                if np.isfinite(val):
                    eff_nodes.append(val)
                val = _finite_float(info.get("fraction_max_gamma_gt_0p9"))
                if np.isfinite(val):
                    frac09.append(val)
                val = _finite_float(info.get("fraction_max_gamma_gt_0p99"))
                if np.isfinite(val):
                    frac099.append(val)

    if basis_conds:
        out["basis_gram_condition_median"] = float(np.median(basis_conds))
        out["basis_gram_condition_p95"] = float(np.percentile(basis_conds, 95))
        out["basis_gram_condition_max"] = float(np.max(basis_conds))
    if normal_medians:
        out["normal_eq_condition_median"] = float(np.median(normal_medians))
    if normal_p95:
        out["normal_eq_condition_p95"] = float(np.max(normal_p95))
    if normal_max:
        out["normal_eq_condition_max"] = float(np.max(normal_max))
    if entropies:
        out["posterior_entropy_mean"] = float(np.mean(entropies))
    if eff_nodes:
        out["effective_quadrature_nodes_mean"] = float(np.mean(eff_nodes))
    if frac09:
        out["fraction_max_gamma_gt_0p9"] = float(np.mean(frac09))
    if frac099:
        out["fraction_max_gamma_gt_0p99"] = float(np.mean(frac099))
    return out


def _masked_halfmap_fsc_curve(report: Path, best_idx: int) -> tuple[list[float], list[float]]:
    metrics_path = report / "candidate_metrics.csv"
    if not metrics_path.exists() or not MASK_PATH.exists():
        return [], []
    candidate_path = None
    with metrics_path.open() as f:
        for row in csv.DictReader(f):
            if row["mode"] == "local_poly_em" and int(row["candidate_index_0based"]) == int(best_idx):
                candidate_path = Path(row["path"])
                break
    if candidate_path is None:
        return [], []
    state_dir = candidate_path.parent.parent
    filename = candidate_path.name
    half1_path = state_dir / "estimates_half1_unfil" / filename
    half2_path = state_dir / "estimates_half2_unfil" / filename
    if not half1_path.exists() or not half2_path.exists():
        return [], []
    half1 = np.asarray(utils.load_mrc(half1_path), dtype=np.float32)
    half2 = np.asarray(utils.load_mrc(half2_path), dtype=np.float32)
    mask = np.asarray(utils.load_mrc(MASK_PATH), dtype=np.float32)
    labels, n_shells = skr._shell_labels(half1.shape)
    ft1 = skr._numpy_dft3(half1 * mask).reshape(-1)
    ft2 = skr._numpy_dft3(half2 * mask).reshape(-1)
    flat_labels = labels.reshape(-1)
    cross = np.bincount(flat_labels, weights=np.real(ft1 * np.conj(ft2)), minlength=n_shells)
    p1 = np.bincount(flat_labels, weights=np.abs(ft1) ** 2, minlength=n_shells)
    p2 = np.bincount(flat_labels, weights=np.abs(ft2) ** 2, minlength=n_shells)
    fsc = cross / np.maximum(np.sqrt(p1 * p2), 1e-30)
    voxel_size = 2.5
    try:
        voxel_size = float(utils.pickle_load(state_dir / "params.pkl")["voxel_size"])
    except Exception:
        pass
    freq = np.arange(n_shells, dtype=np.float64) / (half1.shape[0] * voxel_size)
    return freq.tolist(), fsc.astype(np.float64).tolist()


def _row_by_tag(rows: list[dict], tag: str) -> dict | None:
    for row in rows:
        if row.get("tag") == tag and row.get("status") == "complete":
            return row
    return None


def _row_float(row: dict | None, key: str, default=np.nan) -> float:
    if row is None:
        return default
    return _finite_float(row.get(key), default=default)


def _curve_arrays(curve_payload: dict, tag: str, *, halfmap: bool = False):
    curve = curve_payload.get(tag)
    if curve is None:
        return None, None
    freq_key = "halfmap_frequency_1_per_A" if halfmap else "oracle_frequency_1_per_A"
    fsc_key = "halfmap_fsc" if halfmap else "oracle_fsc"
    freq = np.asarray(curve.get(freq_key) or [], dtype=np.float64)
    fsc = np.asarray(curve.get(fsc_key) or [], dtype=np.float64)
    if freq.size == 0 or fsc.size == 0:
        return None, None
    return freq, fsc


def _plot_curve_group(
    out_path: Path,
    title: str,
    curve_payload: dict,
    rows: list[dict],
    entries: list[tuple[str, str]],
    *,
    halfmap: bool = False,
    ylim=(-0.08, 1.03),
) -> bool:
    plotted = False
    fig, ax = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)
    for idx, (tag, label) in enumerate(entries):
        freq, fsc = _curve_arrays(curve_payload, tag, halfmap=halfmap)
        if freq is None:
            continue
        row = _row_by_tag(rows, tag)
        mean = _row_float(row, "local_poly_em_oracle_mean_fsc")
        shell32 = _row_float(row, "local_poly_em_oracle_shell32")
        suffix = ""
        if np.isfinite(mean):
            suffix = f"  mean={mean:.4f}, s32={shell32:.4f}"
        ax.plot(freq, fsc, linewidth=2.2, label=f"{label}{suffix}")
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.axhline(0.5, color="0.55", linestyle="--", linewidth=0.8)
    ax.axhline(1 / 7, color="0.55", linestyle=":", linewidth=0.8)
    ax.set_xlim(0, 0.20)
    ax.set_ylim(*ylim)
    ax.set_xlabel("spatial frequency (1/A)")
    ylabel = "masked halfmap FSC" if halfmap else "oracle FSC vs GT"
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="lower left")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def _write_readable_plots(out_dir: Path, curve_payload: dict, rows: list[dict]) -> dict[str, str]:
    plot_specs = [
        (
            "fsc_readable_01_baselines.png",
            "Oracle FSC: main baselines",
            [
                ("basis_baseline_d5_i1_q5_mono_none", "d5 q5 EM1 mono none"),
                ("basis_baseline_d5_i1_q7_mono_none", "d5 q7 EM1 mono none"),
                ("basis_baseline_d5_i2_q5_mono_none", "d5 q5 EM2 mono none"),
                ("em_d5_i2_q5_plain_h1_32", "old d5 q5 EM2 plain"),
            ],
            False,
        ),
        (
            "fsc_readable_02_q5_em1_basis_reg.png",
            "Oracle FSC: d5 q5 EM1 basis and regularization",
            [
                ("basis_baseline_d5_i1_q5_mono_none", "mono none"),
                ("basis_d5_i1_q5_monomial_deriv2_eta30", "mono deriv2 eta=30"),
                ("basis_d5_i1_q5_weighted_cholesky_coeff_eta0p3", "w-chol coeff eta=0.3"),
                ("basis_d5_i1_q5_weighted_cholesky_deriv2_eta0p3", "w-chol deriv2 eta=0.3"),
            ],
            False,
        ),
        (
            "fsc_readable_03_d5_hardcases.png",
            "Oracle FSC: d5 hard cases",
            [
                ("basis_baseline_d5_i2_q5_mono_none", "q5 EM2 mono none"),
                ("basis_hard_d5_i2_q5_monomial_deriv2_eta30", "q5 EM2 mono deriv2 eta=30"),
                ("basis_baseline_d5_i1_q7_mono_none", "q7 EM1 mono none"),
                ("basis_hard_d5_i2_q7_monomial_deriv2_eta30", "q7 EM2 mono deriv2 eta=30"),
                ("basis_hard_d5_i2_q7_weighted_cholesky_coeff_eta0p3", "q7 EM2 w-chol coeff eta=0.3"),
            ],
            False,
        ),
        (
            "fsc_readable_04_degree7.png",
            "Oracle FSC: degree 7 checks",
            [
                ("basis_hard_d7_i1_q5_monomial_none", "d7 q5 mono none"),
                ("basis_hard_d7_i1_q5_monomial_coeff_eta1", "d7 q5 mono coeff eta=1"),
                ("basis_hard_d7_i1_q5_weighted_cholesky_coeff_eta0p3", "d7 q5 w-chol coeff eta=0.3"),
                ("basis_hard_d7_i1_q7_monomial_none", "d7 q7 mono none"),
                ("basis_hard_d7_i1_q7_weighted_cholesky_coeff_eta0p3", "d7 q7 w-chol coeff eta=0.3"),
            ],
            False,
        ),
        (
            "fsc_readable_05_halfmap_selected.png",
            "Masked halfmap FSC: selected estimates",
            [
                ("basis_baseline_d5_i2_q5_mono_none", "d5 q5 EM2 mono none"),
                ("basis_hard_d5_i2_q5_monomial_deriv2_eta30", "d5 q5 EM2 mono deriv2 eta=30"),
                ("basis_d5_i1_q5_weighted_cholesky_coeff_eta0p3", "d5 q5 EM1 w-chol coeff eta=0.3"),
                ("basis_hard_d7_i1_q5_monomial_none", "d7 q5 EM1 mono none"),
            ],
            True,
        ),
    ]
    written = {}
    for filename, title, entries, halfmap in plot_specs:
        path = out_dir / filename
        if _plot_curve_group(path, title, curve_payload, rows, entries, halfmap=halfmap):
            written[filename] = str(path)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    return parser.parse_args()


def main() -> None:
    global RUN_DIR, MASK_PATH
    args = parse_args()
    RUN_DIR = args.run_dir
    MASK_PATH = RUN_DIR / "05_masks/mask_crafted_level0p0126_cosine3_128.mrc"
    out_dir = RUN_DIR / "09_local_poly_em_sweep_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    tags = discover_tags()
    rows = []
    curves = []
    curve_payload = {}
    for tag in tags:
        summary = read_summary(tag)
        if summary is None:
            rows.append({"tag": tag, "status": "missing"})
            continue
        oracle = summary["oracle"]
        lpem = oracle["local_poly_em"]
        combined = oracle["combined_fsc_oracle"]
        best = summary["best"]["local_poly_em"]
        row = {
            "tag": tag,
            "status": "complete",
            "local_poly_em_oracle_mean_fsc": lpem["fsc_oracle_mean_0_to_score_max"],
            "local_poly_em_oracle_shell32": lpem["fsc_oracle_shell32"],
            "local_poly_em_error_choice_mean_fsc": lpem["error_choice_fsc_mean_0_to_score_max"],
            "local_poly_em_error_choice_shell32": lpem["error_choice_fsc_shell32"],
            "combined_oracle_mean_fsc": combined["fsc_oracle_mean_0_to_score_max"],
            "combined_oracle_shell32": combined["fsc_oracle_shell32"],
            "best_single_candidate_index": best["best_by_mean_fsc_index_0based"],
            "best_single_candidate_parameter": best["best_by_mean_fsc_parameter"],
            "best_single_candidate_mean_fsc": best["best_by_mean_fsc"],
            "best_single_candidate_shell32": best["best_by_mean_fsc_shell32"],
            "report": str(report_dir(tag)),
        }
        params = _local_poly_em_params(report_dir(tag))
        row.update(_summarize_diagnostics(params))
        rows.append(row)
        choices_path = Path(summary["oracle_shell_choices_csv"])
        with choices_path.open() as f:
            choice_rows = [r for r in csv.DictReader(f) if r["mode"] == "local_poly_em"]
        freq = np.asarray([float(r["frequency_1_per_A"]) for r in choice_rows], dtype=np.float64)
        fsc = np.asarray([float(r["fsc_oracle_fsc"]) for r in choice_rows], dtype=np.float64)
        curves.append((tag, freq, fsc))
        half_freq, half_fsc = _masked_halfmap_fsc_curve(
            report_dir(tag),
            int(row["best_single_candidate_index"]),
        )
        curve_payload[tag] = {
            "oracle_frequency_1_per_A": freq.tolist(),
            "oracle_fsc": fsc.tolist(),
            "halfmap_frequency_1_per_A": half_freq,
            "halfmap_fsc": half_fsc,
        }

    csv_path = out_dir / "summary.csv"
    fieldnames = [
        "tag",
        "status",
        "basis",
        "reg_type",
        "eta",
        "reg_power",
        "degree",
        "q",
        "em_iters",
        "local_poly_em_oracle_mean_fsc",
        "local_poly_em_oracle_shell32",
        "local_poly_em_error_choice_mean_fsc",
        "local_poly_em_error_choice_shell32",
        "combined_oracle_mean_fsc",
        "combined_oracle_shell32",
        "best_single_candidate_index",
        "best_single_candidate_parameter",
        "best_single_candidate_mean_fsc",
        "best_single_candidate_shell32",
        "basis_gram_condition_median",
        "basis_gram_condition_p95",
        "basis_gram_condition_max",
        "normal_eq_condition_median",
        "normal_eq_condition_p95",
        "normal_eq_condition_max",
        "posterior_entropy_mean",
        "effective_quadrature_nodes_mean",
        "fraction_max_gamma_gt_0p9",
        "fraction_max_gamma_gt_0p99",
        "report",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    complete_rows = [r for r in rows if r.get("status") == "complete"]
    best_row = max(complete_rows, key=lambda r: float(r["local_poly_em_oracle_mean_fsc"])) if complete_rows else None

    fig, ax = plt.subplots(figsize=(10.5, 6.0), constrained_layout=True)
    for tag, freq, fsc in curves:
        lw = 2.8 if best_row is not None and tag == best_row["tag"] else 1.3
        alpha = 1.0 if best_row is not None and tag == best_row["tag"] else 0.65
        ax.plot(freq, fsc, linewidth=lw, alpha=alpha, label=tag)
    ax.axhline(0.5, color="0.55", linestyle="--", linewidth=0.8)
    ax.axhline(1 / 7, color="0.55", linestyle=":", linewidth=0.8)
    ax.set_xlim(0, 0.20)
    ax.set_ylim(-0.08, 1.03)
    ax.set_xlabel("spatial frequency (1/A)")
    ax.set_ylabel("local_poly_em oracle FSC vs GT")
    ax.set_title("EM local polynomial sweep: per-shell oracle FSC")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    plot_path = out_dir / "local_poly_em_oracle_fsc_sweep.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    curves_json_path = out_dir / "curves.json"
    curves_json_path.write_text(json.dumps(curve_payload, indent=2, sort_keys=True) + "\n")
    readable_plots = _write_readable_plots(out_dir, curve_payload, rows)

    payload = {
        "run_dir": str(RUN_DIR),
        "summary_csv": str(csv_path),
        "curves_json": str(curves_json_path),
        "plot": str(plot_path),
        "readable_plots": readable_plots,
        "best_by_local_poly_em_oracle_mean_fsc": best_row,
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

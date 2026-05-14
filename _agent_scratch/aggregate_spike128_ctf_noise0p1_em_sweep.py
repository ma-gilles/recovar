from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RUN_DIR = Path("/scratch/gpfs/GILLES/mg6942/runs/spike_grid128_box320_ctf_noise0p1_n200k_raw_index_embsigma100_20260514")
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


def report_dir(tag: str) -> Path:
    return RUN_DIR / f"08_kernel_report_mask_crafted_level0p0126_cosine3_local_poly_{tag}_lazy"


def read_summary(tag: str) -> dict | None:
    path = report_dir(tag) / "summary.json"
    if not path.exists():
        return None
    with path.open() as f:
        summary = json.load(f)
    return summary


def main() -> None:
    out_dir = RUN_DIR / "09_local_poly_em_sweep_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    tags = [BASE_TAG] + SWEEP_TAGS
    rows = []
    curves = []
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
        rows.append(row)
        choices_path = Path(summary["oracle_shell_choices_csv"])
        with choices_path.open() as f:
            choice_rows = [r for r in csv.DictReader(f) if r["mode"] == "local_poly_em"]
        freq = np.asarray([float(r["frequency_1_per_A"]) for r in choice_rows], dtype=np.float64)
        fsc = np.asarray([float(r["fsc_oracle_fsc"]) for r in choice_rows], dtype=np.float64)
        curves.append((tag, freq, fsc))

    csv_path = out_dir / "summary.csv"
    fieldnames = [
        "tag",
        "status",
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

    payload = {
        "run_dir": str(RUN_DIR),
        "summary_csv": str(csv_path),
        "plot": str(plot_path),
        "best_by_local_poly_em_oracle_mean_fsc": best_row,
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

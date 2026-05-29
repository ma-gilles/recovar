from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from recovar.commands import spike_kernel_report as skr
from recovar import utils


def _read_candidate_paths(report_dir: Path) -> dict[str, list[Path]]:
    paths: dict[str, list[Path]] = {}
    with (report_dir / "candidate_metrics.csv").open() as f:
        for row in csv.DictReader(f):
            mode = row["mode"]
            idx = int(row["candidate_index_0based"])
            mode_paths = paths.setdefault(mode, [])
            while len(mode_paths) <= idx:
                mode_paths.append(Path())
            mode_paths[idx] = Path(row["path"])
    for mode, mode_paths in paths.items():
        missing = [idx for idx, path in enumerate(mode_paths) if not path]
        if missing:
            raise RuntimeError(f"Missing candidate paths for {mode}: {missing}")
    return paths


def _read_oracle_rows(report_dir: Path) -> dict[str, list[dict[str, str]]]:
    rows: dict[str, list[dict[str, str]]] = {}
    with (report_dir / "oracle_shell_choices.csv").open() as f:
        for row in csv.DictReader(f):
            rows.setdefault(row["mode"], []).append(row)
    return rows


def _assemble_shell_oracle(
    paths: list[Path],
    shell_choice: np.ndarray,
    out_path: Path,
    voxel_size: float,
) -> np.ndarray:
    first = np.asarray(utils.load_mrc(paths[0]), dtype=np.float32)
    labels, _ = skr._shell_labels(first.shape)
    flat_labels = labels.ravel()
    oracle_ft = np.zeros(first.shape, dtype=np.complex128)
    for idx, path in enumerate(paths):
        selected = shell_choice[flat_labels] == idx
        if not np.any(selected):
            continue
        ft = skr._numpy_dft3(np.asarray(utils.load_mrc(path), dtype=np.float32))
        oracle_ft.ravel()[selected] = ft.ravel()[selected]
    oracle = skr._inverse_numpy_dft3(oracle_ft).astype(np.float32)
    utils.write_mrc(str(out_path), oracle, voxel_size=voxel_size)
    return oracle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target-state", type=int, default=50)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    report_dir = args.report_dir
    out_dir = args.out_dir or (report_dir / "oracle_volumes_for_rsync")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    summary_path = report_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    candidate_paths = _read_candidate_paths(report_dir)
    oracle_rows = _read_oracle_rows(report_dir)
    first_path = next(iter(candidate_paths.values()))[0]
    first = np.asarray(utils.load_mrc(first_path), dtype=np.float32)
    labels, n_shells = skr._shell_labels(first.shape)
    try:
        params_path = first_path.parent.parent / "params.pkl"
        voxel_size = float(utils.pickle_load(params_path)["voxel_size"])
    except Exception:
        voxel_size = 1.0

    manifest: dict[str, object] = {
        "report_dir": str(report_dir),
        "run_dir": str(args.run_dir),
        "summary": str(summary_path),
        "oracle_shell_choices": str(report_dir / "oracle_shell_choices.csv"),
        "candidate_metrics": str(report_dir / "candidate_metrics.csv"),
        "target_state": int(args.target_state),
        "voxel_size_A": voxel_size,
        "outputs": {},
        "notes": "Shell-oracle volumes are assembled from estimates_filt candidate volumes using per-shell GT FSC choices.",
    }

    target_path = args.run_dir / f"04_ground_truth/gt_vol{args.target_state:04d}.mrc"
    shutil.copy2(target_path, out_dir / f"gt_state{args.target_state:04d}.mrc")
    shutil.copy2(args.mask, out_dir / args.mask.name)
    for name in [
        "oracle_fsc_curves_by_method.png",
        "oracle_min_error_choice_fsc_curves_by_method.png",
        "all_candidate_fsc_three_modes.png",
        "best_mean_fsc_candidates.png",
    ]:
        src = report_dir / "plots" / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
    for name in ["summary.json", "oracle_summary.json", "oracle_shell_choices.csv", "candidate_metrics.csv"]:
        src = report_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    mode_choices: dict[str, np.ndarray] = {}
    for mode, rows in oracle_rows.items():
        choice = np.zeros(n_shells, dtype=np.int64)
        for row in rows:
            shell = int(row["shell"])
            if shell < n_shells:
                choice[shell] = int(row["fsc_oracle_candidate_index_0based"])
        mode_choices[mode] = choice
        out_path = out_dir / f"{mode}_fsc_shell_oracle.mrc"
        _assemble_shell_oracle(candidate_paths[mode], choice, out_path, voxel_size)
        manifest["outputs"][mode] = {
            "path": str(out_path),
            "candidate_indices_0based": sorted(int(x) for x in np.unique(choice)),
        }

    combined_choice_mode = []
    combined_choice_idx = np.zeros(n_shells, dtype=np.int64)
    for shell in range(n_shells):
        best_mode = None
        best_idx = 0
        best_fsc = -np.inf
        for mode, rows in oracle_rows.items():
            if shell >= len(rows):
                continue
            fsc = float(rows[shell]["fsc_oracle_fsc"])
            if fsc > best_fsc:
                best_fsc = fsc
                best_mode = mode
                best_idx = int(rows[shell]["fsc_oracle_candidate_index_0based"])
        if best_mode is None:
            best_mode = next(iter(candidate_paths))
        combined_choice_mode.append(best_mode)
        combined_choice_idx[shell] = best_idx

    combined_ft = np.zeros(first.shape, dtype=np.complex128)
    flat_labels = labels.ravel()
    for mode, paths in candidate_paths.items():
        for idx, path in enumerate(paths):
            shell_selected = np.asarray(
                [(m == mode and combined_choice_idx[s] == idx) for s, m in enumerate(combined_choice_mode)],
                dtype=bool,
            )
            selected = shell_selected[flat_labels]
            if not np.any(selected):
                continue
            ft = skr._numpy_dft3(np.asarray(utils.load_mrc(path), dtype=np.float32))
            combined_ft.ravel()[selected] = ft.ravel()[selected]
    combined = skr._inverse_numpy_dft3(combined_ft).astype(np.float32)
    combined_path = out_dir / "combined_all_methods_fsc_shell_oracle.mrc"
    utils.write_mrc(str(combined_path), combined, voxel_size=voxel_size)
    manifest["outputs"]["combined_all_methods"] = {
        "path": str(combined_path),
        "selected_modes": sorted(set(combined_choice_mode)),
    }

    readme = out_dir / "README.txt"
    readme.write_text(
        "Oracle volume bundle for the spike128 CTF/noise0.1 local-poly EM sweep.\n"
        "Use the MRCs directly for visual inspection. The combined volume chooses the highest-GT-FSC method per shell.\n"
        f"Report: {report_dir}\n"
        f"Best row: {summary.get('best', {}).get('local_poly_em', {})}\n"
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out_dir": str(out_dir), "manifest": str(out_dir / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()

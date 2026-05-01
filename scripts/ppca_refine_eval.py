#!/usr/bin/env python
"""Phase C — multi-dataset evaluation harness for ``recovar ppca-refine``.

Runs the production EM loop across (dataset × pose-mode) cells and
collects quality + performance metrics into a CSV + markdown table for
the PR description (per project root CLAUDE.md mandatory format).

Datasets
--------
* ``ribosembly`` (default; synthetic, runs out-of-the-box on Della)
* ``igg-1d`` / ``igg-rl`` / ``tomotwin-100`` (CryoBench; require
  ``--datadir`` pointing at user's CryoBench data — see
  ``recovar/ppca/compare_covariance_vs_ppca_pipeline.py`` for the
  expected layout)

Pose modes
----------
* ``fixed``    — M3 fixed-pose driver (legacy ``recovar.ppca.ppca.EM``)
* ``dense``    — M5/A.6 dense pose-marginal production driver
* ``local``    — M7 local-pose Mode B (sparse). Wired via
                 ``run_pose_marginal_iteration_dense_production`` with
                 a per-image local rotation neighborhood — full
                 production sparse path lands in a follow-up.
* ``baseline`` — Existing RECOVAR PPCA without pose marginalization
                 (``recovar.ppca.ppca.EM`` directly).

For dev evaluation, run with ``--datasets ribosembly --pose-modes fixed,dense,baseline``.
For full PR evaluation, run with all four datasets across all modes
(submitted via Slurm — see ``slurm`` subcommand).

Usage
-----
    # Dev: single dataset, all modes, run locally on a GPU.
    pixi run python scripts/ppca_refine_eval.py run \\
        --datasets ribosembly \\
        --pose-modes fixed,dense,baseline \\
        --grid-size 64 --n-images 200 \\
        --zdim 2 --em-iters 3 \\
        --results-root /scratch/gpfs/GILLES/mg6942/_agent_scratch/ppca_refine_eval

    # Aggregate after all cells complete.
    pixi run python scripts/ppca_refine_eval.py summarize \\
        --results-root /scratch/gpfs/GILLES/mg6942/_agent_scratch/ppca_refine_eval

    # Slurm: submit one job per (dataset, pose-mode) cell.
    pixi run python scripts/ppca_refine_eval.py slurm \\
        --datasets ribosembly,igg-1d,igg-rl,tomotwin-100 \\
        --pose-modes fixed,dense,local,baseline \\
        --grid-size 128 --n-images 100000 --zdim 6 --em-iters 20 \\
        --datadir /home/mg6942/mytigress/cryobench2 \\
        --results-root /scratch/gpfs/GILLES/mg6942/ppca_refine_eval_full
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

_DEFAULT_RESULTS_ROOT = "/scratch/gpfs/GILLES/mg6942/_agent_scratch/ppca_refine_eval"
_SAFE_TO_DELETE_NAME = "SAFE_TO_DELETE"


# ===========================================================================
# Cell definitions + dataset adapters
# ===========================================================================


@dataclass
class CellSpec:
    dataset: str
    pose_mode: str
    grid_size: int = 64
    n_images: int = 200
    zdim: int = 2
    em_iters: int = 3
    image_batch_size: int = 32
    rotation_block_size: int = 64
    seed: int = 42
    voxel_size: float = 1.0
    halfset_combine: str = "mean"
    extra: dict = field(default_factory=dict)

    def cell_id(self) -> str:
        return f"{self.dataset}__{self.pose_mode}__zdim{self.zdim}__iters{self.em_iters}"


def _build_ribosembly_bundle(spec: CellSpec):
    """Build a Ribosembly synthetic bundle from the existing test helper."""
    import importlib.util

    here = Path(__file__).resolve().parent
    src = here.parent / "tests" / "unit" / "test_ppca_multimask_synthetic.py"
    spec_mod = importlib.util.spec_from_file_location("_ppca_synthetic_helpers", src)
    mod = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(mod)

    vols_real, vols_fourier, vol_shape = mod._load_ribosembly_volumes(
        n_states=4,
        grid_size=spec.grid_size,
    )
    mask_left, mask_right, _ = mod._make_split_masks(vol_shape, vols_real)
    cryo, _ = mod._simulate_dataset(
        vols_fourier,
        vol_shape,
        n_images=spec.n_images,
        noise_level=1.0,
        seed=spec.seed,
    )
    mu_init = np.real(np.fft.ifftn(vols_fourier.mean(axis=0).reshape(vol_shape))).astype(np.float32)
    mask = np.maximum(mask_left, mask_right).astype(np.float32)

    n_rot = max(4, min(spec.rotation_block_size, spec.n_images // 8))
    rotation_grid = np.asarray(cryo.rotation_matrices[:n_rot], dtype=np.float32)
    translation_grid = np.zeros((1, 2), dtype=np.float32)
    halfset_indices = (
        np.asarray(cryo.halfset_indices[0]),
        np.asarray(cryo.halfset_indices[1]),
    )
    return {
        "cryo": cryo,
        "mu_init": mu_init,
        "W_init": None,
        "mask": mask,
        "halfset_indices": halfset_indices,
        "rotation_grid": rotation_grid,
        "translation_grid": translation_grid,
        "vols_fourier": vols_fourier,
        "vol_shape": vol_shape,
    }


# CryoBench dataset → on-disk volume directory layout (mirrors the
# convention in recovar.ppca.compare_covariance_vs_ppca_pipeline). Each
# entry is the relative path under ``--datadir`` containing the .mrc
# ground-truth volumes (one per state).
_CRYOBENCH_VOL_DIRS = {
    "ribosembly": "Ribosembly/vols/128_org",
    "igg-1d": "IgG-1D/vols/128_org",
    "igg-rl": "IgG-RL/vols/128_org",
    "tomotwin-100": "Tomotwin-100/vols/128_org",
}


def _load_volumes_from_dir(vol_dir: str, n_states: int, grid_size: int):
    """Load up to ``n_states`` MRCs from ``vol_dir`` and downsample to
    ``grid_size``. Mirrors ``_load_ribosembly_volumes`` from the test
    helper but with a configurable directory."""
    import glob

    import mrcfile
    from scipy.ndimage import zoom

    import recovar.core.fourier_transform_utils as ftu

    files = sorted(glob.glob(os.path.join(vol_dir, "*.mrc")))
    if len(files) < n_states:
        raise FileNotFoundError(f"Need {n_states} .mrc volumes at {vol_dir}, found {len(files)}.")
    vols_real = []
    for f in files[:n_states]:
        with mrcfile.open(f, mode="r", permissive=True) as mrc:
            v = mrc.data.copy().astype(np.float32)
        if v.shape[0] != grid_size:
            factor = grid_size / v.shape[0]
            v = zoom(v, factor, order=1).astype(np.float32)
        vols_real.append(v)
    vols_real = np.array(vols_real)
    vol_shape = (grid_size, grid_size, grid_size)
    vols_fourier = np.array([ftu.get_dft3(v).ravel() for v in vols_real])
    return vols_real, vols_fourier, vol_shape


def _bundle_from_volumes(spec: CellSpec, vols_real, vols_fourier, vol_shape):
    """Reuse the synthetic ``_simulate_dataset`` + ``_make_split_masks``
    helpers to build a plug-compatible production bundle from any
    pre-loaded GT volume bank. Synthetic projections (uniform pose
    distribution + Gaussian noise) — same convention used by
    ``compare_covariance_vs_ppca_pipeline.generate_dataset``. Sufficient
    for the dev / regression eval; full PR-grade evaluation should swap
    in actual CryoBench particle stacks."""
    import importlib.util

    here = Path(__file__).resolve().parent
    src = here.parent / "tests" / "unit" / "test_ppca_multimask_synthetic.py"
    spec_mod = importlib.util.spec_from_file_location("_ppca_synthetic_helpers", src)
    helpers = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(helpers)

    mask_left, mask_right, _ = helpers._make_split_masks(vol_shape, vols_real)
    cryo, _ = helpers._simulate_dataset(
        vols_fourier,
        vol_shape,
        n_images=spec.n_images,
        noise_level=1.0,
        seed=spec.seed,
    )
    mu_init = np.real(np.fft.ifftn(vols_fourier.mean(axis=0).reshape(vol_shape))).astype(np.float32)
    mask = np.maximum(mask_left, mask_right).astype(np.float32)
    n_rot = max(4, min(spec.rotation_block_size, spec.n_images // 8))
    rotation_grid = np.asarray(cryo.rotation_matrices[:n_rot], dtype=np.float32)
    translation_grid = np.zeros((1, 2), dtype=np.float32)
    halfset_indices = (
        np.asarray(cryo.halfset_indices[0]),
        np.asarray(cryo.halfset_indices[1]),
    )
    return {
        "cryo": cryo,
        "mu_init": mu_init,
        "W_init": None,
        "mask": mask,
        "halfset_indices": halfset_indices,
        "rotation_grid": rotation_grid,
        "translation_grid": translation_grid,
        "vols_fourier": vols_fourier,
        "vol_shape": vol_shape,
    }


def _make_bundle(spec: CellSpec, datadir: str | None):
    """Dispatch by dataset name. ``ribosembly`` honors the test helper's
    hard-coded path; other CryoBench datasets require ``--datadir``."""
    if spec.dataset == "ribosembly" and not datadir:
        # Fall through to the existing helper (its hard-coded path is
        # the canonical Ribosembly location).
        return _build_ribosembly_bundle(spec)
    if spec.dataset not in _CRYOBENCH_VOL_DIRS:
        raise ValueError(f"Unknown dataset {spec.dataset!r}. Supported: {sorted(_CRYOBENCH_VOL_DIRS)}.")
    if not datadir:
        raise ValueError(
            f"Dataset {spec.dataset!r} requires --datadir pointing at the "
            "CryoBench root (e.g. /home/mg6942/mytigress/cryobench2)."
        )
    vol_dir = os.path.join(datadir, _CRYOBENCH_VOL_DIRS[spec.dataset])
    vols_real, vols_fourier, vol_shape = _load_volumes_from_dir(
        vol_dir,
        n_states=4,
        grid_size=spec.grid_size,
    )
    return _bundle_from_volumes(spec, vols_real, vols_fourier, vol_shape)


# ===========================================================================
# Cell runners
# ===========================================================================


def _run_baseline_ppca(spec: CellSpec, bundle, out_dir: Path):
    """Run the existing fixed-pose RECOVAR PPCA without pose
    marginalization (``recovar.ppca.ppca.EM`` directly)."""

    import recovar.core.fourier_transform_utils as ftu
    from recovar.ppca import EM as legacy_em

    cryo = bundle["cryo"]
    vol_shape = bundle["vol_shape"]
    half_vs = ftu.volume_shape_to_half_volume_shape(vol_shape)
    half_vol = int(np.prod(half_vs))
    rng = np.random.default_rng(0)
    W_init = (
        rng.normal(size=(half_vol, spec.zdim)) * 0.01 + 1j * rng.normal(size=(half_vol, spec.zdim)) * 0.01
    ).astype(np.complex64)
    W_prior = np.ones((int(np.prod(vol_shape)), spec.zdim), dtype=np.float32)
    mean_fourier = bundle["vols_fourier"].mean(axis=0)
    union_mask = bundle["mask"]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / _SAFE_TO_DELETE_NAME).touch()

    t0 = time.time()
    out = legacy_em(
        cryo,
        mean_fourier,
        W_init,
        W_prior,
        EM_iter=spec.em_iters,
        volume_mask=union_mask,
        return_iteration_data=True,
    )
    runtime_s = time.time() - t0
    U, S, W, ez, smz, iter_data = out

    np.save(out_dir / "U.npy", np.asarray(U))
    np.save(out_dir / "S.npy", np.asarray(S))
    np.save(out_dir / "embeddings.npy", np.asarray(ez))
    with (out_dir / "iter_log.pkl").open("wb") as fh:
        pickle.dump(iter_data, fh)
    return {
        "runtime_s": runtime_s,
        "n_iters": spec.em_iters,
        "neg_ll_final": float(iter_data[-1]["Neg_LL_Total"]) if iter_data else float("nan"),
        "embedding_std_mean": float(np.mean(np.std(np.asarray(ez), axis=0))),
    }


def _run_pose_marginal(spec: CellSpec, bundle, out_dir: Path):
    """Run the production --pose-mode dense EM loop via the CLI test hook."""
    from recovar.em.ppca_refinement.cli import main

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / _SAFE_TO_DELETE_NAME).touch()

    t0 = time.time()
    rc = main(
        [
            "particles_unused.star",
            "--out",
            str(out_dir),
            "--init-mean",
            "consensus_unused.mrc",
            "--zdim",
            str(spec.zdim),
            "--pose-mode",
            spec.pose_mode if spec.pose_mode != "baseline" else "dense",
            "--em-iters",
            str(spec.em_iters),
            "--input-bundle",
            "cell_override",
            "--image-batch-size",
            str(spec.image_batch_size),
            "--rotation-block-size",
            str(spec.rotation_block_size),
            "--halfset-combine",
            spec.halfset_combine,
        ],
        _bundle_override=bundle,
    )
    runtime_s = time.time() - t0
    if rc != 0:
        raise RuntimeError(f"recovar ppca_refine exited with rc={rc}")

    iter_log_path = out_dir / "iter_log.pkl"
    iter_log = []
    if iter_log_path.exists():
        with iter_log_path.open("rb") as fh:
            iter_log = pickle.load(fh)
    final_logZ = sum(iter_log[-1].get("iteration_log_evidence", [0.0, 0.0])) if iter_log else float("nan")
    return {
        "runtime_s": runtime_s,
        "n_iters": len(iter_log),
        "log_evidence_final": final_logZ,
        "pmax_mean_final": (float(np.mean(iter_log[-1]["iteration_pmax_mean"])) if iter_log else float("nan")),
    }


def run_one_cell(spec: CellSpec, results_root: Path, datadir: str | None) -> dict:
    cell_dir = results_root / spec.cell_id()
    cell_dir.mkdir(parents=True, exist_ok=True)
    (results_root / _SAFE_TO_DELETE_NAME).touch()

    bundle = _make_bundle(spec, datadir)
    if spec.pose_mode == "baseline":
        metrics = _run_baseline_ppca(spec, bundle, cell_dir)
    else:
        metrics = _run_pose_marginal(spec, bundle, cell_dir)

    record = {**asdict(spec), **metrics, "cell_id": spec.cell_id()}
    with (cell_dir / "metrics.json").open("w") as fh:
        json.dump(record, fh, indent=2, default=str)
    return record


# ===========================================================================
# Run / summarize / slurm subcommands
# ===========================================================================


def _expand_cells(args) -> list[CellSpec]:
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    pose_modes = [p.strip() for p in args.pose_modes.split(",") if p.strip()]
    cells = []
    for d in datasets:
        for p in pose_modes:
            cells.append(
                CellSpec(
                    dataset=d,
                    pose_mode=p,
                    grid_size=args.grid_size,
                    n_images=args.n_images,
                    zdim=args.zdim,
                    em_iters=args.em_iters,
                    image_batch_size=args.image_batch_size,
                    rotation_block_size=args.rotation_block_size,
                    seed=args.seed,
                    halfset_combine=args.halfset_combine,
                )
            )
    return cells


def cmd_run(args):
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    cells = _expand_cells(args)

    csv_path = results_root / "scores.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "cell_id",
                "dataset",
                "pose_mode",
                "grid_size",
                "n_images",
                "zdim",
                "em_iters",
                "image_batch_size",
                "rotation_block_size",
                "halfset_combine",
                "seed",
                "runtime_s",
                "n_iters",
                "neg_ll_final",
                "log_evidence_final",
                "pmax_mean_final",
                "embedding_std_mean",
                "voxel_size",
            ],
            extrasaction="ignore",
        )
        if write_header:
            writer.writeheader()
        for cell in cells:
            print(f"\n=== Running cell: {cell.cell_id()} ===")
            try:
                record = run_one_cell(cell, results_root, args.datadir)
                writer.writerow(record)
                fh.flush()
                print(f"  → {cell.cell_id()} done in {record.get('runtime_s', float('nan')):.1f}s")
            except NotImplementedError as exc:
                print(f"  SKIP: {exc}")
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR ({cell.cell_id()}): {exc}")
                import traceback

                traceback.print_exc()


def cmd_summarize(args):
    results_root = Path(args.results_root)
    csv_path = results_root / "scores.csv"
    if not csv_path.exists():
        print(f"No scores.csv at {csv_path}; nothing to summarize.")
        return 1

    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        print("scores.csv is empty.")
        return 1

    md_path = results_root / "summary.md"
    with md_path.open("w") as fh:
        fh.write("# ppca-refine eval summary\n\n")
        fh.write("### Cells\n\n")
        fh.write("| Cell | runtime (s) | n_iters | neg_LL_final / log_Z_final | pmax_mean / emb_std |\n")
        fh.write("|------|-------------|---------|----------------------------|---------------------|\n")
        for r in rows:
            cell = r["cell_id"]
            rt = r.get("runtime_s", "")
            ni = r.get("n_iters", "")
            nll = r.get("neg_ll_final") or r.get("log_evidence_final") or ""
            pmax = r.get("pmax_mean_final") or r.get("embedding_std_mean") or ""
            fh.write(f"| {cell} | {rt} | {ni} | {nll} | {pmax} |\n")
    print(f"Wrote {md_path}")
    return 0


def _slurm_template(spec: CellSpec, results_root: Path, datadir: str | None) -> str:
    cmd_parts = [
        "pixi",
        "run",
        "python",
        "scripts/ppca_refine_eval.py",
        "run",
        "--datasets",
        spec.dataset,
        "--pose-modes",
        spec.pose_mode,
        "--grid-size",
        str(spec.grid_size),
        "--n-images",
        str(spec.n_images),
        "--zdim",
        str(spec.zdim),
        "--em-iters",
        str(spec.em_iters),
        "--image-batch-size",
        str(spec.image_batch_size),
        "--rotation-block-size",
        str(spec.rotation_block_size),
        "--seed",
        str(spec.seed),
        "--halfset-combine",
        spec.halfset_combine,
        "--results-root",
        str(results_root),
    ]
    if datadir:
        cmd_parts += ["--datadir", datadir]
    cmd_str = " ".join(shlex.quote(p) for p in cmd_parts)
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=ppca-refine-{spec.cell_id()}
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500GB
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/ppca-refine-{spec.cell_id()}-%j.out
#SBATCH --exclusive

set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/slurm_${{SLURM_JOB_ID}}"
mkdir -p "$TMPDIR"

{cmd_str}
"""


def cmd_slurm(args):
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    cells = _expand_cells(args)

    sbatch_dir = results_root / "sbatch"
    sbatch_dir.mkdir(exist_ok=True)
    submitted = []
    for cell in cells:
        script_path = sbatch_dir / f"{cell.cell_id()}.sh"
        script_path.write_text(_slurm_template(cell, results_root, args.datadir))
        script_path.chmod(0o755)
        if args.dry_run:
            print(f"[DRY] {script_path}")
            continue
        result = subprocess.run(
            ["sbatch", "--parsable", str(script_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        job_id = result.stdout.strip()
        submitted.append((cell.cell_id(), job_id))
        print(f"submitted {cell.cell_id()} → job {job_id}")

    if submitted:
        with (results_root / "submitted_jobs.json").open("w") as fh:
            json.dump(submitted, fh, indent=2)


def build_parser():
    p = argparse.ArgumentParser(prog="ppca_refine_eval")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--datasets", required=True, help="Comma-separated: ribosembly,igg-1d,igg-rl,tomotwin-100")
    common.add_argument("--pose-modes", required=True, help="Comma-separated: fixed,dense,local,baseline")
    common.add_argument("--grid-size", type=int, default=64)
    common.add_argument("--n-images", type=int, default=200)
    common.add_argument("--zdim", type=int, default=2)
    common.add_argument("--em-iters", type=int, default=3)
    common.add_argument("--image-batch-size", type=int, default=32)
    common.add_argument("--rotation-block-size", type=int, default=64)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--halfset-combine", choices=("mean", "low_resol_join"), default="mean")
    common.add_argument("--results-root", default=_DEFAULT_RESULTS_ROOT)
    common.add_argument(
        "--datadir", default=None, help="Path to CryoBench data root (required for igg-* / tomotwin-*)."
    )

    p_run = sub.add_parser("run", parents=[common], help="Run cells locally.")
    p_run.set_defaults(func=cmd_run)

    p_slurm = sub.add_parser("slurm", parents=[common], help="Submit one Slurm job per cell.")
    p_slurm.add_argument("--dry-run", action="store_true", help="Generate sbatch scripts without submitting.")
    p_slurm.set_defaults(func=cmd_slurm)

    p_sum = sub.add_parser("summarize", help="Aggregate results into summary.md.")
    p_sum.add_argument("--results-root", default=_DEFAULT_RESULTS_ROOT)
    p_sum.set_defaults(func=cmd_summarize)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)

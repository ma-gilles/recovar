#!/usr/bin/env python
"""Run two recovar codebases on the same dataset for pipeline/compute_state comparison."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
from pathlib import Path


def infer_grid_size(dataset_dir: Path) -> int:
    patt = re.compile(r"^particles\.(\d+)\.mrcs$")
    for p in dataset_dir.iterdir():
        m = patt.match(p.name)
        if m:
            return int(m.group(1))
    raise ValueError("Could not infer grid size from particles.<grid>.mrcs; pass --grid-size.")


def build_particles_path(dataset_dir: Path, grid_size: int) -> Path:
    mrcs = dataset_dir / f"particles.{grid_size}.mrcs"
    star = dataset_dir / "particles.star"
    if mrcs.exists():
        return mrcs
    if star.exists():
        return star
    raise FileNotFoundError(f"Could not find {mrcs} or {star}")


def run_recovar_command(
    *,
    label: str,
    repo_root: Path,
    python_bin: str,
    subcommand: str,
    args: list[str],
) -> None:
    env = os.environ.copy()
    old_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) if not old_pp else f"{repo_root}:{old_pp}"
    cmd = [python_bin, "-m", "recovar.command_line", subcommand, *args]
    print(f"[{label}] {' '.join(shlex.quote(x) for x in cmd)}")
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare old/new pipeline (and optionally compute_state) on one fixed dataset."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Directory with particles/poses/ctf files.")
    parser.add_argument("--other-repo-root", type=Path, required=True, help="Second repo root to compare against current.")
    parser.add_argument(
        "--current-repo-root",
        type=Path,
        default=Path.cwd(),
        help="Current repo root (default: cwd).",
    )
    parser.add_argument("--output-base", type=Path, required=True, help="Where to write outputs.")
    parser.add_argument("--python-bin", default="python", help="Python interpreter to run commands.")
    parser.add_argument("--grid-size", type=int, default=None, help="Grid size, inferred if omitted.")
    parser.add_argument(
        "--pipeline-extra-args",
        default="--mask from_halfmaps --noise-model radial --correct-contrast",
        help="Extra args appended to pipeline call.",
    )
    parser.add_argument(
        "--run-compute-state",
        action="store_true",
        help="Also run compute_state from both repos using the same latent points.",
    )
    parser.add_argument(
        "--latent-points",
        type=Path,
        default=None,
        help="Latent points file for compute_state (required with --run-compute-state).",
    )
    parser.add_argument(
        "--compute-state-extra-args",
        default="--save-all-estimates",
        help="Extra args appended to compute_state call.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    current_repo = args.current_repo_root.resolve()
    other_repo = args.other_repo_root.resolve()
    output_base = args.output_base.resolve()
    output_base.mkdir(parents=True, exist_ok=True)

    if args.grid_size is None:
        grid_size = infer_grid_size(dataset_dir)
    else:
        grid_size = args.grid_size

    particles = build_particles_path(dataset_dir, grid_size)
    poses = dataset_dir / "poses.pkl"
    ctf = dataset_dir / "ctf.pkl"
    if not poses.exists() or not ctf.exists():
        raise FileNotFoundError(f"Missing poses/ctf in {dataset_dir}")

    cur_out = output_base / "current"
    oth_out = output_base / "other"
    cur_pipe = cur_out / "pipeline_output"
    oth_pipe = oth_out / "pipeline_output"
    cur_out.mkdir(parents=True, exist_ok=True)
    oth_out.mkdir(parents=True, exist_ok=True)

    base_pipe_args = [
        str(particles),
        "--poses",
        str(poses),
        "--ctf",
        str(ctf),
        "-o",
        "",  # placeholder
    ]
    pipe_extra = shlex.split(args.pipeline_extra_args)

    cur_args = base_pipe_args.copy()
    cur_args[6] = str(cur_pipe)
    cur_args.extend(pipe_extra)
    run_recovar_command(
        label="current:pipeline",
        repo_root=current_repo,
        python_bin=args.python_bin,
        subcommand="pipeline",
        args=cur_args,
    )

    oth_args = base_pipe_args.copy()
    oth_args[6] = str(oth_pipe)
    oth_args.extend(pipe_extra)
    run_recovar_command(
        label="other:pipeline",
        repo_root=other_repo,
        python_bin=args.python_bin,
        subcommand="pipeline",
        args=oth_args,
    )

    if args.run_compute_state:
        if args.latent_points is None:
            raise ValueError("--latent-points is required with --run-compute-state")
        latent_points = args.latent_points.resolve()
        if not latent_points.exists():
            raise FileNotFoundError(latent_points)
        cs_extra = shlex.split(args.compute_state_extra_args)

        cur_cs_out = cur_out / "state"
        oth_cs_out = oth_out / "state"
        cur_cs_args = [str(cur_pipe), "-o", str(cur_cs_out), "--latent-points", str(latent_points), *cs_extra]
        oth_cs_args = [str(oth_pipe), "-o", str(oth_cs_out), "--latent-points", str(latent_points), *cs_extra]

        run_recovar_command(
            label="current:compute_state",
            repo_root=current_repo,
            python_bin=args.python_bin,
            subcommand="compute_state",
            args=cur_cs_args,
        )
        run_recovar_command(
            label="other:compute_state",
            repo_root=other_repo,
            python_bin=args.python_bin,
            subcommand="compute_state",
            args=oth_cs_args,
        )

    print(f"Done. Output base: {output_base}")
    print(f"  current pipeline: {cur_pipe}")
    print(f"  other pipeline:   {oth_pipe}")


if __name__ == "__main__":
    main()

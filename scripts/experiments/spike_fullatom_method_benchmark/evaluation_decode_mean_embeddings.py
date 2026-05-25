#!/usr/bin/env python3
"""Decode method volumes at per-label mean embeddings when possible."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


DEFAULT_BENCH_ROOT = Path("/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_benchmark_100k_20260517")
SCRIPT_DIR = Path(__file__).resolve().parent


def _epoch_from_name(path: Path) -> int | None:
    match = re.search(r"epoch(\d+)", path.name)
    if match:
        return int(match.group(1))
    match = re.search(r"\.(\d+)\.", path.name)
    return int(match.group(1)) if match else None


def _find_mean_embedding_files(evaluation_root: Path, method: str) -> list[Path]:
    root = evaluation_root / method
    if not root.exists():
        return []
    return sorted(root.glob("*/mean_embeddings/labels_mean_z*.txt"))


def _matching_weights(bench_root: Path, run_name: str, epoch: int | None) -> Path | None:
    run_dir = bench_root / "cryodrgn" / run_name
    if epoch is not None:
        path = run_dir / f"weights.{epoch}.pkl"
        return path if path.exists() else None
    weights = sorted(run_dir.glob("weights.*.pkl"), key=lambda p: _epoch_from_name(p) or -1)
    return weights[-1] if weights else None


def _run_or_print(cmd: list[str], dry_run: bool, env: dict[str, str]) -> int:
    print("+ " + " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, check=False, env=env).returncode


def _decode_cryodrgn(args: argparse.Namespace) -> list[dict[str, object]]:
    outputs: list[dict[str, object]] = []
    files = _find_mean_embedding_files(args.evaluation_root, "cryodrgn")
    if not files:
        print(f"SKIP cryodrgn: no mean embedding txt files under {args.evaluation_root / 'cryodrgn'}")
        return outputs

    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    for zfile in files:
        run_name = zfile.parents[1].name
        epoch = _epoch_from_name(zfile)
        run_dir = args.bench_root / "cryodrgn" / run_name
        config = run_dir / "config.yaml"
        weights = _matching_weights(args.bench_root, run_name, epoch)
        if weights is None:
            print(f"SKIP cryodrgn/{run_name}: no matching weights for {zfile.name}")
            continue
        if not config.exists():
            print(f"SKIP cryodrgn/{run_name}: missing {config}")
            continue
        out_dir = args.evaluation_root / "cryodrgn" / run_name / "decoded_volumes" / zfile.stem
        done = out_dir / "vol_000.mrc"
        if done.exists() and not args.overwrite:
            print(f"SKIP cryodrgn/{run_name}: decoded volumes already exist in {out_dir}")
        else:
            cmd = [
                sys.executable,
                str(SCRIPT_DIR / "decoding_run_cryodrgn_eval_vol_direct.py"),
                str(weights),
                "-c",
                str(config),
                "-o",
                str(out_dir),
                "--zfile",
                str(zfile),
                "--prefix",
                "gt_label_",
                "--Apix",
                str(args.apix),
            ]
            if args.device is not None:
                cmd.extend(["--device", str(args.device)])
            if args.downsample is not None:
                cmd.extend(["--downsample", str(args.downsample)])
            rc = _run_or_print(cmd, args.dry_run, env)
            if args.dry_run:
                continue
            if rc != 0:
                print(f"ERROR cryodrgn/{run_name}: decoder exited with code {rc}")
                continue
            out_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "method": "cryodrgn",
            "run_name": run_name,
            "zfile": str(zfile),
            "weights": str(weights),
            "config": str(config),
            "decoded_dir": str(out_dir),
            "pattern": "gt_label_*.mrc",
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "decode_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        outputs.append(manifest)
    return outputs


def _report_not_ready(args: argparse.Namespace, method: str, reason: str) -> None:
    root = args.bench_root / method
    mean_files = _find_mean_embedding_files(args.evaluation_root, method)
    if mean_files:
        print(f"SKIP {method}: {reason}; mean files exist but decoder integration is not known yet.")
    else:
        print(f"SKIP {method}: {reason}; no mean embeddings found under {args.evaluation_root / method}.")
    if root.exists():
        print(f"  inspected method root: {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT)
    parser.add_argument("--evaluation-root", type=Path, default=None, help="Default: BENCH_ROOT/evaluation")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["cryodrgn", "cryosparc_3dflex", "dynamight"],
        choices=["cryodrgn", "cryosparc_3dflex", "dynamight"],
    )
    parser.add_argument("--apix", type=float, default=1.0)
    parser.add_argument("--downsample", type=int, default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print decoder commands without running them.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.evaluation_root = args.evaluation_root or (args.bench_root / "evaluation")

    manifests: list[dict[str, object]] = []
    if "cryodrgn" in args.methods:
        manifests.extend(_decode_cryodrgn(args))
    if "cryosparc_3dflex" in args.methods:
        _report_not_ready(args, "cryosparc_3dflex", "CryoSPARC 3DFlex volume-at-latent export is project/job specific")
    if "dynamight" in args.methods:
        _report_not_ready(args, "dynamight", "DynaMight latent-to-volume export is not exposed by the current artifacts")

    args.evaluation_root.mkdir(parents=True, exist_ok=True)
    summary = args.evaluation_root / "decode_summary.json"
    summary.write_text(json.dumps({"decoded": manifests}, indent=2) + "\n")
    print(f"WROTE {summary}")


if __name__ == "__main__":
    main()

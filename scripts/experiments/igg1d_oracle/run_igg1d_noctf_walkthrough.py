#!/usr/bin/env python3
"""Run the IgG-1D oracle walkthrough with simulator CTF fixed to identity."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from recovar.commands import spike_walkthrough
from recovar.core import ctf as core_ctf
from recovar.simulation import simulator


def _validate_noctf(run_dir: Path) -> None:
    ctf_path = run_dir / "03_dataset" / "ctf.pkl"
    sim_info_path = run_dir / "03_dataset" / "simulation_info.pkl"
    ctf_params = np.asarray(pickle.load(ctf_path.open("rb")), dtype=np.float32)
    with sim_info_path.open("rb") as handle:
        sim_info = pickle.load(handle)
    sim_ctf = np.asarray(sim_info["ctf_params"], dtype=np.float32)

    checks = {
        "ctf_pkl_dfu_zero": np.allclose(ctf_params[:, 2], 0.0),
        "ctf_pkl_dfv_zero": np.allclose(ctf_params[:, 3], 0.0),
        "ctf_pkl_dfang_zero": np.allclose(ctf_params[:, 4], 0.0),
        "ctf_pkl_voltage_300": np.allclose(ctf_params[:, 5], 300.0),
        "ctf_pkl_cs_zero": np.allclose(ctf_params[:, 6], 0.0),
        "ctf_pkl_w_minus_one": np.allclose(ctf_params[:, 7], -1.0),
        "ctf_pkl_phase_zero": np.allclose(ctf_params[:, 8], 0.0),
        "sim_contrast_one": np.allclose(sim_ctf[:, core_ctf.CTFParamIndex.CONTRAST], 1.0),
        "sim_bfactor_zero": np.allclose(sim_ctf[:, core_ctf.CTFParamIndex.BFACTOR], 0.0),
    }
    failed = [name for name, ok in checks.items() if not bool(ok)]
    if failed:
        raise RuntimeError(f"No-CTF validation failed: {failed}")

    summary = {
        "description": "No-CTF sanity check. Simulator used dataset_params_option='noctf', yielding CTF == 1.",
        "ctf_pkl": str(ctf_path),
        "simulation_info": str(sim_info_path),
        "n_images": int(ctf_params.shape[0]),
        "checks": checks,
        "ctf_pkl_columns": "[box, voxel_size, DFU, DFV, DFANG, VOLT, CS, W, PHASE_SHIFT]",
    }
    (run_dir / "03_dataset" / "noctf_validation.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = spike_walkthrough.add_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()

    original_generate = simulator.generate_synthetic_dataset

    def generate_noctf(*gen_args, **gen_kwargs):
        gen_kwargs["dataset_params_option"] = "noctf"
        return original_generate(*gen_args, **gen_kwargs)

    simulator.generate_synthetic_dataset = generate_noctf
    try:
        with spike_walkthrough.job_context(args, "igg1d_noctf_walkthrough") as ctx:
            spike_walkthrough.logging.basicConfig(
                format="%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s",
                level=spike_walkthrough.logging.INFO,
                handlers=[
                    spike_walkthrough.RobustFileHandler(str(Path(ctx.output_dir) / "run.log")),
                    spike_walkthrough.RobustStreamHandler(),
                ],
            )
            summary = spike_walkthrough.run_walkthrough(args, Path(ctx.output_dir))
            _validate_noctf(Path(ctx.output_dir))
            spike_walkthrough.logger.info("Finished no-CTF walkthrough. Outputs: %s", summary)
    finally:
        simulator.generate_synthetic_dataset = original_generate


if __name__ == "__main__":
    main()

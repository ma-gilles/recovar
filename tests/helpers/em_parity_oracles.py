"""Admit a K4 capture for the fast case's exact sampling configuration."""
import os
from pathlib import Path

import starfile


def k4_oracle(healpix_order, oversampling, *, environ=None):
    """Return an oracle and replay arguments after checking grid and capture identity.

    Set EM_PARITY_FAST_K4_H{order}_OS{oversampling}_{RELION_DIR,DISPATCH_SCHEDULE}.
    Legacy global variables are accepted only if their capture matches this case.
    Native binary/source admission remains part of the run's external manifest.
    """
    env = os.environ if environ is None else environ
    prefix = f"EM_PARITY_FAST_K4_H{healpix_order}_OS{oversampling}"
    values = [env.get(prefix + suffix) for suffix in ("_RELION_DIR", "_DISPATCH_SCHEDULE")]
    if not any(values):
        values = [env.get("EM_PARITY_FAST_K4" + suffix) for suffix in ("_RELION_DIR", "_DISPATCH_SCHEDULE")]
    if not all(values):
        raise ValueError(f"Missing matched K4 capture: set both {prefix}_RELION_DIR and {prefix}_DISPATCH_SCHEDULE")
    oracle, schedule_path = map(Path, values)
    sampling = starfile.read(oracle / "run_it001_sampling.star", always_dict=True)["sampling_general"]
    optimiser = starfile.read(oracle / "run_it001_optimiser.star", always_dict=True)["optimiser_general"]
    actual = (int(sampling["rlnHealpixOrder"]), int(optimiser["rlnAdaptiveOversampleOrder"]))
    if actual != (healpix_order, oversampling):
        raise ValueError(f"K4 capture grid {actual} differs from requested {(healpix_order, oversampling)}")
    from recovar.em.relion.relion_worker_scale import (
        load_relion_dispatch_schedule,
        verify_relion_dispatch_schedule_oracle,
    )

    schedule = load_relion_dispatch_schedule(schedule_path)
    verify_relion_dispatch_schedule_oracle(schedule, oracle)
    return oracle, ["--relion-dispatch-schedule", str(schedule_path)]

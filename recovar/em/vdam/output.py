"""InitialModel artifact paths, startup metadata, timing and output cadence."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from recovar.em.relion.initial_model_io import _write_data_star, _write_model_star
from recovar.em.vdam.state import InitialModelState, NativeParticleState
from recovar.utils.helpers import write_relion_mrc


def _initial_model_mrc_from_prefix(outputname: str) -> str:
    """Mirror RELION's GUI ``outputname.rstrip("run") + initial_model.mrc``."""

    return outputname.rstrip("run") + "initial_model.mrc"


def _class_mrc_paths(output_prefix: str, iteration: int, K: int) -> tuple[str, ...]:
    return tuple(f"{output_prefix}_it{iteration:03d}_class{k + 1:03d}.mrc" for k in range(K))


class _StageProfile:
    """Optional elapsed-stage report for InitialModel startup and artifact I/O."""

    def __init__(self):
        self.enabled = bool(os.environ.get("RECOVAR_INITIAL_MODEL_PROFILE"))
        self.started = self.stage_started = time.perf_counter()
        self.values = {}

    def record(self, name):
        if self.enabled:
            now = time.perf_counter()
            self.values[f"{name}_time_s"] = float(now - self.stage_started)
            self.stage_started = now

    def report(self, label):
        if self.enabled:
            self.values["total_time_s"] = float(time.perf_counter() - self.started)
            print(f"VDAM {label} profile: {json.dumps(self.values, sort_keys=True)}", flush=True)


def _write_initial_run_metadata(opts, continuation) -> None:
    """Write startup options and the optional native continuation provenance."""

    Path(opts.outputname).parent.mkdir(parents=True, exist_ok=True)
    config_path = f"{opts.outputname}_native_options.json"
    native_options = asdict(opts)
    native_options["resolved_cuda_allocator"] = os.environ.get(
        "TF_GPU_ALLOCATOR",
        "default",
    )
    native_options["jax_compilation_cache_enabled"] = bool(
        os.environ.get("JAX_COMPILATION_CACHE_DIR")
    )
    native_options["jax_compilation_cache_dir"] = os.environ.get(
        "JAX_COMPILATION_CACHE_DIR"
    )
    native_options["jax_persistent_cache_min_compile_time_secs"] = os.environ.get(
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"
    )
    with open(config_path, "w") as f:
        json.dump(native_options, f, indent=2, sort_keys=True)
    if continuation is not None:
        continuation_path = f"{opts.outputname}_diagnostic_continuation.json"
        with open(continuation_path, "w") as f:
            json.dump(
                {
                    "classification": "diagnostic_performance_only",
                    "exactly_one_next_iteration": True,
                    "iteration": int(continuation.iteration),
                    "optimiser_star": str(continuation.optimiser_star),
                    "model_star": str(continuation.model_star),
                    "data_star": str(continuation.data_star),
                    "sampling_star": str(continuation.sampling_star),
                },
                f,
                indent=2,
                sort_keys=True,
            )
            f.write("\n")


def _write_iteration_artifacts(
    output_prefix: str,
    state: InitialModelState,
    iteration: int,
    meta: dict,
    *,
    main_star=None,
    optics_star=None,
    dataset=None,
    particle_state: NativeParticleState | None = None,
) -> None:
    profile = _StageProfile()

    out_dir = Path(output_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    class_mrcs = _class_mrc_paths(output_prefix, iteration, int(state.K))
    profile.record("setup")
    for k, class_mrc in enumerate(class_mrcs):
        write_relion_mrc(class_mrc, np.asarray(state.Iref[k]), voxel_size=float(state.pixel_size))
    profile.record("class_mrc")
    model_star = f"{output_prefix}_it{iteration:03d}_model.star"
    _write_model_star(model_star, state, class_mrcs)
    profile.record("model_star")
    meta_path = f"{output_prefix}_it{iteration:03d}_recovar_meta.json"
    with open(meta_path, "w") as f:
        json.dump(_json_ready(meta), f, indent=2, sort_keys=True)
    profile.record("meta_json")
    if main_star is not None and dataset is not None and particle_state is not None:
        _write_data_star(
            f"{output_prefix}_it{iteration:03d}_data.star",
            main_star,
            optics_star,
            dataset,
            particle_state,
        )
    profile.record("data_star")
    profile.report(f"iteration {iteration} artifact")


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _write_final_outputs(output_prefix: str, state: InitialModelState) -> tuple[str, tuple[str, ...]]:
    iteration = int(state.iter)
    class_mrcs = _class_mrc_paths(output_prefix, iteration, int(state.K))
    out_dir = Path(output_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, class_mrc in enumerate(class_mrcs):
        if not os.path.exists(class_mrc):
            write_relion_mrc(class_mrc, np.asarray(state.Iref[k]), voxel_size=float(state.pixel_size))
    final_mrc = _initial_model_mrc_from_prefix(output_prefix)
    best_class = int(np.argmax(np.asarray(state.pdf_class)))
    write_relion_mrc(final_mrc, np.asarray(state.Iref[best_class]), voxel_size=float(state.pixel_size))
    return final_mrc, class_mrcs

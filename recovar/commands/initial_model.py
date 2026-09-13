"""RELION-equivalent RECOVAR InitialModel / VDAM command."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from recovar.em.vdam.schedules import GuiInitialModelDefaults

DEFAULTS = GuiInitialModelDefaults()

_CONCRETE_RECOVAR_PROVENANCE_MODULES = (
    "recovar.em.vdam.schedules",
    "recovar.em.vdam.driver",
    "recovar.em.vdam.iteration_loop",
    "recovar.em.vdam.dense_adapter",
)


def _assert_expected_repo_imports() -> dict[str, str]:
    """Fail fast when InitialModel resolves through another editable checkout."""
    expected_root_value = os.environ.get("RECOVAR_EXPECTED_REPO_ROOT")
    if not expected_root_value:
        return {}

    expected_root = Path(expected_root_value).expanduser().resolve()
    imported: dict[str, str] = {}
    failures: list[str] = []
    for module_name in _CONCRETE_RECOVAR_PROVENANCE_MODULES:
        module = importlib.import_module(module_name)
        module_file_value = getattr(module, "__file__", None)
        module_file = Path(module_file_value).resolve() if module_file_value else None
        imported[module_name] = str(module_file)
        print(f"InitialModel import provenance: {module_name}={module_file}", flush=True)
        if module_file is None or not module_file.is_relative_to(expected_root):
            failures.append(f"{module_name}={module_file}")
    if failures:
        raise RuntimeError(
            "RECOVAR InitialModel import provenance failure: expected every concrete module under "
            f"{expected_root}, found " + ", ".join(failures)
        )
    return imported


def initial_model_defaults_dict() -> dict[str, object]:
    """Return the public CLI/GUI defaults as a JSON-compatible mapping."""

    return asdict(DEFAULTS)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _disabled_or_pool_aligned_chunk_size(value: str) -> int:
    parsed = _nonnegative_int(value)
    if parsed not in (0,) and parsed < 3:
        raise argparse.ArgumentTypeError("must be 0 (disabled) or at least 3")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _open_unit_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1 (exclusive)")
    return parsed


def _closed_unit_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1 (inclusive)")
    return parsed


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run RECOVAR's RELION-equivalent InitialModel/VDAM refinement.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--i", dest="fn_img", required=True, help="Input particle STAR file")
    parser.add_argument("--o", dest="outputname", default="ab_initio/run", help="Output prefix")
    parser.add_argument("--nr-iter", "--nr_iter", dest="nr_iter", type=_positive_int, default=DEFAULTS.nr_iter)
    parser.add_argument(
        "--grad-write-iter",
        "--grad_write_iter",
        dest="grad_write_iter",
        type=_positive_int,
        default=DEFAULTS.grad_write_iter,
        help="Write trajectory artifacts every N iterations and at the final iteration",
    )
    parser.add_argument("--K", dest="nr_classes", type=_positive_int, default=DEFAULTS.nr_classes)
    parser.add_argument(
        "--tau2-fudge",
        "--tau2_fudge",
        dest="tau2_fudge",
        type=float,
        default=DEFAULTS.tau2_fudge,
    )
    parser.add_argument(
        "--grad-ini-frac",
        "--grad_ini_frac",
        dest="grad_ini_frac",
        type=_open_unit_float,
        default=DEFAULTS.grad_ini_frac,
        help="Fraction of iterations in the initial VDAM phase",
    )
    parser.add_argument(
        "--grad-fin-frac",
        "--grad_fin_frac",
        dest="grad_fin_frac",
        type=_open_unit_float,
        default=DEFAULTS.grad_fin_frac,
        help="Fraction of iterations in the final VDAM phase",
    )
    parser.add_argument(
        "--grad-em-iters",
        "--grad_em_iters",
        dest="grad_em_iters",
        type=_nonnegative_int,
        default=DEFAULTS.grad_em_iters,
        help="Number of terminal iterations using ordinary EM instead of VDAM",
    )
    parser.add_argument(
        "--stepsize",
        type=_positive_float,
        default=DEFAULTS.stepsize,
        help="Base VDAM gradient step size",
    )
    parser.add_argument(
        "--mu",
        type=_closed_unit_float,
        default=DEFAULTS.mu,
        help="VDAM momentum/forgetting factor",
    )
    parser.add_argument("--sym", dest="sym_name", default=DEFAULTS.sym_name)
    parser.add_argument(
        "--run-in-c1",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS.do_run_C1,
        help="Refine in C1 and apply the requested symmetry only to the final output",
    )
    parser.add_argument(
        "--particle-diameter",
        "--particle_diameter",
        dest="particle_diameter",
        type=float,
        default=DEFAULTS.particle_diameter,
        help="Particle diameter in Angstrom",
    )
    parser.add_argument(
        "--solvent",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS.do_solvent,
        help="Apply RELION solvent flattening",
    )
    parser.add_argument(
        "--zero-mask",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS.do_zero_mask,
        help="Zero pixels outside the solvent mask",
    )
    parser.add_argument(
        "--ctf",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS.do_ctf_correction,
        help="Enable CTF correction",
    )
    parser.add_argument("--random-seed", "--random_seed", dest="random_seed", type=int, default=DEFAULTS.random_seed)
    parser.add_argument(
        "--healpix-order",
        "--healpix_order",
        dest="healpix_order",
        type=_nonnegative_int,
        default=DEFAULTS.healpix_order,
    )
    parser.add_argument(
        "--oversampling",
        type=_nonnegative_int,
        default=DEFAULTS.oversampling,
        help="Adaptive angular/translation oversampling order",
    )
    parser.add_argument(
        "--offset-range",
        "--offset_range",
        dest="offset_range",
        type=float,
        default=DEFAULTS.offset_range_px,
        help="Translation search range in pixels",
    )
    parser.add_argument(
        "--offset-step",
        "--offset_step",
        dest="offset_step",
        type=float,
        default=DEFAULTS.offset_step_px,
        help="Translation search step in pixels",
    )
    parser.add_argument(
        "--perturbation-factor",
        "--perturbation_factor",
        dest="perturbation_factor",
        type=float,
        default=DEFAULTS.perturbation_factor,
    )
    parser.add_argument("--random-perturbation", "--random_perturbation", dest="random_perturbation", type=float)
    parser.add_argument(
        "--image-batch-size",
        "--image_batch_size",
        dest="image_batch_size",
        type=_positive_int,
        default=DEFAULTS.image_batch_size,
    )
    parser.add_argument(
        "--rotation-block-size",
        "--rotation_block_size",
        dest="rotation_block_size",
        type=_positive_int,
        default=DEFAULTS.rotation_block_size,
    )
    parser.add_argument(
        "--pass2-engine",
        "--pass2_engine",
        dest="pass2_engine",
        choices=("auto", "local", "compact"),
        default=DEFAULTS.pass2_engine,
        help=(
            "Adaptive pass-2 implementation: auto keeps exact local K=1 and "
            "uses joint compact class-by-pose scoring for K>1"
        ),
    )
    parser.add_argument(
        "--relion-wavg-sequential-cuda",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS.relion_wavg_sequential_cuda,
        help="Use the qualified shared CUDA kernel for RELION-ordered weighted sums",
    )
    parser.add_argument(
        "--exact-local-bucket-radix",
        type=int,
        choices=(2, 4),
        default=DEFAULTS.exact_local_bucket_radix,
        help="Static-shape radix for exact-local K=1 hypothesis buckets",
    )
    parser.add_argument(
        "--exact-local-physical-order-chunk-size",
        type=_disabled_or_pool_aligned_chunk_size,
        default=DEFAULTS.exact_local_physical_order_chunk_size,
        help=(
            "Bound consecutive physical-order K=1 buckets; 0 keeps the "
            "qualified run-global shape"
        ),
    )
    parser.add_argument(
        "--stable-fourier-window-shapes",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS.stable_fourier_window_shapes,
        help=(
            "Experimental K=1 exact-local VDAM policy: reuse low-cardinality "
            "physical Fourier capacities while retaining logical CUDA bounds"
        ),
    )
    parser.add_argument(
        "--bootstrap-min-particles",
        "--bootstrap_min_particles",
        dest="bootstrap_min_particles",
        type=_positive_int,
        default=DEFAULTS.bootstrap_min_particles,
    )
    parser.add_argument(
        "--sigma2-min-particles",
        "--sigma2_min_particles",
        dest="sigma2_min_particles",
        type=_positive_int,
        default=DEFAULTS.sigma2_min_particles,
    )
    parser.add_argument(
        "--translation-sigma-angstrom",
        "--translation_sigma_angstrom",
        dest="translation_sigma_angstrom",
        type=float,
    )
    parser.add_argument(
        "--padding-factor",
        "--padding_factor",
        dest="padding_factor",
        type=_positive_int,
        default=DEFAULTS.padding_factor,
    )
    parser.add_argument(
        "--image-fourier-backend",
        "--image_fourier_backend",
        dest="image_fourier_backend",
        choices=("auto", "host_numpy", "jax_gpu", "relion_cuda"),
        default=DEFAULTS.image_fourier_backend,
    )
    parser.add_argument(
        "--gpu", dest="gpu_ids", default=DEFAULTS.gpu_ids, help="GPU IDs for provenance and backend selection"
    )
    parser.add_argument("--j", dest="nr_threads", type=_positive_int, default=1, help="Compatibility thread count")
    parser.add_argument("--nr-mpi", "--nr_mpi", dest="nr_mpi", type=_positive_int, default=1)
    parser.add_argument("--scratch-dir", "--scratch_dir", dest="scratch_dir", default="")
    parser.add_argument("--datadir", help="Directory used to resolve relative STAR image paths")
    parser.add_argument("--strip-prefix", "--strip_prefix", dest="strip_prefix")
    parser.add_argument(
        "--lazy",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS.lazy,
        help="Load particle images lazily",
    )
    parser.add_argument(
        "--write-iter-artifacts",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS.write_iter_artifacts,
        help="Write RELION-compatible trajectory artifacts",
    )
    parser.add_argument(
        "--require-custom-cuda",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS.require_custom_cuda,
        help="Fail before execution unless the custom CUDA FFI path is ready",
    )
    cuda_mode = parser.add_mutually_exclusive_group()
    cuda_mode.add_argument(
        "--deterministic-cuda",
        action="store_true",
        default=DEFAULTS.deterministic_cuda,
        help="Serialize CUDA launches for diagnostics",
    )
    cuda_mode.add_argument("--allow-async-cuda", action="store_true", help="Explicitly select normal asynchronous CUDA")
    parser.add_argument(
        "--jax-compilation-cache",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS.use_jax_compilation_cache,
        help="Reuse shape-specialized JAX executables across InitialModel runs",
    )
    parser.add_argument(
        "--jax-compilation-cache-dir",
        default=DEFAULTS.jax_compilation_cache_dir,
        help=(
            "Shared JAX compilation-cache directory. Empty uses "
            "$JAX_COMPILATION_CACHE_DIR or the platform user cache."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved native options without running")
    parser.add_argument(
        "--projector-setup-backend", choices=("native", "jax"), default="native",
        help="Reference projector preparation backend.",
    )
    parser.add_argument(
        "--mstep-backend", choices=("native", "jax"), default="native",
        help="VDAM M-step transaction backend.",
    )
    parser.add_argument(
        "--mstep-compute-dtype", choices=("float32", "float64"), default="float64",
        help="M-owned state precision; float32 requires the JAX M-step.",
    )
    parser.add_argument(
        "--diagnostic-continue-optimiser",
        help="Native RELION VDAM checkpoint for exactly one next diagnostic iteration.",
    )
    parser.add_argument(
        "--diagnostic-stop-after-iteration", type=_positive_int,
        help="Diagnostic stopping iteration; nr_iter still controls the full schedule.",
    )
    return parser


def _configure_cuda_launch_blocking(*, deterministic_cuda: bool) -> str:
    value = "1" if bool(deterministic_cuda) else "0"
    os.environ["CUDA_LAUNCH_BLOCKING"] = value
    return value


def _configure_jax_compilation_cache(*, enabled: bool, requested_dir: str) -> dict[str, object]:
    """Resolve and activate the persistent InitialModel compilation cache."""

    if not enabled:
        os.environ.pop("JAX_COMPILATION_CACHE_DIR", None)
        import jax

        jax.config.update("jax_compilation_cache_dir", None)
        return {"enabled": False, "directory": None, "source": "disabled"}

    requested_dir = str(requested_dir).strip()
    environment_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR", "").strip()
    if requested_dir:
        directory = Path(requested_dir).expanduser().resolve()
        source = "command_line"
    elif environment_dir:
        directory = Path(environment_dir).expanduser().resolve()
        source = "environment"
    else:
        cache_home = os.environ.get("XDG_CACHE_HOME", "").strip()
        base = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
        directory = (base / "recovar" / "jax" / "initial_model").resolve()
        source = "automatic_user_cache"

    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(directory)
    min_compile_time_secs = os.environ.setdefault(
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
        "0",
    )
    # Importing ``recovar.em.vdam.schedules`` executes the package
    # initializer, which may import JAX before CLI parsing reaches this point.
    # Update the live config as well as the environment so the cache is active
    # for this process, not merely its children.
    import jax

    jax.config.update("jax_compilation_cache_dir", str(directory))
    jax.config.update(
        "jax_persistent_cache_min_compile_time_secs",
        float(min_compile_time_secs),
    )
    return {"enabled": True, "directory": str(directory), "source": source}


def _require_custom_cuda_runtime() -> dict[str, object]:
    import jax

    import recovar.cuda_backproject as cuda_backproject
    from recovar.core import slicing

    slicing._on_gpu.cache_clear()
    report = {
        "default_backend": jax.default_backend(),
        "device_platforms": [getattr(device, "platform", "") for device in jax.devices()],
        "slicing_on_gpu": bool(slicing._on_gpu()),
        "custom_cuda_requested": bool(cuda_backproject.custom_cuda_requested()),
        "cuda_available": bool(cuda_backproject.cuda_available()),
    }
    print("RECOVAR InitialModel CUDA runtime gate: " + json.dumps(report, sort_keys=True), flush=True)
    if not report["slicing_on_gpu"]:
        raise RuntimeError(f"InitialModel requires a visible GPU for this run: {report}")
    if not report["custom_cuda_requested"] or not report["cuda_available"]:
        raise cuda_backproject.cuda_unavailable_error()
    return report


def _native_options_dict(args: argparse.Namespace) -> dict[str, object]:
    backend = args.image_fourier_backend
    if backend == "auto":
        backend = "relion_cuda" if args.gpu_ids else "host_numpy"
    return {
        "projector_setup_backend": args.projector_setup_backend,
        "mstep_backend": args.mstep_backend,
        "mstep_compute_dtype": args.mstep_compute_dtype,
        "diagnostic_continue_optimiser": args.diagnostic_continue_optimiser,
        "diagnostic_stop_after_iteration": args.diagnostic_stop_after_iteration,
        "fn_img": args.fn_img,
        "outputname": args.outputname,
        "nr_iter": args.nr_iter,
        "nr_classes": args.nr_classes,
        "tau2_fudge": args.tau2_fudge,
        "grad_ini_frac": args.grad_ini_frac,
        "grad_fin_frac": args.grad_fin_frac,
        "grad_em_iters": args.grad_em_iters,
        "stepsize": args.stepsize,
        "mu": args.mu,
        "sym_name": args.sym_name,
        "do_run_C1": args.run_in_c1,
        "particle_diameter": args.particle_diameter,
        "do_solvent": args.solvent,
        "do_zero_mask": args.zero_mask,
        "do_ctf_correction": args.ctf,
        "random_seed": args.random_seed,
        "healpix_order": args.healpix_order,
        "oversampling": args.oversampling,
        "offset_range_px": args.offset_range,
        "offset_step_px": args.offset_step,
        "perturbation_factor": args.perturbation_factor,
        "random_perturbation": args.random_perturbation,
        "image_batch_size": args.image_batch_size,
        "rotation_block_size": args.rotation_block_size,
        "pass2_engine": args.pass2_engine,
        "relion_wavg_sequential_cuda": args.relion_wavg_sequential_cuda,
        "exact_local_bucket_radix": args.exact_local_bucket_radix,
        "exact_local_physical_order_chunk_size": (
            args.exact_local_physical_order_chunk_size
        ),
        "stable_fourier_window_shapes": args.stable_fourier_window_shapes,
        "bootstrap_min_particles": args.bootstrap_min_particles,
        "sigma2_min_particles": args.sigma2_min_particles,
        "padding_factor": args.padding_factor,
        "image_fourier_backend": backend,
        "deterministic_cuda": args.deterministic_cuda,
        "lazy": args.lazy,
        "datadir": args.datadir,
        "strip_prefix": args.strip_prefix,
        "translation_sigma_angstrom": args.translation_sigma_angstrom,
        "write_iter_artifacts": args.write_iter_artifacts,
        "grad_write_iter": args.grad_write_iter,
    }


def main(argv: Sequence[str] | None = None) -> int:
    _assert_expected_repo_imports()
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.mstep_compute_dtype == "float32" and args.mstep_backend != "jax":
        parser.error("--mstep-compute-dtype float32 requires --mstep-backend jax")
    if args.nr_mpi > 1:
        raise SystemExit("ERROR: Gradient refinement is not supported together with MPI.")
    if args.gpu_ids:
        _configure_cuda_launch_blocking(deterministic_cuda=args.deterministic_cuda)
    cache_report = _configure_jax_compilation_cache(
        enabled=args.jax_compilation_cache,
        requested_dir=args.jax_compilation_cache_dir,
    )
    options_dict = _native_options_dict(args)
    if args.dry_run:
        dry_run_payload = dict(options_dict)
        dry_run_payload["resolved_cuda_allocator"] = os.environ.get(
            "TF_GPU_ALLOCATOR",
            "default",
        )
        dry_run_payload["jax_compilation_cache"] = cache_report
        print(json.dumps(dry_run_payload, indent=2, sort_keys=True))
        return 0
    if args.require_custom_cuda:
        _require_custom_cuda_runtime()

    from recovar.em.vdam.driver import NativeInitialModelOptions, run_native_initial_model

    result = run_native_initial_model(NativeInitialModelOptions(**options_dict))
    print(f"recovar InitialModel complete: {result.final_mrc}")
    print(f"Final model STAR: {result.final_model_star}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

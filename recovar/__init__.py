"""RECOVAR: heterogeneity analysis for cryo-EM and cryo-ET."""

import os
import sys


def _initial_model_cli_requested(argv=None, orig_argv=None) -> bool:
    """Return whether this interpreter was launched for InitialModel."""

    argv = tuple(sys.argv if argv is None else argv)
    orig_argv = tuple(getattr(sys, "orig_argv", ()) if orig_argv is None else orig_argv)
    if argv and os.path.basename(argv[0]) == "run_ab_initio.py":
        return True
    if len(argv) > 1 and argv[1] == "initial_model":
        return True
    return any(
        orig_argv[index] == "-m" and orig_argv[index + 1] == "recovar.commands.initial_model"
        for index in range(len(orig_argv) - 1)
    )


def _configure_initial_model_cuda_allocator(*, argv=None, orig_argv=None, environ=None):
    """Select the qualified InitialModel allocator before JAX initializes."""

    environ = os.environ if environ is None else environ
    if not _initial_model_cli_requested(argv=argv, orig_argv=orig_argv):
        return environ.get("TF_GPU_ALLOCATOR")
    requested = environ.get(
        "RECOVAR_INITIAL_MODEL_CUDA_ALLOCATOR",
        "cuda_malloc_async",
    ).strip()
    if requested.lower() not in {"", "default", "none", "off"}:
        environ.setdefault("TF_GPU_ALLOCATOR", requested)
    return environ.get("TF_GPU_ALLOCATOR")


_configure_initial_model_cuda_allocator()

try:
    import recovar.jax_config  # noqa: F401
except ModuleNotFoundError:
    # Allow importing lightweight modules (e.g., utils and FFT helpers)
    # in environments where optional heavy dependencies are absent.
    pass

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "1.0.0b1"

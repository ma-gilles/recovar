#!/usr/bin/env python
"""DEBUG-ONLY: Run recovar refinement with CTF disabled (set to 1.0).

This script monkey-patches recovar.core.ctf._compute_spa_ctf to return
all-ones, so the EM loop reconstructs CTF*proj from CTF-modulated data
without ever deconvolving the CTF. This is what RELION does when
--ctf is NOT passed to relion_refine_mpi.

Purpose: Verify that the dark halo / poor-resolution artifact we see in
RELION reference runs is due to missing --ctf, by reproducing the same
artifact in recovar.

DO NOT USE FOR PRODUCTION RECONSTRUCTIONS.

Usage:
    pixi run python scripts/debug_recovar_no_ctf.py \\
        --data_dir <path> --output <path> [forward all run_full_refinement args]
"""

import sys

import jax.numpy as jnp

# ----------------------------------------------------------------------
# MUST happen BEFORE any recovar.em / engine import: otherwise the JIT
# trace captures the original CTF function. _preprocess_batch is jitted,
# but the call to ``_compute_spa_ctf`` is resolved at trace time via the
# module global, so patching before the first trace is sufficient.
# ----------------------------------------------------------------------
import recovar.core.ctf as _recovar_ctf  # noqa: E402


def _ones_ctf(CTF_params, image_shape, voxel_size, *, half_image=False):
    """Drop-in replacement for _compute_spa_ctf returning all 1.0."""
    n_images = CTF_params.shape[0]
    if half_image:
        n_pixels = image_shape[0] * (image_shape[1] // 2 + 1)
    else:
        n_pixels = image_shape[0] * image_shape[1]
    return jnp.ones((n_images, n_pixels), dtype=jnp.float32)


_recovar_ctf._compute_spa_ctf = _ones_ctf
_recovar_ctf._compute_spa_ctf_antialiased = _ones_ctf

print(
    "[DEBUG NO-CTF] Patched recovar.core.ctf._compute_spa_ctf "
    "and _compute_spa_ctf_antialiased to return all-ones.",
    file=sys.stderr,
    flush=True,
)

# Now hand off to the normal refinement entry point. Add scripts/ to sys.path
# so we can import run_full_refinement as a sibling module.
import os  # noqa: E402

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from run_full_refinement import main  # noqa: E402

main()

"""Fixture-wiring parity test: iter-0 sigma2_noise vs run_it000_model.star.

Reads the 500-particle RELION InitialModel fixture at
  /scratch/gpfs/GILLES/mg6942/tmp/relion_initialmodel_64_20260420_121428_8956_run/
runs recovar's `compute_avg_unaligned_and_sigma2` on the matched stack,
and asserts the resulting per-shell spectrum matches RELION's
`run_it000_model.star` iter-0 sigma2_noise (group 1) to within a tight
absolute tolerance.

The handoff note claims this was achieved to max abs diff ~3.4e-7. We
pin that here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import pytest

from recovar.em.initial_model.avg_unaligned import compute_avg_unaligned_and_sigma2

FIXTURE_DIR = Path("/scratch/gpfs/GILLES/mg6942/tmp/relion_initialmodel_64_20260420_121428_8956_run")
PARTICLES_STAR = Path(
    "/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar/.tmp/"
    "slurm_7178672/pytest-of-mg6942/pytest-0/test_pipeline_spa_gpu0/"
    "gpu_spa/test_dataset/particles.star"
)
PARTICLES_MRCS = PARTICLES_STAR.with_name("particles.64.mrcs")


pytestmark = pytest.mark.unit


requires_fixture = pytest.mark.skipif(
    not FIXTURE_DIR.exists() or not PARTICLES_MRCS.exists(),
    reason="RELION InitialModel fixture not available on this host",
)


def _read_relion_iter0_sigma2(n_shells: int) -> np.ndarray:
    """Parse the `data_model_optics_group_1` sigma2 loop from the fixture."""
    import re

    txt = (FIXTURE_DIR / "run_it000_model.star").read_text()
    block_match = re.search(r"data_model_optics_group_1\n(.*?)(?:\ndata_)", txt, re.DOTALL)
    if not block_match:
        raise RuntimeError("could not find data_model_optics_group_1 in star file")
    block = block_match.group(1)

    values = np.zeros(n_shells, dtype=np.float64)
    for line in block.strip().split("\n"):
        toks = line.split()
        if len(toks) != 3:
            continue
        try:
            idx = int(toks[0])
            val = float(toks[2])
        except ValueError:
            continue
        if 0 <= idx < n_shells:
            values[idx] = val
    return values


def _load_particle_stack_images(mrcs_path: Path, ori_size: int, n_max: int) -> np.ndarray:
    """Load the first `n_max` 2D particle images from the stack.

    Returns an array of shape (N, ori_size, ori_size) with origin at the
    centre pixel (RELION's `setXmippOrigin()` convention).
    """
    import mrcfile

    with mrcfile.open(mrcs_path, permissive=True) as m:
        stack = np.asarray(m.data)
    if stack.ndim != 3:
        raise RuntimeError(f"expected 3D stack, got {stack.shape}")
    if stack.shape[1:] != (ori_size, ori_size):
        raise RuntimeError(f"stack frames {stack.shape[1:]} != expected {(ori_size, ori_size)}")
    return np.asarray(stack[:n_max], dtype=np.float64)


def _image_iter(images: np.ndarray) -> Iterator[Tuple[int, np.ndarray]]:
    for img in images:
        yield 0, img


@requires_fixture
class TestSigma2NoiseIter0Parity:
    ORI_SIZE = 64
    PIXEL_SIZE = 8.5
    PARTICLE_DIAMETER_ANG = 544.0  # from fixture's run_it000_optimiser.star
    WIDTH_MASK_EDGE = 5
    DO_ZERO_MASK = True
    N_PARTICLES = 500  # all particles in the fixture

    def test_sigma2_noise_matches_relion_iter0(self):
        n_shells = self.ORI_SIZE // 2 + 1

        # 1. Load particles from the matched stack
        images = _load_particle_stack_images(PARTICLES_MRCS, self.ORI_SIZE, self.N_PARTICLES)

        # 2. Run recovar's Mavg + sigma2 computation
        Mavg, sigma2_per_group = compute_avg_unaligned_and_sigma2(
            _image_iter(images),
            ori_size=self.ORI_SIZE,
            pixel_size=self.PIXEL_SIZE,
            particle_diameter_ang=self.PARTICLE_DIAMETER_ANG,
            width_mask_edge_px=self.WIDTH_MASK_EDGE,
            do_zero_mask=self.DO_ZERO_MASK,
            nr_optics_groups=1,
            minimum_nr_particles=1000,
        )
        assert Mavg.shape == (self.ORI_SIZE, self.ORI_SIZE)
        assert sigma2_per_group.shape == (1, n_shells)

        recovar_sigma2 = sigma2_per_group[0]

        # 3. Parse RELION's iter 0 sigma2_noise from model.star
        relion_sigma2 = _read_relion_iter0_sigma2(n_shells)

        # 4. Compare
        max_abs = float(np.max(np.abs(recovar_sigma2 - relion_sigma2)))
        max_rel = float(np.max(np.abs(recovar_sigma2 - relion_sigma2) / np.maximum(np.abs(relion_sigma2), 1e-30)))
        # Log for debug
        import logging

        logging.info(
            "sigma2 parity: max_abs=%.3e max_rel=%.3e recovar[0]=%.6e relion[0]=%.6e",
            max_abs,
            max_rel,
            recovar_sigma2[0],
            relion_sigma2[0],
        )
        # The handoff target was max_abs ~3.4e-7. Pin tightly so any
        # regression in the mask / FFT pair surfaces immediately.
        assert max_abs < 1e-6, (
            f"sigma2 parity broke: max_abs={max_abs:.3e}\n"
            f"  recovar head: {recovar_sigma2[:5]}\n"
            f"  relion head:  {relion_sigma2[:5]}"
        )

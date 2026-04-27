"""Phase 5 (S2): Compare RELION's translation grid against recovar.

RELION: Cartesian grid in Angstroms, circular mask (x²+y² ≤ range² + 0.001).
        Oversampled: sub-divide each coarse cell into 2^os × 2^os sub-points.
recovar: Cartesian grid in pixels, circular mask (norm ≤ max_pixel + 0.001).

The grids should match exactly when accounting for the Angstrom/pixel conversion.

Tests:
1. Coarse grid: same points, same ordering
2. Grid shape: circular mask boundary
3. Oversampled sub-grid placement
4. Perturbation offset
"""

import numpy as np
import pytest
from recovar.relion_bind._relion_bind_core import (
    get_coarse_translations,
    get_oversampled_translations,
)


class TestCoarseTranslations:
    """Compare RELION and recovar coarse translation grids."""

    @pytest.mark.parametrize(
        "offset_range,offset_step",
        [
            (3.0, 1.0),
            (5.0, 1.0),
            (10.0, 2.0),
            (6.0, 1.5),
        ],
    )
    def test_grid_matches_recovar(self, offset_range, offset_step):
        from recovar.em.sampling import get_translation_grid

        relion_grid = get_coarse_translations(offset_range, offset_step)
        recovar_grid = get_translation_grid(offset_range, offset_step)

        assert relion_grid.shape[0] == recovar_grid.shape[0], (
            f"Count mismatch: RELION={relion_grid.shape[0]} vs recovar={recovar_grid.shape[0]}"
        )

        max_diff = np.max(np.abs(relion_grid - recovar_grid))
        print(f"\nrange={offset_range}, step={offset_step}: n={relion_grid.shape[0]}, max diff={max_diff:.2e}")
        assert max_diff < 1e-12, f"Grid mismatch: max diff = {max_diff:.2e}"

    def test_circular_boundary(self):
        """Verify both use circular (not square) boundary."""
        relion_grid = get_coarse_translations(5.0, 1.0)
        norms = np.linalg.norm(relion_grid, axis=1)
        assert np.all(norms <= 5.001), "Points outside circular boundary"
        assert np.any(norms > 4.5), "No points near boundary — too conservative"


class TestOversampledTranslations:
    """Compare oversampled translation sub-grids."""

    @pytest.mark.parametrize("oversampling_order", [0, 1, 2])
    def test_oversampled_count(self, oversampling_order):
        """Correct number of sub-points per coarse cell."""
        relion_ov = get_oversampled_translations(5.0, 1.0, 0, oversampling_order, 1.0)
        expected = max(1, 4**oversampling_order)
        if oversampling_order == 0:
            expected = 1
        assert relion_ov.shape[0] == expected, f"OS={oversampling_order}: got {relion_ov.shape[0]}, expected {expected}"

    def test_os0_returns_coarse_point(self):
        """With oversampling=0, returns the coarse grid point in pixels."""
        offset_range, offset_step, pixel_size = 5.0, 1.0, 1.5
        coarse = get_coarse_translations(offset_range, offset_step)

        for itrans in [0, 5, coarse.shape[0] - 1]:
            ov = get_oversampled_translations(offset_range, offset_step, itrans, 0, pixel_size)
            expected = coarse[itrans] / pixel_size
            diff = np.max(np.abs(ov[0] - expected))
            assert diff < 1e-12, f"OS=0 mismatch at itrans={itrans}: diff={diff:.2e}"

    def test_oversampled_centered_in_coarse_cell(self):
        """Oversampled points center around the coarse grid point."""
        offset_range, offset_step, pixel_size = 5.0, 1.0, 1.0
        coarse = get_coarse_translations(offset_range, offset_step)

        itrans = 12
        ov = get_oversampled_translations(offset_range, offset_step, itrans, 1, pixel_size)
        center = np.mean(ov, axis=0)
        expected_center = coarse[itrans] / pixel_size
        diff = np.max(np.abs(center - expected_center))
        assert diff < 1e-12, f"Sub-grid center offset: {diff:.2e}"

    def test_oversampled_spacing(self):
        """Sub-grid spacing is offset_step / (2^os * pixel_size)."""
        offset_step, pixel_size = 2.0, 1.5
        ov = get_oversampled_translations(10.0, offset_step, 0, 1, pixel_size)
        assert ov.shape[0] == 4

        xs = np.sort(np.unique(np.round(ov[:, 0], 10)))
        ys = np.sort(np.unique(np.round(ov[:, 1], 10)))
        expected_spacing = offset_step / (2 * pixel_size)
        actual_x = xs[1] - xs[0] if len(xs) > 1 else 0
        actual_y = ys[1] - ys[0] if len(ys) > 1 else 0
        assert abs(actual_x - expected_spacing) < 1e-10
        assert abs(actual_y - expected_spacing) < 1e-10


class TestTranslationPerturbation:
    """Verify translation perturbation matches RELION's formula."""

    def test_perturbation_offset(self):
        """Perturbation adds random_perturbation * offset_step / pixel_size to both x,y."""
        offset_range, offset_step, pixel_size = 5.0, 1.0, 1.5
        random_pert = 0.3

        ov_no_pert = get_oversampled_translations(offset_range, offset_step, 5, 0, pixel_size, 0.0)
        ov_pert = get_oversampled_translations(offset_range, offset_step, 5, 0, pixel_size, random_pert)

        expected_shift = random_pert * offset_step / pixel_size
        actual_shift_x = ov_pert[0, 0] - ov_no_pert[0, 0]
        actual_shift_y = ov_pert[0, 1] - ov_no_pert[0, 1]

        assert abs(actual_shift_x - expected_shift) < 1e-12
        assert abs(actual_shift_y - expected_shift) < 1e-12

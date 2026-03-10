"""Tests for default max_r sphere clipping and half/full equivalence.

Verifies:
  1. Default max_r = image_shape[0]//2 - 1 is applied when not specified.
  2. Explicit max_r=None disables clipping (backward compat).
  3. Half-vol/half-img vs full-vol/full-img produce equivalent results
     under the default max_r (ensures Nyquist-edge exclusion makes them match).
  4. Upsampling (volume_shape != image_shape) works with max_r.
  5. Project ⟷ backproject adjointness holds with default max_r.
"""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

import recovar.core as core
import recovar.core.slicing as slicing
import recovar.core.fourier_transform_utils as ftu

pytestmark = pytest.mark.unit


# ── Helpers ──────────────────────────────────────────────────────────

def _random_rotations(n, seed=42):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 3, 3))
    q, r = np.linalg.qr(z)
    d = np.sign(np.diagonal(r, axis1=1, axis2=2))
    q = q * d[:, None, :]
    det = np.linalg.det(q)
    q[det < 0] *= -1
    return jnp.array(q, dtype=jnp.float32)


def _hermitian_volume(N, seed=0):
    """Random Hermitian-symmetric full volume (FFT of real data)."""
    rng = np.random.default_rng(seed)
    real_vol = rng.standard_normal((N, N, N)).astype(np.float32)
    ft = np.fft.fftn(real_vol)
    return jnp.array(ft.ravel(), dtype=jnp.complex64)


def _hermitian_volume_rect(volume_shape, seed=0):
    """Random Hermitian-symmetric volume for non-cubic shapes."""
    rng = np.random.default_rng(seed)
    real_vol = rng.standard_normal(volume_shape).astype(np.float32)
    ft = np.fft.fftn(real_vol)
    return jnp.array(ft.ravel(), dtype=jnp.complex64)


def _random_slices(n_images, n_pix, seed=99):
    rng = np.random.default_rng(seed)
    return jnp.array(
        (rng.standard_normal((n_images, n_pix)) +
         1j * rng.standard_normal((n_images, n_pix))).astype(np.complex64)
    )


# ── 1. Default max_r value ──────────────────────────────────────────

class TestDefaultMaxR:
    """Verify default max_r = image_shape[0]//2 - 1."""

    def test_default_max_r_value(self):
        assert slicing._default_max_r((128, 128)) == 63
        assert slicing._default_max_r((64, 64)) == 31
        assert slicing._default_max_r((32, 32)) == 15
        assert slicing._default_max_r((16, 16)) == 7
        assert slicing._default_max_r((8, 8)) == 3

    def test_default_max_r_upsampled(self):
        """With upsampling, image_shape is smaller, max_r based on image."""
        assert slicing._default_max_r((64, 64)) == 31  # ups=2 with 128^3 vol
        assert slicing._default_max_r((32, 32)) == 15  # ups=4 with 128^3 vol

    def test_resolve_auto(self):
        assert slicing._resolve_max_r(slicing._AUTO, (128, 128)) == 63

    def test_resolve_none(self):
        assert slicing._resolve_max_r(None, (128, 128)) is None

    def test_resolve_explicit(self):
        assert slicing._resolve_max_r(5.0, (128, 128)) == 5.0

    def test_default_clips_more_than_none(self):
        """Default max_r should produce more zeros than max_r=None."""
        N = 16
        image_shape = (N, N)
        volume_shape = (N, N, N)
        vol = _hermitian_volume(N)
        rots = _random_rotations(3)

        # Default (auto) — clips at N//2-1 = 7
        out_default = np.asarray(slicing.slice_volume(
            vol, rots, image_shape, volume_shape, "linear_interp"))
        # Explicit None — no clipping
        out_none = np.asarray(slicing.slice_volume(
            vol, rots, image_shape, volume_shape, "linear_interp", max_r=None))

        n_zero_default = np.sum(np.abs(out_default) < 1e-30)
        n_zero_none = np.sum(np.abs(out_none) < 1e-30)
        assert n_zero_default > n_zero_none, (
            f"Default should clip more: {n_zero_default} zeros vs {n_zero_none}"
        )

    def test_explicit_none_matches_old_behavior(self):
        """max_r=None should give same result as the old behavior (no clipping)."""
        N = 16
        image_shape = (N, N)
        volume_shape = (N, N, N)
        vol = _hermitian_volume(N)
        rots = _random_rotations(3)

        # max_r=None should produce no zeros from clipping (only from interpolation)
        out = np.asarray(slicing.slice_volume(
            vol, rots, image_shape, volume_shape, "linear_interp", max_r=None))
        # With a random volume, very few pixels should be exactly zero
        n_nonzero = np.sum(np.abs(out) > 1e-30)
        total = out.size
        assert n_nonzero > 0.8 * total, (
            f"max_r=None should leave most pixels nonzero: {n_nonzero}/{total}"
        )


# ── 2. Half-vol/half-img equivalence under default max_r ────────────

class TestHalfFullEquivalence:
    """All 4 combos of half_volume/half_image should produce equivalent results.

    The default max_r = N//2-1 excludes the Nyquist edge, which is the
    problematic bin for half/full equivalence.
    """

    @pytest.fixture(params=[16, 32], ids=["N16", "N32"])
    def N(self, request):
        return request.param

    def test_project_half_img_matches_full_img(self, N):
        """full-vol+full-img vs full-vol+half-img should be equivalent."""
        image_shape = (N, N)
        volume_shape = (N, N, N)
        vol = _hermitian_volume(N)
        rots = _random_rotations(5, seed=10)

        out_full = np.asarray(slicing.slice_volume(
            vol, rots, image_shape, volume_shape, "linear_interp",
            half_image=False))
        out_half = np.asarray(slicing.slice_volume(
            vol, rots, image_shape, volume_shape, "linear_interp",
            half_image=True))

        # Convert full image to half using proper ftu conversion
        half_from_full = np.asarray(
            ftu.full_image_to_half_image(jnp.array(out_full), image_shape))

        np.testing.assert_allclose(
            out_half, half_from_full, atol=1e-5, rtol=1e-5,
            err_msg=f"N={N}: half_img should match corresponding full_img pixels"
        )

    def test_project_half_vol_matches_full_vol(self, N):
        """half-vol+half-img vs full-vol+half-img."""
        image_shape = (N, N)
        volume_shape = (N, N, N)
        vol_full = _hermitian_volume(N)
        vol_half = ftu.full_volume_to_half_volume(vol_full, volume_shape)
        rots = _random_rotations(5, seed=20)

        out_full = np.asarray(slicing.slice_volume(
            vol_full, rots, image_shape, volume_shape, "linear_interp",
            half_image=True))
        out_half = np.asarray(slicing.slice_volume(
            vol_half, rots, image_shape, volume_shape, "linear_interp",
            half_volume=True, half_image=True))

        np.testing.assert_allclose(
            out_half, out_full, atol=1e-5, rtol=1e-5,
            err_msg=f"N={N}: half_vol should match full_vol projection"
        )

    def test_project_all_four_combos(self, N):
        """All 4 half_vol/half_img combos agree on the half-image pixels."""
        image_shape = (N, N)
        volume_shape = (N, N, N)
        vol_full = _hermitian_volume(N, seed=30)
        vol_half = ftu.full_volume_to_half_volume(vol_full, volume_shape)
        rots = _random_rotations(5, seed=30)

        results = {}
        for hv, hi in [(False, False), (False, True), (True, False), (True, True)]:
            vol = vol_half if hv else vol_full
            out = np.asarray(slicing.slice_volume(
                vol, rots, image_shape, volume_shape, "linear_interp",
                half_volume=hv, half_image=hi))
            if not hi:
                # Convert full image to half using proper ftu conversion
                out = np.asarray(
                    ftu.full_image_to_half_image(jnp.array(out), image_shape))
            results[(hv, hi)] = out

        ref = results[(False, True)]
        for key, val in results.items():
            np.testing.assert_allclose(
                val, ref, atol=1e-4, rtol=1e-4,
                err_msg=f"N={N}: combo {key} differs from (False,True) reference"
            )

    def test_backproject_half_img_matches_full_img(self, N):
        """Backproject with full-img vs half-img should give same volume.

        Half-image backproject uses Hermitian scatter (CONJ_MODE), so the
        input images must be Hermitian symmetric (DFT of real data) for
        the two paths to agree.
        """
        image_shape = (N, N)
        volume_shape = (N, N, N)
        rots = _random_rotations(5, seed=40)

        # Hermitian images: DFT of real data
        rng = np.random.default_rng(40)
        real_imgs = rng.standard_normal((5, N, N)).astype(np.float32)
        full_imgs = jnp.array(ftu.get_dft2(jnp.array(real_imgs))).reshape(5, -1)
        half_imgs = jnp.array(ftu.get_dft2_real(jnp.array(real_imgs))).reshape(5, -1)

        out_full = np.asarray(slicing.adjoint_slice_volume(
            full_imgs, rots, image_shape, volume_shape, "linear_interp",
            half_image=False))
        out_half = np.asarray(slicing.adjoint_slice_volume(
            half_imgs, rots, image_shape, volume_shape, "linear_interp",
            half_image=True))

        np.testing.assert_allclose(
            out_half, out_full, atol=1e-4, rtol=1e-4,
            err_msg=f"N={N}: backproject half_img should match full_img"
        )

    def test_backproject_all_four_combos_self_consistent(self, N):
        """Backproject all 4 combos should produce volumes that re-project consistently.

        Half_vol backproject has 2x weight on interior kz (CONJ_MODE),
        so direct voxel comparison doesn't work. Instead verify that
        the full-vol+full-img and full-vol+half-img backproject paths
        produce the same full volume.
        """
        image_shape = (N, N)
        volume_shape = (N, N, N)
        rots = _random_rotations(5, seed=50)

        # Hermitian images from real data
        rng = np.random.default_rng(50)
        real_imgs = rng.standard_normal((5, N, N)).astype(np.float32)
        full_imgs = jnp.array(ftu.get_dft2(jnp.array(real_imgs))).reshape(5, -1)
        half_imgs = jnp.array(ftu.get_dft2_real(jnp.array(real_imgs))).reshape(5, -1)

        # Full-vol output from full-img vs half-img should match
        out_ff = np.asarray(slicing.adjoint_slice_volume(
            full_imgs, rots, image_shape, volume_shape, "linear_interp",
            half_image=False, half_volume=False))
        out_hf = np.asarray(slicing.adjoint_slice_volume(
            half_imgs, rots, image_shape, volume_shape, "linear_interp",
            half_image=True, half_volume=False))

        np.testing.assert_allclose(
            out_hf, out_ff, atol=1e-4, rtol=1e-4,
            err_msg=f"N={N}: full-vol from half-img should match full-vol from full-img"
        )


# ── 3. Upsampling (padding) tests with max_r ────────────────────────

class TestUpsamplingMaxR:
    """Tests for volume_shape != image_shape (upsampling/padding).

    With upsampling=2, image_shape=(N,N) maps to volume_shape=(2N,2N,2N).
    Image frequencies only reach ±N/2, so max_r = N//2-1 clips just inside.
    """

    @pytest.fixture(params=[
        ((16, 16), (32, 32, 32)),
        ((16, 16), (64, 64, 64)),
        ((32, 32), (64, 64, 64)),
    ], ids=["16to32", "16to64", "32to64"])
    def shapes(self, request):
        return request.param

    def test_project_upsampled_default_max_r(self, shapes):
        """Project with upsampling and default max_r produces valid output."""
        image_shape, volume_shape = shapes
        vol = _hermitian_volume_rect(volume_shape)
        rots = _random_rotations(3, seed=60)

        out = np.asarray(slicing.slice_volume(
            vol, rots, image_shape, volume_shape, "linear_interp"))

        assert out.shape == (3, image_shape[0] * image_shape[1])
        # Should have some nonzero pixels
        assert np.sum(np.abs(out) > 1e-30) > 0

    def test_project_upsampled_half_vol_half_img(self, shapes):
        """Half-vol+half-img with upsampling matches full-vol+half-img."""
        image_shape, volume_shape = shapes
        vol_full = _hermitian_volume_rect(volume_shape)
        vol_half = ftu.full_volume_to_half_volume(vol_full, volume_shape)
        rots = _random_rotations(3, seed=70)

        out_full = np.asarray(slicing.slice_volume(
            vol_full, rots, image_shape, volume_shape, "linear_interp",
            half_image=True))
        out_half = np.asarray(slicing.slice_volume(
            vol_half, rots, image_shape, volume_shape, "linear_interp",
            half_volume=True, half_image=True))

        np.testing.assert_allclose(
            out_half, out_full, atol=1e-4, rtol=1e-4,
            err_msg=f"Upsampled half_vol should match full_vol"
        )

    def test_backproject_upsampled_default_max_r(self, shapes):
        """Backproject with upsampling and default max_r."""
        image_shape, volume_shape = shapes
        rots = _random_rotations(3, seed=80)
        n_pix = image_shape[0] * image_shape[1]
        slices = _random_slices(3, n_pix, seed=80)

        out = np.asarray(slicing.adjoint_slice_volume(
            slices, rots, image_shape, volume_shape, "linear_interp"))

        vol_size = int(np.prod(volume_shape))
        assert out.shape == (vol_size,)
        assert np.sum(np.abs(out) > 1e-30) > 0

    def test_backproject_upsampled_half_img_vs_full_img(self, shapes):
        """Backproject with upsampling: full-vol from half-img vs full-img should match."""
        image_shape, volume_shape = shapes
        N = image_shape[0]
        rots = _random_rotations(3, seed=90)

        # Hermitian images
        rng = np.random.default_rng(90)
        real_imgs = rng.standard_normal((3, N, N)).astype(np.float32)
        full_imgs = jnp.array(ftu.get_dft2(jnp.array(real_imgs))).reshape(3, -1)
        half_imgs = jnp.array(ftu.get_dft2_real(jnp.array(real_imgs))).reshape(3, -1)

        out_full = np.asarray(slicing.adjoint_slice_volume(
            full_imgs, rots, image_shape, volume_shape, "linear_interp",
            half_image=False, half_volume=False))
        out_half = np.asarray(slicing.adjoint_slice_volume(
            half_imgs, rots, image_shape, volume_shape, "linear_interp",
            half_image=True, half_volume=False))

        np.testing.assert_allclose(
            out_half, out_full, atol=1e-4, rtol=1e-4,
            err_msg="Upsampled backproject half_img should match full_img"
        )

    def test_max_r_clips_correctly_with_upsampling(self, shapes):
        """With upsampling, max_r should still zero out high frequencies."""
        image_shape, volume_shape = shapes
        vol = _hermitian_volume_rect(volume_shape)
        rots = _random_rotations(3, seed=100)

        # Small max_r should clip aggressively
        out_small = np.asarray(slicing.slice_volume(
            vol, rots, image_shape, volume_shape, "linear_interp", max_r=3.0))
        out_none = np.asarray(slicing.slice_volume(
            vol, rots, image_shape, volume_shape, "linear_interp", max_r=None))

        n_zero_small = np.sum(np.abs(out_small) < 1e-30)
        n_zero_none = np.sum(np.abs(out_none) < 1e-30)
        assert n_zero_small > n_zero_none


# ── 4. Adjointness test with default max_r ───────────────────────────

class TestAdjointnessWithMaxR:
    """<Ax, y> should equal <x, A^T y> for project/backproject with max_r."""

    @pytest.fixture(params=[16, 32], ids=["N16", "N32"])
    def N(self, request):
        return request.param

    def test_adjointness_default_max_r(self, N):
        """Adjointness holds under default max_r clipping."""
        image_shape = (N, N)
        volume_shape = (N, N, N)
        n_images = 5
        rots = _random_rotations(n_images, seed=110)

        rng = np.random.default_rng(110)
        vol_data = (rng.standard_normal(N**3) + 1j * rng.standard_normal(N**3)).astype(np.complex64)
        vol = jnp.array(vol_data)
        img_data = (rng.standard_normal((n_images, N*N)) + 1j * rng.standard_normal((n_images, N*N))).astype(np.complex64)
        imgs = jnp.array(img_data)

        # Forward: Ax
        Ax = slicing.slice_volume(
            vol, rots, image_shape, volume_shape, "linear_interp")
        # Adjoint: A^T y
        ATy = slicing.adjoint_slice_volume(
            imgs, rots, image_shape, volume_shape, "linear_interp")

        # <Ax, y>
        lhs = float(jnp.real(jnp.vdot(Ax, imgs)))
        # <x, A^T y>
        rhs = float(jnp.real(jnp.vdot(vol, ATy)))

        rel_err = abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-30)
        assert rel_err < 1e-4, f"Adjointness failed: lhs={lhs}, rhs={rhs}, rel_err={rel_err}"

    def test_adjointness_explicit_max_r(self, N):
        """Adjointness with explicit max_r value."""
        image_shape = (N, N)
        volume_shape = (N, N, N)
        max_r = N // 4
        n_images = 5
        rots = _random_rotations(n_images, seed=130)

        rng = np.random.default_rng(130)
        vol = jnp.array((rng.standard_normal(N**3) + 1j * rng.standard_normal(N**3)).astype(np.complex64))
        imgs = jnp.array((rng.standard_normal((n_images, N*N)) + 1j * rng.standard_normal((n_images, N*N))).astype(np.complex64))

        Ax = slicing.slice_volume(
            vol, rots, image_shape, volume_shape, "linear_interp", max_r=max_r)
        ATy = slicing.adjoint_slice_volume(
            imgs, rots, image_shape, volume_shape, "linear_interp", max_r=max_r)

        lhs = float(jnp.real(jnp.vdot(Ax, imgs)))
        rhs = float(jnp.real(jnp.vdot(vol, ATy)))

        rel_err = abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-30)
        assert rel_err < 1e-4, f"Adjointness failed: lhs={lhs}, rhs={rhs}, rel_err={rel_err}"


# ── 5. Batch functions respect default max_r ─────────────────────────

class TestBatchDefaultMaxR:
    """Batch functions should use the same default max_r as single functions."""

    def test_batch_slice_matches_single(self):
        N = 16
        image_shape = (N, N)
        volume_shape = (N, N, N)
        n_vols = 3
        rots = _random_rotations(5, seed=140)

        rng = np.random.default_rng(140)
        vols = jnp.array(
            (rng.standard_normal((n_vols, N**3)) +
             1j * rng.standard_normal((n_vols, N**3))).astype(np.complex64))

        # Batch call with default max_r
        batch_out = np.asarray(slicing.batch_slice_volume(
            vols, rots, image_shape, volume_shape, "linear_interp"))

        # Single calls with default max_r
        for i in range(n_vols):
            single_out = np.asarray(slicing.slice_volume(
                vols[i], rots, image_shape, volume_shape, "linear_interp"))
            np.testing.assert_allclose(
                batch_out[i], single_out, atol=1e-5, rtol=1e-5,
                err_msg=f"Batch vol {i} differs from single"
            )

    def test_batch_adjoint_matches_single(self):
        N = 16
        image_shape = (N, N)
        volume_shape = (N, N, N)
        n_vols = 3
        n_images = 5
        rots = _random_rotations(n_images, seed=150)
        slices_data = _random_slices(n_images, N * N, seed=150)
        batch_slices = jnp.stack([slices_data] * n_vols)

        batch_out = np.asarray(slicing.batch_adjoint_slice_volume(
            batch_slices, rots, image_shape, volume_shape, "linear_interp"))

        for i in range(n_vols):
            single_out = np.asarray(slicing.adjoint_slice_volume(
                slices_data, rots, image_shape, volume_shape, "linear_interp"))
            np.testing.assert_allclose(
                batch_out[i], single_out, atol=1e-5, rtol=1e-5,
                err_msg=f"Batch vol {i} differs from single"
            )

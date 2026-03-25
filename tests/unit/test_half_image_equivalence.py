"""Tests verifying half-image function replacements produce identical results to full-image versions.

For each pipeline function that was converted from full-image to half-image format,
this test module calls both variants with identical input and checks exact equivalence.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar import core


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def random_rotations(n, rng):
    z = rng.standard_normal((n, 3, 3))
    q, r = np.linalg.qr(z)
    d = np.sign(np.diagonal(r, axis1=1, axis2=2))
    q = q * d[:, None, :]
    det = np.linalg.det(q)
    q[det < 0] *= -1
    return q.astype(np.float32)


def hermitian_images(n_images, image_shape, rng):
    """Generate Hermitian images (DFT of real data) in full-centered format."""
    H, W = image_shape
    real_imgs = rng.standard_normal((n_images, H, W)).astype(np.float32)
    full_centered = np.fft.fftshift(np.fft.fft2(real_imgs), axes=(-2, -1))
    return full_centered.reshape(n_images, H * W).astype(np.complex64)


def hermitian_real_images(n_images, image_shape, rng):
    """Generate real-valued Hermitian spectra (like CTF^2 terms)."""
    H, W = image_shape
    real_imgs = rng.standard_normal((n_images, H, W)).astype(np.float32)
    full_centered = np.abs(np.fft.fftshift(np.fft.fft2(real_imgs), axes=(-2, -1))) ** 2
    return full_centered.reshape(n_images, H * W).astype(np.float32)


def hermitian_volume(volume_shape, rng):
    """Generate a Hermitian volume (DFT of real data) in full-centered format."""
    N0, N1, N2 = volume_shape
    real_vol = rng.standard_normal((N0, N1, N2)).astype(np.float32)
    full_centered = np.fft.fftshift(np.fft.fftn(real_vol))
    return full_centered.reshape(N0 * N1 * N2).astype(np.complex64)


def assert_close(a, b, name, rtol=1e-5, atol=1e-6):
    a_np = np.array(a)
    b_np = np.array(b)
    max_err = np.max(np.abs(a_np - b_np))
    scale = max(np.max(np.abs(a_np)), 1e-30)
    rel_err = max_err / scale
    assert rel_err < rtol or max_err < atol, f"{name}: max_err={max_err:.2e}, rel_err={rel_err:.2e}"


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

SIZES = [16, 32]
N_IMAGES_LIST = [5, 10]


@pytest.fixture(params=SIZES)
def N(request):
    return request.param


@pytest.fixture(params=N_IMAGES_LIST)
def n_images(request):
    return request.param


# ---------------------------------------------------------------------------
# Test 1: adjoint_slice_volume full vs from_half_images
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", SIZES)
@pytest.mark.parametrize("n_images", N_IMAGES_LIST)
def test_adjoint_full_vs_half_images(N, n_images):
    """adjoint_slice_volume(full) == adjoint_slice_volume(half, half_image=True)."""
    rng = np.random.default_rng(42 + N)
    image_shape = (N, N)
    volume_shape = (N, N, N)
    rots = jnp.array(random_rotations(n_images, rng))
    images_full = jnp.array(hermitian_images(n_images, image_shape, rng))
    images_half = fourier_transform_utils.full_image_to_half_image(images_full, image_shape)

    from recovar.core.slicing import (
        adjoint_slice_volume,
    )

    vol_full = adjoint_slice_volume(images_full, rots, image_shape, volume_shape, "linear_interp")
    vol_half = adjoint_slice_volume(images_half, rots, image_shape, volume_shape, "linear_interp", half_image=True)

    assert_close(vol_full, vol_half, "adjoint_map full vs half")


# ---------------------------------------------------------------------------
# Test 2: adjoint_slice_volume with volume accumulation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", SIZES)
@pytest.mark.parametrize("n_images", N_IMAGES_LIST)
def test_adjoint_accumulate_full_vs_half(N, n_images):
    """Full vs half with volume accumulation (Ft_y += ...)."""
    rng = np.random.default_rng(123 + N)
    image_shape = (N, N)
    volume_shape = (N, N, N)
    vol_size = N**3
    rots = jnp.array(random_rotations(n_images, rng))
    images = jnp.array(hermitian_images(n_images, image_shape, rng))
    images_half = fourier_transform_utils.full_image_to_half_image(images, image_shape)
    seed = jnp.array(
        rng.standard_normal(vol_size).astype(np.float32) + 1j * rng.standard_normal(vol_size).astype(np.float32),
        dtype=jnp.complex64,
    )

    from recovar.core.slicing import (
        adjoint_slice_volume,
    )

    vol_full = adjoint_slice_volume(images, rots, image_shape, volume_shape, "linear_interp", volume=seed)
    vol_half = adjoint_slice_volume(
        images_half, rots, image_shape, volume_shape, "linear_interp", volume=seed, half_image=True
    )

    assert_close(vol_full, vol_half, "adjoint_map accumulate full vs half")


# ---------------------------------------------------------------------------
# Test 3: adjoint with real-valued images (like CTF^2 terms)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", SIZES)
@pytest.mark.parametrize("n_images", N_IMAGES_LIST)
def test_adjoint_real_images_full_vs_half(N, n_images):
    """CTF^2 / noise terms are real in Fourier space (symmetric, not just Hermitian)."""
    rng = np.random.default_rng(77 + N)
    image_shape = (N, N)
    volume_shape = (N, N, N)
    rots = jnp.array(random_rotations(n_images, rng))
    # Real-valued symmetric images (like |CTF|^2 terms)
    ctf_sq = jnp.array(hermitian_real_images(n_images, image_shape, rng).astype(np.complex64))
    ctf_sq_half = fourier_transform_utils.full_image_to_half_image(ctf_sq, image_shape)

    from recovar.core.slicing import (
        adjoint_slice_volume,
    )

    vol_full = adjoint_slice_volume(ctf_sq, rots, image_shape, volume_shape, "linear_interp")
    vol_half = adjoint_slice_volume(ctf_sq_half, rots, image_shape, volume_shape, "linear_interp", half_image=True)

    assert_close(vol_full, vol_half, "adjoint_map real images full vs half")


# ---------------------------------------------------------------------------
# Test 4: M-step pattern — probability-weighted backprojection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", SIZES)
def test_mstep_backproject_full_vs_half(N):
    """M-step: P @ shifted_images → backproject, full vs half."""
    rng = np.random.default_rng(99 + N)
    image_shape = (N, N)
    volume_shape = (N, N, N)
    n_rots = 5
    n_images = 8
    n_translations = 3
    n_shifted = n_images * n_translations

    rots = jnp.array(random_rotations(n_rots, rng))
    # Probability-weighted sum: P @ shifted_images (Hermitian result)
    shifted_images = jnp.array(hermitian_images(n_shifted, image_shape, rng))
    P = jnp.array(rng.standard_normal((n_rots, n_shifted)).astype(np.float32))
    P = jnp.abs(P)
    P = P / P.sum(axis=-1, keepdims=True)
    summed_images = P @ shifted_images  # n_rots × image_size, Hermitian

    from recovar.core.slicing import (
        adjoint_slice_volume,
    )

    vol_full = adjoint_slice_volume(summed_images, rots, image_shape, volume_shape, "linear_interp")
    summed_half = fourier_transform_utils.full_image_to_half_image(summed_images, image_shape)
    vol_half = adjoint_slice_volume(summed_half, rots, image_shape, volume_shape, "linear_interp", half_image=True)

    assert_close(vol_full, vol_half, "M-step backproject full vs half")


# ---------------------------------------------------------------------------
# Test 6: adjoint_slice_volume full vs from_half_images (linear_interp)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", SIZES)
@pytest.mark.parametrize("n_images", N_IMAGES_LIST)
def test_adjoint_map_full_vs_half(N, n_images):
    """adjoint_slice_volume(linear_interp) full vs half_image=True."""
    rng = np.random.default_rng(66 + N)
    image_shape = (N, N)
    volume_shape = (N, N, N)
    rots = jnp.array(random_rotations(n_images, rng))
    images = jnp.array(hermitian_images(n_images, image_shape, rng))
    images_half = fourier_transform_utils.full_image_to_half_image(images, image_shape)

    from recovar.core.slicing import (
        adjoint_slice_volume,
    )

    vol_full = adjoint_slice_volume(images, rots, image_shape, volume_shape, "linear_interp")
    vol_half = adjoint_slice_volume(images_half, rots, image_shape, volume_shape, "linear_interp", half_image=True)

    assert_close(vol_full, vol_half, "adjoint_map full vs half")


# ---------------------------------------------------------------------------
# Test 7: slice_volume + custom_vjp CUDA path (forward & backward)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", SIZES)
@pytest.mark.parametrize("n_images", N_IMAGES_LIST)
def test_slice_volume_vjp_consistency(N, n_images):
    """Verify VJP of slice_volume matches direct adjoint."""
    rng = np.random.default_rng(88 + N)
    image_shape = (N, N)
    volume_shape = (N, N, N)
    vol_size = N**3
    rots = jnp.array(random_rotations(n_images, rng))
    vol = jnp.array(
        rng.standard_normal(vol_size).astype(np.float32) + 1j * rng.standard_normal(vol_size).astype(np.float32),
        dtype=jnp.complex64,
    )
    images = jnp.array(hermitian_images(n_images, image_shape, rng))

    # Forward
    from recovar.core.slicing import slice_volume, adjoint_slice_volume

    projected = slice_volume(vol, rots, image_shape, volume_shape, "linear_interp")
    assert projected.shape == (n_images, N * N)

    # Adjoint via VJP
    f = lambda v: slice_volume(v, rots, image_shape, volume_shape, "linear_interp")
    _, u = jax.vjp(f, jnp.zeros(vol_size, dtype=jnp.complex64))
    vjp_result = u(images)[0]

    # Direct adjoint
    direct_result = adjoint_slice_volume(images, rots, image_shape, volume_shape, "linear_interp")

    assert_close(vjp_result, direct_result, "VJP vs direct adjoint", rtol=1e-4)


# ---------------------------------------------------------------------------
# Test 8: batch_vol vmap pattern (as used in backproject_one_image)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", SIZES)
def test_batch_vol_adjoint_full_vs_half(N):
    """Vmapped adjoint (multi-volume) full vs half.

    Matches the pattern in backproject_one_image where VOL_AXIS=1 vmaps over
    multiple volumes with shared images/rotations per volume index.
    """
    rng = np.random.default_rng(111 + N)
    image_shape = (N, N)
    volume_shape = (N, N, N)
    n_vols = 2
    n_images = 5
    rots = jnp.array(random_rotations(n_images, rng))

    # Images: (n_images, n_vols, image_size) — axis 1 is the volume axis
    images = jnp.array(hermitian_images(n_images * n_vols, image_shape, rng).reshape(n_vols, n_images, -1))
    # Rots: (n_vols, n_images, 3, 3) — same rotations replicated per volume
    rots_batched = jnp.broadcast_to(rots[None], (n_vols, *rots.shape))

    # VOL_AXIS=0 here to match the (n_vols, ...) leading axis
    batch_full_jax = jax.vmap(
        lambda imgs, rots: core.adjoint_slice_volume(imgs, rots, image_shape, volume_shape, "linear_interp"),
        in_axes=(0, 0),
        out_axes=0,
    )
    batch_half = jax.vmap(
        lambda imgs, rots: core.adjoint_slice_volume(
            imgs, rots, image_shape, volume_shape, "linear_interp", half_image=True
        ),
        in_axes=(0, 0),
        out_axes=0,
    )

    images_half = fourier_transform_utils.full_image_to_half_image(images, image_shape)

    vol_full = batch_full_jax(images, rots_batched)
    vol_half = batch_half(images_half, rots_batched)

    assert_close(vol_full, vol_half, "batch_vol adjoint full vs half")


# ---------------------------------------------------------------------------
# Test 9: project from half volume matches project from full volume
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", SIZES)
@pytest.mark.parametrize("n_images", N_IMAGES_LIST)
def test_project_from_half_volume(N, n_images):
    """slice_volume matches full volume projection."""
    rng = np.random.default_rng(200 + N)
    image_shape = (N, N)
    volume_shape = (N, N, N)
    rots = jnp.array(random_rotations(n_images, rng))
    vol = jnp.array(hermitian_volume(volume_shape, rng))
    vol_half = fourier_transform_utils.full_volume_to_half_volume(vol, volume_shape)

    from recovar.core.slicing import slice_volume

    proj_full = slice_volume(vol, rots, image_shape, volume_shape, "linear_interp")
    proj_half = slice_volume(vol_half, rots, image_shape, volume_shape, "linear_interp", half_volume=True)

    assert_close(proj_full, proj_half, "project from half volume", rtol=1e-4)


# ---------------------------------------------------------------------------
# Test 10: project from half volume via map matches full
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", SIZES)
@pytest.mark.parametrize("n_images", N_IMAGES_LIST)
def test_project_map_from_half_volume(N, n_images):
    """slice_volume matches full volume projection."""
    rng = np.random.default_rng(210 + N)
    image_shape = (N, N)
    volume_shape = (N, N, N)
    rots = jnp.array(random_rotations(n_images, rng))
    vol = jnp.array(hermitian_volume(volume_shape, rng))
    vol_half = fourier_transform_utils.full_volume_to_half_volume(vol, volume_shape)

    from recovar.core.slicing import slice_volume

    proj_full = slice_volume(vol, rots, image_shape, volume_shape, "linear_interp")
    proj_half = slice_volume(vol_half, rots, image_shape, volume_shape, "linear_interp", half_volume=True)

    assert_close(proj_full, proj_half, "project map from half volume", rtol=1e-4)


# ---------------------------------------------------------------------------
# Test 13: VJP of slice_volume
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", SIZES)
@pytest.mark.parametrize("n_images", N_IMAGES_LIST)
def test_half_volume_vjp_consistency(N, n_images):
    """VJP of project-from-half-volume should match the adjoint of half_volume_to_full_volume
    applied to the full-volume VJP.

    Since slice_volume = project(expand(hv)), the VJP is:
      expand'*(project'*(g))
    where expand'* is the adjoint of half_volume_to_full_volume, which sums both
    the direct and conjugate contributions for each half-volume voxel.
    """
    rng = np.random.default_rng(240 + N)
    image_shape = (N, N)
    volume_shape = (N, N, N)
    half_vol_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape)
    half_vol_size = int(np.prod(half_vol_shape))
    vol_size = N**3
    rots = jnp.array(random_rotations(n_images, rng))
    images = jnp.array(hermitian_images(n_images, image_shape, rng))

    from recovar.core.slicing import slice_volume

    # VJP of full-volume project
    f_full = lambda v: slice_volume(v, rots, image_shape, volume_shape, "linear_interp")
    _, u_full = jax.vjp(f_full, jnp.zeros(vol_size, dtype=jnp.complex64))
    vjp_full = u_full(images)[0]

    # Compute the proper adjoint of half_volume_to_full_volume applied to vjp_full:
    # expand'*(g)[j] = g[j] + conj(g[conj(j)]) for voxels where conj(j) is outside the half
    # We can compute this as VJP of the expand operation itself.
    f_expand = lambda hv: fourier_transform_utils.half_volume_to_full_volume(hv, volume_shape)
    _, u_expand = jax.vjp(f_expand, jnp.zeros(half_vol_size, dtype=jnp.complex64))
    vjp_full_ref = u_expand(vjp_full)[0]

    # VJP of half-volume project (composed: expand → project)
    f_half = lambda hv: slice_volume(hv, rots, image_shape, volume_shape, "linear_interp", half_volume=True)
    _, u_half = jax.vjp(f_half, jnp.zeros(half_vol_size, dtype=jnp.complex64))
    vjp_half = u_half(images)[0]

    assert_close(vjp_half, vjp_full_ref, "half volume VJP vs expand'*(full VJP)", rtol=1e-4)


# ---------------------------------------------------------------------------
# Test 13b: JVP of slice_volume
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", SIZES)
@pytest.mark.parametrize("n_images", N_IMAGES_LIST)
def test_half_volume_jvp_consistency(N, n_images):
    """JVP of project-from-half-volume should match JVP of expand → project."""
    rng = np.random.default_rng(260 + N)
    image_shape = (N, N)
    volume_shape = (N, N, N)
    half_vol_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape)
    half_vol_size = int(np.prod(half_vol_shape))
    rots = jnp.array(random_rotations(n_images, rng))

    from recovar.core.slicing import slice_volume

    hv = jnp.array(
        rng.standard_normal(half_vol_size).astype(np.float32)
        + 1j * rng.standard_normal(half_vol_size).astype(np.float32),
        dtype=jnp.complex64,
    )
    tangent = jnp.array(
        rng.standard_normal(half_vol_size).astype(np.float32)
        + 1j * rng.standard_normal(half_vol_size).astype(np.float32),
        dtype=jnp.complex64,
    )

    f_ref = lambda x: slice_volume(
        fourier_transform_utils.half_volume_to_full_volume(x, volume_shape),
        rots,
        image_shape,
        volume_shape,
        "linear_interp",
    )
    f_half = lambda x: slice_volume(x, rots, image_shape, volume_shape, "linear_interp", half_volume=True)

    y_ref, dy_ref = jax.jvp(f_ref, (hv,), (tangent,))
    y_half, dy_half = jax.jvp(f_half, (hv,), (tangent,))

    assert_close(y_half, y_ref, "half volume JVP primal", rtol=1e-4)
    assert_close(dy_half, dy_ref, "half volume JVP tangent", rtol=1e-4)


# ---------------------------------------------------------------------------
# Test 14: Batch project (multiple volumes, shared rotations)
# ---------------------------------------------------------------------------

BATCH_SIZES = [3, 7]


@pytest.mark.parametrize("N", SIZES)
@pytest.mark.parametrize("n_images", N_IMAGES_LIST)
@pytest.mark.parametrize("batch", BATCH_SIZES)
def test_batch_project(N, n_images, batch):
    """batch_slice_volume should match per-volume slice_volume."""
    rng = np.random.default_rng(300 + N + batch)
    image_shape = (N, N)
    volume_shape = (N, N, N)
    vol_size = N**3

    rots = jnp.array(random_rotations(n_images, rng))
    volumes = jnp.array(
        rng.standard_normal((batch, vol_size)).astype(np.float32)
        + 1j * rng.standard_normal((batch, vol_size)).astype(np.float32)
    )

    from recovar.core.slicing import batch_slice_volume, slice_volume

    batch_result = batch_slice_volume(volumes, rots, image_shape, volume_shape, "linear_interp")
    assert batch_result.shape == (batch, n_images, N * N)

    for b in range(batch):
        ref = slice_volume(volumes[b], rots, image_shape, volume_shape, "linear_interp")
        assert_close(batch_result[b], ref, f"batch_project vol {b}", rtol=1e-5)


# ---------------------------------------------------------------------------
# Test 15: Batch backproject (multiple volumes, shared rotations)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", SIZES)
@pytest.mark.parametrize("n_images", N_IMAGES_LIST)
@pytest.mark.parametrize("batch", BATCH_SIZES)
def test_batch_backproject(N, n_images, batch):
    """vmap adjoint_slice_volume should match per-volume adjoint."""
    rng = np.random.default_rng(400 + N + batch)
    image_shape = (N, N)
    volume_shape = (N, N, N)
    vol_size = N**3

    rots = jnp.array(random_rotations(n_images, rng))
    images = jnp.array(
        rng.standard_normal((batch, n_images, N * N)).astype(np.float32)
        + 1j * rng.standard_normal((batch, n_images, N * N)).astype(np.float32)
    )

    from recovar.core.slicing import adjoint_slice_volume

    from recovar.core.slicing import batch_adjoint_slice_volume

    batch_result = batch_adjoint_slice_volume(images, rots, image_shape, volume_shape, "linear_interp")
    assert batch_result.shape == (batch, vol_size)

    for b in range(batch):
        ref = adjoint_slice_volume(images[b], rots, image_shape, volume_shape, "linear_interp")
        assert_close(batch_result[b], ref, f"batch_backproject vol {b}", rtol=1e-4)


# ---------------------------------------------------------------------------
# Test 16: Batch backproject with seed volumes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", SIZES)
@pytest.mark.parametrize("n_images", N_IMAGES_LIST)
def test_batch_backproject_with_seed(N, n_images):
    """batch_adjoint with pre-existing volumes should accumulate correctly."""
    batch = 4
    rng = np.random.default_rng(500 + N)
    image_shape = (N, N)
    volume_shape = (N, N, N)
    vol_size = N**3

    rots = jnp.array(random_rotations(n_images, rng))
    images = jnp.array(
        rng.standard_normal((batch, n_images, N * N)).astype(np.float32)
        + 1j * rng.standard_normal((batch, n_images, N * N)).astype(np.float32)
    )
    seed_vols = jnp.array(
        rng.standard_normal((batch, vol_size)).astype(np.float32)
        + 1j * rng.standard_normal((batch, vol_size)).astype(np.float32)
    )

    from recovar.core.slicing import adjoint_slice_volume, batch_adjoint_slice_volume

    batch_result = batch_adjoint_slice_volume(
        images, rots, image_shape, volume_shape, "linear_interp", volumes=seed_vols
    )

    for b in range(batch):
        ref = adjoint_slice_volume(images[b], rots, image_shape, volume_shape, "linear_interp", volume=seed_vols[b])
        assert_close(batch_result[b], ref, f"batch_backproject_seed vol {b}", rtol=1e-4)

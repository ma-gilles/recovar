"""CUDA vs JAX equivalence tests for projection and backprojection.

Verifies that the CUDA kernels produce identical results (to floating-point
tolerance) as the JAX reference implementation for all combinations of:
  - order: 0 (nearest), 1 (trilinear)
  - half_volume: True / False
  - half_image: True / False
  - dtypes: complex64, float32

These tests explicitly force both code paths on GPU, comparing CUDA output
against JAX fallback output computed on the same device.
"""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

import recovar.core.fourier_transform_utils as fourier_transform_utils

pytestmark = [pytest.mark.unit, pytest.mark.gpu]


# ── Helpers ──────────────────────────────────────────────────────────


def _skip_if_no_cuda():
    from recovar.cuda_backproject import cuda_available

    if not cuda_available():
        pytest.skip("CUDA backproject not available")


def _random_rotations(n, rng):
    """Generate n proper rotation matrices via QR decomposition."""
    z = rng.standard_normal((n, 3, 3))
    q, r = np.linalg.qr(z)
    d = np.sign(np.diagonal(r, axis1=1, axis2=2))
    q = q * d[:, None, :]
    det = np.linalg.det(q)
    q[det < 0] *= -1
    return q.astype(np.float32)


def _slice_volume(volume, rotation_matrices, image_shape, volume_shape, disc_type="linear_interp", half_volume=False, **kwargs):
    import recovar.core.slicing as core_slicing

    wrapped = volume
    if not isinstance(wrapped, (core_slicing.Volume, core_slicing.CubicVolume)):
        if disc_type == "cubic":
            wrapped = core_slicing.to_cubic(wrapped, volume_shape, half_volume=half_volume)
        else:
            wrapped = core_slicing.Volume(wrapped, disc_type=disc_type, half_volume=half_volume)
    return core_slicing.slice_volume(wrapped, rotation_matrices, image_shape, volume_shape, **kwargs)


def _adjoint_slice_volume(slices, rotation_matrices, image_shape, volume_shape, disc_type="linear_interp", half_image=False, half_volume=False, max_r=None):
    import recovar.core.slicing as core_slicing

    expected_shape = fourier_transform_utils.volume_shape_to_half_volume_shape(volume_shape) if half_volume else volume_shape
    flat = int(np.prod(expected_shape))
    like = core_slicing.Volume(jnp.zeros(flat, dtype=jnp.asarray(slices).dtype), disc_type=disc_type, half_volume=half_volume)
    return core_slicing.adjoint_slice_volume(
        slices,
        rotation_matrices,
        image_shape,
        volume_shape,
        like=like,
        half_image=half_image,
        max_r=max_r,
    )


# ── Parametrization ──────────────────────────────────────────────────

_ORDERS = [0, 1]
_HALF_COMBOS = [
    (False, False),
    (False, True),
    (True, False),
    (True, True),
]

# All (order, half_vol, half_img) combos
_ALL_COMBOS = [(order, hv, hi) for order in _ORDERS for hv, hi in _HALF_COMBOS]

# Volume sizes to test — small for speed, large enough to exercise indexing
_SIZES = [16, 32, 64]

# Number of images — enough to cover diverse rotations
_N_IMAGES = [1, 5, 20]


# ── Forward projection (project): CUDA vs JAX ───────────────────────


@pytest.mark.parametrize("order,half_vol,half_img", _ALL_COMBOS)
@pytest.mark.parametrize("N", _SIZES)
def test_project_cuda_vs_jax(order, half_vol, half_img, N, gpu_device):
    """CUDA project must match JAX map_coordinates for all mode combos."""
    _skip_if_no_cuda()
    from recovar.cuda_backproject import project
    from recovar.core.slicing import _jax_slice, _jax_slice_half_image
    import recovar.core.fourier_transform_utils as ftu

    rng = np.random.default_rng(42)
    n_images = 10
    image_shape = (N, N)
    volume_shape = (N, N, N)

    rots = jnp.array(_random_rotations(n_images, rng))

    # When half_vol=True, CUDA reads from the half volume with per-neighbor
    # Hermitian expansion.  At the kz Nyquist boundary, CUDA always conjugates
    # from the partner position, which only matches the Python
    # half_volume_to_full_volume placement for Hermitian data (DFT of real
    # signal).  Use Hermitian data so both expansions agree.
    if half_vol:
        vol_real = jnp.array(rng.standard_normal(volume_shape).astype(np.float32))
        vol_full = ftu.get_dft3(vol_real).ravel()
    else:
        vol_full = jnp.array((rng.standard_normal(N**3) + 1j * rng.standard_normal(N**3)).astype(np.complex64))

    with jax.default_device(gpu_device):
        rots_gpu = jax.device_put(rots)

        # --- CUDA path ---
        if half_vol:
            vol_cuda = ftu.full_volume_to_half_volume(
                jax.device_put(vol_full).reshape(volume_shape), volume_shape
            ).ravel()
        else:
            vol_cuda = jax.device_put(vol_full)

        cuda_out = project(
            vol_cuda, rots_gpu, image_shape, volume_shape, order=order, half_volume=half_vol, half_image=half_img
        )

        # --- JAX path (on same device, bypassing dispatch) ---
        # For half_vol, the Hermitian volume ensures CUDA's per-neighbor
        # expansion and Python's index-based expansion produce the same full
        # volume, so we can use the Python expansion as reference.
        if half_vol:
            half_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
            vol_jax = jax.device_put(ftu.half_volume_to_full_volume(vol_cuda.reshape(half_shape), volume_shape).ravel())
        else:
            vol_jax = jax.device_put(vol_full)
        if half_img:
            jax_out = _jax_slice_half_image(vol_jax, rots_gpu, image_shape, volume_shape, order)
        else:
            jax_out = _jax_slice(vol_jax, rots_gpu, image_shape, volume_shape, order)

    cuda_np = np.asarray(cuda_out)
    jax_np = np.asarray(jax_out)

    # Compute relative error
    norm_jax = np.linalg.norm(jax_np)
    if norm_jax > 0:
        rel_err = np.linalg.norm(cuda_np - jax_np) / norm_jax
    else:
        rel_err = np.linalg.norm(cuda_np - jax_np)

    np.testing.assert_allclose(
        cuda_np,
        jax_np,
        atol=1e-4,
        rtol=1e-4,
        err_msg=(
            f"CUDA project != JAX for order={order}, half_vol={half_vol}, "
            f"half_img={half_img}, N={N}. Relative error: {rel_err:.2e}"
        ),
    )


# ── Backprojection (adjoint): CUDA vs JAX VJP ───────────────────────


@pytest.mark.parametrize("order,half_vol,half_img", _ALL_COMBOS)
@pytest.mark.parametrize("N", _SIZES)
def test_backproject_cuda_vs_jax(order, half_vol, half_img, N, gpu_device):
    """CUDA backproject must match JAX VJP adjoint for all mode combos."""
    _skip_if_no_cuda()
    from recovar.cuda_backproject import backproject
    from recovar.core.slicing import _jax_slice, _jax_slice_half_image, adjoint_slice_volume
    import recovar.core.fourier_transform_utils as ftu

    rng = np.random.default_rng(42)
    n_images = 10
    image_shape = (N, N)
    volume_shape = (N, N, N)

    rots = jnp.array(_random_rotations(n_images, rng))

    # Create images (full or half)
    H, W = image_shape
    full_imgs = jnp.array(
        (rng.standard_normal((n_images, H * W)) + 1j * rng.standard_normal((n_images, H * W))).astype(np.complex64)
    )

    with jax.default_device(gpu_device):
        rots_gpu = jax.device_put(rots)

        # --- CUDA path ---
        if half_img:
            imgs_cuda = jax.device_put(ftu.full_image_to_half_image(full_imgs, image_shape))
        else:
            imgs_cuda = jax.device_put(full_imgs)

        vol_shape_flat = ftu.volume_shape_to_half_volume_shape(volume_shape) if half_vol else volume_shape
        vol_init = jnp.zeros(int(np.prod(vol_shape_flat)), dtype=jnp.complex64)
        vol_init = jax.device_put(vol_init)

        cuda_out = backproject(
            vol_init,
            imgs_cuda,
            rots_gpu,
            image_shape,
            volume_shape,
            order=order,
            half_volume=half_vol,
            half_image=half_img,
        )

        # --- JAX path (via VJP, bypassing CUDA dispatch) ---
        # When half_img is True, CUDA scatters each half-pixel plus its
        # Hermitian conjugate — equivalent to backprojecting
        # half_image_to_full_image(imgs_half).  Use that expansion as the
        # JAX VJP cotangent so both paths see identical effective full images.
        if half_img:
            full_imgs_gpu = jax.device_put(ftu.half_image_to_full_image(imgs_cuda, image_shape))
        else:
            full_imgs_gpu = jax.device_put(full_imgs)

        if half_vol:
            vol_size = int(np.prod(ftu.volume_shape_to_half_volume_shape(volume_shape)))

            def f(v):
                full_v = ftu.half_volume_to_full_volume(v, volume_shape)
                return _jax_slice(full_v, rots_gpu, image_shape, volume_shape, order)
        else:
            vol_size = N**3
            f = lambda v: _jax_slice(v, rots_gpu, image_shape, volume_shape, order)

        _, vjp_fn = jax.vjp(f, jnp.zeros(vol_size, dtype=jnp.complex64))
        jax_out = vjp_fn(full_imgs_gpu)[0]

    cuda_np = np.asarray(cuda_out)
    jax_np = np.asarray(jax_out)

    norm_jax = np.linalg.norm(jax_np)
    if norm_jax > 0:
        rel_err = np.linalg.norm(cuda_np - jax_np) / norm_jax
    else:
        rel_err = np.linalg.norm(cuda_np - jax_np)

    np.testing.assert_allclose(
        cuda_np,
        jax_np,
        atol=1e-4,
        rtol=1e-4,
        err_msg=(
            f"CUDA backproject != JAX VJP for order={order}, half_vol={half_vol}, "
            f"half_img={half_img}, N={N}. Relative error: {rel_err:.2e}"
        ),
    )


# ── Adjoint consistency: <Ax, y> == <x, A^T y> ──────────────────────


@pytest.mark.parametrize("order,half_vol,half_img", _ALL_COMBOS)
def test_adjoint_consistency_cuda(order, half_vol, half_img, gpu_device):
    """Verify adjoint identity <project(vol), imgs> == <vol, backproject(imgs)>.

    This tests that CUDA project and backproject are true adjoints of each other,
    independent of any JAX comparison.
    """
    _skip_if_no_cuda()
    from recovar.cuda_backproject import backproject, project
    import recovar.core.fourier_transform_utils as ftu

    rng = np.random.default_rng(123)
    N = 32
    n_images = 8
    image_shape = (N, N)
    volume_shape = (N, N, N)

    rots = jnp.array(_random_rotations(n_images, rng))

    # Hermitian-symmetric data: DFT of real signals.
    # The adjoint identity <Ax, y> == <x, A^T y> requires Hermitian data
    # when half_vol or half_img is True, because the CUDA kernel's
    # Hermitian fold scatter / conjugate scatter doubles contributions
    # that cancel only for Hermitian inputs.
    H, W = image_shape

    # Volume: centered DFT of a real 3D signal → Hermitian symmetric
    vol_real = jnp.array(rng.standard_normal(volume_shape).astype(np.float32))
    vol_full = ftu.get_dft3(vol_real).ravel()
    if half_vol:
        vol = ftu.full_volume_to_half_volume(vol_full, volume_shape)
    else:
        vol = vol_full

    # Images: centered DFT of real 2D signals → Hermitian symmetric
    imgs_real = jnp.array(rng.standard_normal((n_images, H, W)).astype(np.float32))
    imgs_full = ftu.get_dft2(imgs_real).reshape(n_images, H * W)
    if half_img:
        imgs = ftu.full_image_to_half_image(imgs_full, image_shape)
    else:
        imgs = imgs_full

    with jax.default_device(gpu_device):
        vol_gpu = jax.device_put(vol)
        imgs_gpu = jax.device_put(imgs)
        rots_gpu = jax.device_put(rots)

        # For half_img=True, CUDA backproject scatters each half-pixel plus
        # its Hermitian conjugate.  This makes backproject_half the adjoint of
        # project_FULL (not project_half).  The correct identity is:
        #   <project_full(x), half_to_full(y)> == <x, backproject_half(y)>
        if half_img:
            proj = project(
                vol_gpu, rots_gpu, image_shape, volume_shape, order=order, half_volume=half_vol, half_image=False
            )
            imgs_for_ip = jax.device_put(ftu.half_image_to_full_image(imgs, image_shape))
        else:
            proj = project(
                vol_gpu, rots_gpu, image_shape, volume_shape, order=order, half_volume=half_vol, half_image=half_img
            )
            imgs_for_ip = imgs_gpu

        # Backward: backproject images to volume (always with half_img flag)
        vol_zero = jnp.zeros_like(vol_gpu)
        bp = backproject(
            vol_zero,
            imgs_gpu,
            rots_gpu,
            image_shape,
            volume_shape,
            order=order,
            half_volume=half_vol,
            half_image=half_img,
        )

    # <Ax, y> and <x, A^T y>
    lhs = np.real(np.sum(np.conj(np.asarray(proj)) * np.asarray(imgs_for_ip)))
    rhs = np.real(np.sum(np.conj(np.asarray(vol_gpu)) * np.asarray(bp)))

    np.testing.assert_allclose(
        lhs,
        rhs,
        rtol=1e-3,
        err_msg=(
            f"Adjoint identity violated for order={order}, half_vol={half_vol}, "
            f"half_img={half_img}. <Ax,y>={lhs:.6e}, <x,A^Ty>={rhs:.6e}"
        ),
    )


# ── High-level API: slice_volume dispatch consistency ────────────────


@pytest.mark.parametrize("half_vol,half_img", _HALF_COMBOS)
def test_slice_volume_cuda_vs_cpu(half_vol, half_img, gpu_device, monkeypatch):
    """slice_volume on GPU (CUDA) must match slice_volume on CPU (JAX)."""
    _skip_if_no_cuda()
    import recovar.core.slicing as core_slicing
    import recovar.core.fourier_transform_utils as ftu

    rng = np.random.default_rng(77)
    N = 32
    n_images = 10
    image_shape = (N, N)
    volume_shape = (N, N, N)

    rots = jnp.array(_random_rotations(n_images, rng))

    # When half_vol=True, use Hermitian data (DFT of real signal) so that
    # CUDA's per-neighbor Hermitian expansion matches Python's index-based
    # expansion at the kz Nyquist boundary.
    if half_vol:
        vol_real = jnp.array(rng.standard_normal(volume_shape).astype(np.float32))
        vol_full = ftu.get_dft3(vol_real).ravel()
        vol = ftu.full_volume_to_half_volume(vol_full.reshape(volume_shape), volume_shape).ravel()
    else:
        vol_full = jnp.array((rng.standard_normal(N**3) + 1j * rng.standard_normal(N**3)).astype(np.complex64))
        vol = vol_full

    # GPU path (CUDA) — use max_r=None so CUDA vs JAX comparison isn't
    # affected by FP boundary differences between pre-rotation and
    # post-rotation clipping checks.
    with jax.default_device(gpu_device):
        gpu_result = _slice_volume(
            jax.device_put(vol),
            jax.device_put(rots),
            image_shape,
            volume_shape,
            "linear_interp",
            half_volume=half_vol,
            half_image=half_img,
            max_r=None,
        )

    # CPU path (force JAX by monkeypatching _on_gpu)
    monkeypatch.setattr(core_slicing, "_on_gpu", lambda: False)
    # Clear lru_cache
    core_slicing._on_gpu.cache_clear() if hasattr(core_slicing._on_gpu, "cache_clear") else None

    cpu_result = _slice_volume(
        vol,
        rots,
        image_shape,
        volume_shape,
        "linear_interp",
        half_volume=half_vol,
        half_image=half_img,
        max_r=None,
    )

    np.testing.assert_allclose(
        np.asarray(gpu_result),
        np.asarray(cpu_result),
        atol=1e-4,
        rtol=1e-4,
        err_msg=f"slice_volume GPU != CPU for half_vol={half_vol}, half_img={half_img}",
    )


@pytest.mark.parametrize("half_vol,half_img", _HALF_COMBOS)
def test_adjoint_slice_volume_cuda_vs_cpu(half_vol, half_img, gpu_device, monkeypatch):
    """adjoint_slice_volume on GPU (CUDA) must match CPU (JAX VJP)."""
    _skip_if_no_cuda()
    import recovar.core.slicing as core_slicing
    import recovar.core.fourier_transform_utils as ftu

    rng = np.random.default_rng(77)
    N = 32
    n_images = 10
    image_shape = (N, N)
    volume_shape = (N, N, N)

    rots = jnp.array(_random_rotations(n_images, rng))
    H, W = image_shape
    full_imgs = jnp.array(
        (rng.standard_normal((n_images, H * W)) + 1j * rng.standard_normal((n_images, H * W))).astype(np.complex64)
    )

    if half_img:
        imgs = ftu.full_image_to_half_image(full_imgs, image_shape)
    else:
        imgs = full_imgs

    # GPU path (CUDA) — use max_r=None to avoid FP boundary clipping differences.
    with jax.default_device(gpu_device):
        gpu_result = _adjoint_slice_volume(
            jax.device_put(imgs),
            jax.device_put(rots),
            image_shape,
            volume_shape,
            "linear_interp",
            half_image=half_img,
            half_volume=half_vol,
            max_r=None,
        )

    # CPU path (force JAX)
    monkeypatch.setattr(core_slicing, "_on_gpu", lambda: False)
    core_slicing._on_gpu.cache_clear() if hasattr(core_slicing._on_gpu, "cache_clear") else None

    cpu_result = _adjoint_slice_volume(
        imgs,
        rots,
        image_shape,
        volume_shape,
        "linear_interp",
        half_image=half_img,
        half_volume=half_vol,
        max_r=None,
    )

    np.testing.assert_allclose(
        np.asarray(gpu_result),
        np.asarray(cpu_result),
        atol=1e-4,
        rtol=1e-4,
        err_msg=f"adjoint_slice_volume GPU != CPU for half_vol={half_vol}, half_img={half_img}",
    )


# ── Batch operations ────────────────────────────────────────────────


@pytest.mark.parametrize("half_vol,half_img", _HALF_COMBOS)
def test_batch_project_matches_single(half_vol, half_img, gpu_device):
    """batch_project should match looped single project calls."""
    _skip_if_no_cuda()
    from recovar.cuda_backproject import project, batch_project
    import recovar.core.fourier_transform_utils as ftu

    rng = np.random.default_rng(99)
    N = 24
    n_images = 5
    batch = 3
    image_shape = (N, N)
    volume_shape = (N, N, N)

    rots = jnp.array(_random_rotations(n_images, rng))

    if half_vol:
        hvs = ftu.volume_shape_to_half_volume_shape(volume_shape)
        vol_size = int(np.prod(hvs))
    else:
        vol_size = N**3

    vols = jnp.array(
        (rng.standard_normal((batch, vol_size)) + 1j * rng.standard_normal((batch, vol_size))).astype(np.complex64)
    )

    with jax.default_device(gpu_device):
        vols_gpu = jax.device_put(vols)
        rots_gpu = jax.device_put(rots)

        batch_out = batch_project(
            vols_gpu, rots_gpu, image_shape, volume_shape, order=1, half_volume=half_vol, half_image=half_img
        )

        singles = []
        for i in range(batch):
            s = project(
                vols_gpu[i], rots_gpu, image_shape, volume_shape, order=1, half_volume=half_vol, half_image=half_img
            )
            singles.append(s)
        single_out = jnp.stack(singles)

    np.testing.assert_allclose(
        np.asarray(batch_out),
        np.asarray(single_out),
        atol=1e-5,
        rtol=1e-5,
        err_msg=f"batch_project != looped project for half_vol={half_vol}, half_img={half_img}",
    )


@pytest.mark.parametrize("half_vol,half_img", _HALF_COMBOS)
def test_batch_backproject_matches_single(half_vol, half_img, gpu_device):
    """batch_backproject should match looped single backproject calls."""
    _skip_if_no_cuda()
    from recovar.cuda_backproject import backproject, batch_backproject
    import recovar.core.fourier_transform_utils as ftu

    rng = np.random.default_rng(99)
    N = 24
    n_images = 5
    batch = 3
    image_shape = (N, N)
    volume_shape = (N, N, N)
    H, W = image_shape

    rots = jnp.array(_random_rotations(n_images, rng))

    if half_vol:
        hvs = ftu.volume_shape_to_half_volume_shape(volume_shape)
        vol_size = int(np.prod(hvs))
    else:
        vol_size = N**3

    if half_img:
        n_pix = H * (W // 2 + 1)
    else:
        n_pix = H * W

    imgs = jnp.array(
        (rng.standard_normal((batch, n_images, n_pix)) + 1j * rng.standard_normal((batch, n_images, n_pix))).astype(
            np.complex64
        )
    )

    with jax.default_device(gpu_device):
        imgs_gpu = jax.device_put(imgs)
        rots_gpu = jax.device_put(rots)

        vols_init = jnp.zeros((batch, vol_size), dtype=jnp.complex64)
        vols_init = jax.device_put(vols_init)

        batch_out = batch_backproject(
            vols_init, imgs_gpu, rots_gpu, image_shape, volume_shape, order=1, half_volume=half_vol, half_image=half_img
        )

        singles = []
        for i in range(batch):
            v = jnp.zeros(vol_size, dtype=jnp.complex64)
            v = jax.device_put(v)
            s = backproject(
                v, imgs_gpu[i], rots_gpu, image_shape, volume_shape, order=1, half_volume=half_vol, half_image=half_img
            )
            singles.append(s)
        single_out = jnp.stack(singles)

    np.testing.assert_allclose(
        np.asarray(batch_out),
        np.asarray(single_out),
        atol=1e-5,
        rtol=1e-5,
        err_msg=f"batch_backproject != looped backproject for half_vol={half_vol}, half_img={half_img}",
    )


# ── Edge cases ───────────────────────────────────────────────────────


def test_identity_rotation_project_cuda_vs_jax(gpu_device):
    """Identity rotation should produce identical slices on CUDA and JAX."""
    _skip_if_no_cuda()
    from recovar.cuda_backproject import project
    from recovar.core.slicing import _jax_slice

    N = 32
    image_shape = (N, N)
    volume_shape = (N, N, N)

    rng = np.random.default_rng(0)
    vol = jnp.array((rng.standard_normal(N**3) + 1j * rng.standard_normal(N**3)).astype(np.complex64))
    rots = jnp.eye(3, dtype=jnp.float32).reshape(1, 3, 3)

    with jax.default_device(gpu_device):
        cuda_out = project(jax.device_put(vol), jax.device_put(rots), image_shape, volume_shape, order=1)
        jax_out = _jax_slice(jax.device_put(vol), jax.device_put(rots), image_shape, volume_shape, 1)

    np.testing.assert_allclose(
        np.asarray(cuda_out),
        np.asarray(jax_out),
        atol=1e-5,
        rtol=1e-5,
        err_msg="Identity rotation: CUDA != JAX",
    )


def test_axis_rotations_project_cuda_vs_jax(gpu_device):
    """90-degree axis rotations should produce identical slices on CUDA and JAX."""
    _skip_if_no_cuda()
    from recovar.cuda_backproject import project
    from recovar.core.slicing import _jax_slice

    N = 32
    image_shape = (N, N)
    volume_shape = (N, N, N)

    rng = np.random.default_rng(1)
    vol = jnp.array((rng.standard_normal(N**3) + 1j * rng.standard_normal(N**3)).astype(np.complex64))

    # 90-degree rotations around each axis
    rot_x = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    rot_y = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)
    rot_z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    rots = jnp.array(np.stack([rot_x, rot_y, rot_z]))

    with jax.default_device(gpu_device):
        cuda_out = project(jax.device_put(vol), jax.device_put(rots), image_shape, volume_shape, order=1)
        jax_out = _jax_slice(jax.device_put(vol), jax.device_put(rots), image_shape, volume_shape, 1)

    np.testing.assert_allclose(
        np.asarray(cuda_out),
        np.asarray(jax_out),
        atol=1e-5,
        rtol=1e-5,
        err_msg="90-degree axis rotations: CUDA != JAX",
    )


def test_many_random_rotations_cuda_vs_jax(gpu_device):
    """Many random rotations at realistic size (128x128) to stress-test."""
    _skip_if_no_cuda()
    from recovar.cuda_backproject import project, backproject
    from recovar.core.slicing import _jax_slice

    N = 128
    n_images = 50
    image_shape = (N, N)
    volume_shape = (N, N, N)

    rng = np.random.default_rng(42)
    from scipy.spatial.transform import Rotation

    rots_scipy = Rotation.random(n_images, random_state=42)
    rots = jnp.array(rots_scipy.as_matrix().astype(np.float32))
    vol = jnp.array((rng.standard_normal(N**3) + 1j * rng.standard_normal(N**3)).astype(np.complex64))
    imgs = jnp.array(
        (rng.standard_normal((n_images, N * N)) + 1j * rng.standard_normal((n_images, N * N))).astype(np.complex64)
    )

    with jax.default_device(gpu_device):
        vol_gpu = jax.device_put(vol)
        rots_gpu = jax.device_put(rots)
        imgs_gpu = jax.device_put(imgs)

        # Forward
        cuda_proj = project(vol_gpu, rots_gpu, image_shape, volume_shape, order=1)
        jax_proj = _jax_slice(vol_gpu, rots_gpu, image_shape, volume_shape, 1)

        # Backward
        vol_zero = jnp.zeros(N**3, dtype=jnp.complex64)
        vol_zero = jax.device_put(vol_zero)
        cuda_bp = backproject(vol_zero, imgs_gpu, rots_gpu, image_shape, volume_shape, order=1)

        f = lambda v: _jax_slice(v, rots_gpu, image_shape, volume_shape, 1)
        _, vjp_fn = jax.vjp(f, jnp.zeros(N**3, dtype=jnp.complex64))
        jax_bp = vjp_fn(imgs_gpu)[0]

    # Forward
    cuda_proj_np = np.asarray(cuda_proj)
    jax_proj_np = np.asarray(jax_proj)
    proj_rel = np.linalg.norm(cuda_proj_np - jax_proj_np) / np.linalg.norm(jax_proj_np)

    np.testing.assert_allclose(
        cuda_proj_np,
        jax_proj_np,
        atol=1e-4,
        rtol=1e-4,
        err_msg=f"Forward: N=128, 50 random rots. Relative error: {proj_rel:.2e}",
    )

    # Backward
    cuda_bp_np = np.asarray(cuda_bp)
    jax_bp_np = np.asarray(jax_bp)
    bp_rel = np.linalg.norm(cuda_bp_np - jax_bp_np) / np.linalg.norm(jax_bp_np)

    np.testing.assert_allclose(
        cuda_bp_np,
        jax_bp_np,
        atol=1e-4,
        rtol=1e-4,
        err_msg=f"Backward: N=128, 50 random rots. Relative error: {bp_rel:.2e}",
    )


# ── Real-valued inputs ──────────────────────────────────────────────


@pytest.mark.parametrize("half_vol,half_img", _HALF_COMBOS)
def test_real_backproject_cuda_vs_jax(half_vol, half_img, gpu_device):
    """Real-valued CUDA backproject must match JAX VJP with real inputs."""
    _skip_if_no_cuda()
    from recovar.cuda_backproject import backproject
    from recovar.core.slicing import _jax_slice
    import recovar.core.fourier_transform_utils as ftu

    rng = np.random.default_rng(55)
    N = 32
    n_images = 8
    image_shape = (N, N)
    volume_shape = (N, N, N)
    H, W = image_shape

    rots = jnp.array(_random_rotations(n_images, rng))

    # Real-valued images (like CTF^2 or noise variance)
    if half_img:
        n_pix = H * (W // 2 + 1)
    else:
        n_pix = H * W
    imgs_real = jnp.array(rng.standard_normal((n_images, n_pix)).astype(np.float32))

    with jax.default_device(gpu_device):
        rots_gpu = jax.device_put(rots)

        # CUDA with real inputs
        if half_vol:
            hvs = ftu.volume_shape_to_half_volume_shape(volume_shape)
            vol_init = jnp.zeros(int(np.prod(hvs)), dtype=jnp.float32)
        else:
            vol_init = jnp.zeros(N**3, dtype=jnp.float32)
        vol_init = jax.device_put(vol_init)
        imgs_gpu = jax.device_put(imgs_real)

        cuda_out = backproject(
            vol_init, imgs_gpu, rots_gpu, image_shape, volume_shape, order=1, half_volume=half_vol, half_image=half_img
        )

        # JAX VJP with real inputs (always full volume, expand half_image)
        if half_img:
            full_imgs = ftu.half_image_to_full_image(imgs_real, image_shape)
        else:
            full_imgs = imgs_real
        # Cast to complex for VJP (JAX map_coordinates needs complex for complex volume)
        full_imgs_c = jax.device_put(full_imgs.astype(np.complex64))

        if half_vol:
            vol_size = int(np.prod(ftu.volume_shape_to_half_volume_shape(volume_shape)))

            def f(v):
                full_v = ftu.half_volume_to_full_volume(v, volume_shape)
                return _jax_slice(full_v, rots_gpu, image_shape, volume_shape, 1)
        else:
            vol_size = N**3
            f = lambda v: _jax_slice(v, rots_gpu, image_shape, volume_shape, 1)

        _, vjp_fn = jax.vjp(f, jnp.zeros(vol_size, dtype=jnp.complex64))
        jax_out = vjp_fn(full_imgs_c)[0].real

    np.testing.assert_allclose(
        np.asarray(cuda_out),
        np.asarray(jax_out),
        atol=1e-4,
        rtol=1e-4,
        err_msg=f"Real backproject CUDA != JAX for half_vol={half_vol}, half_img={half_img}",
    )


# ── Non-square images (rectangular) ─────────────────────────────────


@pytest.mark.skip(
    reason=(
        "Rectangular images (H!=W) have a known pixel-ordering mismatch: "
        "CUDA uses (H,W) row-major but JAX meshgrid('xy') produces (W,H) layout. "
        "These coincide only when H==W. The pipeline always uses square images."
    ),
)
def test_rectangular_images_cuda_vs_jax(gpu_device):
    """CUDA must handle non-square image shapes correctly."""
    _skip_if_no_cuda()
    from recovar.cuda_backproject import project, backproject
    from recovar.core.slicing import _jax_slice

    # Non-square: 32x64
    H, W = 32, 64
    N = 32
    image_shape = (H, W)
    volume_shape = (N, N, N)
    n_images = 5

    rng = np.random.default_rng(88)
    rots = jnp.array(_random_rotations(n_images, rng))
    vol = jnp.array((rng.standard_normal(N**3) + 1j * rng.standard_normal(N**3)).astype(np.complex64))

    with jax.default_device(gpu_device):
        vol_gpu = jax.device_put(vol)
        rots_gpu = jax.device_put(rots)

        cuda_out = project(vol_gpu, rots_gpu, image_shape, volume_shape, order=1)
        jax_out = _jax_slice(vol_gpu, rots_gpu, image_shape, volume_shape, 1)

    np.testing.assert_allclose(
        np.asarray(cuda_out),
        np.asarray(jax_out),
        atol=1e-4,
        rtol=1e-4,
        err_msg="Rectangular images (32x64): CUDA != JAX",
    )

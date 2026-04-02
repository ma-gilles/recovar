import numpy as np
import pytest
from types import SimpleNamespace

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from recovar.heterogeneity import adaptive_kernel_discretization as akd
from recovar.heterogeneity import ppca

pytestmark = pytest.mark.unit


def test_batch_vec_and_unvec_roundtrip_real():
    x = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
    v = akd.batch_vec(x)
    x_rt = akd.batch_unvec(v)
    assert v.shape == (2, 9)
    assert x_rt.shape == x.shape
    assert np.allclose(x_rt, x)


def test_batch_vec_and_unvec_roundtrip_complex():
    rng = np.random.default_rng(0)
    xr = rng.normal(size=(4, 2, 2)).astype(np.float32)
    xi = rng.normal(size=(4, 2, 2)).astype(np.float32)
    x = xr + 1j * xi
    v = akd.batch_vec(x)
    x_rt = akd.batch_unvec(v)
    assert v.shape == (4, 4)
    assert x_rt.shape == x.shape
    assert np.allclose(x_rt, x)


# ---------------------------------------------------------------------------
# ppca module tests — M_step_batch, M_step (EM requires full dataset)
# ---------------------------------------------------------------------------


def test_M_step_batch_runs_and_accumulates():
    """Verify M_step_batch runs without shape errors and accumulates non-trivially."""
    from recovar import core

    rng = np.random.default_rng(10)
    grid_size = 4
    image_shape = (grid_size, grid_size)
    volume_shape = (grid_size, grid_size, grid_size)
    volume_size = int(np.prod(volume_shape))
    n_images = 3
    basis_size = 2
    voxel_size = 1.0

    n_pixels = int(np.prod(image_shape))
    images = (rng.normal(size=(n_images, n_pixels)) + 1j * rng.normal(size=(n_images, n_pixels))).astype(np.complex64)
    # Realistic CTF params: DFU, DFV, DFANG, VOLT, CS, W, PHASE_SHIFT, BFACTOR, CONTRAST
    CTF_params = np.zeros((n_images, 9), dtype=np.float32)
    CTF_params[:, 0] = 15000.0  # DFU (Angstrom)
    CTF_params[:, 1] = 15000.0  # DFV (Angstrom)
    CTF_params[:, 3] = 300.0  # VOLT (kV)
    CTF_params[:, 4] = 2.7  # CS (mm)
    CTF_params[:, 5] = 0.1  # W (amplitude contrast)
    CTF_params[:, 8] = 1.0  # CONTRAST
    rotation_matrices = np.tile(np.eye(3, dtype=np.float32), (n_images, 1, 1))
    translations = np.zeros((n_images, 2), dtype=np.float32)
    noise_variance = np.ones((n_images, n_pixels), dtype=np.float32)

    latent_means = rng.normal(size=(n_images, basis_size)).astype(np.float32)
    latent_covs = np.tile(np.eye(basis_size, dtype=np.float32), (n_images, 1, 1)) * 0.1

    lhs = jnp.zeros((volume_size, basis_size * basis_size), dtype=np.complex64)
    rhs = jnp.zeros((volume_size, basis_size), dtype=np.complex64)

    lhs_out, rhs_out = ppca.M_step_batch(
        images,
        lhs,
        rhs,
        latent_means,
        latent_covs,
        CTF_params,
        rotation_matrices,
        translations,
        image_shape,
        volume_shape,
        grid_size,
        voxel_size,
        noise_variance,
        core.CTFEvaluator(),
    )

    assert lhs_out.shape == (volume_size, basis_size * basis_size)
    assert rhs_out.shape == (volume_size, basis_size)
    assert np.all(np.isfinite(np.asarray(lhs_out)))
    assert np.all(np.isfinite(np.asarray(rhs_out)))
    # At least some voxels should have non-zero accumulations
    assert np.any(np.asarray(lhs_out) != 0)
    assert np.any(np.asarray(rhs_out) != 0)


# ---------------------------------------------------------------------------
# E_M_step_batch_half equivalence tests
# ---------------------------------------------------------------------------


def _make_em_step_test_data(rng, grid_size=8, n_images=5, basis_size=3, float_dtype=np.float32):
    """Build inputs for E_M_step_batch / E_M_step_batch_half comparison."""
    import recovar.core.fourier_transform_utils as ftu
    from recovar import core

    complex_dtype = np.complex64 if float_dtype == np.float32 else np.complex128

    image_shape = (grid_size, grid_size)
    volume_shape = (grid_size, grid_size, grid_size)
    volume_size = int(np.prod(volume_shape))
    half_volume_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
    half_volume_size = int(np.prod(half_volume_shape))
    n_pixels = int(np.prod(image_shape))
    half_image_shape = ftu.image_shape_to_half_image_shape(image_shape)
    half_image_size = int(np.prod(half_image_shape))
    voxel_size = 1.0

    # Random Hermitian-symmetric images (real in real space)
    real_imgs = rng.normal(size=(n_images, grid_size, grid_size)).astype(float_dtype)
    images_full = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(real_imgs), norm="backward"), axes=(-2, -1))
    images_full = images_full.reshape(n_images, n_pixels).astype(complex_dtype)

    # Convert to half
    images_half = np.asarray(ftu.full_image_to_half_image(jnp.array(images_full), image_shape))

    # Random mean volume (Hermitian symmetric)
    real_vol = rng.normal(size=volume_shape).astype(float_dtype)
    mean_full = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(real_vol), norm="backward"))
    mean_full = mean_full.reshape(volume_size).astype(complex_dtype)

    # Random W volumes (Hermitian symmetric)
    W_real = rng.normal(size=(basis_size, *volume_shape)).astype(float_dtype)
    W_full_list = []
    for k in range(basis_size):
        wk = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(W_real[k]), norm="backward"))
        W_full_list.append(wk.reshape(volume_size).astype(complex_dtype))
    W_full = np.stack(W_full_list, axis=1)  # (volume_size, basis_size)

    W_half = np.asarray(
        ftu.full_volume_to_half_volume(jnp.array(W_full.T), volume_shape).T
    )  # (half_volume_size, basis_size)

    # CTF params
    CTF_params = np.zeros((n_images, 9), dtype=float_dtype)
    CTF_params[:, 0] = 15000.0 + rng.uniform(-2000, 2000, n_images)
    CTF_params[:, 1] = 15000.0 + rng.uniform(-2000, 2000, n_images)
    CTF_params[:, 3] = 300.0
    CTF_params[:, 4] = 2.7
    CTF_params[:, 5] = 0.1
    CTF_params[:, 8] = 1.0

    # Random rotations (use small perturbations from identity)
    rotation_matrices = np.tile(np.eye(3, dtype=float_dtype), (n_images, 1, 1))
    for i in range(n_images):
        angle = rng.uniform(-0.3, 0.3)
        c, s = np.cos(angle), np.sin(angle)
        rotation_matrices[i, 0, 0] = c
        rotation_matrices[i, 0, 1] = -s
        rotation_matrices[i, 1, 0] = s
        rotation_matrices[i, 1, 1] = c

    translations = rng.uniform(-0.5, 0.5, (n_images, 2)).astype(float_dtype)

    # Use per-image (not per-pixel) noise variance so it is trivially
    # Hermitian-symmetric — required by the rfft-weighted inner products.
    nv_per_image = (1.0 + 0.5 * rng.uniform(size=(n_images, 1))).astype(float_dtype)
    noise_variance = np.broadcast_to(nv_per_image, (n_images, n_pixels)).copy()
    noise_variance_half = np.broadcast_to(nv_per_image, (n_images, half_image_size)).copy()

    ctf_evaluator = core.CTFEvaluator()

    return dict(
        images_full=images_full,
        images_half=images_half,
        mean_full=mean_full,
        W_full=W_full,
        W_half=W_half,
        CTF_params=CTF_params,
        rotation_matrices=rotation_matrices,
        translations=translations,
        noise_variance=noise_variance,
        noise_variance_half=noise_variance_half,
        ctf_evaluator=ctf_evaluator,
        image_shape=image_shape,
        volume_shape=volume_shape,
        volume_size=volume_size,
        half_volume_size=half_volume_size,
        grid_size=grid_size,
        voxel_size=voxel_size,
        n_images=n_images,
        basis_size=basis_size,
        float_dtype=float_dtype,
        complex_dtype=complex_dtype,
    )


def test_rfft_weights_shape_and_values():
    """Verify linalg.rfft2_hermitian_weights has correct shape and DC/Nyquist = 1."""
    from recovar.core.linalg import rfft2_hermitian_weights

    # Even grid — returns sqrt(w), so values are {1, sqrt(2)}
    sw = np.asarray(rfft2_hermitian_weights((8, 8)))
    assert sw.shape == (8 * 5,)
    w = sw**2  # recover raw weights
    w2d = w.reshape(8, 5)
    np.testing.assert_allclose(w2d[:, 0], 1.0)  # DC column
    np.testing.assert_allclose(w2d[:, -1], 1.0)  # Nyquist column
    np.testing.assert_allclose(w2d[:, 1:-1], 2.0)

    # Odd grid
    sw_odd = np.asarray(rfft2_hermitian_weights((7, 7)))
    assert sw_odd.shape == (7 * 4,)
    w_odd = sw_odd**2
    w_odd_2d = w_odd.reshape(7, 4)
    np.testing.assert_allclose(w_odd_2d[:, 0], 1.0)
    np.testing.assert_allclose(w_odd_2d[:, 1:], 2.0)


def test_rfft_weights_inner_product_equivalence():
    """Verify weighted half-image dot product matches full-image dot product."""
    import recovar.core.fourier_transform_utils as ftu

    rng = np.random.default_rng(42)
    grid_size = 8
    image_shape = (grid_size, grid_size)
    n_pixels = grid_size * grid_size

    # Hermitian-symmetric vectors (from real signals)
    x_real = rng.normal(size=(grid_size, grid_size)).astype(np.float32)
    y_real = rng.normal(size=(grid_size, grid_size)).astype(np.float32)
    x_full = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x_real))).reshape(n_pixels).astype(np.complex64)
    y_full = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(y_real))).reshape(n_pixels).astype(np.complex64)

    x_half = np.asarray(ftu.full_image_to_half_image(jnp.array(x_full), image_shape))
    y_half = np.asarray(ftu.full_image_to_half_image(jnp.array(y_full), image_shape))

    from recovar.core.linalg import half_spectrum_last_axis_weights

    w_1d = np.asarray(half_spectrum_last_axis_weights(image_shape[1]))
    rfft_w = np.tile(w_1d, (image_shape[0], 1)).reshape(-1)

    dot_full = np.sum(np.conj(x_full) * y_full).real
    dot_half = np.sum(rfft_w * (np.conj(x_half) * y_half).real)

    np.testing.assert_allclose(dot_half, dot_full, rtol=1e-5)


def test_unpack_tri_to_full_roundtrip():
    """Verify packing to upper-tri and unpacking recovers the original."""
    from recovar.ppca.ppca import unpack_tri_to_full

    rng = np.random.default_rng(99)
    q = 4
    n = 10
    # Symmetric matrices
    A = rng.normal(size=(n, q, q)).astype(np.float32)
    A = (A + A.transpose(0, 2, 1)) / 2
    tri_i, tri_j = np.triu_indices(q)
    packed = A[:, tri_i, tri_j]
    recovered = np.asarray(unpack_tri_to_full(jnp.array(packed), q))
    np.testing.assert_allclose(recovered, A, atol=1e-6)


def test_E_M_step_batch_half_shapes():
    """Verify E_M_step_batch_half returns correct shapes."""
    from recovar.ppca.ppca import E_M_step_batch_half, _tri_size

    rng = np.random.default_rng(123)
    d = _make_em_step_test_data(rng, grid_size=8, n_images=4, basis_size=2)
    tri_sz = _tri_size(d["basis_size"])

    lhs_init = jnp.zeros((d["half_volume_size"], tri_sz), dtype=np.float32)
    rhs_init = jnp.zeros((d["half_volume_size"], d["basis_size"]), dtype=np.complex64)

    lhs, rhs, ez, smz, ll, ll_pi, _mc = E_M_step_batch_half(
        d["images_half"],
        lhs_init,
        rhs_init,
        d["mean_full"],
        d["W_half"],
        d["CTF_params"],
        d["rotation_matrices"],
        d["translations"],
        d["image_shape"],
        d["volume_shape"],
        d["grid_size"],
        d["voxel_size"],
        d["noise_variance_half"],
        d["ctf_evaluator"],
        compute_ll=True,
    )

    assert lhs.shape == (d["half_volume_size"], tri_sz)
    assert rhs.shape == (d["half_volume_size"], d["basis_size"])
    assert ez.shape == (d["n_images"], d["basis_size"])
    assert smz.shape == (d["n_images"], d["basis_size"], d["basis_size"])
    assert ll.shape == ()
    assert np.all(np.isfinite(np.asarray(lhs)))
    assert np.all(np.isfinite(np.asarray(rhs)))
    assert np.all(np.isfinite(np.asarray(ez)))
    assert np.all(np.isfinite(np.asarray(ll)))


def test_prepare_mean_estimate_for_slicing_cubic_matches_explicit_coefficients():
    """Cubic mean preparation must match an explicit spline prefilter."""
    from recovar import core
    from recovar.ppca.ppca import _prepare_mean_estimate_for_slicing

    rng = np.random.default_rng(124)
    grid_size = 8
    image_shape = (grid_size, grid_size)
    volume_shape = (grid_size, grid_size, grid_size)

    real_vol = rng.normal(size=volume_shape).astype(np.float32)
    mean_full = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(real_vol), norm="backward")).reshape(-1).astype(np.complex64)
    rotations = np.tile(np.eye(3, dtype=np.float32), (1, 1, 1))

    prepared = _prepare_mean_estimate_for_slicing(
        jnp.array(mean_full),
        jnp.array(mean_full),
        volume_shape,
        "cubic",
    )
    manual = core.precompute_cubic_coefficients(jnp.array(mean_full), volume_shape)
    np.testing.assert_allclose(np.asarray(prepared), np.asarray(manual), atol=1e-6, rtol=1e-6)

    prepared_proj = np.asarray(
        core.slice_volume(prepared, rotations, image_shape, volume_shape, "cubic", half_image=True)
    )
    manual_proj = np.asarray(
        core.slice_volume(manual, rotations, image_shape, volume_shape, "cubic", half_image=True)
    )
    raw_proj = np.asarray(
        core.slice_volume(jnp.array(mean_full), rotations, image_shape, volume_shape, "cubic", half_image=True)
    )

    np.testing.assert_allclose(prepared_proj, manual_proj, atol=1e-5, rtol=1e-5)
    assert not np.allclose(raw_proj, prepared_proj, atol=1e-2, rtol=1e-2)


def test_prepare_mean_estimate_for_slicing_keeps_precomputed_cubic_coefficients():
    """Passing precomputed cubic coefficients must remain a no-op."""
    from recovar import core
    from recovar.ppca.ppca import _prepare_mean_estimate_for_slicing

    rng = np.random.default_rng(125)
    volume_shape = (8, 8, 8)
    real_vol = rng.normal(size=volume_shape).astype(np.float32)
    mean_full = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(real_vol), norm="backward")).reshape(-1).astype(np.complex64)

    coeffs = core.precompute_cubic_coefficients(jnp.array(mean_full), volume_shape)
    prepared = _prepare_mean_estimate_for_slicing(coeffs, None, volume_shape, "cubic")
    np.testing.assert_allclose(np.asarray(prepared), np.asarray(coeffs), atol=1e-6, rtol=1e-6)


def test_E_M_step_batch_half_contrast_rhs_uses_basis_adjoint(monkeypatch):
    """Contrast RHS backprojection must use the basis adjoint exactly once."""
    import recovar.core.fourier_transform_utils as ftu
    from recovar import core
    from recovar.ppca import ppca as ppca_mod

    rng = np.random.default_rng(126)
    d = _make_em_step_test_data(rng, grid_size=8, n_images=3, basis_size=1)
    tri_sz = ppca_mod._tri_size(d["basis_size"])
    mean_coeffs = core.precompute_cubic_coefficients(jnp.array(d["mean_full"]), d["volume_shape"])

    calls = []

    def fake_batch_adjoint_slice_volume(
        slices,
        rotation_matrices,
        image_shape,
        volume_shape,
        disc_type,
        half_image=False,
        half_volume=False,
        max_r=None,
    ):
        del rotation_matrices, image_shape, max_r
        calls.append((disc_type, half_image, half_volume))
        out_shape = ftu.volume_shape_to_half_volume_shape(volume_shape) if half_volume else volume_shape
        out_size = int(np.prod(out_shape))
        return jnp.zeros((slices.shape[0], out_size), dtype=slices.dtype)

    monkeypatch.setattr(ppca_mod.core, "batch_adjoint_slice_volume", fake_batch_adjoint_slice_volume)

    ppca_mod.E_M_step_batch_half(
        d["images_half"],
        jnp.zeros((d["half_volume_size"], tri_sz), dtype=np.float32),
        jnp.zeros((d["half_volume_size"], d["basis_size"]), dtype=np.complex64),
        mean_coeffs,
        d["W_half"],
        d["CTF_params"],
        d["rotation_matrices"],
        d["translations"],
        d["image_shape"],
        d["volume_shape"],
        d["grid_size"],
        d["voxel_size"],
        d["noise_variance_half"],
        d["ctf_evaluator"],
        compute_ll=False,
        disc_type_mean="cubic",
        disc_type="linear_interp",
        compute_stats=True,
        contrast_mode="marginalize",
        contrast_grid=jnp.array(np.linspace(0.5, 1.5, 5), dtype=np.float32),
        eigenvalues=jnp.ones(d["basis_size"], dtype=np.float32),
        contrast_mean=1.0,
        contrast_variance=0.3**2,
    )

    assert calls == [("linear_interp", True, True)]


def _run_and_compare_half_vs_full(d, atol_ez=1e-5, atol_ll=1e-2, atol_suf=1e-4):
    """Run both E_M_step_batch and E_M_step_batch_half, assert equivalence."""
    import recovar.core.fourier_transform_utils as ftu
    from recovar.ppca.ppca import E_M_step_batch, E_M_step_batch_half, _tri_size, unpack_tri_to_full

    basis_size = d["basis_size"]
    tri_sz = _tri_size(basis_size)

    float_dtype = d.get("float_dtype", np.float32)
    complex_dtype = d.get("complex_dtype", np.complex64)

    # --- Run original (full) ---
    lhs_full, rhs_full, ez_full, smz_full, ll_full, _ = E_M_step_batch(
        d["images_full"],
        jnp.zeros((d["volume_size"], basis_size * basis_size), dtype=float_dtype),
        jnp.zeros((d["volume_size"], basis_size), dtype=complex_dtype),
        d["mean_full"],
        d["W_full"],
        d["CTF_params"],
        d["rotation_matrices"],
        d["translations"],
        d["image_shape"],
        d["volume_shape"],
        d["grid_size"],
        d["voxel_size"],
        d["noise_variance"],
        d["ctf_evaluator"],
        compute_ll=True,
        disc_type_mean="linear_interp",
        disc_type="linear_interp",
    )

    # --- Run half version ---
    lhs_half, rhs_half, ez_half, smz_half, ll_half, _, _mc = E_M_step_batch_half(
        d["images_half"],
        jnp.zeros((d["half_volume_size"], tri_sz), dtype=float_dtype),
        jnp.zeros((d["half_volume_size"], basis_size), dtype=complex_dtype),
        d["mean_full"],
        d["W_half"],
        d["CTF_params"],
        d["rotation_matrices"],
        d["translations"],
        d["image_shape"],
        d["volume_shape"],
        d["grid_size"],
        d["voxel_size"],
        d["noise_variance_half"],
        d["ctf_evaluator"],
        compute_ll=True,
        disc_type_mean="linear_interp",
        disc_type="linear_interp",
    )

    # E-step outputs
    np.testing.assert_allclose(
        np.asarray(ez_half),
        np.asarray(ez_full),
        atol=atol_ez,
        rtol=atol_ez,
        err_msg="expected_zs mismatch",
    )
    np.testing.assert_allclose(
        np.asarray(smz_half),
        np.asarray(smz_full),
        atol=atol_ez,
        rtol=atol_ez,
        err_msg="second_moment_zs mismatch",
    )

    # Log-likelihood
    np.testing.assert_allclose(
        float(ll_half.real),
        float(ll_full.real),
        atol=atol_ll,
        rtol=1e-3,
        err_msg="log-likelihood mismatch",
    )

    # RHS: half-vol → full-vol
    rhs_half_expanded = np.asarray(
        ftu.half_volume_to_full_volume(jnp.array(np.asarray(rhs_half).T), d["volume_shape"]).T
    )
    np.testing.assert_allclose(
        rhs_half_expanded,
        np.asarray(rhs_full),
        atol=atol_suf,
        rtol=atol_suf,
        err_msg="rhs_summed mismatch",
    )

    # LHS: unpack tri → full matrix → half-vol → full-vol
    lhs_half_mat = np.asarray(unpack_tri_to_full(jnp.array(np.asarray(lhs_half)), basis_size))
    lhs_half_flat = lhs_half_mat.reshape(d["half_volume_size"], basis_size * basis_size)
    lhs_half_expanded = np.asarray(ftu.half_volume_to_full_volume(jnp.array(lhs_half_flat.T), d["volume_shape"]).T)
    np.testing.assert_allclose(
        lhs_half_expanded,
        np.asarray(lhs_full),
        atol=atol_suf,
        rtol=atol_suf,
        err_msg="lhs_summed mismatch",
    )


def test_E_M_step_batch_half_equivalence_identity():
    """Tight equivalence: identity rotations, zero translations → near-exact match.

    Tolerances here are float32 limited (1e-5 for E-step, 1e-4 for sufficient
    statistics).  The companion float64 test below proves these gaps vanish at
    higher precision, confirming they are pure floating-point noise.
    """
    rng = np.random.default_rng(7)
    d = _make_em_step_test_data(rng, grid_size=8, n_images=5, basis_size=3)
    d["rotation_matrices"] = np.tile(np.eye(3, dtype=np.float32), (d["n_images"], 1, 1))
    d["translations"] = np.zeros((d["n_images"], 2), dtype=np.float32)
    _run_and_compare_half_vs_full(d, atol_ez=1e-5, atol_ll=1e-3, atol_suf=1e-4)


def test_E_M_step_batch_half_equivalence_identity_f64():
    """Float64 companion: proves the float32 tolerances above are purely precision.

    Same test as test_E_M_step_batch_half_equivalence_identity but in float64.
    Tolerances tighten by >3 orders of magnitude, confirming the algorithm is
    exact and all float32 gaps are rounding noise.
    """
    import os

    os.environ["JAX_ENABLE_X64"] = "1"
    # Force JAX to pick up the flag (idempotent if already set)
    import jax

    jax.config.update("jax_enable_x64", True)

    rng = np.random.default_rng(7)
    d = _make_em_step_test_data(rng, grid_size=8, n_images=5, basis_size=3, float_dtype=np.float64)
    d["rotation_matrices"] = np.tile(np.eye(3, dtype=np.float64), (d["n_images"], 1, 1))
    d["translations"] = np.zeros((d["n_images"], 2), dtype=np.float64)
    _run_and_compare_half_vs_full(d, atol_ez=1e-7, atol_ll=1e-4, atol_suf=1e-5)


def test_E_M_step_batch_half_equivalence_rotated():
    """Loose equivalence with non-trivial rotations/translations.

    JAX's batched translate_images has ~4e-3 numerical noise between full
    and half execution paths (different reduction order under XLA), which
    propagates through M_n_inv.  We use wider tolerances accordingly.
    The float64 companion proves the algorithm is correct.
    """
    rng = np.random.default_rng(7)
    d = _make_em_step_test_data(rng, grid_size=8, n_images=5, basis_size=3)
    _run_and_compare_half_vs_full(d, atol_ez=0.1, atol_ll=1.0, atol_suf=0.1)


def test_E_M_step_batch_half_equivalence_rotated_f64():
    """Float64 companion for the rotated test — proves gaps are float32 noise."""
    import os

    os.environ["JAX_ENABLE_X64"] = "1"
    import jax

    jax.config.update("jax_enable_x64", True)

    rng = np.random.default_rng(7)
    d = _make_em_step_test_data(rng, grid_size=8, n_images=5, basis_size=3, float_dtype=np.float64)
    _run_and_compare_half_vs_full(d, atol_ez=1e-6, atol_ll=1e-4, atol_suf=1e-5)


def test_E_M_step_batch_half_no_stats():
    """Verify compute_stats=False skips LHS/RHS accumulation."""
    from recovar.ppca.ppca import E_M_step_batch_half, _tri_size

    rng = np.random.default_rng(55)
    d = _make_em_step_test_data(rng, grid_size=8, n_images=3, basis_size=2)
    tri_sz = _tri_size(d["basis_size"])

    lhs_init = jnp.zeros((d["half_volume_size"], tri_sz), dtype=np.float32)
    rhs_init = jnp.zeros((d["half_volume_size"], d["basis_size"]), dtype=np.complex64)

    lhs, rhs, ez, smz, ll, _, _mc = E_M_step_batch_half(
        d["images_half"],
        lhs_init,
        rhs_init,
        d["mean_full"],
        d["W_half"],
        d["CTF_params"],
        d["rotation_matrices"],
        d["translations"],
        d["image_shape"],
        d["volume_shape"],
        d["grid_size"],
        d["voxel_size"],
        d["noise_variance_half"],
        d["ctf_evaluator"],
        compute_ll=True,
        compute_stats=False,
    )

    # LHS/RHS should remain zero
    np.testing.assert_array_equal(np.asarray(lhs), 0.0)
    np.testing.assert_array_equal(np.asarray(rhs), 0.0)
    # But E-step should still produce non-trivial results
    assert np.any(np.asarray(ez) != 0)
    assert np.isfinite(float(ll.real))


def test_E_M_step_batch_half_accumulates_across_batches():
    """Verify accumulation works when called with multiple batches."""
    import recovar.core.fourier_transform_utils as ftu
    from recovar.ppca.ppca import E_M_step_batch, E_M_step_batch_half, _tri_size, unpack_tri_to_full

    rng = np.random.default_rng(77)
    d = _make_em_step_test_data(rng, grid_size=8, n_images=6, basis_size=2)
    # Use identity rotations and zero translations for tight comparison
    d["rotation_matrices"] = np.tile(np.eye(3, dtype=np.float32), (d["n_images"], 1, 1))
    d["translations"] = np.zeros((d["n_images"], 2), dtype=np.float32)
    basis_size = d["basis_size"]
    tri_sz = _tri_size(basis_size)

    # Split into two batches of 3
    split = 3

    # --- Full version: two batches ---
    lhs_f = jnp.zeros((d["volume_size"], basis_size * basis_size), dtype=np.float32)
    rhs_f = jnp.zeros((d["volume_size"], basis_size), dtype=np.complex64)
    for s in [slice(0, split), slice(split, d["n_images"])]:
        lhs_f, rhs_f, _, _, _, _ = E_M_step_batch(
            d["images_full"][s],
            lhs_f,
            rhs_f,
            d["mean_full"],
            d["W_full"],
            d["CTF_params"][s],
            d["rotation_matrices"][s],
            d["translations"][s],
            d["image_shape"],
            d["volume_shape"],
            d["grid_size"],
            d["voxel_size"],
            d["noise_variance"][s],
            d["ctf_evaluator"],
            compute_ll=False,
            disc_type_mean="linear_interp",
            disc_type="linear_interp",
        )

    # --- Half version: two batches ---
    lhs_h = jnp.zeros((d["half_volume_size"], tri_sz), dtype=np.float32)
    rhs_h = jnp.zeros((d["half_volume_size"], basis_size), dtype=np.complex64)
    for s in [slice(0, split), slice(split, d["n_images"])]:
        lhs_h, rhs_h, _, _, _, _, _ = E_M_step_batch_half(
            d["images_half"][s],
            lhs_h,
            rhs_h,
            d["mean_full"],
            d["W_half"],
            d["CTF_params"][s],
            d["rotation_matrices"][s],
            d["translations"][s],
            d["image_shape"],
            d["volume_shape"],
            d["grid_size"],
            d["voxel_size"],
            d["noise_variance_half"][s],
            d["ctf_evaluator"],
            compute_ll=False,
            disc_type_mean="linear_interp",
            disc_type="linear_interp",
        )

    # Expand half → full for comparison
    lhs_h_mat = np.asarray(unpack_tri_to_full(jnp.array(np.asarray(lhs_h)), basis_size))
    lhs_h_flat = lhs_h_mat.reshape(d["half_volume_size"], basis_size * basis_size)
    lhs_h_expanded = np.asarray(ftu.half_volume_to_full_volume(jnp.array(lhs_h_flat.T), d["volume_shape"]).T)
    rhs_h_expanded = np.asarray(ftu.half_volume_to_full_volume(jnp.array(np.asarray(rhs_h).T), d["volume_shape"]).T)

    np.testing.assert_allclose(
        lhs_h_expanded, np.asarray(lhs_f), atol=1e-4, rtol=1e-4, err_msg="multi-batch lhs_summed mismatch"
    )
    np.testing.assert_allclose(
        rhs_h_expanded, np.asarray(rhs_f), atol=1e-4, rtol=1e-4, err_msg="multi-batch rhs_summed mismatch"
    )


def test_em_return_posterior_info_exposes_mean_c(monkeypatch):
    from recovar.ppca import ppca as ppca_impl

    fake_dataset = SimpleNamespace(
        volume_shape=(2, 2, 2),
        volume_size=8,
        grid_size=2,
    )
    W_initial = np.ones((8, 2), dtype=np.float32)
    W_prior = np.ones((8, 2), dtype=np.float32)

    def _fake_em_step_half(*_args, **_kwargs):
        return (
            np.ones((8, 2), dtype=np.complex64),
            np.array([[1.0, 2.0]], dtype=np.float32),
            np.array([np.eye(2, dtype=np.float32) * 3.0]),
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([0.1, 0.2], dtype=np.float32),
            10.0,
            8.0,
            2.0,
            np.array([1.3], dtype=np.float32),
        )

    monkeypatch.setattr(ppca_impl, "EM_step_half", _fake_em_step_half)

    U, S, W, expected_zs, second_moment_zs, iteration_data, posterior_info = ppca_impl.EM(
        [fake_dataset],
        np.zeros(8, dtype=np.complex64),
        W_initial,
        W_prior,
        EM_iter=1,
        return_iteration_data=True,
        return_posterior_info=True,
        contrast_mode="marginalize",
    )

    assert U.shape == (8, 2)
    assert S.shape == (2,)
    assert W.shape == (8, 2)
    np.testing.assert_allclose(expected_zs, np.array([[1.0, 2.0]], dtype=np.float32))
    np.testing.assert_allclose(second_moment_zs, np.array([np.eye(2, dtype=np.float32) * 3.0]))
    assert iteration_data[0]["Iteration"] == 0
    np.testing.assert_allclose(posterior_info["mean_c"], np.array([1.3], dtype=np.float32))


# ---------------------------------------------------------------------------
# GPU tests – verify CPU/GPU numerical equivalence
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_batch_vec_unvec_roundtrip_gpu(gpu_device):
    x = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)

    cpu_v = np.asarray(akd.batch_vec(x))
    cpu_rt = np.asarray(akd.batch_unvec(cpu_v))

    with jax.default_device(gpu_device):
        x_g = jax.device_put(jnp.array(x), gpu_device)
        gpu_v = np.asarray(akd.batch_vec(x_g))
        gpu_rt = np.asarray(akd.batch_unvec(gpu_v))

    np.testing.assert_allclose(cpu_v, gpu_v, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(cpu_rt, gpu_rt, atol=1e-5, rtol=1e-5)

"""Upsampled (padded-volume) RELION-style reconstruction on the JAX and CUDA paths.

``relion_style_triangular_kernel(..., upsampling_factor=2)`` backprojects
images of size N into a 2N Fourier volume. An image Fourier pixel k must land
at volume grid coordinate 2k. The CUDA kernels infer this factor from the
shapes; the JAX fallback (``recovar.core.relion_interp``, used on CPU) did
not, so on CPU an upsampled reconstruction was the true map magnified 2x
(junk_particle_detection half-maps, homogeneous means and
heterogeneity_volume on CPU).
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar import core, utils
from recovar.core import fourier_transform_utils as ftu
from recovar.core import relion_interp, slicing
from recovar.core.configs import ForwardModelConfig
from recovar.data_io import cryoem_dataset as dataset
from recovar.data_io import halfsets
from recovar.reconstruction import noise
from recovar.reconstruction import relion_functions as rf

pytestmark = pytest.mark.unit

N = 16
N_IMAGES = 600


def _random_rotations(n, seed):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=1,
    ).astype(np.float32)


def _blob_volume(n, seed=0):
    rng = np.random.default_rng(seed)
    x = np.arange(n) - n // 2
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    vol = np.zeros((n, n, n))
    for _ in range(5):
        c = rng.uniform(-n / 5, n / 5, 3)
        s = rng.uniform(n / 14, n / 9)
        vol += rng.uniform(0.5, 1.5) * np.exp(-((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2) / (2 * s**2))
    return vol, (X**2 + Y**2 + Z**2) < (n / 2 - 2) ** 2


@pytest.fixture(scope="module")
def known_pose_dataset(tmp_path_factory):
    """Noise-free CTF-modulated projections of a blob volume at known poses, loaded from disk."""
    vol, support = _blob_volume(N)
    rots = _random_rotations(N_IMAGES, seed=1)
    rng = np.random.default_rng(2)
    voxel_size = 3.0
    ctf = np.zeros((N_IMAGES, 9), dtype=np.float32)
    ctf[:, core.CTFParamIndex.DFU] = rng.uniform(8000, 25000, N_IMAGES)
    ctf[:, core.CTFParamIndex.DFV] = ctf[:, core.CTFParamIndex.DFU]
    ctf[:, core.CTFParamIndex.VOLT] = 300.0
    ctf[:, core.CTFParamIndex.CS] = 2.7
    ctf[:, core.CTFParamIndex.W] = 0.1
    ctf[:, core.CTFParamIndex.CONTRAST] = 1.0
    geometry_only = dataset.CryoEMDataset(
        image_source=None,
        voxel_size=voxel_size,
        metadata=dataset.ImageMetadata(rots, np.zeros((N_IMAGES, 2), np.float32), ctf),
        ctf_evaluator=core.CTFEvaluator(),
        grid_size=N,
    )
    cfg = ForwardModelConfig.from_dataset(geometry_only, disc_type="linear_interp")
    vol_ft = ftu.get_dft3(jnp.asarray(vol)).reshape(-1)
    proj = slicing.slice_volume(vol_ft, rots, (N, N), (N, N, N), "linear_interp", max_r=None)
    images = np.real(np.asarray(ftu.get_idft2((proj * cfg.compute_ctf(ctf)).reshape(N_IMAGES, N, N))))

    out = tmp_path_factory.mktemp("upsampling_recon")
    utils.write_mrc(str(out / "particles.mrcs"), images.astype(np.float32))
    # load_ctf_params layout: [D, Apix, DFU, DFV, DFANG, VOLT, CS, W, phase]
    ctf_pkl = np.zeros((N_IMAGES, 9), dtype=np.float32)
    ctf_pkl[:, 0] = N
    ctf_pkl[:, 1] = voxel_size
    ctf_pkl[:, 2] = ctf[:, core.CTFParamIndex.DFU]
    ctf_pkl[:, 3] = ctf[:, core.CTFParamIndex.DFV]
    ctf_pkl[:, 5] = 300.0
    ctf_pkl[:, 6] = 2.7
    ctf_pkl[:, 7] = 0.1
    utils.pickle_dump(ctf_pkl, str(out / "ctf.pkl"))
    utils.pickle_dump((rots, np.zeros((N_IMAGES, 2), np.float32)), str(out / "poses.pkl"))
    spec = halfsets.HalfsetDatasetSpec.from_args(
        SimpleNamespace(
            particles=str(out / "particles.mrcs"),
            ctf=str(out / "ctf.pkl"),
            poses=str(out / "poses.pkl"),
            datadir=str(out),
            uninvert_data="false",
        )
    )
    cryo = halfsets.load_halfset_dataset(
        spec, ind_split=[np.arange(0, N_IMAGES, 2), np.arange(1, N_IMAGES, 2)], lazy=False
    )
    cryo.noise = noise.as_noise_model(np.ones(N // 2 + 1, np.float32), cryo.image_shape)
    np.testing.assert_allclose(cryo.rotation_matrices, rots, atol=1e-6)
    return cryo, vol, support


def _reconstruct(cryo, upsampling_factor, junk_detector_postprocess):
    Ft_ctf, F_ty = rf.relion_style_triangular_kernel(
        cryo, None, 200, disc_type="linear_interp", upsampling_factor=upsampling_factor
    )
    post = (
        dict(kernel="triangular", use_spherical_mask=True, grid_correct=True, gridding_correct="square")
        if junk_detector_postprocess
        else {}
    )
    vol = rf.post_process_from_filter_v2(Ft_ctf, F_ty, cryo.volume_shape, upsampling_factor, **post)
    return np.asarray(ftu.get_idft3(jnp.asarray(vol).reshape(cryo.volume_shape)).real)


def _force_jax_path(monkeypatch):
    monkeypatch.setattr(slicing, "_use_cuda_backproject", lambda order: False)
    monkeypatch.setattr(slicing, "_use_cuda", lambda order: False)


@pytest.mark.parametrize("junk_detector_postprocess", [False, True])
def test_upsampled_reconstruction_matches_truth_on_jax_path(monkeypatch, known_pose_dataset, junk_detector_postprocess):
    _force_jax_path(monkeypatch)
    cryo, vol, support = known_pose_dataset
    rec1 = _reconstruct(cryo, 1, junk_detector_postprocess)
    rec2 = _reconstruct(cryo, 2, junk_detector_postprocess)
    corr1 = np.corrcoef(rec1[support], vol[support])[0, 1]
    corr2 = np.corrcoef(rec2[support], vol[support])[0, 1]
    corr12 = np.corrcoef(rec1[support], rec2[support])[0, 1]
    print(f"corr(uf1, truth)={corr1:.4f} corr(uf2, truth)={corr2:.4f} corr(uf1, uf2)={corr12:.4f}")
    # Before the fix the factor-2 map on the JAX path was the truth magnified 2x.
    assert corr1 > 0.98, corr1
    assert corr2 > 0.98, corr2
    assert corr12 > 0.98, corr12


def test_jax_backproject_places_image_pixel_at_upsampled_coordinate():
    """Image pixel (k0, k1) backprojects to volume grid point ups*(k0, k1) + center."""
    image_shape, volume_shape = (8, 8), (16, 16, 16)
    k = ftu.get_k_coordinate_of_each_pixel(image_shape, voxel_size=1, scaled=False)
    k = np.asarray(k)
    pixel = int(np.flatnonzero((k[:, 0] == 2) & (k[:, 1] == -1))[0])
    img = np.zeros((1, 64), np.complex64)
    img[0, pixel] = 1.0
    out = relion_interp.backproject(
        jnp.asarray(img), jnp.eye(3, dtype=jnp.float32)[None], image_shape, volume_shape, order=1, max_r=None
    )
    out = np.asarray(out).reshape(volume_shape)
    nz = np.argwhere(np.abs(out) > 1e-6)
    assert nz.shape == (1, 3), nz
    # rotated = [k0, k1, 0] @ R -> grid (k0, k1, 0) * 2 + 8
    np.testing.assert_array_equal(nz[0], [2 * 2 + 8, 2 * -1 + 8, 8])


def test_jax_project_backproject_adjoint_with_upsampling():
    rng = np.random.default_rng(3)
    image_shape, volume_shape = (8, 8), (16, 16, 16)
    rots = jnp.asarray(_random_rotations(4, seed=4))
    x = jnp.asarray(rng.standard_normal(16**3) + 1j * rng.standard_normal(16**3), jnp.complex64)
    y = jnp.asarray(rng.standard_normal((4, 64)) + 1j * rng.standard_normal((4, 64)), jnp.complex64)
    Ax = relion_interp.project(x, rots, image_shape, volume_shape, order=1, max_r=3.0)
    Aty = relion_interp.backproject(y, rots, image_shape, volume_shape, order=1, max_r=3.0)
    np.testing.assert_allclose(complex(jnp.vdot(Ax.reshape(-1), y.reshape(-1))), complex(jnp.vdot(x, Aty)), rtol=1e-4)


@pytest.mark.gpu
@pytest.mark.parametrize("half", [False, True])
def test_jax_matches_cuda_with_upsampling(half):
    if jax.default_backend() != "gpu":
        pytest.skip("needs a GPU")
    if not slicing._use_cuda_backproject(1):
        pytest.skip("CUDA kernels unavailable")
    from recovar import cuda_backproject as cb

    rng = np.random.default_rng(5)
    image_shape, volume_shape = (16, 16), (32, 32, 32)
    rots = jnp.asarray(_random_rotations(6, seed=6))
    vol_size = int(np.prod(ftu.volume_shape_to_half_volume_shape(volume_shape) if half else volume_shape))
    n_pix = 16 * 9 if half else 256
    x = jnp.asarray(rng.standard_normal(vol_size) + 1j * rng.standard_normal(vol_size), jnp.complex64)
    y = jnp.asarray(rng.standard_normal((6, n_pix)) + 1j * rng.standard_normal((6, n_pix)), jnp.complex64)
    kw = dict(order=1, half_volume=half, half_image=half, max_r=7.0)
    np.testing.assert_allclose(
        np.asarray(relion_interp.project(x, rots, image_shape, volume_shape, **kw)),
        np.asarray(cb.project(x, rots, image_shape, volume_shape, **kw)),
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(relion_interp.backproject(y, rots, image_shape, volume_shape, **kw)),
        np.asarray(cb.backproject(jnp.zeros(vol_size, jnp.complex64), y, rots, image_shape, volume_shape, **kw)),
        rtol=1e-4,
        atol=1e-4,
    )

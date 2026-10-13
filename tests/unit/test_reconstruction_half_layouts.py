"""Reconstruction from half-spectrum layouts must equal reconstruction from full layouts.

A packed half volume (``half_volume=True``) and packed half images
(``half_image=True``) are storage choices.  Backprojecting the same Hermitian
images and reconstructing a map from the accumulators must give the same map in
every layout.  The reference is the full-volume / full-image layout, built with
``adjoint_slice_volume`` and no packing.

Three behaviours are pinned, each by its own test:

* ``test_layout_reconstruction_matches_full_with_prior``: map from accumulators
  built by ``adjoint_slice_volume`` in each ``(half_volume, half_image)`` layout and
  passed to ``post_process_from_filter_v2(input_half_volume=...)``, with a prior
  large enough that the floor is inactive.  This isolates the accumulator
  convention (the half-volume Hermitian fold gives interior ``kz`` twice the
  full-volume value, ``linalg.half_spectrum_last_axis_weights``).
* ``test_layout_reconstruction_matches_full_without_prior``: the same without a prior,
  where the floor can matter (sparse coverage at upsampling 2).
* ``test_relion_style_triangular_kernel_matches_full_layout_reconstruction``: the
  production accumulation entry point (half/half accumulation followed by its own
  expansion to a full volume) against the full-layout reference.

Each runs on the JAX backend (CPU or GPU) and, with a GPU, on the custom CUDA kernels.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import recovar  # noqa: F401  (enables x64)
import recovar.core.fourier_transform_utils as ftu
from recovar.core import slicing
from recovar.core.configs import ForwardModelConfig
from recovar.core.ctf import as_ctf_evaluator
from recovar.reconstruction import relion_functions

pytestmark = pytest.mark.unit

N = 16  # box size: small, but large enough that interior kz exists at every radius used
N_IMAGES = 64
RTOL = {np.float32: 1e-4, np.float64: 1e-10}


# ── Backend selection ────────────────────────────────────────────────


@pytest.fixture(params=["jax", pytest.param("cuda", marks=pytest.mark.gpu)])
def backend(request, monkeypatch):
    """Select the JAX-native or the custom-CUDA projection backend for one test."""
    import recovar.cuda_backproject as cuda_backproject

    if request.param == "jax":
        monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")
    else:
        custom_cuda_lib = request.getfixturevalue("custom_cuda_lib")
        monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
        monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
        monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
        assert slicing._use_cuda_backproject(1), "custom CUDA backproject was not selected"
    # Jitted accumulation helpers pick the backend at trace time: do not reuse the other backend's trace.
    jax.clear_caches()
    yield request.param
    jax.clear_caches()


# ── Synthetic data ───────────────────────────────────────────────────


def _rotations(n, rng):
    q, r = np.linalg.qr(rng.standard_normal((n, 3, 3)))
    q = q * np.sign(np.diagonal(r, axis1=1, axis2=2))[:, None, :]
    q[np.linalg.det(q) < 0, :, 0] *= -1
    return q


def _radial_ctf_evaluator(dtype):
    """Real, even, sign-changing CTF-like function of ``|k|``.

    Radial functions are exactly even, so the half-image grid carries the same
    values as the full grid and no Hermitian inconsistency enters the data.
    """

    def evaluate(params, image_shape, voxel_size, half_image=False, **_):
        h, w = image_shape
        ky = (np.arange(h) - h // 2) / h
        kx = (np.arange(w) - w // 2) / w
        r2 = jnp.asarray((ky[:, None] ** 2 + kx[None, :] ** 2).reshape(1, -1), dtype=dtype)
        ctf = jnp.cos(params[:, :1].astype(dtype) * r2) * jnp.exp(-3.0 * r2)
        return ftu.full_image_to_half_image(ctf, image_shape) if half_image else ctf

    return as_ctf_evaluator(evaluate)


class _Dataset:
    """The part of ``CryoEMDataset`` used by ``relion_style_triangular_kernel``.

    ``tests/helpers/tiny_synthetic`` poses are all the identity, which only touches the
    ``kz = 0`` plane and cannot expose a half-volume problem.
    """

    voxel_size = 1.0

    def __init__(self, images, rotations, ctf_params, ctf_evaluator):
        self.images, self.rotations, self.ctf_params, self.ctf_evaluator = images, rotations, ctf_params, ctf_evaluator
        self.image_shape = tuple(images.shape[-2:])
        self.grid_size = images.shape[-1]

    def iter_batches(self, batch_size, noise_model=None, indices=None, by_image=True):
        for start in range(0, self.images.shape[0], batch_size):
            stop = min(start + batch_size, self.images.shape[0])
            idx = np.arange(start, stop)
            yield (
                self.images[start:stop],
                self.rotations[start:stop],
                jnp.zeros((stop - start, 2), self.images.dtype),
                self.ctf_params[start:stop],
                jnp.asarray(1.0, self.images.dtype),
                idx,
                idx,
            )


class _Problem:
    """Real-space images, poses and the full-layout reference reconstruction inputs."""

    def __init__(self, dtype, upsampling, seed=0):
        rng = np.random.default_rng(seed)
        complex_dtype = np.complex128 if dtype == np.float64 else np.complex64
        self.dtype, self.upsampling = dtype, upsampling
        self.image_shape = (N, N)
        self.original_volume_shape = (N,) * 3
        self.volume_shape = (N * upsampling,) * 3
        self.rotations = jnp.asarray(_rotations(N_IMAGES, rng), dtype=dtype)
        images = jnp.asarray(rng.standard_normal((N_IMAGES, N, N)), dtype=dtype)
        ctf_params = jnp.asarray(rng.uniform(5.0, 30.0, size=(N_IMAGES, 1)), dtype=dtype)
        self.dataset = _Dataset(images, self.rotations, ctf_params, _radial_ctf_evaluator(dtype))
        config = ForwardModelConfig.from_dataset(self.dataset, upsampling_factor=upsampling)
        # Full-layout, noise-weighted data and CTF^2 terms (unit noise): what every layout accumulates.
        ctf = config.compute_ctf(ctf_params)
        self.weighted_images = (ctf * ftu.get_dft2(images).reshape(N_IMAGES, -1)).astype(complex_dtype)
        self.ctf_squared = (ctf**2).astype(complex_dtype)
        self.reference_ft_y = self.accumulate(self.weighted_images, half_volume=False, half_image=False)
        self.reference_ft_ctf = self.accumulate(self.ctf_squared, half_volume=False, half_image=False)

    def accumulate(self, full_images, *, half_volume, half_image):
        images = ftu.full_image_to_half_image(full_images, self.image_shape) if half_image else full_images
        return slicing.adjoint_slice_volume(
            images,
            self.rotations,
            self.image_shape,
            self.volume_shape,
            "linear_interp",
            half_image=half_image,
            half_volume=half_volume,
        )

    def prior(self):
        """Uniform ``tau`` such that ``1/tau`` is half the data weight inside the reconstructed sphere."""
        radius = np.sqrt(
            sum(
                g.astype(float) ** 2
                for g in np.meshgrid(*[np.arange(n) - n // 2 for n in self.volume_shape], indexing="ij")
            )
        )
        weight = np.asarray(self.reference_ft_ctf.real).reshape(self.volume_shape)[radius < N // 2 - 1].mean()
        return jnp.full(int(np.prod(self.original_volume_shape)), 2.0 / weight, dtype=self.dtype)

    def post_process(self, ft_ctf, ft_y, tau, *, half_volume):
        # No spherical mask or grid correction: they are layout-independent and would only dilute the comparison.
        return relion_functions.post_process_from_filter_v2(
            ft_ctf,
            ft_y,
            self.original_volume_shape,
            self.upsampling,
            tau=tau,
            input_half_volume=half_volume,
            return_real_space=True,
            use_spherical_mask=False,
            grid_correct=False,
        )

    def reference_map(self, tau):
        return self.post_process(self.reference_ft_ctf, self.reference_ft_y, tau, half_volume=False)


def _relative_error(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


_LAYOUTS = [
    pytest.param(False, True, id="full_volume-half_image"),
    pytest.param(True, False, id="half_volume-full_image"),
    pytest.param(True, True, id="half_volume-half_image"),
]


# ── Reconstruction through the projection backends ───────────────────


@pytest.mark.parametrize("upsampling", [1, 2])
@pytest.mark.parametrize("half_volume,half_image", _LAYOUTS)
def test_layout_reconstruction_matches_full_with_prior(backend, upsampling, half_volume, half_image):
    """Map from half-layout accumulators and a prior equals the full-layout map."""
    problem = _Problem(np.float64, upsampling)
    tau = problem.prior()
    ft_y = problem.accumulate(problem.weighted_images, half_volume=half_volume, half_image=half_image)
    ft_ctf = problem.accumulate(problem.ctf_squared, half_volume=half_volume, half_image=half_image)

    actual = problem.post_process(ft_ctf, ft_y, tau, half_volume=half_volume)

    assert _relative_error(actual, problem.reference_map(tau)) < RTOL[np.float64]


@pytest.mark.parametrize("upsampling", [1, 2])
@pytest.mark.parametrize("half_volume,half_image", _LAYOUTS)
def test_layout_reconstruction_matches_full_without_prior(backend, upsampling, half_volume, half_image):
    """Map from half-layout accumulators without a prior equals the full-layout map."""
    problem = _Problem(np.float64, upsampling)
    ft_y = problem.accumulate(problem.weighted_images, half_volume=half_volume, half_image=half_image)
    ft_ctf = problem.accumulate(problem.ctf_squared, half_volume=half_volume, half_image=half_image)

    actual = problem.post_process(ft_ctf, ft_y, None, half_volume=half_volume)

    assert _relative_error(actual, problem.reference_map(None)) < RTOL[np.float64]


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("upsampling", [1, 2])
def test_relion_style_triangular_kernel_matches_full_layout_reconstruction(backend, upsampling, dtype):
    """Production accumulation (half/half, expanded to a full volume) reconstructs the full-layout map.

    ``relion_style_triangular_kernel`` returns full-volume accumulators, so
    ``post_process_from_filter_v2`` sees a full layout here; the layouts differ only inside the
    accumulation and its expansion.  The float32 tolerance is limited by single precision; the
    float64 case checks the same comparison six orders tighter.
    """
    problem = _Problem(dtype, upsampling)
    tau = problem.prior()

    ft_ctf, ft_y = relion_functions.relion_style_triangular_kernel(
        problem.dataset, np.ones(N // 2 - 1, np.float32), 16, upsampling_factor=upsampling
    )
    actual = problem.post_process(ft_ctf, ft_y, tau, half_volume=False)

    assert _relative_error(actual, problem.reference_map(tau)) < RTOL[dtype]

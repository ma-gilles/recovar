"""P4-N: the projector backend is a choice of who computes, not of precision.

Two properties are load-bearing and neither was enforced before:

* each backend keeps its own output dtype unless a caller asks, so selecting
  the device path does not silently narrow refinement's complex128 slab to the
  complex64 the InitialModel consumer takes;
* the projector cache distinguishes the backends, so a comparison between them
  cannot be served one backend's slab twice and report perfect agreement.

The native binding is not built in every environment, so the tests that need
it skip rather than fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from recovar.em.refinement.refinement_options import RefinementOptions


def _build(**kwargs):
    from recovar.em.relion.relion_projector_setup import (
        reference_to_relion_projector_half_maps_and_power,
    )

    rng = np.random.default_rng(0)
    reference = rng.standard_normal((1, 32, 32, 32)).astype(np.float64)
    return reference_to_relion_projector_half_maps_and_power(
        reference, current_size=16, padding_factor=2, **kwargs
    )


def _native_available():
    try:
        _build(projector_setup_backend="native")
    except ImportError:
        return False
    return True


native_only = pytest.mark.skipif(
    not _native_available(), reason="the RELION native binding is not built here"
)


def test_refinement_defaults_to_the_device_backend():
    """The device path is the default now that it is qualified.

    Job 14208524 on the 10k/256 fixture: three interleaved arms on one H100,
    the device arm against two host controls. The slabs agree to 6.6e-16
    relative in double, the device arm's end-to-end FSC against RELION sits at
    control-to-control magnitude at every regime position (at order 4, where
    it shares a trajectory with one control, the two agree to 1.2e-5), and it
    removes 65 s of projector build from a 1240 s run. Pass "native" to
    reproduce a run from before that.
    """
    assert RefinementOptions().projector_setup_backend == "jax"


def test_the_host_backend_is_still_reachable():
    """The qualification is reversible by one name, not by editing source."""
    assert RefinementOptions(projector_setup_backend="native").projector_setup_backend == "native"


def test_the_option_accepts_only_the_two_backends():
    from recovar.em.relion.relion_projector_setup import (
        reference_to_relion_projector_half_maps_and_power as build,
    )

    rng = np.random.default_rng(0)
    reference = rng.standard_normal((1, 16, 16, 16)).astype(np.float64)
    with pytest.raises(ValueError, match="Unknown projector_setup_backend"):
        build(reference, current_size=8, padding_factor=2,
              projector_setup_backend="cufft")


def test_jax_backend_keeps_its_complex64_default():
    """The InitialModel consumer's behaviour must not move."""
    slab, _power, _r_max = _build(projector_setup_backend="jax")
    assert np.asarray(slab).dtype == np.complex64


@native_only
def test_native_backend_returns_complex128_by_default():
    slab, _power, _r_max = _build(projector_setup_backend="native")
    assert np.asarray(slab).dtype == np.complex128


@pytest.mark.parametrize("backend", ["native", "jax"])
def test_an_explicit_dtype_is_honoured_by_both_backends(backend):
    """Refinement asks for complex128 and must get it from either path."""
    if backend == "native" and not _native_available():
        pytest.skip("the RELION native binding is not built here")
    slab, _power, _r_max = _build(
        projector_setup_backend=backend, projector_data_dtype="complex128"
    )
    assert np.asarray(slab).dtype == np.complex128


@native_only
def test_the_two_backends_agree_to_double_precision_at_padding_two():
    """Not bitwise across FFT implementations, but within a few ulp.

    Padding factor 2 is what refinement uses. Both are asked for complex128,
    because comparing a double slab against a single one measures the cast
    rather than the transform.
    """
    native, native_power, native_r = _build(
        projector_setup_backend="native", projector_data_dtype="complex128"
    )
    jax_slab, jax_power, jax_r = _build(
        projector_setup_backend="jax", projector_data_dtype="complex128"
    )
    native, jax_slab = np.asarray(native), np.asarray(jax_slab)
    assert native.shape == jax_slab.shape
    assert int(native_r) == int(jax_r)
    scale = max(float(np.abs(native).max()), np.finfo(np.float64).tiny)
    assert float(np.abs(jax_slab - native).max()) / scale < 1e-12
    power_scale = max(float(np.abs(native_power).max()), np.finfo(np.float64).tiny)
    assert float(np.abs(np.asarray(jax_power) - np.asarray(native_power)).max()) / power_scale < 1e-11


@native_only
def test_the_projector_cache_distinguishes_the_backends(tmp_path, monkeypatch):
    """A v1 key would serve one backend's slab to the other.

    That would make any comparison between them report perfect agreement while
    measuring nothing, which is the failure this key change prevents.
    """
    from recovar.em.refinement.projector_preparation import (
        _relion_projector_half_maps_for_scoring,
    )

    monkeypatch.setenv("RECOVAR_RELION_PROJECTOR_CACHE_DIR", str(tmp_path))
    rng = np.random.default_rng(1)
    reference = rng.standard_normal((1, 32, 32, 32)).astype(np.float64)
    # means_k is validated even when real_references supplies the volume, so
    # pass a correctly shaped flat Fourier reference alongside it.
    flat_ft = np.zeros((1, 32 ** 3), dtype=np.complex128)
    built = {}
    for backend in ("native", "jax"):
        slab, _r_max = _relion_projector_half_maps_for_scoring(
            flat_ft,
            volume_shape=(32, 32, 32),
            current_size=16,
            padding_factor=2,
            n_classes=1,
            real_references=reference,
            projector_setup_backend=backend,
        )
        built[backend] = np.asarray(slab)

    cached = sorted(tmp_path.glob("projector_*.npz"))
    assert len(cached) == 2, f"expected one cache entry per backend, got {len(cached)}"
    # Same geometry and same reference, so the slabs agree closely; what the
    # test asserts is that each backend wrote its own entry rather than one
    # reading the other's.
    assert built["native"].shape == built["jax"].shape

"""post_process_from_filter_v2 with a traced logical current size inside a larger static class.

A caller that zero-pads the current-size accumulator to a larger stable class
and passes the class size as the static ``current_size`` and RELION's size as
``logical_current_size`` must get the reconstruction of the unpadded call: the
padded voxels lie outside every logical support rule, so only the program key
changes (relax's stable Fourier windows reuse one program per class).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.reconstruction import relion_functions

pytestmark = pytest.mark.unit


def _accumulators(rng, size):
    shape = (size,) * 3
    weight = rng.uniform(0.5, 2.0, size=shape)
    numerator = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    return weight, numerator


def _pad(values, pad):
    return np.pad(values, [(pad, pad)] * 3)


@pytest.mark.parametrize("logical, physical", [(10, 12), (12, 12)])
def test_padded_class_with_logical_current_size_matches_the_logical_call(logical, physical):
    rng = np.random.default_rng(7)
    vol_shape = (16, 16, 16)
    padding_factor = 2
    logical_size = padding_factor * logical + 3
    physical_size = padding_factor * physical + 3
    weight, numerator = _accumulators(rng, logical_size)
    tau = rng.uniform(0.5, 1.5, size=vol_shape[0] // 2 + 1)
    kwargs = dict(tau=tau, tau2_fudge=2.0, tau_is_1d=True, kernel="triangular", gridding_correct="radial")

    reference = relion_functions.post_process_from_filter_v2(
        jnp.asarray(weight.reshape(-1)),
        jnp.asarray(numerator.reshape(-1)),
        vol_shape,
        padding_factor,
        current_size=logical,
        accumulator_volume_shape=(logical_size,) * 3,
        **kwargs,
    )
    pad = (physical_size - logical_size) // 2
    stable = relion_functions.post_process_from_filter_v2(
        jnp.asarray(_pad(weight, pad).reshape(-1)),
        jnp.asarray(_pad(numerator, pad).reshape(-1)),
        vol_shape,
        padding_factor,
        current_size=physical,
        accumulator_volume_shape=(physical_size,) * 3,
        logical_current_size=jnp.int32(logical),
        **kwargs,
    )
    np.testing.assert_allclose(np.asarray(stable), np.asarray(reference), rtol=1e-10, atol=1e-10)


def test_logical_current_size_needs_a_static_bound():
    with pytest.raises(ValueError, match="needs a positive static current_size"):
        relion_functions.post_process_from_filter_v2(
            jnp.ones(27**3),
            jnp.ones(27**3, dtype=jnp.complex128),
            (16, 16, 16),
            2,
            accumulator_volume_shape=(27,) * 3,
            logical_current_size=jnp.int32(10),
        )

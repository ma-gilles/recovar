"""
JAX-based cubic spline interpolation for multi-dimensional arrays.

Provides cubic spline interpolation functionality with precomputed coefficients.

Uses **periodic boundary conditions** (circulant system solved via FFT), which
preserve Hermitian symmetry of the input: if ``V[k] = conj(V[-k])`` then
``C[k] = conj(C[-k])``.  Output shape equals input shape (no boundary padding).

The periodic circulant system ``[1, 4, 1]`` is solved as:
    C = ifft(fft(V) / eigenvalues)
where eigenvalues_k = 4 + 2*cos(2*pi*k/N).
"""

import functools
import logging
from typing import Sequence

import jax
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, ArrayLike

logger = logging.getLogger(__name__)


@jax.jit
def calculate_spline_coefficients(data: Array) -> Array:
    """Calculate periodic cubic spline coefficients for an array.

    Solves the periodic circulant system ``[1, 4, 1]`` along each dimension
    using FFT.  Output has the same shape as input (no boundary padding).

    Preserves Hermitian symmetry: if ``data[k] = conj(data[-k])`` then
    the coefficients satisfy the same property.

    Args:
        data: Input array (any number of dimensions).

    Returns:
        Spline coefficients with same shape as input.
    """
    result = data
    for axis in range(data.ndim):
        result = _solve_periodic_1d(result, axis)
    return result


def _solve_periodic_1d(data: Array, axis: int) -> Array:
    """Solve the periodic [1,4,1] circulant system along one axis via FFT."""
    N = data.shape[axis]
    k = jnp.arange(N)
    eigenvalues = 4.0 + 2.0 * jnp.cos(2.0 * jnp.pi * k / N)

    # Reshape eigenvalues for broadcasting along the target axis
    shape = [1] * data.ndim
    shape[axis] = N
    eigenvalues = eigenvalues.reshape(shape)

    is_complex = jnp.issubdtype(data.dtype, jnp.complexfloating)
    V_hat = jnp.fft.fft(data, axis=axis)
    C_hat = V_hat / eigenvalues
    result = jnp.fft.ifft(C_hat, axis=axis)
    if is_complex:
        return result.astype(data.dtype)
    else:
        return result.real.astype(data.dtype)


def _cubic_basis(t: Array) -> Array:
    """Evaluate piecewise cubic spline basis function.

    The basis function is non-zero only for |t| <= 2.
    """
    abs_t = jnp.abs(t)

    # Define the two pieces of the basis
    piece_far = lambda t: (2.0 - t) ** 3  # For 1 <= |t| <= 2
    piece_near = lambda t: 4.0 - 6.0 * t**2 + 3.0 * t**3  # For |t| <= 1

    # Combine with conditional logic
    return jnp.where(
        abs_t >= 1.0,
        jnp.where(abs_t <= 2.0, piece_far(abs_t), 0.0),
        jnp.where(abs_t <= 1.0, piece_near(abs_t), 0.0),
    )


def _eval_spline_point_wrap(coeffs: Array, coord: Array) -> Array:
    """Evaluate periodic cubic spline at a single coordinate point.

    Uses wrap (periodic) boundary for index access. For each dimension,
    uses the 4 nearest coefficients with indices wrapped modulo N.
    """
    ndim = coeffs.ndim
    shape = jnp.array(coeffs.shape)

    # Get 4-point stencil base indices
    bases = jnp.floor(coord).astype(jnp.int32)  # (ndim,)

    # Build all 4^ndim index combinations using meshgrid
    offsets = jnp.arange(4)  # [0, 1, 2, 3]
    stencils = [bases[d] + offsets for d in range(ndim)]
    grids = jnp.meshgrid(*stencils, indexing="ij")
    # grids is a list of ndim arrays each of shape (4,)*ndim
    flat_indices = jnp.stack([g.ravel() for g in grids], axis=-1)  # (4^ndim, ndim)

    # Wrap indices for periodic BC
    wrapped = flat_indices % shape[None, :]  # (4^ndim, ndim)

    # Compute basis weights for each stencil point
    def weight_one(idx_flat):
        """Weight = product of _cubic_basis(coord[d] - idx[d] + 1) over dims."""
        return jnp.prod(vmap(_cubic_basis)(coord - idx_flat.astype(coord.dtype) + 1.0))

    weights = vmap(weight_one)(flat_indices.astype(coord.dtype))  # (4^ndim,)

    # Gather coefficient values with wrapped indices
    def gather_one(widx):
        return coeffs[tuple(widx)]

    values = vmap(gather_one)(wrapped)  # (4^ndim,)

    return jnp.sum(weights * values)


@functools.partial(jax.jit, static_argnums=(2, 3))
def interpolate_with_spline(
    spline_coeffs: ArrayLike,
    coords: Sequence[ArrayLike],
    boundary_mode: str = "wrap",
    fill_value: ArrayLike = 0.0,
) -> Array:
    """Interpolate using cubic spline coefficients.

    Args:
        spline_coeffs: Precomputed spline coefficients from calculate_spline_coefficients
        coords: Sequence of coordinate arrays, one per dimension
        boundary_mode: Boundary handling mode ('wrap' for periodic)
        fill_value: Fill value for out-of-bounds when boundary_mode='fill'

    Returns:
        Interpolated values at the specified coordinates
    """
    coeff_array = jnp.asarray(spline_coeffs)

    # Validate dimensions
    if len(coords) != coeff_array.ndim:
        raise ValueError(
            f"Number of coordinate arrays ({len(coords)}) must match "
            f"coefficient dimensions ({coeff_array.ndim})"
        )

    # Stack and flatten coordinates for vectorized processing
    coord_stack = jnp.stack([jnp.asarray(c) for c in coords], axis=0)
    flat_coords = coord_stack.reshape(coeff_array.ndim, -1).T

    if boundary_mode == "wrap":
        evaluator = lambda coord: _eval_spline_point_wrap(coeff_array, coord)
    else:
        evaluator = lambda coord: _eval_spline_point_fill(
            coeff_array, coord, fill_value
        )

    flat_result = vmap(evaluator)(flat_coords)

    # Reshape to match input coordinate shape
    return flat_result.reshape(coord_stack.shape[1:])


def _eval_spline_point_fill(coeffs: Array, coord: Array, fill_value: ArrayLike) -> Array:
    """Evaluate cubic spline at a single point with fill boundary mode."""
    ndim = coeffs.ndim
    shape = jnp.array(coeffs.shape)
    bases = jnp.floor(coord).astype(jnp.int32)

    offsets = jnp.arange(4)
    stencils = [bases[d] + offsets for d in range(ndim)]
    grids = jnp.meshgrid(*stencils, indexing="ij")
    flat_indices = jnp.stack([g.ravel() for g in grids], axis=-1)

    def weight_one(idx_flat):
        return jnp.prod(vmap(_cubic_basis)(coord - idx_flat.astype(coord.dtype) + 1.0))

    weights = vmap(weight_one)(flat_indices.astype(coord.dtype))

    def gather_one(idx):
        in_bounds = jnp.all((idx >= 0) & (idx < shape))
        safe_idx = jnp.clip(idx, 0, shape - 1)
        val = coeffs[tuple(safe_idx)]
        return jnp.where(in_bounds, val, fill_value)

    values = vmap(gather_one)(flat_indices)
    return jnp.sum(weights * values)


# =============================================================================
# Compatibility API
# =============================================================================

def map_coordinates(input, coordinates, order, mode="wrap", cval=0.0):
    """Cubic spline interpolation compatible with scipy/cryojax API.

    Only supports order=3 (cubic spline interpolation).
    Coordinates are shifted by -1 for periodic convention.

    Args:
        input: Input array to interpolate
        coordinates: Sequence of coordinate arrays
        order: Interpolation order (must be 3)
        mode: Boundary mode (default 'wrap' for periodic)
        cval: Fill value for out-of-bounds

    Returns:
        Interpolated values
    """
    if order != 3:
        raise NotImplementedError(
            f"This implementation only supports cubic splines (order=3), got order={order}"
        )

    coeffs = calculate_spline_coefficients(input)
    # Use map_coordinates_with_cubic_spline for consistent coord handling
    return map_coordinates_with_cubic_spline(coeffs, coordinates, mode, cval)


def map_coordinates_with_cubic_spline(
    coefficients, coordinates, mode="wrap", cval=0.0
):
    """Interpolate using precomputed cubic spline coefficients.

    Coordinates are shifted by -1 for periodic convention.

    Args:
        coefficients: Precomputed spline coefficients (ndim-dimensional).
        coordinates: Either a sequence of ndim coordinate arrays, or a single
            array of shape ``(ndim, ...)`` (scipy convention).
        mode: Boundary mode (default 'wrap' for periodic).
        cval: Fill value for out-of-bounds.
    """
    coefficients = jnp.asarray(coefficients)
    ndim = coefficients.ndim
    # Normalize coordinates to list-of-arrays
    if isinstance(coordinates, jnp.ndarray) or (hasattr(coordinates, 'shape') and hasattr(coordinates, 'ndim')):
        coordinates = jnp.asarray(coordinates)
        if coordinates.ndim >= 2 and coordinates.shape[0] == ndim:
            coords_list = [coordinates[i] - 1 for i in range(ndim)]
        else:
            coords_list = [c - 1 for c in coordinates]
    else:
        coords_list = [jnp.asarray(c) - 1 for c in coordinates]
    return interpolate_with_spline(coefficients, coords_list, mode, cval)

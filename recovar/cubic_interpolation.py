"""
JAX-based cubic spline interpolation for multi-dimensional arrays.

Provides cubic spline interpolation functionality with precomputed coefficients.

Reimplementation of the cubic spline interpolation from cryojax.
"""

import functools
from typing import Sequence

import jax
import jax.numpy as jnp
import lineax as lx
from jax import vmap
from jaxtyping import Array, ArrayLike


@jax.jit
def calculate_spline_coefficients(data: Array) -> Array:
    """Calculate cubic spline coefficients for an array.
    
    Solves the spline system along each dimension to get coefficients
    for smooth cubic interpolation.
    
    Args:
        data: Input array
        
    Returns:
        Spline coefficients with same shape as input
    """
    ndim = data.ndim
    result = data
    
    # Process each dimension from last to first
    for dim_idx in range(ndim):
        axis = ndim - dim_idx - 1
        size = result.shape[axis]
        
        # Build tridiagonal system for this dimension
        linear_op = _create_tridiagonal_system(size - 2, result.dtype)
        
        # Create solver function for this dimension
        solver = lambda x: _solve_spline_system(x, linear_op)
        
        # Apply solver along the appropriate axes using vmap
        for vmap_idx in range(ndim - 2, -1, -1):
            axis_adjust = int(vmap_idx >= axis)
            solver = vmap(solver, axis_adjust, axis_adjust)
        
        result = solver(result)
    
    return result


@functools.partial(jax.jit, static_argnums=(2, 3))
def interpolate_with_spline(
    spline_coeffs: ArrayLike,
    coords: Sequence[ArrayLike],
    boundary_mode: str = "fill",
    fill_value: ArrayLike = 0.0,
) -> Array:
    """Interpolate using cubic spline coefficients.
    
    Args:
        spline_coeffs: Precomputed spline coefficients from calculate_spline_coefficients
        coords: Sequence of coordinate arrays, one per dimension
        boundary_mode: Boundary handling mode ('fill', 'wrap', 'clamp', etc.)
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
    
    # Evaluate spline at each coordinate point
    evaluator = lambda coord: _eval_spline_point(
        coeff_array, coord, boundary_mode, fill_value
    )
    
    flat_result = vmap(evaluator)(flat_coords)
    
    # Reshape to match input coordinate shape
    return flat_result.reshape(coord_stack.shape[1:])


def _create_tridiagonal_system(
    size: int,
    dtype,
    diagonal_value: float = 4.0
) -> lx.TridiagonalLinearOperator:
    """Create tridiagonal linear operator for spline system.
    
    The standard cubic spline system has 4 on the main diagonal
    and 1 on the off-diagonals.
    """
    main_diagonal = jnp.full((size,), diagonal_value, dtype=dtype)
    off_diagonal = jnp.full((size - 1,), 1.0, dtype=dtype)
    
    return lx.TridiagonalLinearOperator(main_diagonal, off_diagonal, off_diagonal)


def _build_rhs_vector(data: Array, boundary_c2: Array, boundary_cnm2: Array) -> Array:
    """Construct right-hand side vector for spline system.
    
    Adjusts the first and last elements to account for boundary conditions.
    """
    # Start with interior data points
    rhs = data[1:-1]
    
    # Adjust boundary entries
    rhs = rhs.at[0].set(data[1] - boundary_c2)
    rhs = rhs.at[-1].set(data[-2] - boundary_cnm2)
    
    return rhs


def _solve_spline_system(
    data: Array,
    operator: lx.TridiagonalLinearOperator,
) -> Array:
    """Solve for spline coefficients along one dimension.
    
    Uses natural boundary conditions (second derivative = 0 at endpoints).
    """
    # Boundary coefficients from natural spline conditions
    c_2 = data[0] / 6.0
    c_nm2 = data[-1] / 6.0
    
    # Build and solve system for interior coefficients
    rhs = _build_rhs_vector(data, c_2, c_nm2)
    solution = lx.linear_solve(operator, rhs)
    interior_c = solution.value
    
    # Extrapolate boundary coefficients
    c_1 = 2.0 * c_2 - interior_c[0]
    c_nm1 = 2.0 * c_nm2 - interior_c[-1]
    
    # Assemble complete coefficient array
    return jnp.concatenate([
        jnp.array([c_1, c_2]),
        interior_c,
        jnp.array([c_nm2, c_nm1])
    ])


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


def _eval_coeff_contribution(
    coeffs: Array,
    coord: Array,
    index: Array,
    boundary_mode: str,
    fill_value: ArrayLike,
) -> Array:
    """Evaluate contribution from one coefficient in the spline sum."""
    # Get coefficient at this index
    c_value = coeffs.at[tuple(index)].get(
        mode=boundary_mode, fill_value=fill_value
    )
    
    # Evaluate basis function for each dimension
    basis_fn = vmap(
        lambda x, i: _cubic_basis(x - i + 1.0),
        (0, 0)
    )
    basis_product = basis_fn(coord, index).prod()
    
    return c_value * basis_product


def _eval_spline_point(
    coeffs: Array,
    coord: Array,
    boundary_mode: str,
    fill_value: ArrayLike,
) -> Array:
    """Evaluate cubic spline at a single coordinate point.
    
    For each dimension, uses the 4 nearest coefficients.
    """
    # Get 4-point stencil indices for each dimension
    def get_stencil(x):
        base = jnp.floor(x)
        return (jnp.arange(4) + base).astype(int)
    
    stencil_indices = vmap(get_stencil)(coord)
    
    # Create meshgrid of all coefficient indices to use
    index_mesh = jnp.array(jnp.meshgrid(*stencil_indices, indexing="ij"))
    flat_indices = index_mesh.reshape(coeffs.ndim, -1).T
    
    # Sum contributions from all coefficients in the stencil
    contrib_fn = lambda idx: _eval_coeff_contribution(
        coeffs, coord, idx, boundary_mode, fill_value
    )
    
    return vmap(contrib_fn)(flat_indices).sum()


# =============================================================================
# Compatibility API
# =============================================================================

def map_coordinates(input, coordinates, order, mode="fill", cval=0.0):
    """Cubic spline interpolation compatible with scipy/cryojax API.
    
    Only supports order=3 (cubic spline interpolation).
    
    Args:
        input: Input array to interpolate
        coordinates: Sequence of coordinate arrays
        order: Interpolation order (must be 3)
        mode: Boundary mode ('fill', 'wrap', 'clamp', etc.)
        cval: Fill value for out-of-bounds
        
    Returns:
        Interpolated values
    """
    if order != 3:
        raise NotImplementedError(
            f"This implementation only supports cubic splines (order=3), got order={order}"
        )
    
    coeffs = calculate_spline_coefficients(input)
    return interpolate_with_spline(coeffs, coordinates, mode, cval)


def map_coordinates_with_cubic_spline(
    coefficients, coordinates, mode="fill", cval=0.0
):
    """Interpolate using precomputed cubic spline coefficients.
    
    Compatible with cryojax API.
    """
    return interpolate_with_spline(coefficients, coordinates, mode, cval)


# Alias for external compatibility
compute_spline_coefficients = calculate_spline_coefficients

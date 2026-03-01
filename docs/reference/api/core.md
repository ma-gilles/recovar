# recovar.core

Low-level primitives for cryo-EM forward modeling, geometry, CTF handling,
and Fourier-space operations.

## configs

Equinox-based configuration containers for the forward model, batch data,
and model state.

::: recovar.core.configs
    options:
      members_order: source

## ctf

CTF evaluation and parameter handling.

::: recovar.core.ctf
    options:
      members_order: source

## geometry

Image translation, rotation, and coordinate transforms.

::: recovar.core.geometry
    options:
      members_order: source

## forward

Forward model and adjoint operations for cryo-EM image formation.

::: recovar.core.forward
    options:
      members_order: source

## slicing

Fourier slice extraction and backprojection.

::: recovar.core.slicing
    options:
      members_order: source

## indexing

Frequency-to-volume index mapping and radial shell utilities.

::: recovar.core.indexing
    options:
      members_order: source

## mask

Real-space and Fourier-space mask generation and manipulation.

::: recovar.core.mask
    options:
      members_order: source

## linalg

Linear algebra helpers (batch SVD, QR, eigendecomposition).

::: recovar.core.linalg
    options:
      members_order: source

## fourier_transform_utils

FFT utilities and frequency grid construction.

::: recovar.core.fourier_transform_utils
    options:
      members_order: source

## cubic_interpolation

Cubic spline interpolation on 3-D grids.

::: recovar.core.cubic_interpolation
    options:
      members_order: source

## padding

Real-space zero-padding and unpadding for Fourier oversampling.

::: recovar.core.padding
    options:
      members_order: source

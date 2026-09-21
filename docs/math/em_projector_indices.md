# Compact RELION projector indices

Implemented by [`_validate_centered_relion_projector_pixel_indices`](../../recovar/em/helpers/projection.py), with regression coverage in
[`test_refine_relion_mode.py`](../../tests/unit/test_refine_relion_mode.py) and
[`test_cuda_relion_fine_diff2.py`](../../tests/unit/test_cuda_relion_fine_diff2.py).

For an even image box of width N, the centered packed half-image has N rows
and H = N/2 + 1 columns. A flat index i must satisfy 0 <= i < N*H.
Its row is i // H and its column is i % H.

A projector crop of even width C accepts columns 0 through C/2 and vertical
frequencies -(C/2 - 1) through +C/2. Ordinarily the vertical frequency is
row - N/2. When C equals N, centered row zero is the stored Nyquist row and
uses the RELION label +N/2. When C is smaller than N, that row lies outside
the crop. RELION 5.0.1 uses the positive label in its uncropped half-image
iteration (`src/fftw.h`, FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM); its Fourier
window excludes the negative crop Nyquist row.

This validation matches the existing texture gather and full scatter. It
changes neither the packed storage layout nor the translation convention.

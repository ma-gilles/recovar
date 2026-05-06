"""Denovo-init Iref seeding via random-orient backprojection.

Mirrors the `fn_ref == "None"` branch of
`MlOptimiser::calculateSumOfPowerSpectraAndAverageImage`
(ml_optimiser.cpp:3127-3205) followed by the
`wsum_model.BPref[iclass].reconstruct` call inside
`setSigmaNoiseEstimatesAndSetAverageImage` (ml_optimiser.cpp:3259-3265)
and the subsequent `initialLowPassFilterReferences`
(ml_optimiser.cpp:3336-3372).

For each particle in `sorted_idx` order:

  1. `init_random_generator(random_seed + part_id)` — per-particle RNG reset
  2. `rot = rnd_unif() * 360`
  3. `tilt = rnd_unif() * 180`
  4. `psi = rnd_unif() * 360`
  5. `A = Euler_angles2matrix(rot, tilt, psi, homogeneous=false)`
  6. `iclass = part_id_sorted % nr_classes`
  7. apply softMaskOutsideMap (when do_zero_mask)
  8. FFT(img) with RELION's "normalize=true" forward (divide by H*W)
  9. CenterFFTbySign(Faux)  -> sign flip (-1)^(ky+kx+kz)
 10. windowFourierTransform to current_size (pad=1 at box 64 + ini_high path
     gives current_size = 136 which clamps to ori_size=64, so no-op)
 11. Compute `Fctf` via CTF::getFftwImage with particle's metadata
 12. `Fimg *= Fctf`, `Fctf² = Fctf * Fctf`
 13. `BPref[iclass].set2DFourierTransform(Fimg, A, &Fctf²)`

Then:
  - `BPref[iclass].reconstruct(Iref[iclass], 10, false, dummy_tau2)` (RELION
    calls this with `do_map = false` so no Wiener prior).
  - `initialLowPassFilterReferences` applies a raised-cosine low-pass
    filter at `radius = ori_size * pixel_size / ini_high` pixels,
    tapered over `WIDTH_FMASK_EDGE = 2` shells.

Parity target: `run_it000_class001.mrc` on the 64px fixture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# RELION's fixed low-pass edge width (ml_optimiser.h:91)
WIDTH_FMASK_EDGE: float = 2.0


def _euler_angles_to_matrix(alpha_deg: float, beta_deg: float, gamma_deg: float) -> np.ndarray:
    """Reproduce RELION's `Euler_angles2matrix(alpha, beta, gamma, A, homogeneous=false)`
    (euler.cpp). Returns a 3×3 rotation matrix.
    """
    alpha = np.deg2rad(alpha_deg)
    beta = np.deg2rad(beta_deg)
    gamma = np.deg2rad(gamma_deg)

    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    cc = cb * ca
    cs = cb * sa
    sc = sb * ca
    ss = sb * sa

    A = np.array(
        [
            [cg * cc - sg * sa, cg * cs + sg * ca, -cg * sb],
            [-sg * cc - cg * sa, -sg * cs + cg * ca, sg * sb],
            [sc, ss, cb],
        ],
        dtype=np.float64,
    )
    return A


def _center_fft_by_sign_2d_half(fourier: np.ndarray) -> np.ndarray:
    """Apply RELION's CenterFFTbySign to a 2D half-complex rfft2 output.

    The sign flip is (-1)^((ky + kx) & 1) where ky is the **signed**
    frequency index and kx is the rfft column index.

    For numpy rfft2 output of shape (H, W//2 + 1):
      - kx = [0, 1, ..., W/2]
      - ky = fftfreq(H) * H = [0, 1, ..., H/2-1, -H/2, -H/2+1, ..., -1]
        which as a signed int has parity = i for i in [0, H/2-1] and
        parity = (i - H) for i in [H/2, H-1]. (i - H) & 1 == i & 1 since
        H is even. So parity only depends on the unsigned index i & 1.
    """
    H, W_h = fourier.shape[-2:]
    ky_idx = np.arange(H) & 1
    kx_idx = np.arange(W_h) & 1
    parity = (ky_idx[:, None] ^ kx_idx[None, :]).astype(np.int8)
    sign = np.where(parity, -1.0, 1.0).astype(fourier.dtype)
    return fourier * sign


def _window_fourier_transform_half(Faux: np.ndarray, new_size: int) -> np.ndarray:
    """Crop / pad a half-complex 2D Fourier image to `new_size`.

    Matches RELION's `windowFourierTransform(in, out, new_size)` for 2D
    when the input size already equals new_size (no-op) or when new_size
    > ori_size (no-op via clamping to input). For real cropping, the low
    rows + high rows are kept as in recovar.em.initial_model.e_step.fourier_crop_half.
    """
    H, W_h = Faux.shape[-2:]
    ori_size = H  # assume square
    target_x = new_size // 2 + 1
    if new_size == ori_size:
        return Faux
    if new_size > ori_size:
        # RELION pads; but the bootstrap path calls with current_size >=
        # ori_size so this is effectively a no-op in our path.
        return Faux
    half = new_size // 2
    out = np.zeros((new_size, target_x), dtype=Faux.dtype)
    out[:half, :target_x] = Faux[:half, :target_x]
    out[half:, :target_x] = Faux[ori_size - (new_size - half) :, :target_x]
    return out


def _soft_mask_image_fixture(
    image: np.ndarray,
    radius: float,
    cosine_width: float,
) -> np.ndarray:
    """Thin re-wrapper of avg_unaligned._softmask_outside_map.

    Kept separate so the bootstrap module can be imported without pulling
    in the avg-unaligned test surface.
    """
    from .avg_unaligned import _softmask_outside_map

    return _softmask_outside_map(image, radius, cosine_width)


@dataclass
class ParticleCTF:
    """Per-particle CTF metadata, matching RELION's `_rlnDefocusU` / V / Angle /
    PhaseShift and the optics-group-level voltage/Cs/Q0/angpix/size.
    """

    defU: float  # Å
    defV: float  # Å
    defAngle: float  # degrees
    phase_shift: float = 0.0  # degrees

    # Optics-group params (same for all particles in one group)
    voltage: float = 300.0  # kV
    Cs: float = 2.7  # mm
    Q0: float = 0.07  # amplitude contrast
    angpix: float = 8.5  # Å
    ori_size: int = 64


def reorder_particles_relion_style(
    main_star,
    images: np.ndarray,
    defU: np.ndarray,
    defV: np.ndarray,
    defAngle: np.ndarray,
    phase_shift: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reorder particles to match RELION's internal processing order.

    RELION's `Experiment::read` sorts particles by `_rlnMicrographName`
    lexicographically (stable sort on string). For single-optics-group
    datasets, the first 1000 particles RELION processes for the bootstrap
    are the first 1000 in that sorted order. If recovar feeds particles
    in STAR row order instead, the first 1000 are a DIFFERENT subset,
    breaking iter-0 bootstrap parity for N > 1000.

    Returns reordered (images, defU, defV, defAngle, phase_shift) with
    the stack re-indexed by the image-name's stack frame.
    """
    img_names = main_star["_rlnImageName"].tolist()
    mic_names = main_star["_rlnMicrographName"].tolist()
    order = sorted(range(len(mic_names)), key=lambda i: mic_names[i])
    frame_ids = [int(img_names[i].split("@")[0]) - 1 for i in order]
    return (
        np.ascontiguousarray(images[frame_ids]),
        np.ascontiguousarray(defU[order]),
        np.ascontiguousarray(defV[order]),
        np.ascontiguousarray(defAngle[order]),
        np.ascontiguousarray(phase_shift[order]),
    )


def compute_bootstrap_iref_via_cpp(
    *,
    images: np.ndarray,  # (N, H, W) real-space particles in RELION order
    defU: np.ndarray,
    defV: np.ndarray,
    defAngle: np.ndarray,
    phase_shift: np.ndarray,
    voltage: float,
    Cs: float,
    Q0: float,
    pixel_size: float,
    ori_size: int,
    nr_classes: int,
    particle_diameter_ang: float,
    width_mask_edge_px: float,
    do_zero_mask: bool,
    do_ctf_correction: bool,
    random_seed: int,
    padding_factor: int = 1,
    current_size: int = -1,
    minimum_nr_particles: int = 1000,
) -> np.ndarray:
    """Run the entire RELION InitialModel bootstrap in C++ (recommended).

    Calls `vdam_bootstrap_iref` which reproduces ml_optimiser.cpp:3127-3205
    + the reconstruct call at :3265 using real RELION C++ classes
    (softMaskOutsideMap, FourierTransformer, CenterFFTbySign,
    windowFourierTransform, CTF::getFftwImage, BackProjector).

    RELION GUI InitialModel uses ``--pad 1`` for this bootstrap path. Fresh
    same-build RELION dumps match at |CC| > 0.998 with ``padding_factor=1``
    and the RELION bootstrap current size ``round(0.07 * ori_size)``.

    Returns an Iref array of shape `(nr_classes, ori_size, ori_size, ori_size)`
    in recovar's real-space frame, suitable for ``InitialModelState.Iref`` and
    ``write_relion_mrc``. The C++ binding returns RELION's raw BackProjector
    frame; this wrapper applies the established ``relion_volume_to_recovar``
    axis/sign conversion.
    """
    from recovar.relion_bind import _relion_bind_core as bind
    from recovar.utils.helpers import relion_volume_to_recovar

    if current_size <= 0:
        # RELION's wsum_model.current_size for bootstrap goes through
        # ml_optimiser.cpp:2941 `getPixelFromResolution(1./ini_high)` when
        # ini_high > 0, and ini_high is set to the 0.07-digital-freq Å
        # value at line 2518 for do_average_unaligned. Net result:
        #   current_size = ROUND(0.07 * ori_size) (the SHELL count, not Å)
        # which on the 64-px fixture is 4 (NOT 136). RELION dumps confirm
        # r_max=2, pad_size=7, skip_gridding=1 for the BPref accumulator.
        current_size = int(np.floor(0.07 * ori_size + 0.5))

    iref_relion = np.asarray(
        bind.vdam_bootstrap_iref(
            np.ascontiguousarray(images.astype(np.float64)),
            np.ascontiguousarray(defU.astype(np.float64)),
            np.ascontiguousarray(defV.astype(np.float64)),
            np.ascontiguousarray(defAngle.astype(np.float64)),
            np.ascontiguousarray(phase_shift.astype(np.float64)),
            voltage,
            Cs,
            Q0,
            pixel_size,
            ori_size,
            nr_classes,
            particle_diameter_ang,
            width_mask_edge_px,
            do_zero_mask,
            do_ctf_correction,
            random_seed,
            padding_factor,
            1,  # TRILINEAR
            current_size,
            minimum_nr_particles,
        )
    )
    return np.asarray([relion_volume_to_recovar(vol) for vol in iref_relion], dtype=np.float64)


def postprocess_bootstrap_iref_via_cpp(
    Iref: np.ndarray,
    *,
    pixel_size: float,
    ini_high_ang: float,
    particle_diameter_ang: float,
    width_mask_edge_px: float,
    do_init_blobs: bool = True,
    is_helical_segment: bool = False,
) -> np.ndarray:
    """Apply RELION's post-bootstrap InitialModel reference processing.

    This wraps the real RELION C++ primitives used by
    ``ml_optimiser.cpp:2940-2980``:

      1. ``initialLowPassFilterReferences``
      2. ``SomGraph::make_blobs_3d`` for positive and negative blobs
      3. standard-deviation-preserving ``Iref = blobs_pos - blobs_neg / 2``
      4. a second ``initialLowPassFilterReferences``
      5. ``softMaskOutsideMap``

    ``compute_bootstrap_iref_via_cpp`` leaves RELION's global ``rand()``
    state exactly where the bootstrap loop leaves it. Calling this wrapper
    immediately afterwards preserves that state for the blob draws, matching
    RELION's denovo InitialModel sequence.
    """
    from recovar.relion_bind import _relion_bind_core as bind
    from recovar.utils.helpers import recovar_volume_to_relion, relion_volume_to_recovar

    arr = np.asarray(Iref, dtype=np.float64)
    if arr.ndim != 4 or arr.shape[1] != arr.shape[2] or arr.shape[2] != arr.shape[3]:
        raise ValueError(f"Iref must have shape (K, N, N, N), got {arr.shape}")

    iref_relion = np.asarray([recovar_volume_to_relion(vol) for vol in arr], dtype=np.float64)
    post_relion = np.asarray(
        bind.vdam_postprocess_initial_iref(
            np.ascontiguousarray(iref_relion),
            float(pixel_size),
            float(ini_high_ang),
            float(particle_diameter_ang),
            float(width_mask_edge_px),
            bool(do_init_blobs),
            bool(is_helical_segment),
        ),
        dtype=np.float64,
    )
    return np.asarray([relion_volume_to_recovar(vol) for vol in post_relion], dtype=np.float64)


def compute_bootstrap_iref(
    *,
    images: np.ndarray,  # (N, H, W) real-space particles in stack order
    ctfs: list[ParticleCTF],  # length N
    sorted_idx: np.ndarray,  # (N,) permutation; sorted_idx[i] = part_id
    particle_diameter_ang: float,
    width_mask_edge_px: float,
    do_zero_mask: bool,
    random_seed: int,
    ori_size: int,
    pixel_size: float,
    ini_high: float,
    nr_classes: int,
    do_ctf_correction: bool = True,
    padding_factor: int = 1,
    interpolator: int = 1,  # TRILINEAR
    gridding_nr_iter: int = 10,
) -> np.ndarray:
    """Reproduce RELION's bootstrap Iref generation.

    Returns an array of shape `(nr_classes, ori_size, ori_size, ori_size)`
    in RELION real-space convention (origin at box centre, RELION axis
    order). Callers that want recovar's convention can apply
    `recovar_volume_to_relion` (it's its own inverse).
    """
    from recovar.relion_bind import _relion_bind_core as bind

    N = images.shape[0]
    if images.shape[1:] != (ori_size, ori_size):
        raise ValueError(f"images shape {images.shape} != expected (N, {ori_size}, {ori_size})")
    if len(ctfs) != N:
        raise ValueError(f"ctfs length {len(ctfs)} != N {N}")
    if sorted_idx.shape != (N,):
        raise ValueError(f"sorted_idx shape {sorted_idx.shape} != (N,)")

    radius_px = particle_diameter_ang / (2.0 * pixel_size)

    # `current_size = 1 / getResolution(ROUND(0.07 * ori_size))` with
    # `getResolution(i) = i / (N*px)`. For 64×8.5, shell=4 → current_size = 136.
    # That exceeds ori_size so the windowFourierTransform is effectively
    # a no-op (kept in line as documentation).
    #
    # NOTE: RELION's wsum_model.current_size in bootstrap is this 136, but
    # since BPref.pad_size = ori_size * padding_factor = 64, the actual
    # backprojection bounds are r_max = ori_size/2 = 32.

    # Per-class accumulators: (class -> (images, rotations, weights))
    per_class_images: list[list[np.ndarray]] = [[] for _ in range(nr_classes)]
    per_class_rots: list[list[np.ndarray]] = [[] for _ in range(nr_classes)]
    per_class_weights: list[list[np.ndarray]] = [[] for _ in range(nr_classes)]

    for part_id_sorted, part_id in enumerate(sorted_idx):
        part_id = int(part_id)
        # 1-4. Random Euler angles via RELION's rnd_unif
        u = np.asarray(bind.vdam_rnd_unif_sequence(random_seed + part_id, 3))
        rot = float(u[0]) * 360.0
        tilt = float(u[1]) * 180.0
        psi = float(u[2]) * 360.0

        A = _euler_angles_to_matrix(rot, tilt, psi)

        iclass = part_id_sorted % nr_classes

        # 7. Load and soft-mask
        img = images[part_id].astype(np.float64)
        if do_zero_mask:
            img = _soft_mask_image_fixture(img, radius_px, float(width_mask_edge_px))

        # 8. FFT with RELION's forward normalisation (divide by H*W)
        H, W = img.shape
        Faux = np.fft.rfft2(img, norm=None) / (H * W)

        # 9. CenterFFTbySign
        Faux = _center_fft_by_sign_2d_half(Faux)

        # 10. Window to current_size — no-op on this path (current_size > ori)

        # 11. CTF image
        ctf = ctfs[part_id]
        Fctf = bind.get_ctf_image(
            ctf.defU,
            ctf.defV,
            ctf.defAngle,
            ctf.voltage,
            ctf.Cs,
            ctf.Q0,
            0.0,
            ctf.angpix,
            ctf.ori_size,
            ctf.ori_size,
            False,  # do_ctf_padding
            False,  # do_abs
            True,  # do_damping
            ctf.phase_shift,
            1.0,
        )
        Fctf = np.asarray(Fctf, dtype=np.float64)

        if do_ctf_correction:
            # 12. RELION's Fimg *= Fctf (in place); then Fctf *= Fctf (weight)
            Fimg = Faux * Fctf
            Fweight = Fctf * Fctf
        else:
            Fimg = Faux
            Fweight = np.ones_like(Fctf)

        per_class_images[iclass].append(Fimg.astype(np.complex128))
        per_class_rots[iclass].append(A.astype(np.float64))
        per_class_weights[iclass].append(Fweight.astype(np.float64))

    # Reconstruct per-class
    Iref = np.zeros((nr_classes, ori_size, ori_size, ori_size), dtype=np.float64)
    n_shells = ori_size // 2 + 1
    dummy_tau2 = np.zeros(n_shells, dtype=np.float64)

    for k in range(nr_classes):
        if not per_class_images[k]:
            continue
        imgs = np.stack(per_class_images[k], axis=0)
        rots = np.stack(per_class_rots[k], axis=0)
        wts = np.stack(per_class_weights[k], axis=0)
        vol = bind.backproject_and_reconstruct(
            imgs,
            rots,
            wts,
            dummy_tau2,
            ori_size,
            padding_factor,
            interpolator,
            False,  # do_map
            gridding_nr_iter,
            1.0,
            False,  # skip_gridding
            -1,  # current_size (default in RELION path from initZeros)
        )
        Iref[k] = np.asarray(vol)

    return Iref


def initial_low_pass_filter_references(
    Iref: np.ndarray,
    *,
    ori_size: int,
    pixel_size: float,
    ini_high_ang: float,
) -> np.ndarray:
    """Apply RELION's `initialLowPassFilterReferences` (ml_optimiser.cpp:3336).

    radius = ori_size * pixel_size / ini_high (in Fourier shells)
    radius -= WIDTH_FMASK_EDGE / 2
    radius_p = radius + WIDTH_FMASK_EDGE

    For r < radius: keep
    For radius <= r <= radius_p: multiply by 0.5 - 0.5*cos(π*(radius_p - r)/WIDTH_FMASK_EDGE)
      → 0 at r=radius_p, 1 at r=radius (note: RELION's formula is
      `0.5 - 0.5*cos(...)`, which equals 0 when cos=1 and 1 when cos=-1 —
      so it's an OUTWARD taper from radius to radius_p, going 1→0 as r grows)
    For r > radius_p: zero
    """
    K = Iref.shape[0]
    radius = ori_size * pixel_size / ini_high_ang
    radius -= WIDTH_FMASK_EDGE / 2.0
    radius_p = radius + WIDTH_FMASK_EDGE

    out = np.zeros_like(Iref)
    for k in range(K):
        vol = Iref[k]
        # Forward FFT with RELION normalisation (divide by N^3)
        F = np.fft.rfftn(vol, axes=(0, 1, 2), norm=None) / vol.size
        # Shell indices
        N = vol.shape[0]
        kz = np.fft.fftfreq(N, d=1.0) * N
        ky = np.fft.fftfreq(N, d=1.0) * N
        kx = np.arange(N // 2 + 1, dtype=np.float64)
        r = np.sqrt(kz[:, None, None] ** 2 + ky[None, :, None] ** 2 + kx[None, None, :] ** 2)
        mask = np.zeros_like(r)
        inner = r < radius
        edge = (r >= radius) & (r <= radius_p)
        outer = r > radius_p
        mask[inner] = 1.0
        if WIDTH_FMASK_EDGE > 0:
            mask[edge] = 0.5 - 0.5 * np.cos(np.pi * (radius_p - r[edge]) / WIDTH_FMASK_EDGE)
        # outer -> mask stays 0
        F_filt = F * mask
        # Inverse FFT
        out[k] = np.fft.irfftn(F_filt * vol.size, s=vol.shape, axes=(0, 1, 2), norm=None)
    return out

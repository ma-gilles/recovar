"""Synthetic cryo-ET windows on straight filaments carrying sparse attachments.

Writes the RELION 5 subtomogram (2D-stack) project of
:func:`recovar.simulation.relion_tomo.generate_relion5_tomo_dataset` in which every
particle (a *window* along a filament) has its own volume, plus a ground-truth
folder. Scene model of window ``i`` on filament ``m`` (particle frame, see below)::

    V_i = c_m C_{k(m)} + sum_{attachments j rendered in window i} T(a_ij, phi_j) (mu + U z_j)

- ``C_k``: core of class ``k`` (a smooth hollow cylinder along the particle z axis,
  :func:`core_cross_section`), ``c_m`` the per-filament contrast (lognormal, mean 1).
- Attachments are physical objects on the filament: axial position ``s_j``, azimuth
  ``phi_j``, latent ``z_j ~ N(0, I_q)``. Window ``i`` (centre ``s_i``) renders every
  attachment whose centre lies in its rendering box, at axial offset ``a_ij = s_j - s_i``;
  its *band truth* is the attachment with ``-L/2 <= a_ij < L/2`` (at most one, since
  attachments on a filament are at least ``L`` apart).
- ``mu`` and the columns of ``U`` are linear combinations of analytic atoms (Gaussian
  blobs and their directional derivatives) defined in the canonical frame around
  ``r0 e_x``; ``U = U_orth diag(mode_scales)`` with ``U_orth`` orthonormal in voxel
  space on the nominal (``grid_size``) grid.

Frame and placement convention
------------------------------
Coordinates ``r = (x, y, z)`` are RELION *reference* (particle) coordinates in A, with
the origin at the box centre (array index ``N/2``) and the filament axis along ``+z``.
Density is positive in RELION's convention (protein bright after RELION's CTF). The
placement operator moves a canonical object ``f`` (defined around ``r0 e_x``) by

    T(a, phi) f (r) = f(R_z(phi)^T (r - a e_z)),   R_z(phi) = [[c, -s, 0], [s, c, 0], [0, 0, 1]],

so the canonical point ``r0 e_x`` lands at ``(r0 cos phi, r0 sin phi, a)`` (phi in
degrees in every table, counter-clockwise about ``+z`` from ``+x``). The atoms are
evaluated analytically at the transformed coordinates, so any continuous ``(a, phi)``
is exact (no interpolation). The particle pose ``A_m = Euler(rot_m, tilt_m, psi_m)``
(RELION's ``Euler_angles2matrix``) maps reference to tomogram coordinates:
a reference point ``r`` of window ``i`` sits at ``c_i + A_m r`` in the tomogram, the
filament direction is ``d_m = A_m e_z`` and window centres are ``c_i = P_m + s_i d_m``.
Tilt image ``f`` sees ``r`` at ``(Aproj_f A_m r)[:2]`` (A, image x = column, y = row,
from the image centre). ``rot_m = 0`` (``azimuth_origin="beam"``) puts the particle
``+x`` axis on the tomogram ``+Z`` (beam) axis when ``tilt_m = 90``; that is phi's origin.

In recovar's frame a real-space volume array ``v[i0, i1, i2]`` has axes 0, 1, 2 along
RELION's reference x, y, z and the opposite sign (recovar's CTF is minus RELION's), so
the volumes handed to the image simulator are ``get_dft3(-V)``; truth MRCs are written
in RELION's convention (``mrcfile`` array ``[z, y, x]``, positive density) and load into
recovar's frame with :func:`recovar.utils.load_relion_volume`.
"""

import copy
import dataclasses
import functools
import json
import logging
import math
import os
import subprocess

import jax
import jax.numpy as jnp
import mrcfile
import numpy as np
import pandas as pd

import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar.simulation import relion_tomo

logger = logging.getLogger(__name__)

FRAME_CONVENTION = (
    "Particle (RELION reference) frame, A, origin at box centre (index N/2), filament axis +z. "
    "T(a, phi) f (r) = f(R_z(phi)^T (r - a e_z)), R_z(phi) counter-clockwise about +z, phi in degrees from +x; "
    "the canonical object is defined around r0 e_x, so its reference point lands at (r0 cos phi, r0 sin phi, a). "
    "a_ij = s_j - s_i (attachment minus window axial position). Pose A_m = RELION Euler_angles2matrix(rot, tilt, psi) "
    "maps reference to tomogram coordinates: tomogram position = rlnCenteredCoordinate + A_m r, filament direction "
    "d_m = A_m e_z, window centre c_i = P_m + s_i d_m; rot_m = 0 puts particle +x on tomogram +Z (beam) at tilt 90. "
    "Density positive in RELION's convention. MRCs: RELION array order [z, y, x]; load with "
    "recovar.utils.load_relion_volume to get recovar's frame (array axes x, y, z, sign flipped)."
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FilamentAttachmentConfig:
    """Every generation parameter (lengths in A, angles in degrees)."""

    # Imaging geometry
    grid_size: int = 96
    voxel_size: float = 8.0
    render_grid_size: int | None = None  # None: large enough that tube ends never reach the image
    tube_end_taper: float = 60.0  # cosine taper of the core at the ends of the rendering box
    # Fourier-slice interpolation. "cubic" (spline-prefiltered) places off-centre density to < 0.15 px at the
    # stage-0 geometry; "linear_interp" pulls it towards the box centre by 1-2.5 px in tilted views.
    disc_type: str = "cubic"
    max_tilt: float = 60.0
    tilt_step: float = 3.0
    dose_per_tilt: float = 2.5
    tilt_axis_angle: float = 85.0
    defocus_range: tuple = (20000.0, 40000.0)
    astigmatism: float = 300.0
    tomogram_size: tuple = (2048, 2048, 512)  # bin-1 pixels
    hand: int = -1
    voltage: float = 300.0
    cs: float = 2.7
    amp_contrast: float = 0.1
    snr: float | None = 0.05  # None: noise-free
    noise_model: str = "white"
    image_contrast_std: float = 0.0  # extra per-window image contrast 1 + N(0, sd)
    premultiplied_ctf: bool = False
    seed: int = 0
    # Scene
    n_tomograms: int = 4
    n_filaments: int = 8
    windows_per_filament: int = 15
    segment_spacing: float = 40.0
    filament_tilt_mean: float = 90.0
    filament_tilt_std: float = 5.0
    psi_mode: str = "uniform"  # "uniform" or "preferred"
    psi_preferred: tuple = (0.0, 90.0)
    psi_spread: float = 5.0
    azimuth_origin: str = "beam"  # "beam" (rot = 0) or "random" (rot uniform)
    contrast_log_sd: float = 0.2  # per-filament contrast c_m, lognormal with mean 1
    scale_attachments_with_contrast: bool = False
    filament_halves: bool = True  # rlnRandomSubset per filament (overlapping windows share a half)
    # Core
    core: str = "tube"  # "tube" or "none"
    core_radii: tuple = (125.0,)  # wall-centre radius of each class
    core_class_probabilities: tuple | None = None
    wall_thickness: float = 50.0
    wall_edge_sigma: float = 6.0
    core_amplitude: float = 1.0
    ridge_count: int = 14
    ridge_amplitude: float = 0.0  # protofilament-like modulation (stress option), 0 = off
    # Attachments
    eta: float = 1.0  # fraction of windows whose band is occupied
    band_length: float = 240.0
    min_attachment_spacing: float | None = None  # None: band_length (must be >= band_length)
    r0: float = 180.0
    q: int = 0
    attachment_amplitude: float = 1.0
    # Canonical object, relative to r0 e_x: (dx, dy, dz, sigma, amplitude) per blob
    blobs: tuple = (
        (-5.0, 0.0, 0.0, 14.0, 1.0),
        (25.0, 12.0, 18.0, 10.0, 1.0),
        (15.0, -20.0, -22.0, 8.0, 1.0),
    )
    # Modes, in order: ("shift", blob, (ex, ey, ez), sd in A) or ("amplitude", blob, sd as a fraction)
    modes: tuple = (("shift", 1, (0.0, 0.0, 1.0), 6.0), ("amplitude", 2, 0.3))
    # Outputs
    n_window_volumes_to_save: int = 0
    n_probe_windows: int = 8  # occupied windows whose noise-free component image powers are recorded (0: skip)
    max_volume_bytes: int = 2**31

    def to_json_dict(self):
        return json.loads(json.dumps(dataclasses.asdict(self)))


def stage_config(stage, geometry="toy", **overrides):
    """The preset of a synthetic ladder stage (``stage0`` .. ``stage3``) on ``geometry`` (``toy`` or ``dogic``).

    - ``stage0``: asymmetric landmark only, no core, no noise, every band occupied (sign/axis/centre checks);
    - ``stage1``: core + one rigid attachment per band (``eta = 1``, ``q = 0``);
    - ``stage2``: as stage 1 with sparse occupancy (``eta`` override, default 0.10; grid 0.05, 0.10, 0.30);
    - ``stage3``: as stage 1 with ``q = 2`` known modes.
    """
    if geometry == "toy":
        base = dict(
            grid_size=96, voxel_size=8.0, segment_spacing=40.0, band_length=240.0, tomogram_size=(2048, 2048, 512)
        )
    elif geometry == "dogic":
        base = dict(
            grid_size=128,
            voxel_size=4.8515625,
            segment_spacing=42.2,
            band_length=2 * 42.2,
            tomogram_size=(4096, 4096, 1024),
        )
    else:
        raise ValueError(f"unknown geometry {geometry!r}")
    stages = {
        "stage0": dict(core="none", snr=None, eta=1.0, q=0, contrast_log_sd=0.0, n_window_volumes_to_save=4),
        "stage1": dict(core="tube", eta=1.0, q=0),
        "stage2": dict(core="tube", eta=0.10, q=0),
        "stage3": dict(core="tube", eta=1.0, q=2),
    }
    if stage not in stages:
        raise ValueError(f"unknown stage {stage!r}; choose from {sorted(stages)}")
    return FilamentAttachmentConfig(**{**base, **stages[stage], **overrides})


def object_extent(config):
    """``(axial, radial)`` reach (A) of the canonical attachment: max ``|c_z| + 3 sigma`` and max distance from the axis."""
    blobs = np.asarray(config.blobs, dtype=np.float64).reshape(-1, 5)
    axial = float(np.max(np.abs(blobs[:, 2]) + 3 * blobs[:, 3]))
    radial = float(np.max(np.hypot(config.r0 + blobs[:, 0], blobs[:, 1]) + 3 * blobs[:, 3]))
    return axial, radial


def resolved_render_grid_size(config):
    """Rendering box (px): ``config.render_grid_size``, or the smallest even box whose cut-off content misses every image.

    Content at axial distance ``Z`` and distance ``rho`` from the filament axis projects at least
    ``Z cos(max_tilt) - rho`` from the image centre when the filament lies in the specimen plane.
    The automatic box makes that exceed the inscribed image radius ``N/2`` for every cut: the
    tube's tapered ends (``rho`` = outer wall radius) and the attachments left out because they
    do not fit whole in the box (``rho`` = their radial reach). Image corners may still miss
    far content.
    """
    n = config.grid_size
    if config.render_grid_size is not None:
        m = int(config.render_grid_size)
    else:
        axial, radial = object_extent(config)
        reach = max(radial, max(config.core_radii) + config.wall_thickness / 2 + 3 * config.wall_edge_sigma)
        cos_tilt = math.cos(math.radians(min(config.max_tilt, 80.0)))
        half = (
            (n * config.voxel_size / 2 + reach) / cos_tilt + max(config.tube_end_taper, axial) + 2 * config.voxel_size
        )
        m = max(n, int(math.ceil(2 * half / config.voxel_size)))
        m += (m - n) % 2
    if m < n or (m - n) % 2:
        raise ValueError(f"render_grid_size {m} must be >= grid_size {n} with an even difference")
    return m


# ---------------------------------------------------------------------------
# Attachment object: analytic atoms, mean and modes
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class AttachmentModel:
    """Atoms (canonical frame, relative to ``r0 e_x``) and the mean / mode weights over them.

    Atom ``k`` is ``exp(-|u - c_k|^2 / (2 s_k^2))`` (``kind 0``) or its derivative under a
    shift along ``e_k``, ``((u - c_k) . e_k / s_k^2) exp(...)`` (``kind 1``).
    ``mu = sum_k mean_weights[k] atom_k``; orthonormal mode ``j`` is
    ``sum_k basis_weights[j, k] atom_k``; ``U[:, j] = mode_scales[j]`` times it.
    """

    centres: np.ndarray  # (n_atoms, 3)
    sigmas: np.ndarray  # (n_atoms,)
    directions: np.ndarray  # (n_atoms, 3), zero for plain Gaussians
    kinds: np.ndarray  # (n_atoms,) 0 Gaussian, 1 derivative
    mean_weights: np.ndarray  # (n_atoms,)
    basis_weights: np.ndarray  # (q, n_atoms)
    mode_scales: np.ndarray  # (q,)
    r0: float
    mode_descriptions: list

    @property
    def q(self):
        return self.basis_weights.shape[0]

    def weights(self, z):
        """Atom weights ``(n, n_atoms)`` of ``mu + U z_i`` for latents ``z`` of shape ``(n, q)``."""
        z = np.asarray(z, dtype=np.float64)
        if z.ndim != 2 or z.shape[1] != self.q:
            raise ValueError(f"z must have shape (n, {self.q}), got {z.shape}")
        return self.mean_weights[None, :] + (z * self.mode_scales[None, :]) @ self.basis_weights


# ---------------------------------------------------------------------------
# Exact rendering on a voxel grid
# ---------------------------------------------------------------------------


def _axis_grid(n, voxel_size):
    """Reference coordinate (A) of voxel ``i`` along one axis: ``(i - n // 2) * voxel_size``."""
    return (jnp.arange(n, dtype=jnp.float64) - n // 2) * voxel_size


@functools.partial(jax.jit, static_argnames=("n", "kinds"))
def _render_objects(n, kinds, voxel_size, r0, a, phi_deg, weights, centres, sigmas, directions):
    """``sum_k sum_atoms weights[k, atom] atom(R_z(phi_k)^T (r - a_k e_z) - r0 e_x)`` on an ``n^3`` grid.

    Array axes 0, 1, 2 are the reference x, y, z (A, :func:`_axis_grid`). Gaussians are
    separable into an ``(x, y)`` factor and a ``z`` factor, so each atom costs one
    outer product. ``kinds`` (static) is 0 for a Gaussian and 1 for its directional derivative.
    """
    x = _axis_grid(n, voxel_size)
    xx, yy = x[:, None], x[None, :]
    phi = jnp.deg2rad(phi_deg)

    def body(k, acc):
        c, s = jnp.cos(phi[k]), jnp.sin(phi[k])
        ux = c * xx + s * yy - r0
        uy = -s * xx + c * yy
        uz = x - a[k]
        for atom, kind in enumerate(kinds):
            dx, dy, dz = ux - centres[atom, 0], uy - centres[atom, 1], uz - centres[atom, 2]
            inv = 1.0 / (2.0 * sigmas[atom] ** 2)
            gxy = jnp.exp(-(dx**2 + dy**2) * inv)
            gz = jnp.exp(-(dz**2) * inv)
            if kind == 0:
                value = gxy[:, :, None] * gz[None, None, :]
            else:
                e = directions[atom] / sigmas[atom] ** 2
                value = ((dx * e[0] + dy * e[1]) * gxy)[:, :, None] * gz[None, None, :] + gxy[:, :, None] * (
                    dz * e[2] * gz
                )[None, None, :]
            acc = acc + weights[k, atom] * value
        return acc

    return jax.lax.fori_loop(0, a.shape[0], body, jnp.zeros((n, n, n), dtype=jnp.float64))


def render_objects(model, n, voxel_size, a, phi_deg, weights):
    """Real-space RELION-frame density of objects ``model`` placed at ``(a[k], phi_deg[k])`` with atom weights ``weights[k]``."""
    a = jnp.asarray(np.atleast_1d(a), dtype=jnp.float64)
    phi_deg = jnp.asarray(np.atleast_1d(phi_deg), dtype=jnp.float64)
    weights = jnp.asarray(np.atleast_2d(weights), dtype=jnp.float64)
    return _render_objects(
        int(n),
        tuple(int(k) for k in model.kinds),
        float(voxel_size),
        float(model.r0),
        a,
        phi_deg,
        weights,
        jnp.asarray(model.centres),
        jnp.asarray(model.sigmas),
        jnp.asarray(model.directions),
    )


def core_cross_section(n, voxel_size, radius, thickness, edge_sigma, ridge_count=14, ridge_amplitude=0.0):
    """``(n, n)`` (x, y) profile of a hollow cylinder: wall ``radius +- thickness/2`` with Gaussian-blurred (erf) edges.

    ``ridge_amplitude`` multiplies the wall by ``1 + ridge_amplitude cos(ridge_count theta)``.
    """
    x = _axis_grid(n, voxel_size)
    xx, yy = x[:, None], x[None, :]
    rho = jnp.sqrt(xx**2 + yy**2)
    scale = math.sqrt(2.0) * edge_sigma
    wall = 0.5 * (
        jax.scipy.special.erf((rho - (radius - thickness / 2)) / scale)
        - jax.scipy.special.erf((rho - (radius + thickness / 2)) / scale)
    )
    if ridge_amplitude:
        wall = wall * (1.0 + ridge_amplitude * jnp.cos(ridge_count * jnp.arctan2(yy, xx)))
    return wall


def core_axial_profile(n, voxel_size, flat_half_length, taper):
    """``(n,)`` z profile: 1 for ``|z| <= flat_half_length``, then a cosine^2 fall to 0 over ``taper`` A."""
    z = jnp.abs(_axis_grid(n, voxel_size))
    t = jnp.clip((z - flat_half_length) / max(taper, 1e-9), 0.0, 1.0)
    return jnp.cos(0.5 * jnp.pi * t) ** 2


@functools.partial(jax.jit, static_argnames=("n", "kinds"))
def _window_volume_ft(n, kinds, voxel_size, r0, cross, axial, core_scale, a, phi, weights, centres, sigmas, directions):
    density = core_scale * cross[:, :, None] * axial[None, None, :] + _render_objects(
        n, kinds, voxel_size, r0, a, phi, weights, centres, sigmas, directions
    )
    # recovar's frame: same axes, opposite sign (see the module docstring).
    return fourier_transform_utils.get_dft3(-density).reshape(-1)


# ---------------------------------------------------------------------------
# Attachment model
# ---------------------------------------------------------------------------


def build_attachment_model(config):
    """Atoms, mean and orthonormal modes of the canonical attachment (:class:`AttachmentModel`).

    Mode ``j`` starts as a physical perturbation ``h_j`` (``("shift", b, e, sd)``: blob ``b``
    moved along ``e``, linearised, ``h = A_b d/de G_b``; ``("amplitude", b, sd)``: ``h = A_b G_b``).
    ``[h_1 .. h_q] = Q R`` on the nominal grid at ``T(0, 0)``; the orthonormal modes are ``Q``
    and ``mode_scales[j] = sd_j R_jj`` (the part of ``sd_j h_j`` orthogonal to the earlier modes).
    """
    if config.q > len(config.modes):
        raise ValueError(f"q = {config.q} but only {len(config.modes)} modes are defined")
    blobs = np.asarray(config.blobs, dtype=np.float64).reshape(-1, 5)
    amp = config.attachment_amplitude * blobs[:, 4]
    centres, sigmas = list(blobs[:, :3]), list(blobs[:, 3])
    directions, kinds = [np.zeros(3)] * len(blobs), [0] * len(blobs)
    mode_atoms, physical_sd, descriptions = [], [], []
    for mode in config.modes[: config.q]:
        kind, blob = mode[0], int(mode[1])
        if kind == "shift":
            e = np.asarray(mode[2], dtype=np.float64)
            e = e / np.linalg.norm(e)
            centres.append(blobs[blob, :3])
            sigmas.append(blobs[blob, 3])
            directions.append(e)
            kinds.append(1)
            mode_atoms.append((len(kinds) - 1, amp[blob]))
            physical_sd.append(float(mode[3]))
            descriptions.append(f"shift of blob {blob} along {e.round(4).tolist()}, sd {mode[3]} A (linearised)")
        elif kind == "amplitude":
            mode_atoms.append((blob, amp[blob]))
            physical_sd.append(float(mode[2]))
            descriptions.append(f"amplitude of blob {blob}, sd {mode[2]} of its amplitude")
        else:
            raise ValueError(f"unknown mode kind {kind!r}")
    n_atoms = len(kinds)
    mean_weights = np.zeros(n_atoms)
    mean_weights[: len(blobs)] = amp
    perturbations = np.zeros((config.q, n_atoms))
    for j, (atom, a) in enumerate(mode_atoms):
        perturbations[j, atom] = a
    model = AttachmentModel(
        centres=np.asarray(centres, dtype=np.float64).reshape(n_atoms, 3),
        sigmas=np.asarray(sigmas, dtype=np.float64),
        directions=np.asarray(directions, dtype=np.float64).reshape(n_atoms, 3),
        kinds=np.asarray(kinds, dtype=int),
        mean_weights=mean_weights,
        basis_weights=np.zeros((config.q, n_atoms)),
        mode_scales=np.zeros(config.q),
        r0=float(config.r0),
        mode_descriptions=descriptions,
    )
    if config.q:
        h = np.stack(
            [
                np.asarray(
                    render_objects(model, config.grid_size, config.voxel_size, 0.0, 0.0, perturbations[j])
                ).ravel()
                for j in range(config.q)
            ],
            axis=1,
        )
        _, r = np.linalg.qr(h)
        signs = np.sign(np.diag(r))
        r = r * signs[:, None]
        model.basis_weights = np.linalg.solve(r.T, perturbations)
        model.mode_scales = np.asarray(physical_sd) * np.diag(r)
    extent = np.max(np.linalg.norm(model.centres, axis=1) + 3 * model.sigmas)
    if model.r0 + extent > config.grid_size * config.voxel_size / 2:
        logger.warning("attachment reaches %.0f A from the axis, beyond the half box", model.r0 + extent)
    return model


# ---------------------------------------------------------------------------
# Scene: filaments, windows, physical attachments
# ---------------------------------------------------------------------------


def _attachment_positions(rng, s_lo, s_hi, eta, band_length, min_spacing):
    """Stationary renewal process on ``[s_lo, s_hi]`` with gaps ``min_spacing + Exp(band_length / eta - min_spacing)``.

    With gaps of at least ``min_spacing >= band_length``, the bands of the windows an
    attachment occupies are disjoint, so the occupied fraction of windows is the
    attachment density times ``band_length``, i.e. ``eta``.
    """
    if eta <= 0:
        return np.zeros(0)
    mean_gap = band_length / eta
    extra = mean_gap - min_spacing
    if extra < -1e-9:
        raise ValueError(f"eta {eta} needs a mean spacing {mean_gap} below the minimum spacing {min_spacing}")
    extra = max(extra, 0.0)
    pos = s_lo - (10.0 + rng.random()) * mean_gap
    out = []
    while pos <= s_hi:
        if pos >= s_lo:
            out.append(pos)
        pos += min_spacing + (rng.exponential(extra) if extra > 0 else 0.0)
    return np.asarray(out)


def build_scene(config):
    """Filaments, windows, physical attachments and their window links (all ground truth, no images).

    Returns a dict of DataFrames ``filaments``, ``windows`` (one row per particle, in
    particle order), ``attachments`` and ``window_attachments`` (every rendered pair: the
    attachment fits whole in the window's rendering box),
    plus ``render_grid_size`` and ``latents`` (``(n_attachments, q)``).
    """
    filament_rng, attachment_rng, assign_rng = (
        np.random.default_rng(s) for s in np.random.SeedSequence(config.seed).spawn(3)
    )
    band = float(config.band_length)
    min_spacing = band if config.min_attachment_spacing is None else float(config.min_attachment_spacing)
    if min_spacing < band:
        raise ValueError("min_attachment_spacing must be >= band_length (at most one attachment per band)")
    m_render = resolved_render_grid_size(config)
    render_half = m_render * config.voxel_size / 2
    # Attachments are rendered only when they fit whole in the box (grid z from -m/2 to m/2 - 1 voxels).
    axial_extent, _ = object_extent(config)
    render_limit = (m_render // 2 - 1) * config.voxel_size - axial_extent
    nominal_half = config.grid_size * config.voxel_size / 2
    n_fil, n_win = int(config.n_filaments), int(config.windows_per_filament)
    s_windows = (np.arange(n_win) - (n_win - 1) / 2) * config.segment_spacing

    tomo_px = np.asarray(config.tomogram_size, dtype=np.float64)
    extent = np.array([0.35, 0.35, 0.15]) * tomo_px * config.voxel_size
    tomogram = np.arange(n_fil) % config.n_tomograms
    tilt = np.clip(filament_rng.normal(config.filament_tilt_mean, config.filament_tilt_std, n_fil), 0.0, 180.0)
    if config.psi_mode == "uniform":
        psi = filament_rng.uniform(-180.0, 180.0, n_fil)
    elif config.psi_mode == "preferred":
        psi = np.asarray(config.psi_preferred)[filament_rng.integers(0, len(config.psi_preferred), n_fil)]
        psi = psi + filament_rng.normal(0.0, config.psi_spread, n_fil)
    else:
        raise ValueError(f"unknown psi_mode {config.psi_mode!r}")
    if config.azimuth_origin == "beam":
        rot = np.zeros(n_fil)
    elif config.azimuth_origin == "random":
        rot = filament_rng.uniform(-180.0, 180.0, n_fil)
    else:
        raise ValueError(f"unknown azimuth_origin {config.azimuth_origin!r}")
    anchor = filament_rng.uniform(-extent, extent, size=(n_fil, 3))
    sd = config.contrast_log_sd
    contrast = np.exp(filament_rng.normal(-0.5 * sd**2, sd, n_fil)) if sd > 0 else np.ones(n_fil)
    n_classes = len(config.core_radii)
    probs = None if config.core_class_probabilities is None else np.asarray(config.core_class_probabilities, float)
    core_class = filament_rng.choice(n_classes, size=n_fil, p=probs)
    poses = relion_tomo._relion_euler_matrix(np.stack([rot, tilt, psi], axis=1))  # reference -> tomogram
    direction = poses[:, :, 2]
    half = np.zeros(n_fil, dtype=int)
    for t in range(config.n_tomograms):
        members = np.nonzero(tomogram == t)[0]
        half[assign_rng.permutation(members)] = 1 + np.arange(members.size) % 2
    if not config.filament_halves:
        half[:] = 0  # filled per window below

    filaments = pd.DataFrame(
        {
            "filament_id": np.arange(n_fil),
            "tomogram_index": tomogram,
            "rot": rot,
            "tilt": tilt,
            "psi": psi,
            "direction_x": direction[:, 0],
            "direction_y": direction[:, 1],
            "direction_z": direction[:, 2],
            "anchor_x_angst": anchor[:, 0],
            "anchor_y_angst": anchor[:, 1],
            "anchor_z_angst": anchor[:, 2],
            "contrast": contrast,
            "core_class": core_class,
            "core_radius": np.asarray(config.core_radii, float)[core_class],
            "random_subset": half,
        }
    )

    # Physical attachments, filament by filament.
    att_rows, latents = [], []
    for m in range(n_fil):
        s_att = _attachment_positions(
            attachment_rng, s_windows[0] - render_half, s_windows[-1] + render_half, config.eta, band, min_spacing
        )
        phi = attachment_rng.uniform(0.0, 360.0, s_att.size)
        z = attachment_rng.standard_normal((s_att.size, config.q))
        local = np.stack([config.r0 * np.cos(np.deg2rad(phi)), config.r0 * np.sin(np.deg2rad(phi)), s_att], axis=1)
        centre = anchor[m] + local @ poses[m].T
        for j in range(s_att.size):
            att_rows.append(
                {
                    "filament_id": m,
                    "tomogram_index": tomogram[m],
                    "s_angst": s_att[j],
                    "phi_deg": phi[j],
                    "centre_x_angst": centre[j, 0],
                    "centre_y_angst": centre[j, 1],
                    "centre_z_angst": centre[j, 2],
                }
            )
        latents.append(z)
    attachments = pd.DataFrame(
        att_rows,
        columns=[
            "filament_id",
            "tomogram_index",
            "s_angst",
            "phi_deg",
            "centre_x_angst",
            "centre_y_angst",
            "centre_z_angst",
        ],
    )
    attachments.insert(0, "attachment_id", np.arange(len(attachments)))
    latents = np.concatenate(latents, axis=0) if latents else np.zeros((0, config.q))
    for k in range(config.q):
        attachments[f"z{k}"] = latents[:, k]

    # Windows in particle order: tomogram, filament, position along the filament.
    order = sorted(range(n_fil), key=lambda m: (tomogram[m], m))
    win_rows, link_rows = [], []
    by_filament = {m: attachments.index[attachments["filament_id"] == m].to_numpy() for m in range(n_fil)}
    for m in order:
        for i, s_i in enumerate(s_windows):
            window_id = len(win_rows)
            centre = anchor[m] + s_i * direction[m]
            band_id, band_a = -1, np.nan
            for j in by_filament[m]:
                a = attachments.at[j, "s_angst"] - s_i
                if abs(a) > render_limit:
                    continue
                in_band = -band / 2 <= a < band / 2
                link_rows.append(
                    {
                        "window_id": window_id,
                        "attachment_id": int(attachments.at[j, "attachment_id"]),
                        "a_angst": a,
                        "in_band": bool(in_band),
                        "in_nominal_box": bool(abs(a) <= nominal_half),
                    }
                )
                if in_band:
                    if band_id >= 0:
                        raise AssertionError("two attachments in one band; min spacing violated")
                    band_id, band_a = int(attachments.at[j, "attachment_id"]), a
            row = {
                "window_id": window_id,
                "tomogram_index": tomogram[m],
                "filament_id": m,
                "window_index": i,
                "s_angst": s_i,
                "band_occupied": band_id >= 0,
                "band_attachment_id": band_id,
                "a_angst": band_a,
                "phi_deg": attachments.at[band_id, "phi_deg"] if band_id >= 0 else np.nan,
            }
            for k in range(config.q):
                row[f"z{k}"] = latents[band_id, k] if band_id >= 0 else np.nan
            row.update(
                contrast=contrast[m],
                core_class=core_class[m],
                core_radius=float(np.asarray(config.core_radii, float)[core_class[m]]),
                centre_x_angst=centre[0],
                centre_y_angst=centre[1],
                centre_z_angst=centre[2],
                rot=rot[m],
                tilt=tilt[m],
                psi=psi[m],
                random_subset=half[m],
            )
            win_rows.append(row)
    windows = pd.DataFrame(win_rows)
    window_attachments = pd.DataFrame(
        link_rows, columns=["window_id", "attachment_id", "a_angst", "in_band", "in_nominal_box"]
    )
    if not config.filament_halves:
        windows["random_subset"] = assign_rng.permutation(np.arange(len(windows)) % 2) + 1
    n_rendered = window_attachments.groupby("window_id").size()
    windows["n_rendered_attachments"] = n_rendered.reindex(windows["window_id"]).fillna(0).astype(int).to_numpy()
    in_band = window_attachments[window_attachments["in_band"]].groupby("attachment_id").size()
    attachments["n_band_windows"] = in_band.reindex(attachments["attachment_id"]).fillna(0).astype(int).to_numpy()
    rendered = window_attachments.groupby("attachment_id").size()
    attachments["n_rendered_windows"] = rendered.reindex(attachments["attachment_id"]).fillna(0).astype(int).to_numpy()
    return {
        "filaments": filaments,
        "windows": windows,
        "attachments": attachments,
        "window_attachments": window_attachments,
        "latents": latents,
        "render_grid_size": m_render,
    }


# ---------------------------------------------------------------------------
# Per-window volumes
# ---------------------------------------------------------------------------


class WindowVolumes:
    """Per-window volumes of a scene, built on demand (no per-window file or array is stored).

    ``fourier(indices)`` is the ``particle_volume_fn`` of
    :func:`relion_tomo.generate_relion5_tomo_dataset`: flat centred Fourier volumes in
    recovar's frame on the ``n``-grid. ``real(i)`` is window ``i``'s RELION-frame density.
    ``core_weight`` / ``attachment_weight`` scale the two terms (e.g. 0 for a component-only probe).
    """

    def __init__(self, config, scene, model, n, *, core_weight=1.0, attachment_weight=1.0):
        self.n, self.voxel_size, self.model = int(n), float(config.voxel_size), model
        windows, links = scene["windows"], scene["window_attachments"]
        render_half = scene["render_grid_size"] * config.voxel_size / 2
        flat = render_half - 2 * config.voxel_size - config.tube_end_taper
        self.axial = core_axial_profile(self.n, self.voxel_size, flat, config.tube_end_taper)
        self.cross = [
            core_cross_section(
                self.n,
                self.voxel_size,
                radius,
                config.wall_thickness,
                config.wall_edge_sigma,
                config.ridge_count,
                config.ridge_amplitude,
            )
            for radius in config.core_radii
        ]
        core_on = 0.0 if config.core == "none" else config.core_amplitude * core_weight
        self.core_scale = core_on * windows["contrast"].to_numpy()
        self.core_class = windows["core_class"].to_numpy()
        n_w = len(windows)
        k_max = max(1, int(links.groupby("window_id").size().max()) if len(links) else 1)
        self.a = np.zeros((n_w, k_max))
        self.phi = np.zeros((n_w, k_max))
        self.weights = np.zeros((n_w, k_max, len(model.kinds)))
        att = scene["attachments"].set_index("attachment_id")
        atom_weights = model.weights(scene["latents"]) if len(att) else np.zeros((0, len(model.kinds)))
        scale = attachment_weight * (
            windows["contrast"].to_numpy() if config.scale_attachments_with_contrast else np.ones(n_w)
        )
        for w, group in links.groupby("window_id"):
            ids = group["attachment_id"].to_numpy()
            k = ids.size
            self.a[w, :k] = group["a_angst"].to_numpy()
            self.phi[w, :k] = att.loc[ids, "phi_deg"].to_numpy()
            self.weights[w, :k] = scale[w] * atom_weights[ids]

    def _args(self, i):
        m = self.model
        return (
            self.n,
            tuple(int(k) for k in m.kinds),
            self.voxel_size,
            float(m.r0),
            self.cross[self.core_class[i]],
            self.axial,
            float(self.core_scale[i]),
            jnp.asarray(self.a[i]),
            jnp.asarray(self.phi[i]),
            jnp.asarray(self.weights[i]),
            jnp.asarray(m.centres),
            jnp.asarray(m.sigmas),
            jnp.asarray(m.directions),
        )

    def fourier(self, indices):
        return jnp.stack([_window_volume_ft(*self._args(int(i))) for i in np.atleast_1d(indices)])

    def real(self, i):
        args = self._args(int(i))
        return np.asarray(
            args[6] * args[4][:, :, None] * args[5][None, None, :]
            + _render_objects(args[0], args[1], *args[2:4], *args[7:])
        )


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def write_relion_frame_mrc(path, density, voxel_size):
    """Write a RELION-frame density (array axes x, y, z) as a RELION MRC (array ``[z, y, x]``)."""
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.ascontiguousarray(np.transpose(np.asarray(density), (2, 1, 0))).astype(np.float32))
        mrc.voxel_size = voxel_size


def recovar_git_state():
    """``(sha, dirty)`` of the checkout holding this module, or ``(None, None)`` outside git."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.run(
            ["git", "-C", here, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", here, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return sha, bool(status.strip())


def _component_image_power(output_folder, config, scene, model, info, windows_to_probe):
    """Mean noise-free image power (before RELION normalisation) of the total and attachment-only signal of some windows."""
    from recovar.simulation import optics_groups as optics_groups_sim

    rows = np.nonzero(np.isin(info["flat_rows_particle"], windows_to_probe))[0]
    if rows.size == 0:
        return None
    m = scene["render_grid_size"]
    setting = [{"pixel_size": config.voxel_size, "box_size": config.grid_size, "noise_scale": 1.0}]
    out = {}
    for label, core_weight, att_weight in (("total", 1.0, 1.0), ("attachment", 0.0, 1.0), ("core", 1.0, 0.0)):
        vols = WindowVolumes(config, scene, model, m, core_weight=core_weight, attachment_weight=att_weight)
        images, _, _ = optics_groups_sim.simulate_per_particle_volumes(
            vols.fourier,
            m,
            setting,
            np.zeros(rows.size, dtype=int),
            info["flat_rows_particle"][rows],
            np.asarray(info["flat_rows_rots"])[rows],
            info["flat_rows_ctf_params"][rows],
            np.ones(rows.size),
            ctf_evaluator=relion_tomo.relion_tomo_ctf,
            voxel_size=config.voxel_size,
            grid_size=config.grid_size,
            snr=None,
            noise_model=config.noise_model,
            n_probe_particles=8,
            seed=config.seed,
            disc_type=config.disc_type,
            premultiplied_ctf=False,
            max_volume_bytes=config.max_volume_bytes,
        )
        out[label] = float(np.mean(images**2))
    return out


def generate_filament_attachment_dataset(output_folder, config):
    """Write the RELION 5 project and the ``truth/`` folder of a filament-attachment scene.

    ``config`` is a :class:`FilamentAttachmentConfig` (or a dict of its fields). Outputs:

    - the project of :func:`relion_tomo.generate_relion5_tomo_dataset` (``particles.star``
      with ``rlnHelicalTubeID`` = filament id + 1 and ``rlnHelicalTrackLengthAngst``,
      ``tomograms.star``, ``optimisation_set.star``, ``particles_2d.star``, stacks);
    - ``truth/windows.csv`` (one row per particle, in particle order), ``truth/attachments.csv``
      (one row per physical attachment), ``truth/window_attachments.csv`` (every rendered
      window/attachment pair, with ``a``, band and nominal-box flags), ``truth/filaments.csv``;
    - ``truth/volumes/``: ``core_class<k>.mrc`` (contrast 1), ``mu.mrc`` and
      ``U_orth_mode<j>.mrc`` at ``T(0, 0)`` on the nominal grid, optional ``window_<id>.mrc``,
      all RELION-frame MRCs;
    - ``truth/generation.json``: every parameter, the attachment model, the frame convention,
      the recovar git state, the noise calibration and component image powers.

    Returns a dict with the project paths, the scene tables and the JSON record.
    """
    if isinstance(config, dict):
        config = FilamentAttachmentConfig(**config)
    config = copy.deepcopy(config)
    sha, dirty = recovar_git_state()
    os.makedirs(output_folder, exist_ok=True)
    truth = os.path.join(output_folder, "truth")
    os.makedirs(os.path.join(truth, "volumes"), exist_ok=True)

    model = build_attachment_model(config)
    scene = build_scene(config)
    windows = scene["windows"]
    m = scene["render_grid_size"]
    volumes = WindowVolumes(config, scene, model, m)
    logger.info(
        "Scene: %d filaments, %d windows, %d attachments, occupied fraction %.3f, render box %d px",
        len(scene["filaments"]),
        len(windows),
        len(scene["attachments"]),
        windows["band_occupied"].mean(),
        m,
    )

    eulers = windows[["rot", "tilt", "psi"]].to_numpy()
    coords = windows[["centre_x_angst", "centre_y_angst", "centre_z_angst"]].to_numpy()
    s0 = (1 - config.windows_per_filament) / 2 * config.segment_spacing
    result = relion_tomo.generate_relion5_tomo_dataset(
        output_folder,
        None,
        config.voxel_size,
        len(windows),
        grid_size=config.grid_size,
        n_tomograms=config.n_tomograms,
        optics_groups=[
            {"voltage": config.voltage, "cs": config.cs, "amp_contrast": config.amp_contrast, "noise_scale": 1.0}
        ],
        max_tilt=config.max_tilt,
        tilt_step=config.tilt_step,
        dose_per_tilt=config.dose_per_tilt,
        tilt_axis_angle=config.tilt_axis_angle,
        defocus_range=tuple(config.defocus_range),
        astigmatism=config.astigmatism,
        tomogram_size=tuple(config.tomogram_size),
        hand=config.hand,
        snr=config.snr,
        noise_model=config.noise_model,
        contrast_std=config.image_contrast_std,
        hidden_tilt_fraction=0.0,
        origin_std_angstrom=0.0,
        particle_eulers=eulers,
        premultiplied_ctf=config.premultiplied_ctf,
        seed=config.seed,
        atomic_solvent_correction=False,
        particle_coordinates=coords,
        particle_tomograms=windows["tomogram_index"].to_numpy(),
        particle_random_subset=windows["random_subset"].to_numpy(),
        extra_particle_columns={
            "_rlnHelicalTubeID": windows["filament_id"].to_numpy() + 1,
            "_rlnHelicalTrackLengthAngst": windows["s_angst"].to_numpy() - s0,
        },
        disc_type=config.disc_type,
        particle_volume_fn=volumes.fourier,
        particle_volume_grid_size=m,
        max_volume_bytes=config.max_volume_bytes,
    )
    info = result["simulation_info"]
    tomo_names = [f"TS_{t + 1:02d}" for t in range(config.n_tomograms)]
    windows.insert(1, "rlnTomoParticleName", info["particle_names"])
    windows.insert(2, "tomogram", [tomo_names[t] for t in windows["tomogram_index"]])
    for table in ("filaments", "attachments"):
        scene[table].insert(2, "tomogram", [tomo_names[t] for t in scene[table]["tomogram_index"]])
    links = scene["window_attachments"]
    links.insert(1, "rlnTomoParticleName", windows["rlnTomoParticleName"].to_numpy()[links["window_id"].to_numpy()])
    for name in ("windows", "attachments", "window_attachments", "filaments"):
        scene[name].to_csv(os.path.join(truth, f"{name}.csv"), index=False)

    # Truth volumes on the nominal grid.
    n = config.grid_size
    vol_dir = os.path.join(truth, "volumes")
    nominal = WindowVolumes(config, scene, model, n)
    volume_files = {}
    for k in range(len(config.core_radii)):
        if config.core != "none":
            path = os.path.join(vol_dir, f"core_class{k}.mrc")
            write_relion_frame_mrc(
                path,
                config.core_amplitude * nominal.cross[k][:, :, None] * nominal.axial[None, None, :],
                config.voxel_size,
            )
            volume_files[f"core_class{k}"] = os.path.relpath(path, output_folder)
    mu = np.asarray(render_objects(model, n, config.voxel_size, 0.0, 0.0, model.mean_weights))
    write_relion_frame_mrc(os.path.join(vol_dir, "mu.mrc"), mu, config.voxel_size)
    volume_files["mu"] = "truth/volumes/mu.mrc"
    for j in range(model.q):
        u = np.asarray(render_objects(model, n, config.voxel_size, 0.0, 0.0, model.basis_weights[j]))
        write_relion_frame_mrc(os.path.join(vol_dir, f"U_orth_mode{j}.mrc"), u, config.voxel_size)
        volume_files[f"U_orth_mode{j}"] = f"truth/volumes/U_orth_mode{j}.mrc"
    for w in range(min(config.n_window_volumes_to_save, len(windows))):
        write_relion_frame_mrc(os.path.join(vol_dir, f"window_{w}.mrc"), nominal.real(w), config.voxel_size)
        volume_files[f"window_{w}"] = f"truth/volumes/window_{w}.mrc"

    occupied = windows.index[windows["band_occupied"]].to_numpy()
    n_probe = min(config.n_probe_windows, occupied.size)
    probe = occupied[np.unique(np.linspace(0, occupied.size - 1, n_probe).round().astype(int))] if n_probe else []
    powers = _component_image_power(output_folder, config, scene, model, info, np.asarray(probe, dtype=int))
    calibration = info["noise_calibration"]
    noise_power = calibration.get("target_noise_power")
    record = {
        "config": config.to_json_dict(),
        "render_grid_size": m,
        "recovar_git_sha": sha,
        "recovar_git_dirty": dirty,
        "frame_convention": FRAME_CONVENTION,
        "attachment_model": {
            "atom_centres_relative_to_r0_ex": model.centres.tolist(),
            "atom_sigmas": model.sigmas.tolist(),
            "atom_directions": model.directions.tolist(),
            "atom_kinds": model.kinds.tolist(),
            "mean_weights": model.mean_weights.tolist(),
            "orthonormal_mode_weights": model.basis_weights.tolist(),
            "mode_scales": model.mode_scales.tolist(),
            "mode_descriptions": model.mode_descriptions,
            "U": "U[:, j] = mode_scales[j] * U_orth_mode<j> (voxel-orthonormal on the nominal grid at T(0, 0)); z ~ N(0, I_q)",
            "mu_norm": float(np.linalg.norm(mu)),
        },
        "counts": {
            "filaments": len(scene["filaments"]),
            "windows": len(windows),
            "attachments": len(scene["attachments"]),
            "occupied_fraction": float(windows["band_occupied"].mean()),
            "tilts": int(len(info["flat_rows_particle"]) // max(len(windows), 1)),
        },
        "noise_calibration": calibration,
        "noise_variance_per_pixel": noise_power,
        "probe_windows_occupied": [int(p) for p in probe],
        "image_power_noise_free_occupied_probe": powers,
        "image_snr_occupied_probe": None
        if (powers is None or not noise_power)
        else {k: v / noise_power for k, v in powers.items()},
        "volume_files": volume_files,
        "project": {k: os.path.relpath(v, output_folder) for k, v in result.items() if isinstance(v, str)},
    }
    with open(os.path.join(truth, "generation.json"), "w") as f:
        json.dump(record, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    logger.info("Wrote %s (truth in %s)", output_folder, truth)
    return {**result, "scene": scene, "model": model, "record": record}

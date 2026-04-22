"""End-to-end E-step parity: RELION bindings vs recovar engine_v2 on identical inputs.

Convention bridge (the critical insight):
  RELION uses Minvsigma2 = 1/(2*sigma2) in the diff2 sum, then exp(-diff2).
  recovar uses 1/sigma2 in preprocessing, then score = -0.5*(cross+norm),
  and exp(score) for posteriors. Both yield exp(-sum |res|^2/(2*sigma2)),
  so posteriors match when half_spectrum_scoring=True (all-1 weights).

  With Hermitian weights (w=2 for interior): the half-spectrum sum recovers
  the FULL-spectrum inner product, which is ~2x the half-complex sum.
  This makes the posterior exponentially more peaked — wrong for RELION parity.
  _refine_relion_mode correctly uses half_spectrum_scoring=True.

Exact parity required: rel_err < 1e-10.
"""

import numpy as np
import pytest
from recovar.relion_bind._relion_bind_core import (
    TRILINEAR,
    convert_squared_differences_to_weights,
    euler_angles_to_matrix,
    get_coarse_orientations,
    get_ctf_image,
    project_volume,
    shift_image_in_fourier_transform_2d,
)


def _make_test_volume(N, rng):
    vol = rng.standard_normal((N, N, N))
    ft = np.fft.rfftn(vol)
    kz = np.fft.fftfreq(N, d=1.0 / N)
    ky = np.fft.fftfreq(N, d=1.0 / N)
    kx = np.arange(N // 2 + 1)
    KZ, KY = np.meshgrid(kz, ky, indexing="ij")
    r2 = KZ[:, :, np.newaxis] ** 2 + KY[:, :, np.newaxis] ** 2 + kx[np.newaxis, np.newaxis, :] ** 2
    ft *= np.exp(-r2 / (2 * (N / 6) ** 2))
    return np.fft.irfftn(ft, s=(N, N, N))


def _shell_indices_fftw(N):
    ky = np.arange(N)
    ky = np.where(ky <= N // 2, ky, ky - N)
    kx = np.arange(N // 2 + 1)
    ky2d, kx2d = np.meshgrid(ky, kx, indexing="ij")
    return np.round(np.sqrt(ky2d**2 + kx2d**2)).astype(int)


def _fftw_to_half_flat(img_fftw, N):
    """FFTW half-complex (N, N//2+1) → recovar flat (N_half,)."""
    return img_fftw.reshape(-1)


def _setup_scenario(N=32, n_rot=12, n_trans=5, seed=42):
    """Create synthetic inputs in FFTW convention."""
    rng = np.random.default_rng(seed)
    vol = _make_test_volume(N, rng)

    ctf = get_ctf_image(
        defU=15000.0,
        defV=14000.0,
        defAng=30.0,
        voltage=300.0,
        Cs=2.7,
        Q0=0.07,
        Bfac=0.0,
        angpix=1.5,
        orixdim=N,
        oriydim=N,
        do_ctf_padding=False,
        do_abs=False,
        do_damping=False,
        phase_shift=0.0,
        scale=1.0,
    )

    shell_idx = _shell_indices_fftw(N)
    n_shells = N // 2 + 1
    sigma2 = rng.uniform(0.005, 0.05, n_shells)
    sigma2[0] = 1e10

    orientations = get_coarse_orientations(1)[:n_rot]
    translations = np.array([[0, 0], [1.0, 0.0], [0, 1.0], [-1.5, 0.5], [2.0, -1.0]])[:n_trans]

    projections = []
    for i in range(n_rot):
        rot, tilt, psi = orientations[i]
        A = euler_angles_to_matrix(float(rot), float(tilt), float(psi))
        p = project_volume(
            vol,
            A,
            ori_size=N,
            padding_factor=2,
            interpolator=TRILINEAR,
            current_size=-1,
            do_gridding=True,
        )
        projections.append(p)
    projections = np.array(projections)

    img = ctf * projections[3 % n_rot] + rng.standard_normal(projections[0].shape) * 0.01

    # RELION's Minvsigma2 = 1/(2*sigma2)
    Minvsigma2_2d = np.zeros_like(ctf)
    for sh in range(n_shells):
        mask = shell_idx == sh
        if sigma2[sh] > 0:
            Minvsigma2_2d[mask] = 1.0 / (2.0 * sigma2[sh])

    # recovar's 1/sigma2 convention
    inv_sigma2_2d = np.zeros_like(ctf)
    for sh in range(n_shells):
        mask = shell_idx == sh
        if sigma2[sh] > 0:
            inv_sigma2_2d[mask] = 1.0 / sigma2[sh]

    return dict(
        N=N,
        vol=vol,
        ctf=ctf,
        shell_idx=shell_idx,
        sigma2=sigma2,
        orientations=orientations,
        translations=translations,
        projections=projections,
        img=img,
        n_rot=n_rot,
        n_trans=n_trans,
        n_shells=n_shells,
        Minvsigma2_2d=Minvsigma2_2d,
        inv_sigma2_2d=inv_sigma2_2d,
    )


class TestScoringConventions:
    """Verify the algebraic relationships between RELION and recovar scoring."""

    def test_relion_diff2_convention(self):
        """RELION diff2 uses Minvsigma2 = 1/(2*sigma2).
        diff2 = sum_{half} (1/(2*sigma2)) * |res|^2
        posterior ∝ exp(-diff2) = exp(-sum/(2*sigma2))
        """
        s = _setup_scenario()
        N = s["N"]

        diff2_relion = np.zeros((s["n_rot"], s["n_trans"]))
        for irot in range(s["n_rot"]):
            for itrans in range(s["n_trans"]):
                shifted = shift_image_in_fourier_transform_2d(
                    s["img"],
                    float(N),
                    N,
                    s["translations"][itrans, 0],
                    s["translations"][itrans, 1],
                )
                residual = shifted - s["ctf"] * s["projections"][irot]
                diff2_relion[irot, itrans] = np.sum(s["Minvsigma2_2d"] * np.abs(residual) ** 2)

        # Also compute with recovar's convention: 1/sigma2
        diff2_recovar = np.zeros((s["n_rot"], s["n_trans"]))
        for irot in range(s["n_rot"]):
            for itrans in range(s["n_trans"]):
                shifted = shift_image_in_fourier_transform_2d(
                    s["img"],
                    float(N),
                    N,
                    s["translations"][itrans, 0],
                    s["translations"][itrans, 1],
                )
                residual = shifted - s["ctf"] * s["projections"][irot]
                diff2_recovar[irot, itrans] = np.sum(s["inv_sigma2_2d"] * np.abs(residual) ** 2)

        # diff2_recovar = 2 * diff2_relion (exact)
        ratio = diff2_recovar / (diff2_relion + 1e-30)
        assert np.allclose(ratio, 2.0, rtol=1e-14), f"Convention ratio: {ratio.mean():.6f}"
        print(f"\nConvention: diff2_recovar/diff2_relion = {ratio.mean():.15f} (should be 2.0)")

    def test_all1_weights_recover_half_diff2(self):
        """With all-1 weights on half-spectrum, the GEMM formula gives:
        score = -0.5 * (diff2_recovar_convention - batch_norm_half)
        And since diff2_recovar = 2 * diff2_relion:
        posterior ∝ exp(score) = exp(-diff2_relion) * exp(const) ∝ exp(-diff2_relion)
        """
        s = _setup_scenario()
        N = s["N"]

        # Compute RELION diff2 (with Minvsigma2)
        diff2_relion = np.zeros((s["n_rot"], s["n_trans"]))
        for irot in range(s["n_rot"]):
            for itrans in range(s["n_trans"]):
                shifted = shift_image_in_fourier_transform_2d(
                    s["img"],
                    float(N),
                    N,
                    s["translations"][itrans, 0],
                    s["translations"][itrans, 1],
                )
                residual = shifted - s["ctf"] * s["projections"][irot]
                diff2_relion[irot, itrans] = np.sum(s["Minvsigma2_2d"] * np.abs(residual) ** 2)

        # Compute recovar GEMM scores with all-1 weights
        scores = np.zeros((s["n_rot"], s["n_trans"]))
        for irot in range(s["n_rot"]):
            proj_half = _fftw_to_half_flat(s["projections"][irot], N)
            ctf2_nv_half = _fftw_to_half_flat(s["ctf"] ** 2 * s["inv_sigma2_2d"], N)
            for itrans in range(s["n_trans"]):
                shifted = shift_image_in_fourier_transform_2d(
                    s["img"],
                    float(N),
                    N,
                    s["translations"][itrans, 0],
                    s["translations"][itrans, 1],
                )
                premult_half = _fftw_to_half_flat(s["ctf"] * shifted * s["inv_sigma2_2d"], N)
                cross = -2.0 * np.real(np.conj(premult_half) @ proj_half)  # w=1
                norm = ctf2_nv_half @ np.abs(proj_half) ** 2  # w=1
                scores[irot, itrans] = -0.5 * (cross + norm)

        # score = -0.5 * (diff2_recovar - batch_norm_half)
        # diff2_recovar = 2 * diff2_relion
        # So: score = -diff2_relion + 0.5*batch_norm_half
        # And: -score + 0.5*batch_norm_half = diff2_relion

        # Compute batch_norm_half (constant over translations — Parseval)
        batch_norm_half = np.sum(s["inv_sigma2_2d"] * np.abs(s["img"]) ** 2)

        diff2_from_scores = 0.5 * batch_norm_half - scores
        err = np.max(np.abs(diff2_relion - diff2_from_scores))
        rel = err / (np.max(np.abs(diff2_relion)) + 1e-30)
        print(f"\ndiff2_relion vs (batch_norm/2 - score): rel_err={rel:.2e}")
        assert rel < 1e-12, f"Score→diff2 conversion error: {rel:.2e}"


class TestPosteriorParity:
    """The definitive test: do RELION and recovar produce the same posteriors?"""

    def test_posteriors_match_with_all1_weights(self):
        """With all-1 half weights and correct conventions, posteriors must match."""
        s = _setup_scenario()
        N = s["N"]

        # --- RELION: diff2 with Minvsigma2, then E5 binding ---
        diff2_relion = np.zeros((s["n_rot"], s["n_trans"]))
        for irot in range(s["n_rot"]):
            for itrans in range(s["n_trans"]):
                shifted = shift_image_in_fourier_transform_2d(
                    s["img"],
                    float(N),
                    N,
                    s["translations"][itrans, 0],
                    s["translations"][itrans, 1],
                )
                residual = shifted - s["ctf"] * s["projections"][irot]
                diff2_relion[irot, itrans] = np.sum(s["Minvsigma2_2d"] * np.abs(residual) ** 2)

        orient_prior = np.ones(s["n_rot"])
        offset_prior = np.ones(s["n_trans"])
        posteriors_relion = convert_squared_differences_to_weights(
            diff2_relion,
            orient_prior,
            offset_prior,
            diff2_relion.min(),
        )

        # --- recovar: GEMM scores with all-1 weights, then softmax ---
        scores = np.zeros((s["n_rot"], s["n_trans"]))
        for irot in range(s["n_rot"]):
            proj_half = _fftw_to_half_flat(s["projections"][irot], N)
            ctf2_nv_half = _fftw_to_half_flat(s["ctf"] ** 2 * s["inv_sigma2_2d"], N)
            for itrans in range(s["n_trans"]):
                shifted = shift_image_in_fourier_transform_2d(
                    s["img"],
                    float(N),
                    N,
                    s["translations"][itrans, 0],
                    s["translations"][itrans, 1],
                )
                premult_half = _fftw_to_half_flat(s["ctf"] * shifted * s["inv_sigma2_2d"], N)
                cross = -2.0 * np.real(np.conj(premult_half) @ proj_half)
                norm = ctf2_nv_half @ np.abs(proj_half) ** 2
                scores[irot, itrans] = -0.5 * (cross + norm)

        scores_flat = scores.reshape(-1)
        max_s = scores_flat.max()
        log_Z = max_s + np.log(np.sum(np.exp(scores_flat - max_s)))
        posteriors_recovar = np.exp(scores - log_Z)

        max_diff = np.max(np.abs(posteriors_relion - posteriors_recovar))
        print("\nPosterior parity (all-1 weights, correct conv):")
        print(f"  max_diff: {max_diff:.2e}")
        print(f"  argmax RELION: {np.unravel_index(np.argmax(posteriors_relion), (s['n_rot'], s['n_trans']))}")
        print(f"  argmax recovar: {np.unravel_index(np.argmax(posteriors_recovar), (s['n_rot'], s['n_trans']))}")
        assert max_diff < 1e-14, f"Posterior mismatch: {max_diff:.2e}"

    def test_engine_kernel_posteriors_match(self):
        """Full JAX kernel path → same posteriors as RELION E5."""
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        from recovar.em.dense_single_volume.engine_v2 import _e_step_block_scores

        s = _setup_scenario()
        N = s["N"]
        image_shape = (N, N)
        volume_shape = (N, N, N)

        # RELION posteriors
        diff2_relion = np.zeros((s["n_rot"], s["n_trans"]))
        for irot in range(s["n_rot"]):
            for itrans in range(s["n_trans"]):
                shifted = shift_image_in_fourier_transform_2d(
                    s["img"],
                    float(N),
                    N,
                    s["translations"][itrans, 0],
                    s["translations"][itrans, 1],
                )
                residual = shifted - s["ctf"] * s["projections"][irot]
                diff2_relion[irot, itrans] = np.sum(s["Minvsigma2_2d"] * np.abs(residual) ** 2)

        posteriors_relion = convert_squared_differences_to_weights(
            diff2_relion,
            np.ones(s["n_rot"]),
            np.ones(s["n_trans"]),
            diff2_relion.min(),
        )

        # recovar JAX kernel with all-1 half weights
        hw = jnp.ones(N * (N // 2 + 1), dtype=jnp.float64)

        shifted_imgs = np.zeros((s["n_trans"], N * (N // 2 + 1)), dtype=np.complex128)
        for itrans in range(s["n_trans"]):
            shifted = shift_image_in_fourier_transform_2d(
                s["img"],
                float(N),
                N,
                s["translations"][itrans, 0],
                s["translations"][itrans, 1],
            )
            shifted_imgs[itrans] = _fftw_to_half_flat(s["ctf"] * shifted * s["inv_sigma2_2d"], N)

        shifted_jax = jnp.asarray(shifted_imgs)
        batch_norm_jax = jnp.zeros((1, 1))  # unused by kernel
        ctf2_nv_half = jnp.asarray(_fftw_to_half_flat(s["ctf"] ** 2 * s["inv_sigma2_2d"], N))[None, :]

        proj_half = jnp.asarray(np.array([_fftw_to_half_flat(p, N) for p in s["projections"]]))
        proj_abs2 = jnp.abs(proj_half) ** 2

        scores = np.asarray(
            _e_step_block_scores(
                shifted_jax,
                batch_norm_jax,
                ctf2_nv_half,
                proj_half * hw,
                proj_abs2 * hw,
                hw,
                1,
                s["n_trans"],
                image_shape,
                volume_shape,
            )
        )[0]

        scores_flat = scores.reshape(-1)
        max_s = scores_flat.max()
        log_Z = max_s + np.log(np.sum(np.exp(scores_flat - max_s)))
        posteriors_recovar = np.exp(scores - log_Z)

        max_diff = np.max(np.abs(posteriors_relion - posteriors_recovar))
        print("\nJAX kernel posterior parity:")
        print(f"  max_diff: {max_diff:.2e}")
        assert max_diff < 1e-12, f"JAX kernel posterior mismatch: {max_diff:.2e}"

    def test_hermitian_weights_are_too_peaked(self):
        """With Hermitian weights, posteriors are ~2x more peaked than RELION — wrong."""
        s = _setup_scenario()
        N = s["N"]

        # RELION posteriors
        diff2_relion = np.zeros((s["n_rot"], s["n_trans"]))
        for irot in range(s["n_rot"]):
            for itrans in range(s["n_trans"]):
                shifted = shift_image_in_fourier_transform_2d(
                    s["img"],
                    float(N),
                    N,
                    s["translations"][itrans, 0],
                    s["translations"][itrans, 1],
                )
                residual = shifted - s["ctf"] * s["projections"][irot]
                diff2_relion[irot, itrans] = np.sum(s["Minvsigma2_2d"] * np.abs(residual) ** 2)

        posteriors_relion = convert_squared_differences_to_weights(
            diff2_relion,
            np.ones(s["n_rot"]),
            np.ones(s["n_trans"]),
            diff2_relion.min(),
        )

        # recovar with Hermitian weights
        W_2d = 2.0 * np.ones((N, N // 2 + 1))
        W_2d[:, 0] = 1.0
        W_2d[:, -1] = 1.0
        W_half = W_2d.reshape(-1)

        scores = np.zeros((s["n_rot"], s["n_trans"]))
        for irot in range(s["n_rot"]):
            proj_half = _fftw_to_half_flat(s["projections"][irot], N)
            ctf2_nv_half = _fftw_to_half_flat(s["ctf"] ** 2 * s["inv_sigma2_2d"], N)
            for itrans in range(s["n_trans"]):
                shifted = shift_image_in_fourier_transform_2d(
                    s["img"],
                    float(N),
                    N,
                    s["translations"][itrans, 0],
                    s["translations"][itrans, 1],
                )
                premult_half = _fftw_to_half_flat(s["ctf"] * shifted * s["inv_sigma2_2d"], N)
                cross = -2.0 * np.real(np.conj(premult_half) @ (proj_half * W_half))
                norm = ctf2_nv_half @ (np.abs(proj_half) ** 2 * W_half)
                scores[irot, itrans] = -0.5 * (cross + norm)

        scores_flat = scores.reshape(-1)
        max_s = scores_flat.max()
        log_Z = max_s + np.log(np.sum(np.exp(scores_flat - max_s)))
        posteriors_herm = np.exp(scores - log_Z)

        entropy_relion = -np.sum(posteriors_relion * np.log(posteriors_relion + 1e-30))
        entropy_herm = -np.sum(posteriors_herm * np.log(posteriors_herm + 1e-30))

        print("\nHermitian vs RELION:")
        print(f"  entropy RELION: {entropy_relion:.6f}")
        print(f"  entropy Hermitian: {entropy_herm:.6f}")
        print(f"  Hermitian more peaked: {entropy_herm < entropy_relion}")
        print(f"  max posterior diff: {np.max(np.abs(posteriors_relion - posteriors_herm)):.6f}")

        # Hermitian weights should make the posterior MORE peaked (lower entropy)
        assert entropy_herm < entropy_relion, "Hermitian should be more peaked"

    def test_dc_exclusion_matches_relion(self):
        """RELION sets Minvsigma2[0]=0 (excludes DC from scoring).
        recovar with half_spectrum_scoring=True zeros DC in shifted/ctf arrays.
        Both should produce identical posteriors.
        """
        s = _setup_scenario()
        N = s["N"]

        # RELION with DC excluded: Minvsigma2[DC shell] = 0
        Minv_noDC = s["Minvsigma2_2d"].copy()
        dc_mask = s["shell_idx"] == 0
        Minv_noDC[dc_mask] = 0.0

        diff2_relion = np.zeros((s["n_rot"], s["n_trans"]))
        for irot in range(s["n_rot"]):
            for itrans in range(s["n_trans"]):
                shifted = shift_image_in_fourier_transform_2d(
                    s["img"],
                    float(N),
                    N,
                    s["translations"][itrans, 0],
                    s["translations"][itrans, 1],
                )
                residual = shifted - s["ctf"] * s["projections"][irot]
                diff2_relion[irot, itrans] = np.sum(Minv_noDC * np.abs(residual) ** 2)

        posteriors_relion = convert_squared_differences_to_weights(
            diff2_relion,
            np.ones(s["n_rot"]),
            np.ones(s["n_trans"]),
            diff2_relion.min(),
        )

        # recovar with DC zeroed in preprocessing (matching engine_v2 DC exclusion)
        inv_s2_noDC = s["inv_sigma2_2d"].copy()
        inv_s2_noDC[dc_mask] = 0.0

        scores = np.zeros((s["n_rot"], s["n_trans"]))
        for irot in range(s["n_rot"]):
            proj_half = _fftw_to_half_flat(s["projections"][irot], N)
            ctf2_nv_half = _fftw_to_half_flat(s["ctf"] ** 2 * inv_s2_noDC, N)
            for itrans in range(s["n_trans"]):
                shifted = shift_image_in_fourier_transform_2d(
                    s["img"],
                    float(N),
                    N,
                    s["translations"][itrans, 0],
                    s["translations"][itrans, 1],
                )
                premult_half = _fftw_to_half_flat(s["ctf"] * shifted * inv_s2_noDC, N)
                cross = -2.0 * np.real(np.conj(premult_half) @ proj_half)
                norm = ctf2_nv_half @ np.abs(proj_half) ** 2
                scores[irot, itrans] = -0.5 * (cross + norm)

        scores_flat = scores.reshape(-1)
        max_s = scores_flat.max()
        log_Z = max_s + np.log(np.sum(np.exp(scores_flat - max_s)))
        posteriors_recovar = np.exp(scores - log_Z)

        max_diff = np.max(np.abs(posteriors_relion - posteriors_recovar))
        print(f"\nDC exclusion parity: max_diff={max_diff:.2e}")
        assert max_diff < 1e-14, f"DC exclusion mismatch: {max_diff:.2e}"


class TestWindowingDivergence:
    """Document the known windowing divergence (item #10 in audit)."""

    def test_pixel_count_difference(self):
        """RELION uses rectangular crop, recovar uses radial mask."""
        N = 128
        current_size = 64
        r_max = current_size // 2

        # RELION: rectangular crop to (current_size, current_size//2+1)
        relion_pixels = current_size * (current_size // 2 + 1)

        # recovar: radial mask on (N, N//2+1) half-spectrum
        from recovar.em.dense_single_volume.refine_dev_helpers.fourier_window import make_fourier_window_indices_np

        _, n_windowed = make_fourier_window_indices_np((N, N), current_size)

        print(f"\nWindowing at current_size={current_size} (N={N}):")
        print(f"  RELION rectangular: {relion_pixels} pixels")
        print(f"  recovar radial: {n_windowed} pixels")
        print(
            f"  Difference: {abs(relion_pixels - n_windowed)} ({100 * abs(relion_pixels - n_windowed) / relion_pixels:.1f}%)"
        )

        # Both should include similar number of pixels
        # RELION's rectangular includes corners beyond r_max;
        # recovar's radial excludes corners but stays within r_max
        assert n_windowed < relion_pixels, "Radial mask should have fewer pixels than rectangular"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

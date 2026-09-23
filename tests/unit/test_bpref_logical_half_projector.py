"""Native BPref logical-half input preserves full-cube texture coordinates."""
import jax
import numpy as np
import pytest

from recovar import cuda_backproject as cb
from recovar.em.cuda import kernels as em_cuda_kernels
from test_bpref_optional_denominator import arguments, device

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('shape', [(9, 9, 4), (11, 11, 6), (8, 8, 5)])
def test_invalid_logical_half_rejected_before_cuda(monkeypatch, shape):
    monkeypatch.setattr(cb, '_ensure_ffi', lambda: pytest.fail('unexpected CUDA load'))
    monkeypatch.setattr(em_cuda_kernels, '_ensure_ffi', lambda: pytest.fail('unexpected CUDA load'))
    values = arguments(False, False)
    values['projector_full'] = np.zeros(shape, np.complex64)
    with pytest.raises(TypeError, match='radius-matched logical half slab'):
        em_cuda_kernels.relion_vdam_mstep_fused_projector_x_half.__wrapped__(**device(values))


def test_logical_half_external_host_replay_rejected(monkeypatch):
    monkeypatch.setenv('RECOVAR_VDAM_EXTERNAL_HOST_REPLAY_LIBRARY', '/nonexistent')
    monkeypatch.setattr(cb, '_ensure_ffi', lambda: pytest.fail('unexpected CUDA load'))
    monkeypatch.setattr(em_cuda_kernels, '_ensure_ffi', lambda: pytest.fail('unexpected CUDA load'))
    values = arguments(False, False)
    values['projector_full'] = np.zeros((9, 9, 5), np.complex64)
    with pytest.raises(ValueError, match='requires a full projector cube'):
        em_cuda_kernels.relion_vdam_mstep_fused_projector_x_half.__wrapped__(**device(values))


@pytest.mark.gpu
@pytest.mark.parametrize('padding', [1, 2])
@pytest.mark.parametrize('grouped,stable', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('denominator', [False, True])
def test_logical_half_matches_full_cube_bitwise(padding, grouped, stable, denominator):
    assert jax.default_backend() == 'gpu'
    values = arguments(grouped, stable)
    pad = 2 * 3 * padding + 3
    rng = np.random.default_rng(871)
    half = (rng.normal(size=(pad, pad, pad // 2 + 1))
            + 1j * rng.normal(size=(pad, pad, pad // 2 + 1))).astype(np.complex64)
    # Independent host construction of the established full-cube layout.
    cube = np.zeros((pad, pad, pad), np.complex64)
    cube[pad // 2:, :, :] = half.transpose(2, 1, 0)
    values['projector_full'] = cube
    values['projection_padding_factor'] = padding
    # Fractional, asymmetric sampling; a negative x component exercises conjugation.
    angle = np.float32(0.37)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[-c, s, 0], [s, c, 0], [0, 0, -1]], np.float32)
    values['rotation_matrices'][:] = rotation
    values['translation_angles'][:] = (0.13, -0.27)
    fn = em_cuda_kernels.relion_vdam_mstep_fused_projector_x_half
    full = jax.block_until_ready(fn(**device(values), return_denominator=denominator))
    values['projector_full'] = half
    direct = jax.block_until_ready(fn(**device(values), return_denominator=denominator))
    for old, new in zip(full, direct, strict=True):
        if old is None:
            assert new is None
        else:
            assert np.isfinite(np.asarray(new)).all()
            assert np.asarray(old).tobytes() == np.asarray(new).tobytes()


@pytest.mark.gpu
@pytest.mark.parametrize('grouped,stable', [(False, False), (True, False), (True, True)])
def test_local_physical_carry_consumes_inputs_and_preserves_results(grouped, stable):
    from recovar.em.local.local_physical_grid import _accumulate_relion_vdam_physical_particle_grid

    assert jax.default_backend() == 'gpu'
    values = arguments(grouped, stable)
    rng = np.random.default_rng(177)
    values['projector_full'] = (rng.normal(size=(9, 9, 5))
        + 1j * rng.normal(size=(9, 9, 5))).astype(np.complex64)
    expected = jax.block_until_ready(em_cuda_kernels.relion_vdam_mstep_fused_projector_x_half(
        **device(values), parallel_worker_replay=False))
    v = device(values)
    data, weight = v['data_volume'], v['weight_volume']
    actual = jax.block_until_ready(_accumulate_relion_vdam_physical_particle_grid(
        v['images'], v['ctf'], v['minvsigma2'], v['posterior_over_weight_norm'],
        v['translation_angles'], None, v['rotation_matrices'],
        np.ones(v['rotation_matrices'].shape[:2], bool), data, weight,
        projector_full=v['projector_full'], scoring_rotations=v['rotation_matrices'],
        projector_r_max=v['projector_max_r'], projection_padding_factor=1,
        pixel_indices=v['pixel_indices'], image_shape=v['image_shape'],
        volume_shape=v['volume_shape'], max_r=v['max_r'],
        stable_dense_positions=v.get('stable_dense_positions'),
        logical_current_size=v.get('logical_current_size'),
        reconstruction_group_ids=v.get('reconstruction_group_ids'),
        serial_particle_accumulation=True, consume_accumulators=True,
    ))
    assert data.is_deleted() and weight.is_deleted()
    for old, new in zip(expected[:2], actual, strict=True):
        assert np.asarray(old).tobytes() == np.asarray(new).tobytes()


@pytest.mark.gpu
def test_complete_local_engine_consuming_carry_matches_full_cube(monkeypatch, tmp_path):
    """Two buckets and three translations cover mask broadcasting and carry reuse."""
    from recovar.em.helpers.projection import relion_projector_half_to_texture_full
    from recovar.em.local.local_em_engine import run_local_em_exact
    from recovar.em.local.local_layout import LocalHypothesisLayout
    from test_refine_relion_mode import RawRealImageDataset

    assert jax.default_backend() == 'gpu'
    dataset = RawRealImageDataset(2, np.random.default_rng(412))
    dataset.image_source.backend.relion_fourier_backend = 'relion_cuda'
    dataset.image_source.backend.image_mask_mode = 'relion_background_fill'
    dataset.image_source.backend._relion_image_mask_params = (1.0, 4.0, 1.0)
    star = tmp_path / 'ctf.star'
    star.write_text('''data_optics

loop_
_rlnOpticsGroup #1
_rlnVoltage #2
_rlnSphericalAberration #3
_rlnAmplitudeContrast #4
_rlnImagePixelSize #5
1 300 2.7 0.1 1.0

data_particles

loop_
_rlnOpticsGroup #1
_rlnDefocusU #2
_rlnDefocusV #3
_rlnDefocusAngle #4
_rlnPhaseShift #5
1 12000 13000 17 0
1 15000 14000 31 0
''')
    dataset.particles_file = str(star)
    monkeypatch.delenv('RECOVAR_K1_RELION_EXACT_CTF_STAR', raising=False)
    monkeypatch.delenv('RECOVAR_DISABLE_LOCAL_BIG_JIT', raising=False)
    monkeypatch.setenv('RECOVAR_EXACT_LOCAL_BIG_JIT_MAX_BUCKET_ROTATIONS', '64')
    monkeypatch.setenv('RECOVAR_EXACT_LOCAL_RELION_PROJECTION_CACHE_MAX_GB', '0')
    layout = LocalHypothesisLayout(
        n_global_rotations=48, n_pixels=48, n_psi=1,
        rotation_offsets=np.asarray([0, 16, 48], np.int64),
        rotation_ids_flat=np.arange(48, dtype=np.int32),
        rotations_flat=np.broadcast_to(np.eye(3, dtype=np.float32), (48, 3, 3)).copy(),
        rotation_log_priors_flat=np.zeros(48, np.float32),
        rotation_counts=np.asarray([16, 32], np.int32),
        translation_grid=np.asarray([[-1, 0], [0, 0], [1, 0]], np.float32),
        translation_log_priors=np.zeros((2, 3), np.float32),
    )
    slab = np.zeros((11, 11, 6), np.complex64)
    slab[5, 5, 0] = 1
    slab[6, 5, 1] = .125 + .25j
    kwargs = dict(
        image_batch_size=1, rotation_block_size=32, current_size=4,
        accumulate_noise=True, projection_padding_factor=2, reconstruction_padding_factor=2,
        score_with_masked_images=False, half_spectrum_scoring=True,
        use_float64_scoring=False, use_float64_projections=False,
        projection_relion_texture_interp=True, relion_projector_half=slab,
        relion_projector_r_max=2, mstep_relion_x_half=True, return_profile=True,
        reconstruct_significant_only=True, adaptive_fraction=.999, max_significants=-1,
        relion_exact_fine_diff2=True, relion_exact_score_translation=True,
        relion_exact_bpref_operands=True, relion_wavg_sequential_cuda=True,
        preserve_bpref_particle_order=True, mstep_subtract_ctf_projection=True,
    )
    consume = em_cuda_kernels._relion_vdam_mstep_fused_projector_x_half_consume
    calls = []

    def observed(*args, **options):
        assert args[8].shape == slab.shape
        assert options['return_denominator'] is False
        if reference_mode:
            args = list(args)
            args[8] = relion_projector_half_to_texture_full(args[8])
            result = em_cuda_kernels.relion_vdam_mstep_fused_projector_x_half(*args, **options)
        else:
            result = consume(*args, **options)
            assert args[0].is_deleted() and args[1].is_deleted()
        calls.append(reference_mode)
        return result

    monkeypatch.setattr(em_cuda_kernels, '_relion_vdam_mstep_fused_projector_x_half_consume', observed)
    expected = None
    for reference_mode in (True, False):
        out = run_local_em_exact(
            dataset, np.zeros(1, np.complex64), np.ones(64, np.float32), layout,
            'linear_interp', **kwargs,
        )
        assert int(out.profile['big_jit_bucket_count']) == 2
        assert out.Ft_y.dtype == np.complex64 and out.Ft_ctf.dtype == np.float32
        actual = {name: np.asarray(getattr(out, name)).copy()
                  for name in ('Ft_y', 'Ft_ctf', 'hard_assignments')}
        if reference_mode:
            expected = actual
        else:
            for name, value in actual.items():
                np.testing.assert_array_equal(value, expected[name])
    assert calls == [True, True, False, False]

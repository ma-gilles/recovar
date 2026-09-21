"""Class-segmented projection: each class segment is projected from its own volume."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.local.local_big_jit import (
    _class_volume,
    _project_local_class_segments,
    _project_local_class_segments_flat,
    _project_local_half_spectrum,
)

pytestmark = pytest.mark.unit


def _rotations(rng, batch_size, rows):
    q = rng.standard_normal((batch_size, rows, 3, 3))
    # Orthonormalize so the rotations are valid for the projector.
    u, _, vt = np.linalg.svd(q)
    r = u @ vt
    flip = np.linalg.det(r) < 0
    r[flip, :, 0] *= -1.0
    return jnp.asarray(r, dtype=jnp.float32)


def _volumes(rng, n_classes, volume_shape):
    size = int(np.prod(volume_shape))
    return jnp.asarray(
        rng.standard_normal((n_classes, size)) + 1j * rng.standard_normal((n_classes, size)),
        dtype=jnp.complex64,
    )


def _project(volume, rotations, image_shape, volume_shape):
    return _project_local_half_spectrum(
        volume,
        None,
        rotations,
        None,
        image_shape,
        volume_shape,
        "linear_interp",
        projection_half_volume=False,
        projection_max_r=None,
        relion_projector_output_size=0,
        projection_relion_texture_interp=False,
        projection_force_jax=True,
        use_relion_projector=False,
        relion_projector_r_max=0,
        projection_padding_factor=1,
    )


@pytest.mark.parametrize("n_classes", [1, 2, 3])
def test_class_segments_match_projecting_each_class_separately(n_classes):
    rng = np.random.default_rng(19)
    image_shape, volume_shape = (8, 8), (8, 8, 8)
    batch_size, seg = 3, 4
    rows = n_classes * seg
    local_rotations = _rotations(rng, batch_size, rows)
    volumes = _volumes(rng, n_classes, volume_shape)

    segmented = _project_local_class_segments(
        volumes,
        None,
        local_rotations,
        lambda volume, projector_half, rotations: _project(volume, rotations, image_shape, volume_shape),
        n_classes=n_classes,
        segment_rotation_count=seg,
    )
    assert segmented.shape[0] == batch_size * rows

    per_row = np.asarray(segmented).reshape(batch_size, rows, -1)
    for k in range(n_classes):
        expected = _project(
            volumes[k],
            local_rotations[:, k * seg : (k + 1) * seg].reshape(batch_size * seg, 3, 3),
            image_shape,
            volume_shape,
        )
        np.testing.assert_array_equal(
            per_row[:, k * seg : (k + 1) * seg].reshape(batch_size * seg, -1),
            np.asarray(expected),
        )


def test_one_class_reproduces_the_unsegmented_projection_exactly():
    rng = np.random.default_rng(23)
    image_shape, volume_shape = (8, 8), (8, 8, 8)
    batch_size, rows = 4, 6
    local_rotations = _rotations(rng, batch_size, rows)
    volume = _volumes(rng, 1, volume_shape)

    segmented = _project_local_class_segments(
        volume,
        None,
        local_rotations,
        lambda v, _p, r: _project(v, r, image_shape, volume_shape),
        n_classes=1,
        segment_rotation_count=rows,
    )
    flat = _project(volume[0], local_rotations.reshape(batch_size * rows, 3, 3), image_shape, volume_shape)
    np.testing.assert_array_equal(np.asarray(segmented), np.asarray(flat))


def test_segment_placement_is_class_major_and_row_preserving():
    """A projector that reports (class marker, rotation trace) pins where each row lands."""
    rng = np.random.default_rng(29)
    batch_size, seg, n_classes = 2, 3, 4
    rows = n_classes * seg
    local_rotations = _rotations(rng, batch_size, rows)
    markers = jnp.asarray(np.arange(n_classes, dtype=np.float32)[:, None] + 100.0)

    def fake_project(volume, _projector_half, rotations):
        traces = jnp.trace(rotations, axis1=1, axis2=2)[:, None]
        return jnp.concatenate([jnp.broadcast_to(volume, (rotations.shape[0], 1)), traces], axis=1)

    out = np.asarray(
        _project_local_class_segments(
            markers, None, local_rotations, fake_project,
            n_classes=n_classes, segment_rotation_count=seg,
        )
    ).reshape(batch_size, rows, 2)
    expected_traces = np.asarray(jnp.trace(local_rotations, axis1=2, axis2=3))
    for k in range(n_classes):
        np.testing.assert_array_equal(out[:, k * seg : (k + 1) * seg, 0], np.full((batch_size, seg), 100.0 + k))
    np.testing.assert_allclose(out[:, :, 1], expected_traces, rtol=0, atol=0)


def test_class_volume_rejects_a_mismatched_stack():
    volumes = jnp.zeros((2, 5))
    assert _class_volume(None, 0, 2) is None
    # The unused placeholder operand passes through untouched.
    placeholder = jnp.zeros((1, 1, 1))
    assert _class_volume(placeholder, 0, 2, per_class=False) is placeholder
    np.testing.assert_array_equal(np.asarray(_class_volume(volumes, 1, 2)), np.zeros(5))
    with pytest.raises(ValueError, match="leading class axis"):
        _class_volume(volumes, 0, 3)


def test_segmented_rows_must_factor_into_classes():
    rng = np.random.default_rng(31)
    local_rotations = _rotations(rng, 2, 7)
    with pytest.raises(ValueError, match="class-segmented rows must be"):
        _project_local_class_segments(
            jnp.zeros((2, 4)), None, local_rotations, lambda v, p, r: r.reshape(r.shape[0], -1),
            n_classes=2, segment_rotation_count=3,
        )


def _adjoint_inputs(rng, batch_size, rows, n_pixels, volume_size):
    return (
        jnp.asarray(rng.standard_normal((batch_size * rows, n_pixels))
                    + 1j * rng.standard_normal((batch_size * rows, n_pixels)), dtype=jnp.complex64),
        jnp.asarray(rng.standard_normal((batch_size * rows, n_pixels)), dtype=jnp.float32),
        jnp.zeros((volume_size,), dtype=jnp.complex64),
        jnp.zeros((volume_size,), dtype=jnp.float32),
    )


@pytest.mark.parametrize("n_classes", [2, 3])
def test_class_segment_adjoint_accumulates_each_class_into_its_own_volume(n_classes):
    from recovar.em.local.local_big_jit import (
        _adjoint_local_class_segments,
        _adjoint_local_mstep_volumes,
    )

    rng = np.random.default_rng(37)
    image_shape, volume_shape = (8, 8), (8, 8, 8)
    n_pixels = image_shape[0] * (image_shape[0] // 2 + 1)
    # The local M-step accumulates into half-spectrum volumes.
    volume_size = volume_shape[0] * volume_shape[1] * (volume_shape[2] // 2 + 1)
    batch_size, seg = 2, 3
    rows = n_classes * seg
    summed, ctf_probs, zero_y, zero_ctf = _adjoint_inputs(rng, batch_size, rows, n_pixels, volume_size)
    rotations = _rotations(rng, batch_size, rows).reshape(batch_size * rows, 3, 3)
    # Start from distinct per-class accumulators so a misrouted segment is visible.
    start_y = jnp.stack([zero_y + (k + 1) for k in range(n_classes)])
    start_ctf = jnp.stack([zero_ctf + (k + 1) for k in range(n_classes)])

    got_y, got_ctf = _adjoint_local_class_segments(
        summed, ctf_probs, None, rotations, start_y, start_ctf,
        image_shape, volume_shape, "linear_interp",
        n_classes=n_classes, segment_rotation_count=seg, batch_size=batch_size,
        use_window=False, max_r=None, disable_adjoint_y=False, disable_adjoint_ctf=False,
        relion_x_half_mstep=False,
    )
    assert got_y.shape == (n_classes, volume_size) and got_ctf.shape == (n_classes, volume_size)

    summed_rows = summed.reshape(batch_size, rows, n_pixels)
    ctf_rows = ctf_probs.reshape(batch_size, rows, n_pixels)
    rot_rows = rotations.reshape(batch_size, rows, 3, 3)
    for k in range(n_classes):
        sl = slice(k * seg, (k + 1) * seg)
        want_y, want_ctf = _adjoint_local_mstep_volumes(
            summed_rows[:, sl].reshape(batch_size * seg, n_pixels),
            ctf_rows[:, sl].reshape(batch_size * seg, n_pixels),
            None,
            rot_rows[:, sl].reshape(batch_size * seg, 3, 3),
            start_y[k], start_ctf[k], image_shape, volume_shape, "linear_interp",
            use_window=False, max_r=None, disable_adjoint_y=False, disable_adjoint_ctf=False,
            relion_x_half_mstep=False,
        )
        np.testing.assert_array_equal(np.asarray(got_y[k]), np.asarray(want_y))
        np.testing.assert_array_equal(np.asarray(got_ctf[k]), np.asarray(want_ctf))


def test_class_segment_adjoint_keeps_classes_apart():
    """Rows of one class must not contribute to another class's volume."""
    from recovar.em.local.local_big_jit import _adjoint_local_class_segments

    rng = np.random.default_rng(41)
    image_shape, volume_shape = (8, 8), (8, 8, 8)
    n_pixels = image_shape[0] * (image_shape[0] // 2 + 1)
    volume_size = volume_shape[0] * volume_shape[1] * (volume_shape[2] // 2 + 1)
    batch_size, seg, n_classes = 2, 3, 2
    rows = n_classes * seg
    summed, ctf_probs, zero_y, zero_ctf = _adjoint_inputs(rng, batch_size, rows, n_pixels, volume_size)
    rotations = _rotations(rng, batch_size, rows).reshape(batch_size * rows, 3, 3)
    start_y = jnp.stack([zero_y] * n_classes)
    start_ctf = jnp.stack([zero_ctf] * n_classes)

    # Zero every row of class 1: its volume must stay exactly zero while class 0 fills.
    masked = np.asarray(summed).reshape(batch_size, rows, n_pixels).copy()
    masked[:, seg:] = 0.0
    masked_ctf = np.asarray(ctf_probs).reshape(batch_size, rows, n_pixels).copy()
    masked_ctf[:, seg:] = 0.0
    got_y, got_ctf = _adjoint_local_class_segments(
        jnp.asarray(masked.reshape(batch_size * rows, n_pixels)),
        jnp.asarray(masked_ctf.reshape(batch_size * rows, n_pixels)),
        None, rotations, start_y, start_ctf, image_shape, volume_shape, "linear_interp",
        n_classes=n_classes, segment_rotation_count=seg, batch_size=batch_size,
        use_window=False, max_r=None, disable_adjoint_y=False, disable_adjoint_ctf=False,
        relion_x_half_mstep=False,
    )
    assert np.abs(np.asarray(got_y[0])).max() > 0.0
    np.testing.assert_array_equal(np.asarray(got_y[1]), np.zeros(volume_size, dtype=np.complex64))
    np.testing.assert_array_equal(np.asarray(got_ctf[1]), np.zeros(volume_size, dtype=np.float32))


@pytest.mark.parametrize("n_classes", [2, 4])
def test_class_segment_statistics_match_per_class_reductions(n_classes):
    """Per-class evidence, best score and reconstruction mass come from one joint pass."""
    from recovar.em.local.local_big_jit import _class_segment_statistics

    rng = np.random.default_rng(53)
    batch_size, seg, n_trans = 3, 5, 4
    rows = n_classes * seg
    scores = jnp.asarray(rng.standard_normal((batch_size, rows, n_trans)) * 3.0, dtype=jnp.float32)
    log_z = jax.scipy.special.logsumexp(scores.reshape(batch_size, -1), axis=1)
    probs = jnp.exp(scores - log_z[:, None, None])
    recon = probs * jnp.asarray(rng.random((batch_size, rows, n_trans)) > 0.4, dtype=probs.dtype)

    mass, best, recon_mass, evidence, _winner = _class_segment_statistics(
        probs, scores, recon, n_classes=n_classes, segment_rotation_count=seg,
    )
    for k in range(n_classes):
        sl = slice(k * seg, (k + 1) * seg)
        np.testing.assert_allclose(np.asarray(mass[:, k]), np.asarray(jnp.sum(probs[:, sl], axis=(1, 2))), rtol=0, atol=0)
        np.testing.assert_allclose(np.asarray(best[:, k]), np.asarray(jnp.max(scores[:, sl], axis=(1, 2))), rtol=0, atol=0)
        np.testing.assert_allclose(np.asarray(recon_mass[:, k]), np.asarray(jnp.sum(recon[:, sl], axis=(1, 2))), rtol=0, atol=0)

    # The per-class log evidence is the direct per-class logsumexp.
    direct = np.stack([np.asarray(jax.scipy.special.logsumexp(
        scores[:, k * seg:(k + 1) * seg].reshape(batch_size, -1), axis=1)) for k in range(n_classes)], axis=1)
    np.testing.assert_allclose(np.asarray(evidence), direct, rtol=1e-5, atol=1e-5)
    # And the responsibilities are a partition of unity.
    np.testing.assert_allclose(np.asarray(mass).sum(axis=1), np.ones(batch_size), rtol=1e-6, atol=1e-6)


def test_class_segment_statistics_reject_rows_that_do_not_factor():
    from recovar.em.local.local_big_jit import _class_segment_statistics

    probs = jnp.zeros((2, 7, 3))
    with pytest.raises(ValueError, match="class-segmented rows must be"):
        _class_segment_statistics(probs, probs, None, n_classes=2, segment_rotation_count=3)


@pytest.mark.parametrize("use_float64", [True, False])
def test_class_evidence_survives_posterior_underflow(use_float64):
    """Four populated classes decades apart: masses underflow, evidence must not.

    Reported by the lead against the first implementation, which derived the
    per-class evidence from the joint posterior as log_Z + log(mass) and returned
    -inf for three populated classes in both precisions.
    """
    from recovar.em.local.local_big_jit import _class_segment_statistics

    n_classes, seg, n_trans = 4, 1, 1
    scores = jnp.asarray(np.array([[[0.0], [-1000.0], [-2000.0], [-3000.0]]]), dtype=jnp.float32)
    log_z = jax.scipy.special.logsumexp(scores.reshape(1, -1), axis=1)
    probs = jnp.exp(scores - log_z[:, None, None])

    mass, best, _recon, evidence, _winner = _class_segment_statistics(
        probs, scores, None, n_classes=n_classes, segment_rotation_count=seg,
        use_float64_normalization=use_float64,
    )
    # The masses do underflow; that is correct for a normalized responsibility.
    np.testing.assert_allclose(np.asarray(mass)[0], [1.0, 0.0, 0.0, 0.0], rtol=0, atol=0)
    # The evidence must still be the direct per-class logsumexp.
    np.testing.assert_allclose(np.asarray(evidence)[0], [0.0, -1000.0, -2000.0, -3000.0], rtol=1e-6, atol=1e-4)
    np.testing.assert_allclose(np.asarray(best)[0], [0.0, -1000.0, -2000.0, -3000.0], rtol=0, atol=0)
    assert np.isfinite(np.asarray(evidence)).all()


def test_empty_class_reports_minus_infinity_evidence():
    """A segment with no finite score is the genuinely empty class."""
    from recovar.em.local.local_big_jit import _class_segment_statistics

    scores = jnp.asarray(np.array([[[0.0, -2.0], [-np.inf, -np.inf], [-5.0, -7.0]]]), dtype=jnp.float32)
    probs = jnp.exp(scores - jax.scipy.special.logsumexp(scores.reshape(1, -1), axis=1)[:, None, None])
    probs = jnp.where(jnp.isfinite(probs), probs, 0.0)
    _mass, _best, _recon, evidence, _winner = _class_segment_statistics(
        probs, scores, None, n_classes=3, segment_rotation_count=1,
    )
    got = np.asarray(evidence)[0]
    assert np.isneginf(got[1])
    np.testing.assert_allclose(got[[0, 2]], [
        float(jax.scipy.special.logsumexp(scores[0, 0])),
        float(jax.scipy.special.logsumexp(scores[0, 2])),
    ], rtol=1e-6, atol=1e-5)


def test_class_evidence_matches_direct_logsumexp_on_wide_score_ranges():
    from recovar.em.local.local_big_jit import _class_segment_statistics

    rng = np.random.default_rng(67)
    batch_size, n_classes, seg, n_trans = 3, 4, 3, 2
    # Decade-scale class offsets on top of ordinary within-class spread.
    base = rng.standard_normal((batch_size, n_classes * seg, n_trans)) * 2.0
    offsets = np.repeat(np.array([0.0, -60.0, -400.0, -1500.0]), seg)[None, :, None]
    scores = jnp.asarray(base + offsets, dtype=jnp.float32)
    probs = jnp.exp(scores - jax.scipy.special.logsumexp(scores.reshape(batch_size, -1), axis=1)[:, None, None])
    _mass, _best, _recon, evidence, _winner = _class_segment_statistics(
        probs, scores, None, n_classes=n_classes, segment_rotation_count=seg,
    )
    direct = np.stack([
        np.asarray(jax.scipy.special.logsumexp(scores[:, k * seg:(k + 1) * seg].reshape(batch_size, -1), axis=1))
        for k in range(n_classes)
    ], axis=1)
    np.testing.assert_allclose(np.asarray(evidence), direct, rtol=1e-6, atol=1e-3)


@pytest.mark.parametrize("n_classes", [2, 3])
def test_class_winner_indexes_within_its_own_segment(n_classes):
    """Each class reports its own best (row, translation), as the per-class calls do."""
    from recovar.em.local.local_big_jit import _class_segment_statistics

    rng = np.random.default_rng(71)
    batch_size, seg, n_trans = 4, 3, 5
    rows = n_classes * seg
    scores = jnp.asarray(rng.standard_normal((batch_size, rows, n_trans)) * 4.0, dtype=jnp.float32)
    probs = jnp.exp(scores - jax.scipy.special.logsumexp(scores.reshape(batch_size, -1), axis=1)[:, None, None])

    _mass, best, _recon, _evidence, winner = _class_segment_statistics(
        probs, scores, None, n_classes=n_classes, segment_rotation_count=seg,
    )
    winner = np.asarray(winner)
    assert winner.shape == (batch_size, n_classes)
    for k in range(n_classes):
        segment = np.asarray(scores[:, k * seg:(k + 1) * seg]).reshape(batch_size, -1)
        np.testing.assert_array_equal(winner[:, k], segment.argmax(axis=1))
        # The winner's score is the reported per-class best.
        np.testing.assert_allclose(segment[np.arange(batch_size), winner[:, k]], np.asarray(best[:, k]), rtol=0, atol=0)
        # It decodes to a row inside class k and a valid translation.
        assert ((winner[:, k] // n_trans) < seg).all() and ((winner[:, k] % n_trans) < n_trans).all()


def test_joint_winner_is_the_best_class_winner():
    """The joint argmax and the per-class winners agree on which class wins."""
    from recovar.em.local.local_big_jit import _class_segment_statistics

    rng = np.random.default_rng(73)
    batch_size, n_classes, seg, n_trans = 5, 3, 2, 3
    scores = jnp.asarray(rng.standard_normal((batch_size, n_classes * seg, n_trans)) * 5.0, dtype=jnp.float32)
    probs = jnp.exp(scores - jax.scipy.special.logsumexp(scores.reshape(batch_size, -1), axis=1)[:, None, None])
    _mass, best, _recon, _evidence, winner = _class_segment_statistics(
        probs, scores, None, n_classes=n_classes, segment_rotation_count=seg,
    )
    joint_argmax = np.asarray(jnp.argmax(scores.reshape(batch_size, -1), axis=1))
    joint_class = (joint_argmax // n_trans) // seg
    np.testing.assert_array_equal(joint_class, np.asarray(best).argmax(axis=1))
    for image in range(batch_size):
        k = int(joint_class[image])
        row_in_segment = (int(joint_argmax[image]) // n_trans) - k * seg
        assert int(winner[image, k]) == row_in_segment * n_trans + int(joint_argmax[image]) % n_trans


def _tagging_project_rows(volume, projector_half, rotations):
    """Return each class volume's leading value, so a row reveals which volume made it."""
    tag = jnp.asarray(volume).reshape(-1)[0].real
    return jnp.full((int(rotations.shape[0]), 2), tag, dtype=jnp.float32)


def test_flat_class_segments_project_each_block_from_its_own_volume():
    """Packed class-major rows must not be projected from the wrong class volume.

    `local_big_jit` refuses flat rows at K>1 precisely to avoid "silently projecting a
    row from the wrong class". The flat plan is emitted class-major and reports
    `class_row_counts`, so each class's contiguous block can use its own volume.
    """
    n_classes = 3
    means = jnp.asarray(np.arange(n_classes, dtype=np.complex64).reshape(n_classes, 1) + 1)
    class_row_counts = (5, 2, 3)
    packed = sum(class_row_counts)
    flat_rotations = jnp.asarray(np.zeros((packed, 3, 3), dtype=np.float32))

    out = _project_local_class_segments_flat(
        means, means, flat_rotations, _tagging_project_rows,
        n_classes=n_classes, class_row_counts=class_row_counts,
    )

    expected = np.concatenate(
        [np.full(count, index + 1.0) for index, count in enumerate(class_row_counts)]
    )
    np.testing.assert_allclose(np.asarray(out)[:, 0], expected)


def test_flat_class_segments_keep_the_static_capacity_tail():
    """Capacity padding past the plan's present rows must keep the compiled row shape."""
    n_classes = 2
    means = jnp.asarray(np.arange(n_classes, dtype=np.complex64).reshape(n_classes, 1) + 1)
    class_row_counts = (3, 4)
    tail = 6
    flat_rotations = jnp.asarray(
        np.zeros((sum(class_row_counts) + tail, 3, 3), dtype=np.float32)
    )

    out = _project_local_class_segments_flat(
        means, means, flat_rotations, _tagging_project_rows,
        n_classes=n_classes, class_row_counts=class_row_counts,
    )

    assert int(out.shape[0]) == int(flat_rotations.shape[0])
    np.testing.assert_allclose(np.asarray(out)[-tail:, :], 0.0)


def test_flat_class_segments_reject_a_block_table_that_overruns_the_packed_rows():
    n_classes = 2
    means = jnp.asarray(np.arange(n_classes, dtype=np.complex64).reshape(n_classes, 1) + 1)
    flat_rotations = jnp.asarray(np.zeros((4, 3, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="only 4 are packed"):
        _project_local_class_segments_flat(
            means, means, flat_rotations, _tagging_project_rows,
            n_classes=n_classes, class_row_counts=(3, 3),
        )

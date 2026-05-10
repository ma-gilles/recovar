"""Merge-regression guards for cleanups landed before the EM/VDAM/PPCA-refinement merge.

These tests lock down three cleanups so a future merge cannot silently revert them:

1. ``_cap_W_shell_power`` and the ``cap_W_shell_power`` parameter were deleted from
   ``recovar.em.ppca_refinement.postprocess``. The shell-power cap was a one-sided
   real-space-mask leakage guard that papered over W shrinkage; removing it makes
   W collapse honestly diagnosable.

2. ``score_W_scale`` (and its schedule plumbing) was removed from every public
   library function and from both run scripts. It was a never-empirically-justified
   tempering knob whose only callers were algebraic-identity tests.

3. ``_apply_pmax_guard`` and the ``pmax_mean<0.5`` schedule freeze were removed
   from the run scripts. The pmax magnitude depends on the rotation-grid size and
   has no theoretical link to schedule-advance correctness; gold-standard FSC +
   pose-stability gates already cover the legitimate "don't advance yet" case.

Plus: the dense and local halfset save logic depends on
``state.pose_diagnostics["halfset0"|"halfset1"]`` containing per-halfset
``best_rotation_idx`` / ``best_translation_idx`` / (local) ``image_indices`` /
``best_rotation_matrix`` / ``best_translation``. We assert those keys are
present and shape-correct so the scripts' full-N pose scatter cannot regress
silently.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.core import fourier_transform_utils as ftu
from recovar.em.dense_single_volume.local_layout import LocalHypothesisLayout
from recovar.em.ppca_refinement import postprocess as postprocess_module
from recovar.em.ppca_refinement.dense_dataset import (
    iter_dense_ppca_dataset_blocks,
    prepare_dense_ppca_dataset_inputs,
    run_dense_ppca_fused_em_iteration,
    run_dense_ppca_halfset_fused_em_iteration,
)
from recovar.em.ppca_refinement.local_dataset import (
    run_local_ppca_fused_em_iteration,
    run_local_ppca_halfset_fused_em_iteration,
)
from recovar.em.ppca_refinement.refinement_loop import (
    run_dense_ppca_refinement_loop,
    run_local_ppca_refinement_loop,
)
from recovar.em.ppca_refinement.state import PoseMarginalPPCAEMState
from recovar.em.ppca_refinement.config import (
    GeometryConfig,
    ScheduleConfig,
)

pytestmark = pytest.mark.unit


IMAGE_SHAPE = (4, 4)
VOLUME_SHAPE = (4, 4, 4)
N_HALF = IMAGE_SHAPE[0] * (IMAGE_SHAPE[1] // 2 + 1)
HALF_VOL = int(np.prod(ftu.volume_shape_to_half_volume_shape(VOLUME_SHAPE)))


def _identity_ctf(params, image_shape, voxel_size, *, half_image=False):
    del voxel_size
    if half_image:
        n_pix = image_shape[0] * (image_shape[1] // 2 + 1)
    else:
        n_pix = image_shape[0] * image_shape[1]
    return jnp.ones((params.shape[0], n_pix), dtype=jnp.float32)


def _make_half_fourier_volume(seed):
    rng = np.random.default_rng(seed)
    real = rng.standard_normal(VOLUME_SHAPE).astype(np.float32)
    full = np.fft.fftshift(np.fft.fftn(real)).astype(np.complex64)
    return np.asarray(ftu.full_volume_to_half_volume(full, VOLUME_SHAPE), dtype=np.complex64).reshape(-1)


# Tiny dataset duck-typed to match the existing _TinyPPCAData in
# test_dense_dataset_iteration.py — kept self-contained so this guard test
# doesn't break if the sibling fixture module changes shape.
class _TinyPPCAData:
    def __init__(self, images, *, image_offset=0):
        self.image_shape = IMAGE_SHAPE
        self.volume_shape = VOLUME_SHAPE
        self.grid_size = IMAGE_SHAPE[0]
        self.voxel_size = 1.0
        self.n_images = int(images.shape[0])
        self.n_units = self.n_images
        self.dtype = jnp.complex64
        self.ctf_evaluator = _identity_ctf
        self.CTF_params = np.zeros((self.n_images, 9), dtype=np.float32)
        self._images = np.asarray(images, dtype=np.complex64)
        self._image_offset = int(image_offset)
        self.image_source = self

    def process_images(self, images, apply_image_mask=False):
        del apply_image_mask
        return images

    def process_images_half(self, images, apply_image_mask=False):
        del apply_image_mask
        return images

    @property
    def already_prefetches(self):
        return True

    def iter_batches(self, batch_size, *, indices=None, by_image=False, **kwargs):
        del by_image, kwargs
        if indices is None:
            indices = np.arange(self.n_images)
        indices = np.asarray(indices, dtype=np.int64)
        for start in range(0, indices.size, int(batch_size)):
            idx = indices[start : start + int(batch_size)]
            yield (
                jnp.asarray(self._images[idx]),
                None,
                None,
                jnp.asarray(self.CTF_params[idx]),
                None,
                idx + self._image_offset,
                idx + self._image_offset,
            )

    def get_halfset(self, halfset_id):
        if halfset_id == 0:
            return _TinyPPCAData(self._images[::2], image_offset=0)
        if halfset_id == 1:
            return _TinyPPCAData(self._images[1::2], image_offset=1)
        raise ValueError("halfset_id must be 0 or 1")


def _all_retained_local_layout(dataset, rotations, translations):
    n_images = dataset.n_images
    n_rot = rotations.shape[0]
    offsets = np.arange(n_images + 1, dtype=np.int64) * n_rot
    return LocalHypothesisLayout(
        n_global_rotations=n_rot,
        n_pixels=n_rot,
        n_psi=1,
        rotation_offsets=offsets,
        rotation_ids_flat=np.tile(np.arange(n_rot, dtype=np.int32), n_images),
        rotations_flat=np.tile(rotations[None, :, :, :], (n_images, 1, 1, 1)).reshape(n_images * n_rot, 3, 3),
        rotation_log_priors_flat=np.zeros((n_images * n_rot,), dtype=np.float32),
        rotation_counts=np.full((n_images,), n_rot, dtype=np.int32),
        translation_grid=translations,
        translation_log_priors=np.zeros((n_images, translations.shape[0]), dtype=np.float32),
    )


@pytest.fixture
def tiny_halfset_inputs():
    """4-image dataset; ``get_halfset`` slices even/odd into two 2-image halves."""
    rng = np.random.default_rng(7)
    images = (rng.standard_normal((4, N_HALF)) + 1j * rng.standard_normal((4, N_HALF))).astype(np.complex64)
    dataset = _TinyPPCAData(images)
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (2, 3, 3)).copy()
    translations = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    mu = _make_half_fourier_volume(10)
    W = (_make_half_fourier_volume(11)[:, None] * np.asarray(0.05, dtype=np.float32)).astype(np.complex64)
    return dataset, mu, W, rotations, translations


def _build_state(mu, W):
    return PoseMarginalPPCAEMState(
        mu_half=(jnp.asarray(mu), jnp.asarray(mu)),
        W_half=(jnp.asarray(W), jnp.asarray(W)),
        mu_score=jnp.asarray(mu),
        W_score=jnp.asarray(W),
        W_prior=jnp.ones((HALF_VOL, 1), dtype=jnp.float32) * 5.0,
        mean_prior=jnp.ones((HALF_VOL,), dtype=jnp.float32) * 10.0,
        noise_variance=jnp.ones((N_HALF,), dtype=jnp.float32),
        z_prior_precision_diag=jnp.ones((1,), dtype=jnp.float32),
        schedule_state=None,
    )


# ---------------------------------------------------------------------------
# (1) postprocess cap removed
# ---------------------------------------------------------------------------
def test_postprocess_module_has_no_cap_W_shell_power_function():
    assert not hasattr(postprocess_module, "_cap_W_shell_power"), (
        "shell-power cap should remain deleted; reintroducing it papers over W shrinkage"
    )


def test_postprocess_signature_has_no_cap_W_shell_power_kwarg():
    sig = inspect.signature(postprocess_module.postprocess_ppca_half_volumes)
    assert "cap_W_shell_power" not in sig.parameters


def test_postprocess_diagnostics_no_longer_publish_shell_power_cap_keys():
    """A successful call must not report shell-power-cap diagnostics any more."""
    box = (4, 4, 4)
    half_size = int(np.prod(ftu.volume_shape_to_half_volume_shape(box)))
    mu_half = jnp.zeros((half_size,), dtype=jnp.complex64)
    W_half = jnp.zeros((half_size, 1), dtype=jnp.complex64)
    out = postprocess_module.postprocess_ppca_half_volumes(
        mu_half,
        W_half,
        box,
        config=postprocess_module.PostprocessConfig(
            strategy="mean_and_w_mask",
            mask_radius_px=2.0,
            cosine_width_px=1.0,
            grid_correct=False,
        ),
    )
    forbidden = {
        "postprocess_cap_W_shell_power",
        "postprocess_W_shell_power_scale_min",
        "postprocess_W_shell_power_scale_mean",
        "postprocess_W_shell_power_input_sum",
        "postprocess_W_shell_power_pre_cap_sum",
        "postprocess_W_shell_power_output_sum",
    }
    leaked = forbidden & set(out.diagnostics.keys())
    assert not leaked, f"shell-power cap diagnostics leaked back in: {sorted(leaked)}"


# ---------------------------------------------------------------------------
# (2) score_W_scale removed from every public library function
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "fn",
    [
        prepare_dense_ppca_dataset_inputs,
        iter_dense_ppca_dataset_blocks,
        run_dense_ppca_fused_em_iteration,
        run_dense_ppca_halfset_fused_em_iteration,
        run_local_ppca_fused_em_iteration,
        run_local_ppca_halfset_fused_em_iteration,
        run_dense_ppca_refinement_loop,
        run_local_ppca_refinement_loop,
    ],
)
def test_no_score_W_scale_kwarg_in_public_signatures(fn):
    sig = inspect.signature(fn)
    assert "score_W_scale" not in sig.parameters, (
        f"{fn.__name__} re-grew a score_W_scale kwarg; remove it (no empirical evidence it improves recovery)"
    )


# ---------------------------------------------------------------------------
# (3) script-level cleanups: no _apply_pmax_guard, no score-W-scale CLI flag
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"


@pytest.mark.parametrize(
    "script_name",
    ["run_ppca_dense_from_init_npz.py", "run_ppca_local_from_init_npz.py"],
)
def test_run_scripts_do_not_resurrect_pmax_guard_or_score_W_scale(script_name):
    text = (SCRIPTS_DIR / script_name).read_text()
    forbidden = (
        "_apply_pmax_guard",
        "pmax_guard",
        "score_W_scale",
        "score-W-scale",
        "score_W_tempered",
        "_cap_W_shell_power",
    )
    found = [tok for tok in forbidden if tok in text]
    assert not found, f"{script_name} re-introduced removed symbols: {found} — keep cleanups in place"


# ---------------------------------------------------------------------------
# (4) halfset save invariants — both halfsets emit the per-image keys the
#     scripts' full-N scatter depends on. If these contracts hold, the dense
#     and local halfset save blocks (which are pure data movement that
#     scatters by ``halfset_indices``) cannot silently drop poses.
# ---------------------------------------------------------------------------
def test_dense_halfset_iteration_publishes_per_halfset_pose_keys(tiny_halfset_inputs):
    """The dense halfset iteration must produce pose_diagnostics for both
    halfsets with the per-image keys that the script's full-N scatter
    depends on."""
    dataset, mu, W, rotations, translations = tiny_halfset_inputs
    state = _build_state(mu, W)

    updated = run_dense_ppca_halfset_fused_em_iteration(
        state,
        dataset,
        rotations=rotations,
        translations=translations,
        schedule=ScheduleConfig(image_batch_size=2, rotation_block_size=1),
        geometry=GeometryConfig(current_size=4),
    )

    pd = updated.pose_diagnostics
    for half_idx, key in enumerate(("halfset0", "halfset1")):
        assert key in pd, f"missing {key} in pose_diagnostics"
        diag = pd[key]
        n_per_half = int(dataset.get_halfset(half_idx).n_images)
        for required in ("best_rotation_idx", "best_translation_idx"):
            assert required in diag, f"{key} missing {required!r} (script's full-N scatter relies on this)"
            arr = np.asarray(diag[required])
            assert arr.shape == (n_per_half,), (
                f"{key}.{required} shape {arr.shape} != ({n_per_half},); "
                "shape mismatch will silently drop poses for some images"
            )


def test_local_halfset_iteration_publishes_per_halfset_pose_keys(tiny_halfset_inputs):
    """The local halfset iteration must produce pose_diagnostics for both
    halfsets with all keys the script's full-N scatter (with bucket-order
    argsort) requires."""
    dataset, mu, W, rotations, translations = tiny_halfset_inputs
    state = _build_state(mu, W)

    halfset_datasets = (dataset.get_halfset(0), dataset.get_halfset(1))
    halfset_layouts = tuple(_all_retained_local_layout(half, rotations, translations) for half in halfset_datasets)

    updated = run_local_ppca_halfset_fused_em_iteration(
        state,
        halfset_datasets,
        halfset_layouts,
        geometry=GeometryConfig(current_size=4),
        schedule=ScheduleConfig(mstep_chunk_size=8),
    )

    pd = updated.pose_diagnostics
    for half_idx, key in enumerate(("halfset0", "halfset1")):
        assert key in pd
        diag = pd[key]
        n_per_half = int(dataset.get_halfset(half_idx).n_images)
        for required in (
            "best_rotation_idx",
            "best_translation_idx",
            "best_rotation_matrix",
            "best_translation",
            "image_indices",
        ):
            assert required in diag, f"{key} missing {required!r}"
        rot_idx = np.asarray(diag["best_rotation_idx"])
        trans_idx = np.asarray(diag["best_translation_idx"])
        rot_mat = np.asarray(diag["best_rotation_matrix"])
        trans = np.asarray(diag["best_translation"])
        img_idx = np.asarray(diag["image_indices"])
        assert rot_idx.shape == (n_per_half,)
        assert trans_idx.shape == (n_per_half,)
        assert rot_mat.shape == (n_per_half, 3, 3)
        assert trans.shape == (n_per_half, 2)
        assert img_idx.shape == (n_per_half,)
        # image_indices values must be unique (so argsort produces a valid
        # inverse permutation in the script's bucket-order undo). The actual
        # absolute values depend on the dataset's image_offset; we don't pin them.
        assert len(set(img_idx.tolist())) == n_per_half, "image_indices not unique — bucket undo would lose poses"
        inv_perm = np.argsort(img_idx).astype(np.int64)
        assert sorted(inv_perm.tolist()) == list(range(n_per_half))

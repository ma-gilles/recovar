"""Score prior and validity constraints shared by dense EM bucket paths."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


def _finite_minmax(values) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        finite = np.array([-1e30], dtype=np.float32)
    return float(finite.min()), float(finite.max())


def _is_lazy_candidate_mask(value) -> bool:
    return hasattr(value, "block_mask") and hasattr(value, "shape")


@dataclass(frozen=True)
class DenseScoreConstraints:
    """Dense/global score priors and masks normalized to padded bucket shapes.

    ``candidate_mask`` may be either a shared 2D mask of shape
    ``(n_rot_padded, n_trans)`` or a per-image 3D mask of shape
    ``(n_images, n_rot_padded, n_trans)``.  ``per_image_candidate_mask`` is
    set when the 3D form is used; ``block_inputs`` slices the appropriate
    rotation/image block accordingly.  This is the entry point for
    RELION-style adaptive 2-pass significance pruning: pass-1 produces
    a per-particle significance mask and pass-2 evaluates the fine
    grid with masked-out positions forced to ``-inf`` before softmax.
    """

    rotation_log_prior: object | None
    per_image_rotation_prior: bool
    rotation_prior_minmax: tuple[float, float] | None
    translation_log_prior: object | None
    per_image_translation_prior: bool
    translation_prior_minmax: tuple[float, float] | None
    candidate_mask: object | None
    per_image_candidate_mask: bool
    candidate_mask_count: int | None
    candidate_mask_size: int | None
    n_images: int
    n_rot: int
    n_trans: int
    n_rot_padded: int

    @classmethod
    def from_inputs(
        cls,
        *,
        rotation_log_prior,
        translation_log_prior,
        rotation_translation_mask,
        n_images: int,
        n_rot: int,
        n_trans: int,
        n_rot_padded: int,
    ) -> "DenseScoreConstraints":
        rotation_prior_jnp = None
        per_image_rotation_prior = False
        rotation_prior_minmax = None
        if rotation_log_prior is not None:
            rotation_np = np.asarray(rotation_log_prior, dtype=np.float32)
            if rotation_np.ndim == 1:
                padded = np.full(n_rot_padded, -1e30, dtype=np.float32)
                padded[:n_rot] = rotation_np
            elif rotation_np.ndim == 2:
                if rotation_np.shape != (n_images, n_rot):
                    raise ValueError(
                        "rotation_log_prior must have shape "
                        f"({n_images}, {n_rot}) when image-specific, got {rotation_np.shape}",
                    )
                padded = np.full((n_images, n_rot_padded), -1e30, dtype=np.float32)
                padded[:, :n_rot] = rotation_np
                per_image_rotation_prior = True
            else:
                raise ValueError(
                    f"rotation_log_prior must be 1D or 2D, got {rotation_np.ndim} dimensions",
                )
            rotation_prior_jnp = jnp.asarray(padded)
            rotation_prior_minmax = _finite_minmax(rotation_np)

        translation_prior_jnp = None
        per_image_translation_prior = False
        translation_prior_minmax = None
        if translation_log_prior is not None:
            translation_np = np.asarray(translation_log_prior, dtype=np.float32)
            if translation_np.ndim == 1:
                if translation_np.shape != (n_trans,):
                    raise ValueError(
                        f"translation_log_prior must have shape ({n_trans},), got {translation_np.shape}",
                    )
            elif translation_np.ndim == 2:
                if translation_np.shape != (n_images, n_trans):
                    raise ValueError(
                        "translation_log_prior must have shape "
                        f"({n_images}, {n_trans}) when image-specific, got {translation_np.shape}",
                    )
                per_image_translation_prior = True
            else:
                raise ValueError(
                    f"translation_log_prior must be 1D or 2D, got {translation_np.ndim} dimensions",
                )
            translation_prior_jnp = jnp.asarray(translation_np)
            translation_prior_minmax = _finite_minmax(translation_np)

        candidate_mask_jnp = None
        per_image_candidate_mask = False
        candidate_mask_count = None
        candidate_mask_size = None
        if rotation_translation_mask is not None:
            if _is_lazy_candidate_mask(rotation_translation_mask):
                if tuple(rotation_translation_mask.shape) != (n_images, n_rot, n_trans):
                    raise ValueError(
                        "lazy rotation_translation_mask must have shape "
                        f"({n_images}, {n_rot}, {n_trans}), got {rotation_translation_mask.shape}",
                    )
                candidate_mask_jnp = rotation_translation_mask
                per_image_candidate_mask = True
                candidate_mask_count = getattr(rotation_translation_mask, "valid_count", None)
                candidate_mask_size = getattr(rotation_translation_mask, "size", None)
            else:
                candidate_mask = np.asarray(rotation_translation_mask, dtype=bool)
                if candidate_mask.ndim == 2:
                    if candidate_mask.shape != (n_rot, n_trans):
                        raise ValueError(
                            f"rotation_translation_mask must have shape ({n_rot}, {n_trans}), got {candidate_mask.shape}",
                        )
                    padded_mask = np.zeros((n_rot_padded, n_trans), dtype=bool)
                    padded_mask[:n_rot] = candidate_mask
                elif candidate_mask.ndim == 3:
                    if candidate_mask.shape != (n_images, n_rot, n_trans):
                        raise ValueError(
                            "rotation_translation_mask must have shape "
                            f"({n_images}, {n_rot}, {n_trans}) when image-specific, got {candidate_mask.shape}",
                        )
                    padded_mask = np.zeros((n_images, n_rot_padded, n_trans), dtype=bool)
                    padded_mask[:, :n_rot, :] = candidate_mask
                    per_image_candidate_mask = True
                else:
                    raise ValueError(
                        f"rotation_translation_mask must be 2D or 3D, got {candidate_mask.ndim} dimensions",
                    )
                candidate_mask_jnp = jnp.asarray(padded_mask)
                candidate_mask_count = int(candidate_mask.sum())
                candidate_mask_size = int(candidate_mask.size)

        return cls(
            rotation_log_prior=rotation_prior_jnp,
            per_image_rotation_prior=per_image_rotation_prior,
            rotation_prior_minmax=rotation_prior_minmax,
            translation_log_prior=translation_prior_jnp,
            per_image_translation_prior=per_image_translation_prior,
            translation_prior_minmax=translation_prior_minmax,
            candidate_mask=candidate_mask_jnp,
            per_image_candidate_mask=per_image_candidate_mask,
            candidate_mask_count=candidate_mask_count,
            candidate_mask_size=candidate_mask_size,
            n_images=int(n_images),
            n_rot=int(n_rot),
            n_trans=int(n_trans),
            n_rot_padded=int(n_rot_padded),
        )

    def block_inputs(
        self,
        *,
        r0: int,
        r1: int,
        start: int,
        end: int,
        batch_count: int,
        rotation_block_size: int,
    ):
        actual_count = int(end - start)
        batch_count = int(batch_count)
        if self.rotation_log_prior is None:
            rotation_prior = jnp.zeros((batch_count, rotation_block_size), dtype=jnp.float32)
        elif self.per_image_rotation_prior:
            rotation_prior = self.rotation_log_prior[start:end, r0:r1]
            if batch_count != actual_count:
                rotation_prior = jnp.pad(
                    rotation_prior,
                    ((0, batch_count - actual_count), (0, 0)),
                    constant_values=0,
                )
        else:
            rotation_prior = jnp.broadcast_to(
                self.rotation_log_prior[r0:r1][None, :],
                (batch_count, rotation_block_size),
            )

        if self.translation_log_prior is None:
            translation_prior = jnp.zeros((batch_count, self.n_trans), dtype=jnp.float32)
        elif self.per_image_translation_prior:
            translation_prior = self.translation_log_prior[start:end]
            if batch_count != actual_count:
                translation_prior = jnp.pad(
                    translation_prior,
                    ((0, batch_count - actual_count), (0, 0)),
                    constant_values=0,
                )
        else:
            translation_prior = jnp.broadcast_to(
                self.translation_log_prior[None, :],
                (batch_count, self.n_trans),
            )

        if self.candidate_mask is None:
            candidate_mask = jnp.ones((rotation_block_size, self.n_trans), dtype=bool)
        elif _is_lazy_candidate_mask(self.candidate_mask):
            candidate_mask = self.candidate_mask.block_mask(
                r0=r0,
                r1=r1,
                start=start,
                end=end,
                batch_count=batch_count,
                rotation_block_size=rotation_block_size,
            )
        elif self.per_image_candidate_mask:
            candidate_mask = self.candidate_mask[start:end, r0:r1, :]
            if batch_count != actual_count:
                candidate_mask = jnp.pad(
                    candidate_mask,
                    ((0, batch_count - actual_count), (0, 0), (0, 0)),
                    constant_values=False,
                )
        else:
            candidate_mask = self.candidate_mask[r0:r1]

        valid = max(0, min(rotation_block_size, self.n_rot - int(r0)))
        valid_rotation_mask = jnp.arange(rotation_block_size) < valid
        return rotation_prior, translation_prior, candidate_mask, valid_rotation_mask

    def block_has_candidates(
        self,
        *,
        r0: int,
        start: int,
        end: int,
        batch_count: int,
        rotation_block_size: int,
    ) -> bool:
        """Return whether a block can contain any valid masked candidate."""

        actual_count = int(end - start)
        valid_rot = max(0, min(int(rotation_block_size), self.n_rot - int(r0)))
        if actual_count <= 0 or valid_rot <= 0:
            return False
        if self.candidate_mask is None:
            return True
        if _is_lazy_candidate_mask(self.candidate_mask) and hasattr(self.candidate_mask, "block_has_candidates"):
            return bool(
                self.candidate_mask.block_has_candidates(
                    r0=r0,
                    start=start,
                    end=end,
                    batch_count=batch_count,
                    rotation_block_size=rotation_block_size,
                )
            )
        return True


def apply_dense_score_constraints(
    scores,
    rotation_prior,
    translation_prior,
    candidate_mask,
    valid_rotation_mask,
    valid_image_mask,
    *,
    score_mode: str,
):
    """Apply priors and validity masks to one dense/global score block.

    ``candidate_mask`` may be 2D ``(rot_block, n_trans)`` (shared across
    images) or 3D ``(batch, rot_block, n_trans)`` (per-image, e.g. an
    adaptive 2-pass significance mask).  Both forms broadcast against the
    score tensor of shape ``(batch, rot_block, n_trans)``.
    """

    if score_mode == "gaussian":
        scores = scores + rotation_prior[:, :, None]
        scores = scores + translation_prior[:, None, :]
    if candidate_mask.ndim == 3:
        scores = jnp.where(candidate_mask, scores, -jnp.inf)
    else:
        scores = jnp.where(candidate_mask[None, :, :], scores, -jnp.inf)
    scores = jnp.where(valid_rotation_mask[None, :, None], scores, -jnp.inf)
    return jnp.where(valid_image_mask[:, None, None], scores, 0.0)

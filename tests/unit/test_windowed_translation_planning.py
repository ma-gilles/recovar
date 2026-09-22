import numpy as np
import pytest


def test_windowed_translation_tile_lifts_the_coarse_pass_image_cap():
    """The K-class passes translate inside the current-size window; sizing them by the
    full half-image translation tile capped the 100k/256 K=4 coarse pass at 88 images."""
    from recovar.em.helpers.batch_planning import _estimate_relion_em_batch_sizes

    common = dict(
        requested_image_batch_size=5000,
        requested_rotation_block_size=2000,
        n_rot=576,
        n_trans=29,
        image_shape=(256, 256),
        volume_shape=(256, 256, 256),
        padding_factor=2,
        n_classes=4,
        gpu_memory_gb=80.0,
        current_size=14,
    )
    full = _estimate_relion_em_batch_sizes(**common)
    windowed = _estimate_relion_em_batch_sizes(**common, windowed_translation=True)
    # the full-size tile (29 x 33 024 pixels per image and class) binds well below the
    # request; the 584-pixel window lifts it by more than 2x, and other caps still bind
    assert full.image_batch_size < 5000, full
    assert windowed.image_batch_size > 2 * full.image_batch_size, (full, windowed)
    assert windowed.image_batch_size < 5000, windowed
    # without a current size there is no window: the two estimates agree
    common_full = dict(common, current_size=None)
    assert (
        _estimate_relion_em_batch_sizes(**common_full).image_batch_size
        == _estimate_relion_em_batch_sizes(**common_full, windowed_translation=True).image_batch_size
    )

def test_kclass_adaptive_planner_requests_windowed_translation_when_supported():
    from recovar.em.helpers.batch_planning import _plan_kclass_adaptive_grid_batch_sizes

    seen = []

    def with_kw(n_rot, n_trans, *, classes=None, image_shape_for_batch=None, current_size_for_batch=None, windowed_translation=False):
        seen.append(windowed_translation)
        return 7, 11

    def without_kw(n_rot, n_trans, *, classes=None, image_shape_for_batch=None, current_size_for_batch=None):
        seen.append("no-kw")
        return 7, 11

    for fn in (with_kw, without_kw):
        _plan_kclass_adaptive_grid_batch_sizes(
            coarse_rotations=np.zeros((576, 3, 3)),
            coarse_translations=np.zeros((29, 2)),
            fine_rotations=np.zeros((4608, 3, 3)),
            fine_translations=np.zeros((116, 2)),
            n_classes=4,
            image_shape=(256, 256),
            coarse_current_size=14,
            fine_current_size=38,
            safe_batch_sizes=fn,
        )
    assert seen == [True, True, "no-kw", "no-kw"], seen


@pytest.mark.parametrize("fine_accepts", [False, True])
def test_separate_planners_negotiate_window_keyword_independently(fine_accepts):
    from recovar.em.helpers.batch_planning import _plan_kclass_adaptive_grid_batch_sizes
    seen = []
    def accepts(n_rot, n_trans, *, windowed_translation=False, **kwargs):
        seen.append(windowed_translation)
        return 7, 11
    def legacy(n_rot, n_trans, *, classes, image_shape_for_batch, current_size_for_batch):
        seen.append("legacy")
        return 7, 11
    _plan_kclass_adaptive_grid_batch_sizes(
        coarse_rotations=np.zeros((576, 3, 3)), coarse_translations=np.zeros((29, 2)),
        fine_rotations=np.zeros((4608, 3, 3)), fine_translations=np.zeros((116, 2)),
        n_classes=4, image_shape=(256, 256), coarse_current_size=14, fine_current_size=38,
        safe_batch_sizes=accepts if fine_accepts else legacy,
        significance_safe_batch_sizes=legacy if fine_accepts else accepts,
    )
    assert seen == ([True, "legacy"] if fine_accepts else ["legacy", True])

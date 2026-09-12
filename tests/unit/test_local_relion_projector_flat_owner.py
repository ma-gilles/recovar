"""``_relion_local_projector_flat`` is the one RELION-projector path of the local bucket and packed-noise projections."""

import inspect

from recovar.em.local import local_bucket_stages, local_em_engine


def test_both_local_projections_share_the_relion_projector_owner():
    for fn in (local_em_engine._project_local_bucket, local_em_engine._project_packed_noise_rows):
        src = inspect.getsource(fn)
        assert src.count("_relion_local_projector_flat(") == 1
        assert "prepare_local_projector_slab(" not in src and "_compute_relion_projector_projections_block(" not in src
    owner = inspect.getsource(local_bucket_stages._relion_local_projector_flat)
    assert owner.index("prepare_local_projector_slab(") < owner.index("_compute_relion_projector_projections_block(")
    assert 'raise ValueError("relion_projector_r_max is required' in owner


def test_owner_builds_the_projector_keywords_from_the_window(monkeypatch):
    seen = {}

    def fake_block(half, rotations, image_shape, **kwargs):
        seen.update(kwargs, half=half, rotations=rotations)
        return "FLAT", None

    monkeypatch.setattr(local_bucket_stages, "_compute_relion_projector_projections_block", fake_block)
    monkeypatch.setattr(local_bucket_stages, "prepare_local_projector_slab", lambda h: ("SLAB", h))

    class Window:
        use_window = True
        max_r = 3

        @staticmethod
        def relion_projector_output_size():
            return 12

    out = local_bucket_stages._relion_local_projector_flat(
        "HALF", "ROT", image_shape=(8, 8), relion_projector_r_max=4, projection_padding_factor=2,
        projection_kwargs={"mask_current_image_disk": False, "relion_texture_interp": True}, window_spec=Window(), projection_indices="IDX",
    )
    assert out == "FLAT" and seen["half"] == ("SLAB", "HALF") and seen["r_max"] == 4 and seen["padding_factor"] == 2
    assert seen["projector_output_size"] == 12 and seen["pixel_indices"] == "IDX" and seen["mask_current_image_disk"] is False
    assert seen["relion_texture_interp"] is True and seen["relion_acc_double_floorf_quirk"] is False and seen["centered_rows"] and seen["dense_scale"]

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import starfile

from recovar.em.dense_single_volume.frozen_boundary import (
    FROZEN_BOUNDARY_NUMERICAL_CLASSIFICATION_SCOPE,
    FROZEN_BOUNDARY_PROVENANCE_VERIFICATION_SCOPE,
    V3_REQUIRED_FIXED_SOURCE_NAMES,
    _V3_SCALAR_DTYPES,
)
from scripts.finalize_frozen_boundary_v3 import (
    _runtime_payload,
    _validate_capture_manifest,
    _validate_recovar_source_manifest,
    _validate_runtime_config_against_capture,
    _validate_source_paths,
)
from scripts.run_full_refinement import _particle_stack_paths_from_star


def _runtime_config():
    values = {}
    for field, dtype in _V3_SCALAR_DTYPES.items():
        if not field.startswith("config_"):
            continue
        name = field.removeprefix("config_")
        if dtype is None:
            values[name] = "value"
        elif dtype.kind == "b":
            values[name] = True
        elif dtype.kind in "iu":
            values[name] = 1
        else:
            values[name] = 1.0
    values.update(
        {
            "provenance_verification_scope": FROZEN_BOUNDARY_PROVENANCE_VERIFICATION_SCOPE,
            "numerical_classification_scope": FROZEN_BOUNDARY_NUMERICAL_CLASSIFICATION_SCOPE,
            "declared_relion_base_git_commit": "1" * 40,
            "recovar_git_commit": "2" * 40,
            "projector_boundary_kind": "reconstructed-projector boundary",
            "replay_prefix": "run",
        }
    )
    return values


def test_runtime_payload_requires_exact_typed_config_closure():
    config = _runtime_config()

    payload = _runtime_payload(config)

    assert payload["config_random_seed"].dtype.name == "int64"
    assert payload["config_tau2_fudge"].dtype.name == "float64"
    assert payload["config_do_ctf_correction"].dtype.name == "bool"

    config["do_ctf_correction"] = "false"
    with pytest.raises(ValueError, match="JSON boolean"):
        _runtime_payload(config)


def _source_paths(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    names = {
        *V3_REQUIRED_FIXED_SOURCE_NAMES,
        "particle_stack:0",
        "consumer_map:half1:class1",
        "consumer_map:half2:class1",
    }
    paths = {}
    for name in names:
        path = tmp_path / name.replace(":", "_")
        path.write_text(name, encoding="utf-8")
        paths[name] = path
    for half in (1, 2):
        map_path = paths[f"consumer_map:half{half}:class1"]
        model_path = paths[f"consumer_validation_half{half}_model"]
        starfile.write(
            {"model_classes": pd.DataFrame({"rlnReferenceImage": [str(map_path)]})},
            model_path,
            overwrite=True,
        )
    starfile.write(
        {
            "particles": pd.DataFrame(
                {"rlnImageName": [f"1@{paths['particle_stack:0']}"]}
            )
        },
        paths["particle_star"],
        overwrite=True,
    )
    return paths


def test_source_path_finalizer_closes_consumer_model_map_references(tmp_path):
    paths = _source_paths(tmp_path)

    _validate_source_paths(paths, live_manifest=paths["live_capture_manifest"])

    document = starfile.read(paths["consumer_validation_half2_model"], always_dict=True)
    document["model_classes"].loc[0, "rlnReferenceImage"] = str(tmp_path / "substitute.mrc")
    starfile.write(document, paths["consumer_validation_half2_model"], overwrite=True)
    with pytest.raises(ValueError, match="differs from sealed map"):
        _validate_source_paths(paths, live_manifest=paths["live_capture_manifest"])


def test_source_path_finalizer_rejects_noncontiguous_or_unknown_sources(tmp_path):
    paths = _source_paths(tmp_path)
    paths["particle_stack:2"] = paths.pop("particle_stack:0")
    with pytest.raises(ValueError, match="expected_stacks"):
        _validate_source_paths(paths, live_manifest=paths["live_capture_manifest"])

    paths = _source_paths(tmp_path / "unknown")
    unknown = tmp_path / "unknown_source"
    unknown.write_text("unknown", encoding="utf-8")
    paths["unexpected_source"] = unknown
    with pytest.raises(ValueError, match="unknown"):
        _validate_source_paths(paths, live_manifest=paths["live_capture_manifest"])


def test_source_path_finalizer_rejects_particle_star_stack_substitution(tmp_path):
    paths = _source_paths(tmp_path)
    substitute = tmp_path / "substitute.mrcs"
    substitute.write_bytes(b"substitute")
    starfile.write(
        {"particles": pd.DataFrame({"rlnImageName": [f"1@{substitute}"]})},
        paths["particle_star"],
        overwrite=True,
    )

    with pytest.raises(ValueError, match="stack order differs"):
        _validate_source_paths(paths, live_manifest=paths["live_capture_manifest"])


def test_particle_stack_order_is_resolved_absolute_lexical_not_first_appearance(tmp_path):
    paths = _source_paths(tmp_path)
    stack_a = tmp_path / "a_stack.mrcs"
    stack_z = tmp_path / "z_stack.mrcs"
    stack_a.write_bytes(b"a")
    stack_z.write_bytes(b"z")
    paths["particle_stack:0"] = stack_a
    paths["particle_stack:1"] = stack_z
    starfile.write(
        {
            "particles": pd.DataFrame(
                {
                    "rlnImageName": [
                        f"1@{stack_z}",
                        f"1@{stack_a}",
                        f"2@{stack_z}",
                    ]
                }
            )
        },
        paths["particle_star"],
        overwrite=True,
    )

    _validate_source_paths(paths, live_manifest=paths["live_capture_manifest"])
    assert _particle_stack_paths_from_star(paths["particle_star"]) == (
        stack_a.resolve(),
        stack_z.resolve(),
    )


def test_recovar_source_manifest_must_match_runtime_commit_and_clean_state(tmp_path):
    config = _runtime_config()
    manifest = tmp_path / "source.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "recovar.em.source_manifest.v1",
                "recovar_git_commit": config["recovar_git_commit"],
                "worktree_clean": True,
            }
        ),
        encoding="utf-8",
    )

    _validate_recovar_source_manifest(manifest, config)

    manifest.write_text(
        json.dumps(
            {
                "schema": "recovar.em.source_manifest.v1",
                "recovar_git_commit": "3" * 40,
                "worktree_clean": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source manifest differs"):
        _validate_recovar_source_manifest(manifest, config)


def test_capture_reader_rejects_operand_omitted_from_valid_manifest(tmp_path):
    listed = tmp_path / "listed.bin"
    listed.write_bytes(b"listed")
    omitted = tmp_path / "state_iter2_rank1_device0_class0_iref.bin"
    omitted.write_bytes(b"omitted")
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text(
        f"{hashlib.sha256(listed.read_bytes()).hexdigest()}  {listed}\n",
        encoding="utf-8",
    )

    capture = _validate_capture_manifest(manifest)

    with pytest.raises(ValueError, match="not bound by the live manifest"):
        capture.path(2, 1, "iref")


class _FakeCapture:
    def __init__(self):
        self.values = {
            "control_adaptive_oversampling": 1.0,
            "control_maximum_significants": 1.0,
            "control_width_mask_edge": 1.0,
            "model_tau2_fudge_factor": 1.0,
            "control_random_seed": 1.0,
            "model_nr_classes": 1.0,
            "model_ori_size": 1.0,
            "model_pixel_size": 1.0,
            "projector_padding_factor": 1.0,
            "control_do_ctf_correction": 1.0,
            "control_do_firstiter_cc": 1.0,
            "control_do_norm_correction": 1.0,
            "control_do_scale_correction": 1.0,
            "control_refs_are_ctf_corrected": 1.0,
            "sampling_healpix_order": 1.0,
            "sampling_offset_range": 1.0,
            "sampling_offset_step": 1.0,
            "sampling_perturbation_factor": 1.0,
            "model_data_dim": 2.0,
            "model_ref_dim": 3.0,
            "model_nr_bodies": 1.0,
            "model_nr_optics_groups": 1.0,
        }

    def scalar(self, iteration, rank, name):
        del iteration, rank
        return self.values[name]

    def vector(self, iteration, rank, name, dtype="<f8"):
        del iteration, rank, name
        return np.asarray([1], dtype=dtype)


def test_runtime_config_must_equal_captured_relion_controls(tmp_path):
    optimiser = tmp_path / "completed_optimiser.star"
    model = tmp_path / "completed_half1_model.star"
    starfile.write(
        {
            "optimiser_general": {
                "rlnAutoLocalSearchesHealpixOrder": 1,
                "rlnParticleDiameter": 1.0,
                "rlnJoinHalvesUntilThisResolution": 1.0,
            }
        },
        optimiser,
        overwrite=True,
    )
    starfile.write(
        {"model_general": {"rlnFourierSpaceInterpolator": 1}},
        model,
        overwrite=True,
    )
    config = _runtime_config()
    config.update(
        {
            "diagnostic_arm_id": "real10076.k1.physical_it2.reconstructed_projector.v1",
            "max_iter": 1,
            "skip_final_iteration": True,
            "init_resolution_angstrom": 30.0,
            "offset_range_pixels": 1.0,
            "offset_step_pixels": 1.0,
            "perturb_factor": 1.0,
            "fsc_threshold": 1.0 / 7.0,
            "jax_enable_x64": True,
            "disc_type": "linear_interp",
            "image_fourier_backend": "relion_cuda",
            "local_search_translation_prior_mode": "coarse",
            "projector_boundary_kind": "reconstructed-projector boundary",
        }
    )
    source_paths = {
        "completed_optimiser": optimiser,
        "completed_half1_model": model,
    }

    _validate_runtime_config_against_capture(
        config,
        capture=_FakeCapture(),
        source_paths=source_paths,
        consumer=2,
    )

    config["width_mask_edge_px"] = 2.0
    with pytest.raises(ValueError, match="runtime config width_mask_edge_px"):
        _validate_runtime_config_against_capture(
            config,
            capture=_FakeCapture(),
            source_paths=source_paths,
            consumer=2,
        )

"""Named motion-file reuse must remain valid after native particle selection."""

import hashlib

import numpy as np
import pandas as pd
import pytest
import starfile

from recovar.data_io.relion5_trajectory_validation import validate_named_trajectories

pytestmark = pytest.mark.unit


def _shifts(n=3):
    return pd.DataFrame({f"rlnOrigin{axis}Angst": np.arange(n, dtype=float) + i for i, axis in enumerate("XYZ")})


@pytest.fixture
def motion_case(tmp_path):
    path = tmp_path / "motion.star"
    blocks = {"general": {"rlnParticleNumber": 3}, "tomo/P10": _shifts(), "tomo/P1": _shifts(), "tomo/P2": _shifts()}
    particles = pd.DataFrame(
        {"rlnTomoParticleName": ["tomo/P2", "tomo/P10"], "rlnTomoVisibleFrames": ["[1,0,1]", "[0,1,1]"]}
    )
    starfile.write(blocks, path)
    return path, blocks, particles


def test_named_subset_with_changed_row_order_and_extra_particles(motion_case):
    path, _, particles = motion_case
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    validate_named_trajectories(path, particles)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_full_tomogram_frame_count_not_compact_visible_stack(motion_case):
    path, blocks, particles = motion_case
    blocks["tomo/P2"] = _shifts(2)
    starfile.write(blocks, path, overwrite=True)
    with pytest.raises(ValueError, match="all 3 tomogram frames"):
        validate_named_trajectories(path, particles)


def test_legacy_row_indexed_file_rejected(motion_case):
    path, _, particles = motion_case
    starfile.write(
        {"general": {"rlnParticleNumber": 3}, "0": _shifts(), "1": _shifts(), "2": _shifts()}, path, overwrite=True
    )
    with pytest.raises(ValueError, match="Legacy row-indexed"):
        validate_named_trajectories(path, particles)


def test_missing_named_particle_rejected(motion_case):
    path, blocks, particles = motion_case
    blocks["tomo/other"] = blocks.pop("tomo/P2")
    starfile.write(blocks, path, overwrite=True)
    with pytest.raises(ValueError, match="Missing named trajectories"):
        validate_named_trajectories(path, particles)


def test_duplicate_blocks_rejected_before_dict_parser_discards_them(motion_case):
    path, _, particles = motion_case
    with path.open("a") as handle:
        handle.write("\ndata_tomo/P2\n\nloop_\n_rlnOriginXAngst #1\n_rlnOriginYAngst #2\n_rlnOriginZAngst #3\n0 0 0\n")
    with pytest.raises(ValueError, match="Duplicate"):
        validate_named_trajectories(path, particles)


@pytest.mark.parametrize("missing", ["rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"])
def test_missing_shift_component_rejected(motion_case, missing):
    path, blocks, particles = motion_case
    blocks["tomo/P2"] = blocks["tomo/P2"].drop(columns=missing)
    starfile.write(blocks, path, overwrite=True)
    with pytest.raises(ValueError, match="requires rlnOrigin"):
        validate_named_trajectories(path, particles)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, "not_a_shift"])
def test_invalid_shift_values_rejected(motion_case, bad):
    path, blocks, particles = motion_case
    blocks["tomo/P2"]["rlnOriginXAngst"] = [bad, 0.0, 0.0]
    starfile.write(blocks, path, overwrite=True)
    with pytest.raises(ValueError, match="nonnumeric|non-finite"):
        validate_named_trajectories(path, particles)


@pytest.mark.parametrize("count", [0, 2, 4, 3.5, np.nan, "invalid"])
def test_wrong_particle_count_rejected(motion_case, count):
    path, blocks, particles = motion_case
    blocks["general"]["rlnParticleNumber"] = count
    starfile.write(blocks, path, overwrite=True)
    with pytest.raises(ValueError, match="rlnParticleNumber"):
        validate_named_trajectories(path, particles)


def test_missing_count_rejected(motion_case):
    path, blocks, particles = motion_case
    blocks["general"] = {"other": 3}
    starfile.write(blocks, path, overwrite=True)
    with pytest.raises(ValueError, match="rlnParticleNumber"):
        validate_named_trajectories(path, particles)


def test_general_must_precede_particle_blocks(motion_case):
    path, blocks, particles = motion_case
    blocks["general"] = blocks.pop("general")
    starfile.write(blocks, path, overwrite=True)
    with pytest.raises(ValueError, match="begin with data_general"):
        validate_named_trajectories(path, particles)


@pytest.mark.parametrize("flags", ["[1,2,0]", "[0,0,0]", "invalid", "[]"])
def test_invalid_native_visibility_rejected(motion_case, flags):
    path, _, particles = motion_case
    particles.loc[0, "rlnTomoVisibleFrames"] = flags
    with pytest.raises(ValueError, match="visibility|visible"):
        validate_named_trajectories(path, particles)


def test_duplicate_selected_identity_rejected(motion_case):
    path, _, particles = motion_case
    particles.loc[0, "rlnTomoParticleName"] = "tomo/P10"
    with pytest.raises(ValueError, match="unique"):
        validate_named_trajectories(path, particles)

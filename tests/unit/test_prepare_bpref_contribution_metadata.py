from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from scripts.prepare_bpref_contribution_metadata import prepare_metadata

pytestmark = pytest.mark.unit


def test_prepare_metadata_resolves_identities_and_hashes_frozen_stack(tmp_path):
    stack_path = tmp_path / "particles.mrcs"
    stack_path.write_bytes(b"frozen-stack")
    star_path = tmp_path / "particles.star"
    star_path.write_text(
        "data_particles\n\nloop_\n_rlnImageName #1\n"
        "1@particles.mrcs\n2@particles.mrcs\n"
    )
    image_names_path = tmp_path / "capture" / "image_names.npy"
    manifest_path = tmp_path / "capture" / "manifest.json"

    manifest = prepare_metadata(
        star_path=star_path,
        image_names_path=image_names_path,
        manifest_path=manifest_path,
    )

    identities = np.load(image_names_path, allow_pickle=False)
    assert identities.tolist() == [f"1@{stack_path}", f"2@{stack_path}"]
    assert manifest["source_stack_sha256"] == hashlib.sha256(b"frozen-stack").hexdigest()
    assert json.loads(manifest_path.read_text()) == manifest


def test_prepare_metadata_rejects_multiple_source_stacks(tmp_path):
    (tmp_path / "a.mrcs").write_bytes(b"a")
    (tmp_path / "b.mrcs").write_bytes(b"b")
    star_path = tmp_path / "particles.star"
    star_path.write_text(
        "data_particles\n\nloop_\n_rlnImageName #1\n1@a.mrcs\n1@b.mrcs\n"
    )

    with pytest.raises(ValueError, match="one frozen source stack"):
        prepare_metadata(
            star_path=star_path,
            image_names_path=tmp_path / "image_names.npy",
            manifest_path=tmp_path / "manifest.json",
        )

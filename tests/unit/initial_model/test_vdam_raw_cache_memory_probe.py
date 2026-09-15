from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from recovar import utils
from recovar.data_io import image_loader, starfile
from scripts import probe_vdam_raw_cache_memory as memory_probe

pytestmark = pytest.mark.unit


def _fixture(tmp_path):
    data = np.arange(7 * 5 * 5, dtype=np.float32).reshape(7, 5, 5)
    stack = tmp_path / "particles.mrcs"
    utils.write_mrc(str(stack), data)
    order = np.array([3, 0, 6, 1, 5, 2, 4], dtype=np.int32)
    table = pd.DataFrame(
        {"_rlnImageName": [f"{int(index) + 1}@{stack.name}" for index in order]}
    )
    particles = tmp_path / "particles.star"
    starfile.write_star(str(particles), data=table)
    return data, order, particles


def test_probe_traces_one_cache_and_proves_streamed_byte_identity(monkeypatch, tmp_path):
    data, order, particles = _fixture(tmp_path)
    output = tmp_path / "probe.json"
    monkeypatch.setenv("RECOVAR_CACHE_DIR", "")

    result = memory_probe.probe(
        input_star=particles,
        data_dir=tmp_path,
        output_json=output,
        comparison_batch_size=3,
    )

    assert json.loads(output.read_text()) == result
    assert result["schema"] == memory_probe.SCHEMA
    loader = result["loader"]
    assert loader["cached_shape"] == [7, 5, 5]
    assert loader["cached_nbytes"] == data.nbytes
    assert loader["topology_before"]["mapping_is_contiguous_set"] is True
    assert loader["topology_before"]["mapping_is_strictly_ascending"] is False
    assert loader["topology_before"]["leaf_cached"] == [False]
    assert loader["topology_after"]["leaf_cached"] == [False]
    trace = result["tracemalloc"]
    assert trace["retained_delta_bytes"] >= data.nbytes
    assert trace["peak_above_baseline_bytes"] >= trace["retained_delta_bytes"]
    expected_digest = hashlib.sha256(data[order].tobytes(order="C")).hexdigest()
    exact = result["bitwise_equivalence"]
    assert exact["exact"] is True
    assert exact["cached_sha256"] == expected_digest
    assert exact["streamed_uncached_sha256"] == expected_digest
    assert exact["compared_images"] == 7
    assert exact["batch_count"] == 3
    assert exact["first_mismatch_index"] is None
    assert exact["comparison_loader_cached"] is False


def test_probe_rejects_inherited_staging_cache(monkeypatch, tmp_path):
    _data, _order, particles = _fixture(tmp_path)
    monkeypatch.setenv("RECOVAR_CACHE_DIR", str(tmp_path / "staging"))

    with pytest.raises(RuntimeError, match="RECOVAR_CACHE_DIR must be explicitly empty"):
        memory_probe.probe(
            input_star=particles,
            data_dir=tmp_path,
            output_json=tmp_path / "probe.json",
            comparison_batch_size=3,
        )


def test_probe_records_first_streamed_byte_mismatch(monkeypatch, tmp_path):
    _data, _order, particles = _fixture(tmp_path)
    output = tmp_path / "probe.json"
    monkeypatch.setenv("RECOVAR_CACHE_DIR", "")
    original_loader = image_loader.StarLoader
    constructions = 0

    def loader_factory(*args, **kwargs):
        nonlocal constructions
        loader = original_loader(*args, **kwargs)
        constructions += 1
        if constructions == 2:
            original_get = loader.get

            def corrupted_get(indices):
                batch = original_get(indices).copy()
                batch.view(np.uint8).reshape(-1)[0] ^= np.uint8(1)
                return batch

            loader.get = corrupted_get
        return loader

    monkeypatch.setattr(image_loader, "StarLoader", loader_factory)
    result = memory_probe.probe(
        input_star=particles,
        data_dir=tmp_path,
        output_json=output,
        comparison_batch_size=3,
    )

    evidence = result["bitwise_equivalence"]
    assert evidence["exact"] is False
    assert evidence["first_mismatch_index"] == 0
    assert evidence["cached_sha256"] != evidence["streamed_uncached_sha256"]


def test_probe_rejects_nonpositive_batch_and_existing_output(monkeypatch, tmp_path):
    _data, _order, particles = _fixture(tmp_path)
    output = tmp_path / "probe.json"
    monkeypatch.setenv("RECOVAR_CACHE_DIR", "")

    with pytest.raises(ValueError, match="comparison-batch-size must be positive"):
        memory_probe.probe(
            input_star=particles,
            data_dir=tmp_path,
            output_json=output,
            comparison_batch_size=0,
        )
    output.write_text("occupied\n")
    with pytest.raises(FileExistsError, match="output already exists"):
        memory_probe.probe(
            input_star=particles,
            data_dir=tmp_path,
            output_json=output,
            comparison_batch_size=3,
        )

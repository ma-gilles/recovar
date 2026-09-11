"""Captured rank-local projector state prefixes are discovered by one owner."""

import inspect

import pytest

from recovar.em.dense_single_volume.helpers import relion_projector_capture as cap


def test_owner_discovers_ranks_and_rejects_duplicates(tmp_path):
    with pytest.raises(cap.ProjectorLoadError, match="no captured"):
        cap._captured_rank_prefixes(tmp_path, 3)
    (tmp_path / "state_iter3_rank0_device0_class0_state_schema_version.bin").write_bytes(b"")
    (tmp_path / "state_iter3_rank1_device1_class0_state_schema_version.bin").write_bytes(b"")
    prefixes = cap._captured_rank_prefixes(tmp_path, 3)
    assert sorted(prefixes) == [0, 1] and str(prefixes[1]).endswith("state_iter3_rank1_device1_class0_")
    (tmp_path / "state_iter3_rank1_device2_class0_state_schema_version.bin").write_bytes(b"")
    with pytest.raises(cap.ProjectorLoadError, match="multiple captured devices"):
        cap._captured_rank_prefixes(tmp_path, 3)


def test_loaders_use_the_owner():
    src = inspect.getsource(cap)
    assert src.count("rank_prefixes = _captured_rank_prefixes(dump_dir, iteration)") == 2
    assert src.count('_class0_state_schema_version.bin"') == 1

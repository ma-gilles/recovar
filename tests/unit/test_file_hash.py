"""File identity checks independent of diagnostic acceptance policies."""

import hashlib

import pytest

from recovar.utils.file_hash import sha256_file


@pytest.mark.unit
@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (b"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        (b"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
    ],
)
def test_sha256_file_known_digests(tmp_path, content, expected):
    path = tmp_path / "artifact"
    path.write_bytes(content)
    assert sha256_file(path) == expected


@pytest.mark.unit
def test_sha256_file_includes_data_after_first_block(tmp_path):
    content = bytes(range(256)) * 32768 + b"tail beyond the first block"
    path = tmp_path / "artifact"
    path.write_bytes(content)
    assert sha256_file(path) == hashlib.sha256(content).hexdigest()


@pytest.mark.unit
def test_sha256_file_propagates_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        sha256_file(tmp_path / "missing")


@pytest.mark.unit
def test_sha256_file_propagates_directory_error(tmp_path):
    with pytest.raises(IsADirectoryError):
        sha256_file(tmp_path)

import numpy as np
import pytest

pytest.importorskip("torch")

from recovar import cryodrgn_mrcfile as cmrc


def test_write_parse_mrc_roundtrip(tmp_path):
    vol = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    path = tmp_path / "test.mrc"
    cmrc.write_mrc(str(path), vol, is_vol=True, Apix=1.5)

    parsed, header = cmrc.parse_mrc(str(path))
    assert parsed.shape == vol.shape
    assert np.allclose(parsed, vol)
    assert header.apix == 1.5


def test_get_mrc_header_stack_sets_ispg_zero():
    stack = np.ones((3, 4, 5), dtype=np.float32)
    header = cmrc.get_mrc_header(stack, is_vol=False)
    assert header.fields["ispg"] == 0
    assert header.fields["nz"] == 3


def test_fix_mrc_header_repairs_fields():
    vol = np.ones((4, 4, 4), dtype=np.float32)
    header = cmrc.get_mrc_header(vol, is_vol=True)
    header.fields["cmap"] = b"BAD!"
    header.fields["stamp"] = b"\x00\x00\x00\x00"
    fixed = cmrc.fix_mrc_header(header)
    assert fixed.fields["cmap"] == b"MAP "
    assert fixed.fields["stamp"] in cmrc.MRCHeader.MACHST_FOR_ENDIANNESS.values()


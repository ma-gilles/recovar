"""
Downsampling helpers.

This module previously contained commented-out experimental code copied from
external tooling. It is intentionally left as a small explicit stub until a
maintained in-tree implementation is added.
"""

from __future__ import annotations


def downsample_not_available() -> None:
    """Raise a clear error for removed legacy functionality."""
    raise NotImplementedError(
        "Legacy downsample helpers were removed. Use external cryodrgn tooling "
        "or add a maintained implementation in recovar/downsample.py."
    )

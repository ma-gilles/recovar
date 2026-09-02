#!/usr/bin/env python3
"""Strictly audit a complete K-class RECOVAR/RELION FSC trajectory."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.audit_k4_fsc_trajectory import main  # noqa: E402, I001


if __name__ == "__main__":
    raise SystemExit(main())

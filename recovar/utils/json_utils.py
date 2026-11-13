"""Conversion helpers for numerical JSON reports."""

from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(value: Any):
    """Normalize NumPy values and paths in nested report data.

    Dictionary keys become strings, and tuples become lists. Other types and
    nonfinite values pass through; encoding and caller-specific validation
    determine whether those values are allowed.
    """
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value

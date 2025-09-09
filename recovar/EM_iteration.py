"""Backward-compatible shim for the old EM_iteration API.

This module re-exports the new implementations from `recovar.em` so that
legacy imports like `from recovar import EM_iteration as em` or
`from recovar.EM_iteration import EMState` keep working.
"""

from .em import *  # noqa: F401,F403



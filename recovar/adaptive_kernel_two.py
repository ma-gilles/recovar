"""
Deprecated experimental adaptive kernel module.

This module intentionally exposes no runtime implementation. The previous body
was commented-out prototype code and was removed to keep the codebase clean and
avoid confusion about supported features.
"""

__all__ = ()


def __getattr__(name: str):
    raise AttributeError(
        f"{__name__}.{name} is unavailable. The adaptive kernel prototype "
        "was removed because it was not a supported runtime path."
    )

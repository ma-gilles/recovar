"""Optional nvtx import with no-op fallback.

Usage in any module::

    from recovar.utils.nvtx_shim import nvtx
"""

import functools

try:
    import nvtx  # type: ignore[import-untyped]
except ImportError:

    class _NvtxStub:
        """Drop-in replacement when the ``nvtx`` package is not installed."""

        @staticmethod
        def annotate(msg="", color=None, domain=None):
            class _NoOp:
                def __call__(self, fn):
                    @functools.wraps(fn)
                    def wrapper(*args, **kwargs):
                        return fn(*args, **kwargs)

                    return wrapper

                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

            return _NoOp()

    nvtx = _NvtxStub()  # type: ignore[assignment]

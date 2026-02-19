try:
    import recovar.config  # noqa: F401
except ModuleNotFoundError:
    # Allow importing lightweight modules (e.g., utils and FFT helpers)
    # in environments where optional heavy dependencies are absent.
    pass

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.4.0"

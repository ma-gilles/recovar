"""Unified plot styling for RECOVAR.

Provides consistent colors, sizes, fonts, and helper functions across all
recovar plots. Import this module at the top of any plotting code.

Usage::

    from recovar.output.plot_style import apply_style, safe_savefig, COLORS
    apply_style()
    fig, ax = plt.subplots()
    ...
    safe_savefig(fig, "plots/my_plot.png")
"""

import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Standard dimensions
# ---------------------------------------------------------------------------

FIGSIZE_SMALL = (6, 5)         # Single panel
FIGSIZE_MEDIUM = (10, 8)       # 2x2 grid or moderate detail
FIGSIZE_LARGE = (16, 12)       # Summary panels (4-6 subplots)
FIGSIZE_WIDE = (14, 6)         # Side-by-side comparison

DPI = 150                      # Default save DPI

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

COLORS = {
    "primary": "#2563eb",       # Blue — main data
    "secondary": "#dc2626",     # Red — comparisons, thresholds
    "accent": "#059669",        # Green — good/pass
    "warning": "#d97706",       # Orange — caution
    "muted": "#6b7280",         # Gray — reference lines
    "dark": "#1f2937",          # Near-black — text/axes
    "light": "#f3f4f6",         # Light gray — backgrounds
}

# Categorical palette for clusters/trajectories (color-blind friendly)
CATEGORICAL = [
    "#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed",
    "#db2777", "#0891b2", "#65a30d", "#c2410c", "#4f46e5",
]

# ---------------------------------------------------------------------------
# Font sizes
# ---------------------------------------------------------------------------

FONTSIZE_TITLE = 14
FONTSIZE_SUBTITLE = 12
FONTSIZE_LABEL = 11
FONTSIZE_TICK = 10
FONTSIZE_ANNOTATION = 9

# ---------------------------------------------------------------------------
# Style application
# ---------------------------------------------------------------------------

def apply_style():
    """Set matplotlib defaults for consistent recovar plots.

    Call once at the start of any plotting code. Safe to call multiple times.
    """
    matplotlib.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox_inches": "tight",
        "savefig.pad_inches": 0.1,
        "font.size": FONTSIZE_TICK,
        "axes.titlesize": FONTSIZE_TITLE,
        "axes.labelsize": FONTSIZE_LABEL,
        "xtick.labelsize": FONTSIZE_TICK,
        "ytick.labelsize": FONTSIZE_TICK,
        "legend.fontsize": FONTSIZE_ANNOTATION,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": COLORS["dark"],
        "text.color": COLORS["dark"],
        "xtick.color": COLORS["dark"],
        "ytick.color": COLORS["dark"],
    })


def safe_savefig(fig, path, close=True, **kwargs):
    """Save a figure with NFS-safe error handling.

    Creates parent directories if needed. Handles tight_layout overflow
    errors gracefully.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : str
        Output file path.
    close : bool
        Whether to close the figure after saving.
    **kwargs
        Extra arguments passed to ``fig.savefig()``.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    kwargs.setdefault("dpi", DPI)
    kwargs.setdefault("bbox_inches", "tight")
    try:
        fig.savefig(path, **kwargs)
    except (ValueError, OverflowError):
        # tight_layout can fail with extreme axis ranges
        try:
            fig.savefig(path, dpi=kwargs["dpi"])
        except Exception as e:
            logger.warning("Could not save figure to %s: %s", path, e)
    except (IOError, OSError) as e:
        logger.warning("Could not save figure to %s: %s", path, e)
    if close:
        plt.close(fig)


def volume_slices(vol, title="", cmap="gray"):
    """Plot central slices of a 3D volume along all three axes.

    Parameters
    ----------
    vol : ndarray, shape (D, D, D)
        3D volume.
    title : str
        Figure title.
    cmap : str
        Colormap.

    Returns
    -------
    fig : Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    mid = vol.shape[0] // 2
    for ax, sl, label in zip(axes, [vol[mid], vol[:, mid], vol[:, :, mid]],
                              ["XY", "XZ", "YZ"]):
        ax.imshow(sl, cmap=cmap, origin="lower")
        ax.set_title(label, fontsize=FONTSIZE_SUBTITLE)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=FONTSIZE_TITLE, fontweight="bold")
    fig.tight_layout()
    return fig


def make_summary_figure(n_rows, n_cols, title="", figsize=None):
    """Create a figure with a grid of subplots for summary panels.

    Parameters
    ----------
    n_rows, n_cols : int
        Grid dimensions.
    title : str
        Suptitle.
    figsize : tuple, optional
        Figure size; defaults to scaled FIGSIZE_LARGE.

    Returns
    -------
    fig, axes : Figure, ndarray of Axes
    """
    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if title:
        fig.suptitle(title, fontsize=FONTSIZE_TITLE + 2, fontweight="bold", y=1.02)
    return fig, axes

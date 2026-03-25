"""Centralized output path definitions for RECOVAR pipeline results.

All output file paths are defined here as the single source of truth.
Both the saving side (pipeline.py, analyze.py) and the loading side
(PipelineOutput) should use these definitions to avoid path mismatches.
"""

import os


# ---------------------------------------------------------------------------
# Subdirectory names
# ---------------------------------------------------------------------------

MODEL_DIR = "model"
OUTPUT_DIR = "output"
VOLUMES_DIR = os.path.join(OUTPUT_DIR, "volumes")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")


# ---------------------------------------------------------------------------
# Model file basenames (inside MODEL_DIR)
# ---------------------------------------------------------------------------

PARAMS_FILE = "params.pkl"
EMBEDDINGS_FILE = "embeddings.pkl"
COVARIANCE_COLS_FILE = "covariance_cols.pkl"
HALFSETS_FILE = "halfsets.pkl"
PARTICLES_HALFSETS_FILE = "particles_halfsets.pkl"
ZS_WITH_COMPLEMENT_FILE = "zs_with_complement.pkl"
METADATA_FILE = "metadata.json"


# ---------------------------------------------------------------------------
# Volume file basenames (inside VOLUMES_DIR)
# ---------------------------------------------------------------------------

MEAN_VOLUME = "mean.mrc"
MEAN_HALF1_UNFIL = "mean_half1_unfil.mrc"
MEAN_HALF2_UNFIL = "mean_half2_unfil.mrc"
MEAN_FILTERED = "mean_filt.mrc"
MASK_VOLUME = "mask.mrc"
DILATED_MASK_VOLUME = "dilated_mask.mrc"
FOCUS_MASK_VOLUME = "focus_mask.mrc"
COMPLEMENT_MASK_VOLUME = "complement_mask.mrc"


def eigenvector_filename(index):
    """Return filename for eigenvector volume at given index."""
    return f"eigen_pos{index:04d}.mrc"


def variance_filename(n_eigs):
    """Return filename for variance volume computed from n_eigs eigenvectors."""
    return f"variance{n_eigs}.mrc"


# ---------------------------------------------------------------------------
# ResultPaths class
# ---------------------------------------------------------------------------

class ResultPaths:
    """Single source of truth for all output file paths.

    Usage::

        paths = ResultPaths("/path/to/outdir")
        paths.ensure_dirs()
        utils.pickle_dump(result, paths.params)
        vol = utils.load_mrc(paths.mean_volume)
    """

    def __init__(self, root_dir):
        self.root = root_dir

    # --- Directories ---

    @property
    def model_dir(self):
        return os.path.join(self.root, MODEL_DIR)

    @property
    def output_dir(self):
        return os.path.join(self.root, OUTPUT_DIR)

    @property
    def volumes_dir(self):
        return os.path.join(self.root, VOLUMES_DIR)

    @property
    def plots_dir(self):
        return os.path.join(self.root, PLOTS_DIR)

    def analysis_dir(self, zdim_key):
        return os.path.join(self.root, f"analysis_{zdim_key}")

    # --- Model files ---

    @property
    def params(self):
        return os.path.join(self.model_dir, PARAMS_FILE)

    @property
    def embeddings(self):
        """Legacy path for monolithic embeddings.pkl (backward compat)."""
        return os.path.join(self.model_dir, EMBEDDINGS_FILE)

    def embedding_zdim_dir(self, zdim):
        """Per-zdim embedding directory, e.g. ``model/zdim_4/``."""
        return os.path.join(self.model_dir, f"zdim_{zdim}")

    @property
    def covariance_cols(self):
        return os.path.join(self.model_dir, COVARIANCE_COLS_FILE)

    @property
    def halfsets(self):
        return os.path.join(self.model_dir, HALFSETS_FILE)

    @property
    def particles_halfsets(self):
        return os.path.join(self.model_dir, PARTICLES_HALFSETS_FILE)

    @property
    def zs_with_complement(self):
        return os.path.join(self.model_dir, ZS_WITH_COMPLEMENT_FILE)

    @property
    def metadata(self):
        return os.path.join(self.model_dir, METADATA_FILE)

    # --- Volume files ---

    @property
    def mean_volume(self):
        return os.path.join(self.volumes_dir, MEAN_VOLUME)

    @property
    def mean_half1_unfil(self):
        return os.path.join(self.volumes_dir, MEAN_HALF1_UNFIL)

    @property
    def mean_half2_unfil(self):
        return os.path.join(self.volumes_dir, MEAN_HALF2_UNFIL)

    @property
    def mean_filtered(self):
        return os.path.join(self.volumes_dir, MEAN_FILTERED)

    @property
    def mask_volume(self):
        return os.path.join(self.volumes_dir, MASK_VOLUME)

    @property
    def dilated_mask_volume(self):
        return os.path.join(self.volumes_dir, DILATED_MASK_VOLUME)

    @property
    def focus_mask_volume(self):
        return os.path.join(self.volumes_dir, FOCUS_MASK_VOLUME)

    @property
    def complement_mask_volume(self):
        return os.path.join(self.volumes_dir, COMPLEMENT_MASK_VOLUME)

    def eigenvector(self, index):
        return os.path.join(self.volumes_dir, eigenvector_filename(index))

    def variance(self, n_eigs):
        return os.path.join(self.volumes_dir, variance_filename(n_eigs))

    # --- Other outputs ---

    @property
    def command_txt(self):
        return os.path.join(self.root, "command.txt")

    @property
    def run_log(self):
        return os.path.join(self.root, "run.log")

    # --- Directory creation ---

    def ensure_dirs(self):
        """Create all standard output directories."""
        for d in (self.root, self.model_dir, self.output_dir,
                  self.volumes_dir, self.plots_dir):
            os.makedirs(d, exist_ok=True)

    def ensure_model_dir(self):
        """Create the model directory only."""
        os.makedirs(self.model_dir, exist_ok=True)

    def ensure_volumes_dir(self):
        """Create the volumes directory only."""
        os.makedirs(self.volumes_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# Analysis output naming (downstream: kmeans, trajectories)
# ---------------------------------------------------------------------------

class AnalysisPaths:
    """Path helpers for downstream analysis outputs (kmeans, trajectories).

    Follows RELION-inspired conventions:
    - 1-indexed, zero-padded volume names (center001.mrc, state001.mrc)
    - Primary volumes flat in the output directory
    - Half-maps alongside: center001_half1_unfil.mrc
    - Diagnostics in subdirectories: diagnostics/center001/
    """

    def __init__(self, analysis_dir):
        self.root = analysis_dir

    @property
    def kmeans_dir(self):
        return os.path.join(self.root, "kmeans")

    @property
    def plots_dir(self):
        return os.path.join(self.root, "plots")

    def traj_dir(self, index):
        """Return trajectory directory (1-indexed, zero-padded)."""
        return os.path.join(self.root, f"traj{index:03d}")

    @staticmethod
    def vol_stem(prefix, index):
        """Volume stem without extension, e.g. 'center000'."""
        return f"{prefix}{index:03d}"

    @staticmethod
    def vol_filename(prefix, index):
        """Primary volume filename, e.g. 'center000.mrc'."""
        return f"{prefix}{index:03d}.mrc"

    @staticmethod
    def halfmap_filename(prefix, index, half):
        """Half-map filename, e.g. 'center000_half1_unfil.mrc'."""
        return f"{prefix}{index:03d}_half{half}_unfil.mrc"

    @staticmethod
    def diagnostics_subdir(prefix, index):
        """Diagnostics subdirectory, e.g. 'diagnostics/center000'."""
        return os.path.join("diagnostics", f"{prefix}{index:03d}")


# ---------------------------------------------------------------------------
# Per-volume output paths (used by heterogeneity_volume.py)
# ---------------------------------------------------------------------------

class VolumeOutputPaths:
    """Path abstraction for a single reconstructed volume's output files.

    Primary outputs (filtered volume, half-maps) are placed directly in
    *output_dir*.  Diagnostics (params, local resolution, split choices)
    go into ``output_dir/diagnostics/{prefix}{index:03d}/``.

    This replaces the ad-hoc string concatenation previously used in
    ``heterogeneity_volume.make_volumes_kernel_estimate_local``.

    Usage::

        vp = VolumeOutputPaths("/path/to/kmeans", "center", 0)
        vp.ensure_dirs()
        write_mrc(vp.filtered, volume)
        write_mrc(vp.half1_unfil, half1)
        pickle_dump(params, vp.params)

    Parameters
    ----------
    output_dir : str
        Parent output directory (e.g. the kmeans/ or traj000/ folder).
    prefix : str
        Volume name prefix (e.g. ``"center"``, ``"state"``).
    index : int
        Zero-based volume index.
    """

    def __init__(self, output_dir, prefix, index):
        self.output_dir = output_dir
        self.prefix = prefix
        self.index = index
        self._stem = f"{prefix}{index:03d}"
        self._diag_dir = os.path.join(output_dir, "diagnostics", self._stem)

    @property
    def stem(self):
        """Volume stem without extension, e.g. ``'center000'``."""
        return self._stem

    @property
    def diag_dir(self):
        """Diagnostics subdirectory for this volume."""
        return self._diag_dir

    # --- Primary outputs (in output_dir) ---

    @property
    def filtered(self):
        """Filtered volume: ``{stem}.mrc``."""
        return os.path.join(self.output_dir, f"{self._stem}.mrc")

    @property
    def half1_unfil(self):
        """Unfiltered half-map 1: ``{stem}_half1_unfil.mrc``."""
        return os.path.join(self.output_dir, f"{self._stem}_half1_unfil.mrc")

    @property
    def half2_unfil(self):
        """Unfiltered half-map 2: ``{stem}_half2_unfil.mrc``."""
        return os.path.join(self.output_dir, f"{self._stem}_half2_unfil.mrc")

    @property
    def unfil(self):
        """Unfiltered combined volume: ``{stem}_unfil.mrc``."""
        return os.path.join(self.output_dir, f"{self._stem}_unfil.mrc")

    # --- Diagnostics (in diagnostics/{stem}/) ---

    @property
    def locres(self):
        """Local resolution map."""
        return os.path.join(self._diag_dir, "local_resolution.mrc")

    @property
    def sampling(self):
        """Sampling volume (diagnostic)."""
        return os.path.join(self._diag_dir, "sampling.mrc")

    @property
    def params(self):
        """Heterogeneity parameters pickle."""
        return os.path.join(self._diag_dir, "params.pkl")

    @property
    def split_choice(self):
        """Per-shell bin selection pickle."""
        return os.path.join(self._diag_dir, "split_choice.pkl")

    @property
    def choice(self):
        """Per-voxel bin selection MRC (locmost_likely mode)."""
        return os.path.join(self._diag_dir, "choice.mrc")

    @property
    def choice_smooth(self):
        """Smoothed per-voxel bin selection MRC."""
        return os.path.join(self._diag_dir, "choice_smooth.mrc")

    @property
    def heterogeneity_distances(self):
        """Per-image heterogeneity distances text file."""
        return os.path.join(self._diag_dir, "heterogeneity_distances.txt")

    @property
    def latent_coords(self):
        """Latent coordinates text file for this volume."""
        return os.path.join(self._diag_dir, "latent_coords.txt")

    def estimates_dir(self, half, filtered=False):
        """Directory for all kernel regression estimates (debug only).

        Parameters
        ----------
        half : int
            Half-set index (1 or 2).
        filtered : bool
            If True, return the filtered estimates directory.
        """
        if filtered:
            return os.path.join(self._diag_dir, "estimates_filt")
        return os.path.join(self._diag_dir, f"estimates_half{half}_unfil")

    # --- Debug outputs (in diagnostics/{stem}/) ---

    @property
    def filtered_smooth(self):
        """Smoothed filtered volume (debug)."""
        return os.path.join(self._diag_dir, "filtered_smooth.mrc")

    @property
    def locres_smooth(self):
        """Smoothed local resolution map (debug)."""
        return os.path.join(self._diag_dir, "local_resolution_smooth.mrc")

    @property
    def filtered_before(self):
        """Filter-before-choose volume (debug)."""
        return os.path.join(self._diag_dir, "filtered_before.mrc")

    @property
    def filtered_before_smooth(self):
        """Smoothed filter-before-choose volume (debug)."""
        return os.path.join(self._diag_dir, "filtered_before_smooth.mrc")

    @property
    def cv_half1_unfil(self):
        """Cross-validation estimate, half 1 (debug)."""
        return os.path.join(self._diag_dir, "CV_estimates_half1_unfil.mrc")

    @property
    def cv_half2_unfil(self):
        """Cross-validation estimate, half 2 (debug)."""
        return os.path.join(self._diag_dir, "CV_estimates_half2_unfil.mrc")

    @property
    def cv_noise_half1(self):
        """Cross-validation noise, half 1 (debug)."""
        return os.path.join(self._diag_dir, "CV_noise_half1.mrc")

    @property
    def cv_noise_half2(self):
        """Cross-validation noise, half 2 (debug)."""
        return os.path.join(self._diag_dir, "CV_noise_half2.mrc")

    # --- Directory creation ---

    def ensure_dirs(self):
        """Create output_dir and diagnostics subdirectory."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self._diag_dir, exist_ok=True)


def resolve_volume_diag_path(vol_folder, filename, prefix=None, index=None):
    """Resolve a diagnostic file path with backward compatibility.

    Checks the new ``diagnostics/{stem}/`` layout first, then falls back
    to the old flat layout where files lived directly in *vol_folder*.

    Parameters
    ----------
    vol_folder : str
        The volume output directory (e.g. ``kmeans/`` or a flat diag dir).
    filename : str
        The file to find (e.g. ``"params.pkl"``).
    prefix : str, optional
        Volume prefix for new layout lookup.
    index : int, optional
        Volume index for new layout lookup.

    Returns
    -------
    str
        Resolved path (may not exist if file is missing in both locations).
    """
    # Try new layout: diagnostics/{stem}/{filename}
    if prefix is not None and index is not None:
        stem = f"{prefix}{index:03d}"
        new_path = os.path.join(vol_folder, "diagnostics", stem, filename)
        if os.path.isfile(new_path):
            return new_path

    # Fall back to old flat layout
    flat_path = os.path.join(vol_folder, filename)
    if os.path.isfile(flat_path):
        return flat_path

    # Default to new layout path (even if it doesn't exist yet)
    if prefix is not None and index is not None:
        stem = f"{prefix}{index:03d}"
        return os.path.join(vol_folder, "diagnostics", stem, filename)
    return flat_path

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
        return os.path.join(self.model_dir, EMBEDDINGS_FILE)

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

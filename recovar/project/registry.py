"""Job type registry — maps CLI command names to job type metadata."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class JobType:
    """Metadata for a recovar job type.

    Attributes
    ----------
    name : str
        CamelCase type name (also the directory name), e.g. ``"Pipeline"``.
    command : str
        CLI command name, e.g. ``"pipeline"``.
    produces_volumes : bool
        Whether this job type creates MRC volume files.
    needs_pipeline : bool
        Whether the command requires a pipeline result_dir as input.
    description : str
        One-line description shown in ``recovar project status``.
    """

    name: str
    command: str
    produces_volumes: bool
    needs_pipeline: bool
    description: str

    @property
    def dir_name(self):
        """Directory name under the project root (same as name)."""
        return self.name


# All registered job types, keyed by CLI command name.
JOB_TYPES = {
    "pipeline": JobType(
        "Pipeline", "pipeline", True, False,
        "Mean reconstruction, covariance, PCA, embedding",
    ),
    "analyze": JobType(
        "Analyze", "analyze", True, True,
        "K-means clustering, trajectories, UMAP",
    ),
    "compute_state": JobType(
        "ComputeState", "compute_state", True, True,
        "Volumes at arbitrary latent points",
    ),
    "compute_trajectory": JobType(
        "ComputeTrajectory", "compute_trajectory", True, True,
        "Volumes along a latent-space path",
    ),
    "estimate_conformational_density": JobType(
        "Density", "estimate_conformational_density", False, True,
        "Deconvolved conformational density",
    ),
    "estimate_stable_states": JobType(
        "StableStates", "estimate_stable_states", False, False,
        "Local maxima of conformational density",
    ),
    "junk_particle_detection": JobType(
        "JunkDetection", "junk_particle_detection", False, True,
        "FSC-based junk particle detection",
    ),
    "outlier_detection": JobType(
        "OutlierDetection", "outlier_detection", False, True,
        "Anomaly-based outlier detection",
    ),
    "postprocess": JobType(
        "Postprocess", "postprocess", True, False,
        "FSC filtering of halfmaps",
    ),
    "downsample": JobType(
        "Downsample", "downsample", False, False,
        "Downsample particle images",
    ),
    "extract_image_subset": JobType(
        "ExtractSubset", "extract_image_subset", False, False,
        "Extract image subset by local resolution",
    ),
    "extract_image_subset_from_kmeans": JobType(
        "ExtractSubset", "extract_image_subset_from_kmeans", False, False,
        "Extract image subset by k-means cluster",
    ),
    "pipeline_with_outliers": JobType(
        "PipelineWithOutliers", "pipeline_with_outliers", True, False,
        "Iterative pipeline with outlier removal",
    ),
    "reconstruct_from_external_embedding": JobType(
        "ReconstructExternal", "reconstruct_from_external_embedding", True, False,
        "Volumes from external embeddings",
    ),
}


def get_job_type(command_name: str) -> Optional[JobType]:
    """Look up a job type by CLI command name. Returns None if not found."""
    return JOB_TYPES.get(command_name)

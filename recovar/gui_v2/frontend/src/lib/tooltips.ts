/** Tooltip text for job form parameters. Single source of truth per PHASE1.md. */
export const tooltips: Record<string, string> = {
  "pipeline.particles":
    "Input particle images. Accepts .star (RELION), .cs (cryoSPARC), .mrcs (MRC stack), or .txt (list of .mrcs paths).",
  "pipeline.mask":
    "Solvent mask defining the molecular envelope. Use 'from_halfmaps' to auto-generate from consensus reconstruction, 'sphere' for a spherical mask, or provide a .mrc file.",
  "pipeline.zdim":
    "Latent space dimensions to compute (comma-separated). Each value runs an independent embedding. Start with the default (1,2,4,10,20). Higher values capture more heterogeneity modes but are slower.",
  "pipeline.downsample":
    "Downsample particle images to this box size before processing. Default 256. Lower values are faster but lose high-resolution information.",
  "pipeline.lazy":
    "Enable lazy loading. Required when particle images are too large to fit in memory. Slower due to repeated disk reads.",
  "pipeline.correct_contrast":
    "Correct per-particle contrast variation (amplitude scaling). Recommended for datasets with heterogeneous ice thickness or variable defocus.",
  "pipeline.output_name":
    "Name for the output directory. Will be created under Pipeline/ in the project directory.",
  "pipeline.focus_mask":
    "Optional focus mask for focused classification. Must be a .mrc file with the same box size as particles.",
  "pipeline.datadir":
    "Override data directory for particle images. Use when .star file references relative paths from a different location.",
  "pipeline.n_images":
    "Limit the number of images to use. Useful for quick tests with a subset of the data.",
  "pipeline.halfsets":
    "Column name for halfset assignment in the .star file.",
  "pipeline.poses":
    "Column name prefix for poses in the .star file.",
  "pipeline.ctf":
    "Column name prefix for CTF parameters in the .star file.",
  "pipeline.tilt_series":
    "Enable tilt-series mode for cryo-ET data.",
  "pipeline.strip_prefix":
    "Prefix to strip from particle image paths in the .star file.",
  "analyze.result_dir":
    "Pipeline output directory to analyze. Must contain model/ and output/ subdirectories.",
  "analyze.zdim":
    "Which latent dimension to analyze. Must be one of the zdim values computed by the pipeline.",
  "analyze.n_clusters":
    "Number of k-means clusters for partitioning the latent space. Default 40. More clusters = finer partitioning but smaller per-cluster particle counts.",
  "analyze.n_trajectories":
    "Number of linear trajectories to compute through the latent space. Default 0 (skip). Each trajectory generates a series of volumes.",
  "compute_state.result_dir":
    "Pipeline output directory containing the model to use for volume reconstruction.",
  "compute_state.zdim":
    "Latent dimension to use for reconstruction.",
  "compute_state.latent_coords":
    "Comma-separated latent coordinates specifying the point in z-space to reconstruct.",
  "compute_trajectory.result_dir":
    "Pipeline output directory containing the model to use.",
  "compute_trajectory.zdim":
    "Latent dimension to use for the trajectory.",
  "compute_trajectory.z_start":
    "Comma-separated latent coordinates for the trajectory start point.",
  "compute_trajectory.z_end":
    "Comma-separated latent coordinates for the trajectory end point.",
  "compute_trajectory.n_vols":
    "Number of volumes to compute along the trajectory path. Default 6.",
  "density.result_dir":
    "Pipeline output directory containing results to estimate density from.",
  "density.pca_dim":
    "Dimension of PCA space for density estimation (default 4). Runtime increases exponentially; keep at 5 or below.",
  "density.z_dim_used":
    "Latent dimension to use (default: smallest zdim >= pca_dim). Must be at least as large as pca_dim.",
  "density.percentile_reject":
    "Percentile of particles to reject due to large covariance (default 10%).",
  "density.num_disc_points":
    "Number of grid discretization points per dimension. Default: 50 for dim>3, 100 for dim=3, 200 for dim=2.",
  "density.percentile_bound":
    "Percentile bound for grid extent (default 1%). Rejects coordinates above this percentile when setting grid bounds.",
  "stable_states.density":
    "Path to the density .pkl file output by density estimation (e.g. deconv_density_knee.pkl).",
  "stable_states.percent_top":
    "Percentage of top density points to consider when finding stable states (default 1%).",
  "stable_states.n_local_maxs":
    "Number of local maxima (stable states) to find. If <1, uses automatic detection via HDBSCAN.",
  "postprocess.input":
    "Path to first halfmap (.mrc file) or a directory containing volume subdirectories for batch processing.",
  "postprocess.halfmap2":
    "Path to second halfmap. If not provided, auto-detected from input filename.",
  "postprocess.voxel_size":
    "Voxel size in Angstroms. If not provided, read from the MRC file header.",
  "postprocess.B_factor":
    "B-factor for sharpening in Angstroms^2. Default: no B-factor applied.",
  "postprocess.mask_radius":
    "Radius of spherical mask in Angstroms. Default: no mask.",
  "postprocess.fsc_mask":
    "Path to a mask .mrc file for FSC calculation (optional).",
  "postprocess.batch":
    "Process all volumes in a directory. Input should be a volumes directory.",
  "postprocess.estimate_B_factor":
    "Estimate B-factor from power spectrum decay instead of using a fixed value.",
  "postprocess.local":
    "Use local resolution filtering instead of global filtering.",
  "postprocess.apply_mask":
    "Path to a mask .mrc file to apply to the final filtered map.",
  "downsample.particles":
    "Input particle images (.mrcs, .star, .cs, or .txt file).",
  "downsample.target_D":
    "Target box size in pixels (must be even). Particles will be Fourier-cropped to this size.",
  "downsample.datadir":
    "Path prefix for resolving relative image paths in the input file.",
  "downsample.strip_prefix":
    "Prefix to strip from particle image paths in the star/cs file.",
  "downsample.batch_size":
    "Number of images to process per batch (default 1000).",
};

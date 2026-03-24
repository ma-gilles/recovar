#!/usr/bin/env python3

import argparse
import logging
import os
import glob
import numpy as np
from recovar import utils
from recovar.heterogeneity import locres
import recovar.core.fourier_transform_utils as fourier_transform_utils

logger = logging.getLogger(__name__)

def add_args(parser: argparse.ArgumentParser):
    """Add command line arguments for postprocessing filtering."""
    
    parser.add_argument(
        "input", 
        type=str, 
        help="Path to first halfmap (.mrc file) OR directory containing volume subdirectories"
    )
    
    parser.add_argument(
        "--halfmap2",
        type=str,
        default=None,
        help="Path to second halfmap (if not provided, will auto-detect)"
    )
    
    parser.add_argument(
        "--voxel-size",
        type=float,
        required=False,
        help="Voxel size in Angstroms (if not provided, will be read from the MRC file)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for filtered map (.mrc file) OR output directory for batch processing"
    )
    
    parser.add_argument(
        "--B-factor",
        type=float,
        default=None,
        help="B-factor for sharpening in Angstroms^2 (default: no B-factor)"
    )
    
    parser.add_argument(
        "--mask-radius",
        type=float,
        default=None,
        help="Radius of spherical mask in Angstroms (default: no mask)"
    )
    
    parser.add_argument(
        "--fsc-mask",
        type=str,
        default=None,
        help="Path to a mask .mrc file to use for FSC calculation (optional)"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all volumes in a directory (input should be volumes directory)"
    )
    
    parser.add_argument(
        "--estimate-B-factor",
        action="store_true",
        help="Estimate B-factor by fitting a line to the decay of the power spectrum (log(power) vs. resolution^2) of the input halfmaps."
    )
    
    parser.add_argument(
        "--apply-mask",
        type=str,
        default=None,
        help="Path to a mask .mrc file to apply to the final filtered map (optional)"
    )
    
    # Local filtering options
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local resolution filtering instead of global filtering"
    )
    
    parser.add_argument(
        "--locres-sampling",
        type=float,
        default=25.0,
        help="Sampling distance for local resolution calculation in Angstroms (default: 25.0)"
    )
    
    parser.add_argument(
        "--locres-maskrad",
        type=float,
        default=None,
        help="Radius of local resolution mask in Angstroms (default: 0.5 * locres-sampling)"
    )
    
    parser.add_argument(
        "--locres-edgwidth",
        type=float,
        default=None,
        help="Edge width of local resolution mask in Angstroms (default: locres-sampling)"
    )
    
    parser.add_argument(
        "--locres-minres",
        type=float,
        default=50.0,
        help="Minimum local resolution in Angstroms (default: 50.0)"
    )
    
    parser.add_argument(
        "--fsc-threshold",
        type=float,
        default=1/7,
        help="FSC threshold for resolution determination (default: 1/7)"
    )
    
    parser.add_argument(
        "--filter-edgewidth",
        type=int,
        default=2,
        help="Edge width for FSC filtering in pixels (default: 2)"
    )
    
    parser.add_argument(
        "--use-v2",
        action="store_true",
        help="Use v2 algorithm for local resolution calculation (faster but may be less accurate)"
    )

    from recovar.utils.parser_args import add_project_arg
    add_project_arg(parser)

    return parser


def find_halfmap2(halfmap1_path):
    """Automatically find halfmap2 based on halfmap1 path."""
    
    # Try different common patterns for analyze output and general use
    patterns = [
        ('half1_unfil', 'half2_unfil'),  # analyze output pattern
        ('halfmap1', 'halfmap2'),        # general pattern
        ('half1', 'half2'),              # short pattern
        ('map1', 'map2'),                # map pattern
        ('_1', '_2'),                    # numbered pattern
        ('1', '2')                       # single digit
    ]
    
    for pattern1, pattern2 in patterns:
        if pattern1 in os.path.basename(halfmap1_path):
            halfmap2_path = halfmap1_path.replace(pattern1, pattern2)
            if os.path.exists(halfmap2_path):
                return halfmap2_path
    
    # If no pattern match, try in same directory with common names
    dirname = os.path.dirname(halfmap1_path)
    common_names = ['half2_unfil.mrc', 'halfmap2.mrc', 'half2.mrc', 'map2.mrc']
    
    for name in common_names:
        halfmap2_path = os.path.join(dirname, name)
        if os.path.exists(halfmap2_path):
            return halfmap2_path
    
    return None


def find_volume_directories(volumes_dir):
    """Find all volume subdirectories in a volumes directory."""
    
    # Look for vol* directories (analyze output pattern)
    vol_dirs = glob.glob(os.path.join(volumes_dir, "vol*"))
    if vol_dirs:
        return sorted(vol_dirs)
    
    # Look for any subdirectory that contains halfmaps
    subdirs = []
    for item in os.listdir(volumes_dir):
        item_path = os.path.join(volumes_dir, item)
        if os.path.isdir(item_path):
            # Check if this directory contains halfmaps
            halfmap_patterns = ['half1_unfil.mrc', 'halfmap1.mrc', 'half1.mrc']
            for pattern in halfmap_patterns:
                if os.path.exists(os.path.join(item_path, pattern)):
                    subdirs.append(item_path)
                    break
    
    return sorted(subdirs)


def get_voxel_size_from_mrc(mrc_path):
    import mrcfile
    with mrcfile.open(mrc_path, header_only=True) as mrc:
        vsize = mrc.voxel_size.x
        # Handle tuple/list/array or single float
        return float(vsize)


def estimate_bfactor_from_halfmaps(halfmap1, voxel_size, plot_path=None):
    """
    Robustly estimate B-factor from halfmap power spectrum using Rosenthal & Henderson method.
    
    This implementation follows the approach used in RELION's bfactor_plot.py script,
    which is based on the Rosenthal & Henderson (2003) method for estimating B-factors
    from power spectra.
    
    Parameters:
    -----------
    halfmap1 : np.ndarray
        Input halfmap volume
    voxel_size : float
        Voxel size in Angstroms
    plot_path : str, optional
        Path to save diagnostic plot
        
    Returns:
    --------
    float : Estimated B-factor in Angstrom^2
    
    Raises:
    -------
    RuntimeError : If estimation fails due to insufficient data
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Input validation
    if halfmap1 is None or not isinstance(halfmap1, np.ndarray):
        raise ValueError("halfmap1 must be a numpy array")
    
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    
    if halfmap1.size == 0:
        raise ValueError("halfmap1 cannot be empty")
    
    # Ensure halfmap is 3D
    if halfmap1.ndim != 3:
        raise ValueError(f"halfmap1 must be 3D, got shape {halfmap1.shape}")
    
    # Check for reasonable volume size
    min_size = 32
    if any(s < min_size for s in halfmap1.shape):
        raise ValueError(f"Volume too small for reliable B-factor estimation. Minimum size: {min_size}, got: {halfmap1.shape}")
    
    try:
        # Step 1: Compute power spectrum
        logger.debug("Computing power spectrum...")
        ft = fourier_transform_utils.get_dft3(halfmap1)
        power = np.abs(ft) ** 2
        
        # Step 2: Radial averaging with proper frequency calculation
        logger.debug("Computing radial average...")
        shape = np.array(halfmap1.shape)
        center = shape // 2
        
        # Create coordinate grid
        coords = np.indices(tuple(shape))
        coords = coords - center[:, None, None, None]
        r = np.sqrt(np.sum(coords ** 2, axis=0))
        
        # Flatten arrays for processing
        r_flat = r.flatten()
        power_flat = power.flatten()
        
        # Step 3: Bin by radius with proper frequency calculation
        r_max = np.max(r_flat)
        nbins = min(int(r_max) + 1, shape[0] // 2)  # Limit bins to reasonable number
        
        bin_means = np.full(nbins, np.nan)
        bin_counts = np.zeros(nbins, dtype=int)
        
        # Calculate spatial frequencies properly
        # The frequency step should be 1/(N * voxel_size) where N is the grid size
        # But we need to be careful about the units and scaling
        freq_step = 1.0 / (shape[0] * voxel_size)
        freq_shells = np.arange(nbins) * freq_step
        
        # Debug: Let's also calculate what the actual frequency range should be
        nyquist_freq = 1.0 / (2 * voxel_size)
        logger.debug("Nyquist frequency: %.4f 1/Å", nyquist_freq)
        logger.debug("Frequency step: %.6f 1/Å", freq_step)
        logger.debug("Max frequency in shells: %.4f 1/Å", freq_shells[-1])
        
        for i in range(nbins):
            mask = (r_flat >= i) & (r_flat < i + 1)
            if np.sum(mask) >= 3:  # Require at least 3 points per shell
                bin_means[i] = np.mean(power_flat[mask])
                bin_counts[i] = np.sum(mask)
        
        # Step 4: Filter valid data using RELION-inspired criteria
        logger.debug("Filtering valid data points...")
        
        # Basic validity checks (more permissive)
        valid_basic = (
            np.isfinite(bin_means) & 
            (bin_means > 0) & 
            (bin_counts >= 3) &  # Reduced from 5 to 3
            (freq_shells > 0)
        )
        
        if np.sum(valid_basic) < 3:
            raise RuntimeError(f"Insufficient valid data points: {np.sum(valid_basic)} < 3")
        
        # Compute log power
        log_power = np.log(bin_means)
        
        # Frequency range filtering (more permissive)
        # Exclude only the DC component (zero frequency)
        min_freq = 0.001  # Very permissive - 1000 Angstrom resolution
        max_freq = 1.0 / (2 * voxel_size) * 0.9  # 90% of Nyquist frequency
        
        valid_freq = (
            valid_basic &
            (freq_shells >= min_freq) & 
            (freq_shells <= max_freq)
        )
        
        if np.sum(valid_freq) < 3:
            logger.warning("Frequency filtering too restrictive, using basic filtering")
            valid_freq = valid_basic
        
        # Simple outlier removal (more permissive)
        if np.sum(valid_freq) >= 5:
            valid_indices = np.where(valid_freq)[0]
            log_power_valid = log_power[valid_freq]
            
            # Remove only extreme outliers (beyond 3 standard deviations)
            mean_log = np.mean(log_power_valid)
            std_log = np.std(log_power_valid)
            outlier_mask = np.abs(log_power_valid - mean_log) <= 3.0 * std_log
            
            if np.sum(outlier_mask) >= 3:
                valid_outlier = np.zeros_like(valid_freq)
                valid_outlier[valid_indices[outlier_mask]] = True
                valid_freq = valid_outlier
        
        # Final validation (more permissive)
        if np.sum(valid_freq) < 3:
            raise RuntimeError(f"Final filtering left insufficient data: {np.sum(valid_freq)} < 3")
        
        # Step 5: Prepare data for fitting (following RELION's approach exactly)
        # RELION uses: x = 1/(res * res) where res is resolution in Angstrom
        # and plots log(amplitude) vs 1/resolution²
        resolutions = 1.0 / freq_shells[valid_freq]  # Convert frequency to resolution in Angstrom
        x = 1.0 / (resolutions ** 2)  # This is 1/resolution² (RELION's onepoint.x)
        y = log_power[valid_freq]  # This is log(amplitude) (RELION's onepoint.y)
        logger.info("B-factor estimation: using %s shells from %s total", np.sum(valid_freq), len(freq_shells))
        logger.info("Frequency range: %.4f to %.4f 1/Å", freq_shells[valid_freq].min(), freq_shells[valid_freq].max())
        logger.info("Resolution range: %.1f to %.1f Å", 1/freq_shells[valid_freq].max(), 1/freq_shells[valid_freq].min())
        logger.info("Log power range: %.3f to %.3f", y.min(), y.max())
        logger.info("1/Resolution² range: %.6f to %.6f 1/Å²", x.min(), x.max())
        
        # Step 6: Simple and robust linear fitting
        bfactor = 0.0
        log_p0 = np.max(y)
        fit_method = "none"
        r_squared = 0.0
        
        try:
            # Simple linear regression using numpy
            n = len(x)
            if n < 3:
                raise RuntimeError("Insufficient data points for fitting")
            
            # Use numpy's polyfit for simplicity and robustness
            coeffs = np.polyfit(x, y, 1)
            slope = coeffs[0]
            intercept = coeffs[1]
            
            # Calculate B-factor using RELION's convention exactly
            # RELION uses: global_bfactor = 4. * global_slope
            # The relationship is: log(P) = log(P0) - B * (1/resolution²) / 4
            # So slope = -B/4, therefore B = -4 * slope
            bfactor = -4 * slope
            
            # Calculate R-squared
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Debug: Show the actual fitting data
            logger.info("Fitting data points: %s", len(x))
            logger.info("X range: %.6f to %.6f 1/Å²", x.min(), x.max())
            logger.info("Y range: %.3f to %.3f", y.min(), y.max())
            logger.info("First 5 X values: %s", x[:5])
            logger.info("First 5 Y values: %s", y[:5])
            
            # Check for reasonable B-factor range (RELION typically uses -500 to 500)
            if -500 < bfactor < 500:
                fit_method = "linear_regression"
                log_p0 = intercept
                logger.info("Linear regression fit: B=%.1f Å², R²=%.3f", bfactor, r_squared)
                logger.info("Slope: %.6f, Intercept: %.3f", slope, intercept)
            else:
                logger.warning("B-factor out of reasonable range: %.1f Å²", bfactor)
                fit_method = "none"
                
        except Exception as e:
            logger.warning("Linear regression failed: %s", e)
            fit_method = "none"
        
        # Fallback to zero if fitting failed
        if fit_method == "none":
            logger.warning("Linear regression failed, using B-factor = 0")
            bfactor = 0.0
            log_p0 = np.max(y)
            fit_method = "fallback"
        
        logger.info("Final B-factor estimation: %.1f Angstrom^2 (method: %s, R²=%.3f)", bfactor, fit_method, r_squared)
        
        # Step 7: Generate diagnostic plot
        if plot_path is not None:
            try:
                _create_bfactor_plot_robust(
                    freq_shells, log_power, valid_freq, x, y, 
                    bfactor, log_p0, fit_method, r_squared, plot_path
                )
                logger.info("Diagnostic plot saved: %s", plot_path)
            except Exception as e:
                logger.warning("Failed to create diagnostic plot: %s", e)
        
        return bfactor
        
    except Exception as e:
        logger.error("B-factor estimation failed: %s", e)
        raise RuntimeError(f"B-factor estimation failed: {e}")


def _create_bfactor_plot_robust(freq_shells, log_power, valid_mask, x_fit, y_fit, 
                               bfactor, log_p0, fit_method, r_squared, plot_path):
    """Create diagnostic plot for B-factor estimation (RELION-inspired)."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: All data points
    axes[0].plot(freq_shells, log_power, 'o', markersize=2, alpha=0.3, color='gray', label='All shells')
    axes[0].plot(freq_shells[valid_mask], log_power[valid_mask], 'o', markersize=4, color='red', label='Valid shells')
    axes[0].set_xlabel('Spatial Frequency (1/Å)')
    axes[0].set_ylabel('log(Power)')
    axes[0].set_title('Power Spectrum vs Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Fitting region (RELION-style)
    if len(x_fit) > 0:
        axes[1].plot(x_fit, y_fit, 'o', markersize=4, color='red', label='Fitting data')
        if fit_method == "linear_regression":
            # Plot the fitted line using RELION's convention
            # The relationship is: log(P) = log(P0) - B * (1/resolution²) / 4
            # So y = log_p0 - (bfactor/4) * x
            y_pred = log_p0 - (bfactor / 4) * x_fit
            axes[1].plot(x_fit, y_pred, 'r-', lw=2, 
                        label=f'Fit: B={bfactor:.1f} Å²\nR²={r_squared:.3f}')
        else:
            # Even if fit failed, show the data points
            axes[1].text(0.5, 0.5, f'Fit failed\nB-factor = {bfactor:.1f} Å²\nR² = {r_squared:.3f}', 
                        transform=axes[1].transAxes, ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        axes[1].set_xlabel('1/Resolution² (1/Å²)')
        axes[1].set_ylabel('log(Power)')
        axes[1].set_title(f'B-factor Fitting ({fit_method})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        # If no fitting data, show a message
        axes[1].text(0.5, 0.5, 'No fitting data available', 
                    transform=axes[1].transAxes, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
        axes[1].set_xlabel('1/Resolution² (1/Å²)')
        axes[1].set_ylabel('log(Power)')
        axes[1].set_title('B-factor Fitting (No Data)')
        axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Resolution plot
    valid_freq = freq_shells[valid_mask]
    valid_power = log_power[valid_mask]
    resolutions = 1.0 / valid_freq
    axes[2].plot(resolutions, valid_power, 'o', markersize=4, color='red', label='Valid shells')
    axes[2].set_xlabel('Resolution (Å)')
    axes[2].set_ylabel('log(Power)')
    axes[2].set_title('Power Spectrum vs Resolution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].invert_xaxis()
    
    # Add summary text
    fig.suptitle(f'B-factor Estimation: {bfactor:.1f} Å² (R²={r_squared:.3f})', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()


def _create_bfactor_plot(freq_shells, log_power, valid_mask, x_fit, y_fit, 
                        bfactor, log_p0, fit_method, plot_path):
    """Create diagnostic plot for B-factor estimation (legacy)."""
    import matplotlib.pyplot as plt
    
    def bfactor_model(freq_sq, log_p0, bfactor):
        return log_p0 - bfactor * freq_sq / 4
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: All data points
    axes[0].plot(freq_shells, log_power, 'o', markersize=2, alpha=0.3, color='gray', label='All shells')
    axes[0].plot(freq_shells[valid_mask], log_power[valid_mask], 'o', markersize=4, color='red', label='Valid shells')
    axes[0].set_xlabel('Spatial Frequency (1/Å)')
    axes[0].set_ylabel('log(Power)')
    axes[0].set_title('Power Spectrum vs Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Fitting region
    if len(x_fit) > 0:
        axes[1].plot(x_fit, y_fit, 'o', markersize=4, color='red', label='Fitting data')
        if fit_method != "fallback":
            y_pred = bfactor_model(x_fit, log_p0, bfactor)
            axes[1].plot(x_fit, y_pred, 'r-', lw=2, label=f'Fit: B={bfactor:.1f} Å²')
        axes[1].set_xlabel('Frequency² (1/Å²)')
        axes[1].set_ylabel('log(Power)')
        axes[1].set_title(f'B-factor Fitting ({fit_method})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Resolution plot
    valid_freq = freq_shells[valid_mask]
    valid_power = log_power[valid_mask]
    resolutions = 1.0 / valid_freq
    axes[2].plot(resolutions, valid_power, 'o', markersize=4, color='red', label='Valid shells')
    axes[2].set_xlabel('Resolution (Å)')
    axes[2].set_ylabel('log(Power)')
    axes[2].set_title('Power Spectrum vs Resolution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].invert_xaxis()
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()


def estimate_bfactor_from_fsc_weighted_halfmaps(halfmap1, halfmap2, voxel_size, plot_prefix=None, fsc_mask=None, extra_fscs=None):
    """
    Estimate B-factor using FSC-weighted amplitude spectrum (RELION-style).
    For each FSC (raw, masked if available, etc.), generate a subplot showing:
      - The power/amplitude decay (log amplitude vs frequency)
      - The power decay after FSC weighting (log amplitude vs frequency)
      - The resolution limits used for the B-factor fit (vertical lines)
      - The fit region (highlighted)
      - The B-factor fit line (on the FSC-weighted plot)
    If multiple FSCs are available (e.g., unmasked, masked), plot them all for comparison.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import logging
    logger = logging.getLogger(__name__)
    # Use seaborn for style if available
    try:
        import seaborn as sns
        sns.set_context('notebook')
        sns.set_style('whitegrid')
    except ImportError:
        plt.style.use('seaborn-whitegrid')

    # Helper for FSC weighting
    def fsc_weight_curve(fsc):
        w = np.zeros_like(fsc)
        valid = fsc > 0
        w[valid] = np.sqrt(2 * fsc[valid] / (1 + fsc[valid]))
        return w

    # Prepare list of FSCs to plot: [(label, fsc, mask)]
    fscs_to_plot = []
    if extra_fscs is not None:
        for label, fsc, mask in extra_fscs:
            fscs_to_plot.append((label, fsc, mask))
    # Always include the main FSC (with the provided mask)
    fscs_to_plot.insert(0, ("FSC (main)", None, fsc_mask))

    # Store results for summary
    bfactor_results = []

    # Compute FT of both halfmaps (for all masks)
    ft_sum = 0.5 * (fourier_transform_utils.get_dft3(halfmap1) + fourier_transform_utils.get_dft3(halfmap2))
    shape = np.array(halfmap1.shape)
    center = shape // 2
    coords = np.indices(tuple(shape))
    coords = coords - center[:, None, None, None]
    r = np.sqrt(np.sum(coords ** 2, axis=0))
    r_flat = r.flatten()
    ft_sum_flat = ft_sum.flatten()
    r_max = int(np.max(r_flat))
    nbins = min(r_max + 1, shape[0] // 2)
    freq_step = 1.0 / (shape[0] * voxel_size)
    freq_shells = np.arange(nbins) * freq_step

    # Prepare figure
    n_fscs = len(fscs_to_plot)
    fig, axs = plt.subplots(n_fscs, 2, figsize=(14, 5 * n_fscs))
    if n_fscs == 1:
        axs = np.array([axs])  # Ensure 2D array
    fig.suptitle("RELION-style B-factor Diagnostics: Power Decay and FSC-weighted Fit", fontsize=18, fontweight='bold')

    for idx, (label, fsc_external, mask) in enumerate(fscs_to_plot):
        # Apply mask if provided
        if mask is not None:
            map1 = halfmap1 * mask
            map2 = halfmap2 * mask
        else:
            map1 = halfmap1
            map2 = halfmap2
        # Compute FSC if not provided
        if fsc_external is None:
            from recovar.reconstruction.regularization import get_fsc
            ft1 = fourier_transform_utils.get_dft3(map1)
            ft2 = fourier_transform_utils.get_dft3(map2)
            fsc = get_fsc(ft1, ft2, volume_shape=map1.shape)
        else:
            fsc = fsc_external
        # Amplitude spectra
        amp_unweighted = np.full(nbins, np.nan)
        amp_weighted = np.full(nbins, np.nan)
        bin_counts = np.zeros(nbins, dtype=int)
        fsc_weights = fsc_weight_curve(fsc)
        for i in range(nbins):
            mask_shell = (r_flat >= i) & (r_flat < i + 1)
            if np.sum(mask_shell) >= 3:
                amp_unweighted[i] = np.mean(np.abs(ft_sum_flat[mask_shell]))
                if i < len(fsc):
                    amp_weighted[i] = amp_unweighted[i] * fsc_weights[i]
                bin_counts[i] = np.sum(mask_shell)
        # Prepare for fit
        valid = (
            np.isfinite(amp_weighted) & (amp_weighted > 0) & (bin_counts >= 3) & (freq_shells > 0)
        )
        log_amp = np.log(amp_weighted[valid])
        resolutions = 1.0 / freq_shells[valid]
        x = 1.0 / (resolutions ** 2)
        y = log_amp
        # Fit
        fit_method = "none"
        bfactor = 0.0
        log_p0 = np.max(y)
        r_squared = 0.0
        slope = 0.0
        intercept = 0.0
        fit_range = (x.min(), x.max()) if len(x) > 0 else (0, 0)
        try:
            n = len(x)
            if n >= 3:
                coeffs = np.polyfit(x, y, 1)
                slope = coeffs[0]
                intercept = coeffs[1]
                bfactor = -4 * slope
                y_pred = slope * x + intercept
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                fit_method = "linear_regression"
                log_p0 = intercept
                fit_range = (x.min(), x.max())
        except Exception as e:
            logger.warning("Linear regression failed: %s", e)
            bfactor = 0.0
            fit_method = "fallback"
        bfactor_results.append((label, bfactor, r_squared, fit_range))
        # Plot: Power decay (log amplitude vs frequency)
        ax1 = axs[idx, 0]
        ax1.plot(freq_shells, np.log(amp_unweighted), color='tab:blue', lw=2, label='log(Amplitude, unweighted)')
        ax1.plot(freq_shells, np.log(amp_weighted), color='tab:orange', lw=2, label='log(Amplitude, FSC-weighted)')
        ax1.set_xlabel('Spatial Frequency (1/Å)', fontsize=12)
        ax1.set_ylabel('log(Amplitude)', fontsize=12)
        ax1.set_title(f'{label}: Power Decay', fontsize=14)
        ax1.grid(True, which='both', ls='--', alpha=0.5)
        ax1.legend()
        # Mark fit region
        ax1.axvline(1 / np.sqrt(fit_range[0]), color='gray', ls='--', lw=1, label='Fit region')
        ax1.axvline(1 / np.sqrt(fit_range[1]), color='gray', ls='--', lw=1)
        # Plot: B-factor fit (log amplitude vs 1/res^2)
        ax2 = axs[idx, 1]
        ax2.plot(x, y, 'o', color='tab:orange', label='Data (FSC-weighted)')
        if fit_method == "linear_regression":
            ax2.plot(x, slope * x + intercept, '-', color='tab:red', lw=2, label=f'Fit (B={bfactor:.1f} Å², R²={r_squared:.3f})')
            annotation = f"B-factor = {bfactor:.1f} Å²\nR² = {r_squared:.3f}\nFit region: {fit_range[0]:.3f}-{fit_range[1]:.3f}"
            ax2.annotate(annotation, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12,
                         ha='left', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))
        ax2.set_xlabel('1/Resolution² (1/Å²)', fontsize=12)
        ax2.set_ylabel('log(Amplitude)', fontsize=12)
        ax2.set_title(f'{label}: B-factor Fit', fontsize=14)
        ax2.grid(True, which='both', ls='--', alpha=0.5)
        ax2.legend()
        for ax in (ax1, ax2):
            ax.tick_params(axis='both', which='major', labelsize=11)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    if plot_prefix is not None:
        fig.savefig(f"{plot_prefix}_bfactor_fsc_panels.png", dpi=150)
        fig.savefig(f"{plot_prefix}_bfactor_fsc_panels.pdf")
        plt.close(fig)
    # Return the main B-factor (first one)
    return bfactor_results[0][1] if bfactor_results else 0.0


def local_filter_halfmaps(halfmap1_path, halfmap2_path, voxel_size, output_path, 
                          B_factor=None, mask_radius=None, fsc_mask_path=None, estimate_B_factor=False, apply_mask_path=None,
                          locres_sampling=25.0, locres_maskrad=None, locres_edgwidth=None, locres_minres=50.0, 
                          fsc_threshold=1/7, filter_edgewidth=2):
    """Apply local FSC filtering to a pair of halfmaps."""
    
    # Load halfmaps
    logger.info("Loading halfmaps:")
    logger.info("  Halfmap1: %s", halfmap1_path)
    logger.info("  Halfmap2: %s", halfmap2_path)
    
    halfmap1 = utils.load_mrc(halfmap1_path)
    halfmap2 = utils.load_mrc(halfmap2_path)
    
    # Load FSC mask if provided
    fsc_mask = None
    if fsc_mask_path is not None:
        fsc_mask = utils.load_mrc(fsc_mask_path)
        logger.info("Loaded FSC mask: %s", fsc_mask_path)
    
    # Load apply mask if provided
    apply_mask = None
    if apply_mask_path is not None:
        apply_mask = utils.load_mrc(apply_mask_path)
        logger.info("Loaded apply mask: %s", apply_mask_path)
    
    # If voxel_size is None, get from MRC file
    if voxel_size is None:
        voxel_size = get_voxel_size_from_mrc(halfmap1_path)
        logger.info("Read voxel size %s from %s", voxel_size, halfmap1_path)
    
    # Estimate B-factor if requested
    estimated_bfactor = None
    bfactor_plot_prefix = None
    if estimate_B_factor:
        if output_path.endswith('.mrc'):
            bfactor_plot_prefix = output_path.replace('.mrc', '')
        else:
            bfactor_plot_prefix = output_path
        estimated_bfactor = estimate_bfactor_from_fsc_weighted_halfmaps(
            halfmap1, halfmap2, voxel_size, plot_prefix=bfactor_plot_prefix, fsc_mask=fsc_mask)
        logger.info("Estimated B-factor (RELION-style, FSC-weighted): %.2f Angstrom^2", estimated_bfactor)
        logger.info("Saved B-factor diagnostic plots with prefix: %s", bfactor_plot_prefix)
        if B_factor is None:
            B_factor = estimated_bfactor
        else:
            logger.warning("Using user-supplied B-factor %s, ignoring estimated value %.2f", B_factor, estimated_bfactor)
    
    # Apply local filtering
    logger.info("Applying local FSC filtering...")
    logger.info("Local resolution parameters:")
    logger.info("  Sampling: %s Angstroms", locres_sampling)
    logger.info("  Mask radius: %s Angstroms", locres_maskrad)
    logger.info("  Edge width: %s Angstroms", locres_edgwidth)
    logger.info("  Min resolution: %s Angstroms", locres_minres)
    logger.info("  FSC threshold: %s", fsc_threshold)
    logger.info("  Filter edge width: %s pixels", filter_edgewidth)
    
    # Apply mask to halfmaps if provided
    if apply_mask is not None:
        halfmap1 = np.asarray(halfmap1) * np.asarray(apply_mask)
        halfmap2 = np.asarray(halfmap2) * np.asarray(apply_mask)
        logger.info("Applied mask to halfmaps for local filtering")
    
    # Apply local resolution filtering
    filtered_combined, local_resol_map, local_auc_map, fscs, local_resols = locres.local_resolution(
        halfmap1, halfmap2, B_factor if B_factor is not None else 0, voxel_size,
        locres_sampling=int(locres_sampling),
        locres_maskrad=locres_maskrad,
        locres_edgwidth=locres_edgwidth,
        locres_minres=int(locres_minres),
        use_filter=True,
        fsc_threshold=fsc_threshold,
        filter_edgewidth=filter_edgewidth
    )
    
    # Save filtered map
    utils.write_mrc(output_path, filtered_combined, voxel_size=voxel_size)
    
    # Save local resolution map
    if output_path.endswith('.mrc'):
        local_resol_path = output_path.replace('.mrc', '_local_resol.mrc')
        local_auc_path = output_path.replace('.mrc', '_local_auc.mrc')
    else:
        local_resol_path = output_path + '_local_resol.mrc'
        local_auc_path = output_path + '_local_auc.mrc'
    
    utils.write_mrc(local_resol_path, local_resol_map, voxel_size=voxel_size)
    utils.write_mrc(local_auc_path, local_auc_map, voxel_size=voxel_size)
    
    # Calculate statistics
    valid_mask = local_resol_map > 0
    if np.any(valid_mask):
        median_resol = np.median(local_resol_map[valid_mask])
        mean_resol = np.mean(local_resol_map[valid_mask])
        min_resol = np.min(local_resol_map[valid_mask])
        max_resol = np.max(local_resol_map[valid_mask])
        
        logger.info("Local resolution statistics (valid voxels only):")
        logger.info("  Median: %.2f Angstroms", median_resol)
        logger.info("  Mean: %.2f Angstroms", mean_resol)
        logger.info("  Min: %.2f Angstroms", min_resol)
        logger.info("  Max: %.2f Angstroms", max_resol)
    else:
        logger.warning("No valid local resolution values found")
        median_resol = np.nan
    
    logger.info("Filtered map saved: %s", output_path)
    logger.info("Local resolution map saved: %s", local_resol_path)
    logger.info("Local AUC map saved: %s", local_auc_path)
    
    # Save info file
    if output_path.endswith('.mrc'):
        info_path = output_path.replace('.mrc', '_info.txt')
    else:
        info_path = output_path + '_info.txt'
    with open(info_path, 'w') as f:
        f.write(f"Local Resolution Statistics:\n")
        if not np.isnan(median_resol):
            f.write(f"  Median: {median_resol:.2f} Angstroms\n")
            f.write(f"  Mean: {mean_resol:.2f} Angstroms\n")
            f.write(f"  Min: {min_resol:.2f} Angstroms\n")
            f.write(f"  Max: {max_resol:.2f} Angstroms\n")
        f.write(f"Voxel Size: {voxel_size} Angstroms\n")
        f.write(f"Input Halfmap1: {halfmap1_path}\n")
        f.write(f"Input Halfmap2: {halfmap2_path}\n")
        f.write(f"Local Resolution Parameters:\n")
        f.write(f"  Sampling: {locres_sampling} Angstroms\n")
        f.write(f"  Mask radius: {locres_maskrad} Angstroms\n")
        f.write(f"  Edge width: {locres_edgwidth} Angstroms\n")
        f.write(f"  Min resolution: {locres_minres} Angstroms\n")
        f.write(f"  FSC threshold: {fsc_threshold}\n")
        f.write(f"  Filter edge width: {filter_edgewidth} pixels\n")
        if B_factor is not None:
            f.write(f"B-factor: {B_factor} Angstroms^2\n")
        if estimated_bfactor is not None:
            f.write(f"Estimated B-factor: {estimated_bfactor:.2f} Angstroms^2\n")
        if bfactor_plot_prefix is not None:
            f.write(f"B-factor fit plot: {bfactor_plot_prefix}_bfactor_fsc_panels.png\n")
        if apply_mask_path is not None:
            f.write(f"Apply mask: {apply_mask_path}\n")
        f.write(f"Output files:\n")
        f.write(f"  Filtered map: {output_path}\n")
        f.write(f"  Local resolution map: {local_resol_path}\n")
        f.write(f"  Local AUC map: {local_auc_path}\n")
    
    logger.info("Info saved: %s", info_path)
    
    return median_resol


def postprocess_halfmaps(halfmap1_path, halfmap2_path, voxel_size, output_path, 
                          B_factor=None, mask_radius=None, fsc_mask_path=None, estimate_B_factor=False, apply_mask_path=None):
    """Apply global FSC filtering to a pair of halfmaps."""
    
    # Load halfmaps
    logger.info("Loading halfmaps:")
    logger.info("  Halfmap1: %s", halfmap1_path)
    logger.info("  Halfmap2: %s", halfmap2_path)
    
    halfmap1 = utils.load_mrc(halfmap1_path)
    halfmap2 = utils.load_mrc(halfmap2_path)
    
    # Load FSC mask if provided
    fsc_mask = None
    if fsc_mask_path is not None:
        fsc_mask = utils.load_mrc(fsc_mask_path)
        logger.info("Loaded FSC mask: %s", fsc_mask_path)
    
    # Load apply mask if provided
    apply_mask = None
    if apply_mask_path is not None:
        apply_mask = utils.load_mrc(apply_mask_path)
        logger.info("Loaded apply mask: %s", apply_mask_path)
    
    # If voxel_size is None, get from MRC file
    if voxel_size is None:
        voxel_size = get_voxel_size_from_mrc(halfmap1_path)
        logger.info("Read voxel size %s from %s", voxel_size, halfmap1_path)
    
    # Estimate B-factor if requested
    estimated_bfactor = None
    bfactor_plot_prefix = None
    if estimate_B_factor:
        if output_path.endswith('.mrc'):
            bfactor_plot_prefix = output_path.replace('.mrc', '')
        else:
            bfactor_plot_prefix = output_path
        estimated_bfactor = estimate_bfactor_from_fsc_weighted_halfmaps(
            halfmap1, halfmap2, voxel_size, plot_prefix=bfactor_plot_prefix, fsc_mask=fsc_mask)
        logger.info("Estimated B-factor (RELION-style, FSC-weighted): %.2f Angstrom^2", estimated_bfactor)
        logger.info("Saved B-factor diagnostic plots with prefix: %s", bfactor_plot_prefix)
        if B_factor is None:
            B_factor = estimated_bfactor
        else:
            logger.warning("Using user-supplied B-factor %s, ignoring estimated value %.2f", B_factor, estimated_bfactor)
    
    # Apply global filtering
    logger.info("Applying global FSC filtering...")
    filtered_combined, fsc, global_resol = locres.filter_maps_with_global_fsc(
        halfmap1, halfmap2, voxel_size,
        mask_radius=mask_radius,
        B_factor=B_factor,
        fsc_mask=fsc_mask,
        mask=apply_mask
    )
    
    # Save filtered map
    utils.write_mrc(output_path, filtered_combined, voxel_size=voxel_size)
    
    logger.info("Filtered map saved: %s", output_path)
    logger.info("Global resolution: %.2f Angstroms", global_resol)
    
    # Save info file
    if output_path.endswith('.mrc'):
        info_path = output_path.replace('.mrc', '_info.txt')
    else:
        info_path = output_path + '_info.txt'
    with open(info_path, 'w') as f:
        f.write(f"Global Resolution: {global_resol:.2f} Angstroms\n")
        f.write(f"Voxel Size: {voxel_size} Angstroms\n")
        f.write(f"Input Halfmap1: {halfmap1_path}\n")
        f.write(f"Input Halfmap2: {halfmap2_path}\n")
        if mask_radius is not None:
            f.write(f"Mask Radius: {mask_radius} Angstroms\n")
        if B_factor is not None:
            f.write(f"B-factor: {B_factor} Angstroms^2\n")
        if estimated_bfactor is not None:
            f.write(f"Estimated B-factor: {estimated_bfactor:.2f} Angstroms^2\n")
        if bfactor_plot_prefix is not None:
            f.write(f"B-factor fit plot: {bfactor_plot_prefix}_all_diagnostics.png\n")
        if apply_mask_path is not None:
            f.write(f"Apply mask: {apply_mask_path}\n")
    
    logger.info("Info saved: %s", info_path)
    
    return global_resol


def batch_process_volumes_local(volumes_dir, output_dir, voxel_size, B_factor=None, mask_radius=None, fsc_mask_path=None, estimate_B_factor=False, apply_mask_path=None,
                               locres_sampling=25.0, locres_maskrad=None, locres_edgwidth=None, locres_minres=50.0, 
                               fsc_threshold=1/7, filter_edgewidth=2):
    """Process all volumes in a directory using local filtering."""
    
    logger.info("Batch processing volumes with local filtering in: %s", volumes_dir)
    
    # Find volume directories
    vol_dirs = find_volume_directories(volumes_dir)
    if not vol_dirs:
        raise ValueError(f"No volume directories found in {volumes_dir}")
    
    logger.info("Found %s volume directories to process", len(vol_dirs))
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for vol_dir in vol_dirs:
        vol_name = os.path.basename(vol_dir)
        logger.info("Processing %s", vol_name)
        
        # Find halfmaps in this volume directory
        halfmap1_patterns = ['half1_unfil.mrc', 'halfmap1.mrc', 'half1.mrc']
        halfmap1_path = None
        
        for pattern in halfmap1_patterns:
            test_path = os.path.join(vol_dir, pattern)
            if os.path.exists(test_path):
                halfmap1_path = test_path
                break
        
        if halfmap1_path is None:
            logger.warning("No halfmap1 found in %s, skipping", vol_dir)
            continue
        
        # Find corresponding halfmap2
        halfmap2_path = find_halfmap2(halfmap1_path)
        if halfmap2_path is None:
            logger.warning("No halfmap2 found for %s, skipping", halfmap1_path)
            continue
        
        # Create output path
        output_path = os.path.join(output_dir, f"{vol_name}_local_filtered.mrc")
        
        # If voxel_size is None, get from first halfmap
        vsize = voxel_size
        if vsize is None:
            vsize = get_voxel_size_from_mrc(halfmap1_path)
            logger.info("Read voxel size %s from %s", vsize, halfmap1_path)
        
        try:
            median_resol = local_filter_halfmaps(
                halfmap1_path, halfmap2_path, vsize, output_path,
                B_factor=B_factor, mask_radius=mask_radius, fsc_mask_path=fsc_mask_path, 
                estimate_B_factor=estimate_B_factor, apply_mask_path=apply_mask_path,
                locres_sampling=locres_sampling, locres_maskrad=locres_maskrad, 
                locres_edgwidth=locres_edgwidth, locres_minres=locres_minres,
                fsc_threshold=fsc_threshold, filter_edgewidth=filter_edgewidth
            )
            
            results[vol_name] = {
                'median_resolution': median_resol,
                'filtered_path': output_path,
                'local_resol_path': output_path.replace('.mrc', '_local_resol.mrc'),
                'local_auc_path': output_path.replace('.mrc', '_local_auc.mrc'),
                'success': True
            }
            
        except Exception as e:
            logger.error("Error processing %s: %s", vol_name, str(e))
            results[vol_name] = {
                'error': str(e),
                'success': False
            }
            continue
    
    successful = len([r for r in results.values() if r['success']])
    logger.info("Batch processing completed. Processed %s/%s volumes successfully", successful, len(vol_dirs))
    
    return results


def batch_process_volumes(volumes_dir, output_dir, voxel_size, B_factor=None, mask_radius=None, fsc_mask_path=None, estimate_B_factor=False, apply_mask_path=None):
    """Process all volumes in a directory."""
    
    logger.info("Batch processing volumes in: %s", volumes_dir)
    
    # Find volume directories
    vol_dirs = find_volume_directories(volumes_dir)
    if not vol_dirs:
        raise ValueError(f"No volume directories found in {volumes_dir}")
    
    logger.info("Found %s volume directories to process", len(vol_dirs))
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for vol_dir in vol_dirs:
        vol_name = os.path.basename(vol_dir)
        logger.info("Processing %s", vol_name)
        
        # Find halfmaps in this volume directory
        halfmap1_patterns = ['half1_unfil.mrc', 'halfmap1.mrc', 'half1.mrc']
        halfmap1_path = None
        
        for pattern in halfmap1_patterns:
            test_path = os.path.join(vol_dir, pattern)
            if os.path.exists(test_path):
                halfmap1_path = test_path
                break
        
        if halfmap1_path is None:
            logger.warning("No halfmap1 found in %s, skipping", vol_dir)
            continue
        
        # Find corresponding halfmap2
        halfmap2_path = find_halfmap2(halfmap1_path)
        if halfmap2_path is None:
            logger.warning("No halfmap2 found for %s, skipping", halfmap1_path)
            continue
        
        # Create output path
        output_path = os.path.join(output_dir, f"{vol_name}_filtered.mrc")
        
        # If voxel_size is None, get from first halfmap
        vsize = voxel_size
        if vsize is None:
            vsize = get_voxel_size_from_mrc(halfmap1_path)
            logger.info("Read voxel size %s from %s", vsize, halfmap1_path)
        
        try:
            global_resol = postprocess_halfmaps(
                halfmap1_path, halfmap2_path, vsize, output_path,
                B_factor=B_factor, mask_radius=mask_radius, fsc_mask_path=fsc_mask_path, estimate_B_factor=estimate_B_factor, apply_mask_path=apply_mask_path
            )
            
            results[vol_name] = {
                'global_resolution': global_resol,
                'filtered_path': output_path,
                'success': True
            }
            
        except Exception as e:
            logger.error("Error processing %s: %s", vol_name, str(e))
            results[vol_name] = {
                'error': str(e),
                'success': False
            }
            continue
    
    successful = len([r for r in results.values() if r['success']])
    logger.info("Batch processing completed. Processed %s/%s volumes successfully", successful, len(vol_dirs))
    
    return results


def main():
    """Main entry point for command line usage."""
    parser = argparse.ArgumentParser(
        description="Apply FSC-based filtering to halfmaps (global or local)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single halfmap filtering (global)
  python -m recovar.commands.postprocess half1_unfil.mrc --voxel-size 1.0 --output filtered.mrc
  
  # Single halfmap filtering (local)
  python -m recovar.commands.postprocess half1_unfil.mrc --voxel-size 1.0 --output filtered.mrc --local
  
  # With B-factor sharpening and masking (global)
  python -m recovar.commands.postprocess half1_unfil.mrc --voxel-size 1.0 --output filtered.mrc --B-factor -100 --mask-radius 50
  
  # With B-factor sharpening and masking (local)
  python -m recovar.commands.postprocess half1_unfil.mrc --voxel-size 1.0 --output filtered.mrc --local --B-factor -100 --mask-radius 50
  
  # With custom mask applied to final result
  python -m recovar.commands.postprocess half1_unfil.mrc --voxel-size 1.0 --output filtered.mrc --apply-mask mask.mrc
  
  # Local filtering with custom parameters
  python -m recovar.commands.postprocess half1_unfil.mrc --voxel-size 1.0 --output filtered.mrc --local --locres-sampling 20 --locres-minres 30
  
  # Batch process analyze output volumes directory (global)
  python -m recovar.commands.postprocess /path/to/analysis_output/kmeans_center_volumes --batch --voxel-size 1.0 --output /path/to/filtered_volumes
  
  # Batch process analyze output volumes directory (local)
  python -m recovar.commands.postprocess /path/to/analysis_output/kmeans_center_volumes --batch --local --voxel-size 1.0 --output /path/to/filtered_volumes
  
  # Specify both halfmaps explicitly
  python -m recovar.commands.postprocess half1_unfil.mrc --halfmap2 half2_unfil.mrc --voxel-size 1.0 --output filtered.mrc
        """
    )
    
    parser = add_args(parser)
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    from recovar.project.job_context import job_context
    with job_context(args, "postprocess") as ctx:
        args.output = ctx.output_dir

        # Check if batch processing
        if args.batch:
            if not os.path.isdir(args.input):
                logger.error("For batch processing, input must be a directory: %s", args.input)
                return 1

            try:
                if args.local:
                    results = batch_process_volumes_local(
                        args.input, args.output, args.voxel_size if hasattr(args, 'voxel_size') else None,
                        B_factor=args.B_factor, mask_radius=args.mask_radius, fsc_mask_path=args.fsc_mask,
                        estimate_B_factor=args.estimate_B_factor, apply_mask_path=args.apply_mask,
                        locres_sampling=args.locres_sampling, locres_maskrad=args.locres_maskrad,
                        locres_edgwidth=args.locres_edgwidth, locres_minres=args.locres_minres,
                        fsc_threshold=args.fsc_threshold, filter_edgewidth=args.filter_edgewidth
                    )
                else:
                    results = batch_process_volumes(
                        args.input, args.output, args.voxel_size if hasattr(args, 'voxel_size') else None,
                        B_factor=args.B_factor, mask_radius=args.mask_radius, fsc_mask_path=args.fsc_mask,
                        estimate_B_factor=args.estimate_B_factor, apply_mask_path=args.apply_mask
                    )

                successful = sum(1 for r in results.values() if r['success'])
                total = len(results)
                logger.info("Batch processing completed: %d/%d volumes processed successfully", successful, total)
                return 0

            except Exception as e:
                logger.error("Error in batch processing: %s", e)
                return 1

        else:
            # Single file processing

            # Find halfmap2 if not provided
            if args.halfmap2 is None:
                args.halfmap2 = find_halfmap2(args.input)
                if args.halfmap2 is None:
                    logger.error("Could not find halfmap2 for %s. Please specify --halfmap2 explicitly.", args.input)
                    return 1

            # Check inputs exist
            if not os.path.exists(args.input):
                logger.error("Halfmap1 not found: %s", args.input)
                return 1

            if not os.path.exists(args.halfmap2):
                logger.error("Halfmap2 not found: %s", args.halfmap2)
                return 1

            # Check apply mask exists if provided
            if args.apply_mask is not None and not os.path.exists(args.apply_mask):
                logger.error("Apply mask not found: %s", args.apply_mask)
                return 1

            # Create output directory if needed
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Run filtering
            try:
                if args.local:
                    median_resol = local_filter_halfmaps(
                        args.input, args.halfmap2, args.voxel_size if hasattr(args, 'voxel_size') else None, args.output,
                        B_factor=args.B_factor, mask_radius=args.mask_radius, fsc_mask_path=args.fsc_mask,
                        estimate_B_factor=args.estimate_B_factor, apply_mask_path=args.apply_mask,
                        locres_sampling=args.locres_sampling, locres_maskrad=args.locres_maskrad,
                        locres_edgwidth=args.locres_edgwidth, locres_minres=args.locres_minres,
                        fsc_threshold=args.fsc_threshold, filter_edgewidth=args.filter_edgewidth
                    )

                    if not np.isnan(median_resol):
                        logger.info("Success! Median local resolution: %.2f Angstroms", median_resol)
                    else:
                        logger.info("Success! Local filtering completed (no valid resolution values)")
                    return 0
                else:
                    global_resol = postprocess_halfmaps(
                        args.input, args.halfmap2, args.voxel_size if hasattr(args, 'voxel_size') else None, args.output,
                        B_factor=args.B_factor, mask_radius=args.mask_radius, fsc_mask_path=args.fsc_mask,
                        estimate_B_factor=args.estimate_B_factor, apply_mask_path=args.apply_mask
                    )

                    logger.info("Success! Global resolution: %.2f Angstroms", global_resol)
                    return 0

            except Exception as e:
                logger.error("Error: %s", e)
                return 1


if __name__ == "__main__":
    exit(main()) 
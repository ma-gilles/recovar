"""Visualization helpers: FSC plots, volume slices, embedding scatter."""

import logging

import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar import utils
from recovar.output import metrics
from recovar.reconstruction import regularization

logger = logging.getLogger(__name__)

def plot_noise_profile(pipeline_output, yscale='linear', ax=None):
    """Plot noise power spectrum profiles from pipeline output.

    Args:
        pipeline_output: Pipeline output object with noise variance data.
        yscale: Y-axis scale ('linear' or 'log').
        ax: Optional matplotlib Axes to draw into. If None, creates a new figure.

    Returns:
        Tuple of (fig, ax) matplotlib Figure and Axes objects.
    """
    plt.style.use('default')
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 9))
    else:
        fig = ax.figure

    # Get noise variance - use noise_var_used instead of noise_var
    yy = pipeline_output.get('noise_var_used').copy()

    # Handle different noise model types
    if yy.ndim == 1:
        # Standard radial noise model - 1D array
        if pipeline_output.get('input_args').ignore_zero_frequency:
            yy[0] = 0

        # Plot estimated noise power spectrum
        ax.plot(np.arange(yy.size), yy, 'o-', alpha=0.8, linewidth=2.5, markersize=8,
                label='Estimated noise power spectrum', color='#2E86AB', markerfacecolor='white', markeredgewidth=2)

    elif yy.ndim == 2:
        # Radial-per-tilt noise model - 2D array (n_tilts x n_frequency_shells)
        n_tilts, n_freq = yy.shape

        # Create a beautiful colormap for the different tilts
        tilt_colors = plt.cm.plasma(np.linspace(0, 1, n_tilts))

        # Plot each tilt's noise profile with transparency
        for i in range(n_tilts):
            tilt_noise = yy[i, :]
            if pipeline_output.get('input_args').ignore_zero_frequency:
                tilt_noise[0] = 0

            ax.plot(np.arange(n_freq), tilt_noise, '-', alpha=0.3, linewidth=1,
                    color=tilt_colors[i], label=f'Tilt {i+1}' if i < 5 else None)

        # Plot mean across all tilts with emphasis
        mean_noise = np.mean(yy, axis=0)
        if pipeline_output.get('input_args').ignore_zero_frequency:
            mean_noise[0] = 0

        ax.plot(np.arange(n_freq), mean_noise, 'o-', alpha=1.0, linewidth=4,
                markersize=10, color='#D62828', label='Mean across tilts',
                markerfacecolor='white', markeredgewidth=2, markeredgecolor='#D62828')

    # Plot image power spectrum with error bars if available
    image_PS = pipeline_output.get('image_PS')
    std_image_PS = pipeline_output.get('std_image_PS')

    if image_PS is not None:
        try:
            if (std_image_PS is not None and
                hasattr(std_image_PS, 'size') and
                std_image_PS.size > 0 and
                np.issubdtype(std_image_PS.dtype, np.number) and
                np.all(np.isfinite(std_image_PS))):
                ax.errorbar(x=np.arange(image_PS.size), y=image_PS,
                           yerr=2*std_image_PS, capsize=4, alpha=0.7,
                           label='Image power spectrum', color='#06FFA5', linewidth=2.5,
                           capthick=2, elinewidth=2)
            else:
                ax.plot(np.arange(image_PS.size), image_PS,
                       's-', alpha=0.8, label='Image power spectrum', color='#06FFA5',
                       linewidth=2.5, markersize=8, markerfacecolor='white', markeredgewidth=2)
        except (TypeError, AttributeError, ValueError):
            ax.plot(np.arange(image_PS.size), image_PS,
                   's-', alpha=0.8, label='Image power spectrum', color='#06FFA5',
                   linewidth=2.5, markersize=8, markerfacecolor='white', markeredgewidth=2)

    # Plot masked image power spectrum if available
    masked_image_PS = pipeline_output.get('masked_image_PS')
    std_masked_image_PS = pipeline_output.get('std_masked_image_PS')

    if masked_image_PS is not None:
        try:
            if (std_masked_image_PS is not None and
                hasattr(std_masked_image_PS, 'size') and
                std_masked_image_PS.size > 0 and
                np.issubdtype(std_masked_image_PS.dtype, np.number) and
                np.all(np.isfinite(std_masked_image_PS))):
                ax.errorbar(x=np.arange(masked_image_PS.size), y=masked_image_PS,
                           yerr=2*std_masked_image_PS, capsize=4, alpha=0.7,
                           label='Masked image power spectrum', color='#FF9F1C', linewidth=2.5,
                           capthick=2, elinewidth=2)
            else:
                ax.plot(np.arange(masked_image_PS.size), masked_image_PS,
                       '^-', alpha=0.8, label='Masked image power spectrum', color='#FF9F1C',
                       linewidth=2.5, markersize=8, markerfacecolor='white', markeredgewidth=2)
        except (TypeError, AttributeError, ValueError):
            ax.plot(np.arange(masked_image_PS.size), masked_image_PS,
                   '^-', alpha=0.8, label='Masked image power spectrum', color='#FF9F1C',
                   linewidth=2.5, markersize=8, markerfacecolor='white', markeredgewidth=2)

    # Plot noise variance from high frequency if available
    noise_var_from_hf = pipeline_output.get('noise_var_from_hf')
    if noise_var_from_hf is not None and hasattr(noise_var_from_hf, 'size') and noise_var_from_hf.size > 0:
        ax.plot(np.arange(noise_var_from_hf.size), noise_var_from_hf,
               'D-', alpha=0.8, label='Noise variance from high frequency',
               color='#7209B7', linewidth=2.5, markersize=8, markerfacecolor='white', markeredgewidth=2)

    # Plot radial noise variance outside mask if available
    radial_noise_var_outside_mask = pipeline_output.get('radial_noise_var_outside_mask')
    if radial_noise_var_outside_mask is not None and hasattr(radial_noise_var_outside_mask, 'size') and radial_noise_var_outside_mask.size > 0:
        ax.plot(np.arange(radial_noise_var_outside_mask.size), radial_noise_var_outside_mask,
               'v-', alpha=0.8, label='Radial noise variance outside mask',
               color='#8B4513', linewidth=2.5, markersize=8, markerfacecolor='white', markeredgewidth=2)

    # Add constant noise level line if available
    if yy is not None and yy.size > 0:
        if yy.ndim == 1:
            constant_noise = yy[0]
        else:
            constant_noise = np.mean(yy[:, 0])

        ax.axhline(y=constant_noise, color='#DC143C', linestyle='--', alpha=0.8, linewidth=3,
                   label=f'Constant noise level ({constant_noise:.2e})')

    # Styling
    ax.set_xlabel('Frequency Shell', fontsize=14, fontweight='bold')
    ax.set_ylabel('Power Spectrum', fontsize=14, fontweight='bold')
    ax.set_title('Noise Profile Analysis', fontsize=16, fontweight='bold', pad=20)
    ax.set_yscale(yscale)

    # Enhanced legend
    if yy.ndim == 2 and n_tilts > 5:
        handles, labels = ax.get_legend_handles_labels()
        important_labels = ['Mean across tilts', 'Image power spectrum', 'Masked image power spectrum',
                           'Noise variance from high frequency', 'Radial noise variance outside mask',
                           'Constant noise level']
        important_handles = []
        important_labels_final = []
        for handle, label in zip(handles, labels):
            if any(important in label for important in important_labels):
                important_handles.append(handle)
                important_labels_final.append(label)
        ax.legend(important_handles, important_labels_final, loc='upper right', fontsize=12,
                 frameon=True, fancybox=True, shadow=True, framealpha=0.9)
    else:
        ax.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, shadow=True, framealpha=0.9)

    # Grid and styling
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Background color
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    return fig, ax


def plot_summary_t(pipeline_output, n_eigs=3, filename=None):
    """Plot mean, mask, variance, and top principal component volumes.

    Creates a grid of volume projections and central slices: 3 rows for
    mean/mask/variance plus one row per eigenvolume.

    Args:
        pipeline_output: Pipeline output object with 'mean', 'volume_mask',
            'variance', and eigenvolume data.
        n_eigs: Number of eigenvolumes (principal components) to show.
        filename: Path to save the figure. If None, figure is not saved.
    """
    plt.rcParams.update({})
    font = {'weight' : 'bold',
            'size'   : 22}
    matplotlib.rc('font', **font)

    n_plots = 3 + n_eigs
    fig, axs = plt.subplots(n_plots, 6, figsize = ( 6*3, n_plots * 3))
    global is_first
    is_first = True

    # Infer volume shape from mean volume
    mean_vol = pipeline_output.get('mean')
    D = int(np.round(np.power(mean_vol.size, 1 / 3)))
    volume_shape = (D, D, D)

    def _load_u_real_for_summary(pipeline_output_obj, n_requested):
        n_requested = int(n_requested)
        if n_requested <= 0:
            return np.empty((0, *volume_shape), dtype=np.float32)
        if hasattr(pipeline_output_obj, "get_u_real"):
            return np.asarray(pipeline_output_obj.get_u_real(n_requested))
        return np.asarray(pipeline_output_obj.get('u_real')[:n_requested])

    def get_projection(vol_1d, axis):
        vol_3d = vol_1d.reshape(volume_shape)
        if axis == 0:
            proj = np.sum(vol_3d, axis=0)
        elif axis == 1:
            proj = np.sum(vol_3d, axis=1)
        elif axis == 2:
            proj = np.sum(vol_3d, axis=2)
        if np.iscomplexobj(proj):
            proj = np.real(proj)
        return proj

    def get_slice_real(vol_1d, axis):
        vol_3d = vol_1d.reshape(volume_shape)
        center = vol_3d.shape[axis] // 2
        if axis == 0:
            slice_data = vol_3d[center, :, :]
        elif axis == 1:
            slice_data = vol_3d[:, center, :]
        elif axis == 2:
            slice_data = vol_3d[:, :, center]
        if np.iscomplexobj(slice_data):
            slice_data = np.real(slice_data)
        return slice_data

    def plot_vol(vol, n_plot, from_ft=True, cmap='viridis', name="", symmetric=False):
        if from_ft:
            vol = fourier_transform_utils.get_idft3(vol.reshape(volume_shape)).reshape(-1)
        global is_first

        axs[n_plot,0].set_ylabel(name)

        for k in range(3):
            img = get_projection(vol, axis=k)

            if symmetric:
                vmax = np.max(np.abs(img))
                axs[n_plot,k].imshow(img, cmap=matplotlib.colormaps[cmap], vmin=-vmax, vmax=vmax)
            else:
                axs[n_plot,k].imshow(img, cmap=matplotlib.colormaps[cmap])
            axs[n_plot,k].set_xticklabels([])
            axs[n_plot,k].set_yticklabels([])
            if is_first:
                axs[n_plot,k].set_title(f"projection {k}")

        for k in range(3,6):
            img = get_slice_real(vol, axis=k-3)

            if symmetric:
                vmax = np.max(np.abs(img))
                axs[n_plot,k].imshow(img, cmap=matplotlib.colormaps[cmap], vmin=-vmax, vmax=vmax)
            else:
                axs[n_plot,k].imshow(img, cmap=matplotlib.colormaps[cmap])
            axs[n_plot,k].set_xticklabels([])
            axs[n_plot,k].set_yticklabels([])

            if is_first:
                axs[n_plot,k].set_title(f"slice {k-3}")

        is_first = False
        return

    plot_vol(pipeline_output.get('mean'), 0, from_ft=True, name='mean')
    plot_vol(pipeline_output.get('volume_mask'), 1, from_ft=False, name='mask')
    plot_vol(pipeline_output.get('variance'), 2, from_ft=False, name='variance')

    u = _load_u_real_for_summary(pipeline_output, n_eigs)
    n_eigs_eff = int(min(n_eigs, u.shape[0]))
    for k in range(n_eigs_eff):
        plot_vol(u[k], k+3, from_ft=False, cmap='seismic', name=f"PC {k}", symmetric=True)

    plt.subplots_adjust(wspace=0, hspace=0)
    if filename is not None:
        plt.savefig(filename)
        plt.close()


def plot_cov_results(u, s, max_eig=40, savefile=None):
    """Plot eigenvalue spectra and subspace angle comparison.

    Args:
        u: Dict of eigenvector arrays keyed by method name.
        s: Dict of eigenvalue arrays keyed by method name.
        max_eig: Maximum number of eigenvalues to display.
        savefile: If provided, saves eigenvalue plot to ``{savefile}s.png``
            and subspace angle plot to ``{savefile}u.png``.

    Returns:
        Dict mapping method names to their subspace angle arrays
        (empty if no ground truth key ``'gt'`` is present in *u*).
    """
    plt.rcParams.update({})
    font = {'weight' : 'bold',
            'size'   : 22}

    matplotlib.rc('font', **font)
    m = np.tile(["o", "s", "D", "*", '<','8', '>'], 5)

    m_idx = 0
    plt.figure(figsize = (6,6))

    names_to_plot = dict(zip(list(s.keys()), list(s.keys())))
    names_to_plot['real'] = 'init'
    names_to_plot['rescaled'] = 'SVD'

    for key in s.keys():
        if "+" in key:
            continue

        plt.plot(np.arange(1,s[key][:max_eig].size+1), s[key][:max_eig],  "-"+m[m_idx], label = names_to_plot[key], alpha = 0.5, ms = 15)
        m_idx = (m_idx + 1) % len(m)
        plt.yscale('log')
        plt.legend()

    if 'parallel_analysis' in s:
        plt.semilogy(np.ones_like(s[key][:max_eig]) * s['parallel_analysis'][0], "-"+m[m_idx], label = "par0", alpha = 0.5, ms = 15)

    plt.title('eigenvalues')
    plt.legend()
    if savefile is not None:
        plt.savefig(savefile + 's.png')
        plt.close()

    gt_key = "gt"
    angles ={}
    if gt_key in u:
        plt.figure(figsize = (6,6))
        m = np.repeat(["o", "s", "D", "*", 'x', '>'], 3, 0)
        m_idx = 0
        for key in u.keys():
            if key == gt_key:
                continue
            pkey = key

            max_subspace_size = np.min([15, u[key].shape[-1]])
            angles[key] = metrics.subspace_angles(u[key], u[gt_key], max_subspace_size)
            plt.plot( np.arange(1,1+len(angles[key])), angles[key], "-"+m[m_idx], label = pkey, alpha = 0.5,  ms = 15)
            m_idx = (m_idx + 1) % len(m)
        plt.ylim([-0.05,1.05])
        plt.legend()
        if savefile is not None:
            plt.savefig(savefile + 'u.png')
            plt.close()

    return angles

def plot_mean_fsc(pipeline_output, cryos):
    """Plot FSC curves for the mean reconstruction (masked and unmasked).

    Args:
        pipeline_output: Pipeline output object with 'mean_halfmaps',
            'volume_shape', 'voxel_size', and 'volume_mask'.
        cryos: Unused (kept for backward compatibility).

    Returns:
        Matplotlib Axes with the FSC curves.
    """
    halfmap1, halfmap2 = pipeline_output.get('mean_halfmaps')

    ax, score = plot_fsc_new(halfmap1, halfmap2, pipeline_output.get('volume_shape'), pipeline_output.get('voxel_size'),  curve = None, ax = None, threshold = 1/7, filename = None, name = "unmasked")

    ax, score_masked = plot_fsc_new(halfmap1, halfmap2, pipeline_output.get('volume_shape'), pipeline_output.get('voxel_size'),  curve = None, ax = ax, threshold = 1/7, filename = None, volume_mask = pipeline_output.get('volume_mask'), name = "masked")
    plt.rcParams.update({})
    font = {'weight' : 'bold',
            'size'   : 22}

    ax.set_title("mean estimation", fontsize=20)
    return ax

def plot_fsc(cryo, vol1, vol2, mask=None, threshold=1/7, ax=None, voxel_size=None, volume_shape=None, name="unmasked", fmat="", filename=None):
    """Plot FSC between two volumes using cryo dataset metadata.

    Args:
        cryo: CryoEMDataset providing voxel_size and volume_shape.
        vol1: First half-map (flattened Fourier volume).
        vol2: Second half-map (flattened Fourier volume).
        mask: Optional real-space mask to apply before FSC.
        threshold: FSC resolution threshold (default 1/7).
        ax: Optional Axes to draw into.
        voxel_size: Override voxel size from cryo.
        volume_shape: Override volume shape from cryo.
        name: Label for the curve in the legend.
        fmat: Matplotlib format string for the line.
        filename: Path to save the figure.

    Returns:
        Matplotlib Axes with the FSC curve.
    """
    voxel_size = cryo.voxel_size if voxel_size is None else voxel_size
    volume_shape = cryo.volume_shape if volume_shape is None else volume_shape

    ax, score = plot_fsc_new(vol1, vol2, volume_shape, voxel_size,  curve = None, ax = ax, threshold = threshold, filename = filename, name = name, volume_mask = mask, fmat = fmat)
    logger.info("%s FSC score: %s", name, score)
    return ax


def plot_fsc_new(image1, image2, volume_shape=None, voxel_size=1, curve=None, ax=None, threshold=1/7, filename=None, volume_mask=None, name="", fmat=""):
    """Plot Fourier Shell Correlation between two half-maps.

    Args:
        image1: First half-map (flattened Fourier or real-space volume).
        image2: Second half-map.
        volume_shape: 3-tuple giving the volume dimensions.
        voxel_size: Voxel size in Angstroms.
        curve: Pre-computed FSC curve. If None, computed from the inputs.
        ax: Optional Axes to draw into. If None, creates a new figure.
        threshold: FSC threshold for resolution estimation (default 1/7).
        filename: Path to save the figure.
        volume_mask: Optional real-space mask applied before FSC.
        name: Label prefix for the resolution annotation.
        fmat: Matplotlib format string for the line.

    Returns:
        Tuple of (ax, score) where score is the frequency at which FSC
        crosses the threshold.
    """
    volume_shape = utils.guess_vol_shape_from_vol_size(image1.size) if volume_shape is None else volume_shape

    grid_size = volume_shape[0]
    input_ax_is_none = ax is None
    if input_ax_is_none:
        plt.figure(figsize=(6, 5))
        ax = plt.gca()

    if volume_mask is not None:
        image1 = fourier_transform_utils.get_idft3(image1.reshape(volume_shape))
        image2 = fourier_transform_utils.get_idft3(image2.reshape(volume_shape))
        image1 = fourier_transform_utils.get_dft3(image1 * volume_mask)
        image2 = fourier_transform_utils.get_dft3(image2 * volume_mask)

    if curve is None:
        curve = FSC(np.array(image1).reshape(volume_shape), np.array(image2).reshape(volume_shape))

    freq = fourier_transform_utils.get_1d_frequency_grid(grid_size, voxel_size = voxel_size, scaled = True)
    freq = freq[freq >= 0 ]
    freq = freq[:grid_size//2 ]
    max_idx = min(curve.size, freq.size)
    line, = ax.plot(freq[:max_idx], curve[:max_idx],  fmat, linewidth = 2 )
    color = line.get_color()

    if threshold is not None:
        score = fsc_score(curve, grid_size, voxel_size, threshold = threshold)

        label = name + " "+ "{:.2f}".format(1 / score)+ "\\AA"
        n_dots_in_line = 20
        ax.plot(np.ones(n_dots_in_line) * score, np.linspace(0,1, n_dots_in_line), "-", color = color, label= label)
        ax.plot(freq, threshold * np.ones(freq.size), "k--")
    else:
        score = None

    ax.xaxis.grid(color='gray', linestyle='dashed')
    ax.yaxis.grid(color='gray', linestyle='dashed')

    if input_ax_is_none:

        plt.ylim([0, 1.02])
        plt.xlim([0, np.max(freq)])
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=10)
    if threshold is not None:
        ax.plot(freq, threshold * np.ones(freq.size), "k--")
        ax.legend()

    if filename is not None:
        plt.savefig(filename )
    return ax, score


def FSC(image1, image2, r_dict=None):
    """Compute Fourier Shell Correlation between two 3-D volumes.

    Args:
        image1: First volume as a 3-D numpy array.
        image2: Second volume as a 3-D numpy array.
        r_dict: Unused (kept for backward compatibility).

    Returns:
        1-D array of FSC values per frequency shell.
    """
    from recovar.reconstruction import regularization
    fsc = regularization.get_fsc_gpu(image1, image2, image1.shape, False, frequency_shift = 0 )
    return fsc



def fsc_score(fsc_curve, grid_size, voxel_size, threshold=0.5):
    """Find the frequency at which FSC crosses a threshold.

    Uses linear interpolation between the last shell above threshold
    and the first shell below.

    Args:
        fsc_curve: 1-D array of FSC values per shell.
        grid_size: Number of voxels along one side of the volume.
        voxel_size: Voxel size in Angstroms.
        threshold: FSC threshold (default 0.5).

    Returns:
        Frequency value (in 1/Angstrom) at the threshold crossing.
    """
    freq = fourier_transform_utils.get_1d_frequency_grid(2*grid_size, voxel_size = 0.5*voxel_size, scaled = True)
    freq = freq[freq >= 0 ]

    freq = freq[1:]
    fsc_curve = fsc_curve[1:]

    # First index above threshold
    above_threshold = fsc_curve >= threshold
    if np.all(above_threshold):
        return freq[-1]

    if np.all(~above_threshold):
        return freq[0]


    idx = int(np.max([np.argmin(above_threshold), 0]))
    if idx >= grid_size//2 -1:
        return freq[-1]
    if idx == 0:
        return freq[idx]
    idx = idx-1

    # Linearly interpolate
    from scipy import interpolate
    fsc_curve = np.where(np.isnan(fsc_curve), 0, fsc_curve)
    f = interpolate.interp1d( np.array([fsc_curve[idx], fsc_curve[idx+1]]), np.array([freq[idx], freq[idx+1]]) )

    return np.min([f(threshold),  voxel_size*2])


def plot_latent_space_scatter(z, axes=None, centers=None, labels=None, title="Latent Space Analysis",
                             figsize=(18, 12), save_path=None, show_plot=True):
    """Create scatter plots for latent space visualization.

    Args:
        z: Latent coordinates of shape (n_particles, n_dimensions).
        axes: List of (i, j) tuples specifying which dimensions to plot.
            If None, plots all pairwise combinations of first 4 dimensions.
        centers: Cluster centers to overlay, shape (n_clusters, n_dimensions).
        labels: Labels for cluster centers.
        title: Main title for the plot.
        figsize: Figure size as (width, height).
        save_path: Path to save the plot. If None, figure is not saved.
        show_plot: Whether to display the plot interactively.

    Returns:
        Tuple of (fig, axes_plt) matplotlib Figure and array of Axes.
    """
    if axes is None:
        n_dims = min(4, z.shape[1])
        axes = [(i, j) for i in range(n_dims) for j in range(i+1, n_dims)]

    n_plots = len(axes)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes_plt = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_plots == 1:
        axes_plt = [axes_plt]
    elif n_rows == 1:
        axes_plt = axes_plt.flatten()
    else:
        axes_plt = axes_plt.flatten()

    plot_colors = plt.cm.viridis(np.linspace(0, 1, len(axes)))

    for idx, (i, j) in enumerate(axes):
        if idx < len(axes_plt):
            ax = axes_plt[idx]

            try:
                ax.hexbin(z[:, i], z[:, j], gridsize=30, alpha=0.3, cmap='Blues', mincnt=1)
            except (ValueError, TypeError):
                pass

            ax.scatter(z[:, i], z[:, j], alpha=0.6, s=1, c=plot_colors[idx], edgecolors='none')

            if centers is not None:
                ax.scatter(centers[:, i], centers[:, j], c='red', edgecolor='black',
                          s=100, zorder=3, linewidth=1)

                if labels is not None:
                    for k, label in enumerate(labels):
                        ax.annotate(str(label), centers[k, [i, j]] + np.array([0.1, 0.1]),
                                  fontsize=12, fontweight='bold', color='white',
                                  path_effects=[pe.withStroke(linewidth=3, foreground="black")])

            ax.grid(True, alpha=0.3)
            ax.set_facecolor('white')
            ax.set_xlabel(f'PC{i+1}', fontweight='bold')
            ax.set_ylabel(f'PC{j+1}', fontweight='bold')
            ax.set_title(f'PC{i+1} vs PC{j+1}', fontweight='bold')

    for idx in range(n_plots, len(axes_plt)):
        axes_plt[idx].set_visible(False)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig, axes_plt


def plot_eigenvalues(eigenvalues, ax=None, n_eigs=40):
    """Plot eigenvalue spectrum on a semilogy scale.

    Args:
        eigenvalues: 1-D array of eigenvalues.
        ax: Optional Axes to draw into. If None, creates a new figure.
        n_eigs: Number of eigenvalues to display.

    Returns:
        Matplotlib Axes with the eigenvalue plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(eigenvalues[:n_eigs], '-o', color='#2E86AB', markersize=6,
                markerfacecolor='white', markeredgewidth=1.5)
    ax.set_xlabel('Eigenvalue index', fontsize=12)
    ax.set_ylabel('Eigenvalue', fontsize=12)
    ax.set_title('Eigenvalue Spectrum', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    return ax


def plot_contrast_histogram(contrasts, ax=None, zdim_key=None):
    """Plot histogram of per-particle contrast values.

    Args:
        contrasts: 1-D array of contrast values.
        ax: Optional Axes to draw into. If None, creates a new figure.
        zdim_key: Latent dimension key for the title annotation.

    Returns:
        Matplotlib Axes with the histogram.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(contrasts, bins=50, color='#2E86AB', alpha=0.7, edgecolor='white')
    ax.set_xlabel('Contrast', fontsize=12)
    ax.set_ylabel('Number of particles', fontsize=12)
    title = 'Contrast Histogram'
    if zdim_key is not None:
        title += f' (zdim={zdim_key})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    return ax


def plot_pipeline_summary(po, zdim_key, output_folder):
    """Create a single consolidated summary figure for the pipeline.

    Generates a 3x3 grid showing mean volume, FSC, eigenvalues,
    variance, PC scatter plots, noise profile, and contrast histogram.

    Args:
        po: Pipeline output object.
        zdim_key: Latent dimension key for accessing embeddings/contrasts.
        output_folder: Directory to save the summary PNG.
    """
    import os
    fig = plt.figure(figsize=(20, 18), constrained_layout=True)

    # Use GridSpec for flexible layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # --- Row 0 ---
    # (0,0) Mean volume projections
    ax_mean = fig.add_subplot(gs[0, 0])
    mean_vol = po.get('mean')
    D = int(np.round(np.power(mean_vol.size, 1 / 3)))
    volume_shape = (D, D, D)
    mean_real = np.real(fourier_transform_utils.get_idft3(mean_vol.reshape(volume_shape)))
    ax_mean.imshow(np.sum(mean_real, axis=0), cmap='viridis')
    ax_mean.set_title('Mean Volume', fontsize=12, fontweight='bold')
    ax_mean.set_xticks([])
    ax_mean.set_yticks([])

    # (0,1) FSC curves
    ax_fsc = fig.add_subplot(gs[0, 1])
    halfmap1, halfmap2 = po.get('mean_halfmaps')
    plot_fsc_new(halfmap1, halfmap2, po.get('volume_shape'), po.get('voxel_size'),
                 ax=ax_fsc, threshold=1/7, name="unmasked")
    plot_fsc_new(halfmap1, halfmap2, po.get('volume_shape'), po.get('voxel_size'),
                 ax=ax_fsc, threshold=1/7, volume_mask=po.get('volume_mask'), name="masked")
    ax_fsc.set_title('FSC (Mean)', fontsize=12, fontweight='bold')

    # (0,2) Eigenvalue spectrum
    ax_eig = fig.add_subplot(gs[0, 2])
    plot_eigenvalues(po.get('s'), ax=ax_eig)

    # --- Row 1 ---
    # (1,0) Variance volume projection
    ax_var = fig.add_subplot(gs[1, 0])
    variance = po.get('variance')
    if variance is not None:
        var_vol = np.real(np.array(variance)).reshape(volume_shape)
        ax_var.imshow(np.sum(var_vol, axis=0), cmap='viridis')
    ax_var.set_title('Variance', fontsize=12, fontweight='bold')
    ax_var.set_xticks([])
    ax_var.set_yticks([])

    # Load latent coordinates for PC plots
    z = _load_latent_coords(po)

    # (1,1) PC1 vs PC2
    ax_pc12 = fig.add_subplot(gs[1, 1])
    if z is not None and z.shape[1] >= 2:
        ax_pc12.hexbin(z[:, 0], z[:, 1], gridsize=30, alpha=0.5, cmap='Blues', mincnt=1)
        ax_pc12.scatter(z[:, 0], z[:, 1], alpha=0.3, s=0.5, c='cornflowerblue', edgecolors='none', rasterized=True)
        ax_pc12.set_xlabel('PC1', fontsize=10)
        ax_pc12.set_ylabel('PC2', fontsize=10)
    ax_pc12.set_title('PC1 vs PC2', fontsize=12, fontweight='bold')
    ax_pc12.grid(True, alpha=0.3)

    # (1,2) PC1 vs PC3
    ax_pc13 = fig.add_subplot(gs[1, 2])
    if z is not None and z.shape[1] >= 3:
        ax_pc13.hexbin(z[:, 0], z[:, 2], gridsize=30, alpha=0.5, cmap='Blues', mincnt=1)
        ax_pc13.scatter(z[:, 0], z[:, 2], alpha=0.3, s=0.5, c='cornflowerblue', edgecolors='none', rasterized=True)
        ax_pc13.set_xlabel('PC1', fontsize=10)
        ax_pc13.set_ylabel('PC3', fontsize=10)
    ax_pc13.set_title('PC1 vs PC3', fontsize=12, fontweight='bold')
    ax_pc13.grid(True, alpha=0.3)

    # --- Row 2 ---
    # (2,0) Noise profile
    ax_noise = fig.add_subplot(gs[2, 0])
    try:
        plot_noise_profile(po, ax=ax_noise)
        ax_noise.set_title('Noise Profile', fontsize=12, fontweight='bold')
    except (KeyError, FileNotFoundError, ValueError) as e:
        logger.debug("Could not plot noise profile in summary: %s", e)
        ax_noise.text(0.5, 0.5, 'Noise profile\nnot available', ha='center', va='center', transform=ax_noise.transAxes)

    # (2,1) PC2 vs PC3
    ax_pc23 = fig.add_subplot(gs[2, 1])
    if z is not None and z.shape[1] >= 3:
        ax_pc23.hexbin(z[:, 1], z[:, 2], gridsize=30, alpha=0.5, cmap='Blues', mincnt=1)
        ax_pc23.scatter(z[:, 1], z[:, 2], alpha=0.3, s=0.5, c='cornflowerblue', edgecolors='none', rasterized=True)
        ax_pc23.set_xlabel('PC2', fontsize=10)
        ax_pc23.set_ylabel('PC3', fontsize=10)
    ax_pc23.set_title('PC2 vs PC3', fontsize=12, fontweight='bold')
    ax_pc23.grid(True, alpha=0.3)

    # (2,2) Contrast histogram
    ax_contrast = fig.add_subplot(gs[2, 2])
    try:
        contrasts = po.get('contrasts')[zdim_key]
        plot_contrast_histogram(contrasts, ax=ax_contrast, zdim_key=zdim_key)
    except (KeyError, FileNotFoundError, ValueError) as e:
        logger.debug("Could not plot contrast histogram in summary: %s", e)
        ax_contrast.text(0.5, 0.5, 'Contrast histogram\nnot available', ha='center', va='center', transform=ax_contrast.transAxes)

    fig.suptitle('RECOVAR Pipeline Summary', fontsize=16, fontweight='bold', y=1.01)

    plt.savefig(os.path.join(output_folder, 'pipeline_summary.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _load_latent_coords(po):
    """Load 4-D latent coordinates from pipeline output, or None on failure."""
    try:
        zs_data = po.get('latent_coords')
        if zs_data is None:
            return None
        if isinstance(zs_data, dict):
            if 4 in zs_data:
                z = zs_data[4]
            elif len(zs_data) > 0:
                z = zs_data[list(zs_data.keys())[0]]
            else:
                return None
        elif isinstance(zs_data, (list, tuple)) and len(zs_data) > 4:
            z = zs_data[4]
        elif isinstance(zs_data, (list, tuple)) and len(zs_data) > 0:
            z = zs_data[0]
        else:
            return None
        if z is None or not hasattr(z, 'shape') or z.shape[0] < 10 or z.shape[1] < 2:
            return None
        if np.any(np.isnan(z)) or np.any(np.isinf(z)):
            z = z[~np.any(np.isnan(z) | np.isinf(z), axis=1)]
            if z.shape[0] < 10:
                return None
        return z
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Summary plot functions
# ---------------------------------------------------------------------------

from recovar.output.plot_style import (
    apply_style, safe_savefig, COLORS, CATEGORICAL,
    make_summary_figure, FONTSIZE_TITLE, FONTSIZE_SUBTITLE, FONTSIZE_LABEL,
)


def pipeline_summary(po, save_path):
    """Create a 2x3 overview figure summarizing key pipeline results.

    Panels:
        (0,0) Mean volume — central XY slice of the mean reconstruction.
        (0,1) Eigenvalue spectrum — bar chart of top eigenvalues.
        (0,2) Mean FSC — FSC curve with 1/7 threshold line.
        (1,0) Contrast histogram — distribution of per-particle contrasts.
        (1,1) Variance volume — central XY slice of the variance map.
        (1,2) Mask — central XY slice of the reconstruction mask.

    Each panel is wrapped in try/except so that a single missing quantity
    does not prevent the rest of the summary from being generated.

    Parameters
    ----------
    po : PipelineOutput
        Pipeline output object.
    save_path : str
        Path where the PNG is written.
    """
    apply_style()
    fig, axes = make_summary_figure(2, 3, title="RECOVAR Pipeline Summary",
                                    figsize=(16, 10))

    # -- Helper to infer volume shape from the mean volume --
    def _get_volume_shape():
        mean_vol = po.get('mean')
        if mean_vol is None:
            return None
        D = int(np.round(np.power(mean_vol.size, 1 / 3)))
        return (D, D, D)

    volume_shape = _get_volume_shape()

    # -- (0,0) Mean volume central slice --
    ax = axes[0, 0]
    try:
        mean_vol = po.get('mean')
        if mean_vol is None:
            raise ValueError("mean volume not available")
        mean_real = np.real(
            fourier_transform_utils.get_idft3(mean_vol.reshape(volume_shape))
        )
        mid = mean_real.shape[0] // 2
        ax.imshow(mean_real[mid], cmap='gray', origin='lower')
        ax.set_title("Mean Volume (XY slice)", fontsize=FONTSIZE_SUBTITLE)
        ax.axis('off')
    except Exception as exc:
        logger.debug("pipeline_summary: mean volume panel failed: %s", exc)
        ax.text(0.5, 0.5, "Not available", ha='center', va='center',
                transform=ax.transAxes, fontsize=FONTSIZE_LABEL)
        ax.axis('off')

    # -- (0,1) Eigenvalue spectrum --
    ax = axes[0, 1]
    try:
        eigenvalues = po.get('s')
        if eigenvalues is None:
            # Fall back to params dict
            eigenvalues = po.params.get('s', None)
        if eigenvalues is None:
            raise ValueError("eigenvalues not available")
        eigenvalues = np.asarray(eigenvalues)
        n_eigs = min(len(eigenvalues), 40)
        ax.bar(np.arange(n_eigs), eigenvalues[:n_eigs],
               color=COLORS['primary'], alpha=0.85)
        ax.set_xlabel("Index", fontsize=FONTSIZE_LABEL)
        ax.set_ylabel("Eigenvalue", fontsize=FONTSIZE_LABEL)
        ax.set_title("Eigenvalue Spectrum", fontsize=FONTSIZE_SUBTITLE)
        ax.set_yscale('log')
    except Exception as exc:
        logger.debug("pipeline_summary: eigenvalue panel failed: %s", exc)
        ax.text(0.5, 0.5, "Not available", ha='center', va='center',
                transform=ax.transAxes, fontsize=FONTSIZE_LABEL)

    # -- (0,2) Mean FSC --
    ax = axes[0, 2]
    try:
        halfmap1, halfmap2 = po.get('mean_halfmaps')
        voxel_size = po.get('voxel_size')
        vol_shape = po.get('volume_shape')
        grid_size = vol_shape[0]
        # Compute FSC curve
        fsc_curve = FSC(
            np.array(halfmap1).reshape(vol_shape),
            np.array(halfmap2).reshape(vol_shape),
        )
        freq = fourier_transform_utils.get_1d_frequency_grid(
            grid_size, voxel_size=voxel_size, scaled=True
        )
        freq = freq[freq >= 0][:grid_size // 2]
        max_idx = min(fsc_curve.size, freq.size)
        # Convert frequency to resolution in Angstroms
        with np.errstate(divide='ignore', invalid='ignore'):
            resolution = np.where(freq[:max_idx] > 0, 1.0 / freq[:max_idx], 0)
        ax.plot(resolution, fsc_curve[:max_idx],
                color=COLORS['primary'], linewidth=2, label='FSC')
        ax.axhline(y=1 / 7, color=COLORS['secondary'], linestyle='--',
                   linewidth=1.5, label='1/7 threshold')
        ax.set_xlabel("Resolution (A)", fontsize=FONTSIZE_LABEL)
        ax.set_ylabel("FSC", fontsize=FONTSIZE_LABEL)
        ax.set_title("Mean FSC", fontsize=FONTSIZE_SUBTITLE)
        ax.set_ylim([0, 1.05])
        ax.set_xlim(left=0)
        ax.invert_xaxis()
        ax.legend(fontsize=FONTSIZE_LABEL - 2)
    except Exception as exc:
        logger.debug("pipeline_summary: FSC panel failed: %s", exc)
        ax.text(0.5, 0.5, "Not available", ha='center', va='center',
                transform=ax.transAxes, fontsize=FONTSIZE_LABEL)

    # -- (1,0) Contrast histogram --
    ax = axes[1, 0]
    try:
        contrasts_dict = po.get('contrasts')
        if contrasts_dict is None:
            raise ValueError("contrasts not available")
        # Pick the first available zdim
        if isinstance(contrasts_dict, dict):
            zdim_key = next(iter(contrasts_dict))
            contrasts = contrasts_dict[zdim_key]
        else:
            contrasts = np.asarray(contrasts_dict)
        contrasts = np.asarray(contrasts).ravel()
        ax.hist(contrasts, bins=60, color=COLORS['primary'], alpha=0.8,
                edgecolor='white', linewidth=0.5)
        ax.set_xlabel("Contrast", fontsize=FONTSIZE_LABEL)
        ax.set_ylabel("Count", fontsize=FONTSIZE_LABEL)
        ax.set_title("Contrast Histogram", fontsize=FONTSIZE_SUBTITLE)
    except Exception as exc:
        logger.debug("pipeline_summary: contrast panel failed: %s", exc)
        ax.text(0.5, 0.5, "Not available", ha='center', va='center',
                transform=ax.transAxes, fontsize=FONTSIZE_LABEL)

    # -- (1,1) Variance volume central slice --
    ax = axes[1, 1]
    try:
        variance = po.get('variance')
        if variance is None:
            raise ValueError("variance not available")
        var_vol = np.real(np.asarray(variance)).reshape(volume_shape)
        mid = var_vol.shape[0] // 2
        ax.imshow(var_vol[mid], cmap='viridis', origin='lower')
        ax.set_title("Variance (XY slice)", fontsize=FONTSIZE_SUBTITLE)
        ax.axis('off')
    except Exception as exc:
        logger.debug("pipeline_summary: variance panel failed: %s", exc)
        ax.text(0.5, 0.5, "Not available", ha='center', va='center',
                transform=ax.transAxes, fontsize=FONTSIZE_LABEL)
        ax.axis('off')

    # -- (1,2) Mask central slice --
    ax = axes[1, 2]
    try:
        mask = po.get('volume_mask')
        if mask is None:
            raise ValueError("mask not available")
        mask_vol = np.real(np.asarray(mask)).reshape(volume_shape)
        mid = mask_vol.shape[0] // 2
        ax.imshow(mask_vol[mid], cmap='gray', origin='lower')
        ax.set_title("Mask (XY slice)", fontsize=FONTSIZE_SUBTITLE)
        ax.axis('off')
    except Exception as exc:
        logger.debug("pipeline_summary: mask panel failed: %s", exc)
        ax.text(0.5, 0.5, "Not available", ha='center', va='center',
                transform=ax.transAxes, fontsize=FONTSIZE_LABEL)
        ax.axis('off')

    fig.tight_layout()
    safe_savefig(fig, save_path)


def analyze_summary(zs, centers, labels, save_path, density=None):
    """Create a 2x2 overview figure summarizing k-means analysis results.

    Panels:
        (0,0) Latent space — hexbin density of particles with cluster centers.
        (0,1) Cluster sizes — bar chart of particles per cluster.
        (1,0) Cluster distances — heatmap of pairwise center distances.
        (1,1) Latent variance — histogram of per-particle distances to
              nearest cluster center.

    Parameters
    ----------
    zs : ndarray, shape (n_particles, n_dims)
        Latent coordinates.
    centers : ndarray, shape (n_clusters, n_dims)
        K-means cluster centers.
    labels : ndarray, shape (n_particles,)
        Per-particle cluster assignments.
    save_path : str
        Path where the PNG is written.
    density : ndarray, optional
        Per-particle density values for background coloring (unused if None).
    """
    apply_style()
    fig, axes = make_summary_figure(2, 2, title="RECOVAR Analysis Summary",
                                    figsize=(14, 10))

    zs = np.asarray(zs)
    centers = np.asarray(centers)
    labels = np.asarray(labels)
    n_clusters = centers.shape[0]

    # -- (0,0) Latent space scatter + cluster centers --
    ax = axes[0, 0]
    try:
        ax.hexbin(zs[:, 0], zs[:, 1], gridsize=50, cmap='Blues', mincnt=1,
                  alpha=0.8)
        # Overlay cluster centers
        for k in range(n_clusters):
            color = CATEGORICAL[k % len(CATEGORICAL)]
            ax.scatter(centers[k, 0], centers[k, 1], c=color,
                       edgecolor='black', s=120, zorder=5, linewidth=1.5)
            ax.annotate(str(k), (centers[k, 0], centers[k, 1]),
                        fontsize=FONTSIZE_LABEL - 1, fontweight='bold',
                        color='white', ha='center', va='center',
                        path_effects=[pe.withStroke(linewidth=3,
                                                    foreground='black')])
        ax.set_xlabel("PC 1", fontsize=FONTSIZE_LABEL)
        ax.set_ylabel("PC 2", fontsize=FONTSIZE_LABEL)
        ax.set_title("Latent Space", fontsize=FONTSIZE_SUBTITLE)
    except Exception as exc:
        logger.debug("analyze_summary: latent space panel failed: %s", exc)
        ax.text(0.5, 0.5, "Not available", ha='center', va='center',
                transform=ax.transAxes, fontsize=FONTSIZE_LABEL)

    # -- (0,1) Cluster sizes --
    ax = axes[0, 1]
    try:
        unique_labels, counts = np.unique(labels, return_counts=True)
        sort_idx = np.argsort(-counts)
        sorted_labels = unique_labels[sort_idx]
        sorted_counts = counts[sort_idx]
        bar_colors = [CATEGORICAL[int(l) % len(CATEGORICAL)]
                      for l in sorted_labels]
        ax.bar(np.arange(len(sorted_counts)), sorted_counts, color=bar_colors,
               alpha=0.85)
        ax.set_xlabel("Cluster (sorted)", fontsize=FONTSIZE_LABEL)
        ax.set_ylabel("Particles", fontsize=FONTSIZE_LABEL)
        ax.set_title("Cluster Sizes", fontsize=FONTSIZE_SUBTITLE)
        ax.set_xticks(np.arange(len(sorted_labels)))
        ax.set_xticklabels([str(l) for l in sorted_labels], rotation=45)
    except Exception as exc:
        logger.debug("analyze_summary: cluster sizes panel failed: %s", exc)
        ax.text(0.5, 0.5, "Not available", ha='center', va='center',
                transform=ax.transAxes, fontsize=FONTSIZE_LABEL)

    # -- (1,0) Cluster distances heatmap --
    ax = axes[1, 0]
    try:
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(centers, centers, metric='euclidean')
        im = ax.imshow(dist_matrix, cmap='viridis', origin='lower')
        ax.set_xlabel("Cluster", fontsize=FONTSIZE_LABEL)
        ax.set_ylabel("Cluster", fontsize=FONTSIZE_LABEL)
        ax.set_title("Pairwise Center Distances", fontsize=FONTSIZE_SUBTITLE)
        ax.set_xticks(np.arange(n_clusters))
        ax.set_yticks(np.arange(n_clusters))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    except Exception as exc:
        logger.debug("analyze_summary: distance panel failed: %s", exc)
        ax.text(0.5, 0.5, "Not available", ha='center', va='center',
                transform=ax.transAxes, fontsize=FONTSIZE_LABEL)

    # -- (1,1) Per-particle distance to nearest center --
    ax = axes[1, 1]
    try:
        from scipy.spatial.distance import cdist as _cdist
        dists_to_center = _cdist(zs, centers, metric='euclidean')
        min_dists = dists_to_center[np.arange(len(labels)), labels]
        ax.hist(min_dists, bins=80, color=COLORS['primary'], alpha=0.8,
                edgecolor='white', linewidth=0.5)
        ax.axvline(np.median(min_dists), color=COLORS['secondary'],
                   linestyle='--', linewidth=1.5,
                   label=f"Median = {np.median(min_dists):.3f}")
        ax.set_xlabel("Distance to nearest center", fontsize=FONTSIZE_LABEL)
        ax.set_ylabel("Count", fontsize=FONTSIZE_LABEL)
        ax.set_title("Latent Variance", fontsize=FONTSIZE_SUBTITLE)
        ax.legend(fontsize=FONTSIZE_LABEL - 2)
    except Exception as exc:
        logger.debug("analyze_summary: latent variance panel failed: %s", exc)
        ax.text(0.5, 0.5, "Not available", ha='center', va='center',
                transform=ax.transAxes, fontsize=FONTSIZE_LABEL)

    fig.tight_layout()
    safe_savefig(fig, save_path)


def junk_detection_summary(results_dict, save_path):
    """Create a 2x2 overview figure summarizing junk particle detection.

    Panels:
        (0,0) Cluster quality — histogram of per-cluster FSC AUC values.
        (0,1) Particle distribution — bar chart colored by good/junk status.
        (1,0) Quality summary — text panel with key detection statistics.
        (1,1) FSC curves — top-3 and bottom-3 cluster FSC curves if available.

    Parameters
    ----------
    results_dict : dict
        Must contain at minimum:
            - ``'fsc_aucs'``: per-cluster FSC AUC values (array)
            - ``'n_particles_per_cluster'``: particles in each cluster (array)
            - ``'junk_threshold'``: threshold used to classify junk (float)
            - ``'n_junk'``: number of junk particles (int)
            - ``'n_good'``: number of good particles (int)
        Optionally:
            - ``'fsc_curves'``: dict or list of per-cluster FSC curves
    save_path : str
        Path where the PNG is written.
    """
    apply_style()
    fig, axes = make_summary_figure(2, 2, title="RECOVAR Junk Particle Detection Summary",
                                    figsize=(14, 10))

    fsc_aucs = np.asarray(results_dict.get('fsc_aucs', []))
    n_particles = np.asarray(results_dict.get('n_particles_per_cluster', []))
    threshold = results_dict.get('junk_threshold', None)
    n_junk = results_dict.get('n_junk', None)
    n_good = results_dict.get('n_good', None)

    # -- (0,0) Cluster quality histogram --
    ax = axes[0, 0]
    try:
        if fsc_aucs.size == 0:
            raise ValueError("fsc_aucs empty")
        ax.hist(fsc_aucs, bins=max(10, len(fsc_aucs) // 2),
                color=COLORS['primary'], alpha=0.8, edgecolor='white',
                linewidth=0.5)
        if threshold is not None:
            ax.axvline(threshold, color=COLORS['secondary'], linestyle='--',
                       linewidth=2, label=f"Threshold = {threshold:.3f}")
            ax.legend(fontsize=FONTSIZE_LABEL - 2)
        ax.set_xlabel("FSC AUC", fontsize=FONTSIZE_LABEL)
        ax.set_ylabel("Clusters", fontsize=FONTSIZE_LABEL)
        ax.set_title("Cluster Quality", fontsize=FONTSIZE_SUBTITLE)
    except Exception as exc:
        logger.debug("junk_detection_summary: quality panel failed: %s", exc)
        ax.text(0.5, 0.5, "Not available", ha='center', va='center',
                transform=ax.transAxes, fontsize=FONTSIZE_LABEL)

    # -- (0,1) Particle distribution colored by good/junk --
    ax = axes[0, 1]
    try:
        if n_particles.size == 0:
            raise ValueError("n_particles_per_cluster empty")
        if threshold is not None and fsc_aucs.size == n_particles.size:
            is_good = fsc_aucs >= threshold
            bar_colors = [COLORS['accent'] if g else COLORS['secondary']
                          for g in is_good]
        else:
            bar_colors = COLORS['primary']
        ax.bar(np.arange(len(n_particles)), n_particles, color=bar_colors,
               alpha=0.85)
        ax.set_xlabel("Cluster", fontsize=FONTSIZE_LABEL)
        ax.set_ylabel("Particles", fontsize=FONTSIZE_LABEL)
        ax.set_title("Particle Distribution", fontsize=FONTSIZE_SUBTITLE)
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLORS['accent'], alpha=0.85, label='Good'),
            Patch(facecolor=COLORS['secondary'], alpha=0.85, label='Junk'),
        ]
        ax.legend(handles=legend_elements, fontsize=FONTSIZE_LABEL - 2)
    except Exception as exc:
        logger.debug("junk_detection_summary: distribution panel failed: %s",
                     exc)
        ax.text(0.5, 0.5, "Not available", ha='center', va='center',
                transform=ax.transAxes, fontsize=FONTSIZE_LABEL)

    # -- (1,0) Quality summary text panel --
    ax = axes[1, 0]
    try:
        ax.axis('off')
        lines = []
        if n_good is not None:
            lines.append(f"Good particles:  {n_good:,}")
        if n_junk is not None:
            lines.append(f"Junk particles:  {n_junk:,}")
        if n_good is not None and n_junk is not None and (n_good + n_junk) > 0:
            pct_junk = 100.0 * n_junk / (n_good + n_junk)
            lines.append(f"Junk fraction:   {pct_junk:.1f}%")
        if threshold is not None:
            lines.append(f"Threshold:       {threshold:.4f}")
        if fsc_aucs.size > 0:
            lines.append(f"AUC range:       [{fsc_aucs.min():.3f}, {fsc_aucs.max():.3f}]")
            lines.append(f"AUC median:      {np.median(fsc_aucs):.3f}")
        summary_text = "\n".join(lines) if lines else "No summary data"
        ax.text(0.1, 0.5, summary_text, ha='left', va='center',
                transform=ax.transAxes, fontsize=FONTSIZE_LABEL + 1,
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['light'],
                          edgecolor=COLORS['muted'], alpha=0.9))
        ax.set_title("Quality Summary", fontsize=FONTSIZE_SUBTITLE)
    except Exception as exc:
        logger.debug("junk_detection_summary: summary panel failed: %s", exc)
        ax.text(0.5, 0.5, "Not available", ha='center', va='center',
                transform=ax.transAxes, fontsize=FONTSIZE_LABEL)
        ax.axis('off')

    # -- (1,1) FSC curves for top-3 and bottom-3 clusters --
    ax = axes[1, 1]
    try:
        fsc_curves = results_dict.get('fsc_curves', None)
        if fsc_curves is None or fsc_aucs.size == 0:
            raise ValueError("fsc_curves not provided")
        # Determine top-3 and bottom-3 by AUC
        sorted_idx = np.argsort(fsc_aucs)
        n_show = min(3, len(sorted_idx))
        bottom_idx = sorted_idx[:n_show]
        top_idx = sorted_idx[-n_show:]

        # Access curves (dict or list)
        def _get_curve(idx):
            if isinstance(fsc_curves, dict):
                return np.asarray(fsc_curves[idx])
            return np.asarray(fsc_curves[idx])

        for i, idx in enumerate(top_idx):
            curve = _get_curve(idx)
            ax.plot(curve, color=COLORS['accent'], alpha=0.6 + 0.15 * i,
                    linewidth=1.5,
                    label=f"Best #{i+1} (c{idx})" if i == 0 else None)
        for i, idx in enumerate(bottom_idx):
            curve = _get_curve(idx)
            ax.plot(curve, color=COLORS['secondary'], alpha=0.6 + 0.15 * i,
                    linewidth=1.5, linestyle='--',
                    label=f"Worst #{i+1} (c{idx})" if i == 0 else None)
        ax.set_xlabel("Frequency Shell", fontsize=FONTSIZE_LABEL)
        ax.set_ylabel("FSC", fontsize=FONTSIZE_LABEL)
        ax.set_title("FSC Curves (Best / Worst)", fontsize=FONTSIZE_SUBTITLE)
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=FONTSIZE_LABEL - 2)
    except Exception as exc:
        logger.debug("junk_detection_summary: FSC curves panel failed: %s",
                     exc)
        ax.text(0.5, 0.5, "Not available", ha='center', va='center',
                transform=ax.transAxes, fontsize=FONTSIZE_LABEL)

    fig.tight_layout()
    safe_savefig(fig, save_path)

import logging
from matplotlib import colors as mcolors
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Optional dependency: dataframe_image
try:
    import dataframe_image as dfi
except ImportError:
    dfi = None  # Functions relying on dfi will check for None and skip saving images

import jax.numpy as jnp
import recovar.core.fourier_transform_utils as fourier_transform_utils
from recovar import utils
from recovar.reconstruction import regularization
from recovar.output import metrics
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

# I copy pasted this from an older project. Most of this is probably useless

names_to_show = { "diagonal": "diagonal", "wilson": "Wilson", "diagonal masked": "diagonal masked", "wilson masked": "Wilson masked" }
colors_name = { "diagonal": "cornflowerblue", "wilson": "lightsalmon", "diagonal masked": "blue", "wilson masked": "orangered"  }
use_tex = False
if use_tex:
    plt.rcParams['text.usetex'] = True

def plot_power_spectrum(volume, ax = None):
    input_ax_is_none = ax is None
    if input_ax_is_none:
        plt.figure(figsize=(6, 5))
        ax = plt.gca() 
    avg = regularization.average_over_shells(jnp.abs(volume)**2, volume.shape)
    ax.semilogy(avg)
    return avg

def plot_noise_profile(pipeline_output, yscale = 'linear'):
    # Set up beautiful styling
    plt.style.use('default')
    fig, ax = plt.subplots(figsize = (14, 9))
    
    # Get noise variance - use noise_var_used instead of noise_var
    yy = pipeline_output.get('noise_var_used')
    
    # Handle different noise model types
    if yy.ndim == 1:
        # Standard radial noise model - 1D array
        if pipeline_output.get('input_args').ignore_zero_frequency:
            yy[0] = 0 
        
        # Plot estimated noise power spectrum
        ax.plot(np.arange(yy.size), yy, 'o-', alpha=0.8, linewidth=2.5, markersize=8,
                label='Estimated noise power spectrum', color='#2E86AB', markerfacecolor='white', markeredgewidth=2)
        
    elif yy.ndim == 2:
        # Radial-per-tilt noise model - 2D array (n_tilts × n_frequency_shells)
        n_tilts, n_freq = yy.shape
        
        # Create a beautiful colormap for the different tilts
        colors = plt.cm.plasma(np.linspace(0, 1, n_tilts))
        
        # Plot each tilt's noise profile with transparency
        for i in range(n_tilts):
            tilt_noise = yy[i, :]
            if pipeline_output.get('input_args').ignore_zero_frequency:
                tilt_noise[0] = 0
            
            ax.plot(np.arange(n_freq), tilt_noise, '-', alpha=0.3, linewidth=1, 
                    color=colors[i], label=f'Tilt {i+1}' if i < 5 else None)  # Only label first 5 for clarity
        
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
            constant_noise = yy[0]  # Use first value as constant noise level
        else:
            constant_noise = np.mean(yy[:, 0])  # Mean of first frequency shell across tilts
        
        ax.axhline(y=constant_noise, color='#DC143C', linestyle='--', alpha=0.8, linewidth=3,
                   label=f'Constant noise level ({constant_noise:.2e})')
    
    # Beautiful styling
    ax.set_xlabel('Frequency Shell', fontsize=14, fontweight='bold')
    ax.set_ylabel('Power Spectrum', fontsize=14, fontweight='bold')
    ax.set_title('Noise Profile Analysis', fontsize=16, fontweight='bold', pad=20)
    ax.set_yscale(yscale)
    
    # Enhanced legend
    if yy.ndim == 2 and n_tilts > 5:
        # For many tilts, show a simplified legend
        handles, labels = ax.get_legend_handles_labels()
        # Keep only the mean and other important lines
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

def compare_two_volumes(cryo, vol1, vol2, from_ft_inp = True):
    import matplotlib
    plt.rcParams.update({
        # "text.usetex": True,
        # "font.family": "serif",
        # "font.sans-serif": "Helvetica",
    })
    font = {'weight' : 'bold',
            'size'   : 22}
    matplotlib.rc('font', **font)

    n_plots = 4
    fig, axs = plt.subplots(n_plots, 6, figsize = ( 6*3, n_plots * 3))#, 6*3))
    global is_first
    is_first = True
    
    def plot_vol(vol, n_plot, from_ft = True, cmap = 'viridis', name ="", symmetric = False):
        if not from_ft:
            vol = fourier_transform_utils.get_dft3(vol.reshape(cryo.volume_shape)).reshape(-1)
        global is_first
        
        axs[n_plot,0].set_ylabel(name)

        for k in range(3):

            img = cryo.get_proj(vol, axis =k )
            
            if symmetric:
                vmax = np.max(np.abs(img))
                axs[n_plot,k].imshow(img , cmap=matplotlib.colormaps[cmap], vmin = -vmax, vmax = vmax)
            else:
                axs[n_plot,k].imshow(img , cmap=matplotlib.colormaps[cmap])
            axs[n_plot,k].set_xticklabels([])
            axs[n_plot,k].set_yticklabels([])
            if is_first:            
                axs[n_plot,k].set_title(f"projection {k}")

                # axs[n_plot,k].set_ylabel(name)#f"projection {k}")#, fontsize = 20)

        for k in range(3,6):

            img = cryo.get_slice_real(vol, axis =k-3 )
            
            if symmetric:
                vmax = np.max(np.abs(img))
                axs[n_plot,k].imshow(img , cmap=matplotlib.colormaps[cmap], vmin = -vmax, vmax = vmax)
            else:
                axs[n_plot,k].imshow(img , cmap=matplotlib.colormaps[cmap])
            axs[n_plot,k].set_xticklabels([])
            axs[n_plot,k].set_yticklabels([])
            
            if is_first:            
                axs[n_plot,k].set_title(f"slice {k-3}")#, fontsize = 20)

        is_first = False
        return


    plot_vol(vol1, 0, from_ft = from_ft_inp, name = 'vol1')
    plot_vol(vol2, 1, from_ft = from_ft_inp,name = 'vol2')
    vol_diff = fourier_transform_utils.get_idft3((vol1-vol2).reshape(cryo.volume_shape)).reshape(-1)
    plot_vol(vol_diff,2, from_ft = not from_ft_inp,name = 'diff')
    plot_vol(np.abs(vol_diff),3, from_ft = not from_ft_inp,name = '||diff||')
    
    plt.subplots_adjust(wspace=0, hspace=0)

    return

def plot_summary_t(pipeline_output, n_eigs = 3, filename = None):
    plt.rcParams.update({
        # "text.usetex": True,
        # "font.family": "serif",
        # "font.sans-serif": "Helvetica",
    })
    font = {'weight' : 'bold',
            'size'   : 22}
    matplotlib.rc('font', **font)

    n_plots = 3 + n_eigs
    fig, axs = plt.subplots(n_plots, 6, figsize = ( 6*3, n_plots * 3))#, 6*3))
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
        """Get projection along specified axis"""
        vol_3d = vol_1d.reshape(volume_shape)
        if axis == 0:
            proj = np.sum(vol_3d, axis=0)
        elif axis == 1:
            proj = np.sum(vol_3d, axis=1)
        elif axis == 2:
            proj = np.sum(vol_3d, axis=2)
        
        # Handle complex data by taking the real part
        if np.iscomplexobj(proj):
            proj = np.real(proj)
        return proj
    
    def get_slice_real(vol_1d, axis):
        """Get central slice along specified axis"""
        vol_3d = vol_1d.reshape(volume_shape)
        center = vol_3d.shape[axis] // 2
        if axis == 0:
            slice_data = vol_3d[center, :, :]
        elif axis == 1:
            slice_data = vol_3d[:, center, :]
        elif axis == 2:
            slice_data = vol_3d[:, :, center]
        
        # Handle complex data by taking the real part
        if np.iscomplexobj(slice_data):
            slice_data = np.real(slice_data)
        return slice_data
    
    def plot_vol(vol, n_plot, from_ft = True, cmap = 'viridis', name ="", symmetric = False):
        if from_ft:
            vol = fourier_transform_utils.get_idft3(vol.reshape(volume_shape)).reshape(-1)
        global is_first
        
        axs[n_plot,0].set_ylabel(name)

        for k in range(3):
            img = get_projection(vol, axis=k)
            
            if symmetric:
                vmax = np.max(np.abs(img))
                axs[n_plot,k].imshow(img , cmap=matplotlib.colormaps[cmap], vmin = -vmax, vmax = vmax)
            else:
                axs[n_plot,k].imshow(img , cmap=matplotlib.colormaps[cmap])
            axs[n_plot,k].set_xticklabels([])
            axs[n_plot,k].set_yticklabels([])
            if is_first:            
                axs[n_plot,k].set_title(f"projection {k}")

        for k in range(3,6):
            img = get_slice_real(vol, axis=k-3)
            
            if symmetric:
                vmax = np.max(np.abs(img))
                axs[n_plot,k].imshow(img , cmap=matplotlib.colormaps[cmap], vmin = -vmax, vmax = vmax)
            else:
                axs[n_plot,k].imshow(img , cmap=matplotlib.colormaps[cmap])
            axs[n_plot,k].set_xticklabels([])
            axs[n_plot,k].set_yticklabels([])
            
            if is_first:            
                axs[n_plot,k].set_title(f"slice {k-3}")#, fontsize = 20)

        is_first = False
        return

    plot_vol(pipeline_output.get('mean'), 0, from_ft = True, name = 'mean')
    plot_vol(pipeline_output.get('volume_mask'), 1, from_ft = False,name = 'mask')
    plot_vol(pipeline_output.get('variance'), 2, from_ft = False,name = 'variance')

    u = _load_u_real_for_summary(pipeline_output, n_eigs)
    n_eigs_eff = int(min(n_eigs, u.shape[0]))
    for k in range(n_eigs_eff):
        plot_vol(u[k], k+3, from_ft = False, cmap = 'seismic' ,name = f"PC {k}", symmetric = True)

    plt.subplots_adjust(wspace=0, hspace=0)
    if filename is not None:
        plt.savefig(filename)
    # plt.savefig("summary.png")

    return

def plot_summary(results,cryos, n_eigs = 3):
    plt.rcParams.update({
        # "text.usetex": True,
        # "font.family": "serif",
        # "font.sans-serif": "Helvetica",
    })
    font = {'weight' : 'bold',
            'size'   : 22}
    matplotlib.rc('font', **font)

    n_plots = 2 + n_eigs
    fig, axs = plt.subplots(6, n_plots, figsize = (6*3, n_plots * 3))
    global is_first
    is_first = True
    
    def plot_vol(vol, n_plot, from_ft = True, cmap = 'viridis', name ="", symmetric = False):
        if not from_ft:
            vol = fourier_transform_utils.get_dft3(vol.reshape(cryos[0].volume_shape)).reshape(-1)
        global is_first
        axs[0, n_plot].set_title(name)
        for k in range(3):
            img = cryos[0].get_proj(vol, axis =k )
            
            if symmetric:
                vmax = np.max(np.abs(img))
                axs[k, n_plot].imshow(img , cmap=matplotlib.colormaps[cmap], vmin = -vmax, vmax = vmax)
            else:
                axs[k, n_plot].imshow(img , cmap=matplotlib.colormaps[cmap])
            axs[k, n_plot].set_xticklabels([])
            axs[k, n_plot].set_yticklabels([])
            if is_first:
                axs[k, n_plot].set_ylabel(f"projection {k}")#, fontsize = 20)

        for k in range(3,6):
            img = cryos[0].get_slice_real(vol, axis =k-3 )
            
            if symmetric:
                vmax = np.max(np.abs(img))
                axs[k, n_plot].imshow(img , cmap=matplotlib.colormaps[cmap], vmin = -vmax, vmax = vmax)
            else:
                axs[k, n_plot].imshow(img , cmap=matplotlib.colormaps[cmap])
            axs[k, n_plot].set_xticklabels([])
            axs[k, n_plot].set_yticklabels([])
            
            if is_first:
                axs[k, n_plot].set_ylabel(f"slice {k-3}")#, fontsize = 20)

        is_first = False
        return

    plot_vol(results['means']['combined'], 0, from_ft = True, name = 'mean')
    plot_vol(results['volume_mask'], 1, from_ft = False,name = 'mask')
    for k in range(n_eigs):
        plot_vol(results['u']['rescaled'][:,k], k+2, from_ft = True, cmap = 'seismic' ,name = f"eigen{k}", symmetric = True)

    plt.subplots_adjust(wspace=0, hspace=0)

    
    return
    
def plot_volume_sequence(volumes,cryos):
    plt.rcParams.update({
        # "text.usetex": True,
        # "font.family": "serif",
        # "font.sans-serif": "Helvetica",
    })
    font = {'weight' : 'bold',
            'size'   : 22}
    matplotlib.rc('font', **font)

    n_plots = len(volumes)
    fig, axs = plt.subplots(6, n_plots, figsize = (6*3, n_plots * 3))
    global is_first
    is_first = True
    
    def plot_vol(vol, n_plot, from_ft = True, cmap = 'viridis', name ="", symmetric = False):
        if not from_ft:
            vol = fourier_transform_utils.get_dft3(vol.reshape(cryos[0].volume_shape)).reshape(-1)
        global is_first
        axs[0, n_plot].set_title(name)
        for k in range(3):
            img = cryos[0].get_proj(vol, axis =k )
            
            if symmetric:
                vmax = np.max(np.abs(img))
                axs[k, n_plot].imshow(img , cmap=matplotlib.colormaps[cmap], vmin = -vmax, vmax = vmax)
            else:
                axs[k, n_plot].imshow(img , cmap=matplotlib.colormaps[cmap])
            axs[k, n_plot].set_xticklabels([])
            axs[k, n_plot].set_yticklabels([])
            if is_first:
                axs[k, n_plot].set_ylabel(f"projection {k}")#, fontsize = 20)

        for k in range(3,6):
            img = cryos[0].get_slice_real(vol, axis =k-3 )
            
            if symmetric:
                vmax = np.max(np.abs(img))
                axs[k, n_plot].imshow(img , cmap=matplotlib.colormaps[cmap], vmin = -vmax, vmax = vmax)
            else:
                axs[k, n_plot].imshow(img , cmap=matplotlib.colormaps[cmap])
            axs[k, n_plot].set_xticklabels([])
            axs[k, n_plot].set_yticklabels([])
            
            if is_first:
                axs[k, n_plot].set_ylabel(f"slice {k-3}")#, fontsize = 20)

        is_first = False
            
    for vol_idx,vol in enumerate(volumes):
        plot_vol(vol, vol_idx, from_ft = True, name = f"vol{vol_idx}")

    plt.subplots_adjust(wspace=0, hspace=0)
    
    return
    
    


def plot_cov_results(u,s, max_eig = 40, savefile = None):
    
    plt.rcParams.update({
        # "text.usetex": True,
        # "font.family": "serif",
        # "font.sans-serif": "Helvetica",
    })
    font = {'weight' : 'bold',
            'size'   : 22}

    matplotlib.rc('font', **font)
    m = np.tile(["o", "s", "D", "*", '<','8', '>'], 5)

    m_idx = 0 
    #s['parallel_analysis'] *= 0.8
    plt.figure(figsize = (6,6))
    
    names_to_plot = dict(zip(list(s.keys()), list(s.keys())))
    names_to_plot['real'] = 'init'            
    names_to_plot['rescaled'] = 'SVD'
                        
    for key in s.keys():
        # for key in used_keys:
        if "+" in key:
            continue

        plt.plot(np.arange(1,s[key][:max_eig].size+1), s[key][:max_eig],  "-"+m[m_idx], label = names_to_plot[key], alpha = 0.5, ms = 15)
        m_idx = (m_idx + 1) % len(m)
        plt.yscale('log')
        plt.legend()
        # plt.savefig(output_folder + 'plots/eigenvals.pdf')
        
    if 'parallel_analysis' in s:
        plt.semilogy(np.ones_like(s[key][:max_eig]) * s['parallel_analysis'][0], "-"+m[m_idx], label = "par0", alpha = 0.5, ms = 15)

    plt.title('eigenvalues')
    plt.legend()
    if savefile is not None:
        plt.savefig(savefile + 's.png')

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

    return angles

def plot_mean_fsc(pipeline_output,cryos):

    halfmap1, halfmap2 = pipeline_output.get('mean_halfmaps')

    ax, score = plot_fsc_new(halfmap1, halfmap2, pipeline_output.get('volume_shape'), pipeline_output.get('voxel_size'),  curve = None, ax = None, threshold = 1/7, filename = None, name = "unmasked")

    ax, score_masked = plot_fsc_new(halfmap1, halfmap2, pipeline_output.get('volume_shape'), pipeline_output.get('voxel_size'),  curve = None, ax = ax, threshold = 1/7, filename = None, volume_mask = pipeline_output.get('volume_mask'), name = "masked")
    plt.rcParams.update({
        # "text.usetex": True,
        # "font.family": "serif",
        # "font.sans-serif": "Helvetica",
    })
    font = {'weight' : 'bold',
            'size'   : 22}

    ax.set_title("mean estimation", fontsize=20)
    return ax
    
def plot_fsc(cryo, vol1, vol2, mask = None, threshold = 1/7, ax = None, voxel_size= None, volume_shape= None, name = "unmasked", fmat = "", filename = None):
    voxel_size = cryo.voxel_size if voxel_size is None else voxel_size
    volume_shape = cryo.volume_shape if volume_shape is None else volume_shape

    ax, score = plot_fsc_new(vol1, vol2, volume_shape, voxel_size,  curve = None, ax = ax, threshold = threshold, filename = filename, name = name, volume_mask = mask, fmat = fmat)
    logger.info("%s FSC score: %s", name, score)
    # print(fsc_score)
    return ax
    
    

def plot_mean_result(cryo, means, cov_noise):
    # Check power spectrums
    volume_shape = cryo.volume_shape
    plt.figure(figsize = (6,6))

    for mean_key in means.keys():
        if "1" in mean_key or 'prior' in mean_key:
            continue
        PS = regularization.average_over_shells( np.abs(means[mean_key])**2 , volume_shape)
        plt.semilogy( PS, '-o', label = mean_key, ms = 15, alpha = 0.3)

    noise_level = np.ones_like(PS) * cov_noise
    plt.semilogy( noise_level , '--', label = "noise", ms =15 )

    PS_prior = regularization.average_over_shells(  means["prior"] , volume_shape)
    plt.semilogy( PS_prior , '-.', label = "prior", ms = 15)
    plt.legend()

    # GSFSC
    # plot_fsc_function_paper_simple()  # Requires fsc_curves, grid_size, voxel_size arguments

    score = cryo.plot_FSC(means["corrected0"], means["corrected1"], threshold = 1/7 )
    
    # GSFSC
    if "ground_truth" in means:
        cryo.plot_FSC(means["combined"], means["ground_truth"], threshold = 1/2)
        
    # projection images
    axis = 0
    plt.figure()
    plt.imshow(cryo.get_proj(means["combined"], axis = axis))
    plt.colorbar()
    plt.title('combined')

    if "ground_truth" in means:
        plt.figure()
        plt.imshow(cryo.get_proj(means["ground_truth"], axis = axis))
        plt.title('ground truth')
        plt.colorbar()

        
        
def plot_fsc_new(image1, image2, volume_shape = None, voxel_size=1,  curve = None, ax = None, threshold = 1/7, filename = None, volume_mask = None, name = "", fmat = ""):
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

    # get_fsc_gpu
    if curve is None:
        curve = FSC(np.array(image1).reshape(volume_shape), np.array(image2).reshape(volume_shape))
    
    # Huuuh why is there a 1/2 here??
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
        # plt.legend("AA{-1}) = "+"{:.2f}".format(threshold))
        # plt.title("FSC("  + "{:.2f}".format(1 / score)  + "AA{-1}) = "+ "{:.2f}".format(threshold))
    if threshold is not None:  
        ax.plot(freq, threshold * np.ones(freq.size), "k--")
        ax.legend()

    if filename is not None:
        plt.savefig(filename )
    return ax, score


def plotly_scatter(points_to_plot, opacity = 0.1):

    import plotly
    import plotly.graph_objects as go



    fig = go.Figure(data=[go.Scatter3d(x=points_to_plot[0][:,0], y=points_to_plot[0][:,1], z=points_to_plot[0][:,2],
                                    mode='markers', opacity = opacity)])

    for k in range(1, len(points_to_plot)):
        fig.add_trace(go.Scatter3d(x=points_to_plot[k][:,0], y=points_to_plot[k][:,1], z=points_to_plot[k][:,2],
                                        mode='markers', opacity = opacity))


    fig.update_layout(scene_aspectmode='cube')


    fig.show()

def compute_and_plot_fsc(image1, image2, volume_shape = None, voxel_size =1):
    if volume_shape is not None:
        image1 = image1.reshape(volume_shape)
        image2 = image2.reshape(volume_shape)
        grid_size = volume_shape[0]
    else:
        volume_shape = image1.shape
        grid_size = volume_shape[0]

    fsc = FSC(np.array(image1), np.array(image2), r_dict = None)
    plot_fsc_function_paper_simple([fsc], grid_size, voxel_size)
    return fsc
    
    
### PLOTTING STUFF
def FSC(image1, image2, r_dict = None):
    from recovar.reconstruction import regularization
    fsc = regularization.get_fsc_gpu(image1, image2, image1.shape, False, frequency_shift = 0 )
    return fsc



def fsc_score(fsc_curve, grid_size, voxel_size, threshold = 0.5 ):
    # First index below 0.5
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


def plot_all_images_for_paper(images_to_plot, n_dirs, global_name, scale_image_name, voxel_size, save_to_file):
    for (key, image) in images_to_plot.items():
        if save_to_file:
            import mrcfile
            with mrcfile.new(global_name + key + ".mrc", overwrite=True) as mrc:
                mrc.set_data(image.astype(np.float32))

    for idx in range(n_dirs):
        projections = {}
        for (key, image) in images_to_plot.items():
            projections[key] = np.real(np.sum(image, axis = idx))
        plot_images_on_same_scale(projections, global_name + "proj" + str(idx), scale_image_name,voxel_size, save_to_file)

        slices = {}
        for (key, image) in images_to_plot.items():
            slices[key] = np.real(image.take(image.shape[0]//2, axis = idx))
        plot_images_on_same_scale(slices, global_name + "slice" + str(idx), scale_image_name, voxel_size,save_to_file)

def plot_images_on_same_scale(images_to_plot, global_name, scale_image_name,  voxel_size, save_to_file):
    min_val_all = np.inf
    max_val_all = - np.inf
    for img in images_to_plot.values():
        min_val_all = min(min_val_all, np.min(img))
        max_val_all = max(max_val_all, np.max(img))

    for (name, image) in images_to_plot.items():
        plot_function_with_scale(image, global_name + name, min_val_all, max_val_all, voxel_size, save_to_file = save_to_file, show_colorbar = False)
    if scale_image_name in images_to_plot:
        plot_function_with_scale(images_to_plot[scale_image_name],  global_name + scale_image_name, min_val_all, max_val_all, voxel_size, save_to_file = save_to_file, show_colorbar = True)
    return
    

def plot_function_with_scale(image, name, vmin, vmax, voxel_size, save_to_file = False, show_colorbar = False, show_scalebar = True):
        # Create subplot
        fig, ax = plt.subplots()
        ax.axis("off")
        pos = ax.imshow(image, vmin = vmin, vmax = vmax, cmap='gray')
        
        if show_scalebar:
            # Create scale bar
            scalebar = ScaleBar(voxel_size * 0.1, "nm", length_fraction=0.25 )
            ax.add_artist(scalebar)

        if show_colorbar:
            fig.colorbar(pos, ax = ax)
        if save_to_file:
            plt.savefig(name + ".pdf", bbox_inches='tight')

            
def plot_fsc_function_paper_simple(fsc_curves, grid_size, voxel_size):
    return plot_fsc_function_paper(fsc_curves, "", "", grid_size, voxel_size, save_to_file = False)
                        
            
def plot_fsc_function_paper(fsc_curves, global_name, names, grid_size, voxel_size, save_to_file = False):
    plot_fsc_flag = True
    if plot_fsc_flag:
        plt.figure(figsize=(9, 8))
        ax = plt.gca()
        # plt.rcParams['text.usetex'] = False
        freq = fourier_transform_utils.get_1d_frequency_grid(2*grid_size, voxel_size = 0.5*voxel_size, scaled = True)
        freq = freq[freq >= 0 ]
        freq = freq[:grid_size//2 ]

        lines = []
        line_names = []
        colors_plotted  = []
        names_plotted  = []
        scores = {}
        for name in names:
            curve = fsc_curves[name]
            max_idx = min(curve.size, freq.size)
            plt.plot(freq[:max_idx], curve[:max_idx], color = colors[colors_name[name]] , label = names_to_show[name] , linewidth = 2 )
            colors_plotted.append(colors[colors_name[name]])
            names_plotted.append(names_to_show[name])

            curve = fsc_curves[name + " masked"]
            max_idx = min(curve.size, freq.size)
            line = plt.plot(freq[:max_idx], curve[:max_idx],color =  colors[colors_name[name + " masked"]], label = names_to_show[name + " masked"], linewidth = 2 )
            colors_plotted.append(colors[colors_name[name +" masked"]])
            names_plotted.append(names_to_show[name + " masked"])

            scores[name] = fsc_score(curve, grid_size, voxel_size)

        n_dots_in_line = 20
        for name in names:
            plt.plot(np.ones(n_dots_in_line) * scores[name], np.linspace(0,1, n_dots_in_line), "k-")

        ax.xaxis.grid(color='gray', linestyle='dashed')
        ax.yaxis.grid(color='gray', linestyle='dashed')

        plt.plot(freq, 0.5 * np.ones(freq.size), "k--")
        plt.ylim([0, 1.02])
        plt.xlim([0, np.max(freq)])
        plt.yticks(fontsize=20) 
        plt.xticks(fontsize=20) 

        if save_to_file:
            global_name = global_name + "fsc.pdf"
            plt.savefig(global_name, bbox_inches='tight')

        plt.figure()
        f = lambda m,c: plt.plot([],[], color=c, ls="-")[0]
        handles = [f("-", colors_plotted[i]) for i in range(len(colors_plotted))]
        labels = names_plotted
        legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=True, prop={'size': 20})
        if save_to_file:
            export_legend(legend, filename = global_name + "legend.pdf")
        plt.show()


def export_legend(legend, filename="legend.png", expand=[-5,-5,5,5]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)



# Plotting for sample figures
def plot_samples_on_same_scale(images_to_plot, save_filepath, voxel_size, save_to_file):
    min_val_all = 0
    max_val_all = -np.inf
    for img in images_to_plot:
        min_val_all = min(min_val_all, np.min(img))
        max_val_all = max(max_val_all, np.max(img))

    idx = 0 
    for image in images_to_plot:
        plot_sample_with_scale(image, save_filepath + str(idx), voxel_size, min_val_all, max_val_all, save_to_file = save_to_file, show_colorbar = True)
        idx +=1 
    return

def plot_sample_with_scale(image, save_filepath, voxel_size, vmin , vmax , save_to_file = False, show_colorbar = True, show_scalebar = True):
        fig, ax = plt.subplots(figsize = (4,4))
        ax.axis("off")
        pos = ax.imshow(image, vmin = vmin, vmax = vmax, cmap='gray')
    
        if show_scalebar:
            scalebar = ScaleBar(voxel_size * 0.1, "nm", length_fraction=0.25 )
            ax.add_artist(scalebar)
        plt.show()

        if save_to_file:
            plt.savefig(save_filepath, bbox_inches='tight')

        if show_colorbar:
            fig, ax = plt.subplots(figsize = (4,4))
            plt.gca().set_visible(False)
            cbar = fig.colorbar(pos, orientation="horizontal")
            cbar.ax.tick_params(labelsize=14)
            plt.savefig(save_filepath + "colorbar" , bbox_inches='tight')

def plot_latent_space_scatter(z, axes=None, centers=None, labels=None, title="Latent Space Analysis", 
                             figsize=(18, 12), save_path=None, show_plot=True):
    """
    Create improved scatter plots for latent space visualization similar to the notebook style.
    
    Parameters:
    -----------
    z : np.ndarray
        Latent coordinates of shape (n_particles, n_dimensions)
    axes : list of tuples, optional
        List of (i, j) tuples specifying which dimensions to plot. 
        If None, plots all pairwise combinations of first 4 dimensions
    centers : np.ndarray, optional
        Cluster centers to plot
    labels : list, optional
        Labels for cluster centers
    title : str
        Main title for the plot
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the plot
    show_plot : bool
        Whether to display the plot
    """
    if axes is None:
        # Default to all pairwise combinations of first 4 dimensions
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
    
    # Generate colors for different plots
    colors = plt.cm.viridis(np.linspace(0, 1, len(axes)))
    
    for idx, (i, j) in enumerate(axes):
        if idx < len(axes_plt):
            ax = axes_plt[idx]
            
            # Create hexbin density plot for background
            try:
                ax.hexbin(z[:, i], z[:, j], gridsize=30, alpha=0.3, cmap='Blues', mincnt=1)
            except:
                pass
            
            # Main scatter plot
            ax.scatter(z[:, i], z[:, j], alpha=0.6, s=1, c=colors[idx], edgecolors='none')
            
            # Plot centers if provided
            if centers is not None:
                ax.scatter(centers[:, i], centers[:, j], c='red', edgecolor='black', 
                          s=100, zorder=3, linewidth=1)
                
                # Add labels if provided
                if labels is not None:
                    for k, label in enumerate(labels):
                        ax.annotate(str(label), centers[k, [i, j]] + np.array([0.1, 0.1]), 
                                  fontsize=12, fontweight='bold', color='white',
                                  path_effects=[pe.withStroke(linewidth=3, foreground="black")])
            
            # Improve styling
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('white')
            ax.set_xlabel(f'PC{i+1}', fontweight='bold')
            ax.set_ylabel(f'PC{j+1}', fontweight='bold')
            ax.set_title(f'PC{i+1} vs PC{j+1}', fontweight='bold')
    
    # Hide unused subplots
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

from matplotlib import colors as mcolors
from matplotlib_scalebar.scalebar import ScaleBar
import warnings
import matplotlib.pyplot as plt
import numpy as np
import dataframe_image as dfi
import pandas as pd
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(np)
from recovar import regularization, utils
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
names_to_show = { "diagonal": "diagonal", "wilson": "Wilson", "diagonal masked": "diagonal masked", "wilson masked": "Wilson masked" }
colors_name = { "diagonal": "cornflowerblue", "wilson": "lightsalmon", "diagonal masked": "blue", "wilson masked": "orangered"  }


def plot_cov_results(u,s, savefile = None):
    
    import matplotlib

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    })
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 22}

    matplotlib.rc('font', **font)
    m = np.tile(["o", "s", "D", "*", '<','8', '>'], 5)

    m_idx = 0 
    #s['parallel_analysis'] *= 0.8
    plt.figure(figsize = (12,12))
    
    
    for key in s.keys():
        # for key in used_keys:
        if "+" in key:
            continue

        plt.plot(np.arange(1,s[key][:16].size+1), s[key][:16],  "-"+m[m_idx], label = key, alpha = 0.5, ms = 15)
        m_idx = (m_idx + 1) % len(m)
        plt.yscale('log')
        plt.legend()
        # plt.savefig(output_folder + 'plots/eigenvals.pdf')
        
    if 'parallel_analysis' in s:
        plt.semilogy(np.ones_like(s[key][:20]) * s['parallel_analysis'][0], "-"+m[m_idx], label = "par0", alpha = 0.5, ms = 15)

    if 'rescaled' in s:
        plt.ylim([ s['rescaled'][15]/10 , 5*np.max(s['rescaled']) ])
    # else:
    #     plt.ylim([ s['SVD'][15]/10 , 5*np.max(s['SVD']) ])

    plt.legend()
    if savefile is not None:
        plt.savefig(savefile + 's.png')
    
#     plt.figure(figsize = (10,10))
#     angles = {}
#     gt_key = "gt_col"
#     if gt_key in u:        
#         key_gt = gt_key
#         m_idx = 0
#         for key in u.keys():
#             if key == key_gt:
#                 continue
#             pkey = key  
#             max_subspace_size = np.min([15, u[key].shape[-1]])
#             angles[key] = utils.subspace_angles(u[key], u[key_gt], max_subspace_size)
#             plt.plot( np.arange(1,1+len(angles[key])), angles[key], "-"+m[m_idx], label = pkey, alpha = 0.5,  ms = 15)
#             m_idx = (m_idx + 1) % len(m)
#         plt.legend()
#         if savefile is not None:
#             plt.savefig(savefile + 'u_gt_col.png')
    angles ={}
    plt.figure(figsize = (10,10))
    m = np.repeat(["o", "s", "D", "*", 'x', '>'], 3, 0)
    gt_key = "gt"
    if gt_key in u:
        key_gt = gt_key
        m_idx = 0
        for key in u.keys():
            if key == key_gt:
                continue
            pkey = key  

            max_subspace_size = np.min([15, u[key].shape[-1]])
            angles[key] = utils.subspace_angles(u[key], u[key_gt], max_subspace_size)
            plt.plot( np.arange(1,1+len(angles[key])), angles[key], "-"+m[m_idx], label = pkey, alpha = 0.5,  ms = 15)
            m_idx = (m_idx + 1) % len(m)
        plt.ylim([-0.05,1.05])
        plt.legend()
        if savefile is not None:
            plt.savefig(savefile + 'u.png')


            
    return angles


### PLOTTING
def plot_mean_result(cryo, means, mean_prior, cov_noise):
    # Check power spectrums
    volume_shape = cryo.volume_shape
    plt.figure(figsize = (8,8))

    for mean_key in means.keys():
        if "1" in mean_key or 'prior' in mean_key:
            continue
        PS = regularization.average_over_shells( np.abs(means[mean_key])**2 , volume_shape)
        plt.semilogy( PS, '-o', label = mean_key, ms = 15, alpha = 0.3)

    noise_level = np.ones_like(PS) * cov_noise
    plt.semilogy( noise_level , '--', label = "noise", ms =15 )

    PS_prior = regularization.average_over_shells(  mean_prior , volume_shape)
    plt.semilogy( PS_prior , '-.', label = "prior", ms = 15)
    plt.legend()

    # GSFSC
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


def plot_latent_space(xs, n_things=3, use_gt_label = False, outliers= None, use_real_x = False, gt_labels = None, one_fig_for_each = False, n_clusters = None):
    from sklearn.cluster import KMeans
    # import sklearn
    # from sklearn.svm import LinearSVC
    # from sklearn.pipeline import make_pipeline
    # from sklearn.preprocessing import StandardScaler
    # from sklearn.datasets import make_classification

    if gt_labels is None:
        gt_labels = image_assignment 
    n_clusters = n_clusters if n_clusters is not None else np.max([10, n_gt_molecules])
    # n_clusters = 13
    basis_size = xs.shape[-1]
    if use_real_x:
        xs_split = xs
    else:
        xs_split = jnp.concatenate([ xs.real, xs.imag], axis =-1)
    
    kmeans = KMeans(n_clusters=n_clusters).fit(xs_split)
    centers = kmeans.cluster_centers_
    
    if use_real_x:
        center_coords = centers.T
    else:
        center_real = centers[:,:basis_size]
        center_imag = centers[:,basis_size:]
        center_coords = np.array((center_real + center_imag * 1j).T)
    # center_vols = np.array(basis) @ center_coords + np.array(basis_mean[:,None])
    # import pdb; pdb.set_trace()
    start_idx = 1 if use_real_x else 0
    for offset in range(start_idx,n_things):
        idx = 0
        def pick_first_dim(x):
            return x[...,idx+0].real

        def pick_second_dim(x):
            return x[...,idx+offset].real

        if True:

            markers= [ 'o', '+', '>', 's', '.', 'x']
            bound = 0.1
            xlims = [ np.percentile(pick_first_dim(xs), bound),  np.percentile(pick_first_dim(xs), 100-bound) ]
            ylims = [ np.percentile(pick_second_dim(xs), bound),  np.percentile(pick_second_dim(xs), 100-bound) ]

            plt.figure()
            n_plots = n_clusters# n_gt_molecules if use_gt_label else n_clusters
            for k in range(n_plots):
                if one_fig_for_each:
                    plt.figure()
                    plt.title(str(k))
                # plt.figure()
                if use_gt_label:
                    k_pts = gt_labels ==k # gt_labels[k]
                else:
                    k_pts = kmeans.labels_ == k
                    
                plt.scatter(pick_first_dim(xs[k_pts]), pick_second_dim(xs[k_pts]),alpha=0.1, marker = markers[k%len(markers)] ,  label = 'cluster ' + str(k) )
                plt.xlim(xlims )
                plt.ylim(ylims )

            plt.xlabel("real(U1)")
            plt.ylabel("real(U2)")

            
            # plt.plot( pick_first_dim(gt_coords.T) , pick_second_dim(gt_coords.T), 'ro', label = "gt_coords")
            if outliers is not None:
                plt.scatter(pick_first_dim(xs[outliers]), pick_second_dim(xs[outliers]), alpha= 1, marker = 'x' ,  label = 'outliers?' )

            center_coords_t = center_coords.T
            plt.plot( pick_first_dim(center_coords_t) , pick_second_dim(center_coords_t), 'kx', label = "centers")
            plt.legend()

    return kmeans

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
    # Old verison from me:
    r_dict = ftu.compute_index_dict(image1.shape) if r_dict is None else r_dict
    top_img = image1 * np.conj(image2)
    top = ftu.compute_spherical_average(top_img, r_dict)
    
    # if np.linalg.norm(np.imag(top)) / np.linalg.norm(np.real(top)) > 1e-6:
        #warnings.message("FDC not real. Normalized error:" + str( np.linalg.norm(np.imag(top)) / np.linalg.norm(np.real(top))))
        # warnings.warn("FDC not real. Normalized error:" + str( np.linalg.norm(np.imag(top)) / np.linalg.norm(np.real(top))))
        # print("FDC not real. Normalized error:", np.linalg.norm(np.imag(top)) / np.linalg.norm(np.real(top)))
    top = np.real(top)
    bot = np.sqrt(ftu.compute_spherical_average(np.abs(image1)**2, r_dict) * ftu.compute_spherical_average(np.abs(image2)**2, r_dict) )
    # To get rid of annoying warning.
    bot_pos = np.where( np.abs(bot) > 0, bot, 1)
    bin_fsc = np.where( bot > 0 , top / bot_pos, 0)
    return bin_fsc

def fsc_score(fsc_curve, grid_size, voxel_size, threshold = 0.5 ):
    # First index below 0.5
    freq = ftu.get_1d_frequency_grid(2*grid_size, voxel_size = 0.5*voxel_size, scaled = True)
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
    f = interpolate.interp1d( np.array([fsc_curve[idx], fsc_curve[idx+1]]), np.array([freq[idx], freq[idx+1]]) )
    return np.min([f(threshold), 2 * voxel_size])


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
        plt.rcParams['text.usetex'] = False
        freq = ftu.get_1d_frequency_grid(2*grid_size, voxel_size = 0.5*voxel_size, scaled = True)
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

def make_images_for_paper(denoised_images, clean_image, noisy_image, mask, grid_size, voxel_size, paper_save_directory, experiment_name, log_SNR, save_to_file):
        
    global_name = paper_save_directory  + experiment_name + "_" + str(log_SNR)

    from fourier_transform_utils import get_inverse_fourier_transform, get_fourier_transform
    ### BELOW THIS IS PLOTTING.
    fsc_curves = {};   resolutions = {}
    fsc_curves_masked = {}; 
    names_to_plot = ["diagonal", "wilson"]
    clean_image_ft = get_fourier_transform(clean_image, voxel_size)

    for name in names_to_plot:
        denoised_images_ft = get_fourier_transform(denoised_images[name], voxel_size)
        fsc_curves[name] = FSC(denoised_images_ft, clean_image_ft)

        denoised_masked_ft = get_fourier_transform(denoised_images[name] * mask, voxel_size)
        fsc_curves_masked[name] = FSC(denoised_masked_ft,clean_image_ft)

        score = fsc_score(fsc_curves_masked[name], grid_size, voxel_size)
        resolutions[name] = 1/score


    names_in_denoised = ["diagonal", "wilson"]
    images_to_plot = {}
    for name in names_in_denoised:
        images_to_plot[name] = denoised_images[name]
    images_to_plot["clean"] = clean_image

    plot_all_images_for_paper(images_to_plot, 1,  global_name, scale_image_name = "clean", voxel_size = voxel_size, save_to_file=save_to_file)
    plot_all_images_for_paper({"noisy": noisy_image}, 1, global_name,  scale_image_name = "noisy",voxel_size = voxel_size, save_to_file=save_to_file)


    fsc_curves_to_plot = {}
    plot_names = { "diagonal": "diagonal", "wilson" : "wilson" } #, "wilson_cheat_fixf": "wilson with estimated g"}

    for name in plot_names:
        fsc_curves_to_plot[plot_names[name]] = fsc_curves[name]

    for name in plot_names:
        fsc_curves_to_plot[plot_names[name] + " masked"] = fsc_curves_masked[name]

    plot_fsc_function_paper(fsc_curves_to_plot , global_name, names_in_denoised, grid_size, voxel_size,   save_to_file)


    kk = pd.DataFrame(resolutions, ["resolution"])
    score_df = kk.T
    score_df
    df_styled = score_df.style.highlight_min(subset = ["resolution"], color = "green")
    #plt.savefig('mytable.png')
    display(df_styled)
    if save_to_file:
        dfi.export(df_styled, paper_save_directory + experiment_name + "scores_table"+ str(log_SNR) + ".png")


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

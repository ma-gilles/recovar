from importlib import reload
import recovar.config
import pandas as pd
import numpy as np
import pickle
import subprocess
import os, sys

from cryodrgn import analysis
from cryodrgn import utils
# from cryodrgn import dataset
from cryodrgn import ctf
from recovar import plot_utils
from recovar import output, dataset

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as py
from scipy.spatial.transform import Rotation as RR

# Specify the result dir
recovar_result_dir = '/scratch/gpfs/mg6942/cryodrgn_empiar/empiar10076/inputs/12_02_standard_old_mean/'


results = output.load_results_new(recovar_result_dir)
zdim = 20#list(results['zs'].keys())[-1]
print("available zdim:", list(results['zs'].keys()), "using:", zdim)
z = results['zs'][zdim]
cryos = dataset.load_dataset_from_args(results['input_args'], lazy = False)

output_folder_pub = recovar_result_dir + 'published_lab_2/'

indices =  '/scratch/gpfs/mg6942/cryodrgn_empiar/empiar10076/inputs/published_labels.pkl'
from recovar import utils
gt_indices=utils.pickle_load(indices)
image_assignment = np.concatenate([ gt_indices[cryo.dataset_indices] for cryo in cryos])

for ndim in [20]:
    import matplotlib
    import umap
    from cryodrgn import analysis
    colors = ['C0','C1',
             'C2','C8','green',
             'C3','orangered','maroon','lightcoral',
              'C4','slateblue','magenta','blueviolet','plum']
    rgb_colors = [matplotlib.colors.to_hex(c) for c in colors  ]

    pub_labels = image_assignment
    # get cluster centers for minor classes
    minor_centers = []

    z = results['zs'][ndim]
    mapper = umap.UMAP(n_components = 2).fit(z)
    kk = mapper.embedding_
    
    for i in range(14):
        zsub = z[pub_labels == i]
        minor_centers.append(np.median(zsub,axis=0))
    minor_centers = np.array(minor_centers)

    
    from recovar import embedding, output
    from recovar import output as o
    trajectory_reproj = embedding.generate_conformation_from_reprojection(minor_centers, results['means']['combined'], results['u']['rescaled'][:,:ndim] )
    o.mkdir_safe(output_folder_pub)
    o.save_volumes(trajectory_reproj, output_folder_pub )
    
    from scipy.stats import chi2
    from recovar import latent_density
    likelihood_threshold = latent_density.get_log_likelihood_threshold(k = ndim, q =chi2.cdf(8**2,1))
    weights = latent_density.compute_weights_of_conformation_2(minor_centers, results['zs'][ndim], results['cov_zs'][ndim],likelihood_threshold = likelihood_threshold )
    print(np.sum(weights,axis=0))
    print(np.sum(weights))
    import json
    json.dump(weights.tolist(), open(output_folder_pub + 'weights.json', 'w'))
    # o.plot_loglikelihood_over_scatter(minor_centers, zs, cov_zs, folder, likelihood_threshold = likelihood_threshold)

    o.compute_and_save_volumes_from_z(cryos, results['means'], results['u'], minor_centers, results['zs'][ndim], results['cov_zs'][ndim], results['cov_noise'], output_folder_pub , likelihood_threshold = likelihood_threshold, compute_reproj = False, adaptive = False)

    print('done with means')
    
    o.compute_and_save_volumes_from_z(cryos, results['means'], results['u'], minor_centers[:7], results['zs'][ndim], results['cov_zs'][ndim], results['cov_noise'], output_folder_pub + 'adaptive_p1', likelihood_threshold = likelihood_threshold, compute_reproj = False, adaptive = True)

    o.compute_and_save_volumes_from_z(cryos, results['means'], results['u'], minor_centers[7:], results['zs'][ndim], results['cov_zs'][ndim], results['cov_noise'], output_folder_pub + 'adaptive_p2/', likelihood_threshold = likelihood_threshold, compute_reproj = False, adaptive = True)

    print('done with adaptive')

    # minor_centers_umap, minor_centers_i = analysis.get_nearest_point(z,minor_centers)
    # minor_centers_umap = umap[minor_centers_i]
    minor_centers_umap = mapper.transform(minor_centers)

    n_eigs = z.shape[1]

    import os
    try:
        os.mkdir(output_folder_pub)
    except:
        pass
    #alphal_val = 0.025
    alphal_val = 0.1

    f,x = analysis.plot_by_cluster(kk[:,0],kk[:,1],14,image_assignment,centers=minor_centers_umap,annotate=False, colors=colors, alpha = alphal_val,  s=3, figsize = (10,10))
    x.set_xticks([])
    x.set_yticks([])
    plt.savefig(output_folder_pub + "no_labels.pdf")#, dpi=150)


    f,x = analysis.plot_by_cluster(kk[:,0],kk[:,1],14,image_assignment,centers=minor_centers_umap,annotate=True, colors=colors, alpha = alphal_val, s=3, figsize = (10,10))
    x.set_xticks([])
    x.set_yticks([])
    
    plt.savefig(output_folder_pub + "labels.pdf")#, dpi=150)


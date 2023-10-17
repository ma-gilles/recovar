import jax
import jax.numpy as jnp
import numpy as np
import pickle
from scipy.spatial.transform import Rotation 

from recovar import core, utils
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)
ftu_np = fourier_transform_utils(np)
import preprocessed_datasets

# Maybe should take out these dependencies?
from cryodrgn import mrc

class ExperimentReconstruction():
    def __init__(self, grid_size, voxel_size, padding, exp_name, datadir, vol_datadir, label_file = None, valid_indices = None ):
        volume_params, image_option = preprocessed_datasets.get_synthetic_dataset_input(exp_name, vol_datadir)   
        Xs_vec, voxel_size = generate_ground_truth_volumes(image_option, volume_params, grid_size, voxel_size, padding)
        # cryo.image_stack.D, cryo.voxel_size, cryo.padding)
        
        Xs_vec = np.array(Xs_vec, dtype = 'complex128')
        volume_shape = tuple(3*[grid_size])
        
        # Make sure it's real!?!
        # Is this necessary?
        # Xs_vec = ftu_np.get_dft3(ftu_np.get_idft3(Xs_vec.reshape([-1, *volume_shape])).real)
        # Xs_vec = Xs_vec.reshape([Xs_vec.shape[0], -1])
        self.volumes = Xs_vec
                
        # Get Image assignment
        if label_file is not None:
            with open( label_file, "rb") as f:
                self.image_assignment = pickle.load(f)
        else:
            self.image_assignment = None
        self.valid_indices = valid_indices
        
        self.probs_of_state = None
        self.mean = None
        self.covariance_cols = None

    def filter_indices_(self, indices):
        print("IMPLEMENT THIS")
        
    def get_probs_of_state(self):
        # if self.probs_of_state is None:
        self.compute_probs_of_state()
        return self.probs_of_state

    def get_mean(self):
        # if self.mean is None:
        self.compute_mean_conformation()
        return self.mean
    
    def get_covariance_cols(self, picked_frequencies, contrast_var =0   ):
        # if self.covariance_cols is None:
        #     self.compute_covariance_columns(picked_frequencies)        
        #self.compute_covariance_columns(picked_frequencies, contrast_var = contrast_var )
        u,s = self.get_covariance_eigendecomposition(contrast_variance = contrast_var)
        cov = (u * s) @ (np.conj(u[picked_frequencies,:]).T)
        return cov
    
    def compute_probs_of_state(self):        
        n_gt_molecules = self.volumes.shape[0]
        probs_of_molecule = np.zeros(n_gt_molecules, dtype = np.float32).real
        if self.image_assignment is None:
            probs_of_molecule = np.ones(n_gt_molecules) / n_gt_molecules
        else:
            for k in range(n_gt_molecules):
                probs_of_molecule[k] = np.sum(self.image_assignment[self.valid_indices] == k) / self.image_assignment[self.valid_indices].size
                
        if np.abs(np.sum(probs_of_molecule)  - 1) > 1e-5:
            print("sum of probs not 1!!")
            
        self.probs_of_state = probs_of_molecule
    
    def compute_mean_conformation(self):
        self.mean = np.array(np.sum( self.volumes * self.get_probs_of_state()[:,None], axis = 0 ))
    
    def compute_covariance_columns(self, picked_frequencies, valid_idx =1, contrast_variance = 0  ):
        vols = np.sqrt(1 + contrast_variance) *  valid_idx * ( self.volumes - self.get_mean()[None,:] ) * np.sqrt(self.get_probs_of_state()[:,None])        
        if contrast_variance > 0:
            vols = np.concatenate( [ np.sqrt(contrast_variance) * self.get_mean()[None] * valid_idx, vols ])
        vols = vols.T
        return vols @ np.conj(vols[picked_frequencies,:]).T 
    
    def get_svd(self, contrast_variance = 0 , valid_idx = 1):
        vols = np.sqrt(1 + contrast_variance) *  np.array(valid_idx) * ( self.volumes - self.get_mean()[None,:] ) * np.sqrt(self.get_probs_of_state()[:,None])
        if contrast_variance > 0:
            vols = np.concatenate( [ np.sqrt(contrast_variance) * self.get_mean()[None] * valid_idx, vols ])
        u,s,v = np.linalg.svd(vols.T, full_matrices = False)
        return u,s,v

    def get_covariance_eigendecomposition(self, contrast_variance = 0, valid_idx = 1):
        u,s,v = self.get_svd(contrast_variance = contrast_variance, valid_idx = valid_idx)
        
        zero_freq = core.frequencies_to_vec_indices( jnp.array([[0,0,0]] ), utils.guess_vol_shape_from_vol_size(self.volumes[0].size))
        # u[zero_freq] == np.sum(Fu)== < Fu, 1 > 
        ip = u[zero_freq,:] / np.abs(u[zero_freq,:])
        u /= ip

        return u, s**2
    
def generate_ground_truth_volumes(image_option, volume_params, grid_size, voxel_size, padding):
    if "from_mrc" in image_option:
        # gt_volumes =  generate_volumes_from_mrcs(volume_params)
        gt_volumes, voxel_size = generate_volumes_from_cryodrgn_mrcs(volume_params, grid_size, padding)

    elif "from_pdb" in image_option:
        import generate_synthetic_molecule as gsm
        gt_volumes = gsm.generate_volumes_from_atom_groups(volume_params, voxel_size, grid_size)

    return gt_volumes, voxel_size

def generate_volumes_from_cryodrgn_mrcs(mrc_names, grid_size_i = None, padding= 0 ):
    Xs_vec = []
    first_vol = True
    # grid_size = grid_size_i
    # grid_size = grid_size - padding
    for mrc_name in mrc_names:
        data, header =  mrc.parse_mrc(mrc_name)
        data = np.transpose(data, (2,1,0))
        voxel_size = header.fields['xlen'] / header.fields['nx']
        mrc_grid_size = header.fields['nx']
        
        if grid_size_i is None:
            grid_size = mrc_grid_size
        else:
            grid_size = grid_size_i - padding

            
        if first_vol:
            first_vol = False
            first_voxel_size = voxel_size
        else:
            assert( first_voxel_size == voxel_size)
            
#         data_padded = np.zeros_like(data, shape = np.array(data.shape) + padding )
#         half_pad = padding//2
#         data_padded[half_pad:data.shape[0] + half_pad,half_pad:data.shape[1] + half_pad,half_pad:data.shape[2] + half_pad] = data

        # Zero out grid_sizes outside radius
        X = ftu.get_dft3(data)
        # Downsample
        if mrc_grid_size > grid_size:
            diff_size = mrc_grid_size - grid_size 
            half_diff_size = diff_size//2
            X = X[half_diff_size:-half_diff_size,half_diff_size:-half_diff_size,half_diff_size:-half_diff_size ]
            X = np.array(X)
            X[0,:,:] = 0 
            X[:,0,:] = 0 
            X[:,:,0] = 0 
            X = jnp.asarray(X)
        elif mrc_grid_size < grid_size:
            diff_size = grid_size - mrc_grid_size 
            half_diff_size = diff_size//2            
            X_new = np.zeros( 3*[grid_size] , dtype = X.dtype )
            X_new[half_diff_size:-half_diff_size,half_diff_size:-half_diff_size,half_diff_size:-half_diff_size ] = X
            X = jnp.asarray(X_new)

            
        
        # Pad in real space.
        X = ftu.get_idft3(X)
        X_padded = np.zeros_like(X, shape = np.array(X.shape) + padding )
        half_pad = padding//2
        X_padded[half_pad:X.shape[0] + half_pad,half_pad:X.shape[1] + half_pad,half_pad:X.shape[2] + half_pad] = X
        
        # Back to FT space
        X_padded = ftu.get_dft3(X_padded)

        Xs_vec.append(np.array(X_padded.reshape(-1)))
    voxel_size = voxel_size / grid_size * mrc_grid_size

    return np.stack(Xs_vec), voxel_size


def generate_random_rotations(n_images):
    rotation_matrices = [] 
    for rot_idx in range(n_images):
        random_rot = Rotation.random()
        rotation_matrices.append(random_rot.as_matrix())
    rotation_matrices = jnp.array(rotation_matrices)
    return rotation_matrices


def get_gt_reconstruction(grid_size, voxel_size, padding, exp_name, valid_indices ):
    datadir, vol_datadir, fake_vol_exp_name, fake_vol_datadir, indf, label_file, cov_noise_inp, uninvert_data, ctf_pose_datadir = preprocessed_datasets.get_dataset_params(exp_name, on_della=True)
    gt_reconstruction = ExperimentReconstruction(grid_size, voxel_size, padding, exp_name, datadir, fake_vol_datadir,
                                                     label_file = label_file, valid_indices = valid_indices)
    return gt_reconstruction

def get_col_covariance_for_one_X_one_index(X, Xmean, vec_index, np = jnp):
    return (X - Xmean) * np.conj(X[vec_index] - Xmean[vec_index])

get_col_covariance_for_one_X_many_index = jax.vmap(get_col_covariance_for_one_X_one_index, in_axes = (None, None, 0 ) )
get_col_covariance_for_many_X_one_index = jax.vmap(get_col_covariance_for_one_X_one_index, in_axes = (0, None, None ) )

def get_col_covariance(Xs, X_mean, vec_indices, prob_of_X):
    # Batching over both dimensions

    cov = np.zeros_like(Xs, shape = [X_mean.size, vec_indices.size])
    # for k in range(Xs.shape[0]):
    #     for v_idx,v in enumerate(vec_indices):
    #         cov[:,v_idx] += prob_of_X[k] * get_col_covariance_for_one_X_one_index(Xs[k], X_mean, v, np = np ).T
    
    Xs_j = jnp.array(Xs)
    for v_idx,v in enumerate(vec_indices):
        cov[:,v_idx] = jnp.sum(prob_of_X[...,None] * get_col_covariance_for_one_X_one_index(Xs_j, X_mean, v, np = jnp ), axis =0)

    return cov



# def generate_fake_dataset(exp_name, on_della, cryos, gt_reconstruction, contrast_half_interval = 0.0, load_clean_images = False, disc_type = "linear_interp", use_nufft = False, jax_rand_key = 0, noise_multiplier = 1, input_contrast = None  ):
#     cov_noise = np.array([78090.81])
#     datadir, vol_datadir, fake_vol_exp_name, fake_vol_datadir, indf, label_file, cov_noise, uninvert_data, ctf_pose_datadir = dataset_params.get_dataset_params(exp_name, on_della)
    
#     synthetic_cov_noise = cov_noise * noise_multiplier
#     all_contrasts = [] 
#     for cryo_idx, cryo in enumerate(cryos):

#         vol_datadir = fake_vol_datadir if fake_vol_datadir is not None else vol_datadir    
#         exp_name = fake_vol_exp_name if fake_vol_exp_name is not None else exp_name    

#         volume_params, image_option = gsp.get_synthetic_dataset_input(exp_name, vol_datadir)  
            
#         im_assign = gt_reconstruction.image_assignment[cryo.dataset_indices] 
        

#         if load_clean_images:
#             with open(exp_name + "_synthetic_clean" + str(cryo.grid_size - padding) + ".pkl", "rb") as f:
#                 clean_images = pickle.load(f)
#                 if n_images is not None:
#                     clean_images = clean_images[random_ind]
#         else:

#             clean_images, _ = gsp.generate_synthetic_clean_images(volume_params, image_option, im_assign, 
#                                                                                         cryo.rotation_matrices, cryo.image_stack.D, 
#                                                                                         cryo.voxel_size, use_nufft, 
#                                                                   padding = cryo.image_stack.padding, disc_type = disc_type, gt_reconstruction = gt_reconstruction)
            
#         if input_contrast is not None:
#             contrast = input_contrast[cryo.dataset_indices]
#         else:    
#             assert(contrast_half_interval >= 0)
#             assert(contrast_half_interval < 1)
#             contrast = np.random.uniform(1 - contrast_half_interval, 1 + contrast_half_interval, cryo.n_images)
#         all_contrasts.append(contrast)
            
#         clean_images *= contrast[...,None]
#         jax_rand_key, subkey = jax.random.split(jax_rand_key)
#         image_stack = gsp.generate_fake_data_from_experiment(cryo, volume_params, image_option, im_assign, use_nufft, synthetic_cov_noise, rand_key = subkey, clean_images = clean_images, padding = cryo.padding, disc_type = disc_type )
#         cryo.image_stack = image_stack
#     all_contrasts = np.concatenate(all_contrasts)
#     return all_contrasts, synthetic_cov_noise
#     # return
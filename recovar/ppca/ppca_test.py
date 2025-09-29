#%%

import os


os.environ["CUDA_VISIBLE_DEVICES"] = "2"


import jax
print(f"JAX devices: {jax.devices()}")

from importlib import reload
from importlib.resources import files, as_file

import numpy as np
import jax
import recovar
from recovar import (
    plot_utils, 
    output,
    simulator,
    utils,
    dataset,
    em,
    regularization, 
    synthetic_dataset,
    utils,
    noise
)
from recovar import ppca
from recovar import fourier_transform_utils 
import jax.numpy as jnp
ftu = fourier_transform_utils.fourier_transform_utils(jnp)

import time
import matplotlib.pyplot as plt
import seaborn as sns

reload(simulator)

# Dataset parameters
grid_size = 64
n_images = 10000

generate_data = True
noise_level = 1

# Input/output paths
volume_folder_input = str(files(recovar) / "data" / "vol")
volume_folder_input = '/scratch/gpfs/mg6942/cooperative/models/renamed/'
output_folder = "/tmp/em_test/"

# Dataset generation parameters
volume_distribution = None ##np.array([0.5, 0.5, 0])
voxel_size = 4.25 * 128 / grid_size

# Create output directory
output.mkdir_safe(output_folder)

if generate_data:
    image_stack, sim_info = simulator.generate_synthetic_dataset(output_folder, voxel_size, volume_folder_input, n_images,
                                                                    outlier_file_input = None, grid_size = grid_size,
                                    volume_distribution = volume_distribution,  dataset_params_option = "uniform", noise_level = noise_level,
                                    noise_model = "white", put_extra_particles = False, percent_outliers = 0.00, 
                                    volume_radius = 0.7, trailing_zero_format_in_vol_name = True, noise_scale_std =0, contrast_std = 0  , disc_type = 'nearest', n_tilts = -1 )
    print(f"Finished generating dataset {output_folder}")


# Set up rotations and translation grids
healpix_order = 3
angles = em.sampling.get_rotation_grid(healpix_order)
rotation_grid = utils.R_from_relion(angles)
translation_grid = em.sampling.get_translation_grid(2, 1)

# Load dataset
dataset_dict = dataset.get_default_dataset_option()
dataset_dict.update({
    'ctf_file': output_folder + "ctf.pkl",
    'poses_file': output_folder + "poses.pkl", 
    'particles_file': f"{output_folder}particles.{grid_size}.mrcs"
})
cryo = dataset.load_dataset_from_dict(dataset_dict, lazy=False)

# Load ground truth data
sim_info = utils.pickle_load(output_folder + '/simulation_info.pkl')
gt_recon = synthetic_dataset.load_heterogeneous_reconstruction(sim_info)
gt_vol = gt_recon.volumes
noise_variance = sim_info['noise_variance']

# Get mean volume and signal variance
gt_results = synthetic_dataset.load_heterogeneous_reconstruction(sim_info)
mean_estimate = gt_results.get_mean()

ind_split = [np.arange(n_images//2), np.arange(n_images//2, n_images)]
del dataset_dict['ind']
cryos = dataset.get_split_datasets_from_dict(dataset_dict, ind_split, False)

for cryo in cryos:
    cryo.set_radial_noise_model(noise_variance)

volume_shape = cryos[0].volume_shape
volume_size = cryos[0].volume_size


import pywt

# =============================================================================
# CELL: Wavelet Analysis
# =============================================================================

volume_size = cryos[0].volume_size
mean_real = (ftu.get_idft3(mean_estimate.reshape(volume_shape)) * np.sqrt(volume_size)).real
variance_estimate_coeffs, level_results = ppca.sparse_PCA.wavelet_avg_square_by_level_both(mean_real, wavelet_type='db1')
laplace_coeff_estimate = jnp.sqrt(variance_estimate_coeffs/2)

print("Wavelet analysis complete.")



# =============================================================================
# CELL: ADMM Setup and Proximal Operators
# =============================================================================

from recovar import relion_functions
import recovar
import recovar.ppca.admm_test
reload(recovar.ppca)
reload(recovar.ppca.admm_test)

disc_type = 'linear_interp'
batch_size = 500
noise_variance_radial = utils.make_radial_image(noise_variance, cryos[0].image_shape)

# Generate triangular kernel data
ft_ctf, ft_y = relion_functions.relion_style_triangular_kernel(
    cryo, noise_variance_radial.astype(np.float32), batch_size, disc_type=disc_type
)

# Prepare initial guess and ensure real numbers
X0 = (mean_estimate.reshape(volume_shape))  # Force real
X0_flatten = X0.flatten().reshape(-1, 1)

# For wavelet operations, we need the data in (n_basis, volume_size) format
X0_wavelet_format = X0_flatten.T  # Shape: (1, volume_size)

# Setup proximal operators with correct dimensions
from recovar.ppca.admm_test import LeastSquareFromNormalEqs

# Reshape data for batch processing and ensure real numbers
lhs = (ft_ctf.reshape(volume_size, 1, 1))  # (volume_size, 1, 1) - force real
rhs = (ft_y.reshape(volume_size, 1))       # (volume_size, 1) - force real

rhs = utils.symmetrize_ft_volume(rhs, volume_shape)
lhs = utils.symmetrize_ft_volume(lhs, volume_shape)

# Use scalar sigma for wavelet regularization (ensure real)
sigma_scalar = np.mean(np.real((np.array(1/laplace_coeff_estimate[:,None])))) 

# Create proximal operators with proper dimensions
# For single basis function case, we need (volume_size, 1) shape
normal_size = (volume_size, 1)  # (volume_size, n_basis) where n_basis=1

# Create proximal operators
prox_lstsr = LeastSquareFromNormalEqs(normal_size, lhs, rhs)

ppca.check_imaginary_part(X0_flatten, volume_shape, 'X0_flatten', skip_ft = False )
ppca.check_imaginary_part(rhs / (lhs[..., -1] + 0.001), volume_shape, 'rhs / lhs[..., -1]', skip_ft = False )


# zz = prox_lstsr.prox(X0_flatten, 1)
# ppca.check_imaginary_part(zz, volume_shape, 'zz', skip_ft = False )

# zz = prox_lstsr.prox(zz, 1)
# ppca.check_imaginary_part(zz, volume_shape, 'zz', skip_ft = False )


#%%


prox_wavelet = recovar.ppca.admm_test.WaveletL1(normal_size, volume_shape, 'db1', sigma=sigma_scalar)
from pyproximal.proximal import L1
# prox_wavelet = L1(sigma_scalar)

print("Proximal operators created successfully.")
print(f"LHS shape: {lhs.shape}, RHS shape: {rhs.shape}")
print(f"Normal size: {normal_size}, Volume shape: {volume_shape}")
print(f"Sigma: {sigma_scalar}")

# # Test basic functionality
# print("\nTesting basic proximal operator functionality...")

# # Test least squares proximal operator
# prox_result_ls = prox_lstsr.prox(X0_flatten, 1)
# test_norm_ls = np.linalg.norm(prox_result_ls - X0_flatten)
# print(f"Least squares proximal operator test norm: {test_norm_ls}")
# print(f"Least squares prox result shape: {prox_result_ls.shape}")

# # Test wavelet proximal operator
# prox_result_wav = prox_wavelet.prox(X0_flatten, 1)
# test_norm_wav = np.linalg.norm(prox_result_wav - X0_flatten)
# print(f"Wavelet proximal operator test norm: {test_norm_wav}")
# print(f"Wavelet prox result shape: {prox_result_wav.shape}")

# # Test objective functions
# obj_ls = prox_lstsr(X0_flatten)
# obj_wav = prox_wavelet(X0_flatten)
# print(f"Least squares objective: {obj_ls}")
# print(f"Wavelet objective: {obj_wav}")

# # Test objective decrease property
# print("\nTesting objective decrease property...")
# tau_test = 0.1
# prox_ls_test = prox_lstsr.prox(X0_flatten, tau_test)
# prox_wav_test = prox_wavelet.prox(X0_flatten, tau_test)

# obj_ls_orig = prox_lstsr(X0_flatten)
# obj_ls_prox = prox_lstsr(prox_ls_test)
# obj_wav_orig = prox_wavelet(X0_flatten)
# obj_wav_prox = prox_wavelet(prox_wav_test)

# print(f"LS: orig={obj_ls_orig:.6f}, prox={obj_ls_prox:.6f}, decrease={obj_ls_orig - obj_ls_prox:.6f}")
# print(f"Wav: orig={obj_wav_orig:.6f}, prox={obj_wav_prox:.6f}, decrease={obj_wav_orig - obj_wav_prox:.6f}")

# # Check if proximal operators are actually minimizing the combined objective
# def combined_objective(x, prox_ls, prox_wav, tau):
#     return 0.5 * np.linalg.norm(x - X0_flatten)**2 + tau * (prox_ls(x) + prox_wav(x))

# combined_orig = combined_objective(X0_flatten, prox_lstsr, prox_wavelet, tau_test)
# combined_ls = combined_objective(prox_ls_test, prox_lstsr, prox_wavelet, tau_test)
# combined_wav = combined_objective(prox_wav_test, prox_lstsr, prox_wavelet, tau_test)

# print(f"Combined objective - orig: {combined_orig:.6f}, LS prox: {combined_ls:.6f}, Wav prox: {combined_wav:.6f}")

# # Test the correct proximal operator property
# print("\nTesting correct proximal operator property...")
# def test_proximal_property(prox_op, x, tau, name):
#     """Test if prox_op minimizes 0.5 * ||y - x||^2 + tau * f(y)"""
#     y = prox_op.prox(x, tau)
#     f_x = prox_op(x)
#     f_y = prox_op(y)
    
#     # The proximal operator should minimize: 0.5 * ||y - x||^2 + tau * f(y)
#     F_x = 0.5 * np.linalg.norm(x - x)**2 + tau * f_x  # = tau * f_x
#     F_y = 0.5 * np.linalg.norm(y - x)**2 + tau * f_y
    
#     decrease = F_x - F_y
#     print(f"{name}: F(x)={F_x:.6f}, F(y)={F_y:.6f}, decrease={decrease:.6f}")
#     return F_y <= F_x + 1e-8

# ls_prox_correct = test_proximal_property(prox_lstsr, X0_flatten, tau_test, "LeastSquares")
# wav_prox_correct = test_proximal_property(prox_wavelet, X0_flatten, tau_test, "Wavelet")

# print(f"LeastSquares proximal property: {'PASS' if ls_prox_correct else 'FAIL'}")
# print(f"Wavelet proximal property: {'PASS' if wav_prox_correct else 'FAIL'}")

# # =============================================================================
# # CELL: Proximal Operator Testing
# # =============================================================================

# reload(recovar)
# from recovar import prox_test
# reload(recovar.prox_test)

# print("Testing proximal operators...")

# # Test LeastSquareFromNormalEqs proximal operator
# print("\n=== Testing LeastSquareFromNormalEqs ===")
# results_ls = recovar.prox_test.run_prox_object_tests(prox_lstsr, tau=0.3, x_shape_hint=X0_flatten.shape)
# print("LeastSquareFromNormalEqs tests completed successfully.")

# # Test WaveletL1 proximal operator
# print("\n=== Testing WaveletL1 ===")
# results_wav = recovar.prox_test.run_prox_object_tests(prox_wavelet, tau=0.3, x_shape_hint=X0_flatten.shape)
# print("WaveletL1 tests completed successfully.")

# print("\nProximal operator testing complete.")

# =============================================================================
# CELL: ADMM Optimization
# =============================================================================

print("\nRunning ADMM optimization...")

from pyproximal.optimization.primal import ADMM

from recovar import homogeneous
means, mean_prior, _  = homogeneous.get_mean_conformation_relion(cryos, 2*batch_size, noise_variance = noise_variance_radial,  use_regularization = False)

# from recovar.admm_test import ADMM
# # Run ADMM optimization
X0 = means['combined'].reshape(-1, 1)
# X_rec = ADMM(
#     prox_lstsr, 
#     prox_wavelet, 
#     x0=X0,
#     tau=0.9, 
#     niter=400,
#     show=True,
#     gfirst=True
# )[0]

from recovar.ppca.admm_test import admm_wavelet
reload(recovar.ppca.admm_test)
multiplier = 1e-0
X_rec, Z_rec = recovar.ppca.admm_test.admm_wavelet(lhs, rhs, multiplier * sigma_scalar, 0.9, 40, volume_shape, normal_size, X0)

# X_rec_real = ftu.get_idft3(X_rec.reshape(volume_shape))
# print(np.linalg.norm(X_rec_real.imag) / np.linalg.norm(X_rec_real))

ppca.check_imaginary_part(X_rec, volume_shape, 'X_rec', skip_ft = False )


print("ADMM optimization completed successfully.")
print(f"Reconstruction shape: {X_rec.shape}")

# Test reconstruction quality"
reconstruction_error = np.linalg.norm(X_rec - X0_flatten) / np.linalg.norm(X0_flatten)
print(f"Reconstruction error: {reconstruction_error:.6f}")

print("ADMM optimization complete.")

# %%


# Create figure with 3 subplots side by side
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Get projections
proj1 = cryos[0].get_proj(X_rec)
proj2 = cryos[0].get_proj(mean_estimate)
proj3 = cryos[0].get_proj(means['combined'])

# Plot each with its own scale
im1 = ax1.imshow(proj1)
ax1.set_title('ADMM Wavelet')
plt.colorbar(im1, ax=ax1)

im2 = ax2.imshow(proj2)
ax2.set_title('Mean Estimate') 
plt.colorbar(im2, ax=ax2)

im3 = ax3.imshow(proj3)
ax3.set_title('Combined Means')
plt.colorbar(im3, ax=ax3)

plt.tight_layout()
plt.savefig('/home/mg6942/recovar/_ppca_test.png')

print(prox_lstsr(means['combined'].reshape(1, -1)))
print(prox_lstsr(X_rec))
print(prox_lstsr(mean_estimate.reshape(1, -1)))

# print(np.sum(Z_rec ==0))

ax = plot_utils.plot_fsc(cryos[0], Z_rec, mean_estimate, name = 'ADMM Wavelet Z', threshold = 0.5)
plot_utils.plot_fsc(cryos[0], X_rec, mean_estimate, name = 'ADMM Wavelet', threshold = 0.5, ax = ax)
plot_utils.plot_fsc(cryos[0], means['combined'], mean_estimate, name = 'init estimate', threshold = 0.5, ax = ax)

print('multiplier', multiplier)
print('sigma_scalar.size', sigma_scalar.size)


# %%
# %autoreload 2b

import jax.numpy as jnp
U_gt, s_gt, _ = gt_results.get_vol_svd() 
basis_size = 10
# U_initial = U_initial[:,:basis_size]
# s_initial = s_initial[:basis_size]
# Square root of covar
W_gt = U_gt[:,:basis_size] * s_gt[:basis_size]

w_gt_averaged = regularization.batch_average_over_shells(jnp.abs(W_gt.T)**2, gt_results.volume_shape, 0 )
W_prior = utils.batch_make_radial_image(w_gt_averaged, gt_results.volume_shape, True).T


from recovar import ppca
from recovar import fourier_transform_utils 
ftu = fourier_transform_utils.fourier_transform_utils(jnp)


# More reliable reloading approach
import importlib
import sys

# Remove the modules from cache to force reload
modules_to_reload = ['ppca', 'recovar.ppca', 'recovar', 'recovar.ppca.ppca']
for module_name in modules_to_reload:
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])

# Re-import after reload
import recovar.ppca.ppca as ppca
W_initial = np.random.randn(*W_prior.shape).T
W_initial = W_initial.reshape(W_initial.shape[0], *cryos[0].volume_shape)
W_initial = ftu.get_dft3(W_initial).reshape(W_initial.shape[0], -1).T
prior_multiplier = 1e-1
u_ppca, s_ppca, W, expected_zs, second_moment_zs = recovar.ppca.EM(cryos, mean_estimate, W_initial , W_prior * prior_multiplier, U_gt = U_gt, S_gt = s_gt**2, EM_iter = 20)

# %%
basis_size = W_initial.shape[-1]

sigma_scalars = []
for k in range(W.shape[-1]):
    mean_real = (ftu.get_idft3(W_gt[:,k].reshape(volume_shape)) * np.sqrt(volume_size)).real
    from recovar.ppca import sparse_PCA
    variance_estimate_coeffs, level_results = sparse_PCA.wavelet_avg_square_by_level_both(mean_real, wavelet_type='db1') 
    variance_estimate_coeffs = variance_estimate_coeffs * prior_multiplier
    laplace_coeff_estimate = jnp.sqrt(variance_estimate_coeffs/2)
    sigma_scalar = (np.real((np.array(1/laplace_coeff_estimate[:,None])))) * np.ones(laplace_coeff_estimate.shape)[:,None]
    # sigma_scalar = sigma_scalar.repeat(basis_size, axis = 1) 
    sigma_scalars.append(sigma_scalar)

sigma_scalars = np.concatenate(sigma_scalars, axis = 1) * 1e0


u_ppca, s_ppca, W, expected_zs, second_moment_zs = recovar.ppca.EM(cryos, mean_estimate, W_initial , sigma_scalar , sparse_PCA = True, U_gt = U_gt, S_gt = s_gt**2, EM_iter = 20)
# %%
plt.imshow(mean_real.sum(axis=2))

# %%

max_size_this = np.min([20, u_ppca.shape[-1]])
u = { 'ppca': u_ppca, 'gt': U_initial}
cryos = experiment_dataset
ppca_key = 'ppca'
fig, axes = plt.subplots( len(u.keys()), np.max([2, u[ppca_key].shape[-1]]), figsize=(6, 6))
for i, u_key in enumerate(u.keys()):
    # Plot PPCA components
    for j in range(u[ppca_key].shape[-1]):
        axes[i, j].imshow(cryos[0].get_proj(u[u_key][:,j].reshape(-1)))
        axes[i, j].set_title(f'{u_key} PC{j+1}')


# %%

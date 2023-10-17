import jax.numpy as jnp
import numpy as np
import jax, functools

from recovar import core
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)


# Solves the linear system Dx = b.
def simulate_data(experiment_dataset , cov_noise,  batch_size, volumes, image_assignments, disc_type, mrc, contrasts = None, seed =0 ):
    
    #images = np.zeros(experiment_dataset.image_stack.particles.shape, dtype = experiment_dataset.image_stack.particles.dtype)
    key = jax.random.PRNGKey(seed)
    # center_atoms(prody.parsePDB(mol))
    
    #batch_idft2 = jax.vmap(jax.jit(ftu.get_inverse_fourier_transform,   static_argnums = (1)), in_axes = (0, None) )
    batch_idft2 = jax.vmap(ftu.get_inverse_fourier_transform, in_axes = (0, None) )

    
    for vol_idx in range(len(volumes)):
        img_indices = np.nonzero(image_assignments[experiment_dataset.dataset_indices] == vol_idx)[0]
        n_images = img_indices.size
        
        if disc_type != "pdb":
            vol_real = ftu.get_idft3(volumes[vol_idx].reshape(experiment_dataset.volume_shape))

        for k in range(0, int(np.ceil(n_images/batch_size))):
            batch_st = int(k * batch_size)
            batch_end = int(np.min( [(k+1) * batch_size, n_images]))
            indices = img_indices[batch_st:batch_end]            
            if disc_type == "nufft":
                images_batch = simulate_nufft_data_batch(vol_real, 
                                                 experiment_dataset.rotation_matrices[indices], 
                                                 experiment_dataset.translations[indices], 
                                                 experiment_dataset.CTF_params[indices], 
                                                 experiment_dataset.voxel_size, 
                                                 experiment_dataset.volume_shape, 
                                                 experiment_dataset.image_shape, 
                                                 experiment_dataset.grid_size, 
                                                 disc_type,
                                                 experiment_dataset.CTF_fun)
            elif disc_type == "pdb":
                images_batch = simulate_nufft_data_batch_from_pdb(volumes[vol_idx],
                                                 experiment_dataset.rotation_matrices[indices], 
                                                 experiment_dataset.translations[indices], 
                                                 experiment_dataset.CTF_params[indices], 
                                                 experiment_dataset.voxel_size, 
                                                 experiment_dataset.volume_shape, 
                                                 experiment_dataset.image_shape, 
                                                 experiment_dataset.grid_size, 
                                                 disc_type,
                                                 experiment_dataset.CTF_fun)
                
            else:
                images_batch = simulate_data_batch(volumes[vol_idx],
                                                 experiment_dataset.rotation_matrices[indices], 
                                                 experiment_dataset.translations[indices], 
                                                 experiment_dataset.CTF_params[indices], 
                                                 experiment_dataset.voxel_size, 
                                                 experiment_dataset.volume_shape, 
                                                 experiment_dataset.image_shape, 
                                                 experiment_dataset.grid_size, 
                                                 disc_type,
                                                 experiment_dataset.CTF_fun)
            # if disc_type == "pdb":
            #     print("here")
            #     images_batch = batch_idft2(images_batch.reshape([-1, *experiment_dataset.image_shape]), experiment_dataset.voxel_size).real
            # else:
            #     images_batch = ftu.get_idft2(images_batch.reshape([-1, *experiment_dataset.image_shape])).real

            images_batch = ftu.get_idft2(images_batch.reshape([-1, *experiment_dataset.image_shape])).real
                
            if contrasts is not None:
                images_batch*= contrasts[indices][...,None,None]
            
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, images_batch.shape ) / jnp.sqrt(np.prod(experiment_dataset.image_stack.unpadded_image_shape))
            noise *= jnp.sqrt(cov_noise)
        
            mrc.data[indices] = np.array(images_batch + noise)
            print(k)
            
    return mrc


@functools.partial(jax.jit, static_argnums = [4,5,6,7,8,9])    
def simulate_data_batch(volume, rotation_matrices, translations, CTF_params, voxel_size, volume_shape, image_shape, grid_size, disc_type, CTF_fun ):
    
    CTF = CTF_fun( CTF_params, image_shape, voxel_size)
    corrected_images = core.get_slices(volume, rotation_matrices, image_shape, volume_shape, grid_size, disc_type) * CTF
    
    # Translate back.
    translated_images = core.translate_images(corrected_images, -translations, image_shape)
    
    return translated_images


def simulate_nufft_data_batch(volume, rotation_matrices, translations, CTF_params, voxel_size, volume_shape, image_shape, grid_size, disc_type, CTF_fun ):
    
    CTF = CTF_fun( CTF_params, image_shape, voxel_size)
    corrected_images = core.get_nufft_slices(volume, rotation_matrices, image_shape, volume_shape, grid_size, voxel_size) * CTF
    
    # Translate back.
    translated_images = core.translate_images(corrected_images, -translations, image_shape)
    return translated_images



def simulate_nufft_data_batch_from_pdb(atom_group, rotation_matrices, translations, CTF_params, voxel_size, volume_shape, image_shape, grid_size, disc_type, CTF_fun ):
    
    CTF = CTF_fun( CTF_params, image_shape, voxel_size)    
    plane_coords_mol =  core.batch_get_rotated_plane_coords(rotation_matrices, image_shape, voxel_size, True) 
    slices = core.compute_projections_with_nufft(atom_group, plane_coords_mol, voxel_size)

    corrected_images = slices * CTF
    # Translate back.
    translated_images = core.translate_images(corrected_images, -translations, image_shape)
    return translated_images


# These have not been used forever. DELETE?
def CTF_1D(k, defocus, wavelength, Cs, alpha, B):
    return jnp.sin(-jnp.pi*wavelength*defocus * k**2 + jnp.pi/2 * Cs * wavelength**3 * k **4  - alpha) * jnp.exp(- B * k**2 / 4)

@jax.jit
def CTF(psi, wavelength, defocus, Cs, alpha, B):
    k = jnp.linalg.norm(psi, axis = -1)
    return CTF_1D(k, defocus, wavelength, Cs, alpha, B)


@functools.partial(jax.jit, static_argnums=[1,2])
def get_CTF_single(CTF_params, image_shape, voxel_size):
    psi = core.get_unrotated_plane_coords(image_shape, voxel_size)
    return CTF(psi, *CTF_params)

def get_CTF_constant_single(CTF_params, image_shape, voxel_size):
    psi = core.get_unrotated_plane_coords(image_shape, voxel_size)
    return jnp.ones_like(psi[:,0])

get_CTF = jax.vmap(get_CTF_single, in_axes =(0, None, None) ) 
get_CTF_constant = jax.vmap(get_CTF_constant_single, in_axes =(0, None, None) ) 


@jax.jit
def get_random_CTF(key, plane_coords, wavelength, defocus_bounds, Cs_bounds, alpha_bounds ):
    key, subkey = jax.random.split(key)
    defocus = jax.random.uniform(subkey, minval=defocus_bounds[0], maxval = defocus_bounds[1])
    key, subkey = jax.random.split(key)
    Cs = jax.random.uniform(subkey, minval=Cs_bounds[0], maxval = Cs_bounds[1])
    key, subkey = jax.random.split(key)
    alpha = jax.random.uniform(subkey, minval=alpha_bounds[0], maxval = alpha_bounds[1])
    return CTF(plane_coords, wavelength, defocus, Cs, alpha, B = 0)


def get_random_CTF_params(key, num_params,  wavelength, defocus_bounds, Cs_bounds, alpha_bounds, B = 0  ):
    key, subkey = jax.random.split(key)
    defocus = jax.random.uniform(subkey, minval=defocus_bounds[0], maxval = defocus_bounds[1], shape = (num_params,))
    key, subkey = jax.random.split(key)
    Cs = jax.random.uniform(subkey, minval=Cs_bounds[0], maxval = Cs_bounds[1],shape = (num_params,))
    key, subkey = jax.random.split(key)
    alpha = jax.random.uniform(subkey, minval=alpha_bounds[0], maxval = alpha_bounds[1], shape = (num_params,))
    return jnp.stack( [ jnp.ones(num_params) * wavelength, defocus, Cs, alpha, jnp.ones(num_params) * B  ], axis = -1 )
    

def generate_random_CTF_params(key, n_images ):
    voltage = 200  # Voltage (in KV)
    defocus_bounds = [0.5e4, 3e4]
    Cs_bounds = [1e7, 3e7]  # Spherical aberration
    alpha_bounds = [0.1, 3] # Amplitude contrast
    import recovar.generate_synthetic_molecule as gsm
    wavelength = gsm.voltage_to_wavelength(voltage)
    keys = jax.random.split(key, num=n_images+1)
    key = keys[0]
    CTF_keys = jnp.array(keys[1:])
    CTF_params = get_random_CTF_params(key, n_images,  wavelength, defocus_bounds, Cs_bounds, alpha_bounds )
    return CTF_params
batch_get_random_CTF = jax.vmap(get_random_CTF, in_axes = (0, None, None, None, None, None ))

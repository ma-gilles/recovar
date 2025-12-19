import recovar.config
from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as py
from recovar.fourier_transform_utils import fourier_transform_utils
import jax.numpy as jnp
ftu = fourier_transform_utils(jnp)
from recovar import image_assignment, noise
from sklearn.metrics import confusion_matrix
from recovar import simulate_scattering_potential as ssp
from recovar import simulator, utils, image_assignment, noise, output, dataset
import prody
reload(simulator)

def main():
    ## CHANGE THESE FOLDERS
    output_folder = '/home/mg6942/mytigress/hard_assignment_exp/'
    pdb_folder = './'

    # Some parameters to set
    n_images = 100000
    grid_size = 256
    Bfactor = 60 
    noise_level_tests = np.logspace(1,6,20) # these seem reasonble.
    # This would produce 2 million images of size 256x256

    ## Make volumes from PDB
    pdbs = [ '3down_nogly.pdb', 'up_nogly.pdb']

    voxel_size =1.3 * 256 / grid_size
    volume_shape = tuple(3*[grid_size])

    # Center atoms (but shift by same amount)
    pdb_atoms = [ prody.parsePDB(pdb_folder + '/' +  pdb_i) for pdb_i in pdbs ]
    atoms =pdb_atoms[0]
    coords = atoms.getCoords()
    offset = ssp.get_center_coord_offset(coords)
    # coords = coords - offset
    for atoms in pdb_atoms:
        atoms.setCoords(atoms.getCoords() - offset)

        
    ## Make B-factored volumes (will be considered g.t.)     
    Bfaced_vols = len(pdbs)*[None]
    for idx, atoms in enumerate(pdb_atoms):
        volume = ssp.generate_molecule_spectrum_from_pdb_id(atoms, voxel_size = voxel_size,  grid_size = grid_size, do_center_atoms = False, from_atom_group = True)
        Bfaced_vols[idx] = simulator.Bfactorize_vol(volume.reshape(-1), voxel_size, Bfactor, volume_shape)

    disc_type_sim = 'nufft'
    disc_type_infer = 'cubic'
    # disc_type_sim = 'linear_interp'
    # disc_type_infer = 'linear_interp'


    volume_folder = output_folder + 'true_volumes/'
    output.mkdir_safe(volume_folder)
    output.save_volumes( Bfaced_vols, volume_folder, from_ft= True)

    # plt.imshow(ftu.get_idft3(Bfaced_vols[0].reshape(volume_shape)).sum(axis=0).real)
    # plt.figure()
    # plt.imshow(ftu.get_idft3(Bfaced_vols[1].reshape(volume_shape)).sum(axis=0).real)
    

    error_observed = np.zeros(noise_level_tests.size)
    error_predicted= np.zeros(noise_level_tests.size)

    for idx, noise_level in enumerate(noise_level_tests):
        
        # Generate dataset
        volume_distribution = np.array([0.8,0.2])
        noise_level = noise_level
        dataset_folder = output_folder + f'/dataset{idx}/'
        image_stack, sim_info = simulator.generate_synthetic_dataset(dataset_folder, voxel_size, volume_folder, n_images,
            outlier_file_input = None, grid_size = grid_size,
            volume_distribution = volume_distribution,  dataset_params_option = "uniform", noise_level = noise_level,
            noise_model = "white", put_extra_particles = False, percent_outliers = 0.00, 
            volume_radius = 0.7, trailing_zero_format_in_vol_name = True, noise_scale_std = 0, contrast_std = 0, disc_type = disc_type_sim)
        
        # gt_volumes = np.array(Bfaced_vols) * sim_info['scale_vol']

        # Load datasets and volumes
        # Volumes are scaled so that images are normalized. So they have a slightly different scale for each dataset.
        volumes = simulator.load_volumes_from_folder(sim_info['volumes_path_root'], sim_info['grid_size'] , sim_info['trailing_zero_format_in_vol_name'], normalize=False )
        gt_volumes = volumes * sim_info['scale_vol']

        
        dataset_options = dataset.get_default_dataset_option()
        dataset_options['particles_file'] = dataset_folder + f'particles.{grid_size}.mrcs'
        dataset_options['ctf_file'] = dataset_folder + f'ctf.pkl'
        dataset_options['poses_file'] = dataset_folder + f'poses.pkl'
        cryo = dataset.load_dataset_from_dict(dataset_options, lazy = False)
        
        # Compute hard-assignment
        batch_size = 1000
        image_cov_noise = np.asarray(noise.make_radial_noise(sim_info['noise_variance'], cryo.image_shape))
        log_likelihoods = image_assignment.compute_image_assignment(cryo, gt_volumes,  image_cov_noise, batch_size, disc_type = disc_type_infer)
        assignments = jnp.argmin(log_likelihoods, axis = 0)
        
        confus = confusion_matrix(assignments, sim_info['image_assignment'])
        
        # Compute the gamma from the note.
        
        if confus.size > 1:
            error_observed[idx] = (confus[1,0] + confus[0,1] ) / assignments.size
        else:
            error_observed[idx] = 0
        
        error_predicted[idx] = image_assignment.estimate_false_positive_rate(cryo, gt_volumes,  image_cov_noise, batch_size, disc_type = disc_type_infer)
        print('o', error_observed)
        print('p', error_predicted)
        
        # Checking with the deconvolution formula gives
        observed_pop = np.array([np.mean(assignments==0), np.mean(assignments==1)])
        deconvolve_matrix = np.array( [ [1- error_predicted[idx], error_predicted[idx]], [error_predicted[idx], 1- error_predicted[idx]] ])
        print('Observed pop:', observed_pop)
        print('Deconvolve mat:', deconvolve_matrix)
        print('Deconvolved pop:', np.linalg.solve(deconvolve_matrix, observed_pop))

        # Dump results to file
        result = { 'log_llh': log_likelihoods, 'hard_assignment' : assignments, 'true_assignment' : sim_info['image_assignment'] , 'predicted_error_rate' : error_predicted[idx]   } 
        recovar.utils.pickle_dump( result, dataset_folder + 'result.pkl')
        recovar.utils.pickle_dump( { 'error_observed' : error_observed, \
                                    'error_predicted' : error_predicted, 'noise_level_tests' : noise_level_tests },
                                output_folder + 'curve.pkl') 

        # Make a plot each time.
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.semilogx(noise_level_tests[:idx+1], error_predicted[:idx+1], '-o', label='Analytical', color='blue', marker='o', markersize=6, linewidth=2)
        plt.semilogx(noise_level_tests[:idx+1], error_observed[:idx+1], '-s', label='Observed', color='green', marker='s', markersize=6, linewidth=2)

        plt.xlabel('Noise Level', fontsize=14)
        plt.ylabel('False Positive Rate', fontsize=14)
        plt.title('False Positive Rate vs. Noise Level', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(output_folder + 'curve.png')
        plt.show()

        

if __name__ == '__main__':
    main()
    print("Done")
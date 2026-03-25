RECOVAR Job: compute_trajectory
Command: python /scratch/gpfs/GILLES/mg6942/recovar_wt_output_cleanup/recovar/command_line.py /scratch/gpfs/GILLES/mg6942/recovar_wt_output_cleanup/.tmp/slurm_6060880/pytest-of-mg6942/pytest-0/downstream_gpu0/pipeline_output -o /scratch/gpfs/GILLES/mg6942/recovar_wt_output_cleanup/.tmp/slurm_6060880/pytest-of-mg6942/pytest-0/downstream_gpu0/compute_trajectory_output --zdim 4 --endpts /scratch/gpfs/GILLES/mg6942/recovar_wt_output_cleanup/.tmp/slurm_6060880/pytest-of-mg6942/pytest-0/downstream_gpu0/analysis_output/kmeans/centers.txt --ind 0,1 --n-vols-along-path 3 --lazy
Status: completed
Duration: 3m 26s
Version: 1.0.0b1

VOLUMES:
  state000.mrc
  state000_unfil.mrc
  state001.mrc
  state001_unfil.mrc
  state002.mrc
  state002_unfil.mrc

HALFMAPS:
  state000_half1_unfil.mrc
  state000_half2_unfil.mrc
  state001_half1_unfil.mrc
  state001_half2_unfil.mrc
  state002_half1_unfil.mrc
  state002_half2_unfil.mrc

DIAGNOSTICS:
  diagnostics/state000/heterogeneity_distances.txt
  diagnostics/state000/latent_coords.txt
  diagnostics/state000/local_resolution.mrc
  diagnostics/state000/params.pkl
  diagnostics/state000/sampling.mrc
  diagnostics/state000/split_choice.pkl
  diagnostics/state001/heterogeneity_distances.txt
  diagnostics/state001/latent_coords.txt
  diagnostics/state001/local_resolution.mrc
  diagnostics/state001/params.pkl
  diagnostics/state001/sampling.mrc
  diagnostics/state001/split_choice.pkl
  diagnostics/state002/heterogeneity_distances.txt
  diagnostics/state002/latent_coords.txt
  diagnostics/state002/local_resolution.mrc
  diagnostics/state002/params.pkl
  diagnostics/state002/sampling.mrc
  diagnostics/state002/split_choice.pkl

OTHER:
  latent_coords.txt
  path.json

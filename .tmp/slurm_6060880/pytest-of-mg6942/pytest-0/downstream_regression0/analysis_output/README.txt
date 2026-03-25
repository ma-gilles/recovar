RECOVAR Job: analyze
Command: python /scratch/gpfs/GILLES/mg6942/recovar_wt_output_cleanup/recovar/command_line.py /scratch/gpfs/GILLES/mg6942/recovar_wt_output_cleanup/.tmp/slurm_6060880/pytest-of-mg6942/pytest-0/downstream_regression0/pipeline_output --zdim 4 -o /scratch/gpfs/GILLES/mg6942/recovar_wt_output_cleanup/.tmp/slurm_6060880/pytest-of-mg6942/pytest-0/downstream_regression0/analysis_output --n-clusters 5 --skip-umap --lazy
Status: completed
Duration: 4m 57s
Version: 1.0.0b1

VOLUMES:
  kmeans/center000.mrc
  kmeans/center000_unfil.mrc
  kmeans/center001.mrc
  kmeans/center001_unfil.mrc
  kmeans/center002.mrc
  kmeans/center002_unfil.mrc
  kmeans/center003.mrc
  kmeans/center003_unfil.mrc
  kmeans/center004.mrc
  kmeans/center004_unfil.mrc
  kmeans/diagnostics/center002/local_resolution.mrc
  kmeans/diagnostics/center002/sampling.mrc
  kmeans/diagnostics/center003/local_resolution.mrc
  kmeans/diagnostics/center003/sampling.mrc
  kmeans/diagnostics/center004/local_resolution.mrc
  kmeans/diagnostics/center004/sampling.mrc
  kmeans/diagnostics/center000/local_resolution.mrc
  kmeans/diagnostics/center000/sampling.mrc
  kmeans/diagnostics/center001/local_resolution.mrc
  kmeans/diagnostics/center001/sampling.mrc

HALFMAPS:
  kmeans/center000_half1_unfil.mrc
  kmeans/center000_half2_unfil.mrc
  kmeans/center001_half1_unfil.mrc
  kmeans/center001_half2_unfil.mrc
  kmeans/center002_half1_unfil.mrc
  kmeans/center002_half2_unfil.mrc
  kmeans/center003_half1_unfil.mrc
  kmeans/center003_half2_unfil.mrc
  kmeans/center004_half1_unfil.mrc
  kmeans/center004_half2_unfil.mrc

PLOTS:
  plots/contrast_histogram.png
  plots/PCA/PC_01.png
  plots/PCA/PC_01no_annotate.png
  plots/PCA/PC_02.png
  plots/PCA/PC_02no_annotate.png
  plots/PCA/PC_03.png
  plots/PCA/PC_03no_annotate.png
  plots/PCA/PC_12.png
  plots/PCA/PC_12no_annotate.png

OTHER:
  kmeans/centers.txt
  kmeans/latent_coords.txt
  kmeans/diagnostics/center002/heterogeneity_distances.txt
  kmeans/diagnostics/center002/latent_coords.txt
  kmeans/diagnostics/center002/params.pkl
  kmeans/diagnostics/center002/split_choice.pkl
  kmeans/diagnostics/center003/heterogeneity_distances.txt
  kmeans/diagnostics/center003/latent_coords.txt
  kmeans/diagnostics/center003/params.pkl
  kmeans/diagnostics/center003/split_choice.pkl
  kmeans/diagnostics/center004/heterogeneity_distances.txt
  kmeans/diagnostics/center004/latent_coords.txt
  kmeans/diagnostics/center004/params.pkl
  kmeans/diagnostics/center004/split_choice.pkl
  kmeans/diagnostics/center000/heterogeneity_distances.txt
  kmeans/diagnostics/center000/latent_coords.txt
  kmeans/diagnostics/center000/params.pkl
  kmeans/diagnostics/center000/split_choice.pkl
  kmeans/diagnostics/center001/heterogeneity_distances.txt
  kmeans/diagnostics/center001/latent_coords.txt
  kmeans/diagnostics/center001/params.pkl
  kmeans/diagnostics/center001/split_choice.pkl
  data/kmeans_result.pkl
  data/trajectory_endpoints.pkl

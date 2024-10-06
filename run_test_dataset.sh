RECOVAR_PATH=./
# Generate a small test dataset - should take about 30 sec
python $RECOVAR_PATH/make_test_dataset.py

# Run pipeline, should take about 2 min
python $RECOVAR_PATH/pipeline.py test_dataset/particles.64.mrcs --poses test_dataset/poses.pkl --ctf test_dataset/ctf.pkl --correct-contrast -o test_dataset/pipeline_output --mask=from_halfmaps --lazy

# Run on the 2D embedding with no regularization on latent space (better for density estimation)
# Should take about 5 min
python $RECOVAR_PATH/analyze.py test_dataset/pipeline_output --zdim=2 --no-z-regularization --n-clusters=3 --n-trajectories=0

python  $RECOVAR_PATH/recovar/estimate_conformational_density.py pipeline_output --deconvolved_dim 2

python $RECOVAR_PATH/compute_trajectory.py pipeline_output -o pipeline_output/trajectory1 --endpts pipeline_output/analysis_2_noreg/kmeans_center_coords.txt --density pipeline_output/density/deconv_density_knee.pkl

# option 2
python $RECOVAR_PATH/compute_trajectory.py pipeline_output -o pipeline_output/trajectory1 --endpts pipeline_output/analysis_2_noreg/kmeans_results.pkl --density pipeline_output/density/deconv_density_knee.pkl --kmeans-ind=0,1

# python ~/recovar/compute_trajectory.py test2 --zdim 4 --density test2/density/deconv_density_knee.pkl --kmeans-ind=0,6 --endpts test2/analysis_4_noreg/kmeans_result.pkl -o test2/trajectory

# You may want to delete this directory after running the test. 
# rm -rf $RECOVAR_PATH/test_dataset

## One way to make sure everything went well is that the states in test_dataset/pipeline_output/output/analysis_2_noreg/centers/all_volumes should be similar to the simulated ones in recovar/data/vol*.mrc (the order doesn't matter, though)
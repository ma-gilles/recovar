# RECOVAR: Regularized covariance estimation for cryo-EM heterogeneity analysis

[See preprint here](https://www.biorxiv.org/content/10.1101/2023.10.28.564422v3) and recorded talk [here](https://www.youtube.com/watch?v=cQBQlCCRp8Q&t=740s)

[Installation](#installation)

Running RECOVAR:
* [1. Preprocessing](#i-preprocessing)
* [2. Specifying a mask](#ii-specifying-a-real-space-mask)
* [3. Running the pipeline](#iii-running-recovar-pipeline)
* [4. Analyzing results](#iv-analyzing-results)
* [5. Visualizing results](#v-visualizing-results)


Other:
* [RECOVAR for Cryo-ET](#cryo-et)
* [Testing your installation](#small-test-dataset)
<!-- * [6. Generating trajectories](#vi-generating-additional-trajectories) -->
* [Using kernel regression with other embeddings](#using-kernel-regression-with-other-embeddings)


[TLDR](#tldr)

(OUT OF DATE)
Peak at what output looks like on a [synthetic dataset](output_visualization_simple_synthetic.ipynb) and [real dataset](output_visualization_empiar10076.ipynb).

Also:
[using the source code](#using-the-source-code), 
[limitations](#limitations), 
[contact](#contact)

## Installation 
To run this code, CUDA and [JAX](https://jax.readthedocs.io/en/latest/index.html#) are required. See information about JAX installation [here](https://jax.readthedocs.io/en/latest/installation.html).
Assuming you already have CUDA, installation should take less than 5 minutes.
Below are a set of commands which runs on our university cluster (Della), but may need to be tweaked to run on other clusters.
You may need to load CUDA before installing JAX, E.g., on our university cluster with

    module load cudatoolkit/12.3

Then create an environment, download JAX-cuda (for some reason the latest version is causing issues, so make sure to use 0.4.23), clone the directory and install the requirements (note the --no-deps flag. This is because of some conflict with dependencies of cryodrgn. Will fix it soon.).

    conda create --name recovar python=3.11
    conda activate recovar
    pip install -U "jax[cuda12_pip]"==0.4.23 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    git clone https://github.com/ma-gilles/recovar.git
    pip install --no-deps -r  recovar/recovar_install_requirements.txt
    python -m ipykernel install --user --name=recovar 




<!-- The code was tested on [this commit](https://github.com/ma-gilles/recovar/commit/6388bcc8646c535ae1b121952aa5c04e52402455).

The code for the paper was run on [this commit](https://github.com/ma-gilles/recovar/commit/6388bcc8646c535ae1b121952aa5c04e52402455). -->




## I. Preprocessing 

The input interface of RECOVAR is borrowed directly from the excellent [cryoDRGN toolbox](https://cryodrgn.cs.princeton.edu/). 
Particles, poses and CTF must be prepared in the same way, and below is copy-pasted part of 
[cryoDRGN's documentation](https://github.com/ml-struct-bio/cryodrgn#2-parse-image-poses-from-a-consensus-homogeneous-reconstructiqqon).
<!-- CryoDRGN is a dependency, so you should be able to run the commands below after ``conda activate recovar``. -->
You should first install cryoDRGN, and prepare the dataset as below before going on to step 2.


### cryoDRGN Installation

`cryodrgn` may be installed via `pip`, and we recommend installing `cryodrgn` in a clean conda environment.

    # Create and activate conda environment
    (base) $ conda create --name cryodrgn python=3.9
    (cryodrgn) $ conda activate cryodrgn

    # install cryodrgn
    (cryodrgn) $ pip install cryodrgn

(NOTE: right now you need to install cryoDRGN and RECOVAR in two different environments, will fix soon!) 


You can alternatively install a newer, less stable, development version of `cryodrgn` using our beta release channel:

    (cryodrgn) $ pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ cryodrgn --pre

More installation instructions are found in the [documentation](https://ez-lab.gitbook.io/cryodrgn/installation).



### 1. Preprocess image stack

First resize your particle images using the `cryodrgn downsample` command:

<details><summary><code>$ cryodrgn downsample -h</code></summary>

    usage: cryodrgn downsample [-h] -D D -o MRCS [--is-vol] [--chunk CHUNK]
                               [--datadir DATADIR]
                               mrcs

    Downsample an image stack or volume by clipping fourier frequencies

    positional arguments:
      mrcs               Input images or volume (.mrc, .mrcs, .star, .cs, or .txt)

    optional arguments:
      -h, --help         show this help message and exit
      -D D               New box size in pixels, must be even
      -o MRCS            Output image stack (.mrcs) or volume (.mrc)
      --is-vol           Flag if input .mrc is a volume
      --chunk CHUNK      Chunksize (in # of images) to split particle stack when
                         saving
      --relion31         Flag for relion3.1 star format
      --datadir DATADIR  Optionally provide path to input .mrcs if loading from a
                         .star or .cs file
      --max-threads MAX_THREADS
                         Maximum number of CPU cores for parallelization (default: 16)
      --ind PKL          Filter image stack by these indices

</details>

We recommend first downsampling images to 128x128 since larger images can take much longer to train:

    $ cryodrgn downsample [input particle stack] -D 128 -o particles.128.mrcs

The maximum recommended image size is D=256, so we also recommend downsampling your images to D=256 if your images are larger than 256x256:

    $ cryodrgn downsample [input particle stack] -D 256 -o particles.256.mrcs

The input file format can be a single `.mrcs` file, a `.txt` file containing paths to multiple `.mrcs` files, a RELION `.star` file, or a cryoSPARC `.cs` file. For the latter two options, if the relative paths to the `.mrcs` are broken, the argument `--datadir` can supply the path to where the `.mrcs` files are located.

If there are memory issues with downsampling large particle stacks, add the `--chunk 10000` argument to save images as separate `.mrcs` files of 10k images.


### 2. Parse image poses from a consensus homogeneous reconstruction

CryoDRGN expects image poses to be stored in a binary pickle format (`.pkl`). Use the `parse_pose_star` or `parse_pose_csparc` command to extract the poses from a `.star` file or a `.cs` file, respectively.

Example usage to parse image poses from a RELION 3.1 starfile:

    $ cryodrgn parse_pose_star particles.star -o pose.pkl -D 300

Example usage to parse image poses from a cryoSPARC homogeneous refinement particles.cs file:

    $ cryodrgn parse_pose_csparc cryosparc_P27_J3_005_particles.cs -o pose.pkl -D 300

**Note:** The `-D` argument should be the box size of the consensus refinement (and not the downsampled images from step 1) so that the units for translation shifts are parsed correctly.

### 3. Parse CTF parameters from a .star/.cs file

CryoDRGN expects CTF parameters to be stored in a binary pickle format (`.pkl`). Use the `parse_ctf_star` or `parse_ctf_csparc` command to extract the relevant CTF parameters from a `.star` file or a `.cs` file, respectively.

Example usage for a .star file:

    $ cryodrgn parse_ctf_star particles.star -D 300 --Apix 1.03 -o ctf.pkl

The `-D` and `--Apix` arguments should be set to the box size and Angstrom/pixel of the original `.mrcs` file (before any downsampling).

Example usage for a .cs file:

    $ cryodrgn parse_ctf_csparc cryosparc_P27_J3_005_particles.cs -o ctf.pkl

## II. Specifying a real-space mask

A real space mask is important to boost SNR. Most consensus reconstruction software output a mask, which you can use as input (`--mask=path_to_mask.mrc`). Make sure the mask is not too tight; you can use the input `--dilate-mask-iter` to expand the mask if needed. You may also want to use a focusing mask to focus on heterogeneity in one part of the volume [click here](https://guide.cryosparc.com/processing-data/tutorials-and-case-studies/mask-selection-and-generation-in-ucsf-chimera) to find instructions to generate one with Chimera.

If you don't input a mask, you can ask the software to estimate one using the two halfmaps of the mean ( `--mask=from-halfmaps`). You may also want to run with a loose spherical mask (option `--mask=sphere`) and use the computed variance map to observe which parts have large variance.


## III. Running RECOVAR pipeline

When the input images (.mrcs), poses (.pkl), and CTF parameters (.pkl) have been prepared, RECOVAR can be run with the following command:

    $ python [recovar_directory]/pipeline.py particles.128.mrcs -o output_test --ctf ctf.pkl --poses poses.pkl --mask=[path_to_your_mask.mrc]


<details><summary><code>$ python pipeline.py -h</code></summary>

        usage: pipeline.py [-h] -o OUTDIR [--zdim ZDIM] --poses POSES --ctf pkl --mask mrc [--focus-mask mrc]                                      
                        [--mask-dilate-iter MASK_DILATE_ITER] [--correct-contrast] [--ignore-zero-frequency] [--ind PKL]                        
                        [--uninvert-data UNINVERT_DATA] [--datadir DATADIR] [--n-images N_IMAGES] [--padding PADDING] [--halfsets HALFSETS]     
                        [--keep-intermediate] [--noise-model NOISE_MODEL] [--mean-fn MEAN_FN] [--accept-cpu] [--test-covar-options]             
                        [--low-memory-option] [--dont-use-image-mask] [--do-over-with-contrast] [--tilt-series]                                 
                        [--tilt-series-ctf TILT_SERIES_CTF] [--dose-per-tilt DOSE_PER_TILT] [--angle-per-tilt ANGLE_PER_TILT] [--only-mean]     
                        particles                                                                                                               
                                                                                                                                                
        positional arguments:                                                                                                                      
        particles             Input particles (.mrcs, .star, .cs, or .txt)                                                                       
                                                                                                                                                
        optional arguments:                                                                                                                        
        -h, --help            show this help message and exit                                                                                    
        -o OUTDIR, --outdir OUTDIR                                                                                                               
                                Output directory to save model                                                                                     
        --zdim ZDIM           Dimensions of latent variable. Default=1,2,4,10,20                                                                 
        --poses POSES         Image poses (.pkl)
        --ctf pkl             CTF parameters (.pkl)
        --mask mrc            solvent mask (.mrc).Can solve provide: from_halfmaps, sphere, none                                                
        --focus-mask mrc      focus mask (.mrc)
        --mask-dilate-iter MASK_DILATE_ITER
                                mask options how many iters to dilate solvent and focus mask                                                      
        --correct-contrast    estimate and correct for amplitude scaling (contrast) variation across images                                     
        --ignore-zero-frequency
                                use if you want zero frequency to be ignored. If images have been normalized to 0 mean, this is probably a good   
                                idea
        --tilt-series         Whether to use tilt_series.
        --tilt-series-ctf TILT_SERIES_CTF
                                Whether to use tilt_series ctf. Default = same as tilt_series.                                                    
        --dose-per-tilt DOSE_PER_TILT
        --angle-per-tilt ANGLE_PER_TILT
        --only-mean           Only compute mean

        Dataset loading:
        --ind PKL             Filter particles by these indices
        --uninvert-data UNINVERT_DATA
                                Invert data sign: options: true, false, automatic (default). automatic will swap signs if sum(estimated mean) < 0 
        --datadir DATADIR     Path prefix to particle stack if loading relative paths from a .star or .cs file                                  
    --n-images N_IMAGES   Number of images to use (should only use for quick run)                                                           
    --padding PADDING     Real-space padding
    --halfsets HALFSETS   Path to a file with indices of split dataset (.pkl).
    --keep-intermediate   saves some intermediate result. Probably only useful for debugging
    --noise-model NOISE_MODEL
                            what noise model to use. Options are radial (default) computed from outside the masks, and white computed by
                            power spectrum at high frequencies
    --mean-fn MEAN_FN     which mean function to use. Options are triangular (default), old, triangular_reg
    --accept-cpu          Accept running on CPU if no GPU is found
    --test-covar-options
    --low-memory-option
    --dont-use-image-mask
    --do-over-with-contrast Whether to run again once constrast is estimated
</details>


The required arguments are:

* an input image stack (`.mrcs` or other listed file types)
* `--poses`, image poses (`.pkl`) that correspond to the input images
* `--ctf`, ctf parameters (`.pkl`), unless phase-flipped images are used
* `-o`, a clean output directory for saving results
* `--mask`, a solvent mask (`.mrc`). Can Also provide: from_halfmaps, sphere, none


Additional parameters that are typically set include:
* `--focus-mask` to specify the path to a focus mask path (`.mrc`). Note that if you only have a solvent mask you should pass it with --mask not focus-mask. If you have a focus-mask but not a solvent mask for some reason, you can pass `--mask=sphere`
<!-- * `--mask` to specify which mask to use -->
* `--dilate-mask-iter` to specify the number of dilation iterationof mask (default=0)
* `--zdim`, dimensions of PCA to use for embedding, can submit one integer (`--zdim=20`) or a or a command separated list (`--zdim=10,50,100`). Default (`--zdim=1,2,4,10,20` and using no regulariation).

<!-- * `--uninvert-data`, Use if particles are dark on light (negative stain format) -->



## IV. Analyzing results

After the pipeline is run, you can find the mean, eigenvectors, variance maps, and embeddings in the `outdir/results` directory, where outdir is the option given above by `-o`. You can run some standard analysis by running:

    python analyze.py [pipeline_output_dir] --zdim=10

It will run k-means, generate volumes corresponding to the centers, generate trajectories between pairs of cluster centers, and run UMAP. See more input details below.


<details><summary><code>$ python analyze.py -h</code></summary>
    usage: analyze.py [-h] [-o OUTDIR] [--zdim ZDIM] [--n-clusters N_CLUSTERS] [--n-trajectories N_TRAJECTORIES] [--skip-umap]
                    [--skip-centers] [--n-vols-along-path N_VOLS_ALONG_PATH] [--Bfactor BFACTOR] [--n-bins N_BINS] [--density DENSITY]
                    [--normalize-kmeans] [--no-z-regularization]
                    result_dir

    positional arguments:
    result_dir            result dir (output dir of pipeline)

    optional arguments:
    -h, --help            show this help message and exit
    -o OUTDIR, --outdir OUTDIR
                            Output directory to save model. If not provided, will save in result_dir/output/analysis_zdim/
    --zdim ZDIM           Dimension of latent variable (a single int, not a list)
    --n-clusters N_CLUSTERS
                            number of k-means clusters (default 40)
    --n-trajectories N_TRAJECTORIES
                            number of trajectories to compute between k-means clusters (default 6)
    --skip-umap           whether to skip u-map embedding (can be slow for large dataset)
    --skip-centers        whether to generate the volume of the k-means centers
    --n-vols-along-path N_VOLS_ALONG_PATH
                            number of volumes to compute along each trajectory (default 6)
    --Bfactor BFACTOR     0
    --n-bins N_BINS       number of bins for kernel regression
    --density DENSITY     density saved in .pkl file, with keys 'density' and 'latent_space_bounds'
    --normalize-kmeans    whether to normalize the zs before computing k-means
    --no-z-regularization
                            whether to use z without regularization, e.g. use 2_noreg instead of 2
</details>

To generate volumes at specific place in latent space you can use:

    python compute_state.py [pipeline_output_dir] -o [volume_output_dir] --latent-points [zfiles.txt] --Bfactor=[Bfac]

<details><summary><code>$ python compute_state.py -h</code></summary>

    usage: compute_state.py [-h] [-o OUTDIR] --latent-points LATENT_POINTS [--Bfactor BFACTOR] [--n-bins N_BINS] [--zdim1]
                            [--no-z-regularization]
                            result_dir

    positional arguments:
    result_dir            result dir (output dir of pipeline)

    optional arguments:
    -h, --help            show this help message and exit
    -o OUTDIR, --outdir OUTDIR
                            Output directory to save model
    --latent-points LATENT_POINTS
                            path to latent points (.txt file)
    --Bfactor BFACTOR     0
    --n-bins N_BINS       number of bins for kernel regression
    --zdim1               Whether dimension 1 is used. This is an annoying corner case for np.loadtxt...
    --no-z-regularization
                            Whether to use z regularization

</details>


where pipeline_output_dir is the path provided to the pipeline, latent-points is np.loadtxt-readable file containing the coordinates in latent space, and Bfactor is a b-factor to sharpen (can provide the same as the consensus reconstruction). It should be positive.

The the sharpened volume will be at volume_output_dir/vol000/

To generate a low free-energy trajectory in latent space (and volumes):

    python compute_trajectory.py [pipeline_output_dir] -o [volume_output_dir] --endpts [zfiles.txt] --Bfactor=[Bfac] --density [deconvolved_density.pkl]


<details><summary><code>$ python compute_trajectory.py -h</code></summary>

usage: compute_trajectory.py [-h] [-o OUTDIR] [--zdim ZDIM] [--n-vols-along-path N_VOLS_ALONG_PATH] [--Bfactor BFACTOR]
                             [--n-bins N_BINS] [--density DENSITY] [--no-z-regularization] [--kmeans-ind KMEANS_IND]
                             [--endpts ENDPTS_FILE]
                             result_dir

positional arguments:
  result_dir            result dir (output dir of pipeline)

optional arguments:
  -h, --help            show this help message and exit
  -o OUTDIR, --outdir OUTDIR
                        Output directory to save model
  --zdim ZDIM           Dimension of latent variable (a single int, not a list)
  --n-vols-along-path N_VOLS_ALONG_PATH
                        number of volumes to compute along each trajectory (default 6)
  --Bfactor BFACTOR     0
  --n-bins N_BINS       number of bins for reweighting
  --density DENSITY     density saved in pkl file, key is 'density' and 'latent_space_bounds
  --no-z-regularization
  --kmeans-ind KMEANS_IND
                        indices of k means centers to use as endpoints
  --endpts ENDPTS_FILE  end points file. It storing z values, it should be a .txt file with 2 rows, and if it is from kmeans, it should
                        be a .pkl file (generated by analyze)


</details>


## V. Visualizing results
### Output structure

Assuming you have run the pipeline.py and analyze.py, the output will be saved in the format below (click on the arrow). If you are running on a remote server, I suggest you only copy the [output_dir]/output locally, since the model file will be huge. You can then visualize volumes in ChimeraX.


<details><summary>Output file structure</summary>

    ├── command.txt
    ├── model
    │   ├── covariance_cols.pkl
    │   ├── embeddings.pkl
    │   ├── halfsets.pkl
    │   └── params.pkl
    ├── output
    │   ├── analysis_10
    │   │   ├── centers
    │   │   │   ├── all_volumes
    │   │   │   │   ├── locres000.mrc
    │   │   │   │   ├── locres001.mrc
    │   │   │   │   ├── ...
    │   │   │   │   ├── locres039.mrc
    │   │   │   │   ├── vol000.mrc
    │   │   │   │   ├── vol001.mrc
    │   │   │   │   └── ...
    │   │   │   ├── vol000
    │   │   │   │   ├── ml_optimized_auc.mrc
    │   │   │   │   ├── ml_optimized_half1_unfil.mrc
    │   │   │   │   ├── ml_optimized_half2_unfil.mrc
    │   │   │   │   ├── ml_optimized_locres_filtered.mrc
    │   │   │   │   ├── ml_optimized_locres_filtered_nob.mrc
    │   │   │   │   ├── ml_optimized_locres.mrc
    │   │   │   │   ├── ml_optimized_unfiltered.mrc
    │   │   │   │   ├── ml_params.pkl
    │   │   │   │   └── split_choice.pkl
    │   │   │   ├── ...
    │   │   ├── centers_01no_annotate.png
    │   │   ├── centers.pkl
    │   │   ├── path0
    │   │   │   └── density
    │   │   ├── run.log
    │   │   └── umap
    │   │       ├── centers_no_annotate.png
    │   │       ├── centers_.png
    │   │       ├── embedding.pkl
    │   │       ├── sns_hex.png
    │   │       └── sns.png
    │   └── volumes
    │       ├── dilated_mask.mrc
    │       ├── eigen_neg000.mrc
    │       ├── eigen_neg001.mrc
    │       ├── ...
    │       ├── focus_mask.mrc
    │       ├── mask.mrc
    │       ├── mean_half1_unfil.mrc
    │       ├── mean_half2_unfil.mrc
    │       ├── mean.mrc
    │       ├── variance10.mrc
    │       ├── variance20.mrc
    │       └── variance4.mrc
    └── run.log
</details>


### Visualization in jupyter notebook

You can visualize the results using [this notebook](output_visualization.ipynb), which will show a bunch of results including:
* the FSC of the mean estimation, which you can interpret as an upper bound of the resolution you can expect. 
* decay of eigenvalues to help you pick the right `zdim`
* and standard clustering visualization (borrowed from the cryoDRGN output).

<!-- 
## VI. Generating additional trajectories

Usage example:

    python [recovar_dir]/compute_trajectory.py output-dir --zdim=4 --endpts test-new-mask/output/analysis_4/kmeans_40/centers.pkl --kmeans-ind=0,10 -o path_test

<details><summary><code>$ python compute_trajectory.py -h</code></summary>

    usage: compute_trajectory.py [-h] [-o OUTDIR] [--kmeans-ind KMEANS_IND] [--endpts ENDPTS_FILE] [--zdim ZDIM] [--q <class 'float'>]
                                [--n-vols <class 'int'>]
                                result_dir

    positional arguments:
    result_dir            result dir (output dir of pipeline)

    optional arguments:
    -h, --help            show this help message and exit
    -o OUTDIR, --outdir OUTDIR
                            Output directory to save model
    --kmeans-ind KMEANS_IND
                            indices of k means centers to use as endpoints
    --endpts ENDPTS_FILE  end points file. It storing z values, it should be a .txt file with 2 rows, and if it is from kmeans, it should be a .pkl file
                            (generated by analyze)
    --zdim ZDIM           Dimension of latent variable (a single int, not a list)
    --q <class 'float'>   quantile used for reweighting (default = 0.95)
    --n-vols <class 'int'>
                            number of volumes produced at regular interval along the path
</details> -->

## Small test dataset

If you want to make sure everything is installed properly, you can run the code in `run_test_dataset.sh`, which will generate a small dataset, run the pipeline and generate volumes.

You can do this by: `conda activate recovar` and then  `sh run_test_dataset.sh`. See below for what the script does

<details><summary>sh run_test_dataset.sh</summary>

    RECOVAR_PATH=./
    # Generate a small test dataset - should take about 30 sec
    python $RECOVAR_PATH/make_test_dataset.py

    # Run pipeline, should take about 2 min
    python $RECOVAR_PATH/pipeline.py test_dataset/particles.64.mrcs --poses test_dataset/poses.pkl --ctf test_dataset/ctf.pkl --correct-contrast --o test_dataset/pipeline_output --mask=from_halfmaps

    # Run on the 2D embedding with no regularization on latent space (better for density estimation)
    # Should take about 5 min
    python $RECOVAR_PATH/analyze.py test_dataset/pipeline_output --zdim=2 --no-z-regularization --n-clusters=3 --n-trajectories=0


    # You may want to delete this directory after running the test. 
    # rm -rf $RECOVAR_PATH/test_dataset

    ## One way to make sure everything went well is that the states in test_dataset/pipeline_output/output/analysis_2_noreg/centers/all_volumes should be similar to the simulated ones in recovar/data/vol*.mrc (the order doesn't matter, though)
</details>



## TLDR
 <!-- (WIP - Untested) -->
A short example illustrating the steps to run the code on EMPIAR-10076. Assuming you have downloaded the code and have a GPU, the code should take less than an hour to run, and less than 10 minutes if you downsample to 128 instead (exact running time depends on your hardware). and  Read above for more details:

    # Downloaded poses from here: https://github.com/zhonge/cryodrgn_empiar.git
    git clone https://github.com/zhonge/cryodrgn_empiar.git
    cd cryodrgn_empiar/empiar10076/inputs/

    # Download particles stack from here. https://www.ebi.ac.uk/empiar/EMPIAR-10076/ with your favorite method.
    # My method of choice is to use https://www.globus.org/ 
    # Move the data into cryodrgn_empiar/empiar10076/inputs/

    conda activate recovar
    # Downsample images to D=256
    cryodrgn downsample Parameters.star -D 256 -o particles.256.mrcs --chunk 50000

    # Extract pose and ctf information from cryoSPARC refinement
    cryodrgn parse_ctf_csparc cryosparc_P4_J33_004_particles.cs -o ctf.pkl
    cryodrgn parse_pose_csparc cryosparc_P4_J33_004_particles.cs -D 320 -o poses.pkl
    
    # run recovar
    python [recovar_dir]/pipeline.py particles.256.mrcs --ctf --ctf.pkl -poses poses.pkl --o recovar_test 

    # run analysis
    python [recovar_dir]/analysis.py recovar_test --zdim=20 

    # Open notebook output_visualization.ipynb
    # Change the recovar_result_dir = '[path_to_this_dir]/recovar_test' and 

Note that this is different from the one in the paper. Run the following pipeline command to get the one in the paper (runs on the filtered stack from the cryoDRGN paper, and uses a predefined mask):

    # Download mask
    git clone https://github.com/ma-gilles/recovar_masks.git

    python ~/recovar/pipeline.py particles.256.mrcs --ctf ctf.pkl --poses poses.pkl -o test-mask --mask recovar_masks/mask_10076.mrc --ind filtered.ind.pkl

The output should be the same as [this notebook](output_visualization_empiar10076.ipynb).

## Using kernel regression with other embeddings

You can generate volumes from embedding not generated by RECOVAR using `generate_from_embedding`. E.g., for a cryoDRGN embedding:

    python [recovar_dir/]generate_from_embedding.py particles.256.mrcs --poses poses.pkl --ctf ctf.pkl --embedding 02_cryodrgn256/z.24.pkl --o [output_dir] --target zfile.txt


<details><summary><code>$ python generate_from_embedding.py -h</code></summary>

    usage: generate_from_embedding.py [-h] -o OUTDIR [--zdim ZDIM] --poses POSES --ctf pkl [--ind PKL] [--uninvert-data UNINVERT_DATA]
                                    [--datadir DATADIR] [--n-images N_IMAGES] [--padding PADDING] [--halfsets HALFSETS]
                                    [--noise-model NOISE_MODEL] [--Bfactor BFACTOR] [--n-bins N_BINS] --embedding EMBEDDING --target
                                    TARGET [--zdim1]
                                    particles

    positional arguments:
    particles             Input particles (.mrcs, .star, .cs, or .txt)

    optional arguments:
    -h, --help            show this help message and exit
    -o OUTDIR, --outdir OUTDIR
                            Output directory to save model
    --zdim ZDIM           Dimensions of latent variable. Default=1,2,4,10,20
    --poses POSES         Image poses (.pkl)
    --ctf pkl             CTF parameters (.pkl)
    --Bfactor BFACTOR     0
    --n-bins N_BINS       number of bins for kernel regression
    --embedding EMBEDDING
                            Image embeddings zs (.pkl), e.g. 00_cryodrgn256/z.24.pkl if you want to use a cryoDRGN embedding.
    --target TARGET       Target zs to evaluate the kernel regression (.txt)
    --zdim1               Whether dimension 1 embedding is used. This is an annoying corner case for np.loadtxt...

    Dataset loading:
    --ind PKL             Filter particles by these indices
    --uninvert-data UNINVERT_DATA
                            Invert data sign: options: true, false (default)
    --datadir DATADIR     Path prefix to particle stack if loading relative paths from a .star or .cs file
    --n-images N_IMAGES   Number of images to use (should only use for quick run)
    --padding PADDING     Real-space padding
    --halfsets HALFSETS   Path to a file with indices of split dataset (.pkl).
    --noise-model NOISE_MODEL
                            what noise model to use. Options are radial (default) computed from outside the masks, and white computed by
                            power spectrum at high frequencies

</details>

## Using the source code

I hope some developers find parts of the code useful for their projects. See [this notebook](recovar_coding_tutorial.ipynb) for a short tutorial. (OUT OF DATE, see [cryoJAX](https://github.com/mjo22/cryojax) for a much better documented JAX cryo-EM code.)

Some of the features which may be of interest:
- The basic building block operations of cryo-EM efficiently, in batch and on GPU: shift images, slice volumes, do the adjoint slicing, apply CTF. See [recovar.core](recovar/core.py). Though I have not tried it, all of these operations should be differentiable thus you could use JAX's autodiff.

- A heterogeneity dataset simulator that includes variations in contrast, realistic CTF and pose distribution (loaded from real dataset), junk proteins, outliers, etc. See [recovar.simulator](recovar/simulator.py).


- A code to go from atomic positions to volumes or images. Does not run on GPU. Thanks to [prody](http://prody.csb.pitt.edu/), if you have an internet connection, you can generate the volume from only the PDB ID. E.g., you can do `recovar.simulate_scattering_potential.generate_molecule_spectrum_from_pdb_id('6VXX', 2, 256)` to generate the volume of the spike protein with voxel size 2 on a 256^3 grid. Note that this code exactly evaluates the Fourier transform of the potential, thus it is exact in Fourier space, which can produce some oscillations artifact in the spatial domain. Also see [cryoJAX](https://github.com/mjo22/cryojax)

- Some other features that aren't very well separated from the rest of the code, but I think could easily be stripped out: trajectory computation [recovar.trajectory](recovar/trajectory.py), per-image mask generation [recovar.covariance_core](recovar/covariance_core.py), regularization schemes [recovar.regularization](recovar/regularization.py), various noise estimators [recovar.noise](recovar/noise.py).

- Some features that are not there (but soon, hopefully): pose search, symmetry handling.

## Cryo-ET

I am developping the tomography extension. You can try it out by passing the the ``--tilt-series``, ``--tilt-series-ctf=v2``, ``--angle-per-tilt=[insert angle here]`` to ``pipeline.py``. The input should be identical to cryoDRGN-ET ([see here](https://ez-lab.gitbook.io/cryodrgn/cryodrgn-et-subtomogram-analysis)).

E.g, on the M-file:

    $ python [recovar_dir]/pipeline.py M_particles.star --ctf ctf.pkl --poses pose.pkl -o v2_nocont_$ntilts --datadir=128 --mask=path_to_mask.mrc --tilt-series-ctf=v2  --ntilts=10 --tilt-series  --angle-per-tilt=3.0 --dose-per-tilt=2.93

You can use all tilts by not passing the argument --ntilts.

## Limitations

- *Symmetry*: there is currently no support for symmetry. If you got your poses through symmetric refinement, it will probably not work. It should probably work if you make a symmetry expansion of the particle stack, but I have not tested it.
- *Memory*: RECOVAR uses a lot of memory by default. For a stack of images of size 256, you need approximately 200 GB + size of dataset. You can also use the --lazy option, which will do lazy loading, in which case you need a little more than 200. If you run out of memory, you can use the --low-memory-option, in which case you need 60GB. If even that fails, you can try --very-low-memory-option. Finally, you can downsample the data and run RECOVAR.  
- *ignore-zero-frequency*: I haven't thought much about the best way to do this. I would advise against using it for now.
- *Other ones, probably?*: if you run into issues, please let me know. 


## Citation

If you use this software for analysis, please cite:

    @article{gilles2023bayesian,
      title={A Bayesian Framework for Cryo-EM Heterogeneity Analysis using Regularized Covariance Estimation},
      author={Gilles, Marc Aurele T and Singer, Amit},
      journal={bioRxiv},
      pages={2023--10},
      year={2023},
      publisher={Cold Spring Harbor Laboratory}
    }

## Contact

You can reach me (Marc) at mg6942@princeton.edu with questions or comments.


## Other relevant github repos:

* [ASPIRE](https://github.com/ComputationalCryoEM/ASPIRE-Python)
* [cryoDRGN](https://github.com/ml-struct-bio/cryodrgn)
* [cryoJAX](https://github.com/mjo22/cryojax) 
* [Dynamight](https://github.com/3dem/DynaMight)
* [Flexibility Hub](https://github.com/scipion-em/scipion-em-continuousflex)
* [RELION](https://github.com/3dem/relion)

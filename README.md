# RECOVAR: Regularized covariance estimation for cryo-EM heterogeneity analysis

[Installation](#installation)

Running RECOVAR:
* [1. Preprocessing](#i-preprocessing)
* [2. Specifying a mask](#ii-specifying-a-real-space-mask)
* [3. Running the pipeline](#iii-running-recovar-pipeline)
* [4. Analyzing results](#iv-analyzing-results)
* [5. Visualizing results](#v-visualizing-results)
* [6. Generating trajectories](#vi-generating-additional-trajectories)

[TLDR](#tldr)

Peak at what output looks like on a [synthetic dataset](output_visualization_synthetic.ipynb) and [real dataset](output_visualization_empiar10076.ipynb).

Also:
[using the source code](#using-the-source-code), 
[limitations](#limitations), 
[contact](#contact)

## Installation 
CUDA and [JAX](https://jax.readthedocs.io/en/latest/index.html#) are required to run this code. See information about JAX installation [here](https://jax.readthedocs.io/en/latest/installation.html).

Here is a set of commands which runs on our university cluster (Della), but may need to be tweaked to run on other clusters.

    # module load cudatoolkit/12.2 # You need to load or install CUDA before installing JAX
    conda create --name recovar python=3.9
    conda activate recovar
    pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html # You may need to pass ax[cuda11_pip] if you use cuda v11
    pip install cryodrgn mrcfile scikit-fmm prody finufft scikit-image tensorflow-cpu matplotlib-scalebar dataframe-image umap-learn[plot] sklearn
    git clone https://github.com/ma-gilles/recovar.git
    python -m ipykernel install --user --name=recovar # if you want to use jupyter notebooks


[Jump to TLDR](#tldr)

[See what output format looks like](simple_heterogeneity_output_example.ipynb)


## I. Preprocessing 

The input layer of RECOVAR is borrowed directly from the excellent [cryoDRGN toolbox](https://cryodrgn.cs.princeton.edu/). 
Particles, poses and CTF must be prepared in the same way, and below is copy-pasted part of 
[cryoDRGN's documentation](https://github.com/ml-struct-bio/cryodrgn#2-parse-image-poses-from-a-consensus-homogeneous-reconstructiqqon).
CryoDRGN is a dependency, so you should be able to run the commands below after ``conda activate recovar``.

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

The input file format can be a single `.mrcs` file, a `.txt` file containing paths to multiple `.mrcs` files, a RELION `.star` file, or a cryoSPARC `.cs` file. For the latter two options, if the relative paths to the `.mrcs` are broken, the argument `--datadir` can be used to supply the path to where the `.mrcs` files are located.

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

A real space mask is important to boost SNR. Most consensus reconstruction software output a mask, which you can use as input (`--mask-option=input`). Make sure the mask is not too tight, you can use the input `--dilate-mask-iter` to expand the mask if needed. You may also want to use a focusing mask to focus on heterogeneity on one part of the protein, [click here](https://guide.cryosparc.com/processing-data/tutorials-and-case-studies/mask-selection-and-generation-in-ucsf-chimera) to find instructions to generate one with Chimera.

If you don't input a mask, the software will estimate one using the two halfmap means ( `--mask-option=from-halfmaps`). You may also want to run with a loose spherical mask (option `--mask-option=sphere`) and use the computed variance map to observe which parts have large variance.


## III. Running RECOVAR pipeline

When the input images (.mrcs), poses (.pkl), and CTF parameters (.pkl) have been prepared, RECOVAR can be run with following command:

    $ python [recovar_directory]/pipeline.py particles.128.mrcs -o output_test --ctf ctf.pkl --poses poses.pkl


<details><summary><code>$ python pipeline.py -h</code></summary>

    usage: pipeline.py [-h] -o OUTDIR [--zdim ZDIM] --poses POSES --ctf pkl [--mask mrc] [--mask-option <class 'str'>] [--mask-dilate-iter MASK_DILATE_ITER]
                    [--correct-contrast] [--ind PKL] [--uninvert-data UNINVERT_DATA] [--datadir DATADIR] [--n-images N_IMAGES] [--padding PADDING]
                        [--halfsets HALFSETS]
                        particles

    positional arguments:
    particles             Input particles (.mrcs, .star, .cs, or .txt)

    optional arguments:
    -h, --help            show this help message and exit
    -o OUTDIR, --outdir OUTDIR
                            Output directory to save model
    --zdim ZDIM           Dimension of latent variable
    --poses POSES         Image poses (.pkl)
    --ctf pkl             CTF parameters (.pkl)
    --mask mrc            mask (.mrc)
    --mask-option <class 'str'>
                            mask options: from_halfmaps (default), input, sphere, none
    --mask-dilate-iter MASK_DILATE_ITER
                            mask options how many iters to dilate input mask (only used for input mask)
    --correct-contrast    estimate and correct for amplitude scaling (contrast) variation across images

    Dataset loading:
    --ind PKL             Filter particles by these indices
    --uninvert-data UNINVERT_DATA
                            Invert data sign: options: true, false, automatic (default). automatic will swap signs if sum(estimated mean) < 0
    --datadir DATADIR     Path prefix to particle stack if loading relative paths from a .star or .cs file
    --n-images N_IMAGES   Number of images to use (should only use for quick run)
    --padding PADDING     Real-space padding
    --halfsets HALFSETS   Path to a file with indices of split dataset (.pkl).
</details>


The required arguments are:

* an input image stack (`.mrcs` or other listed file types)
* `--poses`, image poses (`.pkl`) that correspond to the input images
* `--ctf`, ctf parameters (`.pkl`), unless phase-flipped images are used
* `-o`, a clean output directory for saving results

Additional parameters which are typically set include:
* `--zdim`, dimensions of PCA to use for embedding, can submit one integer (`--zdim=20`) or a or a command separated list (`--zdim=10,50,100`). Default (`--zdim=4,10,20`).
* `--mask-option` to specify which mask to use
* `--mask` to specify the mask path (`.mrc`)
* `--dilate-mask-iter` to specify a number of dilation of mask
<!-- * `--uninvert-data`, Use if particles are dark on light (negative stain format) -->


## IV. Analyzing results

After the pipeline is run, you can find the mean, eigenvectors, variance maps and embeddings in the `outdir/results` directory, where outdir the option given above by `-o`. You can run some standard analysis by running 

    python analyze.py [outdir] --zdim=10

It will run k-means, generate volumes corresponding to the centers, generate trajectories between pairs of cluster centers and run UMAP. See more input details below.


<details><summary><code>$ python analyze.py -h</code></summary>

    usage: python analyze.py [-h] [-o OUTDIR] [--zdim ZDIM] [--n-clusters <class 'int'>]
                            [--n-trajectories N_TRAJECTORIES] [--skip-umap] [--q <class 'float'>]
                            [--n-std <class 'float'>]
                            result_dir

    positional arguments:
    result_dir            result dir (output dir of pipeline)

    optional arguments:
    -h, --help            show this help message and exit
    -o OUTDIR, --outdir OUTDIR
                            Output directory to save model
    --zdim ZDIM           Dimension of latent variable (a single int, not a list)
    --n-clusters <class 'int'>
                            mask options: from_halfmaps (default), input, sphere, none
    --n-trajectories N_TRAJECTORIES
                            how many trajectories to compute between k-means clusters
    --skip-umap           whether to skip u-map embedding (can be slow for large dataset)
    --q <class 'float'>   quantile used for reweighting (default = 0.95)
    --n-std <class 'float'>
                            number of standard deviations to use for reweighting (don't set q and this
                            parameter, only one of them)

</details>

## V. Visualizing results
### Output structure

Assuming you have run the pipeline.py and analyze.py, the output will be saved in the format below (click on the arrow). If you are running on a remote server, I would advise you only copy the [output_dir]/output locally, since the model file will be huge. You can then visualize volumes in ChimeraX.


<details><summary>Output file structure</summary>


    [output_dir]
    ├── model
    │   ├── halfsets.pkl # indices of half sets
    │   └── results.pkl # all results, should only be used by analye.py
    ├── output
    │   ├── analysis_4
    │   │   ├── kmeans_40
    │   │   │   ├── centers_01no_annotate.png
    │   │   │   ├── centers_01.png
    │   │   │   ├── ...
    │   │   │   ├── centers.pkl # centers in zformat
    │   │   │   ├── path0
    │   │   │   │   ├── density
    │   │   │   │   │   ├── density_01.png
    │   │   │   │   │   └── ...
    │   │   │   │   ├── density_01.png
    │   │   │   │   ├── ...
    │   │   │   │   ├── path_density_t.png
    │   │   │   │   ├── path.json
    │   │   │   │   ├── reweight_000.mrc # reweighted reconstructions
    │   │   │   │   ├── ...
    │   │   │   │   ├── reweight_halfmap0_000.mrc # halfmaps for each reconstruction
    │   │   │   │   └── ...
    │   │   │   ├── path1
    │   │   │   │   ├── ...
    │   │   │   ├── reweight_000.mrc # volume reconstruction of kmeans 
    │   │   │   ├── ...
    │   │   │   ├── reweight_halfmap1_039.mrc # also halfmaps
    │   │   │   └── trajectory_endpoints.pkl # end points of trajectory used 
    │   │   └── umap_embedding.pkl 
    │   └── volumes
    │       ├── dilated_mask.mrc
    │       ├── eigen_neg000.mrc # Eigenvectors
    │       ├── ...
    │       ├── eigen_pos000.mrc # Negative of eigenvectors. Useful to Chimera visualization to have the two separated, although they clearly contain the same observation
    │       ├── ...
    │       ├── mask.mrc # mask used 
    │       ├── mean.mrc # computed mean
    │       ├── variance10.mrc # compute variance from rank 10 approximation
    │       └── ...
    └── run.log
</details>


### Visualization in jupyter notebook

You can visualize the results using [this notebook](output_visualization.ipynb), which will show a bunch of results including 
* the FSC of the mean estimation to give you an on an upper bound of the resolution you can expect
* decay of eigenvalues to help you pick the right `zdim`
*  and standard clustering visualization (borrowed from the cryoDRGN output).


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
</details>



## TLDR
 (WIP - Untested)
A very short example illustrating the steps to run the code on EMPIAR-10076. Read above for more details:

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

Note that this isn't exactly the one in the paper. Run this analyze command to the one in the paper (runs on filtered stack from cryoDRGN, and uses a predefined mask):

    # Download mask
    git clone https://github.com/ma-gilles/recovar_masks.git

    python ~/recovar/pipeline.py particles.256.mrcs --ctf ctf.pkl --poses poses.pkl -o test-mask --mask recovar_masks/mask_10076.mrc --ind filtered.ind.pkl


## Using the source code

I hope some developpers may find parts of the code useful for their own projects. See [this notebook](recovar_coding_tutorial.ipynb) for a short tutorial.



## Limitations

- *Symmetry*: there is currently no support for symmetry. If you got your poses through symmetric refinement, it will probably not work. If you make a symmetry expansion of the particles stack, it should probably work but I have not tested it.
* *Memory*: you need a lot of memory to run this. For a stack of images of size 256, you probably need 400 GB+.
- *Other ones, probably?*

## Contact

You can reach me (Marc) at mg6942@princeton.edu with questions or comments.
# recovar.data_io

Dataset loading, metadata extraction, and image access for cryo-EM and
cryo-ET data.

## dataset

Core dataset classes and loading functions.

::: recovar.data_io.dataset
    options:
      members_order: source

## cryo_dataset

Google Grain-based dataset backend for efficient batched loading.

::: recovar.data_io.cryo_dataset
    options:
      members_order: source

## metadata_parsing

Extract poses and CTF parameters from RELION `.star` and cryoSPARC `.cs` files.

::: recovar.data_io.metadata_parsing
    options:
      members_order: source

## starfile

RELION `.star` file reading and writing.

::: recovar.data_io.starfile
    options:
      members_order: source

## load_utils

CTF and pose loading utilities (legacy pickle format).

::: recovar.data_io.load_utils
    options:
      members_order: source

## image_loader

Image loading from MRC/MRCS stacks and HDF5 files.

::: recovar.data_io.image_loader
    options:
      members_order: source

## tilt_dataset

Backward-compatible tilt-series dataset wrapper.

::: recovar.data_io.tilt_dataset
    options:
      members_order: source

# recovar.data_io

Dataset loading, metadata extraction, and image access for cryo-EM and
cryo-ET data.

## Flow

```mermaid
flowchart TD
    A[CLI / pipeline args] --> B[halfsets.py<br/>split policy + halfset loading]
    A --> C[cryoem_dataset.py<br/>load_dataset(...)]
    C --> D[image_sources.py<br/>image source assembly]
    C --> E[metadata_readers.py<br/>STAR / CS metadata parsing]
    D --> F[image_backends.py<br/>file-backed SPA / cryo-ET loaders]
    F --> G[image_loader.py<br/>MRC / MRCS / HDF5 I/O]
    E --> H[image_metadata.py<br/>ImageMetadata]
    D --> I[_index_utils.py<br/>image/group remapping]
    B --> I
    D --> J[cryoem_dataset.py<br/>CryoEMDataset]
    H --> J
    I --> J
    B --> J
    J --> K[iter_batches(...)]
    K --> L[pipeline / compute_state / analyze]
```

```text
SPA / cryo-ET load path
  -> cryoem_dataset.load_dataset(...)
       -> image_sources.create_image_source(...)
            -> image_backends.py
            -> image_loader.py
       -> metadata_readers.auto_parse_poses / auto_parse_ctf
            -> image_metadata.ImageMetadata
       -> CryoEMDataset(...)

Halfset / subset path
  -> halfsets.get_split_indices / get_split_tilt_indices
  -> halfsets.load_halfset_dataset / load_halfset_dataset_from_args
  -> CryoEMDataset with halfset_indices

Downstream runtime path
  -> CryoEMDataset.iter_batches(...)
  -> explicit tuples:
     (images, rotation_matrices, translations, ctf_params, noise_variance, particle_indices, image_indices)

Cross-cutting indexing
  _index_utils.py
    - DatasetIndexLayout: local <-> original image/group ids
    - TiltSeriesOriginalIndexMap: particle <-> image ids in the original file
```

Keep these responsibilities separate:

- `image_sources.py` owns raw image access, lazy/eager loading, and subset views.
- `image_metadata.py` owns rotations, translations, and CTF rows.
- `cryoem_dataset.py` is the only high-level coordinator and batch iterator surface.
- `halfsets.py` owns split policy and halfset bookkeeping.
- `_index_utils.py` owns image/group/particle remapping logic.
- `image_backends.py` owns only the low-level stack and tilt-series loaders used underneath image sources.

Public surface used by the main runtime:

- `cryoem_dataset.load_dataset`
- `halfsets.load_halfset_dataset`
- `halfsets.load_halfset_dataset_from_args`
- `CryoEMDataset.iter_batches`
- `CryoEMDataset.subset`

## cryoem_dataset

Core dataset classes and loading functions.

::: recovar.data_io.cryoem_dataset
    options:
      members_order: source

## image_backends

Low-level Grain-backed image backends.

::: recovar.data_io.image_backends
    options:
      members_order: source

## image_sources

Raw image loading abstraction and subset/image-group remapping.

::: recovar.data_io.image_sources
    options:
      members_order: source

## image_metadata

Typed metadata container for poses and CTF rows.

::: recovar.data_io.image_metadata
    options:
      members_order: source

## halfsets

Halfset and split logic for SPA and cryo-ET.

::: recovar.data_io.halfsets
    options:
      members_order: source

## _index_utils

Canonical local/original image, group, and particle index mapping helpers.

::: recovar.data_io._index_utils
    options:
      members_order: source

## metadata_readers

Extract poses and CTF parameters from RELION `.star` and cryoSPARC `.cs` files.

::: recovar.data_io.metadata_readers
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

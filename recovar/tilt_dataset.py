import numpy as np
from collections import Counter, OrderedDict
import logging
import torch
from torch.utils import data
from typing import Optional, Tuple, Union
# from cryodrgn import fft, starfile

from recovar.cryodrgn_source import ImageSource
# from cryodrgn.utils import window_mask
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler
from recovar import mask
from recovar import cryodrgn_starfile as starfile
import time
logger = logging.getLogger(__name__)


# A tilt is a single image (more commonly called a subtilt) 
# A particle is a collection of tilts, supposedly of the same object, that form a tilt series (typically 10-100)
# A tomogram tilt is single big image. Tomogram tilts are never loaded or used directly in this code, but it is occasionally important to know which collection of tilt come from the same tomogram tilt
# A tomogram is a collection of tomogram tilt (typically 10-100)

## Very awkward... Probably should just move things to one file...
def set_standard_mask(D, dtype):
    return mask.window_mask(D, 0.85, 0.99)


class ImageDataset(data.Dataset):
    def __init__(
        self,
        mrcfile,
        lazy=True,
        norm=None,
        keepreal=False,
        invert_data=False,
        ind=None,
        window=True,
        datadir=None,
        window_r=0.85,
        max_threads=16,
        padding=0,
        device: Union[str, torch.device] = "cpu",
    ):
        assert padding == 0, "Padding not implemented yet"
        assert not keepreal, "Not implemented yet"
        datadir = datadir or ""
        self.ind = ind
        self.src = ImageSource.from_file(
            mrcfile,
            lazy=lazy,
            datadir=datadir,
            indices=ind,
            max_threads=max_threads,
        )

        ny = self.src.D
        assert ny % 2 == 0, "Image size must be even."

        self.N = self.src.n
        self.n_images = self.N
        self.D = ny #+ 1  # after symmetrization
        self.invert_data = invert_data
        self.window = None#window_mask(ny, window_r, 0.99).to(device) if window else None
        # norm = norm or self.estimate_normalization()
        # self.norm = [float(x) for x in norm]
        self.device = device
        self.lazy = lazy


        ## Added on top?
        self.dtype = np.complex64 # ???
        self.unpadded_D = ny
        self.unpadded_image_shape = (ny, ny)
        self.image_shape = (ny, ny)
        self.image_size = ny * ny
        self.mask = jnp.array(set_standard_mask(self.unpadded_D, self.dtype))
        self.mult = -1 if invert_data else 1
        self.padding=0

    def process_images(self, images, apply_image_mask = False):
        if apply_image_mask:
            images = images * self.mask
        # logger.warning("CHANGE BACK USE MASK TO FALSE")
        import recovar.padding as pad
        images = pad.padded_dft(images * self.mult,  self.D, self.padding)
        return images.astype(self.dtype)


    def __len__(self):
        return self.N

    def __getitem__(self, index):
        particles = self.src.images(index)
        # this is why it is tricky for index to be allowed to be a list!
        if len(particles.shape) == 2:
            particles = particles[np.newaxis, ...]

        if isinstance(index, (int, np.integer)):
            logger.debug(f"ImageDataset returning images at index ({index})")
        else:
            logger.debug(
                f"ImageDataset returning images for {len(index)} indices ({index[0]}..{index[-1]})"
            )

        return particles, index, index

    def get_slice(
        self, start: int, stop: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return (
            self.src.images(slice(start, stop), require_contiguous=True).numpy(),
            None,
        )

    def get_dataset_generator(self, batch_size, num_workers = 0):
        return NumpyLoader(self, batch_size=batch_size, shuffle=False, num_workers = num_workers)
    
    def get_dataset_subset_generator(self, batch_size, subset_indices, num_workers = 0):
        if subset_indices is None:
            subset_indices = list(range(len(self)))
        return NumpyLoader(torch.utils.data.Subset(self, subset_indices), batch_size=batch_size, shuffle=False, num_workers = num_workers)
        # torch.utils.data.Subset(self, subset_indices)


class TiltSeriesData(ImageDataset):
    """
    Class representing tilt series
    """

    def __init__(
        self,
        tiltstar,
        lazy = True,
        ntilts=None,
        random_tilts=False,
        ind=None,
        voltage=None,
        expected_res=None,
        dose_per_tilt=None,
        angle_per_tilt=None,
        sort_with_Bfac=True,
        **kwargs,
    ):
        self._load_start_time = time.time()
        # Note: ind is the indices of the *tilts*, not the particles
        super().__init__(tiltstar, ind=ind, **kwargs)
        # Parse unique particles from _rlnGroupName
        elapsed = time.time() - self._load_start_time
        print(f"Tilt series loaded in {elapsed:.2f} seconds")

        s = starfile.Starfile.load(tiltstar)
        elapsed = time.time() - self._load_start_time
        print(f"Tilt series loaded in {elapsed:.2f} seconds")

        unique_sorted_group_names = []
        unique_sorted_group_names = list(s.df["_rlnGroupName"].unique())
        # for name in s.df["_rlnGroupName"]:
        #     if name not in unique_sorted_group_names:
        #         unique_sorted_group_names.append(name)

        if ind is not None:
            s.df = s.df.loc[ind]
            
        group_name = list(s.df["_rlnGroupName"])

        particles = OrderedDict()
        self.dataset_tilt_indices = []
        # groups = s.df.groupby("_rlnGroupName").indices
        # particles = [np.array(idx, dtype=int) for idx in groups.values()]
        # self.dataset_tilt_indices = [unique_sorted_group_names.index(gn) for gn in groups.keys()]
        for ii, gn in enumerate(group_name):
            if gn not in particles:
                particles[gn] = []
                self.dataset_tilt_indices.append(unique_sorted_group_names.index(gn))
            particles[gn].append(ii)

        self.particles = [np.asarray(pp, dtype=int) for pp in particles.values()]
        self.Np = len(particles)
        self.ctfscalefactor = np.asarray(s.df["_rlnCtfScalefactor"], dtype=np.float32)

        if '_rlnCtfBfactor' in s.df.columns:
            self.ctfBfactor = np.asarray(s.df["_rlnCtfBfactor"], dtype=np.float32)

        self.tilt_numbers = np.zeros(self.N)
        if sort_with_Bfac:
            logger.info("Sorting tilt series with Bfactor")
        else:
            logger.info("Sorting tilt series with scale factor!")
            logger.warning("Sorting tilt series with scale factor!! May be wrong.")
        for ind in self.particles:
            if sort_with_Bfac:
                sort_idxs = self.ctfBfactor[ind].argsort()
                # logger.info("Sorting with Bfactor")
            else:
                sort_idxs = self.ctfscalefactor[ind].argsort()
                # logger.info("Sorting with ctf scale factor")

            ranks = np.empty_like(sort_idxs)
            ranks[sort_idxs[::-1]] = np.arange(len(ind))
            self.tilt_numbers[ind] = ranks

        # self.tilt_numbers = torch.tensor(self.tilt_numbers).to(self.device)
        logger.info(f"Loaded {self.N} tilts for {self.Np} particles")
        counts = Counter(group_name)
        unique_counts = set(counts.values())
        logger.info(f"{unique_counts} tilts per particle")
        self.counts = counts
        if ntilts is not None:
            assert ntilts <= min(unique_counts)
        self.ntilts = ntilts
        self.random_tilts = random_tilts

        self.voltage = voltage
        self.dose_per_tilt = dose_per_tilt

        # Assumes dose-symmetric tilt scheme
        # As implemented in Hagen, Wan, Briggs J. Struct. Biol. 2017
        self.tilt_angles = None
        # if angle_per_tilt is not None:
            # self.tilt_angles = angle_per_tilt * torch.ceil(self.tilt_numbers / 2)
            # self.tilt_angles = torch.tensor(self.tilt_angles).to(self.device)
        elapsed = time.time() - self._load_start_time
        logger.info(f"Tilt series loaded in {elapsed:.2f} seconds")

    def __len__(self):
        return self.Np

    def __getitem__(self, index):
        # if isinstance(index, list):
        #     index = torch.Tensor(index).to(torch.long)
        tilt_indices = []
        for ii in [index]:
            if self.random_tilts:
                tilt_index = np.random.choice(
                    self.particles[ii], self.ntilts, replace=False
                )
            else:
                # take the first ntilts
                tilt_index = self.particles[ii][0: self.ntilts]
            tilt_indices.append(tilt_index)

        tilt_indices = np.concatenate(tilt_indices)
        images = self.src.images(tilt_indices)
        return images, index, tilt_indices


    @classmethod
    def parse_particle_tilt(
        cls, tiltstar: str
    ) -> Tuple[list[np.ndarray], dict[int, int]]:
        # Parse unique particles from _rlnGroupName
        s = starfile.Starfile.load(tiltstar)
        group_name = list(s.df["_rlnGroupName"])
        particles = OrderedDict()

        for ii, gn in enumerate(group_name):
            particles.setdefault(gn, []).append(ii)
            # if gn not in particles:
            #     particles[gn] = []
            # particles[gn].append(ii)

        particles = [np.asarray(pp, dtype=int) for pp in particles.values()]
        particles_to_tilts = particles
        tilts_to_particles = {}

        for i, j in enumerate(particles):
            for jj in j:
                tilts_to_particles[jj] = i

        return particles_to_tilts, tilts_to_particles


    @classmethod
    def parse_tomogram_tilt(
        cls, tiltstar: str
    ) -> Tuple[list[np.ndarray], dict[int, int]]:
        # Parse unique particles from _rlnGroupName
        s = starfile.Starfile.load(tiltstar)
        group_name = list(s.df["_rlnGroupName"])
        tomogram = OrderedDict()
        for ii, gn in enumerate(group_name):
            tomogram.setdefault(gn, []).append(ii)

        tomogram = [np.asarray(pp, dtype=int) for pp in tomogram.values()]
        tomogram_to_tilts = tomogram
        tilts_to_tomogram = {}

        for i, j in enumerate(tomogram):
            for jj in j:
                tilts_to_tomogram[jj] = i

        return tomogram_to_tilts, tilts_to_tomogram


    @classmethod
    def parse_tomogramtilt_tilt(
        cls, tiltstar: str
    ) -> Tuple[list[np.ndarray], dict[int, int]]:
        # Parse unique particles from _rlnGroupName
        s = starfile.Starfile.load(tiltstar)
        group_name = list(s.df["_rlnGroupName"])
        dose = list(s.df['_rlnCtfBfactor'])
        groups = OrderedDict()
        for i, (gn, d) in enumerate(zip(group_name, dose)):
            key = (gn, d)
            groups.setdefault(key, []).append(i)

        groups = [np.asarray(pp, dtype=int) for pp in groups.values()]
        tomogramtilts_to_tilts = groups
        tilts_to_tomogramtilts = {}

        for i, j in enumerate(groups):
            for jj in j:
                tilts_to_tomogramtilts[jj] = i

        return tomogramtilts_to_tilts, tilts_to_tomogramtilts



    @classmethod
    def particles_to_tilts(
        cls, particles_to_tilts: list[np.ndarray], particles: np.ndarray
    ) -> np.ndarray:
        tilts = [particles_to_tilts[int(i)] for i in particles]
        tilts = np.concatenate(tilts)
        return tilts

    @classmethod
    def tilts_to_particles(cls, tilts_to_particles, tilts):
        particles = [tilts_to_particles[i] for i in tilts]
        particles = np.array(sorted(set(particles)))
        return particles

    def get_tilt(self, index):
        return super().__getitem__(index)

    def get_slice(self, start: int, stop: int) -> Tuple[np.ndarray, np.ndarray]:
        # we have to fetch all the tilts to stay contiguous, and then subset
        tilt_indices = [self.particles[index] for index in range(start, stop)]
        cat_tilt_indices = np.concatenate(tilt_indices)
        images = self.src.images(cat_tilt_indices, require_contiguous=True)

        tilt_masks = []
        for tilt_idx in tilt_indices:
            tilt_mask = np.zeros(len(tilt_idx), dtype=bool)
            if self.random_tilts:
                tilt_mask_idx = np.random.choice(
                    len(tilt_idx), self.ntilts, replace=False
                )
                tilt_mask[tilt_mask_idx] = True
            else:
                # if self.ntilts == -1:
                #     self.ntilts = len(tilt_idx)
                i = 0#(len(tilt_idx) - self.ntilts) // 2
                # if self.n_tilts == -1:
                #     title_mask = np.ones(len(tilt_idx), dtype=bool)
                # else:
                tilt_mask[i: i + self.ntilts] = True
            tilt_masks.append(tilt_mask)

        tilt_mask = np.concatenate(tilt_masks)
        return images.numpy(), tilt_mask

    def get_dataset_generator(self, batch_size, num_workers = 0):
        return make_dataloader(self, batch_size=batch_size, num_workers=num_workers)

    def get_dataset_subset_generator(self, batch_size, subset_indices, num_workers = 0):
        if subset_indices is None:
            return self.get_dataset_generator(batch_size, num_workers = 0)
        return make_dataloader(torch.utils.data.Subset(self, subset_indices), batch_size=batch_size, num_workers=num_workers)

    def get_image_generator(self, batch_size, num_workers=0):
        # This generator iterates over individual images rather than tilt groups.
        class SingleImageDataset(torch.utils.data.Dataset):
            def __init__(self, src):
                self.src = src

            def __len__(self):
                return self.src.n

            def __getitem__(self, index):
                # Fetch a single image. If the image is 2D, add a new axis.
                image = self.src.images(index)
                if image.ndim == 2:
                    image = image[np.newaxis, ...]
                return image, np.inf, index

        return NumpyLoader(SingleImageDataset(self.src), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
    def get_image_subset_generator(self, batch_size, subset_indices, num_workers=0):
        # This generator iterates over individual images rather than tilt groups.
        class SingleImageDataset(torch.utils.data.Dataset):
            def __init__(self, src):
                self.src = src

            def __len__(self):
                return self.src.n

            def __getitem__(self, index):
                # Ensure index is a scalar
                if isinstance(index, (list, tuple, np.ndarray)):
                    index = index[0]
                image = self.src.images(index)
                if image.ndim == 2:
                    image = image[np.newaxis, ...]
                return image, np.inf, index
        
        if subset_indices is None:
            return self.get_image_generator(batch_size, num_workers)
        
        # Convert subset_indices to numpy array and ensure it's 1D
        if not isinstance(subset_indices, np.ndarray):
            subset_indices = np.array(subset_indices)
        if subset_indices.ndim == 0:
            subset_indices = np.array([subset_indices])
        elif subset_indices.ndim > 1:
            subset_indices = subset_indices.flatten()
        
        # Create dataset and dataloader
        dataset = SingleImageDataset(self.src)
        subset_dataset = torch.utils.data.Subset(dataset, subset_indices)
        
        # Print debug info
        # print(f"Dataset length: {len(dataset)}")
        # print(f"Subset indices shape: {subset_indices.shape}")
        # print(f"Subset dataset length: {len(subset_dataset)}")
        
        return NumpyLoader(subset_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def make_dataloader(
    data: ImageDataset,
    *,
    batch_size: int,
    num_workers: int = 0,
    shuffler_size: int = 0,
    shuffle=False,
):
    if shuffler_size > 0 and shuffle:
        assert data.lazy, "Only enable a data shuffler for lazy loading"
        return DataShuffler(data, batch_size=batch_size, buffer_size=shuffler_size)
    else:
        # see https://github.com/zhonge/cryodrgn/pull/221#discussion_r1120711123
        # for discussion of why we use BatchSampler, etc.
        sampler_cls = RandomSampler if shuffle else SequentialSampler
        batch_size=1
        return NumpyLoader(data, batch_size=batch_size, shuffle=False, num_workers = num_workers)

import jax.numpy as jnp

def numpy_collate(batch):
  if isinstance(batch[0], np.ndarray):
    return jnp.concatenate(batch, axis=0)
  elif isinstance(batch[0], (tuple,list)):
    transposed = zip(*batch)
    return [numpy_collate(samples) for samples in transposed]
  elif batch is None:
      return None
  else:
    return jnp.array(batch) 

class NumpyLoader(torch.utils.data.DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)


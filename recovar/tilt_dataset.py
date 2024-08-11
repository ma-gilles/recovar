
import numpy as np
from collections import Counter, OrderedDict
import logging
import torch
from torch.utils import data
from typing import Optional, Tuple, Union
from cryodrgn import fft, starfile
from recovar.cryodrgn_source import ImageSource
# from cryodrgn.utils import window_mask
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler
from recovar import mask

logger = logging.getLogger(__name__)

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
        # self.mask = np.ones(self.image_shape, dtype = np.float32)





    # def estimate_normalization(self, n=1000):
    #     n = min(n, self.N) if n is not None else self.N
    #     indices = range(0, self.N, self.N // n)  # FIXME: what if the data is not IID??
    #     imgs = self.src.images(indices)

    #     particleslist = []
    #     for img in imgs:
    #         particleslist.append(fft.ht2_center(img))
    #     imgs = torch.stack(particleslist)

    #     if self.invert_data:
    #         imgs *= -1

    #     imgs = fft.symmetrize_ht(imgs)
    #     norm = (0, torch.std(imgs))
    #     logger.info("Normalizing HT by {} +/- {}".format(*norm))
    #     return norm

    # def _process(self, data):
    #     if data.ndim == 2:
    #         data = data[np.newaxis, ...]
    #     if self.window is not None:
    #         data *= self.window
    #     data = fft.ht2_center(data)
    #     if self.invert_data:
    #         data *= -1
    #     data = fft.symmetrize_ht(data)
    #     # data = (data - self.norm[0]) / self.norm[1]
    #     images = ftu.get_dft2(images)
    #     images = pad.padded_dft(images * self.mult,  self.image_size, self.padding)

    #     return data

    def process_images(self, images, apply_image_mask = False):
        if apply_image_mask:
            images = images * self.mask
        # logger.warning("CHANGE BACK USE MASK TO FALSE")
        import recovar.padding as pad
        images = pad.padded_dft(images * self.mult,  self.D, self.padding)
        return images


    def __len__(self):
        return self.N

    def __getitem__(self, index):
        # if isinstance(index, list):
        #     index = torch.Tensor(index).to(torch.long)

        #particles = self._process(self.src.images(index).to(self.device))
        particles = self.src.images(index)
        # import pdb; pdb.set_trace() 
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
        # raise NotImplementedError
        # Maybe this would work?
        return NumpyLoader(torch.utils.data.Subset(self, subset_indices), batch_size=batch_size, shuffle=False, num_workers = num_workers)
        # torch.utils.data.Subset(self, subset_indices)


class TiltSeriesData(ImageDataset):
    """
    Class representing tilt series
    """

    def __init__(
        self,
        tiltstar,
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
        # Note: ind is the indices of the *tilts*, not the particles
        super().__init__(tiltstar, ind=ind, **kwargs)

        # Parse unique particles from _rlnGroupName
        s = starfile.Starfile.load(tiltstar)

        unique_sorted_group_names = []
        for name in s.df["_rlnGroupName"]:
            if name not in unique_sorted_group_names:
                unique_sorted_group_names.append(name)

        if ind is not None:
            s.df = s.df.loc[ind]
            
        group_name = list(s.df["_rlnGroupName"])
        
        particles = OrderedDict()
        self.dataset_tilt_indices = []
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
        images = self.src.images(tilt_indices)#.to(self.device))
        return images, index, tilt_indices#, index



    @classmethod
    def parse_particle_tilt(
        cls, tiltstar: str
    ) -> Tuple[list[np.ndarray], dict[int, int]]:
        # Parse unique particles from _rlnGroupName
        s = starfile.Starfile.load(tiltstar)
        group_name = list(s.df["_rlnGroupName"])
        particles = OrderedDict()

        for ii, gn in enumerate(group_name):
            if gn not in particles:
                particles[gn] = []
            particles[gn].append(ii)

        particles = [np.asarray(pp, dtype=int) for pp in particles.values()]
        particles_to_tilts = particles
        tilts_to_particles = {}

        for i, j in enumerate(particles):
            for jj in j:
                tilts_to_particles[jj] = i

        return particles_to_tilts, tilts_to_particles

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

    def get_dataset_subset_generator(self, batch_size, subset_indices , num_workers = 0):
        return make_dataloader(torch.utils.data.Subset(self, subset_indices), batch_size=batch_size, num_workers=num_workers) #torch.utils.data(make_dataloader(self, batch_size=batch_size, num_workers=num_workers), subset_indices)


    # def get_dataset_subset_generator(self, batch_size, subset_indices, num_workers = 0):
    #     return tf.data.Dataset.from_tensor_slices((self.particles[subset_indices], subset_indices)).batch(batch_size, num_parallel_calls = tf.data.AUTOTUNE).as_numpy_iterator()

# def make_data_subset_loader(
#     data: ImageDataset,
#     indices,
#     *,
#     batch_size: int,
#     num_workers: int = 0,
#     shuffler_size: int = 0,
#     shuffle=False,
# ):
#     if shuffler_size > 0 and shuffle:
#         assert data.lazy, "Only enable a data shuffler for lazy loading"
#         return DataShuffler(data, batch_size=batch_size, buffer_size=shuffler_size)
#     else:
#         # see https://github.com/zhonge/cryodrgn/pull/221#discussion_r1120711123
#         # for discussion of why we use BatchSampler, etc.
#         sampler_cls = RandomSampler if shuffle else SequentialSampler
#         batch_size=1

#         dataloader = NumpyLoader(data, batch_size=batch_size, shuffle=False, num_workers = num_workers)
#         return torch.utils.data.Subset(dataloader, indices = indices)



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

        return DataLoader(
            data,
            num_workers=num_workers,
            sampler=BatchSampler(
                sampler_cls(data), batch_size=batch_size, drop_last=False
            ),
            batch_size=None,
            multiprocessing_context="spawn" if num_workers > 0 else None,
        )
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


# class NumpySubsetLoader(torch.utils.data.Subset):
  
#   def __init__(self, dataset, indices, batch_size=1,
#                 shuffle=False, sampler=None,
#                 batch_sampler=None, num_workers=0,
#                 pin_memory=False, drop_last=False,
#                 timeout=0, worker_init_fn=None):
    
#     torch.utils.data.Subset(super(self.__class__, self).__init__(dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         sampler=sampler,
#         batch_sampler=batch_sampler,
#         num_workers=num_workers,
#         collate_fn=numpy_collate,
#         pin_memory=pin_memory,
#         drop_last=drop_last,
#         timeout=timeout,
#         worker_init_fn=worker_init_fn), indices)



# import numpy as np
# from collections import Counter, OrderedDict

# import logging
# import torch
# from torch.utils import data
# from typing import Optional, Tuple, Union
# from cryodrgn import fft, starfile
# from cryodrgn.source import ImageSource
# from cryodrgn.utils import window_mask

# from torch.utils.data import DataLoader
# from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler
# from recovar.fourier_transform_utils import fourier_transform_utils
# ftu = fourier_transform_utils(jnp)

# logger = logging.getLogger(__name__)


# class ImageDataset(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         mrcfile,
#         lazy=True,
#         norm=None,
#         keepreal=False,
#         invert_data=False,
#         ind=None,
#         window=True,
#         datadir=None,
#         window_r=0.85,
#         max_threads=16,
#         device: Union[str, torch.device] = "cpu",
#     ):
#         assert not keepreal, "Not implemented yet"
#         datadir = datadir or ""
#         self.ind = ind
#         self.src = ImageSource.from_file(
#             mrcfile,
#             lazy=lazy,
#             datadir=datadir,
#             indices=ind,
#             max_threads=max_threads,
#         )

#         ny = self.src.D
#         assert ny % 2 == 0, "Image size must be even."

#         self.N = self.src.n
#         self.D = ny + 1  # after symmetrization
#         self.invert_data = invert_data
#         self.window = window_mask(ny, window_r, 0.99).to(device) if window else None
#         norm = norm or self.estimate_normalization()
#         self.norm = [float(x) for x in norm]
#         self.device = device
#         self.lazy = lazy

#     def estimate_normalization(self, n=1000):
#         n = min(n, self.N) if n is not None else self.N
#         indices = range(0, self.N, self.N // n)  # FIXME: what if the data is not IID??
#         imgs = self.src.images(indices)

#         particleslist = []
#         for img in imgs:
#             particleslist.append(fft.ht2_center(img))
#         imgs = torch.stack(particleslist)

#         if self.invert_data:
#             imgs *= -1

#         imgs = fft.symmetrize_ht(imgs)
#         norm = (0, torch.std(imgs))
#         logger.info("Normalizing HT by {} +/- {}".format(*norm))
#         return norm

#     def _process(self, data):
#         if data.ndim == 2:
#             data = data[np.newaxis, ...]
#         if self.window is not None:
#             data *= self.window
#         data = fft.ht2_center(data)
#         if self.invert_data:
#             data *= -1
#         data = fft.symmetrize_ht(data)
#         data = (data - self.norm[0]) / self.norm[1]
#         return data

#     def __len__(self):
#         return self.N

#     def __getitem__(self, index):
#         if isinstance(index, list):
#             index = torch.Tensor(index).to(torch.long)

#         particles = self._process(self.src.images(index).to(self.device))

#         # this is why it is tricky for index to be allowed to be a list!
#         if len(particles.shape) == 2:
#             particles = particles[np.newaxis, ...]

#         if isinstance(index, (int, np.integer)):
#             logger.debug(f"ImageDataset returning images at index ({index})")
#         else:
#             logger.debug(
#                 f"ImageDataset returning images for {len(index)} indices ({index[0]}..{index[-1]})"
#             )

#         return particles, None, index

#     def get_slice(
#         self, start: int, stop: int
#     ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
#         return (
#             self.src.images(slice(start, stop), require_contiguous=True).numpy(),
#             None,
#         )


# class TiltSeriesData(ImageDataset):
#     """
#     Class representing tilt series
#     """

#     def __init__(
#         self,
#         tiltstar,
#         ntilts,
#         random_tilts=False,
#         ind=None,
#         voltage=None,
#         expected_res=None,
#         dose_per_tilt=None,
#         angle_per_tilt=None,
#         **kwargs,
#     ):
#         # Note: ind is the indices of the *tilts*, not the particles
#         super().__init__(tiltstar, ind=ind, **kwargs)

#         # Parse unique particles from _rlnGroupName
#         s = starfile.Starfile.load(tiltstar)
#         if ind is not None:
#             s.df = s.df.loc[ind]
#         group_name = list(s.df["_rlnGroupName"])
#         particles = OrderedDict()
#         for ii, gn in enumerate(group_name):
#             if gn not in particles:
#                 particles[gn] = []
#             particles[gn].append(ii)
#         self.particles = [np.asarray(pp, dtype=int) for pp in particles.values()]
#         self.Np = len(particles)
#         self.ctfscalefactor = np.asarray(s.df["_rlnCtfScalefactor"], dtype=np.float32)
#         self.tilt_numbers = np.zeros(self.N)
#         for ind in self.particles:
#             sort_idxs = self.ctfscalefactor[ind].argsort()
#             ranks = np.empty_like(sort_idxs)
#             ranks[sort_idxs[::-1]] = np.arange(len(ind))
#             self.tilt_numbers[ind] = ranks
#         # self.tilt_numbers = torch.tensor(self.tilt_numbers).to(self.device)
#         logger.info(f"Loaded {self.N} tilts for {self.Np} particles")
#         counts = Counter(group_name)
#         unique_counts = set(counts.values())
#         logger.info(f"{unique_counts} tilts per particle")
#         self.counts = counts
#         assert ntilts <= min(unique_counts)
#         self.ntilts = ntilts
#         self.random_tilts = random_tilts

#         self.voltage = voltage
#         self.dose_per_tilt = dose_per_tilt

#         # Assumes dose-symmetric tilt scheme
#         # As implemented in Hagen, Wan, Briggs J. Struct. Biol. 2017
#         self.tilt_angles = None
#         if angle_per_tilt is not None:
#             self.tilt_angles = angle_per_tilt * torch.ceil(self.tilt_numbers / 2)
#             #self.tilt_angles = torch.tensor(self.tilt_angles).to(self.device)

#     def __len__(self):
#         return self.Np

#     def __getitem__(self, index):
#         if isinstance(index, list):
#             index = np.array(index)#.to(torch.long)
#         tilt_indices = []
#         for ii in index:
#             if self.random_tilts:
#                 tilt_index = np.random.choice(
#                     self.particles[ii], self.ntilts, replace=False
#                 )
#             else:
#                 # take the first ntilts
#                 tilt_index = self.particles[ii][0 : self.ntilts]
#             tilt_indices.append(tilt_index)
#         tilt_indices = np.concatenate(tilt_indices)
#         images = self._process(self.src.images(tilt_indices).to(self.device))
#         return images, tilt_indices, index

#     @classmethod
#     def parse_particle_tilt(
#         cls, tiltstar: str
#     ) -> tuple[list[np.ndarray], dict[np.int64, int]]:
#         # Parse unique particles from _rlnGroupName
#         s = starfile.Starfile.load(tiltstar)
#         group_name = list(s.df["_rlnGroupName"])
#         particles = OrderedDict()

#         for ii, gn in enumerate(group_name):
#             if gn not in particles:
#                 particles[gn] = []
#             particles[gn].append(ii)

#         particles = [np.asarray(pp, dtype=int) for pp in particles.values()]
#         particles_to_tilts = particles
#         tilts_to_particles = {}

#         for i, j in enumerate(particles):
#             for jj in j:
#                 tilts_to_particles[jj] = i

#         return particles_to_tilts, tilts_to_particles

#     @classmethod
#     def particles_to_tilts(
#         cls, particles_to_tilts: list[np.ndarray], particles: np.ndarray
#     ) -> np.ndarray:
#         tilts = [particles_to_tilts[int(i)] for i in particles]
#         tilts = np.concatenate(tilts)

#         return tilts

#     @classmethod
#     def tilts_to_particles(cls, tilts_to_particles, tilts):
#         particles = [tilts_to_particles[i] for i in tilts]
#         particles = np.array(sorted(set(particles)))
#         return particles

#     def get_tilt(self, index):
#         return super().__getitem__(index)

#     def get_slice(self, start: int, stop: int) -> Tuple[np.ndarray, np.ndarray]:
#         # we have to fetch all the tilts to stay contiguous, and then subset
#         tilt_indices = [self.particles[index] for index in range(start, stop)]
#         cat_tilt_indices = np.concatenate(tilt_indices)
#         images = self.src.images(cat_tilt_indices, require_contiguous=True)

#         tilt_masks = []
#         for tilt_idx in tilt_indices:
#             tilt_mask = np.zeros(len(tilt_idx), dtype=np.bool)
#             if self.random_tilts:
#                 tilt_mask_idx = np.random.choice(
#                     len(tilt_idx), self.ntilts, replace=False
#                 )
#                 tilt_mask[tilt_mask_idx] = True
#             else:
#                 i = (len(tilt_idx) - self.ntilts) // 2
#                 tilt_mask[i : i + self.ntilts] = True
#             tilt_masks.append(tilt_mask)
#         tilt_masks = np.concatenate(tilt_masks)
#         selected_images = images[tilt_masks]
#         selected_tilt_indices = cat_tilt_indices[tilt_masks]

#         return selected_images.numpy(), selected_tilt_indices

#     def critical_exposure(self, freq, voltage):
# #         assert (
# #             voltage is not None
# #         ), "Critical exposure calculation requires voltage"

#         # From Grant and Grigorieff, 2015
#         scale_factor = 1
#         if voltage == 200:
#             scale_factor = 0.75
#         scale_factor = jnp.where(jnp.isclose(voltage, 200), 0.75, 1)
# #         critical_exp = jnp.pow(freq, -1.665)
#         critical_exp = freq ** (-1.665)
#         critical_exp = critical_exp *  scale_factor * 0.245
#         return critical_exp +  2.81


#     def optimal_exposure(self, freq):
#         return 2.51284 * self.critical_exposure(freq)


# class DataShuffler:
#     def __init__(
#         self, dataset: ImageDataset, batch_size, buffer_size, dtype=np.float32
#     ):
#         if not all(dataset.src.indices == np.arange(dataset.N)):
#             raise NotImplementedError(
#                 "Sorry dude, --ind is not supported for the data shuffler. "
#                 "The purpose of the shuffler is to load chunks contiguously during lazy loading on huge datasets, which doesn't work with --ind. "
#                 "If you really need this, maybe you should probably use `--ind` during preprocessing (e.g. cryodrgn downsample)."
#             )
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.buffer_size = buffer_size
#         self.dtype = dtype
#         assert self.buffer_size % self.batch_size == 0, (
#             self.buffer_size,
#             self.batch_size,
#         )  # FIXME
#         self.batch_capacity = self.buffer_size // self.batch_size
#         assert self.buffer_size <= len(self.dataset), (
#             self.buffer_size,
#             len(self.dataset),
#         )
#         self.ntilts = getattr(dataset, "ntilts", 1)  # FIXME

#     def __iter__(self):
#         return _DataShufflerIterator(self)


# class _DataShufflerIterator:
#     def __init__(self, shuffler: DataShuffler):
#         self.dataset = shuffler.dataset
#         self.buffer_size = shuffler.buffer_size
#         self.batch_size = shuffler.batch_size
#         self.batch_capacity = shuffler.batch_capacity
#         self.dtype = shuffler.dtype
#         self.ntilts = shuffler.ntilts

#         self.buffer = np.empty(
#             (self.buffer_size, self.ntilts, self.dataset.D - 1, self.dataset.D - 1),
#             dtype=self.dtype,
#         )
#         self.index_buffer = np.full((self.buffer_size,), -1, dtype=np.int64)
#         self.tilt_index_buffer = np.full(
#             (self.buffer_size, self.ntilts), -1, dtype=np.int64
#         )
#         self.num_batches = (
#             len(self.dataset) // self.batch_size
#         )  # FIXME off-by-one? Nah, lets leave the last batch behind
#         self.chunk_order = torch.randperm(self.num_batches)
#         self.count = 0
#         self.flush_remaining = -1  # at the end of the epoch, got to flush the buffer
#         # pre-fill
#         logger.info("Pre-filling data shuffler buffer...")
#         for i in range(self.batch_capacity):
#             chunk, maybe_tilt_indices, chunk_indices = self._get_next_chunk()
#             self.buffer[i * self.batch_size : (i + 1) * self.batch_size] = chunk
#             self.index_buffer[
#                 i * self.batch_size : (i + 1) * self.batch_size
#             ] = chunk_indices
#             if maybe_tilt_indices is not None:
#                 self.tilt_index_buffer[
#                     i * self.batch_size : (i + 1) * self.batch_size
#                 ] = maybe_tilt_indices
#         logger.info(
#             f"Filled buffer with {self.buffer_size} images ({self.batch_capacity} contiguous chunks)."
#         )

#     def _get_next_chunk(self) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
#         chunk_idx = int(self.chunk_order[self.count])
#         self.count += 1
#         particles, maybe_tilt_indices = self.dataset.get_slice(
#             chunk_idx * self.batch_size, (chunk_idx + 1) * self.batch_size
#         )
#         particle_indices = np.arange(
#             chunk_idx * self.batch_size, (chunk_idx + 1) * self.batch_size
#         )
#         particles = particles.reshape(
#             self.batch_size, self.ntilts, *particles.shape[1:]
#         )
#         if maybe_tilt_indices is not None:
#             maybe_tilt_indices = maybe_tilt_indices.reshape(
#                 self.batch_size, self.ntilts
#             )
#         return particles, maybe_tilt_indices, particle_indices

#     def __iter__(self):
#         return self

#     def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """Returns a batch of images, and the indices of those images in the dataset.

#         The buffer starts filled with `batch_capacity` random contiguous chunks.
#         Each time a batch is requested, `batch_size` random images are selected from the buffer,
#         and refilled with the next random contiguous chunk from disk.

#         Once all the chunks have been fetched from disk, the buffer is randomly permuted and then
#         flushed sequentially.
#         """
#         if self.count == self.num_batches and self.flush_remaining == -1:
#             logger.info(
#                 "Finished fetching chunks. Flushing buffer for remaining batches..."
#             )
#             # since we're going to flush the buffer sequentially, we need to shuffle it first
#             perm = np.random.permutation(self.buffer_size)
#             self.buffer = self.buffer[perm]
#             self.index_buffer = self.index_buffer[perm]
#             self.flush_remaining = self.buffer_size

#         if self.flush_remaining != -1:
#             # we're in flush mode, just return chunks out of the buffer
#             assert self.flush_remaining % self.batch_size == 0
#             if self.flush_remaining == 0:
#                 raise StopIteration()
#             particles = self.buffer[
#                 self.flush_remaining - self.batch_size : self.flush_remaining
#             ]
#             particle_indices = self.index_buffer[
#                 self.flush_remaining - self.batch_size : self.flush_remaining
#             ]
#             tilt_indices = self.tilt_index_buffer[
#                 self.flush_remaining - self.batch_size : self.flush_remaining
#             ]
#             self.flush_remaining -= self.batch_size
#         else:
#             indices = np.random.choice(
#                 self.buffer_size, size=self.batch_size, replace=False
#             )
#             particles = self.buffer[indices]
#             particle_indices = self.index_buffer[indices]
#             tilt_indices = self.tilt_index_buffer[indices]

#             chunk, maybe_tilt_indices, chunk_indices = self._get_next_chunk()
#             self.buffer[indices] = chunk
#             self.index_buffer[indices] = chunk_indices
#             if maybe_tilt_indices is not None:
#                 self.tilt_index_buffer[indices] = maybe_tilt_indices

#         particles = torch.from_numpy(particles)
#         particle_indices = torch.from_numpy(particle_indices)
#         tilt_indices = torch.from_numpy(tilt_indices)

#         # merge the batch and tilt dimension
#         particles = particles.view(-1, *particles.shape[2:])
#         tilt_indices = tilt_indices.view(-1, *tilt_indices.shape[2:])

#         particles = self.dataset._process(particles.to(self.dataset.device))
#         # print('ZZZ', particles.shape, tilt_indices.shape, particle_indices.shape)
#         return particles, tilt_indices, particle_indices


# def make_dataloader(
#     data: ImageDataset,
#     *,
#     batch_size: int,
#     num_workers: int = 0,
#     shuffler_size: int = 0,
#     shuffle=True,
# ):
#     if shuffler_size > 0 and shuffle:
#         assert data.lazy, "Only enable a data shuffler for lazy loading"
#         return DataShuffler(data, batch_size=batch_size, buffer_size=shuffler_size)
#     else:
#         # see https://github.com/zhonge/cryodrgn/pull/221#discussion_r1120711123
#         # for discussion of why we use BatchSampler, etc.
#         sampler_cls = RandomSampler if shuffle else SequentialSampler
#         return DataLoader(
#             data,
#             num_workers=num_workers,
#             sampler=BatchSampler(
#                 sampler_cls(data), batch_size=batch_size, drop_last=False
#             ),
#             batch_size=None,
#             multiprocessing_context="spawn" if num_workers > 0 else None,
#         )
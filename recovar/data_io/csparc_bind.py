import numpy as np
from os.path import exists

from recovar import utils
import recovar.core.fourier_transform_utils as fourier_transform_utils


def load_3dva_results(root, dft=False):
    k = 0
    components = []
    zs = []
    metadata = np.load(root + '_particles.cs')
    while exists(f'{root}_component_{k}.mrc'):
        components.append(utils.load_mrc(f'{root}_component_{k}.mrc'))
        zs.append(metadata[f'components_mode_{k}/value'])
        k += 1
    components = np.stack(components)
    zs = np.stack(zs)
    mean = utils.load_mrc(root + '_map.mrc')
    if not dft:
        return mean, components.reshape([components.shape[0], -1]).T, zs

    dft = fourier_transform_utils.get_dft3(components)
    dft = dft.reshape([dft.shape[0], -1])
    dft /= np.sqrt(dft.shape[0])

    return fourier_transform_utils.get_dft3(mean).reshape(-1), dft, zs.T * np.sqrt(dft.shape[0])


# Alias: 3D Flex results use the same file format as 3D VA.
load_3dflex_results = load_3dva_results

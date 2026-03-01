import numpy as np
from recovar import utils
import recovar.core.fourier_transform_utils as fourier_transform_utils
from os.path import exists

def load_3dva_results(root, dft = False):
    k = 0
    components = []; zs = []
    metadata = np.load(root + '_particles.cs')
    while( exists(root + '_component_' + str(k) + '.mrc')):
        components.append((utils.load_mrc(root + '_component_' + str(k) + '.mrc')))
        zs.append(metadata['components_mode_'+ str(k)+'/value'])
        k = k + 1
    components = np.stack(components)
    zs = np.stack(zs)
    mean = (utils.load_mrc(root + '_map.mrc'))
    if dft == False:
        return mean, components.reshape([components.shape[0], -1]).T , zs

    dft = fourier_transform_utils.get_dft3(components) 
    dft = dft.reshape([dft.shape[0], -1])
    dft /= np.sqrt(dft.shape[0])
    
    return fourier_transform_utils.get_dft3(mean).reshape(-1)  , dft , zs.T *  np.sqrt(dft.shape[0])


def load_3dflex_results(root, dft = False):
    k = 0
    components = []; zs = []
    metadata = np.load(root + '_particles.cs')
    while( exists(root + '_component_' + str(k) + '.mrc')):
        components.append((utils.load_mrc(root + '_component_' + str(k) + '.mrc')))
        zs.append(metadata['components_mode_'+ str(k)+'/value'])
        k = k + 1
    components = np.stack(components)
    zs = np.stack(zs)
    mean = (utils.load_mrc(root + '_map.mrc'))
    if dft == False:
        return mean, components.reshape([components.shape[0], -1]).T , zs

    dft = fourier_transform_utils.get_dft3(components) 
    dft = dft.reshape([dft.shape[0], -1])
    dft /= np.sqrt(dft.shape[0])
    
    return fourier_transform_utils.get_dft3(mean).reshape(-1)  , dft , zs.T *  np.sqrt(dft.shape[0])

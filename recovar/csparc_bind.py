import numpy as np
from recovar import utils
import recovar.config
from importlib import reload
import numpy as np
from recovar import plot_utils, utils
from recovar import output, dataset
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as py
from recovar import simulator
import jax
import warnings
from recovar.fourier_transform_utils import fourier_transform_utils
import jax.numpy as jnp
ftu = fourier_transform_utils(jnp)
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

    dft = ftu.get_dft3(components) 
    dft = dft.reshape([dft.shape[0], -1])
    dft /= np.sqrt(dft.shape[0])
    
    return ftu.get_dft3(mean).reshape(-1)  , dft , zs.T *  np.sqrt(dft.shape[0])


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

    dft = ftu.get_dft3(components) 
    dft = dft.reshape([dft.shape[0], -1])
    dft /= np.sqrt(dft.shape[0])
    
    return ftu.get_dft3(mean).reshape(-1)  , dft , zs.T *  np.sqrt(dft.shape[0])

"""
Copy pasted from https://github.com/ml-struct-bio/cryodrgn
"""

from recovar import utils
import logging

# import pickle
from typing import Optional, Tuple, Union, List
import logging
import numpy as np
# import torch
# import torch.nn as nn
# from torch import Tensor
# from cryodrgn import lie_tools, utils

logger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)

def print_ctf_params(params: np.ndarray) -> None:
    assert len(params) == 9
    logger.info("Image size (pix)  : {}".format(int(params[0])))
    logger.info("A/pix             : {}".format(params[1]))
    logger.info("DefocusU (A)      : {}".format(params[2]))
    logger.info("DefocusV (A)      : {}".format(params[3]))
    logger.info("Dfang (deg)       : {}".format(params[4]))
    logger.info("voltage (kV)      : {}".format(params[5]))
    logger.info("cs (mm)           : {}".format(params[6]))
    logger.info("w                 : {}".format(params[7]))
    logger.info("Phase shift (deg) : {}".format(params[8]))

def load_ctf_for_training(D: int, ctf_params_pkl: str) -> np.ndarray:
    assert D % 2 == 0
    ctf_params = utils.pickle_load(ctf_params_pkl)
    assert ctf_params.shape[1] == 9
    # Replace original image size with current dimensions
    Apix = ctf_params[0, 0] * ctf_params[0, 1] / D
    ctf_params[:, 0] = D
    ctf_params[:, 1] = Apix
    print_ctf_params(ctf_params[0])
    # Slice out the first column (D)
    return ctf_params[:, 1:]


# poses_file, dataset.n_images, dataset.unpadded_D, ind = ind
def load_poses(
        infile: Union[str, List[str]],
        Nimg: int,
        D: int,
        # emb_type: Optional[str] = None,
        ind: Optional[np.ndarray] = None,
        # device: Optional[torch.device] = None,
    ):
        """
        Return an instance of PoseTracker

        Inputs:
            infile (str or list):   One or two files, with format options of:
                                    single file with pose pickle
                                    two files with rot and trans pickle
                                    single file with rot pickle
            Nimg:               Number of particles
            D:                  Box size (pixels)
            emb_type:           SO(3) embedding type if refining poses
            ind:                Index array if poses are being filtered
        """
        # load pickle
        if type(infile) is str:
            infile = [infile]
        assert len(infile) in (1, 2)
        if len(infile) == 2:  # rotation pickle, translation pickle
            poses = (utils.pickle_load(infile[0]), utils.pickle_load(infile[1]))
        else:  # rotation pickle or poses pickle
            poses = utils.pickle_load(infile[0])
            if type(poses) != tuple:
                poses = (poses,)

        # rotations
        rots = poses[0]
        if ind is not None:
            if len(rots) > Nimg:  # HACK
                rots = rots[ind]
        assert rots.shape == (
            Nimg,
            3,
            3,
        ), f"Input rotations have shape {rots.shape} but expected ({Nimg},3,3)"

        # translations if they exist
        if len(poses) == 2:
            trans = poses[1]
            if ind is not None:
                if len(trans) > Nimg:  # HACK
                    trans = trans[ind]
            assert trans.shape == (
                Nimg,
                2,
            ), f"Input translations have shape {trans.shape} but expected ({Nimg},2)"
            assert np.all(
                trans <= 1
            ), "ERROR: Old pose format detected. Translations must be in units of fraction of box."
            trans *= D  # convert from fraction to pixels
        else:
            logger.warning("WARNING: No translations provided")
            trans = None

        return rots, trans, D#, emb_type#, device=device)
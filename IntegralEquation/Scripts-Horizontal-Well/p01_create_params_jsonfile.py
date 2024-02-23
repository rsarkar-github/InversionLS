import os
import json
import numpy as np
import shutil
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from ...Utilities.Utils import cosine_taper_2d
from ..Inversion.ScatteringIntegralGeneralVzInversion import ScatteringIntegralGeneralVzInversion2d


if __name__ == "__main__":

    basedir = "InversionLS/Expt/horizontal-well/"
    if not os.path.exists(basedir):
        os.makedirs(basedir)

    # Load horizontal well model
    vp = np.load("InversionLS/Data/vp-log-horizontal-well.npy") / 1000.0
    nz = vp.shape[0]
    nx = 2000

    vp = np.zeros((nz, nx)) + np.reshape(vp, newshape=(nz, 1))
    print(vp.shape)

    def plot(vel, extent, title, file_name=None):
        fig = plt.figure(figsize=(6, 3))  # define figure size
        image = plt.imshow(vel, cmap="jet", interpolation='nearest', extent=extent)

        cbar = plt.colorbar(aspect=10, pad=0.02)
        cbar.set_label('Vp [km/s]', labelpad=10)
        plt.title(title)
        plt.xlabel('x [m]')
        plt.ylabel('z [m]')

        if file_name is not None:
            plt.savefig(file_name, format="pdf", bbox_inches="tight", pad_inches=0.01)

        plt.show()
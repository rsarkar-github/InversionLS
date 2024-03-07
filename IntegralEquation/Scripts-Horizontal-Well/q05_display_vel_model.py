import sys
import numpy as np
import matplotlib.pyplot as plt
from ..Inversion.ScatteringIntegralGeneralVzInversion import ScatteringIntegralGeneralVzInversion2d


if __name__ == "__main__":

    def plot1(vel, extent, title, aspect_ratio=1, aspect_cbar=10, file_name=None, vmin=None, vmax=None):

        if vmin is None:
            vmin = np.min(vel)
        if vmax is None:
            vmax = np.max(vel)

        fig = plt.figure(figsize=(6, 3))  # define figure size
        image = plt.imshow(vel, cmap="jet", interpolation='bicubic', extent=extent, vmin=vmin, vmax=vmax)

        cbar = plt.colorbar(aspect=aspect_cbar, pad=0.02)
        cbar.set_label('Vp [km/s]', labelpad=10)
        plt.title(title)
        plt.xlabel('x [m]')
        plt.ylabel('z [m]')
        plt.gca().set_aspect(aspect_ratio)

        if file_name is not None:
            plt.savefig(file_name, format="pdf", bbox_inches="tight", pad_inches=0.01)

        plt.show()

    with np.load("InversionLS/Data/horizontal-well-new-2d.npz") as f:
        vel = f["arr_0"]

    with np.load("InversionLS/Data/horizontal-well-new-vz-2d.npz") as f:
        vel_vz = f["arr_0"]

    basedir = "InversionLS/Expt/horizontal-well/"
    obj = ScatteringIntegralGeneralVzInversion2d(
        basedir=basedir,
        restart=True,
        restart_code=None
    )

    xmax = 1.0
    zmax = (obj.b - obj.a)
    extent = [0, xmax, zmax, 0]

    plot1(
        vel=vel_vz,
        extent=extent,
        title="Horizontal well v(z) model",
        aspect_ratio=10,
        aspect_cbar=10,
        vmin=2.5,
        vmax=6.0
    )

    plot1(
        vel=vel - vel_vz,
        extent=extent,
        title="Pert",
        aspect_ratio=10,
        aspect_cbar=10
    )
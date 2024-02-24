import os
import json
import numpy as np
from scipy.ndimage import gaussian_filter
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
    vz = np.load("InversionLS/Data/vp-log-horizontal-well.npy") / 1000.0
    nz = vz.shape[0]
    nx = 2000

    vz = np.zeros((nz, nx)) + np.reshape(vz, newshape=(nz, 1))
    print(vz.shape)


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


    # Smooth model
    vz = gaussian_filter(vz, sigma=10)

    # Crop model and calculate some parameters
    dx = dz = 0.1524  # grid spacing in m
    zdim = 80.0
    nz1 = int(zdim / dz) + 1
    nz_start = 1385
    vz = vz[nz_start: nz_start + nz1, :]

    xmax = (nx - 1) * dx
    zmax = (nz1 - 1) * dz
    extent = [0, xmax, zmax, 0]
    # plot(vel=vz, extent=extent, title="Horizontal well v(z) model")

    # Interpolate the vp velocities to 2.5 m rid
    dz_new = 2.0   # grid spacing in m
    dx_new = 2.0   # grid spacing in m
    nz_new = int(zmax / dz_new) + 1   # 80
    nx_new = 501
    print("nz1 = ", nz1, ", nz_new = ", nz_new)

    def func_interp(vel):
        """
        Vel must have shape (nz1,).

        :param vel: Velocity to interpolate on 0.1524m grid.
        :return: Interpolated velocity on 2.5m grid.
        """
        zgrid_input = np.linspace(start=0, stop=zmax, num=nz1, endpoint=True).astype(np.float64)
        interp = RegularGridInterpolator((zgrid_input,), vel.astype(np.float64))

        vel_interp = np.zeros(shape=(nz_new,), dtype=np.float64)

        for i1 in range(nz_new):
                point = np.array([i1 * dz_new])
                vel_interp[i1] = interp(point)

        return vel_interp

    vz_trace = vz[:, 0]
    vz_trace_interp = func_interp(vel=vz_trace)
    vz_interp = np.zeros((nz_new, nx_new)) + np.reshape(vz_trace_interp, newshape=(nz_new, 1))

    xmax_new = (nx_new - 1) * dx_new
    zmax_new = (nz_new - 1) * dz_new
    extent_new = [0, xmax_new, zmax_new, 0]
    plot1(
        vel=vz_interp,
        extent=extent_new,
        title="Horizontal well v(z) model interp",
        aspect_ratio=10,
        aspect_cbar=10,
        vmin=2.5,
        vmax=6.0
    )

    # ---------------------------------------------
    # From this point on, work with vz_interp
    # vz_trace_interp is 1d vz profile
    # ---------------------------------------------
    vp_pert = vz_interp * 0.0
    vp_pert += np.random.normal(size=[vp_pert.shape[0], vp_pert.shape[1]], loc=0.0, scale=0.5)

    vp_pert = gaussian_filter(vp_pert, sigma=[1,10])

    # Apply crop
    vp_pert[0:14, :] = 0
    vp_pert[20:, :] = 0
    vp_pert[:, 0:25] = 0
    vp_pert[:, 475:] = 0

    vp_pert = gaussian_filter(vp_pert, sigma=[2, 10])

    plot1(
        vel=vp_pert,
        extent=extent_new,
        title="Horizontal well v(z) model interp",
        aspect_ratio=10,
        aspect_cbar=10
    )

    # Add vp_pert
    vp_true = vp_pert + vz_interp
    plot1(
        vel=vp_true,
        extent=extent_new,
        title="Horizontal well vp model interp",
        aspect_ratio=10,
        aspect_cbar=10,
        vmin=2.5,
        vmax=6.0
    )

    np.savez("InversionLS/Data/horizontal-well-new-2d.npz", vp_true)
    np.savez("InversionLS/Data/horizontal-well-new-vz-2d.npz", vz_interp)

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from ...Utilities.Utils import cosine_taper_2d


if __name__ == "__main__":

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

    dx = dz = 0.1524  # grid spacing in m
    xmax = (nx - 1) * dx
    zmax = (nz - 1) * dz
    extent = [0, xmax, zmax, 0]
    plot(vel=vp, extent=extent, title="Horizontal well v(z) model")

    # Smooth vp
    vp_smooth = np.zeros(shape=(nz, nx), dtype=float)
    conv_kernel = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    for i in range(nx):
        vp_smooth[:, i] = np.convolve(a=vp[:, i], v=conv_kernel, mode="same")

    plot(vel=vp_smooth, extent=extent, title="Horizontal well v(z) model")

    # Decimate x direction by factor of 2
    vp = vp[:, ::2]
    nz, nx = vp.shape
    dx = dz = 20  # 20m grid spacing

    #
    # xmax = (nx - 1) * dx
    # zmax = (nz - 1) * dz
    # extent = [0, xmax, zmax, 0]
    #
    # def plot(vel, extent, title, file_name=None):
    #     fig = plt.figure(figsize=(6, 3))  # define figure size
    #     image = plt.imshow(vel, cmap="jet", interpolation='nearest', extent=extent)
    #
    #     cbar = plt.colorbar(aspect=10, pad=0.02)
    #     cbar.set_label('Vp [km/s]', labelpad=10)
    #     plt.title(title)
    #     plt.xlabel('x [m]')
    #     plt.ylabel('z [m]')
    #
    #     if file_name is not None:
    #         plt.savefig(file_name, format="pdf", bbox_inches="tight", pad_inches=0.01)
    #
    #     plt.show()
    #
    # plot(vel=vp_vz, extent=extent, title="Marmousi-2 v(z) model")
    #
    # # Interpolate the vp and vp_vz velocities to 15m x 15m grid
    # dz1 = dx1 = 15.0
    # nz_new = int(zmax / dz1) + 1   # 231
    # nx_new = int(xmax / dx1) + 1   # 333
    #
    # def func_interp(vel):
    #     """
    #     Vel must have shape (nz, nx).
    #
    #     :param vel: Velocity to interpolate on 20m x 20m grid.
    #     :return: Interpolated velocity on 15m x 15m grid.
    #     """
    #     zgrid_input = np.linspace(start=0, stop=zmax, num=nz, endpoint=True).astype(np.float64)
    #     xgrid_input = np.linspace(start=0, stop=xmax, num=nx, endpoint=True).astype(np.float64)
    #     interp = RegularGridInterpolator((zgrid_input, xgrid_input), vel.astype(np.float64))
    #
    #     vel_interp = np.zeros(shape=(nz_new, nx_new), dtype=np.float64)
    #
    #     for i1 in range(nz_new):
    #         for i2 in range(nx_new):
    #             point = np.array([i1 * dz1, i2 * dx1])
    #             vel_interp[i1, i2] = interp(point)
    #
    #     return vel_interp
    #
    # vp_interp = func_interp(vp)
    # vp_vz_interp = func_interp(vp_vz)
    #
    # vp_interp = vp_interp.astype(np.float32)
    # vp_vz_interp = vp_vz_interp.astype(np.float32)
    #
    # extent = [0, (nx_new - 1) * dx1, (nz_new - 1) * dz1, 0]
    # plot(vel=vp_interp, extent=extent, title="Marmousi-2 model")
    # plot(
    #     vel=vp_vz_interp,
    #     extent=extent,
    #     title="Marmousi-2 v(z) model",
    #     file_name="InversionLS/Fig/marmousi-vp-vz-interp.pdf"
    # )
    #
    # # Find new water bottom
    # wb = 0
    # for i in range(nz_new):
    #     if vp_interp[i, 0] == 1.5:
    #         wb = i
    #     else:
    #         break
    # wb_depth = wb * dz1
    # print("New water bottom sample = ", wb)  # wb sample = 28
    #
    # # Create perturbation and apply taper
    # skip = 5
    # vp_diff = vp_interp - vp_vz_interp
    # vp_diff1 = vp_diff[skip:nz_new - skip, skip:nx_new - skip] * 1.0
    # cosine_taper_2d(array2d=vp_diff1, ncells_pad_x=20, ncells_pad_z=20)
    # vp_diff1 = vp_diff1.astype(np.float32)
    # vp_diff *= 0
    # vp_diff[skip:nz_new - skip, skip:nx_new - skip] += vp_diff1
    #
    # plot(
    #     vel=vp_vz_interp + vp_diff,
    #     extent=extent,
    #     title="Marmousi-2 model",
    #     file_name="InversionLS/Fig/marmousi-vp-interp-compact.pdf"
    # )
    #
    # # Write files vp_vz to disk
    # np.savez("InversionLS/Data/marmousi-vp-vz-interp.npz", vp_vz_interp[:, 0])
    # np.savez("InversionLS/Data/marmousi-vp-vz-interp-2d.npz", vp_vz_interp)
    # np.savez("InversionLS/Data/marmousi-vp-interp-compact.npz", vp_vz_interp + vp_diff)

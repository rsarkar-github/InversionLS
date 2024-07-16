import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from ..Inversion.ScatteringIntegralGeneralVzInversion import ScatteringIntegralGeneralVzInversion2d


if __name__ == "__main__":

    basedir = "InversionLS/Expt/horizontal-well/"
    obj = ScatteringIntegralGeneralVzInversion2d(
        basedir=basedir,
        restart=True,
        restart_code=None,
        num_procs_check_iter_files=4
    )

    # ------------------------------------
    # Get model parameters
    # ------------------------------------
    scale_fac_inv = obj.scale_fac_inv
    xmax = 1.0
    zmax = (obj.b - obj.a)
    extent = [0, xmax * scale_fac_inv, zmax * scale_fac_inv, 0]

    print("--------------------------------------------")
    print("Printing model parameters and acquisition\n")
    print("nz = ", obj.nz)
    print("n = ", obj.n)
    print("dx = ", xmax / (obj.n - 1))
    print("dz = ", zmax / (obj.nz - 1))
    print("\n")
    print("Num k values = ", obj.num_k_values)
    print("k values = ", np.asarray(obj.k_values) / (2 * np.pi))
    print("\n")
    print("Num receivers = ", obj.num_rec)
    print("Num sources = ", obj.num_sources)
    print("--------------------------------------------")

    # ------------------------------------
    # Read velocity, model pert
    # ------------------------------------
    vz_2d = np.zeros(shape=(obj.nz, obj.n), dtype=np.float32)
    vz_2d = obj.vz

    def plot1(vel, extent, title, aspect_ratio=1, file_name=None, vmin=None, vmax=None):

        if vmin is None:
            vmin = np.min(vel)
        if vmax is None:
            vmax = np.max(vel)

        fig, ax = plt.subplots(1, 1)
        im = ax.imshow(vel, aspect=aspect_ratio, cmap="jet", interpolation='bicubic', extent=extent, vmin=vmin, vmax=vmax)

        ax.set_title(title)
        ax.set_xlabel('x [km]')
        ax.set_ylabel('z [km]')

        axins = inset_axes(ax, width="5%", height="100%", loc='lower left',
                           bbox_to_anchor=(1.02, 0., 1, 1), bbox_transform=ax.transAxes,
                           borderpad=0)
        cbar = fig.colorbar(im, cax=axins)
        cbar.set_label('Vp [km/s]', labelpad=10)

        if file_name is not None:
            plt.savefig(file_name, format="pdf", bbox_inches="tight", pad_inches=0.01)

        plt.show()


    plot1(
        vel=vz_2d,
        extent=extent,
        title="",
        aspect_ratio=4,
        vmin=2.5,
        vmax=6.0
    )
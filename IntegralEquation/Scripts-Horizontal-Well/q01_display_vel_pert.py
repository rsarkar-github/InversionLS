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
    dx = xmax * scale_fac_inv / (obj.n - 1)
    dz = zmax * scale_fac_inv / (obj.nz - 1)

    print("--------------------------------------------")
    print("Printing model parameters and acquisition\n")
    print("nz = ", obj.nz)
    print("n = ", obj.n)
    print("dx = ", xmax * scale_fac_inv/ (obj.n - 1))
    print("dz = ", zmax * scale_fac_inv/ (obj.nz - 1))
    print("\n")
    print("Num k values = ", obj.num_k_values)
    print("k values (Hz) = ", np.asarray(obj.k_values) / (2 * np.pi))
    print("\n")
    print("Num receivers = ", obj.num_rec)
    print("Num sources = ", obj.num_sources)
    print("--------------------------------------------")

    # ------------------------------------
    # Get source and receiver coordinates
    # ------------------------------------
    rec_coords = [[item[0] * dz, item[1] * dx] for item in obj.rec_locs]

    num_sources = 101
    xgrid = np.linspace(start=0, stop=xmax * scale_fac_inv, num=num_sources, endpoint=True)
    zval = 21 * dz
    source_coords = np.zeros(shape=(num_sources, 2), dtype=np.float32)
    source_coords[:, 0] = zval
    source_coords[:, 1] = xgrid

    # ------------------------------------
    # Read velocity v(z), true model pert
    # ------------------------------------
    vz_2d = np.zeros(shape=(obj.nz, obj.n), dtype=np.float32)
    vz_2d = obj.vz
    psi = obj.true_model_pert
    vel = ((1 / (vz_2d ** 2.0)) - psi) ** (-0.5)
    dv = vz_2d - vel


    # -----------------------------------------
    # Plot v(z) velocity with source locations
    # -----------------------------------------
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(vz_2d, aspect=4, cmap="jet", interpolation='bicubic', extent=extent, vmin=2.5, vmax=6.0)
    ax.scatter(source_coords[:, 1], source_coords[:, 0], s=2, c="k", marker="x")
    ax.set_xlabel('x [km]')
    ax.set_ylabel('z [km]')

    axins = inset_axes(ax, width="3%", height="100%", loc='lower left',
                       bbox_to_anchor=(1.02, 0., 1, 1), bbox_transform=ax.transAxes,
                       borderpad=0)
    cbar = fig.colorbar(im, cax=axins)
    cbar.ax.set_title("km/s")
    # cbar.set_label('Vp [km/s]', labelpad=3)


    plt.savefig(
        basedir + "Fig/q01_vz_true_source_overlay.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.01
    )
    plt.show()

    # -----------------------------------------
    # Plot v(z) velocity as trace
    # -----------------------------------------


    # -----------------------------------------
    # Plot perturbation and dv
    # -----------------------------------------
    def plot1(
            vel, extent, title,
            aspect_ratio=1, cmap="jet",
            show_cbar=True,
            label_cbar="",
            file_name=None, vmin=None, vmax=None
    ):

        if vmin is None:
            vmin = np.min(vel)
        if vmax is None:
            vmax = np.max(vel)

        fig, ax = plt.subplots(1, 1)
        im = ax.imshow(vel, aspect=aspect_ratio, cmap=cmap, interpolation='bicubic', extent=extent, vmin=vmin, vmax=vmax)

        ax.set_title(title)
        ax.set_xlabel('x [km]')
        ax.set_ylabel('z [km]')

        if show_cbar:
            axins = inset_axes(ax, width="3%", height="100%", loc='lower left',
                               bbox_to_anchor=(1.02, 0., 1, 1), bbox_transform=ax.transAxes,
                               borderpad=0)
            cbar = fig.colorbar(im, cax=axins)
            cbar.ax.set_title(label_cbar)
            # cbar.set_label(label_cbar, labelpad=3)

        if file_name is not None:
            plt.savefig(file_name, format="pdf", bbox_inches="tight", pad_inches=0.01)

        plt.show()


    plot1(
        vel=dv,
        extent=extent,
        title="",
        aspect_ratio=4,
        cmap="seismic",
        file_name= basedir + "Fig/q01_vel_true_pert.pdf",
        label_cbar="km/s",
        vmin=-0.06,
        vmax=0.06
    )

    plot1(
        vel=psi,
        extent=extent,
        title="",
        aspect_ratio=4,
        cmap="seismic",
        file_name=basedir + "Fig/q01_true_psi.pdf",
        label_cbar=r"$s^2 / km^2$",
        vmin=-0.006,
        vmax=0.006
    )

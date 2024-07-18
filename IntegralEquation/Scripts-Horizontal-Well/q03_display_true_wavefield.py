import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from ..Inversion.ScatteringIntegralGeneralVzInversion import ScatteringIntegralGeneralVzInversion2d


if __name__ == "__main__":

    basedir = "InversionLS/Expt/horizontal-well/"
    obj = ScatteringIntegralGeneralVzInversion2d(
        basedir=basedir,
        restart=True,
        check_iter_files=False,
        restart_code=None,
        num_procs_check_iter_files=4
    )

    # ------------------------------------
    # Read parameters and data to plot
    # ------------------------------------

    # Check arguments
    if len(sys.argv) < 3:
        raise ValueError("Program missing command line arguments.")

    num_k_val = int(sys.argv[1])
    num_source = int(sys.argv[2])

    data = obj.get_true_wavefield(num_k=num_k_val, num_source=num_source)
    output_filename = (basedir + "Fig/q03_true_wavefield_"
                       + "k_num_" + str(num_k_val)
                       + "_sou_num_" + str(num_source) + "_.pdf")

    # ------------------------------------
    # Get model parameters
    # ------------------------------------
    scale_fac_inv = obj.scale_fac_inv
    xmax = 1.0
    zmax = (obj.b - obj.a)
    extent = [0, xmax * scale_fac_inv, zmax * scale_fac_inv, 0]
    dx = xmax * scale_fac_inv / (obj.n - 1)
    dz = zmax * scale_fac_inv / (obj.nz - 1)
    num_k_vals = obj.num_k_values
    k_vals = np.asarray(obj.k_values) / (2 * np.pi)

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


    # -----------------------------------------
    # Plot wavefield
    # -----------------------------------------
    figsize = (11, 4)
    fontsize = 14
    scale = 1e-5

    def plot1(
            vel, extent, title,
            aspect_ratio=1, cmap="jet", figsize=(16, 9),
            show_cbar=True,
            label_cbar="",
            file_name=None, vmin=None, vmax=None,
            show_freq=False,
            freq=None,
            show_source=False,
            sou_coords=None
    ):

        if vmin is None:
            vmin = np.min(vel)
        if vmax is None:
            vmax = np.max(vel)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        im = ax.imshow(vel, aspect=aspect_ratio, cmap=cmap, interpolation='bicubic', extent=extent, vmin=vmin,
                       vmax=vmax)

        ax.set_title(title)
        ax.set_xlabel('x [km]', fontsize=fontsize, fontname="STIXGeneral")
        ax.set_ylabel('z [km]', fontsize=fontsize, fontname="STIXGeneral")

        if show_source is True:
            ax.scatter(sou_coords[1], sou_coords[0], s=25, c="r", marker="x")

        if show_freq is True:
            textstr = r"$\frac{\omega}{2 \pi} = $" + "{:4.1f}".format(freq)
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)

        if show_cbar:
            axins = inset_axes(ax, width="3%", height="100%", loc='lower left',
                               bbox_to_anchor=(1.02, 0., 1, 1), bbox_transform=ax.transAxes,
                               borderpad=0)
            cbar = fig.colorbar(im, cax=axins)
            cbar.ax.set_title(label_cbar, fontname="STIXGeneral")

        if file_name is not None:
            fig.savefig(
                file_name,
                format="pdf",
                pad_inches=0.01
            )

        plt.show()


    plot1(
        vel=np.real(data),
        extent=extent,
        title="",
        aspect_ratio=4,
        cmap="Greys",
        figsize=figsize,
        file_name=output_filename,
        show_cbar=False,
        vmin=-scale,
        vmax=scale,
        show_freq=True,
        freq=k_vals[num_k_val],
        show_source=True,
        sou_coords=source_coords[num_source]
    )
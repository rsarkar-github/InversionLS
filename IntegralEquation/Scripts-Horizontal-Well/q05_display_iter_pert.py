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
        restart_code=None,
        check_iter_files=False,
        num_procs_check_iter_files=4
    )

    # ------------------------------------
    # Read parameters and data to plot
    # ------------------------------------

    # Check arguments
    if len(sys.argv) < 2:
        raise ValueError("Program missing command line arguments.")

    num_iter = int(sys.argv[1])

    # ------------------------------------
    # Get model parameters
    # ------------------------------------
    scale_fac_inv = obj.scale_fac_inv
    xmax = 1.0
    zmax = (obj.b - obj.a)
    extent = [0, xmax * scale_fac_inv, zmax * scale_fac_inv, 0]
    dx = xmax * scale_fac_inv / (obj.n - 1)
    dz = zmax * scale_fac_inv / (obj.nz - 1)

    # ------------------------------------
    # Read model pert
    # ------------------------------------
    if num_iter == -2:
        # Load true pert
        pert = obj.true_model_pert
        output_filename = (basedir + "Fig/q05_true_pert.pdf")

    else:
        pert_fname = obj.model_pert_filename(iter_count=num_iter)
        with np.load(pert_fname) as f:
            pert = f["arr_0"]
        output_filename = (basedir + "Fig/q05_iter_num_" + str(num_iter) + ".pdf")

    # -----------------------------------------
    # Set figsize, fontsize
    # -----------------------------------------
    figsize = (12, 4)
    fontsize = 14


    # -----------------------------------------
    # Plot model updates
    # -----------------------------------------
    def plot1(
            vel, extent, title,
            aspect_ratio=1, cmap="jet", figsize=(16, 9),
            show_cbar=True,
            label_cbar="",
            file_name=None, vmin=None, vmax=None,
            show_iter=False,
            iter_num=None,
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

        if show_iter is True:
            if iter_num == -2:
                textstr = r"$\psi^{\text{true}}$"
            else:
                textstr = "Iter set = " + str(iter_num)
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


    scale = 0.006

    plot1(
        vel=pert,
        extent=extent,
        title="",
        aspect_ratio=4,
        cmap="seismic",
        figsize=figsize,
        file_name=output_filename,
        label_cbar=r"[$s^2 / km^2$]",
        vmin=-scale,
        vmax=scale,
        show_iter=True,
        iter_num=num_iter
    )
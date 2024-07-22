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
    # Get parameters
    # ------------------------------------
    scale_fac_inv = obj.scale_fac_inv
    xmax = 1.0
    zmax = (obj.b - obj.a)
    dx = xmax * scale_fac_inv / (obj.n - 1)
    dz = zmax * scale_fac_inv / (obj.nz - 1)
    num_k_vals = obj.num_k_values
    k_vals = np.asarray(obj.k_values) / (2 * np.pi)

    num_sources = obj.num_sources
    xgrid = np.linspace(start=0, stop=xmax * scale_fac_inv, num=num_sources, endpoint=True)
    zval = 21 * dz
    source_coords = np.zeros(shape=(num_sources, 2), dtype=np.float32)
    source_coords[:, 0] = zval
    source_coords[:, 1] = xgrid

    extent = [source_coords[0, 1], source_coords[num_sources - 1, 1], k_vals[num_k_vals - 1], k_vals[0]]

    # ------------------------------------
    # Load files
    # ------------------------------------
    if num_iter == -1:
        obj2_fname = obj.obj2_filename(iter_count=-1, iter_step=0)
        obj1_fname = obj.obj1_filename(iter_count=-1)

    else:
        obj2_fname = obj.obj2_filename(iter_count=num_iter, iter_step=1)
        obj1_fname = obj.obj1_filename(iter_count=num_iter)

    # ------------------------------------
    # Plot
    # ------------------------------------
    with np.load(obj2_fname) as f:
        obj2 = f["arr_0"]
        output_filename_obj2 = basedir + "Fig/q06_obj2_" + "iter_num_" + str(num_iter) + ".pdf"

    with np.load(obj1_fname) as f:
        obj1 = f["arr_0"]
        output_filename_obj1 = basedir + "Fig/q06_obj1_" + "iter_num_" + str(num_iter) + ".pdf"

    figsize = (9, 5)
    fontsize = 14

    def plot1(
            vel, extent, title,
            aspect_ratio=1, cmap="jet", figsize=(16, 9),
            show_cbar=True,
            label_cbar="",
            file_name=None, vmin=None, vmax=None,
            show_iter=False,
            iter_num=None,
    ):

        if vmin is None:
            vmin = np.min(vel)
        if vmax is None:
            vmax = np.max(vel)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        im = ax.imshow(
            vel, aspect=aspect_ratio, cmap=cmap, interpolation='bicubic',
            extent=extent, vmin=vmin, vmax=vmax
        )

        ax.set_title(title)
        ax.set_xlabel(r'Source $x$ coordinate [km]', fontsize=fontsize, fontname="STIXGeneral")
        ax.set_ylabel(r'$\omega / 2 \pi$   [Hz]', fontsize=fontsize, fontname="STIXGeneral")
        ax.invert_yaxis()

        if show_iter is True:
            if iter_num == -1:
                textstr = "Initial"
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

    scale = 1e-10
    aspect_ratio = 0.02

    plot1(
        vel=obj1 / scale,
        extent=extent,
        title="",
        aspect_ratio=aspect_ratio,
        cmap="Greys",
        figsize=figsize,
        file_name=output_filename_obj1,
        show_cbar=True,
        vmin=0,
        vmax=1,
        show_iter=True,
        iter_num=num_iter,
    )

    scale = 1e-10
    plot1(
        vel=obj2 / scale,
        extent=extent,
        title="Greys",
        aspect_ratio=aspect_ratio,
        cmap="Greys",
        figsize=figsize,
        file_name=output_filename_obj2,
        show_cbar=True,
        vmin=0,
        vmax=1,
        show_iter=True,
        iter_num=num_iter,
    )

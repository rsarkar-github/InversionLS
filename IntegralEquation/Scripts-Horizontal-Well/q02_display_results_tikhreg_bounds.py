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
    # Read model pert
    # ------------------------------------
    model_pert_no_mnorm_no_bounds = np.load(basedir + "data/p06a-update-mnorm-0-bounds-off.npz.npy")
    model_pert_mnorm_no_bounds = np.load(basedir + "data/p06b-update-mnorm-0.001-bounds-off.npz.npy")

    with np.load(obj.model_pert_filename(iter_count=0)) as data:
        model_pert_mnorm_bounds = data["arr_0"]

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
            file_name=None, vmin=None, vmax=None
    ):

        if vmin is None:
            vmin = np.min(vel)
        if vmax is None:
            vmax = np.max(vel)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        im = ax.imshow(vel, aspect=aspect_ratio, cmap=cmap, interpolation='bicubic', extent=extent, vmin=vmin, vmax=vmax)

        ax.set_title(title)
        ax.set_xlabel('x [km]', fontsize=fontsize, fontname="STIXGeneral")
        ax.set_ylabel('z [km]', fontsize=fontsize, fontname="STIXGeneral")

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

    scale = 1e-3

    plot1(
        vel=model_pert_no_mnorm_no_bounds,
        extent=extent,
        title="",
        aspect_ratio=4,
        cmap="seismic",
        figsize=figsize,
        file_name= basedir + "Fig/q02_model_pert_no_mnorm_no_bounds.pdf",
        label_cbar=r"[$s^2 / km^2$]",
        vmin=-scale,
        vmax=scale
    )

    plot1(
        vel=model_pert_mnorm_no_bounds,
        extent=extent,
        title="",
        aspect_ratio=4,
        cmap="seismic",
        figsize=figsize,
        file_name=basedir + "Fig/q02_model_pert_mnorm_no_bounds.pdf",
        label_cbar=r"[$s^2 / km^2$]",
        vmin=-scale,
        vmax=scale
    )

    plot1(
        vel=model_pert_mnorm_bounds,
        extent=extent,
        title="",
        aspect_ratio=4,
        cmap="seismic",
        figsize=figsize,
        file_name=basedir + "Fig/q02_model_pert_mnorm_bounds.pdf",
        label_cbar=r"[$s^2 / km^2$]",
        vmin=-scale,
        vmax=scale
    )

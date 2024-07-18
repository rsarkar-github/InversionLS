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
    num_k_vals = obj.num_k_values

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
    vz_2d += obj.vz
    psi = obj.true_model_pert
    vel = ((1 / (vz_2d ** 2.0)) - psi) ** (-0.5)
    dv = vz_2d - vel


    # -----------------------------------------
    # Plot v(z) velocity with source locations
    # -----------------------------------------

    figsize = (11, 4)
    fontsize = 14

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(vz_2d, aspect=4, cmap="jet", interpolation='bicubic', extent=extent, vmin=2.5, vmax=6.0)
    ax.scatter(source_coords[:, 1], source_coords[:, 0], s=2, c="k", marker="x")
    ax.set_xlabel('x [km]', fontsize=fontsize, fontname="STIXGeneral")
    ax.set_ylabel('z [km]', fontsize=fontsize, fontname="STIXGeneral")

    axins = inset_axes(ax, width="3%", height="100%", loc='lower left',
                       bbox_to_anchor=(1.02, 0., 1, 1), bbox_transform=ax.transAxes,
                       borderpad=0)
    cbar = fig.colorbar(im, cax=axins)
    cbar.ax.set_title("[km/s]", fontname="STIXGeneral")

    fig.savefig(
        basedir + "Fig/q01_vz_true_source_overlay.pdf",
        format="pdf",
        pad_inches=0.01
    )
    plt.show()

    # -----------------------------------------
    # Plot v(z) velocity as trace
    # -----------------------------------------
    zcoords = [dz * i for i in range(obj.nz)]
    fig, ax = plt.subplots(1, 1)
    ax.plot(obj.vz, zcoords, "-r", markersize=4, linewidth=2)
    ax.set_ylabel('z [km]', fontsize=fontsize, fontname="STIXGeneral")
    ax.set_xlabel('Vp [km/s]', fontsize=fontsize, fontname="STIXGeneral")
    ax.invert_yaxis()
    ax.set_xlim((0, 6))
    ax.set_ylim((zmax, 0))
    ax.grid()
    ax.set_aspect(100)
    fig.savefig(
        basedir + "Fig/q01_vz_true_1d.pdf",
        format="pdf",
        pad_inches=0.01
    )
    plt.show()

    # -----------------------------------------
    # Plot perturbation and dv
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


    plot1(
        vel=dv,
        extent=extent,
        title="",
        aspect_ratio=4,
        cmap="seismic",
        figsize=figsize,
        file_name= basedir + "Fig/q01_vel_true_pert.pdf",
        label_cbar="[km/s]",
        vmin=-0.06,
        vmax=0.06
    )

    plot1(
        vel=psi,
        extent=extent,
        title="",
        aspect_ratio=4,
        cmap="seismic",
        figsize=figsize,
        file_name=basedir + "Fig/q01_true_psi.pdf",
        label_cbar=r"$[s^2 / km^2]$",
        vmin=-0.006,
        vmax=0.006
    )

    # -----------------------------------------
    # Plot frequency spectra
    # -----------------------------------------

    # Design amplitudes
    amplitude_list = np.zeros(shape=(num_k_vals,), dtype=obj.precision_real) + 1.0

    if num_k_vals > 10:
        fac = 0.1
        low = 0.1

        n1 = int(num_k_vals * fac)
        da = (1.0 - 0.1) / (n1 - 1)
        for i in range(n1):
            amplitude_list[i] = low + i * da

        for i in range(num_k_vals - 1, num_k_vals - 1 - n1, -1):
            amplitude_list[i] = low + (num_k_vals - 1 - i) * da

    k_vals = np.asarray(obj.k_values)
    freq = (k_vals / scale_fac_inv) / (2 * np.pi)

    fig, ax = plt.subplots(1, 1)
    ax.plot(freq, amplitude_list, "-r*")
    ax.set_xlabel(r"$\omega / 2\pi$   [Hz]", fontname="STIXGeneral", fontsize=fontsize)
    ax.set_ylabel(r"$a(\omega)$", fontname="STIXGeneral", fontsize=fontsize)
    ax.grid()
    ax.set_aspect(10)
    fig.savefig(
        basedir + "Fig/q01_spectra.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.01
    )
    plt.show()
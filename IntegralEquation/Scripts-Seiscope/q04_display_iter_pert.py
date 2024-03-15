import os.path
import sys
import numpy as np
import matplotlib.pyplot as plt
from ..Inversion.ScatteringIntegralGeneralVzInversion import ScatteringIntegralGeneralVzInversion2d


def plot1(vel, extent, title, aspect_ratio=1, aspect_cbar=10, file_name=None, vmin=None, vmax=None):

    if vmin is None:
        vmin = np.min(vel)
    if vmax is None:
        vmax = np.max(vel)

    plt.figure(figsize=(6, 3))  # define figure size
    plt.imshow(vel, cmap="jet", interpolation='bicubic', extent=extent, vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(aspect=aspect_cbar, pad=0.02)
    cbar.set_label('Vp [km/s]', labelpad=10)
    plt.title(title)
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.gca().set_aspect(aspect_ratio)

    if file_name is not None:
        plt.savefig(file_name, format="pdf", bbox_inches="tight", pad_inches=0.01)

    plt.show()


def plot2(vel, extent, title, aspect_ratio=1, aspect_cbar=10, file_name=None, vmin=None, vmax=None):

    if vmin is None and vmax is None:
        vmin = -np.max(np.abs(vel))
        vmax = np.max(np.abs(vel))

    plt.figure(figsize=(6, 3))  # define figure size
    plt.imshow(vel, cmap="seismic", interpolation='bicubic', extent=extent, vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(aspect=aspect_cbar, pad=0.02)
    cbar.set_label('Vp [km/s]', labelpad=10)
    plt.title(title)
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.gca().set_aspect(aspect_ratio)

    if file_name is not None:
        plt.savefig(file_name, format="pdf", bbox_inches="tight", pad_inches=0.01)

    plt.show()


if __name__ == "__main__":

    basedir = "InversionLS/Expt/seiscope/"
    obj = ScatteringIntegralGeneralVzInversion2d(
        basedir=basedir,
        restart=True,
        restart_code=None,
        check_iter_files=True,
        num_procs_check_iter_files=16
    )

    print("Num k values = ", obj.num_k_values, ", Num sources = ", obj.num_sources)

    # Check arguments
    if len(sys.argv) < 3:
        raise ValueError("Program missing command line arguments.")

    num_iter = int(sys.argv[1])
    aspect = int(sys.argv[2])

    # Get perturbation, true perturbation
    pert_fname = obj.model_pert_filename(iter_count=num_iter)
    with np.load(pert_fname) as f:
        pert = f["arr_0"]

    pert_true = obj.true_model_pert

    # Get vz, true vel
    vz_fname = os.path.join(basedir, "vp_vz_2d.npz")
    with np.load(vz_fname) as f:
        vp_vz = f["arr_0"]

    vp_fname = os.path.join(basedir, "vp_true_2d.npz")
    with np.load(vp_fname) as f:
        vp_true = f["arr_0"]

    # Convert to velocity
    def vel_from_pert(psi):
        vp = ((1.0 / vp_vz) ** 2.0) - psi
        vp = (1.0 / vp) ** 0.5
        return vp

    vp_update = vel_from_pert(pert)

    # Plotting
    xmax = 1.0
    zmax = (obj.b - obj.a)
    extent = [0, xmax, zmax, 0]

    plot1(
        vel = vp_update, extent=extent, title="Vp update",
        aspect_ratio=aspect, aspect_cbar=10, file_name=None, vmin=2.5, vmax=6.0
    )

    plot1(
        vel=vp_true, extent=extent, title="Vp true",
        aspect_ratio=aspect, aspect_cbar=10, file_name=None, vmin=2.5, vmax=6.0
    )

    scale = np.max(np.abs(pert_true))

    plot2(
        vel=pert_true, extent=extent, title="True pert", vmin=-scale, vmax=scale,
        aspect_ratio=aspect, aspect_cbar=10, file_name=None
    )
    plot2(
        vel=pert, extent=extent, title="Inverted pert",
        aspect_ratio=aspect, aspect_cbar=10, file_name=None
    )

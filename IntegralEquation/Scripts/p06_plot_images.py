import sys
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # ----------------------------------------------
    # Check arguments and read in parameters
    # ----------------------------------------------
    if len(sys.argv) < 5:
        raise ValueError("Program missing command line arguments.")

    model_mode = int(sys.argv[1])
    freq_mode = int(sys.argv[2])
    solver_mode = int(sys.argv[3])
    mu_mode = int(sys.argv[4])

    if model_mode == 0:
        filepath1 = "InversionLS/Data/sigsbee-new-2d.npz"
        filepath2 = "InversionLS/Data/p02-sigsbee-source.npz"
        filepath4_ = "InversionLS/Data/p04-sigsbee-"
        filepath5_ = "InversionLS/Fig/p06-sigsbee-"

    elif model_mode == 1:
        filepath1 = "InversionLS/Data/marmousi-new-2d.npz"
        filepath2 = "InversionLS/Data/p02-marmousi-source.npz"
        filepath4_ = "InversionLS/Data/p04-marmousi-"
        filepath5_ = "InversionLS/Fig/p06-marmousi-"

    elif model_mode == 2:
        filepath1 = "InversionLS/Data/seiscope-new-2d.npz"
        filepath2 = "InversionLS/Data/p02-seiscope-source.npz"
        filepath4_ = "InversionLS/Data/p04-seiscope-"
        filepath5_ = "InversionLS/Fig/p06-seiscope-"

    else:
        raise ValueError("model mode = ", model_mode, " is not supported. Must be 0, 1, or 2.")

    if freq_mode == 0:
        freq = 5.0  # in Hz
    elif freq_mode == 1:
        freq = 7.5  # in Hz
    elif freq_mode == 2:
        freq = 10.0  # in Hz
    elif freq_mode == 3:
        freq = 15.0  # in Hz
    else:
        raise ValueError("freq mode = ", freq_mode, " is not supported. Must be 0, 1, 2, or 3.")

    if solver_mode == 1:
        solver_name = "lsqr"
    elif solver_mode == 2:
        solver_name = "lsmr"
    else:
        raise ValueError("solver mode = ", solver_mode, " is not supported. Must be 1 or 2.")

    if mu_mode == 0:
        mu_ = 1.0
        scale_ = 5e-5
    elif mu_mode == 1:
        mu_ = 5.0
        scale_ = 1e-5
    elif mu_mode == 2:
        mu_ = 10.0
        scale_ = 1e-6
    else:
        raise ValueError("mu mode = ", mu_mode, " is not supported. Must be 0, 1 or 2.")

    # ----------------------------------------------
    # Load velocities
    # ----------------------------------------------
    with np.load(filepath1) as data:
        vel = data["arr_0"]

    # ----------------------------------------------
    # Load solved wavefields
    # ----------------------------------------------
    with np.load(
            filepath4_ + "sol-" + solver_name + "-" +
            "{:4.2f}".format(freq) + "-mu" + "{:4.2f}".format(mu_) + ".npz"
    ) as data:
        data_sol = data["arr_0"]

    # ----------------------------------------------
    # Plot overlays
    # ----------------------------------------------
    n_ = 351
    nz_ = 251
    dx = 15.0  # something dummy (does not matter)
    dz = 15.0  # something dummy (does not matter)
    lenx = dx * (n_ - 1)
    lenz = dz * (nz_ - 1)

    def plot(data, vel, fname, scale):

        extent = [0, lenx / 1000.0, lenz / 1000.0, 0]
        plt.figure(figsize=(12, 3))  # define figure size
        plt.imshow(data, cmap="Greys", interpolation="none", extent=extent, vmin=-scale, vmax=scale)
        plt.colorbar(aspect=10, pad=0.02)

        plt.imshow(vel, cmap="jet", interpolation='nearest', extent=extent, vmin=1.5, vmax=4.5, alpha=0.25)


        plt.xlabel(r'$x$ [km]')
        plt.ylabel(r'$z$ [km]')

        plt.savefig(fname, format="pdf", bbox_inches="tight", pad_inches=0.01)
        plt.show()


    savefig_fname_sol = (filepath5_ + "sol-" + solver_name + "-" +
                         "{:4.2f}".format(freq) + "-mu" + "{:4.2f}".format(mu_) + ".pdf")
    plot(data=np.real(data_sol), vel=vel, fname=savefig_fname_sol, scale=scale_)

import sys
import numpy as np
import matplotlib.pyplot as plt
from ..Inversion.ScatteringIntegralGeneralVzInversion import ScatteringIntegralGeneralVzInversion2d


if __name__ == "__main__":

    basedir = "InversionLS/Expt/horizontal-well/"
    obj = ScatteringIntegralGeneralVzInversion2d(
        basedir=basedir,
        restart=True,
        restart_code=None
    )

    print("Num k values = ", obj.num_k_values, ", Num sources = ", obj.num_sources)

    # Check arguments
    if len(sys.argv) < 3:
        raise ValueError("Program missing command line arguments.")

    num_iter = int(sys.argv[1])
    aspect = float(sys.argv[2])

    pert_fname = obj.model_pert_filename(iter_count=0)
    with np.load(pert_fname) as f:
        pert = f["arr_0"]

    pert_true = obj.true_model_pert

    scale = np.max(np.abs(pert_true))

    xmax = 1.0
    zmax = (obj.b - obj.a)
    extent = [0, xmax, zmax, 0]

    plt.imshow(np.real(pert_true), cmap="Greys", extent=extent, aspect=aspect, vmin=-scale, vmax=scale)
    plt.xlabel(r'$x_1$ [km]')
    plt.ylabel(r'$z$ [km]')
    plt.show()

    plt.imshow(np.real(pert), cmap="Greys", extent=extent, aspect=aspect, vmin=-scale, vmax=scale)
    plt.xlabel(r'$x_1$ [km]')
    plt.ylabel(r'$z$ [km]')
    plt.show()

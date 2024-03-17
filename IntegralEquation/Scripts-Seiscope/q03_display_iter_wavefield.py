import sys
import numpy as np
import matplotlib.pyplot as plt
from ..Inversion.ScatteringIntegralGeneralVzInversion import ScatteringIntegralGeneralVzInversion2d


if __name__ == "__main__":

    basedir = "InversionLS/Expt/Seiscope/"
    obj = ScatteringIntegralGeneralVzInversion2d(
        basedir=basedir,
        restart=True,
        restart_code=None,
        check_iter_files=True,
        num_procs_check_iter_files=16
    )

    print("Num k values = ", obj.num_k_values, ", Num sources = ", obj.num_sources)

    # Check arguments
    if len(sys.argv) < 5:
        raise ValueError("Program missing command line arguments.")

    num_k_val = int(sys.argv[1])
    num_source = int(sys.argv[2])
    num_iter = int(sys.argv[3])
    aspect = float(sys.argv[4])

    data_true = obj.get_true_wavefield(num_k=num_k_val, num_source=num_source)
    data = obj.get_inverted_wavefield(iter_count=num_iter, num_k=num_k_val, num_source=num_source)
    data1 = obj.get_inverted_wavefield(iter_count=num_iter - 1, num_k=num_k_val, num_source=num_source)

    # if num_iter >= 1:
    #     data_true = obj.get_inverted_wavefield(iter_count=num_iter - 1, num_k=num_k_val, num_source=num_source)
    # else:
    #     data_true = obj.get_true_wavefield(num_k=num_k_val, num_source=num_source)

    scale = np.max(np.abs(data_true))
    scale *= 0.2

    xmax = 1.0
    zmax = (obj.b - obj.a)
    extent = [0, xmax, zmax, 0]

    plt.imshow(np.real(data_true), cmap="Greys", extent=extent, aspect=aspect, vmin=-scale, vmax=scale)
    plt.xlabel(r'$x_1$ [km]')
    plt.ylabel(r'$z$ [km]')
    plt.show()

    plt.imshow(np.real(data), cmap="Greys", extent=extent, aspect=aspect, vmin=-scale, vmax=scale)
    plt.xlabel(r'$x_1$ [km]')
    plt.ylabel(r'$z$ [km]')
    plt.show()

    plt.imshow(np.real(data_true - data), cmap="Greys", extent=extent, aspect=aspect, vmin=-scale, vmax=scale)
    plt.xlabel(r'$x_1$ [km]')
    plt.ylabel(r'$z$ [km]')
    plt.show()

    plt.imshow(np.real(data1 - data), cmap="Greys", extent=extent, aspect=aspect, vmin=-1e-8, vmax=1e-8)
    plt.xlabel(r'$x_1$ [km]')
    plt.ylabel(r'$z$ [km]')
    plt.show()

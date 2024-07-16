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
    scale_fac_inv = obj.scale_fac_inv

    print("Num k values = ", obj.num_k_values, ", Num sources = ", obj.num_sources)

    # Check arguments
    if len(sys.argv) < 3:
        raise ValueError("Program missing command line arguments.")

    num_k_val = int(sys.argv[1])
    num_source = int(sys.argv[2])

    xmax = 1.0
    zmax = (obj.b - obj.a)
    extent = [0, xmax * scale_fac_inv, zmax * scale_fac_inv, 0]

    with np.load(obj.source_filename(num_k=num_k_val)) as data:
        source = data["arr_0"]
    src = source[num_source, :, :]

    scale = 1.0
    plt.imshow(np.real(src), cmap="Greys", extent=extent, aspect=1, vmax=scale, vmin=-scale)
    plt.xlabel(r'$x_1$ [km]')
    plt.ylabel(r'$z$ [km]')
    plt.show()

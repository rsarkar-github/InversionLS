import numpy as np
import matplotlib.pyplot as plt
from ..Inversion.ScatteringIntegralGeneralVzInversion import ScatteringIntegralGeneralVzInversion2d


if __name__ == "__main__":

    basedir = "InversionLS/Expt/marmousi/"
    obj = ScatteringIntegralGeneralVzInversion2d(
        basedir=basedir,
        restart=True,
        restart_code=None
    )

    print("Num k values = ", obj.num_k_values, ", Num sources = ", obj.num_sources)

    num_k_val = 30
    num_source = 10

    data = obj.get_true_wavefield(num_k=num_k_val, num_source=num_source)

    xmax = 1.0
    zmax = (obj.b - obj.a)
    extent = [0, xmax, zmax, 0]
    plt.imshow(np.real(data), cmap="Greys", extent=extent)
    plt.show()

import sys
import numpy as np
import matplotlib.pyplot as plt
from ..Inversion.ScatteringIntegralGeneralVzInversion import ScatteringIntegralGeneralVzInversion2d


if __name__ == "__main__":

    basedir = "InversionLS/Expt/horizontal-well/"
    inv_obj = ScatteringIntegralGeneralVzInversion2d(
        basedir=basedir,
        restart=True,
        restart_code=None
    )

    obj2_fname = inv_obj.obj2_filename(iter_count=-1, iter_step=0)
    with np.load(obj2_fname) as f:
        obj2 = f["arr_0"]

    plt.imshow(obj2, cmap="jet")
    plt.xlabel("Num source")
    plt.ylabel("Num k")
    plt.colorbar()
    plt.show()

    obj1_fname = inv_obj.obj1_filename(iter_count=0)
    with np.load(obj1_fname) as f:
        obj1 = f["arr_0"]

    plt.imshow(obj1, cmap="jet")
    plt.xlabel("Num source")
    plt.ylabel("Num k")
    plt.colorbar()

    plt.show()

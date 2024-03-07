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

    # ---------------------------------------------
    # Initial objective function

    obj2_fname = inv_obj.obj2_filename(iter_count=-1, iter_step=0)
    with np.load(obj2_fname) as f:
        obj2 = f["arr_0"]

    plt.imshow(obj2, cmap="jet")
    plt.title("PDE residual")
    plt.xlabel("Num source")
    plt.ylabel("Num k")
    plt.colorbar()
    plt.show()

    obj1_fname = inv_obj.obj1_filename(iter_count=-1)
    with np.load(obj1_fname) as f:
        obj1 = f["arr_0"]

    scale = np.max(obj1)

    plt.imshow(obj1, cmap="jet", vmin=0, vmax=scale)
    plt.title("Data residual")
    plt.xlabel("Num source")
    plt.ylabel("Num k")
    plt.colorbar()
    plt.show()

    # ---------------------------------------------
    # Iteration 0 objective function

    temp = obj1

    obj2_fname = inv_obj.obj2_filename(iter_count=0, iter_step=0)
    with np.load(obj2_fname) as f:
        obj2 = f["arr_0"]

    plt.imshow(obj2, cmap="jet", vmin=0, vmax=scale)
    plt.title("PDE residual")
    plt.xlabel("Num source")
    plt.ylabel("Num k")
    plt.colorbar()
    plt.show()

    obj1_fname = inv_obj.obj1_filename(iter_count=0)
    with np.load(obj1_fname) as f:
        obj1 = f["arr_0"]

    plt.imshow(obj1, cmap="jet", vmin=0, vmax=scale)
    plt.title("Data residual")
    plt.xlabel("Num source")
    plt.ylabel("Num k")
    plt.colorbar()
    plt.show()

    print(np.linalg.norm(obj1 - obj2))
    print(np.linalg.norm(obj1 - temp))
    print(np.linalg.norm(obj2 - temp))
    print(np.sum(temp), np.sum(obj1), np.sum(obj2))

    obj2_fname = inv_obj.obj2_filename(iter_count=0, iter_step=1)
    with np.load(obj2_fname) as f:
        obj2 = f["arr_0"]

    plt.imshow(obj2, cmap="jet", vmin=0, vmax=scale)
    plt.title("PDE residual")
    plt.xlabel("Num source")
    plt.ylabel("Num k")
    plt.colorbar()
    plt.show()

    print(np.sum(obj2))

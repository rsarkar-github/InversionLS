import os
import numpy as np
from ..Inversion.ScatteringIntegralGeneralVzInversion import ScatteringIntegralGeneralVzInversion2d
import matplotlib.pyplot as plt


if __name__ == "__main__":

    basedir = "InversionLS/Expt/marmousi1/"
    obj = ScatteringIntegralGeneralVzInversion2d(
        basedir=basedir,
        restart=True,
        restart_code=None
    )

    print("\n\n---------------------------------------------")
    print("---------------------------------------------")
    print("Set true perturbation in slowness squared...")
    print("\n")

    with np.load(os.path.join(basedir, "vp_vz_2d.npz")) as data:
        vp_vz = data["arr_0"]
    with np.load(os.path.join(basedir, "vp_true_2d.npz")) as data:
        vp_compact = data["arr_0"]

    phi = (1.0 / (vp_vz ** 2.0)) - (1.0 / (vp_compact ** 2.0))

    xmax = 1.0
    zmax = (obj.b - obj.a)
    extent = [0, xmax, zmax, 0]
    plt.imshow(phi, cmap="Greys", extent=extent)
    plt.colorbar()
    plt.show()
    plt.savefig(os.path.join(basedir, "phi.pdf"), format="pdf", bbox_inches="tight", pad_inches=0.01)

    obj.add_true_model_pert(model_pert=phi)

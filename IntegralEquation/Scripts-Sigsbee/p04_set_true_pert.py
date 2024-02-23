import os
import numpy as np
from ..Inversion.ScatteringIntegralGeneralVzInversion import ScatteringIntegralGeneralVzInversion2d
import matplotlib.pyplot as plt


if __name__ == "__main__":

    basedir = "InversionLS/Expt/sigsbee/"
    obj = ScatteringIntegralGeneralVzInversion2d(
        basedir=basedir,
        restart=True,
        restart_code=None
    )
    scale_fac_inv = obj.scale_fac_inv

    print("\n\n---------------------------------------------")
    print("---------------------------------------------")
    print("Set true perturbation in slowness squared...")
    print("\n")

    with np.load(os.path.join(basedir, "vp_vz_2d.npz")) as data:
        vp_vz = data["arr_0"]
    with np.load(os.path.join(basedir, "vp_true_2d.npz")) as data:
        vp_compact = data["arr_0"]

    psi = (1.0 / (vp_vz ** 2.0)) - (1.0 / (vp_compact ** 2.0))
    psi = psi.astype(obj.precision_real)

    xmax = 1.0
    zmax = (obj.b - obj.a)
    extent = [0, scale_fac_inv * xmax, scale_fac_inv * zmax, 0]

    def plot(psi, extent, title, file_name=None):
        fig = plt.figure(figsize=(6, 3))  # define figure size
        image = plt.imshow(psi, cmap="Greys", interpolation='nearest', extent=extent)

        cbar = plt.colorbar(aspect=10, pad=0.02)
        cbar.set_label('[s' + r'$^2 /$' + 'km' + r'$^2$' +']', labelpad=10)
        plt.title(title)
        plt.xlabel('x [km]')
        plt.ylabel('z [km]')

        if file_name is not None:
            plt.savefig(file_name, format="pdf", bbox_inches="tight", pad_inches=0.01)

        plt.show()

    plot(psi, extent, "True perturbation", os.path.join(basedir, "psi.pdf"))

    obj.add_true_model_pert(model_pert=psi)

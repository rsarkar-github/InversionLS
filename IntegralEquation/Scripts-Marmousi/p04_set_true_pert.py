import os
import numpy as np
from ..Inversion.ScatteringIntegralGeneralVzInversion import ScatteringIntegralGeneralVzInversion2d
import matplotlib.pyplot as plt


def plot(psi, extent, title, aspect_ratio=10, file_name=None):
    plt.figure(figsize=(6, 3))  # define figure size
    plt.imshow(psi, cmap="Greys", interpolation='bicubic', extent=extent)

    cbar = plt.colorbar(aspect=10, pad=0.02)
    cbar.set_label('[s' + r'$^2 /$' + 'km' + r'$^2$' + ']', labelpad=10)
    plt.title(title)
    plt.xlabel('x [km]')
    plt.ylabel('z [km]')
    plt.gca().set_aspect(aspect_ratio)

    if file_name is not None:
        plt.savefig(file_name, format="pdf", bbox_inches="tight", pad_inches=0.01)

    plt.show()


if __name__ == "__main__":

    basedir = "InversionLS/Expt/marmousi/"
    obj = ScatteringIntegralGeneralVzInversion2d(
        basedir=basedir,
        restart=True,
        restart_code=3
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

    lower_bound = -2.0 * np.abs(psi)
    upper_bound = 2.0 * np.abs(psi)

    xmax = 1.0
    zmax = (obj.b - obj.a)
    extent = [0, scale_fac_inv * xmax, scale_fac_inv * zmax, 0]
    plot(psi, extent, "True perturbation", 10, os.path.join(basedir, "Fig/psi.pdf"))
    plot(lower_bound, extent, "Lower bound", 10)
    plot(upper_bound, extent, "Upper bound", 10)

    obj.add_true_model_pert_bounds(model_pert=psi, lower_bound=lower_bound, upper_bound=upper_bound)

import os
import numpy as np
import matplotlib.pyplot as plt
from ..Inversion.ScatteringIntegralGeneralVzInversion import ScatteringIntegralGeneralVzInversion2d


if __name__ == "__main__":

    basedir = "InversionLS/Expt/sigsbee/"
    obj = ScatteringIntegralGeneralVzInversion2d(
        basedir=basedir,
        restart=True,
        restart_code=None
    )

    figdir = os.path.join(basedir, "Fig")
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    num_k_vals = obj.num_k_values
    scale_fac_inv = obj.scale_fac_inv
    k_vals = obj.k_values

    # Design amplitudes
    amplitude_list = np.zeros(shape=(num_k_vals,), dtype=obj.precision_real) + 1.0

    if num_k_vals > 10:
        fac = 0.1
        low = 0.1

        n1 = int(num_k_vals * fac)
        da = (1.0 - 0.1) / (n1 - 1)
        for i in range(n1):
            amplitude_list[i] = low + i * da

        for i in range(num_k_vals - 1, num_k_vals - 1 - n1, -1):
            amplitude_list[i] = low + (num_k_vals - 1 - i) * da

    freq = (k_vals / scale_fac_inv) / (2 * np.pi)

    plt.plot(freq, amplitude_list, "-r*")
    plt.xlabel("Frequency [Hz]", fontname="STIXGeneral", fontsize=12)
    plt.ylabel("Amplitude", fontname="STIXGeneral", fontsize=12)
    plt.grid()

    fname = os.path.join(figdir, "spectra.pdf")
    plt.savefig(fname, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.show()

    print("\n\n---------------------------------------------")
    print("---------------------------------------------")
    print("Creating sources...")

    amplitude_list = amplitude_list.astype(obj.precision)

    nx = obj.n
    nz = obj.nz
    a = obj.a
    b = obj.b
    dx = 1.0 / (nx - 1)
    dz = (b - a) / (nz - 1)
    std = 2 * dx

    num_sources = 71
    xgrid = np.linspace(start=-0.5, stop=0.5, num=num_sources, endpoint=True)
    zval = 1 * dz
    source_coords = np.zeros(shape=(num_sources, 2), dtype=np.float32)
    source_coords[:, 0] = zval
    source_coords[:, 1] = xgrid

    std_list = np.zeros(shape=(num_sources,), dtype=np.float32) + std

    obj.add_sources_gaussian(
        num_sources=num_sources,
        amplitude_list=amplitude_list,
        source_coords=source_coords,
        std_list=std_list
    )

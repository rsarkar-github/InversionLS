import os
import json
import numpy as np
import shutil
from ..Inversion.ScatteringIntegralGeneralVzInversion import ScatteringIntegralGeneralVzInversion2d


if __name__ == "__main__":

    basedir = "InversionLS/Expt/seiscope/"
    if not os.path.exists(basedir):
        os.makedirs(basedir)

    # Load seiscope files
    with np.load("InversionLS/Data/seiscope-new-vz-2d.npz") as data:
        vp_vz_2d = data["arr_0"]
    with np.load("InversionLS/Data/seiscope-new-2d.npz") as data:
        vp_2d = data["arr_0"]

    shutil.copy("InversionLS/Data/seiscope-new-vz-2d.npz", os.path.join(basedir, "vp_vz_2d.npz"))
    shutil.copy("InversionLS/Data/seiscope-new-2d.npz", os.path.join(basedir, "vp_true_2d.npz"))

    dx = dz = 15.0
    nz, nx = vp_vz_2d.shape
    print("nz =", nz, ", nx =", nx)

    # Save 1D vp_vz as numpy array
    vp_vz = np.reshape(vp_vz_2d[:, 0].astype(np.float32), newshape=(nz, 1))
    np.savez(os.path.join(basedir, "vp_vz.npz"), vp_vz)

    # Calculate a, b values
    extent_z = (nz - 1) * dz / 1000
    extent_x = (nx - 1) * dx / 1000
    scale_fac = 1.0 / extent_x

    a = 0.0
    b = a + scale_fac * extent_z
    print("scale_fac = ", scale_fac, ", a = ", a, ", b = ", b)

    # Set m, sigma, num_threads
    m = 4
    sigma = 2 * (1.0 / nx) / m     # approximately 0.0015
    num_threads = 10

    # Set receiver locs
    rec_locs = [(nz - 2, i) for i in range(0, nx)]

    params = {
        "geometry": {
            "a": a,
            "b": b,
            "n": nx,
            "nz": nz,
            "scale_fac_inv": 1.0 / scale_fac
        },
        "precision": "float",
        "greens func": {
            "m": m,
            "sigma": sigma,
            "vz file path": os.path.join(basedir, "vp_vz.npz")
        },
        "rec_locs": rec_locs
    }
    with open(os.path.join(basedir, "params.json"), "w") as file:
        json.dump(params, file, indent=4)

    # Calculate frequencies
    freq_min = 3.0
    freq_max = 10.0
    tmax = 6  # 6s
    dfreq = 1.0 / tmax

    freqs = []
    curr_freq = freq_min
    while curr_freq <= freq_max:
        freqs.append(curr_freq)
        curr_freq += dfreq

    freqs = np.array(freqs)
    freqs = freqs / scale_fac
    k = 2 * np.pi * freqs

    obj = ScatteringIntegralGeneralVzInversion2d(
        basedir=basedir,
        restart=False,
        restart_code=None
    )
    obj.print_params()

    print("\n\n---------------------------------------------")
    print("---------------------------------------------")
    print("Add k values...")
    obj.add_k_values(k_values_list=k)
    print("k values = ", obj.k_values)

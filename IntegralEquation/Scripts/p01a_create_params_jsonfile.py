import os
import json
import numpy as np


if __name__ == "__main__":

    basedir = "InversionLS/Expt/marmousi1"

    # Load Marmousi files
    with np.load("InversionLS/Data/marmousi-vp-vz-interp.npz") as data:
        vp_vz_interp = data["arr_0"]
    with np.load("InversionLS/Data/marmousi-vp-vz-interp-2d.npz") as data:
        vp_vz_interp_2d = data["arr_0"]
    with np.load("InversionLS/Data/marmousi-vp-interp-compact.npz") as data:
        vp_interp_compact = data["arr_0"]

    dx = dz = 15.0
    nz, nx = vp_vz_interp_2d.shape
    print("nz =", nz, ", nx =", nx)

    # Calculate a, b values
    extent_z = (nz - 1) * dz / 1000
    extent_x = (nx - 1) * dx / 1000
    scale_fac = 1.0 / extent_x

    a = 0.0
    b = a + scale_fac * extent_z
    print("scale_fac = ", scale_fac, ", a = ", a, ", b = ", b)

    # Set m, sigma, num_threads
    m = 3
    sigma = 3 * (1.0 / nx) / m
    num_threads = 5

    params = {
        "geometry": {
            "a": a,
            "b": b,
            "n": nx,
            "nz": nz
        },
        "precision": "float",
        "green func": {
            "m": m,
            "sigma": sigma,
            "num threads": num_threads
        }
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

    np.savez(os.path.join(basedir, "k-values.npz"), k)
    
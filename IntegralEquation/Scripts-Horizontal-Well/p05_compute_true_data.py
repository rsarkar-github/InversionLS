import os
import numpy as np
import multiprocessing as mp
from ..Inversion.ScatteringIntegralGeneralVzInversion import ScatteringIntegralGeneralVzInversion2d
import matplotlib.pyplot as plt


if __name__ == "__main__":

    basedir = "InversionLS/Expt/horizontal-well/"
    obj = ScatteringIntegralGeneralVzInversion2d(
        basedir=basedir,
        restart=True,
        restart_code=4
    )

    print("\n\n---------------------------------------------")
    print("---------------------------------------------")
    print("Compute true data...")
    print("\n")

    num_procs = min(obj.num_sources, mp.cpu_count(), 100)
    max_iter = 5000
    tol = 1e-5
    verbose = False
    obj.compute_true_data(num_procs=num_procs, max_iter=max_iter, tol=tol, verbose=verbose)

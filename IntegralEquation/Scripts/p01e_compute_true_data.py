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
    print("Compute true data...")
    print("\n")

    num_procs = obj.num_sources
    max_iter = 200
    tol = 1e-6
    verbose = False
    obj.compute_true_data(num_procs=obj.num_sources, max_iter=max_iter, tol=tol, verbose=verbose)

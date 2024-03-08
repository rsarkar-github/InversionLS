import os
import numpy as np
from ..Inversion.ScatteringIntegralGeneralVzInversion import ScatteringIntegralGeneralVzInversion2d
import matplotlib.pyplot as plt
import multiprocessing as mp


if __name__ == "__main__":

    basedir = "InversionLS/Expt/horizontal-well/"
    obj = ScatteringIntegralGeneralVzInversion2d(
        basedir=basedir,
        restart=True,
        restart_code=5
    )

    print("\n\n---------------------------------------------")
    print("---------------------------------------------")
    print("Set zero initial perturbation and wavefields...")
    print("\n")

    # obj.set_zero_initial_pert_wavefields(num_procs=10)
    num_procs = min(obj.num_sources, mp.cpu_count(), 100)
    obj.set_zero_initial_pert(num_procs=num_procs)

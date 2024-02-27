import os
import numpy as np
from ..Inversion.ScatteringIntegralGeneralVzInversion import ScatteringIntegralGeneralVzInversion2d
import matplotlib.pyplot as plt


if __name__ == "__main__":

    basedir = "InversionLS/Expt/horizontal-well/"
    obj = ScatteringIntegralGeneralVzInversion2d(
        basedir=basedir,
        restart=True,
        restart_code=None
    )

    print("\n\n---------------------------------------------")
    print("---------------------------------------------")
    print("Perform update...")
    print("\n")

    obj1_fname = obj.obj1_filename(iter_count=-1)
    with np.load(obj1_fname) as f:
        obj1 = f["arr_0"]

    obj2_fname = obj.obj2_filename(iter_count=-1, iter_step=0)
    with np.load(obj2_fname) as f:
        obj2 = f["arr_0"]

    obj1_sum = np.sum(obj1)
    obj2_sum = np.sum(obj2)

    fac = obj1_sum / obj2_sum

    print("obj1 = ", obj1_sum, ", obj2 = ", obj2_sum)

    lambda_arr = np.zeros(shape=(obj.num_k_values, obj.num_sources), dtype=np.float32) + 1.0 / fac
    mu_arr = np.zeros(shape=(obj.num_k_values, obj.num_sources), dtype=np.float32) + 1.0

    obj.perform_inversion_update_wavefield(
        iter_count=0,
        lambda_arr=lambda_arr,
        mu_arr=mu_arr,
        max_iter=10,
        solver="lsmr",
        atol=1e-5,
        btol=1e-5,
        num_procs=16,
        clean=True
    )

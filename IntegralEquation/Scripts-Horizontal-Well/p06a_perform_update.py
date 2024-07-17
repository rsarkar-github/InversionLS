import sys
import numpy as np
from IntegralEquation.Inversion.ScatteringIntegralGeneralVzInversion import ScatteringIntegralGeneralVzInversion2d
import multiprocessing as mp


if __name__ == "__main__":

    basedir = "InversionLS/Expt/horizontal-well/"
    num_procs = min(mp.cpu_count(), 20)
    obj = ScatteringIntegralGeneralVzInversion2d(
        basedir=basedir,
        restart=True,
        restart_code=None,
        check_iter_files=True,
        num_procs_check_iter_files=num_procs
    )

    # Check arguments
    if len(sys.argv) < 3:
        raise ValueError("Program missing command line arguments.")

    num_iter = int(sys.argv[1])
    num_outer_iter = int(sys.argv[2])
    num_procs = min(obj.num_sources, mp.cpu_count(), 100)

    lambda_arr = np.zeros(shape=(obj.num_k_values, obj.num_sources), dtype=np.float32) + 1.0
    mu_arr = np.zeros(shape=(obj.num_k_values, obj.num_sources), dtype=np.float32) + 1.0

    obj.perform_inversion_update_wavefield_model_pert(
        iter_count=num_iter, num_outer_iter=num_outer_iter,
        lambda_arr=lambda_arr, mu_arr=mu_arr,
        max_iter=40, solver="cg", atol=1e-5, btol=1e-5,
        max_iter1=20, tol=1e-5, mnorm=0.0, use_bounds=False,
        num_procs=num_procs, clean=True
    )

    with np.load(obj.model_pert_filename(iter_count=num_iter)) as data:
        model_pert = data["arr_0"]

    np.save(basedir + "data/p06a-update-mnorm-0-bounds-off.npz", model_pert)

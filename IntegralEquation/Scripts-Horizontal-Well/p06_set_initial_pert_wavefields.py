import multiprocessing as mp
from ..Inversion.ScatteringIntegralGeneralVzInversion import ScatteringIntegralGeneralVzInversion2d


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

    num_procs = min(obj.num_sources, mp.cpu_count(), 30)
    obj.set_zero_initial_pert(num_procs=num_procs)

import numpy as np
from ..Inversion.ScatteringIntegralGeneralVzInversion import ScatteringIntegralGeneralVzInversion2d


if __name__ == "__main__":

    basedir = "InversionLS/Expt/marmousi1/"
    obj = ScatteringIntegralGeneralVzInversion2d(
        basedir=basedir,
        restart=True,
        restart_code=None
    )

    num_k_vals = obj.num_k_values

    # Design amplitudes
    amplitude_list = np.zeros(shape=(num_k_vals,), dtype=obj.precision) + 1.0

    print("\n\n---------------------------------------------")
    print("---------------------------------------------")
    print("Creating sources...")

    nx = obj.n
    nz = obj.nz
    a = obj.a
    b = obj.b
    dx = 1.0 / (nx - 1)
    dz = (b - a) / (nz - 1)
    std = 3 * dx

    num_sources = 20
    xgrid = np.linspace(start=-0.4, stop=0.4, num=num_sources, endpoint=True)
    zval = 5 * dz
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

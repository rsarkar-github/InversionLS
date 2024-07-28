import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from ..Inversion.ScatteringIntegralGeneralVzInversion import ScatteringIntegralGeneralVzInversion2d


if __name__ == "__main__":


    basedir = "InversionLS/Expt/horizontal-well/"
    obj = ScatteringIntegralGeneralVzInversion2d(
        basedir=basedir,
        restart=True,
        restart_code=None,
        check_iter_files=False,
        num_procs_check_iter_files=4
    )

    # ------------------------------------
    # Get parameters
    # ------------------------------------
    num_k_vals = obj.num_k_values
    num_sources = obj.num_sources
    num_iter = 14

    # ------------------------------------
    # Load files
    # ------------------------------------

    obj1_sum = np.zeros(shape=(14,), dtype=np.float32)
    obj2_sum = np.zeros(shape=(14,), dtype=np.float32)
    obj3_sum = np.zeros(shape=(14,), dtype=np.float32)

    for iter_num in range(num_iter):

        obj1_fname = obj.obj1_filename(iter_count=iter_num)
        with np.load(obj1_fname) as f:
            obj1 = f["arr_0"]
        obj1_sum[iter_num] = np.sum(obj1)

        obj2_fname = obj.obj2_filename(iter_count=iter_num, iter_step=0)
        with np.load(obj2_fname) as f:
            obj2 = f["arr_0"]
        obj2_sum[iter_num] = np.sum(obj2)

        obj3_fname = obj.obj2_filename(iter_count=iter_num, iter_step=1)
        with np.load(obj3_fname) as f:
            obj3 = f["arr_0"]
        obj3_sum[iter_num] = np.sum(obj3)


    x = np.asarray([i for i in range(num_iter)], dtype=np.int32)

    # ------------------------------------
    # Plot
    # ------------------------------------
    figsize = (9, 5)
    fontsize = 20
    scale = 1e-10

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    line1, = ax.semilogy(
        x, obj1_sum / scale, 'r:x',
        linewidth=2, markersize=8,
        label=r"$\mathcal{J}^{\text{data}}_0$"
    )
    line2, = ax.semilogy(
        x, obj2_sum / scale, 'b:x',
        linewidth=2, markersize=8,
        label=r"$\mathcal{J}^{\text{LSE}}_1$"
    )
    line3, = ax.semilogy(
        x, obj3_sum / scale, 'k:x',
        linewidth=2, markersize=8,
        label=r"$\mathcal{J}^{\text{LSE}}_2$"
    )

    ax.set_xlabel('Iteration set', fontsize=fontsize, fontname="STIXGeneral")
    ax.set_ylabel('Objective function', fontsize=fontsize, fontname="STIXGeneral")
    ax.grid()
    ax.set_xlim((0, num_iter - 1))
    ax.legend(handles=[line1, line2, line3])

    fig.savefig(
        basedir + "Fig/q07_obj_total_plots.pdf",
        format="pdf",
        pad_inches=0.01
    )

    plt.show()
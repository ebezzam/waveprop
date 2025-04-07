"""
Show benefits of bandlimiting angular spectrum.

Paper: Band-Limited Angular Spectrum Method for Numerical Simulation of Free-Space Propagation in Far and Near Fields (2009)

"""

import hydra
import os
import numpy as np
import matplotlib.pyplot as plt
from waveprop.util import rect2d, sample_points, plot2d
from waveprop.rs import angular_spectrum, direct_integration
import matplotlib


@hydra.main(version_base=None, config_path="../configs", config_name="square_ap_blas")
def blas(config):

    matplotlib.rc("font", **config.plot.font)
    xlim = config.plot.xlim

    # simulation params
    diam = config.sim.diam  # diameter of aperture [m]
    N = config.sim.N  # number of grid points per size
    L = config.sim.L  # total size of grid
    wv = config.sim.wv  # wavelength
    dz = config.sim.dz  # distance [m]
    d1 = L / N  # source-plane grid spacing

    print("\nPROPAGATION DISTANCE : {} m".format(dz))

    """ discretize aperture """
    x1, y1 = sample_points(N=N, delta=d1)
    u_in = rect2d(x1, y1, diam)

    """ Angular spectrum """
    u_out_asm, x_asm, y_asm = angular_spectrum(u_in=u_in, wv=wv, d1=d1, dz=dz, bandlimit=False)
    u_out_asm_bl, _, _ = angular_spectrum(u_in=u_in, wv=wv, d1=d1, dz=dz, bandlimit=True)

    """ Direct integration (ground truth) """
    u_out_di = direct_integration(u_in, wv, d1, dz, x=x_asm[0], y=[0])

    # plot y2 = 0 cross-section
    idx = y_asm[:, 0] == 0
    plt.figure()
    plt.plot(x_asm[0], np.abs(u_out_asm[idx][0]), label="AS", alpha=config.plot.alpha)
    plt.plot(x_asm[0], np.abs(u_out_asm_bl[idx][0]), label="BLAS", alpha=config.plot.alpha)
    plt.plot(x_asm[0], np.abs(u_out_di[0]), label="direct integration", alpha=config.plot.alpha)

    plt.xlabel("x[m]")
    plt.legend()
    if config.plot.cross_section_log:
        plt.yscale("log")
        plt.title("log amplitude, y = 0")
    else:
        plt.title("amplitude, y = 0")
    if xlim is not None:
        xlim = [max(xlim[0], np.min(x_asm)), min(xlim[1], np.max(x_asm))]
    else:
        xlim = [np.min(x_asm), np.max(x_asm)]
    plt.xlim(xlim)
    plt.savefig("2_cross_section.png", dpi=config.plot.dpi)

    # plot input
    plot2d(x1.squeeze(), y1.squeeze(), u_in, title="Aperture")
    plt.savefig("1_input.png", dpi=config.plot.dpi)

    # plot output
    plot2d(x_asm.squeeze(), y_asm.squeeze(), np.abs(u_out_asm_bl), title="BLAS")
    plt.savefig("4_bandlimited_angular_spectrum.png", dpi=config.plot.dpi)
    plot2d(x_asm.squeeze(), y_asm.squeeze(), np.abs(u_out_asm), title="AS")
    plt.savefig("3_angular_spectrum.png", dpi=config.plot.dpi)

    print(f"\nSaved figures to {os.getcwd()}")


if __name__ == "__main__":
    blas()
